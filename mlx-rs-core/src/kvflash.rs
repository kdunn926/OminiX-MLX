//! KVFlash — bounded-residency KV cache (spike).
//!
//! Port of the *decode-time bounded-attention* core of FlashMemory-style KV
//! paging (lucebox-hub PR #373) to MLX / Apple Silicon.
//!
//! ## What it does
//!
//! Instead of letting the full-attention KV cache grow with the logical
//! context, [`KvFlashCache`] keeps only a bounded **resident pool** of at most
//! `pool` tokens: the first `sink` "attention-sink" tokens (StreamingLLM)
//! plus the most-recent `pool - sink` tokens, evicting the oldest non-sink
//! 64-token chunk when the pool overflows. Decode then attends over a fixed
//! `≤ pool` working set, so decode throughput stays **flat** as the context
//! grows instead of degrading with KV size.
//!
//! ## Why it's exact over the kept set
//!
//! The model bakes RoPE into K *at write time* using the token's true
//! position ([`offset`](KvFlashCache::offset) returns the logical count, so
//! positions keep advancing). RoPE is *relative*, so a query at logical
//! position `N` attending to any kept key at position `p` sees the correct
//! relative offset `N - p` regardless of which middle chunks were dropped.
//! Attention over the resident set is therefore bit-identical to a full-cache
//! run *restricted to those keys* — the only approximation is the dropped
//! chunks' contribution (the LRU/StreamingLLM quality trade-off).
//!
//! ## Spike scope / differences from the source PR
//!
//! - **Decode-bounding only.** Multi-token (prefill) appends accumulate the
//!   full KV so the prefill causal mask stays aligned; eviction triggers only
//!   on single-token decode appends. Bounded chunked *prefill* (the PR's
//!   prefill speedup) is future work.
//! - **No host paging.** Apple Silicon is unified memory, so there is no
//!   discrete-GPU VRAM to page to/from — evicted chunks are simply dropped
//!   from the resident tensor (and from the model's attention). The PR's
//!   99% VRAM reduction does not map; the *decode throughput* win does.
//! - **LRU policy only.** The drafter/scored residency policy that preserves
//!   long-range recall is out of scope; this keeps sink + recent (pure
//!   recency), i.e. StreamingLLM.

use mlx_rs::ops::indexing::{Ellipsis, IndexMutOp, IndexOp};
use mlx_rs::{error::Exception, ops::concatenate_axis, ops::zeros_dtype, Array};

use crate::cache::KeyValueCache;

/// Default eviction chunk size (tokens) — matches the source PR's 64.
pub const DEFAULT_CHUNK: i32 = 64;
/// Default attention-sink tokens kept at the front.
pub const DEFAULT_SINK: i32 = 4;

/// Residency policy: which chunk to evict on overflow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Policy {
    /// Drop the oldest non-sink chunk (StreamingLLM): sink + recent window.
    Lru,
    /// Drop the lowest accumulated-attention chunk among the unprotected
    /// middle (H2O-style heavy-hitter retention): keep sink + recent window +
    /// the most-attended middle chunks. `observe_query` must be wired so the
    /// model's queries accumulate per-token attention mass.
    Scored,
}

/// A bounded-residency KV cache for full-attention layers. See module docs.
#[derive(Debug, Clone)]
pub struct KvFlashCache {
    /// Resident key buffer `[B, Hkv, cap, D]` (step-allocated; `resident`
    /// valid rows). `None` until the first write.
    keys: Option<Array>,
    values: Option<Array>,
    /// Allocation step for the resident buffer.
    step: i32,
    /// Physical tokens currently held (`<= pool` after decode eviction).
    resident: i32,
    /// True logical token count — what `offset()` reports so RoPE positions
    /// keep advancing across evictions.
    logical: i32,
    /// Attention-sink tokens always kept at the front.
    sink: i32,
    /// Max resident tokens.
    pool: i32,
    /// Eviction granularity (tokens).
    chunk: i32,
    /// When true, evict on multi-token (prefill) appends too, so prefill
    /// attention is bounded to the pool (memory bound + O(seq·pool) attention
    /// instead of O(seq²)). Requires chunked prefill with chunk ≤ pool − sink,
    /// and that the just-appended block is the suffix of the resident set —
    /// both hold for the qwen3.6 chunked-prefill path (hardware causal mask).
    bound_prefill: bool,
    /// Eviction policy (LRU or Scored).
    policy: Policy,
    /// Recent tokens always protected from scored eviction (recency window).
    recent: i32,
    /// Per-`observe_query` decay applied to accumulated scores before adding
    /// the new attention mass. `1.0` = plain H2O accumulation; `< 1.0`
    /// recency-weights so recent queries (e.g. the trailing question) dominate
    /// — which is what recovers a needle that only the late query attends.
    decay: f32,
    /// Per-resident-position accumulated attention mass `[resident]` (device).
    /// Maintained in lockstep with `keys`/`values` on append/evict. `None`
    /// until the first write or under `Policy::Lru`.
    scores: Option<Array>,
    /// Number of evictions performed (stat).
    pub evictions: usize,
}

impl KvFlashCache {
    /// `pool`: max resident tokens. `sink`: front tokens always kept.
    /// `chunk`: eviction granularity (oldest non-sink chunk dropped).
    /// `bound_prefill`: also bound prefill (evict on multi-token appends).
    pub fn new(pool: i32, sink: i32, chunk: i32, bound_prefill: bool) -> Self {
        let chunk = chunk.max(1);
        let sink = sink.clamp(0, (pool - chunk).max(0));
        Self {
            keys: None,
            values: None,
            step: 256,
            resident: 0,
            logical: 0,
            sink,
            pool: pool.max(sink + chunk),
            chunk,
            bound_prefill,
            policy: Policy::Lru,
            recent: 0,
            decay: 1.0,
            scores: None,
            evictions: 0,
        }
    }

    /// Switch to H2O-style scored residency: keep sink + a `recent`-token
    /// recency window + the highest accumulated-attention middle chunks.
    /// `decay` (per `observe_query`, `<= 1.0`) recency-weights the
    /// accumulation. Requires the caller to wire [`KeyValueCache::observe_query`].
    pub fn enable_scoring(&mut self, recent: i32, decay: f32) {
        self.policy = Policy::Scored;
        // Leave room for sink + at least one evictable middle chunk.
        self.recent = recent.clamp(0, (self.pool - self.sink - self.chunk).max(0));
        self.decay = decay.clamp(0.0, 1.0);
    }

    pub fn policy(&self) -> Policy {
        self.policy
    }

    pub fn pool(&self) -> i32 {
        self.pool
    }

    pub fn resident(&self) -> i32 {
        self.resident
    }

    /// Grow the resident buffer to hold at least `need` tokens (step-rounded),
    /// preserving the existing `resident` rows.
    fn ensure_capacity(
        &mut self,
        need: i32,
        b: i32,
        hkv: i32,
        k_d: i32,
        v_d: i32,
        k_dtype: mlx_rs::Dtype,
        v_dtype: mlx_rs::Dtype,
    ) -> Result<(), Exception> {
        let cap = self.keys.as_ref().map(|k| k.shape()[2]).unwrap_or(0);
        if need <= cap {
            return Ok(());
        }
        let n_steps = (need + self.step - 1) / self.step;
        let new_size = n_steps * self.step;
        let new_k = zeros_dtype(&[b, hkv, new_size, k_d], k_dtype)?;
        let new_v = zeros_dtype(&[b, hkv, new_size, v_d], v_dtype)?;
        match (self.keys.take(), self.values.take()) {
            (Some(old_k), Some(old_v)) => {
                let old_k = old_k.index((Ellipsis, ..self.resident, ..));
                let old_v = old_v.index((Ellipsis, ..self.resident, ..));
                self.keys = Some(concatenate_axis(&[old_k, new_k], 2)?);
                self.values = Some(concatenate_axis(&[old_v, new_v], 2)?);
            }
            _ => {
                self.keys = Some(new_k);
                self.values = Some(new_v);
            }
        }
        Ok(())
    }

    /// Drop resident positions `[a, b)` from keys, values, and scores,
    /// compacting around the gap (keeps `[0..a]` ++ `[b..resident]`).
    fn drop_range(&mut self, a: i32, b: i32) -> Result<(), Exception> {
        if b <= a {
            return Ok(());
        }
        let r = self.resident;
        let keep = |arr: Array| -> Result<Array, Exception> {
            if a == 0 {
                Ok(arr.index((Ellipsis, b..r, ..)))
            } else {
                concatenate_axis(
                    &[arr.index((Ellipsis, ..a, ..)), arr.index((Ellipsis, b..r, ..))],
                    2,
                )
            }
        };
        let k = self.keys.take().unwrap();
        let v = self.values.take().unwrap();
        self.keys = Some(keep(k)?);
        self.values = Some(keep(v)?);
        if let Some(s) = self.scores.take() {
            // scores is 1-D [resident].
            let kept = if a == 0 {
                s.index((b..r,))
            } else {
                concatenate_axis(&[s.index((..a,)), s.index((b..r,))], 0)?
            };
            self.scores = Some(kept);
        }
        self.resident -= b - a;
        self.evictions += 1;
        Ok(())
    }

    fn evict(&mut self) -> Result<(), Exception> {
        match self.policy {
            Policy::Lru => self.evict_lru(),
            Policy::Scored => self.evict_scored(),
        }
    }

    /// LRU: drop the oldest non-sink chunk(s) so `resident <= pool`.
    fn evict_lru(&mut self) -> Result<(), Exception> {
        if self.resident <= self.pool {
            return Ok(());
        }
        let excess = self.resident - self.pool;
        let mut drop = ((excess + self.chunk - 1) / self.chunk) * self.chunk;
        drop = drop.min(self.resident - self.sink - 1).max(0);
        if drop == 0 {
            return Ok(());
        }
        self.drop_range(self.sink, self.sink + drop)
    }

    /// Scored: while over pool, drop the lowest accumulated-attention chunk in
    /// the unprotected middle `[sink, resident - recent)` (H2O heavy-hitter
    /// retention). Falls back to LRU when no scores have accrued yet.
    fn evict_scored(&mut self) -> Result<(), Exception> {
        while self.resident > self.pool {
            let lo = self.sink;
            let hi = self.resident - self.recent; // exclusive; protected recent tail
            if hi - lo < self.chunk {
                // Middle too small to drop a chunk — protect-everything case;
                // fall back to dropping the oldest evictable chunk.
                let drop = (self.resident - self.pool)
                    .min(self.resident - self.sink - 1)
                    .max(0);
                if drop == 0 {
                    return Ok(());
                }
                return self.drop_range(self.sink, self.sink + drop);
            }
            // Per-chunk accumulated score over the middle; drop the argmin.
            let scores_host: Vec<f32> = match self.scores.as_ref() {
                Some(s) => {
                    let s = s.contiguous()?;
                    mlx_rs::transforms::eval([&s])?;
                    s.as_slice::<f32>().to_vec()
                }
                None => vec![0.0; self.resident as usize],
            };
            let mut best_start = lo;
            let mut best_sum = f32::INFINITY;
            let mut c = lo;
            while c + self.chunk <= hi {
                let sum: f32 = scores_host[c as usize..(c + self.chunk) as usize].iter().sum();
                if sum < best_sum {
                    best_sum = sum;
                    best_start = c;
                }
                c += self.chunk;
            }
            self.drop_range(best_start, best_start + self.chunk)?;
        }
        Ok(())
    }
}

impl KeyValueCache for KvFlashCache {
    /// Logical token count — drives the model's RoPE position for new K/Q.
    fn offset(&self) -> i32 {
        self.logical
    }

    /// Bounded physical key length the next `update_and_fetch` returns.
    fn physical_offset(&self) -> i32 {
        self.resident
    }

    /// `None`: bounding is handled internally via `update_and_fetch`, not via
    /// the caller's sliding-window mask machinery.
    fn max_size(&self) -> Option<i32> {
        None
    }

    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
    ) -> Result<(Array, Array), Exception> {
        let ks = keys.shape();
        let vs = values.shape();
        let (b, hkv, num_new, k_d) = (ks[0], ks[1], ks[2], ks[3]);
        let v_d = vs[3];

        self.ensure_capacity(
            self.resident + num_new,
            b,
            hkv,
            k_d,
            v_d,
            keys.dtype(),
            values.dtype(),
        )?;

        let start = self.resident;
        let end = start + num_new;
        {
            let k = self.keys.as_mut().unwrap();
            let v = self.values.as_mut().unwrap();
            k.index_mut((Ellipsis, start..end, ..), &keys);
            v.index_mut((Ellipsis, start..end, ..), &values);
        }
        self.resident = end;
        self.logical += num_new;

        // Grow the per-position score buffer (zeros for the new tokens) so it
        // stays aligned with the resident set under scored residency.
        if self.policy == Policy::Scored {
            let zeros = zeros_dtype(&[num_new], mlx_rs::Dtype::Float32)?;
            self.scores = Some(match self.scores.take() {
                Some(s) => concatenate_axis(&[s, zeros], 0)?,
                None => zeros,
            });
        }

        // Evict on single-token (decode) appends always; on multi-token
        // (prefill) appends only when `bound_prefill`. Eviction keeps the
        // just-appended block as the resident suffix, so the caller's hardware
        // causal mask (`L_k − L_q` offset) stays correct: the pool prefix is
        // fully visible and the new block is causal among itself.
        if num_new == 1 || self.bound_prefill {
            self.evict()?;
        }

        let k = self.keys.as_ref().unwrap();
        let v = self.values.as_ref().unwrap();
        Ok((
            k.index((Ellipsis, ..self.resident, ..)),
            v.index((Ellipsis, ..self.resident, ..)),
        ))
    }

    fn current_kv(&self) -> Option<(Array, Array)> {
        let k = self.keys.as_ref()?;
        let v = self.values.as_ref()?;
        Some((
            k.index((Ellipsis, ..self.resident, ..)),
            v.index((Ellipsis, ..self.resident, ..)),
        ))
    }

    fn reset(&mut self) {
        self.keys = None;
        self.values = None;
        self.scores = None;
        self.resident = 0;
        self.logical = 0;
        self.evictions = 0;
    }

    /// Accumulate per-resident-position attention mass from the queries `q`
    /// (`[B, Hq, L, D]`) — a cheap mean-head softmax(q·kᵀ) summed over the
    /// query block. Heavy-hitter positions accrue mass and survive scored
    /// eviction. No-op under `Policy::Lru`.
    fn observe_query(&mut self, q: &Array) -> Result<(), Exception> {
        if self.policy != Policy::Scored || self.resident == 0 {
            return Ok(());
        }
        let Some(k) = self.keys.as_ref() else {
            return Ok(());
        };
        let kr = k.index((Ellipsis, ..self.resident, ..)); // [B, Hkv, R, D]
        let d = q.shape()[3];
        // Collapse heads (cheap proxy for the per-head attention).
        let q_mean = q.mean_axes(&[1], false)?; // [B, L, D]
        let k_mean = kr.mean_axes(&[1], false)?; // [B, R, D]
        let scale = (d as f32).powf(-0.5);
        // logits [B, L, R] = q_mean @ k_meanᵀ
        let logits = mlx_rs::ops::matmul(&q_mean, &k_mean.transpose_axes(&[0, 2, 1])?)?
            .multiply(mlx_rs::array!(scale))?;
        let attn = mlx_rs::ops::softmax_axes(&logits, &[-1], None)?; // over R
        // sum over the query block and batch → [R]
        let mass = attn.sum_axes(&[0, 1], false)?; // [R]
        self.scores = Some(match self.scores.take() {
            Some(s) if self.decay < 1.0 => s.multiply(mlx_rs::array!(self.decay))?.add(&mass)?,
            Some(s) => s.add(&mass)?,
            None => mass,
        });
        Ok(())
    }
}

/// Host-paging KVFlash cache: keep **all** chunks in (unified) memory and bound
/// only the attention *working set*. Every `tau` decode steps it reselects the
/// resident set by scoring **all** chunks — including paged-out ones — against
/// the latest query, so a chunk is "paged back in" the moment a query attends
/// it. This recovers the mid-context recall the drop-on-evict policies lose:
/// the first decode reselection uses the trailing query (the question, captured
/// at the end of prefill), which pulls its relevant chunk back into residency.
///
/// On Apple Silicon (unified memory) this trades the memory bound (all chunks
/// stay resident in unified memory) for full recall plus a bounded attention
/// read — the decode-throughput win is kept; the VRAM win is not (there is no
/// separate host RAM to evacuate to).
pub struct KvFlashPagedCache {
    ck: Vec<Array>,     // frozen chunk keys   [B, Hkv, chunk, D]
    cv: Vec<Array>,     // frozen chunk values
    cmean: Vec<Array>,  // per-chunk reduced mean key [D] for cheap scoring
    pk: Option<Array>,  // pending (sub-chunk) tail
    pv: Option<Array>,
    plen: i32,
    logical: i32,
    pool: i32,
    chunk: i32,
    sink_chunks: i32,
    recent_chunks: i32,
    tau: i32,
    steps: i32,
    resident: Vec<usize>, // chunk indices in the attention set (sorted)
    q_mean: Option<Array>, // [D] head-mean of the latest query
    pub pages_in: usize,
}

impl std::fmt::Debug for KvFlashPagedCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KvFlashPagedCache")
            .field("logical", &self.logical)
            .field("chunks", &self.ck.len())
            .field("resident", &self.resident.len())
            .field("pool", &self.pool)
            .field("pages_in", &self.pages_in)
            .finish()
    }
}

impl Clone for KvFlashPagedCache {
    fn clone(&self) -> Self {
        Self {
            ck: self.ck.clone(),
            cv: self.cv.clone(),
            cmean: self.cmean.clone(),
            pk: self.pk.clone(),
            pv: self.pv.clone(),
            plen: self.plen,
            logical: self.logical,
            pool: self.pool,
            chunk: self.chunk,
            sink_chunks: self.sink_chunks,
            recent_chunks: self.recent_chunks,
            tau: self.tau,
            steps: self.steps,
            resident: self.resident.clone(),
            q_mean: self.q_mean.clone(),
            pages_in: self.pages_in,
        }
    }
}

impl KvFlashPagedCache {
    pub fn new(pool: i32, sink: i32, chunk: i32, recent: i32, tau: i32) -> Self {
        let chunk = chunk.max(1);
        let sink_chunks = (sink + chunk - 1) / chunk;
        let recent_chunks = (recent.max(chunk) + chunk - 1) / chunk;
        Self {
            ck: Vec::new(),
            cv: Vec::new(),
            cmean: Vec::new(),
            pk: None,
            pv: None,
            plen: 0,
            logical: 0,
            pool: pool.max((sink_chunks + recent_chunks + 1) * chunk),
            chunk,
            sink_chunks,
            recent_chunks,
            tau: tau.max(1),
            steps: 0,
            resident: Vec::new(),
            q_mean: None,
            pages_in: 0,
        }
    }

    fn n_resident_chunks(&self) -> i32 {
        self.pool / self.chunk
    }

    /// Freeze full `chunk`-sized blocks out of the pending tail.
    fn freeze(&mut self) -> Result<(), Exception> {
        while self.plen >= self.chunk {
            let pk = self.pk.take().unwrap();
            let pv = self.pv.take().unwrap();
            let ck = pk.index((Ellipsis, ..self.chunk, ..));
            let cv = pv.index((Ellipsis, ..self.chunk, ..));
            // reduced mean key [D] (mean over Hkv and chunk positions, batch 0).
            let cm = ck.mean_axes(&[0, 1, 2], false)?; // [D]
            self.ck.push(ck);
            self.cv.push(cv);
            self.cmean.push(cm);
            let rem = self.plen - self.chunk;
            if rem > 0 {
                self.pk = Some(pk.index((Ellipsis, self.chunk.., ..)));
                self.pv = Some(pv.index((Ellipsis, self.chunk.., ..)));
            } else {
                self.pk = None;
                self.pv = None;
            }
            self.plen = rem;
        }
        Ok(())
    }

    /// Reselect the resident chunk set: pin sink + recent chunks, fill the rest
    /// with the highest-scoring middle chunks by `q_mean · cmean`.
    fn reselect(&mut self) -> Result<(), Exception> {
        let n = self.ck.len() as i32;
        let budget = self.n_resident_chunks();
        if n <= budget {
            self.resident = (0..n as usize).collect();
            return Ok(());
        }
        let sink_c = self.sink_chunks.min(n);
        let recent_c = self.recent_chunks.min(n - sink_c);
        let mid_lo = sink_c;
        let mid_hi = n - recent_c; // exclusive
        let mid_budget = (budget - sink_c - recent_c).max(0);

        // Score middle chunks against the latest query.
        let mut chosen: Vec<usize> = Vec::new();
        if mid_budget > 0 && mid_hi > mid_lo {
            if let Some(qm) = self.q_mean.as_ref() {
                // stack middle cmeans → [n_mid, D], score = mids @ qm → [n_mid]
                let mids: Vec<&Array> = (mid_lo..mid_hi)
                    .map(|i| &self.cmean[i as usize])
                    .collect();
                let stacked = mlx_rs::ops::stack_axis(&mids, 0)?; // [n_mid, D]
                let qcol = qm.reshape(&[-1, 1])?; // [D,1]
                let scores = mlx_rs::ops::matmul(&stacked, &qcol)?.reshape(&[-1])?;
                let scores = scores.contiguous()?;
                mlx_rs::transforms::eval([&scores])?;
                let sv = scores.as_slice::<f32>().to_vec();
                let mut order: Vec<usize> =
                    (0..sv.len()).collect();
                order.sort_by(|&a, &b| sv[b].partial_cmp(&sv[a]).unwrap_or(std::cmp::Ordering::Equal));
                for &o in order.iter().take(mid_budget as usize) {
                    chosen.push(mid_lo as usize + o);
                }
            } else {
                // No query yet → take the most recent middle chunks.
                for i in (mid_hi - mid_budget).max(mid_lo)..mid_hi {
                    chosen.push(i as usize);
                }
            }
        }
        let prev_resident: std::collections::HashSet<usize> =
            self.resident.iter().copied().collect();
        let mut res: Vec<usize> = Vec::new();
        for i in 0..sink_c {
            res.push(i as usize);
        }
        res.extend(chosen.iter().copied());
        for i in mid_hi..n {
            res.push(i as usize);
        }
        res.sort_unstable();
        res.dedup();
        // Count newly-paged-in chunks (were not resident last round).
        for &i in &res {
            if !prev_resident.contains(&i) {
                self.pages_in += 1;
            }
        }
        self.resident = res;
        Ok(())
    }

    /// Concatenate the resident chunks (position order) + pending tail.
    fn gather_resident(&self) -> Result<(Array, Array), Exception> {
        let mut ks: Vec<&Array> = self.resident.iter().map(|&i| &self.ck[i]).collect();
        let mut vs: Vec<&Array> = self.resident.iter().map(|&i| &self.cv[i]).collect();
        if let (Some(pk), Some(pv)) = (self.pk.as_ref(), self.pv.as_ref()) {
            ks.push(pk);
            vs.push(pv);
        }
        Ok((concatenate_axis(&ks, 2)?, concatenate_axis(&vs, 2)?))
    }
}

impl KeyValueCache for KvFlashPagedCache {
    fn offset(&self) -> i32 {
        self.logical
    }

    fn physical_offset(&self) -> i32 {
        self.resident.len() as i32 * self.chunk + self.plen
    }

    fn max_size(&self) -> Option<i32> {
        None
    }

    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
    ) -> Result<(Array, Array), Exception> {
        let num_new = keys.shape()[2];
        // Append to the pending tail.
        self.pk = Some(match self.pk.take() {
            Some(p) => concatenate_axis(&[p, keys], 2)?,
            None => keys,
        });
        self.pv = Some(match self.pv.take() {
            Some(p) => concatenate_axis(&[p, values], 2)?,
            None => values,
        });
        self.plen += num_new;
        self.logical += num_new;
        self.freeze()?;

        let n = self.ck.len() as i32;
        if num_new > 1 || n <= self.n_resident_chunks() {
            // Prefill (or still under budget): attend everything.
            self.resident = (0..n as usize).collect();
        } else {
            // Decode over budget: reselect every `tau` steps (and immediately
            // on the first over-budget step, using the trailing prefill query).
            self.steps += 1;
            if self.resident.len() as i32 > self.n_resident_chunks()
                || self.steps % self.tau == 0
            {
                self.reselect()?;
            }
        }
        self.gather_resident()
    }

    fn current_kv(&self) -> Option<(Array, Array)> {
        self.gather_resident().ok()
    }

    fn reset(&mut self) {
        *self = KvFlashPagedCache::new(
            self.pool,
            self.sink_chunks * self.chunk,
            self.chunk,
            self.recent_chunks * self.chunk,
            self.tau,
        );
    }

    fn observe_query(&mut self, q: &Array) -> Result<(), Exception> {
        // Head + seq mean → [D]; drives the next reselection.
        self.q_mean = Some(q.mean_axes(&[0, 1, 2], false)?);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::arange;

    // One token of K/V whose single value == the token id, so we can read back
    // exactly which logical positions remain resident. Shape [1, 1, 1, 1].
    fn tok(id: f32) -> (Array, Array) {
        let a = Array::from_slice(&[id], &[1, 1, 1, 1]);
        (a.clone(), a)
    }

    fn resident_ids(c: &KvFlashCache) -> Vec<i32> {
        let (k, _) = c.current_kv().unwrap();
        let flat = k.flatten(None, None).unwrap();
        flat.as_slice::<f32>().iter().map(|&x| x as i32).collect()
    }

    #[test]
    fn bounds_resident_and_keeps_sink_plus_recent() {
        let _ = arange::<_, f32>(0.0, 1.0, None); // touch mlx to init device
        // pool 8, sink 2, chunk 2, decode-only bounding.
        let mut c = KvFlashCache::new(8, 2, 2, false);
        // Prefill 5 tokens in one shot (multi-token append: no eviction).
        let k = Array::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0], &[1, 1, 5, 1]);
        c.update_and_fetch(k.clone(), k).unwrap();
        assert_eq!(c.offset(), 5);
        assert_eq!(c.resident(), 5);
        // Decode tokens 5..=15 one at a time; resident must never exceed pool.
        for id in 5..=15 {
            let (kk, vv) = tok(id as f32);
            c.update_and_fetch(kk, vv).unwrap();
            assert!(c.resident() <= c.pool(), "resident {} > pool {}", c.resident(), c.pool());
        }
        assert_eq!(c.offset(), 16); // logical advanced for all 16 tokens
        let ids = resident_ids(&c);
        // Sink = first 2 logical positions (0,1) always kept.
        assert_eq!(&ids[..2], &[0, 1], "sink not preserved: {ids:?}");
        // The most-recent token (15) must be present.
        assert_eq!(*ids.last().unwrap(), 15, "recent tail missing: {ids:?}");
        // Resident is bounded.
        assert!(ids.len() as i32 <= c.pool());
        assert!(c.evictions > 0, "expected evictions");
    }

    #[test]
    fn bounded_prefill_caps_resident_during_chunked_append() {
        let _ = arange::<_, f32>(0.0, 1.0, None);
        // pool 8, sink 2, chunk 2, bound_prefill = true.
        let mut c = KvFlashCache::new(8, 2, 2, true);
        // Simulate chunked prefill: 4-token chunks, total 20 tokens.
        let mut next = 0.0f32;
        for _chunk in 0..5 {
            let ids: Vec<f32> = (0..4).map(|_| { let v = next; next += 1.0; v }).collect();
            let k = Array::from_slice(&ids, &[1, 1, 4, 1]);
            c.update_and_fetch(k.clone(), k).unwrap();
            // Resident stays bounded *during* prefill (the whole point).
            assert!(c.resident() <= c.pool(), "prefill resident {} > pool {}", c.resident(), c.pool());
        }
        assert_eq!(c.offset(), 20); // logical advanced for all 20
        let ids = resident_ids(&c);
        assert_eq!(&ids[..2], &[0, 1], "sink not preserved across prefill: {ids:?}");
        assert_eq!(*ids.last().unwrap(), 19, "most-recent prefill token missing: {ids:?}");
        assert!(c.evictions > 0, "expected prefill evictions");
    }

    #[test]
    fn scored_eviction_keeps_the_heavy_hitter() {
        let _ = arange::<_, f32>(0.0, 1.0, None);
        // pool 6, sink 1, recent 2, chunk 1, scored. D = 16 one-hot keys so a
        // query at index 3 attends position 3; V[i] = i so we can read survivors.
        let mut c = KvFlashCache::new(6, 1, 1, false);
        c.enable_scoring(2, 1.0);
        let d = 16usize;
        let q3 = {
            // query == one-hot(3) * 8 (strong), shape [1,1,1,16]
            let mut v = vec![0.0f32; d]; v[3] = 8.0;
            Array::from_slice(&v, &[1, 1, 1, d as i32])
        };
        for id in 0..12i32 {
            let mut kv = vec![0.0f32; d];
            kv[(id as usize) % d] = 1.0; // one-hot(id) key
            let k = Array::from_slice(&kv, &[1, 1, 1, d as i32]);
            let v = Array::from_slice(&vec![id as f32; d], &[1, 1, 1, d as i32]); // V[i]=i
            c.update_and_fetch(k, v).unwrap();
            c.observe_query(&q3).unwrap(); // position 3 keeps getting attention
            assert!(c.resident() <= c.pool());
        }
        // Read survivor ids from V (first column of each resident row; V[i]=i).
        let (_, vv) = c.current_kv().unwrap();
        let col0 = vv.reshape(&[-1, d as i32]).unwrap().index((Ellipsis, ..1)); // [R,1]
        let col0 = col0.contiguous().unwrap();
        mlx_rs::transforms::eval([&col0]).unwrap();
        let ids: Vec<i32> = col0.as_slice::<f32>().iter().map(|&x| x as i32).collect();
        // Heavy hitter (position 3) must survive despite being old & non-recent.
        assert!(ids.contains(&3), "scored eviction dropped the heavy hitter: {ids:?}");
        // Sink (0) and the recent tail (11) are kept too.
        assert!(ids.contains(&0), "sink dropped: {ids:?}");
        assert!(ids.contains(&11), "recent tail dropped: {ids:?}");
    }
}
