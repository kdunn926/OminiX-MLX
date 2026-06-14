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
    /// Number of evictions performed (stat).
    pub evictions: usize,
}

impl KvFlashCache {
    /// `pool`: max resident tokens. `sink`: front tokens always kept.
    /// `chunk`: eviction granularity (oldest non-sink chunk dropped).
    pub fn new(pool: i32, sink: i32, chunk: i32) -> Self {
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
            evictions: 0,
        }
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

    /// Drop the oldest non-sink chunk(s) so `resident <= pool`. Keeps
    /// `[0..sink]` ++ `[sink+drop..resident]` (sink + most-recent tokens).
    fn evict(&mut self) -> Result<(), Exception> {
        if self.resident <= self.pool {
            return Ok(());
        }
        let excess = self.resident - self.pool;
        // Round the drop up to a whole chunk so eviction happens ~once per
        // `chunk` decode steps, not every step; never drop the sink or all of
        // the recent window.
        let mut drop = ((excess + self.chunk - 1) / self.chunk) * self.chunk;
        drop = drop.min(self.resident - self.sink - 1).max(0);
        if drop == 0 {
            return Ok(());
        }
        let k = self.keys.take().unwrap();
        let v = self.values.take().unwrap();
        let keep_k = if self.sink > 0 {
            concatenate_axis(
                &[
                    k.index((Ellipsis, ..self.sink, ..)),
                    k.index((Ellipsis, self.sink + drop..self.resident, ..)),
                ],
                2,
            )?
        } else {
            k.index((Ellipsis, drop..self.resident, ..))
        };
        let keep_v = if self.sink > 0 {
            concatenate_axis(
                &[
                    v.index((Ellipsis, ..self.sink, ..)),
                    v.index((Ellipsis, self.sink + drop..self.resident, ..)),
                ],
                2,
            )?
        } else {
            v.index((Ellipsis, drop..self.resident, ..))
        };
        self.keys = Some(keep_k);
        self.values = Some(keep_v);
        self.resident -= drop;
        self.evictions += 1;
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

        // Evict only on single-token (decode) appends so multi-token prefill
        // keeps a contiguous causal window for the caller's mask.
        if num_new == 1 {
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
        self.resident = 0;
        self.logical = 0;
        self.evictions = 0;
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
        // pool 8, sink 2, chunk 2.
        let mut c = KvFlashCache::new(8, 2, 2);
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
}
