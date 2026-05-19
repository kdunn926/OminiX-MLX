//! Minimal MTPLX speculative decoding session.
//!
//! Cycle (greedy, T=0):
//!   1. Prefill the target on the prompt (uses
//!      `qwen3_6_mlx::Model::forward_last_logits`).
//!   2. Per step:
//!      a. If the MTP head is loaded AND not a stub (`is_stub()` checks
//!         that real weights were materialized), draft 1 token from the
//!         last hidden state + previous-token embedding, then verify it
//!         with one target forward.
//!      b. Else: emit a single autoregressive token (AR fallback).
//!
//! Today only K=1 is implemented (matching Qwen3.6's `mtp_num_hidden_layers=1`
//! config) and only greedy acceptance at T=0. K>1 and probabilistic
//! Leviathan-Chen acceptance are out of scope — see WIP.md.

use std::collections::HashSet;
use std::time::Instant;

use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::{argmax_axis, Array};
use qwen3_6_mlx::{HybridCache, Model};

use crate::acceptance::{accept_speculative, default_rng, AcceptanceMode};
use crate::graph_bank::{GraphBank, GraphKey};
use crate::MtpError;

/// Speculative loop configuration.
#[derive(Debug, Clone)]
pub struct SpeculativeConfig {
    /// Draft block length. For MTP-1 architectures (Qwen3.6) we use K=4
    /// as a reasonable default; the head only predicts +1 token per
    /// invocation but can be unrolled.
    pub block_len: usize,
    /// Hard ceiling on tokens produced (excludes prompt).
    pub max_tokens: usize,
    /// Sampling temperature. For `AcceptanceMode::Greedy` only T=0 is
    /// sensible; for `AcceptanceMode::Speculative` any T>0 is valid.
    pub temp: f32,
    /// Acceptance strategy for verifying drafted tokens. Defaults to
    /// greedy for backwards compatibility.
    pub acceptance: AcceptanceMode,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            block_len: 4,
            max_tokens: 200,
            temp: 0.0,
            acceptance: AcceptanceMode::Greedy,
        }
    }
}

/// Aggregate per-session statistics.
#[derive(Debug, Clone, Default)]
pub struct SessionMetrics {
    pub prefill_s: f64,
    pub decode_s: f64,
    pub total_tokens: usize,
    pub mtp_cycles: usize,
    pub ar_fallback_steps: usize,
    /// Total draft tokens proposed by the MTP head across all cycles.
    /// For K=1 cycles (the current implementation), this equals `mtp_cycles`.
    pub mtp_drafted: usize,
    /// Total draft tokens that matched the target's argmax (greedy) or
    /// passed the speculative-accept probability ratio (sampling).
    pub mtp_accepted: usize,
}

impl SessionMetrics {
    pub fn mtp_acceptance_rate(&self) -> f64 {
        if self.mtp_drafted > 0 {
            self.mtp_accepted as f64 / self.mtp_drafted as f64
        } else {
            0.0
        }
    }
}

impl SessionMetrics {
    pub fn decode_tok_per_s(&self) -> f64 {
        if self.decode_s > 0.0 && self.total_tokens > 0 {
            self.total_tokens as f64 / self.decode_s
        } else {
            0.0
        }
    }
}

/// Speculative-decoding session driving a single Qwen3.6 target.
pub struct MtplxSession {
    model: Model,
    cfg: SpeculativeConfig,
    /// `true` when the loaded MTP head has real weights (not a stub).
    /// We cache it because checking again requires a `&mut Model`.
    mtp_active: bool,
    /// Shape-keyed cache of compiled callables. Currently used in
    /// observation-only mode: we record hit/miss stats per verify
    /// without yet routing the forward through a `CompiledFn`. Real
    /// dispatch is a TODO — see `mtp_cycle`.
    graph_bank: GraphBank,
}

impl MtplxSession {
    /// Build a session around the given (already-loaded) Qwen3.6 target.
    pub fn new(mut model: Model, cfg: SpeculativeConfig) -> Self {
        let mtp_active = model
            .mtp_head()
            .map(|h| !h.is_stub())
            .unwrap_or(false);
        Self {
            model,
            cfg,
            mtp_active,
            graph_bank: GraphBank::new(),
        }
    }

    pub fn has_mtp_head(&self) -> bool {
        self.mtp_active
    }

    /// Snapshot of compiled-graph cache statistics.
    pub fn graph_bank_stats(&self) -> crate::graph_bank::GraphBankStats {
        self.graph_bank.stats()
    }

    /// Generate up to `cfg.max_tokens` tokens for the given prompt token
    /// IDs. Returns the produced token IDs (excluding prompt) plus run
    /// metrics. Stops early on any EOS in `eos`.
    pub fn generate(
        &mut self,
        prompt_ids: &[i32],
        eos: &HashSet<u32>,
    ) -> Result<(Vec<i32>, SessionMetrics), MtpError> {
        let mut metrics = SessionMetrics::default();
        // Reset the MTP head's persistent draft KV cache at the start of
        // every generation so the head's attention starts at position 0
        // for the new sequence (rather than carrying state from a prior
        // request in the same process).
        if let Some(mtp) = self.model.mtp_head() {
            mtp.reset_cache();
        }
        let mut cache: Vec<HybridCache> = self
            .model
            .new_cache(if std::env::var("TURBO_KV").is_ok() {
                qwen3_6_mlx::KVCacheMode::TurboQuant
            } else if std::env::var("QUANTIZE_KV").is_ok() {
                qwen3_6_mlx::KVCacheMode::Quantized
            } else {
                qwen3_6_mlx::KVCacheMode::Standard
            });

        // --- prefill ---
        let prefill_start = Instant::now();
        let prompt_arr =
            Array::from_slice(prompt_ids, &[1, prompt_ids.len() as i32]);
        let logits = self
            .model
            .forward_last_logits(&prompt_arr, &mut cache)?;
        let mut next_id = argmax_id(&logits)?;
        metrics.prefill_s = prefill_start.elapsed().as_secs_f64();

        // --- decode ---
        let decode_start = Instant::now();
        let mut produced: Vec<i32> = Vec::with_capacity(self.cfg.max_tokens);
        produced.push(next_id);
        metrics.total_tokens += 1;
        if eos.contains(&(next_id as u32)) {
            metrics.decode_s = decode_start.elapsed().as_secs_f64();
            return Ok((produced, metrics));
        }

        while metrics.total_tokens < self.cfg.max_tokens {
            if self.mtp_active {
                metrics.mtp_cycles += 1;
                let (extra, next, n_drafted, n_accepted) =
                    self.mtp_cycle(next_id, &mut cache)?;
                metrics.mtp_drafted += n_drafted;
                metrics.mtp_accepted += n_accepted;
                for tok in &extra {
                    produced.push(*tok);
                    metrics.total_tokens += 1;
                    if eos.contains(&(*tok as u32))
                        || metrics.total_tokens >= self.cfg.max_tokens
                    {
                        metrics.decode_s =
                            decode_start.elapsed().as_secs_f64();
                        return Ok((produced, metrics));
                    }
                }
                next_id = next;
                produced.push(next_id);
                metrics.total_tokens += 1;
                if eos.contains(&(next_id as u32)) {
                    break;
                }
            } else {
                // --- AR fallback ---
                metrics.ar_fallback_steps += 1;
                let in_arr = Array::from_slice(&[next_id], &[1, 1]);
                let logits =
                    self.model.forward_last_logits(&in_arr, &mut cache)?;
                next_id = argmax_id(&logits)?;
                produced.push(next_id);
                metrics.total_tokens += 1;
                if eos.contains(&(next_id as u32)) {
                    break;
                }
            }
        }

        metrics.decode_s = decode_start.elapsed().as_secs_f64();
        Ok((produced, metrics))
    }

    /// One MTP-drafted speculative cycle (K=1, greedy):
    ///
    ///   1. Run target forward on `last_token`; capture its post-norm
    ///      hidden state AND its argmax logit. The argmax is the
    ///      "would-have-been-AR" next token — we'll commit it as the
    ///      anchor for this cycle.
    ///   2. Embed `last_token` via the host embed table.
    ///   3. Call `mtp_head.forward(hidden, prev_emb)` → next-token
    ///      logits → argmax = drafted token (one step ahead of the
    ///      AR token).
    ///   4. Verify: run the target on the AR-committed token; its
    ///      argmax is the next "real" target token. If it matches
    ///      the drafted token, we accept the draft (saved one decode
    ///      step). Either way we return one or two committed tokens.
    ///
    /// Returns `(extra_accepted, next_after_accepted)` where
    /// `extra_accepted` is the list of speculatively-accepted tokens
    /// between the previous `next_id` and the new one. With K=1 this is
    /// at most one element.
    /// Multi-step MTP cycle.
    ///
    /// Returns (committed_tokens, next_anchor, n_drafted, n_accepted) where:
    /// * `committed_tokens` is the list of NEW tokens to push to `produced`
    ///   *excluding* the anchor for the next cycle (that's `next_anchor`).
    ///   For K=1 with acceptance, committed_tokens=[ar_next] and next_anchor=drafted.
    ///   For K=N with j accepts, committed_tokens=[ar_next, d_0..d_{j-1}] and
    ///   next_anchor = (j < K) correction : d_{K-1}.
    /// * `n_drafted` = K (regardless of acceptance).
    /// * `n_accepted` = number of drafts that matched target's argmax.
    fn mtp_cycle(
        &mut self,
        last_token: i32,
        cache: &mut Vec<HybridCache>,
    ) -> Result<(Vec<i32>, i32, usize, usize), MtpError> {
        let k = self.cfg.block_len.max(1);

        // Step 1: target forward on `last_token`, keep both the last
        // hidden state and the logits.
        let in_arr = Array::from_slice(&[last_token], &[1, 1]);
        let (hidden, ar_logits) = self
            .model
            .forward_last_hidden_and_logits(&in_arr, cache)?;
        let ar_next = argmax_id(&ar_logits)?;

        // Step 2: K MTP draft steps. Each step feeds the prior MTP block's
        // pre-norm hidden + embedding of the prior just-drafted token,
        // producing the next draft + its pre-norm hidden for the next
        // step. Mirrors llama.cpp PR #22673's draft-context loop.
        //
        // Step 0 anchors on the target's hidden + embed(ar_next).
        // Step i (i>=1) anchors on MTP's own pre-norm hidden + embed(d_{i-1}).
        let mut drafted: Vec<i32> = Vec::with_capacity(k);
        let mut mtp_pre_hidden = hidden.clone();
        let mut next_embed_token = ar_next;
        for _ in 0..k {
            let prev_emb = self.model.embed_tokens(&[next_embed_token])?;
            let (post, pre) = {
                let mtp = self
                    .model
                    .mtp_head()
                    .expect("mtp_cycle entered without an active MTP head");
                mtp.forward_with_pre_norm(&mtp_pre_hidden, &prev_emb)?
            };
            let mtp_logits_3d = self.model.apply_lm_head(&post)?;
            let mtp_logits = {
                let s = mtp_logits_3d.shape();
                if s.len() == 3 {
                    let v = s[s.len() - 1];
                    mtp_logits_3d.reshape(&[1, v])?
                } else {
                    mtp_logits_3d
                }
            };
            let d = argmax_id(&mtp_logits)?;
            drafted.push(d);
            mtp_pre_hidden = pre;
            next_embed_token = d;
        }

        // Step 3: target verify on K tokens: [ar_next, d_0, ..., d_{K-2}].
        // The K-th draft (d_{K-1}) doesn't need to be in the input — we
        // only verify K positions, each predicting the next slot.
        let mut verify_tokens: Vec<i32> = Vec::with_capacity(k);
        verify_tokens.push(ar_next);
        for d in drafted.iter().take(k - 1) {
            verify_tokens.push(*d);
        }
        let verify_in =
            Array::from_slice(&verify_tokens, &[1, verify_tokens.len() as i32]);
        let verify_key = GraphKey::new(
            "verify_forward",
            GraphKey::shape_sig_for(&[&verify_in]),
        );
        self.graph_bank.observe(&verify_key);
        // forward returns per-position logits [B, T, V].
        let verify_logits_full = self.model.forward(&verify_in, cache)?;

        // Walk: find first j where target's argmax at logit position j
        // does not match drafted[j]. Accept j drafts + 1 correction.
        let mut n_accepted: usize = 0;
        let mut correction: i32 = ar_next; // unused fallback
        for j in 0..k {
            let pos_logits = verify_logits_full.index((.., j as i32, ..));
            let target_at = argmax_id(&pos_logits)?;
            let d = drafted[j];
            let accept = target_at == d;
            if std::env::var("MTPLX_DEBUG_ACCEPT").is_ok() {
                eprintln!(
                    "[mtp-dbg] cycle (k={k}) j={j} drafted={d} target={target_at} {}",
                    if accept { "ACCEPT" } else { "REJECT" }
                );
            }
            if accept {
                n_accepted += 1;
            } else {
                correction = target_at;
                break;
            }
        }

        // Roll back caches by the rejected count.
        //  Target cache: verify appended K positions ([ar_next, d_0..d_{K-2}]).
        //  We keep n_accepted accepted-prefix slots; trim (K - n_accepted) - 0_if_full_accept.
        //    - if n_accepted == K (full accept): keep all K verify positions.
        //    - if n_accepted == j < K: keep ar_next + d_0..d_{j-1} = j+1 positions.
        //  Trim count = verify_len - kept = K - (n_accepted_corrected).
        let verify_len = verify_tokens.len() as i32;
        let target_keep: i32 = if n_accepted == k {
            verify_len
        } else {
            // ar_next + n_accepted drafts cached; correction not yet cached.
            (n_accepted as i32) + 1
        };
        let target_trim = verify_len - target_keep;
        if target_trim > 0 {
            for c in cache.iter_mut() {
                let _ = c.trim(target_trim);
            }
        }

        // MTP cache: appended k positions. Keep n_accepted positions; trim
        // (k - n_accepted). When n_accepted == k we keep all — the next
        // cycle's draft step 0 will pick up from MTP cache offset k.
        let mtp_trim = (k - n_accepted) as i32;
        if mtp_trim > 0 {
            if let Some(mtp) = self.model.mtp_head() {
                mtp.trim_cache(mtp_trim);
            }
        }

        // Assemble committed tokens.
        // - ar_next is always committed.
        // - n_accepted drafted tokens follow.
        // - If full accept: next_anchor = d_{K-1} (last drafted, kept).
        //   Otherwise: next_anchor = correction.
        let mut extra: Vec<i32> = Vec::with_capacity(n_accepted + 1);
        extra.push(ar_next);
        for j in 0..n_accepted.saturating_sub(if n_accepted == k { 1 } else { 0 }) {
            extra.push(drafted[j]);
        }
        let next_anchor = if n_accepted == k {
            // All accepted: last draft becomes the anchor; its KV is in
            // target cache so next cycle's step-1 forward will re-prefill
            // it (slight redundancy but simpler than threading it through).
            // Actually — we just kept it in target KV via verify's append
            // of [ar_next, d_0..d_{K-2}], so d_{K-1} is NOT in cache yet.
            // Use it as anchor; next cycle will append it.
            drafted[k - 1]
        } else {
            correction
        };

        Ok((extra, next_anchor, k, n_accepted))
    }

    /// Token-by-token streaming iterator. Emits one `i32` token per `next()`
    /// call. Internally each MTP cycle produces 1-2 tokens which are buffered;
    /// callers see a clean one-at-a-time stream regardless.
    ///
    /// Parameters mirror `SpeculativeConfig` but are passed explicitly so
    /// callers (e.g. OminiX-API) can derive them from the per-request fields
    /// without rebuilding the session. Only greedy acceptance (T=0) is
    /// supported; `max_tokens` overrides `cfg.max_tokens`.
    pub fn generate_iter(
        &mut self,
        prompt_ids: Vec<i32>,
        eos: HashSet<u32>,
        max_tokens: usize,
    ) -> impl Iterator<Item = Result<i32, MtpError>> + '_ {
        let mtp_active = self.mtp_active;
        let mut cache: Option<Vec<HybridCache>> = None;
        let mut initialized = false;
        let mut finished = max_tokens == 0;
        let mut emitted = 0usize;
        let mut pending: std::collections::VecDeque<i32> = std::collections::VecDeque::new();
        let mut last_token: i32 = 0;

        std::iter::from_fn(move || {
            loop {
                // Drain pending buffer first.
                if let Some(tok) = pending.pop_front() {
                    emitted += 1;
                    last_token = tok;
                    if eos.contains(&(tok as u32)) || emitted >= max_tokens {
                        finished = true;
                        pending.clear();
                    }
                    return Some(Ok(tok));
                }

                if finished {
                    return None;
                }

                // Prefill on first call.
                if !initialized {
                    if let Some(mtp) = self.model.mtp_head() {
                        mtp.reset_cache();
                    }
                    let mut c = self.model.new_cache(if std::env::var("TURBO_KV").is_ok() {
                qwen3_6_mlx::KVCacheMode::TurboQuant
            } else if std::env::var("QUANTIZE_KV").is_ok() {
                qwen3_6_mlx::KVCacheMode::Quantized
            } else {
                qwen3_6_mlx::KVCacheMode::Standard
            });
                    let prompt_arr =
                        Array::from_slice(&prompt_ids, &[1, prompt_ids.len() as i32]);
                    let logits = match self.model.forward_last_logits(&prompt_arr, &mut c) {
                        Ok(l) => l,
                        Err(e) => {
                            finished = true;
                            return Some(Err(e.into()));
                        }
                    };
                    let first = match argmax_id(&logits) {
                        Ok(id) => id,
                        Err(e) => {
                            finished = true;
                            return Some(Err(e));
                        }
                    };
                    cache = Some(c);
                    initialized = true;
                    pending.push_back(first);
                    continue;
                }

                // Decode: one MTP cycle or one AR step.
                let c = cache.as_mut().expect("cache initialized");
                if mtp_active {
                    match self.mtp_cycle(last_token, c) {
                        Ok((extra, next, _n_drafted, _n_accepted)) => {
                            for t in extra {
                                pending.push_back(t);
                            }
                            pending.push_back(next);
                        }
                        Err(e) => {
                            finished = true;
                            return Some(Err(e));
                        }
                    }
                } else {
                    let in_arr = Array::from_slice(&[last_token], &[1, 1]);
                    match self.model.forward_last_logits(&in_arr, c) {
                        Ok(logits) => match argmax_id(&logits) {
                            Ok(id) => pending.push_back(id),
                            Err(e) => {
                                finished = true;
                                return Some(Err(e));
                            }
                        },
                        Err(e) => {
                            finished = true;
                            return Some(Err(e.into()));
                        }
                    }
                }
            }
        })
    }
}

fn argmax_id(logits: &Array) -> Result<i32, MtpError> {
    use mlx_rs::ops::indexing::IndexOp;
    let ids = argmax_axis!(logits, -1)?;
    // ids: [B] (B=1) — grab scalar
    let scalar = ids.index(0);
    Ok(scalar.item::<u32>() as i32)
}
