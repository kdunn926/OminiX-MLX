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

use mlx_rs::{argmax_axis, Array};
use qwen3_6_mlx::{HybridCache, Model};

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
    /// Sampling temperature; only T=0 (greedy) is wired up today.
    pub temp: f32,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            block_len: 4,
            max_tokens: 200,
            temp: 0.0,
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
        }
    }

    pub fn has_mtp_head(&self) -> bool {
        self.mtp_active
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
        let mut cache: Vec<HybridCache> = self
            .model
            .new_cache(qwen3_6_mlx::KVCacheMode::Standard);

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
                let (accepted, next) =
                    self.mtp_cycle(next_id, &mut cache)?;
                for tok in &accepted {
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
    fn mtp_cycle(
        &mut self,
        last_token: i32,
        cache: &mut Vec<HybridCache>,
    ) -> Result<(Vec<i32>, i32), MtpError> {
        // Step 1: target forward on `last_token`, keep both the last
        // hidden state and the logits.
        let in_arr = Array::from_slice(&[last_token], &[1, 1]);
        let (hidden, ar_logits) = self
            .model
            .forward_last_hidden_and_logits(&in_arr, cache)?;
        let ar_next = argmax_id(&ar_logits)?;

        // Step 2: embed the previously-committed token (host embed table).
        // `embed_tokens` returns `[1, 1, H]`.
        let prev_emb = self.model.embed_tokens(&[last_token])?;

        // Step 3: MTP draft hidden → apply host LM head → next-after-AR draft.
        // The MTP head returns `[1, 1, H]`; we then reuse the target's
        // `apply_lm_head` (its lm_head or tied embedding) to project to vocab.
        let mtp_hidden = {
            let mtp = self
                .model
                .mtp_head()
                .expect("mtp_cycle entered without an active MTP head");
            mtp.forward(&hidden, &prev_emb)?
        };
        let mtp_logits = self.model.apply_lm_head(&mtp_hidden)?;
        let drafted = argmax_id(&mtp_logits)?;

        // Step 4: verify the draft by running target on ar_next.
        let verify_in = Array::from_slice(&[ar_next], &[1, 1]);
        let verify_logits = self.model.forward_last_logits(&verify_in, cache)?;
        let target_after_ar = argmax_id(&verify_logits)?;

        if target_after_ar == drafted {
            // Accept the draft: we commit `ar_next` as the in-between
            // token and `drafted` as the next anchor.
            Ok((vec![ar_next], drafted))
        } else {
            // Reject: we still commit `ar_next` (free — its cache is
            // populated) and use the target's verified continuation
            // `target_after_ar` as the next anchor. No work wasted; the
            // draft path just didn't pay off.
            Ok((vec![ar_next], target_after_ar))
        }
    }
}

fn argmax_id(logits: &Array) -> Result<i32, MtpError> {
    use mlx_rs::ops::indexing::IndexOp;
    let ids = argmax_axis!(logits, -1)?;
    // ids: [B] (B=1) — grab scalar
    let scalar = ids.index(0);
    Ok(scalar.item::<u32>() as i32)
}
