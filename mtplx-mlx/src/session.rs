//! Minimal MTPLX speculative decoding session.
//!
//! Cycle (greedy, T=0):
//!   1. Prefill the target on the prompt (uses
//!      `qwen3_6_mlx::Model::forward_last_logits`).
//!   2. Per step:
//!      a. If the MTP head is loaded, draft K tokens from the last
//!         hidden state + previous-token embedding.
//!         (Stub today: see below — we never enter this branch in
//!          practice because the stock checkpoint strips MTP weights.)
//!      b. Else: emit a single autoregressive token (AR fallback).
//!   3. Target-verify the K candidates in one forward; accept while
//!      `draft[i] == argmax(target_logits[i])`.
//!
//! The target's `forward_last_logits` only returns the last-position
//! logits, which is exactly what we want for the trailing argmax in the
//! verify pass. For K > 1 candidate verification we'd want all positions
//! — see TODO in `verify_block`.

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
    has_mtp: bool,
}

impl MtplxSession {
    /// Build a session around the given (already-loaded) Qwen3.6 target.
    pub fn new(mut model: Model, cfg: SpeculativeConfig) -> Self {
        let has_mtp = model.mtp_head().is_some();
        Self {
            model,
            cfg,
            has_mtp,
        }
    }

    pub fn has_mtp_head(&self) -> bool {
        self.has_mtp
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
            if self.has_mtp {
                // TODO: when MTP weights are actually loaded, draft K
                // tokens here via `self.model.mtp_head().unwrap().forward`
                // and verify them in a single target pass. We never
                // reach this branch with the stock checkpoint.
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

    /// One MTP-drafted speculative cycle: draft K with the MTP head,
    /// verify with the target, accept greedy prefix.
    ///
    /// **Not implemented**: the stock Qwen3.6 checkpoint strips MTP
    /// weights, so we never construct a real head. This stub returns
    /// a single AR-decoded token so the outer loop still makes progress
    /// if it's ever entered.
    fn mtp_cycle(
        &mut self,
        last_token: i32,
        cache: &mut Vec<HybridCache>,
    ) -> Result<(Vec<i32>, i32), MtpError> {
        // TODO(mtplx): replace with real draft+verify once a checkpoint
        // ships the MTP weights. See WIP.md.
        let in_arr = Array::from_slice(&[last_token], &[1, 1]);
        let logits = self.model.forward_last_logits(&in_arr, cache)?;
        let next = argmax_id(&logits)?;
        Ok((Vec::new(), next))
    }
}

fn argmax_id(logits: &Array) -> Result<i32, MtpError> {
    use mlx_rs::ops::indexing::IndexOp;
    let ids = argmax_axis!(logits, -1)?;
    // ids: [B] (B=1) — grab scalar
    let scalar = ids.index(0);
    Ok(scalar.item::<u32>() as i32)
}
