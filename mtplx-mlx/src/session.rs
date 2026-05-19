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
                metrics.mtp_drafted += 1; // K=1 per cycle
                let (accepted, next, draft_accepted) =
                    self.mtp_cycle(next_id, &mut cache)?;
                if draft_accepted {
                    metrics.mtp_accepted += 1;
                }
                let _ = &accepted;
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
    ) -> Result<(Vec<i32>, i32, bool), MtpError> {
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
        let mtp_logits_3d = self.model.apply_lm_head(&mtp_hidden)?;
        // Squeeze [1,1,V] → [1,V] for parity with verify_logits.
        let mtp_logits = {
            let s = mtp_logits_3d.shape();
            if s.len() == 3 {
                let v = s[s.len() - 1];
                mtp_logits_3d.reshape(&[1, v])?
            } else {
                mtp_logits_3d
            }
        };
        let drafted = argmax_id(&mtp_logits)?;

        // Step 4: verify the draft by running target on ar_next.
        let verify_in = Array::from_slice(&[ar_next], &[1, 1]);
        // Record a hit/miss against the graph cache keyed on the verify
        // input's shape. This is observation-only today: real dispatch
        // through `GraphBank::invoke` requires capturing `&mut self.model`
        // and `&mut cache` in a `'static + Send` closure, which doesn't
        // type-check cleanly without rearchitecting the forward path
        // (likely behind an `Arc<Mutex<...>>` or by exposing a pure
        // `&[Array] -> Vec<Array>` entry point from `qwen3_6_mlx::Model`).
        // TODO(graph-bank): wire the actual compiled forward here.
        let verify_key = GraphKey::new(
            "verify_forward",
            GraphKey::shape_sig_for(&[&verify_in]),
        );
        self.graph_bank.observe(&verify_key);
        let verify_logits = self.model.forward_last_logits(&verify_in, cache)?;

        // Route through the configured acceptance strategy. Both modes
        // commit `ar_next` for free (cache populated, argmax confirmed)
        // and then decide what to anchor the *next* cycle on.
        //
        // K=1 semantics:
        //   * Accept: commit `drafted` as an extra token, next_anchor =
        //     target's verified continuation (greedy: argmax; spec: free
        //     bonus sampled from target distribution).
        //   * Reject: drop the draft, next_anchor = correction (greedy:
        //     target's argmax; spec: residual `(p-q)+` sample).
        let (extra, next_anchor, accepted_flag) = match self.cfg.acceptance {
            AcceptanceMode::Greedy => {
                let target_after_ar = argmax_id(&verify_logits)?;
                if target_after_ar == drafted {
                    (vec![ar_next], drafted, true)
                } else {
                    if let Some(mtp) = self.model.mtp_head() {
                        mtp.trim_cache(1);
                    }
                    (vec![ar_next], target_after_ar, false)
                }
            }
            AcceptanceMode::Speculative => {
                let temp = self.cfg.temp.max(1e-4);
                let res = accept_speculative(
                    &verify_logits,
                    &mtp_logits,
                    &[drafted as u32],
                    temp,
                    default_rng,
                )?;
                if res.all_accepted {
                    let _bonus = res.correction;
                    (vec![ar_next], drafted, true)
                } else {
                    if let Some(mtp) = self.model.mtp_head() {
                        mtp.trim_cache(1);
                    }
                    (vec![ar_next], res.correction as i32, false)
                }
            }
        };

        Ok((extra, next_anchor, accepted_flag))
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
                        Ok((extra, next, _accepted)) => {
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
