use std::path::Path;

use mlx_rs::{
    error::Exception,
    module::Module,
    ops::{concatenate_axis, indexing::IndexOp},
    transforms::eval,
    Array,
};
use mlx_rs_core::utils::AttentionMask;
use qwen3_6_mlx::{
    cache::GdnRollbackSnapshot, model::AttentionLayer, HybridCache, KVCacheMode, Model,
};
use serde::Deserialize;

use crate::engine::spec_epoch::{DraftBlock, DraftModel, TargetModel};

/// Register the verify-qmm Metal kernel with qwen3.6-mlx so that
/// shape-eligible quantized projections (M=16, K%32==0, N%32==0, bits=4)
/// route through `verify_qmm_m16_mma2big` when `OMINIX_VERIFY_QMM=1` is set.
///
/// Called from every `Qwen36TargetAdapter` constructor. `OnceLock::set` is
/// idempotent after the first call.
fn install_verify_qmm_hook() {
    let _ = qwen3_6_mlx::verify_hook::QUANTIZED_VERIFY_QMM_HOOK
        .set(crate::verify_qmm::verify_qmm_dispatch);
}

#[derive(Debug, Clone, Deserialize)]
pub struct DraftCheckpointInfo {
    #[serde(default)]
    pub architectures: Vec<String>,
    #[serde(default)]
    pub block_size: usize,
    #[serde(default)]
    pub hidden_size: i32,
    #[serde(default)]
    pub num_hidden_layers: i32,
    #[serde(default)]
    pub vocab_size: i32,
}

impl DraftCheckpointInfo {
    pub fn load(
        model_dir: impl AsRef<Path>,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let config_path = model_dir.as_ref().join("config.json");
        let config = std::fs::read_to_string(config_path)?;
        Ok(serde_json::from_str(&config)?)
    }

    pub fn is_native_rust_supported(&self) -> bool {
        self.architectures
            .iter()
            .any(|arch| arch == "DFlashDraftModel")
    }
}

pub struct Qwen36TargetAdapter {
    model: Model,
    cache: Vec<HybridCache>,
    verify_snapshot: Option<Vec<HybridCache>>,
    verify_inputs: Option<Array>,
    verify_step: usize,
    step: usize,
    kv_cache_mode: KVCacheMode,
    _temp: f32,
    target_layer_ids: Vec<usize>,
    target_hidden_accumulated: Option<Array>,
    verify_hidden_snapshot: Option<Array>,
    /// Per-layer GDN snapshots captured during the most recent verify pass.
    /// `gdn_snapshots[i]` is `Some` iff layer `i` is a GDN (linear-attention)
    /// layer that recorded a tape; `None` for full-attention layers.
    /// Cleared by `prefill` and `rollback_kv`.
    gdn_snapshots: Vec<Option<GdnRollbackSnapshot>>,
    /// Captures emitted by the most recent verify pass, full block `[B, L, K*H]`.
    /// On rollback, we slice this to `n_keep` rows and append to the
    /// pre-verify accumulator — avoiding the desync where target KV holds
    /// `n_keep` more tokens than the hidden accumulator reflects.
    verify_captures: Option<Array>,
}

impl Qwen36TargetAdapter {
    pub fn new(model: Model, temp: f32) -> Self {
        install_verify_qmm_hook();
        let cache = model.new_cache(KVCacheMode::Standard);
        Self {
            model,
            cache,
            verify_snapshot: None,
            verify_inputs: None,
            verify_step: 0,
            step: 0,
            kv_cache_mode: KVCacheMode::Standard,
            _temp: temp,
            target_layer_ids: Vec::new(),
            target_hidden_accumulated: None,
            verify_hidden_snapshot: None,
            gdn_snapshots: Vec::new(),
            verify_captures: None,
        }
    }

    pub fn with_dflash(model: Model, temp: f32, target_layer_ids: Vec<usize>) -> Self {
        install_verify_qmm_hook();
        // Spike: opt into TurboQuant KV via TURBO_KV=1. The DFlash verify
        // path drives only L=1 decode steps when running outside of the
        // multi-token verify burst; multi-token verify (L > 1) falls
        // back to the standard SDPA path inside `GatedAttention`.
        let paged = matches!(
            std::env::var("OMINIX_PAGED_ATTENTION").ok().as_deref(),
            Some("1") | Some("true") | Some("TRUE")
        );
        let kv_cache_mode = if paged {
            // Paged full-attention KV (per-layer pools). Speculative rollback is
            // CoW-safe: `verify_snapshot.clone()` forks the block tables, so a
            // verify burst's appends copy-on-write rather than clobbering the
            // snapshot; `trim_gdn` → `PagedKvCache::trim_kv` rewinds the offset.
            KVCacheMode::Paged
        } else if std::env::var("TURBO_KV").is_ok() {
            KVCacheMode::TurboQuant
        } else {
            KVCacheMode::Standard
        };
        let cache = model.new_cache(kv_cache_mode);
        Self {
            model,
            cache,
            verify_snapshot: None,
            verify_inputs: None,
            verify_step: 0,
            step: 0,
            kv_cache_mode,
            _temp: temp,
            target_layer_ids,
            target_hidden_accumulated: None,
            verify_hidden_snapshot: None,
            gdn_snapshots: Vec::new(),
            verify_captures: None,
        }
    }

    pub fn new_quantized_kv(model: Model, temp: f32) -> Self {
        install_verify_qmm_hook();
        let cache = model.new_cache(KVCacheMode::Quantized);
        Self {
            model,
            cache,
            verify_snapshot: None,
            verify_inputs: None,
            verify_step: 0,
            step: 0,
            kv_cache_mode: KVCacheMode::Quantized,
            _temp: temp,
            target_layer_ids: Vec::new(),
            target_hidden_accumulated: None,
            verify_hidden_snapshot: None,
            gdn_snapshots: Vec::new(),
            verify_captures: None,
        }
    }

    fn forward_with_hidden_capture(&mut self, inputs: &Array) -> Result<(Array, Array), Exception> {
        if let Some(&layer_id) = self
            .target_layer_ids
            .iter()
            .find(|&&layer_id| layer_id >= self.model.text_model.layers.len())
        {
            return Err(Exception::custom(format!(
                "capture layer index {layer_id} out of range for {} layers",
                self.model.text_model.layers.len()
            )));
        }

        if self.cache.is_empty() {
            self.cache = self.model.new_cache(self.kv_cache_mode);
        }

        let mut h = self.model.text_model.embed_tokens.forward(inputs)?;
        let mask = if h.shape()[1] > 1 {
            Some(AttentionMask::Causal)
        } else {
            None
        };

        let mut captures = Vec::with_capacity(self.target_layer_ids.len());
        for (layer_idx, (layer, cache)) in self
            .model
            .text_model
            .layers
            .iter_mut()
            .zip(self.cache.iter_mut())
            .enumerate()
        {
            h = layer.forward(&h, mask.as_ref(), cache)?;
            if self.target_layer_ids.iter().any(|&id| id == layer_idx) {
                captures.push(h.clone());
            }
        }

        h = self.model.text_model.norm.forward(&h)?;
        let logits = self.model.apply_lm_head(&h)?;
        let capture_refs = captures.iter().collect::<Vec<_>>();
        let captures = concatenate_axis(&capture_refs, 2)?;
        Ok((logits, captures))
    }

    /// Verify-pass forward that captures both hidden states (for downstream
    /// dflash checks) and per-GDN-layer innovation tapes.
    ///
    /// For every linear-attention (GDN) layer with `L > 1`, snapshots the
    /// pre-verify recurrent + conv state and uses
    /// `GatedDeltaNet::forward_prefill_with_tape` so the resulting tape can
    /// later be replayed (via `HybridCache::trim_gdn`) to roll the state
    /// forward by the accepted prefix without a second forward pass.
    ///
    /// Full-attention layers are unchanged — their KV cache is trimmed in
    /// O(1) by `KVCache::trim`.
    fn forward_with_hidden_capture_tape(
        &mut self,
        inputs: &Array,
    ) -> Result<(Array, Array), Exception> {
        if let Some(&layer_id) = self
            .target_layer_ids
            .iter()
            .find(|&&layer_id| layer_id >= self.model.text_model.layers.len())
        {
            return Err(Exception::custom(format!(
                "capture layer index {layer_id} out of range for {} layers",
                self.model.text_model.layers.len()
            )));
        }

        if self.cache.is_empty() {
            self.cache = self.model.new_cache(self.kv_cache_mode);
        }

        let mut h = self.model.text_model.embed_tokens.forward(inputs)?;
        let mask = if h.shape()[1] > 1 {
            Some(AttentionMask::Causal)
        } else {
            None
        };

        let n_layers = self.model.text_model.layers.len();
        let mut gdn_snapshots: Vec<Option<GdnRollbackSnapshot>> =
            (0..n_layers).map(|_| None).collect();

        let target_layer_ids = self.target_layer_ids.clone();
        let mut captures = Vec::with_capacity(target_layer_ids.len());
        for (layer_idx, (layer, cache)) in self
            .model
            .text_model
            .layers
            .iter_mut()
            .zip(self.cache.iter_mut())
            .enumerate()
        {
            let is_gdn_prefill = matches!(layer.attention, AttentionLayer::LinearAttention(_))
                && h.shape()[1] > 1;

            if is_gdn_prefill {
                let rec = match cache {
                    HybridCache::Recurrent(r) => r,
                    _ => {
                        return Err(Exception::custom(
                            "linear-attention layer paired with non-recurrent cache",
                        ));
                    }
                };

                // Snapshot pre-verify recurrent state + conv state + step.
                // `state` is None only on the very first forward (no prior
                // prefill); we materialize a zero state in float32 so the
                // tape-replay path has something to start from.
                let snap_state = match rec.state.clone() {
                    Some(s) => s,
                    None => {
                        let delta = match &layer.attention {
                            AttentionLayer::LinearAttention(d) => d,
                            _ => unreachable!(),
                        };
                        mlx_rs::ops::zeros_dtype(
                            &[
                                h.shape()[0],
                                delta.num_v_heads,
                                delta.key_head_dim,
                                delta.value_head_dim,
                            ],
                            mlx_rs::Dtype::Float32,
                        )?
                    }
                };
                let snap_conv = rec.conv_state.clone();
                let snap_step = rec.step;

                let normed = layer.input_layernorm.forward(&h)?;
                let delta = match &mut layer.attention {
                    AttentionLayer::LinearAttention(d) => d,
                    _ => unreachable!(),
                };
                let (attn_out, capture) = delta.forward_prefill_with_tape(&normed, rec)?;
                let after_attn = h.add(&attn_out)?;
                let normed_h = layer.post_attention_layernorm.forward(&after_attn)?;
                let mlp_out = match &mut layer.ffn {
                    qwen3_6_mlx::model::FfnBlock::Moe(m) => m.forward(&normed_h)?,
                    qwen3_6_mlx::model::FfnBlock::Dense(d) => d.forward(&normed_h)?,
                };
                h = after_attn.add(mlp_out)?;

                gdn_snapshots[layer_idx] = Some(GdnRollbackSnapshot {
                    state: snap_state,
                    conv_state: snap_conv,
                    step: snap_step,
                    capture,
                });
            } else {
                h = layer.forward(&h, mask.as_ref(), cache)?;
            }

            if target_layer_ids.iter().any(|&id| id == layer_idx) {
                captures.push(h.clone());
            }
        }

        self.gdn_snapshots = gdn_snapshots;

        h = self.model.text_model.norm.forward(&h)?;
        let logits = self.model.apply_lm_head(&h)?;
        let captures = if captures.is_empty() {
            Array::zeros::<f32>(&[0])?
        } else {
            let capture_refs = captures.iter().collect::<Vec<_>>();
            concatenate_axis(&capture_refs, 2)?
        };
        Ok((logits, captures))
    }

    fn append_target_hidden(&mut self, captures: Array) -> Result<(), Exception> {
        self.target_hidden_accumulated = Some(match self.target_hidden_accumulated.take() {
            Some(existing) => concatenate_axis(&[&existing, &captures], 1)?,
            None => captures,
        });
        Ok(())
    }
}

impl TargetModel for Qwen36TargetAdapter {
    fn prefill(&mut self, prompt: &Array) -> Result<Array, Exception> {
        let PREFILL_CHUNK: i32 = std::env::var("QWEN36_PREFILL_CHUNK")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n: &i32| n > 0)
            .unwrap_or(64);

        self.cache = self.model.new_cache(self.kv_cache_mode);
        self.verify_snapshot = None;
        self.verify_inputs = None;
        self.verify_step = 0;
        self.step = 0;
        self.target_hidden_accumulated = None;
        self.verify_hidden_snapshot = None;
        self.gdn_snapshots.clear();

        let seq_len = prompt.shape()[1];
        let logits = if seq_len > PREFILL_CHUNK {
            if !self.target_layer_ids.is_empty() {
                let (logits, captures) = self.model.forward_last_logits_with_hidden_capture(
                    prompt,
                    &mut self.cache,
                    &self.target_layer_ids,
                )?;
                self.target_hidden_accumulated = Some(captures);
                logits
            } else {
                let mut pos = 0;
                let mut last_logits = None;
                while pos < seq_len {
                    let end = (pos + PREFILL_CHUNK).min(seq_len);
                    let chunk = prompt.index((.., pos..end));
                    let logits = self.model.forward_last_logits(&chunk, &mut self.cache)?;
                    // Same chunk-eval gating as qwen3.6 AR path:
                    //   QWEN36_SKIP_CHUNK_EVAL=1 — omit
                    //   QWEN36_ASYNC_PREFILL=1   — use async_eval
                    if std::env::var("QWEN36_SKIP_CHUNK_EVAL").is_err() {
                        if std::env::var("QWEN36_ASYNC_PREFILL").is_ok() {
                            mlx_rs::transforms::async_eval([&logits])?;
                        } else {
                            eval([&logits])?;
                        }
                    }
                    last_logits = Some(logits);
                    pos = end;
                }
                last_logits.expect("chunked prefill produced no logits")
            }
        } else if !self.target_layer_ids.is_empty() {
            let (logits, captures) = self.model.forward_last_logits_with_hidden_capture(
                prompt,
                &mut self.cache,
                &self.target_layer_ids,
            )?;
            self.target_hidden_accumulated = Some(captures);
            logits
        } else {
            self.model.forward_last_logits(prompt, &mut self.cache)?
        };

        self.step = seq_len as usize;
        Ok(logits)
    }

    fn verify(&mut self, drafted_tokens: &Array) -> Result<Array, Exception> {
        self.verify_snapshot = Some(self.cache.clone());
        self.verify_inputs = Some(drafted_tokens.clone());
        self.verify_step = self.step;
        self.verify_hidden_snapshot = self.target_hidden_accumulated.clone();

        // Scope-mark this forward as a verify pass so the verify_qmm Metal
        // kernel hook in qwen3.6-mlx can dispatch (and only then). Cleared
        // on Drop — AR/draft forwards remain untouched.
        let _verify_guard = qwen3_6_mlx::verify_hook::VerifyScope::enter();

        // Verify pass routes through the tape-capturing variant whenever the
        // drafted block is long enough to use the GDN prefill path (L > 1),
        // so that `rollback_kv` can replay accepted GDN steps from the
        // pre-verify snapshot instead of cloning + re-forwarding the cache.
        let logits = if drafted_tokens.shape()[1] > 1 {
            let (logits, captures) = self.forward_with_hidden_capture_tape(drafted_tokens)?;
            if !self.target_layer_ids.is_empty() {
                // Save the full-block captures BEFORE appending, so the
                // rollback path can slice to n_keep rows (the append below
                // assumes full acceptance — rollback corrects).
                self.verify_captures = Some(captures.clone());
                self.append_target_hidden(captures)?;
            }
            logits
        } else if !self.target_layer_ids.is_empty() {
            let (logits, captures) = self.forward_with_hidden_capture(drafted_tokens)?;
            self.append_target_hidden(captures)?;
            logits
        } else {
            self.model.forward(drafted_tokens, &mut self.cache)?
        };
        self.step += drafted_tokens.shape()[1] as usize;
        Ok(logits)
    }

    fn step_count(&self) -> usize {
        self.step
    }

    fn rollback_kv(&mut self, n_keep: usize) -> Result<(), Exception> {
        let verify_inputs = self
            .verify_inputs
            .as_ref()
            .ok_or_else(|| Exception::custom("target rollback requested before verify"))?
            .clone();
        let verify_len = verify_inputs.shape()[1] as usize;
        if n_keep > verify_len {
            return Err(Exception::custom(format!(
                "target rollback requested {n_keep} kept tokens, but only {verify_len} verify tokens are available"
            )));
        }

        let n_drop = (verify_len - n_keep) as i32;
        let has_gdn_snapshots = self.gdn_snapshots.iter().any(|s| s.is_some());

        if has_gdn_snapshots {
            // Deterministic per-layer rollback: FA layers trim KV by `n_drop`,
            // GDN layers replay the first `n_keep` recorded tape steps from
            // the snapshotted state.  Avoids cloning the cache and
            // re-forwarding the kept prefix.
            for (layer_idx, cache) in self.cache.iter_mut().enumerate() {
                cache.trim_gdn(
                    n_drop,
                    verify_len as i32,
                    self.gdn_snapshots[layer_idx].as_ref(),
                )?;
            }
            self.step = self.verify_step + n_keep;
            // Rebuild target_hidden_accumulated = pre_verify ++ verify_captures[:n_keep].
            // The bug being fixed: previously this restored to pre-verify only,
            // leaving target KV with `n_keep` more committed positions than the
            // hidden accumulator reflected → next-cycle draft sees stale context
            // → catastrophic acceptance collapse (~0.014 in benches).
            self.target_hidden_accumulated = self.verify_hidden_snapshot.clone();
            if n_keep > 0 && !self.target_layer_ids.is_empty() {
                if let Some(full_caps) = self.verify_captures.as_ref() {
                    let kept_caps = full_caps.index((.., ..n_keep as i32, ..));
                    self.append_target_hidden(kept_caps)?;
                }
            }
        } else {
            self.cache = self.verify_snapshot.clone().ok_or_else(|| {
                Exception::custom("target rollback requested without a verify snapshot")
            })?;
            self.step = self.verify_step;
            self.target_hidden_accumulated = self.verify_hidden_snapshot.clone();

            if n_keep > 0 {
                let kept = verify_inputs.index((.., ..n_keep as i32));
                if !self.target_layer_ids.is_empty() {
                    let (_, captures) = self.forward_with_hidden_capture(&kept)?;
                    self.append_target_hidden(captures)?;
                } else {
                    let _ = self.model.forward_last_logits(&kept, &mut self.cache)?;
                }
                self.step += n_keep;
            }
        }

        self.verify_snapshot = None;
        self.verify_inputs = None;
        self.verify_hidden_snapshot = None;
        self.verify_captures = None;
        self.gdn_snapshots.clear();
        Ok(())
    }

    fn sample(&self, logits: &Array, temp: f32) -> Result<Array, Exception> {
        qwen3_6_mlx::sample(logits, temp)
    }

    fn last_target_hidden(&self) -> Option<Array> {
        self.target_hidden_accumulated.clone()
    }

    fn embed_token(&mut self, id: u32) -> Option<Array> {
        self.model.embed_tokens(&[id as i32]).ok()
    }
}

#[derive(Debug, Clone)]
pub struct MockDraftAdapter {
    vocab_size: u32,
    step: usize,
    last_block_len: usize,
}

impl MockDraftAdapter {
    pub fn new(vocab_size: u32) -> Self {
        Self {
            vocab_size: vocab_size.max(1),
            step: 0,
            last_block_len: 0,
        }
    }
}

impl DraftModel for MockDraftAdapter {
    fn prefill(&mut self, prompt: &Array) -> Result<Array, Exception> {
        self.step = prompt.shape()[1] as usize;
        self.last_block_len = 0;
        Array::zeros::<f32>(&[1, self.vocab_size as i32])
    }

    fn draft_block(
        &mut self,
        last_token: &Array,
        block_len: usize,
    ) -> Result<DraftBlock, Exception> {
        let seed = last_token.index((0, 0)).item::<u32>();
        let mut current = seed;
        let mut tokens = Vec::with_capacity(block_len);
        let stride = (self.step as u32 % 17) + 1;
        for _ in 0..block_len {
            current = current.wrapping_add(stride) % self.vocab_size;
            tokens.push(current);
        }

        self.step += block_len;
        self.last_block_len = block_len;

        Ok(DraftBlock {
            tokens: Array::from_slice(&tokens, &[1, block_len as i32]),
            logits: Array::zeros::<f32>(&[1, block_len as i32, self.vocab_size as i32])?,
        })
    }

    fn rollback(&mut self, n_accepted: usize) -> Result<(), Exception> {
        let rejected = self.last_block_len.saturating_sub(n_accepted);
        self.step = self.step.saturating_sub(rejected);
        self.last_block_len = n_accepted;
        Ok(())
    }
}
