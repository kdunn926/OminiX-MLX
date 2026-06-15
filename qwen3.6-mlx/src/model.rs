use std::collections::{HashMap, HashSet};
use std::path::Path;

use mlx_rs::{
    error::Exception,
    module::{Module, ModuleParameters, Param},
    nn,
    ops::{concatenate_axis, dequantize, indexing::IndexOp},
    quantization::MaybeQuantized,
    Array, Dtype,
};

use mlx_rs_core::{
    cache::{KVCache, QuantizedKVCache, TurboQuantKVCache},
    paged::PagedKvCache,
    error::Error,
    utils::initialize_rope,
};

use crate::attention::{GatedAttention, GatedAttentionInput};
use crate::cache::{GdnRollbackSnapshot, HybridCache, RecurrentState};
use crate::config::{ModelArgs, TextConfig};
use crate::deltanet::GatedDeltaNet;
use crate::moe::{DenseMlp, MoeBlock, QuantizedSwitchLinear, SharedExpert, SwitchGLU};
use crate::mtp::{load_mtp_head, MtpHead};
use crate::vision::VisionTower;

// ============================================================================
// Layer Types
// ============================================================================

pub enum AttentionLayer {
    FullAttention(GatedAttention),
    LinearAttention(GatedDeltaNet),
}

/// FFN for a transformer block — either MoE (35B) or dense MLP (27B).
pub enum FfnBlock {
    Moe(MoeBlock),
    Dense(DenseMlp),
}

pub struct TransformerBlock {
    pub attention: AttentionLayer,
    pub ffn: FfnBlock,
    pub input_layernorm: nn::RmsNorm,
    pub post_attention_layernorm: nn::RmsNorm,
}

impl TransformerBlock {
    #[allow(non_snake_case)]
    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&mlx_rs_core::utils::AttentionMask>,
        cache: &mut HybridCache,
    ) -> Result<Array, Exception> {
        let normed = self.input_layernorm.forward(x)?;

        let attn_out = match (&mut self.attention, cache) {
            (AttentionLayer::FullAttention(attn), HybridCache::KV(kv_cache)) => {
                attn.forward(GatedAttentionInput {
                    x: &normed,
                    mask,
                    cache: Some(kv_cache),
                })?
            }
            (AttentionLayer::FullAttention(attn), HybridCache::QuantizedKV(qkv_cache)) => attn
                .forward(GatedAttentionInput {
                    x: &normed,
                    mask,
                    cache: Some(qkv_cache),
                })?,
            (AttentionLayer::FullAttention(attn), HybridCache::TurboQuantKV(tq_cache)) => attn
                .forward(GatedAttentionInput {
                    x: &normed,
                    mask,
                    cache: Some(tq_cache),
                })?,
            (AttentionLayer::FullAttention(attn), HybridCache::Paged(paged_cache)) => attn
                .forward(GatedAttentionInput {
                    x: &normed,
                    mask,
                    cache: Some(paged_cache),
                })?,
            (AttentionLayer::FullAttention(attn), HybridCache::KvFlash(kf_cache)) => attn
                .forward(GatedAttentionInput {
                    x: &normed,
                    mask,
                    cache: Some(kf_cache),
                })?,
            (AttentionLayer::LinearAttention(delta), HybridCache::Recurrent(rec_cache)) => {
                let L = normed.shape()[1];
                if L > 1 {
                    delta.forward_prefill(&normed, rec_cache)?
                } else {
                    delta.forward_step(&normed, rec_cache)?
                }
            }
            _ => return Err(Exception::custom("Cache type mismatch with layer type")),
        };

        self.ffn_tail(x, attn_out)
    }

    /// Like [`forward`], but for recurrent (GDN) layers in a multi-token
    /// prefill (`L > 1`) it records a [`GdnRollbackSnapshot`] so a speculative
    /// verify pass can be rolled back via [`HybridCache::trim_gdn`]. Returns
    /// `None` for full-attention layers and for single-token steps (where the
    /// plain KV `trim` already rolls back correctly and no GDN tape is needed).
    ///
    /// Kept separate from [`forward`] so the hot path never pays the tape-
    /// recording cost (`deltanet_with_tape` vs the plain recurrence).
    #[allow(non_snake_case)]
    pub fn forward_capture_gdn(
        &mut self,
        x: &Array,
        mask: Option<&mlx_rs_core::utils::AttentionMask>,
        cache: &mut HybridCache,
    ) -> Result<(Array, Option<GdnRollbackSnapshot>), Exception> {
        let normed = self.input_layernorm.forward(x)?;

        let mut snapshot: Option<GdnRollbackSnapshot> = None;
        let attn_out = match (&mut self.attention, cache) {
            (AttentionLayer::FullAttention(attn), HybridCache::KV(kv_cache)) => {
                attn.forward(GatedAttentionInput {
                    x: &normed,
                    mask,
                    cache: Some(kv_cache),
                })?
            }
            (AttentionLayer::FullAttention(attn), HybridCache::QuantizedKV(qkv_cache)) => attn
                .forward(GatedAttentionInput {
                    x: &normed,
                    mask,
                    cache: Some(qkv_cache),
                })?,
            (AttentionLayer::FullAttention(attn), HybridCache::TurboQuantKV(tq_cache)) => attn
                .forward(GatedAttentionInput {
                    x: &normed,
                    mask,
                    cache: Some(tq_cache),
                })?,
            (AttentionLayer::FullAttention(attn), HybridCache::Paged(paged_cache)) => attn
                .forward(GatedAttentionInput {
                    x: &normed,
                    mask,
                    cache: Some(paged_cache),
                })?,
            (AttentionLayer::FullAttention(attn), HybridCache::KvFlash(kf_cache)) => attn
                .forward(GatedAttentionInput {
                    x: &normed,
                    mask,
                    cache: Some(kf_cache),
                })?,
            (AttentionLayer::LinearAttention(delta), HybridCache::Recurrent(rec_cache)) => {
                let L = normed.shape()[1];
                if L > 1 {
                    // Snapshot the recurrent state BEFORE the tape forward
                    // mutates it (it `take`s cache.state and updates step).
                    let B = normed.shape()[0];
                    let pre_state = match &rec_cache.state {
                        Some(s) => s.clone(),
                        None => mlx_rs::ops::zeros_dtype(
                            &[B, delta.num_v_heads, delta.key_head_dim, delta.value_head_dim],
                            Dtype::Float32,
                        )?,
                    };
                    let pre_conv_state = rec_cache.conv_state.clone();
                    let pre_step = rec_cache.step;
                    let (out, capture) = delta.forward_prefill_with_tape(&normed, rec_cache)?;
                    snapshot = Some(GdnRollbackSnapshot {
                        state: pre_state,
                        conv_state: pre_conv_state,
                        step: pre_step,
                        capture,
                    });
                    out
                } else {
                    delta.forward_step(&normed, rec_cache)?
                }
            }
            _ => return Err(Exception::custom("Cache type mismatch with layer type")),
        };

        Ok((self.ffn_tail(x, attn_out)?, snapshot))
    }

    /// Residual + post-attention norm + FFN, shared by [`forward`] and
    /// [`forward_capture_gdn`].
    fn ffn_tail(&mut self, x: &Array, attn_out: Array) -> Result<Array, Exception> {
        let h = x.add(attn_out)?;
        let normed_h = self.post_attention_layernorm.forward(&h)?;
        let mlp_out = match &mut self.ffn {
            FfnBlock::Moe(m) => m.forward(&normed_h)?,
            FfnBlock::Dense(d) => d.forward(&normed_h)?,
        };
        h.add(mlp_out)
    }
}

// ============================================================================
// Full Model
// ============================================================================

/// Selects which KV cache implementation to use for full-attention layers.
#[derive(Debug, Clone, Copy, Default)]
pub enum KVCacheMode {
    /// Standard fp16 KV cache (default).
    #[default]
    Standard,
    /// Mixed-precision cache: K=q8, V=q4.
    Quantized,
    /// Spike: TurboQuant 4-bit K + 8-bit V cache with fused SDPA via
    /// `KeyValueCache::try_fused_attention` (online softmax, V-tile cache).
    TurboQuant,
    /// Paged KV for full-attention layers, drawn from the thread-default
    /// `PagedKvPool` (behind `OMINIX_PAGED_ATTENTION`). Decode runs through
    /// the fused paged-attention kernel; recurrent layers are unaffected.
    Paged,
    /// Spike: KVFlash bounded-residency cache for full-attention layers
    /// (`DFLASH_KVFLASH=<pool>`, `DFLASH_KVFLASH_SINK`). Decode attends a
    /// `<= pool` resident set (sink + recent), so throughput stays flat as
    /// context grows. Lossy (drops cold chunks); no spec-decode rollback.
    KvFlash,
}

pub struct Qwen36TextModel {
    pub embed_tokens: MaybeQuantized<nn::Embedding>,
    pub layers: Vec<TransformerBlock>,
    pub norm: nn::RmsNorm,
    pub layer_types: Vec<String>,
}

impl Qwen36TextModel {
    /// Lazily populate an empty cache vector with the per-layer default
    /// (standard `KVCache` for full-attention layers, `RecurrentState` for
    /// GDN layers). NOTE: this is the STANDARD (fp16) cache mode — callers
    /// who want quantized/paged/TurboQuant KV must pre-build the vector via
    /// `Model::new_cache(mode)`; passing an empty Vec silently selects
    /// standard mode.
    pub fn ensure_cache(&self, cache: &mut Vec<HybridCache>) {
        if cache.is_empty() {
            for layer_type in &self.layer_types {
                if layer_type == "full_attention" {
                    cache.push(HybridCache::KV(KVCache::new()));
                } else {
                    cache.push(HybridCache::Recurrent(RecurrentState::new()));
                }
            }
        }
    }
}

pub struct Model {
    pub args: ModelArgs,
    pub text_model: Qwen36TextModel,
    pub lm_head: Option<MaybeQuantized<nn::Linear>>,
    /// Optional Multi-Token Prediction head. `None` when the checkpoint
    /// strips MTP weights (the common case for `mlx-community` releases).
    pub mtp_head: Option<MtpHead>,
}

impl Model {
    /// Returns the MTP head if one was loaded from the checkpoint.
    /// Returns `None` when MTP weights were absent (the common case for
    /// stock `mlx-community` releases, which strip `mtp.*` keys).
    pub fn mtp_head(&mut self) -> Option<&mut MtpHead> {
        self.mtp_head.as_mut()
    }

    /// Allocate a fresh cache vector appropriate for this model.
    ///
    /// - `mode = Standard`   → full-attention layers get `KVCache`
    /// - `mode = Quantized`  → full-attention layers get `QuantizedKVCache` (K=q8, V=q4)
    pub fn new_cache(&self, mode: KVCacheMode) -> Vec<HybridCache> {
        self.text_model
            .layer_types
            .iter()
            .map(|lt| {
                if lt == "full_attention" {
                    match mode {
                        KVCacheMode::Standard => HybridCache::KV(KVCache::new()),
                        KVCacheMode::Quantized => {
                            // Allow overriding k_bits / v_bits / group_size via env
                            // so callers can match llama.cpp configs like
                            // `--cache-type-k q4_0 --cache-type-v q4_0` (k=4, v=4,
                            // group_size=32). Defaults stay at K=q8 V=q4 gs=64.
                            let k_bits: i32 = std::env::var("KV_K_BITS")
                                .ok()
                                .and_then(|v| v.parse().ok())
                                .unwrap_or(8);
                            let v_bits: i32 = std::env::var("KV_V_BITS")
                                .ok()
                                .and_then(|v| v.parse().ok())
                                .unwrap_or(4);
                            let group_size: i32 = std::env::var("KV_GROUP_SIZE")
                                .ok()
                                .and_then(|v| v.parse().ok())
                                .unwrap_or(64);
                            HybridCache::QuantizedKV(QuantizedKVCache::new(
                                group_size, k_bits, v_bits, 256,
                            ))
                        }
                        KVCacheMode::TurboQuant => {
                            HybridCache::TurboQuantKV(TurboQuantKVCache::new())
                        }
                        // Draws from the thread-default PagedKvPool installed by
                        // the engine; one shared arena backs all full-attn layers.
                        KVCacheMode::Paged => HybridCache::Paged(PagedKvCache::default()),
                        KVCacheMode::KvFlash => {
                            let pool: i32 = std::env::var("DFLASH_KVFLASH")
                                .ok()
                                .and_then(|v| v.parse().ok())
                                .filter(|&n| n > 0)
                                .unwrap_or(4096);
                            let sink: i32 = std::env::var("DFLASH_KVFLASH_SINK")
                                .ok()
                                .and_then(|v| v.parse().ok())
                                .unwrap_or(mlx_rs_core::kvflash::DEFAULT_SINK);
                            // Bound prefill too (memory bound + O(seq·pool)
                            // attention) when DFLASH_KVFLASH_PREFILL=1. Default
                            // off = decode-only bounding (full prefill KV).
                            let bound_prefill = matches!(
                                std::env::var("DFLASH_KVFLASH_PREFILL").ok().as_deref(),
                                Some("1") | Some("true")
                            );
                            HybridCache::KvFlash(mlx_rs_core::kvflash::KvFlashCache::new(
                                pool,
                                sink,
                                mlx_rs_core::kvflash::DEFAULT_CHUNK,
                                bound_prefill,
                            ))
                        }
                    }
                } else {
                    HybridCache::Recurrent(RecurrentState::new())
                }
            })
            .collect()
    }

    /// Like `forward_last_logits` but also returns the last-position
    /// post-norm hidden state. Used by MTP speculative drafting in
    /// `mtplx-mlx`: the MTP head consumes `[B, 1, H]` of host hidden +
    /// the embedded previously-committed token.
    pub fn forward_last_hidden_and_logits(
        &mut self,
        inputs: &Array,
        cache: &mut Vec<HybridCache>,
    ) -> Result<(Array, Array), Exception> {
        // MTP draft heads are trained on the PRE-norm hidden state
        // (matches llama.cpp PR #22673 `llama_get_embeddings_pre_norm_ith`).
        // Passing post-norm hidden shifts the input distribution and
        // tanks draft acceptance — empirically observed 0.17-0.36 on
        // hermes with post-norm vs llama.cpp's 0.72-0.83 with pre-norm.
        let h_pre = self.forward_pre_norm_hidden(inputs, cache)?;
        let last_pre = h_pre.index((.., -1, ..));
        let last_pre_3d = last_pre.index((.., mlx_rs::ops::indexing::NewAxis, ..));
        // LM head consumes post-norm.
        let h_post = self.text_model.norm.forward(&h_pre)?;
        let last_post = h_post.index((.., -1, ..));
        let logits = self.apply_lm_head(&last_post)?;
        Ok((last_pre_3d, logits))
    }

    /// Run the transformer and project only the last sequence position through
    /// the LM head. Avoids a `[B, T, vocab]` matmul during prefill.
    pub fn forward_last_logits(
        &mut self,
        inputs: &Array,
        cache: &mut Vec<HybridCache>,
    ) -> Result<Array, Exception> {
        let h = self.forward_hidden(inputs, cache)?;
        let last = h.index((.., -1, ..));
        self.apply_lm_head(&last)
    }

    #[allow(non_snake_case)]
    pub fn forward(
        &mut self,
        inputs: &Array,
        cache: &mut Vec<HybridCache>,
    ) -> Result<Array, Exception> {
        let h = self.forward_hidden(inputs, cache)?;
        self.apply_lm_head(&h)
    }

    /// Verify forward for speculative decoding. Identical to [`forward`]
    /// (per-position logits `[B, T, V]`) but additionally returns, per layer, a
    /// [`GdnRollbackSnapshot`] for recurrent (GDN) layers so a rejected draft
    /// suffix can be undone with [`HybridCache::trim_gdn`]. Without this, GDN
    /// layers' recurrent state advances during verify and `HybridCache::trim`
    /// (a no-op for recurrent slots) leaves them desynced from the trimmed KV
    /// layers — silently corrupting every subsequent token.
    ///
    /// The returned vector is aligned with `cache` (one entry per layer); entries
    /// are `None` for full-attention layers and for `T == 1`.
    #[allow(non_snake_case)]
    pub fn forward_verify_with_snapshots(
        &mut self,
        inputs: &Array,
        cache: &mut Vec<HybridCache>,
    ) -> Result<(Array, Vec<Option<GdnRollbackSnapshot>>), Exception> {
        let mut h = self.text_model.embed_tokens.forward(inputs)?;
        let seq_len = h.shape()[1];
        let mask = if seq_len > 1 {
            Some(mlx_rs_core::utils::AttentionMask::Causal)
        } else {
            None
        };

        self.text_model.ensure_cache(cache);

        let mut snapshots = Vec::with_capacity(self.text_model.layers.len());
        for (layer, c) in self.text_model.layers.iter_mut().zip(cache.iter_mut()) {
            let (out, snap) = layer.forward_capture_gdn(&h, mask.as_ref(), c)?;
            h = out;
            snapshots.push(snap);
        }

        h = self.text_model.norm.forward(&h)?;
        let logits = self.apply_lm_head(&h)?;
        Ok((logits, snapshots))
    }

    /// Forward pass starting from pre-built embeddings instead of token IDs.
    /// Used for multimodal generation where visual features are spliced in.
    pub fn forward_from_embeds(
        &mut self,
        embeddings: &Array,
        cache: &mut Vec<HybridCache>,
    ) -> Result<Array, Exception> {
        let mut h = embeddings.clone();
        let t = h.shape()[1];
        let mask = if t > 1 {
            Some(mlx_rs_core::utils::AttentionMask::Causal)
        } else {
            None
        };
        self.text_model.ensure_cache(cache);
        for (block, c) in self.text_model.layers.iter_mut().zip(cache.iter_mut()) {
            h = block.forward(&h, mask.as_ref(), c)?;
        }
        h = self.text_model.norm.forward(&h)?;
        let last = h.index((.., -1i32, ..));
        self.apply_lm_head(&last)
    }

    /// Embed a slice of token IDs.
    pub fn embed_tokens(&mut self, ids: &[i32]) -> Result<Array, Exception> {
        let arr = Array::from_slice(ids, &[1, ids.len() as i32]);
        self.text_model.embed_tokens.forward(&arr)
    }

    /// Run all layers, capture hidden states at specified layer indices, return
    /// last-position logits AND concatenated captures [B, T, num_captures * hidden].
    /// `layer_ids` must be valid 0-based layer indices. Captures the hidden state
    /// AFTER the block forward for that layer (before the final norm).
    /// Only projects the last sequence position through lm_head to avoid [B, T, vocab] OOM.
    pub fn forward_last_logits_with_hidden_capture(
        &mut self,
        inputs: &Array,
        cache: &mut Vec<HybridCache>,
        capture_layer_ids: &[usize],
    ) -> Result<(Array, Array), Exception> {
        if capture_layer_ids.is_empty() {
            return Ok((self.forward_last_logits(inputs, cache)?, Array::zeros::<f32>(&[0])?));
        }
        if let Some(&layer_id) = capture_layer_ids
            .iter()
            .find(|&&layer_id| layer_id >= self.text_model.layers.len())
        {
            return Err(Exception::custom(format!(
                "capture layer index {layer_id} out of range for {} layers",
                self.text_model.layers.len()
            )));
        }

        let mut h = self.text_model.embed_tokens.forward(inputs)?;
        let t = h.shape()[1];
        let mask = if t > 1 {
            Some(mlx_rs_core::utils::AttentionMask::Causal)
        } else {
            None
        };

        self.text_model.ensure_cache(cache);

        let mut captures = Vec::with_capacity(capture_layer_ids.len());
        for (layer_idx, (layer, c)) in self
            .text_model
            .layers
            .iter_mut()
            .zip(cache.iter_mut())
            .enumerate()
        {
            h = layer.forward(&h, mask.as_ref(), c)?;
            if capture_layer_ids.iter().any(|&id| id == layer_idx) {
                captures.push(h.clone());
            }
        }

        h = self.text_model.norm.forward(&h)?;
        let logits = self.apply_lm_head(&h.index((.., -1i32, ..)))?;
        let capture_refs = captures.iter().collect::<Vec<_>>();
        let captures = concatenate_axis(&capture_refs, 2)?;
        Ok((logits, captures))
    }

    /// Apply lm_head (or embed_tokens.T if no lm_head) to arbitrary hidden states.
    /// `h` can be any shape ending in hidden_size — result replaces last dim with vocab_size.
    pub fn apply_lm_head(&mut self, h: &Array) -> Result<Array, Exception> {
        match self.lm_head.as_mut() {
            Some(lm_head) => lm_head.forward(h),
            None => match &mut self.text_model.embed_tokens {
                MaybeQuantized::Original(e) => e.as_linear(h),
                MaybeQuantized::Quantized(qe) => qe.as_linear(h),
            },
        }
    }

    /// Extract the lm_head weight matrix [vocab_size, hidden_size] as a contiguous BF16 Array.
    /// For tied weights (no separate lm_head), this dequantizes the embedding weight.
    /// Used to project draft model hidden states through the target's vocabulary projection.
    pub fn get_lm_head_weight(&mut self) -> Result<Array, Exception> {
        let weight = match self.lm_head.as_mut() {
            Some(MaybeQuantized::Original(l)) => l.weight.as_ref().clone(),
            Some(MaybeQuantized::Quantized(ql)) => dequantize(
                &ql.inner.weight,
                &ql.scales,
                &ql.biases,
                ql.group_size,
                ql.bits,
                None::<&str>,
            )?,
            None => match &mut self.text_model.embed_tokens {
                MaybeQuantized::Original(e) => e.weight.as_ref().clone(),
                MaybeQuantized::Quantized(qe) => dequantize(
                    &qe.inner.weight,
                    &qe.scales,
                    &qe.biases,
                    qe.group_size,
                    qe.bits,
                    None::<&str>,
                )?,
            },
        };
        weight.as_dtype(Dtype::Bfloat16)?.contiguous()
    }

    /// Packed quantized lm_head arrays `(weight, scales, biases, group_size,
    /// bits)` when the vocabulary projection is quantized (explicit lm_head or
    /// tied quantized embedding). Returns `None` for non-quantized
    /// checkpoints — callers fall back to [`Self::get_lm_head_weight`].
    /// Lets external drafters (DFlash) run `quantized_matmul` against the
    /// packed weight instead of materializing and re-reading a dequantized
    /// BF16 `[vocab, hidden]` copy every cycle.
    pub fn get_lm_head_quantized(&self) -> Option<(Array, Array, Array, i32, i32)> {
        match self.lm_head.as_ref() {
            Some(MaybeQuantized::Quantized(ql)) => Some((
                ql.inner.weight.as_ref().clone(),
                ql.scales.as_ref().clone(),
                ql.biases.as_ref().clone(),
                ql.group_size,
                ql.bits,
            )),
            Some(MaybeQuantized::Original(_)) => None,
            None => match &self.text_model.embed_tokens {
                MaybeQuantized::Quantized(qe) => Some((
                    qe.inner.weight.as_ref().clone(),
                    qe.scales.as_ref().clone(),
                    qe.biases.as_ref().clone(),
                    qe.group_size,
                    qe.bits,
                )),
                MaybeQuantized::Original(_) => None,
            },
        }
    }

    #[allow(non_snake_case)]
    fn forward_hidden(
        &mut self,
        inputs: &Array,
        cache: &mut Vec<HybridCache>,
    ) -> Result<Array, Exception> {
        let h = self.forward_pre_norm_hidden(inputs, cache)?;
        self.text_model.norm.forward(&h)
    }

    /// Returns the hidden state immediately AFTER the last transformer block
    /// but BEFORE the final RMSNorm. This is the input distribution the MTP
    /// draft head was trained on (mirroring llama.cpp's
    /// `llama_get_embeddings_pre_norm_ith`). Feeding the post-norm output
    /// shifts the distribution and tanks draft acceptance.
    pub fn forward_pre_norm_hidden(
        &mut self,
        inputs: &Array,
        cache: &mut Vec<HybridCache>,
    ) -> Result<Array, Exception> {
        let mut h = self.text_model.embed_tokens.forward(inputs)?;

        let seq_len = h.shape()[1];
        let mask = if seq_len > 1 {
            Some(mlx_rs_core::utils::AttentionMask::Causal)
        } else {
            None
        };

        self.text_model.ensure_cache(cache);

        for (layer, c) in self.text_model.layers.iter_mut().zip(cache.iter_mut()) {
            h = layer.forward(&h, mask.as_ref(), c)?;
        }

        Ok(h)
    }
}

// ============================================================================
// Weight Loading
// ============================================================================

#[derive(Debug, Clone, serde::Deserialize)]
pub struct WeightMap {
    pub weight_map: HashMap<String, String>,
}

fn load_all_weights(model_dir: &Path) -> Result<HashMap<String, Array>, Error> {
    let weights_index = model_dir.join("model.safetensors.index.json");

    let mut all_weights: HashMap<String, Array> = HashMap::new();
    if weights_index.exists() {
        let json = std::fs::read_to_string(weights_index)?;
        let weight_map: WeightMap = serde_json::from_str(&json)?;
        let weight_files: HashSet<&String> = weight_map.weight_map.values().collect();
        for weight_file in weight_files {
            let path = model_dir.join(weight_file);
            let loaded = Array::load_safetensors(&path)?;
            all_weights.extend(loaded);
        }
    } else {
        let path = model_dir.join("model.safetensors");
        let loaded = Array::load_safetensors(&path)?;
        all_weights.extend(loaded);
    }

    // Optional MTP sidecar. MTPLX-Optimized-Speed checkpoints ship MTP
    // weights in a separate `mtp.safetensors` file referenced from
    // `mlx_lm_extra_tensors.mtp_file` in config.json. If the file exists at
    // the conventional path, merge its tensors into the same flat weight map
    // so the existing `mtp::load_mtp_head` flat-key probes pick them up.
    let mtp_sidecar = model_dir.join("mtp.safetensors");
    if mtp_sidecar.exists() {
        let loaded = Array::load_safetensors(&mtp_sidecar)?;
        all_weights.extend(loaded);
    }

    Ok(all_weights)
}

pub(crate) fn get_weight(weights: &HashMap<String, Array>, key: &str) -> Result<Array, Error> {
    weights
        .get(key)
        .cloned()
        .ok_or_else(|| Error::WeightNotFound(key.to_string()))
}

/// Load a `MaybeQuantized<nn::Linear>` from packed checkpoint weights.
///
/// Selects the quantized path when `<prefix>.scales` is present in the
/// weight map, and falls back to a plain BF16/FP16 `nn::Linear` otherwise.
///
/// This is the path Unsloth Dynamic ("UD") MLX 4-bit checkpoints rely on:
/// the global config still declares a uniform 4-bit quantization, but
/// accuracy-sensitive modules (in the Qwen3.6 case the `linear_attn`
/// projections — `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`,
/// `out_proj`) ship with only `.weight` and no `.scales`/`.biases`,
/// meaning they were left at full precision. The canonical model and
/// every per-module loader site that goes through here keeps working
/// unchanged because for canonical checkpoints `.scales` is always
/// present.
pub(crate) fn make_maybe_quantized_linear(
    weights: &HashMap<String, Array>,
    prefix: &str,
    group_size: i32,
    bits: i32,
) -> Result<MaybeQuantized<nn::Linear>, Error> {
    if weights.contains_key(&format!("{}.scales", prefix)) {
        Ok(MaybeQuantized::Quantized(make_quantized_linear(
            weights, prefix, group_size, bits,
        )?))
    } else {
        let weight = get_weight(weights, &format!("{}.weight", prefix))?;
        Ok(MaybeQuantized::Original(nn::Linear {
            weight: Param::new(weight),
            bias: Param::new(None),
        }))
    }
}

/// Derive per-tensor `bits` from the weight/scales shape ratio.
///
/// For MLX packed quantization the relation is
///   weight.shape[-1] = in_features / (32 / bits)
///   scales.shape[-1] = in_features / group_size
/// ⇒ bits = (weight.shape[-1] × 32) / (scales.shape[-1] × group_size)
///
/// Returns the global `fallback` when shapes are unavailable. UD-MLX-4bit
/// ships heterogeneous quant (attention q/k/v/o @ 8-bit, MLP @ 4-bit) under
/// a single `bits=4` global config; the canonical (uniform 4-bit) checkpoint
/// is unaffected — the derivation evaluates to 4 there too.
pub(crate) fn derive_quant_bits(
    weight: &Array,
    scales: &Array,
    group_size: i32,
    fallback: i32,
) -> i32 {
    let wshape = weight.shape();
    let sshape = scales.shape();
    if wshape.len() < 2 || sshape.len() < 2 || group_size <= 0 {
        return fallback;
    }
    let wc = wshape[wshape.len() - 1] as i64;
    let sc = sshape[sshape.len() - 1] as i64;
    if sc <= 0 {
        return fallback;
    }
    let derived = (wc * 32 / (sc * group_size as i64)) as i32;
    // Only 2/3/4/5/6/8 are valid MLX quant bit-widths; anything else is a
    // mismatch in our derivation and we should not trust it.
    if matches!(derived, 2 | 3 | 4 | 5 | 6 | 8) {
        derived
    } else {
        fallback
    }
}

pub(crate) fn make_quantized_linear(
    weights: &HashMap<String, Array>,
    prefix: &str,
    group_size: i32,
    bits: i32,
) -> Result<nn::QuantizedLinear, Error> {
    let weight = get_weight(weights, &format!("{}.weight", prefix))?;
    let scales = get_weight(weights, &format!("{}.scales", prefix))?;
    let biases = get_weight(weights, &format!("{}.biases", prefix))?;

    let derived_bits = derive_quant_bits(&weight, &scales, group_size, bits);

    let inner = nn::Linear {
        weight: Param::new(weight),
        bias: Param::new(None),
    };

    let mut ql = nn::QuantizedLinear {
        group_size,
        bits: derived_bits,
        scales: Param::new(scales),
        biases: Param::new(biases),
        inner,
    };
    ql.freeze_parameters(true);
    Ok(ql)
}

fn make_quantized_embedding(
    weights: &HashMap<String, Array>,
    prefix: &str,
    group_size: i32,
    bits: i32,
) -> Result<nn::QuantizedEmbedding, Error> {
    let weight = get_weight(weights, &format!("{}.weight", prefix))?;
    let scales = get_weight(weights, &format!("{}.scales", prefix))?;
    let biases = get_weight(weights, &format!("{}.biases", prefix))?;

    let derived_bits = derive_quant_bits(&weight, &scales, group_size, bits);

    let inner = nn::Embedding {
        weight: Param::new(weight),
    };

    let mut qe = nn::QuantizedEmbedding {
        group_size,
        bits: derived_bits,
        scales: Param::new(scales),
        biases: Param::new(biases),
        inner,
    };
    qe.freeze_parameters(true);
    Ok(qe)
}

pub(crate) fn load_rms_norm(
    weights: &HashMap<String, Array>,
    key: &str,
    eps: f32,
) -> Result<nn::RmsNorm, Error> {
    Ok(nn::RmsNorm {
        weight: Param::new(get_weight(weights, key)?),
        eps,
    })
}

pub(crate) fn make_quantized_switch_linear(
    weights: &HashMap<String, Array>,
    prefix: &str,
    group_size: i32,
    bits: i32,
) -> Result<QuantizedSwitchLinear, Error> {
    let weight = get_weight(weights, &format!("{}.weight", prefix))?;
    let scales = get_weight(weights, &format!("{}.scales", prefix))?;
    let biases = get_weight(weights, &format!("{}.biases", prefix))?;

    let shape = weight.shape();
    let num_experts = shape[0] as i32;
    let output_dims = shape[1] as i32;
    let scales_shape = scales.shape();
    let input_dims = (scales_shape[2] as i32) * group_size;
    let derived_bits = derive_quant_bits(&weight, &scales, group_size, bits);

    Ok(QuantizedSwitchLinear {
        num_experts,
        input_dims,
        output_dims,
        group_size,
        bits: derived_bits,
        weight: Param::new(weight),
        scales: Param::new(scales),
        biases: Param::new(biases),
    })
}

/// Detect the weight key prefix (VLM vs standalone text model).
fn detect_prefix(weights: &HashMap<String, Array>) -> &'static str {
    if weights.keys().any(|k| k.starts_with("language_model.")) {
        "language_model.model"
    } else {
        "model"
    }
}

fn detect_lm_head_prefix(weights: &HashMap<String, Array>) -> &'static str {
    if weights.contains_key("language_model.lm_head.weight") {
        "language_model.lm_head"
    } else {
        "lm_head"
    }
}

fn build_model_from_weights(
    weights: &HashMap<String, Array>,
    args: &ModelArgs,
) -> Result<Model, Error> {
    let tc = &args.text_config;

    let quant = args.quantization();
    let (group_size, bits) = match quant {
        Some(q) => (q.group_size, q.bits),
        None => {
            return Err(Error::Model(
                "Only quantized models are supported".to_string(),
            ))
        }
    };

    if tc.layer_types.len() != tc.num_hidden_layers as usize {
        return Err(Error::Model(format!(
            "layer_types length ({}) != num_hidden_layers ({})",
            tc.layer_types.len(),
            tc.num_hidden_layers
        )));
    }

    let prefix = detect_prefix(weights);
    let lm_head_prefix = detect_lm_head_prefix(weights);

    let mut layers = Vec::with_capacity(tc.num_hidden_layers as usize);
    for i in 0..tc.num_hidden_layers {
        let layer_prefix = format!("{}.layers.{}", prefix, i);
        let layer_type = &tc.layer_types[i as usize];
        let force_full_attn = layer_type == "full_attention";
        layers.push(load_transformer_block(
            weights,
            &layer_prefix,
            tc,
            group_size,
            bits,
            force_full_attn,
        )?);
    }

    let embed_tokens = MaybeQuantized::Quantized(make_quantized_embedding(
        weights,
        &format!("{}.embed_tokens", prefix),
        group_size,
        bits,
    )?);

    let norm = load_rms_norm(weights, &format!("{}.norm.weight", prefix), tc.rms_norm_eps)?;

    let lm_head = if !args.tie_word_embeddings {
        Some(MaybeQuantized::Quantized(make_quantized_linear(
            weights,
            lm_head_prefix,
            group_size,
            bits,
        )?))
    } else {
        None
    };

    let text_model = Qwen36TextModel {
        embed_tokens,
        layers,
        norm,
        layer_types: tc.layer_types.clone(),
    };

    let mtp_head = load_mtp_head(weights, args)?;
    if let Some(mtp) = &mtp_head {
        eprintln!(
            "Loaded MTP head: {} layers, {} weight keys detected",
            mtp.num_layers,
            mtp.detected_weight_keys.len()
        );
    } else if args.mtp_num_hidden_layers() > 0 {
        eprintln!(
            "Config declares mtp_num_hidden_layers={} but no MTP weights found in checkpoint (this is expected for stock mlx-community releases).",
            args.mtp_num_hidden_layers()
        );
    }

    Ok(Model {
        args: args.clone(),
        text_model,
        lm_head,
        mtp_head,
    })
}

pub fn load_model(model_dir: impl AsRef<Path>) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();

    let config_file = std::fs::File::open(model_dir.join("config.json"))?;
    let args: ModelArgs = serde_json::from_reader(config_file)?;
    let tc = &args.text_config;

    let quant = args.quantization();
    let (_, bits) = match quant {
        Some(q) => (q.group_size, q.bits),
        None => {
            return Err(Error::Model(
                "Only quantized models are supported".to_string(),
            ))
        }
    };

    eprintln!(
        "Loading {}-bit quantized Qwen3.6 ({} layers, {})...",
        bits,
        tc.num_hidden_layers,
        if tc.is_moe() {
            format!(
                "MoE {} experts top-{}",
                tc.num_experts.unwrap_or(0),
                tc.num_experts_per_tok.unwrap_or(0)
            )
        } else {
            "dense MLP".to_string()
        }
    );

    let weights = load_all_weights(model_dir)?;
    let model = build_model_from_weights(&weights, &args)?;

    // Mirror gemma4-mlx: wire the model weight pages into Metal so they
    // stay resident across decode steps. Headroom for KV + activations
    // is env-tunable via QWEN36_WIRED_HEADROOM_GB (default 4).
    let weight_bytes: usize = weights.values().map(|arr| arr.nbytes()).sum();
    let device = mlx_rs_core::memory::get_device_info();
    let headroom_gb: usize = std::env::var("QWEN36_WIRED_HEADROOM_GB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let headroom: usize = headroom_gb * 1024 * 1024 * 1024;
    let target = weight_bytes.saturating_add(headroom);
    let limit = target.min(device.max_recommended_working_set_size);
    let _ = mlx_rs_core::memory::set_wired_limit(limit);

    // Optional MoE expert pre-warming via QWEN36_PREWARM_EXPERTS=1.
    if std::env::var("QWEN36_PREWARM_EXPERTS").is_ok() {
        let refs: Vec<&Array> = weights.values().collect();
        for chunk in refs.chunks(64) {
            let _ = mlx_rs::transforms::eval(chunk.iter().copied());
        }
    }

    Ok(model)
}

pub struct VlModel {
    pub text: Model,
    pub vision: VisionTower,
    pub image_token_id: i32,
    pub vision_start_token_id: i32,
    pub vision_end_token_id: i32,
}

impl VlModel {
    pub fn new_cache(&self, mode: KVCacheMode) -> Vec<HybridCache> {
        self.text.new_cache(mode)
    }

    /// Encode image bytes (JPEG/PNG/etc) into visual feature tokens.
    /// Returns `(visual_features: Array [N_vis, hidden], h_patches, w_patches)`.
    pub fn encode_image_bytes(
        &mut self,
        bytes: &[u8],
    ) -> Result<(Array, i32, i32), mlx_rs_core::error::Error> {
        let img = image::load_from_memory(bytes).map_err(|e| {
            mlx_rs_core::error::Error::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                e.to_string(),
            ))
        })?;
        let patch_size = self
            .text
            .args
            .vision_config
            .as_ref()
            .map(|vc| vc.patch_size)
            .unwrap_or(16);
        let temporal_patch_size = self
            .text
            .args
            .vision_config
            .as_ref()
            .map(|vc| vc.temporal_patch_size)
            .unwrap_or(2);
        let (pixel_array, h_patches, w_patches) =
            crate::vision::preprocess_image(&img, patch_size, temporal_patch_size)?;
        let visual_features = self.vision.forward(&pixel_array, h_patches, w_patches)?;
        Ok((visual_features, h_patches, w_patches))
    }

    /// Prefill with mixed text+image content.
    pub fn prefill_multimodal(
        &mut self,
        input_ids: &[i32],
        visual_features: &Array,
        cache: &mut Vec<HybridCache>,
    ) -> Result<Array, mlx_rs_core::error::Error> {
        let n_visual = visual_features.shape()[0] as usize;
        let hidden_size = visual_features.shape()[1];

        let image_token_id = self.image_token_id;
        let block_start = input_ids.iter().position(|&t| t == image_token_id);

        let logits = if let Some(start) = block_start {
            let block_len = input_ids[start..]
                .iter()
                .take_while(|&&t| t == image_token_id)
                .count();
            if block_len != n_visual {
                return Err(mlx_rs_core::error::Error::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!(
                        "Image token count mismatch: {} placeholders vs {} visual tokens",
                        block_len, n_visual
                    ),
                )));
            }

            let pre_ids = &input_ids[..start];
            let post_ids = &input_ids[start + block_len..];

            let mut parts: Vec<Array> = Vec::new();

            if !pre_ids.is_empty() {
                let pre_emb = self.text.embed_tokens(pre_ids).map_err(|e| {
                    mlx_rs_core::error::Error::Io(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        e.to_string(),
                    ))
                })?;
                parts.push(pre_emb);
            }

            let vis = visual_features.reshape(&[1, n_visual as i32, hidden_size])?;
            parts.push(vis);

            if !post_ids.is_empty() {
                let post_emb = self.text.embed_tokens(post_ids).map_err(|e| {
                    mlx_rs_core::error::Error::Io(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        e.to_string(),
                    ))
                })?;
                parts.push(post_emb);
            }

            let combined = if parts.len() == 1 {
                parts.remove(0)
            } else {
                let refs: Vec<&Array> = parts.iter().collect();
                mlx_rs::ops::concatenate_axis(&refs, 1).map_err(|e| {
                    mlx_rs_core::error::Error::Io(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        e.to_string(),
                    ))
                })?
            };

            self.text
                .forward_from_embeds(&combined, cache)
                .map_err(|e| {
                    mlx_rs_core::error::Error::Io(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        e.to_string(),
                    ))
                })?
        } else {
            let arr = Array::from_slice(input_ids, &[1, input_ids.len() as i32]);
            self.text.forward_last_logits(&arr, cache).map_err(|e| {
                mlx_rs_core::error::Error::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    e.to_string(),
                ))
            })?
        };

        Ok(logits)
    }

    pub fn prefill_text(
        &mut self,
        input_ids: &[i32],
        cache: &mut Vec<HybridCache>,
    ) -> Result<Array, mlx_rs_core::error::Error> {
        let arr = Array::from_slice(input_ids, &[1, input_ids.len() as i32]);
        self.text.forward_last_logits(&arr, cache).map_err(|e| {
            mlx_rs_core::error::Error::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                e.to_string(),
            ))
        })
    }

    /// Single-token decode step. `token_id` is the previously sampled token.
    pub fn decode_token(
        &mut self,
        token_id: i32,
        cache: &mut Vec<HybridCache>,
    ) -> Result<Array, mlx_rs_core::error::Error> {
        let arr = Array::from_slice(&[token_id], &[1, 1]);
        self.text.forward_last_logits(&arr, cache).map_err(|e| {
            mlx_rs_core::error::Error::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                e.to_string(),
            ))
        })
    }
}

pub fn load_vl_model(model_dir: impl AsRef<Path>) -> Result<VlModel, Error> {
    let model_dir = model_dir.as_ref();
    let config_file = std::fs::File::open(model_dir.join("config.json"))?;
    let args: ModelArgs = serde_json::from_reader(config_file)?;

    let vc = args
        .vision_config
        .as_ref()
        .ok_or_else(|| Error::Model("Not a VL model: missing vision_config".to_string()))?;
    let image_token_id = args.image_token_id.unwrap_or(248056);
    let vision_start_token_id = args.vision_start_token_id.unwrap_or(248053);
    let vision_end_token_id = args.vision_end_token_id.unwrap_or(248054);

    if vc.out_hidden_size != args.text_config.hidden_size {
        return Err(Error::Model(format!(
            "Vision out_hidden_size ({}) != text hidden_size ({})",
            vc.out_hidden_size, args.text_config.hidden_size
        )));
    }

    let weights = load_all_weights(model_dir)?;
    let text = build_model_from_weights(&weights, &args)?;
    let vision = crate::vision::load_vision_tower(&weights, vc)?;

    Ok(VlModel {
        text,
        vision,
        image_token_id,
        vision_start_token_id,
        vision_end_token_id,
    })
}

/// Load a transformer block at `layer_prefix`. If `force_full_attention` is
/// `true`, builds a `FullAttention` block regardless of the layer's slot in
/// `tc.layer_types` — useful for MTP heads whose single block is always
/// full-attention.
pub(crate) fn load_transformer_block(
    weights: &HashMap<String, Array>,
    layer_prefix: &str,
    tc: &TextConfig,
    group_size: i32,
    bits: i32,
    force_full_attention: bool,
) -> Result<TransformerBlock, Error> {
    let attention = if force_full_attention {
        AttentionLayer::FullAttention(load_gated_attention(
            weights,
            layer_prefix,
            tc,
            group_size,
            bits,
        )?)
    } else {
        AttentionLayer::LinearAttention(load_gated_deltanet(
            weights,
            layer_prefix,
            tc,
            group_size,
            bits,
        )?)
    };
    let ffn = load_ffn_block(weights, layer_prefix, tc, group_size, bits)?;
    Ok(TransformerBlock {
        attention,
        ffn,
        input_layernorm: load_rms_norm(
            weights,
            &format!("{}.input_layernorm.weight", layer_prefix),
            tc.rms_norm_eps,
        )?,
        post_attention_layernorm: load_rms_norm(
            weights,
            &format!("{}.post_attention_layernorm.weight", layer_prefix),
            tc.rms_norm_eps,
        )?,
    })
}

pub(crate) fn load_ffn_block(
    weights: &HashMap<String, Array>,
    layer_prefix: &str,
    tc: &TextConfig,
    group_size: i32,
    bits: i32,
) -> Result<FfnBlock, Error> {
    if tc.is_moe() {
        let num_experts = tc
            .num_experts
            .ok_or_else(|| Error::Model("MoE config missing num_experts".to_string()))?;
        let config_top_k = tc
            .num_experts_per_tok
            .ok_or_else(|| Error::Model("MoE config missing num_experts_per_tok".to_string()))?;
        // Env-tunable MoE top-k. Default = config value. Clamp so we
        // never exceed the trained-for top-k.
        let top_k = std::env::var("QWEN36_MOE_TOPK")
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
            .map(|n| n.clamp(1, config_top_k))
            .unwrap_or(config_top_k);

        let gate = MaybeQuantized::Quantized(make_quantized_linear(
            weights,
            &format!("{}.mlp.gate", layer_prefix),
            group_size,
            8,
        )?);

        let switch_mlp = SwitchGLU {
            gate_proj: make_quantized_switch_linear(
                weights,
                &format!("{}.mlp.switch_mlp.gate_proj", layer_prefix),
                group_size,
                bits,
            )?,
            up_proj: make_quantized_switch_linear(
                weights,
                &format!("{}.mlp.switch_mlp.up_proj", layer_prefix),
                group_size,
                bits,
            )?,
            down_proj: make_quantized_switch_linear(
                weights,
                &format!("{}.mlp.switch_mlp.down_proj", layer_prefix),
                group_size,
                bits,
            )?,
        };

        let shared_expert = SharedExpert {
            gate_proj: MaybeQuantized::Quantized(make_quantized_linear(
                weights,
                &format!("{}.mlp.shared_expert.gate_proj", layer_prefix),
                group_size,
                bits,
            )?),
            up_proj: MaybeQuantized::Quantized(make_quantized_linear(
                weights,
                &format!("{}.mlp.shared_expert.up_proj", layer_prefix),
                group_size,
                bits,
            )?),
            down_proj: MaybeQuantized::Quantized(make_quantized_linear(
                weights,
                &format!("{}.mlp.shared_expert.down_proj", layer_prefix),
                group_size,
                bits,
            )?),
        };

        let shared_expert_gate = MaybeQuantized::Quantized(make_quantized_linear(
            weights,
            &format!("{}.mlp.shared_expert_gate", layer_prefix),
            group_size,
            8,
        )?);

        Ok(FfnBlock::Moe(MoeBlock {
            num_experts,
            top_k,
            gate,
            switch_mlp,
            shared_expert,
            shared_expert_gate,
        }))
    } else {
        Ok(FfnBlock::Dense(DenseMlp {
            gate_proj: MaybeQuantized::Quantized(make_quantized_linear(
                weights,
                &format!("{}.mlp.gate_proj", layer_prefix),
                group_size,
                bits,
            )?),
            up_proj: MaybeQuantized::Quantized(make_quantized_linear(
                weights,
                &format!("{}.mlp.up_proj", layer_prefix),
                group_size,
                bits,
            )?),
            down_proj: MaybeQuantized::Quantized(make_quantized_linear(
                weights,
                &format!("{}.mlp.down_proj", layer_prefix),
                group_size,
                bits,
            )?),
        }))
    }
}

pub(crate) fn load_gated_attention(
    weights: &HashMap<String, Array>,
    layer_prefix: &str,
    tc: &TextConfig,
    group_size: i32,
    bits: i32,
) -> Result<GatedAttention, Error> {
    let attn_prefix = format!("{}.self_attn", layer_prefix);

    let rope_dims = (tc.head_dim as f32 * tc.rope_parameters.partial_rotary_factor) as i32;
    let rope = initialize_rope(
        rope_dims,
        tc.rope_parameters.rope_theta,
        false,
        &None,
        tc.max_position_embeddings,
    )?;

    Ok(GatedAttention {
        n_heads: tc.num_attention_heads,
        n_kv_heads: tc.num_key_value_heads,
        head_dim: tc.head_dim,
        scale: (tc.head_dim as f32).sqrt().recip(),
        q_proj: MaybeQuantized::Quantized(make_quantized_linear(
            weights,
            &format!("{}.q_proj", attn_prefix),
            group_size,
            bits,
        )?),
        k_proj: MaybeQuantized::Quantized(make_quantized_linear(
            weights,
            &format!("{}.k_proj", attn_prefix),
            group_size,
            bits,
        )?),
        v_proj: MaybeQuantized::Quantized(make_quantized_linear(
            weights,
            &format!("{}.v_proj", attn_prefix),
            group_size,
            bits,
        )?),
        o_proj: MaybeQuantized::Quantized(make_quantized_linear(
            weights,
            &format!("{}.o_proj", attn_prefix),
            group_size,
            bits,
        )?),
        q_norm: load_rms_norm(
            weights,
            &format!("{}.q_norm.weight", attn_prefix),
            tc.rms_norm_eps,
        )?,
        k_norm: load_rms_norm(
            weights,
            &format!("{}.k_norm.weight", attn_prefix),
            tc.rms_norm_eps,
        )?,
        rope,
    })
}

fn load_gated_deltanet(
    weights: &HashMap<String, Array>,
    layer_prefix: &str,
    tc: &TextConfig,
    group_size: i32,
    bits: i32,
) -> Result<GatedDeltaNet, Error> {
    let attn_prefix = format!("{}.linear_attn", layer_prefix);

    let num_k_heads = tc.linear_num_key_heads;
    let num_v_heads = tc.linear_num_value_heads;
    if num_v_heads % num_k_heads != 0 {
        return Err(Error::Model(format!(
            "linear_num_value_heads ({}) must be divisible by linear_num_key_heads ({})",
            num_v_heads, num_k_heads
        )));
    }
    let key_head_dim = tc.linear_key_head_dim;
    let value_head_dim = tc.linear_value_head_dim;
    let key_dim = num_k_heads * key_head_dim;
    let value_dim = num_v_heads * value_head_dim;
    let conv_dim = key_dim * 2 + value_dim;
    let conv_kernel_size = tc.linear_conv_kernel_dim;

    // UD-MLX-4bit checkpoints keep these projections at full precision.
    // `make_maybe_quantized_linear` chooses between the quantized and plain
    // paths based on per-module presence of `.scales`, so the canonical
    // (fully-quantized) checkpoint still loads through the quantized branch.
    Ok(GatedDeltaNet {
        in_proj_qkv: make_maybe_quantized_linear(
            weights,
            &format!("{}.in_proj_qkv", attn_prefix),
            group_size,
            bits,
        )?,
        in_proj_z: make_maybe_quantized_linear(
            weights,
            &format!("{}.in_proj_z", attn_prefix),
            group_size,
            bits,
        )?,
        in_proj_a: make_maybe_quantized_linear(
            weights,
            &format!("{}.in_proj_a", attn_prefix),
            group_size,
            bits,
        )?,
        in_proj_b: make_maybe_quantized_linear(
            weights,
            &format!("{}.in_proj_b", attn_prefix),
            group_size,
            bits,
        )?,
        conv1d_weight: Param::new(get_weight(
            weights,
            &format!("{}.conv1d.weight", attn_prefix),
        )?),
        a_log: Param::new(get_weight(weights, &format!("{}.A_log", attn_prefix))?),
        dt_bias: Param::new(get_weight(weights, &format!("{}.dt_bias", attn_prefix))?),
        norm: load_rms_norm(
            weights,
            &format!("{}.norm.weight", attn_prefix),
            tc.rms_norm_eps,
        )?,
        out_proj: make_maybe_quantized_linear(
            weights,
            &format!("{}.out_proj", attn_prefix),
            group_size,
            bits,
        )?,
        num_k_heads,
        num_v_heads,
        key_head_dim,
        value_head_dim,
        key_dim,
        value_dim,
        conv_dim,
        conv_kernel_size,
    })
}
