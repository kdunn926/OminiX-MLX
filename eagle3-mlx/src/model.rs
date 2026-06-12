//! EAGLE-3 draft model (speculators `Eagle3Speculator` format).
//!
//! A single llama-style decoder layer that drafts in the target's feature
//! space:
//!
//! - Per committed target position `i`, the target's hidden states at the
//!   configured aux layers are concatenated `[B, T, 3H]` and fused through
//!   `fc` to `[B, T, H]`.
//! - The decoder layer input is `concat(input_layernorm(embed(token_{i+1})),
//!   hidden_norm(fused_i))` → `[B, T, 2H]`, so the q/k/v projections take
//!   `2H` inputs.
//! - With `norm_before_residual = true` (RedHat variants) the *normed* fused
//!   hidden is the attention residual; otherwise the raw fused hidden is.
//! - Chain steps (drafting token j+1 from drafted token j) feed the layer's
//!   own pre-norm output back as the next step's hidden input, bypassing
//!   `fc` (its last dim is already `H`).
//! - `lm_head` projects to the reduced draft vocabulary; `d2t` holds
//!   per-draft-id offsets into the target vocabulary:
//!   `target_id = draft_id + d2t[draft_id]`.
//!
//! Reference implementations: vLLM `llama_eagle3.py` and llama.cpp
//! `src/models/eagle3.cpp` (commit 88a3927).

use std::path::Path;

use mlx_rs::{
    builder::Builder,
    error::Exception,
    fast,
    module::Module,
    nn,
    ops::{
        self,
        indexing::{IndexOp, TryIndexOp},
    },
    Array, Dtype,
};
use mlx_rs_core::{
    cache::{KVCache, KeyValueCache},
    error::Error,
    utils::{create_causal_mask, scaled_dot_product_attention, SdpaMask},
};

use crate::config::Eagle3Config;

/// A weight matrix that may be load-time quantized.
///
/// The draft's two big tensors dominate its cost: `lm_head` (`[32k, H]`,
/// 180 MB bf16) is fully re-read by every chain step's logits matmul, and
/// `embed_tokens` (`[262k, H]`, 1.5 GB bf16) dominates resident memory.
/// Quantizing both cuts per-cycle bandwidth ~4x and residency ~3.7x;
/// the decoder-layer projections are small and stay bf16.
pub enum QuantizableWeight {
    Dense(Array),
    Quantized {
        w: Array,
        scales: Array,
        biases: Array,
        group_size: i32,
        bits: i32,
    },
}

impl QuantizableWeight {
    fn quantize(w: Array, bits: i32) -> Result<Self, Exception> {
        let group_size = 64;
        let (wq, scales, biases) = ops::quantize(&w, group_size, bits, None)?;
        mlx_rs::transforms::eval([&wq, &scales, &biases])?;
        Ok(Self::Quantized {
            w: wq,
            scales,
            biases,
            group_size,
            bits,
        })
    }

    /// `x @ w.T` — the lm_head projection.
    fn matmul_t(&self, x: &Array) -> Result<Array, Exception> {
        match self {
            Self::Dense(w) => ops::matmul(x, w.t()),
            Self::Quantized {
                w,
                scales,
                biases,
                group_size,
                bits,
            } => ops::quantized_matmul(
                x,
                w,
                scales,
                biases,
                true,
                *group_size,
                *bits,
                None::<&'static str>,
            ),
        }
    }

    /// Row gather — the embedding lookup. `ids` indexes axis 0.
    fn rows(&self, ids: &Array) -> Result<Array, Exception> {
        match self {
            Self::Dense(w) => w.try_index(ids),
            Self::Quantized {
                w,
                scales,
                biases,
                group_size,
                bits,
            } => ops::dequantize(
                &w.try_index(ids)?,
                &scales.try_index(ids)?,
                &biases.try_index(ids)?,
                *group_size,
                *bits,
                None::<&'static str>,
            ),
        }
    }

    fn shape0(&self) -> i32 {
        match self {
            Self::Dense(w) => w.shape()[0],
            Self::Quantized { w, .. } => w.shape()[0],
        }
    }
}

/// `EAGLE3_QUANT_DRAFT`: `8` (default) or `4` quantize the draft's
/// embed_tokens + lm_head at load; `0`/`off` keeps them bf16.
///
/// Measured on 26B-A4B (300-token greedy): 8-bit is both faster than 4-bit
/// (40.0 vs 39.3 tok/s — 4-bit's acceptance dip 0.482→0.466 costs more than
/// its bandwidth saving) and acceptance-lossless vs bf16.
fn quant_draft_bits() -> Option<i32> {
    match std::env::var("EAGLE3_QUANT_DRAFT").ok().as_deref() {
        Some("0") | Some("off") | Some("none") => None,
        Some("4") => Some(4),
        _ => Some(8),
    }
}

pub struct Eagle3DraftModel {
    pub config: Eagle3Config,

    // [V_target, H] — full target vocab so committed token ids embed directly.
    embed_tokens: QuantizableWeight,
    // [H, 3H] — aux hidden fusion.
    fc: Array,

    // Decoder layer.
    input_layernorm: Array,
    hidden_norm: Array,
    post_attention_layernorm: Array,
    q_proj: Array, // [n_heads*hd, 2H]
    k_proj: Array, // [n_kv*hd, 2H]
    v_proj: Array, // [n_kv*hd, 2H]
    o_proj: Array, // [H, n_heads*hd]
    gate_proj: Array,
    up_proj: Array,
    down_proj: Array,

    // Final norm + reduced-vocab head.
    norm: Array,
    lm_head: QuantizableWeight, // [V_draft, H]
    // [V_draft] i64 offsets: target_id = draft_id + d2t[draft_id].
    d2t: Array,

    rope: nn::Rope,
    n_heads: i32,
    n_kv_heads: i32,
    eps: f32,
    scale: f32,
}

impl std::fmt::Debug for Eagle3DraftModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Eagle3DraftModel")
            .field("verifier", &self.config.verifier_name())
            .finish()
    }
}

fn get(weights: &std::collections::HashMap<String, Array>, key: &str) -> Result<Array, Error> {
    weights
        .get(key)
        .cloned()
        .ok_or_else(|| Error::WeightNotFound(key.to_string()))
}

impl Eagle3DraftModel {
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self, Error> {
        let model_dir = model_dir.as_ref();
        let config = Eagle3Config::load(model_dir)?;
        let weights = Array::load_safetensors(model_dir.join("model.safetensors"))?;

        let tl = &config.transformer_layer_config;
        let rope = nn::RopeBuilder::new(tl.head_dim)
            .traditional(false)
            .base(config.rope_theta())
            .scale(1.0)
            .build()
            .expect("Infallible");

        let quant_bits = quant_draft_bits();
        let wrap = |w: Array| -> Result<QuantizableWeight, Error> {
            match quant_bits {
                Some(bits) => Ok(QuantizableWeight::quantize(w, bits)?),
                None => Ok(QuantizableWeight::Dense(w)),
            }
        };
        if let Some(bits) = quant_bits {
            eprintln!("[eagle3] draft embed_tokens + lm_head quantized to {bits}-bit (EAGLE3_QUANT_DRAFT)");
        }

        let model = Self {
            embed_tokens: wrap(get(&weights, "embed_tokens.weight")?)?,
            fc: get(&weights, "fc.weight")?,
            input_layernorm: get(&weights, "layers.0.input_layernorm.weight")?,
            hidden_norm: get(&weights, "layers.0.hidden_norm.weight")?,
            post_attention_layernorm: get(&weights, "layers.0.post_attention_layernorm.weight")?,
            q_proj: get(&weights, "layers.0.self_attn.q_proj.weight")?,
            k_proj: get(&weights, "layers.0.self_attn.k_proj.weight")?,
            v_proj: get(&weights, "layers.0.self_attn.v_proj.weight")?,
            o_proj: get(&weights, "layers.0.self_attn.o_proj.weight")?,
            gate_proj: get(&weights, "layers.0.mlp.gate_proj.weight")?,
            up_proj: get(&weights, "layers.0.mlp.up_proj.weight")?,
            down_proj: get(&weights, "layers.0.mlp.down_proj.weight")?,
            norm: get(&weights, "norm.weight")?,
            lm_head: wrap(get(&weights, "lm_head.weight")?)?,
            d2t: get(&weights, "d2t")?,
            rope,
            n_heads: tl.num_attention_heads,
            n_kv_heads: tl.num_key_value_heads,
            eps: tl.rms_norm_eps,
            scale: (tl.head_dim as f32).powf(-0.5),
            config,
        };

        // Sanity-check shapes against the config so a mismatched checkpoint
        // fails at load, not mid-generation.
        let h = model.config.transformer_layer_config.hidden_size;
        let n_aux = model.config.eagle_aux_hidden_state_layer_ids.len() as i32;
        if model.fc.shape() != [h, n_aux * h] {
            return Err(Error::InvalidConfig(format!(
                "fc.weight shape {:?} does not match [hidden, n_aux*hidden] = [{h}, {}]",
                model.fc.shape(),
                n_aux * h
            )));
        }
        if model.q_proj.shape()[1] != 2 * h {
            return Err(Error::InvalidConfig(format!(
                "q_proj input dim {} != 2*hidden ({}) — not an EAGLE-3 layer",
                model.q_proj.shape()[1],
                2 * h
            )));
        }
        if model.lm_head.shape0() != model.config.draft_vocab_size {
            return Err(Error::InvalidConfig(format!(
                "lm_head rows {} != draft_vocab_size {}",
                model.lm_head.shape0(),
                model.config.draft_vocab_size
            )));
        }
        Ok(model)
    }

    pub fn hidden_size(&self) -> i32 {
        self.config.transformer_layer_config.hidden_size
    }

    /// Embed committed token ids `[B, T]` (target vocab) → `[B, T, H]`.
    pub fn embed(&self, ids: &Array) -> Result<Array, Exception> {
        self.embed_tokens.rows(ids)
    }

    /// Fuse concatenated aux hidden states `[B, T, n_aux*H]` → `[B, T, H]`.
    pub fn fuse(&self, aux_hidden: &Array) -> Result<Array, Exception> {
        ops::matmul(aux_hidden, self.fc.t())
    }

    /// One decoder-layer forward over `T` positions.
    ///
    /// - `embeds`: `[B, T, H]` token embeddings (token at position `i+1`).
    /// - `hidden`: `[B, T, H]` fused target features (position `i`) for
    ///   ingest, or the previous step's pre-norm output for chain steps.
    /// - `pos_offset`: absolute position of the first row (RoPE offset).
    ///
    /// Returns the pre-norm residual-stream output `[B, T, H]` — feed it to
    /// [`Self::logits`] for drafting and back in as `hidden` for the next
    /// chain step.
    pub fn forward<C: KeyValueCache>(
        &mut self,
        embeds: &Array,
        hidden: &Array,
        cache: &mut C,
        pos_offset: i32,
    ) -> Result<Array, Exception> {
        let b = embeds.shape()[0];
        let t = embeds.shape()[1];

        let e = fast::rms_norm(embeds, &self.input_layernorm, self.eps)?;
        let (h_normed, residual) = if self.config.norm_before_residual {
            let hn = fast::rms_norm(hidden, &self.hidden_norm, self.eps)?;
            (hn.clone(), hn)
        } else {
            (
                fast::rms_norm(hidden, &self.hidden_norm, self.eps)?,
                hidden.clone(),
            )
        };
        let x = ops::concatenate_axis(&[&e, &h_normed], -1)?; // [B, T, 2H]

        // Attention.
        let queries = ops::matmul(&x, self.q_proj.t())?
            .reshape(&[b, t, self.n_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let keys = ops::matmul(&x, self.k_proj.t())?
            .reshape(&[b, t, self.n_kv_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let values = ops::matmul(&x, self.v_proj.t())?
            .reshape(&[b, t, self.n_kv_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let queries = self.rope.forward(nn::RopeInput::from((&queries, pos_offset)))?;
        let keys = self.rope.forward(nn::RopeInput::from((&keys, pos_offset)))?;
        let (keys, values) = cache.update_and_fetch(keys, values)?;

        let mask = if t > 1 {
            let kv_len = keys.shape()[2];
            let m = create_causal_mask(t, Some(kv_len - t), None, None)?
                .as_dtype(x.dtype())?;
            Some(m)
        } else {
            None
        };
        let attn = scaled_dot_product_attention::<&mut C>(
            queries,
            keys,
            values,
            None,
            self.scale,
            mask.as_ref().map(SdpaMask::Array),
        )?
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[b, t, -1])?;
        let attn = ops::matmul(&attn, self.o_proj.t())?;

        // Post-attention norm + MLP, llama style with fused-add semantics:
        // residual2 = residual + attn; out = residual2 + mlp(norm(residual2)).
        let residual2 = residual.add(&attn)?;
        let m_in = fast::rms_norm(&residual2, &self.post_attention_layernorm, self.eps)?;
        let gated = nn::silu(ops::matmul(&m_in, self.gate_proj.t())?)?
            .multiply(ops::matmul(&m_in, self.up_proj.t())?)?;
        let mlp_out = ops::matmul(&gated, self.down_proj.t())?;
        residual2.add(&mlp_out)
    }

    /// Draft-vocab logits for pre-norm hidden rows `[B, T, H]` → `[B, T, Vd]`.
    pub fn logits(&self, prenorm_hidden: &Array) -> Result<Array, Exception> {
        let normed = fast::rms_norm(prenorm_hidden, &self.norm, self.eps)?;
        self.lm_head.matmul_t(&normed)
    }

    /// Greedy-pick a draft token from the last row of `[B, T, Vd]` logits and
    /// map it to the target vocabulary: `target_id = draft_id + d2t[draft_id]`.
    ///
    /// Returns a `[B]` u32 array and does NOT eval — the draft chain stays
    /// GPU-resident; the session's verify-input build is the single host sync.
    pub fn sample_mapped(&self, draft_logits: &Array) -> Result<Array, Exception> {
        let last = draft_logits.index((.., -1, ..)); // [B, Vd]
        let draft_id = mlx_rs::argmax_axis!(&last, -1)?.as_dtype(Dtype::Int64)?;
        let offset = self.d2t.try_index(&draft_id)?;
        draft_id.add(&offset)?.as_dtype(Dtype::Uint32)
    }

    /// Fresh single-layer KV cache for this model.
    pub fn new_cache(&self) -> KVCache {
        KVCache::default()
    }
}
