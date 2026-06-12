//! DiffusionGemma (`model_type = "diffusion_gemma"`) — block-diffusion text
//! generation on the Gemma4 trunk.
//!
//! Ported from mlx-vlm 0.6.3 `models/diffusion_gemma/` +
//! `generate/diffusion.py` (the version that produced the
//! `mlx-community/diffusiongemma-26B-A4B-it-4bit` checkpoint).
//!
//! Architecture: ONE set of transformer weights (`model.decoder.*`) used in
//! two modes:
//!
//! * **Encoder mode** — a causal pass over committed tokens (prompt, then
//!   each finished canvas) that fills per-layer KV caches. Uses per-layer
//!   `model.encoder.language_model.layers.<i>.layer_scalar` overrides.
//! * **Decoder mode** — a bidirectional pass over a `canvas_length`-token
//!   canvas. Canvas Q/K/V attend over [encoder KV ‖ canvas KV] with no
//!   mask (sliding layers instead *slice* the encoder KV to the last
//!   `sliding_window - 1` positions). Nothing is written to the cache.
//!
//! Each decoder layer is the Gemma4 dual-FFN MoE block: dense MLP branch +
//! 128-expert top-8 MoE branch, both geglu (`gelu_approx(gate) * up`),
//! summed and post-normed. Full-attention layers tie K and V (`v_proj` is
//! absent; V = v_norm(raw K)). Logits are the tied embedding transpose with
//! a tanh softcap.
//!
//! Generation (entropy-bound sampler, greedy): start from a random canvas,
//! iterate up to `max_denoising_steps`: forward → divide by a linear
//! temperature schedule → accept the lowest-entropy positions (cumulative
//! entropy budget) → re-randomize the rest → feed softmax(logits)·E soft
//! embeddings back through the `self_conditioning` MLP on the next step.
//! Early-exit when the argmax canvas is stable and mean entropy is below
//! the confidence threshold. Commit the argmax canvas, re-encode it
//! causally into the KV caches, repeat until EOS / max_tokens.

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::{
    array,
    builder::Builder,
    error::Exception,
    module::{Module, Param},
    nn,
    ops::{self, indexing::IndexOp},
    quantization::MaybeQuantized,
    transforms::eval,
    Array, Dtype,
};
use mlx_rs_core::{
    cache::{KVCache, KeyValueCache},
    error::Error,
    utils::{create_causal_mask, scaled_dot_product_attention, SdpaMask},
};
use serde::Deserialize;

use crate::model::{
    get_weight, get_weight_optional, load_all_weights_unfiltered, make_mq_embedding,
    make_mq_linear, make_rms_norm, mq_embedding_as_linear, DenseMlp, GemmaActivation, GemmaRope,
    ProportionalRope, QuantizationConfig, Router, UnscaledRmsNorm,
};
use crate::quant_switch::{QuantizedSwitchLinear, SwitchGLU, SwitchGluExperts};

// ============================================================================
// Config
// ============================================================================

#[derive(Debug, Clone, Deserialize)]
pub struct DiffusionRopeSpec {
    pub rope_theta: f32,
    #[serde(default)]
    pub rope_type: Option<String>,
    #[serde(default = "default_partial_rotary")]
    pub partial_rotary_factor: f32,
}

fn default_partial_rotary() -> f32 {
    1.0
}

#[derive(Debug, Clone, Deserialize)]
pub struct DiffusionRopeParameters {
    pub sliding_attention: DiffusionRopeSpec,
    pub full_attention: DiffusionRopeSpec,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DiffusionTextConfig {
    pub vocab_size: i32,
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub moe_intermediate_size: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    #[serde(default)]
    pub num_global_key_value_heads: Option<i32>,
    pub head_dim: i32,
    #[serde(default)]
    pub global_head_dim: Option<i32>,
    #[serde(default = "default_activation")]
    pub hidden_activation: String,
    pub rms_norm_eps: f32,
    pub layer_types: Vec<String>,
    pub sliding_window: i32,
    #[serde(default = "default_softcap")]
    pub final_logit_softcapping: f32,
    pub num_experts: i32,
    pub top_k_experts: i32,
    pub rope_parameters: DiffusionRopeParameters,
}

fn default_activation() -> String {
    "gelu_pytorch_tanh".to_string()
}

fn default_softcap() -> f32 {
    30.0
}

#[derive(Debug, Clone, Deserialize)]
pub struct DiffusionGemmaConfig {
    pub text_config: DiffusionTextConfig,
    #[serde(default = "default_canvas_length")]
    pub canvas_length: i32,
    #[serde(default)]
    pub eos_token_id: Vec<i64>,
    #[serde(default)]
    pub generation_config: Option<serde_json::Value>,
    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
}

fn default_canvas_length() -> i32 {
    256
}

// ============================================================================
// Attention
// ============================================================================

pub struct DiffusionAttention {
    pub is_sliding: bool,
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub sliding_window: i32,
    pub q_proj: MaybeQuantized<nn::Linear>,
    pub k_proj: MaybeQuantized<nn::Linear>,
    /// Absent on full-attention layers — there K and V are tied
    /// (V = v_norm(raw K projection)).
    pub v_proj: Option<MaybeQuantized<nn::Linear>>,
    pub o_proj: MaybeQuantized<nn::Linear>,
    pub q_norm: nn::RmsNorm,
    pub k_norm: nn::RmsNorm,
    pub v_norm: UnscaledRmsNorm,
    pub rope: GemmaRope,
}

impl DiffusionAttention {
    /// Project Q/K/V for `x` `[B, L, H]` with RoPE at `offset`.
    /// Returns `(q [B,Hq,L,D], k [B,Hkv,L,D], v [B,Hkv,L,D])`.
    fn qkv(&mut self, x: &Array, offset: i32) -> Result<(Array, Array, Array), Exception> {
        let b = x.shape()[0];
        let l = x.shape()[1];

        let q = self
            .q_proj
            .forward(x)?
            .reshape(&[b, l, self.n_heads, self.head_dim])?;
        let q = self.q_norm.forward(&q)?.transpose_axes(&[0, 2, 1, 3])?;
        let q = self.rope.apply(&q, offset)?;

        let k_raw = self
            .k_proj
            .forward(x)?
            .reshape(&[b, l, self.n_kv_heads, self.head_dim])?;
        // Full-attention layers tie K and V: V = raw K projection (pre
        // k_norm), then v_norm.
        let v_raw = match self.v_proj.as_mut() {
            Some(vp) => vp
                .forward(x)?
                .reshape(&[b, l, self.n_kv_heads, self.head_dim])?,
            None => k_raw.clone(),
        };
        let k = self.k_norm.forward(&k_raw)?.transpose_axes(&[0, 2, 1, 3])?;
        let k = self.rope.apply(&k, offset)?;
        let v = self.v_norm.forward(&v_raw)?.transpose_axes(&[0, 2, 1, 3])?;
        Ok((q, k, v))
    }

    /// Causal encoder pass: append K/V to `cache`, attend over the full
    /// (masked) cache.
    pub fn forward_encoder(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: &mut KVCache,
    ) -> Result<Array, Exception> {
        let b = x.shape()[0];
        let l = x.shape()[1];
        let offset = cache.offset();
        let (q, k, v) = self.qkv(x, offset)?;
        let (keys, values) = cache.update_and_fetch(k, v)?;
        let sdpa_mask = match mask {
            Some(m) => Some(SdpaMask::Array(m)),
            None if l > 1 => Some(SdpaMask::Causal),
            None => None,
        };
        let out =
            scaled_dot_product_attention::<KVCache>(q, keys, values, None, 1.0, sdpa_mask)?;
        let out = out
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, l, self.n_heads * self.head_dim])?;
        self.o_proj.forward(&out)
    }

    /// Bidirectional decoder pass over the canvas. Reads the encoder KV
    /// from `cache` (sliding layers: only the last `window - 1` positions)
    /// and never writes to it. No mask: every canvas position attends to
    /// all (windowed) encoder positions and every other canvas position.
    pub fn forward_decoder(&mut self, x: &Array, cache: &KVCache) -> Result<Array, Exception> {
        let b = x.shape()[0];
        let l = x.shape()[1];
        let offset = cache.offset();
        let (q, k, v) = self.qkv(x, offset)?;

        let (keys, values) = match cache.current_kv() {
            Some((mut enc_k, mut enc_v)) => {
                if self.is_sliding {
                    let window = (self.sliding_window - 1).max(0);
                    let enc_len = enc_k.shape()[2];
                    if window > 0 && enc_len > window {
                        enc_k = enc_k.index((.., .., enc_len - window.., ..));
                        enc_v = enc_v.index((.., .., enc_len - window.., ..));
                    }
                }
                (
                    ops::concatenate_axis(&[&enc_k, &k], 2)?,
                    ops::concatenate_axis(&[&enc_v, &v], 2)?,
                )
            }
            None => (k, v),
        };

        let out = scaled_dot_product_attention::<KVCache>(q, keys, values, None, 1.0, None)?;
        let out = out
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, l, self.n_heads * self.head_dim])?;
        self.o_proj.forward(&out)
    }
}

// ============================================================================
// Decoder layer (dual-FFN MoE block)
// ============================================================================

pub struct DiffusionLayer {
    pub is_sliding: bool,
    pub self_attn: DiffusionAttention,
    pub mlp: DenseMlp,
    pub router: Router,
    pub experts: SwitchGluExperts,
    pub input_layernorm: nn::RmsNorm,
    pub post_attention_layernorm: nn::RmsNorm,
    pub pre_feedforward_layernorm: nn::RmsNorm,
    pub post_feedforward_layernorm: nn::RmsNorm,
    pub post_feedforward_layernorm_1: nn::RmsNorm,
    pub post_feedforward_layernorm_2: nn::RmsNorm,
    pub pre_feedforward_layernorm_2: nn::RmsNorm,
    /// Whole-stream multiplier at layer end (decoder mode).
    pub layer_scalar: Array,
    /// Override used in encoder mode (the only encoder-specific weights).
    pub encoder_layer_scalar: Array,
}

impl DiffusionLayer {
    /// Dual-FFN tail shared by both modes: `h` is the post-attention
    /// residual stream; returns `h + post_ffn_norm(mlp_branch + moe_branch)`.
    fn ffn(&mut self, h: &Array) -> Result<Array, Exception> {
        let residual = h;

        let h1 = self.pre_feedforward_layernorm.forward(h)?;
        let h1 = self.mlp.forward(&h1)?;
        let h1 = self.post_feedforward_layernorm_1.forward(&h1)?;

        let shape = h.shape().to_vec();
        let hidden = *shape.last().unwrap();
        let flat = residual.reshape(&[-1, hidden])?;
        let (_, top_k_weights, top_k_index) = self.router.forward(&flat)?;
        let h2 = self.pre_feedforward_layernorm_2.forward(&flat)?;
        let h2 = self
            .experts
            .forward_topk(&h2, &top_k_index, &top_k_weights)?
            .as_dtype(h.dtype())?;
        let h2 = h2.reshape(&shape)?;
        let h2 = self.post_feedforward_layernorm_2.forward(&h2)?;

        let out = self.post_feedforward_layernorm.forward(&h1.add(&h2)?)?;
        residual.add(&out)
    }

    fn attn_block(
        &mut self,
        x: &Array,
        attn_out: Array,
    ) -> Result<Array, Exception> {
        let attn_out = self.post_attention_layernorm.forward(&attn_out)?;
        x.add(&attn_out)
    }

    pub fn forward_encoder(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: &mut KVCache,
    ) -> Result<Array, Exception> {
        let h = self.input_layernorm.forward(x)?;
        let attn = self.self_attn.forward_encoder(&h, mask, cache)?;
        let h = self.attn_block(x, attn)?;
        let h = self.ffn(&h)?;
        let scalar = self.encoder_layer_scalar.as_dtype(h.dtype())?;
        h.multiply(&scalar)
    }

    pub fn forward_decoder(&mut self, x: &Array, cache: &KVCache) -> Result<Array, Exception> {
        let h = self.input_layernorm.forward(x)?;
        let attn = self.self_attn.forward_decoder(&h, cache)?;
        let h = self.attn_block(x, attn)?;
        let h = self.ffn(&h)?;
        let scalar = self.layer_scalar.as_dtype(h.dtype())?;
        h.multiply(&scalar)
    }
}

// ============================================================================
// Model
// ============================================================================

pub struct DiffusionGemmaModel {
    pub config: DiffusionGemmaConfig,
    pub layers: Vec<DiffusionLayer>,
    pub embed_tokens: MaybeQuantized<nn::Embedding>,
    embed_scale: Array,
    pub norm: nn::RmsNorm,
    // SelfConditioning: post_norm(inputs_embeds + mlp(pre_norm(signal)))
    self_cond_pre_norm: nn::RmsNorm,
    self_cond_mlp: DenseMlp,
    self_cond_post_norm: UnscaledRmsNorm,
    softcap: f32,
}

impl DiffusionGemmaModel {
    pub fn new_cache(&self) -> Vec<KVCache> {
        (0..self.layers.len()).map(|_| KVCache::new()).collect()
    }

    pub fn canvas_length(&self) -> i32 {
        self.config.canvas_length
    }

    pub fn vocab_size(&self) -> i32 {
        self.config.text_config.vocab_size
    }

    fn embed(&mut self, ids: &Array) -> Result<Array, Exception> {
        let e = self.embed_tokens.forward(ids)?;
        let scale = self.embed_scale.as_dtype(e.dtype())?;
        e.multiply(&scale)
    }

    /// Causal encoder pass over committed token ids `[1, L]`; appends K/V
    /// to `cache`. The normed hidden output is discarded — only the KV
    /// matters for subsequent canvas decoding.
    pub fn encode(&mut self, ids: &Array, cache: &mut [KVCache]) -> Result<(), Exception> {
        let l = ids.shape()[1];
        let offset = cache[0].offset();
        let mut h = self.embed(ids)?;
        let window = self.config.text_config.sliding_window;
        for (layer, c) in self.layers.iter_mut().zip(cache.iter_mut()) {
            let mask = if l > 1 {
                let win = layer.is_sliding.then_some(window);
                Some(create_causal_mask(l, Some(offset), win, None)?)
            } else {
                None
            };
            h = layer.forward_encoder(&h, mask.as_ref(), c)?;
        }
        Ok(())
    }

    /// Canvas embedding + self-conditioning. NOTE: the self-conditioning
    /// post-norm (unscaled RMS) is applied even when no soft embeddings
    /// are present (the signal is zero, but the norm still runs over the
    /// canvas embeddings).
    fn embed_canvas(
        &mut self,
        canvas: &Array,
        soft_embeddings: Option<&Array>,
    ) -> Result<Array, Exception> {
        let e = self.embed(canvas)?;
        let signal = match soft_embeddings {
            Some(soft) => {
                let normed = self.self_cond_pre_norm.forward(&soft.as_dtype(e.dtype())?)?;
                self.self_cond_mlp.forward(&normed)?
            }
            None => ops::zeros_like(&e)?,
        };
        self.self_cond_post_norm.forward(&e.add(&signal)?)
    }

    /// Bidirectional decoder pass over `canvas` `[1, Lc]` against the
    /// encoder KV. Returns softcapped fp32 logits `[1, Lc, V]`.
    pub fn decode_canvas(
        &mut self,
        canvas: &Array,
        cache: &[KVCache],
        soft_embeddings: Option<&Array>,
    ) -> Result<Array, Exception> {
        let mut h = self.embed_canvas(canvas, soft_embeddings)?;
        for (layer, c) in self.layers.iter_mut().zip(cache.iter()) {
            h = layer.forward_decoder(&h, c)?;
        }
        let h = self.norm.forward(&h)?;
        let logits = mq_embedding_as_linear(&mut self.embed_tokens, &h)?;
        let cap = array!(self.softcap);
        let logits = logits.as_dtype(Dtype::Float32)?;
        ops::tanh(&logits.divide(&cap)?)?.multiply(&cap)
    }

    /// Dequantized `[V, H]` embedding table for self-conditioning soft
    /// embeddings (probs @ E). Dequantized once per generation call —
    /// `quantized_matmul(transpose=false)` is several times slower at this
    /// shape per the mlx-vlm reference.
    pub fn dequant_embed_weight(&self) -> Result<Array, Exception> {
        match &self.embed_tokens {
            MaybeQuantized::Original(e) => Ok(e.weight.as_ref().clone()),
            MaybeQuantized::Quantized(qe) => ops::dequantize(
                &*qe.inner.weight,
                &*qe.scales,
                &*qe.biases,
                qe.group_size,
                qe.bits,
                None::<&str>,
            ),
        }
    }

    pub fn embed_scale_f32(&self) -> Result<f32, Exception> {
        Ok(self.embed_scale.as_dtype(Dtype::Float32)?.item::<f32>())
    }
}

// ============================================================================
// Loading
// ============================================================================

fn make_qsl_split(
    weights: &HashMap<String, Array>,
    prefix: &str,
    num_experts: i32,
    input_dims: i32,
    output_dims: i32,
    group_size: i32,
) -> Result<(QuantizedSwitchLinear, QuantizedSwitchLinear), Exception> {
    // Fused gate_up: weight [E, 2*I, packed_in], scales/biases [E, 2*I, groups].
    // Output rows are independent of the (input-axis) quantization packing, so
    // slicing rows 0..I / I..2I splits gate and up exactly.
    let get = |suffix: &str| -> Result<Array, Exception> {
        weights
            .get(&format!("{prefix}.{suffix}"))
            .cloned()
            .ok_or_else(|| Exception::custom(format!("missing weight: {prefix}.{suffix}")))
    };
    let w = get("weight")?;
    let s = get("scales")?;
    let b = get("biases")?;
    let w_cols = *w.shape().last().unwrap() as i64;
    let s_cols = *s.shape().last().unwrap() as i64;
    let bits = (w_cols * 32 / (s_cols * group_size as i64)) as i32;

    let half = output_dims;
    let split = |arr: &Array| -> Result<(Array, Array), Exception> {
        let gate = arr.index((.., ..half, ..)).contiguous()?;
        let up = arr.index((.., half.., ..)).contiguous()?;
        Ok((gate, up))
    };
    let (gw, uw) = split(&w)?;
    let (gs, us) = split(&s)?;
    let (gb, ub) = split(&b)?;

    let make = |w: Array, s: Array, b: Array| QuantizedSwitchLinear {
        num_experts,
        input_dims,
        output_dims,
        group_size,
        bits,
        weight: Param::new(w),
        scales: Param::new(s),
        biases: Param::new(b),
    };
    Ok((make(gw, gs, gb), make(uw, us, ub)))
}

fn make_qsl(
    weights: &HashMap<String, Array>,
    prefix: &str,
    num_experts: i32,
    input_dims: i32,
    output_dims: i32,
    group_size: i32,
) -> Result<QuantizedSwitchLinear, Exception> {
    let get = |suffix: &str| -> Result<Array, Exception> {
        weights
            .get(&format!("{prefix}.{suffix}"))
            .cloned()
            .ok_or_else(|| Exception::custom(format!("missing weight: {prefix}.{suffix}")))
    };
    let w = get("weight")?;
    let s = get("scales")?;
    let b = get("biases")?;
    let w_cols = *w.shape().last().unwrap() as i64;
    let s_cols = *s.shape().last().unwrap() as i64;
    let bits = (w_cols * 32 / (s_cols * group_size as i64)) as i32;
    Ok(QuantizedSwitchLinear {
        num_experts,
        input_dims,
        output_dims,
        group_size,
        bits,
        weight: Param::new(w),
        scales: Param::new(s),
        biases: Param::new(b),
    })
}

pub fn load_diffusion_model(model_dir: impl AsRef<Path>) -> Result<DiffusionGemmaModel, Error> {
    let model_dir = model_dir.as_ref();
    let config_text = std::fs::read_to_string(model_dir.join("config.json"))
        .map_err(|e| Error::Model(format!("read config.json: {e}")))?;
    let config: DiffusionGemmaConfig = serde_json::from_str(&config_text)
        .map_err(|e| Error::Model(format!("parse diffusion_gemma config: {e}")))?;
    let weights = load_all_weights_unfiltered(model_dir)?;
    build_diffusion_model(config, &weights)
}

/// Build the model from an already-loaded flat weight map. Split out of
/// [`load_diffusion_model`] so tests can drive it with synthetic weights.
pub fn build_diffusion_model(
    config: DiffusionGemmaConfig,
    weights: &HashMap<String, Array>,
) -> Result<DiffusionGemmaModel, Error> {
    let tc = config.text_config.clone();
    let quant = config.quantization.clone();
    let eps = tc.rms_norm_eps;
    let activation = GemmaActivation::from_name(&tc.hidden_activation)?;
    let group_size = quant.as_ref().map(|q| q.group_size).unwrap_or(64);

    let rms = |key: &str| -> Result<nn::RmsNorm, Error> {
        Ok(make_rms_norm(get_weight(weights, key)?, eps))
    };

    let mut layers = Vec::with_capacity(tc.num_hidden_layers as usize);
    for i in 0..tc.num_hidden_layers {
        let p = format!("model.decoder.layers.{i}");
        let layer_type = tc
            .layer_types
            .get(i as usize)
            .map(String::as_str)
            .unwrap_or("full_attention");
        let is_sliding = layer_type == "sliding_attention";
        let (head_dim, n_kv_heads, rope_spec) = if is_sliding {
            (
                tc.head_dim,
                tc.num_key_value_heads,
                &tc.rope_parameters.sliding_attention,
            )
        } else {
            (
                tc.global_head_dim.unwrap_or(tc.head_dim),
                tc.num_global_key_value_heads
                    .unwrap_or(tc.num_key_value_heads),
                &tc.rope_parameters.full_attention,
            )
        };
        let rope = match rope_spec.rope_type.as_deref() {
            Some("proportional") => GemmaRope::Proportional(ProportionalRope::new(
                head_dim,
                rope_spec.rope_theta,
                rope_spec.partial_rotary_factor,
            )),
            _ => GemmaRope::Standard(
                nn::RopeBuilder::new(head_dim)
                    .base(rope_spec.rope_theta)
                    .traditional(false)
                    .build()
                    .map_err(|e| Error::Model(format!("rope build: {e}")))?,
            ),
        };

        let v_proj = if get_weight_optional(weights, &format!("{p}.self_attn.v_proj.weight"))
            .is_some()
        {
            Some(make_mq_linear(
                &weights,
                &format!("{p}.self_attn.v_proj"),
                quant.as_ref(),
            )?)
        } else {
            None
        };

        let self_attn = DiffusionAttention {
            is_sliding,
            n_heads: tc.num_attention_heads,
            n_kv_heads,
            head_dim,
            sliding_window: tc.sliding_window,
            q_proj: make_mq_linear(weights, &format!("{p}.self_attn.q_proj"), quant.as_ref())?,
            k_proj: make_mq_linear(weights, &format!("{p}.self_attn.k_proj"), quant.as_ref())?,
            v_proj,
            o_proj: make_mq_linear(weights, &format!("{p}.self_attn.o_proj"), quant.as_ref())?,
            q_norm: rms(&format!("{p}.self_attn.q_norm.weight"))?,
            k_norm: rms(&format!("{p}.self_attn.k_norm.weight"))?,
            v_norm: UnscaledRmsNorm::new(eps),
            rope,
        };

        let mlp = DenseMlp {
            gate_proj: make_mq_linear(weights, &format!("{p}.mlp.gate_proj"), quant.as_ref())?,
            up_proj: make_mq_linear(weights, &format!("{p}.mlp.up_proj"), quant.as_ref())?,
            down_proj: make_mq_linear(weights, &format!("{p}.mlp.down_proj"), quant.as_ref())?,
            activation,
        };

        let router = Router {
            hidden_size: tc.hidden_size,
            top_k_experts: tc.top_k_experts,
            scalar_root_size: (tc.hidden_size as f32).powf(-0.5),
            proj: make_mq_linear(weights, &format!("{p}.router.proj"), quant.as_ref())?,
            scale: Param::new(get_weight(weights, &format!("{p}.router.scale"))?),
            per_expert_scale: Param::new(get_weight(
                &weights,
                &format!("{p}.router.per_expert_scale"),
            )?),
            norm: UnscaledRmsNorm::new(eps),
        };

        let (gate_proj, up_proj) = make_qsl_split(
            weights,
            &format!("{p}.experts.gate_up_proj"),
            tc.num_experts,
            tc.hidden_size,
            tc.moe_intermediate_size,
            group_size,
        )
        .map_err(|e| Error::Model(e.to_string()))?;
        let down_proj = make_qsl(
            weights,
            &format!("{p}.experts.down_proj"),
            tc.num_experts,
            tc.moe_intermediate_size,
            tc.hidden_size,
            group_size,
        )
        .map_err(|e| Error::Model(e.to_string()))?;
        let experts = SwitchGluExperts {
            hidden_size: tc.hidden_size,
            intermediate_size: tc.moe_intermediate_size,
            switch_glu: SwitchGLU {
                gate_proj,
                up_proj,
                down_proj,
                activation,
            },
        };

        layers.push(DiffusionLayer {
            is_sliding,
            self_attn,
            mlp,
            router,
            experts,
            input_layernorm: rms(&format!("{p}.input_layernorm.weight"))?,
            post_attention_layernorm: rms(&format!("{p}.post_attention_layernorm.weight"))?,
            pre_feedforward_layernorm: rms(&format!("{p}.pre_feedforward_layernorm.weight"))?,
            post_feedforward_layernorm: rms(&format!("{p}.post_feedforward_layernorm.weight"))?,
            post_feedforward_layernorm_1: rms(&format!(
                "{p}.post_feedforward_layernorm_1.weight"
            ))?,
            post_feedforward_layernorm_2: rms(&format!(
                "{p}.post_feedforward_layernorm_2.weight"
            ))?,
            pre_feedforward_layernorm_2: rms(&format!(
                "{p}.pre_feedforward_layernorm_2.weight"
            ))?,
            layer_scalar: get_weight(weights, &format!("{p}.layer_scalar"))?,
            encoder_layer_scalar: get_weight(
                &weights,
                &format!("model.encoder.language_model.layers.{i}.layer_scalar"),
            )?,
        });
    }

    let embed_tokens = make_mq_embedding(weights, "model.decoder.embed_tokens", quant.as_ref())?;
    let embed_scale = Array::from((tc.hidden_size as f32).sqrt())
        .as_dtype(Dtype::Bfloat16)
        .map_err(|e| Error::Model(e.to_string()))?;

    let self_cond_mlp = DenseMlp {
        gate_proj: make_mq_linear(
            weights,
            "model.decoder.self_conditioning.gate_proj",
            quant.as_ref(),
        )?,
        up_proj: make_mq_linear(
            weights,
            "model.decoder.self_conditioning.up_proj",
            quant.as_ref(),
        )?,
        down_proj: make_mq_linear(
            weights,
            "model.decoder.self_conditioning.down_proj",
            quant.as_ref(),
        )?,
        activation,
    };

    Ok(DiffusionGemmaModel {
        softcap: tc.final_logit_softcapping,
        layers,
        embed_tokens,
        embed_scale,
        norm: rms("model.decoder.norm.weight")?,
        self_cond_pre_norm: rms("model.decoder.self_conditioning.pre_norm.weight")?,
        self_cond_mlp,
        self_cond_post_norm: UnscaledRmsNorm::new(eps),
        config,
    })
}

// ============================================================================
// Generation (entropy-bound sampler, greedy)
// ============================================================================

#[derive(Debug, Clone)]
pub struct DiffusionGenerateOptions {
    pub max_tokens: usize,
    pub max_denoising_steps: usize,
    pub min_canvas_length: i32,
    pub t_min: f32,
    pub t_max: f32,
    pub entropy_bound: f32,
    pub confidence_threshold: f32,
    pub stability_threshold: usize,
}

impl Default for DiffusionGenerateOptions {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            max_denoising_steps: 48,
            min_canvas_length: 64,
            t_min: 0.4,
            t_max: 0.8,
            entropy_bound: 0.1,
            confidence_threshold: 0.005,
            stability_threshold: 1,
        }
    }
}

impl DiffusionGenerateOptions {
    /// Defaults overridden by the checkpoint's `generation_config`.
    pub fn from_config(config: &DiffusionGemmaConfig) -> Self {
        let mut opts = Self::default();
        let Some(gc) = config.generation_config.as_ref() else {
            return opts;
        };
        if let Some(v) = gc.get("max_denoising_steps").and_then(|v| v.as_u64()) {
            opts.max_denoising_steps = v as usize;
        }
        if let Some(v) = gc.get("max_new_tokens").and_then(|v| v.as_u64()) {
            opts.max_tokens = v as usize;
        }
        if let Some(v) = gc.get("t_min").and_then(|v| v.as_f64()) {
            opts.t_min = v as f32;
        }
        if let Some(v) = gc.get("t_max").and_then(|v| v.as_f64()) {
            opts.t_max = v as f32;
        }
        if let Some(v) = gc.get("confidence_threshold").and_then(|v| v.as_f64()) {
            opts.confidence_threshold = v as f32;
        }
        if let Some(v) = gc.get("stability_threshold").and_then(|v| v.as_u64()) {
            opts.stability_threshold = v as usize;
        }
        if let Some(v) = gc
            .get("sampler_config")
            .and_then(|s| s.get("entropy_bound"))
            .and_then(|v| v.as_f64())
        {
            opts.entropy_bound = v as f32;
        }
        opts
    }
}

#[derive(Debug, Clone, Default)]
pub struct DiffusionGenStats {
    pub prefill_s: f64,
    pub decode_s: f64,
    pub emitted_tokens: usize,
    pub canvases: usize,
    pub denoise_steps: usize,
    /// canvas_length × steps summed — the actual decoder work done.
    pub work_tokens: usize,
}

impl DiffusionGenStats {
    pub fn tokens_per_sec(&self) -> f64 {
        if self.decode_s > 0.0 {
            self.emitted_tokens as f64 / self.decode_s
        } else {
            0.0
        }
    }
}

fn random_canvas(canvas_len: i32, vocab: i32) -> Result<Array, Exception> {
    mlx_rs::random::randint::<_, i32>(0, vocab, &[1, canvas_len][..], None)
        .map_err(|e| Exception::custom(format!("randint: {e}")))
}

/// Per-position entropy of softcapped fp32 logits `[1, Lc, V]` → `[1, Lc]`.
fn token_entropy(logits: &Array) -> Result<Array, Exception> {
    let lse = ops::logsumexp_axis(logits, -1, true)?;
    let log_probs = logits.subtract(&lse)?;
    let probs = ops::exp(&log_probs)?;
    probs.multiply(&log_probs)?.sum_axis(-1, false)?.negative()
}

/// Entropy-bound transfer mask: sort positions by entropy ascending, accept
/// while `cumsum(entropy) - cummax(entropy) <= bound`. Returns bool `[1, Lc]`.
fn entropy_transfer_mask(entropy: &Array, bound: f32) -> Result<Array, Exception> {
    let sorted_idx = ops::argsort_axis(entropy, -1)?;
    let sorted_entropy = ops::indexing::take_along_axis(entropy, &sorted_idx, -1)?;
    let cum = ops::cumsum(&sorted_entropy, -1, None, None)?;
    let cmax = ops::cummax(&sorted_entropy, -1, None, None)?;
    let sel_sorted = cum.subtract(&cmax)?.le(&array!(bound))?;
    let zeros = ops::zeros_like(&sel_sorted)?;
    ops::indexing::put_along_axis(&zeros, &sorted_idx, &sel_sorted, -1)
}

/// Generate with the entropy-bound sampler at temperature 0 (greedy
/// denoising; the t_min..t_max schedule still shapes the entropy used for
/// acceptance). Calls `on_block(&tokens)` after each committed canvas with
/// the tokens emitted from it (post EOS-truncation).
pub fn diffusion_generate(
    model: &mut DiffusionGemmaModel,
    prompt_ids: &[i32],
    opts: &DiffusionGenerateOptions,
    eos: &[i32],
    mut on_block: impl FnMut(&[i32]),
) -> Result<(Vec<i32>, DiffusionGenStats), Exception> {
    let mut stats = DiffusionGenStats::default();
    let vocab = model.vocab_size();
    let model_canvas = model.canvas_length();
    let embed_scale = model.embed_scale_f32()?;
    let embed_weight = model.dequant_embed_weight()?;

    let mut cache = model.new_cache();

    let prefill_start = std::time::Instant::now();
    let prompt = Array::from_slice(prompt_ids, &[1, prompt_ids.len() as i32]);
    model.encode(&prompt, &mut cache)?;
    // Force the prefill graph so prefill/decode timing is honest.
    for c in &cache {
        if let Some((k, v)) = c.current_kv() {
            eval([&k, &v])?;
        }
    }
    stats.prefill_s = prefill_start.elapsed().as_secs_f64();

    let decode_start = std::time::Instant::now();
    let mut emitted: Vec<i32> = Vec::new();
    let mut finished = false;

    while !finished && emitted.len() < opts.max_tokens {
        stats.canvases += 1;
        let remaining = (opts.max_tokens - emitted.len()) as i32;
        let canvas_len = model_canvas.min(remaining.max(opts.min_canvas_length));

        let mut canvas = random_canvas(canvas_len, vocab)?;
        let mut soft_embeddings: Option<Array> = None;
        let mut history: Vec<Array> = Vec::new();
        let mut argmax_canvas = canvas.clone();

        for cur_step in (1..=opts.max_denoising_steps).rev() {
            stats.denoise_steps += 1;
            stats.work_tokens += canvas_len as usize;

            let logits = model.decode_canvas(&canvas, &cache, soft_embeddings.as_ref())?;
            // Linear temperature schedule: hot early, cool late. Doesn't
            // change the argmax; shapes the entropies the sampler sees.
            let t = opts.t_min
                + (opts.t_max - opts.t_min) * (cur_step as f32 / opts.max_denoising_steps as f32);
            let logits = logits.divide(&array!(t))?;

            argmax_canvas = mlx_rs::argmax_axis!(&logits, -1)?.as_dtype(Dtype::Int32)?;
            if cur_step == 1 {
                break;
            }

            let entropy = token_entropy(&logits)?;
            let accept = entropy_transfer_mask(&entropy, opts.entropy_bound)?;
            canvas = ops::r#where(&accept, &argmax_canvas, &random_canvas(canvas_len, vocab)?)?;

            // Early stop: argmax canvas unchanged across `stability_threshold`
            // prior steps AND mean entropy below the confidence threshold.
            let stable = history.len() == opts.stability_threshold && {
                let mut all_same = true;
                for prev in &history {
                    let same = ops::all(&argmax_canvas.eq(prev)?, None)?.item::<bool>();
                    if !same {
                        all_same = false;
                        break;
                    }
                }
                all_same
            };
            history.push(argmax_canvas.clone());
            if history.len() > opts.stability_threshold {
                history.remove(0);
            }
            if stable {
                let mean_entropy = ops::mean(&entropy, None)?.item::<f32>();
                if mean_entropy < opts.confidence_threshold {
                    break;
                }
            }

            // Self-conditioning soft embeddings for the next step:
            // softmax(logits) @ E * embed_scale.
            let probs = ops::softmax_axis(&logits, -1, Some(true))?;
            let soft = ops::matmul(&probs.as_dtype(embed_weight.dtype())?, &embed_weight)?
                .multiply(&array!(embed_scale))?;
            soft_embeddings = Some(soft);
        }

        // Commit the argmax canvas.
        let row = argmax_canvas.index((0, ..)).contiguous()?;
        eval([&row])?;
        let tokens: Vec<i32> = row.as_slice::<i32>().to_vec();
        let mut block: Vec<i32> = Vec::with_capacity(tokens.len());
        for tok in tokens {
            if eos.contains(&tok) {
                finished = true;
                break;
            }
            block.push(tok);
            if emitted.len() + block.len() >= opts.max_tokens {
                finished = true;
                break;
            }
        }
        emitted.extend_from_slice(&block);
        on_block(&block);

        if !finished {
            // Commit the full canvas into the encoder KV (causal pass) so
            // the next canvas attends to it.
            model.encode(&argmax_canvas, &mut cache)?;
        }
    }

    stats.decode_s = decode_start.elapsed().as_secs_f64();
    stats.emitted_tokens = emitted.len();
    Ok((emitted, stats))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quantize_into(
        weights: &mut HashMap<String, Array>,
        prefix: &str,
        out: i32,
        inp: i32,
        group_size: i32,
        bits: i32,
    ) {
        let dense = mlx_rs::random::uniform::<_, f32>(-0.5, 0.5, &[out, inp][..], None)
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let (w, s, b) =
            mlx_rs::ops::quantize(&dense, group_size, bits, None::<&'static str>).unwrap();
        weights.insert(format!("{prefix}.weight"), w);
        weights.insert(format!("{prefix}.scales"), s);
        weights.insert(format!("{prefix}.biases"), b);
    }

    fn quantize_stacked_into(
        weights: &mut HashMap<String, Array>,
        prefix: &str,
        experts: i32,
        out: i32,
        inp: i32,
        group_size: i32,
        bits: i32,
    ) {
        let dense =
            mlx_rs::random::uniform::<_, f32>(-0.5, 0.5, &[experts, out, inp][..], None)
                .unwrap()
                .as_dtype(Dtype::Bfloat16)
                .unwrap();
        let (w, s, b) =
            mlx_rs::ops::quantize(&dense, group_size, bits, None::<&'static str>).unwrap();
        weights.insert(format!("{prefix}.weight"), w);
        weights.insert(format!("{prefix}.scales"), s);
        weights.insert(format!("{prefix}.biases"), b);
    }

    fn ones_into(weights: &mut HashMap<String, Array>, key: &str, dim: i32) {
        let arr = Array::ones::<f32>(&[dim])
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        weights.insert(key.to_string(), arr);
    }

    fn tiny_config() -> DiffusionGemmaConfig {
        serde_json::from_str(
            r#"{
            "canvas_length": 8,
            "eos_token_id": [1],
            "generation_config": {"max_denoising_steps": 3, "t_min": 0.4, "t_max": 0.8,
                                  "confidence_threshold": 0.005, "stability_threshold": 1,
                                  "sampler_config": {"entropy_bound": 0.1}},
            "quantization": {"group_size": 32, "bits": 4},
            "text_config": {
                "vocab_size": 96, "hidden_size": 64, "intermediate_size": 64,
                "moe_intermediate_size": 32, "num_hidden_layers": 2,
                "num_attention_heads": 4, "num_key_value_heads": 2,
                "num_global_key_value_heads": 1, "head_dim": 16, "global_head_dim": 32,
                "hidden_activation": "gelu_pytorch_tanh", "rms_norm_eps": 1e-6,
                "layer_types": ["sliding_attention", "full_attention"],
                "sliding_window": 4, "final_logit_softcapping": 30.0,
                "num_experts": 4, "top_k_experts": 2,
                "rope_parameters": {
                    "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
                    "full_attention": {"rope_theta": 1000000.0, "rope_type": "proportional",
                                        "partial_rotary_factor": 0.25}
                }
            }
        }"#,
        )
        .expect("tiny config parses")
    }

    fn tiny_weights(cfg: &DiffusionGemmaConfig) -> HashMap<String, Array> {
        let tc = &cfg.text_config;
        let gs = 32;
        let bits = 4;
        let h = tc.hidden_size;
        let mut w = HashMap::new();

        quantize_into(&mut w, "model.decoder.embed_tokens", tc.vocab_size, h, gs, bits);
        ones_into(&mut w, "model.decoder.norm.weight", h);

        ones_into(&mut w, "model.decoder.self_conditioning.pre_norm.weight", h);
        quantize_into(
            &mut w,
            "model.decoder.self_conditioning.gate_proj",
            tc.intermediate_size,
            h,
            gs,
            bits,
        );
        quantize_into(
            &mut w,
            "model.decoder.self_conditioning.up_proj",
            tc.intermediate_size,
            h,
            gs,
            bits,
        );
        quantize_into(
            &mut w,
            "model.decoder.self_conditioning.down_proj",
            h,
            tc.intermediate_size,
            gs,
            bits,
        );

        for i in 0..tc.num_hidden_layers {
            let p = format!("model.decoder.layers.{i}");
            let is_sliding = tc.layer_types[i as usize] == "sliding_attention";
            let (d, kv) = if is_sliding {
                (tc.head_dim, tc.num_key_value_heads)
            } else {
                (
                    tc.global_head_dim.unwrap(),
                    tc.num_global_key_value_heads.unwrap(),
                )
            };
            quantize_into(&mut w, &format!("{p}.self_attn.q_proj"), tc.num_attention_heads * d, h, gs, bits);
            quantize_into(&mut w, &format!("{p}.self_attn.k_proj"), kv * d, h, gs, bits);
            if is_sliding {
                quantize_into(&mut w, &format!("{p}.self_attn.v_proj"), kv * d, h, gs, bits);
            }
            quantize_into(&mut w, &format!("{p}.self_attn.o_proj"), h, tc.num_attention_heads * d, gs, bits);
            ones_into(&mut w, &format!("{p}.self_attn.q_norm.weight"), d);
            ones_into(&mut w, &format!("{p}.self_attn.k_norm.weight"), d);
            for norm in [
                "input_layernorm",
                "post_attention_layernorm",
                "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
                "post_feedforward_layernorm_1",
                "post_feedforward_layernorm_2",
                "pre_feedforward_layernorm_2",
            ] {
                ones_into(&mut w, &format!("{p}.{norm}.weight"), h);
            }
            quantize_into(&mut w, &format!("{p}.mlp.gate_proj"), tc.intermediate_size, h, gs, bits);
            quantize_into(&mut w, &format!("{p}.mlp.up_proj"), tc.intermediate_size, h, gs, bits);
            quantize_into(&mut w, &format!("{p}.mlp.down_proj"), h, tc.intermediate_size, gs, bits);

            quantize_into(&mut w, &format!("{p}.router.proj"), tc.num_experts, h, gs, bits);
            ones_into(&mut w, &format!("{p}.router.scale"), h);
            ones_into(&mut w, &format!("{p}.router.per_expert_scale"), tc.num_experts);

            quantize_stacked_into(
                &mut w,
                &format!("{p}.experts.gate_up_proj"),
                tc.num_experts,
                2 * tc.moe_intermediate_size,
                h,
                gs,
                bits,
            );
            quantize_stacked_into(
                &mut w,
                &format!("{p}.experts.down_proj"),
                tc.num_experts,
                h,
                tc.moe_intermediate_size,
                gs,
                bits,
            );

            let scalar = Array::ones::<f32>(&[1])
                .unwrap()
                .as_dtype(Dtype::Bfloat16)
                .unwrap();
            w.insert(format!("{p}.layer_scalar"), scalar.clone());
            w.insert(
                format!("model.encoder.language_model.layers.{i}.layer_scalar"),
                scalar,
            );
        }
        w
    }

    #[test]
    fn entropy_transfer_mask_selects_lowest_entropy() {
        // `cumsum - cummax <= bound` admits the running-max element for
        // free and budgets the *rest*. Sorted [0.01, 4.0, 5.0]: the diffs
        // are [0, 0.01, 4.01] → 5.0 is rejected, 4.0 still accepted.
        let entropy = Array::from_slice(&[5.0_f32, 4.0, 0.01], &[1, 3]);
        let mask = entropy_transfer_mask(&entropy, 0.1).unwrap();
        eval([&mask]).unwrap();
        assert_eq!(mask.as_slice::<bool>(), &[false, true, true]);
    }

    #[test]
    fn synthetic_diffusion_generate_runs() {
        let _ = mlx_rs::random::seed(7);
        let cfg = tiny_config();
        let weights = tiny_weights(&cfg);
        let mut model = build_diffusion_model(cfg, &weights).expect("build model");

        let opts = DiffusionGenerateOptions {
            max_tokens: 8,
            max_denoising_steps: 3,
            min_canvas_length: 4,
            ..Default::default()
        };
        // EOS id outside the vocab so generation runs to max_tokens.
        let mut blocks = 0;
        let (emitted, stats) =
            diffusion_generate(&mut model, &[2, 3, 4], &opts, &[-1], |_| blocks += 1)
                .expect("generate");
        assert_eq!(emitted.len(), 8);
        assert!(stats.canvases >= 1 && blocks == stats.canvases);
        assert!(stats.denoise_steps >= stats.canvases);
        assert!(emitted.iter().all(|&t| (0..96).contains(&t)));
    }
}
