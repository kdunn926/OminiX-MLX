//! Gemma4 assistant (drafter) model — `Gemma4AssistantForCausalLM`.
//!
//! Port of the upstream `transformers/models/gemma4_assistant/modeling_gemma4_assistant.py`
//! for use as the drafter side of a classical drafter+verifier speculative
//! decoding pair. Distinct from the MTP-head path.
//!
//! Architecture (4-layer recurrent cross-attender, NOT a mini-LM):
//! - Input per drafter step:
//!     `inputs_embeds = concat(target_embed(prev_token), recurrent_hidden)`
//!     shape `[B, 1, 2 * backbone_hidden]`. `recurrent_hidden` is the target's
//!     last hidden on step 0, then this module's own `post_projection` output
//!     thereafter.
//! - `pre_projection`: `[2*5376 → 1024]`. Quantized (Q6 affine g=64) in this
//!   checkpoint.
//! - 4 decoder layers, only `q_proj` + `q_norm` + `o_proj` (no K/V weights).
//!   K and V come from the target's `shared_kv_states` dict, keyed by layer
//!   type. Last `sliding_attention` layer of target → assistant sliding
//!   layers; last `full_attention` layer of target → assistant full layer.
//!   K is consumed **as-is** — the target stored it post-norm, post-rotation;
//!   the assistant does not re-rotate it. V is also as-stored (post v_norm).
//! - Attention is **bidirectional cross-attention** against the target's K/V.
//!   q_len per step is 1, kv_len is the full prompt-so-far. No causal mask.
//! - `post_projection`: `[1024 → 5376]` — produces the recurrent hidden fed
//!   back into the next step's `inputs_embeds`.
//! - Output `logits`: tied `lm_head` over the assistant's own
//!   `embed_tokens [262144, 1024]`. NOT routed through the target's embed.
//! - Per-layer `layer_scalar` (single scalar) multiplied into the final
//!   hidden state at the END of each decoder layer.
//!
//! The model card field `use_ordered_embeddings: false` for our checkpoint
//! means the `MaskedEmbedder` centroid-based output path is unused — we
//! always run a plain tied lm_head over the full 262144-vocab.

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::{
    argmax_axis,
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::{Module, ModuleParameters as _, Param},
    nn,
    ops,
    quantization::MaybeQuantized,
    Array,
};
use mlx_rs_core::{
    utils::{scaled_dot_product_attention, SdpaMask},
    Error,
};
use serde::Deserialize;

use crate::model::{GemmaActivation, GemmaRope, ProportionalRope};

// ============================================================================
// Config
// ============================================================================

#[derive(Debug, Clone, Deserialize)]
pub struct AssistantTextConfig {
    pub vocab_size: i32,
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    #[serde(default)]
    pub num_global_key_value_heads: i32,
    pub head_dim: i32,
    #[serde(default)]
    pub global_head_dim: i32,
    pub sliding_window: i32,
    pub rms_norm_eps: f32,
    pub layer_types: Vec<String>,
    pub hidden_activation: String,
    #[serde(default)]
    pub attention_k_eq_v: bool,
    pub rope_parameters: RopeParameters,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    pub sliding_attention: RopeSpec,
    pub full_attention: RopeSpec,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeSpec {
    pub rope_type: String,
    pub rope_theta: f32,
    #[serde(default = "default_partial")]
    pub partial_rotary_factor: f32,
}

fn default_partial() -> f32 {
    1.0
}

#[derive(Debug, Clone, Deserialize)]
pub struct AssistantConfig {
    pub model_type: String,
    pub backbone_hidden_size: i32,
    pub text_config: AssistantTextConfig,
    pub quantization: Option<QuantSpec>,
    #[serde(default)]
    pub use_ordered_embeddings: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct QuantSpec {
    pub bits: i32,
    pub group_size: i32,
    pub mode: String,
}

impl AssistantConfig {
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self, Error> {
        let path = model_dir.as_ref().join("config.json");
        let txt = std::fs::read_to_string(&path)?;
        Ok(serde_json::from_str(&txt)?)
    }
}

// ============================================================================
// Attention — Q-only with bidirectional cross-attention over borrowed K/V
// ============================================================================

#[derive(Debug, Clone, ModuleParameters)]
pub struct AssistantAttention {
    pub is_sliding: bool,
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub sliding_window: i32,

    #[param]
    pub q_proj: MaybeQuantized<nn::Linear>,
    #[param]
    pub o_proj: MaybeQuantized<nn::Linear>,
    #[param]
    pub q_norm: nn::RmsNorm,

    rope: GemmaRope,
}

impl AssistantAttention {
    /// One drafter step. `hidden` has shape `[B, q_len, hidden_size]`,
    /// `shared_k`/`shared_v` have shape `[B, n_kv_heads, kv_len, head_dim]`
    /// — already-rotated, already-normed by the target. `position_offset`
    /// is the absolute position of the FIRST query token (== `kv_len - q_len`
    /// in the typical recurrent case).
    #[allow(non_snake_case)]
    pub fn forward(
        &mut self,
        hidden: &Array,
        position_offset: i32,
        shared_k: &Array,
        shared_v: &Array,
    ) -> Result<Array, Exception> {
        let shape = hidden.shape();
        let B = shape[0];
        let L = shape[1];

        let queries = self.q_proj.forward(hidden)?;
        let queries = self
            .q_norm
            .forward(&queries.reshape(&[B, L, self.n_heads, self.head_dim])?)?;
        let mut queries = queries.transpose_axes(&[0, 2, 1, 3])?;
        queries = self.rope.apply(&queries, position_offset)?;

        // Bidirectional cross-attention: q_len is small (usually 1), kv_len
        // covers the entire prompt-so-far. No causal mask — the drafter is
        // ALLOWED to attend to future kv positions, but in the recurrent
        // single-step case the kv only contains tokens up to `position_offset`
        // anyway, so the distinction is moot.
        //
        // For sliding-attention layers, restrict to the last `sliding_window`
        // KV positions to match target's training-time attention pattern.
        // Without this, on long-context prompts (kv_len > sliding_window) the
        // drafter mixes positional information target's sliding layer never saw.
        let (sk, sv) = if self.is_sliding
            && self.sliding_window > 0
            && shared_k.shape()[2] > self.sliding_window
            && std::env::var("MTPLX_PAIR_NO_SLIDING_MASK")
                .map(|v| v != "1")
                .unwrap_or(true)
        {
            use mlx_rs::ops::indexing::{Ellipsis, IndexOp, NewAxis};
            let _ = NewAxis;
            let w = self.sliding_window;
            (
                shared_k.index((Ellipsis, -w.., ..)),
                shared_v.index((Ellipsis, -w.., ..)),
            )
        } else {
            (shared_k.clone(), shared_v.clone())
        };
        let attn_out = scaled_dot_product_attention::<mlx_rs_core::cache::KVCache>(
            queries,
            sk,
            sv,
            None,
            self.scale,
            None::<SdpaMask>,
        )?
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[B, L, -1])?;

        self.o_proj.forward(&attn_out)
    }
}

// ============================================================================
// MLP
// ============================================================================

#[derive(Debug, Clone, ModuleParameters)]
pub struct AssistantMlp {
    #[param]
    pub gate_proj: MaybeQuantized<nn::Linear>,
    #[param]
    pub up_proj: MaybeQuantized<nn::Linear>,
    #[param]
    pub down_proj: MaybeQuantized<nn::Linear>,
    pub activation: GemmaActivation,
}

impl Module<&Array> for AssistantMlp {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Self::Output, Self::Error> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let activated = self.activation.apply(&gate)?.multiply(&up)?;
        self.down_proj.forward(&activated)
    }

    fn training_mode(&mut self, _mode: bool) {}
}

// ============================================================================
// Decoder Layer
// ============================================================================

#[derive(Debug, Clone, ModuleParameters)]
pub struct AssistantDecoderLayer {
    #[param]
    pub self_attn: AssistantAttention,
    #[param]
    pub mlp: AssistantMlp,
    #[param]
    pub input_layernorm: nn::RmsNorm,
    #[param]
    pub post_attention_layernorm: nn::RmsNorm,
    #[param]
    pub pre_feedforward_layernorm: nn::RmsNorm,
    #[param]
    pub post_feedforward_layernorm: nn::RmsNorm,
    #[param]
    pub layer_scalar: Param<Array>,
}

impl AssistantDecoderLayer {
    pub fn forward(
        &mut self,
        hidden: &Array,
        position_offset: i32,
        shared_k: &Array,
        shared_v: &Array,
    ) -> Result<Array, Exception> {
        let residual = hidden.clone();
        let attn_in = self.input_layernorm.forward(hidden)?;
        let attn_out =
            self.self_attn
                .forward(&attn_in, position_offset, shared_k, shared_v)?;
        let attn_out = self.post_attention_layernorm.forward(&attn_out)?;
        let mut h = residual.add(&attn_out)?;

        let residual = h.clone();
        let ff_in = self.pre_feedforward_layernorm.forward(&h)?;
        let ff_out = self.mlp.forward(&ff_in)?;
        let ff_out = self.post_feedforward_layernorm.forward(&ff_out)?;
        h = residual.add(&ff_out)?;

        // layer_scalar applied at the very end (matches HF reference).
        let scalar = (&*self.layer_scalar).as_dtype(h.dtype())?;
        h.multiply(&scalar)
    }
}

// ============================================================================
// Shared KV
// ============================================================================

/// K/V borrowed from the target's last layer of each attention type.
/// Each tensor is shape `[B, n_kv_heads_for_type, kv_len, head_dim_for_type]`,
/// already post-norm and post-rotation as stored in the target's KV cache.
#[derive(Debug, Clone)]
pub struct SharedKvStates {
    pub sliding_k: Array,
    pub sliding_v: Array,
    pub full_k: Array,
    pub full_v: Array,
}

impl SharedKvStates {
    pub fn kv_len(&self) -> i32 {
        self.sliding_k.shape()[2]
    }

    /// Slice the kv to the first `n` positions (used when fewer tokens are
    /// accepted than were drafted). Returns a new `SharedKvStates`.
    pub fn slice_kv(&self, n: i32) -> Result<Self, Exception> {
        use mlx_rs::ops::indexing::{Ellipsis, IndexOp};
        Ok(Self {
            sliding_k: self.sliding_k.index((Ellipsis, ..n, ..)),
            sliding_v: self.sliding_v.index((Ellipsis, ..n, ..)),
            full_k: self.full_k.index((Ellipsis, ..n, ..)),
            full_v: self.full_v.index((Ellipsis, ..n, ..)),
        })
    }
}

// ============================================================================
// AssistantModel
// ============================================================================

#[derive(Debug, Clone, ModuleParameters)]
pub struct AssistantModel {
    pub config: AssistantConfig,
    pub layer_types: Vec<LayerKind>,

    #[param]
    pub layers: Vec<AssistantDecoderLayer>,
    #[param]
    pub norm: nn::RmsNorm,
    #[param]
    pub embed_tokens: MaybeQuantized<nn::Embedding>,
    #[param]
    pub pre_projection: MaybeQuantized<nn::Linear>,
    #[param]
    pub post_projection: MaybeQuantized<nn::Linear>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    Sliding,
    Full,
}

impl LayerKind {
    fn parse(s: &str) -> Result<Self, Error> {
        match s {
            "sliding_attention" => Ok(Self::Sliding),
            "full_attention" => Ok(Self::Full),
            other => Err(Error::Model(format!(
                "Unknown layer type '{other}' for gemma4_assistant"
            ))),
        }
    }
}

/// Output of one assistant forward pass.
pub struct AssistantOutput {
    /// `[B, q_len, vocab_size]` — sampling logits from the tied lm_head.
    pub logits: Array,
    /// `[B, q_len, backbone_hidden_size]` — recurrent hidden to feed back
    /// as the second half of next step's `inputs_embeds`.
    pub last_hidden: Array,
}

impl AssistantModel {
    /// `inputs_embeds`: `[B, q_len, 2 * backbone_hidden]`. Caller is
    /// responsible for constructing this as
    /// `concat(target_embed(prev_token), recurrent_hidden)` on the last dim.
    ///
    /// `position_offset`: absolute position of the FIRST query token (this
    /// is the RoPE offset; for q_len=1 it equals `kv_len - 1` after the
    /// last accepted token has been added to the KV).
    pub fn forward(
        &mut self,
        inputs_embeds: &Array,
        position_offset: i32,
        kv: &SharedKvStates,
    ) -> Result<AssistantOutput, Exception> {
        let mut h = self.pre_projection.forward(inputs_embeds)?;

        for (i, layer) in self.layers.iter_mut().enumerate() {
            let (sk, sv) = match self.layer_types[i] {
                LayerKind::Sliding => (&kv.sliding_k, &kv.sliding_v),
                LayerKind::Full => (&kv.full_k, &kv.full_v),
            };
            h = layer.forward(&h, position_offset, sk, sv)?;
        }

        let inner = self.norm.forward(&h)?;
        let last_hidden = self.post_projection.forward(&inner)?;

        // Tied lm_head: logits = inner @ embed_tokens.weight^T
        let logits = match &mut self.embed_tokens {
            MaybeQuantized::Original(e) => {
                // Use weight transpose path.
                let w = e.weight.as_ref();
                inner.matmul(&w.transpose_axes(&[1, 0])?)?
            }
            MaybeQuantized::Quantized(qe) => qe.as_linear(&inner)?,
        };

        Ok(AssistantOutput { logits, last_hidden })
    }
}

// ============================================================================
// Loader
// ============================================================================

fn load_quantized_linear(
    weights: &HashMap<String, Array>,
    prefix: &str,
    group_size: i32,
    bits: i32,
) -> Result<MaybeQuantized<nn::Linear>, Error> {
    let weight = weights
        .get(&format!("{prefix}.weight"))
        .cloned()
        .ok_or_else(|| Error::Model(format!("Weight not found: {prefix}.weight")))?;
    let scales = weights
        .get(&format!("{prefix}.scales"))
        .cloned()
        .ok_or_else(|| Error::Model(format!("Weight not found: {prefix}.scales")))?;
    let biases = weights
        .get(&format!("{prefix}.biases"))
        .cloned()
        .ok_or_else(|| Error::Model(format!("Weight not found: {prefix}.biases")))?;

    let inner = nn::Linear {
        weight: Param::new(weight),
        bias: Param::new(None::<Array>),
    };
    let mut ql = nn::QuantizedLinear {
        group_size,
        bits,
        scales: Param::new(scales),
        biases: Param::new(biases),
        inner,
    };
    ql.freeze_parameters(true);
    Ok(MaybeQuantized::Quantized(ql))
}

fn load_quantized_embedding(
    weights: &HashMap<String, Array>,
    prefix: &str,
    group_size: i32,
    bits: i32,
) -> Result<MaybeQuantized<nn::Embedding>, Error> {
    let weight = weights
        .get(&format!("{prefix}.weight"))
        .cloned()
        .ok_or_else(|| Error::Model(format!("Weight not found: {prefix}.weight")))?;
    let scales = weights
        .get(&format!("{prefix}.scales"))
        .cloned()
        .ok_or_else(|| Error::Model(format!("Weight not found: {prefix}.scales")))?;
    let biases = weights
        .get(&format!("{prefix}.biases"))
        .cloned()
        .ok_or_else(|| Error::Model(format!("Weight not found: {prefix}.biases")))?;

    let inner = nn::Embedding {
        weight: Param::new(weight),
    };
    let mut qe = nn::QuantizedEmbedding {
        group_size,
        bits,
        scales: Param::new(scales),
        biases: Param::new(biases),
        inner,
    };
    qe.freeze_parameters(true);
    Ok(MaybeQuantized::Quantized(qe))
}

fn load_rms_norm(
    weights: &HashMap<String, Array>,
    key: &str,
    eps: f32,
) -> Result<nn::RmsNorm, Error> {
    let weight = weights
        .get(key)
        .cloned()
        .ok_or_else(|| Error::Model(format!("Weight not found: {key}")))?;
    Ok(nn::RmsNorm {
        weight: Param::new(weight),
        eps,
    })
}

fn load_all_weights(model_dir: &Path) -> Result<HashMap<String, Array>, Error> {
    let single = model_dir.join("model.safetensors");
    if !single.exists() {
        return Err(Error::Model(format!(
            "No model.safetensors in {}",
            model_dir.display()
        )));
    }
    let map: HashMap<String, Array> = Array::load_safetensors(&single)?.into_iter().collect();
    Ok(map)
}

fn build_rope(spec: &RopeSpec, head_dim: i32) -> Result<GemmaRope, Error> {
    match spec.rope_type.as_str() {
        "default" => Ok(GemmaRope::Standard(
            nn::RopeBuilder::new(head_dim)
                .base(spec.rope_theta)
                .traditional(false)
                .build()
                .map_err(|e| Error::Model(format!("RoPE build failed: {e:?}")))?,
        )),
        "proportional" => Ok(GemmaRope::Proportional(ProportionalRope::new(
            head_dim,
            spec.rope_theta,
            spec.partial_rotary_factor,
        ))),
        other => Err(Error::Model(format!(
            "Unsupported rope_type '{other}' for assistant"
        ))),
    }
}

pub fn load_assistant_model(model_dir: impl AsRef<Path>) -> Result<AssistantModel, Error> {
    let model_dir = model_dir.as_ref();
    let config = AssistantConfig::load(model_dir)?;
    let text = config.text_config.clone();
    let weights = load_all_weights(model_dir)?;

    let quant = config
        .quantization
        .clone()
        .ok_or_else(|| Error::Model("assistant config missing 'quantization' block".to_string()))?;
    let group_size = quant.group_size;
    let bits = quant.bits;

    let activation = GemmaActivation::from_name(&text.hidden_activation)
        .map_err(|e| Error::Model(format!("Unknown activation: {e:?}")))?;

    let layer_types: Vec<LayerKind> = text
        .layer_types
        .iter()
        .map(|s| LayerKind::parse(s))
        .collect::<Result<Vec<_>, _>>()?;

    let mut layers = Vec::with_capacity(text.num_hidden_layers as usize);
    for layer_idx in 0..text.num_hidden_layers as usize {
        let layer_prefix = format!("model.layers.{layer_idx}");
        let kind = layer_types[layer_idx];
        let is_sliding = matches!(kind, LayerKind::Sliding);
        let head_dim = if is_sliding {
            text.head_dim
        } else if text.global_head_dim > 0 {
            text.global_head_dim
        } else {
            text.head_dim
        };
        let n_kv_heads = if is_sliding {
            text.num_key_value_heads
        } else if text.num_global_key_value_heads > 0 {
            text.num_global_key_value_heads
        } else {
            text.num_key_value_heads
        };
        let rope_spec = if is_sliding {
            &text.rope_parameters.sliding_attention
        } else {
            &text.rope_parameters.full_attention
        };
        let rope = build_rope(rope_spec, head_dim)?;

        let self_attn = AssistantAttention {
            is_sliding,
            n_heads: text.num_attention_heads,
            n_kv_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            sliding_window: text.sliding_window,
            q_proj: load_quantized_linear(
                &weights,
                &format!("{layer_prefix}.self_attn.q_proj"),
                group_size,
                bits,
            )?,
            o_proj: load_quantized_linear(
                &weights,
                &format!("{layer_prefix}.self_attn.o_proj"),
                group_size,
                bits,
            )?,
            q_norm: load_rms_norm(
                &weights,
                &format!("{layer_prefix}.self_attn.q_norm.weight"),
                text.rms_norm_eps,
            )?,
            rope,
        };

        let mlp = AssistantMlp {
            gate_proj: load_quantized_linear(
                &weights,
                &format!("{layer_prefix}.mlp.gate_proj"),
                group_size,
                bits,
            )?,
            up_proj: load_quantized_linear(
                &weights,
                &format!("{layer_prefix}.mlp.up_proj"),
                group_size,
                bits,
            )?,
            down_proj: load_quantized_linear(
                &weights,
                &format!("{layer_prefix}.mlp.down_proj"),
                group_size,
                bits,
            )?,
            activation,
        };

        let layer = AssistantDecoderLayer {
            self_attn,
            mlp,
            input_layernorm: load_rms_norm(
                &weights,
                &format!("{layer_prefix}.input_layernorm.weight"),
                text.rms_norm_eps,
            )?,
            post_attention_layernorm: load_rms_norm(
                &weights,
                &format!("{layer_prefix}.post_attention_layernorm.weight"),
                text.rms_norm_eps,
            )?,
            pre_feedforward_layernorm: load_rms_norm(
                &weights,
                &format!("{layer_prefix}.pre_feedforward_layernorm.weight"),
                text.rms_norm_eps,
            )?,
            post_feedforward_layernorm: load_rms_norm(
                &weights,
                &format!("{layer_prefix}.post_feedforward_layernorm.weight"),
                text.rms_norm_eps,
            )?,
            layer_scalar: Param::new(
                weights
                    .get(&format!("{layer_prefix}.layer_scalar"))
                    .cloned()
                    .ok_or_else(|| {
                        Error::Model(format!("Missing {layer_prefix}.layer_scalar"))
                    })?,
            ),
        };
        layers.push(layer);
    }

    let embed_tokens =
        load_quantized_embedding(&weights, "model.embed_tokens", group_size, bits)?;
    let norm = load_rms_norm(&weights, "model.norm.weight", text.rms_norm_eps)?;
    let pre_projection = load_quantized_linear(&weights, "pre_projection", group_size, bits)?;
    let post_projection = load_quantized_linear(&weights, "post_projection", group_size, bits)?;

    Ok(AssistantModel {
        config,
        layer_types,
        layers,
        norm,
        embed_tokens,
        pre_projection,
        post_projection,
    })
}

// Helper: build a [B, 1, 2*backbone_hidden] inputs_embeds from a [B, 1, H]
// previous-token embedding (from the target's embed) and a [B, 1, H] recurrent
// hidden (target's last hidden on step 0, or this module's `last_hidden`
// thereafter). Both halves must be `backbone_hidden_size` wide.
pub fn build_inputs_embeds(
    prev_token_embed: &Array,
    recurrent_hidden: &Array,
) -> Result<Array, Exception> {
    // Concat order: (recurrent_hidden, prev_token_embed). The previous
    // ordering (prev_token_embed first) produced acceptance ~1% on the
    // gemma4-27B-MTPLX-Optimized-Speed pair, identical to the Qwen3.6
    // MTP concat-order bug — pre_projection's weight matrix expects
    // recurrent first, embed second. Toggle via MTPLX_PAIR_CONCAT_ORDER=embed_first
    // to A/B against the prior bug-for-bug behaviour.
    let recurrent_first = std::env::var("MTPLX_PAIR_CONCAT_ORDER")
        .map(|v| v.as_str() != "embed_first")
        .unwrap_or(true);
    if recurrent_first {
        ops::concatenate_axis(&[recurrent_hidden, prev_token_embed], -1)
    } else {
        ops::concatenate_axis(&[prev_token_embed, recurrent_hidden], -1)
    }
}

/// Quick sample: argmax of last position. Caller can swap in temp/top-k if
/// desired.
pub fn argmax_last(logits: &Array) -> Result<i32, Exception> {
    use mlx_rs::ops::indexing::IndexOp;
    let last = logits.index((.., -1, ..)).reshape(&[-1])?;
    let id = argmax_axis!(last, -1)?.as_dtype(mlx_rs::Dtype::Int32)?;
    Ok(id.item::<i32>())
}
