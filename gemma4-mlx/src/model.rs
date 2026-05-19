//! Gemma 4 text-only model implementation.

use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

use mlx_rs::{
    array,
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::{Module, ModuleParametersExt, Param},
    nn,
    ops::{
        self,
        argsort_axis,
        indexing::{scatter_add_single, take_along_axis, take_axis, IndexOp, NewAxis},
    },
    quantization::MaybeQuantized,
    transforms::eval,
    Array, Dtype,
};
use serde::Deserialize;
use serde_json::Value;
use tokenizers::Tokenizer;

use mlx_rs_core::{
    cache::{KVCache, KeyValueCache},
    error::Error,
    moe_dense_matmul,
    sampler::{DefaultSampler, Sampler},
    utils::{create_causal_mask, scaled_dot_product_attention, SdpaMask},
};

use crate::vision::{
    load_embed_vision, load_vision_model, preprocess_image_gemma4, EmbedVision,
    Gemma4VisionConfig, VisionModel,
};

// ============================================================================
// Configuration
// ============================================================================

/// Deserialize a JSON value that may be `null` or missing as `0`.
fn nullable_i32<'de, D>(deserializer: D) -> Result<i32, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Option::<i32>::deserialize(deserializer).map(|opt| opt.unwrap_or(0))
}

#[derive(Debug, Clone, Deserialize)]
pub struct Gemma4Config {
    pub model_type: String,
    pub text_config: Gemma4TextConfig,
    #[serde(default)]
    pub vision_config: Option<Gemma4VisionConfig>,
    #[serde(default)]
    pub vision_soft_tokens_per_image: usize,
    #[serde(default)]
    pub image_token_id: Option<u32>,
    #[serde(default)]
    pub boi_token_id: Option<u32>,
    #[serde(default)]
    pub eoi_token_id: Option<u32>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// Present when the checkpoint ships pre-quantized
    /// `(.weight U32, .scales BF16, .biases BF16)` triplets. The loader
    /// uses this to build `MaybeQuantized::Quantized` modules so inference
    /// can run native `quantized_matmul` instead of dequantizing to BF16.
    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct QuantizationConfig {
    pub bits: i32,
    pub group_size: i32,
    #[serde(default = "default_quant_mode")]
    pub mode: String,
}

fn default_quant_mode() -> String {
    "affine".to_string()
}

#[derive(Debug, Clone, Deserialize)]
pub struct Gemma4TextConfig {
    pub attention_bias: bool,
    pub attention_dropout: f32,
    pub attention_k_eq_v: bool,
    pub enable_moe_block: bool,
    #[serde(default, deserialize_with = "nullable_i32")]
    pub global_head_dim: i32,
    pub head_dim: i32,
    pub hidden_activation: String,
    pub hidden_size: i32,
    #[serde(default)]
    pub hidden_size_per_layer_input: i32,
    pub intermediate_size: i32,
    pub layer_types: Vec<String>,
    pub max_position_embeddings: i32,
    #[serde(default, deserialize_with = "nullable_i32")]
    pub moe_intermediate_size: i32,
    pub num_attention_heads: i32,
    #[serde(default, deserialize_with = "nullable_i32")]
    pub num_experts: i32,
    #[serde(default, deserialize_with = "nullable_i32")]
    pub num_global_key_value_heads: i32,
    pub num_hidden_layers: i32,
    #[serde(default)]
    pub num_kv_shared_layers: i32,
    pub num_key_value_heads: i32,
    pub rms_norm_eps: f32,
    pub rope_parameters: RopeParameters,
    pub sliding_window: i32,
    pub tie_word_embeddings: bool,
    #[serde(default, deserialize_with = "nullable_i32")]
    pub top_k_experts: i32,
    #[serde(default)]
    pub use_double_wide_mlp: bool,
    pub vocab_size: i32,
    #[serde(default)]
    pub vocab_size_per_layer_input: i32,
    #[serde(default)]
    pub final_logit_softcapping: Option<f32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    pub sliding_attention: RopeSpec,
    pub full_attention: RopeSpec,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeSpec {
    pub rope_theta: f32,
    pub rope_type: String,
    #[serde(default)]
    pub partial_rotary_factor: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct WeightMap {
    pub metadata: HashMap<String, Value>,
    pub weight_map: HashMap<String, String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    SlidingAttention,
    FullAttention,
}

impl LayerType {
    fn from_config(layer_type: &str) -> Result<Self, Error> {
        match layer_type {
            "sliding_attention" => Ok(Self::SlidingAttention),
            "full_attention" => Ok(Self::FullAttention),
            other => Err(Error::Model(format!(
                "Unsupported Gemma4 layer type: {other}"
            ))),
        }
    }

    fn is_sliding(self) -> bool {
        matches!(self, Self::SlidingAttention)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum GemmaActivation {
    GeluPytorchTanh,
    Silu,
}

impl GemmaActivation {
    pub fn from_name(name: &str) -> Result<Self, Error> {
        match name {
            "gelu_pytorch_tanh" => Ok(Self::GeluPytorchTanh),
            "silu" | "swish" => Ok(Self::Silu),
            other => Err(Error::Model(format!(
                "Unsupported Gemma4 activation: {other}"
            ))),
        }
    }

    pub fn apply(self, x: &Array) -> Result<Array, Exception> {
        match self {
            Self::GeluPytorchTanh => nn::gelu_approximate(x),
            Self::Silu => nn::silu(x),
        }
    }
}

#[derive(Debug, Clone)]
pub struct UnscaledRmsNorm {
    pub eps: f32,
}

impl UnscaledRmsNorm {
    pub fn new(eps: f32) -> Self {
        Self { eps }
    }

    pub fn forward(&self, x: &Array) -> Result<Array, Exception> {
        let x_f32 = x.as_dtype(Dtype::Float32)?;
        let variance = x_f32.square()?.mean_axis(-1, true)?;
        let scale = variance.add(&array!(self.eps))?.rsqrt()?;
        x_f32.multiply(&scale)?.as_dtype(x.dtype())
    }
}

// ============================================================================
// Rotary embeddings
// ============================================================================

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
enum RotaryLayout {
    BatchHeadsSeqDim,
    BatchSeqHeadsDim,
}

#[derive(Debug, Clone)]
pub enum GemmaRope {
    Standard(nn::Rope),
    Proportional(ProportionalRope),
}

impl GemmaRope {
    pub fn apply(&mut self, x: &Array, offset: i32) -> Result<Array, Exception> {
        match self {
            Self::Standard(rope) => {
                rope.forward(nn::RopeInputBuilder::new(x).offset(offset).build()?)
            }
            Self::Proportional(rope) => rope.apply(x, offset),
        }
    }

    #[allow(dead_code)]
    fn apply_with_layout(
        &mut self,
        x: &Array,
        offset: i32,
        layout: RotaryLayout,
    ) -> Result<Array, Exception> {
        match self {
            Self::Standard(rope) => match x.shape().len() {
                3 => rope.forward(nn::RopeInputBuilder::new(x).offset(offset).build()?),
                4 => match layout {
                    RotaryLayout::BatchHeadsSeqDim => {
                        rope.forward(nn::RopeInputBuilder::new(x).offset(offset).build()?)
                    }
                    RotaryLayout::BatchSeqHeadsDim => {
                        let transposed = x.transpose_axes(&[0, 2, 1, 3])?;
                        let rotated = rope.forward(
                            nn::RopeInputBuilder::new(&transposed)
                                .offset(offset)
                                .build()?,
                        )?;
                        rotated.transpose_axes(&[0, 2, 1, 3])
                    }
                },
                ndim => Err(Exception::custom(format!(
                    "Gemma standard RoPE expects 3D or 4D input, got {ndim}D"
                ))),
            },
            Self::Proportional(rope) => rope.apply_with_layout(x, offset, layout),
        }
    }

    fn training_mode(&mut self, mode: bool) {
        if let Self::Standard(rope) = self {
            <nn::Rope as Module<nn::RopeInput>>::training_mode(rope, mode);
        }
    }

    /// Per-position RoPE for `x` of shape `[B, H, L, D]` with positions
    /// `[L]` (i64 / int32). Used by DDTree's fused tree-mask verify where
    /// tree nodes at the same depth share the same logical position even
    /// though they sit at different cache slots.
    ///
    /// Falls back to a generic cos/sin path for both Standard and
    /// Proportional variants — the `fast::rope` Metal kernel only accepts
    /// a single starting offset, so per-position rotation has to be done
    /// the slow way (one outer product + cos/sin per call). For
    /// small-L tree forwards (≤ tree_budget tokens) this is dominated by
    /// attention/MLP costs anyway.
    pub fn apply_per_position(
        &self,
        x: &Array,
        positions: &Array,
    ) -> Result<Array, Exception> {
        if x.shape().len() != 4 {
            return Err(Exception::custom(format!(
                "apply_per_position expects 4D [B,H,L,D], got shape {:?}",
                x.shape()
            )));
        }
        let head_dim = x.shape()[3];
        let inv_freq = match self {
            Self::Standard(rope) => standard_inv_freq(rope.base, head_dim)?,
            Self::Proportional(rope) => rope.inv_freq.clone(),
        };
        // Single-dispatch Metal kernel; replaces the prior
        // outer + concat + cos + sin + multiply + rotate-half chain.
        mlx_rs_core::per_position_rope(x, positions, &inv_freq)
    }
}

fn standard_inv_freq(base: f32, head_dim: i32) -> Result<Array, Exception> {
    let half_dim = head_dim / 2;
    let mut inv = Vec::with_capacity(half_dim as usize);
    for i in 0..half_dim {
        inv.push(1.0 / base.powf((2 * i) as f32 / head_dim as f32));
    }
    Ok(Array::from_slice(&inv, &[half_dim]))
}

#[derive(Debug, Clone)]
pub struct ProportionalRope {
    inv_freq: Array,
    head_dim: i32,
}

impl ProportionalRope {
    pub fn new(head_dim: i32, theta: f32, partial_rotary_factor: f32) -> Self {
        let half_dim = head_dim / 2;
        let rope_angles = ((partial_rotary_factor * head_dim as f32) / 2.0).floor() as i32;
        let rope_angles = rope_angles.clamp(0, half_dim);
        let nope_angles = half_dim - rope_angles;

        let mut inv_freq = Vec::with_capacity(half_dim as usize);
        for i in 0..rope_angles {
            inv_freq.push(1.0 / theta.powf((2 * i) as f32 / head_dim as f32));
        }
        inv_freq.extend(std::iter::repeat_n(0.0, nope_angles as usize));

        Self {
            inv_freq: Array::from_slice(&inv_freq, &[half_dim]),
            head_dim,
        }
    }

    pub fn apply(&self, x: &Array, offset: i32) -> Result<Array, Exception> {
        self.apply_with_layout(x, offset, RotaryLayout::BatchHeadsSeqDim)
    }

    #[allow(dead_code)]
    fn apply_with_layout(
        &self,
        x: &Array,
        offset: i32,
        layout: RotaryLayout,
    ) -> Result<Array, Exception> {
        match x.shape().len() {
            3 => self.apply_3d(x, offset),
            4 => self.apply_4d(x, offset, layout),
            ndim => Err(Exception::custom(format!(
                "Gemma proportional RoPE expects 3D or 4D input, got {ndim}D"
            ))),
        }
    }

    fn apply_3d(&self, x: &Array, offset: i32) -> Result<Array, Exception> {
        let seq_len = x.shape()[1];
        let (cos, sin) = self.cos_sin(seq_len, offset, x.dtype(), &[1, seq_len, self.head_dim])?;
        apply_rotary_pos_emb(x, &cos, &sin)
    }

    fn apply_4d(&self, x: &Array, offset: i32, layout: RotaryLayout) -> Result<Array, Exception> {
        let (seq_len, shape) = match layout {
            RotaryLayout::BatchHeadsSeqDim => (x.shape()[2], [1, 1, x.shape()[2], self.head_dim]),
            RotaryLayout::BatchSeqHeadsDim => (x.shape()[1], [1, x.shape()[1], 1, self.head_dim]),
        };
        let (cos, sin) = self.cos_sin(seq_len, offset, x.dtype(), &shape)?;
        apply_rotary_pos_emb(x, &cos, &sin)
    }

    fn cos_sin(
        &self,
        seq_len: i32,
        offset: i32,
        dtype: Dtype,
        shape: &[i32],
    ) -> Result<(Array, Array), Exception> {
        let positions = ops::arange::<_, f32>(offset, offset + seq_len, 1)?;
        let freqs = ops::outer(&positions, &self.inv_freq)?;
        let emb = ops::concatenate_axis(&[&freqs, &freqs], -1)?;
        let cos = ops::cos(&emb)?.as_dtype(dtype)?.reshape(shape)?;
        let sin = ops::sin(&emb)?.as_dtype(dtype)?.reshape(shape)?;
        Ok((cos, sin))
    }
}

fn rotate_half(x: &Array) -> Result<Array, Exception> {
    let half_dim = x.shape()[x.shape().len() - 1] / 2;
    let rotated = match x.shape().len() {
        3 => {
            let x1 = x.index((.., .., ..half_dim));
            let x2 = x.index((.., .., half_dim..));
            let neg_x2 = x2.negative()?;
            ops::concatenate_axis(&[&neg_x2, &x1], -1)?
        }
        4 => {
            let x1 = x.index((.., .., .., ..half_dim));
            let x2 = x.index((.., .., .., half_dim..));
            let neg_x2 = x2.negative()?;
            ops::concatenate_axis(&[&neg_x2, &x1], -1)?
        }
        ndim => {
            return Err(Exception::custom(format!(
                "Gemma rotate_half expects 3D or 4D input, got {ndim}D"
            )))
        }
    };
    Ok(rotated)
}

fn apply_rotary_pos_emb(x: &Array, cos: &Array, sin: &Array) -> Result<Array, Exception> {
    let rotated = rotate_half(x)?;
    x.multiply(cos)?.add(&rotated.multiply(sin)?)
}

// ============================================================================
// Attention
// ============================================================================

#[derive(Debug, Clone, ModuleParameters)]
pub struct Attention {
    pub layer_idx: i32,
    pub layer_type: LayerType,
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub sliding_window: Option<i32>,
    /// If true, this layer shares KV from an earlier layer (no k/v projections).
    pub is_kv_shared: bool,

    #[param]
    pub q_proj: MaybeQuantized<nn::Linear>,
    #[param]
    pub k_proj: Option<MaybeQuantized<nn::Linear>>,
    #[param]
    pub v_proj: Option<MaybeQuantized<nn::Linear>>,
    /// Fused Q-K-V projection (opt-in via `GEMMA4_FUSED_QKV=1`).
    /// Concatenates q/k/v_proj weights along output axis so the forward
    /// pass does ONE matmul instead of 2-3 separate ones. Output is
    /// then split via `fused_splits`. Skipped for KV-shared layers and
    /// when q/k group_size or bits don't match.
    #[param]
    pub qkv_fused: Option<MaybeQuantized<nn::Linear>>,
    /// Per-part output sizes for splitting the fused QKV output:
    /// `[q_out, k_out]` or `[q_out, k_out, v_out]`.
    pub fused_splits: Option<Vec<i32>>,
    #[param]
    pub o_proj: MaybeQuantized<nn::Linear>,
    #[param]
    pub q_norm: nn::RmsNorm,
    #[param]
    pub k_norm: Option<nn::RmsNorm>,
    rope: GemmaRope,

    pub v_norm: Option<UnscaledRmsNorm>,
}

pub struct AttentionInput<'a, C> {
    pub x: &'a Array,
    pub mask: Option<&'a Array>,
    pub cache: &'a mut C,
    /// Pre-computed shared KV from an earlier layer (for KV-shared layers).
    pub shared_kv: Option<(Array, Array)>,
    /// Explicit per-token RoPE positions. When `Some([L])`, each token in
    /// the L-length input gets its Q (and K, if computed) rotated to its
    /// own position instead of the sequential `cache.offset() + i` default.
    /// Used by DDTree's fused tree-mask forward where tree nodes at the
    /// same depth share the same logical position.
    pub position_ids: Option<&'a Array>,
}

impl<C> Module<AttentionInput<'_, C>> for Attention
where
    C: KeyValueCache,
{
    type Output = Array;
    type Error = Exception;

    #[allow(non_snake_case)]
    fn forward(&mut self, input: AttentionInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let AttentionInput {
            x,
            mask,
            cache,
            shared_kv,
            position_ids,
        } = input;

        let shape = x.shape();
        let B = shape[0];
        let L = shape[1];

        let offset = cache.offset();

        let is_shared_kv = shared_kv.is_some();

        // Optional fused Q/K/V projection: one matmul + split, saves 2-3
        // kernel launches per attention forward. Falls back to separate
        // q_proj/k_proj/v_proj calls when fused is unavailable.
        let (raw_q, raw_k_opt, raw_v_opt) = if let Some(qkv) = self.qkv_fused.as_mut() {
            let fused_out = qkv.forward(x)?;
            // Split along last axis using cumulative offsets from `fused_splits`.
            let splits = self.fused_splits.as_ref().expect("fused_splits set when qkv_fused set");
            let q_end = splits[0];
            let k_end = q_end + splits[1];
            let v_end = if splits.len() >= 3 { k_end + splits[2] } else { k_end };
            let q = fused_out.index((.., .., 0_i32..q_end));
            let k = if !is_shared_kv {
                Some(fused_out.index((.., .., q_end..k_end)))
            } else {
                None
            };
            let v = if !is_shared_kv && splits.len() >= 3 {
                Some(fused_out.index((.., .., k_end..v_end)))
            } else {
                None
            };
            (q, k, v)
        } else {
            let q = self.q_proj.forward(x)?;
            (q, None, None)
        };

        let mut queries =
            self.q_norm
                .forward(&raw_q.reshape(&[B, L, self.n_heads, self.head_dim])?)?;
        queries = queries.transpose_axes(&[0, 2, 1, 3])?;

        // Compute new K/V (with RoPE on K) WITHOUT touching the cache yet,
        // so we can opt into the cache's fused-attention fast path
        // (TurboQuant) before falling back to update_and_fetch + SDPA.
        let pending_new_kv: Option<(Array, Array)> = if is_shared_kv {
            None
        } else {
            let raw_keys = if let Some(k) = raw_k_opt {
                k
            } else {
                let k_proj = self.k_proj.as_mut().expect("k_proj required for non-shared layer");
                k_proj.forward(x)?
            };
            let raw_values = if let Some(v) = raw_v_opt {
                v
            } else {
                match self.v_proj.as_mut() {
                    Some(v_proj) => v_proj.forward(x)?,
                    None => raw_keys.clone(),
                }
            };
            let k_norm = self.k_norm.as_mut().expect("k_norm required for non-shared layer");
            let mut new_k =
                k_norm.forward(&raw_keys.reshape(&[B, L, self.n_kv_heads, self.head_dim])?)?;
            let mut new_v = match self.v_norm.as_ref() {
                Some(v_norm) => {
                    v_norm.forward(&raw_values.reshape(&[B, L, self.n_kv_heads, self.head_dim])?)?
                }
                None => raw_values.reshape(&[B, L, self.n_kv_heads, self.head_dim])?,
            };
            new_k = new_k.transpose_axes(&[0, 2, 1, 3])?;
            new_v = new_v.transpose_axes(&[0, 2, 1, 3])?;
            new_k = match position_ids {
                Some(pos) => self.rope.apply_per_position(&new_k, pos)?,
                None => self.rope.apply(&new_k, offset)?,
            };
            Some((new_k, new_v))
        };

        // Apply RoPE on queries. For shared KV layers, derive offset from
        // the borrowed KV length; otherwise use the cache offset before
        // this step's update.
        let rope_offset = if let Some((shared_k, _)) = shared_kv.as_ref() {
            (shared_k.shape()[2] - L) as i32
        } else {
            offset
        };
        queries = match position_ids {
            Some(pos) => self.rope.apply_per_position(&queries, pos)?,
            None => self.rope.apply(&queries, rope_offset)?,
        };

        // Try fused-attention fast path: q_len=1, non-shared KV, no
        // sliding window (spike scope). Falls back to standard SDPA if
        // the cache returns Ok(None) (default impl, or unsupported
        // shapes) or any of the gating conditions fail.
        let fused_eligible = !is_shared_kv && L == 1 && self.sliding_window.is_none();
        if fused_eligible {
            if let Some((new_k, new_v)) = pending_new_kv.as_ref().map(|(k, v)| (k.clone(), v.clone())) {
                let kv_repeat = (self.n_heads / self.n_kv_heads) as i32;
                if let Some(fused_out) = cache.try_fused_attention(
                    &queries, new_k, new_v, self.scale, mask, kv_repeat,
                )? {
                    let out = fused_out
                        .transpose_axes(&[0, 2, 1, 3])?
                        .reshape(&[B, L, -1])?;
                    return self.o_proj.forward(&out);
                }
            }
        }

        // Fallback: standard update_and_fetch + scaled_dot_product_attention.
        let (keys, values) = if let Some((shared_k, shared_v)) = shared_kv {
            (shared_k, shared_v)
        } else {
            let (new_k, new_v) = pending_new_kv.expect("non-shared KV not computed");
            cache.update_and_fetch(new_k, new_v)?
        };

        let sliding_mask = match (mask, self.sliding_window) {
            (None, Some(window)) => {
                Some(create_causal_mask(L, Some(rope_offset), Some(window), None)?)
            }
            _ => None,
        };
        let sdpa_mask = match mask {
            Some(m) => Some(SdpaMask::Array(m)),
            None => match sliding_mask.as_ref() {
                Some(m) => Some(SdpaMask::Array(m)),
                None if L > 1 => Some(SdpaMask::Causal),
                None => None,
            },
        };

        let output = scaled_dot_product_attention::<C>(
            queries, keys, values, None::<C>, self.scale, sdpa_mask,
        )?
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[B, L, -1])?;

        self.o_proj.forward(&output)
    }

    fn training_mode(&mut self, mode: bool) {
        self.q_proj.training_mode(mode);
        if let Some(ref mut k_proj) = self.k_proj {
            k_proj.training_mode(mode);
        }
        if let Some(ref mut v_proj) = self.v_proj {
            v_proj.training_mode(mode);
        }
        self.o_proj.training_mode(mode);
        self.q_norm.training_mode(mode);
        if let Some(ref mut k_norm) = self.k_norm {
            k_norm.training_mode(mode);
        }
        self.rope.training_mode(mode);
    }
}

// ============================================================================
// Feedforward / MoE
// ============================================================================

#[derive(Debug, Clone, ModuleParameters)]
pub struct DenseMlp {
    #[param]
    pub gate_proj: MaybeQuantized<nn::Linear>,
    #[param]
    pub up_proj: MaybeQuantized<nn::Linear>,
    #[param]
    pub down_proj: MaybeQuantized<nn::Linear>,

    pub activation: GemmaActivation,
}

impl Module<&Array> for DenseMlp {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Self::Output, Self::Error> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        // Use the fused Metal kernel for SwiGLU (the common case); fall back to
        // generic activate-then-multiply for other activations like GeluPytorchTanh.
        let activated = match self.activation {
            GemmaActivation::Silu => mlx_rs_core::fused_swiglu(&up, &gate)?,
            _ => self.activation.apply(&gate)?.multiply(&up)?,
        };
        self.down_proj.forward(&activated)
    }

    fn training_mode(&mut self, mode: bool) {
        self.gate_proj.training_mode(mode);
        self.up_proj.training_mode(mode);
        self.down_proj.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub struct Router {
    pub hidden_size: i32,
    pub top_k_experts: i32,
    pub scalar_root_size: f32,

    #[param]
    pub proj: MaybeQuantized<nn::Linear>,
    #[param]
    pub scale: Param<Array>,
    #[param]
    pub per_expert_scale: Param<Array>,

    pub norm: UnscaledRmsNorm,
}

impl Router {
    pub fn forward(&mut self, hidden_states: &Array) -> Result<(Array, Array, Array), Exception> {
        let hidden_states = self.norm.forward(hidden_states)?;
        let scale = (&*self.scale).as_dtype(hidden_states.dtype())?;
        let hidden_states = hidden_states
            .multiply(&scale)?
            .multiply(&array!(self.scalar_root_size))?;

        let expert_scores = self
            .proj
            .forward(&hidden_states)?
            .as_dtype(Dtype::Float32)?;
        let router_probabilities = ops::softmax_axis(&expert_scores, -1, Some(true))?;

        let neg_scores = router_probabilities.negative()?;
        let partitioned_indices = ops::argpartition_axis(&neg_scores, self.top_k_experts - 1, -1)?;
        let top_k_index = partitioned_indices.index((.., ..self.top_k_experts));

        let mut top_k_weights = take_along_axis(&router_probabilities, &top_k_index, -1)?;
        let denom = top_k_weights.sum_axis(-1, true)?;
        top_k_weights = top_k_weights.divide(&denom)?;

        let per_expert_scale = take_axis(
            &(&*self.per_expert_scale).as_dtype(Dtype::Float32)?,
            &top_k_index,
            0,
        )?;
        top_k_weights = top_k_weights.multiply(&per_expert_scale)?;

        Ok((router_probabilities, top_k_weights, top_k_index))
    }

    pub fn training_mode(&mut self, mode: bool) {
        self.proj.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub struct Experts {
    pub hidden_size: i32,
    pub intermediate_size: i32,

    #[param]
    pub gate_up_proj: Param<Array>,
    #[param]
    pub down_proj: Param<Array>,

    pub activation: GemmaActivation,
}

/// One group of (token, k_slot) pairs that all route to the same expert.
/// `token_indices` and `k_slots` are parallel arrays of shape `[n_tokens]`,
/// each entry an i32 index into the original `hidden_states` rows /
/// `top_k_weights` columns respectively.
#[derive(Debug, Clone)]
pub struct ExpertBucket {
    pub expert_id: i32,
    pub token_indices: Array,
    pub k_slots: Array,
    pub n_tokens: i32,
}

/// Group routed (token, k_slot) pairs by expert id. Reads `top_k_index`
/// (shape `[n, k]`, any int dtype) on CPU and returns one
/// [`ExpertBucket`] per expert that received any tokens. Empty experts
/// are omitted from the returned vec.
///
/// Cost: one device→host copy of `n*k` i32 entries. For n=5000, k=4 that's
/// ~80 KB — negligible vs the matmul cost downstream.
pub fn bucket_by_expert(
    top_k_index: &Array,
    num_experts: i32,
) -> Result<Vec<ExpertBucket>, Exception> {
    let idx_i32 = top_k_index.as_dtype(Dtype::Int32)?;
    eval([&idx_i32])?;
    let shape = idx_i32.shape().to_vec();
    let n = shape[0] as usize;
    let k = shape[1] as usize;
    let slice = idx_i32.as_slice::<i32>();

    let mut per_expert: Vec<(Vec<i32>, Vec<i32>)> = (0..num_experts as usize)
        .map(|_| (Vec::new(), Vec::new()))
        .collect();
    for token_idx in 0..n {
        for k_slot in 0..k {
            let expert = slice[token_idx * k + k_slot] as usize;
            if expert >= per_expert.len() {
                return Err(Exception::from(
                    format!(
                        "top_k_index contained expert id {expert} but num_experts is {num_experts}"
                    )
                    .as_str(),
                ));
            }
            per_expert[expert].0.push(token_idx as i32);
            per_expert[expert].1.push(k_slot as i32);
        }
    }

    let mut buckets = Vec::with_capacity(num_experts as usize);
    for (e, (tok_idx, k_slot)) in per_expert.into_iter().enumerate() {
        if tok_idx.is_empty() {
            continue;
        }
        let n_tokens = tok_idx.len() as i32;
        buckets.push(ExpertBucket {
            expert_id: e as i32,
            token_indices: Array::from_slice(&tok_idx, &[n_tokens]),
            k_slots: Array::from_slice(&k_slot, &[n_tokens]),
            n_tokens,
        });
    }
    Ok(buckets)
}

impl Experts {
    pub fn forward_topk(
        &mut self,
        hidden_states: &Array,
        top_k_index: &Array,
        top_k_weights: &Array,
    ) -> Result<Array, Exception> {
        let hidden_dtype = hidden_states.dtype();
        let n = hidden_states.shape()[0];
        let k = top_k_index.shape()[1];

        let hidden_states = hidden_states.reshape(&[n, 1, 1, self.hidden_size])?;

        let gate_up = take_axis(&*self.gate_up_proj, top_k_index, 0)?;
        let gate_up = gate_up.transpose_axes(&[0, 1, 3, 2])?;
        let projected =
            hidden_states
                .matmul(&gate_up)?
                .reshape(&[n, k, 2 * self.intermediate_size])?;
        let split = projected.split(2, -1)?;
        let gate = self.activation.apply(&split[0])?;
        let up = &split[1];

        let activated = gate
            .multiply(up)?
            .reshape(&[n, k, 1, self.intermediate_size])?;

        let down = take_axis(&*self.down_proj, top_k_index, 0)?;
        let down = down.transpose_axes(&[0, 1, 3, 2])?;
        let expert_out = activated
            .matmul(&down)?
            .reshape(&[n, k, self.hidden_size])?
            .as_dtype(Dtype::Float32)?;

        let weighted = expert_out.multiply(&top_k_weights.index((.., .., NewAxis)))?;
        weighted.sum_axis(1, false)?.as_dtype(hidden_dtype)
    }

    /// Expert-major MoE dispatch — phase 2 (#35).
    ///
    /// Computes bucketing entirely on GPU (no CPU sync): flatten
    /// `top_k_index`, argsort to get a permutation that groups identical
    /// experts contiguously, gather X / W in sorted order, run a single
    /// batched matmul, then scatter back to per-token positions.
    ///
    /// With stock MLX matmul this is **structurally equivalent in cost
    /// to `forward_topk`** — the matmul is the same per-(token,k_slot)
    /// shape either way, and we add argsort + extra gathers on top. It
    /// is not a speedup. It is the *enabling layer* for phase 3: a custom
    /// Metal kernel will replace the row-wise matmul with a per-expert
    /// dense `[N_e, H] @ [H, 2I]` tile-reuse kernel (simdgroup_matrix),
    /// fed from this exact sorted layout.
    ///
    /// Gated at the call site by `GEMMA4_EXPERT_MAJOR_MOE=2`. v1
    /// (CPU-bucketed) at `=1`, default token-major at `=0`.
    pub fn forward_topk_expert_major_v2(
        &mut self,
        hidden_states: &Array,
        top_k_index: &Array,
        top_k_weights: &Array,
    ) -> Result<Array, Exception> {
        let hidden_dtype = hidden_states.dtype();
        let n = hidden_states.shape()[0];
        let h = hidden_states.shape()[1];
        let i = self.intermediate_size;
        let k_top = top_k_index.shape()[1];
        let n_k = n * k_top;

        // ── GPU bucketing: sort the (token, k_slot) pairs by expert id. ──
        let flat_experts = top_k_index.reshape(&[-1])?.as_dtype(Dtype::Int32)?;
        let sort_order = argsort_axis(&flat_experts, -1)?.as_dtype(Dtype::Int32)?;
        let sorted_experts = take_axis(&flat_experts, &sort_order, 0)?;
        // Recover (token_idx, k_slot) from the sort order: a position p in
        // the flat layout corresponds to token p / k_top and k_slot p % k_top.
        // We don't materialize a [n_k] arange — instead derive them from
        // sort_order directly: sort_order[i] is the flat position p, so
        //   sorted_token_indices[i] = sort_order[i] / k_top
        //   sorted_k_slots[i]       = sort_order[i] % k_top
        let k_arr = array!(k_top);
        let sorted_token_indices = sort_order.divide(&k_arr)?.as_dtype(Dtype::Int32)?;
        let sorted_k_slots = sort_order.remainder(&k_arr)?.as_dtype(Dtype::Int32)?;
        let _ = sorted_experts; // currently unused by stock-matmul path; phase 3 will consume it

        // ── Promote to fp32 once. ──
        let h_f32 = hidden_states.as_dtype(Dtype::Float32)?;
        let w_f32 = top_k_weights.as_dtype(Dtype::Float32)?;
        let gu_f32 = self.gate_up_proj.as_ref().as_dtype(Dtype::Float32)?;
        let dp_f32 = self.down_proj.as_ref().as_dtype(Dtype::Float32)?;

        // ── Gather X in sorted order: [n_k, H]. ──
        let x_sorted = take_axis(&h_f32, &sorted_token_indices, 0)?;

        // ── Gather gate_up_proj per row using sorted expert ids: [n_k, 2I, H]. ──
        // (sorted_experts contains the i32 expert id at each sorted position.)
        // We re-derive sorted_experts via take(flat_experts, sort_order).
        let sorted_experts_for_gather = take_axis(&flat_experts, &sort_order, 0)?;
        let gu_sorted = take_axis(&gu_f32, &sorted_experts_for_gather, 0)?;

        // ── Batched per-row matmul: [n_k, 1, H] @ [n_k, H, 2I] = [n_k, 1, 2I]. ──
        let x_3d = x_sorted.reshape(&[n_k, 1, h])?;
        let gu_t = gu_sorted.transpose_axes(&[0, 2, 1])?;
        let y = x_3d.matmul(&gu_t)?.reshape(&[n_k, 2 * i])?;

        // ── SwiGLU-style activation: split, gate * up. ──
        let split = y.split(2, -1)?;
        let gate = self.activation.apply(&split[0])?;
        let z = gate.multiply(&split[1])?; // [n_k, I]

        // ── Gather down_proj per row: [n_k, H, I]. ──
        let dp_sorted = take_axis(&dp_f32, &sorted_experts_for_gather, 0)?;
        let dp_t = dp_sorted.transpose_axes(&[0, 2, 1])?;
        let z_3d = z.reshape(&[n_k, 1, i])?;
        let out = z_3d.matmul(&dp_t)?.reshape(&[n_k, h])?;

        // ── Gather routing weights for each (sorted_token_idx, sorted_k_slot). ──
        let flat_idx = sorted_token_indices.multiply(&k_arr)?.add(&sorted_k_slots)?;
        let w_flat = w_f32.reshape(&[-1])?;
        let w_sorted = take_axis(&w_flat, &flat_idx, 0)?.reshape(&[n_k, 1])?;
        let weighted = out.multiply(&w_sorted)?;

        // ── Scatter back to per-token output. ──
        let mut output = ops::zeros::<f32>(&[n, h])?;
        let updates = weighted.reshape(&[n_k, 1, h])?;
        output = scatter_add_single(&output, &sorted_token_indices, &updates, 0)?;

        output.as_dtype(hidden_dtype)
    }

    /// Expert-major MoE dispatch — phase 5 (#35).
    ///
    /// Combines phase 2 GPU bucketing (argsort, no CPU sync on top_k_index)
    /// with phase 3 custom Metal kernel. Only sync: the `[num_experts]` i32
    /// counts vector, ~256 B per layer for E=64.
    ///
    /// Algorithm:
    ///   1. flat_experts = top_k_index.reshape(-1).as_dtype(i32)       [n*k]
    ///   2. sort_order   = argsort(flat_experts)                       [n*k]
    ///   3. sorted_*     = take(*, sort_order, 0) for experts, token_idx, k_slot
    ///   4. counts       = sum(sorted_experts.expand(-1) == arange(E), axis=0)
    ///   5. eval+sync(counts) — single small i32 read
    ///   6. starts[e]    = cumsum(counts) - counts (CPU)
    ///   7. per expert e: slice sorted layout [start..start+count_e]
    ///      → moe_dense_matmul (kernel) for gate_up + down → scatter back
    ///
    /// Gated at the call site by `GEMMA4_EXPERT_MAJOR_MOE=4`.
    pub fn forward_topk_expert_major_v4(
        &mut self,
        hidden_states: &Array,
        top_k_index: &Array,
        top_k_weights: &Array,
    ) -> Result<Array, Exception> {
        let hidden_dtype = hidden_states.dtype();
        let n = hidden_states.shape()[0];
        let h = hidden_states.shape()[1];
        let i = self.intermediate_size;
        let k_top = top_k_index.shape()[1];

        if h % 32 != 0 || (2 * i) % 32 != 0 || i % 8 != 0 {
            return Err(Exception::from(
                "forward_topk_expert_major_v4: shape alignment requirements not met",
            ));
        }

        let num_experts = self.gate_up_proj.as_ref().shape()[0];

        // ── GPU bucketing (Phase 2): sort (token, k_slot) by expert id. ──
        let flat_experts = top_k_index.reshape(&[-1])?.as_dtype(Dtype::Int32)?;
        let sort_order = argsort_axis(&flat_experts, -1)?.as_dtype(Dtype::Int32)?;
        let sorted_experts = take_axis(&flat_experts, &sort_order, 0)?;
        let k_arr = array!(k_top);
        let sorted_token_indices = sort_order.divide(&k_arr)?.as_dtype(Dtype::Int32)?;
        let sorted_k_slots = sort_order.remainder(&k_arr)?.as_dtype(Dtype::Int32)?;

        // ── Per-expert counts on GPU: sum(one_hot(sorted_experts, E), axis=0). ──
        // sorted_experts:               [n_k]
        // sorted_experts.reshape(-1,1): [n_k, 1]
        // arange(E).reshape(1, -1):     [1, E]
        // ==:                           [n_k, E] bool
        // sum axis=0:                   [E] i32
        let n_k = n * k_top;
        let experts_range = ops::arange::<_, i32>(0, num_experts, 1)?
            .as_dtype(Dtype::Int32)?
            .reshape(&[1, num_experts])?;
        let se_col = sorted_experts.reshape(&[n_k, 1])?;
        let mask = se_col
            .eq(&experts_range)?
            .as_dtype(Dtype::Int32)?;
        let counts_arr = mask.sum_axis(0, false)?.as_dtype(Dtype::Int32)?;

        // ── Sync only the [E] counts vector (256 B for E=64). ──
        eval([&counts_arr])?;
        let counts: Vec<i32> = counts_arr.as_slice::<i32>().to_vec();
        let mut starts: Vec<i32> = Vec::with_capacity(num_experts as usize);
        let mut acc = 0;
        for &c in &counts {
            starts.push(acc);
            acc += c;
        }

        // Promote to fp32 once.
        let h_f32 = hidden_states.as_dtype(Dtype::Float32)?;
        let w_f32 = top_k_weights.as_dtype(Dtype::Float32)?;
        let gu_f32 = self.gate_up_proj.as_ref().as_dtype(Dtype::Float32)?;
        let dp_f32 = self.down_proj.as_ref().as_dtype(Dtype::Float32)?;

        // Gather X / weight indices in sorted order on GPU (no sync).
        let x_sorted_all = take_axis(&h_f32, &sorted_token_indices, 0)?; // [n_k, H]
        let w_flat = w_f32.reshape(&[-1])?;
        let flat_routing_idx = sorted_token_indices.multiply(&k_arr)?.add(&sorted_k_slots)?;
        let w_sorted_all = take_axis(&w_flat, &flat_routing_idx, 0)?; // [n_k]

        let mut output = ops::zeros::<f32>(&[n, h])?;

        // ── Per-expert dispatch using the custom Metal kernel. ──
        for e in 0..num_experts as usize {
            let n_e = counts[e];
            if n_e == 0 {
                continue;
            }
            let start = starts[e];

            // Slice sorted layouts for this expert: [N_e, H], [N_e], [N_e].
            let x_e = x_sorted_all.index((start..start + n_e, ..));
            let w_e = w_sorted_all.index((start..start + n_e,));
            let tok_idx_e = sorted_token_indices.index((start..start + n_e,));

            // Pad to multiple of 32.
            let m_padded = ((n_e + 31) / 32) * 32;
            let pad_rows = m_padded - n_e;
            let x_pad = if pad_rows > 0 {
                let zeros = ops::zeros::<f32>(&[pad_rows, h])?;
                ops::concatenate_axis(&[&x_e, &zeros], 0)?
            } else {
                x_e.clone()
            };

            // gate_up_proj[e] is [2I, H] → kernel wants [H, 2I].
            let e_i32 = e as i32;
            let e_arr = Array::from_slice(&[e_i32], &[1]);
            let w_gu = take_axis(&gu_f32, &e_arr, 0)?.reshape(&[2 * i, h])?;
            let w_gu_t = w_gu.transpose_axes(&[1, 0])?;

            let m_buf = Array::from_slice(&[n_e], &[1]);
            let y_padded = moe_dense_matmul(&x_pad, &w_gu_t, &m_buf, h, 2 * i)?;
            let y_e = y_padded.index((..n_e, ..));

            let split = y_e.split(2, -1)?;
            let gate = self.activation.apply(&split[0])?;
            let z_e = gate.multiply(&split[1])?;

            let z_pad = if pad_rows > 0 {
                let zeros = ops::zeros::<f32>(&[pad_rows, i])?;
                ops::concatenate_axis(&[&z_e, &zeros], 0)?
            } else {
                z_e
            };

            let w_d = take_axis(&dp_f32, &e_arr, 0)?.reshape(&[h, i])?;
            let w_d_t = w_d.transpose_axes(&[1, 0])?;
            let out_padded = moe_dense_matmul(&z_pad, &w_d_t, &m_buf, i, h)?;
            let out_e = out_padded.index((..n_e, ..));

            let weighted = out_e.multiply(&w_e.reshape(&[n_e, 1])?)?;
            let updates = weighted.reshape(&[n_e, 1, h])?;
            output = scatter_add_single(&output, &tok_idx_e, &updates, 0)?;
        }

        output.as_dtype(hidden_dtype)
    }

    /// Expert-major MoE dispatch — phase 4 (#35).
    ///
    /// Combines phase 1 CPU bucketing with the phase 3 `moe_dense_matmul`
    /// Metal kernel (simdgroup_matrix<float,8,8> 4x4 acc tiles). Per
    /// expert with N_e routed tokens: pad to multiple of 32, gather X,
    /// transpose gate_up_proj[e] / down_proj[e] for the kernel layout,
    /// invoke kernel twice (gate_up + down), SwiGLU, scatter back.
    ///
    /// Gated at the call site by `GEMMA4_EXPERT_MAJOR_MOE=3`.
    pub fn forward_topk_expert_major_v3(
        &mut self,
        hidden_states: &Array,
        top_k_index: &Array,
        top_k_weights: &Array,
    ) -> Result<Array, Exception> {
        let hidden_dtype = hidden_states.dtype();
        let n = hidden_states.shape()[0];
        let h = hidden_states.shape()[1];
        let i = self.intermediate_size;
        let k_top = top_k_index.shape()[1];

        // Validate K and N alignment for the kernel.
        if h % 8 != 0 {
            return Err(Exception::from(
                "forward_topk_expert_major_v3: hidden_size must be divisible by 8 for moe_dense_matmul",
            ));
        }
        if (2 * i) % 32 != 0 || h % 32 != 0 {
            return Err(Exception::from(
                "forward_topk_expert_major_v3: 2*intermediate_size and hidden_size must be divisible by 32",
            ));
        }

        let num_experts = self.gate_up_proj.as_ref().shape()[0];
        let buckets = bucket_by_expert(top_k_index, num_experts)?;

        // Promote to fp32 once.
        let h_f32 = hidden_states.as_dtype(Dtype::Float32)?;
        let w_f32 = top_k_weights.as_dtype(Dtype::Float32)?;
        let gu_f32 = self.gate_up_proj.as_ref().as_dtype(Dtype::Float32)?;
        let dp_f32 = self.down_proj.as_ref().as_dtype(Dtype::Float32)?;

        let mut output = ops::zeros::<f32>(&[n, h])?;
        let k_top_arr = array!(k_top);
        let w_flat = w_f32.reshape(&[-1])?;

        // Per-expert dispatch.
        for bucket in &buckets {
            let n_e = bucket.n_tokens;
            if n_e == 0 {
                continue;
            }
            let e = bucket.expert_id;

            // Gather X for this expert: [N_e, H].
            let x_e = take_axis(&h_f32, &bucket.token_indices, 0)?;

            // Pad to multiple of 32 rows.
            let m_padded = ((n_e + 31) / 32) * 32;
            let pad_rows = m_padded - n_e;
            let x_pad = if pad_rows > 0 {
                let zeros = ops::zeros::<f32>(&[pad_rows, h])?;
                ops::concatenate_axis(&[&x_e, &zeros], 0)?
            } else {
                x_e
            };

            // gate_up_proj[e]: shape [2I, H] → kernel needs [H, 2I].
            let e_arr = Array::from_slice(&[e], &[1]);
            let w_gu = take_axis(&gu_f32, &e_arr, 0)?.reshape(&[2 * i, h])?;
            let w_gu_t = w_gu.transpose_axes(&[1, 0])?; // [H, 2I]

            let m_buf = Array::from_slice(&[n_e], &[1]);
            let y_padded = moe_dense_matmul(&x_pad, &w_gu_t, &m_buf, h, 2 * i)?;
            let y_e = y_padded.index((..n_e, ..)); // [N_e, 2I]

            // SwiGLU.
            let split = y_e.split(2, -1)?;
            let gate = self.activation.apply(&split[0])?;
            let z_e = gate.multiply(&split[1])?; // [N_e, I]

            // Pad again for second matmul. I must be divisible by 8.
            if i % 8 != 0 {
                return Err(Exception::from(
                    "forward_topk_expert_major_v3: intermediate_size must be divisible by 8",
                ));
            }
            let z_pad = if pad_rows > 0 {
                let zeros = ops::zeros::<f32>(&[pad_rows, i])?;
                ops::concatenate_axis(&[&z_e, &zeros], 0)?
            } else {
                z_e
            };

            // down_proj[e]: shape [H, I] → kernel needs [I, H].
            let w_d = take_axis(&dp_f32, &e_arr, 0)?.reshape(&[h, i])?;
            let w_d_t = w_d.transpose_axes(&[1, 0])?; // [I, H]
            let out_padded = moe_dense_matmul(&z_pad, &w_d_t, &m_buf, i, h)?;
            let out_e = out_padded.index((..n_e, ..)); // [N_e, H]

            // Apply per-(token, k_slot) routing weights.
            let flat_idx = bucket
                .token_indices
                .multiply(&k_top_arr)?
                .add(&bucket.k_slots)?;
            let w_e = take_axis(&w_flat, &flat_idx, 0)?; // [N_e]
            let weighted = out_e.multiply(&w_e.reshape(&[n_e, 1])?)?;

            // Scatter-add into output.
            let updates = weighted.reshape(&[n_e, 1, h])?;
            output = scatter_add_single(&output, &bucket.token_indices, &updates, 0)?;
        }

        output.as_dtype(hidden_dtype)
    }

    /// Expert-major MoE dispatch (#35 phase 1).
    ///
    /// Buckets routed (token, k_slot) pairs by expert, then runs a single
    /// dense matmul per expert over the tokens that routed there. For
    /// prefill with n>>num_experts this consolidates per-token vector
    /// matmuls into per-expert batched ones, which become tile-friendly
    /// for a follow-up simdgroup_matrix kernel (#35 phase 3).
    ///
    /// Correctness target for phase 1: argmax-match against `forward_topk`
    /// to fp32 tolerance. Uses stock MLX matmul (no custom kernel yet).
    /// Gated at the call site by `GEMMA4_EXPERT_MAJOR_MOE=1`; bypassed
    /// per-expert via `EXPERT_MAJOR_MIN_TOKENS` (default 8) below which
    /// the existing token-major matmul is faster.
    pub fn forward_topk_expert_major(
        &mut self,
        hidden_states: &Array,
        top_k_index: &Array,
        top_k_weights: &Array,
    ) -> Result<Array, Exception> {
        let hidden_dtype = hidden_states.dtype();
        let n = hidden_states.shape()[0];
        let h = hidden_states.shape()[1];
        let i = self.intermediate_size;
        let k_top = top_k_index.shape()[1];

        let num_experts = self.gate_up_proj.as_ref().shape()[0];
        let buckets = bucket_by_expert(top_k_index, num_experts)?;

        let min_tokens: i32 = std::env::var("EXPERT_MAJOR_MIN_TOKENS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8);

        // Promote to fp32 once. We accumulate the weighted output in fp32
        // (matches forward_topk's intermediate dtype) and cast back at the
        // end. For 5K-tok prefill, [n, h] fp32 ≈ 5000*5376*4 ≈ 107 MB — ok.
        let h_f32 = hidden_states.as_dtype(Dtype::Float32)?;
        let w_f32 = top_k_weights.as_dtype(Dtype::Float32)?;
        let gu_f32 = self.gate_up_proj.as_ref().as_dtype(Dtype::Float32)?;
        let dp_f32 = self.down_proj.as_ref().as_dtype(Dtype::Float32)?;

        let mut output = ops::zeros::<f32>(&[n, h])?;

        // Flatten weights so we can gather via per-bucket flat indices:
        //   flat_idx[i] = token_indices[i] * k_top + k_slots[i]
        let w_flat = w_f32.reshape(&[-1])?;
        let k_top_arr = array!(k_top);

        for bucket in &buckets {
            if bucket.n_tokens < min_tokens {
                // Skip-gate: defer to per-token kernel for tiny buckets.
                // v1 still routes through this method even for skipped
                // experts — those tokens just get a small batched matmul
                // here. A future revision can carve them out to a sibling
                // per-token path.
            }
            let e = bucket.expert_id;
            let e_arr = Array::from_slice(&[e], &[1]);

            // X_e = hidden_states[token_indices]   -> [N_e, H]
            let x_e = take_axis(&h_f32, &bucket.token_indices, 0)?;

            // W_gu = gate_up_proj[e]               -> [2I, H]
            // take_axis returns [1, 2I, H]; squeeze to [2I, H].
            let w_gu = take_axis(&gu_f32, &e_arr, 0)?.reshape(&[2 * i, h])?;
            let w_gu_t = w_gu.transpose_axes(&[1, 0])?; // [H, 2I]
            let y_e = x_e.matmul(&w_gu_t)?; // [N_e, 2I]

            // Split into gate/up, apply activation+mul.
            let split = y_e.split(2, -1)?;
            let gate = self.activation.apply(&split[0])?;
            let z_e = gate.multiply(&split[1])?; // [N_e, I]

            // W_d = down_proj[e]                    -> [H, I]
            let w_d = take_axis(&dp_f32, &e_arr, 0)?.reshape(&[h, i])?;
            let w_d_t = w_d.transpose_axes(&[1, 0])?; // [I, H]
            let out_e = z_e.matmul(&w_d_t)?; // [N_e, H]

            // Gather routing weights for these (token, k_slot) pairs.
            let flat_idx = bucket
                .token_indices
                .multiply(&k_top_arr)?
                .add(&bucket.k_slots)?;
            let w_e = take_axis(&w_flat, &flat_idx, 0)?; // [N_e]
            let weighted = out_e.multiply(&w_e.reshape(&[bucket.n_tokens, 1])?)?;

            // Scatter-add into output at the bucket's token positions.
            // Multiple buckets may write to the same token row (top_k > 1);
            // scatter_add_single handles the accumulation. MLX expects
            // updates to have shape `indices.shape + (1,) + a.shape[axis+1:]`
            // so reshape [N_e, H] → [N_e, 1, H].
            let updates = weighted.reshape(&[bucket.n_tokens, 1, h])?;
            output = scatter_add_single(&output, &bucket.token_indices, &updates, 0)?;
        }

        output.as_dtype(hidden_dtype)
    }

    pub fn training_mode(&mut self, _mode: bool) {}
}

// ============================================================================
// Decoder / model
// ============================================================================

#[derive(Debug, Clone, ModuleParameters)]
pub struct DecoderLayer {
    pub layer_idx: i32,
    pub enable_moe_block: bool,

    #[param]
    pub self_attn: Attention,
    #[param]
    pub mlp: DenseMlp,
    #[param]
    pub input_layernorm: nn::RmsNorm,
    #[param]
    pub post_attention_layernorm: nn::RmsNorm,
    #[param]
    pub pre_feedforward_layernorm: nn::RmsNorm,
    #[param]
    pub post_feedforward_layernorm: nn::RmsNorm,
    #[param]
    pub router: Option<Router>,
    #[param]
    pub experts: Option<Experts>,
    #[param]
    pub post_feedforward_layernorm_1: Option<nn::RmsNorm>,
    #[param]
    pub post_feedforward_layernorm_2: Option<nn::RmsNorm>,
    #[param]
    pub pre_feedforward_layernorm_2: Option<nn::RmsNorm>,
    #[param]
    pub layer_scalar: Param<Array>,

    // Per-layer embeddings (PLE) — present when hidden_size_per_layer_input > 0
    #[param]
    pub per_layer_input_gate: Option<MaybeQuantized<nn::Linear>>,
    #[param]
    pub per_layer_projection: Option<MaybeQuantized<nn::Linear>>,
    #[param]
    pub post_per_layer_input_norm: Option<nn::RmsNorm>,

    pub activation: GemmaActivation,
}

pub struct DecoderLayerInput<'a, C> {
    pub hidden_states: &'a Array,
    pub mask: Option<&'a Array>,
    pub cache: &'a mut C,
    /// Shared KV from a reference layer (for KV-shared layers).
    pub shared_kv: Option<(Array, Array)>,
    /// Per-layer embedding input for this layer.
    pub per_layer_input: Option<&'a Array>,
    /// Optional per-token RoPE positions; forwarded into `AttentionInput`.
    /// `None` ⇒ standard cache-offset-based sequential rotation.
    pub position_ids: Option<&'a Array>,
}

impl<C> Module<DecoderLayerInput<'_, C>> for DecoderLayer
where
    C: KeyValueCache,
{
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: DecoderLayerInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let DecoderLayerInput {
            hidden_states,
            mask,
            cache,
            shared_kv,
            per_layer_input,
            position_ids,
        } = input;

        let residual = hidden_states.clone();

        let attn_in = self.input_layernorm.forward(hidden_states)?;
        let attn_out = self.self_attn.forward(AttentionInput {
            x: &attn_in,
            mask,
            cache,
            shared_kv,
            position_ids,
        })?;
        let attn_out = self.post_attention_layernorm.forward(&attn_out)?;
        let mut hidden_states = residual.add(&attn_out)?;

        let residual = hidden_states.clone();
        let dense_hidden = self.pre_feedforward_layernorm.forward(&hidden_states)?;
        let mut ff_out = self.mlp.forward(&dense_hidden)?;

        if self.enable_moe_block {
            let mlp_branch = self
                .post_feedforward_layernorm_1
                .as_mut()
                .expect("MoE layer missing post_feedforward_layernorm_1")
                .forward(&ff_out)?;

            let hidden_states_flat = residual.reshape(&[-1, residual.shape()[2]])?;
            let (_, top_k_weights, top_k_index) = self
                .router
                .as_mut()
                .expect("MoE layer missing router")
                .forward(&hidden_states_flat)?;
            let moe_in = self
                .pre_feedforward_layernorm_2
                .as_mut()
                .expect("MoE layer missing pre_feedforward_layernorm_2")
                .forward(&hidden_states_flat)?;
            let experts = self
                .experts
                .as_mut()
                .expect("MoE layer missing experts");
            let mode = std::env::var("GEMMA4_EXPERT_MAJOR_MOE").unwrap_or_default();
            let moe_out = match mode.as_str() {
                "1" => experts.forward_topk_expert_major(&moe_in, &top_k_index, &top_k_weights)?,
                "2" => experts.forward_topk_expert_major_v2(&moe_in, &top_k_index, &top_k_weights)?,
                "3" => experts.forward_topk_expert_major_v3(&moe_in, &top_k_index, &top_k_weights)?,
                "4" => experts.forward_topk_expert_major_v4(&moe_in, &top_k_index, &top_k_weights)?,
                _ => experts.forward_topk(&moe_in, &top_k_index, &top_k_weights)?,
            };
            let moe_out = moe_out.reshape(&residual.shape())?;
            let moe_out = self
                .post_feedforward_layernorm_2
                .as_mut()
                .expect("MoE layer missing post_feedforward_layernorm_2")
                .forward(&moe_out)?;

            ff_out = mlp_branch.add(&moe_out)?;
        }

        let ff_out = self.post_feedforward_layernorm.forward(&ff_out)?;
        hidden_states = residual.add(&ff_out)?;

        // Per-layer embeddings (PLE): gated residual from auxiliary embedding
        if let (Some(gate), Some(proj), Some(norm), Some(ple_input)) = (
            self.per_layer_input_gate.as_mut(),
            self.per_layer_projection.as_mut(),
            self.post_per_layer_input_norm.as_mut(),
            per_layer_input,
        ) {
            let residual = hidden_states.clone();
            let gated = self.activation.apply(&gate.forward(&hidden_states)?)?;
            let gated = gated.multiply(ple_input)?;
            let projected = norm.forward(&proj.forward(&gated)?)?;
            hidden_states = residual.add(&projected)?;
        }

        // Layer scalar applied AFTER PLE (HF applies it as the final operation)
        hidden_states = hidden_states.multiply(&*self.layer_scalar)?;

        Ok(hidden_states)
    }

    fn training_mode(&mut self, mode: bool) {
        <Attention as Module<AttentionInput<'_, C>>>::training_mode(&mut self.self_attn, mode);
        self.mlp.training_mode(mode);
        self.input_layernorm.training_mode(mode);
        self.post_attention_layernorm.training_mode(mode);
        self.pre_feedforward_layernorm.training_mode(mode);
        self.post_feedforward_layernorm.training_mode(mode);
        if let Some(ref mut router) = self.router {
            router.training_mode(mode);
        }
        if let Some(ref mut experts) = self.experts {
            experts.training_mode(mode);
        }
        if let Some(ref mut norm) = self.post_feedforward_layernorm_1 {
            norm.training_mode(mode);
        }
        if let Some(ref mut norm) = self.post_feedforward_layernorm_2 {
            norm.training_mode(mode);
        }
        if let Some(ref mut norm) = self.pre_feedforward_layernorm_2 {
            norm.training_mode(mode);
        }
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub struct LanguageModel {
    pub vocab_size: i32,
    pub num_hidden_layers: i32,
    pub hidden_size_per_layer_input: i32,

    #[param]
    pub embed_tokens: MaybeQuantized<nn::Embedding>,
    #[param]
    pub layers: Vec<DecoderLayer>,
    #[param]
    pub norm: nn::RmsNorm,

    // Per-layer embeddings (PLE)
    #[param]
    pub embed_tokens_per_layer: Option<MaybeQuantized<nn::Embedding>>,
    #[param]
    pub per_layer_model_projection: Option<MaybeQuantized<nn::Linear>>,
    #[param]
    pub per_layer_projection_norm: Option<nn::RmsNorm>,

    /// Maps each layer index to its cache slot. Shared layers point to the
    /// same slot as their reference layer.
    pub kv_cache_map: Vec<usize>,
    /// For layers that store full-length KV for sharing, (layer_idx, cache_slot).
    pub kv_store_layers: HashSet<usize>,

    pub has_moe: bool,

    /// Pre-materialized `sqrt(hidden_size)` scale used by Gemma's embed-scale
    /// multiplication. Caching this as a constant Array avoids the per-step
    /// `Array::from(...).as_dtype(...).as_dtype(...)` allocation/cast chain
    /// that was showing up in AR-decode profiles. We eval() it once at load
    /// time so it's a materialized constant, not a deferred graph node.
    pub embed_scale: Array,
}

pub struct ModelInput<'a, C> {
    pub inputs: &'a Array,
    pub mask: Option<&'a Array>,
    pub cache: &'a mut Vec<C>,
}

impl<C> Module<ModelInput<'_, C>> for LanguageModel
where
    C: KeyValueCache + Default,
{
    type Output = Array;
    type Error = Exception;

    #[allow(non_snake_case)]
    fn forward(&mut self, input: ModelInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let ModelInput {
            inputs,
            mask,
            cache,
        } = input;

        assert!(
            !cache.is_empty(),
            "Cache must be pre-allocated with init_cache() before calling Gemma4 forward",
        );

        let mut hidden_states = self.embed_tokens.forward(inputs)?;

        // Gemma models scale embeddings by sqrt(hidden_size). Use the
        // pre-materialized `embed_scale` constant; cast to the actual hidden
        // dtype only if it differs from bf16 (no-op in the common path).
        let scale = if self.embed_scale.dtype() == hidden_states.dtype() {
            self.embed_scale.clone()
        } else {
            self.embed_scale.as_dtype(hidden_states.dtype())?
        };
        hidden_states = hidden_states.multiply(&scale)?;

        // Compute per-layer embeddings (PLE) if enabled
        let per_layer_inputs = if self.hidden_size_per_layer_input > 0 {
            let ple_dim = self.hidden_size_per_layer_input;
            let num_layers = self.num_hidden_layers;

            // Auxiliary embedding: embed_tokens_per_layer(input_ids)
            let ple_embed = self
                .embed_tokens_per_layer
                .as_mut()
                .expect("PLE embed_tokens_per_layer missing")
                .forward(inputs)?;
            // Scale by sqrt(ple_dim), cast through BF16
            let ple_scale = Array::from((ple_dim as f32).sqrt())
                .as_dtype(Dtype::Bfloat16)?
                .as_dtype(ple_embed.dtype())?;
            let ple_embed = ple_embed.multiply(&ple_scale)?;
            let B = ple_embed.shape()[0];
            let L = ple_embed.shape()[1];
            let ple_embed = ple_embed.reshape(&[B, L, num_layers, ple_dim])?;

            // Model projection: project main embeddings to per-layer space
            let proj = self
                .per_layer_model_projection
                .as_mut()
                .expect("PLE per_layer_model_projection missing")
                .forward(&hidden_states)?;
            // PLE projection inverse-sqrt scaling. Recompute locally since
            // the cached `embed_scale` is the forward sqrt(H), not 1/sqrt(H).
            let proj_scale = array!((hidden_states.shape()[2] as f32).powf(-0.5));
            let proj = proj.multiply(&proj_scale)?;
            let proj = proj.reshape(&[B, L, num_layers, ple_dim])?;
            let proj = self
                .per_layer_projection_norm
                .as_mut()
                .expect("PLE per_layer_projection_norm missing")
                .forward(&proj)?;

            // Combine: (projection + embedding) * 2^-0.5
            let combined = proj.add(&ple_embed)?;
            let input_scale = array!(std::f32::consts::FRAC_1_SQRT_2);
            Some(combined.multiply(&input_scale)?)
        } else {
            None
        };

        // During prefill with MoE, eval each layer to free expert-gather
        // intermediates. Per-layer eval is the original behaviour and
        // turns out to be the right call empirically — coarser strides
        // grow the per-chunk working set enough to trigger MoE memory
        // pressure / thrashing on long prompts (3.4× slower at stride=5
        // on hermes-5K). Override via GEMMA4_EVAL_LAYER_STRIDE if you
        // know the workload fits.
        let is_prefill = inputs.shape()[1] > 1;
        let eval_layer_stride: usize = std::env::var("GEMMA4_EVAL_LAYER_STRIDE")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n: &usize| n > 0)
            .unwrap_or(1);
        let total_layers = self.layers.len();

        // Track shared KV: layers that store full-length KV for later sharing
        let mut shared_kv_store: HashMap<usize, (Array, Array)> = HashMap::new();

        for (i, layer) in self.layers.iter_mut().enumerate() {
            let cache_slot = self.kv_cache_map[i];
            let is_shared = layer.self_attn.is_kv_shared;

            // For KV-shared layers, retrieve pre-computed KV from the store
            let shared_kv = if is_shared {
                shared_kv_store.get(&cache_slot).cloned()
            } else {
                None
            };

            // Extract per-layer input slice for this layer
            let ple_slice = per_layer_inputs
                .as_ref()
                .map(|p| p.index((.., .., i as i32, ..)));

            hidden_states = layer.forward(DecoderLayerInput {
                hidden_states: &hidden_states,
                mask,
                cache: &mut cache[cache_slot],
                shared_kv,
                per_layer_input: ple_slice.as_ref(),
                position_ids: None,
            })?;

            // If this layer stores KV for sharing, snapshot the sliced KV
            if self.kv_store_layers.contains(&i) {
                if let Some(kv) = cache[cache_slot].current_kv() {
                    shared_kv_store.insert(cache_slot, kv);
                }
            }

            if is_prefill
                && self.has_moe
                && ((i + 1) % eval_layer_stride == 0 || i + 1 == total_layers)
            {
                mlx_rs::transforms::eval([&hidden_states])?;
            }
        }

        self.norm.forward(&hidden_states)
    }

    fn training_mode(&mut self, mode: bool) {
        self.embed_tokens.training_mode(mode);
        for layer in &mut self.layers {
            <DecoderLayer as Module<DecoderLayerInput<'_, C>>>::training_mode(layer, mode);
        }
        self.norm.training_mode(mode);
    }
}

impl LanguageModel {
    /// Forward pass from pre-computed embeddings.
    ///
    /// `per_layer_inputs`: optional `[B, L, num_layers, ple_dim]` tensor computed by
    /// `compute_per_layer_inputs`.  Must be provided when `hidden_size_per_layer_input > 0`
    /// (i.e. E4B-style PLE models); may be `None` for non-PLE models.
    pub fn forward_from_embeds<C>(
        &mut self,
        inputs_embeds: &Array,
        mask: Option<&Array>,
        cache: &mut Vec<C>,
        per_layer_inputs: Option<&Array>,
    ) -> Result<Array, Exception>
    where
        C: KeyValueCache + Default,
    {
        assert!(
            !cache.is_empty(),
            "Cache must be pre-allocated with init_cache() before calling Gemma4 forward_from_embeds",
        );

        let mut hidden_states = inputs_embeds.clone();

        let mut shared_kv_store: HashMap<usize, (Array, Array)> = HashMap::new();

        for (i, layer) in self.layers.iter_mut().enumerate() {
            let cache_slot = self.kv_cache_map[i];
            let is_shared = layer.self_attn.is_kv_shared;
            let shared_kv = if is_shared {
                shared_kv_store.get(&cache_slot).cloned()
            } else {
                None
            };

            // Slice per-layer input for this layer: [B, L, ple_dim]
            let ple_slice = per_layer_inputs.map(|p| p.index((.., .., i as i32, ..)));

            hidden_states = layer.forward(DecoderLayerInput {
                hidden_states: &hidden_states,
                mask,
                cache: &mut cache[cache_slot],
                shared_kv,
                per_layer_input: ple_slice.as_ref(),
                position_ids: None,
            })?;

            if self.kv_store_layers.contains(&i) {
                if let Some(kv) = cache[cache_slot].current_kv() {
                    shared_kv_store.insert(cache_slot, kv);
                }
            }
        }

        self.norm.forward(&hidden_states)
    }

    /// Compute the per-layer-embedding (PLE) tensor for the given sequence.
    ///
    /// Returns `Ok(Some(...))` with shape `[B, L, num_layers, ple_dim]` when PLE is
    /// enabled (`hidden_size_per_layer_input > 0`), or `Ok(None)` for non-PLE models.
    ///
    /// `input_ids`: `[B, L]` token IDs (image positions may use `image_token_id`).
    /// `hidden_states`: `[B, L, hidden_size]` scaled input embeddings.
    pub fn compute_per_layer_inputs(
        &mut self,
        input_ids: &Array,
        hidden_states: &Array,
    ) -> Result<Option<Array>, Exception> {
        if self.hidden_size_per_layer_input <= 0 {
            return Ok(None);
        }
        let ple_dim = self.hidden_size_per_layer_input;
        let num_layers = self.num_hidden_layers;
        let hidden_dim = hidden_states.shape()[2] as f32;

        let ple_embed = self
            .embed_tokens_per_layer
            .as_mut()
            .expect("PLE embed_tokens_per_layer missing")
            .forward(input_ids)?;
        let ple_scale = Array::from((ple_dim as f32).sqrt())
            .as_dtype(Dtype::Bfloat16)?
            .as_dtype(ple_embed.dtype())?;
        let ple_embed = ple_embed.multiply(&ple_scale)?;
        let b = ple_embed.shape()[0];
        let l = ple_embed.shape()[1];
        let ple_embed = ple_embed.reshape(&[b, l, num_layers, ple_dim])?;

        let proj = self
            .per_layer_model_projection
            .as_mut()
            .expect("PLE per_layer_model_projection missing")
            .forward(hidden_states)?;
        let proj_scale = array!((hidden_dim).powf(-0.5));
        let proj = proj.multiply(&proj_scale)?;
        let proj = proj.reshape(&[b, l, num_layers, ple_dim])?;
        let proj = self
            .per_layer_projection_norm
            .as_mut()
            .expect("PLE per_layer_projection_norm missing")
            .forward(&proj)?;

        let combined = proj.add(&ple_embed)?;
        let input_scale = array!(std::f32::consts::FRAC_1_SQRT_2);
        Ok(Some(combined.multiply(&input_scale)?))
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub struct Model {
    pub args: Gemma4TextConfig,

    #[param]
    pub model: LanguageModel,
    #[param]
    pub lm_head: Option<MaybeQuantized<nn::Linear>>,
    /// Present when weights ship pre-quantized — needed at runtime by helpers
    /// like `get_lm_head_weight()` that dequantize on demand.
    pub quantization: Option<QuantizationConfig>,
}

impl<C> Module<ModelInput<'_, C>> for Model
where
    C: KeyValueCache + Default,
{
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: ModelInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let out = self.model.forward(input)?;
        let mut logits = match self.lm_head.as_mut() {
            Some(lm_head) => lm_head.forward(&out)?,
            None => mq_embedding_as_linear(&mut self.model.embed_tokens, &out)?,
        };
        if let Some(softcap) = self.args.final_logit_softcapping {
            let cap = array!(softcap);
            logits = ops::tanh(&logits.divide(&cap)?)?.multiply(&cap)?;
        }
        Ok(logits)
    }

    fn training_mode(&mut self, mode: bool) {
        <LanguageModel as Module<ModelInput<'_, C>>>::training_mode(&mut self.model, mode);
        if let Some(ref mut lm_head) = self.lm_head {
            lm_head.training_mode(mode);
        }
    }
}

impl Model {
    /// Project only the last sequence position through the LM head.
    pub fn forward_last_logits<C>(
        &mut self,
        input: ModelInput<'_, C>,
    ) -> Result<Array, Exception>
    where
        C: KeyValueCache + Default,
    {
        let out = self.model.forward(input)?;
        let last = out.index((.., -1, ..));
        let mut logits = match self.lm_head.as_mut() {
            Some(lm_head) => lm_head.forward(&last)?,
            None => mq_embedding_as_linear(&mut self.model.embed_tokens, &last)?,
        };
        if let Some(softcap) = self.args.final_logit_softcapping {
            let cap = array!(softcap);
            logits = ops::tanh(&logits.divide(&cap)?)?.multiply(&cap)?;
        }
        Ok(logits)
    }

    pub fn forward_from_embeds<C>(
        &mut self,
        embeds: &Array,
        cache: &mut Vec<C>,
        per_layer_inputs: Option<&Array>,
    ) -> Result<Array, Exception>
    where
        C: KeyValueCache + Default,
    {
        let out = self.model.forward_from_embeds(embeds, None, cache, per_layer_inputs)?;
        let last = out.index((.., -1, ..));
        let mut logits = match self.lm_head.as_mut() {
            Some(lm_head) => lm_head.forward(&last)?,
            None => mq_embedding_as_linear(&mut self.model.embed_tokens, &last)?,
        };
        if let Some(softcap) = self.args.final_logit_softcapping {
            let cap = array!(softcap);
            logits = ops::tanh(&logits.divide(&cap)?)?.multiply(&cap)?;
        }
        Ok(logits)
    }

    pub fn embed_tokens_slice(&mut self, ids: &[i32]) -> Result<Array, Exception> {
        let input = Array::from_slice(ids, &[1, ids.len() as i32]);
        let embeds = self.model.embed_tokens.forward(&input)?;
        embeds.reshape(&[ids.len() as i32, self.args.hidden_size])
    }

    /// Look up token embeddings for `ids`, returning shape `[1, T, hidden_size]`.
    /// Used by DFlash adapters to stage a single token embedding (shape `[1, 1, H]`).
    pub fn embed_tokens(&mut self, ids: &[i32]) -> Result<Array, Exception> {
        let input = Array::from_slice(ids, &[1, ids.len() as i32]);
        self.model.embed_tokens.forward(&input)
    }

    /// Return the LM head weight matrix `[vocab, hidden_size]`. Prefers the
    /// explicit `lm_head.weight` when present; otherwise falls back to the
    /// tied embedding table. Used by DFlash to project the draft model's
    /// hidden output back to vocab logits.
    pub fn get_lm_head_weight(&self) -> Result<Array, Exception> {
        if let Some(lm_head) = self.lm_head.as_ref() {
            return mq_linear_dequant_weight(lm_head, self.quantization.as_ref());
        }
        mq_embedding_dequant_weight(&self.model.embed_tokens, self.quantization.as_ref())
    }

    /// DDTree fused-tree forward.
    ///
    /// Runs a single target forward over the flat list of tree tokens
    /// `[1, L]` with explicit per-token `position_ids [L]` and a custom
    /// `[L, kv_offset + L]` attention mask (additive log-probs style:
    /// 0 for visible, large-negative for blocked).
    ///
    /// Returns `[1, L, vocab]` per-position logits. The target's KV cache
    /// grows by L slots; caller is responsible for trimming/compacting
    /// rejected tree positions before the next forward.
    /// As `forward_tree` but also returns per-token post-norm hidden so
    /// callers (DDTree) can derive next-cycle root_pred from the
    /// last-accepted-node hidden without an extra LM head call.
    pub fn forward_tree_with_hidden(
        &mut self,
        tokens: &Array,
        position_ids: &Array,
        attention_mask: &Array,
        cache: &mut Vec<KVCache>,
    ) -> Result<(Array, Array), Exception> {
        if self.args.num_kv_shared_layers > 0 {
            return Err(Exception::custom(
                "forward_tree_with_hidden: shared KV layers not supported in the DDTree spike path",
            ));
        }
        if self.model.hidden_size_per_layer_input > 0 {
            return Err(Exception::custom(
                "forward_tree_with_hidden: PLE not supported in the DDTree spike path",
            ));
        }
        let mut h = self.model.embed_tokens.forward(tokens)?;
        let scale = if self.model.embed_scale.dtype() == h.dtype() {
            self.model.embed_scale.clone()
        } else {
            self.model.embed_scale.as_dtype(h.dtype())?
        };
        h = h.multiply(&scale)?;
        for (i, layer) in self.model.layers.iter_mut().enumerate() {
            let cache_slot = self.model.kv_cache_map[i];
            h = layer.forward(DecoderLayerInput {
                hidden_states: &h,
                mask: Some(attention_mask),
                cache: &mut cache[cache_slot],
                shared_kv: None,
                per_layer_input: None,
                position_ids: Some(position_ids),
            })?;
        }
        let normed = self.model.norm.forward(&h)?;
        let mut logits = match self.lm_head.as_mut() {
            Some(lm) => lm.forward(&normed)?,
            None => mq_embedding_as_linear(&mut self.model.embed_tokens, &normed)?,
        };
        if let Some(softcap) = self.args.final_logit_softcapping {
            let cap = array!(softcap);
            logits = ops::tanh(&logits.divide(&cap)?)?.multiply(&cap)?;
        }
        Ok((logits, normed))
    }

    pub fn forward_tree(
        &mut self,
        tokens: &Array,
        position_ids: &Array,
        attention_mask: &Array,
        cache: &mut Vec<KVCache>,
    ) -> Result<Array, Exception> {
        if self.args.num_kv_shared_layers > 0 {
            return Err(Exception::custom(
                "forward_tree: shared KV layers not supported in the DDTree spike path",
            ));
        }
        if self.model.hidden_size_per_layer_input > 0 {
            return Err(Exception::custom(
                "forward_tree: PLE not supported in the DDTree spike path",
            ));
        }
        let mut h = self.model.embed_tokens.forward(tokens)?;
        let scale = if self.model.embed_scale.dtype() == h.dtype() {
            self.model.embed_scale.clone()
        } else {
            self.model.embed_scale.as_dtype(h.dtype())?
        };
        h = h.multiply(&scale)?;

        for (i, layer) in self.model.layers.iter_mut().enumerate() {
            let cache_slot = self.model.kv_cache_map[i];
            h = layer.forward(DecoderLayerInput {
                hidden_states: &h,
                mask: Some(attention_mask),
                cache: &mut cache[cache_slot],
                shared_kv: None,
                per_layer_input: None,
                position_ids: Some(position_ids),
            })?;
        }
        let normed = self.model.norm.forward(&h)?;
        let mut logits = match self.lm_head.as_mut() {
            Some(lm) => lm.forward(&normed)?,
            None => mq_embedding_as_linear(&mut self.model.embed_tokens, &normed)?,
        };
        if let Some(softcap) = self.args.final_logit_softcapping {
            let cap = array!(softcap);
            logits = ops::tanh(&logits.divide(&cap)?)?.multiply(&cap)?;
        }
        Ok(logits)
    }

    /// Project a pre-LM-head hidden state (post-final-norm) through the LM
    /// head to get logits. Used by external verify loops that have already
    /// captured the hidden state and just need the projection.
    pub fn forward_via_hidden(&mut self, hidden: &Array) -> Result<Array, Exception> {
        let mut logits = match self.lm_head.as_mut() {
            Some(lm) => lm.forward(hidden)?,
            None => mq_embedding_as_linear(&mut self.model.embed_tokens, hidden)?,
        };
        if let Some(softcap) = self.args.final_logit_softcapping {
            let cap = array!(softcap);
            logits = ops::tanh(&logits.divide(&cap)?)?.multiply(&cap)?;
        }
        Ok(logits)
    }

    /// Forward pass that captures hidden states after each requested layer index
    /// AND returns last-position logits.  Returns `(logits[B, vocab], captures[B, T, K*H])`
    /// where K = `target_layer_ids.len()` and captures are concatenated along the last axis
    /// in the same order as `target_layer_ids`.
    ///
    /// This is the Rust mirror of the Python `Gemma4TargetOps.forward_with_hidden_capture`
    /// path used by DFlash speculative decoding. It is intended for the 26B-A4B variant
    /// only: it asserts that `hidden_size_per_layer_input == 0` (no PLE) and that
    /// `num_kv_shared_layers == 0` (no KV sharing). Both of these code paths are
    /// deliberately skipped here to keep the function tight; callers with E4B-style
    /// configurations should use the standard forward path.
    /// Forward with per-layer hidden capture + per-position logits `[B, T, V]`.
    /// Used by DFlash verify, which indexes each position's predicted-next-
    /// token distribution. Substantially more expensive than the last-only
    /// variant because the LM head matmul scales linearly with T.
    pub fn forward_with_hidden_capture<C>(
        &mut self,
        input_ids: &Array,
        cache: &mut Vec<C>,
        target_layer_ids: &[usize],
    ) -> Result<(Array, Array), Exception>
    where
        C: KeyValueCache + Default,
    {
        self.forward_with_hidden_capture_impl(input_ids, cache, target_layer_ids, false)
    }

    /// Forward with per-layer hidden capture + **last-position-only** logits
    /// `[B, V]`. Used by DFlash prefill, where only the staged-token logits
    /// are sampled. The LM head matmul cost drops from O(T*V*H) to O(V*H),
    /// which is significant at vocab=262k on Gemma4.
    pub fn forward_last_logits_with_hidden_capture<C>(
        &mut self,
        input_ids: &Array,
        cache: &mut Vec<C>,
        target_layer_ids: &[usize],
    ) -> Result<(Array, Array), Exception>
    where
        C: KeyValueCache + Default,
    {
        self.forward_with_hidden_capture_impl(input_ids, cache, target_layer_ids, true)
    }

    fn forward_with_hidden_capture_impl<C>(
        &mut self,
        input_ids: &Array,
        cache: &mut Vec<C>,
        target_layer_ids: &[usize],
        last_only: bool,
    ) -> Result<(Array, Array), Exception>
    where
        C: KeyValueCache + Default,
    {
        if self.model.hidden_size_per_layer_input > 0 {
            return Err(Exception::custom(
                "forward_with_hidden_capture: PLE (hidden_size_per_layer_input > 0) is not supported \
                 by the DFlash capture path (26B-A4B only)",
            ));
        }
        if self.args.num_kv_shared_layers > 0 {
            return Err(Exception::custom(
                "forward_with_hidden_capture: shared KV layers are not supported by the DFlash \
                 capture path (26B-A4B only)",
            ));
        }
        if let Some(&layer_id) = target_layer_ids
            .iter()
            .find(|&&layer_id| layer_id >= self.model.layers.len())
        {
            return Err(Exception::custom(format!(
                "capture layer index {layer_id} out of range for {} layers",
                self.model.layers.len()
            )));
        }
        assert!(
            !cache.is_empty(),
            "Cache must be pre-allocated with init_cache() before calling forward_with_hidden_capture",
        );

        // Embed and apply Gemma's sqrt(hidden_size) embed scale (matches the Python
        // reference's `h = input_embeddings * embed_scale`). Reuse the
        // pre-materialized constant from LanguageModel.
        let mut hidden_states = self.model.embed_tokens.forward(input_ids)?;
        let scale = if self.model.embed_scale.dtype() == hidden_states.dtype() {
            self.model.embed_scale.clone()
        } else {
            self.model.embed_scale.as_dtype(hidden_states.dtype())?
        };
        hidden_states = hidden_states.multiply(&scale)?;

        // Build a causal mask for prefill, none for single-step decode.
        // The mask offset must reflect any cached prefix already in the KV
        // cache so verify blocks attend to the right key positions:
        //   - fresh prefill (cache empty): offset=0, mask [T, T].
        //   - verify on top of a prefill (cache has P committed tokens):
        //     offset=P, mask [T, P+T] so each query position attends to
        //     all P cached tokens + its causal-prefix verify positions.
        let t = hidden_states.shape()[1];
        let cache_offset: i32 = cache.first().map(|c| c.offset()).unwrap_or(0);
        let mask_arr = if t > 1 {
            Some(create_causal_mask(t, Some(cache_offset), None, None)?)
        } else {
            None
        };
        let mask_ref = mask_arr.as_ref();

        // Per-layer eval was originally needed during long-prompt prefill on
        // MoE to bound peak Metal memory. For short multi-token forwards
        // (DFlash verify blocks of 4-16 tokens) it's catastrophic — 47
        // synchronous barriers per cycle on a 26B model. Only force eval
        // when t is large enough to justify the cost (~ a prefill chunk).
        let is_long_prefill = t > 64;
        // Same per-layer-eval batching as `LanguageModel::forward`.
        // Default 1 (per-layer); higher values trigger MoE thrashing on
        // long prompts.
        let eval_layer_stride: usize = std::env::var("GEMMA4_EVAL_LAYER_STRIDE")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n: &usize| n > 0)
            .unwrap_or(1);
        let total_layers = self.model.layers.len();
        let mut captures = Vec::with_capacity(target_layer_ids.len());

        for (i, layer) in self.model.layers.iter_mut().enumerate() {
            let cache_slot = self.model.kv_cache_map[i];
            hidden_states = layer.forward(DecoderLayerInput {
                hidden_states: &hidden_states,
                mask: mask_ref,
                cache: &mut cache[cache_slot],
                shared_kv: None,
                per_layer_input: None,
                position_ids: None,
            })?;

            if target_layer_ids.iter().any(|&id| id == i) {
                captures.push(hidden_states.clone());
            }

            if is_long_prefill
                && self.model.has_moe
                && ((i + 1) % eval_layer_stride == 0 || i + 1 == total_layers)
            {
                eval([&hidden_states])?;
            }
        }

        let normalized = self.model.norm.forward(&hidden_states)?;
        // last_only: collapse to [B, 1, H] BEFORE the LM head matmul to skip
        // (T-1)/T of the cost. Verify path keeps the full [B, T, H] so each
        // drafted position's logits are available downstream.
        let lm_in = if last_only {
            normalized.index((.., -1, ..))
        } else {
            normalized
        };
        let mut logits = match self.lm_head.as_mut() {
            Some(lm_head) => lm_head.forward(&lm_in)?,
            None => mq_embedding_as_linear(&mut self.model.embed_tokens, &lm_in)?,
        };
        if let Some(softcap) = self.args.final_logit_softcapping {
            let cap = array!(softcap);
            logits = ops::tanh(&logits.divide(&cap)?)?.multiply(&cap)?;
        }

        let captures_concat = if captures.is_empty() {
            Array::zeros::<f32>(&[hidden_states.shape()[0], t, 0])?
        } else {
            let refs: Vec<&Array> = captures.iter().collect();
            ops::concatenate_axis(&refs, 2)?
        };
        Ok((logits, captures_concat))
    }
}

/// Snapshot helper for DFlash KV rollback on gemma4 caches.
///
/// The Python reference implements `_trim_recent_cache(cache, n)` for both
/// `KVCache` (decrement `offset`) and `RotatingKVCache` (rewind `_idx` +
/// re-temporalize the ring buffer). The Rust `mlx_rs_core::cache::KVCache`
/// exposes only `reset()` (offset → 0) and no direct offset setter — and per
/// the DFlash worktree constraints we may not modify `mlx-rs-core`. We
/// therefore implement DFlash KV rollback the same way `Qwen36TargetAdapter`
/// does: snapshot `Vec<KVCache>` via `Clone` before each verify, and on
/// rollback restore the snapshot then replay the kept prefix tokens through
/// the forward pass.
///
/// 26B-A4B does NOT use `RotatingKVCache` — sliding attention is enforced by
/// the per-layer attention mask (`create_causal_mask(..., Some(window), ..)`),
/// not by a ring-buffered KV. So snapshot+replay correctly handles sliding
/// layers without any special casing.
///
/// `snapshot_cache` returns a deep clone (mlx_rs `Array` is reference-counted,
/// so the clone is cheap until the underlying buffer is mutated by
/// `update_and_fetch`, at which point the snapshot keeps the pre-mutation
/// arrays alive).
pub fn snapshot_cache(cache: &[KVCache]) -> Vec<KVCache> {
    cache.to_vec()
}

/// Restore a previously snapshotted cache vector in-place. Mirrors the
/// rollback step of `_trim_recent_cache`. Callers are expected to follow this
/// with a forward pass over the kept-prefix tokens to re-advance the offset.
pub fn restore_cache(cache: &mut Vec<KVCache>, snapshot: Vec<KVCache>) {
    *cache = snapshot;
}

// ============================================================================
// Loading
// ============================================================================

pub fn load_tokenizer(model_dir: impl AsRef<Path>) -> Result<Tokenizer, Error> {
    let file = model_dir.as_ref().join("tokenizer.json");
    Tokenizer::from_file(file).map_err(Into::into)
}

pub fn get_model_args(model_dir: impl AsRef<Path>) -> Result<Gemma4Config, Error> {
    let model_args_filename = model_dir.as_ref().join("config.json");
    let file = std::fs::File::open(model_args_filename)?;
    serde_json::from_reader(file).map_err(Into::into)
}

fn keep_language_weight(key: &str) -> bool {
    key.starts_with("model.language_model.")
        || matches!(
            key,
            "lm_head.weight" | "model.lm_head.weight" | "model.language_model.lm_head.weight"
        )
}

fn load_all_weights(model_dir: &Path) -> Result<HashMap<String, Array>, Error> {
    let weights_index = model_dir.join("model.safetensors.index.json");
    let single_file = model_dir.join("model.safetensors");

    let weight_files: Vec<std::path::PathBuf> = if weights_index.exists() {
        // Sharded: read index to find relevant shard files
        let json = std::fs::read_to_string(&weights_index)?;
        let weight_map: WeightMap = serde_json::from_str(&json)?;
        weight_map
            .weight_map
            .iter()
            .filter(|(key, _)| keep_language_weight(key))
            .map(|(_, file)| model_dir.join(file))
            .collect::<HashSet<_>>()
            .into_iter()
            .collect()
    } else if single_file.exists() {
        // Single file: load it directly
        vec![single_file]
    } else {
        return Err(Error::Model(
            "No model.safetensors or model.safetensors.index.json found".to_string(),
        ));
    };

    let mut all_weights = HashMap::new();
    for weights_filename in weight_files {
        let loaded = Array::load_safetensors(&weights_filename)?;
        for (key, value) in loaded {
            if keep_language_weight(&key) {
                all_weights.insert(key, value);
            }
        }
    }

    Ok(all_weights)
}

pub fn load_all_weights_unfiltered(model_dir: &Path) -> Result<HashMap<String, Array>, Error> {
    let weights_index = model_dir.join("model.safetensors.index.json");
    let single_file = model_dir.join("model.safetensors");

    let weight_files: Vec<std::path::PathBuf> = if weights_index.exists() {
        let json = std::fs::read_to_string(&weights_index)?;
        let weight_map: WeightMap = serde_json::from_str(&json)?;
        weight_map
            .weight_map
            .values()
            .map(|file| model_dir.join(file))
            .collect::<HashSet<_>>()
            .into_iter()
            .collect()
    } else if single_file.exists() {
        vec![single_file]
    } else {
        return Err(Error::Model(
            "No model.safetensors or model.safetensors.index.json found".to_string(),
        ));
    };

    let mut all_weights = HashMap::new();
    for weights_filename in weight_files {
        all_weights.extend(Array::load_safetensors(&weights_filename)?);
    }
    Ok(all_weights)
}

fn get_weight(weights: &HashMap<String, Array>, key: &str) -> Result<Array, Error> {
    weights
        .get(key)
        .cloned()
        .ok_or_else(|| Error::Model(format!("Weight not found: {key}")))
}

fn get_weight_optional(weights: &HashMap<String, Array>, key: &str) -> Option<Array> {
    weights.get(key).cloned()
}

fn get_first_weight(weights: &HashMap<String, Array>, keys: &[&str]) -> Option<Array> {
    keys.iter()
        .find_map(|key| get_weight_optional(weights, key))
}

fn make_linear(weight: Array) -> nn::Linear {
    nn::Linear {
        weight: Param::new(weight),
        bias: Param::new(None::<Array>),
    }
}

/// Build a `MaybeQuantized<nn::Linear>` from weights. If the `(prefix.weight,
/// prefix.scales, prefix.biases)` triplet is present, returns a
/// `Quantized` variant for native `quantized_matmul`. Otherwise returns the
/// `Original` BF16 path.
fn make_mq_linear(
    weights: &HashMap<String, Array>,
    prefix: &str,
    quant: Option<&QuantizationConfig>,
) -> Result<MaybeQuantized<nn::Linear>, Error> {
    let weight = get_weight(weights, &format!("{prefix}.weight"))?;
    if let Some(q) = quant {
        if let (Some(scales), Some(biases)) = (
            get_weight_optional(weights, &format!("{prefix}.scales")),
            get_weight_optional(weights, &format!("{prefix}.biases")),
        ) {
            let inner = make_linear(weight);
            let mut ql = nn::QuantizedLinear {
                group_size: q.group_size,
                bits: q.bits,
                scales: Param::new(scales),
                biases: Param::new(biases),
                inner,
            };
            mlx_rs::module::ModuleParameters::freeze_parameters(&mut ql, true);
            return Ok(MaybeQuantized::Quantized(ql));
        }
    }
    Ok(MaybeQuantized::Original(make_linear(weight)))
}

fn make_mq_linear_optional(
    weights: &HashMap<String, Array>,
    prefix: &str,
    quant: Option<&QuantizationConfig>,
) -> Option<MaybeQuantized<nn::Linear>> {
    if get_weight_optional(weights, &format!("{prefix}.weight")).is_none() {
        return None;
    }
    make_mq_linear(weights, prefix, quant).ok()
}

/// Fuse 2-3 quantized linears into one by concatenating weight/scales/biases
/// along the output axis. All inputs must share `group_size`, `bits`, and
/// input dimension. Used by the fused-QKV opt-in to cut three matmul kernel
/// launches per attention forward down to one.
///
/// Returns a fused `MaybeQuantized<nn::Linear>` and the per-part output
/// sizes so the caller knows where to split the output (Q | K | V).
fn fuse_qkv_linears(
    q: &MaybeQuantized<nn::Linear>,
    k: &MaybeQuantized<nn::Linear>,
    v: Option<&MaybeQuantized<nn::Linear>>,
) -> Result<(MaybeQuantized<nn::Linear>, Vec<i32>), Error> {
    use mlx_rs::ops::concatenate_axis;
    match (q, k) {
        (MaybeQuantized::Quantized(qq), MaybeQuantized::Quantized(qk)) => {
            if qq.group_size != qk.group_size || qq.bits != qk.bits {
                return Err(Error::Model(
                    "fuse_qkv: q/k group_size or bits mismatch".into(),
                ));
            }
            let qw: &Array = qq.inner.weight.as_ref();
            let kw: &Array = qk.inner.weight.as_ref();
            let q_out = qw.shape()[0];
            let k_out = kw.shape()[0];
            let mut weights: Vec<&Array> = vec![qw, kw];
            let qs: &Array = qq.scales.as_ref();
            let ks: &Array = qk.scales.as_ref();
            let mut scales: Vec<&Array> = vec![qs, ks];
            let qb: &Array = qq.biases.as_ref();
            let kb: &Array = qk.biases.as_ref();
            let mut biases: Vec<&Array> = vec![qb, kb];
            let mut splits = vec![q_out, k_out];
            if let Some(MaybeQuantized::Quantized(qv)) = v {
                if qv.group_size != qq.group_size || qv.bits != qq.bits {
                    return Err(Error::Model(
                        "fuse_qkv: v group_size or bits mismatch".into(),
                    ));
                }
                let vw: &Array = qv.inner.weight.as_ref();
                let vs: &Array = qv.scales.as_ref();
                let vb: &Array = qv.biases.as_ref();
                splits.push(vw.shape()[0]);
                weights.push(vw);
                scales.push(vs);
                biases.push(vb);
            }
            let fused_w = concatenate_axis(&weights, 0)
                .map_err(|e| Error::Model(format!("fuse_qkv weight concat: {e}")))?;
            let fused_s = concatenate_axis(&scales, 0)
                .map_err(|e| Error::Model(format!("fuse_qkv scales concat: {e}")))?;
            let fused_b = concatenate_axis(&biases, 0)
                .map_err(|e| Error::Model(format!("fuse_qkv biases concat: {e}")))?;
            let inner = make_linear(fused_w);
            let mut ql = nn::QuantizedLinear {
                group_size: qq.group_size,
                bits: qq.bits,
                scales: Param::new(fused_s),
                biases: Param::new(fused_b),
                inner,
            };
            mlx_rs::module::ModuleParameters::freeze_parameters(&mut ql, true);
            Ok((MaybeQuantized::Quantized(ql), splits))
        }
        // Original (non-quantized) path: concat .weight along axis 0.
        (MaybeQuantized::Original(lq), MaybeQuantized::Original(lk)) => {
            let qw: &Array = lq.weight.as_ref();
            let kw: &Array = lk.weight.as_ref();
            let mut weights: Vec<&Array> = vec![qw, kw];
            let mut splits = vec![qw.shape()[0], kw.shape()[0]];
            if let Some(MaybeQuantized::Original(lv)) = v {
                let vw: &Array = lv.weight.as_ref();
                splits.push(vw.shape()[0]);
                weights.push(vw);
            }
            let fused = concatenate_axis(&weights, 0)
                .map_err(|e| Error::Model(format!("fuse_qkv concat: {e}")))?;
            Ok((MaybeQuantized::Original(make_linear(fused)), splits))
        }
        _ => Err(Error::Model(
            "fuse_qkv: q/k must both be Quantized or both Original".into(),
        )),
    }
}

fn make_mq_embedding(
    weights: &HashMap<String, Array>,
    prefix: &str,
    quant: Option<&QuantizationConfig>,
) -> Result<MaybeQuantized<nn::Embedding>, Error> {
    let weight = get_weight(weights, &format!("{prefix}.weight"))?;
    if let Some(q) = quant {
        if let (Some(scales), Some(biases)) = (
            get_weight_optional(weights, &format!("{prefix}.scales")),
            get_weight_optional(weights, &format!("{prefix}.biases")),
        ) {
            let inner = nn::Embedding {
                weight: Param::new(weight),
            };
            let mut qe = nn::QuantizedEmbedding {
                group_size: q.group_size,
                bits: q.bits,
                scales: Param::new(scales),
                biases: Param::new(biases),
                inner,
            };
            mlx_rs::module::ModuleParameters::freeze_parameters(&mut qe, true);
            return Ok(MaybeQuantized::Quantized(qe));
        }
    }
    Ok(MaybeQuantized::Original(nn::Embedding {
        weight: Param::new(weight),
    }))
}

/// Apply either an embedding or its quantized counterpart as a linear
/// projection (tied lm_head fallback).
fn mq_embedding_as_linear(
    embed: &mut MaybeQuantized<nn::Embedding>,
    x: &Array,
) -> Result<Array, Exception> {
    match embed {
        MaybeQuantized::Original(e) => e.as_linear(x),
        MaybeQuantized::Quantized(qe) => qe.as_linear(x),
    }
}

/// Return the LM head weight as a dequantized BF16 tensor `[vocab, hidden]`.
fn mq_embedding_dequant_weight(
    embed: &MaybeQuantized<nn::Embedding>,
    quant: Option<&QuantizationConfig>,
) -> Result<Array, Exception> {
    match embed {
        MaybeQuantized::Original(e) => Ok(e.weight.as_ref().clone()),
        MaybeQuantized::Quantized(qe) => {
            let q = quant.expect("quantized embedding requires QuantizationConfig");
            ops::dequantize(
                &*qe.inner.weight,
                &*qe.scales,
                &*qe.biases,
                q.group_size,
                q.bits,
                None::<&str>,
            )
        }
    }
}

fn mq_linear_dequant_weight(
    lin: &MaybeQuantized<nn::Linear>,
    quant: Option<&QuantizationConfig>,
) -> Result<Array, Exception> {
    match lin {
        MaybeQuantized::Original(l) => Ok(l.weight.as_ref().clone()),
        MaybeQuantized::Quantized(ql) => {
            let q = quant.expect("quantized linear requires QuantizationConfig");
            ops::dequantize(
                &*ql.inner.weight,
                &*ql.scales,
                &*ql.biases,
                q.group_size,
                q.bits,
                None::<&str>,
            )
        }
    }
}

fn make_rms_norm(weight: Array, eps: f32) -> nn::RmsNorm {
    // HF Gemma4 weights are stored as actual scale values (initialized to ones),
    // NOT as zero-centered offsets. Use them directly.
    nn::RmsNorm {
        weight: Param::new(weight),
        eps,
    }
}

fn build_rope(config: &Gemma4TextConfig, layer_type: LayerType, head_dim: i32) -> GemmaRope {
    let rope = match layer_type {
        LayerType::SlidingAttention => &config.rope_parameters.sliding_attention,
        LayerType::FullAttention => &config.rope_parameters.full_attention,
    };

    match rope.rope_type.as_str() {
        "default" => GemmaRope::Standard(
            nn::RopeBuilder::new(head_dim)
                .base(rope.rope_theta)
                .traditional(false)
                .build()
                .expect("RopeBuilder is infallible for known Gemma4 dimensions"),
        ),
        "proportional" => GemmaRope::Proportional(ProportionalRope::new(
            head_dim,
            rope.rope_theta,
            rope.partial_rotary_factor,
        )),
        other => panic!("Unsupported Gemma4 rope type: {other}"),
    }
}

/// Load model with config overrides applied before building.
/// The `overrides` closure receives the text config for mutation.
pub fn load_model_with_overrides(
    model_dir: impl AsRef<Path>,
    overrides: impl FnOnce(&mut Gemma4TextConfig),
) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    let config = get_model_args(model_dir)?;
    let mut args = config.text_config.clone();
    overrides(&mut args);
    load_model_inner(model_dir, config, args)
}

pub fn load_model(model_dir: impl AsRef<Path>) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    let config = get_model_args(model_dir)?;
    let args = config.text_config.clone();
    load_model_inner(model_dir, config, args)
}

fn load_model_inner(model_dir: &Path, config: Gemma4Config, args: Gemma4TextConfig) -> Result<Model, Error> {
    let weights = load_all_weights(model_dir)?;
    let model = build_model_from_weights(&config, args, &weights)?;

    // Match Python `mlx_lm.utils`: ask Metal to wire enough memory to hold
    // the model weights + a working budget for activations. This pages the
    // weights once at load time and keeps them resident, avoiding per-step
    // page-faults that show up as decode slowdowns under memory pressure.
    //
    // Budget: estimated bytes from the weights HashMap + 4 GB headroom for
    // KV cache + per-step activations, capped at the device's
    // max_recommended_working_set_size.
    let weight_bytes: usize = weights.values().map(|arr| arr.nbytes()).sum();
    let device = mlx_rs_core::memory::get_device_info();
    // Headroom for KV cache + activations. Tunable via env so callers
    // with long prompts / large contexts can bump it without recompiling.
    // Default 16 GB — Sweep 2 on hermes 5K showed -6% TTFT at 16 vs 4 GB
    // on Gemma4-26B-A4B; clamps to device.max_recommended_working_set_size
    // below so it never over-commits.
    let headroom_gb: usize = std::env::var("GEMMA4_WIRED_HEADROOM_GB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16);
    let headroom: usize = headroom_gb * 1024 * 1024 * 1024;
    let target = weight_bytes.saturating_add(headroom);
    let limit = target.min(device.max_recommended_working_set_size);
    let _ = mlx_rs_core::memory::set_wired_limit(limit);

    // Optional MoE expert pre-warming: force-eval every weight tensor so
    // all expert matrices are GPU-resident on first decode rather than
    // being faulted in lazily. Useful before TTFT-sensitive workloads
    // (interactive chat, agent tool calls). Skip for cold-start /
    // throughput workloads where the lazy faulting overlaps with prefill.
    if std::env::var("GEMMA4_PREWARM_EXPERTS").is_ok() {
        let refs: Vec<&Array> = weights.values().collect();
        // Chunk eval to keep the per-eval graph bounded; one giant eval
        // over 26B of weights spikes the lazy-graph builder.
        for chunk in refs.chunks(64) {
            let _ = mlx_rs::transforms::eval(chunk.iter().copied());
        }
    }

    Ok(model)
}

pub fn build_model_from_weights(
    config: &Gemma4Config,
    args: Gemma4TextConfig,
    weights: &HashMap<String, Array>,
) -> Result<Model, Error> {

    if args.use_double_wide_mlp {
        return Err(Error::Model(
            "Gemma4 double-wide MLP is not implemented yet".to_string(),
        ));
    }
    for (layer_name, rope) in [
        ("sliding_attention", &args.rope_parameters.sliding_attention),
        ("full_attention", &args.rope_parameters.full_attention),
    ] {
        if !matches!(rope.rope_type.as_str(), "default" | "proportional") {
            return Err(Error::Model(format!(
                "Unsupported Gemma4 {layer_name} rope type: {}",
                rope.rope_type
            )));
        }
    }

    let activation = GemmaActivation::from_name(&args.hidden_activation)?;

    // Infer global_head_dim from weight shapes when config has null/0.
    // Full attention layers have larger q_proj (more dims per head).
    let mut args = args;
    if args.global_head_dim == 0 {
        // Find the first full_attention layer and infer head_dim from q_proj shape.
        for (i, lt) in args.layer_types.iter().enumerate() {
            if lt == "full_attention" {
                let key = format!("model.language_model.layers.{i}.self_attn.q_proj.weight");
                if let Some(w) = weights.get(&key) {
                    let q_out = w.shape()[0]; // [n_heads * head_dim, hidden_size]
                    args.global_head_dim = q_out / args.num_attention_heads;
                    break;
                }
            }
        }
        // Fallback: use head_dim
        if args.global_head_dim == 0 {
            args.global_head_dim = args.head_dim;
        }
    }

    // Native quantized loading: if config.quantization is set, the loader
    // builds MaybeQuantized::Quantized for every (.weight, .scales, .biases)
    // triplet so inference uses native quantized_matmul. Otherwise everything
    // stays as plain nn::Linear via MaybeQuantized::Original — backwards
    // compatible with the original BF16 path.
    let quant = config.quantization.clone();

    // Compute KV sharing: which layers share KV from earlier layers
    let num_layers = args.num_hidden_layers as usize;
    let first_shared = num_layers.saturating_sub(args.num_kv_shared_layers as usize);
    let non_shared_types: Vec<&str> = args.layer_types[..first_shared]
        .iter()
        .map(|s| s.as_str())
        .collect();

    // Build cache mapping: each layer → cache slot index
    // Non-shared layers get sequential unique slots.
    // Shared layers point to the slot of the last non-shared layer of the same type.
    let mut kv_cache_map = Vec::with_capacity(num_layers);
    let mut kv_store_layers = HashSet::new();
    let mut next_slot = 0usize;
    let mut layer_to_slot: Vec<usize> = Vec::with_capacity(num_layers);

    for i in 0..num_layers {
        if i < first_shared {
            // Non-shared layer: gets its own cache slot
            layer_to_slot.push(next_slot);
            kv_cache_map.push(next_slot);
            next_slot += 1;
        } else {
            // Shared layer: find the last non-shared layer of the same type
            let layer_type_str = args.layer_types[i].as_str();
            let ref_idx = non_shared_types
                .iter()
                .rposition(|t| *t == layer_type_str)
                .ok_or_else(|| {
                    Error::Model(format!(
                        "No reference layer of type {layer_type_str} for shared layer {i}"
                    ))
                })?;
            kv_cache_map.push(layer_to_slot[ref_idx]);
            // Mark the reference layer as one that stores full-length KV
            kv_store_layers.insert(ref_idx);
        }
    }
    let num_cache_slots = next_slot;

    let mut layers = Vec::with_capacity(num_layers);
    for layer_idx_u in 0..num_layers {
        let layer_idx = layer_idx_u as i32;
        let layer_type = LayerType::from_config(&args.layer_types[layer_idx_u])?;
        let layer_prefix = format!("model.language_model.layers.{layer_idx}");
        let is_kv_shared = layer_idx_u >= first_shared && args.num_kv_shared_layers > 0;

        let head_dim = if !layer_type.is_sliding() && args.global_head_dim > 0 {
            args.global_head_dim
        } else {
            args.head_dim
        };
        // For E4B: num_global_key_value_heads is null/0, use num_key_value_heads for all
        let n_kv_heads = if !layer_type.is_sliding() && args.num_global_key_value_heads > 0 {
            args.num_global_key_value_heads
        } else {
            args.num_key_value_heads
        };

        // Q/K/V projections (separate). Fused qkv builds below once we
        // know whether they all exist + match in quant config.
        let q_proj = make_mq_linear(weights, &format!("{layer_prefix}.self_attn.q_proj"), quant.as_ref())?;
        let k_proj = if is_kv_shared {
            None
        } else {
            Some(make_mq_linear(
                weights,
                &format!("{layer_prefix}.self_attn.k_proj"),
                quant.as_ref(),
            )?)
        };
        let v_proj = if is_kv_shared {
            None
        } else {
            make_mq_linear_optional(
                weights,
                &format!("{layer_prefix}.self_attn.v_proj"),
                quant.as_ref(),
            )
        };

        // Optional fused QKV — opt in via env, skip for kv-shared layers.
        // Builds the concat once at load time and stores both the fused
        // linear and the per-part split sizes for runtime splitting.
        let (qkv_fused, fused_splits) =
            if std::env::var("GEMMA4_FUSED_QKV").is_ok() && !is_kv_shared {
                if let Some(k_ref) = k_proj.as_ref() {
                    match fuse_qkv_linears(&q_proj, k_ref, v_proj.as_ref()) {
                        Ok((fused, splits)) => (Some(fused), Some(splits)),
                        Err(_e) => (None, None),
                    }
                } else {
                    (None, None)
                }
            } else {
                (None, None)
            };

        let attention = Attention {
            layer_idx,
            layer_type,
            n_heads: args.num_attention_heads,
            n_kv_heads,
            head_dim,
            scale: 1.0,
            sliding_window: if layer_type.is_sliding() {
                Some(args.sliding_window)
            } else {
                None
            },
            is_kv_shared,
            q_proj,
            k_proj,
            v_proj,
            qkv_fused,
            fused_splits,
            o_proj: make_mq_linear(weights, &format!("{layer_prefix}.self_attn.o_proj"), quant.as_ref())?,
            q_norm: make_rms_norm(
                get_weight(&weights, &format!("{layer_prefix}.self_attn.q_norm.weight"))?,
                args.rms_norm_eps,
            ),
            k_norm: if is_kv_shared {
                None
            } else {
                Some(make_rms_norm(
                    get_weight(&weights, &format!("{layer_prefix}.self_attn.k_norm.weight"))?,
                    args.rms_norm_eps,
                ))
            },
            rope: build_rope(&args, layer_type, head_dim),
            v_norm: if is_kv_shared {
                None
            } else {
                Some(UnscaledRmsNorm::new(args.rms_norm_eps))
            },
        };

        let mlp = DenseMlp {
            gate_proj: make_mq_linear(weights, &format!("{layer_prefix}.mlp.gate_proj"), quant.as_ref())?,
            up_proj: make_mq_linear(weights, &format!("{layer_prefix}.mlp.up_proj"), quant.as_ref())?,
            down_proj: make_mq_linear(weights, &format!("{layer_prefix}.mlp.down_proj"), quant.as_ref())?,
            activation,
        };

        let (router, experts, post_ff_ln_1, post_ff_ln_2, pre_ff_ln_2) = if args.enable_moe_block {
            // Env-tunable MoE top-k. Default = config value. Many MoE
            // papers find top-4 or top-2 captures ~95% of quality at
            // 2-4x throughput. Clamp to [1, top_k_experts] so we never
            // exceed the config's max.
            let top_k_experts = std::env::var("GEMMA4_MOE_TOPK")
                .ok()
                .and_then(|v| v.parse::<i32>().ok())
                .map(|n| n.clamp(1, args.top_k_experts))
                .unwrap_or(args.top_k_experts);
            let router = Router {
                hidden_size: args.hidden_size,
                top_k_experts,
                scalar_root_size: (args.hidden_size as f32).sqrt().recip(),
                proj: make_mq_linear(weights, &format!("{layer_prefix}.router.proj"), quant.as_ref())?,
                scale: Param::new(get_weight(
                    &weights,
                    &format!("{layer_prefix}.router.scale"),
                )?),
                per_expert_scale: Param::new(get_weight(
                    &weights,
                    &format!("{layer_prefix}.router.per_expert_scale"),
                )?),
                norm: UnscaledRmsNorm::new(args.rms_norm_eps),
            };

            let experts = Experts {
                hidden_size: args.hidden_size,
                intermediate_size: args.moe_intermediate_size,
                gate_up_proj: Param::new(get_weight(
                    &weights,
                    &format!("{layer_prefix}.experts.gate_up_proj"),
                )?),
                down_proj: Param::new(get_weight(
                    &weights,
                    &format!("{layer_prefix}.experts.down_proj"),
                )?),
                activation,
            };

            (
                Some(router),
                Some(experts),
                Some(make_rms_norm(
                    get_weight(
                        &weights,
                        &format!("{layer_prefix}.post_feedforward_layernorm_1.weight"),
                    )?,
                    args.rms_norm_eps,
                )),
                Some(make_rms_norm(
                    get_weight(
                        &weights,
                        &format!("{layer_prefix}.post_feedforward_layernorm_2.weight"),
                    )?,
                    args.rms_norm_eps,
                )),
                Some(make_rms_norm(
                    get_weight(
                        &weights,
                        &format!("{layer_prefix}.pre_feedforward_layernorm_2.weight"),
                    )?,
                    args.rms_norm_eps,
                )),
            )
        } else {
            (None, None, None, None, None)
        };

        // PLE per-layer weights
        let per_layer_input_gate = if args.hidden_size_per_layer_input > 0 {
            Some(make_mq_linear(
                weights,
                &format!("{layer_prefix}.per_layer_input_gate"),
                quant.as_ref(),
            )?)
        } else {
            None
        };
        let per_layer_projection = if args.hidden_size_per_layer_input > 0 {
            Some(make_mq_linear(
                weights,
                &format!("{layer_prefix}.per_layer_projection"),
                quant.as_ref(),
            )?)
        } else {
            None
        };
        let post_per_layer_input_norm = if args.hidden_size_per_layer_input > 0 {
            Some(make_rms_norm(
                get_weight(
                    &weights,
                    &format!("{layer_prefix}.post_per_layer_input_norm.weight"),
                )?,
                args.rms_norm_eps,
            ))
        } else {
            None
        };

        layers.push(DecoderLayer {
            layer_idx,
            enable_moe_block: args.enable_moe_block,
            self_attn: attention,
            mlp,
            input_layernorm: make_rms_norm(
                get_weight(&weights, &format!("{layer_prefix}.input_layernorm.weight"))?,
                args.rms_norm_eps,
            ),
            post_attention_layernorm: make_rms_norm(
                get_weight(
                    &weights,
                    &format!("{layer_prefix}.post_attention_layernorm.weight"),
                )?,
                args.rms_norm_eps,
            ),
            pre_feedforward_layernorm: make_rms_norm(
                get_weight(
                    &weights,
                    &format!("{layer_prefix}.pre_feedforward_layernorm.weight"),
                )?,
                args.rms_norm_eps,
            ),
            post_feedforward_layernorm: make_rms_norm(
                get_weight(
                    &weights,
                    &format!("{layer_prefix}.post_feedforward_layernorm.weight"),
                )?,
                args.rms_norm_eps,
            ),
            router,
            experts,
            post_feedforward_layernorm_1: post_ff_ln_1,
            post_feedforward_layernorm_2: post_ff_ln_2,
            pre_feedforward_layernorm_2: pre_ff_ln_2,
            layer_scalar: Param::new(get_weight(
                &weights,
                &format!("{layer_prefix}.layer_scalar"),
            )?),
            per_layer_input_gate,
            per_layer_projection,
            post_per_layer_input_norm,
            activation,
        });
    }

    // PLE model-level weights
    let embed_tokens_per_layer = if args.hidden_size_per_layer_input > 0 {
        Some(make_mq_embedding(
            weights,
            "model.language_model.embed_tokens_per_layer",
            quant.as_ref(),
        )?)
    } else {
        None
    };
    let per_layer_model_projection = if args.hidden_size_per_layer_input > 0 {
        Some(make_mq_linear(
            weights,
            "model.language_model.per_layer_model_projection",
            quant.as_ref(),
        )?)
    } else {
        None
    };
    let per_layer_projection_norm = if args.hidden_size_per_layer_input > 0 {
        Some(make_rms_norm(
            get_weight(
                &weights,
                "model.language_model.per_layer_projection_norm.weight",
            )?,
            args.rms_norm_eps,
        ))
    } else {
        None
    };

    let language_model = LanguageModel {
        vocab_size: args.vocab_size,
        num_hidden_layers: args.num_hidden_layers,
        hidden_size_per_layer_input: args.hidden_size_per_layer_input,
        embed_tokens: make_mq_embedding(
            weights,
            "model.language_model.embed_tokens",
            quant.as_ref(),
        )?,
        layers,
        norm: make_rms_norm(
            get_weight(&weights, "model.language_model.norm.weight")?,
            args.rms_norm_eps,
        ),
        embed_tokens_per_layer,
        per_layer_model_projection,
        per_layer_projection_norm,
        kv_cache_map,
        kv_store_layers,
        has_moe: args.enable_moe_block,
        embed_scale: {
            // Materialize a constant `[1, 1, 1] = sqrt(hidden_size)` once.
            // Broadcasted at multiply time. Eval forces it into a concrete
            // bf16 leaf so each forward consumes a constant, not a deferred
            // graph node.
            let s = Array::from((args.hidden_size as f32).sqrt())
                .as_dtype(Dtype::Bfloat16)?;
            let _ = mlx_rs::transforms::eval([&s]);
            s
        },
    };

    let lm_head = ["lm_head", "model.lm_head", "model.language_model.lm_head"]
        .iter()
        .find(|p| get_weight_optional(weights, &format!("{p}.weight")).is_some())
        .map(|p| make_mq_linear(weights, p, quant.as_ref()))
        .transpose()?;
    if lm_head.is_none() && !args.tie_word_embeddings && !config.tie_word_embeddings {
        return Err(Error::Model(
            "Gemma4 lm_head weights are missing and embeddings are not tied".to_string(),
        ));
    }

    let model = Model {
        args,
        model: language_model,
        lm_head,
        quantization: quant,
    };
    model.eval()?;
    Ok(model)
}

// ============================================================================
// Generation
// ============================================================================

pub fn init_cache<C: KeyValueCache + Default>(num_layers: usize) -> Vec<C> {
    (0..num_layers).map(|_| C::default()).collect()
}

pub struct Gemma4VlModel {
    pub text: Model,
    pub vision: VisionModel,
    pub embed_vision: EmbedVision,
    pub image_token_id: u32,
    pub boi_token_id: u32,
    pub eoi_token_id: u32,
    pub n_vision_tokens: usize,
}

impl Gemma4VlModel {
    pub fn new_cache(&self) -> Vec<KVCache> {
        let num_slots = *self.text.model.kv_cache_map.iter().max().unwrap_or(&0) + 1;
        init_cache::<KVCache>(num_slots)
    }

    pub fn encode_image_bytes(&mut self, bytes: &[u8]) -> Result<Array, Error> {
        let (pixel_values, patch_positions, padding_mask) = preprocess_image_gemma4(bytes)?;
        let num_positions = padding_mask.len() as i32;
        let patch_positions = Array::from_slice(&patch_positions, &[1, num_positions, 2]);
        let padding_bool: Vec<bool> = padding_mask.iter().map(|&v| v != 0).collect();
        let padding_positions = Array::from_slice(&padding_bool, &[1, num_positions]);
        let hidden = self
            .vision
            .forward(&pixel_values, &patch_positions, &padding_positions)?;
        let embeds = self.embed_vision.forward(&hidden)?;
        let s1 = embeds.shape()[1];
        let s2 = embeds.shape()[2];
        embeds.reshape(&[s1, s2]).map_err(Into::into)
    }

    pub fn prefill_multimodal(
        &mut self,
        input_ids: &[i32],
        visual_features: &Array,
        cache: &mut Vec<KVCache>,
    ) -> Result<Array, Error> {
        let image_positions: Vec<usize> = input_ids
            .iter()
            .enumerate()
            .filter_map(|(idx, &id)| (id as u32 == self.image_token_id).then_some(idx))
            .collect();
        if image_positions.is_empty() {
            return self.prefill_text(input_ids, cache);
        }

        let text_embeds = self.text.embed_tokens_slice(input_ids)?;
        let scale = Array::from((self.text.args.hidden_size as f32).sqrt())
            .as_dtype(Dtype::Bfloat16)?
            .as_dtype(text_embeds.dtype())?;
        let text_embeds = text_embeds.multiply(&scale)?;
        let seq_len = input_ids.len();
        let hidden_size = self.text.args.hidden_size as usize;
        let text_f32 = text_embeds.as_dtype(Dtype::Float32)?;
        eval([&text_f32]).map_err(|e| Error::Model(format!("eval text_embeds: {e}")))?;
        let text_slice = text_f32.try_as_slice::<f32>().map_err(|e| {
            Error::Model(format!("text embeddings must be contiguous for multimodal scatter: {e}"))
        })?;

        let vis_len = visual_features.shape()[0] as usize;
        if vis_len == 0 {
            return Err(Error::Model("encode_image_bytes returned zero visual tokens".into()));
        }
        let vis_f32 = visual_features.as_dtype(Dtype::Float32)?;
        eval([&vis_f32]).map_err(|e| Error::Model(format!("eval visual_features: {e}")))?;
        let vis_slice = vis_f32.try_as_slice::<f32>().map_err(|e| {
            Error::Model(format!("visual features must be contiguous for multimodal scatter: {e}"))
        })?;

        let mut combined = text_slice.to_vec();
        for (out_idx, &token_idx) in image_positions.iter().enumerate() {
            let src_row = out_idx % vis_len;
            let dst_start = token_idx * hidden_size;
            let src_start = src_row * hidden_size;
            combined[dst_start..dst_start + hidden_size]
                .copy_from_slice(&vis_slice[src_start..src_start + hidden_size]);
        }

        let combined_f32 = Array::from_slice(
            &combined,
            &[1, seq_len as i32, self.text.args.hidden_size],
        );
        eval([&combined_f32]).map_err(|e| Error::Model(format!("eval combined_f32: {e}")))?;
        let combined = combined_f32.as_dtype(text_embeds.dtype())?;
        eval([&combined]).map_err(|e| Error::Model(format!("eval combined: {e}")))?;

        // For PLE (per-layer embedding) models like E4B, compute the full-sequence PLE
        // tensor [1, seq_len, num_layers, ple_dim] before chunking.  Image positions
        // use image_token_id as the auxiliary token — acceptable because PLE is a
        // secondary modulating input, not the primary feature stream.
        let input_ids_arr = Array::from_slice(input_ids, &[1, input_ids.len() as i32]);
        let per_layer_inputs_full = self
            .text
            .model
            .compute_per_layer_inputs(&input_ids_arr, &combined)
            .map_err(|e| Error::Model(format!("compute_per_layer_inputs: {e}")))?;

        // Process in fixed-size chunks to bound peak GPU memory during MoE dispatch.
        // Single-shot prefill with seq_len=314 would create a ~18 GB intermediate
        // tensor in forward_topk (take_axis on 128 experts × 8 top-k × 314 tokens).
        // Chunked at PREFILL_CHUNK=32 keeps peak intermediate at ~2 GB, matching
        // the text-only path.
        const PREFILL_CHUNK: i32 = 32;
        let total = seq_len as i32;
        let mut pos: i32 = 0;
        while pos < total {
            let end = (pos + PREFILL_CHUNK).min(total);
            let chunk = combined.index((.., pos..end, ..));
            // Slice the PLE tensor to match this chunk's sequence positions.
            let ple_chunk = per_layer_inputs_full
                .as_ref()
                .map(|p| p.index((.., pos..end, .., ..)));
            if end < total {
                // Intermediate chunk: update KV cache, discard hidden states.
                let hidden = self
                    .text
                    .model
                    .forward_from_embeds(&chunk, None, cache, ple_chunk.as_ref())?;
                eval([&hidden])
                    .map_err(|e| Error::Model(format!("prefill chunk {pos}: {e}")))?;
            } else {
                // Final chunk: apply lm_head and return logits.
                return self
                    .text
                    .forward_from_embeds(&chunk, cache, ple_chunk.as_ref())
                    .map_err(Into::into);
            }
            pos = end;
        }
        Err(Error::Model("prefill_multimodal: unexpected empty sequence".into()))
    }

    pub fn prefill_text(
        &mut self,
        input_ids: &[i32],
        cache: &mut Vec<KVCache>,
    ) -> Result<Array, Error> {
        let input = Array::from_slice(input_ids, &[1, input_ids.len() as i32]);
        self.text
            .forward_last_logits(ModelInput {
                inputs: &input,
                mask: None,
                cache,
            })
            .map_err(Into::into)
    }

    pub fn decode_token(
        &mut self,
        token_id: u32,
        cache: &mut Vec<KVCache>,
    ) -> Result<Array, Error> {
        let input = Array::from_slice(&[token_id as i32], &[1, 1]);
        self.text
            .forward_last_logits(ModelInput {
                inputs: &input,
                mask: None,
                cache,
            })
            .map_err(Into::into)
    }
}

pub fn load_vl_model(model_dir: impl AsRef<Path>) -> Result<Gemma4VlModel, Error> {
    let model_dir = model_dir.as_ref();
    let config = get_model_args(model_dir)?;
    let vision_config = config
        .vision_config
        .clone()
        .ok_or_else(|| Error::Model("Not a Gemma4-VL model: missing vision_config".to_string()))?;
    let weights = load_all_weights_unfiltered(model_dir)?;

    let text = build_model_from_weights(&config, config.text_config.clone(), &weights)?;
    let vision = load_vision_model(&weights, &vision_config)?;
    let embed_vision = load_embed_vision(&weights, vision_config.rms_norm_eps)?;

    Ok(Gemma4VlModel {
        text,
        vision,
        embed_vision,
        image_token_id: config.image_token_id.unwrap_or(258880),
        boi_token_id: config.boi_token_id.unwrap_or(255999),
        eoi_token_id: config.eoi_token_id.unwrap_or(258882),
        n_vision_tokens: config.vision_soft_tokens_per_image.max(vision_config.default_output_length as usize),
    })
}

pub struct Generate<'a, C, S: Sampler = DefaultSampler> {
    model: &'a mut Model,
    cache: &'a mut Vec<C>,
    sampler: S,
    temp: f32,
    state: GenerateState<'a>,
    token_count: usize,
}

pub enum GenerateState<'a> {
    Prefill { prompt_token: &'a Array },
    Pipelined { current_y: Array },
    Done,
}

macro_rules! tri {
    ($expr:expr) => {
        match $expr {
            Ok(val) => val,
            Err(e) => return Some(Err(e.into())),
        }
    };
}

impl<'a, C> Generate<'a, C, DefaultSampler>
where
    C: KeyValueCache + Default,
{
    pub fn new(
        model: &'a mut Model,
        cache: &'a mut Vec<C>,
        temp: f32,
        prompt_token: &'a Array,
    ) -> Self {
        Self::with_sampler(model, cache, temp, prompt_token, DefaultSampler)
    }
}

impl<'a, C, S: Sampler> Generate<'a, C, S>
where
    C: KeyValueCache + Default,
{
    pub fn with_sampler(
        model: &'a mut Model,
        cache: &'a mut Vec<C>,
        temp: f32,
        prompt_token: &'a Array,
        sampler: S,
    ) -> Self {
        if cache.is_empty() {
            // Number of cache slots may be less than num_layers due to KV sharing
            let num_slots = *model.model.kv_cache_map.iter().max().unwrap_or(&0) + 1;
            *cache = init_cache(num_slots);
        }

        Self {
            model,
            cache,
            sampler,
            temp,
            state: GenerateState::Prefill { prompt_token },
            token_count: 0,
        }
    }

    fn compute_next(&mut self, y: &Array) -> Result<Array, Exception> {
        let inputs = y.index((.., NewAxis));
        let input = ModelInput {
            inputs: &inputs,
            mask: None,
            cache: self.cache,
        };
        let logits = self.model.forward(input)?;
        // Select last token before sampling to keep output shape [B], not [B, 1].
        self.sampler.sample(&logits.index((.., -1, ..)), self.temp)
    }
}

impl<'a, C, S: Sampler> Iterator for Generate<'a, C, S>
where
    C: KeyValueCache + Default,
{
    type Item = Result<Array, Exception>;

    fn next(&mut self) -> Option<Self::Item> {
        let state = std::mem::replace(&mut self.state, GenerateState::Done);

        match state {
            GenerateState::Prefill { prompt_token } => {
                // Chunked prefill: process the prompt in chunks to limit peak
                // GPU memory from MoE expert gather intermediates. Each chunk
                // updates the KV cache, so attention can look back at all
                // previously processed tokens.
                //
                // Chunk size is env-tunable via `GEMMA4_PREFILL_CHUNK`. The
                // historical default of 32 forces ~162 GPU dispatches on a
                // 5K prompt and is dominated by launch overhead, not by
                // useful compute. Modern Apple GPUs handle far larger
                // chunks comfortably; default raised to 512.
                // Bench (hermes 5183 tok, Gemma4-26B-A4B Q4): chunk=32 →
                // 720s TTFT, chunk=64 → 604s (-16%), chunk=128 → 1905s
                // (MoE expert-gather memory pressure / thrashing, peak
                // GPU 73→91 GB). Sweet spot is 64.
                let prefill_chunk: i32 = std::env::var("GEMMA4_PREFILL_CHUNK")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(64);
                let PREFILL_CHUNK = prefill_chunk;
                let seq_len = prompt_token.shape()[1];

                if seq_len > PREFILL_CHUNK {
                    let mut pos = 0;
                    while pos < seq_len {
                        let end = (pos + PREFILL_CHUNK).min(seq_len);
                        let chunk = prompt_token.index((.., pos..end));
                        let input = ModelInput {
                            inputs: &chunk,
                            mask: None,
                            cache: self.cache,
                        };
                        // forward_last_logits returns `[B, vocab]`; intermediate
                        // chunks discard the logits, the last chunk samples.
                        let logits = tri!(self.model.forward_last_logits(input));
                        // Per-chunk eval frees expert-gather intermediates
                        // but adds a GPU sync per chunk. On machines with
                        // enough wired memory the lazy graph can stretch
                        // safely across all chunks — opt in to async_eval
                        // via GEMMA4_ASYNC_PREFILL=1, or skip the eval
                        // entirely via GEMMA4_SKIP_CHUNK_EVAL=1.
                        if std::env::var("GEMMA4_SKIP_CHUNK_EVAL").is_err() {
                            if std::env::var("GEMMA4_ASYNC_PREFILL").is_ok() {
                                tri!(mlx_rs::transforms::async_eval([&logits]));
                            } else {
                                tri!(mlx_rs::transforms::eval([&logits]));
                            }
                        }
                        pos = end;

                        // On last chunk, sample from the final logits
                        if pos >= seq_len {
                            let y = tri!(self.sampler.sample(&logits, self.temp));
                            tri!(mlx_rs::transforms::eval([&y]));

                            let next_y = tri!(self.compute_next(&y));
                            tri!(mlx_rs::transforms::async_eval([&next_y]));

                            self.state = GenerateState::Pipelined { current_y: next_y };
                            return Some(Ok(y));
                        }
                    }
                    unreachable!()
                } else {
                    let input = ModelInput {
                        inputs: prompt_token,
                        mask: None,
                        cache: self.cache,
                    };
                    let logits = tri!(self.model.forward_last_logits(input));
                    let y = tri!(self.sampler.sample(&logits, self.temp));

                    tri!(mlx_rs::transforms::async_eval([&y]));
                    tri!(mlx_rs::transforms::eval([&y]));

                    let next_y = tri!(self.compute_next(&y));
                    tri!(mlx_rs::transforms::async_eval([&next_y]));

                    self.state = GenerateState::Pipelined { current_y: next_y };
                    Some(Ok(y))
                }
            }
            GenerateState::Pipelined { current_y } => {
                // Optional per-step profiling. GEMMA4_PROFILE_DECODE=1 reports
                // (compute_next_ms, cache_eval_ms) every 16 steps via stderr.
                // Adds two cheap clocks; no-op when env unset.
                let profile = std::env::var("GEMMA4_PROFILE_DECODE").is_ok();
                let t0 = profile.then(std::time::Instant::now);
                let next_y = tri!(self.compute_next(&current_y));
                tri!(mlx_rs::transforms::async_eval([&next_y]));
                let t1 = profile.then(std::time::Instant::now);

                // Per-layer cache eval materializes lazy index chains from
                // update_and_fetch so the next forward pass starts clean.
                for c in self.cache.iter() {
                    tri!(c.eval());
                }
                if profile {
                    let t1 = t1.unwrap();
                    let t0 = t0.unwrap();
                    let t2 = std::time::Instant::now();
                    let next_ms = t1.duration_since(t0).as_secs_f32() * 1000.0;
                    let cache_ms = t2.duration_since(t1).as_secs_f32() * 1000.0;
                    if self.token_count % 16 == 0 {
                        eprintln!(
                            "[gemma4-profile] tok={} next={:.2}ms cache_eval={:.2}ms",
                            self.token_count, next_ms, cache_ms
                        );
                    }
                }

                // Periodically release completed computation graph memory.
                // Default 256-token cadence is a compromise — too frequent
                // = global stalls, too rare = peak memory grows. Env-tune
                // via GEMMA4_CACHE_CLEAR_INTERVAL (0 disables entirely).
                self.token_count += 1;
                let cache_clear_interval: usize = std::env::var("GEMMA4_CACHE_CLEAR_INTERVAL")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(256);
                if cache_clear_interval > 0 && self.token_count % cache_clear_interval == 0 {
                    unsafe {
                        mlx_sys::mlx_clear_cache();
                    }
                }

                self.state = GenerateState::Pipelined { current_y: next_y };
                Some(Ok(current_y))
            }
            GenerateState::Done => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_text_config() -> Gemma4TextConfig {
        Gemma4TextConfig {
            attention_bias: false,
            attention_dropout: 0.0,
            attention_k_eq_v: true,
            enable_moe_block: true,
            global_head_dim: 16,
            head_dim: 4,
            hidden_activation: "gelu_pytorch_tanh".to_string(),
            hidden_size: 16,
            hidden_size_per_layer_input: 0,
            intermediate_size: 32,
            layer_types: vec!["full_attention".to_string()],
            max_position_embeddings: 128,
            moe_intermediate_size: 8,
            num_attention_heads: 2,
            num_experts: 4,
            num_global_key_value_heads: 1,
            num_hidden_layers: 1,
            num_kv_shared_layers: 0,
            num_key_value_heads: 1,
            rms_norm_eps: 1e-6,
            rope_parameters: RopeParameters {
                sliding_attention: RopeSpec {
                    rope_theta: 10_000.0,
                    rope_type: "default".to_string(),
                    partial_rotary_factor: 1.0,
                },
                full_attention: RopeSpec {
                    rope_theta: 1_000_000.0,
                    rope_type: "proportional".to_string(),
                    partial_rotary_factor: 0.25,
                },
            },
            sliding_window: 32,
            tie_word_embeddings: true,
            top_k_experts: 2,
            use_double_wide_mlp: false,
            vocab_size: 256,
            vocab_size_per_layer_input: 0,
            final_logit_softcapping: Some(30.0),
        }
    }

    fn make_test_input(seq_len: i32, head_dim: i32) -> Array {
        let mut values = vec![0.0_f32; seq_len as usize * head_dim as usize];
        let row_start = ((seq_len - 1) * head_dim) as usize;
        for i in 0..head_dim as usize {
            values[row_start + i] = (i + 1) as f32;
        }
        Array::from_slice(&values, &[1, seq_len, head_dim])
    }

    fn make_blnh_test_input(seq_len: i32, n_heads: i32, head_dim: i32) -> Array {
        let mut values = Vec::with_capacity((seq_len * n_heads * head_dim) as usize);
        for pos in 0..seq_len {
            for head in 0..n_heads {
                for dim in 0..head_dim {
                    values.push(((pos + 1) * 100 + (head + 1) * 10 + dim + 1) as f32);
                }
            }
        }
        Array::from_slice(&values, &[1, seq_len, n_heads, head_dim])
    }

    fn rope_angles(head_dim: i32, partial_rotary_factor: f32) -> i32 {
        let rope_angles = (partial_rotary_factor * head_dim as f32 / 2.0).floor() as i32;
        rope_angles.clamp(0, head_dim / 2)
    }

    fn hf_reference_proportional_rope_blnh(
        input: &Array,
        head_dim: i32,
        theta: f32,
        partial_rotary_factor: f32,
        offset: i32,
    ) -> Array {
        let shape = input.shape();
        let batch = shape[0] as usize;
        let seq_len = shape[1] as usize;
        let n_heads = shape[2] as usize;
        let head_dim_usize = head_dim as usize;
        let half_dim = head_dim_usize / 2;
        let rope_angles = rope_angles(head_dim, partial_rotary_factor) as usize;

        let mut inv_freq = vec![0.0_f32; half_dim];
        for (i, freq) in inv_freq.iter_mut().enumerate().take(rope_angles) {
            *freq = 1.0 / theta.powf((2 * i) as f32 / head_dim as f32);
        }

        let input = input.as_slice::<f32>();
        let mut output = vec![0.0_f32; input.len()];
        for b in 0..batch {
            for pos in 0..seq_len {
                let position = (offset as usize + pos) as f32;
                for head in 0..n_heads {
                    let base = ((b * seq_len + pos) * n_heads + head) * head_dim_usize;
                    for lane in 0..half_dim {
                        let angle = position * inv_freq[lane];
                        let cos = angle.cos();
                        let sin = angle.sin();
                        let x1 = input[base + lane];
                        let x2 = input[base + half_dim + lane];
                        output[base + lane] = x1 * cos - x2 * sin;
                        output[base + half_dim + lane] = x2 * cos + x1 * sin;
                    }
                }
            }
        }

        Array::from_slice(&output, shape)
    }

    fn max_abs_diff(lhs: &Array, rhs: &Array) -> f32 {
        lhs.subtract(rhs)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item::<f32>()
    }

    fn noop_lane_diff(
        input: &Array,
        result: &Array,
        token_idx: i32,
        head_dim: i32,
        partial_rotary_factor: f32,
    ) -> f32 {
        let half_dim = head_dim / 2;
        let rope_angles = rope_angles(head_dim, partial_rotary_factor);

        let input_first = input.index((0, token_idx, rope_angles..half_dim));
        let result_first = result.index((0, token_idx, rope_angles..half_dim));
        let input_second = input.index((0, token_idx, (half_dim + rope_angles)..head_dim));
        let result_second = result.index((0, token_idx, (half_dim + rope_angles)..head_dim));

        let first_diff = result_first
            .subtract(&input_first)
            .unwrap()
            .abs()
            .unwrap()
            .sum(None)
            .unwrap()
            .item::<f32>();
        let second_diff = result_second
            .subtract(&input_second)
            .unwrap()
            .abs()
            .unwrap()
            .sum(None)
            .unwrap()
            .item::<f32>();

        first_diff + second_diff
    }

    #[test]
    fn full_attention_prefill_should_preserve_nope_lanes() {
        let config = test_text_config();
        let mut rope = build_rope(&config, LayerType::FullAttention, config.global_head_dim);
        let input = make_test_input(2, config.global_head_dim);
        let result = rope.apply(&input, 0).unwrap();

        let diff = noop_lane_diff(
            &input,
            &result,
            1,
            config.global_head_dim,
            config.rope_parameters.full_attention.partial_rotary_factor,
        );

        assert!(
            diff < 1e-6,
            "expected proportional RoPE to leave Gemma4 nope lanes unchanged during prefill, diff={diff}"
        );
    }

    #[test]
    fn full_attention_decode_should_preserve_nope_lanes_with_offset() {
        let config = test_text_config();
        let mut rope = build_rope(&config, LayerType::FullAttention, config.global_head_dim);
        let input = make_test_input(1, config.global_head_dim);
        let result = rope.apply(&input, 7).unwrap();

        let diff = noop_lane_diff(
            &input,
            &result,
            0,
            config.global_head_dim,
            config.rope_parameters.full_attention.partial_rotary_factor,
        );

        assert!(
            diff < 1e-6,
            "expected proportional RoPE to leave Gemma4 nope lanes unchanged during decode, diff={diff}"
        );
    }

    #[test]
    fn full_attention_prefill_should_match_hf_reference_on_blnh_layout() {
        const HF_ROPE_TOLERANCE: f32 = 5e-5;

        let config = test_text_config();
        let mut rope = build_rope(&config, LayerType::FullAttention, config.global_head_dim);
        let input = make_blnh_test_input(3, config.num_attention_heads, config.global_head_dim);
        let expected = hf_reference_proportional_rope_blnh(
            &input,
            config.global_head_dim,
            config.rope_parameters.full_attention.rope_theta,
            config.rope_parameters.full_attention.partial_rotary_factor,
            0,
        );
        let result = rope
            .apply_with_layout(&input, 0, RotaryLayout::BatchSeqHeadsDim)
            .unwrap();

        let diff = max_abs_diff(&result, &expected);
        assert!(
            diff < HF_ROPE_TOLERANCE,
            "expected BLNH proportional RoPE to match HF reference during prefill, diff={diff}"
        );
    }

    #[test]
    fn full_attention_decode_should_match_hf_reference_on_blnh_layout() {
        const HF_ROPE_TOLERANCE: f32 = 5e-5;

        let config = test_text_config();
        let mut rope = build_rope(&config, LayerType::FullAttention, config.global_head_dim);
        let input = make_blnh_test_input(1, config.num_attention_heads, config.global_head_dim);
        let expected = hf_reference_proportional_rope_blnh(
            &input,
            config.global_head_dim,
            config.rope_parameters.full_attention.rope_theta,
            config.rope_parameters.full_attention.partial_rotary_factor,
            11,
        );
        let result = rope
            .apply_with_layout(&input, 11, RotaryLayout::BatchSeqHeadsDim)
            .unwrap();

        let diff = max_abs_diff(&result, &expected);
        assert!(
            diff < HF_ROPE_TOLERANCE,
            "expected BLNH proportional RoPE to match HF reference during decode, diff={diff}"
        );
    }
}
