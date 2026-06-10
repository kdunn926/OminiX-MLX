//! `glm_ocr_vision` encoder — Glm46V architecture.
//!
//! Reference: HF `Glm4vVisionModel` (parent of `Glm46VVisionModel`).
//! Pipeline:
//!
//! ```text
//!   patches[(grid_t * gh * gw), t*P*P*3]
//!     → patch_embed   : Linear(t*P*P*3 → hidden=1024)    [Conv3d as Linear,
//!                                                          MLX channels-last]
//!     → 24 blocks     : pre-RMSNorm
//!                       → fused QKV (qkv.weight (3*hidden, hidden)) +
//!                         per-head q_norm / k_norm RMSNorms on head_dim +
//!                         2-D rotary on Q/K + bidirectional SDPA + proj
//!                         (with bias)
//!                       pre-RMSNorm
//!                       → SwiGLU MLP (gate/up/down all with bias),
//!                         intermediate=4096
//!     → post_layernorm: RMSNorm(hidden=1024)
//!     → spatial 2×2 reshape (gh/m, gw/m, m, m, hidden) → (n_merged, m, m, hidden)
//!     → downsample    : Conv2d(in=1024, out=1536, kernel=2, stride=2),
//!                        applied as Linear over flattened (m*m*hidden) inputs
//!     → merger        : Linear(out_hidden → out_hidden)
//!                       → GELU(LayerNorm(.))
//!                       → SwiGLU(out_hidden → intermediate=4608 → out_hidden)
//!     → soft_tokens[(gh/m * gw/m), out_hidden=1536]
//! ```
//!
//! The 2-D rotary uses `head_dim/2` rotation channels: half are driven
//! by patch-row index (h_coord), half by patch-col index (w_coord), and
//! the result is concatenated with itself so `rotate_half`'s
//! `dim ↔ dim + d/2` pairing stays within an axis.

use mlx_rs::{
    module::Module,
    nn,
    ops::{self, indexing::IndexOp},
    Array, Dtype,
};
use mlx_rs_core::utils::{scaled_dot_product_attention, SdpaMask};

use crate::config::GlmOcrVisionConfig;
use crate::error::Error;

pub type RmsNorm = nn::RmsNorm;
pub type LayerNorm = nn::LayerNorm;

/// 2-D rotary embedding for the vision encoder. Builds `(n_patches,
/// head_dim)` cos/sin tables from a flat list of patch coords.
pub struct VisionRotary {
    pub head_dim: i32,
    /// `(head_dim / 4,)` — half the channels rotate by `h`, half by `w`.
    pub inv_freq: Array,
}

impl VisionRotary {
    pub fn new(head_dim: i32, rope_theta: f32) -> Self {
        let quarter = (head_dim / 4) as usize;
        let mut data = Vec::with_capacity(quarter);
        for i in 0..quarter {
            let exp = (2 * i) as f32 / (head_dim / 2) as f32;
            data.push(1.0_f32 / rope_theta.powf(exp));
        }
        Self {
            head_dim,
            inv_freq: Array::from_slice(&data, &[quarter as i32]),
        }
    }

    pub fn cos_sin(&self, h_coords: &[i32], w_coords: &[i32]) -> Result<(Array, Array), Error> {
        if h_coords.len() != w_coords.len() {
            return Err(Error::Vision("rotary: h/w coord length mismatch".to_string()));
        }
        let n = h_coords.len() as i32;
        let h = Array::from_slice(h_coords, &[n]);
        let w = Array::from_slice(w_coords, &[n]);
        let inv = self.inv_freq.as_dtype(Dtype::Float32)?.reshape(&[1, -1])?;
        let hf = h.as_dtype(Dtype::Float32)?.reshape(&[-1, 1])?;
        let wf = w.as_dtype(Dtype::Float32)?.reshape(&[-1, 1])?;
        let freq_h = hf.matmul(&inv)?;
        let freq_w = wf.matmul(&inv)?;
        // freqs (n, head_dim/2) → emb (n, head_dim) for rotate_half pairing.
        let freqs = ops::concatenate_axis(&[freq_h, freq_w], -1)?;
        let emb = ops::concatenate_axis(&[freqs.clone(), freqs], -1)?;
        Ok((emb.cos()?, emb.sin()?))
    }
}

fn rotate_half(x: &Array) -> Result<Array, Error> {
    let last = x.shape()[x.ndim() - 1];
    let half = last / 2;
    let x1 = x.index((.., .., ..half));
    let x2 = x.index((.., .., half..));
    let neg_x2 = x2.negative()?;
    Ok(ops::concatenate_axis(&[neg_x2, x1], -1)?)
}

fn apply_rotary(q: &Array, k: &Array, cos: &Array, sin: &Array) -> Result<(Array, Array), Error> {
    // q, k: (n, num_heads, head_dim); cos, sin: (n, head_dim) → unsqueeze for head axis.
    let cos = cos.expand_dims(1)?;
    let sin = sin.expand_dims(1)?;
    let q_rot = rotate_half(q)?;
    let k_rot = rotate_half(k)?;
    let q_out = q.multiply(&cos)?.add(&q_rot.multiply(&sin)?)?;
    let k_out = k.multiply(&cos)?.add(&k_rot.multiply(&sin)?)?;
    Ok((q_out, k_out))
}

/// Vision attention with fused QKV + per-head Q/K RMSNorms.
pub struct VisionAttention {
    pub num_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub qkv: nn::Linear,
    /// Per-head RMSNorm on Q. Weight shape `(head_dim,)`.
    pub q_norm: RmsNorm,
    /// Per-head RMSNorm on K. Weight shape `(head_dim,)`.
    pub k_norm: RmsNorm,
    pub proj: nn::Linear,
}

impl VisionAttention {
    /// `x`: `(n_patches, hidden)`. Returns `(n_patches, hidden)`.
    pub fn forward(&mut self, x: &Array, cos: &Array, sin: &Array) -> Result<Array, Error> {
        let s = x.shape();
        if s.len() != 2 {
            return Err(Error::Vision(format!(
                "VisionAttention: expected (n, hidden); got {:?}",
                s
            )));
        }
        let n = s[0];
        let hidden = s[1];
        let qkv = self.qkv.forward(x)?;
        // qkv shape (n, 3 * hidden) → (n, 3, num_heads, head_dim) → split along dim 1
        let qkv = qkv.reshape(&[n, 3, self.num_heads, self.head_dim])?;
        let q = qkv.index((.., 0, .., ..)); // (n, num_heads, head_dim)
        let k = qkv.index((.., 1, .., ..));
        let v = qkv.index((.., 2, .., ..));
        // Per-head RMSNorm on Q and K before rotary. The norm weight is
        // (head_dim,) so it broadcasts across the n and heads axes.
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;
        let (q, k) = apply_rotary(&q, &k, cos, sin)?;
        // SDPA needs (B, H, N, D); add batch dim then swap N/H.
        let to_bhnd = |a: Array| -> Result<Array, Error> {
            Ok(a.expand_dims(0)?.transpose_axes(&[0, 2, 1, 3])?)
        };
        let q = to_bhnd(q)?;
        let k = to_bhnd(k)?;
        let v = to_bhnd(v)?;
        let attn = scaled_dot_product_attention::<mlx_rs_core::cache::KVCache>(
            q, k, v, None, self.scale, None::<SdpaMask>,
        )?;
        let attn = attn.transpose_axes(&[0, 2, 1, 3])?.reshape(&[n, hidden])?;
        Ok(self.proj.forward(&attn)?)
    }
}

/// SwiGLU MLP with optional biases (the encoder blocks have biases; the
/// merger doesn't).
pub struct VisionMlp {
    pub gate_proj: nn::Linear,
    pub up_proj: nn::Linear,
    pub down_proj: nn::Linear,
}

impl VisionMlp {
    pub fn forward(&mut self, x: &Array) -> Result<Array, Error> {
        let gate = self.gate_proj.forward(x)?;
        let act = nn::silu(&gate)?;
        let up = self.up_proj.forward(x)?;
        Ok(self.down_proj.forward(&act.multiply(&up)?)?)
    }
}

pub struct VisionBlock {
    pub norm1: RmsNorm,
    pub attn: VisionAttention,
    pub norm2: RmsNorm,
    pub mlp: VisionMlp,
}

impl VisionBlock {
    pub fn forward(&mut self, x: &Array, cos: &Array, sin: &Array) -> Result<Array, Error> {
        let normed = self.norm1.forward(x)?;
        let attn = self.attn.forward(&normed, cos, sin)?;
        let h = x.add(&attn)?;
        let normed = self.norm2.forward(&h)?;
        let mlp = self.mlp.forward(&normed)?;
        Ok(h.add(&mlp)?)
    }
}

/// Glm46V merger:
///   proj(x)
///     → GELU(post_projection_norm(.))
///     → SwiGLU(gate_proj, up_proj, down_proj)
///
/// `post_projection_norm` is a **LayerNorm** (weight + bias),
/// distinguishing it from the RMSNorms elsewhere.
pub struct VisionMerger {
    pub proj: nn::Linear,
    pub post_projection_norm: LayerNorm,
    pub gate_proj: nn::Linear,
    pub up_proj: nn::Linear,
    pub down_proj: nn::Linear,
}

impl VisionMerger {
    pub fn forward(&mut self, x: &Array) -> Result<Array, Error> {
        let h = self.proj.forward(x)?;
        let h = self.post_projection_norm.forward(&h)?;
        let h = nn::gelu(&h)?;
        let g = self.gate_proj.forward(&h)?;
        let g = nn::silu(&g)?;
        let u = self.up_proj.forward(&h)?;
        Ok(self.down_proj.forward(&g.multiply(&u)?)?)
    }
}

/// Spatial-2×2 downsample as a Conv2d with stride=kernel=2. The
/// safetensors weight ships as `(out_ch, kH, kW, in_ch)` (MLX
/// channels-last); we evaluate it as a Linear over the merged
/// `(m, m, in_ch)` patches → `(out_ch,)`.
pub struct Downsample {
    /// Reshaped to `(out_hidden, m * m * in_hidden)` at load time.
    pub linear: nn::Linear,
    pub merge_size: i32,
    pub in_hidden: i32,
    pub out_hidden: i32,
}

impl Downsample {
    /// `x`: `(n_patches, in_hidden)` with the implicit `(gh, gw)` layout.
    /// Returns `(n_patches / m², out_hidden)`.
    pub fn forward(&mut self, x: &Array, grid_h: i32, grid_w: i32) -> Result<Array, Error> {
        let m = self.merge_size;
        if grid_h % m != 0 || grid_w % m != 0 {
            return Err(Error::Vision(format!(
                "Downsample: grid ({grid_h}, {grid_w}) not divisible by merge_size {m}"
            )));
        }
        // (gh, gw, in_h) → (gh/m, m, gw/m, m, in_h) → (gh/m, gw/m, m, m, in_h)
        //   → (n_merged, m*m*in_h)
        let merged = x
            .reshape(&[grid_h, grid_w, self.in_hidden])?
            .reshape(&[grid_h / m, m, grid_w / m, m, self.in_hidden])?
            .transpose_axes(&[0, 2, 1, 3, 4])?
            .reshape(&[(grid_h / m) * (grid_w / m), m * m * self.in_hidden])?;
        Ok(self.linear.forward(&merged)?)
    }
}

pub struct VisionEncoder {
    pub config: GlmOcrVisionConfig,
    pub patch_embed: nn::Linear,
    pub rotary: VisionRotary,
    pub blocks: Vec<VisionBlock>,
    pub post_layernorm: RmsNorm,
    pub downsample: Downsample,
    pub merger: VisionMerger,
}

impl VisionEncoder {
    /// Backward-compat stub. Real loader lives in `crate::loader`.
    pub fn load_from_weights(
        _config: GlmOcrVisionConfig,
        _weights: &std::collections::HashMap<String, Array>,
    ) -> Result<Self, Error> {
        Err(Error::Vision(
            "VisionEncoder::load_from_weights deprecated; use loader::load_from_path".into(),
        ))
    }

    /// `patches`: `(n_patches, t*P*P*3)`, `grid`: `(T_raw, gh, gw)`.
    /// Returns `(n_soft, out_hidden)`.
    pub fn forward(&mut self, patches: &Array, grid: (i32, i32, i32)) -> Result<Array, Error> {
        let (_t_raw, grid_h, grid_w) = grid;
        let mut h = self.patch_embed.forward(patches)?;
        // 2-D rotary positions: flat list of (gh, gw) per patch row.
        let mut h_coords = Vec::with_capacity((grid_h * grid_w) as usize);
        let mut w_coords = Vec::with_capacity((grid_h * grid_w) as usize);
        for gh in 0..grid_h {
            for gw in 0..grid_w {
                h_coords.push(gh);
                w_coords.push(gw);
            }
        }
        let (cos, sin) = self.rotary.cos_sin(&h_coords, &w_coords)?;
        for block in self.blocks.iter_mut() {
            h = block.forward(&h, &cos, &sin)?;
        }
        let h = self.post_layernorm.forward(&h)?;
        let h = self.downsample.forward(&h, grid_h, grid_w)?;
        self.merger.forward(&h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotate_half_swaps_halves_with_negation() {
        // (n=1, h=1, d=4) → rotate_half along last axis
        let x = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let r = rotate_half(&x).unwrap();
        mlx_rs::transforms::eval([&r]).unwrap();
        let v = r.try_as_slice::<f32>().unwrap();
        assert_eq!(v, &[-3.0, -4.0, 1.0, 2.0]);
    }

    #[test]
    fn rotary_cos_sin_shape() {
        let rot = VisionRotary::new(64, 10_000.0);
        let h = vec![0_i32, 0, 1, 1];
        let w = vec![0_i32, 1, 0, 1];
        let (cos, sin) = rot.cos_sin(&h, &w).unwrap();
        assert_eq!(cos.shape(), &[4, 64]);
        assert_eq!(sin.shape(), &[4, 64]);
    }
}
