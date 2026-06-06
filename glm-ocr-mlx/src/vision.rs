//! `glm_ocr_vision` encoder.
//!
//! 24-layer ViT mirroring the canonical Qwen2-VL-style vision tower the
//! Glm46V family uses:
//!
//! ```text
//!   PatchEmbed  : Linear(3 * temporal_patch_size * P * P → hidden=1024)
//!                 (consumes the Conv3d-style patchified rows produced by
//!                  `crate::preprocessor::preprocess_image_bytes`)
//!   Block × 24  : RmsNorm
//!                 → GQA attention with 2-D rotary applied to Q/K
//!                   (16 heads, head_dim=64, attention_bias=true), no mask
//!                 RmsNorm
//!                 → SwiGLU MLP (gate_proj, up_proj, down_proj),
//!                   intermediate=4096
//!   Merger      : RmsNorm
//!                 → spatial 2×2 merge (n_patches → n_patches/4, hidden*4)
//!                 → Linear(hidden*4 → hidden*4) → GELU
//!                 → Linear(hidden*4 → out_hidden=1536)
//! ```
//!
//! Forward contract:
//!   `patches[(grid_t * gh * gw), 3*t*P*P]` + `(grid_t, gh, gw)`
//!     → `soft_tokens[(grid_t * gh/m * gw/m), out_hidden]`
//!
//! Weight names will be wired by the safetensors loader in phase 4;
//! this file defines only the modules + their forward semantics.

use mlx_rs::{
    module::Module,
    nn,
    ops::{self, indexing::IndexOp},
    Array, Dtype,
};
use mlx_rs_core::utils::{scaled_dot_product_attention, SdpaMask};

use crate::config::GlmOcrVisionConfig;
use crate::error::Error;

pub type VisionRmsNorm = nn::RmsNorm;

pub struct VisionRotary {
    pub head_dim: i32,
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

    /// Build `(n_patches, head_dim)` cos / sin from coord lists.
    pub fn cos_sin(&self, h_coords: &[i32], w_coords: &[i32]) -> Result<(Array, Array), Error> {
        if h_coords.len() != w_coords.len() {
            return Err(Error::Vision(format!(
                "h_coords ({}) and w_coords ({}) must match",
                h_coords.len(),
                w_coords.len()
            )));
        }
        let n = h_coords.len() as i32;
        let h = Array::from_slice(h_coords, &[n]);
        let w = Array::from_slice(w_coords, &[n]);
        let inv = self.inv_freq.as_dtype(Dtype::Float32)?.reshape(&[1, -1])?;
        let hf = h.as_dtype(Dtype::Float32)?.reshape(&[-1, 1])?;
        let wf = w.as_dtype(Dtype::Float32)?.reshape(&[-1, 1])?;
        let freq_h = hf.matmul(&inv)?;
        let freq_w = wf.matmul(&inv)?;
        // freqs (n, head_dim/2) → emb (n, head_dim) via concat-twice for rotate_half pairing.
        let freqs = ops::concatenate_axis(&[freq_h, freq_w], -1)?;
        let emb = ops::concatenate_axis(&[freqs.clone(), freqs], -1)?;
        let cos = emb.cos()?;
        let sin = emb.sin()?;
        Ok((cos, sin))
    }
}

/// `rotate_half`: x → [-x_{d/2..}, x_{..d/2}] along the last axis.
fn rotate_half(x: &Array) -> Result<Array, Error> {
    let last = x.shape()[x.ndim() - 1];
    let half = last / 2;
    let x1 = x.index((.., ..half));
    let x2 = x.index((.., half..));
    let neg_x2 = x2.negative()?;
    Ok(ops::concatenate_axis(&[neg_x2, x1], -1)?)
}

fn apply_rotary(q: &Array, k: &Array, cos: &Array, sin: &Array) -> Result<(Array, Array), Error> {
    // q, k: (n, num_heads, head_dim); cos, sin: (n, head_dim)
    let cos = cos.expand_dims(1)?;
    let sin = sin.expand_dims(1)?;
    let q_rot = rotate_half(q)?;
    let k_rot = rotate_half(k)?;
    let q_out = q.multiply(&cos)?.add(&q_rot.multiply(&sin)?)?;
    let k_out = k.multiply(&cos)?.add(&k_rot.multiply(&sin)?)?;
    Ok((q_out, k_out))
}

pub struct VisionAttention {
    pub num_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub q_proj: nn::Linear,
    pub k_proj: nn::Linear,
    pub v_proj: nn::Linear,
    pub o_proj: nn::Linear,
}

impl VisionAttention {
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
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;
        let reshape = |a: Array| -> Result<Array, Error> {
            Ok(a.reshape(&[n, self.num_heads, self.head_dim])?)
        };
        let q = reshape(q)?;
        let k = reshape(k)?;
        let v = reshape(v)?;
        let (q, k) = apply_rotary(&q, &k, cos, sin)?;
        let to_bhnd = |a: Array| -> Result<Array, Error> {
            Ok(a.expand_dims(0)?.transpose_axes(&[0, 2, 1, 3])?)
        };
        let q = to_bhnd(q)?;
        let k = to_bhnd(k)?;
        let v = to_bhnd(v)?;
        let attn = scaled_dot_product_attention::<mlx_rs_core::cache::KVCache>(
            q,
            k,
            v,
            None,
            self.scale,
            None::<SdpaMask>,
        )?;
        let attn = attn.transpose_axes(&[0, 2, 1, 3])?.reshape(&[n, hidden])?;
        Ok(self.o_proj.forward(&attn)?)
    }
}

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
        let prod = act.multiply(&up)?;
        Ok(self.down_proj.forward(&prod)?)
    }
}

pub struct VisionBlock {
    pub norm1: VisionRmsNorm,
    pub attn: VisionAttention,
    pub norm2: VisionRmsNorm,
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

pub struct VisionMerger {
    pub norm: VisionRmsNorm,
    pub linear_1: nn::Linear,
    pub linear_2: nn::Linear,
    pub merge_size: i32,
}

impl VisionMerger {
    pub fn forward(&mut self, x: &Array, grid_h: i32, grid_w: i32) -> Result<Array, Error> {
        if grid_h % self.merge_size != 0 || grid_w % self.merge_size != 0 {
            return Err(Error::Vision(format!(
                "VisionMerger: grid ({grid_h}, {grid_w}) not divisible by merge_size {}",
                self.merge_size
            )));
        }
        let normed = self.norm.forward(x)?;
        let s = normed.shape();
        let hidden = s[1];
        let m = self.merge_size;
        let merged = normed
            .reshape(&[grid_h, grid_w, hidden])?
            .reshape(&[grid_h / m, m, grid_w / m, m, hidden])?
            .transpose_axes(&[0, 2, 1, 3, 4])?
            .reshape(&[(grid_h / m) * (grid_w / m), hidden * m * m])?;
        let h = self.linear_1.forward(&merged)?;
        let h = nn::gelu(&h)?;
        Ok(self.linear_2.forward(&h)?)
    }
}

pub struct VisionEncoder {
    pub config: GlmOcrVisionConfig,
    pub patch_embed: nn::Linear,
    pub rotary: VisionRotary,
    pub blocks: Vec<VisionBlock>,
    pub merger: VisionMerger,
}

impl VisionEncoder {
    /// Backward-compat stub for the original scaffold call site. Real
    /// loading lives in the upcoming `loader` module (phase 4).
    pub fn load_from_weights(
        _config: GlmOcrVisionConfig,
        _weights: &std::collections::HashMap<String, Array>,
    ) -> Result<Self, Error> {
        Err(Error::Vision(
            "VisionEncoder::load_from_weights deprecated; use the phase-4 loader"
                .to_string(),
        ))
    }

    /// `patches`: `(n_patches, 3 * t * P * P)`, `grid`: `(T_raw, gh, gw)`.
    /// Returns `(n_soft, out_hidden)`.
    pub fn forward(&mut self, patches: &Array, grid: (i32, i32, i32)) -> Result<Array, Error> {
        let (_t_raw, grid_h, grid_w) = grid;
        let mut h = self.patch_embed.forward(patches)?;
        let n_patches = (grid_h * grid_w) as usize;
        let mut h_coords = Vec::with_capacity(n_patches);
        let mut w_coords = Vec::with_capacity(n_patches);
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
        self.merger.forward(&h, grid_h, grid_w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotate_half_swaps_halves_with_negation() {
        let x = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 4]);
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
