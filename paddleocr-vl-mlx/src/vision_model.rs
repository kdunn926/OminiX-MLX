//! SigLIP-style vision encoder (phase 3).
//!
//! Port of `PaddleOCRVisionModel` / `PaddleOCRVisionTransformer` /
//! `PaddleOCREncoder` in
//! `PaddlePaddle/PaddleOCR-VL-1.5/modeling_paddleocr_vl.py:872..1700`.
//!
//! Scope reductions relative to the HF reference:
//!   * Single image, single sequence (`B = 1`) — no batched packing /
//!     `cu_seqlens` / `sample_indices` machinery.
//!   * `use_rope = False` (the simple path; the OCR processor doesn't
//!     emit `height_position_ids` / `width_position_ids`).
//!   * No pooler head: the `Projector` (phase 4) consumes the pre-pool
//!     `(num_patches, hidden)` features directly. Pooler weights in the
//!     checkpoint can be ignored at load time.
//!   * No window attention, no flash-attn-specific path — plain SDPA
//!     with no mask.
//!
//! What's preserved:
//!   * `Conv2d` patch embedding at `stride = patch_size`.
//!   * Positional embedding lookup at the **canonical** `image_size`
//!     grid (interpolation to variable resolutions is a follow-up; the
//!     forward asserts the input grid matches the canonical one for now).
//!   * 27 standard pre-LayerNorm encoder blocks (LN → MHA → residual →
//!     LN → MLP → residual).
//!   * GELU `pytorch_tanh` (approximate) inside the MLP.
//!   * Final `post_layernorm`.

use mlx_rs::{
    module::{Module, Param},
    nn,
    ops::{self, indexing::IndexOp},
    Array,
};
use mlx_rs_core::error::{Error, Result};
use mlx_rs_core::utils::{scaled_dot_product_attention, SdpaMask};

use crate::config::PaddleOcrVisionConfig;

/// Convenience wrapper around `mlx_rs::fast::layer_norm` so we can store
/// weight/bias as plain `Array`s and reuse the fast Metal kernel.
pub struct VisionLayerNorm {
    pub weight: Array,
    pub bias: Array,
    pub eps: f32,
}

impl VisionLayerNorm {
    pub fn forward(&self, x: &Array) -> Result<Array> {
        mlx_rs::fast::layer_norm(x, Some(&self.weight), Some(&self.bias), self.eps)
            .map_err(Error::from)
    }
}

/// SigLIP-style multi-head self-attention. Separate Q/K/V/Out projections
/// (no fused QKV) to match the canonical PaddleOCR-VL weight layout.
pub struct VisionAttention {
    pub num_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub q_proj: nn::Linear,
    pub k_proj: nn::Linear,
    pub v_proj: nn::Linear,
    pub out_proj: nn::Linear,
}

impl VisionAttention {
    pub fn forward(&mut self, x: &Array) -> Result<Array> {
        let shape = x.shape();
        let b = shape[0];
        let n = shape[1];
        let hidden = shape[2];
        let q = self.q_proj.forward(x).map_err(Error::from)?;
        let k = self.k_proj.forward(x).map_err(Error::from)?;
        let v = self.v_proj.forward(x).map_err(Error::from)?;
        // (B, N, hidden) → (B, num_heads, N, head_dim)
        let reshape = |a: &Array| -> Result<Array> {
            a.reshape(&[b, n, self.num_heads, self.head_dim])
                .map_err(Error::from)?
                .transpose_axes(&[0, 2, 1, 3])
                .map_err(Error::from)
        };
        let q = reshape(&q)?;
        let k = reshape(&k)?;
        let v = reshape(&v)?;
        // SDPA without a cache (this is a one-shot vision pass; nothing to
        // memoise between calls). `_cache: None::<KVCache>` lets the type
        // parameter resolve to a concrete `KeyValueCache` impl that the
        // helper requires; the value isn't consulted because mask = None.
        let attn = scaled_dot_product_attention::<mlx_rs_core::cache::KVCache>(
            q,
            k,
            v,
            None,
            self.scale,
            None::<SdpaMask>,
        )
        .map_err(Error::from)?;
        // (B, H, N, D) → (B, N, H * D)
        let attn = attn
            .transpose_axes(&[0, 2, 1, 3])
            .map_err(Error::from)?
            .reshape(&[b, n, hidden])
            .map_err(Error::from)?;
        self.out_proj.forward(&attn).map_err(Error::from)
    }
}

/// MLP block. PaddleOCR-VL uses `gelu_pytorch_tanh` (the tanh-approximate
/// formulation of GELU). MLX exposes that as `gelu_approximate` — the math
/// is identical up to FP rounding.
pub struct VisionMlp {
    pub fc1: nn::Linear,
    pub fc2: nn::Linear,
}

impl VisionMlp {
    pub fn forward(&mut self, x: &Array) -> Result<Array> {
        let h = self.fc1.forward(x).map_err(Error::from)?;
        let h = nn::gelu_approximate(&h).map_err(Error::from)?;
        self.fc2.forward(&h).map_err(Error::from)
    }
}

/// One pre-LayerNorm encoder block.
pub struct VisionEncoderLayer {
    pub layer_norm1: VisionLayerNorm,
    pub self_attn: VisionAttention,
    pub layer_norm2: VisionLayerNorm,
    pub mlp: VisionMlp,
}

impl VisionEncoderLayer {
    pub fn forward(&mut self, x: &Array) -> Result<Array> {
        let normed = self.layer_norm1.forward(x)?;
        let attn = self.self_attn.forward(&normed)?;
        let h = x.add(&attn).map_err(Error::from)?;
        let normed = self.layer_norm2.forward(&h)?;
        let mlp = self.mlp.forward(&normed)?;
        h.add(&mlp).map_err(Error::from)
    }
}

/// Patch + positional-embedding block.
///
/// `position_embedding.weight` has shape `(num_positions, hidden)` where
/// `num_positions = (image_size / patch_size)²` (the **canonical** grid).
/// Variable-resolution inputs would need bilinear interpolation against
/// this fixed grid — the forward below errors out when the input grid
/// doesn't match `num_positions`, which is enough for the phase-3 smoke;
/// the interpolation hook is a follow-up.
pub struct VisionEmbeddings {
    pub patch_embedding: nn::Conv2d,
    pub position_embedding: nn::Embedding,
    /// Cached `arange(0, num_positions)` — exposed so the canonical-grid
    /// forward doesn't allocate a new ids tensor every call.
    pub position_ids: Array,
    pub num_positions: i32,
    pub embed_dim: i32,
}

impl VisionEmbeddings {
    /// Run the patch projection and add positional embeddings.
    ///
    /// * `pixel_values` — `(B, C, H, W)` channels-first per the HF
    ///   convention. Internally transposed to MLX's channels-last layout
    ///   `(B, H, W, C)` before the Conv2d call. Variable-resolution input
    ///   is supported: when the input patch grid differs from the
    ///   canonical pre-trained grid (`sqrt(num_positions) × sqrt(num_positions)`),
    ///   the position embedding is bilinearly interpolated to match.
    pub fn forward(&mut self, pixel_values: &Array) -> Result<Array> {
        // (B, C, H, W) → (B, H, W, C) for MLX Conv2d.
        let bhwc = pixel_values
            .transpose_axes(&[0, 2, 3, 1])
            .map_err(Error::from)?;
        let conv_out = self.patch_embedding.forward(&bhwc).map_err(Error::from)?;
        // Conv2d output in mlx-rs is channels-last: (B, H', W', out_channels).
        let s = conv_out.shape();
        let b = s[0];
        let h_prime = s[1];
        let w_prime = s[2];
        let oc = s[3];
        let n_patches = h_prime * w_prime;
        let flat = conv_out
            .reshape(&[b, n_patches, oc])
            .map_err(Error::from)?;
        let pos = self.position_embedding_for_grid(h_prime, w_prime)?;
        flat.add(&pos).map_err(Error::from)
    }

    /// Return position embeddings of shape `(num_patches=h*w, hidden)`,
    /// resizing the pre-trained `(sqrt(num_positions) × sqrt(num_positions))`
    /// grid via bilinear interpolation when the input grid differs.
    fn position_embedding_for_grid(&mut self, h: i32, w: i32) -> Result<Array> {
        // Fast path: input grid matches pre-trained grid — no interp.
        let pre_grid = (self.num_positions as f64).sqrt().round() as i32;
        if pre_grid * pre_grid != self.num_positions {
            return Err(Error::Model(format!(
                "VisionEmbeddings: position_embedding's num_positions = {} is not a perfect \
                 square — bilinear interpolation can't proceed (TODO: rect grids)",
                self.num_positions
            )));
        }
        let raw = self
            .position_embedding
            .forward(&self.position_ids)
            .map_err(Error::from)?;
        if h == pre_grid && w == pre_grid {
            return Ok(raw);
        }
        // Reshape (pre_grid², hidden) → (1, pre_grid, pre_grid, hidden) for
        // MLX Upsample (channels-last). Bilinearly resize the spatial dims
        // to (h, w), reshape back to (h*w, hidden), then broadcast against
        // the conv output.
        let grid4 = raw
            .reshape(&[1, pre_grid, pre_grid, self.embed_dim])
            .map_err(Error::from)?;
        let scale_h = h as f32 / pre_grid as f32;
        let scale_w = w as f32 / pre_grid as f32;
        let upsample = mlx_rs::nn::Upsample::new(
            mlx_rs::utils::SingleOrVec::Vec(vec![scale_h, scale_w]),
            mlx_rs::nn::UpsampleMode::Linear { align_corners: false },
        );
        let mut upsample = upsample;
        let resized = upsample.forward(&grid4).map_err(Error::from)?;
        let rs = resized.shape();
        if rs[1] != h || rs[2] != w {
            return Err(Error::Model(format!(
                "VisionEmbeddings: pos-embed Upsample produced ({}, {}) but expected ({h}, {w})",
                rs[1], rs[2]
            )));
        }
        resized.reshape(&[h * w, self.embed_dim]).map_err(Error::from)
    }
}

/// Vision transformer body: embeddings → N layers → post_layernorm.
/// Output shape: `(B, num_patches, hidden)`.
pub struct VisionTransformer {
    pub embeddings: VisionEmbeddings,
    pub layers: Vec<VisionEncoderLayer>,
    pub post_layernorm: VisionLayerNorm,
}

impl VisionTransformer {
    pub fn forward(&mut self, pixel_values: &Array) -> Result<Array> {
        let mut h = self.embeddings.forward(pixel_values)?;
        for layer in self.layers.iter_mut() {
            h = layer.forward(&h)?;
        }
        self.post_layernorm.forward(&h)
    }
}

/// Build an **uninitialised** (random-weight) `VisionTransformer` matching
/// `config`. Production loaders fill the weights from safetensors after.
pub fn build_vision_with_random_weights(
    config: &PaddleOcrVisionConfig,
) -> Result<VisionTransformer> {
    use mlx_rs::random::uniform;

    let h = config.hidden_size;
    let n_h = config.num_attention_heads;
    if h % n_h != 0 {
        return Err(Error::InvalidConfig(format!(
            "vision: hidden_size ({h}) must be divisible by num_attention_heads ({n_h})"
        )));
    }
    let d = h / n_h;
    let inter = config.intermediate_size;
    let scale = 1.0_f32 / (d as f32).sqrt();

    let lin = |in_d: i32, out_d: i32, bias: bool| -> Result<nn::Linear> {
        let lo = -1.0 / (in_d as f32).sqrt();
        let hi = 1.0 / (in_d as f32).sqrt();
        let w = uniform::<_, f32>(lo, hi, &[out_d, in_d], None).map_err(Error::from)?;
        let b = if bias {
            Some(uniform::<_, f32>(lo, hi, &[out_d], None).map_err(Error::from)?)
        } else {
            None
        };
        Ok(nn::Linear {
            weight: Param::new(w),
            bias: Param::new(b),
        })
    };

    let ln = |dim: i32| -> Result<VisionLayerNorm> {
        Ok(VisionLayerNorm {
            weight: ops::ones::<f32>(&[dim]).map_err(Error::from)?,
            bias: ops::zeros::<f32>(&[dim]).map_err(Error::from)?,
            eps: config.layer_norm_eps,
        })
    };

    // Patch embedding: in_channels = num_channels, out_channels = hidden,
    // kernel_size = patch_size, stride = patch_size. Conv2d expects weight
    // shape (out_channels, kernel_h, kernel_w, in_channels).
    let conv_weight_shape = [h, config.patch_size, config.patch_size, config.num_channels];
    let conv_lo = -1.0 / ((config.patch_size * config.patch_size * config.num_channels) as f32).sqrt();
    let conv_hi = -conv_lo;
    let conv_w =
        uniform::<_, f32>(conv_lo, conv_hi, &conv_weight_shape, None).map_err(Error::from)?;
    let conv_b = uniform::<_, f32>(conv_lo, conv_hi, &[h], None).map_err(Error::from)?;
    let patch_embedding = nn::Conv2d {
        weight: Param::new(conv_w),
        bias: Param::new(Some(conv_b)),
        stride: (config.patch_size, config.patch_size),
        padding: (0, 0),
        dilation: (1, 1),
        groups: 1,
    };

    // Position embedding: (num_positions, hidden), where num_positions =
    // (image_size / patch_size)².
    if config.image_size % config.patch_size != 0 {
        return Err(Error::InvalidConfig(format!(
            "vision: image_size ({}) must be divisible by patch_size ({})",
            config.image_size, config.patch_size
        )));
    }
    let grid = config.image_size / config.patch_size;
    let num_positions = grid * grid;
    let pos_w = uniform::<_, f32>(-0.02, 0.02, &[num_positions, h], None).map_err(Error::from)?;
    let position_embedding = nn::Embedding {
        weight: Param::new(pos_w),
    };
    let position_ids: Vec<i32> = (0..num_positions).collect();
    let position_ids = Array::from_slice(&position_ids, &[num_positions]);

    let embeddings = VisionEmbeddings {
        patch_embedding,
        position_embedding,
        position_ids,
        num_positions,
        embed_dim: h,
    };

    let mut layers = Vec::with_capacity(config.num_hidden_layers as usize);
    for _ in 0..config.num_hidden_layers {
        let attn = VisionAttention {
            num_heads: n_h,
            head_dim: d,
            scale,
            q_proj: lin(h, h, true)?,
            k_proj: lin(h, h, true)?,
            v_proj: lin(h, h, true)?,
            out_proj: lin(h, h, true)?,
        };
        let mlp = VisionMlp {
            fc1: lin(h, inter, true)?,
            fc2: lin(inter, h, true)?,
        };
        layers.push(VisionEncoderLayer {
            layer_norm1: ln(h)?,
            self_attn: attn,
            layer_norm2: ln(h)?,
            mlp,
        });
    }

    let post_layernorm = ln(h)?;
    Ok(VisionTransformer {
        embeddings,
        layers,
        post_layernorm,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::random::uniform;

    fn tiny_vision_config() -> PaddleOcrVisionConfig {
        // Compact ViT for shape-only tests: hidden=8, 2 layers, 2 heads,
        // image=8 with patch=4 → grid 2×2 → 4 patches.
        serde_json::from_str::<PaddleOcrVisionConfig>(
            r#"{
              "hidden_size": 8,
              "num_hidden_layers": 2,
              "num_attention_heads": 2,
              "intermediate_size": 16,
              "patch_size": 4,
              "image_size": 8,
              "spatial_merge_size": 2,
              "num_channels": 3,
              "layer_norm_eps": 1e-6,
              "hidden_act": "gelu_pytorch_tanh"
            }"#,
        )
        .unwrap()
    }

    #[test]
    fn forward_shape_at_canonical_resolution() {
        let cfg = tiny_vision_config();
        let mut tower = build_vision_with_random_weights(&cfg).unwrap();
        // Canonical input: (B=1, C=3, H=8, W=8) → 2×2 = 4 patches.
        let pixels = uniform::<_, f32>(-1.0, 1.0, &[1, 3, 8, 8], None).unwrap();
        let out = tower.forward(&pixels).unwrap();
        assert_eq!(out.shape(), &[1, 4, cfg.hidden_size]);
    }

    #[test]
    fn variable_resolution_bilinearly_interpolates() {
        // Tiny ViT: canonical pre_grid = 8/4 = 2, num_positions = 4.
        // Feed an off-canonical input (12×12 → 3×3 grid = 9 patches);
        // bilinear interp should resize the position table from 2×2 to
        // 3×3 and the forward must succeed at shape (1, 9, hidden).
        let cfg = tiny_vision_config();
        let mut tower = build_vision_with_random_weights(&cfg).unwrap();
        let pixels = uniform::<_, f32>(-1.0, 1.0, &[1, 3, 12, 12], None).unwrap();
        let out = tower.forward(&pixels).unwrap();
        assert_eq!(out.shape(), &[1, 9, cfg.hidden_size]);
    }

    #[test]
    fn rejects_bad_config_at_build() {
        // image_size not divisible by patch_size — caught at construction.
        let mut cfg = tiny_vision_config();
        cfg.patch_size = 3; // 8 % 3 != 0
        let r = build_vision_with_random_weights(&cfg);
        assert!(r.is_err());
    }
}
