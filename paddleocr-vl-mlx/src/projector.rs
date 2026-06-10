//! Vision → text projector (phase 4).
//!
//! Port of `Projector` in
//! `PaddlePaddle/PaddleOCR-VL-1.5/modeling_paddleocr_vl.py:813`.
//!
//! Maps the vision encoder's `(num_patches, vision_hidden)` features into
//! the text decoder's `(num_soft_tokens, text_hidden)` embedding space.
//! The "spatial merge" is a `m1 × m2` block-pool: every adjacent
//! `m1 × m2 = spatial_merge_size²` patches collapse into one soft token,
//! dropping the patch count by `m²`. For PaddleOCR-VL-1.5 this is the
//! default `(2, 2)` merge.
//!
//! Forward:
//! ```text
//!   image_feature: (t · h · w, vision_hidden=1152), image_grid_thw=(t,h,w)
//!     pre_norm(LayerNorm, dim=vision_hidden, eps=1e-5)
//!   reshape (t, h_m=h/m1, m1, w_m=w/m2, m2, vision_hidden)
//!     transpose to put (m1, m2) adjacent to the feature dim
//!     reshape to (t · h_m · w_m, m1 · m2 · vision_hidden = 4608)
//!   linear_1: 4608 → 4608 (bias)
//!     GELUActivation (exact, not approx)
//!   linear_2: 4608 → text_hidden=1024 (bias)
//!   → (num_soft_tokens, text_hidden)
//! ```

use mlx_rs::{
    module::{Module, Param},
    nn,
    ops::{self, indexing::IndexOp},
    Array,
};
use mlx_rs_core::error::{Error, Result};

use crate::config::{PaddleOcrVisionConfig, PaddleOcrVlConfig};
use crate::vision_model::VisionLayerNorm;

/// `(merge_kernel_h, merge_kernel_w)`. PaddleOCR-VL-1.5 uses `(2, 2)`.
pub type MergeKernel = (i32, i32);

pub struct Projector {
    pub merge_kernel: MergeKernel,
    pub vision_hidden: i32,
    pub merged_hidden: i32, // = vision_hidden * m1 * m2
    pub text_hidden: i32,
    pub pre_norm: VisionLayerNorm,
    pub linear_1: nn::Linear,
    pub linear_2: nn::Linear,
}

impl Projector {
    /// Project a single image's encoder output into the text-space soft
    /// tokens.
    ///
    /// * `image_feature` — `(num_patches, vision_hidden)` from the vision
    ///   encoder. **B has already been squeezed off** — callers with a
    ///   batch dim should slice it before passing in.
    /// * `image_grid_thw` — the `(T, H, W)` patch grid from the image
    ///   processor. `H` and `W` must be multiples of the merge kernel.
    ///
    /// Output: `(T · (H/m1) · (W/m2), text_hidden)` soft tokens, ready to
    /// splice into the LM embedding stream at the `<|IMAGE_PLACEHOLDER|>`
    /// positions.
    pub fn forward_single(
        &mut self,
        image_feature: &Array,
        image_grid_thw: (i32, i32, i32),
    ) -> Result<Array> {
        let (t, h, w) = image_grid_thw;
        let (m1, m2) = self.merge_kernel;
        if h % m1 != 0 || w % m2 != 0 {
            return Err(Error::InvalidConfig(format!(
                "Projector: grid ({h}, {w}) must be divisible by merge_kernel ({m1}, {m2})"
            )));
        }
        let h_m = h / m1;
        let w_m = w / m2;

        let shape = image_feature.shape();
        let expected = t * h * w;
        if shape.len() != 2 || shape[0] != expected || shape[1] != self.vision_hidden {
            return Err(Error::InvalidConfig(format!(
                "Projector: image_feature shape {:?} doesn't match grid ({t}, {h}, {w}) × \
                 vision_hidden={} (expected ({expected}, {}))",
                shape, self.vision_hidden, self.vision_hidden
            )));
        }

        // Pre-norm (LayerNorm over the feature dim, eps from the
        // canonical 1e-5 used by the Projector — distinct from
        // vision_config.layer_norm_eps which is 1e-6).
        let x = self.pre_norm.forward(image_feature)?;

        // Spatial 2×2 merge via reshape + transpose:
        //   (t · h_m · m1 · w_m · m2, d)
        //   → (t, h_m, m1, w_m, m2, d)
        //   → (t, h_m, w_m, m1, m2, d)   [move m1, m2 next to d]
        //   → (t · h_m · w_m, m1 · m2 · d)
        let d = self.vision_hidden;
        let r1 = x
            .reshape(&[t, h_m, m1, w_m, m2, d])
            .map_err(Error::from)?;
        let r2 = r1
            .transpose_axes(&[0, 1, 3, 2, 4, 5])
            .map_err(Error::from)?;
        let merged = r2
            .reshape(&[t * h_m * w_m, self.merged_hidden])
            .map_err(Error::from)?;

        // Linear → GELU exact → Linear.
        let h1 = self.linear_1.forward(&merged).map_err(Error::from)?;
        let h1 = nn::gelu(&h1).map_err(Error::from)?;
        self.linear_2.forward(&h1).map_err(Error::from)
    }

    /// Convenience: project a batched encoder output `(1, num_patches,
    /// vision_hidden)`. The B=1 case the chat path actually uses.
    pub fn forward_batched(
        &mut self,
        image_feature: &Array,
        image_grid_thw: (i32, i32, i32),
    ) -> Result<Array> {
        let shape = image_feature.shape();
        if shape.len() != 3 || shape[0] != 1 {
            return Err(Error::InvalidConfig(format!(
                "Projector::forward_batched expects (1, N, hidden); got {:?}",
                shape
            )));
        }
        let squeezed = image_feature.index((0, .., ..));
        self.forward_single(&squeezed, image_grid_thw)
    }
}

/// Build a random-weight Projector matching `config`. Used by the unit
/// tests below; production loaders fill the weights from safetensors.
pub fn build_projector_with_random_weights(
    text_config: &PaddleOcrVlConfig,
    vision_config: &PaddleOcrVisionConfig,
) -> Result<Projector> {
    use mlx_rs::random::uniform;

    let m1 = vision_config.spatial_merge_size;
    let m2 = vision_config.spatial_merge_size;
    let vision_hidden = vision_config.hidden_size;
    let merged_hidden = vision_hidden * m1 * m2;
    let text_hidden = text_config.hidden_size;

    let lin = |in_d: i32, out_d: i32| -> Result<nn::Linear> {
        let lo = -1.0 / (in_d as f32).sqrt();
        let hi = -lo;
        let w = uniform::<_, f32>(lo, hi, &[out_d, in_d], None).map_err(Error::from)?;
        let b = uniform::<_, f32>(lo, hi, &[out_d], None).map_err(Error::from)?;
        Ok(nn::Linear {
            weight: Param::new(w),
            bias: Param::new(Some(b)),
        })
    };

    Ok(Projector {
        merge_kernel: (m1, m2),
        vision_hidden,
        merged_hidden,
        text_hidden,
        // The reference uses eps=1e-5 specifically for the Projector's
        // pre_norm (distinct from the vision encoder's 1e-6).
        pre_norm: VisionLayerNorm {
            weight: ops::ones::<f32>(&[vision_hidden]).map_err(Error::from)?,
            bias: ops::zeros::<f32>(&[vision_hidden]).map_err(Error::from)?,
            eps: 1e-5,
        },
        linear_1: lin(merged_hidden, merged_hidden)?,
        linear_2: lin(merged_hidden, text_hidden)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::random::uniform;

    fn tiny_text_config() -> PaddleOcrVlConfig {
        serde_json::from_str(
            r#"{
              "model_type": "paddleocr_vl",
              "hidden_size": 16,
              "num_hidden_layers": 1,
              "num_attention_heads": 2,
              "num_key_value_heads": 1,
              "head_dim": 8,
              "intermediate_size": 32,
              "vocab_size": 20,
              "max_position_embeddings": 64,
              "rope_theta": 10000,
              "rope_scaling": { "mrope_section": [1, 1, 2], "rope_type": "default" },
              "rms_norm_eps": 1e-5,
              "hidden_act": "silu",
              "tie_word_embeddings": false,
              "sliding_window": null,
              "image_token_id": 19,
              "vision_config": {
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "intermediate_size": 16,
                "patch_size": 2,
                "image_size": 8,
                "spatial_merge_size": 2,
                "num_channels": 3,
                "layer_norm_eps": 1e-6,
                "hidden_act": "gelu_pytorch_tanh"
              }
            }"#,
        )
        .unwrap()
    }

    #[test]
    fn output_shape_after_2x2_merge() {
        let text_cfg = tiny_text_config();
        let vis_cfg = text_cfg.vision_config.clone();
        let mut proj = build_projector_with_random_weights(&text_cfg, &vis_cfg).unwrap();

        // T=1, H=4, W=4 → 16 patches in. After 2×2 merge: 4 soft tokens.
        let n_patches = 1 * 4 * 4;
        let features = uniform::<_, f32>(-1.0, 1.0, &[n_patches, vis_cfg.hidden_size], None).unwrap();
        let out = proj.forward_single(&features, (1, 4, 4)).unwrap();
        assert_eq!(out.shape(), &[4, text_cfg.hidden_size]);
    }

    #[test]
    fn batched_wrapper_matches_squeezed() {
        let text_cfg = tiny_text_config();
        let vis_cfg = text_cfg.vision_config.clone();
        let mut proj = build_projector_with_random_weights(&text_cfg, &vis_cfg).unwrap();

        let n_patches = 1 * 4 * 4;
        let features = uniform::<_, f32>(-1.0, 1.0, &[1, n_patches, vis_cfg.hidden_size], None).unwrap();
        let out = proj.forward_batched(&features, (1, 4, 4)).unwrap();
        assert_eq!(out.shape(), &[4, text_cfg.hidden_size]);
    }

    #[test]
    fn rejects_non_multiple_grid() {
        let text_cfg = tiny_text_config();
        let vis_cfg = text_cfg.vision_config.clone();
        let mut proj = build_projector_with_random_weights(&text_cfg, &vis_cfg).unwrap();

        // H=3 with merge_size=2 is not divisible → must reject at gate.
        let n_patches = 1 * 3 * 4;
        let features = uniform::<_, f32>(-1.0, 1.0, &[n_patches, vis_cfg.hidden_size], None).unwrap();
        let err = proj.forward_single(&features, (1, 3, 4));
        assert!(err.is_err());
    }

    #[test]
    fn rejects_feature_shape_mismatch() {
        let text_cfg = tiny_text_config();
        let vis_cfg = text_cfg.vision_config.clone();
        let mut proj = build_projector_with_random_weights(&text_cfg, &vis_cfg).unwrap();

        // Claim grid 1×4×4 (16 patches) but supply 12 — must reject.
        let features = uniform::<_, f32>(-1.0, 1.0, &[12, vis_cfg.hidden_size], None).unwrap();
        let err = proj.forward_single(&features, (1, 4, 4));
        assert!(err.is_err());
    }
}
