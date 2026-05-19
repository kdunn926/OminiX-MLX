//! `glm_ocr_vision` encoder scaffold.
//!
//! 24-layer ViT (1024 hidden, 16 heads, attention_bias=true, RMS norm,
//! SiLU MLP, intermediate 4096). Followed by a spatial-merge step that
//! groups 2×2 patches and a 2-layer MLP projector mapping 1024 →
//! `out_hidden_size` (1536 → matches text decoder hidden).
//!
//! Scaffold only — module structure + load surface stubbed; per-layer
//! forward bodies are TODO. The shape signature is `(patches[T, P*P*3])
//! → image_tokens[T_merged, out_hidden]` where `T_merged = T /
//! (merge_size² * temporal_patch_size)`.

use mlx_rs::{error::Exception, Array};

use crate::config::GlmOcrVisionConfig;
use crate::error::Error;

pub struct VisionEncoder {
    pub config: GlmOcrVisionConfig,
    // TODO(real impl):
    //   - patch_embed: nn::Linear(P*P*3 → hidden)
    //   - position_embed: optional 2D RoPE or learned table
    //   - layers: Vec<ViTBlock>  (each has RmsNorm, attn(QKV bias),
    //                             RmsNorm, SwiGLU)
    //   - post_norm: nn::RmsNorm
    //   - merger: 2-layer MLP that projects merged patches to
    //             out_hidden_size.
    _placeholder: (),
}

impl VisionEncoder {
    /// Load from a weights HashMap. Currently a stub that just stashes
    /// the config — the real load impl must read `vision.*` keys (or
    /// whatever prefix the GLM-OCR checkpoint uses) and construct the
    /// ViT block stack.
    pub fn load_from_weights(
        config: GlmOcrVisionConfig,
        _weights: &std::collections::HashMap<String, Array>,
    ) -> Result<Self, Error> {
        // Validation that comes for free without loading: head_dim
        // divides hidden_size, merge size divides patch grid, etc.
        if config.hidden_size % config.num_heads != 0 {
            return Err(Error::Vision(format!(
                "vision.hidden_size {} not divisible by num_heads {}",
                config.hidden_size, config.num_heads
            )));
        }
        Ok(Self {
            config,
            _placeholder: (),
        })
    }

    /// Forward: `patches[T, P*P*3]` → `image_tokens[T_merged, out_hidden]`.
    /// Stub returns zeros of the expected shape so the downstream
    /// splicing path compiles + the shape contract is testable.
    pub fn forward(&mut self, patches: &Array) -> Result<Array, Exception> {
        let t = patches.shape()[0];
        let merge = self.config.spatial_merge_size * self.config.spatial_merge_size;
        let temporal = self.config.temporal_patch_size;
        let t_merged = t / (merge * temporal).max(1);
        Array::zeros::<f32>(&[t_merged, self.config.out_hidden_size])
    }
}
