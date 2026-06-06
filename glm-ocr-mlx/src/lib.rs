//! GLM-OCR (vision-language OCR) inference on MLX.
//!
//! Full pipeline: image preprocessor → 24-layer ViT (with fused QKV +
//! per-head q/k_norm + 2-D rotary) → downsample (Conv2d 2x2) → SwiGLU
//! merger → Glm4-family 16-layer text decoder with **sandwich-norm**
//! pattern + fused gate_up_proj + MROPE.
//!
//! Entry point: [`loader::load_from_path`] builds a [`GlmOcrModel`]
//! from a HF-format `mlx-community/GLM-OCR-bf16` checkpoint dir.

pub mod config;
pub mod error;
pub mod loader;
pub mod mrope;
pub mod position_ids;
pub mod preprocessor;
pub mod text_decoder;
pub mod vision;

use std::path::Path;

pub use config::{
    load_config, GlmOcrConfig, GlmOcrRopeParameters, GlmOcrTextConfig, GlmOcrVisionConfig,
};
pub use error::Error;
pub use loader::{load_from_path, splice_image_tokens, GlmOcrModel, SpecialTokens};
pub use preprocessor::{load_preprocessor_config, PreprocessedImage, PreprocessorConfig};

/// Load the tokenizer from `model_dir/tokenizer.json`.
pub fn load_tokenizer(model_dir: impl AsRef<Path>) -> Result<tokenizers::Tokenizer, Error> {
    let path = model_dir.as_ref().join("tokenizer.json");
    tokenizers::Tokenizer::from_file(&path).map_err(|e| Error::Tokenizer(e.to_string()))
}
