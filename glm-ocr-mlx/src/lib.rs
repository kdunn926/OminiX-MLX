//! GLM-OCR (vision-language OCR) inference on MLX.
//!
//! Scaffold: config + preprocessor + module shells in place; the heavy
//! lifting (ViT vision encoder forward, glm_ocr_text decoder with
//! mrope + MTP, multimodal token splicing) is staged across the
//! follow-up implementation pass.

pub mod config;
pub mod error;
pub mod mrope;
pub mod position_ids;
pub mod preprocessor;
pub mod text;
pub mod vision;

use std::path::Path;

pub use config::{
    load_config, GlmOcrConfig, GlmOcrRopeParameters, GlmOcrTextConfig, GlmOcrVisionConfig,
};
pub use error::Error;
pub use preprocessor::{load_preprocessor_config, PreprocessedImage, PreprocessorConfig};
pub use text::TextDecoder;
pub use vision::VisionEncoder;

/// Loaded GLM-OCR model: tokenizer + image processor + vision encoder +
/// text decoder. Currently a scaffold container; loading the actual
/// safetensors into the inner modules is the next step.
pub struct GlmOcrModel {
    pub config: GlmOcrConfig,
    pub preprocessor: PreprocessorConfig,
    pub tokenizer: tokenizers::Tokenizer,
    pub vision: VisionEncoder,
    pub text: TextDecoder,
}

pub fn load_tokenizer(model_dir: impl AsRef<Path>) -> Result<tokenizers::Tokenizer, Error> {
    let path = model_dir.as_ref().join("tokenizer.json");
    tokenizers::Tokenizer::from_file(&path).map_err(|e| Error::Tokenizer(e.to_string()))
}

/// Load a GLM-OCR model directory. Scaffold version: parses configs,
/// loads tokenizer + safetensors weight map, constructs the encoder /
/// decoder shells. Encoder/decoder forward is still stubbed — only
/// shape / load-path errors surface here.
pub fn load_model(model_dir: impl AsRef<Path>) -> Result<GlmOcrModel, Error> {
    let model_dir = model_dir.as_ref();
    let config = load_config(model_dir)?;
    if config.model_type != "glm_ocr" {
        return Err(Error::Config(format!(
            "expected model_type=glm_ocr, got {}",
            config.model_type
        )));
    }
    let preprocessor = load_preprocessor_config(model_dir)?;
    let tokenizer = load_tokenizer(model_dir)?;
    let weights = load_weights(model_dir)?;
    let vision = VisionEncoder::load_from_weights(config.vision_config.clone(), &weights)?;
    let text = TextDecoder::load_from_weights(config.text_config.clone(), &weights)?;
    Ok(GlmOcrModel {
        config,
        preprocessor,
        tokenizer,
        vision,
        text,
    })
}

/// Load every safetensors file in the model dir into a single
/// HashMap. Mirrors the gemma4-mlx loader pattern; the per-component
/// loaders (vision, text) pick out the keys they need.
fn load_weights(
    model_dir: &Path,
) -> Result<std::collections::HashMap<String, mlx_rs::Array>, Error> {
    let mut all = std::collections::HashMap::new();
    let mut entries: Vec<_> = std::fs::read_dir(model_dir)
        .map_err(|e| Error::Io(format!("read_dir {}: {e}", model_dir.display())))?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|x| x == "safetensors")
                .unwrap_or(false)
        })
        .collect();
    entries.sort_by_key(|e| e.path());
    if entries.is_empty() {
        return Err(Error::Io(format!(
            "no safetensors in {}",
            model_dir.display()
        )));
    }
    for entry in entries {
        let path = entry.path();
        let map = mlx_rs::Array::load_safetensors(&path)
            .map_err(|e| Error::Safetensors(e.to_string()))?;
        for (k, v) in map {
            all.insert(k, v);
        }
    }
    Ok(all)
}
