//! Config parsing for the `glm_ocr` HF model family.
//!
//! Layout (from `models/GLM-OCR/config.json`):
//! ```json
//! {
//!   "architectures": ["GlmOcrForConditionalGeneration"],
//!   "model_type": "glm_ocr",
//!   "text_config":   { "model_type": "glm_ocr_text",   ... },
//!   "vision_config": { "model_type": "glm_ocr_vision", ... }
//! }
//! ```

use serde::Deserialize;
use std::path::Path;

use crate::error::Error;

#[derive(Debug, Clone, Deserialize)]
pub struct GlmOcrConfig {
    pub model_type: String,
    pub text_config: GlmOcrTextConfig,
    pub vision_config: GlmOcrVisionConfig,
    /// Image / boi / eoi / pad token ids inserted by the processor into
    /// the text stream around vision-encoder outputs.
    #[serde(default)]
    pub image_token_id: Option<i32>,
    #[serde(default)]
    pub image_start_token_id: Option<i32>,
    #[serde(default)]
    pub image_end_token_id: Option<i32>,
    #[serde(default)]
    pub pad_token_id: Option<i32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GlmOcrTextConfig {
    pub model_type: String,
    pub vocab_size: i32,
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    pub num_hidden_layers: i32,
    pub num_key_value_heads: i32,
    pub head_dim: i32,
    pub max_position_embeddings: i32,
    pub rms_norm_eps: f32,
    pub hidden_act: String,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub attention_dropout: f32,
    /// 0 → no MTP head; 1 → single nextn-predict layer (typical for
    /// glm_ocr_text).
    #[serde(default)]
    pub num_nextn_predict_layers: i32,
    pub rope_parameters: GlmOcrRopeParameters,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    pub dtype: Option<String>,
    pub pad_token_id: Option<i32>,
    pub eos_token_id: Option<serde_json::Value>,
}

/// Multimodal RoPE: the rotation dimensions are partitioned into
/// `mrope_section` groups so different axes (text / spatial-h /
/// spatial-w / temporal) use distinct position streams.
#[derive(Debug, Clone, Deserialize)]
pub struct GlmOcrRopeParameters {
    #[serde(default = "default_rope_type")]
    pub rope_type: String,
    pub mrope_section: Option<Vec<i32>>,
    /// Standard RoPE base / theta.
    #[serde(default)]
    pub rope_theta: Option<f32>,
}

fn default_rope_type() -> String {
    "default".to_string()
}

#[derive(Debug, Clone, Deserialize)]
pub struct GlmOcrVisionConfig {
    pub model_type: String,
    pub hidden_size: i32,
    pub depth: i32,
    pub num_heads: i32,
    pub intermediate_size: i32,
    pub image_size: i32,
    pub patch_size: i32,
    pub out_hidden_size: i32,
    pub rms_norm_eps: f32,
    #[serde(default = "default_true")]
    pub attention_bias: bool,
    pub hidden_act: String,
    #[serde(default = "default_one")]
    pub spatial_merge_size: i32,
    #[serde(default = "default_one")]
    pub temporal_patch_size: i32,
}

fn default_true() -> bool {
    true
}
fn default_one() -> i32 {
    1
}

pub fn load_config(model_dir: impl AsRef<Path>) -> Result<GlmOcrConfig, Error> {
    let path = model_dir.as_ref().join("config.json");
    let bytes = std::fs::read(&path).map_err(|e| {
        Error::Io(format!("failed to read {}: {e}", path.display()))
    })?;
    let cfg: GlmOcrConfig =
        serde_json::from_slice(&bytes).map_err(|e| Error::Config(e.to_string()))?;
    Ok(cfg)
}
