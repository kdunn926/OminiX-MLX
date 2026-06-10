//! PaddleOCR-VL-1.5 configuration deserialisers.
//!
//! Parses the HF-style `config.json` shipped by
//! `PaddlePaddle/PaddleOCR-VL-1.5` (and the same layout used by 1.6). The
//! root config carries the text-decoder fields; the vision encoder lives
//! under `vision_config`. Only the fields the inference path actually
//! consumes are made first-class — anything else is tolerated via `serde`
//! defaults.

use std::path::Path;

use serde::Deserialize;

use mlx_rs_core::error::{Error, Result};

/// 3-channel partition for MROPE — see [`crate::mrope::MropePartition`].
fn default_mrope_section() -> Vec<i32> {
    vec![16, 24, 24]
}

fn default_rope_theta() -> f32 {
    500_000.0
}

fn default_rms_norm_eps() -> f32 {
    1e-5
}

fn default_hidden_act() -> String {
    "silu".to_string()
}

fn default_head_dim() -> i32 {
    128
}

/// MROPE-specific section of `rope_scaling`.
///
/// The HF `config.json` ships **both** `rope_type` (newer field) and `type`
/// (HF back-compat). Serde's `alias` rejects that as a duplicate; the raw
/// `Deserialize` impl below tolerates both, preferring `rope_type` when
/// present and falling back to `type` otherwise.
#[derive(Debug, Clone)]
pub struct RopeScaling {
    pub mrope_section: Vec<i32>,
    pub rope_type: Option<String>,
}

impl Default for RopeScaling {
    fn default() -> Self {
        Self {
            mrope_section: default_mrope_section(),
            rope_type: Some("default".to_string()),
        }
    }
}

impl<'de> Deserialize<'de> for RopeScaling {
    fn deserialize<D>(d: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Raw {
            #[serde(default = "default_mrope_section")]
            mrope_section: Vec<i32>,
            #[serde(default)]
            rope_type: Option<String>,
            #[serde(default, rename = "type")]
            ty: Option<String>,
        }
        let raw = Raw::deserialize(d)?;
        Ok(RopeScaling {
            mrope_section: raw.mrope_section,
            rope_type: raw.rope_type.or(raw.ty),
        })
    }
}

/// Vision-encoder config (`vision_config` block).
#[derive(Debug, Clone, Deserialize)]
pub struct PaddleOcrVisionConfig {
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub intermediate_size: i32,
    pub patch_size: i32,
    pub image_size: i32,
    pub spatial_merge_size: i32,
    pub num_channels: i32,
    #[serde(default = "default_vision_layer_norm_eps")]
    pub layer_norm_eps: f32,
    /// `hidden_act` (e.g. `gelu_pytorch_tanh`).
    #[serde(default = "default_vision_hidden_act")]
    pub hidden_act: String,
}

fn default_vision_layer_norm_eps() -> f32 {
    1e-6
}

fn default_vision_hidden_act() -> String {
    "gelu_pytorch_tanh".to_string()
}

/// Text-decoder + top-level config (Ernie 4.5 fields live at the root).
#[derive(Debug, Clone, Deserialize)]
pub struct PaddleOcrVlConfig {
    pub model_type: String,
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    #[serde(default = "default_head_dim")]
    pub head_dim: i32,
    pub intermediate_size: i32,
    /// Pulled from `tokenizer.json` rather than from the config in the
    /// reference checkpoint, but tracked here for completeness — the
    /// canonical 1.5/1.6 ship `vocab_size` on the root config.
    #[serde(default)]
    pub vocab_size: Option<i32>,
    #[serde(default)]
    pub max_position_embeddings: i32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub rope_scaling: RopeScaling,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// `None` for PaddleOCR-VL (no sliding-window attention). Kept for
    /// schema parity with HF.
    #[serde(default)]
    pub sliding_window: Option<i32>,
    pub image_token_id: i32,
    /// Vision-tower sub-config (top of file).
    pub vision_config: PaddleOcrVisionConfig,
}

impl PaddleOcrVlConfig {
    /// Load and parse `<model_dir>/config.json`.
    pub fn from_path(model_dir: impl AsRef<Path>) -> Result<Self> {
        let cfg_path = model_dir.as_ref().join("config.json");
        let f = std::fs::File::open(&cfg_path)?;
        let cfg: PaddleOcrVlConfig = serde_json::from_reader(f).map_err(|e| {
            Error::Model(format!(
                "PaddleOcrVlConfig decode ({}): {e}",
                cfg_path.display()
            ))
        })?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Sanity-check fields the rest of the crate relies on. Catches
    /// config drift across PaddleOCR-VL revisions (e.g. 1.6 introducing
    /// new heads we haven't read about) at load time, before any
    /// matmul shape mismatch surfaces.
    pub fn validate(&self) -> Result<()> {
        if self.model_type != "paddleocr_vl" {
            return Err(Error::InvalidConfig(format!(
                "PaddleOcrVlConfig: model_type='{}' (expected 'paddleocr_vl')",
                self.model_type
            )));
        }
        if self.num_attention_heads <= 0 || self.num_key_value_heads <= 0 {
            return Err(Error::InvalidConfig(format!(
                "PaddleOcrVlConfig: invalid head counts (heads={}, kv_heads={})",
                self.num_attention_heads, self.num_key_value_heads
            )));
        }
        if self.num_attention_heads % self.num_key_value_heads != 0 {
            return Err(Error::InvalidConfig(format!(
                "PaddleOcrVlConfig: num_attention_heads ({}) must be a multiple of \
                 num_key_value_heads ({})",
                self.num_attention_heads, self.num_key_value_heads
            )));
        }
        if self.head_dim <= 0 {
            return Err(Error::InvalidConfig(format!(
                "PaddleOcrVlConfig: head_dim={} must be positive",
                self.head_dim
            )));
        }
        // Intentionally no `num_heads * head_dim == hidden_size` check:
        // GQA Ernie/Llama-style models routinely have Q-dim > hidden_size
        // (Q proj: hidden → n_heads × head_dim; O proj contracts back).
        // The PaddleOCR-VL 1.5 config is exactly this shape:
        // hidden=1024, n_heads=16, head_dim=128 → Q dim = 2048.
        let section_sum: i32 = self.rope_scaling.mrope_section.iter().sum();
        if section_sum * 2 != self.head_dim {
            return Err(Error::InvalidConfig(format!(
                "PaddleOcrVlConfig: 2 * sum(rope_scaling.mrope_section) = {} \
                 must equal head_dim ({})",
                section_sum * 2,
                self.head_dim
            )));
        }
        if self.vision_config.spatial_merge_size <= 0 {
            return Err(Error::InvalidConfig(format!(
                "PaddleOcrVlConfig: vision_config.spatial_merge_size={} must be positive",
                self.vision_config.spatial_merge_size
            )));
        }
        Ok(())
    }

    /// Convenience: `num_attention_heads / num_key_value_heads`. GQA group
    /// count used by the attention forward to repeat K/V heads.
    pub fn kv_groups(&self) -> i32 {
        self.num_attention_heads / self.num_key_value_heads
    }
}

/// Load `<model_dir>/tokenizer.json` via the `tokenizers` crate.
pub fn load_tokenizer(model_dir: impl AsRef<Path>) -> Result<tokenizers::Tokenizer> {
    let path = model_dir.as_ref().join("tokenizer.json");
    tokenizers::Tokenizer::from_file(&path).map_err(|e| {
        Error::Model(format!(
            "load_tokenizer ({}): {e}",
            path.display()
        ))
    })
}

/// Special tokens the PaddleOCR-VL chat / vision-splice paths key off.
///
/// IDs are pulled from the loaded tokenizer (or the values in
/// `added_tokens.json` if the field isn't surfaced through the
/// `tokenizers` API). Stored once at model load time and never
/// re-resolved.
#[derive(Debug, Clone)]
pub struct SpecialTokens {
    pub image_token_id: u32,
    /// `<|IMAGE_START|>` — wraps each image's soft-token block.
    pub image_start_id: u32,
    /// `<|IMAGE_END|>`
    pub image_end_id: u32,
    /// `<|IMAGE_SEP|>` — separator between multiple images in a single
    /// turn (not commonly used in the OCR chat template but reserved).
    pub image_sep_id: Option<u32>,
    /// `<|image_pad|>` — placeholder that some processors emit. Not used
    /// by our preprocessor but tracked for completeness.
    pub image_pad_id: Option<u32>,
}

impl SpecialTokens {
    /// Resolve special tokens from a loaded tokenizer + the
    /// `image_token_id` value in the config.
    pub fn resolve(
        tokenizer: &tokenizers::Tokenizer,
        image_token_id_from_config: i32,
    ) -> Result<Self> {
        let lookup = |s: &str| -> Option<u32> { tokenizer.token_to_id(s) };
        let image_start_id = lookup("<|IMAGE_START|>").ok_or_else(|| {
            Error::Model("tokenizer missing <|IMAGE_START|>".to_string())
        })?;
        let image_end_id = lookup("<|IMAGE_END|>").ok_or_else(|| {
            Error::Model("tokenizer missing <|IMAGE_END|>".to_string())
        })?;
        // Resolve image_token_id from the tokenizer when possible; fall
        // back to the config's value (they must agree on a well-formed
        // checkpoint, but the config is what get_rope_index uses).
        let image_token_id = lookup("<|IMAGE_PLACEHOLDER|>")
            .unwrap_or(image_token_id_from_config as u32);
        if image_token_id as i32 != image_token_id_from_config {
            return Err(Error::Model(format!(
                "SpecialTokens: tokenizer-side <|IMAGE_PLACEHOLDER|> id ({}) disagrees \
                 with config.image_token_id ({}); refusing to load",
                image_token_id, image_token_id_from_config,
            )));
        }
        Ok(Self {
            image_token_id,
            image_start_id,
            image_end_id,
            image_sep_id: lookup("<|IMAGE_SEP|>"),
            image_pad_id: lookup("<|image_pad|>"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn canonical_config_json() -> &'static str {
        // Subset of the 1.5 config that exercises every parsed field +
        // validates the partition/head_dim arithmetic.
        r#"{
          "model_type": "paddleocr_vl",
          "hidden_size": 1024,
          "num_hidden_layers": 18,
          "num_attention_heads": 16,
          "num_key_value_heads": 2,
          "head_dim": 128,
          "intermediate_size": 3072,
          "vocab_size": 103424,
          "max_position_embeddings": 131072,
          "rope_theta": 500000,
          "rope_scaling": {
            "mrope_section": [16, 24, 24],
            "rope_type": "default",
            "type": "default"
          },
          "rms_norm_eps": 1e-5,
          "hidden_act": "silu",
          "tie_word_embeddings": false,
          "sliding_window": null,
          "image_token_id": 100295,
          "vision_config": {
            "hidden_size": 1152,
            "num_hidden_layers": 27,
            "num_attention_heads": 16,
            "intermediate_size": 4304,
            "patch_size": 14,
            "image_size": 384,
            "spatial_merge_size": 2,
            "num_channels": 3,
            "layer_norm_eps": 1e-6,
            "hidden_act": "gelu_pytorch_tanh"
          }
        }"#
    }

    #[test]
    fn parses_canonical_config() {
        let cfg: PaddleOcrVlConfig = serde_json::from_str(canonical_config_json()).unwrap();
        assert_eq!(cfg.model_type, "paddleocr_vl");
        assert_eq!(cfg.hidden_size, 1024);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.rope_scaling.mrope_section, vec![16, 24, 24]);
        assert_eq!(cfg.rope_scaling.rope_type.as_deref(), Some("default"));
        assert_eq!(cfg.kv_groups(), 8);
        assert_eq!(cfg.image_token_id, 100295);
        assert_eq!(cfg.vision_config.patch_size, 14);
        assert_eq!(cfg.vision_config.spatial_merge_size, 2);
        cfg.validate().unwrap();
    }

    #[test]
    fn validate_catches_partition_mismatch() {
        let mut cfg: PaddleOcrVlConfig = serde_json::from_str(canonical_config_json()).unwrap();
        cfg.rope_scaling.mrope_section = vec![16, 24, 23]; // sums to 63, *2 = 126 ≠ 128
        let r = cfg.validate();
        assert!(r.is_err(), "must reject mrope_section sum * 2 != head_dim");
    }

    #[test]
    fn validate_catches_zero_head_dim() {
        let mut cfg: PaddleOcrVlConfig = serde_json::from_str(canonical_config_json()).unwrap();
        cfg.head_dim = 0;
        let r = cfg.validate();
        assert!(r.is_err(), "must reject head_dim = 0");
    }

    #[test]
    fn validate_catches_gqa_misalignment() {
        let mut cfg: PaddleOcrVlConfig = serde_json::from_str(canonical_config_json()).unwrap();
        cfg.num_key_value_heads = 3; // 16 not divisible by 3
        let r = cfg.validate();
        assert!(r.is_err(), "must reject num_attention_heads % num_key_value_heads != 0");
    }

    #[test]
    fn validate_catches_wrong_model_type() {
        let mut cfg: PaddleOcrVlConfig = serde_json::from_str(canonical_config_json()).unwrap();
        cfg.model_type = "paddleocr_vl_assistant".to_string();
        let r = cfg.validate();
        assert!(r.is_err(), "must reject unexpected model_type");
    }
}
