//! Parser for speculators-format EAGLE-3 draft checkpoints (`config.json`).
//!
//! Format reference: vllm-project/speculators `Eagle3SpeculatorConfig`
//! (`speculators_model_type: "eagle3"`), e.g.
//! `RedHatAI/gemma-4-26B-A4B-it-speculator.eagle3`.

use std::path::Path;

use mlx_rs_core::error::Error;
use serde::Deserialize;

/// Llama-style single decoder layer hyperparameters for the draft.
#[derive(Debug, Clone, Deserialize)]
pub struct TransformerLayerConfig {
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub head_dim: i32,
    pub rms_norm_eps: f32,
    /// Full (target) vocabulary size — the draft's `embed_tokens` covers it.
    pub vocab_size: i32,
    #[serde(default)]
    pub rope_parameters: Option<RopeParameters>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub rope_type: Option<String>,
}

fn default_rope_theta() -> f32 {
    10000.0
}

#[derive(Debug, Clone, Deserialize)]
pub struct SpeculatorsVerifier {
    pub name_or_path: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ProposalMethod {
    #[serde(default)]
    pub speculative_tokens: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SpeculatorsConfig {
    pub algorithm: String,
    pub verifier: SpeculatorsVerifier,
    #[serde(default)]
    pub proposal_methods: Vec<ProposalMethod>,
}

/// Top-level speculators EAGLE-3 config.
#[derive(Debug, Clone, Deserialize)]
pub struct Eagle3Config {
    pub speculators_model_type: String,
    pub draft_vocab_size: i32,
    /// Target layer ids whose hidden states feed the `fc` fusion layer.
    /// vLLM convention: hidden state at the *input* of layer `i`
    /// (== residual stream output of layer `i-1`).
    pub eagle_aux_hidden_state_layer_ids: Vec<usize>,
    #[serde(default)]
    pub norm_before_fc: bool,
    #[serde(default)]
    pub norm_before_residual: bool,
    /// Target hidden size; `null` means same as the draft `hidden_size`.
    #[serde(default)]
    pub target_hidden_size: Option<i32>,
    pub transformer_layer_config: TransformerLayerConfig,
    pub speculators_config: SpeculatorsConfig,
}

impl Eagle3Config {
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self, Error> {
        let path = model_dir.as_ref().join("config.json");
        let text = std::fs::read_to_string(&path)?;
        let config: Eagle3Config = serde_json::from_str(&text)?;
        if config.speculators_model_type != "eagle3" {
            return Err(Error::InvalidConfig(format!(
                "{} is not an EAGLE-3 speculators checkpoint (speculators_model_type={})",
                path.display(),
                config.speculators_model_type
            )));
        }
        if config.speculators_config.algorithm != "eagle3" {
            return Err(Error::InvalidConfig(format!(
                "unsupported speculators algorithm {:?} (expected \"eagle3\")",
                config.speculators_config.algorithm
            )));
        }
        Ok(config)
    }

    pub fn rope_theta(&self) -> f32 {
        self.transformer_layer_config
            .rope_parameters
            .as_ref()
            .map(|r| r.rope_theta)
            .unwrap_or_else(default_rope_theta)
    }

    /// Default draft chain length recommended by the checkpoint.
    pub fn default_speculative_tokens(&self) -> usize {
        self.speculators_config
            .proposal_methods
            .first()
            .and_then(|m| m.speculative_tokens)
            .unwrap_or(3)
    }

    /// The verifier (target) model this draft was trained against,
    /// e.g. `google/gemma-4-26B-A4B-it`.
    pub fn verifier_name(&self) -> &str {
        &self.speculators_config.verifier.name_or_path
    }
}
