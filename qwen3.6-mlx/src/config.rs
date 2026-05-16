use serde::Deserialize;

#[derive(Debug, Clone, Deserialize, Default)]
pub struct QuantizationConfig {
    #[serde(default = "default_group_size")]
    pub group_size: i32,
    #[serde(default = "default_bits")]
    pub bits: i32,
}

fn default_group_size() -> i32 {
    64
}
fn default_bits() -> i32 {
    4
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f32,
    #[serde(default)]
    pub rope_type: Option<String>,
}

fn default_rope_theta() -> f32 {
    10_000_000.0
}
fn default_partial_rotary_factor() -> f32 {
    0.25
}

#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub head_dim: i32,
    pub rms_norm_eps: f32,
    pub vocab_size: i32,
    #[serde(default = "default_max_pos")]
    pub max_position_embeddings: i32,
    pub layer_types: Vec<String>,

    // Linear attention (DeltaNet) config
    pub linear_num_key_heads: i32,
    pub linear_num_value_heads: i32,
    pub linear_key_head_dim: i32,
    pub linear_value_head_dim: i32,
    pub linear_conv_kernel_dim: i32,

    // RoPE
    pub rope_parameters: RopeParameters,

    // Output gate for full attention
    #[serde(default)]
    pub attn_output_gate: bool,

    // MoE config — absent for dense models
    #[serde(default)]
    pub num_experts: Option<i32>,
    #[serde(default)]
    pub num_experts_per_tok: Option<i32>,
    #[serde(default)]
    pub moe_intermediate_size: Option<i32>,
    #[serde(default)]
    pub shared_expert_intermediate_size: Option<i32>,

    // Quantization (sometimes inside text_config)
    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,

    // Multi-Token Prediction (MTP) head config — used by mtplx-mlx.
    // Qwen3.6-35B-A3B declares `mtp_num_hidden_layers: 1` and
    // `mtp_use_dedicated_embeddings: false`. Most released checkpoints
    // strip the MTP weights, so loaders treat these as best-effort hints.
    #[serde(default)]
    pub mtp_num_hidden_layers: Option<i32>,
    #[serde(default)]
    pub mtp_use_dedicated_embeddings: Option<bool>,
}

fn default_max_pos() -> i32 {
    262144
}

impl TextConfig {
    /// Returns true if this config describes a Mixture-of-Experts model.
    pub fn is_moe(&self) -> bool {
        self.num_experts.unwrap_or(0) > 0
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub text_config: TextConfig,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
    // VL fields (absent for text-only models)
    #[serde(default)]
    pub vision_config: Option<VisionConfig>,
    #[serde(default)]
    pub image_token_id: Option<i32>,
    #[serde(default)]
    pub vision_start_token_id: Option<i32>,
    #[serde(default)]
    pub vision_end_token_id: Option<i32>,
    #[serde(default)]
    pub language_model_only: Option<bool>,

    // Top-level mirrors of MTP config (sometimes Qwen ships these at the
    // top level instead of / in addition to inside `text_config`).
    #[serde(default)]
    pub mtp_num_hidden_layers: Option<i32>,
    #[serde(default)]
    pub mtp_use_dedicated_embeddings: Option<bool>,
}

impl ModelArgs {
    /// Get the effective quantization config (top-level takes precedence)
    pub fn quantization(&self) -> Option<&QuantizationConfig> {
        self.quantization
            .as_ref()
            .or(self.text_config.quantization.as_ref())
    }

    /// Number of MTP head layers (0 if disabled). Checks top-level first,
    /// then `text_config`.
    pub fn mtp_num_hidden_layers(&self) -> i32 {
        self.mtp_num_hidden_layers
            .or(self.text_config.mtp_num_hidden_layers)
            .unwrap_or(0)
    }

    /// Whether the MTP head has its own dedicated token embeddings.
    pub fn mtp_use_dedicated_embeddings(&self) -> bool {
        self.mtp_use_dedicated_embeddings
            .or(self.text_config.mtp_use_dedicated_embeddings)
            .unwrap_or(false)
    }
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct VisionConfig {
    #[serde(default = "default_vis_depth")]
    pub depth: usize,
    #[serde(default = "default_vis_hidden")]
    pub hidden_size: i32,
    #[serde(default = "default_vis_heads")]
    pub num_heads: i32,
    #[serde(default = "default_vis_intermediate")]
    pub intermediate_size: i32,
    #[serde(default = "default_vis_patch_size")]
    pub patch_size: i32,
    #[serde(default = "default_temporal_patch_size")]
    pub temporal_patch_size: i32,
    #[serde(default = "default_in_channels")]
    pub in_channels: i32,
    #[serde(default = "default_out_hidden_size")]
    pub out_hidden_size: i32,
    #[serde(default = "default_num_pos_emb")]
    pub num_position_embeddings: i32,
    #[serde(default = "default_spatial_merge_size")]
    pub spatial_merge_size: i32,
    #[serde(default = "default_deepstack_indexes")]
    pub deepstack_visual_indexes: Vec<usize>,
}

fn default_vis_depth() -> usize {
    24
}
fn default_vis_hidden() -> i32 {
    1152
}
fn default_vis_heads() -> i32 {
    16
}
fn default_vis_intermediate() -> i32 {
    4096
}
fn default_vis_patch_size() -> i32 {
    16
}
fn default_temporal_patch_size() -> i32 {
    2
}
fn default_in_channels() -> i32 {
    3
}
fn default_out_hidden_size() -> i32 {
    5120
}
fn default_num_pos_emb() -> i32 {
    2304
}
fn default_spatial_merge_size() -> i32 {
    2
}
fn default_deepstack_indexes() -> Vec<usize> {
    vec![5, 11, 17]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_text_config(extras: &str) -> TextConfig {
        let json = format!(
            r#"{{
                "hidden_size": 5120,
                "num_hidden_layers": 4,
                "num_attention_heads": 24,
                "num_key_value_heads": 4,
                "head_dim": 256,
                "rms_norm_eps": 1e-6,
                "vocab_size": 248320,
                "layer_types": ["linear_attention","linear_attention","linear_attention","full_attention"],
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 48,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
                "rope_parameters": {{"rope_theta": 10000000.0, "partial_rotary_factor": 0.25}}
                {}
            }}"#,
            if extras.is_empty() {
                String::new()
            } else {
                format!(", {}", extras)
            }
        );
        serde_json::from_str(&json).expect("minimal TextConfig parse failed")
    }

    #[test]
    fn is_moe_false_for_dense_config() {
        let tc = minimal_text_config("");
        assert!(!tc.is_moe(), "dense config should not be MoE");
        assert_eq!(tc.num_experts, None);
        assert_eq!(tc.num_experts_per_tok, None);
    }

    #[test]
    fn is_moe_true_for_moe_config() {
        let tc = minimal_text_config(
            r#""num_experts": 256, "num_experts_per_tok": 8, "moe_intermediate_size": 2048"#,
        );
        assert!(tc.is_moe(), "MoE config should report is_moe() true");
        assert_eq!(tc.num_experts, Some(256));
        assert_eq!(tc.num_experts_per_tok, Some(8));
    }

    #[test]
    fn is_moe_false_when_num_experts_is_zero() {
        let tc = minimal_text_config(r#""num_experts": 0"#);
        assert!(!tc.is_moe(), "num_experts=0 should not be MoE");
    }

    #[test]
    fn quantization_top_level_takes_precedence() {
        let json = r#"{
            "text_config": {
                "hidden_size": 5120, "num_hidden_layers": 4, "num_attention_heads": 24,
                "num_key_value_heads": 4, "head_dim": 256, "rms_norm_eps": 1e-6,
                "vocab_size": 248320,
                "layer_types": ["linear_attention","linear_attention","linear_attention","full_attention"],
                "linear_num_key_heads": 16, "linear_num_value_heads": 48,
                "linear_key_head_dim": 128, "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
                "rope_parameters": {"rope_theta": 10000000.0, "partial_rotary_factor": 0.25},
                "quantization": {"group_size": 64, "bits": 4}
            },
            "quantization": {"group_size": 32, "bits": 8}
        }"#;
        let args: ModelArgs = serde_json::from_str(json).unwrap();
        assert_eq!(
            args.quantization().unwrap().bits,
            8,
            "top-level quant should win"
        );
    }
}
