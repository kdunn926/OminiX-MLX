//! Multi-Token Prediction (MTP) head for Qwen3.6.
//!
//! Qwen3.6 declares `mtp_num_hidden_layers` and `mtp_use_dedicated_embeddings`
//! in its config (`text_config.mtp_num_hidden_layers: 1` for the 35B-A3B
//! checkpoint). The published `mlx-community` checkpoints strip the MTP
//! weights — see `mlx_lm.models.qwen3_5.Qwen3_5MoEModel.sanitize`, which
//! drops every key containing `"mtp."` — so most loads will find no
//! weights and return `None`.
//!
//! There is no canonical MTP head reference in `mlx_lm` or in the
//! `qwen3_next` companion module (both drop `mtp.*` keys). For Qwen3.6's
//! `mtp_num_hidden_layers == 1` setting we model the head as a single
//! transformer block:
//!
//!   - `enorm` (RMSNorm over previous-token embedding)
//!   - `hnorm` (RMSNorm over previous hidden state)
//!   - `eh_proj` (Linear: concat([h, e_prev]) -> hidden)
//!   - one transformer-style block (input_layernorm + self_attn + post_attention_layernorm + mlp)
//!   - `shared_head.norm` + (optional) `shared_head.head` projection.
//!
//! This module currently stops at parsing the config flags and probing
//! the safetensors for any `mtp.*` / `model.mtp_layers.*` prefix. If
//! none is present (the common case), `load_mtp_head` returns `Ok(None)`
//! and `mtplx-mlx` falls back to autoregressive decoding.
//!
//! If/when a checkpoint ships the weights, fill in the block construction
//! in `load_mtp_head_from_weights` — the public API (`MtpHead::forward`)
//! is intentionally narrow so the caller doesn't need to know the
//! internal layer shape.

use std::collections::HashMap;

use mlx_rs::{error::Exception, Array};

use crate::config::ModelArgs;

/// MTP head for drafting `num_layers` extra speculative tokens per cycle.
///
/// For Qwen3.6's `mtp_num_hidden_layers == 1` config this is a single
/// fused projection + transformer block. The internal representation is
/// intentionally opaque to callers — only `forward` is part of the API.
pub struct MtpHead {
    /// Number of MTP layers (matches `mtp_num_hidden_layers`).
    pub num_layers: i32,
    /// Hidden size (matches the host model).
    pub hidden_size: i32,
    /// Whether this head carries its own embedding table; if false the
    /// caller should pass embeddings from the host model.
    pub use_dedicated_embeddings: bool,
    /// Names of weight keys that were detected in the checkpoint. Empty
    /// when no MTP weights were present (stub mode).
    pub detected_weight_keys: Vec<String>,
}

impl MtpHead {
    /// Draft `k` next-token logits given the last hidden state and the
    /// previously-emitted token's embedding.
    ///
    /// This is a stub: until a checkpoint ships MTP weights we have no
    /// learned parameters to evaluate. Callers (`mtplx-mlx`) check for
    /// the stub case via `is_stub()` and fall back to AR.
    pub fn forward(
        &mut self,
        _hidden: &Array,
        _prev_token_emb: &Array,
    ) -> Result<Array, Exception> {
        Err(Exception::custom(
            "MtpHead::forward called on a stub head — no MTP weights were loaded.",
        ))
    }

    /// True when no real MTP weights were found in the checkpoint.
    pub fn is_stub(&self) -> bool {
        self.detected_weight_keys.is_empty()
    }
}

/// Attempt to load the MTP head from a flat weight map.
///
/// Returns `Ok(None)` (not an error) when:
///   * the config does not declare any MTP layers, OR
///   * no `mtp.*` / `model.mtp_layers.*` keys are present in the weights.
///
/// Most released Qwen3.6 checkpoints strip the MTP block, so the common
/// path is `Ok(None)`.
pub fn load_mtp_head(
    weights: &HashMap<String, Array>,
    args: &ModelArgs,
) -> Result<Option<MtpHead>, mlx_rs_core::error::Error> {
    let num_layers = args.mtp_num_hidden_layers();
    if num_layers <= 0 {
        return Ok(None);
    }

    let detected: Vec<String> = weights
        .keys()
        .filter(|k| is_mtp_key(k))
        .cloned()
        .collect();

    if detected.is_empty() {
        // The checkpoint omitted MTP weights — graceful no-op.
        return Ok(None);
    }

    // TODO: when a checkpoint ships these weights, decode them into a
    // concrete transformer block here. For now we surface that they were
    // detected so the caller can decide what to do.
    Ok(Some(MtpHead {
        num_layers,
        hidden_size: args.text_config.hidden_size,
        use_dedicated_embeddings: args.mtp_use_dedicated_embeddings(),
        detected_weight_keys: detected,
    }))
}

/// True when a weight key belongs to the MTP block. Matches the suffix
/// conventions used across the Qwen3.5 / Qwen3-Next / DeepSeek MTP forks:
///   * `mtp.<...>` (top-level)
///   * `model.mtp.<...>` / `model.mtp_layers.<i>.<...>`
///   * `language_model.model.mtp.<...>` (VLM-prefixed)
///   * `<prefix>.mtp_layers.<i>.<...>`
fn is_mtp_key(k: &str) -> bool {
    k.contains(".mtp.") || k.contains(".mtp_layers.") || k.starts_with("mtp.")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_mtp_key_variants() {
        assert!(is_mtp_key("mtp.layers.0.input_layernorm.weight"));
        assert!(is_mtp_key("model.mtp.0.eh_proj.weight"));
        assert!(is_mtp_key("model.mtp_layers.0.shared_head.norm.weight"));
        assert!(is_mtp_key(
            "language_model.model.mtp_layers.0.self_attn.q_proj.weight"
        ));
        assert!(!is_mtp_key("model.layers.0.self_attn.q_proj.weight"));
        assert!(!is_mtp_key("model.embed_tokens.weight"));
    }
}
