//! UD-MLX-4bit Gemma4 loader (preserve-quant path).
//!
//! The Unsloth "UD" (Unsloth Dynamic) MLX 4-bit quantization for Gemma4
//! ships with three differences from the canonical mlx-community
//! gemma-4-26B-A4B-it format:
//!
//! 1. **Key prefix**: tensors are named `language_model.model.*` rather
//!    than `model.language_model.*`. Resolved by `translate_key`.
//! 2. **MoE layout**: experts use the `switch_glu` submodule with three
//!    separate per-expert projections (`gate_proj`, `up_proj`,
//!    `down_proj`) rather than the canonical fused `gate_up_proj` +
//!    `down_proj`. `build_model_from_weights` now detects this layout
//!    and builds `SwitchGluExperts` (gather_qmm dispatch) instead of
//!    the fused `Experts`.
//! 3. **Heterogeneous quantization**: q/k/v/o @ Q8, mlp @ Q4, embed @ Q6
//!    — all sharing `group_size=64`. `make_mq_linear` /
//!    `make_mq_embedding` now derive per-tensor `bits` from the
//!    weight/scales shape ratio, so the single global
//!    `QuantizationConfig` is sufficient.
//!
//! Memory: ~15 GB resident (matches the on-disk size — no dequant).
//! Inference dispatches through native `quantized_matmul` for attention
//! and `gather_qmm` for MoE.

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::Array;
use mlx_rs_core::Error;

use crate::model::{
    build_model_from_weights, get_model_args, load_all_weights_unfiltered, Gemma4VlModel, Model,
};
use crate::vision::{load_embed_vision, load_vision_model};

/// `language_model.model.X` → `model.language_model.X`
/// `language_model.lm_head.X` → `lm_head.X`
fn translate_key(key: &str) -> String {
    if let Some(rest) = key.strip_prefix("language_model.model.") {
        format!("model.language_model.{rest}")
    } else if let Some(rest) = key.strip_prefix("language_model.lm_head.") {
        format!("lm_head.{rest}")
    } else {
        key.to_string()
    }
}

pub(crate) fn translate_keys(raw: HashMap<String, Array>) -> HashMap<String, Array> {
    raw.into_iter()
        .map(|(k, v)| (translate_key(&k), v))
        .collect()
}

/// Load a UD-MLX-4bit-format Gemma4 model. The returned `Model` keeps
/// every quantized tensor in its packed form (Q4/Q6/Q8 according to
/// the heterogeneous per-tensor config); attention runs through native
/// `quantized_matmul` and the MoE block runs through `gather_qmm` via
/// `SwitchGluExperts`.
///
/// Drop-in replacement for `load_model`: the returned `Model` exposes
/// the same API, can be passed to `Generate`, mtplx_chat, etc.
pub fn load_ud_mlx_4bit(model_dir: impl AsRef<Path>) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    let config = get_model_args(model_dir)?;
    let args = config.text_config.clone();

    let raw = load_all_weights_unfiltered(model_dir)?;
    let weights = translate_keys(raw);

    build_model_from_weights(&config, args, &weights)
}

/// UD-MLX-4bit key translation extended for the vision side of Gemma4-VL
/// checkpoints. Same text-side rewrites as `translate_key`, plus a `model.`
/// prefix prepended to the unprefixed `vision_tower.*` and `embed_vision.*`
/// tensors that `load_vision_model` / `load_embed_vision` expect under
/// `model.vision_tower.*` and `model.embed_vision.*`.
fn translate_key_vl(key: &str) -> String {
    if let Some(rest) = key.strip_prefix("language_model.model.") {
        format!("model.language_model.{rest}")
    } else if let Some(rest) = key.strip_prefix("language_model.lm_head.") {
        format!("lm_head.{rest}")
    } else if key.starts_with("vision_tower.") || key.starts_with("embed_vision.") {
        format!("model.{key}")
    } else {
        key.to_string()
    }
}

fn translate_keys_vl(raw: HashMap<String, Array>) -> HashMap<String, Array> {
    raw.into_iter()
        .map(|(k, v)| (translate_key_vl(&k), v))
        .collect()
}

/// Load a UD-MLX-4bit-format Gemma4-VL model — the multimodal counterpart
/// to [`load_ud_mlx_4bit`]. The UD checkpoint flattens the multimodal
/// naming (`language_model.model.*`, `vision_tower.*`, `embed_vision.*`)
/// instead of the canonical nested form (`model.language_model.*`,
/// `model.vision_tower.*`, `model.embed_vision.*`); this loader translates
/// the prefixes and otherwise delegates to the same `build_model_from_weights`
/// (text, with 4-bit dequant routing) and `load_vision_model` /
/// `load_embed_vision` (vision, BF16 with `.linear.weight` wrapping) that
/// the canonical loader uses.
pub fn load_ud_mlx_4bit_vl(model_dir: impl AsRef<Path>) -> Result<Gemma4VlModel, Error> {
    let model_dir = model_dir.as_ref();
    let config = get_model_args(model_dir)?;
    let vision_config = config
        .vision_config
        .clone()
        .ok_or_else(|| Error::Model("Not a Gemma4-VL model: missing vision_config".to_string()))?;

    let raw = load_all_weights_unfiltered(model_dir)?;
    let weights = translate_keys_vl(raw);

    let text = build_model_from_weights(&config, config.text_config.clone(), &weights)?;
    let vision = load_vision_model(&weights, &vision_config)?;
    let embed_vision = load_embed_vision(&weights, vision_config.rms_norm_eps, config.quantization.as_ref())?;

    Ok(Gemma4VlModel {
        text,
        vision,
        embed_vision,
        image_token_id: config.image_token_id.unwrap_or(258880),
        boi_token_id: config.boi_token_id.unwrap_or(255999),
        eoi_token_id: config.eoi_token_id.unwrap_or(258882),
        n_vision_tokens: config
            .vision_soft_tokens_per_image
            .max(vision_config.default_output_length as usize),
    })
}
