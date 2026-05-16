//! MTPLX-Optimized-Speed target loader for Gemma4.
//!
//! The MTPLX preconversion of `google/gemma-4-31B-it` ships with one
//! deviation from `mlx-community`'s convention that gemma4-mlx's standard
//! loader expects: tensors are named `language_model.model.*` rather than
//! `model.language_model.*`. We translate this at load time and hand the
//! result to `build_model_from_weights`, which now natively understands
//! the `(.weight U32, .scales BF16, .biases BF16)` triplets and builds
//! `MaybeQuantized::Quantized` modules so inference runs native
//! `quantized_matmul` (no dequantization to BF16).
//!
//! Memory: weights stay in their packed Q4 form (~13GB resident for the
//! 27B target, vs ~54GB for the BF16 dequant path).

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::Array;
use mlx_rs_core::Error;

use crate::model::{build_model_from_weights, get_model_args, load_all_weights_unfiltered, Model};

/// MTPLX → canonical key prefix translation. Identity on already-canonical
/// keys.
fn translate_prefix(key: &str) -> String {
    if let Some(rest) = key.strip_prefix("language_model.model.") {
        format!("model.language_model.{rest}")
    } else if let Some(rest) = key.strip_prefix("language_model.lm_head.") {
        format!("lm_head.{rest}")
    } else {
        key.to_string()
    }
}

fn translate_keys(raw: HashMap<String, Array>) -> HashMap<String, Array> {
    raw.into_iter()
        .map(|(k, v)| (translate_prefix(&k), v))
        .collect()
}

/// Load the MTPLX-Optimized-Speed Gemma4 target as a native quantized model
/// (no dequantization at load time). The returned `Model` is identical in
/// API to one produced by `gemma4_mlx::load_model`, but its quantized
/// linears stay packed and use `quantized_matmul` at inference time.
pub fn load_mtplx_target(model_dir: impl AsRef<Path>) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    let config = get_model_args(model_dir)?;
    let args = config.text_config.clone();

    let raw = load_all_weights_unfiltered(model_dir)?;
    let weights = translate_keys(raw);

    build_model_from_weights(&config, args, &weights)
}
