//! MTPLX-Optimized-Speed target loader for Gemma4.
//!
//! The MTPLX preconversion of `google/gemma-4-31B-it` uses the same tensor
//! layout as the Unsloth UD-MLX-4bit checkpoints: `language_model.model.*`
//! key prefixes and packed `(.weight U32, .scales BF16, .biases BF16)`
//! quantized triplets. Loading is therefore identical to
//! [`crate::ud_loader::load_ud_mlx_4bit`]; this alias exists so MTPLX call
//! sites keep a descriptive entry point.
//!
//! Memory: weights stay in their packed Q4 form (~13GB resident for the
//! 27B target, vs ~54GB for the BF16 dequant path).

use std::path::Path;

use mlx_rs_core::Error;

use crate::model::Model;

/// Load the MTPLX-Optimized-Speed Gemma4 target as a native quantized model
/// (no dequantization at load time). The returned `Model` is identical in
/// API to one produced by `gemma4_mlx::load_model`, but its quantized
/// linears stay packed and use `quantized_matmul` at inference time.
pub fn load_mtplx_target(model_dir: impl AsRef<Path>) -> Result<Model, Error> {
    crate::ud_loader::load_ud_mlx_4bit(model_dir)
}
