//! Smoke test for the encoder-free unified-vision embedder
//! ([`gemma4_mlx::unified_vision`]).
//!
//! Loads a Gemma 4 12B `gemma4_unified_vision` checkpoint, builds the
//! `UnifiedVisionEmbedder` + the shared `EmbedVision` projection from raw
//! weights, then runs a forward with **synthetic** zero-valued patches and
//! a small position grid. Verifies the output shape — does NOT verify
//! pixel-correct embedding values (Phase 2 image preprocessor + a real
//! image are required for that, and the upstream sglang PR has only been
//! merged this week so the reference path itself is still being shaken
//! down).
//!
//! Usage:
//!   cargo run --release -p gemma4-mlx --example unified_vision_smoke \
//!     -- ./models/gemma-4-12B-it-4bit

use std::env;
use std::path::PathBuf;

use anyhow::{anyhow, Result};
use gemma4_mlx::unified_vision::{load_unified_4bit_vl, Gemma4UnifiedVlModel};
use mlx_rs::{ops, Array, Dtype};

fn main() -> Result<()> {
    let model_dir: PathBuf = env::args()
        .nth(1)
        .ok_or_else(|| anyhow!("usage: unified_vision_smoke <model_dir>"))?
        .into();

    eprintln!("loading {} …", model_dir.display());
    let Gemma4UnifiedVlModel {
        vision: mut embedder,
        embed_vision: mut projector,
        n_vision_tokens,
        image_token_id,
        ..
    } = load_unified_4bit_vl(&model_dir).map_err(|e| anyhow!(e.to_string()))?;
    eprintln!(
        "loaded: image_token_id={image_token_id} n_vision_tokens={n_vision_tokens}"
    );

    // Patch layout: 14×14 = 196 model-sized patches, each
    // model_patch_size×model_patch_size×3 = 6912 raw pixel values.
    let n_patches = 196_i32;
    let patch_pixels = 48 * 48 * 3;
    let pixel_values = ops::zeros::<f32>(&[1, n_patches, patch_pixels]).unwrap();

    // Synthetic position grid: row-major 14×14 over (X, Y); no padding.
    let mut coords = Vec::with_capacity((n_patches as usize) * 2);
    for y in 0..14_i32 {
        for x in 0..14_i32 {
            coords.push(x);
            coords.push(y);
        }
    }
    let pos_ids = Array::from_slice(&coords, &[1, n_patches, 2]);

    eprintln!(
        "forward: pixel_values={:?} pos_ids={:?}",
        pixel_values.shape(),
        pos_ids.shape()
    );
    let hidden = embedder
        .forward(&pixel_values, &pos_ids)
        .map_err(|e| anyhow!(e.to_string()))?;
    eprintln!(
        "after vision_embedder: shape={:?} dtype={:?}",
        hidden.shape(),
        hidden.dtype()
    );

    let projected = projector
        .forward(&hidden)
        .map_err(|e| anyhow!(e.to_string()))?;
    eprintln!(
        "after embed_vision:    shape={:?} dtype={:?}",
        projected.shape(),
        projected.dtype()
    );

    let proj_shape = projected.shape();
    assert_eq!(proj_shape.len(), 3, "expected 3D projection output");
    assert_eq!(proj_shape[0], 1);
    assert_eq!(proj_shape[1], n_patches);
    assert!(
        matches!(projected.dtype(), Dtype::Bfloat16 | Dtype::Float16 | Dtype::Float32),
        "unexpected output dtype {:?}",
        projected.dtype()
    );

    eprintln!("smoke ok — shape matches expectations");
    Ok(())
}
