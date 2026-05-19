//! `Glm46VImageProcessor` port.
//!
//! Pipeline:
//!   1. Decode image to RGB.
//!   2. Dynamic resize so the smaller edge is at least `min_side` and
//!      the area is within `[shortest_edge, longest_edge]` (from
//!      `preprocessor_config.json`).  The resized H/W are also rounded
//!      to a multiple of `patch_size * spatial_merge_size` so the ViT
//!      can patchify without padding.
//!   3. ImageNet-style normalize: `(x - mean) / std`.
//!   4. Patchify: reshape `[3, H, W]` → `[T, P*P*3]` where T = (H/P)*(W/P).
//!      For glm_ocr the temporal axis is always 1 (single image), so
//!      `temporal_patch_size=2` reuses the same patch twice.
//!
//! Output: a single `Array` `[T, P*P*3]` ready to feed the ViT, plus
//! the spatial grid `(grid_h, grid_w)` so the merger / position-id
//! generator know how to lay out the tokens.

use mlx_rs::Array;

use crate::error::Error;

#[derive(Debug, Clone)]
pub struct PreprocessorConfig {
    pub patch_size: i32,
    pub temporal_patch_size: i32,
    pub merge_size: i32,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    /// Shortest-edge / longest-edge area bounds (in pixels²).
    pub shortest_edge: i32,
    pub longest_edge: i32,
}

impl Default for PreprocessorConfig {
    fn default() -> Self {
        Self {
            patch_size: 14,
            temporal_patch_size: 2,
            merge_size: 2,
            image_mean: [0.48145466, 0.4578275, 0.40821073],
            image_std: [0.26862954, 0.26130258, 0.27577711],
            shortest_edge: 12544,
            longest_edge: 9_633_792,
        }
    }
}

pub fn load_preprocessor_config(
    model_dir: impl AsRef<std::path::Path>,
) -> Result<PreprocessorConfig, Error> {
    let path = model_dir.as_ref().join("preprocessor_config.json");
    if !path.exists() {
        return Ok(PreprocessorConfig::default());
    }
    let bytes = std::fs::read(&path)
        .map_err(|e| Error::Io(format!("read {}: {e}", path.display())))?;
    let v: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|e| Error::Config(e.to_string()))?;
    let size = v.get("size").cloned().unwrap_or(serde_json::Value::Null);
    let shortest_edge = size
        .get("shortest_edge")
        .and_then(|x| x.as_i64())
        .unwrap_or(12544) as i32;
    let longest_edge = size
        .get("longest_edge")
        .and_then(|x| x.as_i64())
        .unwrap_or(9_633_792) as i32;
    let patch_size = v.get("patch_size").and_then(|x| x.as_i64()).unwrap_or(14) as i32;
    let temporal_patch_size = v
        .get("temporal_patch_size")
        .and_then(|x| x.as_i64())
        .unwrap_or(2) as i32;
    let merge_size = v.get("merge_size").and_then(|x| x.as_i64()).unwrap_or(2) as i32;
    let mean = read_triplet(&v, "image_mean", [0.48145466, 0.4578275, 0.40821073]);
    let std = read_triplet(&v, "image_std", [0.26862954, 0.26130258, 0.27577711]);
    Ok(PreprocessorConfig {
        patch_size,
        temporal_patch_size,
        merge_size,
        image_mean: mean,
        image_std: std,
        shortest_edge,
        longest_edge,
    })
}

fn read_triplet(v: &serde_json::Value, key: &str, default: [f32; 3]) -> [f32; 3] {
    v.get(key)
        .and_then(|x| x.as_array())
        .and_then(|a| {
            if a.len() != 3 {
                return None;
            }
            Some([
                a[0].as_f64().unwrap_or(default[0] as f64) as f32,
                a[1].as_f64().unwrap_or(default[1] as f64) as f32,
                a[2].as_f64().unwrap_or(default[2] as f64) as f32,
            ])
        })
        .unwrap_or(default)
}

/// Loaded image patches ready for the vision encoder.
pub struct PreprocessedImage {
    /// Shape `[T_patches, P*P*3]`, dtype f32.
    pub patches: Array,
    /// Patch grid (rows, cols) after spatial_merge.
    pub grid_h: i32,
    pub grid_w: i32,
}

/// Preprocess a single image file. Stub: produces a dummy zero tensor
/// matching the expected shape so downstream wiring can compile and
/// integration-test against shapes. Real pixel decoding + resize +
/// patchify lands as part of the full vision pipeline impl.
pub fn preprocess_image_stub(
    cfg: &PreprocessorConfig,
    grid_h: i32,
    grid_w: i32,
) -> Result<PreprocessedImage, Error> {
    let p = cfg.patch_size;
    let t = cfg.temporal_patch_size;
    let token_count = (grid_h * grid_w) * t;
    let patches = mlx_rs::Array::zeros::<f32>(&[token_count, p * p * 3])
        .map_err(Error::Mlx)?;
    Ok(PreprocessedImage {
        patches,
        grid_h,
        grid_w,
    })
}
