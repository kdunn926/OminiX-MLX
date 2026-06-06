//! `Glm46VImageProcessor` port.
//!
//! Pipeline (matching the HF reference's `_preprocess` for a single image):
//!
//!   1. Decode bytes → RGB `image::DynamicImage`.
//!   2. **smart_resize** to a target `(h_bar, w_bar)` such that both dims
//!      are multiples of `patch_size × spatial_merge_size` and the total
//!      pixel count lands in `[shortest_edge, longest_edge]`. Bicubic.
//!   3. Rescale `pixel /= 255` → `[0, 1]`.
//!   4. Normalize per-channel with ImageNet `mean` / `std`.
//!   5. Patchify: pack each (gh, gw) patch as a flattened
//!      `(C=3, t=temporal_patch_size, P, P)` cube → length `3*t*P*P`.
//!      The output matrix is `(grid_t * gh * gw, 3*t*P*P)` where
//!      `grid_t = 1` for stills. The temporal dim is **embedded inside
//!      each row's patch features** (matching the Qwen2-VL convention
//!      Glm46V's Conv3d-style patch_embed consumes); the same spatial
//!      patch is replicated across the `t` slots within a row so the
//!      Conv3d kernel sees identical frames.
//!
//! Output: an `Array` of shape `(grid_t * gh * gw, 3 * t * P * P)` ready
//! for the ViT's patch-embed Linear, plus an
//! `image_grid_thw = (T_raw, H_grid, W_grid)` descriptor where
//! `T_raw = temporal_patch_size`. The grid drives the position-id
//! builder and the post-encoder spatial merger.

use image::imageops::FilterType;
use mlx_rs::Array;

use crate::error::Error;

#[derive(Debug, Clone)]
pub struct PreprocessorConfig {
    pub patch_size: i32,
    pub temporal_patch_size: i32,
    pub merge_size: i32,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    /// Pixel-area lower bound (defaults to `28*28*16 = 12544`).
    pub shortest_edge: i32,
    /// Pixel-area upper bound (defaults to `28*28*12288 = 9_633_792`).
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
    /// Shape `(grid_t * grid_h * grid_w, 3 * t * P * P)`, dtype f32, where
    /// `grid_t = 1` for stills. Each row is one patch flattened in
    /// `(C, t, P, P)` order so the patch-embed Linear can ingest it as if
    /// it were the input to a Conv3d kernel.
    pub patches: Array,
    /// `(T_raw, grid_h, grid_w)` — `T_raw = temporal_patch_size` for stills.
    /// This is the descriptor the position-id builder + spatial merger consume.
    pub image_grid_thw: (i32, i32, i32),
}

/// Decode → smart-resize → normalize → patchify pipeline.
pub fn preprocess_image_bytes(
    bytes: &[u8],
    cfg: &PreprocessorConfig,
) -> Result<PreprocessedImage, Error> {
    let img = image::load_from_memory(bytes)
        .map_err(|e| Error::Image(format!("decode failed: {e}")))?;
    let rgb = img.to_rgb8();
    let (w0, h0) = rgb.dimensions();
    let factor = cfg.patch_size * cfg.merge_size;
    let (h_bar, w_bar) = smart_resize(
        h0 as i32,
        w0 as i32,
        factor,
        cfg.shortest_edge,
        cfg.longest_edge,
    )?;
    let resized = image::imageops::resize(
        &rgb,
        w_bar as u32,
        h_bar as u32,
        FilterType::CatmullRom,
    );

    // Channels-first normalize: shape (3, H, W), f32, ImageNet stats.
    let h = h_bar as usize;
    let w = w_bar as usize;
    let chan_stride = h * w;
    let mut chw = vec![0.0_f32; 3 * chan_stride];
    let inv = [
        1.0 / 255.0 / cfg.image_std[0],
        1.0 / 255.0 / cfg.image_std[1],
        1.0 / 255.0 / cfg.image_std[2],
    ];
    let offset = [
        -cfg.image_mean[0] / cfg.image_std[0],
        -cfg.image_mean[1] / cfg.image_std[1],
        -cfg.image_mean[2] / cfg.image_std[2],
    ];
    for y in 0..h {
        for x in 0..w {
            let px = resized.get_pixel(x as u32, y as u32).0;
            let idx = y * w + x;
            chw[idx] = (px[0] as f32) * inv[0] + offset[0];
            chw[chan_stride + idx] = (px[1] as f32) * inv[1] + offset[1];
            chw[2 * chan_stride + idx] = (px[2] as f32) * inv[2] + offset[2];
        }
    }

    // Patchify: (3, H, W) → (grid_t * grid_h * grid_w, 3 * t * P * P)
    // with `grid_t = 1` for stills. Each row holds one patch flattened in
    // (C, t, P, P) order: outer axis = channel, then t-slot, then row,
    // then col. For still images the t-axis is filled by replicating the
    // spatial patch across `t` slots so the Conv3d kernel (which sees
    // `t = temporal_patch_size` frames simultaneously) gets identical
    // frames — matches the HF Glm46VImageProcessor `_preprocess`.
    let p = cfg.patch_size as usize;
    let grid_h = (h_bar / cfg.patch_size) as usize;
    let grid_w = (w_bar / cfg.patch_size) as usize;
    let t_raw = cfg.temporal_patch_size as usize;
    let grid_t: usize = 1;
    let patch_len = 3 * t_raw * p * p;
    let n_patches = grid_t * grid_h * grid_w;
    let mut patches = vec![0.0_f32; n_patches * patch_len];

    // Row layout: (t, P, P, c) — MLX channels-last Conv3d weight has axis
    // order (out_ch, kT, kH, kW, in_ch), so the input row must flatten in
    // (t, h, w, c) to match. For each (gh, gw) we emit `t_raw` slots of
    // identical (P, P, c) blocks (still-image temporal replication).
    for gh in 0..grid_h {
        for gw in 0..grid_w {
            let row = gh * grid_w + gw;
            let dst_start = row * patch_len;
            // (P, P, 3) block: outer y, then x, then channel.
            let mut block = vec![0.0_f32; p * p * 3];
            for py in 0..p {
                for px in 0..p {
                    let src_y = gh * p + py;
                    let src_x = gw * p + px;
                    for c in 0..3 {
                        let src = c * chan_stride + src_y * w + src_x;
                        block[(py * p + px) * 3 + c] = chw[src];
                    }
                }
            }
            let block_len = p * p * 3;
            for t in 0..t_raw {
                let dst = dst_start + t * block_len;
                patches[dst..dst + block_len].copy_from_slice(&block);
            }
        }
    }

    let patches_arr = Array::from_slice(
        &patches,
        &[n_patches as i32, patch_len as i32],
    );
    Ok(PreprocessedImage {
        patches: patches_arr,
        image_grid_thw: (cfg.temporal_patch_size, grid_h as i32, grid_w as i32),
    })
}

/// Pure host-side smart-resize (port of `smart_resize` in
/// `image_processing_glm46v.py`).
pub fn smart_resize(
    mut height: i32,
    mut width: i32,
    factor: i32,
    min_pixels: i32,
    max_pixels: i32,
) -> Result<(i32, i32), Error> {
    if factor <= 0 || min_pixels <= 0 || max_pixels <= 0 || min_pixels > max_pixels {
        return Err(Error::Image(format!(
            "smart_resize: invalid params (factor={factor}, min_pixels={min_pixels}, \
             max_pixels={max_pixels})"
        )));
    }
    if height < factor {
        width = ((width as f32 * factor as f32) / height.max(1) as f32).round() as i32;
        height = factor;
    }
    if width < factor {
        height = ((height as f32 * factor as f32) / width.max(1) as f32).round() as i32;
        width = factor;
    }
    let aspect = (height.max(width) as f32) / (height.min(width).max(1) as f32);
    if aspect > 200.0 {
        return Err(Error::Image(format!(
            "smart_resize: aspect ratio must be < 200, got {aspect}"
        )));
    }
    let round_to_factor = |v: f32| -> i32 {
        ((v / factor as f32).round() as i32).max(1) * factor
    };
    let floor_to_factor = |v: f32| -> i32 {
        ((v / factor as f32).floor() as i32).max(1) * factor
    };
    let ceil_to_factor = |v: f32| -> i32 {
        ((v / factor as f32).ceil() as i32).max(1) * factor
    };
    let mut h_bar = round_to_factor(height as f32);
    let mut w_bar = round_to_factor(width as f32);
    let total = h_bar * w_bar;
    if total > max_pixels {
        let beta = ((height as f32 * width as f32) / max_pixels as f32).sqrt();
        h_bar = floor_to_factor(height as f32 / beta);
        w_bar = floor_to_factor(width as f32 / beta);
    } else if total < min_pixels {
        let beta = (min_pixels as f32 / (height as f32 * width as f32)).sqrt();
        h_bar = ceil_to_factor(height as f32 * beta);
        w_bar = ceil_to_factor(width as f32 * beta);
    }
    Ok((h_bar, w_bar))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smart_resize_rounds_to_factor() {
        let (h, w) = smart_resize(300, 500, 28, 28 * 28, 28 * 28 * 1280).unwrap();
        assert_eq!(h % 28, 0);
        assert_eq!(w % 28, 0);
    }

    #[test]
    fn smart_resize_clamps_to_max_pixels() {
        let max = 28 * 28 * 100;
        let (h, w) = smart_resize(4000, 4000, 28, 28 * 28, max).unwrap();
        assert!(h * w <= max);
    }

    #[test]
    fn smart_resize_inflates_to_min_pixels() {
        let min = 28 * 28 * 50;
        let (h, w) = smart_resize(40, 40, 28, min, 28 * 28 * 1280).unwrap();
        assert!(h * w >= min);
    }

    #[test]
    fn preprocess_produces_expected_patch_shape() {
        // (grid_t * gh * gw, 3 * t * P * P) with grid_t = 1 for stills.
        let w = 300;
        let h = 200;
        let mut buf = image::RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                buf.put_pixel(x, y, image::Rgb([(x % 256) as u8, (y % 256) as u8, 0]));
            }
        }
        let mut png = Vec::new();
        image::DynamicImage::ImageRgb8(buf)
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();
        let cfg = PreprocessorConfig::default();
        let out = preprocess_image_bytes(&png, &cfg).unwrap();
        let s = out.patches.shape();
        assert_eq!(s.len(), 2);
        let (t_raw, grid_h, grid_w) = out.image_grid_thw;
        assert_eq!(t_raw, cfg.temporal_patch_size);
        assert_eq!(s[0], grid_h * grid_w); // grid_t = 1 for stills
        assert_eq!(s[1], 3 * t_raw * cfg.patch_size * cfg.patch_size);
    }

    #[test]
    fn duplicated_temporal_slots_inside_row_are_identical() {
        // Each row is laid out as t consecutive (P, P, 3) blocks; for a
        // still image the t blocks must be byte-identical.
        let w = 56;
        let h = 56;
        let mut buf = image::RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                buf.put_pixel(x, y, image::Rgb([(x + y) as u8, x as u8, y as u8]));
            }
        }
        let mut png = Vec::new();
        image::DynamicImage::ImageRgb8(buf)
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();
        let cfg = PreprocessorConfig::default();
        let out = preprocess_image_bytes(&png, &cfg).unwrap();
        let (_, gh, gw) = out.image_grid_thw;
        let t_raw = cfg.temporal_patch_size as usize;
        let p = cfg.patch_size as usize;
        let block_len = p * p * 3;
        let patch_len = t_raw * block_len;
        let f32_patches = out
            .patches
            .as_dtype(mlx_rs::Dtype::Float32)
            .unwrap();
        mlx_rs::transforms::eval([&f32_patches]).unwrap();
        let slice = f32_patches.try_as_slice::<f32>().unwrap();
        for row in 0..(gh * gw) as usize {
            let row_start = row * patch_len;
            let slot0 = row_start;
            for t in 1..t_raw {
                let slott = row_start + t * block_len;
                for k in 0..block_len {
                    assert!(
                        (slice[slot0 + k] - slice[slott + k]).abs() < 1e-6,
                        "row={row} t={t} k={k} differs",
                    );
                }
            }
        }
    }

}
