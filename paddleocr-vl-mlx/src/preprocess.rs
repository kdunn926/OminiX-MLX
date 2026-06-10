//! CPU image preprocessor (phase 5).
//!
//! Port of `PaddleOCRVLImageProcessor` in
//! `PaddlePaddle/PaddleOCR-VL-1.5/image_processing_paddleocr_vl.py`.
//!
//! Steps (matching the HF processor's `_preprocess` path for the canonical
//! image-only flow):
//!
//!   1. Decode bytes → `image::DynamicImage`. Convert to RGB.
//!   2. **smart_resize**: pick a target `(h_bar, w_bar)` such that both
//!      dims are multiples of `factor = patch_size × spatial_merge_size`
//!      (28 for PaddleOCR-VL-1.5), aspect ratio is preserved as closely
//!      as possible, and the total pixel count lands in
//!      `[min_pixels, max_pixels]`. Bicubic resampling.
//!   3. Rescale `pixel /= 255` → `[0, 1]`.
//!   4. Normalize `(pixel − 0.5) / 0.5` → `[-1, 1]` (the canonical config
//!      ships `mean = [0.5, 0.5, 0.5]`, `std = [0.5, 0.5, 0.5]`).
//!   5. Pack into a channels-first `(1, 3, h_bar, w_bar)` `f32` MLX
//!      array; emit `image_grid_thw = (1, h_bar / patch_size,
//!      w_bar / patch_size)` for downstream consumers.

use image::imageops::FilterType;
use mlx_rs::Array;
use mlx_rs_core::error::{Error, Result};

use crate::config::PaddleOcrVisionConfig;

/// Canonical processor parameters from
/// `preprocessor_config.json`. The values come from the 1.5 checkpoint;
/// callers can override per-request if needed (e.g. shrinking
/// `max_pixels` for memory-constrained hosts).
#[derive(Debug, Clone, Copy)]
pub struct PreprocessParams {
    /// Round-to multiple for both H and W after resize. Equals
    /// `patch_size × spatial_merge_size = 14 × 2 = 28` on the canonical
    /// 1.5 config.
    pub factor: i32,
    /// Lower bound on `h_bar × w_bar` after resize.
    pub min_pixels: i32,
    /// Upper bound on `h_bar × w_bar` after resize.
    pub max_pixels: i32,
    /// `patch_size` from the vision config (14 on 1.5). Used only to
    /// build the returned `image_grid_thw`.
    pub patch_size: i32,
    /// Per-channel mean (length 3).
    pub mean: [f32; 3],
    /// Per-channel std (length 3).
    pub std: [f32; 3],
}

impl PreprocessParams {
    /// Build the canonical 1.5 parameter set from the vision config.
    /// `mean = std = 0.5` matches the shipped `preprocessor_config.json`.
    pub fn from_vision_config(vc: &PaddleOcrVisionConfig) -> Self {
        let factor = vc.patch_size * vc.spatial_merge_size;
        // The shipped config sets min_pixels = 28 * 28 * 144 and
        // max_pixels = 28 * 28 * 1280 (the python `smart_resize`
        // defaults). Reproduce the same numbers here so the resize
        // selects the same target dims as the HF processor.
        let min_pixels = factor * factor * 144;
        let max_pixels = factor * factor * 1280;
        Self {
            factor,
            min_pixels,
            max_pixels,
            patch_size: vc.patch_size,
            mean: [0.5, 0.5, 0.5],
            std: [0.5, 0.5, 0.5],
        }
    }
}

/// Pixel tensor + patch grid produced by [`preprocess_image_bytes`].
pub struct PreprocessOutput {
    /// `(1, 3, h_bar, w_bar)` `f32` MLX array, channels-first per the HF
    /// convention. The Vision encoder's `forward` transposes to BHWC
    /// internally before its Conv2d call.
    pub pixel_values: Array,
    /// `(T, H, W)` patch grid passed downstream to the position-id
    /// builder, the vision encoder's positional embedding lookup, and
    /// the Projector. `T = 1` always for images.
    pub image_grid_thw: (i32, i32, i32),
}

/// Decode → smart-resize → rescale → normalize → pack.
///
/// Returns both the pixel tensor and the `image_grid_thw` descriptor the
/// rest of the pipeline keys off.
pub fn preprocess_image_bytes(
    bytes: &[u8],
    params: &PreprocessParams,
) -> Result<PreprocessOutput> {
    let img = image::load_from_memory(bytes).map_err(|e| {
        Error::Model(format!("preprocess: decode failed: {e}"))
    })?;
    let rgb = img.to_rgb8();
    let (w0, h0) = rgb.dimensions();
    let (h_bar, w_bar) = smart_resize(
        h0 as i32,
        w0 as i32,
        params.factor,
        params.min_pixels,
        params.max_pixels,
    )?;
    let resized = image::imageops::resize(
        &rgb,
        w_bar as u32,
        h_bar as u32,
        FilterType::CatmullRom,
    );
    let h = h_bar as usize;
    let w = w_bar as usize;
    // Pack as channels-first (1, 3, H, W) so callers and the rest of
    // the crate see the HF convention.
    let mut packed = vec![0.0_f32; 3 * h * w];
    let chan_stride = h * w;
    let inv = [
        1.0 / 255.0 / params.std[0],
        1.0 / 255.0 / params.std[1],
        1.0 / 255.0 / params.std[2],
    ];
    let offset = [
        -params.mean[0] / params.std[0],
        -params.mean[1] / params.std[1],
        -params.mean[2] / params.std[2],
    ];
    for y in 0..h {
        for x in 0..w {
            let px = resized.get_pixel(x as u32, y as u32).0;
            let idx = y * w + x;
            packed[idx] = (px[0] as f32) * inv[0] + offset[0];
            packed[chan_stride + idx] = (px[1] as f32) * inv[1] + offset[1];
            packed[2 * chan_stride + idx] = (px[2] as f32) * inv[2] + offset[2];
        }
    }
    let pixel_values = Array::from_slice(&packed, &[1, 3, h_bar, w_bar]);
    let grid = (1, h_bar / params.patch_size, w_bar / params.patch_size);
    Ok(PreprocessOutput {
        pixel_values,
        image_grid_thw: grid,
    })
}

/// Pure host-side `smart_resize` from
/// `image_processing_paddleocr_vl.py:128`.
///
/// Returns `(h_bar, w_bar)` such that:
///   - Both dims are multiples of `factor`.
///   - `h_bar * w_bar ∈ [min_pixels, max_pixels]`.
///   - Aspect ratio is preserved as closely as possible.
pub fn smart_resize(
    mut height: i32,
    mut width: i32,
    factor: i32,
    min_pixels: i32,
    max_pixels: i32,
) -> Result<(i32, i32)> {
    if factor <= 0 || min_pixels <= 0 || max_pixels <= 0 || min_pixels > max_pixels {
        return Err(Error::InvalidConfig(format!(
            "smart_resize: invalid params (factor={factor}, min_pixels={min_pixels}, \
             max_pixels={max_pixels})"
        )));
    }
    // Bump degenerate-tiny dims up to `factor` so the round-up math has
    // something to work with — matches the python `if height < factor`
    // and `if width < factor` branches.
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
        return Err(Error::InvalidConfig(format!(
            "smart_resize: absolute aspect ratio must be < 200, got {aspect}"
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
    use mlx_rs::Dtype;

    #[test]
    fn smart_resize_rounds_to_factor() {
        // factor=28; arbitrary 500x300 image well inside the pixel
        // budget should be rounded to the nearest multiple of 28.
        let (h, w) = smart_resize(300, 500, 28, 28 * 28, 28 * 28 * 1280).unwrap();
        assert_eq!(h % 28, 0);
        assert_eq!(w % 28, 0);
        assert!(h > 0 && w > 0);
        // Aspect should be preserved within one factor step in either
        // dim — 500/300 ≈ 1.667.
        let aspect_in = 500.0_f32 / 300.0;
        let aspect_out = w as f32 / h as f32;
        assert!(
            (aspect_in - aspect_out).abs() < 0.2,
            "aspect drifted from {aspect_in} to {aspect_out}",
        );
    }

    #[test]
    fn smart_resize_clamps_to_max_pixels() {
        // Huge input; total should land at-or-below max_pixels.
        let max_pixels = 28 * 28 * 100;
        let (h, w) = smart_resize(4000, 4000, 28, 28 * 28, max_pixels).unwrap();
        assert!(
            h * w <= max_pixels,
            "h*w = {} must be ≤ max_pixels = {max_pixels}",
            h * w
        );
    }

    #[test]
    fn smart_resize_inflates_to_min_pixels() {
        // Tiny input; total should land at-or-above min_pixels.
        let min_pixels = 28 * 28 * 50;
        let (h, w) = smart_resize(40, 40, 28, min_pixels, 28 * 28 * 1280).unwrap();
        assert!(
            h * w >= min_pixels,
            "h*w = {} must be ≥ min_pixels = {min_pixels}",
            h * w
        );
    }

    #[test]
    fn smart_resize_rejects_extreme_aspect() {
        // 1000:1 aspect — should be rejected.
        let r = smart_resize(1, 1000, 28, 28 * 28, 28 * 28 * 1280);
        assert!(r.is_err());
    }

    #[test]
    fn preprocess_emits_correct_shape_and_grid() {
        // Synthesise a 200×300 RGB image, encode it as PNG, then
        // preprocess. The output shape must be (1, 3, h_bar, w_bar)
        // for some valid (h_bar, w_bar) tuple, and the grid must be
        // (1, h_bar/14, w_bar/14).
        let w = 300;
        let h = 200;
        let mut buf = image::RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let r = (x % 256) as u8;
                let g = (y % 256) as u8;
                let b = ((x ^ y) % 256) as u8;
                buf.put_pixel(x, y, image::Rgb([r, g, b]));
            }
        }
        let mut png = Vec::new();
        {
            let mut cursor = std::io::Cursor::new(&mut png);
            image::DynamicImage::ImageRgb8(buf)
                .write_to(&mut cursor, image::ImageFormat::Png)
                .unwrap();
        }

        // Fabricate vision config matching the canonical 1.5 layout.
        let vc: PaddleOcrVisionConfig = serde_json::from_str(
            r#"{
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
            }"#,
        )
        .unwrap();
        let params = PreprocessParams::from_vision_config(&vc);
        let out = preprocess_image_bytes(&png, &params).unwrap();
        let s = out.pixel_values.shape();
        assert_eq!(s.len(), 4);
        assert_eq!(s[0], 1);
        assert_eq!(s[1], 3);
        let h_bar = s[2];
        let w_bar = s[3];
        assert_eq!(h_bar % params.factor, 0);
        assert_eq!(w_bar % params.factor, 0);
        let (t, gh, gw) = out.image_grid_thw;
        assert_eq!(t, 1);
        assert_eq!(gh, h_bar / params.patch_size);
        assert_eq!(gw, w_bar / params.patch_size);
        // Range check: with mean/std = 0.5, pixels land in [-1, 1].
        let absmax = out
            .pixel_values
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap()
            .item::<f32>();
        assert!(
            absmax <= 1.0_f32 + 1e-3,
            "pixel range must be ≤ 1 after normalization; got |max| = {absmax}"
        );
    }
}
