//! End-to-end smoke: load a real PaddleOCR-VL-1.5 checkpoint, run the full
//! pipeline (preprocess → vision tower → projector → text decoder), and
//! print one forward step's argmax token.
//!
//! Pixel-correct output requires the bilinear pos-embed interpolation
//! follow-up; for now we ask the smoke to use a canonical-size image
//! (384×384) so the vision encoder doesn't reject it.
//!
//! Usage:
//!   cargo run --release -p paddleocr-vl-mlx --example load_and_forward \
//!     -- <model_dir> [image_path]

use std::env;
use std::path::PathBuf;

use anyhow::{anyhow, Result};
use mlx_rs::{ops, Array, Dtype};
use paddleocr_vl_mlx::{loader::splice_image_tokens, load_from_path};

fn main() -> Result<()> {
    let model_dir: PathBuf = env::args()
        .nth(1)
        .ok_or_else(|| anyhow!("usage: load_and_forward <model_dir> [image_path]"))?
        .into();
    let image_path: Option<PathBuf> = env::args().nth(2).map(Into::into);

    eprintln!("loading {} …", model_dir.display());
    let mut model = load_from_path(&model_dir).map_err(|e| anyhow!(e.to_string()))?;
    eprintln!(
        "loaded: text_layers={}, text_hidden={}, vision_layers={}, vision_hidden={}, vocab={}, image_token_id={}",
        model.config.num_hidden_layers,
        model.config.hidden_size,
        model.config.vision_config.num_hidden_layers,
        model.config.vision_config.hidden_size,
        model.config.vocab_size.unwrap_or(0),
        model.config.image_token_id,
    );
    eprintln!(
        "special tokens: image_start={} image_end={} image_token={}",
        model.special_tokens.image_start_id,
        model.special_tokens.image_end_id,
        model.special_tokens.image_token_id,
    );

    // Generate or read an image. The vision encoder currently asserts the
    // canonical 384×384 (729-position pos table); make sure our input
    // produces 27×27 patches.
    let pixel_grid = 384_u32;
    let png_bytes = match image_path {
        Some(p) => {
            eprintln!("reading {}", p.display());
            std::fs::read(&p)?
        }
        None => {
            eprintln!("generating a {pixel_grid}×{pixel_grid} synthetic image …");
            let mut img = image::RgbImage::new(pixel_grid, pixel_grid);
            for y in 0..pixel_grid {
                for x in 0..pixel_grid {
                    let r = (x % 256) as u8;
                    let g = (y % 256) as u8;
                    let b = ((x ^ y) % 256) as u8;
                    img.put_pixel(x, y, image::Rgb([r, g, b]));
                }
            }
            let mut buf = Vec::new();
            image::DynamicImage::ImageRgb8(img).write_to(
                &mut std::io::Cursor::new(&mut buf),
                image::ImageFormat::Png,
            )?;
            buf
        }
    };

    eprintln!("encoding image → soft tokens …");
    let (soft_tokens, image_grid_thw) = model
        .encode_image_bytes(&png_bytes)
        .map_err(|e| anyhow!(e.to_string()))?;
    let soft_shape: Vec<i32> = soft_tokens.shape().to_vec();
    eprintln!(
        "  vision soft tokens: shape={:?} grid={:?}",
        soft_shape, image_grid_thw
    );
    let n_soft = soft_shape[0] as usize;

    // Build a minimal prompt: <|IMAGE_START|> + <|IMAGE_PLACEHOLDER|> ×
    // n_soft + <|IMAGE_END|> + a one-word instruction encoded with the
    // tokenizer.
    let preamble = "Extract the text:";
    let tail: Vec<i32> = model
        .tokenizer
        .encode(preamble, false)
        .map_err(|e| anyhow!("tokenize preamble: {e}"))?
        .get_ids()
        .iter()
        .map(|&id| id as i32)
        .collect();
    let mut input_ids: Vec<i32> = Vec::with_capacity(2 + n_soft + tail.len());
    input_ids.push(model.special_tokens.image_start_id as i32);
    for _ in 0..n_soft {
        input_ids.push(model.special_tokens.image_token_id as i32);
    }
    input_ids.push(model.special_tokens.image_end_id as i32);
    input_ids.extend_from_slice(&tail);
    eprintln!(
        "input_ids len = {} (n_soft={} preamble={})",
        input_ids.len(),
        n_soft,
        tail.len()
    );

    // Build position ids and embeddings; splice soft tokens.
    let position_ids = model
        .build_position_ids(&input_ids, &[image_grid_thw])
        .map_err(|e| anyhow!(e.to_string()))?;
    let ids_arr = Array::from_slice(&input_ids, &[1, input_ids.len() as i32]);
    let text_embeds = model
        .llm
        .model
        .embed(&ids_arr)
        .map_err(|e| anyhow!(e.to_string()))?;
    let spliced = splice_image_tokens(
        &text_embeds,
        &input_ids,
        &soft_tokens,
        model.config.image_token_id,
    )
    .map_err(|e| anyhow!(e.to_string()))?;

    let mut cache = model.new_cache();
    eprintln!("running forward (prefill all positions) …");
    let logits = model
        .llm
        .forward_last_logits_from_embeds(&spliced, &position_ids, &mut cache)
        .map_err(|e| anyhow!(e.to_string()))?;
    let argmax = mlx_rs::argmax_axis!(&logits, -1)?
        .as_dtype(Dtype::Int32)?;
    let _ = ops::ones::<f32>(&[1])?; // touch ops to keep stream live
    mlx_rs::transforms::eval([&argmax])?;
    use mlx_rs::ops::indexing::IndexOp;
    let next_id = argmax.index(0).item::<i32>();
    eprintln!("next-token argmax = {next_id}");
    if let Ok(decoded) = model.tokenizer.decode(&[next_id as u32], true) {
        eprintln!("  decodes to: {:?}", decoded);
    }
    Ok(())
}
