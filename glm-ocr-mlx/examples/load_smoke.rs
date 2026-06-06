//! Smoke: load the real checkpoint and try one prefill step.
use std::env;
use std::path::PathBuf;

use anyhow::{anyhow, Result};
use glm_ocr_mlx::{load_from_path, splice_image_tokens};
use mlx_rs::Array;

fn main() -> Result<()> {
    let model_dir: PathBuf = env::args()
        .nth(1)
        .ok_or_else(|| anyhow!("usage: load_smoke <model_dir> [image_path]"))?
        .into();
    let image_path: Option<PathBuf> = env::args().nth(2).map(Into::into);

    eprintln!("loading {} …", model_dir.display());
    let mut model = load_from_path(&model_dir).map_err(|e| anyhow!(e.to_string()))?;
    eprintln!(
        "loaded: text_layers={} text_hidden={} vision_depth={} vision_hidden={} vocab={}",
        model.config.text_config.num_hidden_layers,
        model.config.text_config.hidden_size,
        model.config.vision_config.depth,
        model.config.vision_config.hidden_size,
        model.config.text_config.vocab_size,
    );

    // Synthesize an image if none supplied.
    let bytes = match image_path {
        Some(p) => std::fs::read(p)?,
        None => {
            let mut img = image::RgbImage::new(336, 336);
            for y in 0..336 {
                for x in 0..336 {
                    img.put_pixel(x, y, image::Rgb([(x % 256) as u8, (y % 256) as u8, ((x ^ y) % 256) as u8]));
                }
            }
            let mut buf = Vec::new();
            image::DynamicImage::ImageRgb8(img)
                .write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)?;
            buf
        }
    };

    eprintln!("encoding image …");
    let (soft, grid) = model.encode_image_bytes(&bytes).map_err(|e| anyhow!(e.to_string()))?;
    let sshape = soft.shape().to_vec();
    eprintln!("  soft tokens shape={:?} grid={:?}", sshape, grid);
    let n_soft = sshape[0] as usize;

    // Build a tiny prompt around the image placeholders.
    let preamble = "Extract:";
    let tail: Vec<i32> = model
        .tokenizer
        .encode(preamble, false)
        .map_err(|e| anyhow!("tok: {e}"))?
        .get_ids()
        .iter()
        .map(|&id| id as i32)
        .collect();
    let mut ids: Vec<i32> = Vec::with_capacity(2 + n_soft + tail.len());
    ids.push(model.special_tokens.image_start_id);
    for _ in 0..n_soft {
        ids.push(model.special_tokens.image_token_id);
    }
    ids.push(model.special_tokens.image_end_id);
    ids.extend_from_slice(&tail);
    eprintln!("input_ids len={}", ids.len());

    let position_ids = model
        .build_position_ids(&ids, &[grid])
        .map_err(|e| anyhow!(e.to_string()))?;
    let ids_arr = Array::from_slice(&ids, &[1, ids.len() as i32]);
    let text_embeds = model
        .llm
        .model
        .embed(&ids_arr)
        .map_err(|e| anyhow!(e.to_string()))?;
    let spliced =
        splice_image_tokens(&text_embeds, &ids, &soft, model.special_tokens.image_token_id)
            .map_err(|e| anyhow!(e.to_string()))?;

    let mut cache = model.new_cache();
    eprintln!("prefill …");
    let logits = model
        .llm
        .forward_last_logits_from_embeds(&spliced, &position_ids, &mut cache)
        .map_err(|e| anyhow!(e.to_string()))?;
    let argmax = mlx_rs::argmax_axis!(&logits, -1)?.as_dtype(mlx_rs::Dtype::Int32)?;
    mlx_rs::transforms::eval([&argmax])?;
    use mlx_rs::ops::indexing::IndexOp;
    let next = argmax.index(0).item::<i32>();
    eprintln!("argmax next token = {next}");
    if let Ok(s) = model.tokenizer.decode(&[next as u32], true) {
        eprintln!("  → {:?}", s);
    }
    Ok(())
}
