//! Minimal CLI that runs Qwen3-VL on (image, prompt) and prints the answer.
//!
//! Usage:
//!   cargo run --release -p qwen3-vl-mlx --example describe_image -- \
//!     models/Qwen3-VL-4B-Instruct-4bit \
//!     path/to/image.png \
//!     "Describe what you see."
//!
//! The model dir must contain `model.safetensors` (sharded or single-file),
//! `tokenizer.json`, and `config.json`.

use std::{env, error::Error, fs, path::PathBuf};

use qwen3_vl_mlx::{generate, load_model};

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let argv: Vec<String> = env::args().collect();
    let model_dir = argv
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/Qwen3-VL-4B-Instruct-4bit"));
    let image_path = argv
        .get(2)
        .map(PathBuf::from)
        .ok_or("missing image path; usage: describe_image <model_dir> <image_path> [prompt]")?;
    let prompt = argv
        .get(3)
        .cloned()
        .unwrap_or_else(|| "Describe this image in one sentence.".to_string());

    let max_tokens: usize = env::var("CHAT_MAX_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);
    let temperature: f32 = env::var("CHAT_TEMP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);

    eprintln!("[describe_image] loading model from {}", model_dir.display());
    let mut model = load_model(&model_dir)?;

    let image_bytes = fs::read(&image_path)?;
    let tokenizer_json = fs::read(model_dir.join("tokenizer.json"))?;

    eprintln!("[describe_image] image: {} ({} bytes)", image_path.display(), image_bytes.len());
    eprintln!("[describe_image] prompt: {prompt:?}");

    let t = std::time::Instant::now();
    let answer = generate(
        &mut model,
        &image_bytes,
        &prompt,
        &tokenizer_json,
        max_tokens,
        temperature,
    )?;
    let elapsed = t.elapsed().as_secs_f64();

    println!("{answer}");
    eprintln!("[describe_image] {} chars in {:.2}s", answer.len(), elapsed);

    Ok(())
}
