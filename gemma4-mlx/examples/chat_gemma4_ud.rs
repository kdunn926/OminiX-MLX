//! Chat with the UD-MLX-4bit Gemma4 variant (preserve-quant loader).
//!
//! Same flow as `chat_gemma4` but routes through `ud_loader::load_ud_mlx_4bit`
//! which keeps the heterogeneous Q4/Q6/Q8 weights packed (no dequant
//! at load time, ~15 GB resident vs ~50 GB for the bf16 model).
//!
//! Usage:
//!   cargo run --release -p gemma4-mlx --example chat_gemma4_ud -- \
//!     models/gemma4-26B-a4b-it-UD-MLX-4bit "Hello, how are you?"

use std::{env, error::Error, path::PathBuf};

use gemma4_mlx::{
    load_tokenizer, ud_loader::load_ud_mlx_4bit, Generate, KVCache, EOS_TOKEN_IDS,
};
use mlx_rs::{
    ops::indexing::{IndexOp, NewAxis},
    Array,
};

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let args: Vec<String> = env::args().collect();
    let model_dir = args
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/gemma4-26B-a4b-it-UD-MLX-4bit"));
    let prompt = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "Write a short haiku about Gemma 4.".to_string());

    let mut model = load_ud_mlx_4bit(&model_dir)?;
    let tokenizer = load_tokenizer(&model_dir)?;

    let encoding = tokenizer.encode(prompt, true)?;
    let prompt_tokens = Array::from(encoding.get_ids()).index(NewAxis);

    let mut cache = Vec::<KVCache>::new();
    let generator = Generate::new(&mut model, &mut cache, 0.0, &prompt_tokens);

    let max_tokens: usize = std::env::var("CHAT_MAX_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2048);
    let mut emitted = 0_usize;
    let t_start = std::time::Instant::now();
    for token in generator.take(max_tokens) {
        let token = token?;
        let token_id = token.item::<u32>();
        if EOS_TOKEN_IDS.contains(&token_id) {
            break;
        }
        print!("{}", tokenizer.decode(&[token_id], true)?);
        emitted += 1;
    }
    let elapsed = t_start.elapsed().as_secs_f64();
    eprintln!(
        "\n[chat_gemma4_ud] emitted {} tok in {:.2}s = {:.2} tok/s",
        emitted,
        elapsed,
        emitted as f64 / elapsed.max(1e-6)
    );

    Ok(())
}
