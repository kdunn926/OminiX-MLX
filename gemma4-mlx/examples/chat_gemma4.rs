use std::{env, error::Error, path::PathBuf};

use gemma4_mlx::{load_model, load_tokenizer, Generate, KVCache, EOS_TOKEN_IDS};
use mlx_rs::{
    ops::indexing::{IndexOp, NewAxis},
    Array,
};

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let args: Vec<String> = env::args().collect();
    let model_dir = args
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/gemma-4-26B-A4B-it"));
    let prompt = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "Write a short haiku about Gemma 4.".to_string());

    let mut model = load_model(&model_dir)?;
    let tokenizer = load_tokenizer(&model_dir)?;

    let encoding = tokenizer.encode(prompt, true)?;
    let prompt_tokens = Array::from(encoding.get_ids()).index(NewAxis);

    let mut cache = Vec::<KVCache>::new();
    let generator = Generate::new(&mut model, &mut cache, 0.0, &prompt_tokens);

    let max_tokens: usize = std::env::var("CHAT_MAX_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2048);
    let mut emitted = 0usize;
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
        "\n[chat_gemma4] emitted {} tok in {:.2}s = {:.2} tok/s",
        emitted, elapsed, emitted as f64 / elapsed.max(1e-6)
    );

    Ok(())
}
