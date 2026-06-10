//! Minimal text chat with a Gemma 4 instruction-tuned model.
//!
//! Wraps the user prompt in the Gemma 4 turn markers (`<|turn>` / `<turn|>`)
//! so instruct-tuned checkpoints produce coherent answers. `chat_gemma4`
//! feeds the raw prompt, which only works for base models.
//!
//! Usage:
//!   cargo run --release -p gemma4-mlx --example chat_text -- \
//!     models/gemma-4-12B-it-bf16 "What is the capital of France?"

use std::{env, error::Error, path::PathBuf};

use gemma4_mlx::{
    load_model, load_tokenizer, mixed_cache::init_layered_cache, ud_loader::load_ud_mlx_4bit,
    Generate, EOS_TOKEN_IDS,
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
        .unwrap_or_else(|| PathBuf::from("models/gemma-4-12B-it-bf16"));
    let prompt = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "What is the capital of France? Answer in one sentence.".to_string());

    // Auto-detect: heterogeneous-quantized checkpoints (config.json has a
    // `quantization` block with per-tensor overrides) need the UD loader,
    // which derives per-tensor `bits` from each weight/scales shape ratio.
    // Plain bf16 checkpoints use the canonical loader.
    let is_quantized = {
        let cfg_path = model_dir.join("config.json");
        let raw: serde_json::Value = serde_json::from_reader(std::fs::File::open(&cfg_path)?)?;
        raw.get("quantization")
            .map(|q| q.is_object())
            .unwrap_or(false)
    };
    eprintln!(
        "[chat_text] loader: {}",
        if is_quantized { "load_ud_mlx_4bit (quantized)" } else { "load_model (bf16)" }
    );
    let mut model = if is_quantized {
        load_ud_mlx_4bit(&model_dir)?
    } else {
        load_model(&model_dir)?
    };
    let tokenizer = load_tokenizer(&model_dir)?;

    // Look up the Gemma 4 turn markers once. The model emits `<turn|>` (106)
    // at the end of its reply; EOS_TOKEN_IDS already covers it.
    let bos_id = tokenizer
        .token_to_id("<bos>")
        .ok_or("tokenizer missing <bos>")? as i32;
    let turn_start = tokenizer
        .token_to_id("<|turn>")
        .ok_or("tokenizer missing <|turn>")? as i32;
    let turn_end = tokenizer
        .token_to_id("<turn|>")
        .ok_or("tokenizer missing <turn|>")? as i32;
    let newline = tokenizer
        .token_to_id("\n")
        .ok_or("tokenizer missing newline")? as i32;

    // <bos><|turn>user\n{prompt}<turn|>\n<|turn>model\n
    let encode = |s: &str| -> Result<Vec<i32>, Box<dyn Error + Send + Sync>> {
        Ok(tokenizer
            .encode(s, false)?
            .get_ids()
            .iter()
            .map(|&i| i as i32)
            .collect())
    };
    let mut ids: Vec<i32> = vec![bos_id, turn_start];
    ids.extend(encode("user\n")?);
    ids.extend(encode(&prompt)?);
    ids.push(turn_end);
    ids.push(newline);
    ids.push(turn_start);
    ids.extend(encode("model\n")?);

    let prompt_tokens = Array::from_slice(&ids, &[ids.len() as i32]).index(NewAxis);

    let mut cache = init_layered_cache(&model);
    let generator = Generate::new(&mut model, &mut cache, 0.0, &prompt_tokens);

    let max_tokens: usize = std::env::var("CHAT_MAX_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);
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
        "\n[chat_text] emitted {} tok in {:.2}s = {:.2} tok/s",
        emitted, elapsed, emitted as f64 / elapsed.max(1e-6)
    );

    Ok(())
}
