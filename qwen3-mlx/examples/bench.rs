//! Inference benchmark with separated prefill / TTFT / decode timing.
//!
//! Modes:
//!   bench  - measure load, prefill, TTFT, decode tok/s for a given prompt length
//!   parity - deterministic greedy (temp=0) generation, prints first N token IDs
//!
//! Usage:
//!   cargo run --release -p qwen3-mlx --example bench -- bench <model_dir> <prompt_len> <max_new>
//!   cargo run --release -p qwen3-mlx --example bench -- parity <model_dir> <num_tokens>

use std::env;
use std::time::Instant;

use mlx_rs::ops::indexing::{IndexOp, NewAxis};
use mlx_rs::transforms::eval;
use qwen3_mlx::{load_model, load_tokenizer, Error, Generate, KVCache};

const PARITY_PROMPT: &str = "The capital of France is";

fn main() -> Result<(), Error> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        usage(&args[0]);
        std::process::exit(2);
    }

    match args[1].as_str() {
        "bench" => {
            if args.len() < 5 {
                usage(&args[0]);
                std::process::exit(2);
            }
            let prompt_len: usize = args[3].parse().expect("prompt_len must be an integer");
            let max_new: usize = args[4].parse().expect("max_new must be an integer");
            run_bench(&args[2], prompt_len, max_new)
        }
        "parity" => {
            if args.len() < 4 {
                usage(&args[0]);
                std::process::exit(2);
            }
            let num_tokens: usize = args[3].parse().expect("num_tokens must be an integer");
            run_parity(&args[2], num_tokens)
        }
        other => {
            eprintln!("unknown mode: {other}");
            usage(&args[0]);
            std::process::exit(2);
        }
    }
}

fn usage(prog: &str) {
    eprintln!("Usage:");
    eprintln!("  {prog} bench  <model_dir> <prompt_len> <max_new>");
    eprintln!("  {prog} parity <model_dir> <num_tokens>");
}

fn run_bench(model_dir: &str, prompt_len: usize, max_new: usize) -> Result<(), Error> {
    let t_load = Instant::now();
    let tokenizer = load_tokenizer(model_dir)?;
    let load_tok_ms = t_load.elapsed().as_secs_f64() * 1000.0;

    let t_model = Instant::now();
    let mut model = load_model(model_dir)?;
    let load_model_ms = t_model.elapsed().as_secs_f64() * 1000.0;

    let prompt_ids = synthesize_prompt(&tokenizer, prompt_len)?;
    let actual_len = prompt_ids.len();
    let prompt_array = mlx_rs::Array::from(prompt_ids.as_slice()).index(NewAxis);

    let mut cache: Vec<Option<KVCache>> = Vec::new();
    let mut gen = Generate::<KVCache>::new(&mut model, &mut cache, 0.0, &prompt_array);

    let t_prefill = Instant::now();
    let first = gen.next().expect("first token")?;
    eval([&first])?;
    let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
    let ttft_ms = prefill_ms;

    let t_decode = Instant::now();
    let mut decoded = 1usize;
    let mut last_token = first;
    for _ in 1..max_new {
        let tok = match gen.next() {
            Some(Ok(t)) => t,
            Some(Err(e)) => return Err(e.into()),
            None => break,
        };
        last_token = tok;
        decoded += 1;
    }
    eval([&last_token])?;
    let decode_ms = t_decode.elapsed().as_secs_f64() * 1000.0;
    let decode_steps = decoded.saturating_sub(1).max(1);
    let decode_tok_s = decode_steps as f64 / (decode_ms / 1000.0);
    let total_ms = prefill_ms + decode_ms;

    println!(
        "model={} prompt_len={} max_new={} \
         load_model_ms={:.1} load_tok_ms={:.1} \
         prefill_ms={:.1} ttft_ms={:.1} \
         decode_steps={} decode_ms={:.1} decode_tok_s={:.1} \
         total_ms={:.1}",
        model_dir,
        actual_len,
        max_new,
        load_model_ms,
        load_tok_ms,
        prefill_ms,
        ttft_ms,
        decode_steps,
        decode_ms,
        decode_tok_s,
        total_ms,
    );

    Ok(())
}

fn run_parity(model_dir: &str, num_tokens: usize) -> Result<(), Error> {
    let tokenizer = load_tokenizer(model_dir)?;
    let mut model = load_model(model_dir)?;

    let encoding = tokenizer.encode(PARITY_PROMPT, true)?;
    let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
    let prompt_array = mlx_rs::Array::from(prompt_ids.as_slice()).index(NewAxis);

    let mut cache: Vec<Option<KVCache>> = Vec::new();
    let gen = Generate::<KVCache>::new(&mut model, &mut cache, 0.0, &prompt_array);

    let mut ids: Vec<u32> = Vec::with_capacity(num_tokens);
    for tok in gen.take(num_tokens) {
        let tok = tok?;
        ids.push(tok.item::<u32>());
    }

    println!("model={} prompt={:?}", model_dir, PARITY_PROMPT);
    println!("prompt_ids={:?}", prompt_ids);
    println!("greedy_ids={:?}", ids);
    let text = tokenizer.decode(&ids, true)?;
    println!("greedy_text={:?}", text);
    Ok(())
}

/// Build a prompt with approximately `target_len` tokens by repeating a stock paragraph
/// and trimming. Returns the encoded ids.
fn synthesize_prompt(
    tokenizer: &tokenizers::Tokenizer,
    target_len: usize,
) -> Result<Vec<u32>, Error> {
    let unit = "The quick brown fox jumps over the lazy dog. ";
    let mut text = String::new();
    while text.len() < target_len * 6 {
        text.push_str(unit);
    }
    let encoding = tokenizer.encode(text.as_str(), true)?;
    let mut ids: Vec<u32> = encoding.get_ids().to_vec();
    if ids.len() < target_len {
        while ids.len() < target_len {
            let pad = ids.last().copied().unwrap_or(0);
            ids.push(pad);
        }
    } else {
        ids.truncate(target_len);
    }
    Ok(ids)
}
