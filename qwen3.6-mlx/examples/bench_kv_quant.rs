//! KV-cache quantization benchmark.
//!
//! Compares standard fp16 KV cache vs mixed-precision QuantizedKVCache (K=q8, V=q4)
//! across a range of context window sizes on a loaded Qwen3.6 model.
//!
//! Usage:
//!   cargo run --example bench_kv_quant --release -- <model_dir> [decode_tokens]
//!
//! Example:
//!   cargo run --example bench_kv_quant --release -- \
//!     ../OminiX-MLX/models/Qwen3.6-35B-A3B-4bit 50

use std::collections::HashSet;
use mlx_rs::{
    ops::indexing::{IndexOp, NewAxis},
    Array,
};
use qwen3_6_mlx::{load_model, load_tokenizer, Generate};

struct BenchResult {
    ctx_tokens: usize,
    ttft_secs: f64,
    decode_tps: f64,
    n_generated: usize,
}

fn run_bench(
    model_dir: &str,
    ctx_tokens: usize,
    decode_tokens: usize,
    quantized: bool,
    eos_tokens: &HashSet<u32>,
) -> anyhow::Result<BenchResult> {
    let tokenizer = load_tokenizer(model_dir)?;
    let mut model = load_model(model_dir)?;

    // Build a synthetic prompt of `ctx_tokens` tokens
    let phrase = "The quick brown fox jumps over the lazy dog. ";
    let mut prompt_text = String::new();
    while prompt_text.len() < ctx_tokens * 6 {
        prompt_text.push_str(phrase);
    }

    let encoding = tokenizer
        .encode(prompt_text.as_str(), false)
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    let ids = encoding.get_ids();
    let ids = &ids[..ctx_tokens.min(ids.len())];
    let prompt = Array::from(ids);
    let prompt = prompt.index(NewAxis);

    let decode_start = std::time::Instant::now();
    let gen = if quantized {
        Generate::new_quantized_kv(&mut model, 0.0, &prompt)
    } else {
        Generate::new(&mut model, 0.0, &prompt)
    };

    let mut tokens = Vec::new();
    let mut ttft: Option<f64> = None;

    for result in gen.take(decode_tokens) {
        let tok = result?;
        if ttft.is_none() {
            ttft = Some(decode_start.elapsed().as_secs_f64());
        }
        let id = tok.item::<u32>();
        tokens.push(id);
        if eos_tokens.contains(&id) {
            break;
        }
    }

    let total_secs = decode_start.elapsed().as_secs_f64();
    let decode_tps = tokens.len() as f64 / total_secs;

    Ok(BenchResult {
        ctx_tokens,
        ttft_secs: ttft.unwrap_or(0.0),
        decode_tps,
        n_generated: tokens.len(),
    })
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let model_dir = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "./models/Qwen3.6-35B-A3B-4bit".to_string());
    let decode_tokens: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(50);

    eprintln!("Model: {model_dir}");
    eprintln!("Decode tokens per run: {decode_tokens}");
    eprintln!();

    // Load EOS tokens once
    let eos_tokens: HashSet<u32> = {
        let cfg: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(std::path::Path::new(&model_dir).join("config.json"))?,
        )?;
        match &cfg["eos_token_id"] {
            serde_json::Value::Array(ids) => ids
                .iter()
                .filter_map(|v| v.as_u64().map(|n| n as u32))
                .collect(),
            serde_json::Value::Number(n) => {
                std::iter::once(n.as_u64().unwrap_or(248044) as u32).collect()
            }
            _ => std::iter::once(248044u32).collect(),
        }
    };

    // Theoretical KV-cache memory savings table (27B dense, fp16 vs q8k/q4v)
    // Per token: fp16 = 64 layers * 2 * 4 heads * 256 head_dim * 2 bytes = 256 KB
    //            q8k/q4v ≈ 104 KB (59% savings)
    eprintln!("Theoretical KV-cache memory (27B dense):");
    eprintln!("{:<8} {:>10} {:>12} {:>10}", "ctx(K)", "fp16(GB)", "q8k/q4v(GB)", "savings%");
    for k in [1, 4, 8, 16, 32, 64, 128, 256] {
        let fp16 = k as f64 * 1024.0 * 256.0 / 1e9;
        let qkv = k as f64 * 1024.0 * 104.0 / 1e9;
        let pct = (1.0 - qkv / fp16) * 100.0;
        eprintln!("{:<8} {:>10.2} {:>12.2} {:>9.0}%", k, fp16, qkv, pct);
    }
    eprintln!();

    // Context sizes to test (tokens)
    let ctx_sizes = [256usize, 1024, 4096];

    println!(
        "{:<10} {:<8} {:>10} {:>14} {:>10}",
        "cache", "ctx(tok)", "TTFT(s)", "decode(tok/s)", "generated"
    );
    println!("{}", "-".repeat(58));

    for &ctx in &ctx_sizes {
        for &quantized in &[false, true] {
            let label = if quantized { "q8k/q4v" } else { "fp16" };
            match run_bench(&model_dir, ctx, decode_tokens, quantized, &eos_tokens) {
                Ok(r) => {
                    println!(
                        "{:<10} {:<8} {:>10.3} {:>14.1} {:>10}",
                        label, r.ctx_tokens, r.ttft_secs, r.decode_tps, r.n_generated
                    );
                }
                Err(e) => {
                    println!("{:<10} {:<8} ERROR: {e}", label, ctx);
                }
            }
        }
    }

    Ok(())
}
