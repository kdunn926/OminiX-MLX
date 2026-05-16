//! Benchmark MTPLX speculative decoding on a Qwen3.6 target.
//!
//! Usage:
//!   cargo run --release -p mtplx-mlx --example bench_mtplx -- \
//!     --target /path/to/Qwen3.6-35B-A3B-4bit \
//!     [--prompt "..."] [--max-tokens 100] [--temp 0.0]
//!
//! With the stock checkpoint (MTP weights stripped) the session falls
//! back to autoregressive — the program prints a clear notice and runs
//! AR greedy generation. No panic on missing MTP weights.

use std::collections::HashSet;
use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};
use mtplx_mlx::{AcceptanceMode, MtplxSession, SpeculativeConfig};
use qwen3_6_mlx::{load_model, load_tokenizer};

const DEFAULT_PROMPT: &str = "The theory of general relativity";

#[derive(Debug)]
struct Args {
    target: PathBuf,
    prompt: String,
    max_tokens: usize,
    temp: f32,
    /// `Some(true)` → force speculative; `Some(false)` → force greedy;
    /// `None` → auto-pick based on temp.
    speculative: Option<bool>,
}

fn main() -> Result<()> {
    let args = parse_args()?;

    let tokenizer = load_tokenizer(&args.target)?;
    let encoding = tokenizer
        .encode(args.prompt.as_str(), false)
        .map_err(|e| anyhow!(e.to_string()))?;
    let prompt_ids: Vec<i32> = encoding.get_ids().iter().map(|&id| id as i32).collect();

    eprintln!("Loading target from {}...", args.target.display());
    let model = load_model(&args.target).map_err(|e| anyhow!("{e}"))?;

    let acceptance = match args.speculative {
        Some(true) => AcceptanceMode::Speculative,
        Some(false) => AcceptanceMode::Greedy,
        None => {
            if args.temp > 0.0 {
                AcceptanceMode::Speculative
            } else {
                AcceptanceMode::Greedy
            }
        }
    };
    eprintln!(
        "[mtplx] acceptance={:?} temp={}",
        acceptance, args.temp
    );
    let cfg = SpeculativeConfig {
        block_len: 4,
        max_tokens: args.max_tokens,
        temp: args.temp,
        acceptance,
    };
    let mut session = MtplxSession::new(model, cfg);

    if session.has_mtp_head() {
        eprintln!("[mtplx] MTP head present — drafting K=4 per cycle");
    } else {
        eprintln!(
            "[mtplx] target.mtp_head() is None — falling back to autoregressive"
        );
    }

    let eos = load_eos_tokens(&args.target)?;

    let (tokens, metrics) = session
        .generate(&prompt_ids, &eos)
        .map_err(|e| anyhow!("{e}"))?;

    // Decode generated tokens for display.
    let toks_u32: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
    let text = tokenizer
        .decode(&toks_u32, true)
        .map_err(|e| anyhow!(e.to_string()))?;

    println!("--- generated ---");
    println!("{}", text);
    println!("--- metrics ---");
    println!(
        "prefill_s={:.3} decode_s={:.3} decode_tok_per_s={:.2}",
        metrics.prefill_s,
        metrics.decode_s,
        metrics.decode_tok_per_s()
    );
    println!(
        "total_tokens={} mtp_cycles={} ar_fallback_steps={}",
        metrics.total_tokens, metrics.mtp_cycles, metrics.ar_fallback_steps
    );

    Ok(())
}

fn load_eos_tokens(model_dir: &std::path::Path) -> Result<HashSet<u32>> {
    let config_path = model_dir.join("config.json");
    let config: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&config_path)
            .with_context(|| format!("failed to read {}", config_path.display()))?,
    )?;
    let eos = match &config["eos_token_id"] {
        serde_json::Value::Array(ids) => ids
            .iter()
            .filter_map(|value| value.as_u64().map(|id| id as u32))
            .collect(),
        serde_json::Value::Number(id) => {
            let mut set = HashSet::new();
            set.insert(id.as_u64().unwrap_or(248046) as u32);
            set
        }
        _ => {
            let mut set = HashSet::new();
            set.insert(248046u32);
            set
        }
    };
    Ok(eos)
}

fn parse_args() -> Result<Args> {
    let mut args = std::env::args().skip(1);
    let mut target = PathBuf::from("models/Qwen3.6-35B-A3B-4bit");
    let mut prompt = DEFAULT_PROMPT.to_string();
    let mut max_tokens = 100usize;
    let mut temp = 0.0f32;
    let mut speculative: Option<bool> = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--target" => {
                target = PathBuf::from(
                    args.next()
                        .ok_or_else(|| anyhow!("--target requires a path"))?,
                )
            }
            "--prompt" => {
                prompt = args
                    .next()
                    .ok_or_else(|| anyhow!("--prompt requires text"))?
            }
            "--max-tokens" => {
                max_tokens = args
                    .next()
                    .ok_or_else(|| anyhow!("--max-tokens requires a value"))?
                    .parse()
                    .context("invalid --max-tokens value")?
            }
            "--temp" => {
                temp = args
                    .next()
                    .ok_or_else(|| anyhow!("--temp requires a value"))?
                    .parse()
                    .context("invalid --temp value")?
            }
            "--speculative" => speculative = Some(true),
            "--greedy" => speculative = Some(false),
            "--help" | "-h" => {
                println!(
                    "Usage: cargo run --release -p mtplx-mlx --example bench_mtplx -- --target /path/to/Qwen3.6-35B-A3B-4bit [--prompt \"...\"] [--max-tokens 100] [--temp 0.0]"
                );
                std::process::exit(0);
            }
            other => return Err(anyhow!("unknown argument: {other}")),
        }
    }

    Ok(Args {
        target,
        prompt,
        max_tokens,
        temp,
        speculative,
    })
}
