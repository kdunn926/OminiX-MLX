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

    // Detect target family from config.json so we can route Gemma4 down
    // its AR path (no MTP head, paired-model speculation deferred) and
    // Qwen3.6 through the existing MtplxSession.
    let is_gemma4 = {
        let cfg_path = args.target.join("config.json");
        if cfg_path.exists() {
            let cfg: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(&cfg_path)?)?;
            cfg.get("model_type")
                .and_then(|v| v.as_str())
                .map(|s| s.starts_with("gemma"))
                .unwrap_or(false)
        } else {
            false
        }
    };
    if is_gemma4 {
        return run_gemma4(&args);
    }
    let tokenizer = load_tokenizer(&args.target)?;
    // If --prompt points to a .json file, treat it as an OpenAI-format
    // chat fixture and render to Qwen ChatML (matches the hermes-gateway
    // fixtures in OminiX-API/tests/fixtures).
    let prompt_text = if std::path::Path::new(&args.prompt).exists()
        && args.prompt.ends_with(".json")
    {
        let bytes = std::fs::read(&args.prompt)?;
        let v: serde_json::Value = serde_json::from_slice(&bytes)?;
        let msgs = v
            .get("messages")
            .and_then(|m| m.as_array())
            .ok_or_else(|| anyhow!("fixture has no messages array"))?;
        let mut out = String::new();
        for m in msgs {
            let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("user");
            let content = match m.get("content") {
                Some(serde_json::Value::String(s)) => s.clone(),
                Some(serde_json::Value::Array(parts)) => parts
                    .iter()
                    .filter_map(|p| p.get("text").and_then(|t| t.as_str()).map(String::from))
                    .collect::<Vec<_>>()
                    .join(""),
                _ => String::new(),
            };
            out.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
        }
        out.push_str("<|im_start|>assistant\n");
        out
    } else {
        args.prompt.clone()
    };
    let encoding = tokenizer
        .encode(prompt_text.as_str(), false)
        .map_err(|e| anyhow!(e.to_string()))?;
    let prompt_ids: Vec<i32> = encoding.get_ids().iter().map(|&id| id as i32).collect();
    eprintln!("[mtplx] prompt: {} tokens", prompt_ids.len());

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
        block_len: std::env::var("MTPLX_BLOCK_LEN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4),
        max_tokens: args.max_tokens,
        temp: args.temp,
        acceptance,
    };
    let block_len = cfg.block_len;
    let mut session = MtplxSession::new(model, cfg);

    if session.has_mtp_head() {
        eprintln!("[mtplx] MTP head present — drafting K={} per cycle", block_len);
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
        "total_tokens={} mtp_cycles={} ar_fallback_steps={} mtp_drafted={} mtp_accepted={} acceptance_rate={:.3}",
        metrics.total_tokens,
        metrics.mtp_cycles,
        metrics.ar_fallback_steps,
        metrics.mtp_drafted,
        metrics.mtp_accepted,
        metrics.mtp_acceptance_rate(),
    );

    Ok(())
}

/// Gemma4 path: AR-only generation through `gemma4_mlx::Generate`, with
/// the same prefill/decode/peak-mem reporting as the qwen3.6 MtplxSession
/// path. Paired-model speculation (target + assistant) is a follow-up.
fn run_gemma4(args: &Args) -> Result<()> {
    use mlx_rs::ops::indexing::{IndexOp, NewAxis};
    use std::time::Instant;

    eprintln!("Loading Gemma4 target from {}...", args.target.display());
    let tokenizer = gemma4_mlx::load_tokenizer(&args.target)
        .map_err(|e| anyhow!("{e}"))?;
    // Try the MTPLX target loader first (paired layout); fall back to
    // plain load_model for the single-dir 26B-A4B layout.
    let load_start = Instant::now();
    let mut model = if args.target.join("mtplx_artifact.json").exists() {
        gemma4_mlx::mtplx_target::load_mtplx_target(&args.target)
            .map_err(|e| anyhow!("{e}"))?
    } else {
        gemma4_mlx::load_model(&args.target).map_err(|e| anyhow!("{e}"))?
    };
    eprintln!(
        "  Gemma4 target loaded in {:.1}s",
        load_start.elapsed().as_secs_f32()
    );

    // Render prompt (string or chat-fixture JSON).
    let prompt_text = render_prompt_arg(&args.prompt)?;
    let encoding = tokenizer
        .encode(prompt_text.as_str(), false)
        .map_err(|e| anyhow!(e.to_string()))?;
    let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
    eprintln!("[mtplx-gemma4] prompt: {} tokens", prompt_ids.len());

    let prompt_arr =
        mlx_rs::Array::from_slice(&prompt_ids, &[1, prompt_ids.len() as i32]);
    let prompt_arr = prompt_arr.index(NewAxis); // [1, 1, T]
    let _ = prompt_arr; // unused, gemma4_mlx::Generate handles its own indexing

    let eos = load_eos_tokens(&args.target)?;
    let num_slots = *model.model.kv_cache_map.iter().max().unwrap_or(&0) + 1;
    let mut cache: Vec<gemma4_mlx::KVCache> =
        gemma4_mlx::init_cache::<gemma4_mlx::KVCache>(num_slots);
    let prompt_arr2 = mlx_rs::Array::from_slice(
        &prompt_ids.iter().map(|&t| t as i32).collect::<Vec<_>>(),
        &[1, prompt_ids.len() as i32],
    );

    let gen_start = Instant::now();
    let mut first_token_time: Option<f32> = None;
    let mut decode_tokens: usize = 0;
    let mut total_tokens: usize = 0;

    let generator = gemma4_mlx::Generate::new(
        &mut model,
        &mut cache,
        args.temp,
        &prompt_arr2,
    );
    let mut text_pieces: Vec<u32> = Vec::new();
    for tok in generator.take(args.max_tokens) {
        let tok = tok.map_err(|e| anyhow!("{e}"))?;
        let token_id = tok.item::<u32>();
        if first_token_time.is_none() {
            first_token_time = Some(gen_start.elapsed().as_secs_f32());
        }
        if eos.contains(&token_id) {
            break;
        }
        text_pieces.push(token_id);
        total_tokens += 1;
        decode_tokens += 1;
    }
    let total_s = gen_start.elapsed().as_secs_f32();
    let ttft_s = first_token_time.unwrap_or(total_s);
    let decode_s = (total_s - ttft_s).max(1e-6);
    let decode_tps = if decode_tokens > 1 {
        (decode_tokens - 1) as f32 / decode_s
    } else {
        0.0
    };

    let text = tokenizer
        .decode(&text_pieces, true)
        .map_err(|e| anyhow!(e.to_string()))?;
    println!("--- generated ---");
    println!("{}", text);
    println!("--- metrics ---");
    println!(
        "prefill_s={:.3} decode_s={:.3} decode_tok_per_s={:.2}",
        ttft_s, decode_s, decode_tps
    );
    println!(
        "total_tokens={} mtp_cycles=0 ar_fallback_steps={} (gemma4 AR — paired-model MTP TBD)",
        total_tokens, total_tokens
    );
    Ok(())
}

/// Shared prompt-arg rendering (string or chat fixture).
fn render_prompt_arg(arg: &str) -> Result<String> {
    if std::path::Path::new(arg).exists() && arg.ends_with(".json") {
        let bytes = std::fs::read(arg)?;
        let v: serde_json::Value = serde_json::from_slice(&bytes)?;
        let msgs = v
            .get("messages")
            .and_then(|m| m.as_array())
            .ok_or_else(|| anyhow!("fixture has no messages array"))?;
        let mut out = String::new();
        for m in msgs {
            let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("user");
            let content = match m.get("content") {
                Some(serde_json::Value::String(s)) => s.clone(),
                Some(serde_json::Value::Array(parts)) => parts
                    .iter()
                    .filter_map(|p| p.get("text").and_then(|t| t.as_str()).map(String::from))
                    .collect::<Vec<_>>()
                    .join(""),
                _ => String::new(),
            };
            // Both Qwen3.6 and Gemma4 use ChatML for the bench fixtures
            // here; if needed, future templates can branch on model_type.
            out.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
        }
        out.push_str("<|im_start|>assistant\n");
        Ok(out)
    } else {
        Ok(arg.to_string())
    }
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
