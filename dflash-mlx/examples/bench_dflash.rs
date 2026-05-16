//! Benchmark DFlash speculative decoding vs autoregressive on Qwen3.6-35B-A3B-4bit.
//!
//! Usage:
//!   cargo run --release --example bench_dflash -- \
//!     --target /path/to/Qwen3.6-35B-A3B-4bit \
//!     [--draft /path/to/DFlash-draft-model] \
//!     [--prompt "..."] \
//!     [--max-tokens 200] \
//!     [--temp 0.7]

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use dflash_mlx::{
    DFlashDraftAdapter, DFlashDraftModel, DFlashSession, DraftCheckpointInfo, MockDraftAdapter,
    Qwen36TargetAdapter, SessionMetrics, SpeculativeCycleConfig,
};
use qwen3_6_mlx::{load_model, load_tokenizer, Generate};

const DEFAULT_PROMPT: &str =
    "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n";

#[derive(Debug)]
struct Args {
    target: PathBuf,
    draft: Option<PathBuf>,
    prompt: String,
    max_tokens: usize,
    temp: f32,
    cpu: bool,
}

#[derive(Debug, Clone)]
struct RunStats {
    prefill_s: f64,
    decode_tok_s: f64,
    total_tokens: usize,
}

fn main() -> Result<()> {
    let args = parse_args()?;
    if args.cpu {
        mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
        eprintln!("Using CPU device");
    } else {
        eprintln!("Using GPU device");
    }

    let tokenizer = load_tokenizer(&args.target)?;
    let encoding = tokenizer
        .encode(args.prompt.as_str(), false)
        .map_err(|e| anyhow!(e.to_string()))?;
    let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
    let prompt = mlx_rs::Array::from_slice(&prompt_ids, &[1, prompt_ids.len() as i32]);
    let eos_tokens = load_eos_tokens(&args.target)?;

    eprintln!("Loading target model from {}...", args.target.display());
    let load_start = Instant::now();
    let mut model = load_model(&args.target)?;
    eprintln!(
        "Target loaded in {:.1}s",
        load_start.elapsed().as_secs_f64()
    );

    let ar_stats =
        run_autoregressive(&mut model, &prompt, args.max_tokens, args.temp, &eos_tokens)?;
    println!(
        "Autoregressive: prefill_s={:.3} decode_tok_s={:.2} total_tokens={}",
        ar_stats.prefill_s, ar_stats.decode_tok_s, ar_stats.total_tokens
    );

    match resolve_draft_mode(args.draft.as_deref()) {
        DraftMode::Missing(note) => {
            println!("{note}");
        }
        DraftMode::Mock(note) => {
            println!("{note}");
            let vocab_size = model.args.text_config.vocab_size as u32;
            let target = Qwen36TargetAdapter::new(model, args.temp);
            let draft = MockDraftAdapter::new(vocab_size);
            let mut session = DFlashSession::new(target, draft, SpeculativeCycleConfig::default());
            let (dflash_stats, metrics) = run_dflash(
                &mut session,
                prompt_ids,
                args.max_tokens,
                args.temp,
                &eos_tokens,
            )?;
            println!(
                "DFlash: prefill_s={:.3} decode_tok_s={:.2} acceptance_ratio={:.3} avg_block_len={:.2} total_tokens={}",
                dflash_stats.prefill_s,
                dflash_stats.decode_tok_s,
                metrics.acceptance_ratio,
                metrics.avg_block_len,
                dflash_stats.total_tokens
            );
            let speedup = if ar_stats.decode_tok_s > 0.0 {
                dflash_stats.decode_tok_s / ar_stats.decode_tok_s
            } else {
                0.0
            };
            println!("Speedup ratio: {:.2}x", speedup);
        }
        DraftMode::Real(draft_path) => {
            println!("Loading DFlash draft model from {}...", draft_path.display());
            let draft_model = DFlashDraftModel::load_from_path(&draft_path)?;
            let block_size = draft_model.args.block_size();
            let target_layer_ids = draft_model.args.target_layer_ids();
            let mask_token_id = draft_model.args.mask_token_id();
            let lm_head_weight = model.get_lm_head_weight()?;
            let mask_emb = model.embed_tokens(&[mask_token_id as i32])?;
            let target = Qwen36TargetAdapter::with_dflash(model, args.temp, target_layer_ids);
            let draft = DFlashDraftAdapter::new(draft_model, mask_emb, lm_head_weight);
            let spec_config = SpeculativeCycleConfig {
                block_len: block_size,
                min_block_tokens: block_size,
                ..Default::default()
            };
            let mut session = DFlashSession::new(target, draft, spec_config);
            let (dflash_stats, metrics) = run_dflash_real(
                &mut session,
                prompt_ids,
                args.max_tokens,
                args.temp,
                &eos_tokens,
            )?;
            println!(
                "DFlash: prefill_s={:.3} decode_tok_s={:.2} acceptance_ratio={:.3} avg_block_len={:.2} total_tokens={}",
                dflash_stats.prefill_s,
                dflash_stats.decode_tok_s,
                metrics.acceptance_ratio,
                metrics.avg_block_len,
                dflash_stats.total_tokens
            );
            let speedup = if ar_stats.decode_tok_s > 0.0 {
                dflash_stats.decode_tok_s / ar_stats.decode_tok_s
            } else {
                0.0
            };
            println!("Speedup ratio: {:.2}x", speedup);
        }
    }

    Ok(())
}

fn run_autoregressive(
    model: &mut qwen3_6_mlx::Model,
    prompt: &mlx_rs::Array,
    max_tokens: usize,
    temp: f32,
    eos_tokens: &HashSet<u32>,
) -> Result<RunStats> {
    let start = Instant::now();
    let mut ttft = None;
    let mut total_tokens = 0usize;

    for token in Generate::new(model, temp, prompt).take(max_tokens) {
        let token = token?;
        if ttft.is_none() {
            ttft = Some(start.elapsed().as_secs_f64());
        }
        let token_id = token.item::<u32>();
        if eos_tokens.contains(&token_id) {
            break;
        }
        total_tokens += 1;
    }

    Ok(finalize_stats(start, ttft, total_tokens))
}

fn run_dflash(
    session: &mut DFlashSession<Qwen36TargetAdapter, MockDraftAdapter>,
    prompt_tokens: Vec<u32>,
    max_tokens: usize,
    temp: f32,
    eos_tokens: &HashSet<u32>,
) -> Result<(RunStats, SessionMetrics)> {
    run_dflash_session(session, prompt_tokens, max_tokens, temp, eos_tokens)
}

fn run_dflash_real(
    session: &mut DFlashSession<Qwen36TargetAdapter, DFlashDraftAdapter>,
    prompt_tokens: Vec<u32>,
    max_tokens: usize,
    temp: f32,
    eos_tokens: &HashSet<u32>,
) -> Result<(RunStats, SessionMetrics)> {
    run_dflash_session(session, prompt_tokens, max_tokens, temp, eos_tokens)
}

fn run_dflash_session<Target, Draft>(
    session: &mut DFlashSession<Target, Draft>,
    prompt_tokens: Vec<u32>,
    max_tokens: usize,
    temp: f32,
    eos_tokens: &HashSet<u32>,
) -> Result<(RunStats, SessionMetrics)>
where
    Target: dflash_mlx::TargetModel,
    Draft: dflash_mlx::DraftModel,
{
    let start = Instant::now();
    let mut ttft = None;
    let mut total_tokens = 0usize;
    let eos_vec: Vec<u32> = eos_tokens.iter().copied().collect();

    for token in session.run_generate(prompt_tokens, max_tokens, temp, &eos_vec) {
        let token_id = token?;
        if ttft.is_none() {
            ttft = Some(start.elapsed().as_secs_f64());
        }
        if eos_tokens.contains(&token_id) {
            break;
        }
        total_tokens += 1;
    }

    Ok((
        finalize_stats(start, ttft, total_tokens),
        session.metrics().clone(),
    ))
}

fn finalize_stats(start: Instant, ttft: Option<f64>, total_tokens: usize) -> RunStats {
    let elapsed = start.elapsed().as_secs_f64();
    let prefill_s = ttft.unwrap_or(elapsed);
    let decode_time = (elapsed - prefill_s).max(0.0);
    let decode_tokens = total_tokens.saturating_sub(1);
    let decode_tok_s = if decode_tokens > 0 && decode_time > 0.0 {
        decode_tokens as f64 / decode_time
    } else {
        0.0
    };
    RunStats {
        prefill_s,
        decode_tok_s,
        total_tokens,
    }
}

enum DraftMode {
    Missing(String),
    Mock(String),
    Real(PathBuf),
}

fn resolve_draft_mode(cli_draft: Option<&Path>) -> DraftMode {
    let default_path = Path::new("models/Qwen3.6-35B-A3B-DFlash");
    let draft_path = cli_draft
        .map(PathBuf::from)
        .or_else(|| default_path.exists().then(|| default_path.to_path_buf()));

    match draft_path {
        Some(path) => match DraftCheckpointInfo::load(&path) {
            Ok(info) if info.is_native_rust_supported() => DraftMode::Real(path),
            Ok(info) => DraftMode::Mock(format!(
                "Draft checkpoint at {} uses architectures {:?}; current Rust integration falls back to MockDraftAdapter for pipeline benchmarking.",
                path.display(),
                info.architectures
            )),
            Err(err) => DraftMode::Mock(format!(
                "Could not inspect draft checkpoint at {} ({err}); falling back to MockDraftAdapter.",
                path.display()
            )),
        },
        None => DraftMode::Missing(
            "No draft model found. Download z-lab/Qwen3.6-35B-A3B-DFlash into models/ and re-run, or pass --draft /path/to/draft.".to_string(),
        ),
    }
}

fn load_eos_tokens(model_dir: &Path) -> Result<HashSet<u32>> {
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
    let mut draft = None;
    let mut prompt = DEFAULT_PROMPT.to_string();
    let mut max_tokens = 200usize;
    let mut temp = 0.7f32;
    let mut cpu = false;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--target" => {
                target = PathBuf::from(
                    args.next()
                        .ok_or_else(|| anyhow!("--target requires a path"))?,
                )
            }
            "--draft" => {
                draft = Some(PathBuf::from(
                    args.next()
                        .ok_or_else(|| anyhow!("--draft requires a path"))?,
                ))
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
            "--cpu" => cpu = true,
            "--help" | "-h" => {
                println!(
                    "Usage: cargo run --release --example bench_dflash -- --target /path/to/Qwen3.6-35B-A3B-4bit [--draft /path/to/DFlash-draft-model] [--prompt \"...\"] [--max-tokens 200] [--temp 0.7] [--cpu]"
                );
                std::process::exit(0);
            }
            other => return Err(anyhow!("unknown argument: {other}")),
        }
    }

    Ok(Args {
        target,
        draft,
        prompt,
        max_tokens,
        temp,
        cpu,
    })
}
