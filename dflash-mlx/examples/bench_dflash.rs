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
    discover_draft_for_target, DFlashDraftAdapter, DFlashDraftModel, DFlashSession,
    DraftCheckpointInfo, Gemma4TargetAdapter, MockDraftAdapter, Qwen36TargetAdapter,
    SessionMetrics, SpeculativeCycleConfig,
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
    let target_kind = detect_target_kind(&args.target)?;
    eprintln!("Detected target model_type: {target_kind:?}");

    if matches!(target_kind, TargetKind::Gemma4) {
        return run_gemma4(
            &args,
            prompt_ids,
            eos_tokens,
        );
    }

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

    match resolve_draft_mode(&args.target, args.draft.as_deref()) {
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
            // NOTE: min_block_tokens must be strictly less than block_len for
            // adaptive sizing to engage. Setting them equal (as the prior config
            // did with block_size=16 for both) neutered the adaptive policy —
            // current_block_len() returns block_len.max(min_block_tokens) =
            // block_len in every cycle. Python's reference uses 4 as the
            // reduced size, matching `SpeculativeCycleConfig::default()`.
            let spec_config = SpeculativeCycleConfig {
                block_len: block_size,
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

fn resolve_draft_mode(target: &Path, cli_draft: Option<&Path>) -> DraftMode {
    let draft_path = cli_draft
        .map(PathBuf::from)
        .or_else(|| discover_draft_for_target(target));

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
        None => DraftMode::Missing(format!(
            "No DFlash draft found alongside target at {}. Expected a sibling '<target>-DFlash' dir, a 'dflash_draft_path' key in the target config, or --draft /path/to/draft.",
            target.display()
        )),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TargetKind {
    Qwen36,
    Gemma4,
}

fn detect_target_kind(model_dir: &Path) -> Result<TargetKind> {
    let config_path = model_dir.join("config.json");
    let config: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&config_path)
            .with_context(|| format!("failed to read {}", config_path.display()))?,
    )?;
    let model_type = config
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if model_type == "gemma4" || model_type.starts_with("gemma4") {
        Ok(TargetKind::Gemma4)
    } else {
        Ok(TargetKind::Qwen36)
    }
}

/// Gemma4 (26B-A4B) bench routing path.
///
/// The DFlash draft for Gemma4 is `z-lab/gemma-4-26B-A4B-it-DFlash`, which is
/// a gated repo and typically not present locally. This function therefore
/// only requires Real/Mock/Missing routing to compile; runtime behavior for
/// the Real branch is unverified until a draft is available.
fn run_gemma4(
    args: &Args,
    prompt_ids: Vec<u32>,
    eos_tokens: HashSet<u32>,
) -> Result<()> {
    eprintln!(
        "Loading Gemma4 target model from {}...",
        args.target.display()
    );
    let load_start = Instant::now();
    let mut model = gemma4_mlx::load_model(&args.target)?;
    eprintln!(
        "Gemma4 target loaded in {:.1}s",
        load_start.elapsed().as_secs_f64()
    );

    // Autoregressive baseline is not implemented for Gemma4 here — gemma4-mlx
    // does not yet expose a standalone `Generate` iterator analogous to
    // qwen3_6_mlx's. Skip AR baseline; users can compare against the existing
    // chat_gemma4 example for AR throughput. The DFlash speedup ratio is
    // therefore omitted.
    let _ = (&mut model, &prompt_ids, &eos_tokens);

    match resolve_draft_mode(&args.target, args.draft.as_deref()) {
        DraftMode::Missing(note) => {
            println!("{note}");
            println!(
                "Gemma4 DFlash draft is gated (z-lab/gemma-4-26B-A4B-it-DFlash); supply --draft \
                 to a local copy to exercise the Real path."
            );
        }
        DraftMode::Mock(note) => {
            println!("{note}");
            let vocab_size = model.args.vocab_size as u32;
            // No target_layer_ids — mock draft does not consume hidden states.
            let target = Gemma4TargetAdapter::with_dflash(model, args.temp, Vec::new());
            let draft = MockDraftAdapter::new(vocab_size);
            let mut session = DFlashSession::new(target, draft, SpeculativeCycleConfig::default());
            let (dflash_stats, metrics) = run_dflash_session(
                &mut session,
                prompt_ids,
                args.max_tokens,
                args.temp,
                &eos_tokens,
            )?;
            println!(
                "DFlash(gemma4): prefill_s={:.3} decode_tok_s={:.2} acceptance_ratio={:.3} avg_block_len={:.2} total_tokens={}",
                dflash_stats.prefill_s,
                dflash_stats.decode_tok_s,
                metrics.acceptance_ratio,
                metrics.avg_block_len,
                dflash_stats.total_tokens
            );
        }
        DraftMode::Real(draft_path) => {
            println!(
                "Loading DFlash draft model from {} (gemma4 target)...",
                draft_path.display()
            );
            let draft_model = DFlashDraftModel::load_from_path(&draft_path)?;
            let block_size = draft_model.args.block_size();
            let target_layer_ids = draft_model.args.target_layer_ids();
            let mask_token_id = draft_model.args.mask_token_id();
            // Gemma4 LM head: prefer the explicit `lm_head` weight when
            // present, otherwise tie to `embed_tokens` (matches the Python
            // reference's tied-embedding path).
            //
            // TODO: gemma4-mlx does not yet expose a `get_lm_head_weight`
            // accessor analogous to Qwen36's. For the bench-routing
            // milestone we approximate via the embedding-tied weight, which
            // is what the Real DFlash draft would need anyway for the
            // mask-token embedding lookup.
            let mask_emb = model
                .embed_tokens(&[mask_token_id as i32])
                .map_err(|e| anyhow!(e.to_string()))?;
            // Use the embed table as the LM head weight (tied-embedding
            // assumption). This is consistent with gemma4 fallback path
            // `embed_tokens.as_linear` at model.rs:1285.
            let lm_head_weight = mask_emb.clone(); // placeholder; see TODO above
            let target = Gemma4TargetAdapter::with_dflash(model, args.temp, target_layer_ids);
            let draft = DFlashDraftAdapter::new(draft_model, mask_emb, lm_head_weight);
            let spec_config = SpeculativeCycleConfig {
                block_len: block_size,
                ..Default::default()
            };
            let mut session = DFlashSession::new(target, draft, spec_config);
            let (dflash_stats, metrics) = run_dflash_session(
                &mut session,
                prompt_ids,
                args.max_tokens,
                args.temp,
                &eos_tokens,
            )?;
            println!(
                "DFlash(gemma4): prefill_s={:.3} decode_tok_s={:.2} acceptance_ratio={:.3} avg_block_len={:.2} total_tokens={}",
                dflash_stats.prefill_s,
                dflash_stats.decode_tok_s,
                metrics.acceptance_ratio,
                metrics.avg_block_len,
                dflash_stats.total_tokens
            );
        }
    }
    Ok(())
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
