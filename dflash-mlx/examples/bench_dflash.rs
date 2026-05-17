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
    build_tree, discover_draft_for_target, topk_per_position, verify_tree_naive,
    DFlashDraftAdapter, DFlashDraftModel, DFlashSession, DraftCheckpointInfo, DraftModel,
    Gemma4TargetAdapter, MockDraftAdapter, Qwen36TargetAdapter, SessionMetrics,
    SpeculativeCycleConfig, TargetModel,
};
use dflash_mlx::engine::ddtree::{verify_tree_fused, GemmaTreeTarget};
use mlx_rs::ops::indexing::IndexOp;
use qwen3_6_mlx::{load_model, load_tokenizer, Generate};

const DEFAULT_PROMPT: &str =
    "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n";

#[derive(Debug)]
struct DDTreeArgs {
    enabled: bool,
    budget: usize,
    topk: usize,
    fused: bool,
}

struct Args {
    target: PathBuf,
    draft: Option<PathBuf>,
    prompt: String,
    max_tokens: usize,
    temp: f32,
    cpu: bool,
    ddtree: DDTreeArgs,
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

/// DDTree run loop. Builds a budget-N tree from draft logits per cycle and
/// verifies each tree branch sequentially (snapshot/restore between
/// branches) before walking to the longest accepted path.
fn run_ddtree<T: TargetModel + GemmaTreeTarget, D: DraftModel>(
    target: &mut T,
    draft: &mut D,
    prompt_tokens: Vec<u32>,
    max_tokens: usize,
    temp: f32,
    eos_tokens: &HashSet<u32>,
    block_size: usize,
    tree_budget: usize,
    tree_topk: usize,
    fused: bool,
) -> Result<(RunStats, SessionMetrics)> {
    let start = Instant::now();
    let prompt_arr = mlx_rs::Array::from_slice(&prompt_tokens, &[1, prompt_tokens.len() as i32]);
    let target_logits = target.prefill(&prompt_arr)?;
    let _ = draft.prefill(&prompt_arr)?;
    if let Some(h) = target.last_target_hidden() {
        draft.set_target_hidden(h);
    }
    let first_tok_arr = target
        .sample(&target_logits, temp)?
        .as_dtype(mlx_rs::Dtype::Uint32)?;
    let first_tok_arr = first_tok_arr.contiguous()?;
    mlx_rs::transforms::eval([&first_tok_arr])?;
    let first_token = first_tok_arr.item::<u32>();
    let ttft = start.elapsed().as_secs_f64();

    // Install first_token in target's KV cache and capture the predicted
    // token AFTER first_token. Used by fused DDTree as the "root_pred" for
    // accept_path's seed comparison.
    let first_arr = mlx_rs::Array::from_slice(&[first_token], &[1, 1]);
    let first_post_logits = target.verify(&first_arr)?;
    mlx_rs::transforms::eval([&first_post_logits])?;
    let root_pred_arr = mlx_rs::argmax_axis!(
        first_post_logits.index((.., -1, ..)).reshape(&[-1])?,
        -1
    )?
    .as_dtype(mlx_rs::Dtype::Uint32)?;
    mlx_rs::transforms::eval([&root_pred_arr])?;
    let mut root_pred = root_pred_arr.item::<u32>();

    let mut emitted: Vec<u32> = vec![first_token];
    let mut last_token = first_token;
    let mut metrics = SessionMetrics::default();

    while emitted.len() < max_tokens && !eos_tokens.contains(&last_token) {
        let remaining = max_tokens - emitted.len();
        if remaining < 2 {
            break;
        }
        let block_len = block_size.min(remaining).max(2);

        // Staged embedding for DFlash drafter alignment.
        if let Some(staged_emb) = target.embed_token(last_token) {
            draft.set_staged_embedding(staged_emb);
        }
        if let Some(h) = target.last_target_hidden() {
            draft.set_target_hidden(h);
        }
        let last_arr = mlx_rs::Array::from_slice(&[last_token], &[1, 1]);
        let drafted = draft.draft_block(&last_arr, block_len)?;
        // drafted.logits: [1, block_len-1, vocab] — per-position next-token
        // logits over the diffusion block.
        let block_logits = drafted.logits.index((0, .., ..));
        mlx_rs::transforms::eval([&block_logits])?;
        let (top_ids, top_lps) = topk_per_position(&block_logits, tree_topk)?;
        let tree = build_tree(&top_ids, &top_lps, tree_budget);
        let kv_offset = target.step_count() as i32;
        let start_pos = kv_offset; // seed token at absolute position kv_offset

        let (accepted, bonus) = if fused {
            // One fused tree forward → per-node logits + per-node hidden.
            let (tokens, position_ids, mask) = dflash_mlx::engine::ddtree::compile_tree_inputs(
                last_token, &tree, start_pos, kv_offset, mlx_rs::Dtype::Bfloat16,
            )?;
            let (tree_logits, tree_hidden) = target
                .verify_tree_with_hidden_call(&tokens, &position_ids, &mask)?;
            mlx_rs::transforms::eval([&tree_logits, &tree_hidden])?;

            // Acceptance walk.
            let preds = mlx_rs::argmax_axis!(&tree_logits, -1)?
                .as_dtype(mlx_rs::Dtype::Uint32)?
                .reshape(&[-1])?;
            mlx_rs::transforms::eval([&preds])?;
            let preds_slice = preds.as_slice::<u32>();
            let predicted_after_node: Vec<u32> = preds_slice.to_vec();
            let (acc, bonus_tok) =
                dflash_mlx::engine::ddtree::accept_path(&tree, &predicted_after_node, root_pred);

            // In-place interior KV compaction: keep just the accepted
            // node slots (positions kv_offset + acc[0], +acc[1], ...).
            // Then capture bonus's "next root pred" from the LAST
            // accepted node's hidden via lm_head, or fall back to the
            // tree's seed-based root_pred when nothing accepted (bonus
            // is then just root_pred and the cycle commits only that).
            if acc.is_empty() {
                // Nothing accepted ⇒ keep no tree slots. Bonus must still
                // get its K/V installed before the next cycle.
                target.compact_cache_call(
                    kv_offset,
                    &mlx_rs::Array::from_slice::<i32>(&[], &[0]),
                )?;
                let bonus_arr = mlx_rs::Array::from_slice(&[bonus_tok], &[1, 1]);
                let bonus_logits = target.verify(&bonus_arr)?;
                mlx_rs::transforms::eval([&bonus_logits])?;
                let next_root_arr = mlx_rs::argmax_axis!(
                    bonus_logits.index((.., -1, ..)).reshape(&[-1])?,
                    -1
                )?
                .as_dtype(mlx_rs::Dtype::Uint32)?;
                mlx_rs::transforms::eval([&next_root_arr])?;
                root_pred = next_root_arr.item::<u32>();
            } else {
                let keep_idx_i32: Vec<i32> = acc.iter().map(|&i| i as i32).collect();
                let keep_arr = mlx_rs::Array::from_slice(
                    &keep_idx_i32,
                    &[keep_idx_i32.len() as i32],
                );
                target.compact_cache_call(kv_offset, &keep_arr)?;
                // Now feed bonus alone (1 token) to install its K/V and
                // grab next-cycle root_pred from its post-norm hidden.
                let bonus_arr = mlx_rs::Array::from_slice(&[bonus_tok], &[1, 1]);
                let bonus_logits = target.verify(&bonus_arr)?;
                mlx_rs::transforms::eval([&bonus_logits])?;
                let next_root_arr = mlx_rs::argmax_axis!(
                    bonus_logits.index((.., -1, ..)).reshape(&[-1])?,
                    -1
                )?
                .as_dtype(mlx_rs::Dtype::Uint32)?;
                mlx_rs::transforms::eval([&next_root_arr])?;
                root_pred = next_root_arr.item::<u32>();
            }
            (acc, bonus_tok)
        } else {
            let (acc, bonus_tok, _accepted_tokens) =
                verify_tree_naive(target, last_token, &tree)?;
            (acc, bonus_tok)
        };
        let n_accepted = accepted.len();
        metrics.total_cycles += 1;
        metrics.total_drafted += tree.len();
        metrics.total_accepted += n_accepted;

        // Commit accepted path + bonus.
        for &idx in &accepted {
            let tok = tree[idx].token_id;
            emitted.push(tok);
            if eos_tokens.contains(&tok) || emitted.len() >= max_tokens {
                break;
            }
        }
        if !eos_tokens.contains(emitted.last().unwrap()) && emitted.len() < max_tokens {
            emitted.push(bonus);
        }
        last_token = *emitted.last().unwrap();
        // Sync hidden capture for next cycle.
        if let Some(h) = target.last_target_hidden() {
            draft.set_target_hidden(h);
        }
    }

    if metrics.total_drafted > 0 {
        metrics.acceptance_ratio = metrics.total_accepted as f32 / metrics.total_drafted as f32;
        metrics.avg_block_len = metrics.total_accepted as f32 / metrics.total_cycles as f32;
    }
    metrics.total_tokens = emitted.len();
    Ok((finalize_stats(start, Some(ttft), emitted.len()), metrics))
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
            // Gemma4 LM head: explicit lm_head when present, else the tied
            // embedding table. Matches the Python reference path.
            let mask_emb = model
                .embed_tokens(&[mask_token_id as i32])
                .map_err(|e| anyhow!(e.to_string()))?;
            let lm_head_weight = model
                .get_lm_head_weight()
                .map_err(|e| anyhow!(e.to_string()))?;
            let mut target =
                Gemma4TargetAdapter::with_dflash(model, args.temp, target_layer_ids);
            let mut draft = DFlashDraftAdapter::new(draft_model, mask_emb, lm_head_weight);
            let (dflash_stats, metrics) = if args.ddtree.enabled {
                println!(
                    "DDTree mode: budget={} topk={} block_size={} fused={}",
                    args.ddtree.budget, args.ddtree.topk, block_size, args.ddtree.fused
                );
                run_ddtree(
                    &mut target,
                    &mut draft,
                    prompt_ids,
                    args.max_tokens,
                    args.temp,
                    &eos_tokens,
                    block_size,
                    args.ddtree.budget,
                    args.ddtree.topk,
                    args.ddtree.fused,
                )?
            } else {
                let spec_config = SpeculativeCycleConfig {
                    block_len: block_size,
                    ..Default::default()
                };
                let mut session = DFlashSession::new(target, draft, spec_config);
                run_dflash_session(
                    &mut session,
                    prompt_ids,
                    args.max_tokens,
                    args.temp,
                    &eos_tokens,
                )?
            };
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
    let mut ddtree_enabled = false;
    let mut ddtree_budget = 16usize;
    let mut ddtree_topk = 4usize;
    let mut ddtree_naive = false;

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
            "--ddtree" => ddtree_enabled = true,
            "--tree-budget" => {
                ddtree_budget = args
                    .next()
                    .ok_or_else(|| anyhow!("--tree-budget requires a value"))?
                    .parse()
                    .context("invalid --tree-budget value")?
            }
            "--tree-naive" => ddtree_naive = true,
            "--tree-topk" => {
                ddtree_topk = args
                    .next()
                    .ok_or_else(|| anyhow!("--tree-topk requires a value"))?
                    .parse()
                    .context("invalid --tree-topk value")?
            }
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
        ddtree: DDTreeArgs {
            enabled: ddtree_enabled,
            budget: ddtree_budget,
            topk: ddtree_topk,
            fused: !ddtree_naive,
        },
    })
}
