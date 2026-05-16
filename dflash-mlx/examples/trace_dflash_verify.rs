use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use dflash_mlx::{
    DraftBlock, DraftModel, DFlashDraftAdapter, DFlashDraftModel, DFlashSession,
    DraftCheckpointInfo, Qwen36TargetAdapter, SpeculativeCycleConfig, TargetModel,
};
use mlx_rs::{
    argmax_axis,
    error::Exception,
    module::Module,
    ops::{concatenate_axis, indexing::IndexOp, zeros_dtype},
    transforms::eval,
    Array, Dtype,
};
use qwen3_6_mlx::{load_model, load_tokenizer, model::AttentionLayer};

const DEFAULT_PROMPT: &str = "The theory of general relativity";
const EXACT_SMALL_PROJ_QKV_PAD_M: i32 = 6;
const EXACT_SMALL_PROJ_Z_PAD_M: i32 = 10;

#[derive(Debug)]
struct Args {
    target: PathBuf,
    draft: PathBuf,
    prompt: String,
    max_tokens: usize,
    temp: f32,
    cpu: bool,
}

struct TraceTarget {
    inner: Qwen36TargetAdapter,
    verify_lens: Vec<usize>,
    verify_inputs: Vec<Vec<u32>>,
    verify_posteriors: Vec<Vec<u32>>,
}

struct TraceDraft {
    inner: DFlashDraftAdapter,
    draft_context_lens: Vec<usize>,
    drafted_blocks: Vec<Vec<u32>>,
}

impl TargetModel for TraceTarget {
    fn prefill(&mut self, prompt: &Array) -> Result<Array, Exception> {
        self.inner.prefill(prompt)
    }

    fn verify(&mut self, drafted_tokens: &Array) -> Result<Array, Exception> {
        self.verify_lens.push(drafted_tokens.shape()[1] as usize);
        self.verify_inputs
            .push(array_to_vec_u32(&drafted_tokens.index((0, ..)))?);
        let logits = self.inner.verify(drafted_tokens)?;
        let posterior = argmax_axis!(&logits, -1)?.as_dtype(Dtype::Uint32)?;
        self.verify_posteriors
            .push(array_to_vec_u32(&posterior.index((0, ..)))?);
        Ok(logits)
    }

    fn step_count(&self) -> usize {
        self.inner.step_count()
    }

    fn rollback_kv(&mut self, n_keep: usize) -> Result<(), Exception> {
        self.inner.rollback_kv(n_keep)
    }

    fn sample(&self, logits: &Array, temp: f32) -> Result<Array, Exception> {
        self.inner.sample(logits, temp)
    }

    fn last_target_hidden(&self) -> Option<Array> {
        self.inner.last_target_hidden()
    }

    fn embed_token(&mut self, id: u32) -> Option<Array> {
        self.inner.embed_token(id)
    }
}

impl DraftModel for TraceDraft {
    fn prefill(&mut self, prompt: &Array) -> Result<Array, Exception> {
        self.inner.prefill(prompt)
    }

    fn draft_block(&mut self, last_token: &Array, block_len: usize) -> Result<DraftBlock, Exception> {
        let context_len = self
            .inner
            .target_hidden()
            .map(|hidden| hidden.shape()[1] as usize)
            .ok_or_else(|| Exception::custom("TraceDraft: target_hidden missing before draft_block"))?;
        self.draft_context_lens.push(context_len);
        let block = self.inner.draft_block(last_token, block_len)?;
        self.drafted_blocks
            .push(array_to_vec_u32(&block.tokens.index((0, ..)))?);
        Ok(block)
    }

    fn rollback(&mut self, n_accepted: usize) -> Result<(), Exception> {
        self.inner.rollback(n_accepted)
    }

    fn set_target_hidden(&mut self, h: Array) {
        self.inner.set_target_hidden(h);
    }

    fn set_staged_embedding(&mut self, emb: Array) {
        self.inner.set_staged_embedding(emb);
    }
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
    let prompt_len = prompt_ids.len();
    let eos_tokens = load_eos_tokens(&args.target)?;

    let mut model = load_model(&args.target)?;
    let draft_model = DFlashDraftModel::load_from_path(&args.draft)?;
    let block_size = draft_model.args.block_size();
    let target_layer_ids = draft_model.args.target_layer_ids();
    let mask_token_id = draft_model.args.mask_token_id();
    let lm_head_weight = model.get_lm_head_weight()?;
    let mask_emb = model.embed_tokens(&[mask_token_id as i32])?;

    let target = TraceTarget {
        inner: Qwen36TargetAdapter::with_dflash(model, args.temp, target_layer_ids),
        verify_lens: Vec::new(),
        verify_inputs: Vec::new(),
        verify_posteriors: Vec::new(),
    };
    let draft = TraceDraft {
        inner: DFlashDraftAdapter::new(draft_model, mask_emb, lm_head_weight),
        draft_context_lens: Vec::new(),
        drafted_blocks: Vec::new(),
    };
    let spec_config = SpeculativeCycleConfig {
        block_len: block_size,
        min_block_tokens: block_size,
        ..Default::default()
    };
    let mut session = DFlashSession::new(target, draft, spec_config);
    let eos_vec: Vec<u32> = eos_tokens.iter().copied().collect();
    let mut emitted = 0usize;
    for token in session.run_generate(prompt_ids, args.max_tokens, args.temp, &eos_vec) {
        let token = token?;
        if eos_tokens.contains(&token) {
            break;
        }
        emitted += 1;
    }

    let mut hist = BTreeMap::<usize, usize>::new();
    for len in &session.target().verify_lens {
        *hist.entry(*len).or_default() += 1;
    }
    println!("emitted_tokens={emitted}");
    println!("verify_lengths={:?}", session.target().verify_lens);
    println!("verify_hist={hist:?}");
    print_short_window_projection_analysis(&args.target, &session.target().verify_inputs)?;
    print_context_contract_analysis(prompt_len, session.target(), session.draft());
    println!("metrics={:?}", session.metrics());
    Ok(())
}

fn print_context_contract_analysis(
    prompt_len: usize,
    target: &TraceTarget,
    draft: &TraceDraft,
) {
    println!("context_contract_analysis:");
    let mut expected_context_len = prompt_len;
    for cycle_idx in 0..draft.draft_context_lens.len() {
        let context_len = draft.draft_context_lens[cycle_idx];
        let drafted = &draft.drafted_blocks[cycle_idx];
        let posterior = &target.verify_posteriors[cycle_idx];
        let drafted_count = drafted.len();
        let acceptance_len = drafted
            .iter()
            .zip(posterior.iter().take(drafted_count))
            .take_while(|(drafted_token, posterior_token)| drafted_token == posterior_token)
            .count();
        let commit_count = acceptance_len + 1;
        let expected_next_context = context_len + commit_count;
        let next_context = draft.draft_context_lens.get(cycle_idx + 1).copied();
        let progression = match next_context {
            Some(next_len) => format!("{next_len} (expected {expected_next_context})"),
            None => "n/a".to_string(),
        };
        println!(
            "  cycle={} context_len={} expected_context_len={} drafted={} verify_len={} acceptance_len={} commit_count={} next_context_len={}",
            cycle_idx + 1,
            context_len,
            expected_context_len,
            drafted_count,
            target.verify_lens[cycle_idx],
            acceptance_len,
            commit_count,
            progression,
        );
        expected_context_len += commit_count;
    }
}

fn print_short_window_projection_analysis(target: &Path, verify_inputs: &[Vec<u32>]) -> Result<()> {
    let interesting: Vec<_> = verify_inputs
        .iter()
        .enumerate()
        .filter(|(_, tokens)| tokens.len() < 16)
        .collect();
    if interesting.is_empty() {
        println!("short_verify_projection_analysis=[]");
        return Ok(());
    }

    let mut model = load_model(target)?;
    println!("short_verify_projection_analysis:");
    for (cycle_idx, tokens) in interesting {
        let analysis = analyze_layer0_window(&mut model, tokens)?;
        println!(
            "  cycle={} len={} qkv_exact={} z_exact={} qkv_direct_vs_contig=max:{:.6} mean:{:.6} qkv_direct_vs_exact=max:{:.6} mean:{:.6} z_direct_vs_contig=max:{:.6} mean:{:.6} z_direct_vs_exact=max:{:.6} mean:{:.6} tokens={:?}",
            cycle_idx + 1,
            tokens.len(),
            tokens.len() < EXACT_SMALL_PROJ_QKV_PAD_M as usize,
            tokens.len() < EXACT_SMALL_PROJ_Z_PAD_M as usize,
            analysis.qkv_direct_vs_contig.0,
            analysis.qkv_direct_vs_contig.1,
            analysis.qkv_direct_vs_exact.0,
            analysis.qkv_direct_vs_exact.1,
            analysis.z_direct_vs_contig.0,
            analysis.z_direct_vs_contig.1,
            analysis.z_direct_vs_exact.0,
            analysis.z_direct_vs_exact.1,
            tokens,
        );
    }
    Ok(())
}

#[derive(Debug)]
struct WindowProjectionAnalysis {
    qkv_direct_vs_contig: (f32, f32),
    qkv_direct_vs_exact: (f32, f32),
    z_direct_vs_contig: (f32, f32),
    z_direct_vs_exact: (f32, f32),
}

fn analyze_layer0_window(
    model: &mut qwen3_6_mlx::Model,
    tokens: &[u32],
) -> Result<WindowProjectionAnalysis> {
    let token_ids: Vec<i32> = tokens.iter().map(|id| *id as i32).collect();
    let input = Array::from_slice(&token_ids, &[1, token_ids.len() as i32]);
    let embeddings = model.text_model.embed_tokens.forward(&input)?;
    let layer = model
        .text_model
        .layers
        .get_mut(0)
        .ok_or_else(|| anyhow!("model has no layers"))?;
    let layer0_input_norm = layer.input_layernorm.forward(&embeddings)?;

    let (qkv_direct, qkv_contig, qkv_exact, z_direct, z_contig, z_exact) =
        match &mut layer.attention {
            AttentionLayer::LinearAttention(delta) => {
                let input_contig = layer0_input_norm.contiguous()?;
                let qkv_direct = delta.in_proj_qkv.forward(&layer0_input_norm)?;
                let qkv_contig = delta.in_proj_qkv.forward(&input_contig)?;
                let qkv_exact = exact_small_proj_with_pad_m(
                    &layer0_input_norm,
                    EXACT_SMALL_PROJ_QKV_PAD_M,
                    |x| delta.in_proj_qkv.forward(x),
                )?;
                let z_direct = delta.in_proj_z.forward(&layer0_input_norm)?;
                let z_contig = delta.in_proj_z.forward(&input_contig)?;
                let z_exact = exact_small_proj_with_pad_m(
                    &layer0_input_norm,
                    EXACT_SMALL_PROJ_Z_PAD_M,
                    |x| delta.in_proj_z.forward(x),
                )?;
                (qkv_direct, qkv_contig, qkv_exact, z_direct, z_contig, z_exact)
            }
            AttentionLayer::FullAttention(_) => {
                return Err(anyhow!("layer 0 is not linear attention"));
            }
        };

    Ok(WindowProjectionAnalysis {
        qkv_direct_vs_contig: diff_stats(&qkv_direct, &qkv_contig)?,
        qkv_direct_vs_exact: diff_stats(&qkv_direct, &qkv_exact)?,
        z_direct_vs_contig: diff_stats(&z_direct, &z_contig)?,
        z_direct_vs_exact: diff_stats(&z_direct, &z_exact)?,
    })
}

fn exact_small_proj_with_pad_m<F>(x: &Array, pad_m: i32, mut forward: F) -> Result<Array>
where
    F: FnMut(&Array) -> Result<Array, Exception>,
{
    let seq_len = x.shape()[1];
    if seq_len < pad_m {
        let pad = zeros_dtype(&[x.shape()[0], pad_m - seq_len, x.shape()[2]], x.dtype())?;
        let padded = concatenate_axis(&[x, &pad], 1)?;
        let out = forward(&padded)?;
        Ok(out.index((.., ..seq_len, ..)))
    } else {
        Ok(forward(x)?)
    }
}

fn diff_stats(lhs: &Array, rhs: &Array) -> Result<(f32, f32)> {
    let lhs = lhs.as_dtype(Dtype::Float32)?.contiguous()?;
    let rhs = rhs.as_dtype(Dtype::Float32)?.contiguous()?;
    eval([&lhs, &rhs])?;
    let lhs = lhs.as_slice::<f32>();
    let rhs = rhs.as_slice::<f32>();
    if lhs.len() != rhs.len() {
        return Err(anyhow!(
            "shape mismatch in diff_stats: lhs={} rhs={}",
            lhs.len(),
            rhs.len()
        ));
    }
    let mut max_diff = 0.0f32;
    let mut sum_diff = 0.0f32;
    for (left, right) in lhs.iter().zip(rhs.iter()) {
        let diff = (left - right).abs();
        max_diff = max_diff.max(diff);
        sum_diff += diff;
    }
    let mean_diff = if lhs.is_empty() {
        0.0
    } else {
        sum_diff / lhs.len() as f32
    };
    Ok((max_diff, mean_diff))
}

fn array_to_vec_u32(array: &Array) -> Result<Vec<u32>, Exception> {
    let contiguous = array.contiguous()?;
    eval([&contiguous])?;
    Ok(contiguous.as_slice::<u32>().to_vec())
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
    let mut draft = PathBuf::from("models/Qwen3.6-35B-A3B-DFlash");
    let mut prompt = DEFAULT_PROMPT.to_string();
    let mut max_tokens = 40usize;
    let mut temp = 0.0f32;
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
                draft = PathBuf::from(
                    args.next()
                        .ok_or_else(|| anyhow!("--draft requires a path"))?,
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
            "--cpu" => cpu = true,
            other => return Err(anyhow!("unknown argument: {other}")),
        }
    }

    if let Err(err) = DraftCheckpointInfo::load(&draft) {
        return Err(anyhow!("failed to inspect draft checkpoint {}: {err}", draft.display()));
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
