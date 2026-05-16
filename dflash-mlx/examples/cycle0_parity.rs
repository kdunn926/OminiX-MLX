use std::{collections::HashMap, fs, path::PathBuf};

use anyhow::{anyhow, Context, Result};
use dflash_mlx::{DFlashDraftModel, Qwen36TargetAdapter, TargetModel};
use mlx_rs::{
    argmax_axis,
    ops::{broadcast_to, concatenate_axis, indexing::IndexOp, matmul},
    Array, Dtype,
};
use qwen3_6_mlx::{load_model, load_tokenizer};

const DEFAULT_PROMPT: &str =
    "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n";

#[derive(Debug)]
struct Args {
    target: PathBuf,
    draft: PathBuf,
    prompt: String,
    out: PathBuf,
    cpu: bool,
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
    if prompt_ids.is_empty() {
        return Err(anyhow!("prompt tokenization produced no ids"));
    }

    let prompt = Array::from_slice(&prompt_ids, &[1, prompt_ids.len() as i32]);

    let mut target_model = load_model(&args.target)?;
    let mut draft_model = DFlashDraftModel::load_from_path(&args.draft)?;
    let target_layer_ids = draft_model.args.target_layer_ids();
    let block_len = draft_model.args.block_size();
    let mask_token_id = draft_model.args.mask_token_id();
    let lm_head_weight = target_model.get_lm_head_weight()?;
    let mask_embedding = target_model.embed_tokens(&[mask_token_id as i32])?;

    let mut target = Qwen36TargetAdapter::with_dflash(target_model, 0.0, target_layer_ids);
    let prefill_logits = target.prefill(&prompt)?;
    let staged_token = target
        .sample(&prefill_logits, 0.0)?
        .as_dtype(Dtype::Uint32)?
        .item::<u32>();
    let raw_target_hidden = target
        .last_target_hidden()
        .ok_or_else(|| anyhow!("target adapter did not retain hidden captures after prefill"))?;
    let staged_embedding = target
        .embed_token(staged_token)
        .ok_or_else(|| anyhow!("failed to embed staged token {staged_token}"))?;

    let noise_emb = build_noise_embedding(&staged_embedding, &mask_embedding, block_len)?;
    let projected_target_hidden = draft_model.project_target_hidden(&raw_target_hidden)?;
    let ctx_offset = raw_target_hidden.shape()[1] as usize;
    let first_layer_debug =
        draft_model.debug_first_layer(&noise_emb, &projected_target_hidden, ctx_offset)?;
    let draft_hidden =
        draft_model.forward_projected_context(&noise_emb, &projected_target_hidden, ctx_offset)?;

    let prediction_hidden = draft_hidden.index((.., 1.., ..));
    let lm_head_t = lm_head_weight.t();
    let draft_logits = matmul(&prediction_hidden, &lm_head_t)?;
    let draft_tokens = argmax_axis!(&draft_logits, -1)?.as_dtype(Dtype::Uint32)?;

    let mut tensors = HashMap::new();
    tensors.insert("prompt_ids".to_string(), prompt.clone());
    tensors.insert(
        "block_len".to_string(),
        Array::from_slice(&[block_len as i32], &[1]),
    );
    tensors.insert(
        "staged_token".to_string(),
        Array::from_slice(&[staged_token], &[1]),
    );
    tensors.insert("prefill_logits".to_string(), prefill_logits);
    tensors.insert("raw_target_hidden".to_string(), raw_target_hidden);
    tensors.insert(
        "projected_target_hidden".to_string(),
        projected_target_hidden,
    );
    tensors.insert("noise_emb".to_string(), noise_emb);
    tensors.insert("draft_hidden".to_string(), draft_hidden);
    tensors.insert("draft_logits".to_string(), draft_logits);
    tensors.insert("draft_tokens".to_string(), draft_tokens);
    for (name, value) in first_layer_debug {
        tensors.insert(name, value);
    }

    if let Some(parent) = args.out.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
    }

    let metadata = HashMap::from([
        ("target".to_string(), args.target.display().to_string()),
        ("draft".to_string(), args.draft.display().to_string()),
        ("prompt".to_string(), args.prompt),
    ]);
    Array::save_safetensors(&tensors, Some(&metadata), &args.out)
        .with_context(|| format!("failed to save {}", args.out.display()))?;

    println!("Saved Rust cycle-0 parity tensors to {}", args.out.display());
    println!("staged_token={staged_token} block_len={block_len}");

    Ok(())
}

fn build_noise_embedding(
    staged_embedding: &Array,
    mask_embedding: &Array,
    block_len: usize,
) -> Result<Array> {
    if block_len < 2 {
        return Err(anyhow!("block_len must be at least 2, got {block_len}"));
    }
    let hidden_size = mask_embedding.shape()[2];
    let mask_tail = broadcast_to(mask_embedding, &[1, block_len as i32 - 1, hidden_size])?;
    Ok(concatenate_axis(&[staged_embedding, &mask_tail], 1)?)
}

fn parse_args() -> Result<Args> {
    let mut args = std::env::args().skip(1);
    let mut target = PathBuf::from("models/Qwen3.6-35B-A3B-4bit");
    let mut draft = PathBuf::from("models/Qwen3.6-35B-A3B-DFlash");
    let mut prompt = DEFAULT_PROMPT.to_string();
    let mut out = PathBuf::from("/tmp/dflash_cycle0_rust.safetensors");
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
            "--out" => {
                out = PathBuf::from(
                    args.next()
                        .ok_or_else(|| anyhow!("--out requires a path"))?,
                )
            }
            "--cpu" => cpu = true,
            "--help" | "-h" => {
                println!(
                    "Usage: cargo run --release --example cycle0_parity -- [--target /path/to/Qwen3.6-35B-A3B-4bit] [--draft /path/to/Qwen3.6-35B-A3B-DFlash] [--prompt \"...\"] [--out /tmp/dflash_cycle0_rust.safetensors] [--cpu]"
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
        out,
        cpu,
    })
}
