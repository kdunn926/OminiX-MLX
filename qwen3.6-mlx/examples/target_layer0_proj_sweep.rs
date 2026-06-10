use std::{collections::HashMap, fs, path::PathBuf};

use anyhow::{anyhow, Context, Result};
use mlx_rs::module::Module;
use mlx_rs::ops::{concatenate_axis, indexing::IndexOp, zeros_dtype};
use mlx_rs::Array;
use qwen3_6_mlx::{load_model, load_tokenizer, model::AttentionLayer};

const DEFAULT_PROMPT: &str = "The theory of general relativity";

#[derive(Debug)]
struct Args {
    target: PathBuf,
    prompt: String,
    out: PathBuf,
    cpu: bool,
    max_len: i32,
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
    let mut model = load_model(&args.target)?;
    let embeddings = model.text_model.embed_tokens.forward(&prompt)?;
    let layer0_input_norm = match model.text_model.layers.get_mut(0) {
        Some(layer) => layer.input_layernorm.forward(&embeddings)?,
        None => return Err(anyhow!("model has no layers")),
    };

    let mut tensors = HashMap::new();
    tensors.insert("prompt_ids".to_string(), prompt);
    tensors.insert("layer0_input_norm_full".to_string(), layer0_input_norm.clone());

    let max_len = args.max_len.min(layer0_input_norm.shape()[1]).max(1);
    match model.text_model.layers.get_mut(0) {
        Some(layer) => match &mut layer.attention {
            AttentionLayer::LinearAttention(delta) => {
                for seq_len in 1..=max_len {
                    let prefix = layer0_input_norm.index((.., ..seq_len, ..));
                    let prefix_contig = prefix.contiguous()?;
                    let pad = if seq_len < 16 {
                        let zeros = zeros_dtype(
                            &[prefix.shape()[0], 16 - seq_len, prefix.shape()[2]],
                            prefix.dtype(),
                        )?;
                        concatenate_axis(&[&prefix, &zeros], 1)?
                    } else {
                        prefix.clone()
                    };

                    let qkv_direct = delta.in_proj_qkv.forward(&prefix)?;
                    let qkv_contig = delta.in_proj_qkv.forward(&prefix_contig)?;
                    let qkv_padded = delta.in_proj_qkv.forward(&pad)?.index((.., ..seq_len, ..));
                    let z_direct = delta.in_proj_z.forward(&prefix)?;
                    let z_contig = delta.in_proj_z.forward(&prefix_contig)?;
                    let z_padded = delta.in_proj_z.forward(&pad)?.index((.., ..seq_len, ..));

                    tensors.insert(format!("len_{seq_len}_prefix"), prefix);
                    tensors.insert(format!("len_{seq_len}_qkv_direct"), qkv_direct);
                    tensors.insert(format!("len_{seq_len}_qkv_contig"), qkv_contig);
                    tensors.insert(format!("len_{seq_len}_qkv_padded"), qkv_padded);
                    tensors.insert(format!("len_{seq_len}_z_direct"), z_direct);
                    tensors.insert(format!("len_{seq_len}_z_contig"), z_contig);
                    tensors.insert(format!("len_{seq_len}_z_padded"), z_padded);
                }
            }
            AttentionLayer::FullAttention(_) => {
                return Err(anyhow!("layer 0 is not linear attention"));
            }
        },
        None => return Err(anyhow!("model has no layers")),
    }

    if let Some(parent) = args.out.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
    }

    let metadata = HashMap::from([
        ("target".to_string(), args.target.display().to_string()),
        ("prompt".to_string(), args.prompt),
    ]);
    Array::save_safetensors(&tensors, Some(&metadata), &args.out)
        .with_context(|| format!("failed to save {}", args.out.display()))?;
    eprintln!("Saved layer-0 projection sweep tensors to {}", args.out.display());
    Ok(())
}

fn parse_args() -> Result<Args> {
    let mut target = PathBuf::from("models/Qwen3.6-35B-A3B-4bit");
    let mut prompt = DEFAULT_PROMPT.to_string();
    let mut out = PathBuf::from("/tmp/qwen_target_layer0_proj_sweep.safetensors");
    let mut cpu = false;
    let mut max_len = 16;

    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--target" => {
                target = PathBuf::from(it.next().ok_or_else(|| anyhow!("missing value for --target"))?);
            }
            "--prompt" => {
                prompt = it.next().ok_or_else(|| anyhow!("missing value for --prompt"))?;
            }
            "--out" => {
                out = PathBuf::from(it.next().ok_or_else(|| anyhow!("missing value for --out"))?);
            }
            "--max-len" => {
                max_len = it
                    .next()
                    .ok_or_else(|| anyhow!("missing value for --max-len"))?
                    .parse()
                    .context("invalid --max-len")?;
            }
            "--cpu" => cpu = true,
            other => return Err(anyhow!("unknown argument: {other}")),
        }
    }

    Ok(Args {
        target,
        prompt,
        out,
        cpu,
        max_len,
    })
}
