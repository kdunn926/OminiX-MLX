use std::{collections::HashMap, fs, path::PathBuf};

use anyhow::{anyhow, Context, Result};
use mlx_rs::module::Module;
use mlx_rs::ops::concatenate_axis;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::Array;
use qwen3_6_mlx::{
    cache::RecurrentState,
    load_model, load_tokenizer,
    model::AttentionLayer,
};

const DEFAULT_PROMPT: &str =
    "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n";

#[derive(Debug)]
struct Args {
    target: PathBuf,
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
    let mut model = load_model(&args.target)?;
    let embeddings = model.text_model.embed_tokens.forward(&prompt)?;
    let layer0_input_norm = match model.text_model.layers.get_mut(0) {
        Some(layer) => layer.input_layernorm.forward(&embeddings)?,
        None => return Err(anyhow!("model has no layers")),
    };

    let mut cache = RecurrentState::new();
    let (debug, qkv_direct, qkv_contig, qkv_padded, z_direct, z_contig, z_padded) =
        match model.text_model.layers.get_mut(0) {
        Some(layer) => match &mut layer.attention {
            AttentionLayer::LinearAttention(delta) => {
                let input_contig = layer0_input_norm.contiguous()?;
                let qkv_direct = delta.in_proj_qkv.forward(&layer0_input_norm)?;
                let qkv_contig = delta.in_proj_qkv.forward(&input_contig)?;
                let z_direct = delta.in_proj_z.forward(&layer0_input_norm)?;
                let z_contig = delta.in_proj_z.forward(&input_contig)?;

                let seq_len = layer0_input_norm.shape()[1];
                let hidden_dim = layer0_input_norm.shape()[2];
                let padded_input = if seq_len < 16 {
                    let pad = mlx_rs::ops::zeros_dtype(
                        &[layer0_input_norm.shape()[0], 16 - seq_len, hidden_dim],
                        layer0_input_norm.dtype(),
                    )?;
                    concatenate_axis(&[&layer0_input_norm, &pad], 1)?
                } else {
                    layer0_input_norm.clone()
                };
                let qkv_padded_full = delta.in_proj_qkv.forward(&padded_input)?;
                let z_padded_full = delta.in_proj_z.forward(&padded_input)?;
                let qkv_padded = qkv_padded_full.index((.., ..seq_len, ..));
                let z_padded = z_padded_full.index((.., ..seq_len, ..));

                (
                    delta.debug_prefill_tensors(&layer0_input_norm, &mut cache)?,
                    qkv_direct,
                    qkv_contig,
                    qkv_padded,
                    z_direct,
                    z_contig,
                    z_padded,
                )
            }
            AttentionLayer::FullAttention(_) => {
                return Err(anyhow!("layer 0 is not linear attention"));
            }
        },
        None => return Err(anyhow!("model has no layers")),
    };

    let mut tensors = HashMap::new();
    tensors.insert("prompt_ids".to_string(), prompt);
    tensors.insert("embeddings".to_string(), embeddings);
    tensors.insert("layer0_input_norm".to_string(), layer0_input_norm);
    tensors.insert("qkv_direct".to_string(), qkv_direct);
    tensors.insert("qkv_contig".to_string(), qkv_contig);
    tensors.insert("qkv_padded".to_string(), qkv_padded);
    tensors.insert("z_direct".to_string(), z_direct);
    tensors.insert("z_contig".to_string(), z_contig);
    tensors.insert("z_padded".to_string(), z_padded);
    for (name, value) in debug {
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
        ("prompt".to_string(), args.prompt),
    ]);
    Array::save_safetensors(&tensors, Some(&metadata), &args.out)
        .with_context(|| format!("failed to save {}", args.out.display()))?;
    eprintln!("Saved layer-0 target debug tensors to {}", args.out.display());
    Ok(())
}

fn parse_args() -> Result<Args> {
    let mut target = PathBuf::from("models/Qwen3.6-35B-A3B-4bit");
    let mut prompt = DEFAULT_PROMPT.to_string();
    let mut out = PathBuf::from("/tmp/qwen_target_layer0_debug.safetensors");
    let mut cpu = false;

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
            "--cpu" => cpu = true,
            other => return Err(anyhow!("unknown argument: {other}")),
        }
    }

    Ok(Args {
        target,
        prompt,
        out,
        cpu,
    })
}
