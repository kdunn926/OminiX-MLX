use std::{collections::HashMap, fs, path::PathBuf};

use anyhow::{anyhow, Context, Result};
use mlx_rs::{
    Dtype,
    module::Module,
    ops::{concatenate_axis, indexing::IndexOp, zeros_dtype},
    Array,
};
use qwen3_6_mlx::{load_model, model::AttentionLayer};

const EXACT_SMALL_PROJ_QKV_PAD_M: i32 = 6;
const EXACT_SMALL_PROJ_Z_PAD_M: i32 = 10;

#[derive(Debug)]
struct Args {
    target: PathBuf,
    tokens: Vec<u32>,
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

    let token_ids: Vec<i32> = args.tokens.iter().map(|id| *id as i32).collect();
    let input = Array::from_slice(&token_ids, &[1, token_ids.len() as i32]);
    let mut model = load_model(&args.target)?;
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

    let mut tensors = HashMap::new();
    tensors.insert("tokens".to_string(), input);
    tensors.insert("embeddings".to_string(), embeddings.as_dtype(Dtype::Float32)?);
    tensors.insert(
        "layer0_input_norm".to_string(),
        layer0_input_norm.as_dtype(Dtype::Float32)?,
    );
    tensors.insert("qkv_direct".to_string(), qkv_direct.as_dtype(Dtype::Float32)?);
    tensors.insert("qkv_contig".to_string(), qkv_contig.as_dtype(Dtype::Float32)?);
    tensors.insert("qkv_exact".to_string(), qkv_exact.as_dtype(Dtype::Float32)?);
    tensors.insert("z_direct".to_string(), z_direct.as_dtype(Dtype::Float32)?);
    tensors.insert("z_contig".to_string(), z_contig.as_dtype(Dtype::Float32)?);
    tensors.insert("z_exact".to_string(), z_exact.as_dtype(Dtype::Float32)?);

    if let Some(parent) = args.out.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
    }

    let metadata = HashMap::from([("target".to_string(), args.target.display().to_string())]);
    Array::save_safetensors(&tensors, Some(&metadata), &args.out)
        .with_context(|| format!("failed to save {}", args.out.display()))?;
    println!("saved={}", args.out.display());
    Ok(())
}

fn exact_small_proj_with_pad_m<F>(x: &Array, pad_m: i32, mut forward: F) -> Result<Array>
where
    F: FnMut(&Array) -> Result<Array, mlx_rs::error::Exception>,
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

fn parse_args() -> Result<Args> {
    let mut target = PathBuf::from("models/Qwen3.6-35B-A3B-4bit");
    let mut tokens = None;
    let mut out = PathBuf::from("/tmp/qwen_target_layer0_token_dump.safetensors");
    let mut cpu = false;

    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--target" => {
                target =
                    PathBuf::from(it.next().ok_or_else(|| anyhow!("missing value for --target"))?);
            }
            "--tokens" => {
                let raw = it.next().ok_or_else(|| anyhow!("missing value for --tokens"))?;
                let parsed: Result<Vec<u32>, _> = raw
                    .split(',')
                    .filter(|part| !part.is_empty())
                    .map(|part| part.parse::<u32>())
                    .collect();
                let parsed = parsed.context("invalid --tokens list")?;
                if parsed.is_empty() {
                    return Err(anyhow!("--tokens must not be empty"));
                }
                tokens = Some(parsed);
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
        tokens: tokens.ok_or_else(|| anyhow!("--tokens is required"))?,
        out,
        cpu,
    })
}
