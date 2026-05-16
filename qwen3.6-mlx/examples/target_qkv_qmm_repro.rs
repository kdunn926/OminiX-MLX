use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};
use mlx_rs::{
    module::Module,
    ops::{dequantize, indexing::IndexOp, matmul, quantized_matmul},
    quantization::MaybeQuantized,
    Array, Dtype,
};
use qwen3_6_mlx::{load_model, model::AttentionLayer};

#[derive(Debug)]
struct Args {
    target: PathBuf,
    tokens: Vec<u32>,
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

    let report = match &mut layer.attention {
        AttentionLayer::LinearAttention(delta) => {
            let ql = match &mut delta.in_proj_qkv {
                MaybeQuantized::Quantized(ql) => ql,
                MaybeQuantized::Original(_) => {
                    return Err(anyhow!("layer0 in_proj_qkv is not quantized"));
                }
            };
            run_qmm_repro(&layer0_input_norm, ql)?
        }
        AttentionLayer::FullAttention(_) => {
            return Err(anyhow!("layer 0 is not linear attention"));
        }
    };

    println!("tokens={:?}", args.tokens);
    println!("seq_len={}", args.tokens.len());
    println!("{}", report);
    Ok(())
}

fn run_qmm_repro(x: &Array, ql: &mut mlx_rs::nn::QuantizedLinear) -> Result<String> {
    let direct = ql.forward(x)?;
    let qmm_direct = quantized_matmul(
        x,
        &ql.inner.weight,
        &ql.scales,
        ql.biases.as_ref(),
        true,
        ql.group_size,
        ql.bits,
        None::<&str>,
    )?;

    let x_contig = x.contiguous()?;
    let qmm_contig = quantized_matmul(
        &x_contig,
        &ql.inner.weight,
        &ql.scales,
        ql.biases.as_ref(),
        true,
        ql.group_size,
        ql.bits,
        None::<&str>,
    )?;

    let seq_len = x.shape()[1];
    let hidden = x.shape()[2];
    let flat = x_contig.reshape(&[seq_len, hidden])?;
    let qmm_flat = quantized_matmul(
        &flat,
        &ql.inner.weight,
        &ql.scales,
        ql.biases.as_ref(),
        true,
        ql.group_size,
        ql.bits,
        None::<&str>,
    )?
    .reshape(&[1, seq_len, -1])?;

    let deq = dequantize(
        &ql.inner.weight,
        &ql.scales,
        ql.biases.as_ref(),
        ql.group_size,
        ql.bits,
        None::<&str>,
    )?;
    let deq_t = deq.t();
    let ref_matmul = matmul(&x.as_dtype(Dtype::Float32)?, &deq_t)?;

    let ref_flat = matmul(&flat.as_dtype(Dtype::Float32)?, &deq_t)?.reshape(&[1, seq_len, -1])?;

    Ok(format!(
        concat!(
            "direct_vs_qmm_direct={}\n",
            "direct_vs_qmm_contig={}\n",
            "direct_vs_qmm_flat={}\n",
            "direct_vs_ref_matmul={}\n",
            "qmm_direct_vs_ref_matmul={}\n",
            "qmm_flat_vs_ref_flat={}\n",
            "sample_direct_head={}\n",
            "sample_qmm_flat_head={}\n",
            "sample_ref_head={}\n"
        ),
        fmt_stats(diff_stats(&direct, &qmm_direct)?),
        fmt_stats(diff_stats(&direct, &qmm_contig)?),
        fmt_stats(diff_stats(&direct, &qmm_flat)?),
        fmt_stats(diff_stats(&direct, &ref_matmul)?),
        fmt_stats(diff_stats(&qmm_direct, &ref_matmul)?),
        fmt_stats(diff_stats(&qmm_flat, &ref_flat)?),
        fmt_slice(&direct.index((0, 0, 0..8))),
        fmt_slice(&qmm_flat.index((0, 0, 0..8))),
        fmt_slice(&ref_matmul.index((0, 0, 0..8))),
    ))
}

fn diff_stats(lhs: &Array, rhs: &Array) -> Result<(f32, f32)> {
    let lhs = lhs.as_dtype(Dtype::Float32)?.contiguous()?;
    let rhs = rhs.as_dtype(Dtype::Float32)?.contiguous()?;
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

fn fmt_stats((max_diff, mean_diff): (f32, f32)) -> String {
    format!("max:{max_diff:.6} mean:{mean_diff:.6}")
}

fn fmt_slice(x: &Array) -> String {
    match x.as_dtype(Dtype::Float32).and_then(|a| a.contiguous()) {
        Ok(v) => {
            let vals = v
                .as_slice::<f32>()
                .iter()
                .map(|v| format!("{v:.6}"))
                .collect::<Vec<_>>();
            format!("[{}]", vals.join(", "))
        }
        Err(_) => "<unavailable>".to_string(),
    }
}

fn parse_args() -> Result<Args> {
    let mut target = PathBuf::from("models/Qwen3.6-35B-A3B-4bit");
    let mut tokens = None;
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
            "--cpu" => cpu = true,
            other => return Err(anyhow!("unknown argument: {other}")),
        }
    }

    Ok(Args {
        target,
        tokens: tokens.ok_or_else(|| anyhow!("--tokens is required"))?,
        cpu,
    })
}
