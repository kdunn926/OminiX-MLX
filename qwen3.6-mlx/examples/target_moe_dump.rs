//! Dump layer-K MoE routing (gates, top-k indices, top-k scores, final output)
//! on a fixed input — bypasses the prefill stack so we can isolate MoE-specific
//! divergence from Python.
//!
//! Uses layer-0's `input_layernorm(embed_tokens(prompt))` as the MoE input —
//! that tensor is bit-identical between Rust and Python at mlx 0.31.2, so any
//! divergence in the dumped MoE outputs is attributable to the MoE block.
//!
//! Usage:
//!   target/release/examples/target_moe_dump --target models/Qwen3.6-35B-A3B-4bit \
//!     --layer 1 --out /tmp/qwen_moe_layer1_rs.safetensors

use std::{collections::HashMap, fs, path::PathBuf};

use anyhow::{anyhow, Context, Result};
use mlx_rs::{
    module::Module, ops::argpartition_axis, ops::indexing::IndexOp, ops::indexing::take_along_axis,
    ops::softmax_axis, Array, Dtype,
};
use qwen3_6_mlx::{load_model, load_tokenizer, model::FfnBlock};

const DEFAULT_PROMPT: &str =
    "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n";

fn main() -> Result<()> {
    let mut target = PathBuf::from("models/Qwen3.6-35B-A3B-4bit");
    let mut out = PathBuf::from("/tmp/qwen_moe_dump_rs.safetensors");
    let mut layer_idx: usize = 1;
    let mut prompt = DEFAULT_PROMPT.to_string();

    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--target" => target = PathBuf::from(it.next().context("missing --target value")?),
            "--out" => out = PathBuf::from(it.next().context("missing --out value")?),
            "--layer" => layer_idx = it.next().context("missing --layer value")?.parse()?,
            "--prompt" => prompt = it.next().context("missing --prompt value")?,
            other => return Err(anyhow!("unknown argument: {other}")),
        }
    }

    let tokenizer = load_tokenizer(&target)?;
    let encoding = tokenizer
        .encode(prompt.as_str(), false)
        .map_err(|e| anyhow!(e.to_string()))?;
    let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
    let prompt_arr = Array::from_slice(&prompt_ids, &[1, prompt_ids.len() as i32]);

    let mut model = load_model(&target)?;
    let embeddings = model.text_model.embed_tokens.forward(&prompt_arr)?;
    let l0_input_norm = model
        .text_model
        .layers
        .get_mut(0)
        .ok_or_else(|| anyhow!("no layers"))?
        .input_layernorm
        .forward(&embeddings)?;

    // Use the bit-exact l0_input_norm as the synthetic MoE input.
    let mlp_in = l0_input_norm;

    let num_layers = model.text_model.layers.len();
    let layer = model
        .text_model
        .layers
        .get_mut(layer_idx)
        .ok_or_else(|| anyhow!("model has only {num_layers} layers"))?;
    let moe = match &mut layer.ffn {
        FfnBlock::Moe(m) => m,
        _ => return Err(anyhow!("layer {layer_idx} is not MoE")),
    };

    let gates_raw = moe.gate.forward(&mlp_in)?;
    let gates = softmax_axis(&gates_raw, -1, true)?;
    let neg_gates = gates.negative()?;
    let partitioned = argpartition_axis(&neg_gates, moe.top_k - 1, -1)?;
    let top_k_indices = partitioned.index((.., .., ..moe.top_k));
    let top_k_scores_raw = take_along_axis(&gates, &top_k_indices, -1)?;
    let score_sum = top_k_scores_raw.sum_axis(-1, true)?;
    let top_k_scores = top_k_scores_raw.divide(&score_sum)?;
    let moe_out = moe.forward(&mlp_in)?;

    let mut tensors = HashMap::new();
    tensors.insert("mlp_in".to_string(), mlp_in.as_dtype(Dtype::Float32)?);
    tensors.insert("gates_raw".to_string(), gates_raw.as_dtype(Dtype::Float32)?);
    tensors.insert("gates".to_string(), gates.as_dtype(Dtype::Float32)?);
    tensors.insert(
        "top_k_indices".to_string(),
        top_k_indices.as_dtype(Dtype::Int32)?,
    );
    tensors.insert(
        "top_k_scores".to_string(),
        top_k_scores.as_dtype(Dtype::Float32)?,
    );
    tensors.insert("moe_out".to_string(), moe_out.as_dtype(Dtype::Float32)?);

    if let Some(parent) = out.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).ok();
        }
    }
    Array::save_safetensors(
        &tensors,
        Some(&HashMap::from([("layer".to_string(), layer_idx.to_string())])),
        &out,
    )?;
    println!("saved={} layer={}", out.display(), layer_idx);
    Ok(())
}
