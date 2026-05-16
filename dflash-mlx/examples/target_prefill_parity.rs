use std::{collections::HashMap, fs, path::PathBuf};

use anyhow::{anyhow, Context, Result};
use mlx_rs::module::Module;
use mlx_rs::Array;
use qwen3_6_mlx::{load_model, load_tokenizer};

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
    let capture_layer_ids: Vec<usize> = (0..model.text_model.layers.len()).collect();
    let (prefill_logits, all_hidden) =
        model.forward_last_logits_with_hidden_capture(&prompt, &mut Vec::new(), &capture_layer_ids)?;

    let mut tensors = HashMap::new();
    tensors.insert("prompt_ids".to_string(), prompt);
    tensors.insert("embeddings".to_string(), embeddings);
    tensors.insert("layer0_input_norm".to_string(), layer0_input_norm);
    tensors.insert("prefill_logits".to_string(), prefill_logits);
    tensors.insert("all_hidden".to_string(), all_hidden);
    tensors.insert(
        "num_layers".to_string(),
        Array::from_slice(&[capture_layer_ids.len() as i32], &[1]),
    );

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
    eprintln!("Saved target parity tensors to {}", args.out.display());
    Ok(())
}

fn parse_args() -> Result<Args> {
    let mut target = PathBuf::from("models/Qwen3.6-35B-A3B-4bit");
    let mut prompt = DEFAULT_PROMPT.to_string();
    let mut out = PathBuf::from("/tmp/qwen_target_prefill_parity.safetensors");
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
