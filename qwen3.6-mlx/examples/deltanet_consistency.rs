use anyhow::{anyhow, Result};
use mlx_rs::{
    module::Module,
    ops::{concatenate_axis, indexing::IndexOp},
};
use qwen3_6_mlx::{
    cache::RecurrentState,
    load_model, load_tokenizer,
    model::AttentionLayer,
};

const DEFAULT_PROMPT: &str =
    "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n";

fn main() -> Result<()> {
    let use_cpu = std::env::args().any(|a| a == "--cpu");
    if use_cpu {
        mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
        eprintln!("Using CPU device");
    } else {
        eprintln!("Using GPU device");
    }

    let args: Vec<String> = std::env::args().filter(|a| a != "--cpu").collect();
    let model_dir = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "models/Qwen3.6-35B-A3B-4bit".to_string());
    let prompt = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| DEFAULT_PROMPT.to_string());

    let tokenizer = load_tokenizer(&model_dir)?;
    let encoding = tokenizer
        .encode(prompt.as_str(), false)
        .map_err(|e| anyhow!(e.to_string()))?;
    let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
    let input_ids = mlx_rs::Array::from_slice(&prompt_ids, &[1, prompt_ids.len() as i32]);

    let mut prefill_model = load_model(&model_dir)?;
    let mut step_model = load_model(&model_dir)?;

    let prefill_embeds = prefill_model.text_model.embed_tokens.forward(&input_ids)?;
    let step_embeds = step_model.text_model.embed_tokens.forward(&input_ids)?;

    let mut prefill_cache = RecurrentState::new();
    let mut step_cache = RecurrentState::new();

    let prefill_layer = match prefill_model.text_model.layers.get_mut(0) {
        Some(layer) => layer,
        None => return Err(anyhow!("model has no layers")),
    };
    let step_layer = match step_model.text_model.layers.get_mut(0) {
        Some(layer) => layer,
        None => return Err(anyhow!("model has no layers")),
    };

    let prefill_out = match &mut prefill_layer.attention {
        AttentionLayer::LinearAttention(delta) => delta.forward_prefill(&prefill_embeds, &mut prefill_cache)?,
        AttentionLayer::FullAttention(_) => return Err(anyhow!("layer 0 is not linear attention")),
    };

    let mut step_outputs = Vec::with_capacity(prompt_ids.len());
    match &mut step_layer.attention {
        AttentionLayer::LinearAttention(delta) => {
            for t in 0..prompt_ids.len() as i32 {
                let token_embed = step_embeds.index((.., t..t + 1, ..));
                let out = delta.forward_step(&token_embed, &mut step_cache)?;
                step_outputs.push(out);
            }
        }
        AttentionLayer::FullAttention(_) => return Err(anyhow!("layer 0 is not linear attention")),
    }
    let step_refs = step_outputs.iter().collect::<Vec<_>>();
    let step_out = concatenate_axis(&step_refs, 1)?;

    let out_diff = prefill_out
        .subtract(&step_out)?
        .abs()?
        .as_dtype(mlx_rs::Dtype::Float32)?;
    let out_max = out_diff.max(None)?.item::<f32>();
    let out_mean = out_diff.mean(None)?.item::<f32>();

    let state_diff = match (&prefill_cache.state, &step_cache.state) {
        (Some(a), Some(b)) => {
            let diff = a.subtract(b)?.abs()?.as_dtype(mlx_rs::Dtype::Float32)?;
            Some((diff.max(None)?.item::<f32>(), diff.mean(None)?.item::<f32>()))
        }
        _ => None,
    };

    println!("layer0_output max_diff={out_max:.6} mean_diff={out_mean:.6}");
    if let Some((state_max, state_mean)) = state_diff {
        println!("layer0_state max_diff={state_max:.6} mean_diff={state_mean:.6}");
    }

    Ok(())
}
