//! DiffusionGemma block-diffusion text generation.
//!
//! Usage:
//!   cargo run --release -p gemma4-mlx --example diffusiongemma_chat -- \
//!     models/diffusiongemma-26B-A4B-it-4bit "Why is the sky blue?" [max_tokens]
//!
//! Env:
//!   DIFFUSION_STEPS=N   override max denoising steps (default from config, 48)
//!   DIFFUSION_RAW=1     skip the chat template (raw continuation)
//!   DIFFUSION_SEED=N    seed MLX RNG (canvas init is random)

use std::env;
use std::path::PathBuf;

use anyhow::{anyhow, Result};

use gemma4_mlx::diffusion::{
    diffusion_generate, load_diffusion_model, DiffusionGenerateOptions,
};
use gemma4_mlx::{load_tokenizer, Gemma4ChatTemplate, Gemma4Message};

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let model_dir = args
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/diffusiongemma-26B-A4B-it-4bit"));
    let prompt = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "Why is the sky blue?".to_string());
    let max_tokens: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(256);

    if let Ok(seed) = env::var("DIFFUSION_SEED") {
        let seed: u64 = seed.parse().map_err(|_| anyhow!("bad DIFFUSION_SEED"))?;
        mlx_rs::random::seed(seed).map_err(|e| anyhow!("seed: {e}"))?;
    }

    let tokenizer = load_tokenizer(&model_dir)?;
    let use_chat = env::var("DIFFUSION_RAW").is_err();
    let rendered = if use_chat {
        let template = Gemma4ChatTemplate::load(&model_dir)?;
        template.render_prompt(&[Gemma4Message::user(&prompt)], &[], true)?
    } else {
        prompt.clone()
    };
    let enc = tokenizer
        .encode(rendered.as_str(), !use_chat)
        .map_err(|e| anyhow!("encode: {e}"))?;
    let prompt_ids: Vec<i32> = enc.get_ids().iter().map(|&i| i as i32).collect();

    eprintln!("[diffusion] loading {} ...", model_dir.display());
    let t0 = std::time::Instant::now();
    let mut model = load_diffusion_model(&model_dir).map_err(|e| anyhow!("load: {e}"))?;
    eprintln!(
        "[diffusion] loaded in {:.1}s  ({} layers, canvas {}, vocab {})",
        t0.elapsed().as_secs_f64(),
        model.layers.len(),
        model.canvas_length(),
        model.vocab_size(),
    );

    let mut opts = DiffusionGenerateOptions::from_config(&model.config);
    opts.max_tokens = max_tokens;
    if let Ok(steps) = env::var("DIFFUSION_STEPS") {
        opts.max_denoising_steps = steps.parse().map_err(|_| anyhow!("bad DIFFUSION_STEPS"))?;
    }
    let eos: Vec<i32> = if model.config.eos_token_id.is_empty() {
        vec![1, 106, 50]
    } else {
        model.config.eos_token_id.iter().map(|&v| v as i32).collect()
    };
    eprintln!(
        "[diffusion] prompt {} tok | max_tokens {} steps {} entropy_bound {} t {}..{}",
        prompt_ids.len(),
        opts.max_tokens,
        opts.max_denoising_steps,
        opts.entropy_bound,
        opts.t_min,
        opts.t_max,
    );

    let tok_stream = tokenizer.clone();
    let (emitted, stats) = diffusion_generate(&mut model, &prompt_ids, &opts, &eos, |block| {
        if block.is_empty() {
            return;
        }
        let ids: Vec<u32> = block.iter().map(|&t| t as u32).collect();
        if let Ok(text) = tok_stream.decode(&ids, true) {
            print!("{text}");
            use std::io::Write;
            let _ = std::io::stdout().flush();
        }
    })
    .map_err(|e| anyhow!("generate: {e}"))?;
    println!();

    eprintln!(
        "\n[diffusion] prefill {:.2}s | {} tok in {:.2}s = {:.2} tok/s | {} canvases, {} denoise steps ({:.1} steps/canvas), {} work tokens ({:.0} work tok/s)",
        stats.prefill_s,
        emitted.len(),
        stats.decode_s,
        stats.tokens_per_sec(),
        stats.canvases,
        stats.denoise_steps,
        stats.denoise_steps as f64 / stats.canvases.max(1) as f64,
        stats.work_tokens,
        stats.work_tokens as f64 / stats.decode_s.max(1e-9),
    );
    Ok(())
}
