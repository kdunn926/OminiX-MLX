//! Sliding-window KV cache benchmark.
//!
//! Compares end-to-end throughput and peak MLX active memory between:
//!   - **Baseline**: `Vec<KVCache>` (unbounded). Every layer — including
//!     the 40 (12B) / 35 (e4b) / 25 (26B-A4B) sliding-attention layers —
//!     stores full-length KV; sliding behavior is mask-only.
//!   - **Layered**: `Vec<MixedKvCache>` built via [`init_layered_cache`].
//!     Sliding-attention layers use `SlidingKVCache` which physically
//!     caps the buffer at the model's per-type `sliding_window`.
//!
//! For each `(mode, prompt_len, gen_len)` combination, the bench reports:
//!   - tokens-per-second during the decode phase (post-prefill)
//!   - peak `mlx_get_active_memory` observed during the run
//!
//! Run:
//! ```
//! cargo run --release -p gemma4-mlx --example sliding_bench -- \
//!     models/gemma-4-12B-it-4bit
//! ```
//!
//! Override the prompt/gen lengths via env:
//!   BENCH_PROMPT_LENS=1024,4096,16384,32768
//!   BENCH_GEN_TOKENS=128

use std::{env, error::Error, path::PathBuf, time::Instant};

use gemma4_mlx::{
    init_cache, init_layered_cache, load_model, ud_loader::load_ud_mlx_4bit, KVCache,
    KeyValueCache, MixedKvCache, Model, ModelInput,
};
use mlx_rs::{
    argmax_axis,
    module::Module,
    ops::indexing::{IndexOp, NewAxis},
    transforms::eval,
    Array, Dtype,
};

#[derive(Debug, Clone, Copy)]
enum Mode {
    Baseline,
    Layered,
}

impl Mode {
    fn label(self) -> &'static str {
        match self {
            Mode::Baseline => "baseline (unbounded Vec<KVCache>)",
            Mode::Layered => "layered (sliding-trim Vec<MixedKvCache>)",
        }
    }
}

fn active_memory_bytes() -> usize {
    let mut bytes: usize = 0;
    unsafe {
        let _ = mlx_sys::mlx_get_active_memory(&mut bytes);
    }
    bytes
}

fn mb(bytes: usize) -> f64 {
    bytes as f64 / 1024.0 / 1024.0
}

/// Run one decode session: prefill `prompt_len` tokens, then generate
/// `gen_len` tokens. Returns (decode_tok_per_sec, peak_active_memory_bytes).
fn run_one<C>(
    model: &mut Model,
    cache: &mut Vec<C>,
    prompt_len: usize,
    gen_len: usize,
) -> Result<(f64, usize), Box<dyn Error + Send + Sync>>
where
    C: KeyValueCache + Default,
{
    // Synthetic prompt: token id 1 repeated. Content doesn't matter for a
    // pure throughput bench — we're measuring the cache + attention path,
    // not generation quality. (We deliberately bypass the chat template
    // here; this is a forward-pass microbench.)
    let prompt_ids: Vec<i32> = vec![1; prompt_len];
    let prompt_arr = Array::from_slice(&prompt_ids, &[prompt_ids.len() as i32]).index(NewAxis);

    // Track peak active memory across the whole run (prefill + decode).
    let mut peak = active_memory_bytes();

    // Prefill — chunked through `GEMMA4_PREFILL_CHUNK` (default 256) by
    // the model forward when L > chunk; we just hand the whole prompt and
    // let the forward chunk. For huge prompts a manual chunk loop would
    // bound the prefill transient — kept simple here.
    let input = ModelInput {
        inputs: &prompt_arr,
        mask: None,
        cache,
    };
    let logits = model.forward(input)?;
    eval([&logits])?;
    peak = peak.max(active_memory_bytes());

    // Greedy: take argmax of last position. Suppresses sampler overhead so
    // the bench reflects forward-pass cost.
    let last = logits.index((.., -1, ..));
    let mut next = argmax_axis!(last, -1)?.as_dtype(Dtype::Int32)?;
    eval([&next])?;

    let t_start = Instant::now();
    for _ in 0..gen_len {
        let next_arr = next.index((.., NewAxis));
        let input = ModelInput {
            inputs: &next_arr,
            mask: None,
            cache,
        };
        let logits = model.forward(input)?;
        let last = logits.index((.., -1, ..));
        next = argmax_axis!(last, -1)?.as_dtype(Dtype::Int32)?;
        eval([&next])?;
        peak = peak.max(active_memory_bytes());
    }
    let elapsed = t_start.elapsed().as_secs_f64();
    let tok_per_sec = gen_len as f64 / elapsed.max(1e-9);
    Ok((tok_per_sec, peak))
}

fn parse_csv_usize(var: &str, default: &[usize]) -> Vec<usize> {
    env::var(var)
        .ok()
        .map(|s| {
            s.split(',')
                .filter_map(|t| t.trim().parse::<usize>().ok())
                .filter(|&n| n > 0)
                .collect::<Vec<_>>()
        })
        .filter(|v: &Vec<usize>| !v.is_empty())
        .unwrap_or_else(|| default.to_vec())
}

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let args: Vec<String> = env::args().collect();
    let model_dir: PathBuf = args
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/gemma-4-12B-it-4bit"));

    // Reasonable defaults: spans short context (where any per-step trim
    // overhead would dominate) through long context (where the trim
    // savings should compound).
    let prompt_lens = parse_csv_usize("BENCH_PROMPT_LENS", &[1024, 4096, 16384]);
    let gen_len = parse_csv_usize("BENCH_GEN_TOKENS", &[128])[0];

    eprintln!("[sliding_bench] model: {}", model_dir.display());
    eprintln!(
        "[sliding_bench] prompt_lens={:?}  gen_tokens={}",
        prompt_lens, gen_len
    );

    // Auto-detect quant variant the same way `chat_text` does.
    let is_quantized = {
        let cfg_path = model_dir.join("config.json");
        let raw: serde_json::Value =
            serde_json::from_reader(std::fs::File::open(&cfg_path)?)?;
        raw.get("quantization")
            .map(|q| q.is_object())
            .unwrap_or(false)
    };
    eprintln!(
        "[sliding_bench] loader: {}",
        if is_quantized {
            "load_ud_mlx_4bit"
        } else {
            "load_model (bf16)"
        }
    );
    let mut model = if is_quantized {
        load_ud_mlx_4bit(&model_dir)?
    } else {
        load_model(&model_dir)?
    };

    // Establish a memory baseline AFTER weights are loaded — the per-mode
    // peaks below subtract this so the reported memory is "what the cache
    // + attention transients added on top of weights".
    let weights_baseline = active_memory_bytes();
    eprintln!(
        "[sliding_bench] post-load active memory: {:.1} MB (weights baseline)",
        mb(weights_baseline)
    );

    println!();
    println!(
        "{:<10} {:>10} {:>10} {:>12} {:>14}",
        "mode", "prompt", "gen_tok/s", "peak_MB", "Δ_vs_weights"
    );
    println!("{}", "-".repeat(60));

    for &p_len in &prompt_lens {
        for &mode in &[Mode::Baseline, Mode::Layered] {
            // Fresh cache per run — no prefix-cache reuse; we're measuring
            // the cold-prefill + steady-decode shape, not warm-cache hits.
            let (tok_per_sec, peak) = match mode {
                Mode::Baseline => {
                    let num_slots =
                        *model.model.kv_cache_map.iter().max().unwrap_or(&0) + 1;
                    let mut cache: Vec<KVCache> = init_cache(num_slots);
                    run_one(&mut model, &mut cache, p_len, gen_len)?
                }
                Mode::Layered => {
                    let mut cache: Vec<MixedKvCache> = init_layered_cache(&model);
                    run_one(&mut model, &mut cache, p_len, gen_len)?
                }
            };
            let delta = peak.saturating_sub(weights_baseline);
            println!(
                "{:<10} {:>10} {:>10.2} {:>12.1} {:>14.1}",
                match mode {
                    Mode::Baseline => "baseline",
                    Mode::Layered => "layered",
                },
                p_len,
                tok_per_sec,
                mb(peak),
                mb(delta),
            );
        }
        println!();
    }

    eprintln!("[sliding_bench] modes: baseline = {}; layered = {}", Mode::Baseline.label(), Mode::Layered.label());
    Ok(())
}
