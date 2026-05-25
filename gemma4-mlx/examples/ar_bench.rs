//! Minimal Rust AR throughput sanity check for Gemma4 (no DFlash).
//! Used to baseline the gemma4 verify-path bottleneck and to compare KV
//! backends. Set `PAGED_KV=1` to run full-attention layers through the paged
//! KV cache (fused paged-attention kernel); default is the standard fp16 cache.

use anyhow::Result;
use gemma4_mlx::mixed_cache::init_mixed_paged_cache;
use gemma4_mlx::{load_model, load_tokenizer, Generate, Model};
use mlx_rs::Array;
use mlx_rs_core::cache::{KVCache, KeyValueCache};
use std::time::Instant;

/// Run AR generation with a pre-built cache and report timing.
fn run_with_cache<C: KeyValueCache + Default>(
    model: &mut Model,
    mut cache: Vec<C>,
    max_tokens: usize,
    prompt: &Array,
) -> Result<(f64, f64, usize)> {
    let t0 = Instant::now();
    let mut prefill = 0.0;
    let mut count = 0usize;
    let gen = Generate::new(model, &mut cache, 0.0, prompt);
    for (i, tok) in gen.take(max_tokens).enumerate() {
        let _ = tok.map_err(|e| anyhow::anyhow!(e.to_string()))?;
        if i == 0 {
            prefill = t0.elapsed().as_secs_f64();
        }
        count += 1;
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let decode_s = (elapsed - prefill).max(0.001);
    let decode_tok_s = (count.saturating_sub(1) as f64) / decode_s;
    Ok((prefill, decode_tok_s, count))
}

fn main() -> Result<()> {
    let target = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/gemma-4-26B-A4B-it".to_string());
    let max_tokens: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);
    let prompt_str = std::env::args()
        .nth(3)
        .unwrap_or_else(|| "The theory of general relativity".to_string());

    let tokenizer = load_tokenizer(&target).map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let mut model = load_model(&target).map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let enc = tokenizer
        .encode(prompt_str.as_str(), false)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let ids: Vec<i32> = enc.get_ids().iter().map(|&i| i as i32).collect();
    eprintln!("prompt_len={}", ids.len());
    let prompt = Array::from_slice(&ids, &[1, ids.len() as i32]);
    let num_slots = *model.model.kv_cache_map.iter().max().unwrap_or(&0) + 1;

    let paged = std::env::var("PAGED_KV").is_ok();
    eprintln!(
        "kv_backend: {}",
        if paged { "paged (full-attn layers only; sliding stay contiguous)" } else { "standard fp16" }
    );
    let (prefill, decode_tok_s, count) = if paged {
        // Mixed cache: page only full-attention layers; sliding layers stay
        // on the contiguous KVCache (paging them regresses badly).
        let cache = init_mixed_paged_cache(&model);
        run_with_cache(&mut model, cache, max_tokens, &prompt)?
    } else {
        let cache: Vec<KVCache> = (0..num_slots).map(|_| Default::default()).collect();
        run_with_cache(&mut model, cache, max_tokens, &prompt)?
    };

    let decode_s = (count.saturating_sub(1) as f64) / decode_tok_s.max(0.001);
    println!(
        "Rust AR gemma4: prefill_s={prefill:.2} decode_s={decode_s:.2} decode_tok_s={decode_tok_s:.2} total={count}",
    );
    Ok(())
}
