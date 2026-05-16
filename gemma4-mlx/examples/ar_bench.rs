//! Minimal Rust AR throughput sanity check for Gemma4 (no DFlash).
//! Used to baseline the gemma4 verify-path bottleneck.

use anyhow::Result;
use gemma4_mlx::{load_model, load_tokenizer, Generate};
use mlx_rs::Array;
use std::time::Instant;

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
    let mut cache: Vec<mlx_rs_core::cache::KVCache> =
        (0..num_slots).map(|_| Default::default()).collect();

    let t0 = Instant::now();
    let mut prefill = 0.0;
    let mut count = 0usize;
    let gen = Generate::new(&mut model, &mut cache, 0.0, &prompt);
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
    println!(
        "Rust AR gemma4: prefill_s={prefill:.2} decode_s={decode_s:.2} decode_tok_s={decode_tok_s:.2} total={count}",
    );
    Ok(())
}
