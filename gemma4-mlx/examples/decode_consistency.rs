//! Decode-correctness check: KV-cached greedy decode must be token-for-token
//! identical to a no-cache re-forward of the growing sequence (both deterministic
//! greedy). This isolates the KV-cache + RoPE-offset + per-layer mask machinery
//! and does not depend on any external reference.
//!
//! Usage:
//!   cargo run --release -p gemma4-mlx --example decode_consistency -- \
//!     <MODEL_DIR> [K=8]

use std::path::Path;

use anyhow::Result;
use gemma4_mlx::{init_cache, load_model, ud_loader::load_ud_mlx_4bit, Generate, KVCache, ModelInput};
use mlx_rs::{Array, Dtype};

fn argmax_array(logits: &Array) -> Result<i32> {
    let f = logits.as_dtype(Dtype::Float32)?;
    f.eval()?;
    let v: Vec<f32> = f.as_slice::<f32>().to_vec();
    Ok(v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as i32)
        .unwrap_or(0))
}

fn is_quantized(model_dir: &Path) -> Result<bool> {
    let raw: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(model_dir.join("config.json"))?)?;
    Ok(raw.get("quantization").map(|q| q.is_object()).unwrap_or(false))
}

fn main() -> Result<()> {
    let dir = std::env::args()
        .nth(1)
        .expect("usage: decode_consistency <MODEL_DIR> [K]");
    let k: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let model_dir = Path::new(&dir);

    // Fixed prompt — matches the model_parity / dump_gemma4_logits tokens.
    let prompt: [i32; 6] = [2, 1024, 2048, 4096, 8192, 16384];
    // Gemma 4 EOS / end-of-turn — stop greedy on these so the comparison
    // does not run past the model's natural stopping point.
    const EOS: [i32; 2] = [1, 106];

    let quant = is_quantized(model_dir)?;
    println!(
        "loader: {}",
        if quant { "load_ud_mlx_4bit (quantized)" } else { "load_model (bf16)" }
    );
    let mut model = if quant { load_ud_mlx_4bit(model_dir)? } else { load_model(model_dir)? };
    let num_slots = *model.model.kv_cache_map.iter().max().unwrap_or(&0) + 1;

    // ── A) KV-cached greedy decode via the Generate iterator ──────────────────
    let mut cache: Vec<KVCache> = init_cache(num_slots);
    let prompt_arr = Array::from_slice(&prompt, &[1, prompt.len() as i32]);
    let mut cache_ids: Vec<i32> = Vec::with_capacity(k);
    {
        let gen = Generate::new(&mut model, &mut cache, 0.0, &prompt_arr);
        for tok in gen.take(k) {
            let id = tok?.item::<u32>() as i32;
            if EOS.contains(&id) {
                break;
            }
            cache_ids.push(id);
        }
    }
    drop(cache);

    // ── B) No-cache greedy: fresh cache each step, re-forward growing sequence
    let mut seq: Vec<i32> = prompt.to_vec();
    let mut nocache_ids: Vec<i32> = Vec::with_capacity(k);
    for _ in 0..k {
        let mut fresh_cache: Vec<KVCache> = init_cache(num_slots);
        let toks = Array::from_slice(&seq, &[1, seq.len() as i32]);
        let last = model.forward_last_logits(ModelInput {
            inputs: &toks,
            mask: None,
            cache: &mut fresh_cache,
        })?;
        let next = argmax_array(&last)?;
        if EOS.contains(&next) {
            break;
        }
        nocache_ids.push(next);
        seq.push(next);
    }

    println!("cache   greedy: {cache_ids:?}");
    println!("nocache greedy: {nocache_ids:?}");
    let matched = cache_ids == nocache_ids;
    println!("TOKEN-FOR-TOKEN MATCH (cache == no-cache): {matched}");
    if !matched {
        eprintln!("DECODE INCONSISTENT — KV-cache path diverges from no-cache re-forward");
        std::process::exit(1);
    }
    println!("PASS — KV-cache decode is correct (matches no-cache re-forward).");
    Ok(())
}
