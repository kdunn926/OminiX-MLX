//! Minimal Rust AR throughput sanity check for Gemma4 (no DFlash).
//! Used to baseline the gemma4 verify-path bottleneck and to compare KV
//! backends.
//!
//! KV backend (default: layered — physical sliding-window trim):
//!   FLAT_KV=1   unbounded Vec<KVCache> (legacy baseline)
//!   PAGED_KV=1  full-attn layers paged, sliding layers contiguous
//!
//! Loader: defaults to the canonical `load_model`. Try
//! `LOADER=ud ar_bench <model_dir> …` to route through the UD-MLX-4bit
//! loader for checkpoints that use the `language_model.model.*` prefix
//! (Unsloth Dynamic, including the multimodal gemma-4-e4b-it-4bit
//! checkpoint whose text-side weights match the UD layout).

use anyhow::Result;
use gemma4_mlx::mixed_cache::{init_layered_cache, init_mixed_paged_cache};
use gemma4_mlx::ud_loader::load_ud_mlx_4bit;
use gemma4_mlx::{load_model, load_tokenizer, Generate, Model};
use mlx_rs::Array;
use mlx_rs_core::cache::{KVCache, KeyValueCache};
use std::time::Instant;

#[path = "reranker_drafter.rs"]
mod reranker_drafter;

/// Run AR generation with a pre-built cache and report timing + generated ids.
fn run_with_cache<C: KeyValueCache + Default>(
    model: &mut Model,
    mut cache: Vec<C>,
    max_tokens: usize,
    prompt: &Array,
) -> Result<(f64, f64, usize, Vec<u32>)> {
    let t0 = Instant::now();
    let mut prefill = 0.0;
    let mut count = 0usize;
    let mut out_ids: Vec<u32> = Vec::with_capacity(max_tokens);
    let gen = Generate::new(model, &mut cache, 0.0, prompt);
    for (i, tok) in gen.take(max_tokens).enumerate() {
        let tok = tok.map_err(|e| anyhow::anyhow!(e.to_string()))?;
        out_ids.push(tok.item::<u32>());
        if i == 0 {
            prefill = t0.elapsed().as_secs_f64();
        }
        count += 1;
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let decode_s = (elapsed - prefill).max(0.001);
    let decode_tok_s = (count.saturating_sub(1) as f64) / decode_s;
    Ok((prefill, decode_tok_s, count, out_ids))
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
    let loader_choice = std::env::var("LOADER").unwrap_or_default();
    let mut model: Model = if loader_choice == "ud" {
        eprintln!("loader: UD-MLX-4bit (load_ud_mlx_4bit)");
        load_ud_mlx_4bit(&target).map_err(|e| anyhow::anyhow!(e.to_string()))?
    } else {
        eprintln!("loader: canonical (load_model)");
        load_model(&target).map_err(|e| anyhow::anyhow!(e.to_string()))?
    };
    let enc = tokenizer
        .encode(prompt_str.as_str(), false)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    // Gemma models require a leading <bos>. We encode raw prompts with
    // add_special_tokens=false (so callers control the template), which omits
    // it — and the 12B unified checkpoints are acutely BOS-sensitive: without
    // <bos> their AR output degenerates to a `<start_of_turn>model` loop (the
    // 26B tolerates its absence on long prompts, which masked this). Prepend it
    // unless already present; NO_BOS=1 opts out for A/B testing.
    let mut ids: Vec<i32> = enc.get_ids().iter().map(|&i| i as i32).collect();
    if std::env::var("NO_BOS").is_err() {
        if let Some(bos) = tokenizer.token_to_id("<bos>") {
            if ids.first() != Some(&(bos as i32)) {
                ids.insert(0, bos as i32);
                eprintln!("prepended <bos> (id {bos})");
            }
        }
    }
    eprintln!("prompt_len={}", ids.len());
    let prompt = Array::from_slice(&ids, &[1, ids.len() as i32]);
    let num_slots = *model.model.kv_cache_map.iter().max().unwrap_or(&0) + 1;

    let paged = std::env::var("PAGED_KV").is_ok();
    let flat = std::env::var("FLAT_KV").is_ok();
    let kvflash = std::env::var("KVFLASH").is_ok() || std::env::var("DFLASH_KVFLASH").is_ok();
    let paging = std::env::var("DFLASH_KVFLASH_PAGING").as_deref() == Ok("1");
    let drafter = std::env::var("DFLASH_KVFLASH_DRAFTER").as_deref() == Ok("1");

    // Reranker "drafter": before decode, rerank the prompt's 64-token chunks once
    // and pin the most-relevant resident for the whole generation (recall). The
    // reranker scores (query, chunk_text) text pairs, so the gemma/reranker
    // tokenizer mismatch is irrelevant — we chunk by gemma's own tokens (matching
    // the cache) and decode each to text. Needs paging (host-resident chunks).
    if kvflash && paging && drafter {
        let chunk = 64usize;
        let chunk_texts: Vec<String> = ids
            .chunks(chunk)
            .map(|c| {
                let cids: Vec<u32> = c.iter().map(|&x| x as u32).collect();
                tokenizer.decode(&cids, false).unwrap_or_default()
            })
            .collect();
        let qn = std::env::var("DFLASH_KVFLASH_DRAFTER_QTOK")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(160usize)
            .min(ids.len());
        let qids: Vec<u32> = ids[ids.len() - qn..].iter().map(|&x| x as u32).collect();
        let query = tokenizer.decode(&qids, true).unwrap_or_default();
        let rr_dir = std::env::var("DFLASH_KVFLASH_RERANKER")
            .unwrap_or_else(|_| "models/Qwen3-Reranker-0.6B-4bit".into());
        eprintln!("drafter: reranking {} chunks vs query ({qn} tok) with {rr_dir}", chunk_texts.len());
        let t0 = Instant::now();
        let mut rr = reranker_drafter::RerankerDrafter::load(&rr_dir)?;
        let order = rr.rank(&query, &chunk_texts)?;
        drop(rr);
        eprintln!(
            "drafter: ranked in {:.1}s; top chunks {:?}",
            t0.elapsed().as_secs_f64(),
            &order[..order.len().min(8)]
        );
        mlx_rs_core::kvflash::set_drafter_pins(order);
    }

    let kv_label = if kvflash && paging {
        let pool = std::env::var("DFLASH_KVFLASH").unwrap_or_else(|_| "4096".into());
        let how = if drafter { "drafter pins" } else { "q·k" };
        format!("kvflash host-paging ({how}, global pool={pool})")
    } else if kvflash {
        let pool = std::env::var("DFLASH_KVFLASH").unwrap_or_else(|_| "4096".into());
        format!("kvflash (global pool={pool})")
    } else if paged { "paged".into() } else if flat { "flat (unbounded)".into() } else { "layered (sliding-trim)".into() };
    eprintln!("kv_backend: {kv_label}");
    let (prefill, decode_tok_s, count, out_ids) = if kvflash && paging {
        let cache = gemma4_mlx::init_kvflash_paged_cache(&model);
        run_with_cache(&mut model, cache, max_tokens, &prompt)?
    } else if kvflash {
        let cache = gemma4_mlx::init_kvflash_cache(&model);
        run_with_cache(&mut model, cache, max_tokens, &prompt)?
    } else if paged {
        let cache = init_mixed_paged_cache(&model);
        run_with_cache(&mut model, cache, max_tokens, &prompt)?
    } else if flat {
        let cache: Vec<KVCache> = (0..num_slots).map(|_| Default::default()).collect();
        run_with_cache(&mut model, cache, max_tokens, &prompt)?
    } else {
        let cache = init_layered_cache(&model);
        run_with_cache(&mut model, cache, max_tokens, &prompt)?
    };
    if let Ok(text) = tokenizer.decode(&out_ids, true) {
        eprintln!("--- output ---\n{text}\n--------------");
    }

    let decode_s = (count.saturating_sub(1) as f64) / decode_tok_s.max(0.001);
    println!(
        "Rust AR gemma4: prefill_s={prefill:.2} decode_s={decode_s:.2} decode_tok_s={decode_tok_s:.2} total={count}",
    );
    Ok(())
}
