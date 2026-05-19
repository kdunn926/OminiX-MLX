//! Standalone autoregressive chat on the MTPLX 27B Gemma4 target.
//!
//! Loads `models/Gemma4-27B-MTPLX-Optimized-Speed/target/` via
//! `mtplx_target::load_mtplx_target` (Q4 → BF16 dequant) and runs the
//! standard `Generate` iterator. No drafter, no speculation — this is the
//! sanity check that the target model alone produces a coherent
//! continuation under gemma4-mlx's existing AR path.
//!
//! Usage:
//!   cargo run --release -p gemma4-mlx --example mtplx_chat -- \
//!     models/Gemma4-27B-MTPLX-Optimized-Speed/target \
//!     "The quick brown fox" 64

use std::{env, path::PathBuf};

use anyhow::{anyhow, Result};
use mlx_rs::{
    ops::indexing::{IndexOp, NewAxis},
    transforms::eval,
    Array,
};

use gemma4_mlx::{
    load_tokenizer, mtplx_target::load_mtplx_target, Gemma4ChatTemplate, Gemma4Message, Generate,
    KVCache, KeyValueCache, QuantizedKVCache, TurboQuantKVCache,
};

/// EOS token IDs from generation_config.json.
const EOS_TOKEN_IDS: &[i32] = &[1, 106, 50];

fn run_decode<C: KeyValueCache + Default>(
    model: &mut gemma4_mlx::Model,
    cache: &mut Vec<C>,
    prompt_arr: &Array,
    temp: f32,
    max_new: usize,
    t_gen: &std::time::Instant,
    t_first: &mut Option<std::time::Duration>,
    emitted: &mut Vec<i32>,
) -> Result<()> {
    let generator = Generate::new(model, cache, temp, prompt_arr);
    print!("Generating: ");
    for (i, tok) in generator.enumerate() {
        if i >= max_new {
            break;
        }
        let tok_arr = tok.map_err(|e| anyhow!("gen step: {e:?}"))?;
        eval([&tok_arr])?;
        if t_first.is_none() {
            *t_first = Some(t_gen.elapsed());
        }
        let id: i32 = tok_arr.reshape(&[-1])?.item::<i32>();
        if EOS_TOKEN_IDS.contains(&id) {
            println!("\n[hit eos at step {i}: {id}]");
            break;
        }
        emitted.push(id);
    }
    Ok(())
}
fn temperature() -> f32 {
    std::env::var("TEMP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0)
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let target_dir = args
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from("models/Gemma4-27B-MTPLX-Optimized-Speed/target")
        });
    let prompt_arg = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "The quick brown fox".to_string());
    // If the prompt argument points to an existing JSON file, treat it as
    // an OpenAI-format chat fixture (e.g. hermes-gateway fixtures from
    // OminiX-API/tests/fixtures). Otherwise it's a literal prompt string.
    let (prompt, fixture_messages) = if std::path::Path::new(&prompt_arg).exists()
        && prompt_arg.ends_with(".json")
    {
        let bytes = std::fs::read(&prompt_arg)?;
        let v: serde_json::Value = serde_json::from_slice(&bytes)?;
        let msgs = v
            .get("messages")
            .and_then(|m| m.as_array())
            .ok_or_else(|| anyhow!("fixture {prompt_arg} has no messages array"))?;
        let parsed: Vec<Gemma4Message> = msgs
            .iter()
            .filter_map(|m| {
                let role = m.get("role")?.as_str()?;
                let content = match m.get("content") {
                    Some(serde_json::Value::String(s)) => s.clone(),
                    Some(serde_json::Value::Array(parts)) => parts
                        .iter()
                        .filter_map(|p| p.get("text").and_then(|t| t.as_str()).map(String::from))
                        .collect::<Vec<_>>()
                        .join(""),
                    _ => return None,
                };
                Some(match role {
                    "system" => Gemma4Message::system(content),
                    "user" => Gemma4Message::user(content),
                    "assistant" => Gemma4Message::assistant(content),
                    "tool" => {
                        let name = m.get("name").and_then(|n| n.as_str()).unwrap_or("tool");
                        Gemma4Message::tool(name, content)
                    }
                    _ => return None,
                })
            })
            .collect();
        (String::new(), Some(parsed))
    } else {
        (prompt_arg, None)
    };
    let max_new: usize = args
        .get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);

    println!("target_dir : {}", target_dir.display());
    println!("prompt     : {:?}", prompt);
    println!("max_new    : {max_new}");
    let temp = temperature();
    println!("temperature: {temp}");
    println!("eos        : {EOS_TOKEN_IDS:?}");
    println!();

    println!("Loading tokenizer + chat template…");
    let tokenizer = load_tokenizer(&target_dir)?;
    let template = Gemma4ChatTemplate::load(&target_dir)?;
    let rendered = if let Some(msgs) = &fixture_messages {
        template.render_prompt(msgs, &[], true)?
    } else {
        template.render_prompt(&[Gemma4Message::user(&prompt)], &[], true)?
    };
    println!("Rendered prompt ({} chars).", rendered.len());

    println!("Loading MTPLX target (native Q4; ~13GB resident)…");
    let t0 = std::time::Instant::now();
    let mut model = load_mtplx_target(&target_dir)?;
    println!("  target loaded in {:.1}s", t0.elapsed().as_secs_f32());

    let enc = tokenizer
        .encode(rendered.as_str(), false)
        .map_err(|e| anyhow!("encode: {e}"))?;
    let prompt_ids: Vec<i32> = enc.get_ids().iter().map(|&i| i as i32).collect();
    println!(
        "Prompt tokens ({} ids): {:?}",
        prompt_ids.len(),
        &prompt_ids[..prompt_ids.len().min(16)]
    );
    let prompt_arr = Array::from(prompt_ids.as_slice()).index(NewAxis);

    // Cache type selector: set QUANTIZE_KV=1 to use mixed-precision Q8K/Q4V
    // KV cache (mlx_rs_core::QuantizedKVCache) instead of the default
    // bf16 KVCache. The generic `Generate` iterator and `Model::forward`
    // path both work with any `KeyValueCache + Default`, so this is a
    // type-level switch only.
    // Cache backend selector via env vars (mutually exclusive; first wins):
    //   TURBO_KV=1     → TurboQuantKVCache (4-bit Lloyd-Max + Hadamard on K, BF16 V)
    //   QUANTIZE_KV=1  → mlx_rs_core::QuantizedKVCache (Q8 K, Q4 V)
    //   default        → KVCache (BF16 K, BF16 V)
    let backend = if std::env::var("TURBO_KV").is_ok() {
        "turboquant"
    } else if std::env::var("QUANTIZE_KV").is_ok() {
        "quantized"
    } else {
        "bf16"
    };
    println!("kv_backend : {backend}");

    let t_gen = std::time::Instant::now();
    let mut t_first: Option<std::time::Duration> = None;
    let mut emitted: Vec<i32> = Vec::with_capacity(max_new);
    match backend {
        "turboquant" => {
            let mut cache: Vec<TurboQuantKVCache> = Vec::new();
            run_decode(&mut model, &mut cache, &prompt_arr, temp, max_new, &t_gen, &mut t_first, &mut emitted)?;
        }
        "quantized" => {
            let mut cache: Vec<QuantizedKVCache> = Vec::new();
            run_decode(&mut model, &mut cache, &prompt_arr, temp, max_new, &t_gen, &mut t_first, &mut emitted)?;
        }
        _ => {
            // Prompt-cache prefix: when GEMMA4_PROMPT_CACHE_DIR is set,
            // try to load a previously-saved KV state for the longest
            // matching prefix of this prompt and skip prefilling that
            // portion. After generation, save the post-prompt KV so
            // future runs with the same prompt prefix start warm.
            // bf16-KVCache only — TQ / Quantized caches don't support
            // safetensors save/load yet.
            let cache_dir = std::env::var("GEMMA4_PROMPT_CACHE_DIR").ok();
            let (mut cache, prefix_skipped, suffix_prompt) = if let Some(ref d) = cache_dir {
                match KVCache::try_load_kv_caches(&prompt_ids, d) {
                    Ok(Some((caches, n_cached))) => {
                        println!(
                            "prompt-cache: HIT — reusing {} cached tokens, prefilling {} suffix",
                            n_cached,
                            prompt_ids.len() - n_cached
                        );
                        let suffix_slice = &prompt_ids[n_cached..];
                        let suffix_arr = Array::from(suffix_slice).index(NewAxis);
                        (caches, n_cached, suffix_arr)
                    }
                    Ok(None) => {
                        println!("prompt-cache: miss — prefilling and saving");
                        (Vec::<KVCache>::new(), 0usize, prompt_arr.clone())
                    }
                    Err(e) => {
                        eprintln!("prompt-cache: load error ({e}); falling back to full prefill");
                        (Vec::<KVCache>::new(), 0usize, prompt_arr.clone())
                    }
                }
            } else {
                (Vec::<KVCache>::new(), 0usize, prompt_arr.clone())
            };
            run_decode(&mut model, &mut cache, &suffix_prompt, temp, max_new, &t_gen, &mut t_first, &mut emitted)?;
            // Save after a *successful* full-prompt prefill (i.e. when
            // we didn't hit the cache and produced the full KV state).
            // Skip writeback on cache hits to avoid disk churn.
            if let Some(d) = cache_dir {
                if prefix_skipped == 0 {
                    if let Err(e) = KVCache::save_kv_caches(&cache, &prompt_ids, &d) {
                        eprintln!("prompt-cache: save error: {e}");
                    } else {
                        println!("prompt-cache: saved {} tokens worth of KV state to {d}", prompt_ids.len());
                    }
                }
            }
        }
    }
    let total = t_gen.elapsed();
    let ttft = t_first.unwrap_or(total);
    let decode_secs = (total - ttft).as_secs_f32();
    let decoded_tokens = emitted.len().saturating_sub(1) as f32; // exclude first
    let decode_tps = if decoded_tokens > 0.0 {
        decoded_tokens / decode_secs.max(1e-6)
    } else {
        0.0
    };
    let prompt_tps = prompt_ids.len() as f32 / ttft.as_secs_f32().max(1e-6);

    let u32_ids: Vec<u32> = emitted.iter().map(|&i| i as u32).collect();
    let text = tokenizer
        .decode(&u32_ids, false)
        .unwrap_or_else(|_| "<decode error>".to_string());
    println!();
    println!("\n=== Timing ===");
    println!("  TTFT (prefill + 1st token): {:.2}s", ttft.as_secs_f32());
    println!(
        "  Prefill throughput:         {:.1} prompt-tok/s",
        prompt_tps
    );
    println!(
        "  Decode throughput:          {:.1} tok/s  ({} tokens in {:.2}s)",
        decode_tps, decoded_tokens as usize, decode_secs
    );
    println!("  Total:                      {:.2}s", total.as_secs_f32());
    println!("\n=== Output ===");
    println!("{prompt}{text}");
    Ok(())
}
