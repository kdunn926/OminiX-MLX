use mlx_rs::ops::indexing::{IndexOp, NewAxis};
use mlx_rs_core::cache::KeyValueCache;
use qwen3_6_mlx::{load_model, load_tokenizer, Generate};
use std::collections::HashSet;

fn main() -> anyhow::Result<()> {
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
        .unwrap_or_else(|| "./models/Qwen3.6-35B-A3B-4bit".to_string());

    eprintln!("Loading model from {}...", model_dir);
    let start = std::time::Instant::now();
    let tokenizer = load_tokenizer(&model_dir)?;
    let mut model = load_model(&model_dir)?;
    eprintln!("Model loaded in {:.1}s", start.elapsed().as_secs_f64());

    let prompt = args.get(2).cloned().unwrap_or_else(|| {
        "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n"
            .to_string()
    });

    let encoding = tokenizer
        .encode(prompt.as_str(), false)
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    let prompt_ids_full: Vec<i32> = encoding.get_ids().iter().map(|&t| t as i32).collect();
    let prompt_tokens_full = mlx_rs::Array::from_slice(&prompt_ids_full, &[1, prompt_ids_full.len() as i32]);
    eprintln!("Prompt: {} tokens", prompt_ids_full.len());

    // Prompt-cache prefix: QWEN36_PROMPT_CACHE_DIR=path enables save/load
    // of a per-prompt prefix KV state. Standard `KVCache` only — TQ /
    // Quantized variants don't support safetensors save/load.
    let qwen_cache_dir = std::env::var("QWEN36_PROMPT_CACHE_DIR").ok();
    let prompt_cache_load: Option<(Vec<qwen3_6_mlx::HybridCache>, usize)> = qwen_cache_dir
        .as_ref()
        .and_then(|d| {
            match mlx_rs_core::cache::KVCache::try_load_kv_caches(&prompt_ids_full, d) {
                Ok(Some((kv_caches, n))) => {
                    eprintln!(
                        "prompt-cache: HIT — reusing {} cached tokens, prefilling {} suffix",
                        n,
                        prompt_ids_full.len() - n
                    );
                    let hybrid: Vec<qwen3_6_mlx::HybridCache> = kv_caches
                        .into_iter()
                        .map(qwen3_6_mlx::HybridCache::KV)
                        .collect();
                    Some((hybrid, n))
                }
                Ok(None) => {
                    eprintln!("prompt-cache: miss — prefilling and saving");
                    None
                }
                Err(e) => {
                    eprintln!("prompt-cache: load error ({e}); full prefill");
                    None
                }
            }
        });
    let (prompt_tokens, prefix_skipped) = match prompt_cache_load.as_ref() {
        Some((_, n)) => {
            let suffix_arr = mlx_rs::Array::from_slice(
                &prompt_ids_full[*n..],
                &[1, (prompt_ids_full.len() - n) as i32],
            );
            (suffix_arr, *n)
        }
        None => (prompt_tokens_full.clone(), 0usize),
    };
    let _ = &prompt_tokens_full; // keep for later save call

    let max_tokens: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(500);

    let temp: f32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0.7);

    let eos_tokens: HashSet<u32> = {
        let config_path = std::path::Path::new(&model_dir).join("config.json");
        let config: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&config_path)?)?;
        match &config["eos_token_id"] {
            serde_json::Value::Array(ids) => ids
                .iter()
                .filter_map(|v| v.as_u64().map(|n| n as u32))
                .collect(),
            serde_json::Value::Number(n) => {
                let mut s = HashSet::new();
                s.insert(n.as_u64().unwrap_or(248044) as u32);
                s
            }
            _ => {
                let mut s = HashSet::new();
                s.insert(248044u32);
                s
            }
        }
    };

    let mut gen = if let Some((hybrid_caches, _)) = prompt_cache_load {
        eprintln!("kv_backend: standard fp16 (warm-start from prompt cache)");
        Generate::new_with_cache(&mut model, hybrid_caches, temp, &prompt_tokens)
    } else if std::env::var("TURBO_KV").is_ok() {
        eprintln!("kv_backend: turboquant");
        Generate::new_turboquant_kv(&mut model, temp, &prompt_tokens)
    } else if std::env::var("QUANTIZE_KV").is_ok() {
        eprintln!("kv_backend: quantized (K=q8, V=q4)");
        Generate::new_quantized_kv(&mut model, temp, &prompt_tokens)
    } else if std::env::var("PAGED_KV").is_ok() {
        eprintln!("kv_backend: paged");
        Generate::new_paged_kv(&mut model, temp, &prompt_tokens)
    } else {
        eprintln!("kv_backend: standard fp16");
        Generate::new(&mut model, temp, &prompt_tokens)
    };

    let start = std::time::Instant::now();
    let mut token_count = 0;
    let mut ttft: Option<f64> = None;

    for token_result in (&mut gen).take(max_tokens) {
        let token = token_result?;
        if ttft.is_none() {
            ttft = Some(start.elapsed().as_secs_f64());
        }
        let token_id = token.item::<u32>();
        if eos_tokens.contains(&token_id) {
            break;
        }

        let text = tokenizer
            .decode(&[token_id], true)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        print!("{}", text);
        token_count += 1;
    }
    println!();

    // Optional prompt-cache writeback: on cache-miss runs, persist the
    // post-prefill KV state so subsequent same-prompt runs hit warm.
    // Only writes the basic-KV (HybridCache::KV) slots — TQ / Quantized
    // / Recurrent slots are skipped and treated as empty on load.
    if let Some(d) = qwen_cache_dir.as_ref() {
        if prefix_skipped == 0 {
            let post_cache = gen.into_cache();
            let mut kv_only: Vec<mlx_rs_core::cache::KVCache> = Vec::new();
            for h in post_cache {
                if let qwen3_6_mlx::HybridCache::KV(mut kv) = h {
                    let off = kv.offset();
                    let want = prompt_ids_full.len() as i32;
                    if off > want {
                        kv.trim(off - want);
                    }
                    kv_only.push(kv);
                } else {
                    kv_only.push(mlx_rs_core::cache::KVCache::new());
                }
            }
            if let Err(e) = mlx_rs_core::cache::KVCache::save_kv_caches(
                &kv_only,
                &prompt_ids_full,
                d,
            ) {
                eprintln!("prompt-cache: save error: {e}");
            } else {
                eprintln!(
                    "prompt-cache: saved {} tokens of KV state to {d}",
                    prompt_ids_full.len()
                );
            }
        }
    }
    let elapsed = start.elapsed();
    let decode_time = elapsed.as_secs_f64() - ttft.unwrap_or(0.0);
    let decode_tokens = if token_count > 1 { token_count - 1 } else { 0 };
    let decode_tps = if decode_tokens > 0 {
        decode_tokens as f64 / decode_time
    } else {
        0.0
    };
    eprintln!(
        "TTFT: {:.2}s | Decode: {} tok in {:.2}s ({:.1} tok/s) | Total: {} tok in {:.2}s",
        ttft.unwrap_or(0.0),
        decode_tokens,
        decode_time,
        decode_tps,
        token_count,
        elapsed.as_secs_f64(),
    );

    Ok(())
}
