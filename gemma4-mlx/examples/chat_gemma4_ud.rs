//! Chat with the UD-MLX-4bit Gemma4 variant (preserve-quant loader).
//!
//! Same flow as `chat_gemma4` but routes through `ud_loader::load_ud_mlx_4bit`
//! which keeps the heterogeneous Q4/Q6/Q8 weights packed (no dequant
//! at load time, ~15 GB resident vs ~50 GB for the bf16 model).
//!
//! Usage:
//!   cargo run --release -p gemma4-mlx --example chat_gemma4_ud -- \
//!     models/gemma4-26B-a4b-it-UD-MLX-4bit "Hello, how are you?"

use std::{env, error::Error, path::PathBuf};

use gemma4_mlx::{
    load_tokenizer, ud_loader::load_ud_mlx_4bit, Generate, KVCache, EOS_TOKEN_IDS,
};
// NOTE: this example stays on Vec<KVCache> (flat/unbounded) because the
// prompt-cache API (save_kv_caches / try_load_kv_caches) is typed to KVCache.
// Switching to init_layered_cache would require extending the save/load API to
// handle SlidingKVCache snapshots — tracked separately. Use chat_text for the
// layered-cache default path on e4b / 12B.
use mlx_rs::{
    ops::indexing::{IndexOp, NewAxis},
    Array,
};
use mlx_rs_core::cache::KVCache as CoreKVCache;

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let args: Vec<String> = env::args().collect();
    let model_dir = args
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/gemma4-26B-a4b-it-UD-MLX-4bit"));
    let prompt = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "Write a short haiku about Gemma 4.".to_string());

    let mut model = load_ud_mlx_4bit(&model_dir)?;
    let tokenizer = load_tokenizer(&model_dir)?;

    let encoding = tokenizer.encode(prompt, true)?;
    let prompt_ids: Vec<i32> = encoding.get_ids().iter().map(|&i| i as i32).collect();

    // ── Prefix cache (#31): if GEMMA4_PROMPT_CACHE_DIR is set, try to
    // reuse a previously-saved KV state for the longest matching prefix
    // of this prompt. Each subsequent tool-use turn in a session
    // re-prefills only the new suffix instead of the full 5-12K-token
    // history. Saves ~5-10s of TTFT per turn on hermes-class workloads.
    // ──
    let cache_dir = std::env::var("GEMMA4_PROMPT_CACHE_DIR").ok();
    let (mut cache, prefix_skipped, suffix_arr) = if let Some(ref d) = cache_dir {
        match CoreKVCache::try_load_kv_caches(&prompt_ids, d) {
            Ok(Some((caches, n_cached))) => {
                eprintln!(
                    "[prompt-cache] HIT — reusing {n_cached} cached tokens, prefilling {} suffix",
                    prompt_ids.len() - n_cached
                );
                let suffix_slice = &prompt_ids[n_cached..];
                let suffix_arr = Array::from(suffix_slice).index(NewAxis);
                (caches, n_cached, suffix_arr)
            }
            Ok(None) => {
                eprintln!("[prompt-cache] miss — full prefill, will save");
                let full = Array::from(prompt_ids.as_slice()).index(NewAxis);
                (Vec::<KVCache>::new(), 0_usize, full)
            }
            Err(e) => {
                eprintln!("[prompt-cache] load error ({e}) — falling back to full prefill");
                let full = Array::from(prompt_ids.as_slice()).index(NewAxis);
                (Vec::<KVCache>::new(), 0_usize, full)
            }
        }
    } else {
        let full = Array::from(prompt_ids.as_slice()).index(NewAxis);
        (Vec::<KVCache>::new(), 0_usize, full)
    };
    let generator = Generate::new(&mut model, &mut cache, 0.0, &suffix_arr);

    let max_tokens: usize = std::env::var("CHAT_MAX_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2048);
    let mut emitted = 0_usize;
    let t_start = std::time::Instant::now();
    for token in generator.take(max_tokens) {
        let token = token?;
        let token_id = token.item::<u32>();
        if EOS_TOKEN_IDS.contains(&token_id) {
            break;
        }
        print!("{}", tokenizer.decode(&[token_id], true)?);
        emitted += 1;
    }
    let elapsed = t_start.elapsed().as_secs_f64();
    eprintln!(
        "\n[chat_gemma4_ud] emitted {} tok in {:.2}s = {:.2} tok/s",
        emitted,
        elapsed,
        emitted as f64 / elapsed.max(1e-6)
    );

    // Save the post-prefill KV cache for reuse on the next turn. Skip
    // writeback on cache hits to avoid disk churn — the existing
    // saved cache already covers the prefix that matched.
    if let Some(d) = cache_dir {
        if prefix_skipped == 0 {
            match CoreKVCache::save_kv_caches(&cache, &prompt_ids, &d) {
                Ok(()) => eprintln!(
                    "[prompt-cache] saved {} tokens worth of KV state to {d}",
                    prompt_ids.len()
                ),
                Err(e) => eprintln!("[prompt-cache] save error: {e}"),
            }
        }
    }

    Ok(())
}
