//! Qwen3.6 hybrid inference on Apple Silicon with MLX.
//!
//! This crate implements Qwen3.6 models using a hybrid attention architecture:
//! - **Linear attention layers** (GatedDeltaNet — fixed-size recurrent state)
//! - **Full attention layers** (GatedAttention — with partial RoPE and output gate)
//!
//! Supported variants:
//! - **Qwen3.6-35B-A3B**: 256 routed experts (top-8) + 1 shared expert per layer (MoE)
//! - **Qwen3.6-27B**: Standard dense MLP (gate/up/down_proj) per layer
//!
//! Supports 4-bit quantized models from the `mlx-community` HuggingFace hub.

pub mod attention;
pub mod cache;
pub mod config;
pub mod deltanet;
pub mod model;
pub mod moe;
pub mod mtp;
pub mod verify_hook;
pub mod vision;

pub use cache::{GdnRollbackSnapshot, HybridCache, RecurrentState};
pub use deltanet::GdnTapeCapture;
pub use config::{ModelArgs, VisionConfig};
pub use mlx_rs_core::{cache::QuantizedKVCache, error::Error, load_tokenizer};
pub use model::{load_model, load_vl_model, KVCacheMode, Model, VlModel};
pub use vision::{preprocess_image, VisionTower};

use mlx_rs::{
    argmax_axis, array, categorical,
    error::Exception,
    ops::indexing::{IndexOp, NewAxis},
    Array,
};

/// A single message in a multi-turn VL chat, with pre-formatted text content.
pub struct VlChatMessage {
    pub role: String,
    pub text: String,
    /// Number of image placeholder tokens to insert (from visual_features.shape()[0]).
    pub n_visual_tokens: Option<usize>,
}

pub fn build_chat_tokens_for_messages(
    tokenizer: &tokenizers::Tokenizer,
    messages: &[VlChatMessage],
    image_token_id: i32,
    vision_start_token_id: i32,
    vision_end_token_id: i32,
) -> Result<Vec<i32>, mlx_rs_core::error::Error> {
    let im_start = tokenizer
        .token_to_id("<|im_start|>")
        .ok_or_else(|| mlx_rs_core::error::Error::Tokenizer("Missing <|im_start|>".into()))?
        as i32;
    let im_end = tokenizer
        .token_to_id("<|im_end|>")
        .ok_or_else(|| mlx_rs_core::error::Error::Tokenizer("Missing <|im_end|>".into()))?
        as i32;
    let newline = tokenizer.token_to_id("\n").unwrap_or(198) as i32;

    let encode = |text: &str| -> Result<Vec<i32>, mlx_rs_core::error::Error> {
        let enc = tokenizer.encode(text, false)?;
        Ok(enc.get_ids().iter().map(|&id| id as i32).collect())
    };

    let mut tokens: Vec<i32> = Vec::new();

    for msg in messages {
        let role_tokens = encode(&msg.role)?;
        let text_tokens = if msg.text.is_empty() {
            vec![]
        } else {
            encode(&msg.text)?
        };

        tokens.push(im_start);
        tokens.extend_from_slice(&role_tokens);
        tokens.push(newline);

        if let Some(n_visual) = msg.n_visual_tokens {
            tokens.push(vision_start_token_id);
            for _ in 0..n_visual {
                tokens.push(image_token_id);
            }
            tokens.push(vision_end_token_id);
            tokens.push(newline);
        }

        tokens.extend_from_slice(&text_tokens);
        tokens.push(im_end);
        tokens.push(newline);
    }

    let assistant_tokens = encode("assistant")?;
    tokens.push(im_start);
    tokens.extend_from_slice(&assistant_tokens);
    tokens.push(newline);

    Ok(tokens)
}

// ============================================================================
// Sampling
// ============================================================================

pub fn sample(logits: &Array, temp: f32) -> Result<Array, Exception> {
    match temp {
        t if t == 0.0 => argmax_axis!(logits, -1).map_err(Into::into),
        _ => {
            let logits = logits.multiply(array!(1.0 / temp))?;
            categorical!(logits).map_err(Into::into)
        }
    }
}

// ============================================================================
// Generation Iterator
// ============================================================================

pub struct Generate<'a> {
    model: &'a mut Model,
    cache: Vec<HybridCache>,
    temp: f32,
    state: GenerateState<'a>,
    prefetched: Option<Array>,
    token_count: usize,
}

enum GenerateState<'a> {
    Prefill { prompt: &'a Array },
    Decode,
}

impl<'a> Generate<'a> {
    pub fn new(model: &'a mut Model, temp: f32, prompt: &'a Array) -> Self {
        Self {
            model,
            cache: Vec::new(),
            temp,
            state: GenerateState::Prefill { prompt },
            prefetched: None,
            token_count: 0,
        }
    }

    /// Same as `new` but pre-allocates mixed-precision KV caches (K=q8, V=q4).
    pub fn new_quantized_kv(model: &'a mut Model, temp: f32, prompt: &'a Array) -> Self {
        let cache = model.new_cache(KVCacheMode::Quantized);
        Self {
            model,
            cache,
            temp,
            state: GenerateState::Prefill { prompt },
            prefetched: None,
            token_count: 0,
        }
    }

    /// Spike: TurboQuant 4-bit K + 8-bit V cache. Uses the fused
    /// online-softmax SDPA path via `KeyValueCache::try_fused_attention`.
    pub fn new_turboquant_kv(model: &'a mut Model, temp: f32, prompt: &'a Array) -> Self {
        let cache = model.new_cache(KVCacheMode::TurboQuant);
        Self {
            model,
            cache,
            temp,
            state: GenerateState::Prefill { prompt },
            prefetched: None,
            token_count: 0,
        }
    }

    /// Paged KV for full-attention layers: each layer draws blocks from its own
    /// private `PagedKvPool` and decode runs through the fused paged-attention
    /// kernel. Recurrent (GDN) layers are unaffected. Used to benchmark the
    /// paged-attention path against the standard contiguous KV cache.
    pub fn new_paged_kv(model: &'a mut Model, temp: f32, prompt: &'a Array) -> Self {
        let cache = model.new_cache(KVCacheMode::Paged);
        Self {
            model,
            cache,
            temp,
            state: GenerateState::Prefill { prompt },
            prefetched: None,
            token_count: 0,
        }
    }

    /// Build a Generate with a pre-populated cache (e.g. from a
    /// prompt-cache prefix load). Caller wraps each loaded `KVCache`
    /// in `HybridCache::KV(...)` and supplies the suffix prompt.
    pub fn new_with_cache(
        model: &'a mut Model,
        cache: Vec<HybridCache>,
        temp: f32,
        prompt: &'a Array,
    ) -> Self {
        Self {
            model,
            cache,
            temp,
            state: GenerateState::Prefill { prompt },
            prefetched: None,
            token_count: 0,
        }
    }

    /// Consume the iterator and return the populated cache vector.
    /// Used by the prompt-cache feature to persist the post-prefill
    /// KV state after generation completes.
    pub fn into_cache(self) -> Vec<HybridCache> {
        self.cache
    }

    /// Borrow the cache vector. Useful for inspecting cached offsets or
    /// snapshotting mid-stream.
    pub fn cache(&self) -> &[HybridCache] {
        &self.cache
    }

    fn compute_next(&mut self, y: &Array) -> Result<Array, Exception> {
        let inputs = y.index((.., NewAxis)); // [B, 1]
        let logits = self.model.forward(&inputs, &mut self.cache)?;
        sample(&logits.index((.., -1, ..)), self.temp)
    }
}

macro_rules! tri {
    ($expr:expr) => {
        match $expr {
            Ok(val) => val,
            Err(e) => return Some(Err(e.into())),
        }
    };
}

impl Iterator for Generate<'_> {
    type Item = Result<Array, Exception>;

    fn next(&mut self) -> Option<Self::Item> {
        use mlx_rs::transforms::{async_eval, eval};

        match &self.state {
            GenerateState::Prefill { prompt } => {
                let prompt = *prompt;
                // Chunked prefill keeps peak memory bounded while still filling
                // KV/recurrent caches across the full prompt. Chunk size is
                // env-tunable via QWEN36_PREFILL_CHUNK (default 64). Same
                // shape as the gemma4 path so the env knobs match.
                let prefill_chunk: i32 = std::env::var("QWEN36_PREFILL_CHUNK")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .filter(|&n: &i32| n > 0)
                    .unwrap_or(64);
                let seq_len = prompt.shape()[1];
                // Chunked prefill: only the final chunk needs full vocab logits;
                // earlier chunks just populate the cache, so use forward_last_logits
                // (cheap `[B, vocab]` slice) for them too — the throwaway logits
                // are immediately discarded. Per-chunk eval mode is env-tunable:
                //   QWEN36_SKIP_CHUNK_EVAL=1 — omit chunk-boundary eval entirely.
                //   QWEN36_ASYNC_PREFILL=1   — use async_eval to overlap.
                let skip_chunk_eval = std::env::var("QWEN36_SKIP_CHUNK_EVAL").is_ok();
                let async_prefill = std::env::var("QWEN36_ASYNC_PREFILL").is_ok();
                let logits = if seq_len > prefill_chunk {
                    let mut pos = 0;
                    let mut last_logits = None;
                    while pos < seq_len {
                        let end = (pos + prefill_chunk).min(seq_len);
                        let chunk = prompt.index((.., pos..end));
                        let logits = tri!(self.model.forward_last_logits(&chunk, &mut self.cache));
                        if !skip_chunk_eval {
                            if async_prefill {
                                tri!(async_eval([&logits]));
                            } else {
                                tri!(eval([&logits]));
                            }
                        }
                        last_logits = Some(logits);
                        pos = end;
                    }
                    last_logits.expect("chunked prefill produced no logits")
                } else {
                    tri!(self.model.forward_last_logits(prompt, &mut self.cache))
                };
                let y = tri!(sample(&logits, self.temp));

                let _ = async_eval([&y]);
                let next_y = tri!(self.compute_next(&y));
                let _ = async_eval([&next_y]);
                let _ = eval([&y]);

                self.prefetched = Some(next_y);
                self.state = GenerateState::Decode;
                self.token_count = 1;

                Some(Ok(y))
            }
            GenerateState::Decode => {
                let current = self.prefetched.take()?;
                // QWEN36_PROFILE_DECODE=1 logs per-step compute_next timing
                // every 16 tokens.
                let profile = std::env::var("QWEN36_PROFILE_DECODE").is_ok();
                let t0 = profile.then(std::time::Instant::now);
                let next_y = tri!(self.compute_next(&current));
                let _ = mlx_rs::transforms::async_eval([&next_y]);
                if let Some(t0) = t0 {
                    let ms = t0.elapsed().as_secs_f32() * 1000.0;
                    if self.token_count % 16 == 0 {
                        eprintln!(
                            "[qwen36-profile] tok={} next={:.2}ms",
                            self.token_count, ms
                        );
                    }
                }

                self.prefetched = Some(next_y);
                self.token_count += 1;

                let cache_clear_interval: usize = std::env::var("QWEN36_CACHE_CLEAR_INTERVAL")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(256);
                if cache_clear_interval > 0 && self.token_count % cache_clear_interval == 0 {
                    unsafe {
                        mlx_sys::mlx_clear_cache();
                    }
                }

                Some(Ok(current))
            }
        }
    }
}
