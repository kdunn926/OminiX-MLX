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
pub mod moe;
pub mod model;

pub use cache::HybridCache;
pub use config::ModelArgs;
pub use model::{load_model, Model};
pub use mlx_rs_core::{error::Error, load_tokenizer};

use mlx_rs::{
    argmax_axis, array, categorical,
    error::Exception,
    ops::indexing::{IndexOp, NewAxis},
    Array,
};

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
                // KV/recurrent caches across the full prompt.
                const PREFILL_CHUNK: i32 = 64;
                let seq_len = prompt.shape()[1];
                // Chunked prefill: only the final chunk needs full vocab logits;
                // earlier chunks just populate the cache, so use forward_last_logits
                // (cheap `[B, vocab]` slice) for them too — the throwaway logits
                // are immediately discarded.
                let logits = if seq_len > PREFILL_CHUNK {
                    let mut pos = 0;
                    let mut last_logits = None;
                    while pos < seq_len {
                        let end = (pos + PREFILL_CHUNK).min(seq_len);
                        let chunk = prompt.index((.., pos..end));
                        let logits = tri!(self.model.forward_last_logits(&chunk, &mut self.cache));
                        // Materialize and free intermediates between chunks.
                        tri!(eval([&logits]));
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
                let next_y = tri!(self.compute_next(&current));
                let _ = mlx_rs::transforms::async_eval([&next_y]);

                self.prefetched = Some(next_y);
                self.token_count += 1;

                if self.token_count % 256 == 0 {
                    unsafe {
                        mlx_sys::mlx_clear_cache();
                    }
                }

                Some(Ok(current))
            }
        }
    }
}
