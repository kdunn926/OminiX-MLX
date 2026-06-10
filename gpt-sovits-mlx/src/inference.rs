//! End-to-end inference pipeline for GPT-SoVITS
//!
//! This module provides the complete TTS inference pipeline:
//! 1. Text preprocessing (text -> phonemes)
//! 2. BERT encoding (text -> bert features)
//! 3. HuBERT encoding (reference audio -> audio features)
//! 4. T2S generation (phonemes + bert + ref_audio -> semantic tokens)
//! 5. SoVITS vocoding (semantic tokens -> audio waveform)
//!
//! # Example
//!
//! ```ignore
//! use mlx_rs_lm::inference::{GenerationConfig, generate_semantic_tokens};
//!
//! // Generate semantic tokens
//! let (tokens, finished) = generate_semantic_tokens(
//!     &mut t2s_model,
//!     &phoneme_ids,
//!     &bert_features,
//!     &config,
//! )?;
//! ```

use mlx_rs::{
    ops::{concatenate_axis, indexing::IndexOp},
    Array,
};
use serde::{Deserialize, Serialize};

use crate::{
    cache::KeyValueCache,
    error::Error,
    models::t2s::{T2SModel, T2SInput},
    sampling::{Sampler, SamplingConfig},
    text::{PreprocessorConfig, TextPreprocessor, symbols_to_ids},
};

/// Configuration for semantic token generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Maximum number of tokens to generate
    pub max_tokens: usize,
    /// Minimum number of tokens before allowing EOS
    pub min_tokens: usize,
    /// Temperature for sampling (0 = greedy)
    pub temperature: f32,
    /// Top-k sampling (0 = disabled)
    pub top_k: usize,
    /// Top-p (nucleus) sampling threshold
    pub top_p: f32,
    /// Repetition penalty (1.0 = disabled)
    pub repetition_penalty: f32,
    /// EOS token ID for semantic tokens
    pub eos_token_id: i32,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 500,
            min_tokens: 10,
            temperature: 0.8,
            top_k: 3,
            top_p: 0.95,
            repetition_penalty: 1.0,
            eos_token_id: 1024,
        }
    }
}

impl GenerationConfig {
    /// Greedy decoding configuration
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            ..Default::default()
        }
    }

    /// Sampling configuration with default parameters
    pub fn sampling() -> Self {
        Self::default()
    }
}

/// Output from semantic token generation
#[derive(Debug)]
pub struct GenerationOutput {
    /// Generated semantic token IDs [batch, seq]
    pub tokens: Array,
    /// Number of tokens generated
    pub num_tokens: usize,
    /// Whether generation finished with EOS
    pub finished_with_eos: bool,
}

/// Preprocess text to phoneme IDs
///
/// Returns (phoneme_ids, phoneme_strings, word2ph, text_normalized)
pub fn preprocess_text(text: &str) -> (Array, Vec<String>, Vec<i32>, String) {
    preprocess_text_with_lang(text, None)
}

/// Preprocess text with explicit language override
pub fn preprocess_text_with_lang(text: &str, language: Option<crate::text::Language>) -> (Array, Vec<String>, Vec<i32>, String) {
    // GPT-SoVITS format: no BOS/EOS, punctuation at end serves as sentence delimiter
    let config = PreprocessorConfig {
        add_bos: false,
        add_eos: false,
        ..PreprocessorConfig::default()
    };
    let preprocessor = TextPreprocessor::new(config);

    // Convert text to phonemes
    let output = preprocessor.preprocess(text, language);

    // Use phonemes directly - Python doesn't add any end marker,
    // the trailing punctuation (comma, period, etc.) is the sentence delimiter
    let phonemes = output.phonemes.clone();
    let word2ph = output.word2ph.clone();
    let text_normalized = output.text_normalized.clone();

    // Convert to IDs
    let phoneme_refs: Vec<&str> = phonemes.iter().map(|s| s.as_str()).collect();
    let ids = symbols_to_ids(&phoneme_refs);

    let phoneme_ids = Array::from_slice(&ids, &[1, ids.len() as i32]);

    (phoneme_ids, phonemes, word2ph, text_normalized)
}

/// Build a stateful sampler honoring all GenerationConfig sampling knobs
/// (top_k, top_p, temperature, repetition_penalty).
fn build_sampler(config: &GenerationConfig) -> Sampler {
    Sampler::new(SamplingConfig {
        top_k: config.top_k as i32,
        top_p: config.top_p,
        // Sampler treats temperature 0 as ~greedy (clamped to 1e-5)
        temperature: config.temperature,
        repetition_penalty: config.repetition_penalty,
        eos_token: config.eos_token_id,
    })
}

/// Generate semantic tokens autoregressively
///
/// # Arguments
///
/// * `model` - T2S model
/// * `phoneme_ids` - Phoneme token IDs [batch, seq]
/// * `bert_features` - BERT features [batch, seq, 1024]
/// * `config` - Generation configuration
pub fn generate_semantic_tokens<C>(
    model: &mut T2SModel,
    phoneme_ids: &Array,
    bert_features: &Array,
    config: &GenerationConfig,
) -> Result<GenerationOutput, Error>
where
    C: KeyValueCache + Default,
{
    use mlx_rs::module::Module;

    let batch_size = phoneme_ids.shape()[0];

    // Initialize with start token (0)
    let current_token = Array::zeros::<i32>(&[batch_size, 1])
        .map_err(|e| Error::Message(format!("Failed to create start token: {e}")))?;
    let mut all_tokens = vec![current_token.clone()];

    // Create KV caches
    let num_layers = model.config.num_layers as usize;
    let mut caches: Vec<Option<C>> = (0..num_layers).map(|_| None).collect();

    // Prefill: process phonemes and BERT features
    let input = T2SInput {
        phoneme_ids,
        semantic_ids: &current_token,
        bert_features,
        cache: &mut caches,
    };

    let logits = model.forward(input)
        .map_err(|e| Error::Message(format!("T2S prefill failed: {e}")))?;

    // Get logits for next token (take last position)
    let seq_len = logits.shape()[1];
    let next_logits = logits.index((.., seq_len - 1.., ..))
        .squeeze()
        .map_err(|e| Error::Message(format!("Failed to squeeze logits: {e}")))?;

    // Sampler honoring top_k / top_p / temperature / repetition_penalty
    let mut sampler = build_sampler(config);

    // Sample first token; mask EOS if we have not reached min_tokens yet
    // (the min_tokens gate applies to the first sampled token too).
    let (first_id, _) = if config.min_tokens > 0 {
        sampler.sample_with_eos_mask(&next_logits)?
    } else {
        sampler.sample(&next_logits)?
    };
    sampler.add_token(first_id);
    let mut next_token = Array::from_slice(&[first_id], &[batch_size, 1]);
    all_tokens.push(next_token.clone());

    let mut finished = first_id == config.eos_token_id;

    // Autoregressive generation
    for step in 1..config.max_tokens {
        if finished {
            break;
        }

        // Process only the new token
        let input = T2SInput {
            phoneme_ids,
            semantic_ids: &next_token,
            bert_features,
            cache: &mut caches,
        };

        let logits = model.forward(input)
            .map_err(|e| Error::Message(format!("T2S step {step} failed: {e}")))?;

        let seq_len = logits.shape()[1];
        let next_logits = logits.index((.., seq_len - 1.., ..))
            .squeeze()
            .map_err(|e| Error::Message(format!("Failed to squeeze: {e}")))?;

        // Sample next token; mask EOS until min_tokens is reached
        let (token_id, _) = if step < config.min_tokens {
            sampler.sample_with_eos_mask(&next_logits)?
        } else {
            sampler.sample(&next_logits)?
        };
        sampler.add_token(token_id);
        next_token = Array::from_slice(&[token_id], &[batch_size, 1]);
        all_tokens.push(next_token.clone());

        // Check for EOS
        if step >= config.min_tokens && token_id == config.eos_token_id {
            finished = true;
        }
    }

    // Concatenate all tokens
    let token_refs: Vec<&Array> = all_tokens.iter().collect();
    let tokens = concatenate_axis(&token_refs, 1)
        .map_err(|e| Error::Message(format!("Failed to concatenate tokens: {e}")))?;

    // Remove start token
    let tokens = tokens.index((.., 1..));

    let num_tokens = tokens.shape()[1] as usize;

    Ok(GenerationOutput {
        tokens,
        num_tokens,
        finished_with_eos: finished,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generation_config_default() {
        let config = GenerationConfig::default();
        assert_eq!(config.max_tokens, 500);
        assert_eq!(config.temperature, 0.8);
        assert_eq!(config.top_k, 3);
        assert_eq!(config.eos_token_id, 1024);
    }

    #[test]
    fn test_generation_config_greedy() {
        let config = GenerationConfig::greedy();
        assert_eq!(config.temperature, 0.0);
        assert_eq!(config.top_k, 0);
        assert_eq!(config.top_p, 1.0);
    }

    #[test]
    fn test_preprocess_text() {
        let (ids, phonemes, word2ph, _norm_text) = preprocess_text("你好");
        assert!(!phonemes.is_empty());
        // GPT-SoVITS format: just phonemes, no extra markers (matches Python's cut0)
        // "你好" -> 4 phonemes: n, i3, h, ao3
        assert_eq!(phonemes.len(), 4);
        assert_eq!(ids.shape()[0], 1); // batch size
        // word2ph should have 2 entries: 2 Chinese characters
        assert_eq!(word2ph.len(), 2);
    }
}
