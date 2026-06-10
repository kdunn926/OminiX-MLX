//! Integration tests for GPT-SoVITS voice cloning
//!
//! These tests verify the end-to-end functionality of the voice cloning pipeline.
//! Some tests require model files and are skipped if models aren't available.

use gpt_sovits_mlx::voice_clone::{VoiceCloner, VoiceClonerConfig, SynthesisOptions};
use gpt_sovits_mlx::error::Error;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// Root directory for test models.
///
/// Reads `OMINIX_MODELS_DIR`, falling back to `~/.OminiX/models`.
fn model_root() -> PathBuf {
    if let Ok(dir) = std::env::var("OMINIX_MODELS_DIR") {
        return PathBuf::from(dir);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    PathBuf::from(home).join(".OminiX/models")
}

/// Path to the reference audio used by the model-gated tests
fn reference_audio_path() -> PathBuf {
    model_root().join("moyoyo/ref_audios/doubao_ref_mix_new.wav")
}

/// Test helper: check if default model paths exist; prints a loud skip
/// message (with the missing paths) when they don't.
fn models_available() -> bool {
    let config = VoiceClonerConfig::default();
    let missing: Vec<&String> = [&config.t2s_weights, &config.bert_weights, &config.vits_weights]
        .into_iter()
        .filter(|p| !Path::new(p.as_str()).exists())
        .collect();
    if missing.is_empty() {
        true
    } else {
        eprintln!(
            "=== SKIPPING test: model files not available: {:?} \
             (set GPT_SOVITS_MODEL_DIR or place models under ~/.OminiX/models/gpt-sovits-mlx) ===",
            missing
        );
        false
    }
}

/// Test helper: check if reference audio exists; prints a loud skip message
/// when it doesn't.
fn reference_audio_available() -> bool {
    let path = reference_audio_path();
    if path.exists() {
        true
    } else {
        eprintln!(
            "=== SKIPPING test: reference audio not found at {} \
             (set OMINIX_MODELS_DIR to override the model root) ===",
            path.display()
        );
        false
    }
}

// ============================================================================
// Config Tests
// ============================================================================

#[test]
fn test_config_default() {
    let config = VoiceClonerConfig::default();

    // Check sampling parameters are reasonable
    assert!(config.top_k > 0, "top_k should be positive");
    assert!(config.top_p > 0.0 && config.top_p <= 1.0, "top_p should be in (0, 1]");
    assert!(config.temperature > 0.0, "temperature should be positive");
    assert!(config.repetition_penalty >= 1.0, "repetition_penalty should be >= 1.0");

    // Check sample rate
    assert_eq!(config.sample_rate, 32000, "default sample rate should be 32kHz");
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[test]
fn test_error_empty_input() {
    if !models_available() {
        return;
    }

    let config = VoiceClonerConfig::default();
    let cloner_result = VoiceCloner::new(config);

    if let Ok(mut cloner) = cloner_result {
        // Should fail with empty input
        let result = cloner.synthesize("");
        assert!(matches!(result, Err(Error::EmptyInput)));

        // Should also fail with whitespace-only input
        let result = cloner.synthesize("   \n\t  ");
        assert!(matches!(result, Err(Error::EmptyInput)));
    }
}

#[test]
fn test_error_reference_not_set() {
    if !models_available() {
        return;
    }

    let config = VoiceClonerConfig::default();
    if let Ok(mut cloner) = VoiceCloner::new(config) {
        // Should fail when no reference is set
        let result = cloner.synthesize("Hello");
        assert!(matches!(result, Err(Error::ReferenceNotSet)));
    }
}

#[test]
fn test_error_text_too_long() {
    if !models_available() {
        return;
    }

    let config = VoiceClonerConfig::default();
    if let Ok(mut cloner) = VoiceCloner::new(config) {
        // Create text that's too long (> 10000 chars)
        let long_text = "你好".repeat(6000);
        let result = cloner.synthesize(&long_text);
        assert!(matches!(result, Err(Error::TextTooLong { .. })));
    }
}

// ============================================================================
// Synthesis Options Tests
// ============================================================================

#[test]
fn test_synthesis_options_default() {
    let options = SynthesisOptions::default();
    assert!(options.timeout.is_none());
    assert!(options.cancel_token.is_none());
    assert!(options.max_tokens_per_chunk.is_none());
}

#[test]
fn test_synthesis_options_with_timeout() {
    let options = SynthesisOptions::with_timeout(Duration::from_secs(30));
    assert_eq!(options.timeout, Some(Duration::from_secs(30)));
}

#[test]
fn test_synthesis_options_with_cancel_token() {
    let token = Arc::new(AtomicBool::new(false));
    let options = SynthesisOptions::with_cancel_token(token.clone());

    assert!(options.cancel_token.is_some());

    // Test cancellation
    token.store(true, Ordering::Relaxed);
    assert!(options.cancel_token.as_ref().unwrap().load(Ordering::Relaxed));
}

// ============================================================================
// Full Pipeline Tests (require models)
// ============================================================================

#[test]
#[ignore = "Requires model files and reference audio"]
fn test_synthesize_chinese() {
    if !models_available() || !reference_audio_available() {
        return;
    }

    let config = VoiceClonerConfig::default();
    let mut cloner = VoiceCloner::new(config).expect("Failed to create VoiceCloner");

    cloner.set_reference_audio(reference_audio_path())
        .expect("Failed to set reference audio");

    let audio = cloner.synthesize("你好世界").expect("Synthesis failed");

    // Verify output
    assert!(!audio.samples.is_empty(), "Output should not be empty");
    assert!(audio.samples.len() > 16000, "Output should be at least 0.5s at 32kHz");
    assert_eq!(audio.sample_rate, 32000, "Sample rate should be 32kHz");
    assert!(audio.num_tokens > 0, "Should have generated some tokens");

    // Check samples are in valid range
    let max_sample = audio.samples.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min_sample = audio.samples.iter().cloned().fold(f32::INFINITY, f32::min);
    assert!(max_sample <= 1.0 && min_sample >= -1.0, "Samples should be in [-1, 1] range");

    // Check it's not silence
    let energy: f32 = audio.samples.iter().map(|s| s * s).sum();
    assert!(energy > 0.01, "Output should not be silence");
}

#[test]
#[ignore = "Requires model files and reference audio"]
fn test_synthesize_mixed_language() {
    if !models_available() || !reference_audio_available() {
        return;
    }

    let config = VoiceClonerConfig::default();
    let mut cloner = VoiceCloner::new(config).expect("Failed to create VoiceCloner");

    cloner.set_reference_audio(reference_audio_path())
        .expect("Failed to set reference audio");

    // Mixed Chinese and English
    let audio = cloner.synthesize("Hello世界，这是一个test。").expect("Synthesis failed");

    assert!(!audio.samples.is_empty(), "Output should not be empty");
    assert!(audio.samples.len() > 16000, "Output should be at least 0.5s");
}

#[test]
#[ignore = "Requires model files and reference audio"]
fn test_synthesize_with_cancellation() {
    if !models_available() || !reference_audio_available() {
        return;
    }

    let config = VoiceClonerConfig::default();
    let mut cloner = VoiceCloner::new(config).expect("Failed to create VoiceCloner");

    cloner.set_reference_audio(reference_audio_path())
        .expect("Failed to set reference audio");

    // Create a pre-cancelled token
    let cancel_token = Arc::new(AtomicBool::new(true));
    let options = SynthesisOptions::with_cancel_token(cancel_token);

    let result = cloner.synthesize_with_options("你好", options);
    assert!(matches!(result, Err(Error::Cancelled)));
}

// ============================================================================
// Audio Output Tests
// ============================================================================

#[test]
fn test_audio_output_to_i16() {
    use gpt_sovits_mlx::voice_clone::AudioOutput;

    let audio = AudioOutput {
        samples: vec![0.0, 0.5, -0.5, 1.0, -1.0],
        sample_rate: 32000,
        duration: 0.0,
        num_tokens: 0,
    };

    let i16_samples = audio.to_i16_samples();

    assert_eq!(i16_samples.len(), 5);
    assert_eq!(i16_samples[0], 0);  // 0.0 -> 0
    assert!(i16_samples[1] > 16000);  // 0.5 -> ~16383
    assert!(i16_samples[2] < -16000);  // -0.5 -> ~-16384
    assert_eq!(i16_samples[3], 32767);  // 1.0 -> 32767
    assert_eq!(i16_samples[4], -32767);  // -1.0 -> -32767
}
