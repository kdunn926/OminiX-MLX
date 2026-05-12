//! Integration tests for model loading — both 35B MoE and 27B dense variants.
//!
//! These tests gate on the existence of local model directories; they are
//! skipped automatically in CI environments that lack the model weights.

use std::path::Path;

const MODEL_35B: &str = "../models/Qwen3.6-35B-A3B-4bit";
const MODEL_27B: &str = "../models/Qwen3.6-27B-4bit";

// ============================================================================
// Config deserialization (no weights needed)
// ============================================================================

#[test]
fn config_35b_is_moe() {
    let config_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(MODEL_35B)
        .join("config.json");
    if !config_path.exists() {
        eprintln!("SKIP: 35B config not found at {:?}", config_path);
        return;
    }
    let file = std::fs::File::open(&config_path).expect("open config.json");
    let args: qwen3_6_mlx::ModelArgs = serde_json::from_reader(file).expect("parse config.json");
    assert!(
        args.text_config.is_moe(),
        "35B model should be identified as MoE"
    );
}

#[test]
fn config_27b_is_dense() {
    let config_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(MODEL_27B)
        .join("config.json");
    if !config_path.exists() {
        eprintln!("SKIP: 27B config not found at {:?}", config_path);
        return;
    }
    let file = std::fs::File::open(&config_path).expect("open config.json");
    let args: qwen3_6_mlx::ModelArgs = serde_json::from_reader(file).expect("parse config.json");
    assert!(
        !args.text_config.is_moe(),
        "27B model should NOT be identified as MoE"
    );
    // Verify expected architecture dimensions
    assert_eq!(args.text_config.hidden_size, 5120);
    assert_eq!(args.text_config.num_hidden_layers, 64);
    assert_eq!(args.text_config.vocab_size, 248320);
}

// ============================================================================
// Full model load (requires weights on disk)
// ============================================================================

#[test]
fn load_35b_moe_model() {
    let model_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(MODEL_35B);
    if !model_dir.exists() {
        eprintln!("SKIP: 35B model dir not found at {:?}", model_dir);
        return;
    }
    let result = qwen3_6_mlx::load_model(&model_dir);
    assert!(
        result.is_ok(),
        "35B MoE model load failed: {:?}",
        result.err()
    );
    let model = result.unwrap();
    assert!(model.lm_head.is_some() || !model.args.tie_word_embeddings);
    eprintln!("35B MoE model loaded OK");
}

#[test]
fn load_27b_dense_model() {
    let model_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(MODEL_27B);
    if !model_dir.exists() {
        eprintln!("SKIP: 27B model dir not found at {:?}", model_dir);
        return;
    }
    let result = qwen3_6_mlx::load_model(&model_dir);
    assert!(
        result.is_ok(),
        "27B dense model load failed: {:?}",
        result.err()
    );
    let model = result.unwrap();
    // All layers should exist
    assert_eq!(
        model.text_model.layers.len(),
        64,
        "27B should have 64 layers"
    );
    eprintln!("27B dense model loaded OK");
}
