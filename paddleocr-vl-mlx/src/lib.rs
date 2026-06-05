//! PaddleOCR-VL-1.5 inference on Apple Silicon with MLX.
//!
//! In-progress port: only the multimodal RoPE (MROPE) kernel is implemented
//! today as a phase-2-de-risking spike. The rest of the model (Ernie 4.5
//! text decoder, SigLIP-style vision encoder, 2x2 spatial-merge Projector,
//! variable-resolution preprocessor) lives in `docs/paddleocr-vl-1.5-plan.md`.

pub mod config;
pub mod mrope;
pub mod position_ids;
pub mod text_model;
pub mod vision_model;

pub use config::{
    load_tokenizer, PaddleOcrVisionConfig, PaddleOcrVlConfig, RopeScaling, SpecialTokens,
};
pub use text_model::{
    build_with_random_weights, AttentionInput, Ernie45Attention, Ernie45DecoderLayer,
    Ernie45ForCausalLM, Ernie45Mlp, Ernie45Model, LayerInput,
};
pub use vision_model::{
    build_vision_with_random_weights, VisionAttention, VisionEmbeddings, VisionEncoderLayer,
    VisionLayerNorm, VisionMlp, VisionTransformer,
};
