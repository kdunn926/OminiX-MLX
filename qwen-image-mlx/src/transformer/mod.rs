//! Transformer components for Qwen-Image (unquantized building blocks).
//!
//! Reference: diffusers/models/transformers/transformer_qwenimage.py
//!
//! NOTE: the `QwenImagePipeline` built on these is not exercised by any
//! example — `qwen_quantized` is the production path; treat this
//! implementation as unvalidated.

mod norm;
mod attention;
mod feedforward;
mod block;
mod embeddings;
mod rope;
mod transformer;

pub use norm::{QwenLayerNorm, QwenAdaLayerNormContinuous};
pub use attention::QwenTransformerAttention;
pub use feedforward::QwenFeedForward;
pub use block::QwenTransformerBlock;
pub use embeddings::{QwenTimesteps, QwenTimestepEmbedding, QwenTimeTextEmbed};
pub use rope::{QwenEmbedRope, apply_rope};
pub use transformer::{QwenTransformer, QwenTransformerConfig};
