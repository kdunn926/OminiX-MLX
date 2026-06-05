//! PaddleOCR-VL-1.5 inference on Apple Silicon with MLX.
//!
//! In-progress port: only the multimodal RoPE (MROPE) kernel is implemented
//! today as a phase-2-de-risking spike. The rest of the model (Ernie 4.5
//! text decoder, SigLIP-style vision encoder, 2x2 spatial-merge Projector,
//! variable-resolution preprocessor) lives in `docs/paddleocr-vl-1.5-plan.md`.

pub mod mrope;
