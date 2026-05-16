//! MTPLX — Multi-Token Prediction speculative decoding for MLX Rust.
//!
//! Uses the target model's MTP head (when present) to draft K tokens per
//! cycle. The target then verifies the K draft candidates in a single
//! forward pass; greedy acceptance compares draft tokens to the target's
//! argmax.
//!
//! ## Current capability
//!
//! The stock `mlx-community/Qwen3.6-35B-A3B-4bit` checkpoint **strips**
//! the MTP weights (see `mlx_lm.models.qwen3_5.Qwen3_5MoEModel.sanitize`,
//! which deletes every key containing `"mtp."`). As a result this crate
//! detects `target.mtp_head().is_none()` at session start and falls back
//! to autoregressive decoding. The plumbing is in place for the day a
//! checkpoint ships the weights.
//!
//! ## Out of scope
//!
//! See `WIP.md` for the planned, but not-yet-implemented, work:
//!   - Leviathan-Chen probability ratio acceptance + residual `(p-q)+` sampling
//!   - Deterministic GDN-replay Metal kernel
//!   - GraphBank compiled-graph cache
//!   - DeepSeek / GLM / MiMo / Nemotron support

pub mod acceptance;
pub mod session;

pub use acceptance::{
    accept_greedy, accept_speculative, default_rng, seed_default_rng, AcceptanceMode,
    AcceptanceResult,
};
pub use session::{MtplxSession, SessionMetrics, SpeculativeConfig};

/// Errors surfaced by `mtplx-mlx`.
#[derive(Debug, thiserror::Error)]
pub enum MtpError {
    #[error("target model has no MTP head — checkpoint stripped the weights")]
    NoMtpHead,
    #[error("mlx exception: {0}")]
    Mlx(String),
    #[error("core error: {0}")]
    Core(#[from] mlx_rs_core::error::Error),
}

impl From<mlx_rs::error::Exception> for MtpError {
    fn from(e: mlx_rs::error::Exception) -> Self {
        MtpError::Mlx(format!("{e:?}"))
    }
}

/// Returns `true` when the target's MTP head is available for drafting.
///
/// This is a runtime check (does the loaded checkpoint actually carry
/// MTP weights?) rather than a compile-time capability gate. Replaces
/// the old `NoMtpHead` capability error.
pub fn target_supports_mtp(model: &mut qwen3_6_mlx::Model) -> bool {
    model.mtp_head().is_some()
}
