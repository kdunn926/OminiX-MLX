//! EAGLE-3 speculative decoding for MLX Rust.
//!
//! Loads speculators-format EAGLE-3 draft checkpoints (e.g.
//! `RedHatAI/gemma-4-26B-A4B-it-speculator.eagle3`) and runs them against a
//! Gemma4 target through the shared `dflash_mlx::DFlashSession` cycle:
//! the target's per-layer hidden captures feed the draft's `fc` fusion, the
//! single-layer draft chains greedily in feature space, and the target
//! verifies each block.
//!
//! EAGLE-3 is the **default** decode path for targets with a discoverable
//! draft checkpoint; opt out with `OMINIX_EAGLE3=0` — see [`session`] for
//! the knob table and discovery rules. Reference semantics: vLLM
//! `llama_eagle3.py` / llama.cpp commit 88a3927.

pub mod adapter;
pub mod config;
pub mod model;
pub mod session;

pub use adapter::Eagle3DraftAdapter;
pub use config::Eagle3Config;
pub use model::Eagle3DraftModel;
pub use session::{discover_draft, env_enabled, Eagle3Session};
