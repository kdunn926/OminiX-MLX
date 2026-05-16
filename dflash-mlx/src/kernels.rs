//! GDN replay-rollback Metal kernels.
//!
//! The actual kernels live in `mlx-rs-core::metal_kernels` so that they can be
//! shared with `qwen3.6-mlx` (which previously could not depend on
//! `dflash-mlx` without creating a circular dependency).  This module
//! preserves the original `dflash-mlx::kernels::{gated_delta_with_tape,
//! tape_replay}` symbols as thin shims that delegate to the shared
//! implementations.
//!
//! Shape contract (matches `mlx_rs_core::deltanet_recurrence`):
//!   q, k     : [B, H, L, K]
//!   v        : [B, H, L, V]
//!   decay,
//!   beta     : [B, H, L]
//!   state_in : [B, H, K, V]
//! Returns:
//!   (output [B, H, L, V], state_out [B, H, K, V], tape [B, H, L, V]).
//!
//! `tape_replay` consumes `(tape, k, decay, state_in)` and returns
//! `state_out [B, H, K, V]`.

use mlx_rs::{error::Exception, Array};

/// Re-export of `mlx_rs_core::deltanet_with_tape`.
///
/// Runs the gated-delta recurrent scan with the same decayed-state retrieval
/// semantics as `mlx_rs_core::deltanet_recurrence`, additionally returning
/// the per-step `delta` (innovation) tape so the state can later be replayed
/// from a snapshot.
pub fn gated_delta_with_tape(
    q: &Array,
    k: &Array,
    v: &Array,
    decay: &Array,
    beta: &Array,
    state_in: &Array,
) -> Result<(Array, Array, Array), Exception> {
    mlx_rs_core::deltanet_with_tape(q, k, v, decay, beta, state_in)
}

/// Re-export of `mlx_rs_core::deltanet_tape_replay`.
///
/// Advances the recurrent state by L tape steps from a snapshotted
/// `state_in` using the recorded innovation tape, key tensor, and decay
/// gates.  Used by speculative-decoding rollback to skip rerunning the
/// scan for accepted tokens.
pub fn tape_replay(
    tape: &Array,
    k: &Array,
    decay: &Array,
    state_in: &Array,
) -> Result<Array, Exception> {
    mlx_rs_core::deltanet_tape_replay(tape, k, decay, state_in)
}
