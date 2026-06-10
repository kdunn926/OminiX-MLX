//! Per-draft-layer cache of projected (k_proj/k_norm/RoPE-applied) context
//! key/value tensors, mirroring Python's `ContextOnlyDraftKVCache`.
//!
//! The draft model's attention does cross-attention from a small noise block
//! to a (potentially very long) target context. Without this cache, every
//! cycle re-projects the entire accumulated target hidden through fc +
//! hidden_norm + per-layer k_proj/v_proj/k_norm/RoPE — that's O(context)
//! work per cycle, and at 18K context dominates the cycle cost. With this
//! cache, each cycle only projects the **delta** of newly committed
//! positions and appends to the cached k/v.

use mlx_rs::{error::Exception, ops::concatenate_axis, Array};

/// Cached post-RoPE k_ctx / v_ctx for one draft layer.
#[derive(Debug, Clone, Default)]
pub struct ProjectedContextCache {
    keys: Option<Array>,   // [B, n_kv_heads, cached_len, head_dim]
    values: Option<Array>, // [B, n_kv_heads, cached_len, head_dim]
    offset: usize,         // number of positions currently cached
}

impl ProjectedContextCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of positions currently stored in the cache.
    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn keys(&self) -> Option<&Array> {
        self.keys.as_ref()
    }

    pub fn values(&self) -> Option<&Array> {
        self.values.as_ref()
    }

    /// Append already-RoPE'd k/v for `appended_positions` new tokens.
    pub fn append(
        &mut self,
        keys: Array,
        values: Array,
        appended_positions: usize,
    ) -> Result<(), Exception> {
        if appended_positions == 0 {
            return Ok(());
        }
        match (self.keys.take(), self.values.take()) {
            (Some(prev_k), Some(prev_v)) => {
                self.keys = Some(concatenate_axis(&[&prev_k, &keys], 2)?);
                self.values = Some(concatenate_axis(&[&prev_v, &values], 2)?);
            }
            _ => {
                self.keys = Some(keys);
                self.values = Some(values);
            }
        }
        self.offset += appended_positions;
        Ok(())
    }

    /// Reset the cache to empty (e.g. start of a new sequence).
    pub fn reset(&mut self) {
        self.keys = None;
        self.values = None;
        self.offset = 0;
    }
}
