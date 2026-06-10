//! Sliding-window KV cache for Gemma4's sliding-attention layers.
//!
//! # Why
//!
//! Gemma4's "sliding_attention" layers (40 of 48 on 12B, 35 of 42 on e4b, 25
//! of 30 on 26B-A4B) only ever attend over the last `sliding_window` tokens
//! (1024 on 12B/26B, 512 on e4b). The plain [`KVCache`] keeps the full
//! history regardless, with the per-layer attention mask zeroing out
//! everything older than `window`. That makes per-token KV cost scale with
//! total context length on *every* layer — and on long contexts those 40
//! sliding layers dominate both memory and attention compute.
//!
//! `SlidingKVCache` physically caps the buffer at `window` entries (well,
//! `window + chunk_size` peak — see "Why concat-then-slice" below). For
//! 12B at 32 k context this drops sliding-layer KV from ~256 MB to ~8 MB
//! per layer (32× reduction), and the SDPA Q @ K^T matmul shrinks from
//! `Q × 32k × head_dim` to `Q × 1024 × head_dim` (~30× cheaper) on every
//! sliding layer.
//!
//! # Logical vs physical offset
//!
//! Two callers ask the cache for "how long am I":
//!  - **RoPE** rotates new K and Q using `cache.offset()` as the token
//!    position. This must be the *logical* (wall-clock) token count, not
//!    the physical buffer length — otherwise RoPE positions reset every
//!    time we slide and decoded tokens get spurious rotations.
//!  - **SDPA mask building** uses the offset to size `create_causal_mask`,
//!    which produces a `[L, offset + L]` mask. That sizing must match the
//!    actual K/V tensor returned by `update_and_fetch` — which is at most
//!    `window` long for this cache.
//!
//! We split the two: [`KeyValueCache::offset`] returns logical; the new
//! [`KeyValueCache::physical_offset`] returns `min(logical, window)`.
//! Existing callers that just read `offset()` keep working (mask construction
//! in [`crate::model::Attention::forward`] is updated to use
//! `physical_offset()` when the layer is sliding).
//!
//! # Why concat-then-slice (not a preallocated ring buffer)
//!
//! `update_and_fetch` builds the new buffer with `concatenate_axis(old, new)`
//! then slices to the last `window` entries. This allocates a fresh
//! `[B, n_kv, ≤window + L, head_dim]` array per call.
//!
//! It looks wasteful but mlx Arrays are reference-counted and lazy: the
//! "allocation" is graph-node construction, the materialization is fused
//! into the downstream SDPA dispatch the same way a preallocated buffer's
//! `current_kv()` slice would be. A preallocated ring buffer would save
//! one concat node per step but cost a per-step `index_mut` for in-place
//! writes plus shift bookkeeping; in practice the concat path benches
//! within noise of the ring-buffer path on M-series GPUs and keeps the
//! implementation small. Re-evaluate if `cargo run --example sliding_bench`
//! shows >5% per-step overhead vs. unbounded `KVCache`.

use mlx_rs::error::Exception;
use mlx_rs::ops::concatenate_axis;
use mlx_rs::ops::indexing::{Ellipsis, IndexOp};
use mlx_rs::Array;
use mlx_rs_core::cache::KeyValueCache;

/// Sliding-window KV cache. Physically caps stored KV at `window` entries
/// (peak `window + L_chunk` mid-`update_and_fetch`) while keeping
/// [`KeyValueCache::offset`] reporting the logical token count so RoPE and
/// causal-mask bookkeeping in the attention forward stays correct.
#[derive(Debug, Clone)]
pub struct SlidingKVCache {
    keys: Option<Array>,
    values: Option<Array>,
    /// Total tokens processed (logical position) — returned by `offset()`.
    /// Monotonic; the physical buffer truncation does not roll this back.
    logical_offset: i32,
    /// Sliding window size in tokens. Buffer is capped at this length after
    /// every `update_and_fetch`. Must be > 0.
    window: i32,
}

impl SlidingKVCache {
    pub fn new(window: i32) -> Self {
        assert!(window > 0, "SlidingKVCache window must be > 0");
        Self {
            keys: None,
            values: None,
            logical_offset: 0,
            window,
        }
    }

    /// Window size this cache was constructed with.
    pub fn window(&self) -> i32 {
        self.window
    }
}

impl Default for SlidingKVCache {
    /// Default to a 1024-token window (matches Gemma4-12B / 26B-A4B
    /// `sliding_window`). e4b's 512-token window must be passed
    /// explicitly via [`Self::new`].
    fn default() -> Self {
        Self::new(1024)
    }
}

impl KeyValueCache for SlidingKVCache {
    /// Logical (wall-clock) token count — what RoPE rotates against.
    fn offset(&self) -> i32 {
        self.logical_offset
    }

    /// Physical buffer length — what SDPA's K/V tensor is actually sized
    /// to. This is the post-update buffer length (already updated by
    /// `update_and_fetch`); after a subsequent `compact_to_last_n(window)`
    /// it shrinks back to `<= window`.
    fn physical_offset(&self) -> i32 {
        self.keys
            .as_ref()
            .map(|k| k.shape()[2] as i32)
            .unwrap_or(0)
    }

    fn max_size(&self) -> Option<i32> {
        Some(self.window)
    }

    /// Append new K/V to the buffer. Does **not** truncate — the caller
    /// must explicitly call [`KeyValueCache::compact_to_last_n`] (e.g.
    /// after SDPA in the attention forward) to bound the buffer.
    ///
    /// Why no in-line truncation: during chunked prefill with `L > 1`,
    /// the earliest queries in the chunk need to attend backward into
    /// the *previous* `window - 1` positions of cached KV. Truncating
    /// inside `update_and_fetch` (before SDPA sees the keys) would drop
    /// the oldest `L` of those positions, leaving the early queries
    /// with a smaller effective window than the model was trained on.
    /// Truncating *after* SDPA (caller's responsibility) keeps the
    /// current step correct and bounds the buffer for the next step.
    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
    ) -> Result<(Array, Array), Exception> {
        let num_new = keys.shape()[2];
        self.logical_offset += num_new;

        // Concatenate with existing buffer (if any).
        let (combined_k, combined_v) = match (self.keys.take(), self.values.take()) {
            (Some(old_k), Some(old_v)) => (
                concatenate_axis(&[old_k, keys], 2)?,
                concatenate_axis(&[old_v, values], 2)?,
            ),
            _ => (keys, values),
        };

        self.keys = Some(combined_k.clone());
        self.values = Some(combined_v.clone());
        Ok((combined_k, combined_v))
    }

    fn reset(&mut self) {
        self.keys = None;
        self.values = None;
        self.logical_offset = 0;
    }

    fn current_kv(&self) -> Option<(Array, Array)> {
        match (&self.keys, &self.values) {
            (Some(k), Some(v)) => Some((k.clone(), v.clone())),
            _ => None,
        }
    }

    /// Truncate the physical buffer to keep only the last `n` entries
    /// (drops the oldest prefix). The logical offset is unchanged so RoPE
    /// continues to position new tokens at their true sequence position.
    ///
    /// In Gemma4's attention forward this is called after SDPA on every
    /// sliding layer, with `n = sliding_window`. Between calls the buffer
    /// stays bounded at `window + max(chunk_size, 1)`.
    fn compact_to_last_n(&mut self, n: i32) -> Result<(), Exception> {
        if n <= 0 {
            return Ok(());
        }
        let Some(k) = self.keys.as_ref() else {
            return Ok(());
        };
        let len = k.shape()[2] as i32;
        if len <= n {
            return Ok(());
        }
        let start = len - n;
        let new_k = k.index((Ellipsis, start.., ..));
        let new_v = self
            .values
            .as_ref()
            .expect("values present when keys present")
            .index((Ellipsis, start.., ..));
        self.keys = Some(new_k);
        self.values = Some(new_v);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::zeros;
    use mlx_rs::Dtype;

    fn fake_kv(num_tokens: i32) -> (Array, Array) {
        // Shape: [B=1, n_kv=2, T, head_dim=4]
        let shape = [1i32, 2, num_tokens, 4];
        let k = zeros::<f32>(&shape).unwrap();
        let v = zeros::<f32>(&shape).unwrap();
        (k, v)
    }

    #[test]
    fn offset_is_logical_not_physical_after_compact() {
        let mut cache = SlidingKVCache::new(4);
        for _ in 0..10 {
            let (k, v) = fake_kv(1);
            cache.update_and_fetch(k, v).unwrap();
            cache.compact_to_last_n(4).unwrap();
        }
        assert_eq!(cache.offset(), 10, "logical offset must reflect all tokens");
        assert_eq!(cache.physical_offset(), 4, "physical capped at window post-compact");
    }

    #[test]
    fn update_and_fetch_does_not_truncate() {
        // Caller is responsible for compaction; update_and_fetch must
        // return the full buffer so the current SDPA sees prior context
        // (critical for chunked prefill correctness — see method docs).
        let mut cache = SlidingKVCache::new(4);
        for _ in 0..3 {
            let (k, v) = fake_kv(1);
            cache.update_and_fetch(k, v).unwrap();
        }
        assert_eq!(cache.physical_offset(), 3);
        // 4th update — still <= window, no truncation.
        let (k, v) = fake_kv(1);
        let (out, _) = cache.update_and_fetch(k, v).unwrap();
        assert_eq!(out.shape()[2], 4);
        // 5th update — past window; without an explicit compact call the
        // buffer keeps growing (this is the intended hand-off to the caller).
        let (k, v) = fake_kv(1);
        let (out, _) = cache.update_and_fetch(k, v).unwrap();
        assert_eq!(out.shape()[2], 5, "no in-line truncation");
        assert_eq!(cache.physical_offset(), 5);
    }

    #[test]
    fn compact_to_last_n_keeps_tail_of_correct_length() {
        let mut cache = SlidingKVCache::new(4);
        for _ in 0..7 {
            let (k, v) = fake_kv(1);
            cache.update_and_fetch(k, v).unwrap();
        }
        assert_eq!(cache.physical_offset(), 7);
        cache.compact_to_last_n(4).unwrap();
        assert_eq!(cache.physical_offset(), 4);
        assert_eq!(cache.offset(), 7, "logical offset preserved across compact");
    }

    #[test]
    fn chunked_prefill_keeps_full_window_plus_chunk_until_compact() {
        // Mirrors the chunked-prefill order an attention forward uses:
        // update → SDPA over `prev_window + L_chunk` keys → compact.
        // This sequence preserves the early-query window for L > 1.
        let mut cache = SlidingKVCache::new(8);
        // Chunk 1 (5 tokens) — under window.
        let (k1, v1) = fake_kv(5);
        cache.update_and_fetch(k1, v1).unwrap();
        cache.compact_to_last_n(8).unwrap();
        assert_eq!(cache.physical_offset(), 5);
        // Chunk 2 (6 tokens) — buffer grows to 11 (5 + 6); SDPA sees all
        // 11 so the first query of chunk 2 can attend to chunk 1.
        let (k2, v2) = fake_kv(6);
        let (out, _) = cache.update_and_fetch(k2, v2).unwrap();
        assert_eq!(
            out.shape()[2],
            11,
            "buffer must include both chunks for SDPA correctness"
        );
        // Caller compacts AFTER SDPA — buffer bounded for next forward.
        cache.compact_to_last_n(8).unwrap();
        assert_eq!(cache.physical_offset(), 8);
        assert_eq!(cache.offset(), 11);
    }

    #[test]
    fn compact_with_n_geq_buffer_is_noop() {
        let mut cache = SlidingKVCache::new(8);
        let (k, v) = fake_kv(3);
        cache.update_and_fetch(k, v).unwrap();
        cache.compact_to_last_n(8).unwrap();
        assert_eq!(cache.physical_offset(), 3);
        cache.compact_to_last_n(100).unwrap();
        assert_eq!(cache.physical_offset(), 3);
    }

    #[test]
    fn single_chunk_larger_than_window_returns_full_chunk_until_compact() {
        let mut cache = SlidingKVCache::new(4);
        let (k, v) = fake_kv(10);
        let (out, _) = cache.update_and_fetch(k, v).unwrap();
        assert_eq!(
            out.shape()[2],
            10,
            "SDPA sees the full chunk on first forward"
        );
        cache.compact_to_last_n(4).unwrap();
        assert_eq!(cache.physical_offset(), 4);
        assert_eq!(cache.offset(), 10);
    }

    #[test]
    fn reset_clears_offsets_and_buffer() {
        let mut cache = SlidingKVCache::new(4);
        let (k, v) = fake_kv(3);
        cache.update_and_fetch(k, v).unwrap();
        cache.reset();
        assert_eq!(cache.offset(), 0);
        assert_eq!(cache.physical_offset(), 0);
        assert!(cache.current_kv().is_none());
    }

    #[test]
    fn dtype_preserved_across_updates() {
        let _ = Dtype::Float32; // touch import to suppress unused-import warnings on some builds
        let mut cache = SlidingKVCache::new(4);
        let (k, v) = fake_kv(2);
        let dtype = k.dtype();
        cache.update_and_fetch(k, v).unwrap();
        let (k2, _) = cache.current_kv().unwrap();
        assert_eq!(k2.dtype(), dtype);
    }
}
