//! Per-layer mixed KV cache for Gemma4.
//!
//! Gemma4 interleaves sliding-window and full-attention layers. Paged KV only
//! makes sense for the **full-attention** layers — sliding layers should keep a
//! contiguous `KVCache` (paging them gathers their whole history into a
//! contiguous copy every decode step, which regresses throughput badly, worst
//! at long context). But Gemma4's cache is a monomorphic `Vec<C>`, so it cannot
//! mix per layer. This enum is the mixed cache type: build it with
//! [`init_mixed_paged_cache`] so full-attention slots use `PagedKvCache` and
//! sliding slots use the contiguous `KVCache`. It implements [`KeyValueCache`]
//! by delegating to the active variant, so `Generate<MixedKvCache>` works
//! unchanged.

use mlx_rs::{error::Exception, Array};
use mlx_rs_core::cache::{KVCache, KeyValueCache};
use mlx_rs_core::paged::PagedKvCache;

use crate::model::Model;
use crate::sliding_cache::SlidingKVCache;

/// A KV cache slot that is one of:
///   - [`MixedKvCache::Kv`]: contiguous [`KVCache`] — full-attention layers
///     when paging is disabled, or sliding-attention layers when the
///     sliding-trim optimization is disabled (legacy behavior).
///   - [`MixedKvCache::Paged`]: [`PagedKvCache`] — full-attention layers
///     with `OMINIX_PAGED_ATTENTION=1`.
///   - [`MixedKvCache::Sliding`]: [`SlidingKVCache`] — sliding-attention
///     layers when [`init_layered_cache`] is used. Caps physical buffer
///     at `window` entries while keeping `offset()` reporting the logical
///     token count (so RoPE stays correct).
#[derive(Debug, Clone)]
pub enum MixedKvCache {
    Kv(KVCache),
    Paged(PagedKvCache),
    Sliding(SlidingKVCache),
}

impl Default for MixedKvCache {
    fn default() -> Self {
        MixedKvCache::Kv(KVCache::default())
    }
}

impl KeyValueCache for MixedKvCache {
    fn offset(&self) -> i32 {
        match self {
            MixedKvCache::Kv(c) => c.offset(),
            MixedKvCache::Paged(c) => c.offset(),
            MixedKvCache::Sliding(c) => c.offset(),
        }
    }

    fn physical_offset(&self) -> i32 {
        match self {
            MixedKvCache::Kv(c) => c.physical_offset(),
            MixedKvCache::Paged(c) => c.physical_offset(),
            MixedKvCache::Sliding(c) => c.physical_offset(),
        }
    }

    fn max_size(&self) -> Option<i32> {
        match self {
            MixedKvCache::Kv(c) => c.max_size(),
            MixedKvCache::Paged(c) => c.max_size(),
            MixedKvCache::Sliding(c) => c.max_size(),
        }
    }

    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
    ) -> Result<(Array, Array), Exception> {
        match self {
            MixedKvCache::Kv(c) => c.update_and_fetch(keys, values),
            MixedKvCache::Paged(c) => c.update_and_fetch(keys, values),
            MixedKvCache::Sliding(c) => c.update_and_fetch(keys, values),
        }
    }

    fn reset(&mut self) {
        match self {
            MixedKvCache::Kv(c) => c.reset(),
            MixedKvCache::Paged(c) => c.reset(),
            MixedKvCache::Sliding(c) => c.reset(),
        }
    }

    fn try_fused_attention(
        &mut self,
        q: &Array,
        k_new: Array,
        v_new: Array,
        scale: f32,
        mask: Option<&Array>,
        kv_repeat: i32,
    ) -> Result<Option<Array>, Exception> {
        match self {
            MixedKvCache::Kv(c) => {
                c.try_fused_attention(q, k_new, v_new, scale, mask, kv_repeat)
            }
            MixedKvCache::Paged(c) => {
                c.try_fused_attention(q, k_new, v_new, scale, mask, kv_repeat)
            }
            // SlidingKVCache uses the default (returns None → fall back to
            // standard `update_and_fetch + SDPA`). Fused-attention would
            // need a sliding-mask-aware kernel.
            MixedKvCache::Sliding(c) => {
                c.try_fused_attention(q, k_new, v_new, scale, mask, kv_repeat)
            }
        }
    }

    fn current_kv(&self) -> Option<(Array, Array)> {
        match self {
            MixedKvCache::Kv(c) => c.current_kv(),
            MixedKvCache::Paged(c) => c.current_kv(),
            MixedKvCache::Sliding(c) => c.current_kv(),
        }
    }

    fn eval(&self) -> Result<(), Exception> {
        match self {
            MixedKvCache::Kv(c) => c.eval(),
            MixedKvCache::Paged(c) => c.eval(),
            MixedKvCache::Sliding(c) => c.eval(),
        }
    }

    fn trim_kv(&mut self, n_drop: i32) -> Result<(), Exception> {
        match self {
            MixedKvCache::Kv(c) => c.trim_kv(n_drop),
            MixedKvCache::Paged(c) => c.trim_kv(n_drop),
            // SlidingKVCache rejects trim_kv: DFlash-style rollback by
            // last-N is undefined for a buffer that has already dropped
            // its older tail. Callers using DFlash with sliding layers
            // must snapshot+restore the SlidingKVCache via `Clone` instead.
            MixedKvCache::Sliding(c) => c.trim_kv(n_drop),
        }
    }

    fn compact_kv(&mut self, past_length: i32, keep_indices: &Array) -> Result<(), Exception> {
        match self {
            MixedKvCache::Kv(c) => c.compact_kv(past_length, keep_indices),
            MixedKvCache::Paged(c) => c.compact_kv(past_length, keep_indices),
            MixedKvCache::Sliding(c) => c.compact_kv(past_length, keep_indices),
        }
    }

    fn compact_to_last_n(&mut self, n: i32) -> Result<(), Exception> {
        match self {
            MixedKvCache::Kv(c) => c.compact_to_last_n(n),
            MixedKvCache::Paged(c) => c.compact_to_last_n(n),
            MixedKvCache::Sliding(c) => c.compact_to_last_n(n),
        }
    }
}

/// Build a per-layer mixed cache: full-attention slots get a `PagedKvCache`,
/// sliding-window slots keep a contiguous `KVCache`. A slot shared by any
/// full-attention layer is paged (so shared KV is preserved correctly).
pub fn init_mixed_paged_cache(model: &Model) -> Vec<MixedKvCache> {
    let inner = &model.model;
    let num_slots = *inner.kv_cache_map.iter().max().unwrap_or(&0) + 1;
    // A slot is "full attention" (→ paged) if any layer using it is non-sliding.
    let mut slot_full = vec![false; num_slots];
    for (i, layer) in inner.layers.iter().enumerate() {
        if layer.self_attn.sliding_window.is_none() {
            slot_full[inner.kv_cache_map[i]] = true;
        }
    }
    // Optional block-size override (PAGED_BLOCK_SIZE) for tuning the paged
    // arena's block granularity. Affects memory packing / sharing granularity;
    // the fused decode kernel's per-position cost is block-size-independent.
    let block_size: Option<i32> = std::env::var("PAGED_BLOCK_SIZE")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n| n > 0);
    slot_full
        .into_iter()
        .map(|full| {
            if full {
                match block_size {
                    Some(bs) => MixedKvCache::Paged(PagedKvCache::new(bs)),
                    None => MixedKvCache::Paged(PagedKvCache::default()),
                }
            } else {
                MixedKvCache::Kv(KVCache::default())
            }
        })
        .collect()
}

/// Build a per-layer mixed cache where **full-attention** slots get an
/// unbounded contiguous [`KVCache`] (today's behavior) and **sliding-
/// attention** slots get a window-bounded [`SlidingKVCache`] sized to that
/// layer's `sliding_window`.
///
/// Use this instead of `init_cache::<KVCache>` for any Gemma4 backend that
/// wants per-layer KV memory + attention-compute savings on long contexts.
/// The returned `Vec<MixedKvCache>` slots into `model.forward` unchanged —
/// the dispatch is via the `KeyValueCache` impl on the enum.
///
/// **Slot-sharing rule** (mirrors [`init_mixed_paged_cache`]): a slot is
/// "full attention" if *any* layer using it is non-sliding. Such a slot
/// uses the unbounded [`KVCache`] so shared KV stays consistent for the
/// full-attention reader. Slots used exclusively by sliding layers (the
/// vast majority on 12B/26B) get the [`SlidingKVCache`].
///
/// **Shared-KV rule**: any slot written by a `kv_store_layers` layer
/// (e4b-style KV sharing) also stays on the unbounded [`KVCache`], even
/// when all its layers are sliding. Shared readers derive their RoPE
/// offset from the *physical* KV length of the snapshot
/// (`shared_k.shape()[2] - L`) and take the snapshot after post-SDPA
/// compaction — a compacted `SlidingKVCache` would pin that offset at
/// `window - 1` while the stored keys keep their true positions,
/// corrupting attention on every reader layer at long context.
///
/// The window per sliding slot is the `sliding_window` of the first layer
/// that uses that slot — Gemma4's config sets it uniformly per layer-type,
/// so per-slot windows match across all referencing layers.
pub fn init_layered_cache(model: &Model) -> Vec<MixedKvCache> {
    let inner = &model.model;
    let num_slots = *inner.kv_cache_map.iter().max().unwrap_or(&0) + 1;

    // Pass 1: identify slot category and (for sliding slots) the window.
    let mut slot_full = vec![false; num_slots];
    let mut slot_window: Vec<Option<i32>> = vec![None; num_slots];
    for (i, layer) in inner.layers.iter().enumerate() {
        let slot = inner.kv_cache_map[i];
        // Shared-KV store slots must stay unbounded: readers RoPE their
        // queries from the snapshot's physical length, which a compacting
        // SlidingKVCache caps at `window` (see doc comment above).
        if inner.kv_store_layers.contains(&i) {
            slot_full[slot] = true;
        }
        match layer.self_attn.sliding_window {
            None => slot_full[slot] = true,
            Some(w) => {
                // Record the first sliding window seen for this slot; the
                // Gemma4 config keeps `sliding_window` uniform across all
                // layers of the same type, so any conflicting w later would
                // indicate a config bug. We don't error — just take the
                // first to keep this helper infallible.
                if slot_window[slot].is_none() {
                    slot_window[slot] = Some(w);
                }
            }
        }
    }

    (0..num_slots)
        .map(|slot| {
            if slot_full[slot] {
                MixedKvCache::Kv(KVCache::default())
            } else {
                let window = slot_window[slot].unwrap_or(1024);
                MixedKvCache::Sliding(SlidingKVCache::new(window))
            }
        })
        .collect()
}
