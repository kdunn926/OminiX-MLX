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

/// A KV cache slot that is either a contiguous `KVCache` (sliding layers) or a
/// `PagedKvCache` (full-attention layers).
#[derive(Debug)]
pub enum MixedKvCache {
    Kv(KVCache),
    Paged(PagedKvCache),
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
        }
    }

    fn max_size(&self) -> Option<i32> {
        match self {
            MixedKvCache::Kv(c) => c.max_size(),
            MixedKvCache::Paged(c) => c.max_size(),
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
        }
    }

    fn reset(&mut self) {
        match self {
            MixedKvCache::Kv(c) => c.reset(),
            MixedKvCache::Paged(c) => c.reset(),
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
        }
    }

    fn current_kv(&self) -> Option<(Array, Array)> {
        match self {
            MixedKvCache::Kv(c) => c.current_kv(),
            MixedKvCache::Paged(c) => c.current_kv(),
        }
    }

    fn eval(&self) -> Result<(), Exception> {
        match self {
            MixedKvCache::Kv(c) => c.eval(),
            MixedKvCache::Paged(c) => c.eval(),
        }
    }

    fn trim_kv(&mut self, n_drop: i32) -> Result<(), Exception> {
        match self {
            MixedKvCache::Kv(c) => c.trim_kv(n_drop),
            MixedKvCache::Paged(c) => c.trim_kv(n_drop),
        }
    }

    fn compact_kv(&mut self, past_length: i32, keep_indices: &Array) -> Result<(), Exception> {
        match self {
            MixedKvCache::Kv(c) => c.compact_kv(past_length, keep_indices),
            MixedKvCache::Paged(c) => c.compact_kv(past_length, keep_indices),
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
    slot_full
        .into_iter()
        .map(|full| {
            if full {
                MixedKvCache::Paged(PagedKvCache::default())
            } else {
                MixedKvCache::Kv(KVCache::default())
            }
        })
        .collect()
}
