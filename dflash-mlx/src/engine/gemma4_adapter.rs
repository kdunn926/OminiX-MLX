//! DFlash target adapter for `gemma4_mlx::Model` (26B-A4B variant only).
//!
//! Mirrors the structure of `Qwen36TargetAdapter` but for the Gemma4 model
//! family. Scoped to the 26B-A4B (dense + MoE, no PLE, no shared-KV) variant
//! — the E4B path (PLE + shared KV) is intentionally not supported here and
//! will require a separate adapter (TODO).
//!
//! Implementation notes:
//! - KV rollback is done via `gemma4_mlx::snapshot_cache` / `restore_cache`,
//!   followed by replaying the kept-prefix tokens through
//!   `Model::forward_with_hidden_capture`. The Python reference's
//!   `_trim_recent_cache` is not directly available because `mlx-rs-core`'s
//!   `KVCache` has no public offset setter, and the DFlash lane is forbidden
//!   from touching `mlx-rs-core`. Snapshot+replay is the same strategy
//!   `Qwen36TargetAdapter` uses; see `gemma4_mlx::snapshot_cache` docs.
//! - `verify` uses `Model::forward_with_hidden_capture` (last_only=false), so
//!   it DOES return full per-position logits `[B, T, V]` — spec_epoch's
//!   `verify_logits[.., t, ..]` indexing across the block is correct. (An
//!   earlier revision narrowed to last-position logits; that is fixed.)
//!   What remains open is acceptance-ratio PARITY with the Python
//!   `target_gemma4.py`: gemma4-specific numerics (final_logit_softcapping,
//!   the sqrt(hidden) embed scale, sliding-window masks, norm eps) make the
//!   Rust target's hidden states / logits diverge subtly from what the DFlash
//!   draft was trained against, which depresses acceptance (observed ~0.11 vs
//!   ~0.32 on Qwen3.6). That is the 3-6 week deferred parity investigation —
//!   a numerical gap, NOT a wiring/shape bug (target_layer_ids align with the
//!   30-layer target and the draft loads as a real DFlashDraftModel).

use mlx_rs::{error::Exception, ops::concatenate_axis, Array};

use gemma4_mlx::{KVCache, Model};
use mlx_rs_core::cache::{KeyValueCache, TurboQuantKVCache};
use mlx_rs::{argmax_axis, ops::indexing::IndexOp};

use crate::engine::spec_epoch::TargetModel;

/// DFlash target adapter for the 26B-A4B Gemma4 variant.
///
/// Holds the model, its KV cache, target capture layer ids, and the
/// verify-snapshot fields needed to roll back on partial acceptance.
pub struct Gemma4TargetAdapter<C: KeyValueCache + Default = KVCache> {
    model: Model,
    cache: Vec<C>,
    target_layer_ids: Vec<usize>,
    step: usize,
    _temp: f32,

    // Per-cycle captured hidden states. Stored as a Vec<Array> of segments
    // (one per committed verify or prefill chunk) plus a position-prefix-sum
    // so we can serve "captures since offset X" in O(remaining segments)
    // instead of O(total) — avoiding the quadratic-concat cost the previous
    // eager-concat accumulator paid as generation length grew.
    target_hidden_segments: Vec<Array>,
    target_hidden_positions: Vec<usize>, // cumulative end-position of each segment
    // Cached fully-concatenated view, lazily computed on `last_target_hidden`.
    // Invalidated (set to None) on every segment append.
    target_hidden_full_cache: Option<Array>,

    // Verify snapshot — captured at the start of `verify`, restored in
    // `rollback_kv`. With trim-based KV rollback (KVCache::trim), the cache
    // itself isn't cloned; only the segment count is recorded so rollback
    // can drop the latest segment if needed.
    verify_inputs: Option<Array>,
    verify_step: usize,
    verify_segment_count_pre: usize,
}

impl<C: KeyValueCache + Default> Gemma4TargetAdapter<C> {
    /// Construct a Gemma4 target adapter with DFlash hidden-capture wiring.
    ///
    /// `target_layer_ids` corresponds to the draft model's
    /// `target_layer_ids` field (i.e. which layers the DFlash draft consumes
    /// hidden states from). Passing an empty vec disables capture and the
    /// adapter behaves as plain target-only.
    pub fn with_dflash(model: Model, temp: f32, target_layer_ids: Vec<usize>) -> Self {
        let cache = Self::fresh_cache(&model);
        Self {
            model,
            cache,
            target_layer_ids,
            step: 0,
            _temp: temp,
            target_hidden_segments: Vec::new(),
            target_hidden_positions: Vec::new(),
            target_hidden_full_cache: None,
            verify_inputs: None,
            verify_step: 0,
            verify_segment_count_pre: 0,
        }
    }

    fn fresh_cache(model: &Model) -> Vec<C> {
        let num_slots = *model.model.kv_cache_map.iter().max().unwrap_or(&0) + 1;
        gemma4_mlx::init_cache::<C>(num_slots)
    }

    fn append_target_hidden(&mut self, captures: Array) -> Result<(), Exception> {
        let seg_len = captures.shape()[1] as usize;
        let prev_total = self.target_hidden_positions.last().copied().unwrap_or(0);
        self.target_hidden_segments.push(captures);
        self.target_hidden_positions.push(prev_total + seg_len);
        self.target_hidden_full_cache = None;
        // Cap the number of retained hidden-state segments to bound
        // memory growth across long generations. The drafter reads
        // `last_target_hidden` which concats all segments — capping
        // drops the oldest. Sweep 6 on Qwen3.6-27B + 100-token DFlash
        // showed cap=8 is identical in throughput AND acceptance to
        // unbounded retention (acceptance 0.265 in both). Larger caps
        // (32, 128) regressed slightly due to list-management overhead.
        // Default 8; `DFLASH_MAX_HIDDEN_SEGS=0` disables the cap.
        //
        // SEMANTICS: once the cap engages, `last_target_hidden()` is a
        // *sliding window* over the most recent segments, no longer the
        // full context from position 0. The drafter's delta computation
        // and the ProjectedContextCache's cache-relative RoPE offsets
        // then condition on window-relative positions rather than true
        // absolute positions. The sweep above found no acceptance
        // regression on the tested workload, but this is a semantic
        // approximation that may be workload-dependent — disable the cap
        // when debugging acceptance-rate anomalies.
        let cap_default: usize = 8;
        let cap_opt: Option<usize> = std::env::var("DFLASH_MAX_HIDDEN_SEGS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .or(Some(cap_default))
            .filter(|&n| n > 0);
        if let Some(cap) = cap_opt {
            while self.target_hidden_segments.len() > cap {
                self.target_hidden_segments.remove(0);
                self.target_hidden_positions.remove(0);
            }
        }
        Ok(())
    }

    fn truncate_latest_segment_to(&mut self, n_keep: usize) -> Result<(), Exception> {
        if n_keep == 0 {
            self.target_hidden_segments.pop();
            self.target_hidden_positions.pop();
            self.target_hidden_full_cache = None;
            return Ok(());
        }
        let Some(seg) = self.target_hidden_segments.last_mut() else {
            return Ok(());
        };
        let seg_len = seg.shape()[1] as usize;
        if n_keep >= seg_len {
            return Ok(());
        }
        *seg = seg.index((.., ..n_keep as i32, ..));
        let prev_total = if self.target_hidden_positions.len() >= 2 {
            self.target_hidden_positions[self.target_hidden_positions.len() - 2]
        } else {
            0
        };
        let last = self.target_hidden_positions.last_mut().unwrap();
        *last = prev_total + n_keep;
        self.target_hidden_full_cache = None;
        Ok(())
    }

    fn rebuild_full_hidden(&self) -> Result<Option<Array>, Exception> {
        if self.target_hidden_segments.is_empty() {
            return Ok(None);
        }
        if self.target_hidden_segments.len() == 1 {
            return Ok(Some(self.target_hidden_segments[0].clone()));
        }
        let refs: Vec<&Array> = self.target_hidden_segments.iter().collect();
        Ok(Some(concatenate_axis(&refs, 1)?))
    }
}

// DDTree only supports the plain `KVCache` adapter — it relies on
// `cache.compact(...)` which TurboQuant doesn't implement.
impl crate::engine::ddtree::GemmaTreeTarget for Gemma4TargetAdapter<KVCache> {
    fn verify_tree_call(
        &mut self,
        tokens: &Array,
        position_ids: &Array,
        attention_mask: &Array,
    ) -> Result<Array, Exception> {
        self.verify_tree(tokens, position_ids, attention_mask)
    }
    fn compact_cache_call(
        &mut self,
        past_length: i32,
        keep_indices: &Array,
    ) -> Result<(), Exception> {
        self.compact_cache(past_length, keep_indices)
    }
    fn verify_tree_with_hidden_call(
        &mut self,
        tokens: &Array,
        position_ids: &Array,
        attention_mask: &Array,
    ) -> Result<(Array, Array), Exception> {
        self.verify_tree_with_hidden(tokens, position_ids, attention_mask)
    }
    fn lm_head_call(&mut self, hidden: &Array) -> Result<Array, Exception> {
        self.model.forward_via_hidden(hidden)
    }
}

/// Convenience constructor for the TurboQuant KV variant. Mirrors
/// `Gemma4TargetAdapter::<KVCache>::with_dflash` but allocates
/// `TurboQuantKVCache` per layer so the fused online-softmax SDPA path
/// can engage. Linear DFlash only (no DDTree).
impl Gemma4TargetAdapter<TurboQuantKVCache> {
    pub fn with_dflash_turboquant(
        model: Model,
        temp: f32,
        target_layer_ids: Vec<usize>,
    ) -> Self {
        <Gemma4TargetAdapter<TurboQuantKVCache>>::with_dflash(model, temp, target_layer_ids)
    }
}

// DDTree-specific impl (compact, verify_tree, etc.). KVCache only.
impl Gemma4TargetAdapter<KVCache> {
    /// Compact every layer's KV cache by keeping only the positions in
    /// `keep_indices` (offsets into the appended window starting at
    /// `past_length`). Drives interior-slot deletion for DDTree's tree
    /// accept path so we can avoid a full rollback + re-feed of the
    /// accepted prefix.
    pub fn compact_cache(
        &mut self,
        past_length: i32,
        keep_indices: &Array,
    ) -> Result<(), Exception> {
        // step counter follows offset: keep them in sync.
        let new_offset = past_length + keep_indices.shape()[0];
        for c in self.cache.iter_mut() {
            c.compact(past_length, keep_indices)?;
        }
        self.step = new_offset as usize;
        // Hidden segments: the just-completed verify pushed one segment.
        // For DDTree's path we don't currently rely on per-cycle hidden
        // capture (DFlash drafter feeds from the most recent committed
        // hidden which gets refreshed by the post-cycle single-token
        // bonus verify). Drop the latest segment so target_hidden_segments
        // doesn't accumulate stale tree captures.
        if self.target_hidden_segments.len() > self.verify_segment_count_pre {
            self.target_hidden_segments
                .truncate(self.verify_segment_count_pre);
            self.target_hidden_positions
                .truncate(self.verify_segment_count_pre);
            self.target_hidden_full_cache = None;
        }
        self.verify_inputs = None;
        Ok(())
    }

    /// Like `verify_tree` but also returns per-node post-norm hidden so
    /// the caller can derive the next-cycle root_pred from the
    /// last-accepted-node hidden without a separate LM-head call.
    pub fn verify_tree_with_hidden(
        &mut self,
        tokens: &Array,
        position_ids: &Array,
        attention_mask: &Array,
    ) -> Result<(Array, Array), Exception> {
        self.verify_inputs = Some(tokens.clone());
        self.verify_step = self.step;
        self.verify_segment_count_pre = self.target_hidden_segments.len();
        let (logits, hidden) = self.model.forward_tree_with_hidden(
            tokens,
            position_ids,
            attention_mask,
            &mut self.cache,
        )?;
        self.step += tokens.shape()[1] as usize;
        Ok((logits, hidden))
    }

    /// DDTree fused tree verify.
    ///
    /// Runs `Model::forward_tree` over a flat tree of tokens with explicit
    /// per-node `position_ids` and a bidirectional tree-visibility mask.
    /// Returns `[1, L, vocab]` per-node logits. The target's KV cache grows
    /// by L slots (one per tree node). Callers must call `rollback_kv` to
    /// drop the entire tree before installing the accepted prefix linearly.
    ///
    /// This skips DFlash's hidden-state capture (DDTree currently uses the
    /// DFlash drafter without target-hidden conditioning per-cycle — the
    /// drafter still works from the most-recently-committed hidden, which
    /// gets refreshed by the post-cycle linear forward). If a future
    /// integration needs per-tree-node hidden capture, extend forward_tree
    /// to also return captures.
    pub fn verify_tree(
        &mut self,
        tokens: &Array,
        position_ids: &Array,
        attention_mask: &Array,
    ) -> Result<Array, Exception> {
        // Treat the tree verify as a single verify "block" for rollback
        // accounting: record the input length so rollback_kv(n_keep) trims
        // (tree_len - n_keep) entries.
        self.verify_inputs = Some(tokens.clone());
        self.verify_step = self.step;
        self.verify_segment_count_pre = self.target_hidden_segments.len();
        let logits = self
            .model
            .forward_tree(tokens, position_ids, attention_mask, &mut self.cache)?;
        self.step += tokens.shape()[1] as usize;
        Ok(logits)
    }

}

impl<C: KeyValueCache + Default> TargetModel for Gemma4TargetAdapter<C> {
    fn prefill(&mut self, prompt: &Array) -> Result<Array, Exception> {
        // Reset cache, step, hidden accumulator, and verify state.
        self.cache = Self::fresh_cache(&self.model);
        self.step = 0;
        self.target_hidden_segments.clear();
        self.target_hidden_positions.clear();
        self.target_hidden_full_cache = None;
        self.verify_inputs = None;
        self.verify_step = 0;
        self.verify_segment_count_pre = 0;

        // Same env knob as the AR path: GEMMA4_prefill_chunk tunes the
        // per-chunk token count. Default 64 matches the AR path's
        // tested sweet spot (chunk=128 thrashes MoE expert gather).
        let prefill_chunk: i32 = std::env::var("GEMMA4_prefill_chunk")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n: &i32| n > 0)
            .unwrap_or(64);
        let seq_len = prompt.shape()[1];

        if seq_len <= prefill_chunk {
            // Short prompt — single forward (matches the prior path).
            let (logits, captures) = self.model.forward_last_logits_with_hidden_capture(
                prompt,
                &mut self.cache,
                &self.target_layer_ids,
            )?;
            self.append_target_hidden(captures)?;
            self.step = seq_len as usize;
            return Ok(logits);
        }

        // Chunked prefill — bounds peak Metal allocation (MoE expert
        // intermediates + attention activations grow with chunk size, not
        // total prompt length). Intermediate chunks pay the per-position
        // LM head cost wastefully, which is acceptable: at vocab=262k on
        // a 26B model the LM head is ~10% of the per-chunk compute and
        // the memory headroom recovered is the dominant win on long
        // prompts that would otherwise OOM.
        let mut last_logits: Option<Array> = None;
        let mut pos = 0;
        while pos < seq_len {
            let end = (pos + prefill_chunk).min(seq_len);
            let chunk = prompt.index((.., pos..end));
            let is_last = end == seq_len;
            let (logits, captures) = if is_last {
                self.model.forward_last_logits_with_hidden_capture(
                    &chunk,
                    &mut self.cache,
                    &self.target_layer_ids,
                )?
            } else {
                // For non-last chunks the logits are discarded — use the
                // `last_only` variant which collapses the LM head matmul
                // to a single position. On vocab=262k Gemma4 this drops
                // ~33 MB of unused per-position logits per chunk and
                // ~2-3x the prefill cost (the LM head dominates per-chunk
                // compute when full per-position logits are materialised
                // but never read).
                let (last_logits, captures) = self.model.forward_last_logits_with_hidden_capture(
                    &chunk,
                    &mut self.cache,
                    &self.target_layer_ids,
                )?;
                // Per-chunk eval frees expert-gather intermediates but
                // synchronizes the GPU. GEMMA4_SKIP_CHUNK_EVAL=1 omits
                // it entirely (faster on machines with enough wired
                // memory); GEMMA4_ASYNC_PREFILL=1 uses async_eval to
                // overlap the eval wait with the next chunk's launch.
                if std::env::var("GEMMA4_SKIP_CHUNK_EVAL").is_err() {
                    if std::env::var("GEMMA4_ASYNC_PREFILL").is_ok() {
                        mlx_rs::transforms::async_eval([&last_logits, &captures])?;
                    } else {
                        mlx_rs::transforms::eval([&last_logits, &captures])?;
                    }
                }
                (last_logits, captures)
            };
            self.append_target_hidden(captures)?;
            if is_last {
                last_logits = Some(logits);
            }
            pos = end;
        }
        self.step = seq_len as usize;
        Ok(last_logits.expect("chunked prefill produced no logits"))
    }

    fn verify(&mut self, drafted_tokens: &Array) -> Result<Array, Exception> {
        // Trim-based rollback: KV cache trims by offset, no clone needed.
        // Hidden segments are tracked by count — rollback truncates the
        // last (just-appended verify) segment in place.
        self.verify_inputs = Some(drafted_tokens.clone());
        self.verify_step = self.step;
        self.verify_segment_count_pre = self.target_hidden_segments.len();

        let (logits, captures) = self.model.forward_with_hidden_capture(
            drafted_tokens,
            &mut self.cache,
            &self.target_layer_ids,
        )?;
        self.append_target_hidden(captures)?;
        self.step += drafted_tokens.shape()[1] as usize;
        Ok(logits)
    }

    fn step_count(&self) -> usize {
        self.step
    }

    fn rollback_kv(&mut self, n_keep: usize) -> Result<(), Exception> {
        let verify_inputs = self
            .verify_inputs
            .as_ref()
            .ok_or_else(|| {
                Exception::custom("gemma4 target rollback requested before verify")
            })?
            .clone();
        let verify_len = verify_inputs.shape()[1] as usize;
        if n_keep > verify_len {
            return Err(Exception::custom(format!(
                "gemma4 target rollback requested {n_keep} kept tokens, but only {verify_len} verify tokens are available"
            )));
        }

        // Trim-based rollback: rewind each layer's KV cache by the number of
        // verify positions NOT kept (`verify_len - n_keep`). Restore the
        // hidden accumulator + step counter to pre-verify + n_keep state.
        // No re-forward needed — the kept KV entries are already correct
        // (they're the first n_keep positions of the just-completed verify
        // forward); the rejected ones become overwritable garbage that the
        // next verify cycle's `update_and_fetch` overwrites in place.
        let n_drop = (verify_len - n_keep) as i32;
        if n_drop > 0 {
            for cache in self.cache.iter_mut() {
                cache.trim_kv(n_drop)?;
            }
        }
        self.step = self.verify_step + n_keep;

        // Roll back the hidden-state segments: verify() pushed exactly one
        // segment (the verify captures). If 0 accepted, drop that segment
        // entirely; otherwise truncate it in place to n_keep rows.
        if self.target_hidden_segments.len() > self.verify_segment_count_pre {
            // Verify added the latest segment; truncate or drop it.
            self.truncate_latest_segment_to(n_keep)?;
        }

        self.verify_inputs = None;
        Ok(())
    }

    fn sample(&self, logits: &Array, temp: f32) -> Result<Array, Exception> {
        // gemma4-mlx does not expose a top-level `sample` helper, so we
        // implement temp==0 argmax and a basic temperature-scaled categorical
        // sampler inline. This matches the semantics of `qwen3_6_mlx::sample`
        // (see `qwen3.6-mlx/src/lib.rs:104-112`).
        if temp == 0.0 {
            argmax_axis!(logits, -1).map_err(Into::into)
        } else {
            let scaled = logits.multiply(mlx_rs::array!(1.0f32 / temp))?;
            mlx_rs::categorical!(scaled).map_err(Into::into)
        }
    }

    fn last_target_hidden(&self) -> Option<Array> {
        // Concat-on-demand from the Vec<Array> segments. Caller pattern is
        // 1-3 reads per cycle, each O(total_segments_size) — same asymptotic
        // as the old eager-concat path, but the append cost dropped from
        // O(total) to O(1), so the per-cycle work is roughly halved.
        match self.rebuild_full_hidden() {
            Ok(h) => h,
            Err(e) => {
                // Don't collapse a real concat failure into the misleading
                // "target_hidden not set before draft_block" downstream error.
                eprintln!("[dflash] rebuild_full_hidden failed: {e}");
                None
            }
        }
    }

    fn embed_token(&mut self, id: u32) -> Option<Array> {
        self.model.embed_tokens(&[id as i32]).ok()
    }
}
