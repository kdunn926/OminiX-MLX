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
//! - `Model::forward_with_hidden_capture` currently narrows to last-position
//!   logits before the LM head (see `gemma4-mlx/src/model.rs:1281-1286`),
//!   which differs from `Qwen36TargetAdapter`'s full-sequence logits. For
//!   verify, spec_epoch indexes `verify_logits[.., t, ..]` across the block,
//!   so this shape mismatch is a known TODO: acceptance-ratio parity against
//!   the Python `target_gemma4.py` is the 3-6 week deferred investigation
//!   called out in the task scope. Compilation and routing are unaffected.

use mlx_rs::{error::Exception, ops::concatenate_axis, Array};

use gemma4_mlx::{KVCache, Model};
use mlx_rs::{argmax_axis, ops::indexing::IndexOp};

use crate::engine::spec_epoch::TargetModel;

/// DFlash target adapter for the 26B-A4B Gemma4 variant.
///
/// Holds the model, its KV cache, target capture layer ids, and the
/// verify-snapshot fields needed to roll back on partial acceptance.
pub struct Gemma4TargetAdapter {
    model: Model,
    cache: Vec<KVCache>,
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

impl Gemma4TargetAdapter {
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

}

impl crate::engine::ddtree::GemmaTreeTarget for Gemma4TargetAdapter {
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

impl Gemma4TargetAdapter {
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

    fn fresh_cache(model: &Model) -> Vec<KVCache> {
        let num_slots = *model.model.kv_cache_map.iter().max().unwrap_or(&0) + 1;
        gemma4_mlx::init_cache::<KVCache>(num_slots)
    }

    fn append_target_hidden(&mut self, captures: Array) -> Result<(), Exception> {
        let seg_len = captures.shape()[1] as usize;
        let prev_total = self.target_hidden_positions.last().copied().unwrap_or(0);
        self.target_hidden_segments.push(captures);
        self.target_hidden_positions.push(prev_total + seg_len);
        // Invalidate the fully-concat cache.
        self.target_hidden_full_cache = None;
        Ok(())
    }

    fn drop_segments_after(&mut self, keep_segments: usize) {
        self.target_hidden_segments.truncate(keep_segments);
        self.target_hidden_positions.truncate(keep_segments);
        self.target_hidden_full_cache = None;
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

impl TargetModel for Gemma4TargetAdapter {
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

        const PREFILL_CHUNK: i32 = 64;
        let seq_len = prompt.shape()[1];

        if seq_len <= PREFILL_CHUNK {
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
            let end = (pos + PREFILL_CHUNK).min(seq_len);
            let chunk = prompt.index((.., pos..end));
            let is_last = end == seq_len;
            let (logits, captures) = if is_last {
                self.model.forward_last_logits_with_hidden_capture(
                    &chunk,
                    &mut self.cache,
                    &self.target_layer_ids,
                )?
            } else {
                let (per_pos_logits, captures) = self.model.forward_with_hidden_capture(
                    &chunk,
                    &mut self.cache,
                    &self.target_layer_ids,
                )?;
                // Force eval to free per-chunk intermediates before the
                // next chunk starts.
                mlx_rs::transforms::eval([&per_pos_logits, &captures])?;
                (per_pos_logits, captures)
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
                cache.trim(n_drop);
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
        self.rebuild_full_hidden().ok().flatten()
    }

    fn embed_token(&mut self, id: u32) -> Option<Array> {
        self.model.embed_tokens(&[id as i32]).ok()
    }
}
