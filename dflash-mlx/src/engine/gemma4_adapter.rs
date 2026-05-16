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

    // Accumulated captured hidden states across all committed forward passes,
    // shape `[B, total_T, K*H]`. Matches the Qwen36 adapter's accumulator.
    target_hidden_accumulated: Option<Array>,

    // Verify snapshot — captured at the start of `verify`, restored in
    // `rollback_kv`. Cache snapshot uses gemma4's `snapshot_cache` helper
    // (deep clone of the `Vec<KVCache>`).
    verify_snapshot: Option<Vec<KVCache>>,
    verify_inputs: Option<Array>,
    verify_step: usize,
    verify_hidden_snapshot: Option<Array>,
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
            target_hidden_accumulated: None,
            verify_snapshot: None,
            verify_inputs: None,
            verify_step: 0,
            verify_hidden_snapshot: None,
        }
    }

    fn fresh_cache(model: &Model) -> Vec<KVCache> {
        let num_slots = *model.model.kv_cache_map.iter().max().unwrap_or(&0) + 1;
        gemma4_mlx::init_cache::<KVCache>(num_slots)
    }

    fn append_target_hidden(&mut self, captures: Array) -> Result<(), Exception> {
        // `forward_with_hidden_capture` returns `[B, T, 0]` zeros when
        // `target_layer_ids` is empty; the concatenate-along-T below remains
        // valid either way.
        self.target_hidden_accumulated = Some(match self.target_hidden_accumulated.take() {
            Some(existing) => concatenate_axis(&[&existing, &captures], 1)?,
            None => captures,
        });
        Ok(())
    }
}

impl TargetModel for Gemma4TargetAdapter {
    fn prefill(&mut self, prompt: &Array) -> Result<Array, Exception> {
        // Reset cache, step, hidden accumulator, and verify snapshots.
        self.cache = Self::fresh_cache(&self.model);
        self.step = 0;
        self.target_hidden_accumulated = None;
        self.verify_snapshot = None;
        self.verify_inputs = None;
        self.verify_step = 0;
        self.verify_hidden_snapshot = None;

        let seq_len = prompt.shape()[1];

        // TODO: chunked prefill. The Qwen36 adapter chunks at 64 tokens to
        // reduce peak Metal allocation. `forward_with_hidden_capture` in
        // gemma4-mlx does not currently support chunked prefill with hidden
        // accumulation across chunks (would need to slice each chunk's
        // captured `[B, chunk_T, K*H]` and stitch). For v1 we forward the
        // whole prompt at once; for long prompts this may OOM on large
        // models. Acceptable for the bench-routing milestone.
        let (logits, captures) = self.model.forward_with_hidden_capture(
            prompt,
            &mut self.cache,
            &self.target_layer_ids,
        )?;
        self.append_target_hidden(captures)?;
        self.step = seq_len as usize;
        // forward_with_hidden_capture returns per-position logits [B, T, V] so
        // the verify path can index per drafted position. Prefill consumers
        // (DFlashSession::run_generate's initial sample) want just the last
        // position's logits for staged-token selection.
        Ok(logits.index((.., -1, ..)))
    }

    fn verify(&mut self, drafted_tokens: &Array) -> Result<Array, Exception> {
        // Trim-based rollback: instead of cloning the entire Vec<KVCache>
        // before each verify (expensive on a 26B MoE model), we just record
        // the verify span and snapshot the hidden accumulator. On rollback,
        // `KVCache::trim(n_drop)` rewinds the offset in O(1) per layer; no
        // re-replay of accepted tokens is needed because trim only drops
        // positions past `n_keep` — those past-n_keep KV entries become
        // overwritable garbage that the next verify cycle's
        // `update_and_fetch` will reuse in place.
        self.verify_snapshot = None; // trim path doesn't need a Vec snapshot
        self.verify_inputs = Some(drafted_tokens.clone());
        self.verify_step = self.step;
        self.verify_hidden_snapshot = self.target_hidden_accumulated.clone();

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

        // Truncate the hidden accumulator to keep only the accepted portion
        // of this verify pass. Pre-verify hidden + first n_keep verify
        // captures.
        let pre_verify_hidden = self.verify_hidden_snapshot.take();
        self.target_hidden_accumulated = match (pre_verify_hidden, n_keep) {
            (snap, 0) => snap,
            (snap, n) => {
                // Re-derive the kept-portion captures by slicing the live
                // accumulator: it currently holds pre-verify + all verify
                // captures (per `append_target_hidden` from verify()). Take
                // the first (pre_verify_len + n_keep) positions.
                let pre_len = snap
                    .as_ref()
                    .map(|h| h.shape()[1] as usize)
                    .unwrap_or(0);
                let total_keep = pre_len + n;
                self.target_hidden_accumulated.as_ref().map(|acc| {
                    acc.index((.., ..total_keep as i32, ..))
                })
            }
        };

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
        self.target_hidden_accumulated.clone()
    }

    fn embed_token(&mut self, id: u32) -> Option<Array> {
        self.model.embed_tokens(&[id as i32]).ok()
    }
}
