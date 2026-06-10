use mlx_rs::{error::Exception, ops::{broadcast_to, concatenate_axis, indexing::IndexOp}, Array, Dtype};

use crate::cache::ProjectedContextCache;
use crate::engine::spec_epoch::{DraftBlock, DraftModel};
use crate::model::DFlashDraftModel;

pub struct DFlashDraftAdapter {
    pub model: DFlashDraftModel,
    target_hidden: Option<Array>,
    staged_embedding: Option<Array>,
    mask_token_embedding: Array,
    lm_head_weight: Array,
    /// Per-layer ProjectedContextCache. Holds post-fc/hidden_norm/k_proj/k_norm/RoPE
    /// k_ctx and v_ctx for all already-committed target positions, so each
    /// draft cycle only has to project & RoPE the new committed delta
    /// (typically 4 tokens), not the entire accumulated context.
    caches: Vec<ProjectedContextCache>,
}

impl DFlashDraftAdapter {
    pub fn new(model: DFlashDraftModel, mask_token_embedding: Array, lm_head_weight: Array) -> Self {
        let num_layers = model.layers.len();
        Self {
            model,
            target_hidden: None,
            staged_embedding: None,
            mask_token_embedding,
            lm_head_weight,
            caches: (0..num_layers).map(|_| ProjectedContextCache::new()).collect(),
        }
    }

    pub fn set_target_hidden_impl(&mut self, h: Array) {
        self.target_hidden = Some(h);
    }

    pub fn target_hidden(&self) -> Option<&Array> {
        self.target_hidden.as_ref()
    }
}

impl DraftModel for DFlashDraftAdapter {
    fn prefill(&mut self, _prompt: &Array) -> Result<Array, Exception> {
        for cache in &mut self.caches {
            cache.reset();
        }
        let vocab = self.lm_head_weight.shape()[0];
        Array::zeros::<f32>(&[1, vocab])
    }

    fn draft_block(&mut self, _last_token: &Array, block_len: usize) -> Result<DraftBlock, Exception> {
        let target_hidden = self.target_hidden.as_ref().ok_or_else(|| {
            Exception::custom("DFlashDraftAdapter: target_hidden not set before draft_block")
        })?;

        let total_ctx_len = target_hidden.shape()[1] as usize;
        let cached_len = self
            .caches
            .first()
            .map(|c| c.offset())
            .unwrap_or(0);
        // Caches grow monotonically with each committed cycle. Target rollback
        // can shrink target_hidden — when that happens, restart cache from
        // scratch on this cycle so we never advertise more cached positions
        // than the target actually committed.
        if cached_len > total_ctx_len {
            for cache in &mut self.caches {
                cache.reset();
            }
        }
        let cached_len = self.caches.first().map(|c| c.offset()).unwrap_or(0);
        let raw_delta = if total_ctx_len > cached_len {
            target_hidden.index((
                ..,
                cached_len as i32..total_ctx_len as i32,
                ..,
            ))
        } else {
            // No new committed positions to cache — pass an empty delta.
            target_hidden.index((.., 0..0, ..))
        };

        let hidden_size = self.mask_token_embedding.shape()[2];

        // DFlash alignment: noise = [staged_emb, mask × (block_len-1)] = block_len tokens total.
        // The staged token's embedding sits at noise[0] (position ctx_offset); the draft model
        // predicts the next block_len-1 tokens from noise[1..] outputs.  Skipping draft_hidden[:,0,:]
        // gives block_len-1 predictions that align with posterior[0..block_len-2].
        // The DFlash draft model was trained with exactly block_len noise tokens, so the mask tail
        // must be block_len-1 (not block_len) to keep total noise length == block_len.
        let staged_emb = self.staged_embedding.take().ok_or_else(|| {
            Exception::custom("DFlashDraftAdapter: staged_embedding not set before draft_block; \
                call set_staged_embedding with the current staged token's embedding first")
        })?;
        let mask_tail = broadcast_to(
            &self.mask_token_embedding,
            &[1, block_len as i32 - 1, hidden_size],
        )?;
        let noise_emb = concatenate_axis(&[&staged_emb, &mask_tail], 1)?;

        let draft_hidden =
            self.model
                .forward_with_caches(&noise_emb, &raw_delta, &mut self.caches)?;

        let lm_head_t = self.lm_head_weight.t();

        // Skip output[0]: staged token at noise[0]; predictions are at output[1..block_len].
        let prediction_hidden = draft_hidden.index((.., 1.., ..));

        let logits = mlx_rs::ops::matmul(&prediction_hidden, &lm_head_t)?;
        let tokens = mlx_rs::argmax_axis!(&logits, -1)?.as_dtype(Dtype::Uint32)?;

        Ok(DraftBlock { tokens, logits })
    }

    fn rollback(&mut self, _n_accepted: usize) -> Result<(), Exception> {
        Ok(())
    }

    fn set_target_hidden(&mut self, h: Array) {
        self.target_hidden = Some(h);
    }

    fn set_staged_embedding(&mut self, emb: Array) {
        self.staged_embedding = Some(emb);
    }
}
