use mlx_rs::{error::Exception, ops::{broadcast_to, concatenate_axis, indexing::IndexOp}, Array, Dtype};

use crate::engine::spec_epoch::{DraftBlock, DraftModel};
use crate::model::DFlashDraftModel;

pub struct DFlashDraftAdapter {
    pub model: DFlashDraftModel,
    target_hidden: Option<Array>,
    staged_embedding: Option<Array>,
    mask_token_embedding: Array,
    lm_head_weight: Array,
}

impl DFlashDraftAdapter {
    pub fn new(model: DFlashDraftModel, mask_token_embedding: Array, lm_head_weight: Array) -> Self {
        Self {
            model,
            target_hidden: None,
            staged_embedding: None,
            mask_token_embedding,
            lm_head_weight,
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
        let vocab = self.lm_head_weight.shape()[0];
        Array::zeros::<f32>(&[1, vocab])
    }

    fn draft_block(&mut self, _last_token: &Array, block_len: usize) -> Result<DraftBlock, Exception> {
        let target_hidden = self.target_hidden.as_ref().ok_or_else(|| {
            Exception::custom("DFlashDraftAdapter: target_hidden not set before draft_block")
        })?;

        // ctx_offset = RoPE position of the first draft/noise token.
        let ctx_offset = target_hidden.shape()[1] as usize;
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

        let draft_hidden = self.model.forward(&noise_emb, target_hidden, ctx_offset)?;

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
