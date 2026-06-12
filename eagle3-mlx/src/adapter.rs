//! `DraftModel` adapter wiring [`Eagle3DraftModel`] into the
//! `dflash_mlx::DFlashSession` speculative cycle.
//!
//! ## Alignment invariants (see vLLM `llm_base_proposer.py`, llama.cpp
//! `common/speculative.cpp`)
//!
//! The draft processes one row per *committed hidden state*: row `i` pairs
//! the token at sequence position `i+1` with the target's fused aux hidden
//! from position `i`, at RoPE position `i`. Hidden states therefore lag
//! committed tokens by exactly one:
//!
//! ```text
//! committed_tokens.len() == hidden_len() + 1     (during generation)
//! ```
//!
//! - `committed_tokens` is maintained exclusively through
//!   `observe_committed` (plus the prompt at `prefill`), so it stays correct
//!   on CopySpec short-circuit cycles where `draft_block` never runs.
//! - `hidden_len()` is derived as `committed_tokens.len() - 1`, which equals
//!   the number of hidden rows the target has committed (prompt rows from
//!   prefill + the kept rows of each verify block).
//! - The draft KV cache persists across cycles. Speculative chain rows are
//!   trimmed in `rollback`; the next cycle's ingest pass re-processes the
//!   newly committed span with *target-derived* hiddens (the llama.cpp
//!   `seq_rm` + re-seed strategy).

use mlx_rs::{
    error::Exception,
    ops::{concatenate_axis, indexing::IndexOp},
    Array,
};
use mlx_rs_core::cache::{KVCache, KeyValueCache};

use dflash_mlx::{DraftBlock, DraftModel};

use crate::model::Eagle3DraftModel;

pub struct Eagle3DraftAdapter {
    pub model: Eagle3DraftModel,
    cache: KVCache,
    /// Prompt + every generated token, in order. Maintained via
    /// `prefill` + `observe_committed`.
    committed_tokens: Vec<u32>,
    /// True once the first generated token has been observed (before that,
    /// every committed token also has a hidden row, so len() == hidden_len).
    generation_started: bool,
    /// Absolute count of committed positions already ingested into the
    /// draft KV cache.
    ingested_abs: usize,
    /// Speculative chain rows currently in the draft KV cache (trimmed on
    /// rollback).
    chain_rows: usize,
    /// Latest committed-hidden window from the target: `[1, W, n_aux*H]`,
    /// covering absolute positions `[hidden_len - W, hidden_len)`.
    target_hidden: Option<Array>,
    window_gap_warned: bool,
}

impl Eagle3DraftAdapter {
    pub fn new(model: Eagle3DraftModel) -> Self {
        Self {
            model,
            cache: KVCache::default(),
            committed_tokens: Vec::new(),
            generation_started: false,
            ingested_abs: 0,
            chain_rows: 0,
            target_hidden: None,
            window_gap_warned: false,
        }
    }

    /// Number of hidden rows the target has committed so far.
    fn hidden_len(&self) -> usize {
        if self.generation_started {
            self.committed_tokens.len().saturating_sub(1)
        } else {
            self.committed_tokens.len()
        }
    }
}

impl DraftModel for Eagle3DraftAdapter {
    fn prefill(&mut self, prompt: &Array) -> Result<Array, Exception> {
        self.cache.reset();
        let flat = prompt.index((0, ..)).contiguous()?;
        mlx_rs::transforms::eval([&flat])?;
        self.committed_tokens = flat.as_slice::<u32>().to_vec();
        self.generation_started = false;
        self.ingested_abs = 0;
        self.chain_rows = 0;
        self.target_hidden = None;
        // The actual draft prefill is deferred to the first draft_block: the
        // session hands us the target's prompt hiddens (set_target_hidden)
        // only after this call.
        Array::zeros::<f32>(&[1, self.model.config.draft_vocab_size])
    }

    fn draft_block(
        &mut self,
        last_token: &Array,
        block_len: usize,
    ) -> Result<DraftBlock, Exception> {
        let window = self.target_hidden.as_ref().ok_or_else(|| {
            Exception::custom("Eagle3DraftAdapter: target_hidden not set before draft_block")
        })?;
        let last_tok = {
            let v = last_token.index((0, 0)).contiguous()?;
            mlx_rs::transforms::eval([&v])?;
            v.item::<u32>()
        };
        if self.committed_tokens.last() != Some(&last_tok) {
            return Err(Exception::custom(format!(
                "Eagle3DraftAdapter: draft_block last_token {last_tok} does not match the \
                 latest committed token {:?} — observe_committed bookkeeping is out of sync",
                self.committed_tokens.last()
            )));
        }
        if self.chain_rows > 0 {
            // Defensive: rollback should have trimmed these.
            self.cache.trim_kv(self.chain_rows as i32)?;
            self.chain_rows = 0;
        }

        let hidden_len = self.hidden_len();
        let window_len = window.shape()[1] as usize;
        let window_start = hidden_len.saturating_sub(window_len);

        // Ingest the newly committed span [start, hidden_len): fused target
        // hiddens paired with the next committed token at each position.
        let start = if self.ingested_abs < window_start {
            // The target's hidden accumulator dropped positions we never
            // ingested (segment cap on very long prompts). Skip the gap —
            // the draft attends over a suffix of the context, a documented
            // approximation that degrades acceptance, not correctness.
            if !self.window_gap_warned {
                eprintln!(
                    "[eagle3] hidden window dropped {} unconsumed positions; \
                     draft context is now a suffix (raise DFLASH_MAX_HIDDEN_SEGS to avoid)",
                    window_start - self.ingested_abs
                );
                self.window_gap_warned = true;
            }
            window_start
        } else {
            self.ingested_abs
        };
        let delta = hidden_len - start;
        if delta == 0 {
            return Err(Exception::custom(
                "Eagle3DraftAdapter: no new committed hidden rows to ingest — \
                 draft_block called twice without an intervening verify?",
            ));
        }

        let w0 = (start - window_start) as i32;
        let aux = window.index((.., w0..w0 + delta as i32, ..));
        let fused = self.model.fuse(&aux)?; // [1, delta, H]

        // Tokens at positions start+1 ..= hidden_len (the last is last_tok).
        let ids: Vec<u32> = self.committed_tokens[start + 1..=hidden_len].to_vec();
        let ids = Array::from_slice(&ids, &[1, delta as i32]);
        let embeds = self.model.embed(&ids)?; // [1, delta, H]

        let prenorm = self
            .model
            .forward(&embeds, &fused, &mut self.cache, start as i32)?;
        self.ingested_abs = hidden_len;

        // Chain: the ingest pass's last row drafts token 1; each further step
        // feeds (embed(mapped token), previous pre-norm output) at the next
        // position. Fully GPU-resident — the session's verify-input build is
        // the single host sync per cycle.
        let mut last_prenorm = prenorm.index((.., -1.., ..)); // [1, 1, H]
        let mut tokens: Vec<Array> = Vec::with_capacity(block_len);
        let mut chain_logits: Vec<Array> = Vec::with_capacity(block_len);

        let logits = self.model.logits(&last_prenorm)?; // [1, 1, Vd]
        tokens.push(self.model.sample_mapped(&logits)?);
        chain_logits.push(logits);

        for j in 1..block_len {
            let tok = tokens[j - 1].reshape(&[1, 1])?;
            let emb = self.model.embed(&tok)?;
            last_prenorm = self.model.forward(
                &emb,
                &last_prenorm,
                &mut self.cache,
                (hidden_len + j - 1) as i32,
            )?;
            self.chain_rows += 1;
            let logits = self.model.logits(&last_prenorm)?;
            tokens.push(self.model.sample_mapped(&logits)?);
            chain_logits.push(logits);
        }

        let token_refs: Vec<&Array> = tokens.iter().collect();
        let tokens = concatenate_axis(&token_refs, 0)?.reshape(&[1, block_len as i32])?;
        let logit_refs: Vec<&Array> = chain_logits.iter().collect();
        let logits = concatenate_axis(&logit_refs, 1)?; // [1, block_len, Vd]

        Ok(DraftBlock { tokens, logits })
    }

    fn rollback(&mut self, _n_accepted: usize) -> Result<(), Exception> {
        // Drop ALL speculative chain rows — including accepted ones. The
        // accepted span gets re-ingested next cycle from the target's true
        // aux hiddens (chain rows were keyed on the draft's own features).
        if self.chain_rows > 0 {
            self.cache.trim_kv(self.chain_rows as i32)?;
            self.chain_rows = 0;
        }
        Ok(())
    }

    fn set_target_hidden(&mut self, h: Array) {
        self.target_hidden = Some(h);
    }

    fn observe_committed(&mut self, tokens: &[u32]) {
        self.generation_started = true;
        self.committed_tokens.extend_from_slice(tokens);
    }
}
