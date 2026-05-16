use std::collections::{HashSet, VecDeque};

use mlx_rs::{
    argmax_axis, error::Exception, ops::indexing::IndexOp, transforms::eval, Array, Dtype,
};

use super::{
    acceptance::match_acceptance_length,
    config::{AdaptiveBlockPolicy, SpeculativeCycleConfig},
    copyspec::CopySpecIndex,
};

pub trait TargetModel {
    /// Prefill the model on `prompt` tokens [B, T], populate caches, return last-token logits [B, vocab].
    fn prefill(&mut self, prompt: &Array) -> Result<Array, Exception>;

    /// Verify aligned decode inputs [B, block_len], return per-position logits [B, block_len, vocab].
    fn verify(&mut self, drafted_tokens: &Array) -> Result<Array, Exception>;

    /// Get the current decode step count (for the full model, this tracks how many tokens processed).
    fn step_count(&self) -> usize;

    /// Reset/rollback caches to the kept verified prefix length.
    fn rollback_kv(&mut self, n_keep: usize) -> Result<(), Exception>;

    /// Sample token from logits [B, vocab] → [B] u32.
    fn sample(&self, logits: &Array, temp: f32) -> Result<Array, Exception>;

    fn last_target_hidden(&self) -> Option<Array> {
        None
    }

    /// Look up the embedding for a single token id, returning shape [1, 1, hidden_size].
    /// Default returns None; implement for target models that support DFlash staged embedding.
    fn embed_token(&mut self, _id: u32) -> Option<Array> {
        None
    }
}

pub trait DraftModel {
    /// Prefill the draft model on `prompt` tokens, populate caches, return last-token logits.
    fn prefill(&mut self, prompt: &Array) -> Result<Array, Exception>;

    /// Generate draft tokens starting from `last_token` [B, 1].
    /// `block_len` is the requested noise block size (e.g. 16 for DFlash).
    /// The returned `DraftBlock.tokens` may have fewer than `block_len` tokens
    /// (DFlash returns `block_len - 1`); callers must use `tokens.shape()[0]` as
    /// the authoritative drafted count rather than assuming it equals `block_len`.
    fn draft_block(
        &mut self,
        last_token: &Array,
        block_len: usize,
    ) -> Result<DraftBlock, Exception>;

    /// Roll back state to post-prefill, then replay only `n_accepted` draft steps.
    fn rollback(&mut self, n_accepted: usize) -> Result<(), Exception>;

    fn set_target_hidden(&mut self, _h: Array) {}

    /// Provide the staged token's embedding (shape [1, 1, hidden]) to the draft model so it
    /// can place the actual staged token at noise position 0 instead of the mask token.
    /// This is the DFlash alignment fix: Python's `draft_greedy` uses the staged token
    /// embedding at noise[0] and skips output[0], making draft predictions align with
    /// posterior targets.  Default is a no-op for adapters that don't use it.
    fn set_staged_embedding(&mut self, _emb: Array) {}
}

#[derive(Debug, Clone)]
pub struct DraftBlock {
    pub tokens: Array,
    pub logits: Array,
}

/// A running DFlash speculative decoding session.
#[derive(Debug)]
pub struct DFlashSession<Target, Draft> {
    target: Target,
    draft: Draft,
    config: SpeculativeCycleConfig,
    adaptive_policy: AdaptiveBlockPolicy,
    copyspec: Option<CopySpecIndex>,
    metrics: SessionMetrics,
}

#[derive(Debug, Clone, Default)]
pub struct SessionMetrics {
    pub total_tokens: usize,
    pub total_cycles: usize,
    pub total_accepted: usize,
    pub total_drafted: usize,
    pub acceptance_ratio: f32,
    pub avg_block_len: f32,
}

/// Emitted event from run_generate iterator.
#[derive(Debug, Clone)]
pub enum GenerateEvent {
    Token { token_id: u32 },
    Summary(SessionMetrics),
}

impl<Target, Draft> DFlashSession<Target, Draft> {
    pub fn new(target: Target, draft: Draft, config: SpeculativeCycleConfig) -> Self {
        let adaptive_policy = AdaptiveBlockPolicy::new(config.clone());
        Self {
            target,
            draft,
            config,
            adaptive_policy,
            copyspec: None,
            metrics: SessionMetrics::default(),
        }
    }

    pub fn config(&self) -> &SpeculativeCycleConfig {
        &self.config
    }

    pub fn adaptive_policy(&self) -> &AdaptiveBlockPolicy {
        &self.adaptive_policy
    }

    pub fn metrics(&self) -> &SessionMetrics {
        &self.metrics
    }

    pub fn target(&self) -> &Target {
        &self.target
    }

    pub fn draft(&self) -> &Draft {
        &self.draft
    }

    pub fn set_copyspec(&mut self, copyspec: CopySpecIndex) {
        self.copyspec = Some(copyspec);
    }
}

impl<Target: TargetModel, Draft: DraftModel> DFlashSession<Target, Draft> {
    pub fn run_generate(
        &mut self,
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
        temp: f32,
        eos_token_ids: &[u32],
    ) -> impl Iterator<Item = Result<u32, Exception>> + '_ {
        self.metrics = SessionMetrics::default();
        self.adaptive_policy = AdaptiveBlockPolicy::new(self.config.clone());

        let eos_tokens: HashSet<u32> = eos_token_ids.iter().copied().collect();
        let mut prompt_tokens = Some(prompt_tokens);
        let mut initialized = false;
        let mut finished = max_tokens == 0;
        let mut emitted = 0usize;
        let mut last_emitted: Option<u32> = None;
        let mut pending = VecDeque::<Result<u32, Exception>>::new();

        std::iter::from_fn(move || loop {
            if let Some(item) = pending.pop_front() {
                match &item {
                    Ok(token) => {
                        emitted += 1;
                        self.metrics.total_tokens = emitted;
                        last_emitted = Some(*token);
                        if emitted >= max_tokens || eos_tokens.contains(token) {
                            finished = true;
                            pending.clear();
                        }
                    }
                    Err(_) => {
                        finished = true;
                        pending.clear();
                    }
                }
                return Some(item);
            }

            if finished {
                return None;
            }

            if !initialized {
                let prompt = prompt_tokens.take().unwrap_or_default();
                if prompt.is_empty() {
                    finished = true;
                    return Some(Err(Exception::custom("prompt_tokens must not be empty")));
                }

                let prompt_array = Array::from_slice(&prompt, &[1, prompt.len() as i32]);
                let target_logits = match self.target.prefill(&prompt_array) {
                    Ok(logits) => logits,
                    Err(err) => {
                        finished = true;
                        return Some(Err(err));
                    }
                };
                if let Err(err) = self.draft.prefill(&prompt_array) {
                    finished = true;
                    return Some(Err(err));
                }
                if let Some(h) = self.target.last_target_hidden() {
                    self.draft.set_target_hidden(h);
                }

                match self
                    .target
                    .sample(&target_logits, temp)
                    .and_then(|token| scalar_token(&token))
                {
                    Ok(token) => {
                        pending.push_back(Ok(token));
                        initialized = true;
                        continue;
                    }
                    Err(err) => {
                        finished = true;
                        return Some(Err(err));
                    }
                }
            }

            let last_token = match last_emitted {
                Some(token) => token,
                None => {
                    finished = true;
                    return Some(Err(Exception::custom(
                        "generation state is missing the last emitted token",
                    )));
                }
            };

            let remaining = max_tokens.saturating_sub(emitted);
            if remaining == 0 {
                finished = true;
                return None;
            }

            if remaining <= 2 {
                // Too few tokens left to run a DFlash draft cycle (need at least block_len >= 2
                // to produce >= 1 draft token after the block_len-1 mask fix).  Fall back to a
                // single target step.
                let input = Array::from_slice(&[last_token], &[1, 1]);
                let logits = match self.target.verify(&input) {
                    Ok(logits) => logits,
                    Err(err) => {
                        finished = true;
                        return Some(Err(err));
                    }
                };
                let next_logits = logits.index((.., 0, ..));
                match self
                    .target
                    .sample(&next_logits, temp)
                    .and_then(|token| scalar_token(&token))
                {
                    Ok(token) => {
                        pending.push_back(Ok(token));
                        continue;
                    }
                    Err(err) => {
                        finished = true;
                        return Some(Err(err));
                    }
                }
            }

            let block_len = self.adaptive_policy.current_block_len().min(remaining - 1);
            let last_token_array = Array::from_slice(&[last_token], &[1, 1]);

            // Provide the staged token's actual embedding as noise[0] so the draft model's
            // position predictions align with the target's posterior (the DFlash alignment fix).
            if let Some(staged_emb) = self.target.embed_token(last_token) {
                self.draft.set_staged_embedding(staged_emb);
            } else {
            }

            let drafted = match self.draft.draft_block(&last_token_array, block_len) {
                Ok(block) => block,
                Err(err) => {
                    finished = true;
                    return Some(Err(err));
                }
            };

            let drafted_tokens = drafted.tokens.index((0, ..));
            // drafted_count = block_len - 1: DFlash returns one fewer token than the noise
            // block length because noise[0] is the staged token and its output is skipped.
            let drafted_count = drafted_tokens.shape()[0] as usize;
            let verify_inputs = match build_verify_inputs(last_token, &drafted_tokens) {
                Ok(inputs) => inputs,
                Err(err) => {
                    finished = true;
                    return Some(Err(err));
                }
            };
            let verify_logits = match self.target.verify(&verify_inputs) {
                Ok(logits) => logits,
                Err(err) => {
                    finished = true;
                    return Some(Err(err));
                }
            };
            let posterior =
                match argmax_axis!(&verify_logits, -1).and_then(|a| a.as_dtype(Dtype::Uint32)) {
                    Ok(tokens) => tokens,
                    Err(err) => {
                        finished = true;
                        return Some(Err(err.into()));
                    }
                };
            let posterior_tokens = posterior.index((0, ..drafted_count as i32));

            let n_accepted = match match_acceptance_length(&drafted_tokens, &posterior_tokens) {
                Ok(n) => n,
                Err(err) => {
                    finished = true;
                    return Some(Err(err));
                }
            };

            let drafted_vec = match array_to_vec_u32(&drafted_tokens) {
                Ok(tokens) => tokens,
                Err(err) => {
                    finished = true;
                    return Some(Err(err));
                }
            };
            for token in drafted_vec.iter().take(n_accepted) {
                pending.push_back(Ok(*token));
            }

            let target_token = if n_accepted < drafted_count {
                let correction_logits = verify_logits.index((.., n_accepted as i32, ..));
                let token = match self
                    .target
                    .sample(&correction_logits, temp)
                    .and_then(|token| scalar_token(&token))
                {
                    Ok(token) => token,
                    Err(err) => {
                        finished = true;
                        return Some(Err(err));
                    }
                };
                if let Err(err) = self.draft.rollback(n_accepted) {
                    finished = true;
                    return Some(Err(err));
                }
                if let Err(err) = self.target.rollback_kv(1 + n_accepted) {
                    finished = true;
                    return Some(Err(err));
                }
                if let Some(h) = self.target.last_target_hidden() {
                    self.draft.set_target_hidden(h);
                }
                token
            } else {
                // Full acceptance: correction logit is at index drafted_count (the position
                // after the last drafted token in verify_logits).
                let correction_logits = verify_logits.index((.., drafted_count as i32, ..));
                if let Err(err) = self.draft.rollback(n_accepted) {
                    finished = true;
                    return Some(Err(err));
                }
                if let Some(h) = self.target.last_target_hidden() {
                    self.draft.set_target_hidden(h);
                }
                match self
                    .target
                    .sample(&correction_logits, temp)
                    .and_then(|token| scalar_token(&token))
                {
                    Ok(token) => token,
                    Err(err) => {
                        finished = true;
                        return Some(Err(err));
                    }
                }
            };
            pending.push_back(Ok(target_token));

            self.metrics.total_cycles += 1;
            self.metrics.total_accepted += n_accepted;
            self.metrics.total_drafted += drafted_count;
            self.metrics.acceptance_ratio = if self.metrics.total_drafted == 0 {
                0.0
            } else {
                self.metrics.total_accepted as f32 / self.metrics.total_drafted as f32
            };
            self.metrics.avg_block_len = if self.metrics.total_cycles == 0 {
                0.0
            } else {
                self.metrics.total_drafted as f32 / self.metrics.total_cycles as f32
            };
            self.adaptive_policy.update(
                n_accepted as f32 / drafted_count as f32,
                (n_accepted + 1) as f32,
            );
        })
    }
}

fn build_verify_inputs(last_token: u32, drafted_tokens: &Array) -> Result<Array, Exception> {
    if drafted_tokens.shape().len() != 1 {
        return Err(Exception::custom(format!(
            "build_verify_inputs expects a 1D drafted token array, got {:?}",
            drafted_tokens.shape()
        )));
    }

    let drafted = array_to_vec_u32(drafted_tokens)?;
    if drafted.is_empty() {
        return Err(Exception::custom(
            "build_verify_inputs requires at least one drafted token",
        ));
    }

    // Include all drafted tokens: [last_token, draft[0], ..., draft[block_len-1]]
    // This gives block_len+1 verify logits so index block_len yields the correction
    // token for the full-acceptance case without an extra forward pass.
    let mut verify_inputs = Vec::with_capacity(drafted.len() + 1);
    verify_inputs.push(last_token);
    verify_inputs.extend(drafted.iter().copied());
    Ok(Array::from_slice(
        &verify_inputs,
        &[1, verify_inputs.len() as i32],
    ))
}

fn array_to_vec_u32(array: &Array) -> Result<Vec<u32>, Exception> {
    let contiguous = array.contiguous()?;
    eval([&contiguous])?;
    Ok(contiguous.as_slice::<u32>().to_vec())
}

fn scalar_token(array: &Array) -> Result<u32, Exception> {
    let contiguous = array.contiguous()?;
    eval([&contiguous])?;
    Ok(contiguous.item::<u32>())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    #[derive(Debug)]
    struct ScriptedTarget {
        vocab_size: i32,
        prefill_token: u32,
        verify_sequences: VecDeque<Vec<u32>>,
        rollback_history: Vec<usize>,
        step: usize,
    }

    impl ScriptedTarget {
        fn new(prefill_token: u32, verify_sequences: Vec<Vec<u32>>) -> Self {
            Self {
                vocab_size: 64,
                prefill_token,
                verify_sequences: verify_sequences.into(),
                rollback_history: Vec::new(),
                step: 0,
            }
        }
    }

    impl TargetModel for ScriptedTarget {
        fn prefill(&mut self, prompt: &Array) -> Result<Array, Exception> {
            self.step = prompt.shape()[1] as usize;
            Ok(logits_for_token(self.prefill_token, self.vocab_size))
        }

        fn verify(&mut self, drafted_tokens: &Array) -> Result<Array, Exception> {
            let expected_len = drafted_tokens.shape()[1] as usize;
            let sequence = self
                .verify_sequences
                .pop_front()
                .ok_or_else(|| Exception::custom("missing scripted verify sequence"))?;
            if sequence.len() != expected_len {
                return Err(Exception::custom(format!(
                    "scripted verify length mismatch: expected {expected_len}, got {}",
                    sequence.len()
                )));
            }
            self.step += expected_len;
            Ok(logits_for_sequence(&sequence, self.vocab_size))
        }

        fn step_count(&self) -> usize {
            self.step
        }

        fn rollback_kv(&mut self, n_keep: usize) -> Result<(), Exception> {
            self.rollback_history.push(n_keep);
            Ok(())
        }

        fn sample(&self, logits: &Array, _temp: f32) -> Result<Array, Exception> {
            argmax_axis!(logits, -1).map_err(Into::into)
        }
    }

    #[derive(Debug)]
    struct ScriptedDraft {
        vocab_size: i32,
        blocks: VecDeque<Vec<u32>>,
        rollback_history: Vec<usize>,
    }

    impl ScriptedDraft {
        fn new(blocks: Vec<Vec<u32>>) -> Self {
            Self {
                vocab_size: 64,
                blocks: blocks.into(),
                rollback_history: Vec::new(),
            }
        }
    }

    impl DraftModel for ScriptedDraft {
        fn prefill(&mut self, _prompt: &Array) -> Result<Array, Exception> {
            Array::zeros::<f32>(&[1, self.vocab_size])
        }

        fn draft_block(
            &mut self,
            _last_token: &Array,
            block_len: usize,
        ) -> Result<DraftBlock, Exception> {
            let block = self
                .blocks
                .pop_front()
                .ok_or_else(|| Exception::custom("missing scripted draft block"))?;
            if block.len() != block_len {
                return Err(Exception::custom(format!(
                    "scripted draft block length mismatch: expected {block_len}, got {}",
                    block.len()
                )));
            }
            Ok(DraftBlock {
                tokens: Array::from_slice(&block, &[1, block_len as i32]),
                logits: Array::zeros::<f32>(&[1, block_len as i32, self.vocab_size])?,
            })
        }

        fn rollback(&mut self, n_accepted: usize) -> Result<(), Exception> {
            self.rollback_history.push(n_accepted);
            Ok(())
        }
    }

    fn logits_for_token(token: u32, vocab_size: i32) -> Array {
        let mut data = vec![0.0f32; vocab_size as usize];
        data[token as usize] = 1.0;
        Array::from_slice(&data, &[1, vocab_size])
    }

    fn logits_for_sequence(tokens: &[u32], vocab_size: i32) -> Array {
        let mut data = vec![0.0f32; tokens.len() * vocab_size as usize];
        for (index, token) in tokens.iter().enumerate() {
            data[index * vocab_size as usize + *token as usize] = 1.0;
        }
        Array::from_slice(&data, &[1, tokens.len() as i32, vocab_size])
    }

    #[test]
    fn build_verify_inputs_shifts_previous_token() {
        let _guard = crate::mlx_test_guard();
        let drafted = Array::from_slice(&[11u32, 12, 13, 14], &[4]);
        let inputs = build_verify_inputs(10, &drafted).unwrap();
        // Now includes all drafted tokens: [last_token, draft[0..n]]
        assert_eq!(inputs.shape(), &[1, 5]);
        assert_eq!(
            array_to_vec_u32(&inputs.index((0, ..))).unwrap(),
            vec![10, 11, 12, 13, 14]
        );
    }

    #[test]
    fn run_generate_rolls_back_on_mismatch() {
        let _guard = crate::mlx_test_guard();
        // verify_sequences now need 3-element entries since verify_inputs includes all drafted tokens
        let target = ScriptedTarget::new(10, vec![vec![11, 21, 55], vec![30]]);
        let draft = ScriptedDraft::new(vec![vec![11, 12]]);
        let config = SpeculativeCycleConfig {
            block_len: 2,
            min_block_tokens: 2,
            ..Default::default()
        };
        let mut session = DFlashSession::new(target, draft, config);

        let tokens: Vec<u32> = session
            .run_generate(vec![1, 2, 3], 4, 0.0, &[])
            .map(|item| item.unwrap())
            .collect();

        assert_eq!(tokens, vec![10, 11, 21, 30]);
        assert_eq!(session.metrics().total_cycles, 1);
        assert_eq!(session.metrics().total_accepted, 1);
        assert_eq!(session.metrics().total_drafted, 2);
        assert_eq!(session.target().rollback_history, vec![2]);
        assert_eq!(session.draft().rollback_history, vec![1]);
    }

    #[test]
    fn run_generate_emits_extra_target_token_when_all_drafts_match() {
        let _guard = crate::mlx_test_guard();
        // verify_sequences: first entry has block_len+1 elements (correction at index block_len);
        // single-step fallback at remaining==1 still uses 1-element sequences.
        let target = ScriptedTarget::new(10, vec![vec![11, 12, 13], vec![14]]);
        let draft = ScriptedDraft::new(vec![vec![11, 12], vec![14]]);
        let config = SpeculativeCycleConfig {
            block_len: 2,
            min_block_tokens: 2,
            ..Default::default()
        };
        let mut session = DFlashSession::new(target, draft, config);

        let tokens: Vec<u32> = session
            .run_generate(vec![1, 2, 3], 5, 0.0, &[])
            .map(|item| item.unwrap())
            .collect();

        assert_eq!(tokens, vec![10, 11, 12, 13, 14]);
        assert_eq!(session.metrics().total_cycles, 1);
        assert_eq!(session.metrics().total_accepted, 2);
        assert_eq!(session.metrics().total_drafted, 2);
        assert!(session.target().rollback_history.is_empty());
        assert_eq!(session.draft().rollback_history, vec![2]);
    }
}
