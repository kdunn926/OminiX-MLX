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

    /// Notify the draft of the tokens committed this cycle, in order: the
    /// `n_accepted` accepted drafted tokens followed by the target's
    /// correction/bonus token. Fires on every cycle of `run_generate`,
    /// including CopySpec short-circuit cycles where `draft_block` was never
    /// called — drafters that maintain token-aligned state (e.g. EAGLE-style
    /// feature caches) rely on this rather than on their own drafted tokens.
    /// Default is a no-op for stateless drafters.
    fn observe_committed(&mut self, _tokens: &[u32]) {}
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
    /// Number of cycles whose draft block came from the CopySpec
    /// prompt-tail index instead of the draft model.
    pub copyspec_hits: usize,
    /// Total tokens proposed via CopySpec (sum over `copyspec_hits` cycles).
    pub copyspec_tokens: usize,
    /// Per-mode cycle tallies for the v0.1.7 adaptive verify state
    /// machine. Lets you see whether a workload sat in Reduced the
    /// whole time or recovered to Large via probes.
    pub cycles_large: usize,
    pub cycles_reduced: usize,
    pub cycles_probe: usize,
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

                // Build the CopySpec N-gram index over the prompt if one
                // hasn't been provided explicitly. Free drafts for prompts
                // that echo themselves (code, math reasoning, structured
                // outputs, repetition).
                if self.copyspec.is_none() {
                    self.copyspec = Some(CopySpecIndex::new(&prompt));
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
                        self.draft.observe_committed(&[token]);
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
                        // The single-token verify above appended one hidden
                        // row; keep token-aligned drafters in sync with it.
                        self.draft.observe_committed(&[token]);
                        pending.push_back(Ok(token));
                        continue;
                    }
                    Err(err) => {
                        finished = true;
                        return Some(Err(err));
                    }
                }
            }

            // Wall-clock start of this cycle, so v0.1.7's adaptive-verify
            // update() can compare real tok/s across modes.
            let cycle_started_at = std::time::Instant::now();
            match self.adaptive_policy.mode() {
                crate::engine::config::BlockMode::Large => self.metrics.cycles_large += 1,
                crate::engine::config::BlockMode::Reduced => self.metrics.cycles_reduced += 1,
                crate::engine::config::BlockMode::Probe => self.metrics.cycles_probe += 1,
            }
            let block_len = self.adaptive_policy.current_block_len().min(remaining - 1);
            let last_token_array = Array::from_slice(&[last_token], &[1, 1]);

            // Provide the staged token's actual embedding as noise[0] so the draft model's
            // position predictions align with the target's posterior (the DFlash alignment fix).
            if let Some(staged_emb) = self.target.embed_token(last_token) {
                self.draft.set_staged_embedding(staged_emb);
            }

            // CopySpec short-circuit: if the prompt-tail index has a hit
            // for (committed tail + last_token), use those prompt tokens as
            // the draft block and skip the draft model forward entirely.
            // The target verifier still gates every proposed token.
            let copyspec_hit = self
                .copyspec
                .as_ref()
                .and_then(|c| c.draft_after(last_token, block_len - 1, None));
            let drafted = match copyspec_hit {
                Some(tokens) => {
                    self.metrics.copyspec_hits += 1;
                    self.metrics.copyspec_tokens += tokens.len();
                    let n = tokens.len() as i32;
                    match Array::zeros::<f32>(&[1, n, 1]) {
                        Ok(zeros) => DraftBlock {
                            tokens: Array::from_slice(&tokens, &[1, n]),
                            // logits unused downstream (only `tokens`).
                            logits: zeros,
                        },
                        Err(err) => {
                            finished = true;
                            return Some(Err(err));
                        }
                    }
                }
                None => match self.draft.draft_block(&last_token_array, block_len) {
                    Ok(block) => block,
                    Err(err) => {
                        finished = true;
                        return Some(Err(err));
                    }
                },
            };

            let drafted_tokens = drafted.tokens.index((0, ..));
            // drafted_count = block_len - 1: DFlash returns one fewer token than the noise
            // block length because noise[0] is the staged token and its output is skipped.
            let drafted_count = drafted_tokens.shape()[0] as usize;
            let (verify_inputs, drafted_vec) =
                match build_verify_inputs(last_token, &drafted_tokens) {
                    Ok(pair) => pair,
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

            // Acceptance. At T=0 use greedy longest-prefix matching (accepted
            // tokens == target argmax). At T>0 use distribution-preserving
            // speculative acceptance: accept drafted[i] with probability
            // p_i(drafted[i]) and sample the correction from the residual, so
            // the committed tokens follow the target distribution rather than
            // the argmax (greedy accept + tempered correction was biased).
            let (n_accepted, target_token) = if temp <= 0.0 {
                let posterior = match argmax_axis!(&verify_logits, -1)
                    .and_then(|a| a.as_dtype(Dtype::Uint32))
                {
                    Ok(tokens) => tokens,
                    Err(err) => {
                        finished = true;
                        return Some(Err(err.into()));
                    }
                };
                let posterior_tokens = posterior.index((0, ..drafted_count as i32));
                let n = match match_acceptance_length(&drafted_tokens, &posterior_tokens) {
                    Ok(n) => n,
                    Err(err) => {
                        finished = true;
                        return Some(Err(err));
                    }
                };
                // Correction at the reject slot, or the bonus slot on full accept.
                let corr_idx = if n < drafted_count { n } else { drafted_count } as i32;
                let corr_logits = verify_logits.index((.., corr_idx, ..));
                let token = match self
                    .target
                    .sample(&corr_logits, temp)
                    .and_then(|t| scalar_token(&t))
                {
                    Ok(token) => token,
                    Err(err) => {
                        finished = true;
                        return Some(Err(err));
                    }
                };
                (n, token)
            } else {
                let mut rng = crate::engine::acceptance::default_rng;
                match crate::engine::acceptance::speculative_accept(
                    &verify_logits,
                    &drafted_vec,
                    temp,
                    &mut rng,
                ) {
                    Ok(res) => (res.n_accepted, res.correction),
                    Err(err) => {
                        finished = true;
                        return Some(Err(err));
                    }
                }
            };

            for token in drafted_vec.iter().take(n_accepted) {
                pending.push_back(Ok(*token));
            }

            // Roll back the rejected draft suffix. The target's KV is trimmed
            // only on a partial accept (full acceptance keeps all verify
            // positions); the draft context is rolled back either way, then the
            // post-rollback target hidden is handed to the drafter.
            if let Err(err) = self.draft.rollback(n_accepted) {
                finished = true;
                return Some(Err(err));
            }
            if n_accepted < drafted_count {
                if let Err(err) = self.target.rollback_kv(1 + n_accepted) {
                    finished = true;
                    return Some(Err(err));
                }
            }
            if let Some(h) = self.target.last_target_hidden() {
                self.draft.set_target_hidden(h);
            }
            pending.push_back(Ok(target_token));

            // Everything that got committed this cycle: the n_accepted
            // drafted tokens that survived verification + the target's
            // correction/stage token. Drives both the CopySpec index and
            // the drafter's committed-token bookkeeping.
            let mut committed: Vec<u32> =
                drafted_vec.iter().take(n_accepted).copied().collect();
            committed.push(target_token);
            self.draft.observe_committed(&committed);
            if let Some(copyspec) = self.copyspec.as_mut() {
                copyspec.append_committed(&committed);
            }

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
            // v0.1.7 adaptive verify needs real per-cycle wall time. This cycle's
            // draft + verify GPU work is already forced before we get here:
            // `array_to_vec_u32` (drafted tokens) and either `scalar_token` (greedy
            // correction) or `speculative_accept`'s host-side softmax read all call
            // `eval`/`as_slice`, which materialize the verify logits. So
            // `cycle_wall_s` reflects real GPU work, not just MLX graph build.
            let cycle_wall_s = cycle_started_at.elapsed().as_secs_f32();
            self.adaptive_policy.update(
                n_accepted as f32 / drafted_count as f32,
                (n_accepted + 1) as f32,
                cycle_wall_s,
            );
        })
    }
}

// ============================================================================
// DDTree (tree-shaped speculative decoding) integration
// ============================================================================

impl<Target, Draft> DFlashSession<Target, Draft>
where
    Target: TargetModel + crate::engine::ddtree::GemmaTreeTarget,
    Draft: DraftModel,
{
    /// DDTree variant of `run_generate`.
    ///
    /// Uses the DFlash drafter to produce per-position logits, builds a
    /// budget-N tree from them, runs a single fused tree forward on the
    /// target with a custom visibility mask + per-token position_ids,
    /// walks the tree to accept the longest matching path, then compacts
    /// the cache in-place (keeping only the accepted slots) and runs a
    /// 1-token verify on the bonus to install its K/V and capture the
    /// next-cycle root_pred.
    ///
    /// Falls back to plain `run_generate` semantics if `config.ddtree`
    /// is None.
    pub fn run_generate_ddtree(
        &mut self,
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
        temp: f32,
        eos_token_ids: &[u32],
    ) -> impl Iterator<Item = Result<u32, Exception>> + '_ {
        let ddtree_cfg = match self.config.ddtree {
            Some(cfg) => cfg,
            None => {
                // No-op pass-through; rely on the caller to use
                // `run_generate` instead. We return an iterator that just
                // delegates by buffering all tokens. To keep this simple
                // and avoid lifetime gymnastics, surface an error.
                return Box::new(std::iter::once(Err(Exception::custom(
                    "run_generate_ddtree called but SpeculativeCycleConfig.ddtree is None",
                ))))
                    as Box<dyn Iterator<Item = Result<u32, Exception>>>;
            }
        };
        let block_len = self.config.block_len;
        self.metrics = SessionMetrics::default();
        let eos_tokens: HashSet<u32> = eos_token_ids.iter().copied().collect();

        let mut prompt_tokens = Some(prompt_tokens);
        let mut initialized = false;
        let mut finished = max_tokens == 0;
        let mut emitted = 0usize;
        let mut last_token: u32 = 0;
        let mut root_pred: u32 = 0;
        let mut pending = VecDeque::<Result<u32, Exception>>::new();

        Box::new(std::iter::from_fn(move || loop {
            if let Some(item) = pending.pop_front() {
                if let Ok(t) = &item {
                    emitted += 1;
                    self.metrics.total_tokens = emitted;
                    last_token = *t;
                    if emitted >= max_tokens || eos_tokens.contains(t) {
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
                // Build CopySpec index over the prompt if not provided —
                // matches run_generate's behavior, enables free drafts on
                // prompts that echo themselves.
                if self.copyspec.is_none() {
                    self.copyspec = Some(CopySpecIndex::new(&prompt));
                }
                let prompt_arr = Array::from_slice(&prompt, &[1, prompt.len() as i32]);
                let target_logits = match self.target.prefill(&prompt_arr) {
                    Ok(l) => l,
                    Err(e) => {
                        finished = true;
                        return Some(Err(e));
                    }
                };
                if let Err(e) = self.draft.prefill(&prompt_arr) {
                    finished = true;
                    return Some(Err(e));
                }
                if let Some(h) = self.target.last_target_hidden() {
                    self.draft.set_target_hidden(h);
                }
                let first_token = match self
                    .target
                    .sample(&target_logits, temp)
                    .and_then(|t| scalar_token(&t))
                {
                    Ok(t) => t,
                    Err(e) => {
                        finished = true;
                        return Some(Err(e));
                    }
                };
                // Install first_token in target KV + capture root_pred.
                let first_arr = Array::from_slice(&[first_token], &[1, 1]);
                let first_post = match self.target.verify(&first_arr) {
                    Ok(l) => l,
                    Err(e) => {
                        finished = true;
                        return Some(Err(e));
                    }
                };
                if let Err(e) = eval([&first_post]) {
                    finished = true;
                    return Some(Err(e));
                }
                let root_pred_token = match argmax_axis!(
                    first_post.index((.., -1, ..)).reshape(&[-1]).unwrap(),
                    -1
                )
                .and_then(|a| a.as_dtype(Dtype::Uint32))
                .and_then(|a| {
                    let c = a.contiguous()?;
                    eval([&c])?;
                    Ok(c.item::<u32>())
                }) {
                    Ok(t) => t,
                    Err(e) => {
                        finished = true;
                        return Some(Err(e));
                    }
                };
                root_pred = root_pred_token;
                pending.push_back(Ok(first_token));
                initialized = true;
                continue;
            }

            let remaining = max_tokens.saturating_sub(emitted);
            if remaining == 0 {
                finished = true;
                return None;
            }
            if remaining == 1 {
                // A tree cycle needs >= 2 tokens of headroom, but the last
                // verify already captured the target's greedy prediction for
                // the next position (`root_pred`) — emit it as the final
                // token instead of stopping one short of max_tokens. KV for
                // it is never needed (generation ends here).
                finished = true;
                if initialized {
                    pending.push_back(Ok(root_pred));
                    continue;
                }
                return None;
            }
            let block = block_len.min(remaining).max(2);

            // CopySpec short-circuit: if the prompt-tail index has a hit
            // for (committed tail + last_token), use those prompt tokens as
            // a degenerate "chain tree" (one node per depth, no branching)
            // and skip the draft model forward entirely. The target verify
            // still gates every proposed token via the same accept_path
            // walk used for real trees.
            let copyspec_hit = self
                .copyspec
                .as_ref()
                .and_then(|c| c.draft_after(last_token, block - 1, None));
            let (tree, _from_copyspec) = if let Some(chain) = copyspec_hit {
                self.metrics.copyspec_hits += 1;
                self.metrics.copyspec_tokens += chain.len();
                let chain_tree: Vec<crate::engine::ddtree::TreeNode> = chain
                    .iter()
                    .enumerate()
                    .map(|(i, &tok)| crate::engine::ddtree::TreeNode {
                        token_id: tok,
                        parent: if i == 0 { usize::MAX } else { i - 1 },
                        depth: i + 1,
                        joint_logp: 0.0,
                    })
                    .collect();
                (chain_tree, true)
            } else {
                // Staged embedding + target hidden for the DFlash drafter.
                if let Some(emb) = self.target.embed_token(last_token) {
                    self.draft.set_staged_embedding(emb);
                }
                if let Some(h) = self.target.last_target_hidden() {
                    self.draft.set_target_hidden(h);
                }
                let last_arr = Array::from_slice(&[last_token], &[1, 1]);
                let drafted = match self.draft.draft_block(&last_arr, block) {
                    Ok(b) => b,
                    Err(e) => {
                        finished = true;
                        return Some(Err(e));
                    }
                };
                let block_logits = drafted.logits.index((0, .., ..));
                if let Err(e) = eval([&block_logits]) {
                    finished = true;
                    return Some(Err(e));
                }
                let (top_ids, top_lps) = match crate::engine::ddtree::topk_per_position(
                    &block_logits,
                    ddtree_cfg.tree_topk,
                ) {
                    Ok(v) => v,
                    Err(e) => {
                        finished = true;
                        return Some(Err(e));
                    }
                };
                (
                    crate::engine::ddtree::build_tree(&top_ids, &top_lps, ddtree_cfg.tree_budget),
                    false,
                )
            };

            let kv_offset = self.target.step_count() as i32;
            let start_pos = kv_offset;

            // Fused tree verify.
            let (tokens, position_ids, mask) =
                match crate::engine::ddtree::compile_tree_inputs(
                    last_token,
                    &tree,
                    start_pos,
                    kv_offset,
                    Dtype::Bfloat16,
                ) {
                    Ok(t) => t,
                    Err(e) => {
                        finished = true;
                        return Some(Err(e));
                    }
                };
            let tree_logits = match self
                .target
                .verify_tree_call(&tokens, &position_ids, &mask)
            {
                Ok(l) => l,
                Err(e) => {
                    finished = true;
                    return Some(Err(e));
                }
            };
            if let Err(e) = eval([&tree_logits]) {
                finished = true;
                return Some(Err(e));
            }
            let preds = match argmax_axis!(&tree_logits, -1)
                .and_then(|a| a.as_dtype(Dtype::Uint32))
                .and_then(|a| a.reshape(&[-1]))
            {
                Ok(p) => p,
                Err(e) => {
                    finished = true;
                    return Some(Err(e.into()));
                }
            };
            if let Err(e) = eval([&preds]) {
                finished = true;
                return Some(Err(e));
            }
            let preds_vec = preds.as_slice::<u32>().to_vec();
            let (accepted, bonus) =
                crate::engine::ddtree::accept_path(&tree, &preds_vec, root_pred);

            // Compact in-place to just the accepted node slots, then
            // 1-token verify on bonus to install its K/V + capture next
            // cycle's root_pred from the last-position logits.
            let keep_i32: Vec<i32> = accepted.iter().map(|&i| i as i32).collect();
            let keep_arr = Array::from_slice(
                &keep_i32,
                &[keep_i32.len() as i32],
            );
            if let Err(e) = self.target.compact_cache_call(kv_offset, &keep_arr) {
                finished = true;
                return Some(Err(e));
            }
            let bonus_arr = Array::from_slice(&[bonus], &[1, 1]);
            let bonus_logits = match self.target.verify(&bonus_arr) {
                Ok(l) => l,
                Err(e) => {
                    finished = true;
                    return Some(Err(e));
                }
            };
            if let Err(e) = eval([&bonus_logits]) {
                finished = true;
                return Some(Err(e));
            }
            let next_root = match argmax_axis!(
                bonus_logits.index((.., -1, ..)).reshape(&[-1]).unwrap(),
                -1
            )
            .and_then(|a| a.as_dtype(Dtype::Uint32))
            .and_then(|a| {
                let c = a.contiguous()?;
                eval([&c])?;
                Ok(c.item::<u32>())
            }) {
                Ok(t) => t,
                Err(e) => {
                    finished = true;
                    return Some(Err(e));
                }
            };
            root_pred = next_root;

            // Commit accepted prefix + bonus.
            let mut committed_this: Vec<u32> = Vec::with_capacity(accepted.len() + 1);
            for &i in &accepted {
                let t = tree[i].token_id;
                pending.push_back(Ok(t));
                committed_this.push(t);
            }
            pending.push_back(Ok(bonus));
            committed_this.push(bonus);

            // Extend CopySpec index with everything committed this cycle so
            // future cycles can short-circuit when the model echoes
            // previously-emitted spans (common in code, math, structured
            // outputs).
            if let Some(cs) = self.copyspec.as_mut() {
                cs.append_committed(&committed_this);
            }

            self.metrics.total_cycles += 1;
            self.metrics.total_accepted += accepted.len();
            self.metrics.total_drafted += tree.len();
            if self.metrics.total_drafted > 0 {
                self.metrics.acceptance_ratio =
                    self.metrics.total_accepted as f32 / self.metrics.total_drafted as f32;
            }
            self.metrics.avg_block_len =
                self.metrics.total_accepted as f32 / self.metrics.total_cycles.max(1) as f32;
        })) as Box<dyn Iterator<Item = Result<u32, Exception>>>
    }
}

/// Returns the verify input array `[1, drafted+1]` AND the drafted tokens as
/// a host vec. Pulling the drafted tokens to host is the cycle's first hard
/// GPU sync — returning the vec lets the caller reuse it instead of paying a
/// second eval + device→host copy on the same array.
fn build_verify_inputs(
    last_token: u32,
    drafted_tokens: &Array,
) -> Result<(Array, Vec<u32>), Exception> {
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
    let inputs = Array::from_slice(&verify_inputs, &[1, verify_inputs.len() as i32]);
    Ok((inputs, drafted))
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
        let (inputs, drafted_vec) = build_verify_inputs(10, &drafted).unwrap();
        // Now includes all drafted tokens: [last_token, draft[0..n]]
        assert_eq!(inputs.shape(), &[1, 5]);
        assert_eq!(
            array_to_vec_u32(&inputs.index((0, ..))).unwrap(),
            vec![10, 11, 12, 13, 14]
        );
        assert_eq!(drafted_vec, vec![11, 12, 13, 14]);
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
