use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub struct SpeculativeCycleConfig {
    pub block_len: usize,
    pub prefill_step_size: usize,
    pub capture_layer_ids: Vec<usize>,
    pub min_block_tokens: usize,
    pub drop_acceptance_threshold: f32,
    pub tpc_threshold: f32,
    pub adaptive_window: usize,
    pub adaptive_cooldown: usize,
}

impl Default for SpeculativeCycleConfig {
    fn default() -> Self {
        Self {
            block_len: 16,
            prefill_step_size: 2048,
            capture_layer_ids: Vec::new(),
            min_block_tokens: 4,
            drop_acceptance_threshold: 0.75,
            tpc_threshold: 3.5,
            adaptive_window: 4,
            adaptive_cooldown: 64,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AdaptiveBlockPolicy {
    config: SpeculativeCycleConfig,
    window_acceptance: VecDeque<f32>,
    window_tpc: VecDeque<f32>,
    reduced_burst_remaining: usize,
}

impl AdaptiveBlockPolicy {
    pub fn new(config: SpeculativeCycleConfig) -> Self {
        Self {
            config,
            window_acceptance: VecDeque::new(),
            window_tpc: VecDeque::new(),
            reduced_burst_remaining: 0,
        }
    }

    pub fn current_block_len(&self) -> usize {
        if self.reduced_burst_remaining > 0 {
            self.config.min_block_tokens.min(self.config.block_len)
        } else {
            self.config.block_len.max(self.config.min_block_tokens)
        }
    }

    pub fn update(&mut self, acceptance: f32, tokens_per_cycle: f32) {
        push_window(
            &mut self.window_acceptance,
            acceptance,
            self.config.adaptive_window.max(1),
        );
        push_window(
            &mut self.window_tpc,
            tokens_per_cycle,
            self.config.adaptive_window.max(1),
        );

        if self.reduced_burst_remaining > 0 {
            self.reduced_burst_remaining -= 1;
        }

        let avg_acceptance = mean(&self.window_acceptance);
        let avg_tpc = mean(&self.window_tpc);
        if avg_acceptance < self.config.drop_acceptance_threshold
            || avg_tpc < self.config.tpc_threshold
        {
            self.reduced_burst_remaining = self.config.adaptive_cooldown;
        }
    }
}

fn push_window(window: &mut VecDeque<f32>, value: f32, limit: usize) {
    if window.len() == limit {
        window.pop_front();
    }
    window.push_back(value);
}

fn mean(window: &VecDeque<f32>) -> f32 {
    if window.is_empty() {
        0.0
    } else {
        window.iter().sum::<f32>() / window.len() as f32
    }
}
