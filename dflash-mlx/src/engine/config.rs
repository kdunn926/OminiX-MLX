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
    /// DDTree (Diffusion Draft Tree) override. When `Some`, sessions whose
    /// target implements `GemmaTreeTarget` can use `run_generate_ddtree`
    /// for tree-shaped speculative decoding. The plain `run_generate`
    /// path ignores this field.
    pub ddtree: Option<DDTreeConfig>,
    /// Number of consecutive cycles to stay in `Reduced` mode before the
    /// adaptive policy schedules a `Probe` cycle that briefly tries the
    /// large block again. Ported from python dflash-mlx v0.1.7. 0 keeps
    /// the legacy v0.1.6 behaviour (no probe, stay reduced for the
    /// whole `adaptive_cooldown` window).
    pub probe_interval_cycles: usize,
    /// Minimum number of cycles to spend in `Reduced` mode before the
    /// first probe fires. Avoids probing immediately after dropping,
    /// when the workload hasn't stabilised yet. Default 16.
    pub probe_min_reduced_cycles: usize,
}

/// State machine for the adaptive verify policy (python dflash v0.1.7).
/// Tracks the size of the verify block the next cycle should use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlockMode {
    /// Normal large-block path (uses `block_len`).
    Large,
    /// Reduced-block burst path (uses `min_block_tokens`).
    Reduced,
    /// A single probe cycle inside a Reduced phase that temporarily
    /// runs the large block to measure whether the workload would
    /// benefit from returning to Large.
    Probe,
}

/// Knobs for the DDTree variant of the spec-decoding cycle.
#[derive(Debug, Clone, Copy)]
pub struct DDTreeConfig {
    /// Maximum number of tree nodes per cycle (excludes the implicit
    /// root / seed). Larger budgets explore more candidates but cost
    /// proportionally more per-cycle verify.
    pub tree_budget: usize,
    /// Per-depth top-k token count for tree expansion. The heap pops at
    /// most `topk` candidates per depth before expanding deeper.
    pub tree_topk: usize,
}

impl Default for DDTreeConfig {
    fn default() -> Self {
        Self {
            tree_budget: 6,
            tree_topk: 2,
        }
    }
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
            ddtree: None,
            probe_interval_cycles: 32,
            probe_min_reduced_cycles: 16,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AdaptiveBlockPolicy {
    config: SpeculativeCycleConfig,
    window_acceptance: VecDeque<f32>,
    window_tpc: VecDeque<f32>,
    /// Real wall-clock tokens-per-second window (v0.1.7) — the actual
    /// throughput metric used to decide between modes. acceptance and
    /// tpc inform the LARGE→REDUCED drop; this window informs the
    /// REDUCED→LARGE promotion via probe comparison.
    window_real_tps: VecDeque<f32>,

    /// State machine (python dflash v0.1.7).
    mode: BlockMode,
    /// Cycles spent in the current mode so far.
    cycles_in_mode: usize,
    /// Wall-clock real-tps observed during the Reduced phase before a
    /// probe fires. Captured at probe-launch so we can compare after.
    reduced_real_tps_baseline: Option<f32>,
    /// True when the next `update` call should compare the probe's
    /// real_tps against the captured baseline.
    probe_pending_compare: bool,
}

impl AdaptiveBlockPolicy {
    pub fn new(config: SpeculativeCycleConfig) -> Self {
        Self {
            config,
            window_acceptance: VecDeque::new(),
            window_tpc: VecDeque::new(),
            window_real_tps: VecDeque::new(),
            mode: BlockMode::Large,
            cycles_in_mode: 0,
            reduced_real_tps_baseline: None,
            probe_pending_compare: false,
        }
    }

    /// Current state machine mode (Large / Reduced / Probe).
    pub fn mode(&self) -> BlockMode {
        self.mode
    }

    pub fn current_block_len(&self) -> usize {
        match self.mode {
            // Large + Probe both run the large block; Probe is just a
            // labelled Large cycle inside a Reduced phase.
            BlockMode::Large | BlockMode::Probe => {
                self.config.block_len.max(self.config.min_block_tokens)
            }
            BlockMode::Reduced => {
                self.config.min_block_tokens.min(self.config.block_len)
            }
        }
    }

    /// Update the policy with the just-completed cycle's stats.
    ///
    /// `acceptance` = n_accepted / drafted_count.
    /// `tokens_per_cycle` = committed tokens this cycle (including the
    /// target's correction/bonus token).
    /// `cycle_wall_s` = wall-clock seconds spent on this cycle.
    pub fn update(&mut self, acceptance: f32, tokens_per_cycle: f32, cycle_wall_s: f32) {
        let real_tps = tokens_per_cycle / cycle_wall_s.max(1e-6);
        let w = self.config.adaptive_window.max(1);
        push_window(&mut self.window_acceptance, acceptance, w);
        push_window(&mut self.window_tpc, tokens_per_cycle, w);
        push_window(&mut self.window_real_tps, real_tps, w);

        self.cycles_in_mode += 1;
        let avg_acc = mean(&self.window_acceptance);
        let avg_tpc = mean(&self.window_tpc);
        let avg_tps = mean(&self.window_real_tps);

        match self.mode {
            BlockMode::Large => {
                // Drop into Reduced when the large block stops paying.
                if avg_acc < self.config.drop_acceptance_threshold
                    || avg_tpc < self.config.tpc_threshold
                {
                    self.mode = BlockMode::Reduced;
                    self.cycles_in_mode = 0;
                    self.reduced_real_tps_baseline = None;
                    self.probe_pending_compare = false;
                }
            }
            BlockMode::Reduced => {
                // After we've stayed reduced for at least the warm-up
                // window, fire a single probe every `probe_interval`
                // cycles. The legacy v0.1.6 behaviour (no probe) is
                // recovered by setting probe_interval_cycles = 0.
                let probe_ok = self.config.probe_interval_cycles > 0
                    && self.cycles_in_mode >= self.config.probe_min_reduced_cycles
                    && (self.cycles_in_mode - self.config.probe_min_reduced_cycles)
                        % self.config.probe_interval_cycles
                        == 0;
                if probe_ok {
                    self.reduced_real_tps_baseline = Some(avg_tps);
                    self.mode = BlockMode::Probe;
                    self.probe_pending_compare = true;
                }
            }
            BlockMode::Probe => {
                if self.probe_pending_compare {
                    // Compare probe's real_tps against the Reduced
                    // baseline. Promote back to Large only if the probe
                    // beat it; otherwise stay Reduced.
                    let baseline = self.reduced_real_tps_baseline.unwrap_or(0.0);
                    self.mode = if real_tps > baseline {
                        BlockMode::Large
                    } else {
                        BlockMode::Reduced
                    };
                    self.cycles_in_mode = 0;
                    self.probe_pending_compare = false;
                    self.reduced_real_tps_baseline = None;
                }
            }
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
