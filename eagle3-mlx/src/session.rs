//! Env-gated EAGLE-3 session for Gemma4 targets.
//!
//! ## Env knobs
//!
//! | var | effect |
//! |-----|--------|
//! | `OMINIX_EAGLE3=1` | master gate; sessions refuse to load without it |
//! | `EAGLE3_DRAFT_DIR` | default draft checkpoint dir for examples |
//! | `EAGLE3_BLOCK` | draft chain length (default: checkpoint's `speculative_tokens`, 3) |
//!
//! EAGLE-3 is only wired for model pairs it was trained on: the draft
//! checkpoint's `speculators_config.verifier` must name the Gemma4 family,
//! and the target's hidden size / layer count must match the draft's
//! expectations. Other model families fail at load with a clear error.

use std::path::Path;

use mlx_rs::error::Exception;
use mlx_rs_core::error::Error;

use dflash_mlx::{DFlashSession, Gemma4TargetAdapter, SessionMetrics, SpeculativeCycleConfig};
use gemma4_mlx::KVCache;

use crate::adapter::Eagle3DraftAdapter;
use crate::model::Eagle3DraftModel;

/// Master env gate: `OMINIX_EAGLE3=1` (or `true`/`yes`/`on`).
pub fn env_enabled() -> bool {
    matches!(
        std::env::var("OMINIX_EAGLE3").ok().as_deref(),
        Some("1") | Some("true") | Some("yes") | Some("on")
    )
}

/// A running EAGLE-3 speculative decoding session over a Gemma4 target.
pub struct Eagle3Session {
    inner: DFlashSession<Gemma4TargetAdapter<KVCache>, Eagle3DraftAdapter>,
    pub block_len: usize,
}

impl Eagle3Session {
    /// Load target + draft and validate that the pair is one EAGLE-3 is
    /// trained for. Errors unless `OMINIX_EAGLE3=1` is set.
    pub fn load(
        target_dir: impl AsRef<Path>,
        draft_dir: impl AsRef<Path>,
        temp: f32,
    ) -> Result<Self, Error> {
        if !env_enabled() {
            return Err(Error::Model(
                "EAGLE-3 is env-gated: set OMINIX_EAGLE3=1 to enable".into(),
            ));
        }

        // The EAGLE-3 ingest pass needs hidden rows back to the last
        // committed position it consumed; the DFlash segment cap turns the
        // accumulator into a sliding window that can drop prompt rows before
        // the first draft cycle. Disable it unless the user overrode it.
        if std::env::var("DFLASH_MAX_HIDDEN_SEGS").is_err() {
            std::env::set_var("DFLASH_MAX_HIDDEN_SEGS", "0");
        }

        let draft = Eagle3DraftModel::load(&draft_dir)?;
        let cfg = &draft.config;

        // "Relevant models" gate: this crate only wires the Gemma4 target
        // adapter, and the draft head is only valid for the verifier it was
        // trained against.
        let verifier = cfg.verifier_name().to_ascii_lowercase();
        if !verifier.contains("gemma-4") && !verifier.contains("gemma4") {
            return Err(Error::Model(format!(
                "EAGLE-3 draft was trained for verifier {:?}; only Gemma4 targets are wired \
                 (load the matching speculator for other families)",
                cfg.verifier_name()
            )));
        }
        // vLLM convention: aux layer id i = residual stream at the INPUT of
        // target layer i. gemma4-mlx captures at layer OUTPUT, so shift by 1.
        // Id 0 (the embedding stream) has no layer-output equivalent.
        let mut capture_ids = Vec::with_capacity(cfg.eagle_aux_hidden_state_layer_ids.len());
        for &id in &cfg.eagle_aux_hidden_state_layer_ids {
            if id == 0 {
                return Err(Error::Model(
                    "EAGLE-3 aux layer id 0 (embedding stream) is not supported by the \
                     gemma4 capture path"
                        .into(),
                ));
            }
            capture_ids.push(id - 1);
        }

        let target = gemma4_mlx::load_model(&target_dir)?;
        let t_hidden = target.args.hidden_size;
        let d_hidden = cfg.transformer_layer_config.hidden_size;
        let t_layers = target.args.num_hidden_layers as usize;
        let max_aux = *cfg.eagle_aux_hidden_state_layer_ids.iter().max().unwrap_or(&0);
        if cfg.target_hidden_size.unwrap_or(d_hidden) != t_hidden {
            return Err(Error::Model(format!(
                "target hidden_size {t_hidden} != draft's expected target hidden size {}",
                cfg.target_hidden_size.unwrap_or(d_hidden)
            )));
        }
        if max_aux >= t_layers {
            return Err(Error::Model(format!(
                "aux layer id {max_aux} out of range for a {t_layers}-layer target"
            )));
        }

        let block_len = std::env::var("EAGLE3_BLOCK")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n >= 2)
            .unwrap_or_else(|| cfg.default_speculative_tokens().max(2));

        let target = Gemma4TargetAdapter::<KVCache>::with_dflash(target, temp, capture_ids);
        let draft = Eagle3DraftAdapter::new(draft);
        let config = SpeculativeCycleConfig {
            block_len,
            min_block_tokens: 2,
            ..Default::default()
        };
        Ok(Self {
            inner: DFlashSession::new(target, draft, config),
            block_len,
        })
    }

    pub fn generate(
        &mut self,
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
        temp: f32,
        eos_token_ids: &[u32],
    ) -> impl Iterator<Item = Result<u32, Exception>> + '_ {
        self.inner
            .run_generate(prompt_tokens, max_tokens, temp, eos_token_ids)
    }

    pub fn metrics(&self) -> &SessionMetrics {
        self.inner.metrics()
    }
}
