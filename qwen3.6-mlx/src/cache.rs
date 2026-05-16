use mlx_rs::{error::Exception, Array};
use mlx_rs_core::cache::{KVCache, KeyValueCache, QuantizedKVCache};

/// Recurrent state for DeltaNet layers.
///
/// Stores a fixed-size state matrix and a conv1d sliding window buffer.
/// Unlike KV cache which grows with sequence length, this has constant size.
#[derive(Debug, Clone)]
pub struct RecurrentState {
    /// Delta rule state: [B, num_v_heads, k_dim, v_dim]
    pub state: Option<Array>,
    /// Conv1d sliding window: [B, conv_dim, kernel_size - 1]
    pub conv_state: Option<Array>,
    /// Number of tokens processed (for position tracking)
    pub step: i32,
}

impl RecurrentState {
    pub fn new() -> Self {
        Self {
            state: None,
            conv_state: None,
            step: 0,
        }
    }
}

impl Default for RecurrentState {
    fn default() -> Self {
        Self::new()
    }
}

/// Unified cache for hybrid model layers.
///
/// Full attention layers use KV cache or quantized KV cache;
/// DeltaNet layers use recurrent state.
#[derive(Debug, Clone)]
pub enum HybridCache {
    KV(KVCache),
    QuantizedKV(QuantizedKVCache),
    Recurrent(RecurrentState),
}

impl HybridCache {
    pub fn offset(&self) -> i32 {
        match self {
            HybridCache::KV(kv) => kv.offset(),
            HybridCache::QuantizedKV(qkv) => qkv.offset(),
            HybridCache::Recurrent(rec) => rec.step,
        }
    }

    /// Deterministic rollback for a speculative-decoding verify pass.
    ///
    /// Full-attention layers drop the trailing `n_drop` positions from the
    /// KV cache in O(1) via `KVCache::trim` (matching the pre-verify offset).
    /// Recurrent (GDN) layers replay the first `(verify_len - n_drop)`
    /// captured tape entries from `snapshot` to reconstruct the post-accept
    /// recurrent state, avoiding a full re-forward over the accepted prefix.
    ///
    /// `verify_len` is the number of tokens that were fed to the verify pass.
    /// For GDN layers, `snapshot` MUST be supplied (the pre-verify state +
    /// recorded tape); for non-GDN layers it is ignored.
    pub fn trim_gdn(
        &mut self,
        n_drop: i32,
        verify_len: i32,
        snapshot: Option<&GdnRollbackSnapshot>,
    ) -> Result<(), Exception> {
        match self {
            HybridCache::KV(kv) => {
                kv.trim(n_drop);
                Ok(())
            }
            HybridCache::QuantizedKV(_) => {
                // QuantizedKVCache currently lacks an O(1) trim hook. Callers
                // that mix quantized KV with speculative rollback must fall
                // back to the snapshot+re-forward path until that lands.
                Err(Exception::custom(
                    "HybridCache::trim_gdn: QuantizedKVCache trim is not yet implemented",
                ))
            }
            HybridCache::Recurrent(rec) => {
                let snap = snapshot.ok_or_else(|| {
                    Exception::custom(
                        "HybridCache::trim_gdn: GDN layer requires a snapshot to roll back",
                    )
                })?;
                let n_keep = (verify_len - n_drop).max(0);
                let replayed = snap.capture.replay_prefix(&snap.state, n_keep as usize)?;
                rec.state = Some(replayed);
                rec.conv_state = snap.conv_state.clone();
                rec.step = snap.step + n_keep;
                Ok(())
            }
        }
    }
}

/// Pre-verify snapshot for a single GDN layer, used by `HybridCache::trim_gdn`.
#[derive(Debug, Clone)]
pub struct GdnRollbackSnapshot {
    /// Pre-verify recurrent state [B, H, K, V].
    pub state: Array,
    /// Pre-verify conv1d sliding window [B, conv_dim, kernel_size - 1].
    pub conv_state: Option<Array>,
    /// `cache.step` at the moment the snapshot was taken.
    pub step: i32,
    /// Innovation tape + (k, decay) captured by `forward_prefill_with_tape`.
    pub capture: crate::deltanet::GdnTapeCapture,
}
