use mlx_rs::{error::Exception, Array};
use mlx_rs_core::cache::{KVCache, KeyValueCache, QuantizedKVCache, TurboQuantKVCache};
use mlx_rs_core::kvflash::KvFlashCache;
use mlx_rs_core::paged::PagedKvCache;

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

impl RecurrentState {
    /// Persist the recurrent state to a safetensors file. Populated
    /// `state` and `conv_state` Arrays are stored under those names;
    /// `step` is stored in metadata. Empty state (step==0) is allowed
    /// and produces a manifest-only file (no tensors).
    pub fn save_to_path(&self, path: impl AsRef<std::path::Path>) -> Result<(), Exception> {
        let mut tensors: Vec<(&str, &Array)> = Vec::new();
        if let Some(t) = self.state.as_ref() {
            tensors.push(("state", t));
        }
        if let Some(t) = self.conv_state.as_ref() {
            tensors.push(("conv_state", t));
        }
        let mut meta = std::collections::HashMap::new();
        meta.insert("step".to_string(), self.step.to_string());
        Array::save_safetensors(tensors, Some(&meta), path.as_ref())
            .map_err(|e| Exception::custom(format!("save_safetensors: {e}")))?;
        Ok(())
    }

    pub fn load_from_path(path: impl AsRef<std::path::Path>) -> Result<Self, Exception> {
        let (map, meta) = Array::load_safetensors_with_metadata(path.as_ref())
            .map_err(|e| Exception::custom(format!("load_safetensors_with_metadata: {e}")))?;
        let step = meta
            .get("step")
            .and_then(|v| v.parse::<i32>().ok())
            .unwrap_or(0);
        Ok(Self {
            state: map.get("state").cloned(),
            conv_state: map.get("conv_state").cloned(),
            step,
        })
    }
}

// ============================================================================
// HybridCache disk persistence (Qwen3.6 prefix-cache)
//
// Each layer is one safetensors file with a `kind` metadata tag
// identifying the variant. The HybridCache::Recurrent variant is
// supported alongside ::KV and ::QuantizedKV (the modes Qwen3.6 uses
// in practice). ::TurboQuantKV is currently rejected — its 9-tensor
// state is more invasive to serialize and isn't on the default
// OminiX-API code path.
// ============================================================================

impl HybridCache {
    pub fn save_to_path(&self, path: impl AsRef<std::path::Path>) -> Result<(), Exception> {
        let path = path.as_ref();
        match self {
            HybridCache::KV(kv) => {
                // Write a tiny kind marker alongside the KVCache file.
                kv.save_to_path(path)?;
                let kind_path = path.with_extension("kind");
                std::fs::write(&kind_path, b"kv")
                    .map_err(|e| Exception::custom(format!("write kind: {e}")))?;
                Ok(())
            }
            HybridCache::QuantizedKV(qkv) => {
                qkv.save_to_path(path)?;
                std::fs::write(path.with_extension("kind"), b"qkv")
                    .map_err(|e| Exception::custom(format!("write kind: {e}")))?;
                Ok(())
            }
            HybridCache::Recurrent(rec) => {
                rec.save_to_path(path)?;
                std::fs::write(path.with_extension("kind"), b"rec")
                    .map_err(|e| Exception::custom(format!("write kind: {e}")))?;
                Ok(())
            }
            HybridCache::TurboQuantKV(_) => Err(Exception::custom(
                "HybridCache::save_to_path: TurboQuantKV not yet supported",
            )),
            // Paged caches live in an in-memory pool and are never disk-persisted
            // (the API forces an in-memory prefix cache when paging is enabled).
            HybridCache::Paged(_) => Err(Exception::custom(
                "HybridCache::save_to_path: Paged not supported (in-memory only)",
            )),
            HybridCache::KvFlash(_) => Err(Exception::custom(
                "HybridCache::save_to_path: KvFlash not supported (lossy resident pool)",
            )),
        }
    }

    pub fn load_from_path(path: impl AsRef<std::path::Path>) -> Result<Self, Exception> {
        let path = path.as_ref();
        let kind_path = path.with_extension("kind");
        let kind = std::fs::read_to_string(&kind_path)
            .map_err(|e| Exception::custom(format!("read kind {}: {e}", kind_path.display())))?;
        match kind.trim() {
            "kv" => Ok(HybridCache::KV(KVCache::load_from_path(path)?)),
            "qkv" => Ok(HybridCache::QuantizedKV(QuantizedKVCache::load_from_path(path)?)),
            "rec" => Ok(HybridCache::Recurrent(RecurrentState::load_from_path(path)?)),
            other => Err(Exception::custom(format!(
                "HybridCache::load_from_path: unknown kind '{other}'"
            ))),
        }
    }

    /// Save a Vec<HybridCache> + token sequence to a directory.
    /// Mirrors `mlx_rs_core::cache::KVCache::save_kv_caches`.
    pub fn save_hybrid_caches(
        caches: &[HybridCache],
        tokens: &[i32],
        dir: impl AsRef<std::path::Path>,
    ) -> Result<(), Exception> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir)
            .map_err(|e| Exception::custom(format!("create_dir_all {}: {e}", dir.display())))?;
        for (i, c) in caches.iter().enumerate() {
            if c.offset() == 0 {
                continue;
            }
            c.save_to_path(dir.join(format!("cache_{i}.safetensors")))?;
        }
        let manifest = serde_json::json!({
            "tokens": tokens,
            "n_caches": caches.len(),
        });
        std::fs::write(
            dir.join("manifest.json"),
            serde_json::to_vec(&manifest)
                .map_err(|e| Exception::custom(format!("manifest serialize: {e}")))?,
        )
        .map_err(|e| Exception::custom(format!("write manifest: {e}")))?;
        Ok(())
    }

    /// Try to load a HybridCache session whose token sequence is a strict
    /// prefix of `prompt_tokens`. Returns `Some((caches, n_cached))` on
    /// hit. Returns `None` if no manifest is present or the cached tokens
    /// don't prefix-match.
    pub fn try_load_hybrid_caches(
        prompt_tokens: &[i32],
        dir: impl AsRef<std::path::Path>,
    ) -> Result<Option<(Vec<HybridCache>, usize)>, Exception> {
        let dir = dir.as_ref();
        let manifest_path = dir.join("manifest.json");
        if !manifest_path.exists() {
            return Ok(None);
        }
        let bytes = std::fs::read(&manifest_path)
            .map_err(|e| Exception::custom(format!("read manifest: {e}")))?;
        let manifest: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|e| Exception::custom(format!("manifest parse: {e}")))?;
        let cached_tokens: Vec<i32> = manifest
            .get("tokens")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_i64().map(|n| n as i32)).collect())
            .unwrap_or_default();
        let n = cached_tokens.len();
        if n == 0 || prompt_tokens.len() < n {
            return Ok(None);
        }
        if prompt_tokens[..n] != cached_tokens[..] {
            return Ok(None);
        }
        let n = n.min(prompt_tokens.len() - 1);
        if n == 0 {
            return Ok(None);
        }
        let n_caches = manifest
            .get("n_caches")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let mut caches = Vec::with_capacity(n_caches);
        for i in 0..n_caches {
            let path = dir.join(format!("cache_{i}.safetensors"));
            let kind_path = path.with_extension("kind");
            let cache = if kind_path.exists() {
                HybridCache::load_from_path(&path)?
            } else {
                // No kind marker → empty slot at save time (recurrent
                // layer that hadn't been touched; or skipped).
                HybridCache::Recurrent(RecurrentState::new())
            };
            caches.push(cache);
        }
        // The caller re-prefills `prompt_tokens[n..]`, so the loaded caches
        // must hold exactly `n` positions. When the cached sequence equals
        // the whole prompt, `n` was capped to len-1 above but the snapshot
        // still holds all len positions — without trimming, the final token
        // would enter the KV cache twice (RoPE offsets len-1 AND len) and
        // run through the GDN recurrence a second time.
        let excess = cached_tokens.len() - n;
        if excess > 0 {
            for cache in caches.iter_mut() {
                match cache {
                    HybridCache::Recurrent(rec) => {
                        if rec.step > n as i32 {
                            // Recurrent state can't be rolled back without a
                            // tape — treat the snapshot as a miss.
                            return Ok(None);
                        }
                    }
                    other => {
                        if other.offset() > n as i32 {
                            let drop = other.offset() - n as i32;
                            if other.trim(drop).is_err() {
                                return Ok(None);
                            }
                        }
                    }
                }
            }
        }
        Ok(Some((caches, n)))
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
    /// Spike: TurboQuant 4-bit K + 8-bit V cache with fused single-dispatch
    /// SDPA path. Wired via `KVCacheMode::TurboQuant`.
    TurboQuantKV(TurboQuantKVCache),
    Recurrent(RecurrentState),
    /// Paged KV for full-attention layers (behind `OMINIX_PAGED_ATTENTION`).
    /// Cloning forks the block table (shared prefix blocks, CoW on divergence);
    /// dropping releases blocks back to the shared pool.
    Paged(PagedKvCache),
    /// Spike: KVFlash bounded-residency cache (sink + recent window, LRU
    /// chunk eviction at decode). Wired via `KVCacheMode::KvFlash`
    /// (`DFLASH_KVFLASH=<pool>`). Decode attends a `<= pool` working set so
    /// throughput stays flat as context grows.
    KvFlash(KvFlashCache),
}

impl HybridCache {
    pub fn offset(&self) -> i32 {
        match self {
            HybridCache::KV(kv) => kv.offset(),
            HybridCache::QuantizedKV(qkv) => qkv.offset(),
            HybridCache::TurboQuantKV(tq) => tq.offset(),
            HybridCache::Recurrent(rec) => rec.step,
            HybridCache::Paged(p) => p.offset(),
            HybridCache::KvFlash(kf) => kf.offset(),
        }
    }

    /// Trim trailing `n_drop` positions from the cache. Used by multi-step
    /// MTP drafting to roll back rejected draft positions on the target's
    /// KV cache. Recurrent (GDN) slots are no-op — speculation paths
    /// that include GDN layers should use `trim_gdn` with a captured
    /// snapshot instead.
    pub fn trim(&mut self, n_drop: i32) -> Result<(), Exception> {
        if n_drop <= 0 {
            return Ok(());
        }
        match self {
            HybridCache::KV(kv) => {
                kv.trim(n_drop);
                Ok(())
            }
            HybridCache::TurboQuantKV(tq) => tq.trim(n_drop),
            HybridCache::QuantizedKV(qkv) => qkv.trim_kv(n_drop),
            HybridCache::Recurrent(_) => Ok(()), // no-op; see trim_gdn
            HybridCache::Paged(p) => p.trim_kv(n_drop),
            // Spec-decode rollback isn't supported on the lossy resident pool
            // (a trimmed token may already be evicted); KvFlash is decode-only.
            HybridCache::KvFlash(_) => Err(Exception::custom(
                "HybridCache::trim: KvFlash does not support speculative rollback",
            )),
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
            HybridCache::QuantizedKV(qkv) => {
                // Residual-only trim (F20): rejected draft tokens are recent and
                // live in the unquantized residual. Errors if a rollback reaches
                // into a quantized block (unsupported), rather than corrupting.
                qkv.trim_kv(n_drop)
            }
            HybridCache::TurboQuantKV(tq) => {
                tq.trim(n_drop)?;
                Ok(())
            }
            HybridCache::Paged(p) => p.trim_kv(n_drop),
            HybridCache::KvFlash(_) => Err(Exception::custom(
                "HybridCache::trim_gdn: KvFlash does not support speculative rollback",
            )),
            HybridCache::Recurrent(rec) => {
                let snap = snapshot.ok_or_else(|| {
                    Exception::custom(
                        "HybridCache::trim_gdn: GDN layer requires a snapshot to roll back",
                    )
                })?;
                let n_keep = (verify_len - n_drop).max(0);
                let replayed = snap.capture.replay_prefix(&snap.state, n_keep as usize)?;
                rec.state = Some(replayed);
                // Rebuild conv1d sliding window for the post-`n_keep` position.
                // snap.conv_state is the PRE-verify window; we need the window
                // after the first n_keep accepted verify tokens have flowed
                // through conv1d. Replay = take the last (k-1) elements of
                // concat([pre_conv_state, qkv_cf[:, :, :n_keep]], -1).
                rec.conv_state = Some(
                    snap.capture
                        .rolled_conv_state(snap.conv_state.as_ref(), n_keep as usize)?,
                );
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
    ///
    /// This is the window BEFORE the verify pass consumed any tokens.
    /// `trim_gdn` combines this with `capture.qkv_cf[:, :, :n_keep]` to
    /// reconstruct the post-`n_keep` window.
    pub conv_state: Option<Array>,
    /// `cache.step` at the moment the snapshot was taken.
    pub step: i32,
    /// Innovation tape + (k, decay) captured by `forward_prefill_with_tape`.
    pub capture: crate::deltanet::GdnTapeCapture,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deltanet::GdnTapeCapture;
    use mlx_rs::ops::zeros_dtype;
    use mlx_rs::Dtype;

    // F1 regression: on a speculative-decoding rollback, a recurrent (GDN)
    // layer MUST be rolled back. `HybridCache::trim` is a no-op for recurrent
    // slots (leaving GDN state advanced past the accepted prefix while KV
    // layers are trimmed — silent corruption); `trim_gdn` replays the accepted
    // prefix from a snapshot and rolls the step back. This guards that the MTP
    // verify path uses `trim_gdn`, not `trim`, for recurrent layers.
    #[test]
    fn trim_is_noop_for_recurrent_but_trim_gdn_rolls_back() {
        let (b, h, kdim, vdim, l) = (1i32, 1i32, 32i32, 32i32, 4i32);
        let conv_dim = 96i32;
        let post_state = zeros_dtype(&[b, h, kdim, vdim], Dtype::Float32).unwrap();
        let rec = RecurrentState {
            state: Some(post_state),
            conv_state: None,
            step: 10 + l, // advanced by the verify of `l` tokens
        };

        // `trim` leaves the recurrent step advanced — documents the bug.
        let mut c_trim = HybridCache::Recurrent(rec.clone());
        c_trim.trim(2).unwrap();
        assert_eq!(
            c_trim.offset(),
            10 + l,
            "trim must be a no-op for recurrent layers",
        );

        // `trim_gdn` rolls the layer back to snapshot.step + (verify_len - n_drop).
        let capture = GdnTapeCapture {
            tape: zeros_dtype(&[b, h, l, vdim], Dtype::Float32).unwrap(),
            k: zeros_dtype(&[b, h, l, kdim], Dtype::Float32).unwrap(),
            decay: zeros_dtype(&[b, h, l], Dtype::Float32).unwrap(),
            qkv_cf: zeros_dtype(&[b, conv_dim, l], Dtype::Float32).unwrap(),
            conv_kernel_size: 4,
        };
        let snap = GdnRollbackSnapshot {
            state: zeros_dtype(&[b, h, kdim, vdim], Dtype::Float32).unwrap(),
            conv_state: None,
            step: 10,
            capture,
        };
        let mut c_gdn = HybridCache::Recurrent(rec);
        c_gdn.trim_gdn(2, l, Some(&snap)).unwrap();
        assert_eq!(c_gdn.offset(), 10 + (l - 2)); // 12: rolled back to accepted prefix
    }
}
