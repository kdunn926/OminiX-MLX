//! Block-hashed disk tier for the paged KV pool — a cross-request,
//! cross-restart cache of prefilled KV blocks to cut TTFT.
//!
//! Each *full* `block_size`-token block of a prompt is given a **chained
//! content hash** `h[i] = fnv(h[i-1], tokens[block i])`, so a block's hash
//! encodes its entire prefix (vLLM-style). Block KV is stored on disk keyed by
//! `(hash, layer)`. On a new request, the longest run of leading block hashes
//! already on disk is loaded straight into the pool, skipping that much
//! prefill; after generation, the prompt's full blocks are written back.
//!
//! Prototype scope: wired into the Qwen3 dense paged path. Keying is per model
//! (KV is model/dtype/tokenizer specific) and uses exact token-prefix hashes.

use std::path::{Path, PathBuf};

use mlx_rs::{error::Exception, Array};

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv1a(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// Chained content hashes for each *full* block of `tokens`. `hashes[i]`
/// identifies the prefix `tokens[..(i+1)*block_size]`. Deterministic across
/// runs (FNV-1a), so the cache survives restarts.
pub fn block_hashes(tokens: &[u32], block_size: usize) -> Vec<u64> {
    let mut hashes = Vec::new();
    let mut h = FNV_OFFSET;
    let n_full = tokens.len() / block_size;
    let mut buf = Vec::with_capacity(block_size * 4);
    for b in 0..n_full {
        buf.clear();
        for &t in &tokens[b * block_size..(b + 1) * block_size] {
            buf.extend_from_slice(&t.to_le_bytes());
        }
        h = fnv1a(h, &buf);
        hashes.push(h);
    }
    hashes
}

fn sanitize(s: &str) -> String {
    let out: String = s
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect();
    if out.is_empty() {
        "default".to_string()
    } else {
        out
    }
}

/// Per-model on-disk store of KV blocks keyed by `(chained block hash, layer)`.
pub struct PagedBlockDiskCache {
    dir: PathBuf,
    block_size: usize,
}

impl PagedBlockDiskCache {
    /// Create (or open) the store under `base_dir/<model>/paged_b<block_size>`.
    /// Namespaced per model + block size — KV is model/dtype/tokenizer specific.
    pub fn open(base_dir: impl AsRef<Path>, model_key: &str, block_size: usize) -> std::io::Result<Self> {
        let dir = base_dir
            .as_ref()
            .join(sanitize(model_key))
            .join(format!("paged_b{block_size}"));
        std::fs::create_dir_all(&dir)?;
        Ok(Self { dir, block_size })
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    fn path(&self, hash: u64, layer: usize) -> PathBuf {
        self.dir.join(format!("{hash:016x}_L{layer}.safetensors"))
    }

    /// Whether block `(hash, layer)` is on disk.
    pub fn has(&self, hash: u64, layer: usize) -> bool {
        self.path(hash, layer).exists()
    }

    /// Length of the longest leading run of `hashes` present on disk for
    /// `layer` (probing one layer is enough — all layers of a block are saved
    /// together).
    pub fn cached_prefix_blocks(&self, hashes: &[u64], layer: usize) -> usize {
        let mut n = 0;
        for &h in hashes {
            if self.has(h, layer) {
                n += 1;
            } else {
                break;
            }
        }
        n
    }

    /// Persist a block's K/V (`[B, H, block_size, D]`). Atomic (temp + rename).
    pub fn save(&self, hash: u64, layer: usize, k: &Array, v: &Array) -> Result<(), Exception> {
        let final_path = self.path(hash, layer);
        // Temp name must keep the `.safetensors` extension (mlx infers format
        // from it); the leading dot + pid keeps concurrent writers distinct.
        let tmp = self.dir.join(format!(
            ".{hash:016x}_L{layer}.{}.partial.safetensors",
            std::process::id()
        ));
        mlx_rs::Array::save_safetensors([("k", k), ("v", v)], None, &tmp)
            .map_err(|e| Exception::custom(format!("paged disk save: {e}")))?;
        std::fs::rename(&tmp, &final_path)
            .map_err(|e| Exception::custom(format!("paged disk rename: {e}")))?;
        Ok(())
    }

    /// Load a block's K/V if present.
    pub fn load(&self, hash: u64, layer: usize) -> Result<Option<(Array, Array)>, Exception> {
        let p = self.path(hash, layer);
        if !p.exists() {
            return Ok(None);
        }
        let map = mlx_rs::Array::load_safetensors(&p)
            .map_err(|e| Exception::custom(format!("paged disk load: {e}")))?;
        match (map.get("k"), map.get("v")) {
            (Some(k), Some(v)) => Ok(Some((k.clone(), v.clone()))),
            _ => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fresh unique temp dir (avoids a `tempfile` dev-dependency).
    fn tmp_dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir()
            .join(format!("ominix_paged_disk_{}_{}", name, std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn chained_hashes_are_deterministic_and_prefix_sensitive() {
        let a = block_hashes(&[1, 2, 3, 4, 5, 6, 7, 8], 4); // 2 full blocks
        let b = block_hashes(&[1, 2, 3, 4, 5, 6, 7, 8], 4);
        assert_eq!(a, b, "deterministic");
        assert_eq!(a.len(), 2);
        // Same first block ⇒ same first hash; different second block ⇒ differs.
        let c = block_hashes(&[1, 2, 3, 4, 9, 9, 9, 9], 4);
        assert_eq!(a[0], c[0], "shared prefix block ⇒ shared hash");
        assert_ne!(a[1], c[1], "diverging block ⇒ different hash");
        // A changed first block changes the chained hash of the second too.
        let d = block_hashes(&[0, 2, 3, 4, 5, 6, 7, 8], 4);
        assert_ne!(a[0], d[0]);
        assert_ne!(a[1], d[1], "chaining: parent change propagates");
        // Partial trailing block is not hashed.
        assert_eq!(block_hashes(&[1, 2, 3, 4, 5], 4).len(), 1);
    }

    #[test]
    fn save_load_roundtrip() {
        let dir = tmp_dir("roundtrip");
        let cache = PagedBlockDiskCache::open(&dir, "m", 4).unwrap();
        let k = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 1, 4, 1]);
        let v = Array::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[1, 1, 4, 1]);
        assert!(!cache.has(0xABCD, 0));
        cache.save(0xABCD, 0, &k, &v).unwrap();
        assert!(cache.has(0xABCD, 0));
        let (lk, lv) = cache.load(0xABCD, 0).unwrap().unwrap();
        let close = lk.all_close(&k, None, None, None).unwrap();
        assert!(close.as_slice::<bool>()[0], "k roundtrips");
        let close = lv.all_close(&v, None, None, None).unwrap();
        assert!(close.as_slice::<bool>()[0], "v roundtrips");
        assert!(cache.load(0x9999, 0).unwrap().is_none(), "miss → None");
    }

    #[test]
    fn cached_prefix_blocks_stops_at_first_miss() {
        let dir = tmp_dir("prefix");
        let cache = PagedBlockDiskCache::open(&dir, "m", 4).unwrap();
        let z = Array::from_slice(&[0.0f32], &[1, 1, 1, 1]);
        let hashes = [10u64, 11, 12, 13];
        cache.save(10, 0, &z, &z).unwrap();
        cache.save(11, 0, &z, &z).unwrap();
        // 12 missing → run stops at 2.
        cache.save(13, 0, &z, &z).unwrap();
        assert_eq!(cache.cached_prefix_blocks(&hashes, 0), 2);
    }
}
