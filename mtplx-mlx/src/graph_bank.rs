//! Shape-keyed cache of type-erased compiled MLX graphs.
//!
//! The first iteration of this module was observation-only: it counted
//! hits/misses per `GraphKey` and stored an opaque `Box<dyn Any + Send>`
//! placeholder so callers could simulate the cache shape without actually
//! reusing graphs.
//!
//! This revision swaps the placeholder for [`mlx_rs::transforms::CompiledFn`],
//! a type-erased `FnMut(&[Array]) -> Result<Vec<Array>, Exception>`. The
//! cache now holds a real callable per shape signature. Real wiring of the
//! verify-forward path into this cache still requires further work (see
//! `session.rs`); the observation API is preserved for callers that only
//! want stats.

use std::collections::HashMap;

use mlx_rs::error::Exception;
use mlx_rs::transforms::CompiledFn;
use mlx_rs::Array;

/// Type-erased compiled graph slot stored in the bank.
pub type CompiledGraph = CompiledFn;

/// Stable identifier for a compiled graph. Today we key on a free-form
/// label plus an input-shape signature; layer index / dtype can be folded
/// into the label by the caller.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GraphKey {
    /// Caller-supplied tag, e.g. `"verify_forward"` or `"mtp_draft"`.
    pub tag: String,
    /// Concatenated input shapes: `[[1,1], [1,1,4096]]` → `"1x1|1x1x4096"`.
    pub shape_sig: String,
}

impl GraphKey {
    pub fn new(tag: impl Into<String>, shape_sig: impl Into<String>) -> Self {
        Self {
            tag: tag.into(),
            shape_sig: shape_sig.into(),
        }
    }

    /// Build a shape signature from a slice of input arrays.
    pub fn shape_sig_for(inputs: &[&Array]) -> String {
        let mut out = String::new();
        for (i, arr) in inputs.iter().enumerate() {
            if i > 0 {
                out.push('|');
            }
            let shape = arr.shape();
            for (j, d) in shape.iter().enumerate() {
                if j > 0 {
                    out.push('x');
                }
                out.push_str(&d.to_string());
            }
        }
        out
    }
}

/// Aggregate hit/miss counters reported by the bank.
#[derive(Debug, Clone, Copy, Default)]
pub struct GraphBankStats {
    pub hits: u64,
    pub misses: u64,
}

impl GraphBankStats {
    pub fn total(&self) -> u64 {
        self.hits + self.misses
    }

    pub fn hit_rate(&self) -> f64 {
        let t = self.total();
        if t == 0 {
            0.0
        } else {
            self.hits as f64 / t as f64
        }
    }
}

/// Shape-keyed cache of compiled callables.
#[derive(Default)]
pub struct GraphBank {
    entries: HashMap<GraphKey, CompiledGraph>,
    stats: GraphBankStats,
}

impl std::fmt::Debug for GraphBank {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraphBank")
            .field("entry_count", &self.entries.len())
            .field("stats", &self.stats)
            .finish()
    }
}

impl GraphBank {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn stats(&self) -> GraphBankStats {
        self.stats
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Legacy observation-only API. Bumps the hit counter when the key is
    /// already known, miss otherwise. Does NOT populate the slot — that's
    /// the caller's job via [`Self::get_or_compile`] or [`Self::invoke`].
    pub fn observe(&mut self, key: &GraphKey) {
        if self.entries.contains_key(key) {
            self.stats.hits += 1;
        } else {
            self.stats.misses += 1;
        }
    }

    /// Fetch the cached callable for `key`, building it via `build` on
    /// miss. Returns `&mut CompiledGraph` so the caller can drive it.
    pub fn get_or_compile<F>(
        &mut self,
        key: GraphKey,
        build: F,
    ) -> Result<&mut CompiledGraph, Exception>
    where
        F: FnOnce() -> Result<CompiledGraph, Exception>,
    {
        if self.entries.contains_key(&key) {
            self.stats.hits += 1;
        } else {
            self.stats.misses += 1;
            let graph = build()?;
            self.entries.insert(key.clone(), graph);
        }
        Ok(self
            .entries
            .get_mut(&key)
            .expect("entry just inserted or already present"))
    }

    /// Convenience: get-or-compile, then immediately call the slot with
    /// `inputs`. Mirrors how callers will typically use the bank.
    pub fn invoke<F>(
        &mut self,
        key: GraphKey,
        build: F,
        inputs: &[Array],
    ) -> Result<Vec<Array>, Exception>
    where
        F: FnOnce() -> Result<CompiledGraph, Exception>,
    {
        let slot = self.get_or_compile(key, build)?;
        slot.call(inputs)
    }
}
