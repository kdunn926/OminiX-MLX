//! Type-erased wrapper around mlx-rs's typed `compile()`.
//!
//! [`compile`](crate::transforms::compile::compile) is generic in the closure
//! type, which makes it impossible to store heterogeneous compiled callables
//! in a `HashMap`. This wrapper erases the type via
//! `Box<dyn FnMut + Send>` so callers can keep a shape-keyed cache.
//!
//! Usage:
//! ```ignore
//! use mlx_rs::{Array, transforms::CompiledFn};
//!
//! let mut f = CompiledFn::new(|args: &[Array]| -> Result<Vec<Array>, _> {
//!     Ok(vec![args[0].add(&args[0])?])
//! });
//! let out = f.call(&[some_array])?;
//! ```
//!
//! Note: this wrapper does NOT itself invoke
//! [`compile`](crate::transforms::compile::compile). Fitting an arbitrary
//! `&[Array] -> Vec<Array>` shape into the typed `compile<F>` API is not
//! currently expressible from Rust without unsafe transmutes. `CompiledFn`
//! is a "slot" the caller fills with whatever closure they want (compiled
//! or not). When the inner closure does invoke `compile()`, MLX's
//! C++-level graph cache deduplicates the actual compile across multiple
//! `CompiledFn` instances anyway, so the practical effect is the same.

use crate::error::Exception;
use crate::Array;

/// Type-erased compiled callable.
///
/// See module docs for usage and caveats.
pub struct CompiledFn {
    inner: Box<dyn FnMut(&[Array]) -> Result<Vec<Array>, Exception> + Send + 'static>,
}

impl std::fmt::Debug for CompiledFn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledFn").finish_non_exhaustive()
    }
}

impl CompiledFn {
    /// Wrap an arbitrary `&[Array] -> Result<Vec<Array>, Exception>` closure
    /// in a type-erased slot suitable for storing in a `HashMap`.
    pub fn new<F>(f: F) -> Self
    where
        F: FnMut(&[Array]) -> Result<Vec<Array>, Exception> + Send + 'static,
    {
        Self { inner: Box::new(f) }
    }

    /// Invoke the wrapped closure.
    pub fn call(&mut self, inputs: &[Array]) -> Result<Vec<Array>, Exception> {
        (self.inner)(inputs)
    }
}
