//! Rust → Core ML predict bridge (spike).
//!
//! Owns an opaque handle to a Core ML model loaded on macOS via the Swift
//! shim in `ane_runner.swift`. The shim is compiled and linked by
//! `build.rs`; on non-mac targets the crate compiles to a stub returning
//! `Unsupported` from every call.

use std::ffi::CString;
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CoreMlError {
    #[error("Core ML predict failed (code {0})")]
    Predict(i32),
    #[error("Core ML load failed")]
    Load,
    #[error("invalid path: {0}")]
    Path(String),
    #[error("output buffer too small: needed {needed}, had {had}")]
    BufferTooSmall { needed: usize, had: usize },
    #[error("not supported on this platform")]
    Unsupported,
}

/// Compute-unit selector. Matches Core ML's `MLComputeUnits` enum plus a
/// macOS 14+ "neural-engine-preferred" variant we map to
/// `.cpuAndNeuralEngine` (Core ML has no `.neuralEngineOnly`).
#[repr(i32)]
#[derive(Debug, Clone, Copy)]
pub enum ComputeUnits {
    All = 0,
    CpuOnly = 1,
    CpuAndGpu = 2,
    CpuAndNeuralEngine = 3,
    NeuralEnginePreferred = 4,
}

#[cfg(target_os = "macos")]
extern "C" {
    fn coreml_load(path: *const std::os::raw::c_char, compute_units: i32) -> *mut std::ffi::c_void;
    fn coreml_predict(
        handle: *mut std::ffi::c_void,
        pixels: *const f32,
        pixel_count: i32,
        out_buffer: *mut f32,
        out_capacity: i32,
        out_actual_len: *mut i32,
    ) -> i32;
    fn coreml_predict_zero_copy(
        handle: *mut std::ffi::c_void,
        pixels: *mut f32,
        pixel_count: i32,
        out_buffer: *mut f32,
        out_capacity: i32,
        out_actual_len: *mut i32,
    ) -> i32;
    fn coreml_free(handle: *mut std::ffi::c_void);
}

pub struct CoreMlModel {
    #[cfg(target_os = "macos")]
    handle: *mut std::ffi::c_void,
}

impl CoreMlModel {
    pub fn load(path: impl AsRef<Path>, units: ComputeUnits) -> Result<Self, CoreMlError> {
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (path, units);
            return Err(CoreMlError::Unsupported);
        }
        #[cfg(target_os = "macos")]
        unsafe {
            let path_str = path
                .as_ref()
                .to_str()
                .ok_or_else(|| CoreMlError::Path(format!("{}", path.as_ref().display())))?;
            let c = CString::new(path_str).map_err(|_| CoreMlError::Path(path_str.to_string()))?;
            let handle = coreml_load(c.as_ptr(), units as i32);
            if handle.is_null() {
                return Err(CoreMlError::Load);
            }
            Ok(Self { handle })
        }
    }

    /// Zero-copy predict. Wraps `pixels` as MLMultiArray with a no-op
    /// deallocator on the Swift side — saves one CPU copy per inference
    /// (especially worthwhile when pixels were just produced by an MLX
    /// graph and we want to avoid bouncing through a temporary buffer).
    ///
    /// SAFETY: caller must keep `pixels` alive across the call. The
    /// `&mut [f32]` borrow guarantees that, and we transmute it to a
    /// raw mutable pointer for the FFI.
    pub fn predict_zero_copy(
        &self,
        pixels: &mut [f32],
        out: &mut [f32],
    ) -> Result<usize, CoreMlError> {
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (pixels, out);
            return Err(CoreMlError::Unsupported);
        }
        #[cfg(target_os = "macos")]
        unsafe {
            let mut actual: i32 = 0;
            let rc = coreml_predict_zero_copy(
                self.handle,
                pixels.as_mut_ptr(),
                pixels.len() as i32,
                out.as_mut_ptr(),
                out.len() as i32,
                &mut actual,
            );
            match rc {
                0 => Ok(actual as usize),
                -6 => Err(CoreMlError::BufferTooSmall {
                    needed: actual as usize,
                    had: out.len(),
                }),
                code => Err(CoreMlError::Predict(code)),
            }
        }
    }

    pub fn predict(&self, pixels: &[f32], out: &mut [f32]) -> Result<usize, CoreMlError> {
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (pixels, out);
            return Err(CoreMlError::Unsupported);
        }
        #[cfg(target_os = "macos")]
        unsafe {
            let mut actual: i32 = 0;
            let rc = coreml_predict(
                self.handle,
                pixels.as_ptr(),
                pixels.len() as i32,
                out.as_mut_ptr(),
                out.len() as i32,
                &mut actual,
            );
            match rc {
                0 => Ok(actual as usize),
                -6 => Err(CoreMlError::BufferTooSmall {
                    needed: actual as usize,
                    had: out.len(),
                }),
                code => Err(CoreMlError::Predict(code)),
            }
        }
    }
}

#[cfg(target_os = "macos")]
impl Drop for CoreMlModel {
    fn drop(&mut self) {
        unsafe {
            if !self.handle.is_null() {
                coreml_free(self.handle);
                self.handle = std::ptr::null_mut();
            }
        }
    }
}

// SAFETY: the Swift shim's MLModel handle is itself thread-safe for
// concurrent predict calls per Apple's MLModel docs. We mark Send so a
// model can be moved between threads; Sync requires Apple's internal
// thread-safety guarantees so we leave it off and let callers wrap in
// Arc<Mutex<...>> for shared concurrent access.
#[cfg(target_os = "macos")]
unsafe impl Send for CoreMlModel {}
