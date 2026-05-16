//! Optional dispatch hook for specialized verify-time quantized matmul kernels.
//!
//! When `OMINIX_VERIFY_QMM=1` AND a hook has been registered via
//! `QUANTIZED_VERIFY_QMM_HOOK.set(...)`, the wrapper
//! `quantized_linear_forward` routes through the registered kernel for
//! shape-eligible quantized projections (M=16, K%32==0, N%32==0, bits=4).
//! Otherwise it forwards normally through `MaybeQuantized::forward`.
//!
//! The hook is set by `dflash-mlx::Qwen36TargetAdapter::with_dflash(...)` to
//! the `mma2big` Metal kernel ported from Python `dflash_mlx.verify_qmm`.
//! qwen3.6-mlx itself does not depend on dflash-mlx — the hook is a plain
//! function pointer registered at runtime to avoid the circular dep.

use std::cell::Cell;
use std::sync::OnceLock;

use mlx_rs::{
    error::Exception, module::Module, nn, quantization::MaybeQuantized, Array, Dtype,
};

thread_local! {
    /// Per-thread "we are currently inside a target verify pass" flag. When
    /// set, eligible quantized projections (M ∈ [4, 16], aligned K/N, bits=4,
    /// bf16/fp16) route through the verify-qmm kernel; pad up to M=16 if
    /// needed. When unset (the default, including AR and draft forwards),
    /// every projection takes the stock `MaybeQuantized::forward` path.
    static VERIFY_ACTIVE: Cell<bool> = const { Cell::new(false) };
}

/// RAII guard: set the verify flag for the duration of a forward pass.
/// Use from `Qwen36TargetAdapter::verify` to scope the kernel dispatch.
pub struct VerifyScope {
    prev: bool,
}

impl VerifyScope {
    pub fn enter() -> Self {
        let prev = VERIFY_ACTIVE.with(|f| {
            let p = f.get();
            f.set(true);
            p
        });
        Self { prev }
    }
}

impl Drop for VerifyScope {
    fn drop(&mut self) {
        let prev = self.prev;
        VERIFY_ACTIVE.with(|f| f.set(prev));
    }
}

fn verify_active() -> bool {
    VERIFY_ACTIVE.with(|f| f.get())
}

/// Signature: `(x, w_packed, scales, biases, group_size, bits) -> Result<Array>`.
///
/// `x` is `[M=16, K]`, `w_packed` is the inner Linear's packed-uint32 weight,
/// `scales` and `biases` come straight from `QuantizedLinear`'s `scales` /
/// `biases` params. Output is `[16, N]`.
pub type QmmHook =
    fn(&Array, &Array, &Array, &Array, i32, i32) -> Result<Array, Exception>;

/// Optional global hook. Set once at program start (or per session adapter
/// construction). Subsequent `set` calls are ignored.
pub static QUANTIZED_VERIFY_QMM_HOOK: OnceLock<QmmHook> = OnceLock::new();

fn env_flag_enabled() -> bool {
    // Cache the env-var lookup to avoid syscalls on every projection call.
    // Even with `OMINIX_VERIFY_QMM=1`, dispatch only fires inside a
    // `VerifyScope` — the env var is the master switch, the scope guards
    // each verify call site.
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        std::env::var("OMINIX_VERIFY_QMM")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

/// Forward through a `MaybeQuantized<Linear>`, dispatching to the verify-qmm
/// hook when the env var is set, the linear is quantized, and the shape is
/// eligible. Falls back to `linear.forward(x)` otherwise.
pub fn quantized_linear_forward(
    linear: &mut MaybeQuantized<nn::Linear>,
    x: &Array,
) -> Result<Array, Exception> {
    if !env_flag_enabled() || !verify_active() {
        return linear.forward(x);
    }
    let hook = match QUANTIZED_VERIFY_QMM_HOOK.get() {
        Some(h) => *h,
        None => return linear.forward(x),
    };
    let q = match linear {
        MaybeQuantized::Quantized(q) => q,
        MaybeQuantized::Original(_) => return linear.forward(x),
    };

    let x_shape = x.shape();
    if x_shape.is_empty() {
        return linear.forward(x);
    }
    // M = product of all but the last axis; K = last axis.
    let k = *x_shape.last().unwrap_or(&0);
    let m: i32 = x_shape[..x_shape.len() - 1].iter().product();
    // weight is stored as [N, K/8] uint32 for bits=4.
    let n = q.inner.weight.as_ref().shape()[0];
    // Kernel requires bf16/fp16 inputs and matching dtype on scales/biases.
    // Fall back for any other dtype (e.g. fp32 intermediates from DeltaNet).
    if !matches!(x.dtype(), Dtype::Bfloat16 | Dtype::Float16) {
        return linear.forward(x);
    }
    if q.scales.as_ref().dtype() != x.dtype() || q.biases.as_ref().dtype() != x.dtype() {
        return linear.forward(x);
    }
    // M=16 kernel only. Padding M<16 up to 16 was tried and is a net
    // regression in practice: with adaptive block sizing pinning block_len
    // at 4, 4x wasted FLOPs from the pad can't be recouped by the M=16
    // kernel's throughput advantage. Small-M cases need a small-M kernel
    // (port of Python's `m4_ksplit_np`) — tracked as a follow-up.
    if m != 16 || k % 32 != 0 || n % 32 != 0 || q.bits != 4 {
        return linear.forward(x);
    }

    // Hook wants 2-D input; reshape if needed.
    let needs_reshape = x_shape.len() != 2;
    let x2 = if needs_reshape {
        x.reshape(&[m, k])?
    } else {
        x.clone()
    };
    let out_2d = hook(
        &x2,
        q.inner.weight.as_ref(),
        q.scales.as_ref(),
        q.biases.as_ref(),
        q.group_size,
        q.bits,
    )?;
    if needs_reshape {
        let mut out_shape: Vec<i32> = x_shape[..x_shape.len() - 1].to_vec();
        out_shape.push(n);
        out_2d.reshape(&out_shape)
    } else {
        Ok(out_2d)
    }
}
