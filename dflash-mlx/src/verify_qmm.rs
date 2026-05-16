// Copyright 2026 OminiX-MLX
// Licensed under the Apache License, Version 2.0
//
// Rust port of dflash_mlx.verify_qmm — quantized matmul Metal kernels used by
// the target-side verify path. The Python reference lives at
// `dflash_mlx/verify_qmm.py` (installed under miniforge3).
//
// Variants and what triggers each (see `_auto_variant` / `_resolve_m16_ktmpl_variant`
// in the Python source, lines ~21-51):
//
//   mma2big (PORTED)
//     - M <= 16, N % 32 == 0, K % 32 == 0, bits == 4
//     - The simplest variant; one Metal kernel, no K-splitting or pipelining.
//     - Python: `_build_kernel_mma2big`, lines ~448-547.
//
//   mma2big_pipe (TODO)
//     - Same shape gates as `mma2big`, but triggered when K >= 8192 or N <= 8192.
//     - Splits K into `K_PARTS` (default 8) for better SMEM pipelining; reduces
//       partials on host. Python: `_build_kernel_mma2big_pipe`.
//
//   m16_combo_ktmpl (TODO)
//     - bits == 4, K % 256 == 0, N % 16 == 0, and (K >= 8192 or N <= 5120).
//     - Tree-reduction with K as template constant. Python:
//       `_build_kernel_m16_combo_ktmpl`.
//
//   m16_super_tree_fp16_ktmpl (TODO)
//     - bits == 4, K % 256 == 0, N % 16 == 0, !combo_ktmpl gate.
//     - Wider super-tree reduction. Python:
//       `_build_kernel_m16_super_tree_fp16_ktmpl`.
//
//   m4_ksplit_np (TODO)
//     - M == 4, bits == 4, N % 4 == 0, K % 32 == 0.
//     - Tiny-M decode path. Python: `_build_kernel_m4_ksplit_np`.

use mlx_rs::{error::Exception, Array, Dtype};
use std::collections::HashMap;
use std::ffi::CString;
use std::sync::Mutex;

// ---------------------------------------------------------------------------
// Kernel cache
// ---------------------------------------------------------------------------

struct KernelState {
    kernel: mlx_sys::mlx_fast_metal_kernel,
    input_names: mlx_sys::mlx_vector_string,
    output_names: mlx_sys::mlx_vector_string,
}

unsafe impl Send for KernelState {}
unsafe impl Sync for KernelState {}

impl Drop for KernelState {
    fn drop(&mut self) {
        unsafe {
            mlx_sys::mlx_fast_metal_kernel_free(self.kernel);
            mlx_sys::mlx_vector_string_free(self.input_names);
            mlx_sys::mlx_vector_string_free(self.output_names);
        }
    }
}

fn kernel_cache() -> &'static Mutex<HashMap<(i32, Dtype), KernelState>> {
    use std::sync::OnceLock;
    static CACHE: OnceLock<Mutex<HashMap<(i32, Dtype), KernelState>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn dtype_tag(dt: Dtype) -> &'static str {
    match dt {
        Dtype::Bfloat16 => "bf16",
        Dtype::Float16 => "fp16",
        _ => "unk",
    }
}

// ---------------------------------------------------------------------------
// Metal source — mma2big (4-bit, M<=16, BN=32, BK=32)
//
// Ported verbatim from Python `_build_kernel_mma2big`, lines ~453-537.
// The only literal substitution is `GS` (group_size). M=BM=16, K and N are
// taken from scalar Array inputs `K_size`, `N_size` (M_size is unused inside
// the kernel body in Python; we still accept it for input-list parity).
// ---------------------------------------------------------------------------

fn build_mma2big_source(group_size: i32) -> String {
    format!(
        r#"
        using namespace metal;
        constexpr int BM = 16;
        constexpr int BN = 32;
        constexpr int BK = 32;
        constexpr int BK_SUB = 8;
        constexpr int GS = {gs};

        uint tid   = thread_position_in_threadgroup.x;
        uint sg_id = tid / 32;
        uint tg_n  = threadgroup_position_in_grid.y;

        int K = int(K_size);
        int N = int(N_size);
        int K_by_8  = K / 8;
        int K_by_gs = K / GS;
        int n0 = int(tg_n) * BN;

        threadgroup T B_tile[BK * BN];

        simdgroup_matrix<T, 8, 8> a_top, a_bot, b_L, b_R;
        simdgroup_matrix<float, 8, 8> c_tL = simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<float, 8, 8> c_tR = simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<float, 8, 8> c_bL = simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<float, 8, 8> c_bR = simdgroup_matrix<float, 8, 8>(0.0f);

        int t_a = int(tid);
        int t_b = int(tid) + 64;
        int dq_k_a = t_a / BN, dq_n_a = t_a % BN;
        int dq_k_b = t_b / BN, dq_n_b = t_b % BN;

        int sg_n_off = int(sg_id) * 16;

        for (int k0 = 0; k0 < K; k0 += BK) {{
            {{
                int n_global = n0 + dq_n_a;
                int k_base = k0 + dq_k_a * 8;
                uint32_t packed = w_q[n_global * K_by_8 + (k_base >> 3)];
                float s = float(scales[n_global * K_by_gs + (k_base / GS)]);
                float b = float(biases[n_global * K_by_gs + (k_base / GS)]);
                for (int ki = 0; ki < 8; ++ki) {{
                    uint32_t nib = (packed >> (ki * 4)) & 0xFu;
                    B_tile[(dq_k_a * 8 + ki) * BN + dq_n_a] = T(float(nib) * s + b);
                }}
            }}
            {{
                int n_global = n0 + dq_n_b;
                int k_base = k0 + dq_k_b * 8;
                uint32_t packed = w_q[n_global * K_by_8 + (k_base >> 3)];
                float s = float(scales[n_global * K_by_gs + (k_base / GS)]);
                float b = float(biases[n_global * K_by_gs + (k_base / GS)]);
                for (int ki = 0; ki < 8; ++ki) {{
                    uint32_t nib = (packed >> (ki * 4)) & 0xFu;
                    B_tile[(dq_k_b * 8 + ki) * BN + dq_n_b] = T(float(nib) * s + b);
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (int ks = 0; ks < BK / BK_SUB; ++ks) {{
                simdgroup_load(a_top, x + k0 + ks * BK_SUB,                  K);
                simdgroup_load(a_bot, x + 8 * K + k0 + ks * BK_SUB,          K);
                simdgroup_load(b_L, B_tile + ks * BK_SUB * BN + sg_n_off,         BN);
                simdgroup_load(b_R, B_tile + ks * BK_SUB * BN + sg_n_off + 8,     BN);
                simdgroup_multiply_accumulate(c_tL, a_top, b_L, c_tL);
                simdgroup_multiply_accumulate(c_tR, a_top, b_R, c_tR);
                simdgroup_multiply_accumulate(c_bL, a_bot, b_L, c_bL);
                simdgroup_multiply_accumulate(c_bR, a_bot, b_R, c_bR);
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        simdgroup_matrix<T, 8, 8> c_tL_T, c_tR_T, c_bL_T, c_bR_T;
        c_tL_T.thread_elements()[0] = T(c_tL.thread_elements()[0]);
        c_tL_T.thread_elements()[1] = T(c_tL.thread_elements()[1]);
        c_tR_T.thread_elements()[0] = T(c_tR.thread_elements()[0]);
        c_tR_T.thread_elements()[1] = T(c_tR.thread_elements()[1]);
        c_bL_T.thread_elements()[0] = T(c_bL.thread_elements()[0]);
        c_bL_T.thread_elements()[1] = T(c_bL.thread_elements()[1]);
        c_bR_T.thread_elements()[0] = T(c_bR.thread_elements()[0]);
        c_bR_T.thread_elements()[1] = T(c_bR.thread_elements()[1]);
        simdgroup_store(c_tL_T, y + n0 + sg_n_off,                  N);
        simdgroup_store(c_tR_T, y + n0 + sg_n_off + 8,              N);
        simdgroup_store(c_bL_T, y + 8 * N + n0 + sg_n_off,          N);
        simdgroup_store(c_bR_T, y + 8 * N + n0 + sg_n_off + 8,      N);
    "#,
        gs = group_size,
    )
}

fn create_mma2big_kernel(group_size: i32, dtype: Dtype) -> KernelState {
    let source = build_mma2big_source(group_size);
    let name = format!("verify_mma2big_gs{}_{}", group_size, dtype_tag(dtype));

    unsafe {
        let input_names = mlx_sys::mlx_vector_string_new();
        for input in ["x", "w_q", "scales", "biases", "M_size", "K_size", "N_size"] {
            let c = CString::new(input).unwrap();
            mlx_sys::mlx_vector_string_append_value(input_names, c.as_ptr());
        }

        let output_names = mlx_sys::mlx_vector_string_new();
        let out_c = CString::new("y").unwrap();
        mlx_sys::mlx_vector_string_append_value(output_names, out_c.as_ptr());

        let source_c = CString::new(source).unwrap();
        let header_c = CString::new("").unwrap();
        let name_c = CString::new(name).unwrap();
        let kernel = mlx_sys::mlx_fast_metal_kernel_new(
            name_c.as_ptr(),
            input_names,
            output_names,
            source_c.as_ptr(),
            header_c.as_ptr(),
            true,
            false,
        );

        KernelState {
            kernel,
            input_names,
            output_names,
        }
    }
}

// ---------------------------------------------------------------------------
// Public entry: mma2big
// ---------------------------------------------------------------------------

/// Gating heuristic for the mma2big M=16 path.
///
/// Returns true if the input shape and bitness are eligible for the
/// `mma2big` kernel. Mirrors `_auto_variant`/dispatch gates in
/// `verify_linear.py` (lines ~141-148).
pub fn mma2big_eligible(m: i32, k: i32, n: i32, bits: i32) -> bool {
    bits == 4 && m > 0 && m <= 16 && n > 0 && k > 0 && n % 32 == 0 && k % 32 == 0
}

/// Run the `mma2big` verify quantized matmul kernel.
///
/// Layout matches Python verify_qmm: `x` is `[M, K]` (M<=16, padded to 16),
/// `w_packed` is `[N, K/8]` uint32, scales/biases are `[N, K/group_size]`
/// in `dtype`. Output is `[16, N]` in `dtype`.
///
/// Equivalent to `mlx_rs::ops::quantized_matmul(x, w_packed, scales, biases,
/// transpose=true, group_size, bits=4)` but bypasses the general path.
pub fn verify_qmm_m16_mma2big(
    x: &Array,
    w_packed: &Array,
    scales: &Array,
    biases: &Array,
    group_size: i32,
    bits: i32,
) -> Result<Array, Exception> {
    if bits != 4 {
        return Err(Exception::custom(format!(
            "verify_qmm_m16_mma2big: only bits==4 supported, got {bits}"
        )));
    }
    let x_shape = x.shape();
    let w_shape = w_packed.shape();
    if x_shape.len() != 2 || w_shape.len() != 2 {
        return Err(Exception::custom(format!(
            "verify_qmm_m16_mma2big expects 2-D x and w_packed, got x={x_shape:?} w={w_shape:?}"
        )));
    }
    let m = x_shape[0];
    let k = x_shape[1];
    let n = w_shape[0];
    let k_by_8 = w_shape[1];
    if k_by_8 * 8 != k {
        return Err(Exception::custom(format!(
            "verify_qmm_m16_mma2big: w_packed inner dim {k_by_8} != K/8 (K={k})"
        )));
    }
    if !mma2big_eligible(m, k, n, bits) {
        return Err(Exception::custom(format!(
            "verify_qmm_m16_mma2big: ineligible shape m={m} k={k} n={n} (need m<=16, k%32==0, n%32==0, bits==4)"
        )));
    }
    if m != 16 {
        return Err(Exception::custom(format!(
            "verify_qmm_m16_mma2big: kernel requires x to be padded to M=16 rows, got M={m}"
        )));
    }
    if k % group_size != 0 {
        return Err(Exception::custom(format!(
            "verify_qmm_m16_mma2big: K ({k}) not divisible by group_size ({group_size})"
        )));
    }

    let dtype = x.dtype();
    if dtype != Dtype::Bfloat16 && dtype != Dtype::Float16 {
        return Err(Exception::custom(format!(
            "verify_qmm_m16_mma2big: x dtype must be bf16 or fp16, got {:?}",
            dtype
        )));
    }
    if scales.dtype() != dtype || biases.dtype() != dtype {
        return Err(Exception::custom(format!(
            "verify_qmm_m16_mma2big: scales/biases dtype must match x ({:?}), got scales={:?} biases={:?}",
            dtype,
            scales.dtype(),
            biases.dtype()
        )));
    }

    let dtype_u32: u32 = dtype.into();

    // Ensure cached kernel exists for (group_size, dtype). Keyed by
    // (group_size, dtype) — K and N flow through scalar Array inputs at apply time.
    {
        let mut cache = kernel_cache().lock().unwrap();
        cache
            .entry((group_size, dtype))
            .or_insert_with(|| create_mma2big_kernel(group_size, dtype));
    }

    // Scalar inputs (M, K, N) as 0-D int32 arrays.
    let m_scalar = Array::from_int(m);
    let k_scalar = Array::from_int(k);
    let n_scalar = Array::from_int(n);

    unsafe {
        let stream = mlx_sys::mlx_default_gpu_stream_new();
        let config = mlx_sys::mlx_fast_metal_kernel_config_new();

        // Template arg: T dtype.
        let t_name = CString::new("T").unwrap();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
            config,
            t_name.as_ptr(),
            dtype_u32,
        );

        // Grid: (64, N/32, 1), threadgroup: (64, 1, 1). See verify_linear.py:206-207.
        mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, 64, n / 32, 1);
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, 64, 1, 1);

        // Output: [16, N] in dtype.
        let y_shape = vec![16i32, n];
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            y_shape.as_ptr(),
            y_shape.len(),
            dtype_u32,
        );

        let inputs = mlx_sys::mlx_vector_array_new();
        mlx_sys::mlx_vector_array_append_value(inputs, x.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, w_packed.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, scales.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, biases.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, m_scalar.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, k_scalar.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, n_scalar.as_ptr());

        // Re-acquire kernel pointer under lock (it stays valid for process lifetime).
        let kernel_ptr = {
            let cache = kernel_cache().lock().unwrap();
            cache.get(&(group_size, dtype)).unwrap().kernel
        };

        let mut outputs = mlx_sys::mlx_vector_array_new();
        let ret = mlx_sys::mlx_fast_metal_kernel_apply(
            &mut outputs,
            kernel_ptr,
            inputs,
            config,
            stream,
        );
        if ret != 0 {
            mlx_sys::mlx_fast_metal_kernel_config_free(config);
            mlx_sys::mlx_vector_array_free(inputs);
            mlx_sys::mlx_vector_array_free(outputs);
            mlx_sys::mlx_stream_free(stream);
            return Err(Exception::custom(
                "verify_qmm_m16_mma2big Metal kernel execution failed",
            ));
        }

        let mut y_ptr = mlx_sys::mlx_array_new();
        mlx_sys::mlx_vector_array_get(&mut y_ptr, outputs, 0);

        mlx_sys::mlx_fast_metal_kernel_config_free(config);
        mlx_sys::mlx_vector_array_free(inputs);
        mlx_sys::mlx_vector_array_free(outputs);
        mlx_sys::mlx_stream_free(stream);

        Ok(Array::from_ptr(y_ptr))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::{quantize, quantized_matmul};

    fn max_abs_diff_bf16(a: &Array, b: &Array) -> f32 {
        let a32 = a.as_dtype(Dtype::Float32).unwrap();
        let b32 = b.as_dtype(Dtype::Float32).unwrap();
        mlx_rs::transforms::eval([&a32, &b32]).unwrap();
        let a_c = a32.contiguous().unwrap();
        let b_c = b32.contiguous().unwrap();
        mlx_rs::transforms::eval([&a_c, &b_c]).unwrap();
        let av = a_c.as_slice::<f32>();
        let bv = b_c.as_slice::<f32>();
        av.iter()
            .zip(bv.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn mma2big_matches_quantized_matmul_bf16() {
        let _guard = crate::mlx_test_guard();

        // Smallest valid shape: M=16, K=64, N=32, group_size=64, bits=4.
        let m: i32 = 16;
        let k: i32 = 64;
        let n: i32 = 32;
        let group_size: i32 = 64;
        let bits: i32 = 4;

        // Deterministic small inputs.
        let key_w = mlx_rs::random::key(0).unwrap();
        let key_x = mlx_rs::random::key(1).unwrap();
        let w_f32 =
            mlx_rs::random::normal::<f32>(&[n, k][..], None, None, &key_w).unwrap();
        let w_bf16 = w_f32.as_dtype(Dtype::Bfloat16).unwrap();
        let (w_packed, scales, biases) =
            quantize(&w_bf16, Some(group_size), Some(bits), None).unwrap();

        let x_f32 =
            mlx_rs::random::normal::<f32>(&[m, k][..], None, None, &key_x).unwrap();
        let x_bf16 = x_f32.as_dtype(Dtype::Bfloat16).unwrap();

        // Reference: standard quantized_matmul with transpose=true.
        let y_ref = quantized_matmul(
            &x_bf16,
            &w_packed,
            &scales,
            &biases,
            Some(true),
            Some(group_size),
            Some(bits),
            None,
        )
        .unwrap();

        // Kernel under test.
        let y_kern = verify_qmm_m16_mma2big(
            &x_bf16, &w_packed, &scales, &biases, group_size, bits,
        )
        .unwrap();

        assert_eq!(y_kern.shape(), &[16, n]);
        let diff = max_abs_diff_bf16(&y_kern, &y_ref);
        assert!(
            diff < 1e-2,
            "verify_qmm mma2big vs quantized_matmul: max diff {diff} >= 1e-2"
        );
    }

    #[test]
    fn mma2big_eligible_gating() {
        assert!(mma2big_eligible(16, 64, 32, 4));
        assert!(mma2big_eligible(8, 256, 64, 4));
        assert!(!mma2big_eligible(16, 64, 32, 8)); // bits != 4
        assert!(!mma2big_eligible(16, 64, 16, 4)); // n % 32 != 0
        assert!(!mma2big_eligible(16, 48, 32, 4)); // k % 32 != 0
        assert!(!mma2big_eligible(17, 64, 32, 4)); // m > 16
    }
}
