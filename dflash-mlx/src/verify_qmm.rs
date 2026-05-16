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
//   m4_ksplit_np (PORTED)
//     - M ∈ [1, 4], bits == 4, N % 4 == 0, K % 32 == 0.
//     - Tiny-M decode path with K split into K_PARTS segments per threadgroup
//       for better parallelism. Python: `_build_kernel_m4_ksplit_np`,
//       lines ~53-194 of `verify_qmm.py`.

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

// ---------------------------------------------------------------------------
// Metal source — m4_ksplit_np (4-bit, M=4, BN=4, K_PARTS split)
//
// Ported verbatim from Python `_build_kernel_m4_ksplit_np`, lines 53-194 of
// `verify_qmm.py`. Literal substitutions are `GS` (group_size) and `K_PARTS`
// (number of K-splits) — both baked into kernel source as constexprs.
// Cache key: (group_size, dtype, k_parts) since k_parts is a build constant.
// ---------------------------------------------------------------------------

fn build_m4_ksplit_np_source(group_size: i32, k_parts: i32) -> String {
    format!(
        r#"
        using namespace metal;
        constexpr int M = 4;
        constexpr int BN = 4;
        constexpr int K_PARTS = {kp};
        constexpr int GS = {gs};

        uint part = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint tg_n = threadgroup_position_in_grid.y;

        int K = int(K_size);
        int N = int(N_size);
        int K_by_8 = K / 8;
        int K_by_gs = K / GS;
        int n0 = int(tg_n) * BN;
        int packs_per_part = K_by_8 / K_PARTS;
        int pack_start = int(part) * packs_per_part;
        int pack_end = (int(part) == K_PARTS - 1) ? K_by_8 : pack_start + packs_per_part;

        float acc[BN * M];
        for (int i = 0; i < BN * M; ++i) {{
            acc[i] = 0.0f;
        }}

        using Vec8 = vec<T, 8>;
        const device Vec8 *xv = (const device Vec8*)x;

        for (int pack = pack_start + int(lane); pack < pack_end; pack += 32) {{
            int k_base = pack * 8;
            Vec8 v0 = xv[(0 * K + k_base) / 8];
            Vec8 v1 = xv[(1 * K + k_base) / 8];
            Vec8 v2 = xv[(2 * K + k_base) / 8];
            Vec8 v3 = xv[(3 * K + k_base) / 8];
            uint32_t p0 = w_q[(n0 + 0) * K_by_8 + pack];
            uint32_t p1 = w_q[(n0 + 1) * K_by_8 + pack];
            uint32_t p2 = w_q[(n0 + 2) * K_by_8 + pack];
            uint32_t p3 = w_q[(n0 + 3) * K_by_8 + pack];
            float s0 = float(scales[(n0 + 0) * K_by_gs + (k_base / GS)]);
            float s1 = float(scales[(n0 + 1) * K_by_gs + (k_base / GS)]);
            float s2 = float(scales[(n0 + 2) * K_by_gs + (k_base / GS)]);
            float s3 = float(scales[(n0 + 3) * K_by_gs + (k_base / GS)]);
            float b0 = float(biases[(n0 + 0) * K_by_gs + (k_base / GS)]);
            float b1 = float(biases[(n0 + 1) * K_by_gs + (k_base / GS)]);
            float b2 = float(biases[(n0 + 2) * K_by_gs + (k_base / GS)]);
            float b3 = float(biases[(n0 + 3) * K_by_gs + (k_base / GS)]);

            {{
                uint32_t packed = p0;
                float s = s0;
                float b = b0;
                for (int ki = 0; ki < 8; ++ki) {{
                    float wv = float((packed >> (ki * 4)) & 0xFu) * s + b;
                    acc[0 * M + 0] += float(v0[ki]) * wv;
                    acc[0 * M + 1] += float(v1[ki]) * wv;
                    acc[0 * M + 2] += float(v2[ki]) * wv;
                    acc[0 * M + 3] += float(v3[ki]) * wv;
                }}
            }}
            {{
                uint32_t packed = p1;
                float s = s1;
                float b = b1;
                for (int ki = 0; ki < 8; ++ki) {{
                    float wv = float((packed >> (ki * 4)) & 0xFu) * s + b;
                    acc[1 * M + 0] += float(v0[ki]) * wv;
                    acc[1 * M + 1] += float(v1[ki]) * wv;
                    acc[1 * M + 2] += float(v2[ki]) * wv;
                    acc[1 * M + 3] += float(v3[ki]) * wv;
                }}
            }}
            {{
                uint32_t packed = p2;
                float s = s2;
                float b = b2;
                for (int ki = 0; ki < 8; ++ki) {{
                    float wv = float((packed >> (ki * 4)) & 0xFu) * s + b;
                    acc[2 * M + 0] += float(v0[ki]) * wv;
                    acc[2 * M + 1] += float(v1[ki]) * wv;
                    acc[2 * M + 2] += float(v2[ki]) * wv;
                    acc[2 * M + 3] += float(v3[ki]) * wv;
                }}
            }}
            {{
                uint32_t packed = p3;
                float s = s3;
                float b = b3;
                for (int ki = 0; ki < 8; ++ki) {{
                    float wv = float((packed >> (ki * 4)) & 0xFu) * s + b;
                    acc[3 * M + 0] += float(v0[ki]) * wv;
                    acc[3 * M + 1] += float(v1[ki]) * wv;
                    acc[3 * M + 2] += float(v2[ki]) * wv;
                    acc[3 * M + 3] += float(v3[ki]) * wv;
                }}
            }}
        }}

        for (int i = 0; i < BN * M; ++i) {{
            acc[i] = simd_sum(acc[i]);
        }}

        threadgroup float partial[K_PARTS * BN * M];
        if (lane == 0) {{
            for (int i = 0; i < BN * M; ++i) {{
                partial[int(part) * BN * M + i] = acc[i];
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (part == 0 && lane < BN * M) {{
            float total = 0.0f;
            for (int p = 0; p < K_PARTS; ++p) {{
                total += partial[p * BN * M + int(lane)];
            }}
            int j = int(lane) / M;
            int row = int(lane) - j * M;
            int n_global = n0 + j;
            if (n_global < N) {{
                y[row * N + n_global] = T(total);
            }}
        }}
    "#,
        gs = group_size,
        kp = k_parts,
    )
}

fn create_m4_ksplit_np_kernel(group_size: i32, dtype: Dtype, k_parts: i32) -> KernelState {
    let source = build_m4_ksplit_np_source(group_size, k_parts);
    let name = format!(
        "verify_m4_ksplit_np_kp{}_gs{}_{}",
        k_parts,
        group_size,
        dtype_tag(dtype)
    );

    unsafe {
        let input_names = mlx_sys::mlx_vector_string_new();
        for input in ["x", "w_q", "scales", "biases", "K_size", "N_size"] {
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

fn m4_ksplit_kernel_cache() -> &'static Mutex<HashMap<(i32, Dtype, i32), KernelState>> {
    use std::sync::OnceLock;
    static CACHE: OnceLock<Mutex<HashMap<(i32, Dtype, i32), KernelState>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Eligibility for `m4_ksplit_np`. Mirrors `_m4_ksplit_np_shape`
/// (Python `verify_qmm.py` line 28-29) plus the M range used at the dispatch
/// call site. Accepts M ∈ [1, 4]; rows below 4 are zero-padded by the caller.
pub fn m4_ksplit_np_eligible(m: i32, k: i32, n: i32, bits: i32) -> bool {
    bits == 4 && m >= 1 && m <= 4 && n > 0 && k > 0 && n % 4 == 0 && k % 32 == 0
}

/// Pick the number of K-splits for a given output dim. Mirrors
/// `_m4_ksplit_np_kparts` (Python line 31-32).
pub fn m4_ksplit_np_kparts(n: i32) -> i32 {
    if n >= 4096 {
        2
    } else {
        4
    }
}

/// Run the `m4_ksplit_np` verify quantized matmul kernel.
///
/// Layout: `x` is `[M, K]` with M ∈ [1, 4] (rows beyond M are read by the
/// kernel as M is baked to 4 in source — callers passing M<4 must zero-pad
/// before calling). `w_packed` is `[N, K/8]` uint32. Output is `[M, N]`.
///
/// Equivalent to `mlx_rs::ops::quantized_matmul(x_padded, w_packed, scales,
/// biases, transpose=true, group_size, bits=4)[..M]`.
pub fn verify_qmm_m4_ksplit_np(
    x: &Array,
    w_packed: &Array,
    scales: &Array,
    biases: &Array,
    group_size: i32,
    bits: i32,
) -> Result<Array, Exception> {
    if bits != 4 {
        return Err(Exception::custom(format!(
            "verify_qmm_m4_ksplit_np: only bits==4 supported, got {bits}"
        )));
    }
    let x_shape = x.shape();
    let w_shape = w_packed.shape();
    if x_shape.len() != 2 || w_shape.len() != 2 {
        return Err(Exception::custom(format!(
            "verify_qmm_m4_ksplit_np expects 2-D x and w_packed, got x={x_shape:?} w={w_shape:?}"
        )));
    }
    let m_orig = x_shape[0];
    let k = x_shape[1];
    let n = w_shape[0];
    let k_by_8 = w_shape[1];
    if k_by_8 * 8 != k {
        return Err(Exception::custom(format!(
            "verify_qmm_m4_ksplit_np: w_packed inner dim {k_by_8} != K/8 (K={k})"
        )));
    }
    if !m4_ksplit_np_eligible(m_orig, k, n, bits) {
        return Err(Exception::custom(format!(
            "verify_qmm_m4_ksplit_np: ineligible shape m={m_orig} k={k} n={n} (need 1<=m<=4, k%32==0, n%4==0, bits==4)"
        )));
    }
    if k % group_size != 0 {
        return Err(Exception::custom(format!(
            "verify_qmm_m4_ksplit_np: K ({k}) not divisible by group_size ({group_size})"
        )));
    }

    let dtype = x.dtype();
    if dtype != Dtype::Bfloat16 && dtype != Dtype::Float16 {
        return Err(Exception::custom(format!(
            "verify_qmm_m4_ksplit_np: x dtype must be bf16 or fp16, got {:?}",
            dtype
        )));
    }
    if scales.dtype() != dtype || biases.dtype() != dtype {
        return Err(Exception::custom(format!(
            "verify_qmm_m4_ksplit_np: scales/biases dtype must match x ({:?}), got scales={:?} biases={:?}",
            dtype,
            scales.dtype(),
            biases.dtype()
        )));
    }

    // Zero-pad x up to M=4 if needed (the kernel unconditionally reads
    // 4 rows of x). Outputs corresponding to padded rows are discarded.
    let needs_pad = m_orig < 4;
    let x_padded: Array = if needs_pad {
        let pad_rows = 4 - m_orig;
        let zeros = Array::zeros::<f32>(&[pad_rows, k])?.as_dtype(dtype)?;
        mlx_rs::ops::concatenate_axis(&[x, &zeros], 0)?
    } else {
        x.clone()
    };
    let x_padded = x_padded.contiguous()?;

    let k_parts = m4_ksplit_np_kparts(n);
    let dtype_u32: u32 = dtype.into();

    {
        let mut cache = m4_ksplit_kernel_cache().lock().unwrap();
        cache
            .entry((group_size, dtype, k_parts))
            .or_insert_with(|| create_m4_ksplit_np_kernel(group_size, dtype, k_parts));
    }

    let k_scalar = Array::from_int(k);
    let n_scalar = Array::from_int(n);

    let y_4 = unsafe {
        let stream = mlx_sys::mlx_default_gpu_stream_new();
        let config = mlx_sys::mlx_fast_metal_kernel_config_new();

        let t_name = CString::new("T").unwrap();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
            config,
            t_name.as_ptr(),
            dtype_u32,
        );

        // grid: (32 * k_parts, N/4, 1); threadgroup: (32 * k_parts, 1, 1).
        // Python verify_qmm.py:945-946.
        mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, 32 * k_parts, n / 4, 1);
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, 32 * k_parts, 1, 1);

        // Output: [M=4, N] in dtype.
        let y_shape = vec![4i32, n];
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            y_shape.as_ptr(),
            y_shape.len(),
            dtype_u32,
        );

        let inputs = mlx_sys::mlx_vector_array_new();
        mlx_sys::mlx_vector_array_append_value(inputs, x_padded.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, w_packed.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, scales.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, biases.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, k_scalar.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, n_scalar.as_ptr());

        let kernel_ptr = {
            let cache = m4_ksplit_kernel_cache().lock().unwrap();
            cache.get(&(group_size, dtype, k_parts)).unwrap().kernel
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
                "verify_qmm_m4_ksplit_np Metal kernel execution failed",
            ));
        }

        let mut y_ptr = mlx_sys::mlx_array_new();
        mlx_sys::mlx_vector_array_get(&mut y_ptr, outputs, 0);

        mlx_sys::mlx_fast_metal_kernel_config_free(config);
        mlx_sys::mlx_vector_array_free(inputs);
        mlx_sys::mlx_vector_array_free(outputs);
        mlx_sys::mlx_stream_free(stream);

        Array::from_ptr(y_ptr)
    };

    if needs_pad {
        // Slice back to [m_orig, N].
        use mlx_rs::ops::indexing::IndexOp;
        Ok(y_4.index((0..m_orig, ..)))
    } else {
        Ok(y_4)
    }
}

/// Unified dispatch: route based on M to the right verify-qmm kernel.
///
/// - M ∈ [1, 4]: `verify_qmm_m4_ksplit_np` (zero-pads to 4 internally)
/// - M == 16: `verify_qmm_m16_mma2big`
/// - otherwise: returns an error (caller should fall back to stock qmm).
///
/// This is the entry point registered as the `QUANTIZED_VERIFY_QMM_HOOK` so
/// the hook signature does not need to change.
pub fn verify_qmm_dispatch(
    x: &Array,
    w_packed: &Array,
    scales: &Array,
    biases: &Array,
    group_size: i32,
    bits: i32,
) -> Result<Array, Exception> {
    let m = x.shape().first().copied().unwrap_or(0);
    let k = x.shape().get(1).copied().unwrap_or(0);
    let n = w_packed.shape().first().copied().unwrap_or(0);
    if m == 16 && mma2big_eligible(m, k, n, bits) {
        verify_qmm_m16_mma2big(x, w_packed, scales, biases, group_size, bits)
    } else if m >= 1 && m <= 4 && m4_ksplit_np_eligible(m, k, n, bits) {
        verify_qmm_m4_ksplit_np(x, w_packed, scales, biases, group_size, bits)
    } else {
        Err(Exception::custom(format!(
            "verify_qmm_dispatch: no eligible kernel for m={m} k={k} n={n} bits={bits}"
        )))
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

    /// numpy-allclose style relative tolerance check for bf16 GEMM outputs.
    /// Different summation orders (simdgroup matrix vs scalar+simd_sum) yield
    /// slightly different bf16 roundings — absolute diff scales with value
    /// magnitude. Threshold: `|a - b| <= atol + rtol * max(|a|, |b|)`.
    fn allclose_bf16(a: &Array, b: &Array, atol: f32, rtol: f32) -> (bool, f32, f32) {
        let a32 = a.as_dtype(Dtype::Float32).unwrap().contiguous().unwrap();
        let b32 = b.as_dtype(Dtype::Float32).unwrap().contiguous().unwrap();
        mlx_rs::transforms::eval([&a32, &b32]).unwrap();
        let av = a32.as_slice::<f32>();
        let bv = b32.as_slice::<f32>();
        let mut max_abs = 0.0f32;
        let mut max_rel = 0.0f32;
        let mut ok = true;
        for (x, y) in av.iter().zip(bv.iter()) {
            let d = (x - y).abs();
            let m = x.abs().max(y.abs());
            let allowed = atol + rtol * m;
            if d > allowed {
                ok = false;
            }
            max_abs = max_abs.max(d);
            if m > 1e-6 {
                max_rel = max_rel.max(d / m);
            }
        }
        (ok, max_abs, max_rel)
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
    fn m4_ksplit_np_matches_quantized_matmul_bf16() {
        let _guard = crate::mlx_test_guard();

        // M=4, K=64, N=32, group_size=64, bits=4. n=32 -> k_parts = 4.
        let m: i32 = 4;
        let k: i32 = 64;
        let n: i32 = 32;
        let group_size: i32 = 64;
        let bits: i32 = 4;

        let key_w = mlx_rs::random::key(2).unwrap();
        let key_x = mlx_rs::random::key(3).unwrap();
        let w_f32 =
            mlx_rs::random::normal::<f32>(&[n, k][..], None, None, &key_w).unwrap();
        let w_bf16 = w_f32.as_dtype(Dtype::Bfloat16).unwrap();
        let (w_packed, scales, biases) =
            quantize(&w_bf16, Some(group_size), Some(bits), None).unwrap();

        let x_f32 =
            mlx_rs::random::normal::<f32>(&[m, k][..], None, None, &key_x).unwrap();
        let x_bf16 = x_f32.as_dtype(Dtype::Bfloat16).unwrap();

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

        let y_kern = verify_qmm_m4_ksplit_np(
            &x_bf16, &w_packed, &scales, &biases, group_size, bits,
        )
        .unwrap();

        assert_eq!(y_kern.shape(), &[m, n]);
        // bf16 GEMM tolerance: the kernel uses scalar float accumulators with
        // simd_sum reduction; stock quantized_matmul uses a different
        // summation order. Accumulated bf16 rounding for K=64 random normal
        // terms is empirically up to ~0.15 absolute. Compare with mixed
        // tolerance: |a-b| <= atol + rtol * max(|a|, |b|).
        let (ok, max_abs, max_rel) = allclose_bf16(&y_kern, &y_ref, 0.2, 0.05);
        assert!(
            ok,
            "verify_qmm m4_ksplit_np vs quantized_matmul: max abs diff {max_abs}, max rel diff {max_rel} (atol=0.2, rtol=0.05)"
        );
    }

    #[test]
    fn m4_ksplit_np_matches_quantized_matmul_bf16_m2() {
        // Test M<4 padding path. M=2 caller, kernel pads to 4 then slices.
        let _guard = crate::mlx_test_guard();
        let m: i32 = 2;
        let k: i32 = 64;
        let n: i32 = 32;
        let group_size: i32 = 64;
        let bits: i32 = 4;

        let key_w = mlx_rs::random::key(4).unwrap();
        let key_x = mlx_rs::random::key(5).unwrap();
        let w_bf16 = mlx_rs::random::normal::<f32>(&[n, k][..], None, None, &key_w)
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let (w_packed, scales, biases) =
            quantize(&w_bf16, Some(group_size), Some(bits), None).unwrap();
        let x_bf16 = mlx_rs::random::normal::<f32>(&[m, k][..], None, None, &key_x)
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();

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
        let y_kern = verify_qmm_m4_ksplit_np(
            &x_bf16, &w_packed, &scales, &biases, group_size, bits,
        )
        .unwrap();
        assert_eq!(y_kern.shape(), &[m, n]);
        let (ok, max_abs, max_rel) = allclose_bf16(&y_kern, &y_ref, 0.2, 0.05);
        assert!(
            ok,
            "verify_qmm m4_ksplit_np M=2 vs quantized_matmul: max abs diff {max_abs}, max rel diff {max_rel}"
        );
    }

    #[test]
    fn m4_ksplit_np_eligible_gating() {
        assert!(m4_ksplit_np_eligible(4, 64, 32, 4));
        assert!(m4_ksplit_np_eligible(1, 64, 32, 4));
        assert!(m4_ksplit_np_eligible(3, 256, 64, 4));
        assert!(!m4_ksplit_np_eligible(4, 64, 32, 8)); // bits != 4
        assert!(!m4_ksplit_np_eligible(4, 64, 6, 4)); // n % 4 != 0
        assert!(!m4_ksplit_np_eligible(4, 48, 32, 4)); // k % 32 != 0
        assert!(!m4_ksplit_np_eligible(5, 64, 32, 4)); // m > 4
        assert!(!m4_ksplit_np_eligible(0, 64, 32, 4)); // m < 1
    }

    #[test]
    fn m4_ksplit_np_kparts_picks() {
        assert_eq!(m4_ksplit_np_kparts(32), 4);
        assert_eq!(m4_ksplit_np_kparts(1024), 4);
        assert_eq!(m4_ksplit_np_kparts(4096), 2);
        assert_eq!(m4_ksplit_np_kparts(8192), 2);
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
