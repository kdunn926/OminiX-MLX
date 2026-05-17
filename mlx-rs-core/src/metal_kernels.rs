//! Custom Metal kernels for fused operations
//!
//! Provides:
//! - fused_swiglu: 10-12x faster than separate silu + multiply (for MoE models)
//! - fused_modulate: Fused LayerNorm + modulation for DiT transformers
//! - deltanet_recurrence: GPU-side delta-rule scan for Qwen3.5/Qwen3.6 prefill
//! - kv_compact: single-dispatch interior KV-cache compaction for tree-shaped
//!   speculative decoding (DDTree)

use mlx_rs::{Array, Dtype, error::Exception};
use std::ffi::CString;
use std::sync::OnceLock;

// =============================================================================
// KV compaction kernel
// =============================================================================
//
// Used by DDTree-style speculative decoding to drop rejected tree branches
// from the interior of a KV cache buffer in a single Metal dispatch. The
// kernel writes a fresh output cache of shape `[B, H, past_length +
// keep_count, D]`:
//   positions [0..past_length)            ← copied straight from input
//   positions [past_length..pl+keep_cnt)  ← gathered from input[past_length + keep_indices[i]]
//
// This replaces the prior implementation that did `take_axis` (one Metal
// dispatch into a temp tensor) + `index_mut` (a second dispatch to copy
// back), so the per-cycle compact cost is roughly halved on long caches
// where the prefix copy dominates.
const KV_COMPACT_KERNEL_SOURCE: &str = r#"
    // Grid layout: (D * out_S, H, B) — one thread per output element.
    uint d = thread_position_in_grid.x % uint(D);
    uint t = thread_position_in_grid.x / uint(D);
    uint h = thread_position_in_grid.y;
    uint b = thread_position_in_grid.z;
    if (t >= uint(out_S)) return;

    // Source slot in the input cache.
    int src_t;
    if (int(t) < past_length) {
        src_t = int(t);
    } else {
        src_t = past_length + keep_indices[int(t) - past_length];
    }

    uint in_stride_t  = uint(D);
    uint in_stride_h  = uint(S) * in_stride_t;
    uint in_stride_b  = uint(H) * in_stride_h;
    uint out_stride_t = uint(D);
    uint out_stride_h = uint(out_S) * out_stride_t;
    uint out_stride_b = uint(H) * out_stride_h;

    uint src = b * in_stride_b  + h * in_stride_h  + uint(src_t) * in_stride_t + d;
    uint dst = b * out_stride_b + h * out_stride_h + t * out_stride_t          + d;
    out[dst] = in_buf[src];
"#;

static KV_COMPACT_KERNEL: OnceLock<MetalKernel> = OnceLock::new();

// =============================================================================
// Per-position RoPE kernel
// =============================================================================
//
// The stock `mlx::fast::rope` Metal kernel takes a single starting offset
// and rotates positions [offset, offset+1, ..., offset+L-1] sequentially.
// For DDTree's fused tree forward we need DIFFERENT positions per token
// (siblings at the same tree depth share a position), so the spike has
// been using a slow Rust path that computes cos/sin via outer product +
// generic ops.
//
// This kernel does the per-position rotation in one Metal dispatch.
// Layout: each thread computes ONE output element using the
// non-traditional rotate-half formulation (the same one MLX's
// `apply_rotary_pos_emb` uses):
//   for d in 0..half_dim:
//       angle = positions[t] * inv_freq[d]
//       out[..t, d]            = x[..t, d]            * cos(angle)
//                              - x[..t, d + half_dim] * sin(angle)
//       out[..t, d + half_dim] = x[..t, d + half_dim] * cos(angle)
//                              + x[..t, d]            * sin(angle)
const PER_POSITION_ROPE_KERNEL_SOURCE: &str = r#"
    uint d = thread_position_in_grid.x % uint(D);
    uint t = thread_position_in_grid.x / uint(D);
    uint h = thread_position_in_grid.y;
    uint b = thread_position_in_grid.z;
    if (t >= uint(L)) return;

    uint half_dim = uint(D) / 2;
    uint d_lo = (d < half_dim) ? d : (d - half_dim);

    float pos   = positions[t];
    float invf  = inv_freq[d_lo];
    float angle = pos * invf;
    float cos_v = metal::cos(angle);
    float sin_v = metal::sin(angle);

    uint stride_t = uint(D);
    uint stride_h = uint(L) * stride_t;
    uint stride_b = uint(H) * stride_h;
    uint base = b * stride_b + h * stride_h + t * stride_t;

    T x_lo = in_buf[base + d_lo];
    T x_hi = in_buf[base + d_lo + half_dim];
    T out_v;
    if (d < half_dim) {
        out_v = T(float(x_lo) * cos_v - float(x_hi) * sin_v);
    } else {
        out_v = T(float(x_hi) * cos_v + float(x_lo) * sin_v);
    }
    out[base + d] = out_v;
"#;

static PER_POSITION_ROPE_KERNEL: OnceLock<MetalKernel> = OnceLock::new();

// =============================================================================
// TurboQuant key compression / decompression kernels
// =============================================================================
//
// Port of github.com/onur-gokyildiz-bhi/tq-kv's CUDA tq_compress + matching
// decompress, sized for Gemma4-class workloads (head_dim ≤ 512, kv_heads
// up to 32). Each kernel processes one key vector per grid block.
//
// Compress pipeline (kernel `tq_compress_4bit`):
//   1. Load key vector × sign mask into threadgroup memory.
//   2. Per-token mean subtraction (parallel sum reduction).
//   3. In-place Fast Walsh-Hadamard Transform (butterfly).
//   4. Normalize by 1/sqrt(head_dim).
//   5. Compute norm + sigma = norm / sqrt(head_dim).
//   6. Quantize each coordinate by linear scan over 15 boundaries → 4-bit
//      index in [0, 15]. Pack 8 indices per u32 word.
//   7. Store (packed_indices [head_dim/8 u32], sigma f32, mean f32).
//
// Decompress pipeline (kernel `tq_decompress_4bit`):
//   1. Unpack 4-bit index → centroid * sigma.
//   2. Inverse Hadamard (= forward Hadamard, since it's self-inverse).
//   3. Add mean back, undo sign flip.
//   4. Cast to output dtype (BF16/F16/F32 via template T).
//
// Grid layout for both: x = batch * n_kv_heads * tokens, threads = 64.

const TQ_COMPRESS_4BIT_KERNEL: &str = r#"
    uint tid = thread_position_in_threadgroup.x;
    uint vec_idx = threadgroup_position_in_grid.x;
    if (vec_idx >= uint(n_vectors)) return;

    threadgroup float s_data[512]; // head_dim ≤ 512
    threadgroup float s_red[64];

    // Pre-loaded constants:
    //   boundaries[15] — Lloyd-Max boundaries for N(0,1)
    //   signs[D]       — randomized Hadamard signs
    //   keys_in[N*D]   — input keys, row-major
    //   packed_out, sigma_out, mean_out — outputs

    uint d_in = vec_idx * uint(D);

    // Step 1: load key * sign.
    for (uint i = tid; i < uint(D); i += uint(64)) {
        s_data[i] = float(keys_in[d_in + i]) * signs[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: per-token mean removal.
    float local_sum = 0.0f;
    for (uint i = tid; i < uint(D); i += uint(64)) {
        local_sum += s_data[i];
    }
    s_red[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) { s_red[tid] += s_red[tid + 32]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 16) { s_red[tid] += s_red[tid + 16]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 8)  { s_red[tid] += s_red[tid +  8]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 4)  { s_red[tid] += s_red[tid +  4]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 2)  { s_red[tid] += s_red[tid +  2]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { s_red[0] += s_red[1]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean = s_red[0] / float(D);
    for (uint i = tid; i < uint(D); i += uint(64)) {
        s_data[i] -= mean;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: in-place Walsh-Hadamard butterfly.
    for (uint step = 1u; step < uint(D); step <<= 1u) {
        for (uint i = tid; i < uint(D) / 2u; i += uint(64)) {
            uint j = (i / step) * (step * 2u) + (i % step);
            uint k = j + step;
            float a = s_data[j];
            float b = s_data[k];
            s_data[j] = a + b;
            s_data[k] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Step 4: normalize by 1/sqrt(D).
    float scale = metal::rsqrt(float(D));
    for (uint i = tid; i < uint(D); i += uint(64)) {
        s_data[i] *= scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 5: per-vector sigma = norm / sqrt(D).
    float local_sq = 0.0f;
    for (uint i = tid; i < uint(D); i += uint(64)) {
        local_sq += s_data[i] * s_data[i];
    }
    s_red[tid] = local_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) { s_red[tid] += s_red[tid + 32]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 16) { s_red[tid] += s_red[tid + 16]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 8)  { s_red[tid] += s_red[tid +  8]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 4)  { s_red[tid] += s_red[tid +  4]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 2)  { s_red[tid] += s_red[tid +  2]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { s_red[0] += s_red[1]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float norm = metal::sqrt(s_red[0]);
    float sigma = norm * metal::rsqrt(float(D));
    float inv_sigma = (sigma > 1e-10f) ? (1.0f / sigma) : 0.0f;
    if (tid == 0) {
        sigma_out[vec_idx] = sigma;
        mean_out[vec_idx]  = mean;
    }

    // Step 6: quantize + pack 8 indices into one u32.
    // packed_out is [n_vectors, D/8] u32.
    uint words_per_vec = uint(D) / 8u;
    for (uint w = tid; w < words_per_vec; w += uint(64)) {
        uint packed = 0u;
        for (uint k = 0u; k < 8u; ++k) {
            uint d = w * 8u + k;
            float v = s_data[d] * inv_sigma;
            // Linear scan over 15 boundaries — fully unrolled, hot in registers.
            uint idx = 0u;
            if (v > boundaries[0])  idx = 1u;
            if (v > boundaries[1])  idx = 2u;
            if (v > boundaries[2])  idx = 3u;
            if (v > boundaries[3])  idx = 4u;
            if (v > boundaries[4])  idx = 5u;
            if (v > boundaries[5])  idx = 6u;
            if (v > boundaries[6])  idx = 7u;
            if (v > boundaries[7])  idx = 8u;
            if (v > boundaries[8])  idx = 9u;
            if (v > boundaries[9])  idx = 10u;
            if (v > boundaries[10]) idx = 11u;
            if (v > boundaries[11]) idx = 12u;
            if (v > boundaries[12]) idx = 13u;
            if (v > boundaries[13]) idx = 14u;
            if (v > boundaries[14]) idx = 15u;
            packed |= (idx & 0xFu) << (k * 4u);
        }
        packed_out[vec_idx * words_per_vec + w] = packed;
    }
"#;

const TQ_DECOMPRESS_4BIT_KERNEL: &str = r#"
    uint tid = thread_position_in_threadgroup.x;
    uint vec_idx = threadgroup_position_in_grid.x;
    if (vec_idx >= uint(n_vectors)) return;

    threadgroup float s_data[512];

    float sigma = sigma_in[vec_idx];
    float mean  = mean_in[vec_idx];
    uint words_per_vec = uint(D) / 8u;

    // Step 1: unpack indices → centroid * sigma.
    for (uint w = tid; w < words_per_vec; w += uint(64)) {
        uint packed = packed_in[vec_idx * words_per_vec + w];
        for (uint k = 0u; k < 8u; ++k) {
            uint idx = (packed >> (k * 4u)) & 0xFu;
            s_data[w * 8u + k] = centroids[idx] * sigma;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: inverse Hadamard (self-inverse modulo normalization).
    for (uint step = 1u; step < uint(D); step <<= 1u) {
        for (uint i = tid; i < uint(D) / 2u; i += uint(64)) {
            uint j = (i / step) * (step * 2u) + (i % step);
            uint k = j + step;
            float a = s_data[j];
            float b = s_data[k];
            s_data[j] = a + b;
            s_data[k] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float scale = metal::rsqrt(float(D));
    for (uint i = tid; i < uint(D); i += uint(64)) {
        s_data[i] *= scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: add mean back, undo sign flip, write out as T.
    uint d_out = vec_idx * uint(D);
    for (uint i = tid; i < uint(D); i += uint(64)) {
        float v = (s_data[i] + mean) * signs[i];
        keys_out[d_out + i] = T(v);
    }
"#;

static TQ_COMPRESS_4BIT_KERNEL_HANDLE: OnceLock<MetalKernel> = OnceLock::new();
static TQ_DECOMPRESS_4BIT_KERNEL_HANDLE: OnceLock<MetalKernel> = OnceLock::new();

fn create_tq_compress_4bit_kernel() -> MetalKernel {
    unsafe {
        let keys_in = CString::new("keys_in").unwrap();
        let signs = CString::new("signs").unwrap();
        let boundaries = CString::new("boundaries").unwrap();
        let packed_out = CString::new("packed_out").unwrap();
        let sigma_out = CString::new("sigma_out").unwrap();
        let mean_out = CString::new("mean_out").unwrap();
        let inputs = mlx_sys::mlx_vector_string_new();
        mlx_sys::mlx_vector_string_append_value(inputs, keys_in.as_ptr());
        mlx_sys::mlx_vector_string_append_value(inputs, signs.as_ptr());
        mlx_sys::mlx_vector_string_append_value(inputs, boundaries.as_ptr());
        let outputs = mlx_sys::mlx_vector_string_new();
        mlx_sys::mlx_vector_string_append_value(outputs, packed_out.as_ptr());
        mlx_sys::mlx_vector_string_append_value(outputs, sigma_out.as_ptr());
        mlx_sys::mlx_vector_string_append_value(outputs, mean_out.as_ptr());
        let source = CString::new(TQ_COMPRESS_4BIT_KERNEL).unwrap();
        let header = CString::new("").unwrap();
        let name = CString::new("tq_compress_4bit").unwrap();
        let kernel = mlx_sys::mlx_fast_metal_kernel_new(
            name.as_ptr(),
            inputs,
            outputs,
            source.as_ptr(),
            header.as_ptr(),
            true,
            false,
        );
        MetalKernel { kernel, input_names: inputs, output_names: outputs }
    }
}

fn create_tq_decompress_4bit_kernel() -> MetalKernel {
    unsafe {
        let packed_in = CString::new("packed_in").unwrap();
        let sigma_in = CString::new("sigma_in").unwrap();
        let mean_in = CString::new("mean_in").unwrap();
        let signs = CString::new("signs").unwrap();
        let centroids = CString::new("centroids").unwrap();
        let keys_out = CString::new("keys_out").unwrap();
        let inputs = mlx_sys::mlx_vector_string_new();
        mlx_sys::mlx_vector_string_append_value(inputs, packed_in.as_ptr());
        mlx_sys::mlx_vector_string_append_value(inputs, sigma_in.as_ptr());
        mlx_sys::mlx_vector_string_append_value(inputs, mean_in.as_ptr());
        mlx_sys::mlx_vector_string_append_value(inputs, signs.as_ptr());
        mlx_sys::mlx_vector_string_append_value(inputs, centroids.as_ptr());
        let outputs = mlx_sys::mlx_vector_string_new();
        mlx_sys::mlx_vector_string_append_value(outputs, keys_out.as_ptr());
        let source = CString::new(TQ_DECOMPRESS_4BIT_KERNEL).unwrap();
        let header = CString::new("").unwrap();
        let name = CString::new("tq_decompress_4bit").unwrap();
        let kernel = mlx_sys::mlx_fast_metal_kernel_new(
            name.as_ptr(),
            inputs,
            outputs,
            source.as_ptr(),
            header.as_ptr(),
            true,
            false,
        );
        MetalKernel { kernel, input_names: inputs, output_names: outputs }
    }
}

/// Compress an `[N, D]` (or any flattened `[*, D]`) BF16/F16/F32 keys
/// tensor into TurboQuant 4-bit form. Returns
/// `(packed [N, D/8] u32, sigma [N] f32, mean [N] f32)`.
///
/// `signs` is a `[D]` f32 tensor of +/- 1 values (use
/// `turboquant::cached_signs(dim, seed)`).
pub fn tq_compress_4bit(
    keys: &Array,
    signs: &Array,
    boundaries: &Array,
) -> Result<(Array, Array, Array), Exception> {
    let shape = keys.shape();
    if shape.len() < 2 {
        return Err(Exception::custom(format!(
            "tq_compress_4bit expects rank >= 2 input [..., D], got {:?}",
            shape
        )));
    }
    let d = *shape.last().unwrap();
    if d % 8 != 0 {
        return Err(Exception::custom(format!(
            "tq_compress_4bit requires head_dim divisible by 8, got {d}"
        )));
    }
    if d > 512 {
        return Err(Exception::custom(format!(
            "tq_compress_4bit shared-mem cap is 512 floats, got D={d}"
        )));
    }
    let n_vectors: i32 = shape.iter().take(shape.len() - 1).product();
    let packed_cols = d / 8;
    let kernel = TQ_COMPRESS_4BIT_KERNEL_HANDLE.get_or_init(create_tq_compress_4bit_kernel);

    unsafe {
        let stream = mlx_sys::mlx_default_gpu_stream_new();
        let config = mlx_sys::mlx_fast_metal_kernel_config_new();
        let d_name = CString::new("D").unwrap();
        let n_name = CString::new("n_vectors").unwrap();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config, d_name.as_ptr(), d,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config, n_name.as_ptr(), n_vectors,
        );
        // Grid is in TOTAL threads (matches fused_swiglu pattern): we want
        // `n_vectors` threadgroups × 64 threads = 64 * n_vectors threads.
        mlx_sys::mlx_fast_metal_kernel_config_set_grid(
            config, 64 * n_vectors.max(1), 1, 1,
        );
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, 64, 1, 1);

        // Outputs in declaration order: packed_out (u32), sigma_out (f32), mean_out (f32).
        let packed_shape: [i32; 2] = [n_vectors, packed_cols];
        let sigma_shape: [i32; 1] = [n_vectors];
        let mean_shape: [i32; 1] = [n_vectors];
        let u32_dtype: u32 = mlx_rs::Dtype::Uint32.into();
        let f32_dtype: u32 = mlx_rs::Dtype::Float32.into();
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config, packed_shape.as_ptr(), packed_shape.len(), u32_dtype,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config, sigma_shape.as_ptr(), sigma_shape.len(), f32_dtype,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config, mean_shape.as_ptr(), mean_shape.len(), f32_dtype,
        );

        // Ensure inputs are f32 — the kernel hardcodes float reads.
        let keys_f32 = keys.as_dtype(mlx_rs::Dtype::Float32)?;
        let signs_f32 = signs.as_dtype(mlx_rs::Dtype::Float32)?;
        let boundaries_f32 = boundaries.as_dtype(mlx_rs::Dtype::Float32)?;
        let inputs = mlx_sys::mlx_vector_array_new();
        mlx_sys::mlx_vector_array_append_value(inputs, keys_f32.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, signs_f32.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, boundaries_f32.as_ptr());

        let mut outputs = mlx_sys::mlx_vector_array_new();
        let ret = mlx_sys::mlx_fast_metal_kernel_apply(
            &mut outputs, kernel.kernel, inputs, config, stream,
        );
        if ret != 0 {
            mlx_sys::mlx_fast_metal_kernel_config_free(config);
            mlx_sys::mlx_vector_array_free(inputs);
            mlx_sys::mlx_vector_array_free(outputs);
            mlx_sys::mlx_stream_free(stream);
            return Err(Exception::custom("tq_compress_4bit kernel failed"));
        }
        let mut packed = mlx_sys::mlx_array_new();
        let mut sigma = mlx_sys::mlx_array_new();
        let mut mean = mlx_sys::mlx_array_new();
        mlx_sys::mlx_vector_array_get(&mut packed, outputs, 0);
        mlx_sys::mlx_vector_array_get(&mut sigma, outputs, 1);
        mlx_sys::mlx_vector_array_get(&mut mean, outputs, 2);
        mlx_sys::mlx_fast_metal_kernel_config_free(config);
        mlx_sys::mlx_vector_array_free(inputs);
        mlx_sys::mlx_vector_array_free(outputs);
        mlx_sys::mlx_stream_free(stream);
        Ok((
            Array::from_ptr(packed),
            Array::from_ptr(sigma),
            Array::from_ptr(mean),
        ))
    }
}

/// Decompress a TurboQuant 4-bit key block into a `[N, D]` tensor of
/// dtype `out_dtype` (BF16/F16/F32). Shape `out_shape` should match the
/// original shape passed to `tq_compress_4bit`.
pub fn tq_decompress_4bit(
    packed: &Array,
    sigma: &Array,
    mean: &Array,
    signs: &Array,
    centroids: &Array,
    out_shape: &[i32],
    out_dtype: mlx_rs::Dtype,
) -> Result<Array, Exception> {
    if out_shape.len() < 2 {
        return Err(Exception::custom(format!(
            "tq_decompress_4bit out_shape must be rank >= 2, got {out_shape:?}",
        )));
    }
    let d = *out_shape.last().unwrap();
    if d % 8 != 0 || d > 512 {
        return Err(Exception::custom(format!(
            "tq_decompress_4bit: D must be ≤ 512 and div by 8, got {d}"
        )));
    }
    let n_vectors: i32 = out_shape.iter().take(out_shape.len() - 1).product();
    let kernel = TQ_DECOMPRESS_4BIT_KERNEL_HANDLE.get_or_init(create_tq_decompress_4bit_kernel);
    let dtype: u32 = out_dtype.into();

    unsafe {
        let stream = mlx_sys::mlx_default_gpu_stream_new();
        let config = mlx_sys::mlx_fast_metal_kernel_config_new();
        let type_name = CString::new("T").unwrap();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
            config, type_name.as_ptr(), dtype,
        );
        let d_name = CString::new("D").unwrap();
        let n_name = CString::new("n_vectors").unwrap();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config, d_name.as_ptr(), d,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config, n_name.as_ptr(), n_vectors,
        );
        mlx_sys::mlx_fast_metal_kernel_config_set_grid(
            config, 64 * n_vectors.max(1), 1, 1,
        );
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, 64, 1, 1);
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config, out_shape.as_ptr(), out_shape.len(), dtype,
        );

        let signs_f32 = signs.as_dtype(mlx_rs::Dtype::Float32)?;
        let centroids_f32 = centroids.as_dtype(mlx_rs::Dtype::Float32)?;
        let sigma_f32 = sigma.as_dtype(mlx_rs::Dtype::Float32)?;
        let mean_f32 = mean.as_dtype(mlx_rs::Dtype::Float32)?;
        let inputs = mlx_sys::mlx_vector_array_new();
        mlx_sys::mlx_vector_array_append_value(inputs, packed.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, sigma_f32.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, mean_f32.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, signs_f32.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, centroids_f32.as_ptr());

        let mut outputs = mlx_sys::mlx_vector_array_new();
        let ret = mlx_sys::mlx_fast_metal_kernel_apply(
            &mut outputs, kernel.kernel, inputs, config, stream,
        );
        if ret != 0 {
            mlx_sys::mlx_fast_metal_kernel_config_free(config);
            mlx_sys::mlx_vector_array_free(inputs);
            mlx_sys::mlx_vector_array_free(outputs);
            mlx_sys::mlx_stream_free(stream);
            return Err(Exception::custom("tq_decompress_4bit kernel failed"));
        }
        let mut result = mlx_sys::mlx_array_new();
        mlx_sys::mlx_vector_array_get(&mut result, outputs, 0);
        mlx_sys::mlx_fast_metal_kernel_config_free(config);
        mlx_sys::mlx_vector_array_free(inputs);
        mlx_sys::mlx_vector_array_free(outputs);
        mlx_sys::mlx_stream_free(stream);
        Ok(Array::from_ptr(result))
    }
}

fn create_per_position_rope_kernel() -> MetalKernel {
    unsafe {
        let in_name = CString::new("in_buf").unwrap();
        let inv_name = CString::new("inv_freq").unwrap();
        let pos_name = CString::new("positions").unwrap();
        let out_name = CString::new("out").unwrap();
        let inputs = mlx_sys::mlx_vector_string_new();
        mlx_sys::mlx_vector_string_append_value(inputs, in_name.as_ptr());
        mlx_sys::mlx_vector_string_append_value(inputs, inv_name.as_ptr());
        mlx_sys::mlx_vector_string_append_value(inputs, pos_name.as_ptr());
        let outputs = mlx_sys::mlx_vector_string_new();
        mlx_sys::mlx_vector_string_append_value(outputs, out_name.as_ptr());
        let source = CString::new(PER_POSITION_ROPE_KERNEL_SOURCE).unwrap();
        let header = CString::new("").unwrap();
        let name = CString::new("per_position_rope").unwrap();
        let kernel = mlx_sys::mlx_fast_metal_kernel_new(
            name.as_ptr(),
            inputs,
            outputs,
            source.as_ptr(),
            header.as_ptr(),
            true,
            false,
        );
        MetalKernel { kernel, input_names: inputs, output_names: outputs }
    }
}

/// Apply rotary positional embedding to `x` of shape `[B, H, L, D]` using
/// a per-token `positions[L]` array and a `inv_freq[D/2]` table (both
/// f32). Non-traditional (rotate-half) variant. Returns a fresh `[B, H,
/// L, D]` of the same dtype as `x`.
///
/// Single Metal dispatch — replaces the prior outer-product + cos + sin
/// + rotate-half chain (~6 MLX kernels) that the spike used.
pub fn per_position_rope(
    x: &Array,
    positions: &Array,
    inv_freq: &Array,
) -> Result<Array, Exception> {
    let shape = x.shape();
    if shape.len() != 4 {
        return Err(Exception::custom(format!(
            "per_position_rope expects [B,H,L,D], got {:?}",
            shape
        )));
    }
    let b = shape[0];
    let h = shape[1];
    let l = shape[2];
    let d = shape[3];
    if d % 2 != 0 {
        return Err(Exception::custom(format!(
            "per_position_rope requires even head_dim, got D={d}"
        )));
    }
    let dtype: u32 = x.dtype().into();
    let positions = positions.as_dtype(Dtype::Float32)?;
    let inv_freq = inv_freq.as_dtype(Dtype::Float32)?;

    let kernel = PER_POSITION_ROPE_KERNEL.get_or_init(create_per_position_rope_kernel);
    unsafe {
        let stream = mlx_sys::mlx_default_gpu_stream_new();
        let config = mlx_sys::mlx_fast_metal_kernel_config_new();
        let type_name = CString::new("T").unwrap();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
            config, type_name.as_ptr(), dtype,
        );
        for (name, value) in [("D", d), ("H", h), ("L", l)] {
            let cname = CString::new(name).unwrap();
            mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
                config, cname.as_ptr(), value,
            );
        }
        mlx_sys::mlx_fast_metal_kernel_config_set_grid(
            config,
            (d * l).max(1),
            h.max(1),
            b.max(1),
        );
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, 64, 1, 1);
        let out_shape: [i32; 4] = [b, h, l, d];
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            out_shape.as_ptr(),
            out_shape.len(),
            dtype,
        );
        let inputs = mlx_sys::mlx_vector_array_new();
        mlx_sys::mlx_vector_array_append_value(inputs, x.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, inv_freq.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, positions.as_ptr());
        let mut outputs = mlx_sys::mlx_vector_array_new();
        let ret = mlx_sys::mlx_fast_metal_kernel_apply(
            &mut outputs, kernel.kernel, inputs, config, stream,
        );
        if ret != 0 {
            mlx_sys::mlx_fast_metal_kernel_config_free(config);
            mlx_sys::mlx_vector_array_free(inputs);
            mlx_sys::mlx_vector_array_free(outputs);
            mlx_sys::mlx_stream_free(stream);
            return Err(Exception::custom("per_position_rope kernel execution failed"));
        }
        let mut result = mlx_sys::mlx_array_new();
        mlx_sys::mlx_vector_array_get(&mut result, outputs, 0);
        mlx_sys::mlx_fast_metal_kernel_config_free(config);
        mlx_sys::mlx_vector_array_free(inputs);
        mlx_sys::mlx_vector_array_free(outputs);
        mlx_sys::mlx_stream_free(stream);
        Ok(Array::from_ptr(result))
    }
}

fn create_kv_compact_kernel() -> MetalKernel {
    unsafe {
        let in_name = CString::new("in_buf").unwrap();
        let idx_name = CString::new("keep_indices").unwrap();
        let out_name = CString::new("out").unwrap();
        let inputs = mlx_sys::mlx_vector_string_new();
        mlx_sys::mlx_vector_string_append_value(inputs, in_name.as_ptr());
        mlx_sys::mlx_vector_string_append_value(inputs, idx_name.as_ptr());
        let outputs = mlx_sys::mlx_vector_string_new();
        mlx_sys::mlx_vector_string_append_value(outputs, out_name.as_ptr());
        let source = CString::new(KV_COMPACT_KERNEL_SOURCE).unwrap();
        let header = CString::new("").unwrap();
        let name = CString::new("kv_compact_gather").unwrap();
        let kernel = mlx_sys::mlx_fast_metal_kernel_new(
            name.as_ptr(),
            inputs,
            outputs,
            source.as_ptr(),
            header.as_ptr(),
            true,
            false,
        );
        MetalKernel { kernel, input_names: inputs, output_names: outputs }
    }
}

/// Compact a `[B, H, S, D]` cache buffer by keeping only the slots in
/// `keep_indices` (offsets into the appended window starting at
/// `past_length`). Returns a fresh `[B, H, past_length + keep_count, D]`
/// buffer in ONE Metal dispatch.
///
/// Replaces the prior stock-MLX path which dispatched `take_axis` into a
/// temp tensor (one Metal kernel) and then `index_mut` to scatter back
/// into the cache buffer (a second Metal kernel) — two dispatches, two
/// allocations. The fused kernel does both regions (prefix copy + window
/// gather) in a single grid.
pub fn kv_compact(
    in_buf: &Array,
    past_length: i32,
    keep_indices: &Array,
) -> Result<Array, Exception> {
    let shape = in_buf.shape();
    if shape.len() != 4 {
        return Err(Exception::custom(format!(
            "kv_compact expects [B,H,S,D], got {:?}",
            shape
        )));
    }
    let b = shape[0];
    let h = shape[1];
    let s = shape[2];
    let d = shape[3];
    let keep_count = keep_indices.shape()[0];
    let out_s = past_length + keep_count;
    let dtype: u32 = in_buf.dtype().into();
    let keep_indices = if keep_indices.dtype() != Dtype::Int32 {
        keep_indices.as_dtype(Dtype::Int32)?
    } else {
        keep_indices.clone()
    };

    let kernel = KV_COMPACT_KERNEL.get_or_init(create_kv_compact_kernel);
    unsafe {
        let stream = mlx_sys::mlx_default_gpu_stream_new();
        let config = mlx_sys::mlx_fast_metal_kernel_config_new();

        let type_name = CString::new("T").unwrap();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
            config, type_name.as_ptr(), dtype,
        );
        for (name, value) in [
            ("D", d),
            ("H", h),
            ("S", s),
            ("out_S", out_s),
            ("past_length", past_length),
            ("keep_count", keep_count),
        ] {
            let cname = CString::new(name).unwrap();
            mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
                config, cname.as_ptr(), value,
            );
        }
        mlx_sys::mlx_fast_metal_kernel_config_set_grid(
            config,
            (d * out_s).max(1),
            h.max(1),
            b.max(1),
        );
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, 64, 1, 1);

        let out_shape: [i32; 4] = [b, h, out_s, d];
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            out_shape.as_ptr(),
            out_shape.len(),
            dtype,
        );

        let inputs = mlx_sys::mlx_vector_array_new();
        mlx_sys::mlx_vector_array_append_value(inputs, in_buf.as_ptr());
        // Sentinel keep_indices if empty (kernel still receives a valid pointer).
        let keep_for_kernel = if keep_count == 0 {
            mlx_rs::Array::from_slice::<i32>(&[0i32], &[1])
        } else {
            keep_indices
        };
        mlx_sys::mlx_vector_array_append_value(inputs, keep_for_kernel.as_ptr());

        let mut outputs = mlx_sys::mlx_vector_array_new();
        let ret = mlx_sys::mlx_fast_metal_kernel_apply(
            &mut outputs, kernel.kernel, inputs, config, stream,
        );
        if ret != 0 {
            mlx_sys::mlx_fast_metal_kernel_config_free(config);
            mlx_sys::mlx_vector_array_free(inputs);
            mlx_sys::mlx_vector_array_free(outputs);
            mlx_sys::mlx_stream_free(stream);
            return Err(Exception::custom("kv_compact kernel execution failed"));
        }

        let mut result = mlx_sys::mlx_array_new();
        mlx_sys::mlx_vector_array_get(&mut result, outputs, 0);

        mlx_sys::mlx_fast_metal_kernel_config_free(config);
        mlx_sys::mlx_vector_array_free(inputs);
        mlx_sys::mlx_vector_array_free(outputs);
        mlx_sys::mlx_stream_free(stream);

        Ok(Array::from_ptr(result))
    }
}

const SWIGLU_KERNEL_SOURCE: &str = r#"
    uint elem = thread_position_in_grid.x;
    T gate_val = gate[elem];
    T x_val = x[elem];
    // silu(gate) = gate / (1 + exp(-gate))
    T silu_gate = gate_val / (T(1) + metal::exp(-gate_val));
    out[elem] = silu_gate * x_val;
"#;

// Fused LayerNorm + Modulation kernel for DiT transformers
// Computes: (1 + scale) * LayerNorm(x) + shift
// where LayerNorm has no learnable parameters (elementwise_affine=False)
//
// This kernel uses parallel reduction within each threadgroup to compute
// mean and variance efficiently.
//
// IMPORTANT: Always launch exactly 256 threads per threadgroup for correct reduction.
const MODULATE_KERNEL_SOURCE: &str = r#"
    // Each threadgroup handles one row (one position in the sequence)
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    constexpr uint THREADS = 256;

    // Shared memory for parallel reduction
    threadgroup T shared_sum[256];
    threadgroup T shared_sum_sq[256];

    // Initialize shared memory to 0 (all threads do this)
    shared_sum[tid] = T(0);
    shared_sum_sq[tid] = T(0);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each thread accumulates partial sums over its portion of the row
    T local_sum = T(0);
    T local_sum_sq = T(0);

    uint base = row * dim;
    for (uint i = tid; i < dim; i += THREADS) {
        T val = x[base + i];
        local_sum += val;
        local_sum_sq += val * val;
    }

    // Store to shared memory
    shared_sum[tid] = local_sum;
    shared_sum_sq[tid] = local_sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction - fully unrolled for 256 threads
    if (tid < 128) { shared_sum[tid] += shared_sum[tid + 128]; shared_sum_sq[tid] += shared_sum_sq[tid + 128]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 64) { shared_sum[tid] += shared_sum[tid + 64]; shared_sum_sq[tid] += shared_sum_sq[tid + 64]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) { shared_sum[tid] += shared_sum[tid + 32]; shared_sum_sq[tid] += shared_sum_sq[tid + 32]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 16) { shared_sum[tid] += shared_sum[tid + 16]; shared_sum_sq[tid] += shared_sum_sq[tid + 16]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 8) { shared_sum[tid] += shared_sum[tid + 8]; shared_sum_sq[tid] += shared_sum_sq[tid + 8]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 4) { shared_sum[tid] += shared_sum[tid + 4]; shared_sum_sq[tid] += shared_sum_sq[tid + 4]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 2) { shared_sum[tid] += shared_sum[tid + 2]; shared_sum_sq[tid] += shared_sum_sq[tid + 2]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { shared_sum[0] += shared_sum[1]; shared_sum_sq[0] += shared_sum_sq[1]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ALL threads read the final sums and compute mean/inv_std locally
    // (avoids issues with scalar threadgroup variable broadcast)
    T sum_val = shared_sum[0];
    T sum_sq_val = shared_sum_sq[0];
    T mean = sum_val / T(dim);
    T var = sum_sq_val / T(dim) - mean * mean;
    // Clamp variance to avoid NaN from numerical precision issues
    var = max(var, T(0));
    T inv_std = rsqrt(var + T(1e-6));

    // Apply normalization and modulation: (1 + scale) * normalized + shift
    for (uint i = tid; i < dim; i += THREADS) {
        T normalized = (x[base + i] - mean) * inv_std;
        T scale_val = scale[i];
        T shift_val = shift[i];
        out[base + i] = (T(1) + scale_val) * normalized + shift_val;
    }
"#;

// =============================================================================
// DeltaNet recurrence kernel
// =============================================================================
//
// Runs the per-timestep delta-rule scan for Gated DeltaNet entirely on the GPU
// in one dispatch, replacing ~13 MLX ops per timestep with a single Metal
// kernel launch per (B, H, V_pos) work item.
//
// Threadgroup layout
// ------------------
//   grid = (V, H, B)         one threadgroup per (batch, head, v-position)
//   group = (32, 1, 1)       exactly one SIMD-group per threadgroup
//
// Each thread inside a group owns K/32 consecutive K-indices (e.g. K=128 →
// 4 K-entries per thread). The full per-(b,h,v) state column lives in thread
// registers across the 32 threads; the only cross-thread communication is a
// `simd_sum` per timestep to fold the K-dimension dot products. State never
// touches global memory between timesteps.
//
// Algorithm (per threadgroup, per timestep t)
//   1. Load q_t[k_range], k_t[k_range] for this thread's K-stride.
//      Load v_t[v_pos], decay_t, beta_t (broadcast — every thread reads same).
//   2. local_kv  = sum_{k in stride} state[k] * k_t[k]      // per-thread
//   3. kv_mem    = simd_sum(local_kv)                       // reduces across 32 threads
//   4. delta     = (v_t[v_pos] - kv_mem) * beta_t           // identical in every thread
//   5. state[k]  = state[k] * decay_t + k_t[k] * delta      // per-thread update
//   6. local_out = sum_{k in stride} state[k] * q_t[k]      // per-thread
//   7. output_t  = simd_sum(local_out)                      // identical in every thread
//   8. thread 0 of the simdgroup writes output[b, h, t, v_pos] to global mem
//
// After the t-loop, every thread writes its K-stride of `state` back to
// `state_out[b, h, *, v_pos]`.
//
// All arrays are row-contiguous Float32. Shape contract:
//   q, k     : [B, H, L, K]
//   v        : [B, H, L, V]
//   decay,
//   beta     : [B, H, L]
//   state_in : [B, H, K, V]
//   output   : [B, H, L, V]   (kernel output 0)
//   state_out: [B, H, K, V]   (kernel output 1)
//
// Template arguments: T (dtype), K (key head_dim), V (value head_dim), L (seq len).
const DELTANET_RECURRENCE_KERNEL_SOURCE: &str = r#"
    constexpr uint THREADS = 32;
    constexpr uint K_PER_THREAD = K / THREADS;

    uint v_pos = threadgroup_position_in_grid.x;
    uint h     = threadgroup_position_in_grid.y;
    uint b     = threadgroup_position_in_grid.z;
    uint tid   = thread_position_in_threadgroup.x;
    uint k_base = tid * K_PER_THREAD;

    uint bh_seq_base = (b * H + h) * L;
    uint state_bh_base = ((b * H) + h) * K * V;
    uint v_base_bh_l = bh_seq_base * V;

    // Load this thread's stripe of state[b, h, k_base..k_base+K_PER_THREAD, v_pos]
    T state_local[K_PER_THREAD];
    for (uint kk = 0; kk < K_PER_THREAD; ++kk) {
        uint k_idx = k_base + kk;
        state_local[kk] = state_in[state_bh_base + k_idx * V + v_pos];
    }

    for (uint t = 0; t < L; ++t) {
        // Per-timestep scalars (broadcast across simdgroup — every thread reads same)
        T decay_t = decay[bh_seq_base + t];
        T beta_t  = beta[bh_seq_base + t];
        T v_t     = v_in[v_base_bh_l + t * V + v_pos];

        // Per-thread stripe of q_t[k], k_t[k]
        T q_local[K_PER_THREAD];
        T k_local[K_PER_THREAD];
        uint qk_base = bh_seq_base * K + t * K + k_base;
        for (uint kk = 0; kk < K_PER_THREAD; ++kk) {
            q_local[kk] = q_in[qk_base + kk];
            k_local[kk] = k_in[qk_base + kk];
        }

        // 1) Apply decay first, then compute kv_mem = sum_k (decay*state[k]) * k_t[k].
        //    The correct delta rule retrieves from the already-decayed state so that
        //    delta = v_t - (alpha * S_{t-1})^T k_t.  Using the undecayed state here
        //    produces wrong corrections and causes the model to collapse to EOS.
        T local_kv = T(0);
        for (uint kk = 0; kk < K_PER_THREAD; ++kk) {
            state_local[kk] *= decay_t;
            local_kv += state_local[kk] * k_local[kk];
        }
        T kv_mem = simd_sum(local_kv);

        T delta = (v_t - kv_mem) * beta_t;

        // 2) state[k] += k_t[k] * delta  (decay already applied above)
        // 3) output local accumulator: sum_k state[k] * q_t[k] (post-update)
        T local_out = T(0);
        for (uint kk = 0; kk < K_PER_THREAD; ++kk) {
            state_local[kk] += k_local[kk] * delta;
            local_out += state_local[kk] * q_local[kk];
        }
        T out_val = simd_sum(local_out);

        // Single lane writes the output to avoid 32 redundant stores.
        if (tid == 0) {
            output[v_base_bh_l + t * V + v_pos] = out_val;
        }
    }

    // Write back this thread's stripe of the post-scan state.
    for (uint kk = 0; kk < K_PER_THREAD; ++kk) {
        uint k_idx = k_base + kk;
        state_out[state_bh_base + k_idx * V + v_pos] = state_local[kk];
    }
"#;

// =============================================================================
// DeltaNet recurrence with innovation tape (for speculative-decoding rollback)
// =============================================================================
//
// Same delta-rule scan as `deltanet_recurrence` (decayed-state retrieval) but
// also records the per-timestep `delta` ("innovation") tensor to global memory
// so that the recurrent state can be cheaply rolled forward from a snapshot
// after the verifier accepts a prefix of drafted tokens.
//
// Shape contract (identical to `deltanet_recurrence`):
//   q, k     : [B, H, L, K]
//   v        : [B, H, L, V]
//   decay,
//   beta     : [B, H, L]
//   state_in : [B, H, K, V]
//   output   : [B, H, L, V]   (kernel output 0)
//   state_out: [B, H, K, V]   (kernel output 1)
//   tape     : [B, H, L, V]   (kernel output 2 — `delta` per (b,h,t,v_pos))
const DELTANET_WITH_TAPE_KERNEL_SOURCE: &str = r#"
    constexpr uint THREADS = 32;
    constexpr uint K_PER_THREAD = K / THREADS;

    uint v_pos = threadgroup_position_in_grid.x;
    uint h     = threadgroup_position_in_grid.y;
    uint b     = threadgroup_position_in_grid.z;
    uint tid   = thread_position_in_threadgroup.x;
    uint k_base = tid * K_PER_THREAD;

    uint bh_seq_base = (b * H + h) * L;
    uint state_bh_base = ((b * H) + h) * K * V;
    uint v_base_bh_l = bh_seq_base * V;

    T state_local[K_PER_THREAD];
    for (uint kk = 0; kk < K_PER_THREAD; ++kk) {
        uint k_idx = k_base + kk;
        state_local[kk] = state_in[state_bh_base + k_idx * V + v_pos];
    }

    for (uint t = 0; t < L; ++t) {
        T decay_t = decay[bh_seq_base + t];
        T beta_t  = beta[bh_seq_base + t];
        T v_t     = v_in[v_base_bh_l + t * V + v_pos];

        T q_local[K_PER_THREAD];
        T k_local[K_PER_THREAD];
        uint qk_base = bh_seq_base * K + t * K + k_base;
        for (uint kk = 0; kk < K_PER_THREAD; ++kk) {
            q_local[kk] = q_in[qk_base + kk];
            k_local[kk] = k_in[qk_base + kk];
        }

        T local_kv = T(0);
        for (uint kk = 0; kk < K_PER_THREAD; ++kk) {
            state_local[kk] *= decay_t;
            local_kv += state_local[kk] * k_local[kk];
        }
        T kv_mem = simd_sum(local_kv);

        T delta = (v_t - kv_mem) * beta_t;

        T local_out = T(0);
        for (uint kk = 0; kk < K_PER_THREAD; ++kk) {
            state_local[kk] += k_local[kk] * delta;
            local_out += state_local[kk] * q_local[kk];
        }
        T out_val = simd_sum(local_out);

        if (tid == 0) {
            output[v_base_bh_l + t * V + v_pos] = out_val;
            tape[v_base_bh_l + t * V + v_pos] = delta;
        }
    }

    for (uint kk = 0; kk < K_PER_THREAD; ++kk) {
        uint k_idx = k_base + kk;
        state_out[state_bh_base + k_idx * V + v_pos] = state_local[kk];
    }
"#;

// Replay only the state update (no output) using a recorded innovation tape.
// Used by speculative-decoding rollback to advance state by a prefix of
// accepted tokens without re-deriving deltas.
//
// Inputs:
//   tape  : [B, H, L, V]
//   k     : [B, H, L, K]
//   decay : [B, H, L]
//   state_in : [B, H, K, V]
// Output:
//   state_out : [B, H, K, V]
const DELTANET_TAPE_REPLAY_KERNEL_SOURCE: &str = r#"
    constexpr uint THREADS = 32;
    constexpr uint K_PER_THREAD = K / THREADS;

    uint v_pos = threadgroup_position_in_grid.x;
    uint h     = threadgroup_position_in_grid.y;
    uint b     = threadgroup_position_in_grid.z;
    uint tid   = thread_position_in_threadgroup.x;
    uint k_base = tid * K_PER_THREAD;

    uint bh_seq_base = (b * H + h) * L;
    uint state_bh_base = ((b * H) + h) * K * V;
    uint v_base_bh_l = bh_seq_base * V;

    T state_local[K_PER_THREAD];
    for (uint kk = 0; kk < K_PER_THREAD; ++kk) {
        uint k_idx = k_base + kk;
        state_local[kk] = state_in[state_bh_base + k_idx * V + v_pos];
    }

    for (uint t = 0; t < L; ++t) {
        T decay_t = decay[bh_seq_base + t];
        T delta_t = tape[v_base_bh_l + t * V + v_pos];

        T k_local[K_PER_THREAD];
        uint qk_base = bh_seq_base * K + t * K + k_base;
        for (uint kk = 0; kk < K_PER_THREAD; ++kk) {
            k_local[kk] = k_in[qk_base + kk];
        }

        for (uint kk = 0; kk < K_PER_THREAD; ++kk) {
            state_local[kk] = state_local[kk] * decay_t + k_local[kk] * delta_t;
        }
    }

    for (uint kk = 0; kk < K_PER_THREAD; ++kk) {
        uint k_idx = k_base + kk;
        state_out[state_bh_base + k_idx * V + v_pos] = state_local[kk];
    }
"#;

static SWIGLU_KERNEL: OnceLock<MetalKernel> = OnceLock::new();
static MODULATE_KERNEL: OnceLock<MetalKernel> = OnceLock::new();
static DELTANET_KERNEL: OnceLock<MetalKernel> = OnceLock::new();
static DELTANET_WITH_TAPE_KERNEL: OnceLock<MetalKernel> = OnceLock::new();
static DELTANET_TAPE_REPLAY_KERNEL: OnceLock<MetalKernel> = OnceLock::new();

struct MetalKernel {
    kernel: mlx_sys::mlx_fast_metal_kernel,
    input_names: mlx_sys::mlx_vector_string,
    output_names: mlx_sys::mlx_vector_string,
}

unsafe impl Send for MetalKernel {}
unsafe impl Sync for MetalKernel {}

impl Drop for MetalKernel {
    fn drop(&mut self) {
        unsafe {
            mlx_sys::mlx_fast_metal_kernel_free(self.kernel);
            mlx_sys::mlx_vector_string_free(self.input_names);
            mlx_sys::mlx_vector_string_free(self.output_names);
        }
    }
}

fn create_swiglu_kernel() -> MetalKernel {
    unsafe {
        let x_name = CString::new("x").unwrap();
        let gate_name = CString::new("gate").unwrap();
        let out_name = CString::new("out").unwrap();

        let input_names = mlx_sys::mlx_vector_string_new();
        mlx_sys::mlx_vector_string_append_value(input_names, x_name.as_ptr());
        mlx_sys::mlx_vector_string_append_value(input_names, gate_name.as_ptr());

        let output_names = mlx_sys::mlx_vector_string_new();
        mlx_sys::mlx_vector_string_append_value(output_names, out_name.as_ptr());

        let source = CString::new(SWIGLU_KERNEL_SOURCE).unwrap();
        let header = CString::new("").unwrap();
        let name = CString::new("fused_swiglu").unwrap();

        let kernel = mlx_sys::mlx_fast_metal_kernel_new(
            name.as_ptr(),
            input_names,
            output_names,
            source.as_ptr(),
            header.as_ptr(),
            true,
            false,
        );

        MetalKernel { kernel, input_names, output_names }
    }
}

fn create_deltanet_kernel() -> MetalKernel {
    unsafe {
        let q_name = CString::new("q_in").unwrap();
        let k_name = CString::new("k_in").unwrap();
        let v_name = CString::new("v_in").unwrap();
        let decay_name = CString::new("decay").unwrap();
        let beta_name = CString::new("beta").unwrap();
        let state_in_name = CString::new("state_in").unwrap();

        let output_name = CString::new("output").unwrap();
        let state_out_name = CString::new("state_out").unwrap();

        let input_names = mlx_sys::mlx_vector_string_new();
        mlx_sys::mlx_vector_string_append_value(input_names, q_name.as_ptr());
        mlx_sys::mlx_vector_string_append_value(input_names, k_name.as_ptr());
        mlx_sys::mlx_vector_string_append_value(input_names, v_name.as_ptr());
        mlx_sys::mlx_vector_string_append_value(input_names, decay_name.as_ptr());
        mlx_sys::mlx_vector_string_append_value(input_names, beta_name.as_ptr());
        mlx_sys::mlx_vector_string_append_value(input_names, state_in_name.as_ptr());

        let output_names = mlx_sys::mlx_vector_string_new();
        mlx_sys::mlx_vector_string_append_value(output_names, output_name.as_ptr());
        mlx_sys::mlx_vector_string_append_value(output_names, state_out_name.as_ptr());

        let source = CString::new(DELTANET_RECURRENCE_KERNEL_SOURCE).unwrap();
        let header = CString::new("").unwrap();
        let name = CString::new("deltanet_recurrence").unwrap();

        let kernel = mlx_sys::mlx_fast_metal_kernel_new(
            name.as_ptr(),
            input_names,
            output_names,
            source.as_ptr(),
            header.as_ptr(),
            true,  // ensure_row_contiguous
            false, // atomic_outputs
        );

        MetalKernel {
            kernel,
            input_names,
            output_names,
        }
    }
}

fn create_modulate_kernel() -> MetalKernel {
    unsafe {
        let x_name = CString::new("x").unwrap();
        let scale_name = CString::new("scale").unwrap();
        let shift_name = CString::new("shift").unwrap();
        let out_name = CString::new("out").unwrap();

        let input_names = mlx_sys::mlx_vector_string_new();
        mlx_sys::mlx_vector_string_append_value(input_names, x_name.as_ptr());
        mlx_sys::mlx_vector_string_append_value(input_names, scale_name.as_ptr());
        mlx_sys::mlx_vector_string_append_value(input_names, shift_name.as_ptr());

        let output_names = mlx_sys::mlx_vector_string_new();
        mlx_sys::mlx_vector_string_append_value(output_names, out_name.as_ptr());

        let source = CString::new(MODULATE_KERNEL_SOURCE).unwrap();
        let header = CString::new("").unwrap();
        let name = CString::new("fused_modulate").unwrap();

        let kernel = mlx_sys::mlx_fast_metal_kernel_new(
            name.as_ptr(),
            input_names,
            output_names,
            source.as_ptr(),
            header.as_ptr(),
            true,  // ensure_row_contiguous
            false, // atomic_outputs
        );

        MetalKernel { kernel, input_names, output_names }
    }
}

/// Fused SwiGLU activation using custom Metal kernel
///
/// Computes: silu(gate) * x = (gate / (1 + exp(-gate))) * x
///
/// This is ~10-12x faster than separate silu() + multiply() calls.
/// Critical for MoE models which have many SwiGLU calls per forward pass.
pub fn fused_swiglu(x: &Array, gate: &Array) -> Result<Array, Exception> {
    let kernel = SWIGLU_KERNEL.get_or_init(create_swiglu_kernel);

    let shape = x.shape();
    let total_elements: usize = shape.iter().map(|&s| s as usize).product();
    let dtype: u32 = x.dtype().into();

    unsafe {
        let stream = mlx_sys::mlx_default_gpu_stream_new();
        let config = mlx_sys::mlx_fast_metal_kernel_config_new();

        let type_name = CString::new("T").unwrap();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
            config, type_name.as_ptr(), dtype);

        mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, total_elements as i32, 1, 1);
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1);

        let shape_i32: Vec<i32> = shape.iter().map(|&s| s as i32).collect();
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config, shape_i32.as_ptr(), shape.len(), dtype);

        let inputs = mlx_sys::mlx_vector_array_new();
        mlx_sys::mlx_vector_array_append_value(inputs, x.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, gate.as_ptr());

        let mut outputs = mlx_sys::mlx_vector_array_new();
        let ret = mlx_sys::mlx_fast_metal_kernel_apply(
            &mut outputs, kernel.kernel, inputs, config, stream);

        if ret != 0 {
            mlx_sys::mlx_fast_metal_kernel_config_free(config);
            mlx_sys::mlx_vector_array_free(inputs);
            mlx_sys::mlx_vector_array_free(outputs);
            mlx_sys::mlx_stream_free(stream);
            return Err(Exception::custom("Metal kernel execution failed"));
        }

        let mut result = mlx_sys::mlx_array_new();
        mlx_sys::mlx_vector_array_get(&mut result, outputs, 0);

        mlx_sys::mlx_fast_metal_kernel_config_free(config);
        mlx_sys::mlx_vector_array_free(inputs);
        mlx_sys::mlx_vector_array_free(outputs);
        mlx_sys::mlx_stream_free(stream);

        Ok(Array::from_ptr(result))
    }
}

/// Fused LayerNorm + Modulation using custom Metal kernel
///
/// Computes: (1 + scale) * LayerNorm(x) + shift
/// where LayerNorm has no learnable parameters (elementwise_affine=False)
///
/// This fuses 7+ operations into a single Metal kernel:
/// - mean computation
/// - variance computation
/// - normalization
/// - scale application (1 + scale)
/// - shift application
///
/// Critical for DiT (Diffusion Transformer) models which call modulate
/// 4x per block × 60 blocks × 40 forward passes = 9600 times per generation.
///
/// # Arguments
/// * `x` - Input tensor of shape [batch, seq, dim] or [seq, dim]
/// * `shift` - Shift tensor, will be flattened to [dim]
/// * `scale` - Scale tensor, will be flattened to [dim]
///
/// # Returns
/// Output tensor of same shape as `x`
pub fn fused_modulate(x: &Array, shift: &Array, scale: &Array) -> Result<Array, Exception> {
    let kernel = MODULATE_KERNEL.get_or_init(create_modulate_kernel);

    let shape = x.shape();
    if shape.len() < 2 {
        return Err(Exception::custom("fused_modulate requires at least 2D input"));
    }

    let dim = shape[shape.len() - 1] as i32;
    let num_rows: i32 = shape.iter().take(shape.len() - 1).map(|&s| s as i32).product();
    let dtype: u32 = x.dtype().into();

    // Ensure shift and scale are contiguous [dim] arrays
    // Use flatten to handle any input shape
    let shift_flat = shift.flatten(None, None)?;
    let scale_flat = scale.flatten(None, None)?;

    // Verify dimensions match
    if shift_flat.shape()[0] != dim || scale_flat.shape()[0] != dim {
        return Err(Exception::custom(format!(
            "fused_modulate: shift/scale dim {} doesn't match x dim {}",
            shift_flat.shape()[0], dim
        )));
    }

    unsafe {
        let stream = mlx_sys::mlx_default_gpu_stream_new();
        let config = mlx_sys::mlx_fast_metal_kernel_config_new();

        // Template argument for dtype
        let type_name = CString::new("T").unwrap();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
            config, type_name.as_ptr(), dtype);

        // Constant argument for dimension size
        let dim_name = CString::new("dim").unwrap();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config, dim_name.as_ptr(), dim);

        // Grid: total threads = num_rows * 256 (so we get exactly num_rows threadgroups)
        // Threadgroup: 256 threads per group for parallel reduction
        // This gives threadgroup_position_in_grid.x ranging from 0 to num_rows-1
        let total_threads = num_rows * 256;
        mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, total_threads, 1, 1);
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1);

        // Output shape same as input
        let shape_i32: Vec<i32> = shape.iter().map(|&s| s as i32).collect();
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config, shape_i32.as_ptr(), shape.len(), dtype);

        // Input arrays
        let inputs = mlx_sys::mlx_vector_array_new();
        mlx_sys::mlx_vector_array_append_value(inputs, x.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, scale_flat.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, shift_flat.as_ptr());

        let mut outputs = mlx_sys::mlx_vector_array_new();
        let ret = mlx_sys::mlx_fast_metal_kernel_apply(
            &mut outputs, kernel.kernel, inputs, config, stream);

        if ret != 0 {
            mlx_sys::mlx_fast_metal_kernel_config_free(config);
            mlx_sys::mlx_vector_array_free(inputs);
            mlx_sys::mlx_vector_array_free(outputs);
            mlx_sys::mlx_stream_free(stream);
            return Err(Exception::custom("fused_modulate Metal kernel execution failed"));
        }

        let mut result = mlx_sys::mlx_array_new();
        mlx_sys::mlx_vector_array_get(&mut result, outputs, 0);

        mlx_sys::mlx_fast_metal_kernel_config_free(config);
        mlx_sys::mlx_vector_array_free(inputs);
        mlx_sys::mlx_vector_array_free(outputs);
        mlx_sys::mlx_stream_free(stream);

        Ok(Array::from_ptr(result))
    }
}

/// Gated-DeltaNet recurrence (delta-rule) executed entirely on the GPU.
///
/// Replaces a per-timestep Rust loop of ~13 MLX ops with a single Metal
/// kernel launch. Each (batch, head, v_position) is mapped to one
/// threadgroup (32-thread SIMD-group). The KV-state column for that
/// triple is held in thread registers across the t=0..L scan; only the
/// K-dimension dot products cross threads (via `simd_sum`).
///
/// Shape contract (all row-contiguous Float32):
///   q, k     : [B, H, L, K]
///   v        : [B, H, L, V]
///   decay,
///   beta     : [B, H, L]
///   state_in : [B, H, K, V]
///
/// Returns (output [B, H, L, V], state_out [B, H, K, V]).
/// Requires K % 32 == 0.
pub fn deltanet_recurrence(
    q: &Array,
    k: &Array,
    v: &Array,
    decay: &Array,
    beta: &Array,
    state_in: &Array,
) -> Result<(Array, Array), Exception> {
    let kernel = DELTANET_KERNEL.get_or_init(create_deltanet_kernel);

    let q_shape = q.shape();
    let v_shape = v.shape();
    if q_shape.len() != 4 || v_shape.len() != 4 {
        return Err(Exception::custom(format!(
            "deltanet_recurrence: expected 4D q and v, got q={:?} v={:?}",
            q_shape, v_shape
        )));
    }
    let b = q_shape[0] as i32;
    let h = q_shape[1] as i32;
    let l = q_shape[2] as i32;
    let kdim = q_shape[3] as i32;
    let vdim = v_shape[3] as i32;
    if kdim % 32 != 0 || vdim == 0 {
        return Err(Exception::custom(format!(
            "deltanet_recurrence: K must be a multiple of 32 (got K={kdim}, V={vdim})"
        )));
    }

    let dtype: u32 = q.dtype().into();

    unsafe {
        let stream = mlx_sys::mlx_default_gpu_stream_new();
        let config = mlx_sys::mlx_fast_metal_kernel_config_new();

        let t_name = CString::new("T").unwrap();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
            config, t_name.as_ptr(), dtype);
        let k_name = CString::new("K").unwrap();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(config, k_name.as_ptr(), kdim);
        let v_name = CString::new("V").unwrap();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(config, v_name.as_ptr(), vdim);
        let l_name = CString::new("L").unwrap();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(config, l_name.as_ptr(), l);
        let h_name = CString::new("H").unwrap();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(config, h_name.as_ptr(), h);

        // grid = (V*32, H, B); one simdgroup of 32 threads per (b, h, v_pos).
        let total_x = vdim * 32;
        mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, total_x, h, b);
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, 32, 1, 1);

        let out_shape: Vec<i32> = vec![b, h, l, vdim];
        let state_shape: Vec<i32> = vec![b, h, kdim, vdim];
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config, out_shape.as_ptr(), out_shape.len(), dtype);
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config, state_shape.as_ptr(), state_shape.len(), dtype);

        let inputs = mlx_sys::mlx_vector_array_new();
        mlx_sys::mlx_vector_array_append_value(inputs, q.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, k.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, v.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, decay.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, beta.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, state_in.as_ptr());

        let mut outputs = mlx_sys::mlx_vector_array_new();
        let ret = mlx_sys::mlx_fast_metal_kernel_apply(
            &mut outputs, kernel.kernel, inputs, config, stream);

        if ret != 0 {
            mlx_sys::mlx_fast_metal_kernel_config_free(config);
            mlx_sys::mlx_vector_array_free(inputs);
            mlx_sys::mlx_vector_array_free(outputs);
            mlx_sys::mlx_stream_free(stream);
            return Err(Exception::custom(
                "deltanet_recurrence Metal kernel execution failed"));
        }

        // mlx_fast_metal_kernel_apply creates lazy arrays in MLX's computation
        // graph — the kernel runs when eval() is triggered downstream, not here.
        // mlx_synchronize is therefore a no-op here but is harmless to keep.
        mlx_sys::mlx_synchronize(stream);

        let mut out_result = mlx_sys::mlx_array_new();
        mlx_sys::mlx_vector_array_get(&mut out_result, outputs, 0);
        let mut state_result = mlx_sys::mlx_array_new();
        mlx_sys::mlx_vector_array_get(&mut state_result, outputs, 1);

        mlx_sys::mlx_fast_metal_kernel_config_free(config);
        mlx_sys::mlx_vector_array_free(inputs);
        mlx_sys::mlx_vector_array_free(outputs);
        mlx_sys::mlx_stream_free(stream);

        Ok((Array::from_ptr(out_result), Array::from_ptr(state_result)))
    }
}

fn create_deltanet_with_tape_kernel() -> MetalKernel {
    unsafe {
        let q_name = CString::new("q_in").unwrap();
        let k_name = CString::new("k_in").unwrap();
        let v_name = CString::new("v_in").unwrap();
        let decay_name = CString::new("decay").unwrap();
        let beta_name = CString::new("beta").unwrap();
        let state_in_name = CString::new("state_in").unwrap();

        let output_name = CString::new("output").unwrap();
        let state_out_name = CString::new("state_out").unwrap();
        let tape_name = CString::new("tape").unwrap();

        let input_names = mlx_sys::mlx_vector_string_new();
        mlx_sys::mlx_vector_string_append_value(input_names, q_name.as_ptr());
        mlx_sys::mlx_vector_string_append_value(input_names, k_name.as_ptr());
        mlx_sys::mlx_vector_string_append_value(input_names, v_name.as_ptr());
        mlx_sys::mlx_vector_string_append_value(input_names, decay_name.as_ptr());
        mlx_sys::mlx_vector_string_append_value(input_names, beta_name.as_ptr());
        mlx_sys::mlx_vector_string_append_value(input_names, state_in_name.as_ptr());

        let output_names = mlx_sys::mlx_vector_string_new();
        mlx_sys::mlx_vector_string_append_value(output_names, output_name.as_ptr());
        mlx_sys::mlx_vector_string_append_value(output_names, state_out_name.as_ptr());
        mlx_sys::mlx_vector_string_append_value(output_names, tape_name.as_ptr());

        let source = CString::new(DELTANET_WITH_TAPE_KERNEL_SOURCE).unwrap();
        let header = CString::new("").unwrap();
        let name = CString::new("deltanet_with_tape").unwrap();

        let kernel = mlx_sys::mlx_fast_metal_kernel_new(
            name.as_ptr(),
            input_names,
            output_names,
            source.as_ptr(),
            header.as_ptr(),
            true,
            false,
        );

        MetalKernel {
            kernel,
            input_names,
            output_names,
        }
    }
}

fn create_deltanet_tape_replay_kernel() -> MetalKernel {
    unsafe {
        let tape_name = CString::new("tape").unwrap();
        let k_name = CString::new("k_in").unwrap();
        let decay_name = CString::new("decay").unwrap();
        let state_in_name = CString::new("state_in").unwrap();
        let state_out_name = CString::new("state_out").unwrap();

        let input_names = mlx_sys::mlx_vector_string_new();
        mlx_sys::mlx_vector_string_append_value(input_names, tape_name.as_ptr());
        mlx_sys::mlx_vector_string_append_value(input_names, k_name.as_ptr());
        mlx_sys::mlx_vector_string_append_value(input_names, decay_name.as_ptr());
        mlx_sys::mlx_vector_string_append_value(input_names, state_in_name.as_ptr());

        let output_names = mlx_sys::mlx_vector_string_new();
        mlx_sys::mlx_vector_string_append_value(output_names, state_out_name.as_ptr());

        let source = CString::new(DELTANET_TAPE_REPLAY_KERNEL_SOURCE).unwrap();
        let header = CString::new("").unwrap();
        let name = CString::new("deltanet_tape_replay").unwrap();

        let kernel = mlx_sys::mlx_fast_metal_kernel_new(
            name.as_ptr(),
            input_names,
            output_names,
            source.as_ptr(),
            header.as_ptr(),
            true,
            false,
        );

        MetalKernel {
            kernel,
            input_names,
            output_names,
        }
    }
}

/// Same as `deltanet_recurrence`, but additionally records the per-timestep
/// `delta` ("innovation") tensor.
///
/// Returns `(output, state_out, tape)`:
///   output: [B, H, L, V]
///   state_out: [B, H, K, V]
///   tape: [B, H, L, V]
///
/// The tape can later be fed to `deltanet_tape_replay` together with a
/// snapshot of the pre-scan state and any prefix of `k`/`decay` to
/// cheaply advance the recurrent state by an accepted prefix of tokens
/// (used by speculative-decoding rollback).  Requires K % 32 == 0.
pub fn deltanet_with_tape(
    q: &Array,
    k: &Array,
    v: &Array,
    decay: &Array,
    beta: &Array,
    state_in: &Array,
) -> Result<(Array, Array, Array), Exception> {
    let kernel = DELTANET_WITH_TAPE_KERNEL.get_or_init(create_deltanet_with_tape_kernel);

    let q_shape = q.shape();
    let v_shape = v.shape();
    if q_shape.len() != 4 || v_shape.len() != 4 {
        return Err(Exception::custom(format!(
            "deltanet_with_tape: expected 4D q and v, got q={:?} v={:?}",
            q_shape, v_shape
        )));
    }
    let b = q_shape[0] as i32;
    let h = q_shape[1] as i32;
    let l = q_shape[2] as i32;
    let kdim = q_shape[3] as i32;
    let vdim = v_shape[3] as i32;
    if kdim % 32 != 0 || vdim == 0 {
        return Err(Exception::custom(format!(
            "deltanet_with_tape: K must be a multiple of 32 (got K={kdim}, V={vdim})"
        )));
    }

    let dtype: u32 = q.dtype().into();

    unsafe {
        let stream = mlx_sys::mlx_default_gpu_stream_new();
        let config = mlx_sys::mlx_fast_metal_kernel_config_new();

        let t_name = CString::new("T").unwrap();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
            config, t_name.as_ptr(), dtype);
        let k_name = CString::new("K").unwrap();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(config, k_name.as_ptr(), kdim);
        let v_name = CString::new("V").unwrap();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(config, v_name.as_ptr(), vdim);
        let l_name = CString::new("L").unwrap();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(config, l_name.as_ptr(), l);
        let h_name = CString::new("H").unwrap();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(config, h_name.as_ptr(), h);

        let total_x = vdim * 32;
        mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, total_x, h, b);
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, 32, 1, 1);

        let out_shape: Vec<i32> = vec![b, h, l, vdim];
        let state_shape: Vec<i32> = vec![b, h, kdim, vdim];
        let tape_shape: Vec<i32> = vec![b, h, l, vdim];
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config, out_shape.as_ptr(), out_shape.len(), dtype);
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config, state_shape.as_ptr(), state_shape.len(), dtype);
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config, tape_shape.as_ptr(), tape_shape.len(), dtype);

        let inputs = mlx_sys::mlx_vector_array_new();
        mlx_sys::mlx_vector_array_append_value(inputs, q.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, k.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, v.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, decay.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, beta.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, state_in.as_ptr());

        let mut outputs = mlx_sys::mlx_vector_array_new();
        let ret = mlx_sys::mlx_fast_metal_kernel_apply(
            &mut outputs, kernel.kernel, inputs, config, stream);

        if ret != 0 {
            mlx_sys::mlx_fast_metal_kernel_config_free(config);
            mlx_sys::mlx_vector_array_free(inputs);
            mlx_sys::mlx_vector_array_free(outputs);
            mlx_sys::mlx_stream_free(stream);
            return Err(Exception::custom(
                "deltanet_with_tape Metal kernel execution failed"));
        }

        mlx_sys::mlx_synchronize(stream);

        let mut out_result = mlx_sys::mlx_array_new();
        mlx_sys::mlx_vector_array_get(&mut out_result, outputs, 0);
        let mut state_result = mlx_sys::mlx_array_new();
        mlx_sys::mlx_vector_array_get(&mut state_result, outputs, 1);
        let mut tape_result = mlx_sys::mlx_array_new();
        mlx_sys::mlx_vector_array_get(&mut tape_result, outputs, 2);

        mlx_sys::mlx_fast_metal_kernel_config_free(config);
        mlx_sys::mlx_vector_array_free(inputs);
        mlx_sys::mlx_vector_array_free(outputs);
        mlx_sys::mlx_stream_free(stream);

        Ok((
            Array::from_ptr(out_result),
            Array::from_ptr(state_result),
            Array::from_ptr(tape_result),
        ))
    }
}

/// Roll the recurrent state forward by replaying `L` steps of recorded
/// `(tape, k, decay)` from a snapshotted `state_in`.  No q/v are required
/// because the per-step output is not needed for state rollback.
///
/// Shape contract:
///   tape   : [B, H, L, V]
///   k      : [B, H, L, K]
///   decay  : [B, H, L]
///   state_in : [B, H, K, V]
/// Returns state_out : [B, H, K, V].
/// Requires K % 32 == 0.
pub fn deltanet_tape_replay(
    tape: &Array,
    k: &Array,
    decay: &Array,
    state_in: &Array,
) -> Result<Array, Exception> {
    let kernel = DELTANET_TAPE_REPLAY_KERNEL.get_or_init(create_deltanet_tape_replay_kernel);

    let tape_shape = tape.shape();
    let k_shape = k.shape();
    if tape_shape.len() != 4 || k_shape.len() != 4 {
        return Err(Exception::custom(format!(
            "deltanet_tape_replay: expected 4D tape and k, got tape={:?} k={:?}",
            tape_shape, k_shape
        )));
    }
    let b = tape_shape[0] as i32;
    let h = tape_shape[1] as i32;
    let l = tape_shape[2] as i32;
    let vdim = tape_shape[3] as i32;
    let kdim = k_shape[3] as i32;
    if k_shape[0] != b || k_shape[1] != h || k_shape[2] != l {
        return Err(Exception::custom(format!(
            "deltanet_tape_replay: tape/k mismatch tape={:?} k={:?}",
            tape_shape, k_shape
        )));
    }
    if kdim % 32 != 0 || vdim == 0 {
        return Err(Exception::custom(format!(
            "deltanet_tape_replay: K must be a multiple of 32 (got K={kdim}, V={vdim})"
        )));
    }

    let dtype: u32 = tape.dtype().into();

    unsafe {
        let stream = mlx_sys::mlx_default_gpu_stream_new();
        let config = mlx_sys::mlx_fast_metal_kernel_config_new();

        let t_name = CString::new("T").unwrap();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
            config, t_name.as_ptr(), dtype);
        let k_name = CString::new("K").unwrap();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(config, k_name.as_ptr(), kdim);
        let v_name = CString::new("V").unwrap();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(config, v_name.as_ptr(), vdim);
        let l_name = CString::new("L").unwrap();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(config, l_name.as_ptr(), l);
        let h_name = CString::new("H").unwrap();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(config, h_name.as_ptr(), h);

        let total_x = vdim * 32;
        mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, total_x, h, b);
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, 32, 1, 1);

        let state_shape: Vec<i32> = vec![b, h, kdim, vdim];
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config, state_shape.as_ptr(), state_shape.len(), dtype);

        let inputs = mlx_sys::mlx_vector_array_new();
        mlx_sys::mlx_vector_array_append_value(inputs, tape.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, k.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, decay.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, state_in.as_ptr());

        let mut outputs = mlx_sys::mlx_vector_array_new();
        let ret = mlx_sys::mlx_fast_metal_kernel_apply(
            &mut outputs, kernel.kernel, inputs, config, stream);

        if ret != 0 {
            mlx_sys::mlx_fast_metal_kernel_config_free(config);
            mlx_sys::mlx_vector_array_free(inputs);
            mlx_sys::mlx_vector_array_free(outputs);
            mlx_sys::mlx_stream_free(stream);
            return Err(Exception::custom(
                "deltanet_tape_replay Metal kernel execution failed"));
        }

        mlx_sys::mlx_synchronize(stream);

        let mut state_result = mlx_sys::mlx_array_new();
        mlx_sys::mlx_vector_array_get(&mut state_result, outputs, 0);

        mlx_sys::mlx_fast_metal_kernel_config_free(config);
        mlx_sys::mlx_vector_array_free(inputs);
        mlx_sys::mlx_vector_array_free(outputs);
        mlx_sys::mlx_stream_free(stream);

        Ok(Array::from_ptr(state_result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::indexing::IndexOp;
    use mlx_rs::Array;

    /// Reference implementation of the delta rule recurrence in pure Rust/MLX ops.
    ///
    /// Matches the fallback loop in deltanet.rs (forward_prefill):
    ///   - decay applied first
    ///   - kv_mem retrieved from decayed state
    ///   - state updated with k ⊗ delta (no second decay)
    ///
    /// Inputs:
    ///   q, k: [B, H, L, K]   (float32)
    ///   v:    [B, H, L, V]   (float32)
    ///   decay: [B, H, L]     (float32, pre-exponentiated α)
    ///   beta:  [B, H, L]     (float32)
    ///   state_in: [B, H, K, V] (float32)
    ///
    /// Returns (output [B, H, L, V], state_out [B, H, K, V])
    fn delta_rule_reference(
        q: &Array,
        k: &Array,
        v: &Array,
        decay: &Array,
        beta: &Array,
        state_in: &Array,
    ) -> Result<(Array, Array), mlx_rs::error::Exception> {
        let shape = q.shape();
        let (b, h, l, k_dim) = (shape[0], shape[1], shape[2], shape[3]);
        let v_dim = v.shape()[3];

        // state_t: [B, H, V, K] (transposed for matmul convenience)
        let mut state_t = state_in.transpose_axes(&[0, 1, 3, 2])?;

        let decay_5d = decay.reshape(&[b, h, l, 1, 1])?;
        let k_col = k.reshape(&[b, h, l, k_dim, 1])?;
        let q_col = q.reshape(&[b, h, l, k_dim, 1])?;
        let v_col = v.reshape(&[b, h, l, v_dim, 1])?;
        let beta_col = beta.reshape(&[b, h, l, 1, 1])?;
        let k_row = k_col.transpose_axes(&[0, 1, 2, 4, 3])?;

        let mut outputs = Vec::with_capacity(l as usize);
        for t in 0..l {
            let decay_t = decay_5d.index((.., .., t, .., ..));
            let k_t   = k_col.index((.., .., t, .., ..));
            let v_t   = v_col.index((.., .., t, .., ..));
            let beta_t = beta_col.index((.., .., t, .., ..));
            let q_t   = q_col.index((.., .., t, .., ..));
            let k_r   = k_row.index((.., .., t, .., ..));

            state_t = state_t.multiply(&decay_t)?;
            let kv_mem = state_t.matmul(&k_t)?;
            let delta = v_t.subtract(&kv_mem)?.multiply(&beta_t)?;
            state_t = state_t.add(delta.matmul(&k_r)?)?;
            outputs.push(state_t.matmul(&q_t)?);
        }

        let stacked = mlx_rs::ops::stack_axis(&outputs, 2)?;
        let out_bhlv = stacked.reshape(&[b, h, l, v_dim])?;
        let state_out = state_t.transpose_axes(&[0, 1, 3, 2])?;
        Ok((out_bhlv, state_out))
    }

    /// Assert two arrays are element-wise close (max absolute diff < tol).
    fn assert_arrays_close(a: &Array, b: &Array, tol: f32, label: &str) {
        mlx_rs::transforms::eval([a, b]).unwrap();
        let a_cont = a.contiguous().unwrap();
        let b_cont = b.contiguous().unwrap();
        mlx_rs::transforms::eval([&a_cont, &b_cont]).unwrap();
        let a_slice = a_cont.as_slice::<f32>();
        let b_slice = b_cont.as_slice::<f32>();
        assert_eq!(a_slice.len(), b_slice.len(), "{}: array sizes differ", label);
        let max_diff = a_slice
            .iter()
            .zip(b_slice.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < tol,
            "{}: max absolute difference {} exceeds tolerance {}",
            label, max_diff, tol,
        );
    }

    /// Verify the Metal delta-rule kernel matches the reference fallback for a
    /// small (B=1, H=2, L=4, K=32, V=32) input.  K=32 satisfies the K%32==0
    /// condition so the fast kernel path is exercised.
    ///
    /// This test guards against the bug where the kernel retrieved kv_mem from
    /// the *undecayed* state instead of the decayed state, causing the model to
    /// collapse to EOS on the first decode token.
    #[test]
    fn test_deltanet_kernel_matches_reference() {
        let b = 1i32;
        let h = 2i32;
        let l = 4i32;
        let k_dim = 32i32;
        let v_dim = 32i32;

        let q_data: Vec<f32> = (0..(b * h * l * k_dim) as usize)
            .map(|i| ((i as f32) * 0.01 - 0.5).tanh())
            .collect();
        let k_data: Vec<f32> = (0..(b * h * l * k_dim) as usize)
            .map(|i| ((i as f32) * 0.013 - 0.3).tanh())
            .collect();
        let v_data: Vec<f32> = (0..(b * h * l * v_dim) as usize)
            .map(|i| ((i as f32) * 0.007).sin() * 0.5)
            .collect();
        // decay in (0, 1): pre-exponentiated α values
        let decay_data: Vec<f32> = (0..(b * h * l) as usize)
            .map(|i| (-((i as f32) * 0.1 + 0.1).exp()).exp())
            .collect();
        let beta_data: Vec<f32> = (0..(b * h * l) as usize)
            .map(|i| 1.0 / (1.0 + (-((i as f32) * 0.2)).exp()))
            .collect();
        let state_data = vec![0.0f32; (b * h * k_dim * v_dim) as usize];

        let q = Array::from_slice(&q_data, &[b, h, l, k_dim]);
        let k = Array::from_slice(&k_data, &[b, h, l, k_dim]);
        let v = Array::from_slice(&v_data, &[b, h, l, v_dim]);
        let decay = Array::from_slice(&decay_data, &[b, h, l]);
        let beta = Array::from_slice(&beta_data, &[b, h, l]);
        let state_in = Array::from_slice(&state_data, &[b, h, k_dim, v_dim]);

        let (ref_out, ref_state) =
            delta_rule_reference(&q, &k, &v, &decay, &beta, &state_in)
                .expect("reference recurrence failed");

        let (kernel_out, kernel_state) =
            deltanet_recurrence(&q, &k, &v, &decay, &beta, &state_in)
                .expect("Metal kernel recurrence failed");

        assert_arrays_close(&kernel_out, &ref_out, 1e-4, "output");
        assert_arrays_close(&kernel_state, &ref_state, 1e-4, "state_out");
    }

    /// Verify that `deltanet_with_tape` produces the same output/state as
    /// `deltanet_recurrence`, and that replaying the recorded tape from the
    /// original `state_in` (via `deltanet_tape_replay`) reproduces the final
    /// state.  Guards against tape semantics regressions in the rollback path.
    #[test]
    fn test_deltanet_with_tape_and_replay() {
        let b = 1i32;
        let h = 2i32;
        let l = 4i32;
        let k_dim = 32i32;
        let v_dim = 32i32;

        let q_data: Vec<f32> = (0..(b * h * l * k_dim) as usize)
            .map(|i| ((i as f32) * 0.01 - 0.5).tanh())
            .collect();
        let k_data: Vec<f32> = (0..(b * h * l * k_dim) as usize)
            .map(|i| ((i as f32) * 0.013 - 0.3).tanh())
            .collect();
        let v_data: Vec<f32> = (0..(b * h * l * v_dim) as usize)
            .map(|i| ((i as f32) * 0.007).sin() * 0.5)
            .collect();
        let decay_data: Vec<f32> = (0..(b * h * l) as usize)
            .map(|i| (-((i as f32) * 0.1 + 0.1).exp()).exp())
            .collect();
        let beta_data: Vec<f32> = (0..(b * h * l) as usize)
            .map(|i| 1.0 / (1.0 + (-((i as f32) * 0.2)).exp()))
            .collect();
        let state_data = vec![0.0f32; (b * h * k_dim * v_dim) as usize];

        let q = Array::from_slice(&q_data, &[b, h, l, k_dim]);
        let k = Array::from_slice(&k_data, &[b, h, l, k_dim]);
        let v = Array::from_slice(&v_data, &[b, h, l, v_dim]);
        let decay = Array::from_slice(&decay_data, &[b, h, l]);
        let beta = Array::from_slice(&beta_data, &[b, h, l]);
        let state_in = Array::from_slice(&state_data, &[b, h, k_dim, v_dim]);

        let (ref_out, ref_state) =
            deltanet_recurrence(&q, &k, &v, &decay, &beta, &state_in)
                .expect("reference recurrence failed");
        let (tape_out, tape_state, tape) =
            deltanet_with_tape(&q, &k, &v, &decay, &beta, &state_in)
                .expect("tape kernel failed");

        assert_arrays_close(&tape_out, &ref_out, 1e-4, "tape output vs reference");
        assert_arrays_close(&tape_state, &ref_state, 1e-4, "tape state vs reference");

        // Full-length replay should reproduce the same final state.
        let replayed = deltanet_tape_replay(&tape, &k, &decay, &state_in)
            .expect("tape replay failed");
        assert_arrays_close(&replayed, &ref_state, 1e-4, "full tape replay");

        // Partial replay: keep first 2 steps. Compare to fresh recurrence on
        // q[:,:,:2], k[:,:,:2], v[:,:,:2] from the same state_in.
        let keep = 2i32;
        let q_p = q.index((.., .., ..keep, ..));
        let k_p = k.index((.., .., ..keep, ..));
        let v_p = v.index((.., .., ..keep, ..));
        let decay_p = decay.index((.., .., ..keep));
        let beta_p = beta.index((.., .., ..keep));
        let (_pref_out, pref_state) =
            deltanet_recurrence(&q_p, &k_p, &v_p, &decay_p, &beta_p, &state_in)
                .expect("partial reference failed");
        let tape_p = tape.index((.., .., ..keep, ..));
        let replayed_p = deltanet_tape_replay(&tape_p, &k_p, &decay_p, &state_in)
            .expect("partial tape replay failed");
        assert_arrays_close(&replayed_p, &pref_state, 1e-4, "partial tape replay");
    }
}
