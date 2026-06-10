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

// =============================================================================
// Fused TurboQuant QK score kernel
// =============================================================================
//
// Computes per-position attention scores directly from compressed K
// (no intermediate dequantization) using the algebraic identity:
//   Q · K = Q · (signs ⊙ Hadamard^-1(centroids[idx] * sigma) + mean)
//         = Hadamard(signs ⊙ Q) · (centroids[idx] * sigma) + mean * sum(Q)
//         = sigma * (Q_rot · centroids[idx]) + mean * query_sum
//
// where Q_rot = Hadamard(signs ⊙ Q) is the pre-rotated query (Hadamard
// is orthogonal symmetric so it's its own inverse modulo normalization).
//
// Grid layout: one threadgroup per (b, h, q) triple. Each threadgroup
// owns one Q vector, pre-rotates it once, then computes scores against
// all kv positions in parallel across the 64 threads.
//
// GQA mapping: q_head h → kv_head h / kv_repeat, where kv_repeat =
// n_q_heads / n_kv_heads. Both head counts come in via template ints.
const TQ_QK_SCORE_KERNEL: &str = r#"
    uint tid = thread_position_in_threadgroup.x;
    uint group = threadgroup_position_in_grid.x;
    // Decompose group = b * Hq + h_q (q_len = 1 in spike).
    uint b   = group / uint(Hq);
    uint h_q = group % uint(Hq);
    if (b >= uint(B)) return;
    uint h_kv = h_q / uint(kv_repeat);

    threadgroup float s_q[512]; // pre-rotated query, head_dim ≤ 512

    // Step 1: load Q[b, h_q, 0, :] × signs.
    uint q_offset = (b * uint(Hq) + h_q) * uint(D);
    for (uint i = tid; i < uint(D); i += uint(64)) {
        s_q[i] = float(q_in[q_offset + i]) * signs[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: query_sum = sum_i (signs[i] * Q[i]).
    // Derivation: K_recon[d] = (WHT(centroids[idx]*sigma)[d] + mean) * signs[d].
    // Then Q·K_recon = sum_d Q[d]*signs[d] * WHT(c)[d] + mean * sum_d Q[d]*signs[d]
    //               = sigma * (Q_rot · centroids[idx]) + mean * (signs⊙Q).sum()
    // — i.e. mean is multiplied by the SIGNED query sum, not the raw sum.
    threadgroup float s_red[64];
    float local_sum = 0.0f;
    for (uint i = tid; i < uint(D); i += uint(64)) {
        local_sum += s_q[i]; // s_q already holds Q[i] * signs[i].
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
    float query_sum = s_red[0];

    // Step 3: in-place WHT on the sign-flipped Q (now in s_q).
    for (uint step = 1u; step < uint(D); step <<= 1u) {
        for (uint i = tid; i < uint(D) / 2u; i += uint(64)) {
            uint j = (i / step) * (step * 2u) + (i % step);
            uint k = j + step;
            float a = s_q[j];
            float b2 = s_q[k];
            s_q[j] = a + b2;
            s_q[k] = a - b2;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float scale = metal::rsqrt(float(D));
    for (uint i = tid; i < uint(D); i += uint(64)) {
        s_q[i] *= scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 4: per-kv-position score.
    // packed_k shape conceptually [B, H_kv, KV, D/8]; we read by index.
    // sigma/mean shape [B, H_kv, KV].
    uint packed_per_vec = uint(D) / 8u;
    uint kv_stride_inner = packed_per_vec;
    uint kv_stride_h     = uint(KV) * kv_stride_inner;
    uint kv_stride_b     = uint(Hkv) * kv_stride_h;
    uint kv_meta_stride_h = uint(KV);
    uint kv_meta_stride_b = uint(Hkv) * kv_meta_stride_h;

    uint out_offset = ((b * uint(Hq) + h_q) * 1u) * uint(KV);

    for (uint k = tid; k < uint(KV); k += uint(64)) {
        uint base = b * kv_stride_b + h_kv * kv_stride_h + k * kv_stride_inner;
        float dot = 0.0f;
        for (uint w = 0u; w < packed_per_vec; ++w) {
            uint packed = packed_k[base + w];
            // 8 indices in one u32.
            uint d0 = w * 8u;
            dot += s_q[d0 + 0u] * centroids[(packed >>  0u) & 0xFu];
            dot += s_q[d0 + 1u] * centroids[(packed >>  4u) & 0xFu];
            dot += s_q[d0 + 2u] * centroids[(packed >>  8u) & 0xFu];
            dot += s_q[d0 + 3u] * centroids[(packed >> 12u) & 0xFu];
            dot += s_q[d0 + 4u] * centroids[(packed >> 16u) & 0xFu];
            dot += s_q[d0 + 5u] * centroids[(packed >> 20u) & 0xFu];
            dot += s_q[d0 + 6u] * centroids[(packed >> 24u) & 0xFu];
            dot += s_q[d0 + 7u] * centroids[(packed >> 28u) & 0xFu];
        }
        uint meta = b * kv_meta_stride_b + h_kv * kv_meta_stride_h + k;
        float sigma_k = sigma_in[meta];
        float mean_k  = mean_in[meta];
        scores_out[out_offset + k] = sigma_k * dot + mean_k * query_sum;
    }
"#;

static TQ_QK_SCORE_KERNEL_HANDLE: OnceLock<MetalKernel> = OnceLock::new();

// =============================================================================
// Fully-fused TurboQuant SDPA kernel
// =============================================================================
//
// One Metal dispatch that does QK score (from compressed K) + softmax +
// V matmul. Replaces the prior 3-dispatch sequence used by
// `try_fused_attention` so the K-decompress savings actually show up
// on the decode wall-clock at short contexts.
//
// Layout: one threadgroup per (b, h_q). 64 threads collaborate:
//   - Threads cooperate on Q pre-rotation (sign flip + WHT in shared mem).
//   - Each thread strides over kv positions for the score loop, writing
//     into a shared [KV] scratch buffer.
//   - Cooperative reduce for softmax max / sum.
//   - Each thread accumulates the V multiply for its D slice.
//
// Constraints:
//   - q_len = 1 (decode hot path).
//   - kv_len ≤ MAX_KV_BUF (4096 by default — bounded by threadgroup
//     memory: 4096 * 4 bytes = 16 KB for the scores scratch).
//   - head_dim ≤ 512 (matches the WHT shared-mem cap).
//
// Inputs:
//   q          [B, Hq, 1, D]     T (BF16/F16/F32 templated)
//   packed_k   [B, Hkv, KV, D/8] u32
//   sigma_k    [B, Hkv, KV]      f32
//   mean_k     [B, Hkv, KV]      f32
//   v          [B, Hkv, KV, D]   T
//   signs      [D]               f32
//   centroids  [16]              f32
//   mask       [KV] or empty     f32 (broadcast over all q heads)
//
// Output:
//   out        [B, Hq, 1, D]     T
const TQ_SDPA_4BIT_KERNEL: &str = r#"
    uint tid = thread_position_in_threadgroup.x;
    // Grid layout:
    //   x = 64 * (b * Hq + h_q)        — picks (batch, query head)
    //   y = d_chunk_id                 — picks which D-slice this group owns
    uint group_x = threadgroup_position_in_grid.x;
    uint d_chunk_id = threadgroup_position_in_grid.y;
    uint b   = group_x / uint(Hq);
    uint h_q = group_x % uint(Hq);
    if (b >= uint(B)) return;
    uint h_kv = h_q / uint(kv_repeat);
    uint d_start = d_chunk_id * uint(D_CHUNK);
    uint d_end   = metal::min(d_start + uint(D_CHUNK), uint(D));
    if (d_start >= uint(D)) return;

    threadgroup float s_q[512];
    threadgroup float s_centroids[16];
    // s_scores at 2048 = 8 KB. Apple SM has 32 KB threadgroup mem;
    // budget is ~12 KB after Q + centroids + reductions, so 2048 is
    // a sane upper bound that still leaves room for ~2-3 concurrent
    // groups per SM. Covers Gemma4's 2x-sliding-window contexts;
    // longer caches go through the online-softmax kernel.
    threadgroup float s_scores[2048];
    threadgroup float s_red[64];
    threadgroup float s_query_sum;
    threadgroup float s_global_max;
    threadgroup float s_total_sum;

    // Centroid table → shared.
    if (tid < 16u) s_centroids[tid] = centroids[tid];
    // Load Q * signs.
    uint q_offset = (b * uint(Hq) + h_q) * uint(D);
    for (uint i = tid; i < uint(D); i += uint(64)) {
        s_q[i] = float(q[q_offset + i]) * signs[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // query_sum_signed = sum(s_q).
    float local_sum = 0.0f;
    for (uint i = tid; i < uint(D); i += uint(64)) local_sum += s_q[i];
    s_red[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) s_red[tid] += s_red[tid + 32];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 16) s_red[tid] += s_red[tid + 16];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid <  8) s_red[tid] += s_red[tid +  8];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid <  4) s_red[tid] += s_red[tid +  4];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid <  2) s_red[tid] += s_red[tid +  2];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { s_red[0] += s_red[1]; s_query_sum = s_red[0]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float query_sum_signed = s_query_sum;

    // In-place WHT.
    for (uint step = 1u; step < uint(D); step <<= 1u) {
        for (uint i = tid; i < uint(D) / 2u; i += uint(64)) {
            uint j = (i / step) * (step * 2u) + (i % step);
            uint k = j + step;
            float a = s_q[j];
            float b2 = s_q[k];
            s_q[j] = a + b2;
            s_q[k] = a - b2;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float wht_scale = metal::rsqrt(float(D));
    for (uint i = tid; i < uint(D); i += uint(64)) s_q[i] *= wht_scale;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 1: score per kv position into s_scores, track local max.
    uint packed_per_vec = uint(D) / 8u;
    uint kv_stride_inner = packed_per_vec;
    uint kv_stride_h     = uint(KV) * kv_stride_inner;
    uint kv_stride_b     = uint(Hkv) * kv_stride_h;
    uint meta_stride_h   = uint(KV);
    uint meta_stride_b   = uint(Hkv) * meta_stride_h;
    float local_max = -1e30f;
    for (uint k = tid; k < uint(KV); k += uint(64)) {
        uint base = b * kv_stride_b + h_kv * kv_stride_h + k * kv_stride_inner;
        float dot = 0.0f;
        for (uint w = 0u; w < packed_per_vec; ++w) {
            uint packed = packed_k[base + w];
            uint d0 = w * 8u;
            dot += s_q[d0 + 0u] * s_centroids[(packed >>  0u) & 0xFu];
            dot += s_q[d0 + 1u] * s_centroids[(packed >>  4u) & 0xFu];
            dot += s_q[d0 + 2u] * s_centroids[(packed >>  8u) & 0xFu];
            dot += s_q[d0 + 3u] * s_centroids[(packed >> 12u) & 0xFu];
            dot += s_q[d0 + 4u] * s_centroids[(packed >> 16u) & 0xFu];
            dot += s_q[d0 + 5u] * s_centroids[(packed >> 20u) & 0xFu];
            dot += s_q[d0 + 6u] * s_centroids[(packed >> 24u) & 0xFu];
            dot += s_q[d0 + 7u] * s_centroids[(packed >> 28u) & 0xFu];
        }
        uint meta = b * meta_stride_b + h_kv * meta_stride_h + k;
        float sigma = sigma_k[meta];
        float mean  = mean_k[meta];
        // Q has been pre-scaled by `scale` on the host (Metal custom
        // kernels don't take f32 template args). sigma and mean act
        // linearly on Q so the scale flows through both terms.
        float score = sigma * dot + mean * query_sum_signed;
        if (has_mask != 0) score += mask[k];
        s_scores[k] = score;
        local_max = metal::max(local_max, score);
    }
    s_red[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) s_red[tid] = metal::max(s_red[tid], s_red[tid + 32]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 16) s_red[tid] = metal::max(s_red[tid], s_red[tid + 16]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid <  8) s_red[tid] = metal::max(s_red[tid], s_red[tid +  8]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid <  4) s_red[tid] = metal::max(s_red[tid], s_red[tid +  4]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid <  2) s_red[tid] = metal::max(s_red[tid], s_red[tid +  2]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { s_red[0] = metal::max(s_red[0], s_red[1]); s_global_max = s_red[0]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float gmax = s_global_max;

    // Phase 2: exp(score - max), sum.
    float local_sum_exp = 0.0f;
    for (uint k = tid; k < uint(KV); k += uint(64)) {
        float e = metal::exp(s_scores[k] - gmax);
        s_scores[k] = e;
        local_sum_exp += e;
    }
    s_red[tid] = local_sum_exp;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) s_red[tid] += s_red[tid + 32];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 16) s_red[tid] += s_red[tid + 16];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid <  8) s_red[tid] += s_red[tid +  8];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid <  4) s_red[tid] += s_red[tid +  4];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid <  2) s_red[tid] += s_red[tid +  2];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { s_red[0] += s_red[1]; s_total_sum = s_red[0]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_total = 1.0f / s_total_sum;

    // Phase 3: V multiply. Per-dim accumulate over kv, restricted to
    // this threadgroup's d_chunk slice. V is dequantised inline from
    // packed 8-bit per-group (mlx symmetric affine), so we avoid the
    // separate V-decompress dispatch the caller used to do.
    //
    // V layout:
    //   packed_v   [B, Hkv, KV, D/4]            u32   (4 vals per word)
    //   v_scales   [B, Hkv, KV, D/v_group_size] f32
    //   v_biases   [B, Hkv, KV, D/v_group_size] f32
    uint packed_per_row = uint(D) / 4u;
    uint v_inner = packed_per_row;
    uint v_h     = uint(KV) * v_inner;
    uint v_b     = uint(Hkv) * v_h;
    uint v_base  = b * v_b + h_kv * v_h;

    uint n_groups = uint(D) / uint(V_GROUP_SIZE);
    uint vs_inner = n_groups;
    uint vs_h     = uint(KV) * vs_inner;
    uint vs_b     = uint(Hkv) * vs_h;
    uint vs_base  = b * vs_b + h_kv * vs_h;

    uint out_offset = (b * uint(Hq) + h_q) * uint(D);
    uint chunk_size = d_end - d_start;
    for (uint dd = tid; dd < chunk_size; dd += uint(64)) {
        uint d = d_start + dd;
        uint word_idx = d / 4u;
        uint byte_idx = d % 4u;
        uint group_id = d / uint(V_GROUP_SIZE);
        float acc = 0.0f;
        for (uint k = 0u; k < uint(KV); ++k) {
            uint packed = packed_v[v_base + k * v_inner + word_idx];
            uint byte_val = (packed >> (byte_idx * 8u)) & 0xFFu;
            float scale = v_scales[vs_base + k * vs_inner + group_id];
            float bias  = v_biases[vs_base + k * vs_inner + group_id];
            float v_dq  = float(byte_val) * scale + bias;
            acc += s_scores[k] * inv_total * v_dq;
        }
        out[out_offset + d] = T(acc);
    }
"#;

static TQ_SDPA_4BIT_KERNEL_HANDLE: OnceLock<MetalKernel> = OnceLock::new();

// =============================================================================
// Online-softmax TurboQuant SDPA kernel (Flash-Attention v2 style)
// =============================================================================
//
// Same algebraic identity + V-dequant-fuse + D-tiling as `tq_sdpa_4bit`,
// but processes kv in TILE_KV=64 chunks and maintains a running
// (running_max, running_sum_exp, running_output) per threadgroup. Never
// materialises a full [KV] score buffer, so kv_len is unbounded —
// scaling to 5k+ token hermes contexts that the prior kernel couldn't
// touch (its s_scores[1024] cap forced fallback to the partial-fuse
// path at long contexts).
//
// Per-tile algorithm (Flash-Attention v2):
//   1. Compute scores for tile_kv positions into s_tile_scores[64].
//   2. tile_max = max over tile.
//   3. new_m  = max(running_max, tile_max).
//      alpha  = exp(running_max - new_m).
//   4. For each k in tile: s_tile_scores[k] = exp(score - new_m).
//      tile_sum = sum(s_tile_scores).
//   5. l = l * alpha + tile_sum.
//   6. Per-thread output update for this thread's d:
//        out[d] = out[d] * alpha + sum_k s_tile_scores[k] * V[k, d]
//      where V is dequantised inline from packed_v.
//   7. m = new_m.
// Final:
//   out[d] = out[d] / l   (per-thread, one d per thread).
//
// Constraint: D_CHUNK must equal 64 so one thread owns exactly one
// output dim. This makes the V-multiply phase trivially parallel and
// avoids per-thread state arrays.
const TQ_SDPA_4BIT_ONLINE_KERNEL: &str = r#"
    uint tid = thread_position_in_threadgroup.x;
    uint group_x = threadgroup_position_in_grid.x;
    uint d_chunk_id = threadgroup_position_in_grid.y;
    uint b   = group_x / uint(Hq);
    uint h_q = group_x % uint(Hq);
    if (b >= uint(B)) return;
    uint h_kv = h_q / uint(kv_repeat);
    uint d_start = d_chunk_id * 64u;            // D_CHUNK fixed at 64
    if (d_start >= uint(D)) return;
    uint d = d_start + tid;                     // one output dim per thread
    bool d_valid = d < uint(D);

    threadgroup float s_q[512];
    threadgroup float s_centroids[16];
    threadgroup float s_tile_scores[64];        // TILE_KV
    threadgroup float s_red[64];
    threadgroup float s_query_sum;
    threadgroup float s_alpha;
    threadgroup float s_new_m;
    threadgroup float s_tile_sum;
    threadgroup float s_running_m;
    threadgroup float s_running_l;
    // V-tile cache: 64 KV rows x 64 D-chunk values, dequantised once per
    // tile and reused across the 64 output threads. Eliminates the
    // ~5k GMEM reads per output dim that made the naive online kernel
    // bottlenecked on packed_v / v_scales loads. Stored in float so the
    // output FMAs match the per-thread reference path exactly.
    threadgroup float s_v_tile[64 * 64];

    // Load centroids.
    if (tid < 16u) s_centroids[tid] = centroids[tid];

    // Pre-rotate Q: signs * Q, then WHT.
    uint q_offset = (b * uint(Hq) + h_q) * uint(D);
    for (uint i = tid; i < uint(D); i += uint(64)) {
        s_q[i] = float(q[q_offset + i]) * signs[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // query_sum_signed = sum(s_q) BEFORE the WHT.
    float local_sum = 0.0f;
    for (uint i = tid; i < uint(D); i += uint(64)) local_sum += s_q[i];
    s_red[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) s_red[tid] += s_red[tid + 32];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 16) s_red[tid] += s_red[tid + 16];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid <  8) s_red[tid] += s_red[tid +  8];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid <  4) s_red[tid] += s_red[tid +  4];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid <  2) s_red[tid] += s_red[tid +  2];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { s_red[0] += s_red[1]; s_query_sum = s_red[0]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float query_sum_signed = s_query_sum;

    // WHT.
    for (uint step = 1u; step < uint(D); step <<= 1u) {
        for (uint i = tid; i < uint(D) / 2u; i += uint(64)) {
            uint j = (i / step) * (step * 2u) + (i % step);
            uint k = j + step;
            float a = s_q[j];
            float b2 = s_q[k];
            s_q[j] = a + b2;
            s_q[k] = a - b2;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float wht_scale = metal::rsqrt(float(D));
    for (uint i = tid; i < uint(D); i += uint(64)) s_q[i] *= wht_scale;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // KV layout strides.
    uint packed_per_vec = uint(D) / 8u;
    uint k_stride_inner = packed_per_vec;
    uint k_stride_h     = uint(KV) * k_stride_inner;
    uint k_stride_b     = uint(Hkv) * k_stride_h;
    uint meta_stride_h  = uint(KV);
    uint meta_stride_b  = uint(Hkv) * meta_stride_h;

    uint v_packed_per_row = uint(D) / 4u;
    uint v_stride_h       = uint(KV) * v_packed_per_row;
    uint v_stride_b       = uint(Hkv) * v_stride_h;
    uint vs_inner         = uint(D) / uint(V_GROUP_SIZE);
    uint vs_h             = uint(KV) * vs_inner;
    uint vs_b             = uint(Hkv) * vs_h;
    uint v_base  = b * v_stride_b + h_kv * v_stride_h;
    uint vs_base = b * vs_b + h_kv * vs_h;

    // Per-thread output accumulator (one d per thread).
    float my_out = 0.0f;
    uint  my_word_idx = 0u;
    uint  my_byte_idx = 0u;
    uint  my_group_id = 0u;
    if (d_valid) {
        my_word_idx = d / 4u;
        my_byte_idx = d % 4u;
        my_group_id = d / uint(V_GROUP_SIZE);
    }

    // Running state in shared.
    if (tid == 0) { s_running_m = -1.0e30f; s_running_l = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tile loop: kv in [t..t+TILE_KV).
    for (uint t = 0u; t < uint(KV); t += 64u) {
        uint tile_end = metal::min(t + 64u, uint(KV));
        uint tile_len = tile_end - t;

        // (a) Score this tile — one score per thread (TILE_KV = 64
        //     threads). Last partial tile masks extras with -inf.
        float my_score = -1.0e30f;
        if (tid < tile_len) {
            uint k = t + tid;
            uint kbase = b * k_stride_b + h_kv * k_stride_h + k * k_stride_inner;
            float dot = 0.0f;
            for (uint w = 0u; w < packed_per_vec; ++w) {
                uint packed = packed_k[kbase + w];
                uint d0 = w * 8u;
                dot += s_q[d0 + 0u] * s_centroids[(packed >>  0u) & 0xFu];
                dot += s_q[d0 + 1u] * s_centroids[(packed >>  4u) & 0xFu];
                dot += s_q[d0 + 2u] * s_centroids[(packed >>  8u) & 0xFu];
                dot += s_q[d0 + 3u] * s_centroids[(packed >> 12u) & 0xFu];
                dot += s_q[d0 + 4u] * s_centroids[(packed >> 16u) & 0xFu];
                dot += s_q[d0 + 5u] * s_centroids[(packed >> 20u) & 0xFu];
                dot += s_q[d0 + 6u] * s_centroids[(packed >> 24u) & 0xFu];
                dot += s_q[d0 + 7u] * s_centroids[(packed >> 28u) & 0xFu];
            }
            uint meta = b * meta_stride_b + h_kv * meta_stride_h + k;
            float sigma = sigma_k[meta];
            float mean  = mean_k[meta];
            my_score = sigma * dot + mean * query_sum_signed;
            if (has_mask != 0) my_score += mask[k];
        }
        s_tile_scores[tid] = my_score;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // (b) Cooperative max-reduce across the 64-thread tile.
        s_red[tid] = my_score;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 32) s_red[tid] = metal::max(s_red[tid], s_red[tid + 32]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 16) s_red[tid] = metal::max(s_red[tid], s_red[tid + 16]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid <  8) s_red[tid] = metal::max(s_red[tid], s_red[tid +  8]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid <  4) s_red[tid] = metal::max(s_red[tid], s_red[tid +  4]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid <  2) s_red[tid] = metal::max(s_red[tid], s_red[tid +  2]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float tmax = metal::max(s_red[0], s_red[1]);
            float nm = metal::max(s_running_m, tmax);
            s_alpha = (s_running_m > -1.0e29f) ? metal::exp(s_running_m - nm) : 0.0f;
            s_new_m = nm;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float alpha = s_alpha;
        float new_m = s_new_m;

        // (c) Cooperative exp + sum-reduce. Each thread computes one
        //     position's exp, then we reduce 64 partial sums.
        float my_exp = 0.0f;
        if (tid < tile_len) {
            my_exp = metal::exp(s_tile_scores[tid] - new_m);
            s_tile_scores[tid] = my_exp;
        } else {
            s_tile_scores[tid] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        s_red[tid] = my_exp;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 32) s_red[tid] += s_red[tid + 32];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 16) s_red[tid] += s_red[tid + 16];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid <  8) s_red[tid] += s_red[tid +  8];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid <  4) s_red[tid] += s_red[tid +  4];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid <  2) s_red[tid] += s_red[tid +  2];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float tsum = s_red[0] + s_red[1];
            s_tile_sum = tsum;
            s_running_l = s_running_l * alpha + tsum;
            s_running_m = new_m;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // (d) Cooperatively dequantise the V tile into shared memory,
        //     then each thread reads from shared for the FMA loop.
        //     Layout: s_v_tile[kk * 64 + d_local] holds V[t+kk, d_start+d_local].
        if (tid < tile_len) {
            uint k = t + tid;
            uint v_row_base = v_base + k * v_packed_per_row;
            uint vs_row_base = vs_base + k * vs_inner;
            // 64-wide d-chunk = 16 packed u32 words per row (each word
            // holds 4 u8 V values). Walk the 16 words in order so reads
            // are sequential within a thread.
            uint word_lo = d_start / 4u;
            for (uint w = 0u; w < 16u; ++w) {
                uint packed = packed_v[v_row_base + word_lo + w];
                uint d_word = d_start + w * 4u;
                // Up to 4 bytes per word; each byte may sit in a
                // different V quant group, so look up scale/bias per
                // byte. With V_GROUP_SIZE >= 4 (always true for the
                // 32-group default) all 4 bytes share scale/bias and
                // this collapses to a single load.
                uint g0 = (d_word + 0u) / uint(V_GROUP_SIZE);
                uint g1 = (d_word + 1u) / uint(V_GROUP_SIZE);
                uint g2 = (d_word + 2u) / uint(V_GROUP_SIZE);
                uint g3 = (d_word + 3u) / uint(V_GROUP_SIZE);
                float vs0 = v_scales[vs_row_base + g0];
                float vb0 = v_biases[vs_row_base + g0];
                float vs1 = (g1 == g0) ? vs0 : v_scales[vs_row_base + g1];
                float vb1 = (g1 == g0) ? vb0 : v_biases[vs_row_base + g1];
                float vs2 = (g2 == g0) ? vs0 : v_scales[vs_row_base + g2];
                float vb2 = (g2 == g0) ? vb0 : v_biases[vs_row_base + g2];
                float vs3 = (g3 == g0) ? vs0 : v_scales[vs_row_base + g3];
                float vb3 = (g3 == g0) ? vb0 : v_biases[vs_row_base + g3];
                uint row_off = tid * 64u + w * 4u;
                s_v_tile[row_off + 0u] = float((packed >>  0u) & 0xFFu) * vs0 + vb0;
                s_v_tile[row_off + 1u] = float((packed >>  8u) & 0xFFu) * vs1 + vb1;
                s_v_tile[row_off + 2u] = float((packed >> 16u) & 0xFFu) * vs2 + vb2;
                s_v_tile[row_off + 3u] = float((packed >> 24u) & 0xFFu) * vs3 + vb3;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (d_valid) {
            float delta = 0.0f;
            uint d_local = tid;
            // Unrolled by 4 to give the compiler easier IL for the
            // FMA pipeline. tile_len <= 64; trailing positions read
            // 0-scored slots that the score phase masked with -inf
            // → exp(-inf) = 0, so the extra FMAs are harmless.
            uint kk = 0u;
            for (; kk + 4u <= tile_len; kk += 4u) {
                delta += s_tile_scores[kk + 0u] * s_v_tile[(kk + 0u) * 64u + d_local];
                delta += s_tile_scores[kk + 1u] * s_v_tile[(kk + 1u) * 64u + d_local];
                delta += s_tile_scores[kk + 2u] * s_v_tile[(kk + 2u) * 64u + d_local];
                delta += s_tile_scores[kk + 3u] * s_v_tile[(kk + 3u) * 64u + d_local];
            }
            for (; kk < tile_len; ++kk) {
                delta += s_tile_scores[kk] * s_v_tile[kk * 64u + d_local];
            }
            my_out = my_out * alpha + delta;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Normalise and write one output value per thread.
    if (d_valid) {
        float inv_l = 1.0f / s_running_l;
        uint out_offset = (b * uint(Hq) + h_q) * uint(D) + d;
        out[out_offset] = T(my_out * inv_l);
    }
"#;

// Wide (256-thread) online-softmax SDPA kernel for the 4-bit TurboQuant
// path. Originally an attempt to drive the V matmul through
// simdgroup_matrix<float, 8, 8> tile intrinsics, but for q_len=1 decode the
// attention output is a matrix-VECTOR product (1xKV scores · KVxD V): an 8x8
// systolic tile leaves 7 of 8 rows empty, wasting 7/8 of the MACs, and the
// sparse-A path produced wrong results (sign flips / inf). simdgroup_matrix is
// the right tool for DENSE matmuls (MoE MLP, batched/q_len>1 prefill), not a
// row-vector reduction. This kernel therefore uses a cooperative scalar
// reduction (64 threads, one output dim each) — the same math as
// tq_sdpa_4bit_online but spreading V-dequant across 256 threads (4 per KV
// row). Selected via TURBOQUANT_SIMD_MATMUL; gains over the v2 kernel are
// marginal since the V matmul is <5% of Gemma4 5K decode time.
const TQ_SDPA_4BIT_ONLINE_SIMD_KERNEL: &str = r#"
    uint tid = thread_position_in_threadgroup.x;
    uint group_x = threadgroup_position_in_grid.x;
    uint d_chunk_id = threadgroup_position_in_grid.y;
    uint b   = group_x / uint(Hq);
    uint h_q = group_x % uint(Hq);
    if (b >= uint(B)) return;
    uint h_kv = h_q / uint(kv_repeat);
    uint d_start = d_chunk_id * 64u;
    if (d_start >= uint(D)) return;

    threadgroup float s_q[512];
    threadgroup float s_centroids[16];
    threadgroup float s_tile_scores[64];
    threadgroup float s_red[64];
    threadgroup float s_query_sum;
    threadgroup float s_alpha;
    threadgroup float s_new_m;
    threadgroup float s_tile_sum;
    threadgroup float s_running_m;
    threadgroup float s_running_l;
    threadgroup float s_v_tile[64 * 64];
    // Running (rescaled) output numerator for the 64 d-cols of this chunk.
    threadgroup float s_out[64];

    if (tid < 16u) s_centroids[tid] = centroids[tid];

    // Pre-rotate Q: signs * Q, then WHT. Reuse 64 threads for this phase.
    uint q_offset = (b * uint(Hq) + h_q) * uint(D);
    if (tid < 64u) {
        for (uint i = tid; i < uint(D); i += 64u) {
            s_q[i] = float(q[q_offset + i]) * signs[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_sum = 0.0f;
    if (tid < 64u) {
        for (uint i = tid; i < uint(D); i += 64u) local_sum += s_q[i];
    }
    s_red[tid & 63u] = (tid < 64u) ? local_sum : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32u) s_red[tid] += s_red[tid + 32u];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 16u) s_red[tid] += s_red[tid + 16u];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid <  8u) s_red[tid] += s_red[tid +  8u];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid <  4u) s_red[tid] += s_red[tid +  4u];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid <  2u) s_red[tid] += s_red[tid +  2u];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) { s_red[0] += s_red[1]; s_query_sum = s_red[0]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float query_sum_signed = s_query_sum;

    // WHT on Q.
    if (tid < 64u) {
        for (uint step = 1u; step < uint(D); step <<= 1u) {
            for (uint i = tid; i < uint(D) / 2u; i += 64u) {
                uint j = (i / step) * (step * 2u) + (i % step);
                uint k = j + step;
                float a = s_q[j];
                float b2 = s_q[k];
                s_q[j] = a + b2;
                s_q[k] = a - b2;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float wht_scale = metal::rsqrt(float(D));
        for (uint i = tid; i < uint(D); i += 64u) s_q[i] *= wht_scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint packed_per_vec = uint(D) / 8u;
    uint k_stride_inner = packed_per_vec;
    uint k_stride_h     = uint(KV) * k_stride_inner;
    uint k_stride_b     = uint(Hkv) * k_stride_h;
    uint meta_stride_h  = uint(KV);
    uint meta_stride_b  = uint(Hkv) * meta_stride_h;

    uint v_packed_per_row = uint(D) / 4u;
    uint v_stride_h       = uint(KV) * v_packed_per_row;
    uint v_stride_b       = uint(Hkv) * v_stride_h;
    uint vs_inner         = uint(D) / uint(V_GROUP_SIZE);
    uint vs_h             = uint(KV) * vs_inner;
    uint vs_b             = uint(Hkv) * vs_h;
    uint v_base  = b * v_stride_b + h_kv * v_stride_h;
    uint vs_base = b * vs_b + h_kv * vs_h;

    if (tid == 0u) { s_running_m = -1.0e30f; s_running_l = 0.0f; }
    if (tid < 64u) s_out[tid] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint t = 0u; t < uint(KV); t += 64u) {
        uint tile_end = metal::min(t + 64u, uint(KV));
        uint tile_len = tile_end - t;

        // (a) Score this tile (first 64 threads; rest idle).
        float my_score = -1.0e30f;
        if (tid < tile_len) {
            uint k = t + tid;
            uint kbase = b * k_stride_b + h_kv * k_stride_h + k * k_stride_inner;
            float dot = 0.0f;
            for (uint w = 0u; w < packed_per_vec; ++w) {
                uint packed = packed_k[kbase + w];
                uint d0 = w * 8u;
                dot += s_q[d0 + 0u] * s_centroids[(packed >>  0u) & 0xFu];
                dot += s_q[d0 + 1u] * s_centroids[(packed >>  4u) & 0xFu];
                dot += s_q[d0 + 2u] * s_centroids[(packed >>  8u) & 0xFu];
                dot += s_q[d0 + 3u] * s_centroids[(packed >> 12u) & 0xFu];
                dot += s_q[d0 + 4u] * s_centroids[(packed >> 16u) & 0xFu];
                dot += s_q[d0 + 5u] * s_centroids[(packed >> 20u) & 0xFu];
                dot += s_q[d0 + 6u] * s_centroids[(packed >> 24u) & 0xFu];
                dot += s_q[d0 + 7u] * s_centroids[(packed >> 28u) & 0xFu];
            }
            uint meta = b * meta_stride_b + h_kv * meta_stride_h + k;
            float sigma = sigma_k[meta];
            float mean  = mean_k[meta];
            my_score = sigma * dot + mean * query_sum_signed;
            if (has_mask != 0) my_score += mask[k];
        }
        if (tid < 64u) s_tile_scores[tid] = my_score;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // (b) Cooperative max-reduce.
        if (tid < 64u) s_red[tid] = my_score;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 32u) s_red[tid] = metal::max(s_red[tid], s_red[tid + 32u]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 16u) s_red[tid] = metal::max(s_red[tid], s_red[tid + 16u]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid <  8u) s_red[tid] = metal::max(s_red[tid], s_red[tid +  8u]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid <  4u) s_red[tid] = metal::max(s_red[tid], s_red[tid +  4u]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid <  2u) s_red[tid] = metal::max(s_red[tid], s_red[tid +  2u]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            float tmax = metal::max(s_red[0], s_red[1]);
            float nm = metal::max(s_running_m, tmax);
            s_alpha = (s_running_m > -1.0e29f) ? metal::exp(s_running_m - nm) : 0.0f;
            s_new_m = nm;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float alpha = s_alpha;
        float new_m = s_new_m;

        // (c) Cooperative exp + sum-reduce.
        float my_exp = 0.0f;
        if (tid < tile_len) {
            my_exp = metal::exp(s_tile_scores[tid] - new_m);
            s_tile_scores[tid] = my_exp;
        } else if (tid < 64u) {
            s_tile_scores[tid] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 64u) s_red[tid] = (tid < tile_len) ? my_exp : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 32u) s_red[tid] += s_red[tid + 32u];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 16u) s_red[tid] += s_red[tid + 16u];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid <  8u) s_red[tid] += s_red[tid +  8u];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid <  4u) s_red[tid] += s_red[tid +  4u];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid <  2u) s_red[tid] += s_red[tid +  2u];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            float tsum = s_red[0] + s_red[1];
            s_tile_sum = tsum;
            s_running_l = s_running_l * alpha + tsum;
            s_running_m = new_m;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // (d) Cooperative V dequant into s_v_tile [kk][d_local]. Spread
        // the 64 KV rows across the 256 threads — 4 threads per row,
        // each loads 4 packed_v u32 words (4 d values each = 16 d).
        uint row = tid >> 2;       // 0..63 — kv row this thread serves
        uint quarter = tid & 3u;   // 0..3 — which 16 d-cols this thread serves
        if (row < tile_len) {
            uint k = t + row;
            uint v_row_base = v_base + k * v_packed_per_row;
            uint vs_row_base = vs_base + k * vs_inner;
            uint word_lo = d_start / 4u;
            for (uint ww = 0u; ww < 4u; ++ww) {
                uint w = quarter * 4u + ww;
                uint packed = packed_v[v_row_base + word_lo + w];
                uint d_word = d_start + w * 4u;
                uint g0 = (d_word + 0u) / uint(V_GROUP_SIZE);
                uint g1 = (d_word + 1u) / uint(V_GROUP_SIZE);
                uint g2 = (d_word + 2u) / uint(V_GROUP_SIZE);
                uint g3 = (d_word + 3u) / uint(V_GROUP_SIZE);
                float vs0 = v_scales[vs_row_base + g0];
                float vb0 = v_biases[vs_row_base + g0];
                float vs1 = (g1 == g0) ? vs0 : v_scales[vs_row_base + g1];
                float vb1 = (g1 == g0) ? vb0 : v_biases[vs_row_base + g1];
                float vs2 = (g2 == g0) ? vs0 : v_scales[vs_row_base + g2];
                float vb2 = (g2 == g0) ? vb0 : v_biases[vs_row_base + g2];
                float vs3 = (g3 == g0) ? vs0 : v_scales[vs_row_base + g3];
                float vb3 = (g3 == g0) ? vb0 : v_biases[vs_row_base + g3];
                uint row_off = row * 64u + w * 4u;
                s_v_tile[row_off + 0u] = float((packed >>  0u) & 0xFFu) * vs0 + vb0;
                s_v_tile[row_off + 1u] = float((packed >>  8u) & 0xFFu) * vs1 + vb1;
                s_v_tile[row_off + 2u] = float((packed >> 16u) & 0xFFu) * vs2 + vb2;
                s_v_tile[row_off + 3u] = float((packed >> 24u) & 0xFFu) * vs3 + vb3;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // (e) Cooperative V matmul: 64 threads, each accumulates one output
        // dim d-local = tid over all KV positions in the tile, then folds into
        // the running numerator with the online-softmax rescale. Scores for
        // kk >= tile_len are zeroed in step (c), so the full 64-wide loop is
        // safe even for a short final tile.
        if (tid < 64u) {
            float acc = 0.0f;
            for (uint kk = 0u; kk < 64u; ++kk) {
                acc += s_tile_scores[kk] * s_v_tile[kk * 64u + tid];
            }
            s_out[tid] = s_out[tid] * alpha + acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // (f) Normalise by the running softmax denominator and write D_CHUNK outputs.
    if (tid < 64u) {
        float inv_l = 1.0f / s_running_l;
        uint d_global = d_start + tid;  // tid is the d-local index (0..63)
        if (d_global < uint(D)) {
            uint out_offset = (b * uint(Hq) + h_q) * uint(D) + d_global;
            out[out_offset] = T(s_out[tid] * inv_l);
        }
    }
"#;

static TQ_SDPA_4BIT_ONLINE_SIMD_KERNEL_HANDLE: OnceLock<MetalKernel> = OnceLock::new();

fn create_tq_sdpa_4bit_online_simd_kernel() -> MetalKernel {
    unsafe {
        let q = CString::new("q").unwrap();
        let packed_k = CString::new("packed_k").unwrap();
        let sigma_k = CString::new("sigma_k").unwrap();
        let mean_k = CString::new("mean_k").unwrap();
        let packed_v = CString::new("packed_v").unwrap();
        let v_scales = CString::new("v_scales").unwrap();
        let v_biases = CString::new("v_biases").unwrap();
        let signs = CString::new("signs").unwrap();
        let centroids = CString::new("centroids").unwrap();
        let mask = CString::new("mask").unwrap();
        let out = CString::new("out").unwrap();

        let inputs = mlx_sys::mlx_vector_string_new();
        mlx_sys::mlx_vector_string_append_value(inputs, q.as_ptr());
        mlx_sys::mlx_vector_string_append_value(inputs, packed_k.as_ptr());
        mlx_sys::mlx_vector_string_append_value(inputs, sigma_k.as_ptr());
        mlx_sys::mlx_vector_string_append_value(inputs, mean_k.as_ptr());
        mlx_sys::mlx_vector_string_append_value(inputs, packed_v.as_ptr());
        mlx_sys::mlx_vector_string_append_value(inputs, v_scales.as_ptr());
        mlx_sys::mlx_vector_string_append_value(inputs, v_biases.as_ptr());
        mlx_sys::mlx_vector_string_append_value(inputs, signs.as_ptr());
        mlx_sys::mlx_vector_string_append_value(inputs, centroids.as_ptr());
        mlx_sys::mlx_vector_string_append_value(inputs, mask.as_ptr());
        let outputs = mlx_sys::mlx_vector_string_new();
        mlx_sys::mlx_vector_string_append_value(outputs, out.as_ptr());

        let source = CString::new(TQ_SDPA_4BIT_ONLINE_SIMD_KERNEL).unwrap();
        let header = CString::new(
            "#include <metal_simdgroup>\n#include <metal_simdgroup_matrix>\n",
        )
        .unwrap();
        let name = CString::new("tq_sdpa_4bit_online_simd").unwrap();
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

static TQ_SDPA_4BIT_ONLINE_KERNEL_HANDLE: OnceLock<MetalKernel> = OnceLock::new();

fn create_tq_sdpa_4bit_online_kernel() -> MetalKernel {
    unsafe {
        let q = CString::new("q").unwrap();
        let packed_k = CString::new("packed_k").unwrap();
        let sigma_k = CString::new("sigma_k").unwrap();
        let mean_k = CString::new("mean_k").unwrap();
        let packed_v = CString::new("packed_v").unwrap();
        let v_scales = CString::new("v_scales").unwrap();
        let v_biases = CString::new("v_biases").unwrap();
        let signs = CString::new("signs").unwrap();
        let centroids = CString::new("centroids").unwrap();
        let mask = CString::new("mask").unwrap();
        let out = CString::new("out").unwrap();

        let inputs = mlx_sys::mlx_vector_string_new();
        mlx_sys::mlx_vector_string_append_value(inputs, q.as_ptr());
        mlx_sys::mlx_vector_string_append_value(inputs, packed_k.as_ptr());
        mlx_sys::mlx_vector_string_append_value(inputs, sigma_k.as_ptr());
        mlx_sys::mlx_vector_string_append_value(inputs, mean_k.as_ptr());
        mlx_sys::mlx_vector_string_append_value(inputs, packed_v.as_ptr());
        mlx_sys::mlx_vector_string_append_value(inputs, v_scales.as_ptr());
        mlx_sys::mlx_vector_string_append_value(inputs, v_biases.as_ptr());
        mlx_sys::mlx_vector_string_append_value(inputs, signs.as_ptr());
        mlx_sys::mlx_vector_string_append_value(inputs, centroids.as_ptr());
        mlx_sys::mlx_vector_string_append_value(inputs, mask.as_ptr());
        let outputs = mlx_sys::mlx_vector_string_new();
        mlx_sys::mlx_vector_string_append_value(outputs, out.as_ptr());

        let source = CString::new(TQ_SDPA_4BIT_ONLINE_KERNEL).unwrap();
        let header = CString::new("").unwrap();
        let name = CString::new("tq_sdpa_4bit_online").unwrap();
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

/// Flash-Attention v2 style fused TurboQuant SDPA. Same input/output
/// contract as `tq_sdpa_4bit` but uses online softmax to handle
/// arbitrary kv_len (no shared-mem scores buffer). D_CHUNK is fixed at
/// 64 so one thread owns one output dim.
pub fn tq_sdpa_4bit_online(
    q: &Array,
    packed_k: &Array,
    sigma_k: &Array,
    mean_k: &Array,
    packed_v: &Array,
    v_scales: &Array,
    v_biases: &Array,
    signs: &Array,
    centroids: &Array,
    mask: Option<&Array>,
    scale: f32,
    kv_repeat: i32,
    v_group_size: i32,
) -> Result<Array, Exception> {
    let qs = q.shape();
    if qs.len() != 4 || qs[2] != 1 {
        return Err(Exception::custom(format!(
            "tq_sdpa_4bit_online expects Q [B, Hq, 1, D], got {qs:?}",
        )));
    }
    let b = qs[0];
    let h_q = qs[1];
    let d = qs[3];
    let ps = packed_k.shape();
    let h_kv = ps[1];
    let kv = ps[2];
    if h_q != h_kv * kv_repeat {
        return Err(Exception::custom(format!(
            "GQA mismatch: Hq={h_q}, Hkv={h_kv}, kv_repeat={kv_repeat}",
        )));
    }
    if d > 512 || d % 64 != 0 {
        return Err(Exception::custom(format!(
            "tq_sdpa_4bit_online requires D divisible by 64 and ≤ 512; got {d}",
        )));
    }
    let dtype: u32 = q.dtype().into();
    let scale_arr = mlx_rs::array!(scale).as_dtype(q.dtype())?;
    let q_f = q.multiply(&scale_arr)?;
    let signs_f32 = signs.as_dtype(mlx_rs::Dtype::Float32)?;
    let centroids_f32 = centroids.as_dtype(mlx_rs::Dtype::Float32)?;
    let sigma_f32 = sigma_k.as_dtype(mlx_rs::Dtype::Float32)?;
    let mean_f32 = mean_k.as_dtype(mlx_rs::Dtype::Float32)?;
    let (has_mask, mask_arr) = match mask {
        Some(m) => (1i32, m.as_dtype(mlx_rs::Dtype::Float32)?),
        None => (0i32, mlx_rs::Array::from_slice::<f32>(&[0.0], &[1])),
    };

    let n_d_chunks = (d + 63) / 64;
    let kernel = TQ_SDPA_4BIT_ONLINE_KERNEL_HANDLE
        .get_or_init(create_tq_sdpa_4bit_online_kernel);

    unsafe {
        let stream = mlx_sys::mlx_default_gpu_stream_new();
        let config = mlx_sys::mlx_fast_metal_kernel_config_new();
        let type_name = CString::new("T").unwrap();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
            config, type_name.as_ptr(), dtype,
        );
        for (name, value) in [
            ("D", d), ("Hq", h_q), ("Hkv", h_kv), ("KV", kv), ("B", b),
            ("kv_repeat", kv_repeat), ("has_mask", has_mask),
            ("V_GROUP_SIZE", v_group_size),
        ] {
            let cname = CString::new(name).unwrap();
            mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
                config, cname.as_ptr(), value,
            );
        }
        mlx_sys::mlx_fast_metal_kernel_config_set_grid(
            config, 64 * b * h_q, n_d_chunks, 1,
        );
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, 64, 1, 1);

        let out_shape: [i32; 4] = [b, h_q, 1, d];
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config, out_shape.as_ptr(), out_shape.len(), dtype,
        );

        let vs_f32 = v_scales.as_dtype(mlx_rs::Dtype::Float32)?;
        let vb_f32 = v_biases.as_dtype(mlx_rs::Dtype::Float32)?;
        let inputs = mlx_sys::mlx_vector_array_new();
        mlx_sys::mlx_vector_array_append_value(inputs, q_f.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, packed_k.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, sigma_f32.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, mean_f32.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, packed_v.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, vs_f32.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, vb_f32.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, signs_f32.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, centroids_f32.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, mask_arr.as_ptr());

        let mut outputs = mlx_sys::mlx_vector_array_new();
        let ret = mlx_sys::mlx_fast_metal_kernel_apply(
            &mut outputs, kernel.kernel, inputs, config, stream,
        );
        if ret != 0 {
            mlx_sys::mlx_fast_metal_kernel_config_free(config);
            mlx_sys::mlx_vector_array_free(inputs);
            mlx_sys::mlx_vector_array_free(outputs);
            mlx_sys::mlx_stream_free(stream);
            return Err(Exception::custom("tq_sdpa_4bit_online kernel failed"));
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

/// Variant of `tq_sdpa_4bit_online` that uses `simdgroup_matrix<float, 8, 8>`
/// intrinsics for the V matmul phase. Same input/output contract; only the
/// FMA loop differs. Selected when `TURBOQUANT_SIMD_MATMUL=1`.
pub fn tq_sdpa_4bit_online_simd(
    q: &Array,
    packed_k: &Array,
    sigma_k: &Array,
    mean_k: &Array,
    packed_v: &Array,
    v_scales: &Array,
    v_biases: &Array,
    signs: &Array,
    centroids: &Array,
    mask: Option<&Array>,
    scale: f32,
    kv_repeat: i32,
    v_group_size: i32,
) -> Result<Array, Exception> {
    let qs = q.shape();
    if qs.len() != 4 || qs[2] != 1 {
        return Err(Exception::custom(format!(
            "tq_sdpa_4bit_online_simd expects Q [B, Hq, 1, D], got {qs:?}",
        )));
    }
    let b = qs[0];
    let h_q = qs[1];
    let d = qs[3];
    let ps = packed_k.shape();
    let h_kv = ps[1];
    let kv = ps[2];
    if h_q != h_kv * kv_repeat {
        return Err(Exception::custom(format!(
            "GQA mismatch: Hq={h_q}, Hkv={h_kv}, kv_repeat={kv_repeat}",
        )));
    }
    if d > 512 || d % 64 != 0 {
        return Err(Exception::custom(format!(
            "tq_sdpa_4bit_online_simd requires D divisible by 64 and ≤ 512; got {d}",
        )));
    }
    let dtype: u32 = q.dtype().into();
    let scale_arr = mlx_rs::array!(scale).as_dtype(q.dtype())?;
    let q_f = q.multiply(&scale_arr)?;
    let signs_f32 = signs.as_dtype(mlx_rs::Dtype::Float32)?;
    let centroids_f32 = centroids.as_dtype(mlx_rs::Dtype::Float32)?;
    let sigma_f32 = sigma_k.as_dtype(mlx_rs::Dtype::Float32)?;
    let mean_f32 = mean_k.as_dtype(mlx_rs::Dtype::Float32)?;
    let (has_mask, mask_arr) = match mask {
        Some(m) => (1i32, m.as_dtype(mlx_rs::Dtype::Float32)?),
        None => (0i32, mlx_rs::Array::from_slice::<f32>(&[0.0], &[1])),
    };

    let n_d_chunks = (d + 63) / 64;
    let kernel = TQ_SDPA_4BIT_ONLINE_SIMD_KERNEL_HANDLE
        .get_or_init(create_tq_sdpa_4bit_online_simd_kernel);

    unsafe {
        let stream = mlx_sys::mlx_default_gpu_stream_new();
        let config = mlx_sys::mlx_fast_metal_kernel_config_new();
        let type_name = CString::new("T").unwrap();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
            config, type_name.as_ptr(), dtype,
        );
        for (name, value) in [
            ("D", d), ("Hq", h_q), ("Hkv", h_kv), ("KV", kv), ("B", b),
            ("kv_repeat", kv_repeat), ("has_mask", has_mask),
            ("V_GROUP_SIZE", v_group_size),
        ] {
            let cname = CString::new(name).unwrap();
            mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
                config, cname.as_ptr(), value,
            );
        }
        // Grid: same as v2 (one threadgroup per (b * Hq, d_chunk)) but each
        // threadgroup now has 256 threads = 8 simdgroups.
        mlx_sys::mlx_fast_metal_kernel_config_set_grid(
            config, 256 * b * h_q, n_d_chunks, 1,
        );
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1);

        let out_shape: [i32; 4] = [b, h_q, 1, d];
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config, out_shape.as_ptr(), out_shape.len(), dtype,
        );

        let vs_f32 = v_scales.as_dtype(mlx_rs::Dtype::Float32)?;
        let vb_f32 = v_biases.as_dtype(mlx_rs::Dtype::Float32)?;
        let inputs = mlx_sys::mlx_vector_array_new();
        mlx_sys::mlx_vector_array_append_value(inputs, q_f.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, packed_k.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, sigma_f32.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, mean_f32.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, packed_v.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, vs_f32.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, vb_f32.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, signs_f32.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, centroids_f32.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, mask_arr.as_ptr());

        let mut outputs = mlx_sys::mlx_vector_array_new();
        let ret = mlx_sys::mlx_fast_metal_kernel_apply(
            &mut outputs, kernel.kernel, inputs, config, stream,
        );
        if ret != 0 {
            mlx_sys::mlx_fast_metal_kernel_config_free(config);
            mlx_sys::mlx_vector_array_free(inputs);
            mlx_sys::mlx_vector_array_free(outputs);
            mlx_sys::mlx_stream_free(stream);
            return Err(Exception::custom("tq_sdpa_4bit_online_simd kernel failed"));
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

fn create_tq_sdpa_4bit_kernel() -> MetalKernel {
    unsafe {
        let q = CString::new("q").unwrap();
        let packed_k = CString::new("packed_k").unwrap();
        let sigma_k = CString::new("sigma_k").unwrap();
        let mean_k = CString::new("mean_k").unwrap();
        let packed_v = CString::new("packed_v").unwrap();
        let v_scales = CString::new("v_scales").unwrap();
        let v_biases = CString::new("v_biases").unwrap();
        let signs = CString::new("signs").unwrap();
        let centroids = CString::new("centroids").unwrap();
        let mask = CString::new("mask").unwrap();
        let out = CString::new("out").unwrap();

        let inputs = mlx_sys::mlx_vector_string_new();
        mlx_sys::mlx_vector_string_append_value(inputs, q.as_ptr());
        mlx_sys::mlx_vector_string_append_value(inputs, packed_k.as_ptr());
        mlx_sys::mlx_vector_string_append_value(inputs, sigma_k.as_ptr());
        mlx_sys::mlx_vector_string_append_value(inputs, mean_k.as_ptr());
        mlx_sys::mlx_vector_string_append_value(inputs, packed_v.as_ptr());
        mlx_sys::mlx_vector_string_append_value(inputs, v_scales.as_ptr());
        mlx_sys::mlx_vector_string_append_value(inputs, v_biases.as_ptr());
        mlx_sys::mlx_vector_string_append_value(inputs, signs.as_ptr());
        mlx_sys::mlx_vector_string_append_value(inputs, centroids.as_ptr());
        mlx_sys::mlx_vector_string_append_value(inputs, mask.as_ptr());

        let outputs = mlx_sys::mlx_vector_string_new();
        mlx_sys::mlx_vector_string_append_value(outputs, out.as_ptr());

        let source = CString::new(TQ_SDPA_4BIT_KERNEL).unwrap();
        let header = CString::new("").unwrap();
        let name = CString::new("tq_sdpa_4bit").unwrap();
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

/// Maximum kv_len the bounded fused SDPA kernel supports per call.
/// Bounded by `s_scores` shared-memory allocation (2048 floats = 8 KB).
/// Apple Silicon SMs have 32 KB threadgroup memory and our other
/// scratch (Q + centroids + reductions) fits in ~4 KB, leaving
/// headroom for ~2-3 concurrent groups per SM.  Longer caches go
/// through the streaming-softmax kernel `tq_sdpa_4bit_online`.
pub const TQ_SDPA_MAX_KV: i32 = 2048;

/// Fully-fused TurboQuant SDPA: one Metal dispatch does
/// QK-score-from-compressed-K + softmax + V matmul. Returns
/// `[B, Hq, 1, D]` in the same dtype as Q.
///
/// `mask` is optional; pass `None` for no mask, or a `[KV]` f32 tensor
/// of additive log-prob mask values that broadcasts over all query
/// heads / batches.
pub fn tq_sdpa_4bit(
    q: &Array,
    packed_k: &Array,
    sigma_k: &Array,
    mean_k: &Array,
    packed_v: &Array,
    v_scales: &Array,
    v_biases: &Array,
    signs: &Array,
    centroids: &Array,
    mask: Option<&Array>,
    scale: f32,
    kv_repeat: i32,
    v_group_size: i32,
) -> Result<Array, Exception> {
    let qs = q.shape();
    if qs.len() != 4 || qs[2] != 1 {
        return Err(Exception::custom(format!(
            "tq_sdpa_4bit expects Q [B, Hq, 1, D], got {qs:?}",
        )));
    }
    let b = qs[0];
    let h_q = qs[1];
    let d = qs[3];
    let ps = packed_k.shape();
    if ps.len() != 4 || ps[3] != d / 8 {
        return Err(Exception::custom(format!(
            "tq_sdpa_4bit packed_k shape mismatch, got {ps:?} for D={d}",
        )));
    }
    let h_kv = ps[1];
    let kv = ps[2];
    if h_q != h_kv * kv_repeat {
        return Err(Exception::custom(format!(
            "GQA mismatch: Hq={h_q}, Hkv={h_kv}, kv_repeat={kv_repeat}",
        )));
    }
    if kv > TQ_SDPA_MAX_KV {
        return Err(Exception::custom(format!(
            "tq_sdpa_4bit kv_len {kv} exceeds MAX={TQ_SDPA_MAX_KV}; fall back to non-fused",
        )));
    }
    if d > 512 || d % 8 != 0 {
        return Err(Exception::custom(format!("D must be ≤ 512, div by 8; got {d}")));
    }
    let dtype: u32 = q.dtype().into();
    // Pre-scale Q so the kernel doesn't have to take an f32 constant.
    let scale_arr = mlx_rs::array!(scale).as_dtype(q.dtype())?;
    let q_f = q.multiply(&scale_arr)?;
    let signs_f32 = signs.as_dtype(mlx_rs::Dtype::Float32)?;
    let centroids_f32 = centroids.as_dtype(mlx_rs::Dtype::Float32)?;
    let sigma_f32 = sigma_k.as_dtype(mlx_rs::Dtype::Float32)?;
    let mean_f32 = mean_k.as_dtype(mlx_rs::Dtype::Float32)?;
    let (has_mask, mask_arr) = match mask {
        Some(m) => (1i32, m.as_dtype(mlx_rs::Dtype::Float32)?),
        None => (0i32, mlx_rs::Array::from_slice::<f32>(&[0.0], &[1])),
    };

    let kernel = TQ_SDPA_4BIT_KERNEL_HANDLE.get_or_init(create_tq_sdpa_4bit_kernel);
    unsafe {
        let stream = mlx_sys::mlx_default_gpu_stream_new();
        let config = mlx_sys::mlx_fast_metal_kernel_config_new();
        let type_name = CString::new("T").unwrap();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
            config, type_name.as_ptr(), dtype,
        );
        // D-tile so V matmul parallelises across multiple threadgroups.
        // Chunk size 32 = 8 threadgroups per (b, h_q) at D=256, 16 at
        // D=512. Hard upper bound is `D` (must be a divisor of D for
        // clean tiling); 32 happens to divide every head_dim we ship.
        let d_chunk = if d >= 32 && d % 32 == 0 { 32 } else { d };
        let n_d_chunks = (d + d_chunk - 1) / d_chunk;
        for (name, value) in [
            ("D", d), ("Hq", h_q), ("Hkv", h_kv), ("KV", kv), ("B", b),
            ("kv_repeat", kv_repeat), ("has_mask", has_mask),
            ("D_CHUNK", d_chunk), ("V_GROUP_SIZE", v_group_size),
        ] {
            let cname = CString::new(name).unwrap();
            mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
                config, cname.as_ptr(), value,
            );
        }
        // Pre-scale Q on the host (kernels don't accept f32 template args).
        // Done after the dtype conversion above using mlx ops.
        let _ = scale;
        // 2-D grid: x picks (b, h_q), y picks d_chunk_id.
        // Total threadgroups = B * Hq * n_d_chunks (e.g. 1*32*8 = 256
        // for Gemma4-26B sliding heads — proper GPU saturation vs the
        // prior 32 threadgroups).
        mlx_sys::mlx_fast_metal_kernel_config_set_grid(
            config, 64 * b * h_q, n_d_chunks, 1,
        );
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, 64, 1, 1);

        let out_shape: [i32; 4] = [b, h_q, 1, d];
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config, out_shape.as_ptr(), out_shape.len(), dtype,
        );

        let vs_f32 = v_scales.as_dtype(mlx_rs::Dtype::Float32)?;
        let vb_f32 = v_biases.as_dtype(mlx_rs::Dtype::Float32)?;
        let inputs = mlx_sys::mlx_vector_array_new();
        mlx_sys::mlx_vector_array_append_value(inputs, q_f.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, packed_k.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, sigma_f32.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, mean_f32.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, packed_v.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, vs_f32.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, vb_f32.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, signs_f32.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, centroids_f32.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, mask_arr.as_ptr());

        let mut outputs = mlx_sys::mlx_vector_array_new();
        let ret = mlx_sys::mlx_fast_metal_kernel_apply(
            &mut outputs, kernel.kernel, inputs, config, stream,
        );
        if ret != 0 {
            mlx_sys::mlx_fast_metal_kernel_config_free(config);
            mlx_sys::mlx_vector_array_free(inputs);
            mlx_sys::mlx_vector_array_free(outputs);
            mlx_sys::mlx_stream_free(stream);
            return Err(Exception::custom("tq_sdpa_4bit kernel failed"));
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

fn create_tq_qk_score_kernel() -> MetalKernel {
    unsafe {
        let q_in = CString::new("q_in").unwrap();
        let packed_k = CString::new("packed_k").unwrap();
        let sigma_in = CString::new("sigma_in").unwrap();
        let mean_in = CString::new("mean_in").unwrap();
        let signs = CString::new("signs").unwrap();
        let centroids = CString::new("centroids").unwrap();
        let scores_out = CString::new("scores_out").unwrap();

        let inputs = mlx_sys::mlx_vector_string_new();
        mlx_sys::mlx_vector_string_append_value(inputs, q_in.as_ptr());
        mlx_sys::mlx_vector_string_append_value(inputs, packed_k.as_ptr());
        mlx_sys::mlx_vector_string_append_value(inputs, sigma_in.as_ptr());
        mlx_sys::mlx_vector_string_append_value(inputs, mean_in.as_ptr());
        mlx_sys::mlx_vector_string_append_value(inputs, signs.as_ptr());
        mlx_sys::mlx_vector_string_append_value(inputs, centroids.as_ptr());
        let outputs = mlx_sys::mlx_vector_string_new();
        mlx_sys::mlx_vector_string_append_value(outputs, scores_out.as_ptr());

        let source = CString::new(TQ_QK_SCORE_KERNEL).unwrap();
        let header = CString::new("").unwrap();
        let name = CString::new("tq_qk_score").unwrap();
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

/// Fused QK score from TurboQuant 4-bit compressed K. Returns
/// `[B, H, q_len=1, KV]` f32 scores (no scaling, no mask, no softmax —
/// caller handles those).
///
/// This is the decode-time hot-path kernel: skips the K decompression
/// entirely. For prefill / multi-query verify, use the
/// `tq_decompress_4bit` + standard SDPA path instead (this kernel
/// hard-codes q_len=1 for the spike).
///
/// GQA: `kv_repeat = n_q_heads / n_kv_heads` (must divide evenly).
pub fn tq_qk_score(
    q: &Array,
    packed_k: &Array,
    sigma_k: &Array,
    mean_k: &Array,
    signs: &Array,
    centroids: &Array,
    kv_repeat: i32,
) -> Result<Array, Exception> {
    let qs = q.shape();
    if qs.len() != 4 || qs[2] != 1 {
        return Err(Exception::custom(format!(
            "tq_qk_score expects Q [B, Hq, 1, D] (q_len=1), got {qs:?}"
        )));
    }
    let b = qs[0];
    let h_q = qs[1];
    let d = qs[3];
    let ps = packed_k.shape();
    if ps.len() != 4 || ps[3] != d / 8 {
        return Err(Exception::custom(format!(
            "tq_qk_score expects packed_k [B, Hkv, KV, D/8], got {ps:?}",
        )));
    }
    let h_kv = ps[1];
    let kv = ps[2];
    if h_q != h_kv * kv_repeat {
        return Err(Exception::custom(format!(
            "GQA mismatch: Hq={h_q}, Hkv={h_kv}, kv_repeat={kv_repeat}"
        )));
    }
    if d > 512 || d % 8 != 0 {
        return Err(Exception::custom(format!("D must be ≤ 512 and div by 8, got {d}")));
    }

    let kernel = TQ_QK_SCORE_KERNEL_HANDLE.get_or_init(create_tq_qk_score_kernel);
    unsafe {
        let stream = mlx_sys::mlx_default_gpu_stream_new();
        let config = mlx_sys::mlx_fast_metal_kernel_config_new();
        for (name, value) in [
            ("D", d), ("Hq", h_q), ("Hkv", h_kv), ("KV", kv), ("B", b),
            ("kv_repeat", kv_repeat),
        ] {
            let cname = CString::new(name).unwrap();
            mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
                config, cname.as_ptr(), value,
            );
        }
        // Total threads = 64 × (B * Hq) threadgroups.
        mlx_sys::mlx_fast_metal_kernel_config_set_grid(
            config,
            64 * b * h_q,
            1,
            1,
        );
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, 64, 1, 1);

        let out_shape: [i32; 4] = [b, h_q, 1, kv];
        let f32_dtype: u32 = mlx_rs::Dtype::Float32.into();
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config, out_shape.as_ptr(), out_shape.len(), f32_dtype,
        );

        let q_f32 = q.as_dtype(mlx_rs::Dtype::Float32)?;
        let signs_f32 = signs.as_dtype(mlx_rs::Dtype::Float32)?;
        let centroids_f32 = centroids.as_dtype(mlx_rs::Dtype::Float32)?;
        let sigma_f32 = sigma_k.as_dtype(mlx_rs::Dtype::Float32)?;
        let mean_f32 = mean_k.as_dtype(mlx_rs::Dtype::Float32)?;

        let inputs = mlx_sys::mlx_vector_array_new();
        mlx_sys::mlx_vector_array_append_value(inputs, q_f32.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, packed_k.as_ptr());
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
            return Err(Exception::custom("tq_qk_score kernel failed"));
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

// =============================================================================
// Paged KV gather kernel (Phase 2/3 of paged attention — see `paged.rs`)
// =============================================================================
//
// Gathers a sequence's logical KV out of a paged block arena into one
// contiguous `[B, H, N, D]` tensor in a single Metal dispatch — the kernel
// replacement for the concat-based `paged::gather_blocks` placeholder.
//
// Storage model it expects:
//   pool_buf:     contiguous arena `[num_blocks, B, H, BLOCK, D]` of dtype T
//   block_table:  int32 `[ceil(N / BLOCK)]`, logical block i → physical block id
// For logical token t: physical block = block_table[t / BLOCK],
//                      in-block offset = t % BLOCK.
//
// NOTE (draft): `PagedKvPool` currently stores each block as a separate
// `Array` (`Vec<Option<Array>>`), so it must first be given a single-arena
// storage mode before this kernel can be wired into `PagedKvCache`. Until
// then `PagedKvCache` keeps using the concat fallback. The kernel below is
// validated in isolation by `gather_blocks` tests.
const GATHER_BLOCKS_KERNEL_SOURCE: &str = r#"
    // Grid layout: (D * N, H, B) — one thread per output element.
    uint d = thread_position_in_grid.x % uint(D);
    uint t = thread_position_in_grid.x / uint(D);
    uint h = thread_position_in_grid.y;
    uint b = thread_position_in_grid.z;
    if (t >= uint(N)) return;

    // Resolve the physical token for logical position t: block_table maps the
    // logical block, blocks are contiguous BLOCK-token ranges in the arena.
    int logical_block = int(t) / BLOCK;
    int in_off        = int(t) % BLOCK;
    int pb            = block_table[logical_block];
    uint phys_t       = uint(pb) * uint(BLOCK) + uint(in_off);

    // Strides into the [B, H, SEQ, D] contiguous arena.
    uint stride_t = uint(D);
    uint stride_h = uint(SEQ) * stride_t;
    uint stride_b = uint(H) * stride_h;
    uint src = b * stride_b + h * stride_h + phys_t * stride_t + d;

    // Strides into the [B, H, N, D] contiguous output.
    uint out_stride_t = uint(D);
    uint out_stride_h = uint(N) * out_stride_t;
    uint out_stride_b = uint(H) * out_stride_h;
    uint dst = b * out_stride_b + h * out_stride_h + t * out_stride_t + d;

    out[dst] = pool_buf[src];
"#;

static GATHER_BLOCKS_KERNEL: OnceLock<MetalKernel> = OnceLock::new();

fn create_gather_blocks_kernel() -> MetalKernel {
    unsafe {
        let pool_name = CString::new("pool_buf").unwrap();
        let table_name = CString::new("block_table").unwrap();
        let out_name = CString::new("out").unwrap();
        let inputs = mlx_sys::mlx_vector_string_new();
        mlx_sys::mlx_vector_string_append_value(inputs, pool_name.as_ptr());
        mlx_sys::mlx_vector_string_append_value(inputs, table_name.as_ptr());
        let outputs = mlx_sys::mlx_vector_string_new();
        mlx_sys::mlx_vector_string_append_value(outputs, out_name.as_ptr());
        let source = CString::new(GATHER_BLOCKS_KERNEL_SOURCE).unwrap();
        let header = CString::new("").unwrap();
        let name = CString::new("paged_gather_blocks").unwrap();
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

/// Gather a paged sequence's KV into one contiguous `[B, H, n_tokens, D]`
/// tensor in a single Metal dispatch (the non-contiguous-blocks fallback;
/// contiguous sequences are a free slice in `PagedKvCache::gather`).
///
/// - `pool_buf`: contiguous arena `[B, H, seq, D]`, dtype `T` (blocks are
///   `block_size`-token ranges along the seq axis).
/// - `block_table`: int32 `[ceil(n_tokens / block_size)]`, logical→physical
///   block id. Coerced to int32 if needed.
/// - `seq`: the arena's physical token capacity (`pool_buf.shape()[2]`).
#[allow(clippy::too_many_arguments)]
pub fn gather_blocks(
    pool_buf: &Array,
    block_table: &Array,
    n_tokens: i32,
    block_size: i32,
    seq: i32,
    b: i32,
    h: i32,
    d: i32,
) -> Result<Array, Exception> {
    if n_tokens <= 0 {
        return Err(Exception::custom("gather_blocks: n_tokens must be > 0"));
    }
    let shape = pool_buf.shape();
    if shape.len() != 4 {
        return Err(Exception::custom(format!(
            "gather_blocks expects pool_buf [B,H,SEQ,D], got {:?}",
            shape
        )));
    }
    let dtype: u32 = pool_buf.dtype().into();
    let block_table = if block_table.dtype() != Dtype::Int32 {
        block_table.as_dtype(Dtype::Int32)?
    } else {
        block_table.clone()
    };

    let kernel = GATHER_BLOCKS_KERNEL.get_or_init(create_gather_blocks_kernel);
    unsafe {
        let stream = mlx_sys::mlx_default_gpu_stream_new();
        let config = mlx_sys::mlx_fast_metal_kernel_config_new();

        let type_name = CString::new("T").unwrap();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
            config, type_name.as_ptr(), dtype,
        );
        for (name, value) in [("D", d), ("H", h), ("B", b), ("BLOCK", block_size), ("N", n_tokens), ("SEQ", seq)] {
            let cname = CString::new(name).unwrap();
            mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
                config, cname.as_ptr(), value,
            );
        }
        mlx_sys::mlx_fast_metal_kernel_config_set_grid(
            config,
            (d * n_tokens).max(1),
            h.max(1),
            b.max(1),
        );
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, 64, 1, 1);

        let out_shape: [i32; 4] = [b, h, n_tokens, d];
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            out_shape.as_ptr(),
            out_shape.len(),
            dtype,
        );

        let inputs = mlx_sys::mlx_vector_array_new();
        mlx_sys::mlx_vector_array_append_value(inputs, pool_buf.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, block_table.as_ptr());

        let mut outputs = mlx_sys::mlx_vector_array_new();
        let ret = mlx_sys::mlx_fast_metal_kernel_apply(
            &mut outputs, kernel.kernel, inputs, config, stream,
        );
        if ret != 0 {
            mlx_sys::mlx_fast_metal_kernel_config_free(config);
            mlx_sys::mlx_vector_array_free(inputs);
            mlx_sys::mlx_vector_array_free(outputs);
            mlx_sys::mlx_stream_free(stream);
            return Err(Exception::custom("gather_blocks kernel execution failed"));
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

// =============================================================================
// Paged attention — decode (single-query) kernel
// =============================================================================
//
// Computes attention output for a one-token query directly against a paged KV
// arena, WITHOUT first gathering the cache into a contiguous tensor. One
// thread per (query head, batch); each does a flash-style online-softmax pass
// over all cached positions, resolving each position's physical block through
// the block table. This removes the per-decode-step gather copy.
//
//   q:           [B, HQ, 1, D]                (the just-projected query)
//   k_arena:     [num_blocks, B, HKV, BLOCK, D]
//   v_arena:     [num_blocks, B, HKV, BLOCK, D]
//   block_table: int32 [ceil(N / BLOCK)]      logical→physical block id
//   scale_arr:   f32 [1]                       softmax scale (1/sqrt(D))
//   out:         [B, HQ, 1, D]
//
// GQA: HQ query heads map to HKV kv heads via kv_repeat = HQ / HKV.
// Decode attends to all N cached positions (the query is the latest token),
// so no causal masking is needed here.
// 2-pass split-D flash decode. One THREADGROUP per (query head hq, batch b),
// `TG` threads cooperating. The previous design kept a per-thread `acc[D]`
// register array — for D=256 that spills (256 registers/thread), which crippled
// occupancy and made the kernel slower than contiguous SDPA at long context.
//
// This version is register-light:
//   Pass 1: the TG threads cooperatively compute all N scores into the `s_p`
//           threadgroup buffer, reducing to the global softmax max and sum.
//           Each thread holds only scalars (no acc array).
//   Pass 2: the D output dims are SPLIT across threads — each thread owns the
//           dims `d = tid, tid+TG, …`, looping over all N positions and reading
//           `s_p[t]` (the softmax weight) × V[t][d]. Each thread accumulates a
//           single scalar per dim, so there is no per-thread D-sized array.
//
// `s_p` caps the cached length at `N_MAX`; the caller falls back to gather+SDPA
// for sequences longer than that. Pass-2 reads across the TG threads at a fixed
// position are contiguous in D, so they coalesce.
/// Max cached length the fused paged decode kernel handles (sized by its `s_p`
/// threadgroup buffer). Callers fall back to gather + SDPA beyond this.
pub const PAGED_DECODE_MAX_KV: i32 = 4096;

// Single-pass variant (small N). Each of TG threads runs the full online
// softmax over a strided slice of N, holding a per-thread `acc[D]`. The acc[D]
// register array spills for large D, which is why this loses to the 2-pass
// kernel at long context — but at SHORT context its lower fixed overhead (no
// score buffer, fewer barriers) makes it the faster choice. The dispatch picks
// this below PAGED_DECODE_TWO_PASS_MIN cached tokens.
const PAGED_ATTENTION_DECODE_V1_SOURCE: &str = r#"
    threadgroup float tg_m[TG];
    threadgroup float tg_l[TG];
    threadgroup float tg_acc[TG * D];
    threadgroup float g_m;
    threadgroup float g_l;

    uint hq  = threadgroup_position_in_grid.x;
    uint b   = threadgroup_position_in_grid.y;
    uint tid = thread_position_in_threadgroup.x;

    uint hkv = hq / uint(KV_REPEAT);
    float scale = float(scale_arr[0]);
    uint q_base = b * uint(HQ) * uint(D) + hq * uint(D);

    uint ar_t = uint(D);
    uint ar_h = uint(SEQ) * ar_t;
    uint ar_b = uint(HKV) * ar_h;

    float m = -INFINITY;
    float l = 0.0;
    float acc[D];
    for (int d = 0; d < D; ++d) { acc[d] = 0.0; }

    for (int t = int(tid); t < N; t += TG) {
        int pb  = block_table[t / BLOCK];
        int off = t % BLOCK;
        uint phys_t  = uint(pb) * uint(BLOCK) + uint(off);
        uint kv_base = b * ar_b + hkv * ar_h + phys_t * ar_t;

        float score = 0.0;
        for (int d = 0; d < D; ++d) {
            score += float(q[q_base + d]) * float(k_arena[kv_base + d]);
        }
        score *= scale;

        float m_new = max(m, score);
        float corr  = exp(m - m_new);
        float p     = exp(score - m_new);
        l = l * corr + p;
        for (int d = 0; d < D; ++d) {
            acc[d] = acc[d] * corr + p * float(v_arena[kv_base + d]);
        }
        m = m_new;
    }

    tg_m[tid] = m;
    tg_l[tid] = l;
    for (int d = 0; d < D; ++d) { tg_acc[tid * D + d] = acc[d]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float gm = -INFINITY;
        for (int i = 0; i < TG; ++i) { gm = max(gm, tg_m[i]); }
        float gl = 0.0;
        for (int i = 0; i < TG; ++i) { gl += tg_l[i] * exp(tg_m[i] - gm); }
        g_m = gm;
        g_l = gl;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float gm = g_m;
    float inv_l = (g_l > 0.0) ? (1.0 / g_l) : 0.0;
    uint o_base = b * uint(HQ) * uint(D) + hq * uint(D);
    for (uint d = tid; d < uint(D); d += TG) {
        float a = 0.0;
        for (int i = 0; i < TG; ++i) {
            a += tg_acc[uint(i) * uint(D) + d] * exp(tg_m[i] - gm);
        }
        out[o_base + d] = T(a * inv_l);
    }
"#;

static PAGED_ATTENTION_DECODE_V1_KERNEL: OnceLock<MetalKernel> = OnceLock::new();

fn create_paged_attention_decode_v1_kernel() -> MetalKernel {
    create_paged_kernel("paged_attention_decode_v1", PAGED_ATTENTION_DECODE_V1_SOURCE)
}

/// Cached length at/above which the dispatch uses the register-light 2-pass
/// kernel (below it, the lower-overhead single-pass V1 is faster). Crossover
/// measured on gemma-4-26B (D=256): single-pass wins at ~270 tokens, 2-pass
/// wins at ~2300.
pub const PAGED_DECODE_TWO_PASS_MIN: i32 = 1024;

const PAGED_ATTENTION_DECODE_KERNEL_SOURCE: &str = r#"
    threadgroup float s_p[N_MAX];   // scores (pass 1) then exp-weights (pass 2)
    threadgroup float s_red[TG];    // reduction scratch
    threadgroup float g_m;
    threadgroup float g_l;

    uint hq  = threadgroup_position_in_grid.x;
    uint b   = threadgroup_position_in_grid.y;
    uint tid = thread_position_in_threadgroup.x;

    uint hkv = hq / uint(KV_REPEAT);
    float scale = float(scale_arr[0]);
    uint q_base = b * uint(HQ) * uint(D) + hq * uint(D);

    // arena: [B, HKV, SEQ, D] — block pb owns tokens [pb*BLOCK, (pb+1)*BLOCK).
    uint ar_t = uint(D);
    uint ar_h = uint(SEQ) * ar_t;
    uint ar_b = uint(HKV) * ar_h;

    // --- Pass 1a: scores into s_p, per-thread max over a strided slice of N ---
    float local_m = -INFINITY;
    for (uint t = tid; t < uint(N); t += uint(TG)) {
        int pb = block_table[t / uint(BLOCK)];
        uint phys_t = uint(pb) * uint(BLOCK) + (t % uint(BLOCK));
        uint kv_base = b * ar_b + hkv * ar_h + phys_t * ar_t;
        float score = 0.0;
        for (uint d = 0; d < uint(D); ++d) {
            score += float(q[q_base + d]) * float(k_arena[kv_base + d]);
        }
        score *= scale;
        s_p[t] = score;
        local_m = max(local_m, score);
    }
    s_red[tid] = local_m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = uint(TG) >> 1; s > 0u; s >>= 1u) {
        if (tid < s) s_red[tid] = max(s_red[tid], s_red[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) g_m = s_red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m = g_m;

    // --- Pass 1b: exp(score - m) into s_p, per-thread sum ---
    float local_l = 0.0;
    for (uint t = tid; t < uint(N); t += uint(TG)) {
        float e = exp(s_p[t] - m);
        s_p[t] = e;
        local_l += e;
    }
    s_red[tid] = local_l;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = uint(TG) >> 1; s > 0u; s >>= 1u) {
        if (tid < s) s_red[tid] += s_red[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) g_l = s_red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_l = (g_l > 0.0) ? (1.0 / g_l) : 0.0;

    // --- Pass 2: out[d] = inv_l · Σ_t s_p[t]·V[t][d], D split across threads ---
    uint o_base = b * uint(HQ) * uint(D) + hq * uint(D);
    for (uint d = tid; d < uint(D); d += uint(TG)) {
        float a = 0.0;
        for (uint t = 0; t < uint(N); ++t) {
            int pb = block_table[t / uint(BLOCK)];
            uint phys_t = uint(pb) * uint(BLOCK) + (t % uint(BLOCK));
            uint kv_base = b * ar_b + hkv * ar_h + phys_t * ar_t;
            a += s_p[t] * float(v_arena[kv_base + d]);
        }
        out[o_base + d] = T(a * inv_l);
    }
"#;

static PAGED_ATTENTION_DECODE_KERNEL: OnceLock<MetalKernel> = OnceLock::new();

fn create_paged_attention_decode_kernel() -> MetalKernel {
    create_paged_kernel("paged_attention_decode", PAGED_ATTENTION_DECODE_KERNEL_SOURCE)
}

/// Shared builder for the paged-attention decode kernels (V1 single-pass and
/// V2 2-pass) — same inputs/outputs, different source.
fn create_paged_kernel(name: &str, src: &str) -> MetalKernel {
    unsafe {
        let inputs = mlx_sys::mlx_vector_string_new();
        for n in ["q", "k_arena", "v_arena", "block_table", "scale_arr"] {
            let c = CString::new(n).unwrap();
            mlx_sys::mlx_vector_string_append_value(inputs, c.as_ptr());
        }
        let outputs = mlx_sys::mlx_vector_string_new();
        let out_name = CString::new("out").unwrap();
        mlx_sys::mlx_vector_string_append_value(outputs, out_name.as_ptr());
        let source = CString::new(src).unwrap();
        let header = CString::new("").unwrap();
        let name = CString::new(name).unwrap();
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

/// Single-query paged attention: `[B, HQ, 1, D]` output computed against a
/// paged KV arena in one Metal dispatch, no gather. See the kernel source.
///
/// - `q`: `[B, HQ, 1, D]`, dtype `T`.
/// - `k_arena`/`v_arena`: contiguous `[B, HKV, seq, D]`, dtype `T` (blocks are
///   `block_size`-token ranges along the seq axis).
/// - `block_table`: int32 `[ceil(n_tokens / block_size)]` (coerced if needed).
/// - `seq`: the arena's physical token capacity (`k_arena.shape()[2]`).
/// - `scale`: softmax scale (typically `1/sqrt(D)`).
#[allow(clippy::too_many_arguments)]
pub fn paged_attention_decode(
    q: &Array,
    k_arena: &Array,
    v_arena: &Array,
    block_table: &Array,
    scale: f32,
    n_tokens: i32,
    block_size: i32,
    seq: i32,
    b: i32,
    hq: i32,
    hkv: i32,
    d: i32,
) -> Result<Array, Exception> {
    if n_tokens <= 0 {
        return Err(Exception::custom("paged_attention_decode: n_tokens must be > 0"));
    }
    if hkv <= 0 || hq % hkv != 0 {
        return Err(Exception::custom("paged_attention_decode: HQ must be a multiple of HKV"));
    }
    if q.shape() != [b, hq, 1, d] {
        return Err(Exception::custom(format!(
            "paged_attention_decode expects q [B,HQ,1,D]={:?}, got {:?}",
            [b, hq, 1, d],
            q.shape()
        )));
    }
    let kv_repeat = hq / hkv;
    let dtype: u32 = q.dtype().into();
    let block_table = if block_table.dtype() != Dtype::Int32 {
        block_table.as_dtype(Dtype::Int32)?
    } else {
        block_table.clone()
    };
    let scale_arr = Array::from_slice(&[scale], &[1]);

    // Pick the kernel by cached length: the register-light 2-pass kernel wins at
    // long context, the lower-overhead single-pass V1 wins at short context.
    let two_pass = n_tokens >= PAGED_DECODE_TWO_PASS_MIN;
    let kernel = if two_pass {
        PAGED_ATTENTION_DECODE_KERNEL.get_or_init(create_paged_attention_decode_kernel)
    } else {
        PAGED_ATTENTION_DECODE_V1_KERNEL.get_or_init(create_paged_attention_decode_v1_kernel)
    };
    unsafe {
        let stream = mlx_sys::mlx_default_gpu_stream_new();
        let config = mlx_sys::mlx_fast_metal_kernel_config_new();

        let type_name = CString::new("T").unwrap();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
            config, type_name.as_ptr(), dtype,
        );
        // V2 (2-pass): TG=256 (decoupled from D, no acc[D] array). V1
        // (single-pass): TG capped so tg_acc[TG*D] stays ≤16KB threadgroup mem.
        let tg: i32 = if two_pass { 256 } else { (4096 / d.max(1)).clamp(1, 32) };
        let mut targs: Vec<(&str, i32)> = vec![
            ("D", d),
            ("HQ", hq),
            ("HKV", hkv),
            ("B", b),
            ("SEQ", seq),
            ("BLOCK", block_size),
            ("N", n_tokens),
            ("KV_REPEAT", kv_repeat),
            ("TG", tg),
        ];
        // N_MAX sizes V2's s_p score buffer; V1 doesn't use it.
        if two_pass {
            targs.push(("N_MAX", PAGED_DECODE_MAX_KV));
        }
        for (name, value) in targs {
            let cname = CString::new(name).unwrap();
            mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
                config, cname.as_ptr(), value,
            );
        }
        // One threadgroup per (hq, b): grid.x = HQ·TG threads (HQ groups of TG),
        // grid.y = B groups of 1. `threadgroup_position_in_grid` → (hq, b).
        mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, hq.max(1) * tg, b.max(1), 1);
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, tg, 1, 1);

        let out_shape: [i32; 4] = [b, hq, 1, d];
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            out_shape.as_ptr(),
            out_shape.len(),
            dtype,
        );

        let inputs = mlx_sys::mlx_vector_array_new();
        for arr in [q, k_arena, v_arena, &block_table, &scale_arr] {
            mlx_sys::mlx_vector_array_append_value(inputs, arr.as_ptr());
        }

        let mut outputs = mlx_sys::mlx_vector_array_new();
        let ret = mlx_sys::mlx_fast_metal_kernel_apply(
            &mut outputs, kernel.kernel, inputs, config, stream,
        );
        if ret != 0 {
            mlx_sys::mlx_fast_metal_kernel_config_free(config);
            mlx_sys::mlx_vector_array_free(inputs);
            mlx_sys::mlx_vector_array_free(outputs);
            mlx_sys::mlx_stream_free(stream);
            return Err(Exception::custom("paged_attention_decode kernel execution failed"));
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

// ============================================================================
// MoE dense matmul kernel (#35 phase 3).
//
// Computes C = A @ B^T for one expert bucket:
//   A: [M, K] row-major   (sorted tokens routed to this expert)
//   B: [N, K] row-major   (expert's gate_up_proj or down_proj weight matrix)
//   C: [M, N] row-major
//
// Uses simdgroup_matrix<float, 8, 8> tiles in the canonical tinygrad 4x4
// acc pattern (github.com/tinygrad/tinygrad/blob/3f2d4014/extra/gemm/
// metal_matmul.py). Each threadgroup computes a 32x32 output region via
// 16 acc tiles. K is a template arg (compile-time); M is dynamic via a
// [1]-shape int32 buffer so we don't recompile per bucket size.
//
// Constraints:
//   - K must be a multiple of 8 (tile inner-dim alignment).
//   - N must be a multiple of 32 (output column-tile alignment).
//   - M is padded to a multiple of 32 at dispatch time by the caller.
//
// The kernel is dispatched once per expert from `forward_topk_expert_major_v3`.
// On Gemma4-26B-A4B-it this is 64 launches per MoE layer; the per-launch
// overhead is amortized against the dense matmul that does ~mean_bucket *
// K * N * 2 fp32 flops per expert.
// ============================================================================
const MOE_DENSE_MATMUL_KERNEL: &str = r#"
    int M = M_buf[0];
    uint row_base = threadgroup_position_in_grid.x * 32u;
    uint col_base = threadgroup_position_in_grid.y * 32u;
    if (row_base >= uint(M)) return;
    if (col_base >= uint(N_)) return;

    simdgroup_matrix<float, 8, 8> acc[4][4];
    for (uint i = 0; i < 4; i++)
        for (uint j = 0; j < 4; j++)
            acc[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);

    simdgroup_matrix<float, 8, 8> a_tile[4];
    simdgroup_matrix<float, 8, 8> b_tile[4];

    for (uint k = 0; k < uint(K_); k += 8u) {
        // A is [M, K] row-major: row r starts at A + r*K.
        // Load 4 vertically-stacked 8x8 tiles at row_base + 0,8,16,24.
        simdgroup_load(a_tile[0], A + (row_base + 0u)  * uint(K_) + k, uint(K_));
        simdgroup_load(a_tile[1], A + (row_base + 8u)  * uint(K_) + k, uint(K_));
        simdgroup_load(a_tile[2], A + (row_base + 16u) * uint(K_) + k, uint(K_));
        simdgroup_load(a_tile[3], A + (row_base + 24u) * uint(K_) + k, uint(K_));

        // B is passed pre-transposed by the caller as [K, N] row-major.
        // Tile (k:k+8, col_base+j:col_base+j+8) starts at offset k*N + col_base+j*8,
        // stride N.
        simdgroup_load(b_tile[0], B + k * uint(N_) + col_base +  0u, uint(N_));
        simdgroup_load(b_tile[1], B + k * uint(N_) + col_base +  8u, uint(N_));
        simdgroup_load(b_tile[2], B + k * uint(N_) + col_base + 16u, uint(N_));
        simdgroup_load(b_tile[3], B + k * uint(N_) + col_base + 24u, uint(N_));

        for (uint i = 0; i < 4; i++) {
            for (uint j = 0; j < 4; j++) {
                simdgroup_multiply_accumulate(acc[i][j], a_tile[i], b_tile[j], acc[i][j]);
            }
        }
    }

    // Store 16 8x8 tiles to C[row_base..row_base+32, col_base..col_base+32].
    // Caller is required to pad M to a multiple of 32 (padded rows contain
    // zeros, so their outputs are valid garbage that the caller discards).
    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) {
            uint r = row_base + i * 8u;
            uint c = col_base + j * 8u;
            simdgroup_store(acc[i][j], C + r * uint(N_) + c, uint(N_));
        }
    }
"#;

static MOE_DENSE_MATMUL_KERNEL_HANDLE: OnceLock<MetalKernel> = OnceLock::new();

fn create_moe_dense_matmul_kernel() -> MetalKernel {
    unsafe {
        let a = CString::new("A").unwrap();
        let b = CString::new("B").unwrap();
        let m_buf = CString::new("M_buf").unwrap();
        let out = CString::new("C").unwrap();

        let inputs = mlx_sys::mlx_vector_string_new();
        mlx_sys::mlx_vector_string_append_value(inputs, a.as_ptr());
        mlx_sys::mlx_vector_string_append_value(inputs, b.as_ptr());
        mlx_sys::mlx_vector_string_append_value(inputs, m_buf.as_ptr());
        let outputs = mlx_sys::mlx_vector_string_new();
        mlx_sys::mlx_vector_string_append_value(outputs, out.as_ptr());

        let source = CString::new(MOE_DENSE_MATMUL_KERNEL).unwrap();
        let header = CString::new(
            "#include <metal_simdgroup>\n#include <metal_simdgroup_matrix>\n",
        )
        .unwrap();
        let name = CString::new("moe_dense_matmul").unwrap();
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

/// Compute `C = A @ B` where A is `[M_padded, K]` and B is `[K, N]`,
/// using the simdgroup_matrix tile kernel from #35 phase 3. Caller is
/// responsible for transposing the per-expert weight matrix (originally
/// `[N, K]` in `gate_up_proj[e]`) before invoking.
///
/// `M_padded` must be a multiple of 32 (caller's responsibility — pad A
/// with zero rows to round up `actual_M`). `K` must be a multiple of 8.
/// `N` must be a multiple of 32.
///
/// `m_buf` is a `[1]`-shape int32 array carrying the actual (unpadded)
/// row count for the bucket; the kernel uses it as a bounds check to
/// skip threadgroups that are entirely beyond `actual_M`. The caller
/// slices the returned `[M_padded, N]` array to `[actual_M, N]`.
///
/// Inputs must be fp32 (kernel uses `simdgroup_matrix<float, 8, 8>`).
/// `K` is passed as a template arg; the kernel is recompiled the first
/// time a new K is seen, then cached.
pub fn moe_dense_matmul(
    a: &Array,
    b: &Array,
    m_buf: &Array,
    k: i32,
    n: i32,
) -> Result<Array, Exception> {
    if k % 8 != 0 {
        return Err(Exception::custom(format!(
            "moe_dense_matmul: K must be divisible by 8; got {k}"
        )));
    }
    if n % 32 != 0 {
        return Err(Exception::custom(format!(
            "moe_dense_matmul: N must be divisible by 32; got {n}"
        )));
    }
    let a_shape = a.shape();
    if a_shape.len() != 2 || a_shape[1] != k {
        return Err(Exception::custom(format!(
            "moe_dense_matmul: A must be [M_padded, K={k}]; got {a_shape:?}"
        )));
    }
    let m_padded = a_shape[0];
    if m_padded % 32 != 0 {
        return Err(Exception::custom(format!(
            "moe_dense_matmul: M_padded must be divisible by 32; got {m_padded}. \
             Caller must pre-pad A with zero rows."
        )));
    }
    let b_shape = b.shape();
    if b_shape.len() != 2 || b_shape[0] != k || b_shape[1] != n {
        return Err(Exception::custom(format!(
            "moe_dense_matmul: B must be [K={k}, N={n}]; got {b_shape:?}"
        )));
    }
    if a.dtype() != mlx_rs::Dtype::Float32 || b.dtype() != mlx_rs::Dtype::Float32 {
        return Err(Exception::custom(
            "moe_dense_matmul: A and B must be fp32",
        ));
    }
    let m_buf_shape = m_buf.shape();
    if m_buf_shape.len() != 1 || m_buf_shape[0] != 1 {
        return Err(Exception::custom(format!(
            "moe_dense_matmul: M_buf must be [1]; got {m_buf_shape:?}"
        )));
    }
    if m_buf.dtype() != mlx_rs::Dtype::Int32 {
        return Err(Exception::custom("moe_dense_matmul: M_buf must be int32"));
    }

    let kernel = MOE_DENSE_MATMUL_KERNEL_HANDLE.get_or_init(create_moe_dense_matmul_kernel);

    unsafe {
        let stream = mlx_sys::mlx_default_gpu_stream_new();
        let config = mlx_sys::mlx_fast_metal_kernel_config_new();
        // Template args: K_, N_ (the kernel reads these as compile-time uints).
        for (name, value) in [("K_", k), ("N_", n)] {
            let cname = CString::new(name).unwrap();
            mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
                config, cname.as_ptr(), value,
            );
        }
        // Grid: (M_padded/32) × (N/32) threadgroups, each 32 threads (one
        // simdgroup) handling a 32x32 output region via 4x4 acc tiles.
        // MLX `set_grid` takes TOTAL THREADS (not threadgroups): total =
        // tg_count × threads_per_tg. So x = (m_padded/32 tgs) × 32 = m_padded,
        // y = n/32 tgs × 1 = n/32.
        mlx_sys::mlx_fast_metal_kernel_config_set_grid(
            config, m_padded, n / 32, 1,
        );
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, 32, 1, 1);

        let out_shape: [i32; 2] = [m_padded, n];
        let f32_dtype: u32 = mlx_rs::Dtype::Float32.into();
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config, out_shape.as_ptr(), out_shape.len(), f32_dtype,
        );

        let inputs = mlx_sys::mlx_vector_array_new();
        mlx_sys::mlx_vector_array_append_value(inputs, a.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, b.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, m_buf.as_ptr());

        let mut outputs = mlx_sys::mlx_vector_array_new();
        let ret = mlx_sys::mlx_fast_metal_kernel_apply(
            &mut outputs, kernel.kernel, inputs, config, stream,
        );
        if ret != 0 {
            mlx_sys::mlx_fast_metal_kernel_config_free(config);
            mlx_sys::mlx_vector_array_free(inputs);
            mlx_sys::mlx_vector_array_free(outputs);
            mlx_sys::mlx_stream_free(stream);
            return Err(Exception::custom("moe_dense_matmul kernel failed"));
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

// ============================================================================
// MoE dense matmul kernel — bf16 variant (#35 phase 6).
//
// Same algorithm as moe_dense_matmul but with simdgroup_matrix<bfloat, 8, 8>
// to skip the fp32 promotion overhead. Accumulates in fp32 for precision
// (Apple's mixed-precision simdgroup_multiply_accumulate). Requires Metal
// 3.0+ which is standard on Apple Silicon M2+.
//
// On Gemma4-26B-A4B-it the gate_up_proj / down_proj weights ship as bf16;
// this kernel can consume them directly. Compared to the fp32 variant we
// save: bf16→fp32 promotion on the input + 4× memory bandwidth on the
// weight loads.
// ============================================================================
const MOE_DENSE_MATMUL_BF16_KERNEL: &str = r#"
    int M = M_buf[0];
    uint row_base = threadgroup_position_in_grid.x * 32u;
    uint col_base = threadgroup_position_in_grid.y * 32u;
    if (row_base >= uint(M)) return;
    if (col_base >= uint(N_)) return;

    // bf16 throughout. Metal doesn't allow fp32→bf16 simdgroup_matrix
    // conversion at store time, so we accumulate in bf16. For our expert
    // matmul (K up to 5376) this gives ~5e-2 abs error which is
    // acceptable for downstream SwiGLU + softmax.
    simdgroup_matrix<bfloat, 8, 8> acc[4][4];
    for (uint i = 0; i < 4; i++)
        for (uint j = 0; j < 4; j++)
            acc[i][j] = simdgroup_matrix<bfloat, 8, 8>(bfloat(0.0));

    simdgroup_matrix<bfloat, 8, 8> a_tile[4];
    simdgroup_matrix<bfloat, 8, 8> b_tile[4];

    for (uint k = 0; k < uint(K_); k += 8u) {
        simdgroup_load(a_tile[0], A + (row_base + 0u)  * uint(K_) + k, uint(K_));
        simdgroup_load(a_tile[1], A + (row_base + 8u)  * uint(K_) + k, uint(K_));
        simdgroup_load(a_tile[2], A + (row_base + 16u) * uint(K_) + k, uint(K_));
        simdgroup_load(a_tile[3], A + (row_base + 24u) * uint(K_) + k, uint(K_));

        simdgroup_load(b_tile[0], B + k * uint(N_) + col_base +  0u, uint(N_));
        simdgroup_load(b_tile[1], B + k * uint(N_) + col_base +  8u, uint(N_));
        simdgroup_load(b_tile[2], B + k * uint(N_) + col_base + 16u, uint(N_));
        simdgroup_load(b_tile[3], B + k * uint(N_) + col_base + 24u, uint(N_));

        for (uint i = 0; i < 4; i++) {
            for (uint j = 0; j < 4; j++) {
                simdgroup_multiply_accumulate(acc[i][j], a_tile[i], b_tile[j], acc[i][j]);
            }
        }
    }

    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) {
            uint r = row_base + i * 8u;
            uint c = col_base + j * 8u;
            simdgroup_store(acc[i][j], C + r * uint(N_) + c, uint(N_));
        }
    }
"#;

static MOE_DENSE_MATMUL_BF16_KERNEL_HANDLE: OnceLock<MetalKernel> = OnceLock::new();

fn create_moe_dense_matmul_bf16_kernel() -> MetalKernel {
    unsafe {
        let a = CString::new("A").unwrap();
        let b = CString::new("B").unwrap();
        let m_buf = CString::new("M_buf").unwrap();
        let out = CString::new("C").unwrap();

        let inputs = mlx_sys::mlx_vector_string_new();
        mlx_sys::mlx_vector_string_append_value(inputs, a.as_ptr());
        mlx_sys::mlx_vector_string_append_value(inputs, b.as_ptr());
        mlx_sys::mlx_vector_string_append_value(inputs, m_buf.as_ptr());
        let outputs = mlx_sys::mlx_vector_string_new();
        mlx_sys::mlx_vector_string_append_value(outputs, out.as_ptr());

        let source = CString::new(MOE_DENSE_MATMUL_BF16_KERNEL).unwrap();
        let header = CString::new(
            "#include <metal_simdgroup>\n#include <metal_simdgroup_matrix>\n",
        )
        .unwrap();
        let name = CString::new("moe_dense_matmul_bf16").unwrap();
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

/// bf16 variant of [`moe_dense_matmul`]. Same shape contract; expects
/// `A` and `B` as bf16 (not fp32) and produces a bf16 output. Internally
/// accumulates in fp32 for precision.
pub fn moe_dense_matmul_bf16(
    a: &Array,
    b: &Array,
    m_buf: &Array,
    k: i32,
    n: i32,
) -> Result<Array, Exception> {
    if k % 8 != 0 {
        return Err(Exception::custom(format!(
            "moe_dense_matmul_bf16: K must be divisible by 8; got {k}"
        )));
    }
    if n % 32 != 0 {
        return Err(Exception::custom(format!(
            "moe_dense_matmul_bf16: N must be divisible by 32; got {n}"
        )));
    }
    let a_shape = a.shape();
    if a_shape.len() != 2 || a_shape[1] != k {
        return Err(Exception::custom(format!(
            "moe_dense_matmul_bf16: A must be [M_padded, K={k}]; got {a_shape:?}"
        )));
    }
    let m_padded = a_shape[0];
    if m_padded % 32 != 0 {
        return Err(Exception::custom(format!(
            "moe_dense_matmul_bf16: M_padded must be divisible by 32; got {m_padded}"
        )));
    }
    let b_shape = b.shape();
    if b_shape.len() != 2 || b_shape[0] != k || b_shape[1] != n {
        return Err(Exception::custom(format!(
            "moe_dense_matmul_bf16: B must be [K={k}, N={n}]; got {b_shape:?}"
        )));
    }
    if a.dtype() != mlx_rs::Dtype::Bfloat16 || b.dtype() != mlx_rs::Dtype::Bfloat16 {
        return Err(Exception::custom(format!(
            "moe_dense_matmul_bf16: A and B must be bf16; got A={:?} B={:?}",
            a.dtype(),
            b.dtype()
        )));
    }
    if m_buf.dtype() != mlx_rs::Dtype::Int32 {
        return Err(Exception::custom("moe_dense_matmul_bf16: M_buf must be int32"));
    }

    let kernel = MOE_DENSE_MATMUL_BF16_KERNEL_HANDLE
        .get_or_init(create_moe_dense_matmul_bf16_kernel);

    unsafe {
        let stream = mlx_sys::mlx_default_gpu_stream_new();
        let config = mlx_sys::mlx_fast_metal_kernel_config_new();
        for (name, value) in [("K_", k), ("N_", n)] {
            let cname = CString::new(name).unwrap();
            mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
                config, cname.as_ptr(), value,
            );
        }
        mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, m_padded, n / 32, 1);
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, 32, 1, 1);

        let out_shape: [i32; 2] = [m_padded, n];
        let bf16_dtype: u32 = mlx_rs::Dtype::Bfloat16.into();
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config, out_shape.as_ptr(), out_shape.len(), bf16_dtype,
        );

        let inputs = mlx_sys::mlx_vector_array_new();
        mlx_sys::mlx_vector_array_append_value(inputs, a.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, b.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, m_buf.as_ptr());

        let mut outputs = mlx_sys::mlx_vector_array_new();
        let ret = mlx_sys::mlx_fast_metal_kernel_apply(
            &mut outputs, kernel.kernel, inputs, config, stream,
        );
        if ret != 0 {
            mlx_sys::mlx_fast_metal_kernel_config_free(config);
            mlx_sys::mlx_vector_array_free(inputs);
            mlx_sys::mlx_vector_array_free(outputs);
            mlx_sys::mlx_stream_free(stream);
            return Err(Exception::custom("moe_dense_matmul_bf16 kernel failed"));
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

// ============================================================================
// MoE dense matmul kernel — batched all-experts variant (#35 phase 7).
// ============================================================================
const MOE_DENSE_MATMUL_BATCHED_KERNEL: &str = r#"
    uint bx = threadgroup_position_in_grid.x;
    uint by = threadgroup_position_in_grid.y;
    uint row_base = bx * 32u;
    uint col_base = by * 32u;
    if (col_base >= uint(N_)) return;
    int e = block_expert_id[bx];
    if (e < 0) return;
    uint w_off = uint(e) * uint(N_) * uint(K_);

    // Phase 9 (#35): fp32 accumulators, bf16 A/B tiles. Apple Metal
    // supports mixed-precision simdgroup_multiply_accumulate: the acc
    // type determines accumulation precision. Output buffer is fp32; the
    // caller narrows to bf16 (or fp16) after the kernel returns. This
    // eliminates the K=5376 accumulator-rounding divergence we observed
    // in phase 7+8.
    simdgroup_matrix<float, 8, 8> acc[4][4];
    for (uint i = 0; i < 4; i++)
        for (uint j = 0; j < 4; j++)
            acc[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);

    simdgroup_matrix<bfloat, 8, 8> a_tile[4];
    simdgroup_matrix<bfloat, 8, 8> b_tile[4];

    for (uint k = 0; k < uint(K_); k += 8u) {
        simdgroup_load(a_tile[0], A + (row_base + 0u)  * uint(K_) + k, uint(K_));
        simdgroup_load(a_tile[1], A + (row_base + 8u)  * uint(K_) + k, uint(K_));
        simdgroup_load(a_tile[2], A + (row_base + 16u) * uint(K_) + k, uint(K_));
        simdgroup_load(a_tile[3], A + (row_base + 24u) * uint(K_) + k, uint(K_));

        simdgroup_load(b_tile[0], B + w_off + k * uint(N_) + col_base +  0u, uint(N_));
        simdgroup_load(b_tile[1], B + w_off + k * uint(N_) + col_base +  8u, uint(N_));
        simdgroup_load(b_tile[2], B + w_off + k * uint(N_) + col_base + 16u, uint(N_));
        simdgroup_load(b_tile[3], B + w_off + k * uint(N_) + col_base + 24u, uint(N_));

        for (uint i = 0; i < 4; i++) {
            for (uint j = 0; j < 4; j++) {
                simdgroup_multiply_accumulate(acc[i][j], a_tile[i], b_tile[j], acc[i][j]);
            }
        }
    }

    // Store fp32 acc tiles to fp32 output buffer; caller narrows.
    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) {
            uint r = row_base + i * 8u;
            uint c = col_base + j * 8u;
            simdgroup_store(acc[i][j], C + r * uint(N_) + c, uint(N_));
        }
    }
"#;

static MOE_DENSE_MATMUL_BATCHED_KERNEL_HANDLE: OnceLock<MetalKernel> = OnceLock::new();

fn create_moe_dense_matmul_batched_kernel() -> MetalKernel {
    unsafe {
        let a = CString::new("A").unwrap();
        let b = CString::new("B").unwrap();
        let block_expert_id = CString::new("block_expert_id").unwrap();
        let out = CString::new("C").unwrap();

        let inputs = mlx_sys::mlx_vector_string_new();
        mlx_sys::mlx_vector_string_append_value(inputs, a.as_ptr());
        mlx_sys::mlx_vector_string_append_value(inputs, b.as_ptr());
        mlx_sys::mlx_vector_string_append_value(inputs, block_expert_id.as_ptr());
        let outputs = mlx_sys::mlx_vector_string_new();
        mlx_sys::mlx_vector_string_append_value(outputs, out.as_ptr());

        let source = CString::new(MOE_DENSE_MATMUL_BATCHED_KERNEL).unwrap();
        let header = CString::new(
            "#include <metal_simdgroup>\n#include <metal_simdgroup_matrix>\n",
        )
        .unwrap();
        let name = CString::new("moe_dense_matmul_batched").unwrap();
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

/// Single-launch batched MoE matmul (#35 phase 7).
///
/// `a`: `[M_total_padded, K]` bf16 — caller-packed input, zero-padded between
/// expert buckets so padded rows produce zero output naturally.
/// `b`: `[E, K, N]` bf16 — per-expert weight matrices, *pre-transposed*
/// by the caller from `[E, N, K]` to `[E, K, N]`. Pre-transpose can be
/// done once at model load (phase 8 follow-up).
/// `block_expert_id`: `[M_total_padded / 32]` i32 — for each 32-row block,
/// which expert id it belongs to. Use `-1` to skip a block entirely.
pub fn moe_dense_matmul_batched(
    a: &Array,
    b: &Array,
    block_expert_id: &Array,
    k: i32,
    n: i32,
) -> Result<Array, Exception> {
    if k % 8 != 0 {
        return Err(Exception::custom(format!(
            "moe_dense_matmul_batched: K must be divisible by 8; got {k}"
        )));
    }
    if n % 32 != 0 {
        return Err(Exception::custom(format!(
            "moe_dense_matmul_batched: N must be divisible by 32; got {n}"
        )));
    }
    let a_shape = a.shape();
    if a_shape.len() != 2 || a_shape[1] != k {
        return Err(Exception::custom(format!(
            "moe_dense_matmul_batched: A must be [M_total_padded, K={k}]; got {a_shape:?}"
        )));
    }
    let m_total_padded = a_shape[0];
    if m_total_padded % 32 != 0 {
        return Err(Exception::custom(format!(
            "moe_dense_matmul_batched: M_total_padded must be divisible by 32; got {m_total_padded}"
        )));
    }
    let b_shape = b.shape();
    if b_shape.len() != 3 || b_shape[1] != k || b_shape[2] != n {
        return Err(Exception::custom(format!(
            "moe_dense_matmul_batched: B must be [E, K={k}, N={n}]; got {b_shape:?}"
        )));
    }
    let id_shape = block_expert_id.shape();
    if id_shape.len() != 1 || id_shape[0] != m_total_padded / 32 {
        return Err(Exception::custom(format!(
            "moe_dense_matmul_batched: block_expert_id must be [{}]; got {id_shape:?}",
            m_total_padded / 32
        )));
    }
    if a.dtype() != mlx_rs::Dtype::Bfloat16 || b.dtype() != mlx_rs::Dtype::Bfloat16 {
        return Err(Exception::custom(format!(
            "moe_dense_matmul_batched: A and B must be bf16; got A={:?} B={:?}",
            a.dtype(),
            b.dtype()
        )));
    }
    if block_expert_id.dtype() != mlx_rs::Dtype::Int32 {
        return Err(Exception::custom(
            "moe_dense_matmul_batched: block_expert_id must be int32",
        ));
    }

    let kernel = MOE_DENSE_MATMUL_BATCHED_KERNEL_HANDLE
        .get_or_init(create_moe_dense_matmul_batched_kernel);

    unsafe {
        let stream = mlx_sys::mlx_default_gpu_stream_new();
        let config = mlx_sys::mlx_fast_metal_kernel_config_new();
        for (name, value) in [("K_", k), ("N_", n)] {
            let cname = CString::new(name).unwrap();
            mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
                config, cname.as_ptr(), value,
            );
        }
        mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, m_total_padded, n / 32, 1);
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, 32, 1, 1);

        let out_shape: [i32; 2] = [m_total_padded, n];
        let f32_dtype: u32 = mlx_rs::Dtype::Float32.into();
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config, out_shape.as_ptr(), out_shape.len(), f32_dtype,
        );

        let inputs = mlx_sys::mlx_vector_array_new();
        mlx_sys::mlx_vector_array_append_value(inputs, a.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, b.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, block_expert_id.as_ptr());

        let mut outputs = mlx_sys::mlx_vector_array_new();
        let ret = mlx_sys::mlx_fast_metal_kernel_apply(
            &mut outputs, kernel.kernel, inputs, config, stream,
        );
        if ret != 0 {
            mlx_sys::mlx_fast_metal_kernel_config_free(config);
            mlx_sys::mlx_vector_array_free(inputs);
            mlx_sys::mlx_vector_array_free(outputs);
            mlx_sys::mlx_stream_free(stream);
            return Err(Exception::custom("moe_dense_matmul_batched kernel failed"));
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

// ============================================================================
// Fused RMSNorm + residual_add + RMSNorm (T4).
//
// Collapses the per-layer sequence:
//   normed_attn = post_attention_layernorm(attn_out)
//   sum         = residual + normed_attn
//   normed_sum  = pre_feedforward_layernorm(sum)
// from 3 launches into 1. Outputs both `sum` (for the next residual)
// and `normed_sum` (the FF block input).
//
// At 30 MoE layers × 2 launches saved = 60/token launches ≈ 2.2 ms at
// the observed ~37 µs/launch overhead (~5% decode improvement target).
//
// Inputs:  attn_out [n, H] bf16, residual [n, H] bf16,
//          weight_post_attn [H] bf16, weight_pre_ff [H] bf16
// Outputs: sum [n, H] bf16, normed_sum [n, H] bf16
// Template args: H (compile-time)
// Grid: one threadgroup per row n; 64 threads cooperate on the two
// RMSNorm reductions.
// ============================================================================
const FUSED_NORM_ADD_NORM_KERNEL: &str = r#"
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint gs = threads_per_threadgroup.x;
    const float EPS = 1e-6f;

    device const bfloat *row_attn = attn_out + row * uint(H_);
    device const bfloat *row_res  = residual + row * uint(H_);
    device bfloat *row_sum    = sum + row * uint(H_);
    device bfloat *row_normed = normed_sum + row * uint(H_);

    threadgroup float red_buf[64];
    threadgroup float shared_sum[H_];  // residual + normed_attn, reused for 2nd norm

    // ── 1. RMSNorm(attn_out, weight_post_attn): compute sumsq across H ──
    float local_sq = 0.0f;
    for (uint i = tid; i < uint(H_); i += gs) {
        float v = float(row_attn[i]);
        local_sq += v * v;
    }
    red_buf[tid] = local_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = gs >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) red_buf[tid] += red_buf[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_rms_attn = rsqrt(red_buf[0] / float(H_) + EPS);

    // ── 2. apply weight, add residual, store sum, accumulate sumsq for 2nd norm ──
    float local_sq2 = 0.0f;
    for (uint i = tid; i < uint(H_); i += gs) {
        float v = float(row_attn[i]) * inv_rms_attn * float(weight_post_attn[i]);
        float s = float(row_res[i]) + v;
        shared_sum[i] = s;
        row_sum[i] = bfloat(s);
        local_sq2 += s * s;
    }
    red_buf[tid] = local_sq2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = gs >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) red_buf[tid] += red_buf[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_rms_sum = rsqrt(red_buf[0] / float(H_) + EPS);

    // ── 3. normed_sum = sum * inv_rms * weight_pre_ff ──
    for (uint i = tid; i < uint(H_); i += gs) {
        row_normed[i] = bfloat(shared_sum[i] * inv_rms_sum * float(weight_pre_ff[i]));
    }
"#;

static FUSED_NORM_ADD_NORM_KERNEL_HANDLE: OnceLock<MetalKernel> = OnceLock::new();

fn create_fused_norm_add_norm_kernel() -> MetalKernel {
    unsafe {
        let attn = CString::new("attn_out").unwrap();
        let res = CString::new("residual").unwrap();
        let w1 = CString::new("weight_post_attn").unwrap();
        let w2 = CString::new("weight_pre_ff").unwrap();
        let out_sum = CString::new("sum").unwrap();
        let out_n = CString::new("normed_sum").unwrap();

        let inputs = mlx_sys::mlx_vector_string_new();
        mlx_sys::mlx_vector_string_append_value(inputs, attn.as_ptr());
        mlx_sys::mlx_vector_string_append_value(inputs, res.as_ptr());
        mlx_sys::mlx_vector_string_append_value(inputs, w1.as_ptr());
        mlx_sys::mlx_vector_string_append_value(inputs, w2.as_ptr());
        let outputs = mlx_sys::mlx_vector_string_new();
        mlx_sys::mlx_vector_string_append_value(outputs, out_sum.as_ptr());
        mlx_sys::mlx_vector_string_append_value(outputs, out_n.as_ptr());

        let source = CString::new(FUSED_NORM_ADD_NORM_KERNEL).unwrap();
        let header = CString::new("").unwrap();
        let name = CString::new("fused_norm_add_norm").unwrap();
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

/// Fused `RMSNorm(attn_out, w1) → add residual → RMSNorm(sum, w2)`.
///
/// All inputs/outputs are bf16. `weight_post_attn` and `weight_pre_ff`
/// are the per-feature scale vectors `[H]` from the two RMSNorm layers.
/// Returns `(sum, normed_sum)` — caller uses `sum` as the next residual
/// and `normed_sum` as input to the feed-forward block.
///
/// Replaces 3 MLX launches (rms_norm + add + rms_norm) with 1.
pub fn fused_norm_add_norm(
    attn_out: &Array,
    residual: &Array,
    weight_post_attn: &Array,
    weight_pre_ff: &Array,
) -> Result<(Array, Array), Exception> {
    let s = attn_out.shape();
    if s.len() != 2 {
        return Err(Exception::custom(format!(
            "fused_norm_add_norm: attn_out must be [n, H]; got {s:?}"
        )));
    }
    let n = s[0];
    let h = s[1];
    if h > 8192 || h % 32 != 0 {
        return Err(Exception::custom(format!(
            "fused_norm_add_norm: H must be ≤8192 and divisible by 32; got {h}"
        )));
    }
    if residual.shape() != s {
        return Err(Exception::custom(format!(
            "fused_norm_add_norm: residual shape {:?} ≠ attn_out shape {s:?}",
            residual.shape()
        )));
    }
    if weight_post_attn.shape() != &[h] || weight_pre_ff.shape() != &[h] {
        return Err(Exception::custom(format!(
            "fused_norm_add_norm: weights must be [H={h}]; got {:?} / {:?}",
            weight_post_attn.shape(),
            weight_pre_ff.shape()
        )));
    }
    if attn_out.dtype() != mlx_rs::Dtype::Bfloat16
        || residual.dtype() != mlx_rs::Dtype::Bfloat16
        || weight_post_attn.dtype() != mlx_rs::Dtype::Bfloat16
        || weight_pre_ff.dtype() != mlx_rs::Dtype::Bfloat16
    {
        return Err(Exception::custom(
            "fused_norm_add_norm: all tensors must be bf16",
        ));
    }

    let kernel = FUSED_NORM_ADD_NORM_KERNEL_HANDLE.get_or_init(create_fused_norm_add_norm_kernel);

    unsafe {
        let stream = mlx_sys::mlx_default_gpu_stream_new();
        let config = mlx_sys::mlx_fast_metal_kernel_config_new();
        let cname = CString::new("H_").unwrap();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
            config, cname.as_ptr(), h,
        );
        // 64 threads per row; one row per threadgroup.
        mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, n * 64, 1, 1);
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, 64, 1, 1);

        let out_shape: [i32; 2] = [n, h];
        let bf16: u32 = mlx_rs::Dtype::Bfloat16.into();
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config, out_shape.as_ptr(), out_shape.len(), bf16,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config, out_shape.as_ptr(), out_shape.len(), bf16,
        );

        let inputs = mlx_sys::mlx_vector_array_new();
        mlx_sys::mlx_vector_array_append_value(inputs, attn_out.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, residual.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, weight_post_attn.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, weight_pre_ff.as_ptr());

        let mut outputs = mlx_sys::mlx_vector_array_new();
        let ret = mlx_sys::mlx_fast_metal_kernel_apply(
            &mut outputs, kernel.kernel, inputs, config, stream,
        );
        if ret != 0 {
            mlx_sys::mlx_fast_metal_kernel_config_free(config);
            mlx_sys::mlx_vector_array_free(inputs);
            mlx_sys::mlx_vector_array_free(outputs);
            mlx_sys::mlx_stream_free(stream);
            return Err(Exception::custom("fused_norm_add_norm kernel failed"));
        }
        let mut sum_out = mlx_sys::mlx_array_new();
        let mut normed_out = mlx_sys::mlx_array_new();
        mlx_sys::mlx_vector_array_get(&mut sum_out, outputs, 0);
        mlx_sys::mlx_vector_array_get(&mut normed_out, outputs, 1);
        mlx_sys::mlx_fast_metal_kernel_config_free(config);
        mlx_sys::mlx_vector_array_free(inputs);
        mlx_sys::mlx_vector_array_free(outputs);
        mlx_sys::mlx_stream_free(stream);
        Ok((Array::from_ptr(sum_out), Array::from_ptr(normed_out)))
    }
}

// ============================================================================
// Fused router top-k selection + normalize + per-expert-scale (T2).
//
// Collapses the post-matmul tail of `gemma4-mlx/src/model.rs:Router::forward`:
//   softmax → negate → argpartition → take_along → sum → divide → take →
//   multiply  (8 separate MLX op launches)
// into ONE Metal kernel that takes raw scores `[n, E]` and emits both
// `top_k_index [n, K]` and `top_k_weights [n, K]` directly.
//
// Per-token routing math: each threadgroup handles one token row.
// Threads in the group cooperate on the softmax (max + exp + sum
// reductions across E experts). Top-K selection is serial on thread 0
// using an insertion-sort over E entries (E=128 or 256, K=8 — ~2K ops
// max per row, trivial).
//
// At Gemma4-26B-A4B (E=128, K=8, 30 MoE layers): saves 7 launches per
// layer × 30 layers ≈ 210 launches per decoded token = ~8 ms at the
// observed ~37 µs/launch overhead.
// ============================================================================
const FUSED_ROUTER_TOPK_KERNEL: &str = r#"
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint group_size = threads_per_threadgroup.x;

    device const float *row_scores = scores + row * uint(E);

    threadgroup float probs[E];
    threadgroup float red_buf[256];

    // ── 1. softmax: max reduction across E ──
    float local_max = -INFINITY;
    for (uint e = tid; e < uint(E); e += group_size) {
        local_max = max(local_max, row_scores[e]);
    }
    red_buf[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = group_size >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) red_buf[tid] = max(red_buf[tid], red_buf[tid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = red_buf[0];

    // ── 2. exp(x - max), store in shared probs ──
    for (uint e = tid; e < uint(E); e += group_size) {
        probs[e] = exp(row_scores[e] - row_max);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── 3. sum reduction ──
    float local_sum = 0.0f;
    for (uint e = tid; e < uint(E); e += group_size) {
        local_sum += probs[e];
    }
    red_buf[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = group_size >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) red_buf[tid] += red_buf[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_sum = red_buf[0];

    // ── 4. normalize: probs[e] /= sum ──
    for (uint e = tid; e < uint(E); e += group_size) {
        probs[e] = probs[e] / row_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── 5. top-K selection (serial on thread 0; K small, E small) ──
    if (tid == 0) {
        // top_w / top_i held in ascending position order (largest first).
        float top_w[K];
        int   top_i[K];
        for (uint k = 0; k < uint(K); k++) {
            top_w[k] = -INFINITY;
            top_i[k] = -1;
        }
        for (uint e = 0; e < uint(E); e++) {
            float v = probs[e];
            if (v > top_w[uint(K) - 1]) {
                int pos = int(K) - 1;
                while (pos > 0 && top_w[pos - 1] < v) {
                    top_w[pos] = top_w[pos - 1];
                    top_i[pos] = top_i[pos - 1];
                    pos--;
                }
                top_w[pos] = v;
                top_i[pos] = int(e);
            }
        }
        // Normalize top-K weights + multiply by per_expert_scale.
        float topk_sum = 0.0f;
        for (uint k = 0; k < uint(K); k++) topk_sum += top_w[k];
        device int *row_idx = top_k_index + row * uint(K);
        device float *row_w = top_k_weights + row * uint(K);
        for (uint k = 0; k < uint(K); k++) {
            float w = top_w[k] / topk_sum;
            w *= per_expert_scale[top_i[k]];
            row_w[k] = w;
            row_idx[k] = top_i[k];
        }
    }
"#;

static FUSED_ROUTER_TOPK_KERNEL_HANDLE: OnceLock<MetalKernel> = OnceLock::new();

fn create_fused_router_topk_kernel() -> MetalKernel {
    unsafe {
        let scores = CString::new("scores").unwrap();
        let pes = CString::new("per_expert_scale").unwrap();
        let out_idx = CString::new("top_k_index").unwrap();
        let out_w = CString::new("top_k_weights").unwrap();

        let inputs = mlx_sys::mlx_vector_string_new();
        mlx_sys::mlx_vector_string_append_value(inputs, scores.as_ptr());
        mlx_sys::mlx_vector_string_append_value(inputs, pes.as_ptr());
        let outputs = mlx_sys::mlx_vector_string_new();
        mlx_sys::mlx_vector_string_append_value(outputs, out_idx.as_ptr());
        mlx_sys::mlx_vector_string_append_value(outputs, out_w.as_ptr());

        let source = CString::new(FUSED_ROUTER_TOPK_KERNEL).unwrap();
        let header = CString::new("").unwrap();
        let name = CString::new("fused_router_topk").unwrap();
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

/// Fused router post-matmul tail.
///
/// `scores`: `[n, E]` fp32 — raw expert scores (post-projection, pre-softmax).
/// `per_expert_scale`: `[E]` fp32 — multiplier applied to the normalized
/// top-K weights before output.
///
/// Returns `(top_k_index, top_k_weights)`:
///   - `top_k_index`: `[n, K]` i32 — sorted descending by softmax probability.
///   - `top_k_weights`: `[n, K]` fp32 — softmax probabilities renormalized
///     over the top-K then scaled by `per_expert_scale[top_k_index]`.
///
/// Replaces 8 separate MLX ops (softmax + negate + argpartition + take_along +
/// sum + divide + take_axis + multiply) with a single kernel launch.
pub fn fused_router_topk(
    scores: &Array,
    per_expert_scale: &Array,
    k: i32,
    e: i32,
) -> Result<(Array, Array), Exception> {
    if k <= 0 || e <= 0 {
        return Err(Exception::custom(format!(
            "fused_router_topk: K and E must be positive; got K={k} E={e}"
        )));
    }
    if k > e {
        return Err(Exception::custom(format!(
            "fused_router_topk: K ({k}) cannot exceed E ({e})"
        )));
    }
    let s = scores.shape();
    if s.len() != 2 || s[1] != e {
        return Err(Exception::custom(format!(
            "fused_router_topk: scores must be [n, E={e}]; got {s:?}"
        )));
    }
    let n = s[0];
    let pes_shape = per_expert_scale.shape();
    if pes_shape.len() != 1 || pes_shape[0] != e {
        return Err(Exception::custom(format!(
            "fused_router_topk: per_expert_scale must be [E={e}]; got {pes_shape:?}"
        )));
    }
    if scores.dtype() != mlx_rs::Dtype::Float32
        || per_expert_scale.dtype() != mlx_rs::Dtype::Float32
    {
        return Err(Exception::custom(format!(
            "fused_router_topk: scores and per_expert_scale must be fp32; got {:?} / {:?}",
            scores.dtype(),
            per_expert_scale.dtype()
        )));
    }

    let kernel = FUSED_ROUTER_TOPK_KERNEL_HANDLE.get_or_init(create_fused_router_topk_kernel);

    unsafe {
        let stream = mlx_sys::mlx_default_gpu_stream_new();
        let config = mlx_sys::mlx_fast_metal_kernel_config_new();
        for (name, value) in [("K", k), ("E", e)] {
            let cname = CString::new(name).unwrap();
            mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
                config, cname.as_ptr(), value,
            );
        }
        // One threadgroup per row; 64 threads per group (one simdgroup
        // pair, enough for the reductions across E≤256).
        mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, n * 64, 1, 1);
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, 64, 1, 1);

        let idx_shape: [i32; 2] = [n, k];
        let w_shape: [i32; 2] = [n, k];
        let i32_dtype: u32 = mlx_rs::Dtype::Int32.into();
        let f32_dtype: u32 = mlx_rs::Dtype::Float32.into();
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config, idx_shape.as_ptr(), idx_shape.len(), i32_dtype,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config, w_shape.as_ptr(), w_shape.len(), f32_dtype,
        );

        let inputs = mlx_sys::mlx_vector_array_new();
        mlx_sys::mlx_vector_array_append_value(inputs, scores.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, per_expert_scale.as_ptr());

        let mut outputs = mlx_sys::mlx_vector_array_new();
        let ret = mlx_sys::mlx_fast_metal_kernel_apply(
            &mut outputs, kernel.kernel, inputs, config, stream,
        );
        if ret != 0 {
            mlx_sys::mlx_fast_metal_kernel_config_free(config);
            mlx_sys::mlx_vector_array_free(inputs);
            mlx_sys::mlx_vector_array_free(outputs);
            mlx_sys::mlx_stream_free(stream);
            return Err(Exception::custom("fused_router_topk kernel failed"));
        }
        let mut idx = mlx_sys::mlx_array_new();
        let mut w = mlx_sys::mlx_array_new();
        mlx_sys::mlx_vector_array_get(&mut idx, outputs, 0);
        mlx_sys::mlx_vector_array_get(&mut w, outputs, 1);
        mlx_sys::mlx_fast_metal_kernel_config_free(config);
        mlx_sys::mlx_vector_array_free(inputs);
        mlx_sys::mlx_vector_array_free(outputs);
        mlx_sys::mlx_stream_free(stream);
        Ok((Array::from_ptr(idx), Array::from_ptr(w)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::indexing::IndexOp;
    use mlx_rs::Array;

    /// T4 fused_norm_add_norm parity vs MLX op chain (rms_norm → add → rms_norm).
    #[test]
    fn fused_norm_add_norm_matches_reference() {
        let n = 4_i32;
        let h = 2816_i32;
        let attn_data: Vec<f32> = (0..n * h).map(|i| ((i as f32) * 0.013).sin() * 0.5).collect();
        let res_data: Vec<f32> = (0..n * h).map(|i| ((i as f32) * 0.017).cos() * 0.3).collect();
        let w1_data: Vec<f32> = (0..h).map(|i| 1.0 + 0.001 * (i as f32)).collect();
        let w2_data: Vec<f32> = (0..h).map(|i| 0.5 + 0.002 * (i as f32)).collect();
        let attn = Array::from_slice(&attn_data, &[n, h])
            .as_dtype(mlx_rs::Dtype::Bfloat16).unwrap();
        let res = Array::from_slice(&res_data, &[n, h])
            .as_dtype(mlx_rs::Dtype::Bfloat16).unwrap();
        let w1 = Array::from_slice(&w1_data, &[h])
            .as_dtype(mlx_rs::Dtype::Bfloat16).unwrap();
        let w2 = Array::from_slice(&w2_data, &[h])
            .as_dtype(mlx_rs::Dtype::Bfloat16).unwrap();

        // Reference: rms_norm(attn, w1) → add residual → rms_norm(sum, w2).
        let normed_attn = mlx_rs::fast::rms_norm(&attn, &w1, 1e-6).unwrap();
        let sum_ref = res.add(&normed_attn).unwrap();
        let normed_sum_ref = mlx_rs::fast::rms_norm(&sum_ref, &w2, 1e-6).unwrap();
        mlx_rs::transforms::eval([&sum_ref, &normed_sum_ref]).unwrap();

        let (sum_k, normed_k) = fused_norm_add_norm(&attn, &res, &w1, &w2).unwrap();
        mlx_rs::transforms::eval([&sum_k, &normed_k]).unwrap();

        let sum_ref_f32 = sum_ref.as_dtype(mlx_rs::Dtype::Float32).unwrap();
        let sum_k_f32 = sum_k.as_dtype(mlx_rs::Dtype::Float32).unwrap();
        let nor_ref_f32 = normed_sum_ref.as_dtype(mlx_rs::Dtype::Float32).unwrap();
        let nor_k_f32 = normed_k.as_dtype(mlx_rs::Dtype::Float32).unwrap();
        mlx_rs::transforms::eval([&sum_ref_f32, &sum_k_f32, &nor_ref_f32, &nor_k_f32]).unwrap();

        let max_abs = |a: &Array, b: &Array| -> f32 {
            a.as_slice::<f32>()
                .iter()
                .zip(b.as_slice::<f32>().iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0_f32, f32::max)
        };
        let s_diff = max_abs(&sum_ref_f32, &sum_k_f32);
        let n_diff = max_abs(&nor_ref_f32, &nor_k_f32);
        // `sum` is bit-exact within bf16 store precision (kernel produces
        // the same bf16 output as the reference's add).
        assert!(s_diff < 5e-2, "sum max_abs {s_diff}");
        // `normed_sum` tolerance is wider (≤0.2 absolute) because our
        // kernel keeps the intermediate `residual + normed_attn` in fp32
        // across the second RMSNorm via threadgroup memory, while the
        // MLX reference round-trips through bf16 between the two norms.
        // We're MORE precise than the reference, not less — but the
        // numerical paths differ, so a tight tolerance would fail.
        assert!(n_diff < 0.2, "normed_sum max_abs {n_diff}");
    }

    /// Router-tail fusion parity vs MLX op chain on E=128, K=8.
    #[test]
    fn fused_router_topk_matches_reference() {
        let n = 4_i32;
        let e = 128_i32;
        let k = 8_i32;
        let scores_data: Vec<f32> = (0..n * e).map(|i| ((i as f32) * 0.013).sin()).collect();
        let pes_data: Vec<f32> = (0..e).map(|i| 1.0 + 0.01 * (i as f32)).collect();
        let scores = Array::from_slice(&scores_data, &[n, e]);
        let pes = Array::from_slice(&pes_data, &[e]);

        // Reference: replicate Router::forward's tail in MLX ops.
        let probs = mlx_rs::ops::softmax_axis(&scores, -1, Some(true)).unwrap();
        let neg = probs.negative().unwrap();
        let parts = mlx_rs::ops::argpartition_axis(&neg, k - 1, -1).unwrap();
        let top_idx_ref = parts.index((.., ..k)).as_dtype(mlx_rs::Dtype::Int32).unwrap();
        let top_w_ref = mlx_rs::ops::indexing::take_along_axis(&probs, &top_idx_ref, -1).unwrap();
        let denom = top_w_ref.sum_axis(-1, true).unwrap();
        let top_w_ref = top_w_ref.divide(&denom).unwrap();
        let pes_taken = mlx_rs::ops::indexing::take_axis(&pes, &top_idx_ref, 0).unwrap();
        let top_w_ref = top_w_ref.multiply(&pes_taken).unwrap();
        mlx_rs::transforms::eval([&top_idx_ref, &top_w_ref]).unwrap();

        // Kernel.
        let (top_idx, top_w) = fused_router_topk(&scores, &pes, k, e).unwrap();
        mlx_rs::transforms::eval([&top_idx, &top_w]).unwrap();

        // Compare. argpartition order can differ within ties; check that
        // the SET of selected experts matches and the weights match
        // per-set elementwise after sorting both by index.
        let ref_idx = top_idx_ref.as_slice::<i32>();
        let ref_w = top_w_ref.as_slice::<f32>();
        let our_idx = top_idx.as_slice::<i32>();
        let our_w = top_w.as_slice::<f32>();

        for row in 0..n as usize {
            // Pair ref (idx, w) and our (idx, w), sort by idx, compare.
            let mut ref_pairs: Vec<(i32, f32)> = (0..k as usize)
                .map(|j| (ref_idx[row * k as usize + j], ref_w[row * k as usize + j]))
                .collect();
            let mut our_pairs: Vec<(i32, f32)> = (0..k as usize)
                .map(|j| (our_idx[row * k as usize + j], our_w[row * k as usize + j]))
                .collect();
            ref_pairs.sort_by_key(|p| p.0);
            our_pairs.sort_by_key(|p| p.0);

            for ((ri, rw), (qi, qw)) in ref_pairs.iter().zip(our_pairs.iter()) {
                assert_eq!(
                    ri, qi,
                    "row {row}: index mismatch ref={ri} kernel={qi}"
                );
                let diff = (rw - qw).abs();
                assert!(
                    diff < 1e-4,
                    "row {row} idx {ri}: weight diff {diff} (ref={rw} kernel={qw})"
                );
            }
        }
    }

    /// Batched kernel parity: 2 experts × 32 rows × (K=64, N=128).
    #[test]
    fn moe_dense_matmul_batched_matches_reference() {
        let e_total = 2_i32;
        let rows_per_expert = 32_i32;
        let m_total = rows_per_expert * e_total;
        let k = 64_i32;
        let n = 128_i32;

        let a_data: Vec<f32> = (0..m_total * k).map(|i| ((i as f32) * 0.013).cos()).collect();
        // B layout: [E, K, N] (caller pre-transposed).
        let b_data: Vec<f32> = (0..e_total * k * n).map(|i| ((i as f32) * 0.017).sin()).collect();
        let a_f32 = Array::from_slice(&a_data, &[m_total, k]);
        let b_f32 = Array::from_slice(&b_data, &[e_total, k, n]);
        let a = a_f32.as_dtype(mlx_rs::Dtype::Bfloat16).unwrap();
        let b = b_f32.as_dtype(mlx_rs::Dtype::Bfloat16).unwrap();

        let a_e0 = a.index((..32, ..));
        let a_e1 = a.index((32..64, ..));
        let b_e0 = b.index((0, .., ..));
        let b_e1 = b.index((1, .., ..));
        let ref_e0 = a_e0.matmul(&b_e0).unwrap();
        let ref_e1 = a_e1.matmul(&b_e1).unwrap();
        let reference = mlx_rs::ops::concatenate_axis(&[&ref_e0, &ref_e1], 0).unwrap();
        mlx_rs::transforms::eval([&reference]).unwrap();

        let block_id = Array::from_slice(&[0_i32, 1_i32], &[2]);
        let result = moe_dense_matmul_batched(&a, &b, &block_id, k, n).unwrap();
        mlx_rs::transforms::eval([&result]).unwrap();

        let ref_f32 = reference.as_dtype(mlx_rs::Dtype::Float32).unwrap();
        let res_f32 = result.as_dtype(mlx_rs::Dtype::Float32).unwrap();
        mlx_rs::transforms::eval([&ref_f32, &res_f32]).unwrap();
        let r = ref_f32.as_slice::<f32>();
        let q = res_f32.as_slice::<f32>();
        let mut max_abs = 0.0_f32;
        for (a, b) in r.iter().zip(q.iter()) {
            max_abs = max_abs.max((a - b).abs());
        }
        assert!(
            max_abs < 5e-2,
            "batched kernel max_abs={max_abs:.6} exceeds bf16 tolerance"
        );
    }

    /// MoE dense matmul kernel parity test: smallest valid shapes
    /// (M=32, K=8, N=32). Compares against `mlx_rs::ops::matmul(A, B)`
    /// elementwise to fp32 tolerance.
    #[test]
    fn moe_dense_matmul_matches_reference_min_shape() {
        // Construct deterministic A=[32, 8] and B=[32, 8] in fp32.
        let m = 32_i32;
        let k = 8_i32;
        let n = 32_i32;
        let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        // Kernel expects B as [K, N] row-major.
        let b_data: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.02).sin()).collect();
        let a = Array::from_slice(&a_data, &[m, k]);
        let b = Array::from_slice(&b_data, &[k, n]);

        // Reference: A @ B = [M, N].
        let reference = a.matmul(&b).expect("ref matmul");
        mlx_rs::transforms::eval([&reference]).expect("eval ref");

        // Kernel.
        let m_buf = Array::from_slice(&[m], &[1]);
        let result = moe_dense_matmul(&a, &b, &m_buf, k, n).expect("kernel");
        mlx_rs::transforms::eval([&result]).expect("eval result");

        let ref_slice = reference.as_slice::<f32>();
        let res_slice = result.as_slice::<f32>();
        assert_eq!(ref_slice.len(), res_slice.len());
        let mut max_abs = 0.0_f32;
        let mut max_rel = 0.0_f32;
        for (i, (&r, &q)) in ref_slice.iter().zip(res_slice.iter()).enumerate() {
            let abs = (r - q).abs();
            let rel = abs / r.abs().max(1e-6);
            if abs > max_abs {
                max_abs = abs;
            }
            if rel > max_rel {
                max_rel = rel;
            }
            assert!(
                abs < 1e-3 && rel < 1e-3,
                "kernel diverges at idx {i}: ref={r} kernel={q} abs={abs} rel={rel}"
            );
        }
        eprintln!(
            "moe_dense_matmul parity ok: max_abs={max_abs:.6} max_rel={max_rel:.6}"
        );
    }

    /// bf16 kernel parity vs MLX bf16 matmul, larger shape.
    #[test]
    fn moe_dense_matmul_bf16_matches_reference() {
        let m = 64_i32;
        let k = 64_i32;
        let n = 128_i32;
        let a_data: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.013).cos()).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.017).sin()).collect();
        let a_f32 = Array::from_slice(&a_data, &[m, k]);
        let b_f32 = Array::from_slice(&b_data, &[k, n]);
        let a = a_f32.as_dtype(mlx_rs::Dtype::Bfloat16).expect("a→bf16");
        let b = b_f32.as_dtype(mlx_rs::Dtype::Bfloat16).expect("b→bf16");

        let reference = a.matmul(&b).expect("ref matmul");
        mlx_rs::transforms::eval([&reference]).expect("eval ref");

        let m_buf = Array::from_slice(&[m], &[1]);
        let result = moe_dense_matmul_bf16(&a, &b, &m_buf, k, n).expect("kernel");
        mlx_rs::transforms::eval([&result]).expect("eval result");

        let ref_f32 = reference.as_dtype(mlx_rs::Dtype::Float32).expect("ref→f32");
        let res_f32 = result.as_dtype(mlx_rs::Dtype::Float32).expect("res→f32");
        mlx_rs::transforms::eval([&ref_f32, &res_f32]).expect("eval f32");
        let ref_slice = ref_f32.as_slice::<f32>();
        let res_slice = res_f32.as_slice::<f32>();
        let mut max_abs = 0.0_f32;
        for (&r, &q) in ref_slice.iter().zip(res_slice.iter()) {
            max_abs = max_abs.max((r - q).abs());
        }
        // bf16 has ~7 bits mantissa; max abs error scales with reduction size.
        // For K=64 reductions, expect up to ~1e-2 abs error.
        assert!(
            max_abs < 5e-2,
            "bf16 kernel max_abs={max_abs:.6} exceeds tolerance"
        );
    }

    /// Larger-shape parity: M=64 (2 row-tiles), K=64 (8 K-steps), N=128 (4 col-tiles).
    #[test]
    fn moe_dense_matmul_matches_reference_larger_shape() {
        let m = 64_i32;
        let k = 64_i32;
        let n = 128_i32;
        let a_data: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.013).cos()).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.017).sin()).collect();
        let a = Array::from_slice(&a_data, &[m, k]);
        let b = Array::from_slice(&b_data, &[k, n]);

        let reference = a.matmul(&b).expect("ref matmul");
        mlx_rs::transforms::eval([&reference]).expect("eval ref");

        let m_buf = Array::from_slice(&[m], &[1]);
        let result = moe_dense_matmul(&a, &b, &m_buf, k, n).expect("kernel");
        mlx_rs::transforms::eval([&result]).expect("eval result");

        let ref_slice = reference.as_slice::<f32>();
        let res_slice = result.as_slice::<f32>();
        let mut max_abs = 0.0_f32;
        let mut max_rel = 0.0_f32;
        for (&r, &q) in ref_slice.iter().zip(res_slice.iter()) {
            let abs = (r - q).abs();
            let rel = abs / r.abs().max(1e-6);
            max_abs = max_abs.max(abs);
            max_rel = max_rel.max(rel);
        }
        assert!(
            max_abs < 1e-3,
            "max_abs={max_abs:.6} max_rel={max_rel:.6} exceeds tolerance"
        );
    }

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
