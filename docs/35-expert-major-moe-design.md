# #35 — Expert-Major MoE MLP — design doc for a dedicated session

**Status**: pending, scope refined 2026-05-18. Estimated 1–2 days of focused
work. Unblocked from #27 conceptually (simdgroup_matrix shines on dense
tiles, not row-vector workloads).

## Motivation

Hermes 5K prefill on Gemma4-26B-A4B-it currently runs at **7.4–8.6
prompt-tok/s** (see `perf-benchmarks-hermes.md`). Profiling indicates the
MoE block dominates prefill cost — specifically, `Experts::forward_topk`
in `gemma4-mlx/src/model.rs:765-800`, which routes `n×k` token-expert
pairs through per-token vector-matrix matmuls.

For decode (n=1, k=4), this path is fine — each expert sees ≤1 token and
no batching helps. The optimization targets **prefill** where, on hermes
5K with 64 experts and top_k=4, each expert receives an average of
`n × k / num_experts = 5000 × 4 / 64 ≈ 313 tokens` per layer. That's a
perfect dense-matmul tile workload for `simdgroup_matrix<float, 8, 8>`.

**Conservative target**: 2× hermes-5K prefill throughput (15–17 tok/s).
**Stretch**: 3× (22–25 tok/s) if memory bandwidth allows.

## Current path (token-major)

```rust
// gemma4-mlx/src/model.rs:765
fn forward_topk(&mut self, hidden_states, top_k_index, top_k_weights) {
    // hidden_states: [n, H]
    // top_k_index:   [n, k]    — expert ids
    // top_k_weights: [n, k]    — routing weights

    // Gather per-(token,expert) weight slices:
    let gate_up = take_axis(&self.gate_up_proj, top_k_index, 0)?;
    // gate_up: [n, k, 2I, H] (after transpose [0,1,3,2]: [n, k, H, 2I])

    // Per-token vector-matrix matmul:
    let projected = hidden_states.reshape(&[n, 1, 1, H])
                                 .matmul(&gate_up)?;     // → [n, k, 1, 2I]
    // ...split, activate, down_proj, weighted sum...
}
```

Each `(token, expert)` pair triggers a `1×H @ H×2I` matmul. For n=5000,
k=4, that's 20 000 separate row-vector matmuls per MoE layer. MLX
batches them into a single 4D matmul, but the underlying kernels can't
exploit tile reuse across tokens that hit the same expert.

## Proposed path (expert-major)

```
Step 1: bucket tokens by expert
   Input:  top_k_index [n, k]
   Output: for each expert e, a list of (token_idx, k_slot) pairs
           plus token_count[e]

Step 2: per-expert dense matmul (simdgroup_matrix tiles)
   For each expert e with N_e routed tokens:
     gather  X_e: [N_e, H]     ← hidden_states[token_idx_list]
     matmul  Y_e: [N_e, 2I] = X_e @ gate_up_proj[e]    (dense, tile-friendly)
     split   gate, up = Y_e.chunk(2, dim=-1)
     act     gate = GeLU(gate)
     mul     z_e = gate * up                            [N_e, I]
     matmul  out_e: [N_e, H] = z_e @ down_proj[e]      (dense)

Step 3: scatter with routing weights
   For each (e, j) in expert e's token list:
     output[token_idx[j]] += top_k_weights[token_idx[j], k_slot[j]] * out_e[j]
```

## Implementation breakdown

### Phase 1 — Rust scaffolding (½ day)

Files to touch:
- `gemma4-mlx/src/model.rs` — new `Experts::forward_topk_expert_major()`
  method behind `GEMMA4_EXPERT_MAJOR_MOE=1` env gate. Existing
  `forward_topk` untouched (fallback path).
- Add `bucket_by_expert(top_k_index: &Array, num_experts: i32) ->
  Result<Vec<ExpertBucket>>` helper. Returns:
  ```rust
  struct ExpertBucket {
      expert_id: i32,
      token_indices: Array,    // [N_e] i32
      k_slots: Array,          // [N_e] i32 (which of the top-k slots)
      n_tokens: i32,
  }
  ```
- Use `argsort` + `cumsum` on flattened `top_k_index` to compute offsets
  without per-expert Python-side loops.

### Phase 1 + 2 — Status: LANDED, no speedup yet, by design

Phase 1 (`7b98d9f`) — `forward_topk_expert_major` with CPU bucketing
behind `GEMMA4_EXPERT_MAJOR_MOE=1`. Validated correct; ~1.6× slower
than `forward_topk` due to per-layer GPU→CPU sync forced by reading
`top_k_index` to a Vec for the bucket loop.

Phase 2 (`65e999e`) — `forward_topk_expert_major_v2` with GPU
bucketing (argsort + take, no CPU sync) behind
`GEMMA4_EXPERT_MAJOR_MOE=2`. Validated correct; performance
equivalent to `forward_topk`. **No speedup achieved, by design.**

**Empirical lesson:** stock MLX matmul on the sorted layout is
structurally equivalent in cost to `forward_topk`. MLX's `take_axis`
+ batched matmul fuses into the same per-row vector-matrix kernel
regardless of whether the input rows are in token-major or
expert-sorted order. The Phase 1 design assumption that "per-expert
MLX dispatch validates the algorithm and shows the speedup" was
incorrect: stock MLX matmul cannot exploit "many tokens share one
expert weight matrix" without a custom kernel that uses tile reuse.

Phase 2's actual deliverable is the **sorted input layout**:
`(sorted_token_indices, sorted_k_slots, sorted_experts)` produced
entirely on GPU. This is the input format for the Phase 3 custom
Metal kernel.

Knob finalized at the call site:
- unset / `=0` → `forward_topk` (token-major, fastest with stock MLX)
- `=1` → v1 CPU bucketing (slower; kept as a per-bucket gate
  reference for the kernel work)
- `=2` → v2 GPU bucketing (correctness-equivalent to forward_topk;
  produces the layout Phase 3 needs)

### Phase 3 — Dense Metal kernel (½–1 day)

Replace the `x_e @ w_gu` matmul with a custom Metal kernel using
`simdgroup_matrix<float, 8, 8>` tiles. Reference:
[tinygrad's metal_matmul.py](https://github.com/tinygrad/tinygrad/blob/3f2d401464461972e62ef5ace365bf3621b962c0/extra/gemm/metal_matmul.py)
demonstrates the canonical pattern:

```metal
simdgroup_float8x8 acc[4][4];  // 4×4 = 16 8×8 output tiles per threadgroup
simdgroup_float8x8 A[4], B[4];
for (uint k = 0; k < K; k += 8) {
    simdgroup_load(A[0], a_ptr + k +  0*K, K, ulong2(0, 0));
    simdgroup_load(A[1], a_ptr + k +  8*K, K, ulong2(0, 0));
    simdgroup_load(A[2], a_ptr + k + 16*K, K, ulong2(0, 0));
    simdgroup_load(A[3], a_ptr + k + 24*K, K, ulong2(0, 0));
    simdgroup_load(B[0], b_ptr + 0 + k*N, N, ulong2(0, 0));
    // ... B[1..3]
    simdgroup_multiply_accumulate(acc[i][j], A[i], B[j], acc[i][j]);
    // 16 accumulate calls
}
// store 16 tiles to output [32 rows × 32 cols]
```

Grid: `(M/32, N/32, 1)` threadgroups, each `(32, LID=2, 1)` threads —
processes a 32×(32*LID) output tile per threadgroup.

For our shapes per expert:
- Gemma4-26B-A4B: `H=5376, I=14336` (so `2I=28672`).
- Smallest N_e for the kernel: 32 rows. Smaller → fall back to MLX matmul.
- Largest expected N_e on hermes 5K: ~500 rows.

Two kernels needed:
1. `expert_gate_up_matmul`: `[N_e, H] @ [H, 2I]` (output 28672 cols)
2. `expert_down_matmul`: `[N_e, I] @ [I, H]` (output 5376 cols)

Both follow the same tile loop. The activation + element-wise multiply
between them is a separate small kernel (or use MLX ops on the
intermediate `[N_e, 2I]` result).

**Validation**: side-by-side against MLX `matmul` on a fixed input;
require max abs error < 1e-3 in bf16.

### Phase 4 — Bench + lock in (¼ day)

- Run hermes 5K prefill with `GEMMA4_EXPERT_MAJOR_MOE=1` vs unset.
- Update `perf-benchmarks-hermes.md` with prefill tok/s, peak GPU,
  acceptance ratios (should be identical — this is correctness-preserving).
- If ≥1.5× speedup, make `GEMMA4_EXPERT_MAJOR_MOE=1` default for
  prefill (decode auto-falls-back via Phase-2 skip-gate).

## Risk register

1. **Bucket-and-scatter overhead** could erase gains for small n.
   Mitigation: skip-gate at `n < EXPERT_MAJOR_MIN_PROMPT_TOKENS` (default
   64) for the entire MoE block, not just per-expert.

2. **GPU memory pressure** — gathering `x_e` per expert duplicates token
   activations. For n=5000, top_k=4, that's 20 000 copies of H=5376 fp16
   = 200 MB extra. Should fit easily in 96 GB.

3. **Kernel correctness regressions** — the tinygrad reference is for
   contiguous row-major fp32. We need bf16 with strided slices (the
   gather may produce non-contiguous outputs). Mitigation: validate
   on the contiguous fast path first; fall back to MLX matmul for
   non-contiguous inputs in v1.

4. **Quantized weights** — Gemma4-26B's `gate_up_proj` is fp16 in the
   checkpoint we benchmark, but the MTPLX-Optimized variants ship Q4.
   v1 only handles fp16/bf16. Q4 expert MLP needs dequant inside the
   kernel (or a separate path). Track as a follow-up.

## Files to read first

- `gemma4-mlx/src/model.rs:751-803` — current `Experts::forward_topk`.
- `gemma4-mlx/src/model.rs:705-744` — `Router::forward` (provides
  `top_k_index` and `top_k_weights`).
- `mlx-rs-core/src/metal_kernels.rs:948-1240` — existing
  `tq_sdpa_4bit_online_simd` kernel as a layout/grid reference (and a
  cautionary tale for row-vector workloads — see #27).
- [tinygrad metal_matmul.py](https://github.com/tinygrad/tinygrad/blob/3f2d401464461972e62ef5ace365bf3621b962c0/extra/gemm/metal_matmul.py) —
  canonical 4×4 acc tile loop.

## Out-of-scope for this task

- Decode optimization (n=1) — no win available, skip-gate handles it.
- Q4 expert weights — track as a follow-up after the fp16 path is proven.
- Hardware-specific tuning (M2/M3/M4 simdgroup width variations).
- Auto-grad / training support — inference only.
