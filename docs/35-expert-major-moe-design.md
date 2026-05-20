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

### Phase 3 — LANDED, kernel correct, no speedup

Phases 3-6 (`9dfd86a`, `0981756`, `1efb9ac`, `a02db78`, `69d86cf`) shipped:
- `moe_dense_matmul` (fp32, simdgroup_matrix<float,8,8>, bit-exact parity)
- `moe_dense_matmul_bf16` (bf16, same kernel logic, ~5e-2 abs tol)
- `forward_topk_expert_major_v3`: CPU bucketing + fp32 kernel
- `forward_topk_expert_major_v4`: GPU bucketing + fp32 kernel
- `forward_topk_expert_major_v5`: GPU bucketing + bf16 kernel

All five expert-major variants land at the same wall time as
`forward_topk` (~3.4 tok/s decode on gemma-4-26B-A4B-it).

**Diagnosis:** the bottleneck is **per-expert kernel-launch overhead**.
With 64 experts × 64 MoE layers = 4096 kernel launches per decode step,
at ~100µs Metal launch overhead each, that's ~400ms/token just in
launches — accounting for ~100% of the 294ms/token decode time.

### Phase 7 (FUTURE) — single batched all-experts kernel launch

This is the only remaining lever and requires:

1. **GPU input packing.** Currently each expert's bucket is processed
   separately, requiring per-expert padding within the per-launch call.
   Move to a global packed layout `[M_total_padded, H]` where each
   expert's contiguous block is padded to a multiple of 32 rows at
   construction time. Compute `padded_starts[E]` via cumsum on GPU;
   scatter sorted tokens into the packed layout.

2. **Per-block expert metadata.** Build `block_expert_id[M_total_padded/32]`
   on GPU — a lookup table saying which expert each 32-row block belongs
   to. Build it from `padded_starts` via a `searchsorted`-style pattern
   or a per-block arange comparison.

3. **New kernel variant `moe_dense_matmul_batched`.** Accepts:
   - `packed_x: [M_total_padded, H]`
   - `weights: [E, N, K]` (gate_up_proj or down_proj)
   - `block_expert_id: [M_total_padded/32]`
   - `expert_counts: [E]` (for bounds masking padded rows)
   Each threadgroup reads its expert id from the lookup table, computes
   the per-expert weight offset (`e * N * K`), and runs the same
   simdgroup_matrix 4×4 acc-tile pattern. Single Metal launch covers
   all experts.

4. **Scatter back.** After the matmul, scatter from packed layout back
   to per-token output positions, masking out padded rows.

**Expected impact**: 4096 launches/step → 2 launches/step (one for
gate_up + one for down_proj per MoE layer = 128 launches/step
total). Should reclaim the ~400ms/token launch overhead. Realistic
target: 2-3× decode speedup on Gemma4-26B-A4B-it MoE workloads.

### Phase 7 — LANDED (`bec97ca`), regressed 4.2× — root cause: weight transpose per call

Bench on gemma-4-26B-A4B-it with `CHAT_MAX_TOKENS=64`, prompt "Hi"
(`77d8f69` added the harness):

| Variant | wall (64 tok) | decode tok/s |
|---|---|---|
| default (forward_topk, token-major) | 18.6s | 3.4 |
| v6 (phase 7 batched single launch) | 78.1s | 0.82 |

v6's user CPU time dropped from 55s (v4) → 16s (v6), confirming launch
overhead IS reduced as designed. But wall time regressed because the
Rust setup cost per MoE layer now dominates:

1. `transpose_axes([0,2,1])` on `[E, 2I, H]` weights materializes
   ~5.5 GB of bf16 memory bandwidth per layer → 350 GB / decode step
   at 64 layers, ~1 second/token at the memory-bandwidth limit.
2. `scatter_add_single` for packed_x: n_k × H = ~50 MB/layer written.
3. `take_axis` gather of n_k real rows from packed output: another
   50 MB/layer.

The kernel itself is fast and correct; the per-layer dispatch
overhead from transposing the per-expert weight matrices wipes out
all the launch-overhead savings.

### Phase 8 (FUTURE) — fix the transpose

Two options:

1. **Cache the [E, K, N] view at model load.** Adds a parallel
   `gate_up_proj_kn` and `down_proj_kn` field that's the
   pre-transposed view, materialized once. Memory cost: 10.8 GB of
   additional bf16 storage (gate_up_proj is `[64, 16384, 5376] bf16`).
   On a 96 GB machine that's acceptable for the prefill/decode
   speedup it would unlock.

2. **Modify the kernel to read [E, N, K] directly via simdgroup_load's
   transpose flag.** Eliminates the materialization entirely. Risk:
   the transpose flag had unclear behavior in our earlier
   single-expert kernel attempts (zero output). Now that we know the
   `set_grid` convention, those earlier failures may have been the
   real culprit and transpose-load might just work. Worth trying.

Option 2 is cheaper and more elegant. Option 1 is the safe fallback.

Estimated time: ½ day for option 2 (rewrite kernel + parity test).
If it fails, fall back to option 1 (~½ day for the loader changes).

### Phase 8 — LANDED (`5c33a9d`), revealing deeper architectural issues

Option 2 (`e369569`) was attempted first and failed — the transpose-flag
on simdgroup_load produces the same max_abs=0.5 broken output we saw
in the #27 experiment. Confirmed unreliable.

Option 1 (`5c33a9d`) landed: lazy-cached `[E, K, N]` views in
`gate_up_proj_kt` / `down_proj_kt` fields on `Experts`. Materialized
once on first v6 call; subsequent calls read the cached views with
no transpose overhead. Memory cost: ~22 GB additional bf16 storage.

Bench (64 tokens, prompt 'Hi'):
| variant                                    | tok/s | output |
|--------------------------------------------|-------|--------|
| default forward_topk                       | 7.78  | HiHi…  |
| v6 (phase 7 batched) + phase 8 cache       | 0.95  | 1111…  |
| v6 + cache + skip-gate (commit `5c33a9d`)  | 7.78  | HiHi…  |

The cache fixed the per-call transpose materialization but revealed
two deeper issues v6 still hits:

1. **Decode is the wrong workload.** With batch=1 (n=1 token, k=4
   experts), v6's 32-row bucket padding produces 32× redundant FLOPS
   per bucket vs forward_topk's per-token dispatch. Dense expert-major
   matmul only wins when `tokens_per_expert >> 1` — that's prefill,
   not decode.

2. **bf16 accumulator divergence.** Kernel accumulates in bf16
   (Metal blocks fp32→bf16 simdgroup_matrix narrowing). For K=H=5376
   reductions, bf16's 7 mantissa bits accumulate enough rounding error
   to flip argmax decisions. v6 produced `1111…` while baseline
   produced `HiHi…` — both are valid model continuations for slightly
   different logits, but they prove the precision floor is too high.

`5c33a9d` adds `EXPERT_MAJOR_MIN_PROMPT_TOKENS` (default 256) — when
`n < min_tokens`, v6 falls through to `forward_topk`. Decode bench
matches default tok/s + output. Prefill path remains gated; actual
benefit requires a prefill-dominated workload to measure.

### Phase 9 — LANDED (`c333c45`), **11× prefill speedup validated**

Mixed-precision accumulation (bf16 A/B + fp32 acc + fp32 output)
works on Apple Metal — `simdgroup_multiply_accumulate(fp32_acc,
bf16_a, bf16_b, fp32_acc)` is a native overload. Eliminates the
K=5376 bf16 accumulator-rounding divergence.

Bench (hermes 5K prefill, gemma-4-26B-A4B-it, max_tokens=2):

| Variant                              | wall    | prompt-tok/s | speedup |
|--------------------------------------|---------|--------------|---------|
| default `forward_topk` (token-major) | **487s**| ~10.6        | 1×      |
| v6 phase 9 batched + fp32 acc        | **42s** | **~123**     | **~11×**|

Both produced the same first emitted token ('I'). Correctness
preserved at argmax level.

The phase 7 (batched single-launch) + phase 8 (pre-transpose cache)
+ phase 9 (fp32 mixed-precision accumulation) architecture is now
**validated end-to-end** on the workload it was designed for: long
prefill on MoE models.

#### Phase 9 follow-up — chunk-size sweep + skinny path + Qwen3.6

**Chunk-size sweep on Gemma4-26B-A4B-it hermes 5K prefill:**

| chunk | default forward_topk | v6           | v6 speedup |
|-------|---------------------|--------------|------------|
| 64    | 537.40s             | **46.66s**   | **11.5×**  |
| 128   | (DNF / slower)      | 27.10s       | ≥20×       |
| 256   | (DNF / slower)      | 16.66s       | ≥30×       |

Larger chunks consistently better for v6 (more amortization per launch).
Default `GEMMA4_PREFILL_CHUNK=64` is sub-optimal — should be raised
in conjunction with v6.

**v7 (skinny-path variant)** at `=7` — splits experts into fat
(n_e ≥ 32) and skinny (<32). Fat → kernel, skinny → MLX matmul.
Designed to eliminate v6's padding-to-32 FLOPs waste on tiny buckets.

Empirically v7 is **2.5-2.7× slower than v6** across all chunks:

| chunk | v6     | v7      |
|-------|--------|---------|
| 64    | 48.70s | 133.61s |
| 128   | 29.04s |  73.34s |
| 256   | 17.36s |  41.83s |

Root cause: at chunk=64, all 64 experts have n_e ≈ 4 (all skinny),
so v7 dispatches 64 × 40 layers × ~80 chunks = ~327k MLX matmul calls
per prefill. v6 does ~5k batched kernel calls total. Dispatch
overhead trumps padding waste at this scale.

Conclusion: **v6 padded approach is the right architecture for
chunked production prefill.** v7 kept at `=7` for future workloads
where most buckets are fat; not the default.

**Qwen3.6-35B-A3B-4bit:** investigated, NOT ported.
`qwen3.6-mlx/src/moe.rs:120` already uses MLX's `QuantizedSwitchLinear`
which internally implements `gather_qmm` — MLX's optimized expert-major
batched matmul on 4-bit quantized weights. The existing code does:
  1. `gather_sort(x, indices)` — sorts tokens by expert (≈ our Phase 2)
  2. `quantized_switch_linear.apply(x_sorted, indices_sorted, true)` —
     batched Q4 matmul per expert
  3. `scatter_unsort` to restore per-token output

This is structurally equivalent to v6 but stays in Q4 throughout
(no bf16 cache, no fp32 promotion). Qwen3.6-35B-A3B already lands
at 362 prompt-tok/s on hermes 5K — ~3× faster than Gemma4 + v6 even
though Gemma4 just got an 11× speedup from v6.

**The takeaway**: #35 was specifically valuable for Gemma4's
non-quantized MoE path. Models routing through MLX's
QuantizedSwitchLinear already have the equivalent optimization.
Porting v6 to Qwen3.6 would require either:
- A Q4-aware kernel variant (kernel rewrite + MLX 4-bit semantics)
- Pre-dequantizing weights at load (≈ 40 GB of bf16 cache for
  Qwen3.6-35B-A3B — risky on 96 GB systems alongside everything else)

Neither is justified given Qwen3.6 is already faster than v6-Gemma4.

#### Phase 10 (FUTURE FOLLOW-UP) — optimization not validation

To make v6 production-ready for prefill:

1. **fp32 accumulator workaround.** Two options:
   (a) Store fp32 acc tiles to a `[M_total_padded, N]` fp32 output
       buffer; do a separate fp32→bf16 narrowing pass on the result.
       Doubles output bandwidth but preserves precision.
   (b) Cast acc tiles through threadgroup memory: `simdgroup_store(acc,
       tg_buf, stride)` then per-thread fp32→bf16 conversion + `simdgroup_load`
       of the bf16 view. ~2× ops but stays in registers.

2. **Prefill-targeted bench.** `chat_gemma4` with hermes 5K prompt at
   `--max-tokens 1` measures pure prefill. Compare default vs v6 at
   that workload. The phase 7 launch-reduction + phase 8 cache should
   show 1.5-2× prefill speedup if the architecture is sound.

3. **Fuse the packed-x scatter into the sorted gather.** Currently:
   sorted_x = take(h_bf16, sorted_token_indices); then scatter into
   packed_x at target_in_packed. The take produces a contiguous
   [n_k, H] intermediate that we immediately re-scatter — wasteful.
   Direct gather into the packed layout would halve the scatter
   bandwidth.

Estimated: ½ day for prefill bench (1), ½ day for fp32 acc workaround
(2), ½ day for scatter fusion (3). Total ~1.5 days to validate the
phase 7+8 architecture actually delivers on prefill.

**Cost**: ~300 LOC of GPU bucketing + new kernel + dispatch wrapper.
Same complexity tier as the original phase 3 kernel implementation.

Files to touch:
- `mlx-rs-core/src/metal_kernels.rs`: add `moe_dense_matmul_batched`
  kernel + dispatch wrapper. Parallel to `moe_dense_matmul_bf16`.
- `gemma4-mlx/src/model.rs`: add `forward_topk_expert_major_v6` using
  the batched kernel. Gate at `GEMMA4_EXPERT_MAJOR_MOE=6`.

This is the single highest-leverage remaining task for Gemma4 MoE
prefill/decode performance.

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
