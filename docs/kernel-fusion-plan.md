# Kernel-fusion plan — Gemma4 / Qwen3.6 long-context decode

**Goal**: reduce per-decode-token Metal launch overhead, which is the
dominant cost at >5K context (~80 ms/token observed on UD-MLX-4bit at
12.7K context vs ~2 ms theoretical bandwidth limit).

**Hardware**: Apple Silicon GPU via MLX. Per-launch overhead ~50–100 µs.

## Inventory: ops per layer per decode token

Counted statically from `DecoderLayer::forward` in
`gemma4-mlx/src/model.rs`. Numbers below are per layer per decoded
token, MoE-enabled layer:

| Block | Ops | Detail |
|---|---|---|
| Outer residual stream | 6 | 4 RMSNorms + 2 adds |
| Attention | 8–9 | q/k/v_proj (3) + q_norm + k_norm + RoPE-q + RoPE-k + SDPA + o_proj |
| Dense MLP | 4 | gate_proj + up_proj + fused_swiglu (1) + down_proj |
| MoE block | 14 | pre/post layernorms (3) + router (4) + 3 gather_qmm + fused_swiglu + scatter/reduce ops (3) |
| **Total per layer** | **~32** | |

Gemma4-26B-A4B has 30 layers. Decode token cost:
**~900 Metal launches per token**.

At 50–100 µs per launch overhead: **45–90 ms/token launch budget alone**.
Observed: 80 ms/token at 12.7K context. **Matches almost exactly.**

This is the bottleneck. Bandwidth-bound work for the same step is
~2 ms. Compute is ~5–10 ms. Launches dominate.

## Fusion targets, ranked by launch-count savings

### T1: Fused RMSNorm + Q/K/V projection
- Current: `input_layernorm → q_proj → k_proj → v_proj` = 4 launches
- Fused: 1 kernel produces normalized hidden + 3 stacked outputs
- Savings: **3 launches/layer × 30 layers = 90/token (≈ 5–9 ms)**
- Kernel reads: hidden_states once (instead of 4×); single scan
- Implementation: ~300 LOC Metal (Q4-aware variant for UD; bf16 variant for fallback)

### T2: Fused router (softmax + top-k + normalize + per-expert-scale)
- Current: `scores → softmax → argpartition → take_along + take → divide + multiply` = 5–6 launches in `Router::forward`
- Fused: 1 kernel does the full routing pipeline (input: scores → output: top_k_index, top_k_weights)
- Savings: **4–5 launches/MoE-layer × ~30 MoE layers = 120–150/token (≈ 6–15 ms)**
- E=128/256 experts, top_k=8 — small kernel, easy partial sort
- Implementation: ~200 LOC Metal; pure routing math, no model-weight reads

### T3: Fused gate_proj + up_proj + swiglu activation (dense MLP)
- Current: `gate_proj + up_proj + fused_swiglu` = 3 launches
- Fused: 1 kernel computes gate, up, multiplies through SwiGLU
- Savings: **2 launches/layer × 30 layers = 60/token (≈ 3–6 ms)**
- For UD: Q4 gate_proj + Q4 up_proj — need Q4-dequant inside fused matmul. Complex.
- For bf16: simpler.
- Implementation: 200–500 LOC depending on Q4 awareness

### T4: Fused post_attention_layernorm + residual_add + pre_feedforward_layernorm
- Current: `post_attn_ln → add → pre_ff_ln` = 3 launches
- Fused: 1 kernel reads attn_out and residual once, emits normalized hidden for FF block
- Savings: **2 launches/layer × 30 = 60/token (≈ 3–6 ms)**
- Implementation: ~150 LOC Metal (pure elementwise, no GEMM)
- Similar fusion before attention block: `add(residual, ff_out) → input_layernorm_next_layer` — but cross-layer, awkward.

### T5: Fused Q-norm + RoPE (per head, per Q and K)
- Current: q_norm + apply_rope on Q, k_norm + apply_rope on K = 4 launches (mlx-rs-core has `per_position_rope` which combines RoPE with the implicit reshape — need to check)
- Fused: combined q-norm-then-rope kernel for both Q and K
- Savings: **2 launches/layer × 30 = 60/token (≈ 3–6 ms)**
- Implementation: extension of existing `per_position_rope` to optionally apply RMSNorm before rotation; ~100 LOC

### T6: Fused MoE bookkeeping (post-down scatter + post_feedforward_layernorm_2 + mlp_branch.add)
- Current: scatter → layernorm → add = 3 launches
- Fused: 1 kernel
- Savings: **2 launches/MoE-layer × 30 = 60/token (≈ 3–6 ms)**

### T7: Fused SwitchGLU non-sort path (gather_qmm × 3 → fused gate/up/down on per-token routes)
- Current at decode (n=1, k=8): 3 gather_qmm calls
- Fused: 1 kernel does all three projections in one pass, walking the
  k=8 routes inline. Q4-aware. **Hardest fusion**; potential biggest win
  since it touches the dominant per-MoE-layer ops.
- Savings: **2 launches/MoE-layer × 30 = 60/token (≈ 3–6 ms)**
- Implementation: ~600 LOC Metal; needs Q4 unpacking + per-route matmul accumulation.

## Total potential

Summing T1–T6 launches saved per token: **~450–500 launches/token**
≈ **25–50 ms saved** on the 80 ms/token observed.

Realistic outcome with the easy 4 (T1, T2, T4, T5): **~270 launches
saved ≈ 15–30 ms/token = decode rate 12 → 16–22 tok/s** at long
context. ~30–80% improvement.

T3 and T7 each individually add a few more tok/s but require
Q4-aware kernels — much bigger work.

## Order of attack

1. **T2 (router fusion)** — first. Highest leverage per LOC. Pure
   routing math, no weight reads, no Q4 awareness needed, works for
   both bf16 and UD paths identically. ~200 LOC. **~1 day.**

2. **T4 (post_attn_ln + add + pre_ff_ln fusion)** — second. Pure
   elementwise, no GEMM, ~150 LOC. **~½ day.** Same pattern is reusable
   for the post-FF residual fusion.

3. **T5 (q_norm + RoPE)** — third. Extend `per_position_rope` to take
   an optional pre-norm. ~100 LOC. **~½ day.**

4. **T1 (RMSNorm + QKV fused)** — fourth. Requires Q4-aware kernel
   for UD path AND bf16 variant. The biggest single launch savings
   target but also the most complex. ~500 LOC. **~2 days.**

5. **T3 / T7 / T6** — only if T1–T5 don't close the gap to bandwidth.

## Validation method

Add a profiling env knob: `GEMMA4_PROFILE_DECODE=1` logs decode
ms/token at known context lengths. Bench before and after each
fusion lands. Target: decode at 12.7K hermes-helical context goes
from 12.3 tok/s → 18–22 tok/s.

Parity tests for each fused kernel: bit-exact (or within bf16/fp16
tolerance) against the existing op chain on small random inputs.

## Files to touch

- `mlx-rs-core/src/metal_kernels.rs` — new kernel modules per target
- `gemma4-mlx/src/model.rs` — opt-in dispatch (env-gated) for each fusion
- `gemma4-mlx/examples/` — bench harness updates if needed
- `docs/kernel-fusion-plan.md` — keep this doc updated with measured impact

## Caveats

- MLX may already do some of these fusions internally via lazy graph
  optimization. **Need to verify with Instruments before assuming
  the launch counts above.** The Metal frame debugger will show
  actual command-buffer dispatches; if MLX is already collapsing
  some ops, the savings shrink.
- Q4-aware kernels (T1, T3, T7) need to understand UD's per-tensor
  group_size/bits which we derive from shapes.
- Some of these fusions may not compose cleanly with the v6 bf16
  batched-MoE path — that's fine since UD-MLX-4bit is the
  recommended path and doesn't use v6.
