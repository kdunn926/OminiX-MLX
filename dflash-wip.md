# DFlash WIP Summary

## Goal

Port the Python `dflash-mlx` speculative decoding path to Rust in `OminiX-MLX`, then drive the Rust implementation toward functional parity with the Python reference on Qwen3.6-35B-A3B.

The guiding rule for the current phase is:

1. make **target parity** the first hard gate,
2. especially around **short-sequence quantized projection parity** on real verify windows,
3. only then revisit the **projected-context/cache contract** for cycle 1+.

## High-level status

- The Rust DFlash path is implemented enough to run real speculative decode.
- Best standing benchmark result on the current prompt:
  - prompt: `"The theory of general relativity"`
  - `acceptance_ratio=0.103`
  - `avg_block_len≈13.31`
- Python reference is still far higher, around `~0.5` acceptance.
- The dominant unresolved problem is still **target-side mismatch before the draft model**.

## What has been tried so far

### 1. Initial DFlash port and integration work

Implemented and/or wired:

- `dflash-mlx` crate
- target/draft adapter traits
- DFlash session loop
- Qwen3.6 target adapter
- Rust DFlash draft model loading and forward path
- benchmark example
- recurrent rollback/cache infrastructure
- staged-token alignment logic

Separate review/spike work also concluded:

- **Do not switch** OminiX-MLX to `MisterEkole/mlx-rs`
- wrapper-swapping alone does not improve MLX performance
- ANE work is severable but not a realistic path for full Qwen3.6-35B offload

### 2. Draft-side debugging that has mostly been resolved

The following were investigated and are no longer the leading suspects:

- staged token / noise embedding alignment
- draft projection (`fc -> hidden_norm`)
- hidden-capture ordering
- YaRN RoPE implementation
- `mx.fast.rope` behavior on matched inputs

Established results:

- staged token and `noise_emb` match Python
- draft projection matches Python when given the same raw captured hidden
- hidden-capture ordering already matches Python
- Rust draft RoPE now matches Python nearly exactly on matched bf16 inputs

Conclusion: the remaining large mismatch is **not primarily draft-side**.

### 3. Target-side parity narrowing

Target-only diagnostics showed:

- target drift starts at **layer 0**
- embeddings match Python exactly
- layer-0 input RMSNorm output matches Python exactly
- target drift compounds through the stack after layer 0

The first visible internal mismatch is in the raw layer-0 quantized projections:

- `qkv`
- `z`
- `a`
- `b`

### 4. Things tried and either kept or rejected

#### Kept

1. **DeltaNet out_proj dtype fix**
   - Cast final `out_proj` result back to input dtype before residual add.
   - Acceptance improved from **0.083 -> 0.092**.

2. **Short-sequence exact-small-proj thresholds**
   - Originally broadened to `qkv + z`, later narrowed to measured ranges:
     - `qkv`: pad only for lengths `< 6`
     - `z`: pad only for lengths `< 10`
     - `a/b`: keep `< 16`
   - Best benchmark remains **0.103**.

#### Rejected / not explanatory

1. Python-style grouped-query SDPA reshaping
   - Did not materially move parity or acceptance.

2. Manual RoPE replacement
   - Made acceptance worse; reverted.

3. Forcing DeltaNet prefill to bypass fused recurrence kernel
   - Did not materially change parity or acceptance.

4. More Python-like RMSNorm-style Q/K normalization experiment
   - Improved some early diffs but worsened acceptance back to `0.083`; reverted.

## Diagnostics and harnesses added

### DFlash-side

- `dflash-mlx/examples/cycle0_parity.rs`
  - Main cycle-0 parity harness
  - Dumps staged token, raw target captures, projected context, noise embedding, draft tensors, logits, and draft tokens

- `dflash-mlx/examples/target_prefill_parity.rs`
  - Dumps target prefill logits and all captured hidden layers
  - Also dumps embeddings and layer-0 input norm

- `dflash-mlx/examples/trace_dflash_verify.rs`
  - Logs real verify-window lengths
  - Now also traces per-cycle draft context length and acceptance progression

### Target-model-side

- `qwen3.6-mlx/examples/deltanet_consistency.rs`
  - Confirms layer-0 DeltaNet `forward_prefill` vs repeated `forward_step` are already very close

- `qwen3.6-mlx/examples/target_layer0_debug.rs`
  - Dumps detailed layer-0 internals

- `qwen3.6-mlx/examples/target_layer0_proj_sweep.rs`
  - Sweeps prefix lengths and compares direct/contiguous/padded projection paths

- `qwen3.6-mlx/examples/target_layer0_token_dump.rs`
  - Dumps arbitrary token windows for direct Python-vs-Rust projection comparison

- `qwen3.6-mlx/examples/target_qkv_qmm_repro.rs`
  - Minimal reproducer for raw `qkv` quantized matmul behavior

## Key findings so far

### Acceptance progression

- near-zero -> `~0.083` after earlier alignment fixes
- `0.083 -> 0.092` from DeltaNet `out_proj` dtype fix
- `0.092 -> 0.103` from short-sequence `qkv + z` exact-small-proj behavior

### Real runtime verify windows

On the standing benchmark prompt, real verify lengths are:

- `[16 x13, 9, 7, 5, 1]`

This was critical because it showed the acceptance improvement is not coming from the first cycle, but from **later short verify windows**.

### Short-sequence shape sensitivity

On the earlier sweep:

- Rust `qkv` matches Python for lengths `1..5`, then diverges at `6..15`
- Rust `z` matches Python for lengths `1..9`, then diverges at `10..15`

### Exact runtime-window behavior

Using the actual traced `9/7/5/1` verify windows:

- `len 9/7`
  - `qkv_direct == qkv_exact`
  - `z_direct != z_exact`

- `len 5/1`
  - both `qkv_direct != qkv_exact`
  - and `z_direct != z_exact`

- `direct == contiguous` on all four runtime short windows

Conclusion: the live effect is coming from the **exact-small-proj padding threshold**, not ordinary contiguity.

### Cross-language compare on exact runtime windows

Using the same token windows in Python and Rust:

- `layer0_input_norm` matches exactly on all traced runtime windows

#### `qkv`, len `9/7`

- Python direct differs from Rust direct
- Rust direct already equals the **dequantized reference**
- Python direct also differs from that dequantized reference

#### `z`, len `9/7`

- Python direct == Rust direct
- both differ from the dequantized reference
- Rust exact-small-proj moves `z` to the dequantized reference

#### `qkv` and `z`, len `5/1`

- Python direct == Rust direct
- both differ from the dequantized reference
- Rust exact-small-proj moves them to the dequantized reference

### Interpretation of the current workaround

This is the most important current conclusion:

The current `exact_small_proj()` threshold workaround is **not a Python-parity fix**.

Instead, on the real runtime windows it mostly pushes Rust toward the **dequantized reference path**, and in some cases away from Python’s live quantized path, even though acceptance improves.

So the `0.103` gain is best understood as a **behavior-shaping workaround**, not as evidence that target parity is solved.

## Minimal `qkv` quantized-matmul reproducer result

The next highest-value investigation was a minimal reproducer around the `qkv len=9/7` path.

That reproducer established:

- In **Rust**, all of the following are the same on `len=9/7`:
  - `QuantizedLinear::forward`
  - raw `ops::quantized_matmul`
  - contiguous `3D` input
  - flattened `2D` input

- In **Python**, all of the following are also the same on `len=9/7`:
  - `QuantizedLinear.__call__`
  - contiguous `3D` input
  - flattened `2D` `mx.quantized_matmul`

This rules out:

- 3D vs 2D lowering
- non-contiguous input handling
- the Rust `QuantizedLinear` wrapper above `quantized_matmul`

## Current best interpretation

The strongest new meta-suspect is now **runtime provenance / MLX implementation skew**.

Current comparison is between:

- **Python**
  - `mlx 0.31.2`
  - `mlx-lm 0.31.3`
  - `dflash-mlx 0.1.6`

- **Rust**
  - vendored `mlx-c` via `mlx-sys 0.2.0`
  - local vendored commit `169ade3...`

That means the parity investigation is currently comparing two different MLX runtime lineages. This is now a plausible explanation for why the same packed quantized tensors produce different `qkv` outputs at `len=9/7`.

## Projected-context/cache-contract status

This remains important, but it has been deliberately deprioritized behind target parity.

### Python contract

Python:

- projects target captures at feature-store ingress
- stores projected committed deltas in `TargetFeatureStore`
- relies on draft-side projected-context cache semantics (`ContextOnlyDraftKVCache`)

### Rust contract

Rust currently:

- stores raw captured hidden in `Qwen36TargetAdapter`
- projects inside `DFlashDraftModel::forward`
- recomputes full effective projected context each cycle
- does **not** currently mirror Python’s projected-context cache contract

### Important nuance

Rust’s current Strategy-B bookkeeping is internally coherent:

- `draft_context_len == prompt_len + committed_tokens_so_far`
- `next_context_len == current_context_len + (acceptance_len + 1)` on every traced cycle

So the projected-context problem is currently about **equivalence to Python’s semantics**, not an obvious off-by-one or accumulation bug in Rust’s own loop.

## Current state of the worktree

Notable current local changes / added tools include:

- `qwen3.6-mlx/src/deltanet.rs`
- `dflash-mlx/examples/trace_dflash_verify.rs`
- `qwen3.6-mlx/examples/target_layer0_debug.rs`
- `qwen3.6-mlx/examples/target_layer0_proj_sweep.rs`
- `qwen3.6-mlx/examples/target_layer0_token_dump.rs`
- `qwen3.6-mlx/examples/target_qkv_qmm_repro.rs`
- `qwen3.6-mlx/examples/deltanet_consistency.rs`
- `dflash-mlx/examples/cycle0_parity.rs`
- `dflash-mlx/examples/target_prefill_parity.rs`

The current best-known benchmark configuration is still the targeted `qkv/z` threshold path with acceptance `0.103`.

## Recommended future investigation plan

### Immediate next step

1. **Align MLX runtime provenance**
   - Establish an apples-to-apples MLX comparison if possible.
   - Determine whether Rust vendored `mlx-c 0.2.0` and Python `mlx 0.31.2` are executing materially different quantized-matmul implementations.
   - Until that is resolved, be careful about interpreting the `qkv len=9/7` split as an OminiX-only bug.

### After runtime provenance is clarified

2. **Continue target parity as the first hard gate**
   - Keep using:
     - `cycle0_parity.rs`
     - `target_prefill_parity.rs`
     - `target_layer0_debug.rs`
     - `target_layer0_token_dump.rs`
     - `target_qkv_qmm_repro.rs`
   - Do not treat acceptance movement alone as proof of correctness.

3. **Resolve the remaining target-side quantized projection story**
   - Explain:
     - why `qkv len=9/7` already splits cross-language without exact-small-proj
     - why `z len=9/7` does not
     - why `qkv + z len=5/1` match cross-language but both differ from dequantized reference

4. **Only after target parity is credible, revisit projected-context semantics**
   - Choose between:
     - **Strategy A:** mirror Python projected-context cache semantics
     - **Strategy B:** keep Rust full-context recompute and prove equivalent behavior end-to-end

5. **Only after that, reevaluate the remaining acceptance gap**
   - If cycle 0 parity is still wrong, later cache/rollback changes will not recover Python-level acceptance.

## Current outstanding investigation priorities

1. Align MLX runtime provenance / implementation
2. Continue target-side quantized parity work
3. Revisit projected-context/cache equivalence only after target hard gate is understood

---

## Update — post-runtime-bump investigation

### What was resolved

1. **Runtime provenance aligned.** Vendored `mlx-c` upgraded from `0.4.1` to upstream main (post-`v0.6.0`, pins `mlx v0.31.2` — same as Python). Wrapper fixes: `mlx_quantize`/`mlx_dequantize` `global_scale`, FFT `mlx_fft_norm`, `mlx_metal_device_info` → key-value `mlx_device_info_*` API.
2. **Layer-0 quantized projections now bit-exact to Python on GPU.** Verified via `scripts/python_layer0_token_dump.py` ↔ `qwen3.6-mlx/examples/target_layer0_token_dump.rs` at len=9: `embeddings`, `layer0_input_norm`, `qkv_direct`, `z_direct`, `a_direct`, `b_direct` all `max_abs=0`.
3. **`exact_small_proj` workaround removed.** It was a workaround for the mlx 0.30.1 Metal qmm kernel bug at `(M=small, N=8192)`. With mlx 0.31.2 it is redundant and actively harmful on `z` (padded path drifts by 0.125 max_abs vs direct). Side-benefit: AR decode 30 → 67 tok/s on Qwen3.6-35B-A3B-4bit.

### What was disconfirmed

1. **MLX-runtime-skew hypothesis as primary acceptance cause.** Bumping Rust to mlx 0.31.1 (one patch off Python) left acceptance unchanged at 0.128. Bumping to mlx 0.31.2 (= Python) left it at 0.120. Runtime alignment fixes the layer-0 numerics but not the acceptance ratio.
2. **Projected-context cache contract as a parity issue.** Walked Python `ContextOnlyDraftKVCache` end-to-end. Every op in the projected-context pipeline (`fc`, `hidden_norm`, `k_proj`/`v_proj`, `k_norm`, RoPE) is per-token, so caching post-projection k/v vs recomputing is bit-exact equivalent. This is a perf optimization, not a parity fix.
3. **Fused DeltaNet Metal kernel as the drift source.** Forcing the standard-MLX-ops fallback (gating `mlx_rs_core::deltanet_recurrence` off) produced **bit-identical** layer-0 output to the fused path on layers 0/9/18/27. Acceptance moved 0.120 → 0.134 — within noise.

### New leading hypothesis — drift is inside DeltaNet, downstream of in_proj_*, not in the recurrent scan

Cycle-0 end-to-end diff (`scripts/python_cycle0_parity.py` ↔ `dflash-mlx/examples/cycle0_parity.rs`):

| Tensor | rel max | Notes |
|---|---|---|
| `prompt_ids`, `noise_emb`, `staged_token` | 0 | bit-exact |
| `layer0_input_norm`, `qkv_direct`, `z_direct`, `a_direct`, `b_direct` | 0 | bit-exact (separate len=9 dump) |
| `raw_target_hidden`[layer 0 output] | **3.8%** | linear_attention (DeltaNet) |
| `raw_target_hidden`[layers 9/18/27/36] | ~0.3% each | |
| `prefill_logits` | 2.8% | propagated drift |
| `draft_hidden` | 5.6% | |
| `draft_logits` | 13% | lm_head amplification |
| `draft_tokens` | 13/15 agree | |

Remaining suspects inside layer-0 DeltaNet, between `in_proj_*` and layer output:

- `conv1d` over qkv
- L2 normalization of q / k
- `decay` computation (`softplus(a + dt_bias) → exp`)
- `RMSNormGated` (output gate by `z`)
- **`out_proj`** — quantized matmul, shape `4096 → 2048`. Strongest hypothesis because the previous mlx-0.30.1 qmm bug was also shape-specific, and we have not yet verified `out_proj` at this shape between Rust and Python.

### Next-step plan

Add a `debug_first_layer` analog to `LinearAttention` in `qwen3.6-mlx/src/deltanet.rs` that returns per-step intermediates (post-conv, post-l2-norm, decay, beta, post-recurrence, post-rmsnorm-gated, post-out_proj). Mirror in Python; diff on the same prefill state to find which sub-step diverges first.

