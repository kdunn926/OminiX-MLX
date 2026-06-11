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

### DeltaNet substep audit — completed

Used the existing `target_layer0_debug.rs` + a Python mirror
(`scripts/python_layer0_debug.py`) to diff every layer-0 DeltaNet
intermediate.

Findings, in order along the forward pipeline:

| Op | Status | Action |
|---|---|---|
| `embeddings`, `input_layernorm` | bit-exact | — |
| `in_proj_qkv` / `z` / `a` / `b` | bit-exact (post mlx 0.31.2 bump) | runtime alignment |
| `conv1d` (`qkv_after_conv`) | 0.6% drift | switched manual kernel-tap loop to fused `mlx_rs::ops::conv1d` — bit-exact |
| q/k normalization | k_norm 3.9% drift | switched `l2_normalize + scale` to Python's `rms_norm(., None, 1e-6) + scale` — k_norm 0.2% |
| recurrent scan (`output`) | 0.2% — within BF16 noise | — |
| `out_proj` (quantized 4096→2048) | 0.37% | not actionable (kernel noise) |

### MoE audit — completed

Layer-1 MoE dump on bit-exact synthetic input (`layer0_input_norm`,
which is bit-identical Rust vs Python) using
`qwen3.6-mlx/examples/target_moe_dump.rs` ↔ `scripts/python_moe_dump.py`:

- `gates_raw`, `gates` (post-softmax): bit-exact
- top-k expert SETS per token: SET-MATCH on all 15 tokens
- `moe_out`: 1.0% relmax — entirely from the expert matmuls
  (`gate_proj` / `up_proj` / `down_proj` in `SwitchGLU` and the shared
  expert), not from routing

Aligned Rust's argpartition pattern with Python's literally
(`argpartition(gates, kth=N-k)[..., N-k:]` instead of
`argpartition(-gates, k-1)[..., :k]`) to eliminate a "different code,
same output" footgun. Numerically neutral on this example.

### Synthesis — per-op parity floor reached

Every individual op we've probed is either bit-exact or has been
*made* bit-exact by a targeted fix. The remaining drift is uniformly
small (~0.4–1% per quantized matmul) and is BF16 kernel noise — not
something OminiX can fix without changing precision. With ~25 DeltaNet
layers × `out_proj` + 40 layers × MoE expert matmuls, this compounds
to the ~3% `prefill_logits` relmax we measure end-to-end.

**Despite all per-op fixes, DFlash acceptance has not budged from
~0.12.** AR throughput improved 30 → 67 tok/s (2.2×) as a side-effect
of the runtime bump + workaround removal — a substantial standalone
perf win, but unrelated to the acceptance gap.

### New leading hypothesis — DFlash session-level logic

Per-op parity is solid. The acceptance gap (Rust ~0.12 vs Python ~0.5)
must therefore live in DFlash's session/protocol layer, not in the
target or draft forward computations. Specifically:

1. **Staged-token / noise embedding alignment** — claimed resolved by
   the original wip doc, never validated against acceptance.
2. **Projected-context cache / target_hidden accumulation across
   cycles** — Rust uses Strategy B full-recompute every cycle, Python
   uses incremental `ContextOnlyDraftKVCache`. We confirmed earlier
   the two are mathematically equivalent at infinite precision, but
   the in-practice BF16 differences across many cycles have not been
   measured.
3. **Acceptance comparison logic** — how Rust decides which drafted
   tokens to accept vs reject against the target's verification logits.
4. **Rollback bookkeeping** — DeltaNet recurrent state + KV cache
   snapshot/restore on rejection.

### Resolution — adaptive block sizing was silently neutered

Comparing Python `CycleCompleteEvent` stream against Rust's
`trace_dflash_verify` per-cycle output revealed the smoking gun on
cycle 5:

- Python's `block_len` dropped from 16 → 4 starting cycle 5 (the
  adaptive policy kicked in after the 4-cycle window saw poor
  acceptance).
- Rust's `block_len` stayed at 16 for the entire run.

Root cause: `dflash-mlx/examples/{bench_dflash, trace_dflash_verify}.rs`
were constructing `SpeculativeCycleConfig` with
`min_block_tokens: block_size` (= 16, same as `block_len`). With both
bounds set to 16, `AdaptiveBlockPolicy::current_block_len()` returns
`block_len.max(min_block_tokens)` = 16 even in reduced mode — the
adaptive reduction code path was dead.

The fix is a one-line removal — let `..Default::default()` provide the
correct `min_block_tokens=4`. Effects on Qwen3.6-35B-A3B-4bit on the
benchmark prompt:

| metric                | before | after |
|-----------------------|-------:|------:|
| acceptance (a/d)      |   0.12 |  0.43 |
| avg_block_len         |   14.6 |  3.12 |
| DFlash decode tok/s   |   25.0 |  38.0 |
| verify tokens per gen |   5.05 |  ~1.4 |
| accepted / generated  |  0.625 | 0.55  |

Rust's per-cycle trajectory now mirrors Python's: one or two full
(block_len=16) cycles at the start, then steady-state block_len=4 for
the rest of generation, ending with a short tail.

### Lesson learned

The entire prior parity investigation was looking in the wrong place.
Numerical drift across layers, MoE routing, q/k normalization, conv1d
kernel choice — every per-op fix was a real correctness improvement
but **none of them moved acceptance** because acceptance was being
crushed by a config typo that disabled the adaptive policy. A
multi-cycle behavior comparison (rather than per-op numerical diffs)
would have found this on day one.

For future similar investigations: **diff the cycle-level behavior
first**, before chasing per-op BF16 floor.

### Open work post-fix

1. **AR is still faster than DFlash on this specific config**
   (Qwen3.6-35B-A3B-4bit + matched mlx 0.31.2 runtime — AR is 70 tok/s,
   DFlash is 38 tok/s). On this MoE-sparse model with already-fast AR
   decode, DFlash may not be a net throughput win at all. Worth
   measuring on (a) longer prompts where target prefill amortizes
   better, (b) dense Qwen3.6-27B where AR is slower.
2. **OminiX-API integration** of DFlash (Phase 3 from the original
   plan) — now that the path actually works as designed, it can be
   wired in as a real backend.
3. **Acceptance-ratio metric reporting** is currently `accepted /
   drafted` in Rust vs `accepted / generated` in Python. Worth
   standardizing in the Rust metrics so cross-language comparisons
   aren't misleading.

---

## Update 2026-06-11 — full-stack review: why DFlash loses to AR

A correctness/perf review of the whole DFlash path (spec_epoch, adapters,
acceptance, draft model, rollback, bench) found the protocol sound — greedy
posterior/correction row indexing, staged-token alignment, GDN tape rollback
(incl. conv-window reconstruction), and the incremental
`ProjectedContextCache` are all correct. The "no speedup" decomposes into
avoidable overhead and MoE physics:

### New measurements

`bench_dflash`, standing prompt "The theory of general relativity",
200 tokens, Qwen3.6-35B-A3B-4bit:

| temp | AR tok/s | DFlash tok/s | acceptance (a/d) | speedup |
|---|---|---|---|---|
| 0.0 | 73.0 | 47.1 | 0.375 | 0.65× |
| 0.7 | 73.0 | 37.2 | 0.241 | 0.51× |

The historical 0.43 was measured at the bench's default temp 0.7 with
stochastic acceptance, which understates drafter quality vs the Python
greedy ~0.5 reference; greedy gives 0.375 — same ballpark, small residual
gap (BF16 kernel-noise floor documented above).

`trace_dflash_verify` on dense **Qwen3.6-27B-4bit** (temp 0, 64 tok):
**acceptance 0.396** (38/96), 2.67 tokens committed/cycle, context
bookkeeping exact on every cycle. So the 27B's 7.27-vs-17.58 tok/s deficit
is NOT an acceptance bug — it is pure implementation overhead.

### Cost accounting (per steady-state Reduced cycle, block 4)

- Draft forward: ~0.95 GB (35B) / **3.46 GB (27B)** bf16 weight reads —
  the draft is unquantized.
- Draft lm_head: matmul against a **dequantized bf16 [vocab, H] weight**
  (`get_lm_head_weight`) — 1.02 GB (35B) / **2.54 GB (27B)** read per
  cycle, plus a permanently-resident dequant buffer.
- 3 GPU sync points per cycle (one redundant — `drafted_tokens` is pulled
  to host twice: spec_epoch.rs:342 and :356); no `async_eval` overlap,
  unlike the AR baseline's pipelined loop.
- temp>0 acceptance copies the full `[dc+1, vocab]` fp32 softmax to host
  (~4 MB/cycle at block 4).
- `verify_qmm` is dead in measured configs: shape gate accepts m==16/m≤4
  but real verify rows are 5 or 17.

### Verdict

- **35B MoE: fundamentally capped at ≈ break-even.** A verify of L tokens
  reads ~`256·(1−e^(−8L/256))/8`× one token's routed-expert bytes (4.6× at
  L=5), so at the block-4 steady state both Rust and Python collapse to,
  even a perfect implementation lands ~55-70 tok/s vs 73 AR. Wrong tool
  for this target.
- **Dense 27B: the winnable case.** Verify ≈ free (bandwidth-bound), so
  with a 4-bit draft (+~2.6 GB/cycle saved) and a quantized lm_head
  (+~1.9 GB/cycle saved) the cost model gives **~1.5-2× over AR** at the
  measured 0.40 acceptance. This is where optimization effort should go.

### Implemented (same day): quantized lm_head + 4-bit draft + sync dedup

- `DraftLmHead::Quantized` — draft logits via `quantized_matmul` against
  the target's packed lm_head/tied-embed arrays
  (`Model::get_lm_head_quantized`), replacing the dequantized-BF16 matmul.
- `DFLASH_QUANT_DRAFT=4|8` — on-load quantization of all draft linears
  (q/k/v/o, gate/up/down, fc) to 4- or 8-bit group-64 via
  `MaybeQuantized<nn::Linear>`. Costs ~0.06 acceptance on the 27B trace
  prompt (0.396 → 0.337) but nets positive throughput.
- `build_verify_inputs` now returns the drafted host vec, deleting the
  duplicate eval + device→host copy per cycle.

Results, dense 27B, standing prompt, temp 0, 200 tokens (short-prompt
in-process comparison; the historical 0.41× was on the 5K hermes fixture):

| config | DFlash tok/s | AR tok/s | speedup |
|---|---|---|---|
| before (bf16 draft + dequant lm_head, 5K prompt) | 7.27 | 17.58 | 0.41× |
| quantized lm_head only | 16.18 | 19.12 | 0.85× |
| + 4-bit draft | 17.08 | 19.06 | 0.90× |
| + sync dedup | **17.49** | 19.01 | **0.92×** |

Remaining gap is latency, not bytes: ~195 ms/cycle vs ~63 ms of predicted
weight traffic for ~3.3 committed tokens. The serial structure dominates —
2 remaining hard syncs/cycle with no `async_eval` overlap (the AR baseline
pipelines), the 48-GDN-layer tape-capture verify path, and per-rejection
tape replay (48 dispatches). Closing it means overlapping the draft graph
build with verify via `async_eval` and trimming the GDN tape path — a
structural change, not a knob.

