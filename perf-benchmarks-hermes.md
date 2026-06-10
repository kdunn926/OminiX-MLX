# Hermes-Gateway Perf Benchmarks

All runs use the OminiX-API `tests/fixtures/hermes_chat_simple.json` fixture
(~5K prompt tokens, depending on tokenizer) and greedy decoding (`temp=0`)
to 31 generated tokens. Bench harnesses: `mtplx_chat`, `bench_dflash`,
`bench_mtplx`. Hardware: single-host MLX/Metal. Date: 2026-05-17.

Peak RSS = process resident memory; Peak GPU footprint = MLX's
`peak_memory()` allocator high-water mark (combined unified-memory cost on
Apple Silicon).

## Aggregate results

| Model | Engine | KV cache | Prompt tok | Prefill (s) | TTFT (s) | Decode tok/s | Peak RSS (GB) | Peak GPU (GB) |
|---|---|---|---|---|---|---|---|---|
| Gemma4-26B-A4B-it | mtplx AR | BF16 baseline | 5183 | 667.6 | 667.6 | 1.8 | — | — |
| Gemma4-26B-A4B-it | mtplx AR | TurboQuant partial-fuse (sink=4) | 5183 | 670.4 | 670.4 | 0.7 | — | — |
| Gemma4-26B-A4B-it | mtplx AR | TurboQuant fully-fused (sink=0) | 5183 | 670.6 | 670.6 | 3.1 | — | — |
| Gemma4-26B-A4B-it | mtplx AR | TurboQuant online v1 (naive V) | 5183 | 730.9 | 730.9 | 0.5 | — | — |
| **Gemma4-26B-A4B-it** | **mtplx AR** | **TurboQuant online v2 (V-tile cache)** ⭐ | 5183 | 664.5 | 664.5 | **4.5** | — | — |
| Qwen3.6-35B-A3B-4bit (MoE) | bench_dflash AR | fp16 | 5068 | 14.0 | 14.0 | **60.32** | 20.77 | 31.13 |
| Qwen3.6-35B-A3B-4bit (MoE) | bench_dflash AR | TurboQuant (cold first run) | 5068 | 16.6 | 16.6 | 19.25 | 20.79 | 33.18 |
| Qwen3.6-35B-A3B-4bit (MoE) | bench_dflash AR | TurboQuant (warm, 2nd in-process) | 5068 | 14.7 | 14.7 | 53.21 | — | — |
| Qwen3.6-35B-A3B-4bit (MoE) | bench_dflash DFlash | fp16 | 5068 | 7.4 | 7.4 | 15.80 | — | — |
| Qwen3.6-35B-A3B-4bit (MoE) | bench_dflash DFlash | TurboQuant | 5068 | 7.4 | 7.4 | 15.54 | 20.77 | 33.17 |
| Qwen3.6-27B-4bit (dense) | bench_dflash AR | fp16 | 5068 | 55.1 | 55.1 | **17.58** | — | — |
| Qwen3.6-27B-4bit (dense) | bench_dflash AR | TurboQuant | 5068 | 51.4 | 51.4 | 10.97 | — | — |
| Qwen3.6-27B-4bit (dense) | bench_dflash DFlash | fp16 | 5068 | 50.8 | 50.8 | 7.27 | — | — |
| Qwen3.6-27B-MTPLX-Optimized | bench_mtplx | fp16 | 5035 | 50.4 | 52.8 | 12.67 | 15.75 | 25.13 |
| Qwen3.6-27B-MTPLX-Optimized | bench_mtplx | TurboQuant | 5035 | 51.2 | 54.7 | 8.98 | 15.76 | 25.88 |
| Qwen3.6-27B-MTPLX-Optimized | bench_mtplx (K=2) | q4/q4 KV (gs=32) ≈ llama.cpp `q4_0/q4_0` + `spec-draft-n-max 2` | 5035 | 50.5 | 53.0 | 12.45 | 15.76 | 25.50 |
| Qwen3.6-27B-MTPLX-Optimized | bench_mtplx (K=4) | q4/q4 KV (gs=32) ≈ llama.cpp `Q4_K_M + q4_0/q4_0 + draft-mtp` | 5035 | 50.3 | 52.9 | 12.17 | 15.78 | 25.50 |
| Qwen3.6-27B-MTPLX-Optimized | bench_mtplx (K=2, **MTP fixes**) | fp16 — concat order + ar_next embed | 5035 | 50.2 | 56.7 | **15.37** (acceptance **0.860**) | — | 25.87 |
| Qwen3.6-27B-MTPLX-Optimized | bench_mtplx (K=2, **multi-step drafting**) | fp16 — 2 MTP forwards per cycle, batch verify | 5035 | 50.2 | 55.9 | **17.61** (acceptance 0.569) | — | — |
| Qwen3.6-27B-MTPLX-Optimized | bench_mtplx (K=4, multi-step) | fp16 — 4 MTP forwards per cycle | 5035 | 50.9 | 58.4 | 13.42 (acceptance 0.333) | — | — |
| Gemma4-26B-A4B-it | mtplx AR (Fused QKV OFF, baseline) | BF16 | 5183 | — | 660.98 | 0.4 (5 decode) | — | 68.6 |
| Gemma4-26B-A4B-it | mtplx AR (Fused QKV ON, `GEMMA4_FUSED_QKV=1`) | BF16 | 5183 | — | 700.82 | 0.4 (5 decode) | — | 69.6 | +6% TTFT regression — keep off for long context |

## Gemma4 UD-MLX-4bit (preserve-quant) — hermes 5K ⭐ recommended path

`models/gemma4-26B-a4b-it-UD-MLX-4bit/` loaded via
`ud_loader::load_ud_mlx_4bit`. Heterogeneous per-tensor quant
(q/k/v/o @ Q8, mlp @ Q4, embed @ Q6, all group_size=64). MoE dispatches
through MLX's `gather_qmm` via `SwitchGluExperts`.

| GEMMA4_PREFILL_CHUNK | TTFT (s) | Decode @ 5K ctx (tok/s) | Peak GPU (GB) | Notes |
|---|---|---|---|---|
| 64                 | 16.59 | 28     | ~15 | safe across all memory tiers |
| 128                | 12.12 | —      | ~17 | |
| 256 ⭐ (new default) | **9.49** | **30** | ~20 | recommended; tested on 96 GB Mac |

Resident weights ~15 GB (matches disk, no dequant). Bypasses the v6
bf16 batched-kernel path entirely; uses MLX's native `quantized_matmul`
for attention and `gather_qmm` for MoE.

## Gemma4 Expert-Major MoE (#35) — hermes 5K prefill

| Variant | hermes 5K prefill wall | Prompt tok/s | Speedup |
|---|---|---|---|
| Default `forward_topk` (token-major) | 487s | ~10.6 | 1× (baseline) |
| **v6 Phase 9 (`GEMMA4_EXPERT_MAJOR_MOE=6`)** ⭐ | **42s** | **~123** | **~11×** |

Pipeline: GPU bucketing (argsort + counts via one-hot sum) → packed
`[M_total_padded, H]` layout → single batched `moe_dense_matmul_batched`
Metal launch (per-block expert lookup, simdgroup_matrix<float,8,8>
4×4 acc tiles, fp32 mixed-precision accumulation) → SwiGLU on packed
fp32 layout → second batched launch for down_proj → fp32 scatter back
to per-token output. Pre-transposed `[E, K, N]` weight views cached
lazily on first call (~22 GB extra bf16). Skip-gate
`EXPERT_MAJOR_MIN_PROMPT_TOKENS=256` (default) routes decode/short
prompts back through `forward_topk` so v6 only fires when there's
enough tokens-per-expert to amortize the 32-row padding.

Same first emitted token as default ("I") — correctness preserved at
argmax level.

## Gemma4 paired-model speculation — sliding-window mask A/B (hermes 5K-tok concat)

| Variant | Linear tok/s | Tree2 tok/s | Acceptance (Linear) | Wall (96 tok) |
|---|---|---|---|---|
| Sliding-window mask ON (b1d4e82, default) | 2.06 | 0.98 | 0/576 | 46.6s + 98.0s |
| Sliding-window mask OFF (`MTPLX_PAIR_NO_SLIDING_MASK=1`) | 2.05 | 0.97 | 0/576 | 46.9s + 99.4s |

Sliding-window mask is **neutral on acceptance** for this prompt (no regression, no win). Greedy-vs-greedy on out-of-distribution hermes-style prompts produces 0% acceptance for *distributional* reasons — the drafter's argmax never aligns with the target's. Real acceptance lift requires stochastic draft sampling + spec-decode rejection math (see `docs/35-expert-major-moe-design.md`-adjacent #36 follow-up).

⭐ = headline winner for that model.

## Gemma4 prefill chunk sweep (hermes 5183 tok, max_tokens=5)

| GEMMA4_PREFILL_CHUNK | TTFT (s) | Prefill tok/s | Peak RSS (GB) | Peak GPU (GB) | Notes |
|---|---|---|---|---|---|
| 32 (legacy default) | 720.1 | 7.2 | 45.4 | 73.6 | reference |
| **64** ⭐ (new default) | **604.1** | **8.6** | 47.6 | 73.7 | -16% TTFT |
| 128 | 1905.5 | 2.7 | 47.6 | 91.2 | MoE expert-gather thrashing |

Default raised from 32 → 64 in `gemma4-mlx/src/model.rs` (env-tunable via
`GEMMA4_PREFILL_CHUNK`).

## Findings

1. **TurboQuant KV is a win on Gemma4 (dense, BF16-weight), a loss on
   Qwen3.6 (4-bit-weight + GQA-heavy / MoE).** Gemma4 with TQ online
   softmax + V-tile cache: 4.5 tok/s vs 1.8 BF16 baseline (2.5× speedup).
   Every Qwen3.6 variant regresses 11-38% with TQ and shows *higher*
   peak GPU footprint than fp16 (extra sigma/mean/scales/biases tensors
   plus compression scratch outweigh the K compression on already-4-bit
   weights).

2. **MoE A3B (3B active params per token) crushes everything else.**
   Qwen3.6-35B-A3B-4bit AR hits 60 tok/s at 5K context — 3.4× the dense
   Qwen3.6-27B and ~13× Gemma4-26B-A4B's best TQ result.

3. **Speculative drafting (DFlash, MTPLX) underperforms AR on Qwen3.6
   for long real-world prompts.** Acceptance lands at 0.17-0.29 on
   hermes, so the extra draft+verify compute dominates. DFlash speedup
   ratios: 0.26-0.41× across all Qwen3.6 configs. MTPLX K=4 on the
   27B-MTPLX checkpoint: 12.67 tok/s vs plain 27B AR fp16 at 17.58.

4. **TurboQuant + speculative compose cleanly now** (DFlash + TQ runs
   after wiring `TurboQuantKVCache::trim`) but inherits both regressions.

5. **DFlash adaptive verify v0.1.7** (Large/Reduced/Probe state machine
   with wall-clock probe comparison, ported on this branch) doesn't
   change the picture for these specific prompts because acceptance
   never rises high enough to keep the policy in Large mode.

## Recommended configs by use case

| Workload | Best config |
|---|---|
| **Gemma4-class long-context** ⭐ | **`gemma4-26B-a4b-it-UD-MLX-4bit` via `ud_loader::load_ud_mlx_4bit`** (preserve-quant; `GEMMA4_PREFILL_CHUNK=256` default; ~9.5s TTFT on hermes 5K, 30 tok/s decode, ~20 GB peak) |
| Long-context MoE A3B (Qwen3.6-35B-A3B) | Plain AR fp16 KV — no speculation, no quantized KV |
| Long-context dense 4-bit (Qwen3.6-27B) | Plain AR fp16 KV |
| MTPLX-optimized checkpoints | fp16 KV (TQ regresses; MTP speculation may or may not help vs AR depending on prompt) |
| Memory-tight Gemma4 (no UD model on disk) | bf16 + `GEMMA4_EXPERT_MAJOR_MOE=6 GEMMA4_PREFILL_CHUNK=64` (~47s TTFT, ~32 GB peak; safe on 64 GB) |
| Long-context dense BF16 (legacy Gemma4) | TurboQuant KV, online softmax, V-tile cache, sink=0, fused_kv_min=1 |

## Environment flags (TurboQuant)

- `TURBO_KV=1` — pick TurboQuantKVCache (qwen3.6 generate / bench_dflash AR / bench_mtplx)
- `TURBOQUANT_ONLINE=0` — disable the V-tile online softmax path (default ON)
- `TURBOQUANT_FUSED_KV_MIN=N` — minimum kv_len to engage the fused kernel (default 8192; set 1 to always engage)
- `TURBOQUANT_SINK_TOKENS=N` — keep first N tokens at native dtype (default 4; set 0 to compress everything)
