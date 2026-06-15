# Model Leaderboard — best decode configs

Per-model decode throughput (tok/s) and the **best config** to run each model,
split by context regime. Apple Silicon (M-series), greedy `temp=0`. Two regimes
matter because the winning config *flips* with context length:

- **Short context** (≤ a few K tokens): the KV read is cheap, so the win comes
  from **speculative decoding** (EAGLE-3 / MTP-pair / DFlash) — more tokens per
  target forward.
- **Long context** (≳8K, growing): the **global-attention KV read** dominates
  each decode step, so the win comes from **kvflash bounded residency** — and
  `host-paging + reranker drafter` adds mid-context **recall** at the same
  bounded read. See [`kvflash-spike.md`](./kvflash-spike.md).

> ⚠️ The short-context **spec ratios** below are inherited from the 2026-06-12
> same-session bench and are **cold-biased** (AR measured first, cold MoE-gather
> JIT; spec measured warm) — a warm 35B re-bench flipped DFlash from 1.40× to
> 0.95×. Trust the **AR decode** columns; treat spec ratios as *needs warm
> re-bench* unless marked otherwise. The **long-context kvflash** numbers below
> were each measured warm, in-process, this session.

## Long-context decode — kvflash is the best config (NEW)

Decode tok/s, default cache (baseline) → `kvflash host-paging + drafter`. All
configs recall an 8.3K/18K mid-context needle; the drafter (Qwen3-Reranker-0.6B)
ranks the needle chunk #1 and pins it resident. `pool=2048`.

| model | ctx | baseline | **paging+drafter** | ratio | notes |
|---|---|---|---|---|---|
| **gemma4-26B-A4B** (UD) | 8.3K | 28.4 | **31.4** | 1.11× | peak (short) 33.5 |
| | 18K | 26.3 | **30.2** | **1.15×** | gap widens with ctx |
| | 15.9K | 26.9 | 32.1¹ | 1.19× | ¹kvflash pool=1024, no drafter |
| **gemma4-12B** (UD) | 8.3K | 19.8 | **21.2** | 1.07× | peak 22.6; 8 global layers |
| | 18K | 19.5 | **21.2** | 1.09× | drafter dead-flat 21.2→21.2 |
| **qwen3.6-27B** (dense) | 7.6K | 18.1 | 18.6 | — | standard decode sags with ctx |
| | 21.6K | 15.9 | **18.9** | **1.19×** | kvflash holds flat |

**Reading it:** paging+drafter tracks the model's *peak* decode rate (90–94% held
at 18K) because the global-KV read is pool-capped, while the baseline keeps
paying for the growing cache (78–86% of peak). The gap widens monotonically with
context. One-time cost: a reranker pass before decode (~15s for 281 chunks @18K,
batched), amortized over the generation. Prefill is unchanged (paging bounds
decode, not prefill).

**Use it when:** long context (≳8K) **and** you need either steady long-context
throughput or mid-context recall under a bounded KV. For short prompts kvflash is
byte-identical to the default cache (nothing is evicted), so it's safe as a
default for long-context workloads.

## Short-context decode — AR baseline & best spec config

Decode tok/s at ~6–18 tok prompt (2026-06-12, same-session). `AR` = plain
autoregressive; `best spec` = fastest speculative config found.

| model | AR | best spec (config) | ratio† |
|---|---|---|---|
| gemma4-e4b 4bit (non-QAT) | 60.8 | — | — |
| Qwen3.6-35B-A3B-4bit (MoE) | 69.6‡ | DFlash 66.3 | 0.95× ‡warm: spec LOSES |
| gemma4-26B-A4B UD-4bit (MoE) | 32.2 | EAGLE-3 40.0 | 1.24× (acc 0.48) |
| gemma4-12B QAT-4bit (dense) | 22.4 | MTP-pair 31.0 | 1.38× (acc 0.52) |
| Qwen3.6-27B-4bit (dense 64L) | 17.4 | DFlash 13.2 | 0.76× (LOSES) |
| Gemma4-27B-MTPLX 4bit (60L dense) | 8.08 | MTP-pair 16.48 | 2.04× (acc 0.55) |

† cold-biased — see warning above; re-bench warm before trusting.
‡ warm re-bench (the one config verified warm): spec does **not** help the 35B.

## Recommendations

- **Fast daily driver, short prompts:** gemma4-e4b (60.8) or the 35B-A3B MoE
  (~70 warm AR — run it plain AR, spec doesn't help warm).
- **Long-document / RAG / large context:** add `DFLASH_KVFLASH=<pool>` (and
  `DFLASH_KVFLASH_PAGING=1 DFLASH_KVFLASH_DRAFTER=1` if you need mid-context
  recall) — best decode throughput past ~8K and the only config that recalls an
  evicted mid-context fact at a bounded KV read.
- **Biggest spec win (short ctx):** Gemma4-27B-MTPLX pair (2.04×) — but the
  spec ratios need a warm re-bench to confirm.

Cross-refs: [`kvflash-spike.md`](./kvflash-spike.md),
[`performance-comparison.md`](./performance-comparison.md) (Rust-vs-Python parity).
