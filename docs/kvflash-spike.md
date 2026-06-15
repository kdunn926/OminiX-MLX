# KVFlash spike — bounded-residency KV for MLX

**Branch:** `spike/kvflash` · **Status:** spike (decode-bounding core only)

Port of the decode-time bounded-attention core of FlashMemory-style KV paging
([lucebox-hub#373](https://github.com/Luce-Org/lucebox-hub/pull/373)) to
Rust/MLX on Apple Silicon.

## Idea

Instead of letting the full-attention KV cache grow with context, keep a fixed
**resident pool** of at most `pool` tokens — the first `sink` attention-sink
tokens (StreamingLLM) plus the most-recent `pool - sink` — and evict the oldest
non-sink 64-token chunk when it overflows. Decode then attends a `≤ pool`
working set, so **decode throughput stays flat as context grows** instead of
degrading with KV size.

### Why it's exact over the kept set

The model bakes RoPE into K at write time using the token's true position;
`offset()` returns the logical count so positions keep advancing. RoPE is
*relative*, so a query at position `N` attending any kept key at position `p`
sees the correct offset `N − p` regardless of which middle chunks were dropped.
Attention over the resident set is therefore **bit-identical to a full-cache
run restricted to those keys** — the only loss is the dropped chunks'
contribution (the LRU/StreamingLLM quality trade-off).

## What's implemented

- `mlx-rs-core/src/kvflash.rs` — `KvFlashCache` (impls `KeyValueCache`):
  step-buffered K/V, chunk-granular LRU eviction on decode, returns the bounded
  resident set from `update_and_fetch`.
- `qwen3.6-mlx` — `KVCacheMode::KvFlash` / `HybridCache::KvFlash`,
  `Generate::new_kvflash`, env `DFLASH_KVFLASH=<pool>` /
  `DFLASH_KVFLASH_SINK` / `KVFLASH=1`, generate-example selection.

Plugs into the existing SDPA path: the bounded resident set is just what
`update_and_fetch` returns, so no attention-kernel changes.

```bash
DFLASH_KVFLASH=2048 cargo run --release -p qwen3-6-mlx --example generate -- \
    models/Qwen3.6-27B-4bit "<long prompt>" 64 0
```

## Spike scope / differences from the source PR

- **Bounded prefill (opt-in, `DFLASH_KVFLASH_PREFILL=1`).** With it on, eviction
  also runs on the multi-token chunked-prefill appends, so each prefill chunk
  attends a `≤ pool` resident set instead of the growing prefix — bounding
  prefill **memory** to `O(pool)` and prefill attention to `O(seq·pool)`
  instead of `O(seq²)`. Works with **zero mask changes**: qwen3.6 chunked
  prefill passes a hardware causal mask (offset `L_k − L_q`), and the new chunk
  is always the resident suffix, so the pool prefix is fully visible and the
  chunk is causal among itself. Requires `chunk ≤ pool − sink` (default
  64 ≤ 2044). Off by default = decode-only bounding (prefill accumulates full
  KV, TTFT unchanged).
- **No host paging.** Apple Silicon is unified memory — there's no discrete-GPU
  VRAM to page to/from, so evicted chunks are simply dropped. The PR's ~99%
  VRAM reduction doesn't map; the *decode throughput* win (bounded KV read)
  does.
- **LRU policy only.** The drafter/scored residency that preserves long-range
  recall is out of scope; this is sink + recent (pure recency = StreamingLLM),
  so mid-context facts are lost once evicted (see needle result below).
- **No spec-decode rollback** on the lossy pool.

## Results (Qwen3.6-27B-4bit, M-series, greedy)

Validated: unit test (resident bounded ≤ pool, sink + recent kept, logical
offset advances); short context (< pool) **byte-identical** to standard fp16.

Long-context decode (32-token greedy, needle = a code planted at position ~0):

| context | mode | prefill (TTFT) | decode tok/s | needle recall |
|---|---|---|---|---|
| 7.6K  | standard fp16     | 75.7s  | 18.1 | ✓ |
| 7.6K  | kvflash pool=2048 | 75.7s  | 18.6 | ✗ |
| 21.6K | standard fp16          | 220.7s | **15.9** | ✓ |
| 21.6K | kvflash pool=2048      | 220.1s | **18.9** | ✗ |
| 21.6K | kvflash pool=2048 +prefill-bound | 210.9s | 18.7 | ✗ |

Bounded prefill (`DFLASH_KVFLASH_PREFILL=1`) keeps **decode flat** (18.7) and
shaves TTFT only ~4% (220→211s) — on this hybrid model the sequential DeltaNet
recurrence (48/64 layers, which kvflash doesn't touch) dominates prefill, so
bounding the 16 attention layers moves TTFT little. Its real benefit is the
**prefill memory bound**: resident KV stays `≤ pool` (2048 tokens) through
prefill instead of holding the full 21.6K — the property that lets 64K–256K
contexts fit at all. (Output stays coherent; with the needle evicted the model
correctly reports the code isn't present rather than hallucinating.)

**The core property holds.** Standard decode **degrades with context**
(18.1 → 15.9 tok/s as the KV read grows 7.6K → 21.6K), while kvflash stays
**flat** (~18.6–18.9) because it reads a bounded 2048-token pool regardless of
logical length — a **1.19× decode speedup at 21.6K** that widens with context
(the PR's full-cache curve keeps falling to 13.1 tok/s at 256K; kvflash would
stay ~flat). Prefill (TTFT) is identical in both modes — kvflash doesn't bound
prefill in this spike.

**The recall trade-off is real.** With pure LRU (sink=4 + recent), the planted
code sits past the sink and is evicted once context exceeds the pool, so
kvflash answers wrong while the full cache recalls it.

## Scored residency (H2O-style) — and why it doesn't recover recall *here*

`DFLASH_KVFLASH_POLICY=scored` adds query-aware eviction: a
`KeyValueCache::observe_query` hook feeds the model's post-RoPE queries to the
cache, which accumulates a cheap mean-head `softmax(q·kᵀ)` per resident
position and evicts the **lowest-attention** middle chunk (keep sink + recency
window + heavy-hitters) instead of the oldest. `DFLASH_KVFLASH_DECAY<1`
recency-weights so the trailing query dominates. A unit test confirms the
**mechanism**: a position that keeps getting attention survives eviction while
old unattended positions are dropped.

But on a **mid-context needle** (a code at ~50% depth in repetitive filler;
Qwen3.6-27B, 8.3K tokens, pool=2048), no in-model scoring variant recovers it:

| policy | needle recall |
|---|---|
| standard fp16 (full cache) | ✓ |
| kvflash LRU | ✗ |
| kvflash scored (plain) | ✗ |
| kvflash scored decay=0.90 | ✗ |
| kvflash scored decay=0.97 | ✗ |

**Why — a structural limit of the no-paging spike.** The needle's importance
only manifests when the model *generates the answer* (decode), but the first
eviction (8.3K→2048) happens at the start of decode, *before* that retrieval
attention accumulates — and with no host paging, an evicted chunk is gone for
good. During prefill the needle is attended only weakly (the question encodes
causally but doesn't "look up" yet) and is swamped by the local attention every
filler token gets, so it never ranks as a heavy hitter; recency-weighting only
extends the recent window, which can't reach 4K tokens back.

This is exactly why the source PR pairs two things this spike doesn't have:
(1) a **drafter** that scores chunks by *query relevance* proactively (not the
model's own incidental attention), and (2) **host paging** to *recall* an
evicted chunk once it becomes relevant. Cheap in-model attention scoring +
drop-on-evict is enough to keep *naturally heavy-hitter* tokens (the unit-test
case, and real documents where early facts are referenced throughout) but not
an adversarial needle whose relevance is deferred to answer time.

### Gemma4 (the better showcase)

Wired into gemma4 too (`init_kvflash_cache`: global/full-attention slots get
`KvFlashCache`, sliding slots keep their already-window-bounded
`SlidingKVCache`, shared-KV store slots stay unbounded). Gemma4-26B-A4B is an
all-attention MoE — no DeltaNet — so prefill is ~10× faster than qwen3.6 and
the global layers are the clear decode bottleneck at long context.

| gemma4-26B-A4B @ 15.9K | prefill | decode tok/s |
|---|---|---|
| flat (all unbounded)              | 33.2s | 26.9 |
| layered (sliding-trim, default)   | 26.0s | 26.9 |
| **kvflash global pool=1024**      | 26.4s | **32.1** |
| **kvflash pool=1024 +prefill**    | **23.9s** | 31.9 |

**1.19× decode** vs both the default layered cache and flat. Note layered and
flat tie on decode (26.9): layered bounds the *sliding* layers but leaves the
*global* layers unbounded, and at 16K the global layers are the decode
bottleneck — so only kvflash (which bounds them) speeds decode up. Bounded
prefill is also the **fastest prefill** (23.9s, beating even layered) because
gemma4's global layers are real attention whose `O(seq²)` prefill cost the pool
bounds — the qwen3.6 case below doesn't show this because DeltaNet dominates
its prefill.

### Hardware caveat

On this **hybrid** arch (48 GatedDeltaNet + 16 full-attention layers) the
DeltaNet recurrence dominates **prefill** and is sequential — ~76 s to prefill
7.6K tokens on M-series. So (a) prefill isn't the part kvflash speeds up here
(that's bounded *chunked* prefill, out of spike scope), and (b) the decode KV
read only becomes a large fraction of the step around ~30K+ tokens, where
prefilling repeatedly for A/B is expensive. The documented 2.9× decode speedup
is at 64K–256K on a discrete GPU; the same bandwidth dynamic holds on MLX but
the absolute context where it pays off is large and slow to set up on this
hybrid model.

## Next steps if pursued

1. ~~Bounded chunked prefill~~ — **done** (`DFLASH_KVFLASH_PREFILL=1`; see the
   bounded-prefill row above). On this hybrid model the DeltaNet recurrence
   (48/64 layers, unaffected by KV bounding) dominates prefill, so the TTFT win
   is bounded by the attention share; the **prefill memory bound** (`O(pool)`)
   is the load-bearing benefit for fitting 64K–256K contexts.
2. ~~Scored residency~~ — H2O-style attention scoring is **done**
   (`DFLASH_KVFLASH_POLICY=scored`), but recovering an *adversarial* mid-context
   needle needs the source PR's drafter (proactive query-relevance) + **host
   paging** (recall evicted chunks); see the scored-residency section above.
3. Wire into gemma4 (full-attention layers only; SWA layers already ring-buffer).
