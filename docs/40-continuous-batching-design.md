# #40 — Continuous Batching — design doc for a dedicated session

**Status**: pending, scope captured 2026-05-19. Estimated 1–2 weeks of
focused work for a correctness-first v1; another week for production
polish (prefill-chunking interleaving, admission control). Largest
remaining open task on the perf list.

## Motivation

Single-stream decode numbers on hermes 5K (current best):

| Model | Engine | Decode tok/s (single stream) |
|---|---|---|
| Qwen3.6-35B-A3B-4bit (MoE) | bench_dflash AR | **60.3** |
| Qwen3.6-27B-MTPLX | bench_mtplx K=2 + fixes | 15.4 |
| Gemma4-26B-A4B-it (TurboQuant online v2) | mtplx AR | 4.5 |

Decode at any batch size on these models is **memory-bandwidth-bound**
(weights are large, activations are small). Read-once-per-layer weight
traffic is identical for batch=1 and batch=8 — only the activation
matmul widens. Empirically on Apple Silicon with 96 GB unified memory
and ~400 GB/s memory bandwidth, batch=8 decode runs at ~1.05–1.2× the
batch=1 wall time per step.

That means: serving 8 concurrent users at 60 tok/s each (480 tok/s
aggregate) is achievable *if* the scheduler keeps the GPU saturated.
Today's path serves 8 concurrent users at 60 tok/s **only for the
first one** — the other 7 queue and wait, because each request spawns
its own inference worker on a one-at-a-time GPU.

**Conservative target**: 3× aggregate throughput on Qwen3.6-35B-A3B
hermes-class loads (60 → 180 tok/s aggregate at concurrency=4).
**Stretch**: 5× (300 tok/s at concurrency=8) once attention is
varlen-packed.

This is the single highest-leverage remaining optimization for any
multi-tenant deployment. For single-user CLI workloads it returns
nothing — pick #35 (expert-major MoE) instead.

## Current path (static, batch=1, one worker per request)

```
OminiX-API/HTTP -> spawn Worker(req) -> Worker:
                                        load model (cached)
                                        prefill(prompt, cache)
                                        loop:
                                            decode(cache)
                                            yield token
                                        finish
```

Properties:
- `KVCache` per layer is a single `[1, T, n_kv_heads, head_dim]` tensor
  that grows monotonically. `KVCache::trim` releases tail entries on
  rollback (DFlash / MTP).
- SDPA expects uniform `kv_len` across the (size-1) batch dim.
- No request can join an in-flight forward pass; multi-tenant
  serialization happens at the OminiX-API layer (mutex on the worker).

## Proposed path (continuous batching)

```
OminiX-API/HTTP -> push(req) onto SchedQueue

InferenceLoop (single worker, owns the model + cache pool):
  state: running = [Seq]               -- in-flight sequences
         pending = SchedQueue          -- waiting prompts
         cache_pool: SlotAllocator     -- per-layer paged or slotted KV
  step():
    admit pending into running while token budget allows
    if any seq is in PrefillState:
        chunked-prefill subset, advance their state to Decoding
    decode_step over all Decoding seqs (varlen attention)
    for each seq:
        sample next token; check stop/EOS/max_tokens
        push token to seq's stream channel
        if finished: release cache slots, remove from running
  loop step() forever
```

## Component breakdown

### Phase 1 — Slot-indexed KV cache (~3 days)

The single hardest piece. Three viable layouts:

**Option A. Fixed-slot allocator.** Allocate
`[max_seqs=16, max_kv_len=8192, n_kv_heads, head_dim]` per layer up
front (= ~24 GB on Qwen3.6-35B-A3B). Each `Seq` owns a slot index;
writes go to `cache[slot, write_ptr]`. Simple, fast, no fragmentation
math — but wastes memory on short sequences and caps concurrency at
`max_seqs`.

**Option B. Paged attention (vLLM-style).** Allocate `block_size=16`
KV-token pages globally; each `Seq` holds a vec of page indices.
Attention reads pages via gather. Maximum memory utilization, supports
thousands of concurrent short sequences. Significantly more complex —
needs a custom Metal kernel for the gather path because MLX's SDPA
doesn't accept page tables.

**Option C. Packed varlen tensor.** All running sequences' K/V live
in a single `[total_tokens, n_kv_heads, head_dim]` tensor with
`cu_seqlens` cumulative-length offsets. Standard FlashAttention varlen
layout. Compact, but inserting a new sequence requires reshuffling
(or trailing-only append).

**Recommendation: Option A for v1.** Cap concurrency at 8 (Qwen3.6-35B-A3B
RSS is ~20 GB → 16 slots × 4K kv × 16 KV heads × 128 head_dim × 2
bytes ≈ 8.4 GB — fits in 96 GB). Migrate to Option B in v2 once the
scheduler is proven.

Files to touch:
- `mlx-rs-core/src/cache.rs` — new `SlottedKVCache` impl, parallel to
  `KVCache` and `TurboQuantKVCache`. Implements the existing
  `KeyValueCache` trait with per-slot `current_kv(slot)`,
  `update(slot, k, v)`, `trim(slot, n)`.
- `mlx-rs-core/src/cache.rs:KeyValueCache` trait — extend with
  `update_slot`, `current_kv_slot`, `release_slot`, `acquire_slot`.

### Phase 2 — Variable-length attention (~3 days)

Each step's input is `[total_active_tokens, n_heads, head_dim]` (Q
packed) and per-layer KV reads need per-slot `kv_len`. MLX's stock
SDPA broadcasts uniform kv_len, so we need a custom path:

**v1: block-diagonal mask.** Build a `[batch, q_len_padded, kv_len_padded]`
mask that's `-inf` off-diagonal, run stock SDPA with the mask. Wastes
attention compute proportional to (max_kv_len / mean_kv_len)² — on
hermes 5K with one new request and one nearly-done request,
≈ (5000 / 2500)² = 4× wasted. Acceptable for v1; bypasses Metal kernel
work.

**v2: varlen-packed SDPA kernel.** A new Metal kernel that takes
`cu_seqlens` and computes attention per-segment with no padding.
Similar to FlashAttention's varlen path. Significant work.

Files to touch:
- `gemma4-mlx/src/model.rs` and `qwen3.6-mlx/src/model.rs` —
  `Attention::forward` paths gain a "batched / per-seq kv_len" mode.
- `mlx-rs-core/src/metal_kernels.rs` — eventually a varlen SDPA kernel
  for v2 (out of scope for v1).

### Phase 3 — Scheduler loop (~2 days)

Single async worker, no lock contention with HTTP layer:

```rust
struct Seq {
    slot: SlotId,
    state: State,                     // Prefill { remaining_chunks } | Decoding
    tokens: Vec<i32>,
    write_ptr: usize,                 // position in slot
    stop_tokens: SmallVec<[i32; 4]>,
    max_tokens: usize,
    stream: mpsc::Sender<Token>,
    sampler: Sampler,
}

struct Scheduler {
    cache_pool: SlottedKVCache,
    running: Vec<Seq>,                // max 8 entries
    pending: VecDeque<PendingReq>,
    token_budget: usize,              // per-step soft cap
}

impl Scheduler {
    fn step(&mut self, model: &mut Model) -> Result<()> {
        self.admit_pending();
        if let Some(prefill_batch) = self.gather_prefill() {
            self.run_prefill(model, prefill_batch)?;
        }
        if let Some(decode_batch) = self.gather_decode() {
            self.run_decode(model, decode_batch)?;
        }
        self.harvest_finished();
        Ok(())
    }
}
```

**Admission policy** (v1): admit a pending request when
`running.len() < max_seqs` AND `running.iter().map(|s| s.write_ptr).sum::<usize>() + new_prompt_tokens < global_token_budget`.

**Prefill-decode interleaving** (v1): a step that has a prefill admit
chunks it to ≤ `prefill_chunk_size` tokens and shares the GPU pass
with decode. The chunk-size value already exists for Gemma4 (`GEMMA4_PREFILL_CHUNK=64`).

Files to touch:
- New crate: `omnix-scheduler` under workspace root.
- `gemma4-mlx`, `qwen3.6-mlx`: expose `prefill_chunk(tokens, cache, slot)`
  and `decode_step(tokens, cache, slots) -> [logits_per_slot]` entry
  points.

### Phase 4 — OminiX-API wiring (~3 days)

OminiX-API today: HTTP handler → spawn worker → block on completion.
After: HTTP handler → push to scheduler queue → forward stream channel
to the SSE response.

Files to touch:
- `OminiX-API` (separate repo at `/Users/kyle/repos/OminiX-API`) —
  `routes/chat_completions.rs` etc.: replace per-request worker spawn
  with `scheduler.submit(req).await -> stream`.
- Cancellation: when client drops the SSE connection, send a signal
  to drop the matching `Seq` and release its slot.
- Stop-sequence handling: each `Seq` carries its `stop_tokens` set;
  enforced in `harvest_finished`.

### Phase 5 — Bench + lock in (~2 days)

- Single-stream parity bench: ensure batch=1 continuous-batching
  matches today's single-stream tok/s within 5% (no regression).
- Multi-stream bench: 4 concurrent identical hermes 5K requests; expect
  aggregate ≥ 3× single-stream. 8-concurrent; expect aggregate ≥ 5×.
- Long-prompt-mixed-with-short bench: one in-flight 5K prefill +
  one short 200-tok prefill; verify the short one doesn't get starved
  > 250 ms.

## Risk register

1. **Memory pressure under concurrency.** 8 slots × 4K kv-tokens on
   Qwen3.6-35B = ~8 GB just for KV; plus model RSS 20 GB + activations
   leaves ~65 GB headroom on a 96 GB machine. Tight but workable.
   **Mitigation**: cap `max_kv_len` per slot at 4096 for v1; reject
   prompts that exceed it.

2. **Variable kv_len waste with block-diagonal mask.** Worst case
   ~4× attention compute. For Qwen3.6-35B-A3B where attention is
   ~10% of the layer time (MoE dominates), the wasted compute is
   acceptable. For dense Gemma4 it's worse. **Mitigation**: prioritize
   varlen kernel (Phase 2 v2) for Gemma4-class deployments.

3. **DFlash + continuous batching incompatibility.** DFlash today does
   per-sequence rollback via `KVCache::trim(n)`. Slotted cache must
   support per-slot trim. **Mitigation**: the slotted cache trait
   already includes `trim(slot, n)` in the design.

4. **MTPLX + continuous batching.** MTPLX's draft+verify cycle
   complicates batching: drafted tokens must speculatively extend the
   slot, then either commit or roll back. Per-slot speculative state
   adds complexity. **Mitigation**: v1 disables MTPLX in batched mode
   (fallback to AR per-slot). MTPLX-in-batched-mode is a follow-up.

5. **Prefill-decode contention.** A 5K-prompt prefill takes ~50 s on
   Gemma4 (8.6 tok/s prefill); during that time, in-flight decodes
   stall unless interleaved. **Mitigation**: prefill chunks of 64–128
   tokens, interleaved with decode steps; net prefill time goes up
   ~10–20% but no decode starvation.

6. **MoE expert routing under batching.** With batch=8 and top_k=4,
   each step routes 32 token-expert pairs through 64 experts → mean
   0.5 token/expert/step. That's WORSE than batch=1 for token-major
   MoE (more dispatch overhead, no tile reuse). **Mitigation**:
   pairs cleanly with #35 (expert-major) — both wins compose.

## Files to read first

- `mlx-rs-core/src/cache.rs` — current `KVCache` + `TurboQuantKVCache`,
  the `KeyValueCache` trait, `trim`, `compact_kv`. Slotted version
  parallels these.
- `gemma4-mlx/src/model.rs:Attention::forward` and `qwen3.6-mlx/src/
  model.rs:Attention::forward` — the SDPA call sites that need a
  per-seq kv_len mode.
- `mtplx-mlx/src/session.rs` — example of a long-lived inference
  loop (single stream); the scheduler is structurally similar but
  multi-seq.
- vLLM v0.1 codebase
  (https://github.com/vllm-project/vllm/tree/v0.1.0) — reference
  implementation of the scheduler + paged attention pattern.
  Note this is GPL/Apache mixed; read for design only, don't copy
  code.
- HuggingFace TGI's scheduler
  (https://github.com/huggingface/text-generation-inference/tree/main/router)
  — simpler reference for the admission/scheduling logic.

## Out-of-scope for this task

- **Paged attention (Option B).** v2 work after the slotted v1 ships.
- **Varlen SDPA Metal kernel.** v2 work; v1 uses block-diagonal mask.
- **Speculative decoding in batched mode.** v2 work; v1 falls back to
  AR per-slot.
- **Prefix caching across requests.** Already partially done via
  `cached chat-prefix prefill` (#31); cross-slot reuse is its own
  task.
- **Tensor parallelism / multi-GPU.** Apple Silicon is single-GPU per
  machine; not relevant for our target deployment.
- **Speculative *target* sharing across sequences.** Theoretical win
  if all in-flight sequences are decoding the same prefix; in practice
  rare for chat workloads.

## Strategic notes

- **Skip this if the goal is single-user CLI latency.** Continuous
  batching returns nothing for batch=1. Pick #35 (expert-major MoE)
  instead.
- **Pick this if the goal is multi-tenant serving throughput.** Single
  highest-leverage remaining task; 3-5× aggregate win on memory-bound
  decode.
- **Compose with #35 for MoE models.** Expert-major MLP wins more
  with batched decode than with single-stream (more token-expert
  pairs per layer); the two are orthogonal and additive.
- **Order of work**: do #35 first if MoE is the target deployment
  (Qwen3.6-35B-A3B); do #40 first if dense models (Gemma4, Qwen3.6-27B
  dense) are the target. The MoE case benefits from #35's prefill win
  to amortize concurrent prefill cost; the dense case has no #35
  analog so #40 is the only lever.
