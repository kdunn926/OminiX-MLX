# MTP decoding for `gemma-4-12B-it` — implementation plan

**Status:** design / exploration. No code committed yet.

**Reference:** `sgl-project/sglang` `python/sglang/srt/models/gemma4_mtp.py`
(merged with PR #27167), specifically `Gemma4AssistantForCausalLM` and the
`bind_frozen_kv_context` / centroid-masking machinery.

**Reference checkpoint:** `mlx-community/gemma-4-12B-it-assistant-4bit`
(`model_type = gemma4_unified_assistant`).

---

## 1. Architecture summary

The Gemma 4 unified 12B is served with a "**Frozen-KV Multi-Token Prediction**"
speculative drafter. The drafter is a tiny (4-layer, hidden=1024) Gemma 4
transformer that lives next to the target (48-layer, hidden=3840) and predicts
the next token. Speedup comes from amortising K drafted tokens across one
target verify forward.

### 1.1 Distinguishing features vs. existing infrastructure

| Property | Qwen3.6 in-model MTP (`qwen3.6-mlx/src/mtp.rs`) | Gemma4 unified Frozen-KV MTP (this plan) |
|---|---|---|
| Where the head lives | Inside the target model checkpoint (`mtp.safetensors`) | Separate model repo (`-assistant-4bit`) |
| Number of layers | 1 (single hidden→logits head) | **4** (full transformer blocks) |
| KV cache | Shares the target's main cache (it's the same model) | **Frozen-KV bridge** — assistant layers read target K/V, never write |
| Input | last hidden + last token embedding | `pre_projection(cat([token_embed, prev_hidden]))` (2× backbone fan-in) |
| Output projection | none (head directly produces logits) | `post_projection(hidden) → backbone_hidden_size` then `lm_head` |
| LM head | Standard Linear / tied embeddings | **Centroid-masked head** (256 centroids, top-K=32, vocab permuted) |
| K (draft length) | 1 | configurable (default 1; unrolls for K>1) |

The bolded items are all new infrastructure relative to what gemma4-mlx ships.

### 1.2 Forward pass (per draft step, K=1)

Inputs:
- `prev_hidden ∈ ℝ^[B, S, 3840]` — target's last hidden state from the previous step
- `input_ids ∈ ℤ^[B, S]` — token IDs for the assistant's position (one per draft slot)
- Target's KV pool (shared, read-only)

Steps:
```
token_embed = embed_tokens(input_ids) * sqrt(backbone_hidden_size)    # (B,S,3840), Gemma-style scale
z = pre_projection(concat(token_embed, prev_hidden, axis=-1))         # (B,S, 1024) — pre_projection: 7680 → 1024
hidden = transformer(z, kv_from_target_pool)                          # (B,S, 1024) — 4 Gemma blocks
projected = post_projection(hidden)                                   # (B,S, 3840)
logits = centroid_masked_head(projected, hidden)                      # (B,S, 262144) — see §1.3
return logits, projected   # `projected` becomes prev_hidden for the next step
```

### 1.3 Centroid-masked LM head

The assistant config declares `use_ordered_embeddings: true` and ships:
- `centroids.weight ∈ ℝ^[256, 1024]` — centroid classifier
- `lm_head.weight ∈ ℝ^[262144, 1024]` — tied with `embed_tokens` (vocab × hidden)
- `token_ordering ∈ ℤ^[262144]` — buffer mapping centroid-grouped index → real vocab index, materialized at load via `_reorder_embedding_to_centroid_order`

Forward (per token in batch):
```
scores       = centroids(flat_hidden)                       # (N, 256)
top_k_idx    = argtopk(scores, k=32)                        # (N, 32) — centroid indices

# Reshape lm_head as (centroids × vocab_per_centroid × hidden) = (256 × 1024 × 1024)
# Gather the 32 selected centroid groups: shape (N, 32, 1024, 1024)
# Materialised contiguously as (N, 32*1024, 1024).
selected_E   = lm_head.weight.view(256, 1024, 1024)[top_k_idx]  # (N, 32, 1024, 1024)
selected_E   = selected_E.reshape(N, 32*1024, 1024)

selected_l   = (flat_hidden.unsqueeze(1) @ selected_E.transpose(-2, -1)).squeeze(1)  # (N, 32*1024)

# Scatter selected logits to real vocab via token_ordering reorder
real_idx     = token_ordering.view(256, 1024)[top_k_idx].reshape(N, 32*1024)        # (N, 32*1024)
out          = full((N, 262144), -inf)
out.scatter_(-1, real_idx, selected_l)
return out
```

This drops the per-token LM-head FLOPs by **256/32 = 8×** vs. a dense head while
keeping correctness in expectation (the masked-out positions are tokens the
centroid classifier didn't think were plausible).

`vocab_size_per_centroid = 262144 / 256 = 1024 = hidden_size` — convenient
numeric coincidence that simplifies the reshape.

### 1.4 Frozen-KV bridge

The assistant doesn't own a KV cache. Each of the 4 assistant layers is bound
to a specific target physical layer at load time
(`bind_frozen_kv_context(ctx)`):

```
for assistant_logical, layer in enumerate(assistant.layers):
    target_phys = ctx.get_physical_layer_id(assistant_logical)
    layer.self_attn.is_kv_shared_layer = True
    layer.self_attn.kv_shared_layer_index = target_phys
```

`ctx.get_physical_layer_id` collapses HF's two-hop indirection (typed-layer
mapping → KV-shared-layer redirect) into a direct pointer.

During the assistant's attention forward, K/V are loaded from the target's
KV pool at the bound layer; the assistant's K/V projections are computed and
discarded (suppressed write). The assistant's Q is used as-is.

This requires:
- A cross-pool reference in our MLX KV cache types — currently
  `gemma4_mlx::KVCache` / `MixedKvCache` are owned per-model.
- A mode flag on `gemma4_mlx::Attention::forward` that skips KV updates
  and loads K/V from a borrowed source pool.

### 1.5 Speculative cycle (K = 1, greedy at T = 0)

```
target.prefill(prompt_ids) → (hidden_L, logits_L)
prev_hidden = hidden_L
token = argmax(logits_L)
emit(token)

while not eos and len(emitted) < max_tokens:
    # 1. Assistant drafts the next token from (token, prev_hidden)
    draft_logits, draft_projected = assistant.forward([token], prev_hidden)
    draft_token = argmax(draft_logits)

    # 2. Target verifies by computing the real next step
    target_hidden, target_logits = target.decode(token)
    target_token = argmax(target_logits)

    # 3. Compare: greedy accept = draft == target
    if draft_token == target_token:
        emit(target_token)
        token = target_token
        prev_hidden = draft_projected   # use assistant's projected hidden as the recurrence input
    else:
        emit(target_token)
        token = target_token
        prev_hidden = target_hidden     # fall back to target's real hidden

stats: accept_rate = accepted_drafts / total_verifies   # ~50–80% on typical English
```

For K > 1 the assistant drafts K tokens before a single target verify forward
that processes positions [L+1..L+K] in one chunk. Probabilistic acceptance
(Leviathan-Chen) replaces the greedy `==` check.

Speedup on a single sequence at K=1 is bounded by
`E[accept] / (1 + assistant_cost / target_cost)`. With a 12B target and a
4-layer/1024-hidden assistant the cost ratio is ~3% (4×1024² vs 48×3840²);
expected speedup with 60% acceptance is ~1.55×. K=2-4 doubles the
opportunity at the cost of acceptance dropping geometrically.

---

## 2. Implementation roadmap

Order matters: each step builds on the previous. Suggested execution as
separate commits / PRs so we can pause and benchmark between them.

### Phase 1 — `gemma4_unified_assistant` loader

Files: `gemma4-mlx/src/unified_assistant.rs` (new),
`gemma4-mlx/src/lib.rs` (re-export).

Components:
- `Gemma4UnifiedAssistantConfig` deserializer (parses the assistant repo's
  `config.json` — top-level keys `backbone_hidden_size`, `num_centroids`,
  `centroid_intermediate_top_k`, `use_ordered_embeddings`, plus nested
  `text_config` of 4 standard Gemma 4 layers).
- `Gemma4UnifiedAssistant` struct:
  - `pre_projection: nn::Linear` (7680 → 1024, no bias, **plain not quantized** per sglang's ReplicatedLinear with `quant_config=None`)
  - `text: gemma4_mlx::Model` reused (4 layers, hidden 1024, sliding/full
    pattern per `text_config.layer_types`)
  - `post_projection: nn::Linear` (1024 → 3840, no bias, plain)
  - `lm_head: MaybeQuantized<nn::Linear>` tied to `text.embed_tokens` when
    `tie_word_embeddings`
  - `centroids: nn::Linear` (1024 → 256, no bias)
  - `token_ordering: Array` (i32 vocab permutation)
- `load_unified_assistant(dir)` — dequant-to-BF16 on the small projections,
  preserve packed-4-bit on the 4 transformer layers (use the existing
  `make_mq_linear` from `gemma4-mlx/src/model.rs`).

Smoke: load the 4-bit assistant, assert each module's shapes match the config.

### Phase 2 — Target hidden-state hook

Files: `gemma4-mlx/src/model.rs` (modify).

The target's `Model::forward_from_embeds` returns logits. We need to also
surface the pre-`lm_head` hidden state for the recurrence. Add:

```rust
impl Model {
    /// Like `forward_from_embeds`, but returns the per-position hidden state
    /// in addition to the lm-head logits. The hidden state is what the
    /// Frozen-KV MTP assistant consumes as `prev_hidden`.
    pub fn forward_with_hidden<C>(...) -> Result<(Array /*hidden*/, Array /*logits*/), Exception>
    where C: KeyValueCache + Default;
}
```

Implementation: branch out from `forward_from_embeds` at the point where it
takes the final-position last hidden state. Return both.

For `prefill_multimodal` and `prefill_text` (and the unified VL equivalents),
add `with_hidden=true` variants that return `(hidden_last, logits)`.

### Phase 3 — Frozen-KV bridge on `gemma4_mlx::Attention`

Files: `gemma4-mlx/src/model.rs` (attention forward).

Modify `Attention::forward` to accept an optional "borrow source" for K/V.
When set, the layer:
- Computes Q from `z`
- **Reads K/V from the borrowed cache** at the bound target_physical_layer_id
- **Skips writing the local cache**
- Runs SDPA with (Q_local, K_borrowed, V_borrowed)

Add a `FrozenKvBinding` struct that maps assistant_logical → (target_pool_ref,
target_physical_layer_id). Stored on `Gemma4UnifiedAssistant` per layer.

Concretely:
```rust
pub struct AttentionInput<'a, C: KeyValueCache + Default> {
    pub inputs: &'a Array,
    pub mask: Option<&'a Array>,
    pub cache: &'a mut C,
    /// When `Some`, this layer reads K/V from the borrowed cache at the
    /// given physical index and suppresses writes to `self.cache`.
    pub frozen_kv: Option<(&'a C, usize)>,   // NEW
}
```

Most call sites pass `frozen_kv: None`; the assistant's forward populates
this from its `FrozenKvBinding` map.

### Phase 4 — Assistant forward pass

Files: `gemma4-mlx/src/unified_assistant.rs`.

Implements:
```rust
impl Gemma4UnifiedAssistant {
    pub fn forward<C>(
        &mut self,
        input_ids: &[i32],
        prev_hidden: &Array,            // (B, S, 3840), from target
        target_cache: &C,               // borrowed read-only target KV
        layer_id_map: &[usize],         // assistant_logical → target_phys
        positions: &Array,
    ) -> Result<(Array /*logits*/, Array /*projected_hidden*/), Exception>
    where C: KeyValueCache + Default;
}
```

Key concern: the target's `prefill_multimodal` runs through 48 layers with a
PLE precompute. The assistant's 4 layers need the SAME PLE (since they
share the target's residual stream conceptually). Look at the sglang
`_get_text_config` clone — it sets `num_kv_shared_layers = 0` on the
assistant config copy. We need to wire PLE accordingly.

### Phase 5 — Centroid LM head

Files: `gemma4-mlx/src/unified_assistant.rs`.

```rust
fn centroid_masked_logits(&self, hidden: &Array) -> Result<Array, Exception> {
    let scores = self.centroids.forward(hidden)?;          // (N, 256)
    let top_k = argtopk(scores, self.top_k)?;              // (N, 32)
    let lm_view = self.lm_head.weight                       // (262144, 1024)
        .reshape(&[self.num_centroids, self.vpc, self.hidden])?;
    let selected_e = take_axis(&lm_view, &top_k, 0)?;       // (N, 32, vpc, 1024)
    let selected_e = selected_e.reshape(&[N, top_k * vpc, hidden])?;
    let selected_logits = mlx_rs::matmul(
        &hidden.expand_dims(1)?,                            // (N, 1, hidden)
        &selected_e.transpose(&[0, 2, 1])?,                 // (N, hidden, top_k*vpc)
    )?.squeeze(1)?;
    let real_idx = take_axis(&self.token_ordering_view, &top_k, 0)?
        .reshape(&[N, top_k * vpc])?;
    let mut out = full((N, vocab), neg_inf);
    out = scatter(&out, &real_idx, &selected_logits, axis=-1)?;
    Ok(out)
}
```

Risks here:
- MLX `argtopk` exists but check whether it's stable / GPU-resident
- `scatter` semantics: MLX provides `put_along_axis` — confirm semantics match
- The reshape `lm_head.weight: (262144, 1024) → (256, 1024, 1024)` only works
  if `lm_head` has already been reordered via `token_ordering` at load time
  (see §1.3 footnote and sglang's `_reorder_embedding_to_centroid_order`).

### Phase 6 — Speculative session

Files: `mtplx-mlx/src/unified_session.rs` (new — analogous to existing
`MtplxSession` for Qwen3.6 but with an external assistant).

```rust
pub struct Gemma4UnifiedMtpSession {
    target: gemma4_mlx::Gemma4UnifiedVlModel,
    assistant: gemma4_mlx::Gemma4UnifiedAssistant,
    layer_id_map: Vec<usize>,
    cfg: SpeculativeConfig,        // reuse: block_len, max_tokens, temp, acceptance
}

impl Gemma4UnifiedMtpSession {
    pub fn generate<C>(
        &mut self,
        prompt_ids: &[i32],
        eos: &HashSet<u32>,
    ) -> Result<(Vec<i32>, SessionMetrics), MtpError>;
}
```

Implements the K=1 greedy cycle from §1.5. Reuses
`mtplx_mlx::acceptance::accept_greedy` and `accept_speculative`. K>1 is a
follow-up.

### Phase 7 — Loader detection + API backend variant

Files: `gemma4-mlx/src/ud_loader.rs` (extend), `OminiX-API/src/engines/llm.rs`.

Auto-detection: when a target model dir contains a sidecar
`assistant/config.json` with `model_type = "gemma4_unified_assistant"`, load
the pair into a new backend variant:

```rust
ModelBackend::Gemma4UnifiedMtp {
    session: mtplx_mlx::Gemma4UnifiedMtpSession,
    template: gemma4_mlx::Gemma4ChatTemplate,
    eos_tokens: Vec<u32>,
}
```

Wire all 13 match sites (mirroring the unified VL wire-up from
`cfca98f`). Streaming / paged-KV / Responses API come for free if the
session API matches.

### Phase 8 — Benchmark

Files: `gemma4-mlx/examples/unified_mtp_bench.rs` (new),
`results/gemma4_unified_mtp.csv` (new).

Measure:
- prefill_s
- decode_tps **without** MTP (vanilla 12B)
- decode_tps **with** MTP at K=1 (greedy, T=0)
- acceptance_rate
- speedup ratio

Repeat for short and long prompts. Append to the existing `bench_sweep.csv`
schema via a new harness analogous to `bench_dflash`.

---

## 3. Effort estimate (rough)

| Phase | LOC | Wall-clock |
|---|---|---|
| 1. Loader | 200 | 0.5 day |
| 2. Hidden-state hook | 80 | 0.5 day |
| 3. Frozen-KV bridge | 200 | 1 day (touches all attention paths) |
| 4. Assistant forward | 150 | 0.5 day |
| 5. Centroid LM head | 200 | **1–2 days** (MLX op verification, perf) |
| 6. Speculative session | 250 | 1 day |
| 7. API backend variant | 200 | 0.5 day |
| 8. Benchmark | 100 | 0.5 day |
| **Total** | **~1380** | **~6 days focused work** |

Bottleneck risks (would expand the estimate):
- **Phase 3** if MLX KV cache types resist generalisation to cross-pool reads
  (may need a new cache trait + every cache impl updated).
- **Phase 5** if MLX's `scatter` / `topk` aren't a clean fit (may need to
  implement the centroid head as a custom Metal kernel for performance).
- **Phase 6** if PLE precompute interacts badly with the assistant's
  zero-KV-shared-layer override.

---

## 4. Open questions

1. **PLE for the assistant.** The assistant's `text_config` has
   `num_kv_shared_layers = 0`, but the parent unified config has PLE turned
   on for the target (E4B PLE pattern). Does the assistant compute its own
   PLE, or skip it entirely (since it doesn't own its full state)? The
   sglang code passes `per_layer_inputs=None` in the assistant's model call
   — strongly suggesting **PLE is bypassed in the assistant**. Confirm.

2. **Sliding-window layers.** The assistant has 4 layers with
   `layer_types` mixing `sliding_attention` and `full_attention`. Frozen-KV
   reads must respect this — a sliding-window assistant layer reading from
   a full-attention target layer would read tokens it isn't supposed to see.
   sglang's `bind_frozen_kv_context` says "HF Gemma 4 ties each typed
   (sliding/full) assistant layer to the target's last layer of the same
   type" — verify this mapping rule on the 12B checkpoint.

3. **K > 1 viability.** At K = 1 the per-cycle target verify already
   dominates wall time. The K > 1 benefit is bounded by the assistant
   acceptance rate decaying. The existing `mtplx-mlx::acceptance` module
   supports `Speculative` (Leviathan-Chen) — confirm whether the centroid
   head's masked logits are numerically stable enough for the acceptance
   ratio computation (the `−inf/2` mask values must not poison softmax).

4. **MTP for the canonical Gemma4 (27B) too?** sglang's
   `Gemma4AssistantForCausalLM` covers both the canonical and unified
   assistant variants (the unified one is just a thin subclass). Phase 1's
   loader could be parameterised to handle both, retiring the older
   `gemma4-mlx/src/mtplx_target.rs` stub at the same time. Worth scoping
   into Phase 7.

---

## 5. Next action

**Phase 1 (loader) is the natural first commit.** It's standalone and
unblocks the rest:
- No changes to existing infrastructure required
- Compiles and runs in isolation against the
  `mlx-community/gemma-4-12B-it-assistant-4bit` checkpoint
- Provides a smoke test that the config / weight layout we read from sglang
  matches the actual MLX checkpoint
- After it lands we can re-scope phases 2–8 with one verified data point.

Decision needed: proceed to Phase 1 implementation, or defer until other
priorities land?
