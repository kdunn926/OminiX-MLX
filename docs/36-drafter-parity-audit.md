# #36 — Gemma4 Paired-Model Drafter Parity Audit

Date: 2026-05-19. Branch: `perf/inference-optimization`.

## TL;DR

The Rust port of `Gemma4AssistantForCausalLM` in
`gemma4-mlx/src/assistant.rs` is architecturally and weight-wise sound.
All 94 safetensors are loaded; every per-layer shape derives correctly
from the assistant config; quantization packing math
(Q6 affine, group_size=64) checks out; per-layer-type RoPE and head
dimensions are wired; the sliding-window mask is applied on sliding
layers; concat order is correct; the recurrent loop propagates
`post_projection(inner)` as expected.

The observed drafter top-1 match rate of **3–10% against the MTPLX
target** is therefore **not from a structural port bug**. The remaining
delta vs the published `mtplx_pair.json` claim of `ratio: 0.981` at
`block_size=6` would require side-by-side intermediate-tensor
comparison with the upstream HF Python implementation — but
`gemma4_assistant` is not present in upstream `transformers`; it's a
custom MTPLX architecture that ships only with the model checkpoint.

This audit clears every check that can be performed without that
reference. Further investigation should pivot to runtime comparison
against a reference implementation (Python or HF custom code, if it
can be obtained).

## Weight inventory

`models/Gemma4-27B-MTPLX-Optimized-Speed/assistant/model.safetensors`
contains **94 tensors** total, all of which are loaded by
`load_assistant_model` in `gemma4-mlx/src/assistant.rs`:

- Per layer (×4 layers, 21 tensors each):
  - `input_layernorm.weight`
  - `layer_scalar`                       (per-layer learnable scalar)
  - `mlp.{down, gate, up}_proj.{weight, scales, biases}`
  - `post_attention_layernorm.weight`
  - `post_feedforward_layernorm.weight`
  - `pre_feedforward_layernorm.weight`
  - `self_attn.o_proj.{weight, scales, biases}`
  - `self_attn.q_norm.weight`
  - `self_attn.q_proj.{weight, scales, biases}`
- Model-level (10 tensors):
  - `model.embed_tokens.{weight, scales, biases}`
  - `model.norm.weight`
  - `pre_projection.{weight, scales, biases}`
  - `post_projection.{weight, scales, biases}`

Notably **absent** (confirming our architectural assumptions):
- No `self_attn.k_proj` / `v_proj` / `k_norm` → all K/V comes from the
  target's KV cache (cross-attention drafter).
- No centroid tensors → the `num_centroids: 2048` /
  `centroid_intermediate_top_k: 32` config fields are unused in this
  checkpoint. Our implementation correctly skips the centroid path.

## Shape audit

Verified via reading the safetensors header directly (no torch needed):

| Tensor | Shape | Decoded | Matches config? |
|---|---|---|---|
| L0 `q_norm.weight` | `[256]` | sliding `head_dim` | ✅ (text_config.head_dim=256) |
| L1 `q_norm.weight` | `[256]` | sliding `head_dim` | ✅ |
| L2 `q_norm.weight` | `[256]` | sliding `head_dim` | ✅ |
| L3 `q_norm.weight` | `[512]` | full `global_head_dim` | ✅ (text_config.global_head_dim=512) |
| L0 `q_proj.weight` | `[8192, 192]` U32 | `out=8192 = 32×256`, `in=192×32/6=1024` (hidden) | ✅ |
| L3 `q_proj.weight` | `[16384, 192]` U32 | `out=16384 = 32×512`, `in=1024` | ✅ |
| L0 `o_proj.weight` | `[1024, 1536]` U32 | `out=1024 (hidden)`, `in=1536×32/6=8192=32×256` | ✅ |
| L3 `o_proj.weight` | `[1024, 3072]` U32 | `out=1024`, `in=3072×32/6=16384=32×512` | ✅ |
| `pre_projection.weight` | `[1024, 2016]` U32 | `out=1024 (hidden)`, `in=2016×32/6=10752=2×5376` | ✅ |
| `post_projection.weight` | `[5376, 192]` U32 | `out=5376 (backbone_hidden)`, `in=1024` | ✅ |
| `embed_tokens.weight` | `[262144, 192]` U32 | `vocab=262144`, `embed_dim=1024` | ✅ |
| `model.norm.weight` | `[1024]` BF16 | hidden_size | ✅ |
| `layer_scalar` (each layer) | `[1]` BF16 | scalar broadcast | ✅ |

## Forward-pass audit

Walked through `AssistantModel::forward` and
`AssistantDecoderLayer::forward` line by line against the assistant
config and Gemma4 conventions:

1. `inputs_embeds`: `[B, q_len, 2*5376]` from
   `concat(recurrent_hidden, prev_token_embed)`. Concat order was
   the source of the earlier 1%→15% acceptance lift (commit `8b30953`).
2. `pre_projection`: `[B, q_len, 10752] → [B, q_len, 1024]`. ✓
3. For each of 4 decoder layers:
   - `residual = h`
   - `attn_in = input_layernorm(h)`
   - `attn_out = self_attn(attn_in, position_offset, shared_k, shared_v)`
   - `attn_out = post_attention_layernorm(attn_out)` (Gemma double-norm)
   - `h = residual + attn_out`
   - `residual = h`
   - `ff_in = pre_feedforward_layernorm(h)`
   - `ff_out = mlp(ff_in)`
   - `ff_out = post_feedforward_layernorm(ff_out)` (Gemma double-norm)
   - `h = residual + ff_out`
   - `h = h * layer_scalar`
4. `inner = model.norm(h)` (final RMSNorm)
5. `last_hidden = post_projection(inner)`: `[B, q_len, 1024] → [B, q_len, 5376]`
6. `logits = embed_tokens.as_linear(inner)` (tied lm_head, no softcap)

Inside `AssistantAttention::forward`:
1. `queries = q_proj(hidden)` → `[B, q_len, n_heads × head_dim]`
2. `queries = q_norm(reshape [B, q_len, n_heads, head_dim])` — per-head norm
3. `queries = transpose(0, 2, 1, 3)` → `[B, n_heads, q_len, head_dim]`
4. `queries = rope.apply(queries, position_offset)` (per-layer-type RoPE:
   sliding=default θ=10K, full=proportional θ=1M `partial_rotary_factor=0.25`)
5. Sliding-window slice: if `is_sliding && kv_len > sliding_window=1024`,
   restrict to last 1024 KV positions (commit `b1d4e82`)
6. SDPA: bidirectional cross-attention, no causal mask, scale =
   `1/sqrt(head_dim)`. GQA broadcast handled by MLX.
7. `o_proj(attn_out)` → `[B, q_len, hidden]`

All matches the assistant config and the Gemma4 conventions we ported
from the target.

## Suspect items investigated and cleared

| Suspect | Verdict |
|---|---|
| Concat order `[recurrent, prev_embed]` vs `[prev_embed, recurrent]` | Fixed in `8b30953`; current order is correct (confirmed by 1%→15% lift) |
| Sliding-window mask missing on sliding layers | Fixed in `b1d4e82`; correctness-only on hermes-class prompts (kv_len > 1024) |
| Per-layer-type head_dim wired correctly | ✅ checked against q_norm shapes (256 vs 512) |
| Per-layer-type RoPE (default vs proportional, theta, partial_rotary) | ✅ `build_rope` dispatches on `rope_type` from config |
| KV-head asymmetry (16 sliding, 4 full) | ✅ `n_kv_heads` chosen per layer type in loader (lines 545-551 of assistant.rs) |
| Layer scalar applied at end (not in residual stream) | ✅ matches HF Gemma-class pattern |
| Tied embed for LM head | ✅ via `MaybeQuantized<nn::Embedding>::as_linear` |
| Target hidden source: post-norm vs pre-norm | A/B via `MTPLX_PAIR_PRE_NORM_SEED=1` — neither produced a dramatic lift |
| Position offset semantics (single fixed `kv_offset-1` vs advancing) | A/B via `MTPLX_PAIR_POS=kv|advance|zero` — defaults to `kv_offset-1` |
| Embed scaling on `prev_token_embed` (`sqrt(target.hidden_size)`) | ✅ applied via `scaled_token_embed` (matches target's input convention) |
| Centroid lookup wired up | Confirmed unused in this checkpoint (no centroid weights present) |
| Acceptance math (greedy vs Leviathan-Chen) | LC implemented in `6284404`; correct math but acceptance bounded by joint distribution |

## What an HF-reference comparison would test

If an HF Python implementation of `gemma4_assistant` becomes
available, the following intermediate-tensor comparisons would
isolate any remaining numerical divergence (run on a fixed seeded
prompt; record bf16 tensors with `model.eval()` + `torch.no_grad()`):

1. **Input concat**: `inputs_embeds[0, 0, :16]` and `[0, 0, -16:]`
   should bit-match between Python (concatenate
   `(recurrent_hidden, prev_embed)`) and Rust
   (`build_inputs_embeds`).
2. **pre_projection output**: `h[0, 0, :16]` after the 10752→1024
   projection.
3. **Layer 0 attn output**: post-q_proj, post-q_norm, post-RoPE
   query at `[0, 0, head_0, :16]`. Same for `[0, n_heads-1, :16]`.
4. **Layer 0 attn weights / SDPA output**: trickier without
   intercepting MLX-internal SDPA; instead compare `attn_out` after
   `o_proj` at `[0, 0, :16]`.
5. **Per-layer hidden after `layer_scalar`**: `h[0, 0, :16]` at the
   exit of each decoder layer.
6. **Final `inner` (post `model.norm`)**: `[0, 0, :16]`.
7. **Logits**: top-10 token ids + their logits values. If logits
   diverge at this stage despite earlier layers matching, the LM
   head wiring is at fault.
8. **last_hidden (post_projection output)**: `[0, 0, :16]`.

An off-by-one in RoPE phase, a missed activation, or a
transposition mismatch would surface in step 3, 4, or 5.

## Recommendations

1. **Treat #36 as architecturally complete**. No further blind-port
   changes to `assistant.rs` are likely to move the needle.
2. **Audit the trained drafter's joint distribution against the
   target empirically** (no HF reference needed): on a fixed prompt,
   dump `argmax(drafter_logits)` and `argmax(target_logits)` for the
   first 50 decode positions. If they diverge >85% of the time, the
   drafter just wasn't trained against this particular target's
   distribution (possible — MTPLX pairs are tuned for specific
   prompt classes like the "flappy" suite mentioned in
   `mtplx_pair.json`).
3. **Try the `flappy` prompt suite** at the published config
   (T=1.0, top_p=0.95, top_k=64, block_size=6, seed=0,
   max_tokens=1000) — that's the *exact* setup the published
   `ratio: 0.981` was measured on (see `mtplx_pair.json`).
   Hermes-style chat prompts are likely out-of-distribution.
   **Our current LC implementation has top_k but not top_p**;
   adding nucleus truncation is a one-function addition similar
   to `truncate_top_k_inplace`. Without it, residual sampling
   bleeds probability mass into long tails even with top_k=64.
4. **If LC + chat-template hermes still <20%, mark #36 as
   "ported correctly, drafter quality insufficient for our
   workload" and pivot resources to #35 (expert-major MoE) or #40
   (continuous batching)** — both have clearer upside.

## Files

- `gemma4-mlx/src/assistant.rs` — the port being audited
- `gemma4-mlx/examples/draft_verify_spike.rs` — bench harness
- `gemma4-mlx/examples/pair_smoke.rs` — drafter-only smoke
- `gemma4-mlx/examples/assistant_smoke.rs` — assistant standalone
  load+forward smoke
- `models/Gemma4-27B-MTPLX-Optimized-Speed/assistant/config.json` —
  drafter config
- `models/Gemma4-27B-MTPLX-Optimized-Speed/assistant/generation_config.json` —
  drafter sampling config (`num_assistant_tokens: 6`, top_k=64, top_p=0.95)
- `models/Gemma4-27B-MTPLX-Optimized-Speed/mtplx_pair.json` — published
  reference (`ratio: 0.981`, `block_size: 6`, `flappy` prompts)
