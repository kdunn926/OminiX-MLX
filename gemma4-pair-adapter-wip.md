# Gemma4 target+assistant pair adapter — WIP

Status: **end-to-end working** (2026-05-16, same session as kickoff).
Branch `feat/gemma4-pair-adapter` still at `perf/inference-optimization`
HEAD — work is local, not yet committed.

What works:

- `examples/mtplx_chat.rs` — standalone AR generation on the MTPLX 27B
  target. Validated on `"What is 2 + 3?"` → `"2 + 3 = 5"` and on
  `"Write a one-sentence haiku about autumn."` → `"Gold leaves drift
  away."`. Uses the existing `Gemma4ChatTemplate` for proper prompt
  formatting and the existing `Generate` iterator for AR decoding;
  greedy decode terminates at the `<channel|>` EOS marker as expected.

Pair adapter (`cargo run --release -p gemma4-mlx --example pair_smoke`):
- `gemma4_mlx::assistant` — full port of `Gemma4AssistantForCausalLM`
  (4 layers, Q-only bidirectional cross-attn, Q6-affine-G64 loader).
  Validated standalone by `examples/assistant_smoke.rs`.
- `gemma4_mlx::mtplx_target` — loader for the MTPLX-Optimized-Speed
  target (Q4 weights with `language_model.model.*` key prefix). Now a
  thin prefix-translation wrapper; weights stay packed Q4 and use native
  `quantized_matmul` (no BF16 blow-up). Memory ~13GB resident, decode
  ~12 tok/s on a 96GB M-class machine.
- Underlying refactor: `gemma4-mlx`'s `Attention`/`DenseMlp`/`Router`/
  `Model`/`LanguageModel`/`DecoderLayer` field types switched from
  `nn::Linear` / `nn::Embedding` to `MaybeQuantized<...>`. The loader
  builds `Quantized` variants whenever the checkpoint provides
  `(.weight, .scales, .biases)` triplets and `Gemma4Config.quantization`
  is set; non-quantized BF16 checkpoints still load via the same path
  as `Original` variants. Backwards-compatible with the regular
  `gemma-4-26B-A4B-it` model.
- `examples/pair_smoke.rs` — full pipeline:
  prompt → tokenize → target prefill → extract last sliding/full layer
  K/V via `KVCache::current_kv()` → assistant recurrent loop with
  argmax sampling → decode. First validated run on prompt
  `"Hello, how are you?"` produced
  `" How are you, how are you, how are you, how are you,"` — coherent
  English continuation, repetitive because we used pure argmax with no
  temperature/top-p.

Known limitations:
- Repetition is expected at temp=0 (argmax) for an unverified drafter.
  Real spec-decoding will replace argmax with target-verified sampling.
- Target prefill takes ~150s on a short prompt due to BF16-dequantized
  weights + first-shot Metal kernel compile. Subsequent runs of the
  same binary are faster.
- The Q4 → BF16 dequant blow-up (~16GB → ~54GB resident) is fine for
  this demo on a 96GB machine but unacceptable for production. Replace
  with a quantized-native DecoderLayer in a follow-up.

Safetensors inspection summary and all architectural details below
remain accurate; the §Wiring section reflects the upstream HF reference
and was the basis for the implementation.

## Architecture (confirmed from `assistant/model.safetensors` header)

All weights are Q6 affine, group_size=64. Decoding U32 column dim: real
`in_features = cols * 32 / 6`. Embedding `[262144, 1024]`.

```
input draft-token id ─► embed_tokens (262144→1024, Q6)  ┐
                                                         ├─► (combine — TBD how)
target backbone hidden ─► pre_projection (10752→1024)   ┘
                                       │
                                       ▼
                  ┌───── 4× DecoderLayer ─────┐
                  │  input_layernorm           │
                  │  self_attn:                │
                  │    q_proj only (no K/V!)   │  K,V borrowed from TARGET
                  │    q_norm                  │  (num_kv_shared_layers=4 ⇒ all)
                  │    o_proj                  │
                  │  layer_scalar (bf16 [1])   │  per-layer residual scale
                  │  pre_feedforward_layernorm │
                  │  mlp (gate/up/down, 1024↔8192)
                  │  post_feedforward_layernorm│
                  │  post_attention_layernorm  │
                  └────────────────────────────┘
                                       │
                                       ▼
                              model.norm (1024)
                                       │
                                       ▼
                       post_projection (1024→5376, Q6)
                                       │
                                       ▼
                  TARGET's tied embed (262144×5376) → logits
```

Per-layer attention shapes:
- Layers 0-2 (sliding, window=1024): `q_proj 1024→8192` = 32h × 256 head_dim; `q_norm[256]`; `o_proj 8192→1024`.
- Layer 3 (full attention): `q_proj 1024→16384` = 32h × 512 head_dim (`global_head_dim`); `q_norm[512]`; `o_proj 16384→1024`.

Key facts the original doc got wrong:
- **No centroid table tensor exists** in the safetensors. `num_centroids=2048`,
  `centroid_intermediate_top_k=32` are runtime hyperparams only — likely a
  fast top-k sampler over `post_projection · target_embed^T`, NOT an
  architectural lookup. Can be ignored for the v1 port (use plain argmax/top-k
  over the full 262144-vocab logits to validate correctness first).
- **K and V are not in the assistant's weights at all** — `attention_k_eq_v`
  + `num_kv_shared_layers=4` (= all layers) means every assistant layer
  borrows K/V from the target's KV cache. The adapter must wire target
  KV-cache slices through to assistant attention. The mapping (which target
  layer → which assistant layer) is **not** in any config file inspected so
  far; needs HF source.
- **The LM head is the target's tied embed**, not the assistant's. The
  assistant's own `embed_tokens` is used only on the input side (to embed
  the previously emitted draft token). `post_projection` lifts 1024→5376
  so that the target's `[262144,5376]` tied embed can decode.
- **`pre_projection` accepts 10752-dim**, exactly `2 × 5376`. Strong evidence
  the input concatenates hidden from **two target layers** (or hidden +
  residual, or hidden across two timesteps). HF source needed to disambiguate.

## Wiring (resolved from `transformers/models/gemma4_assistant/modeling_gemma4_assistant.py` + `generation/candidate_generator.py::Gemma4AssistantCandidateGenerator`)

### Per-drafter-step inputs

```python
last_token_embedding = target_input_embeddings(last_token_id)   # [B,1,5376] — TARGET's embed
inputs_embeds = cat([last_token_embedding, last_hidden_state], dim=-1)  # [B,1,10752]
# step 0: last_hidden_state = target.last_layer_hidden[B, n_last_matches:n_last_matches+1, :]
# step k>0: last_hidden_state = outputs.last_hidden_state  (= drafter's post_projection output)
```

So `pre_projection`'s 10752 input = `[target_embed(prev_token) || recurrent_hidden]`,
where the recurrent hidden is **the target's last layer hidden** on the first
step and **the drafter's own `post_projection` output** on every subsequent
step. This is the GRU-style recurrence that lets the drafter run AR without
its own KV.

### K/V wiring — `shared_kv_states`

```python
shared_kv_states = model_outputs.shared_kv_states  # produced by target with return_shared_kv_states=True
# dict[str, (K, V)] keyed by "full_attention" and "sliding_attention"
# Each is taken from the TARGET's LAST layer of that type
# Shape: [B, num_kv_heads, kv_len, head_dim], cropped to kv_len=current_length each iteration
# FROZEN across the entire draft block — only inputs_embeds rotates per step
```

The assistant attention is **bidirectional cross-attention** against these
K/V (see `create_attention_masks` — uses `create_bidirectional_mask` /
`create_bidirectional_sliding_window_mask`, NOT causal). Assistant layers
0-2 (sliding) attend over `shared_kv_states["sliding_attention"]`; layer 3
(full, `global_head_dim=512`) attends over `shared_kv_states["full_attention"]`.
The drafter never builds its own KV — Q comes from `pre_projection`'s output
flowing through `q_proj`, K and V come straight from the target's cache.

### Outputs

```python
@dataclass class Gemma4AssistantOutput:
    last_hidden_state = post_projection(inner_last_hidden)   # [B,1,5376] — recurrent
    logits            = lm_head(inner_last_hidden)            # [B,1,262144] — sampling head
```

`lm_head` is tied to the assistant's own `embed_tokens` `[262144, 1024]`
(_tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}). So:
- The **logits** path uses the assistant's own 1024-d tied embed.
- The **recurrent hidden** path uses `post_projection` to lift 1024 → 5376
  so it can replace the target's hidden in the next step's concat.

### Items that supersede earlier guesses

- ~~"LM head is the target's tied embed"~~ — **wrong**. Logits come from
  assistant's own tied lm_head; `post_projection` is for the recurrent
  hidden only.
- ~~"`embed_tokens` is used to embed the previously emitted draft token"~~
  — **wrong**. It's used *only* through the tied lm_head on the output side.
  The previous token is embedded by the **target's** embeddings on the
  input side (`target_model_input_embeddings(last_token_id)`).
- ~~"Need to figure out which target layers feed pre_projection"~~ —
  resolved. Last layer hidden of the target (step 0) then drafter's own
  recurrent output (step k>0).
- ~~"Need to figure out which target K/V each assistant layer borrows"~~ —
  resolved. Last `sliding_attention` layer of target → all sliding assistant
  layers (0-2); last `full_attention` layer of target → assistant layer 3.

### Remaining minor unknowns

1. **`layer_scalar` placement** — single bf16 scalar per layer. Lives on
   `Gemma4AssistantDecoderLayer` (not in the assistant modeling file; in
   the inner `gemma4_text` model that `AutoModel.from_config(text_config)`
   instantiates). Need to grep `modeling_gemma4.py` for `layer_scalar`.
2. **Rotary position embeddings** — assistant has rope params per
   layer-type (`rope_theta=10000` for sliding, `1000000` with
   `proportional` type and `partial_rotary_factor=0.25` for full). The
   `position_ids` passed to the drafter is `[[input_ids.shape[1]-1]]`
   (a single absolute position per step). Confirm Q is rotated, K is not
   (since K is borrowed already-rotated from target — or is it rotated
   again? Almost certainly the target's stored K is **post-rotation**,
   so assistant must NOT re-rotate; verify in `Gemma4Attention`).
3. **`return_shared_kv_states` on the target** — our existing
   `Gemma4TargetAdapter` captures hidden states but does NOT currently
   surface per-layer-type K/V. We need to (a) identify the "last sliding"
   and "last full" layer indices in the 26B/27B target and (b) expose
   their K/V buffers from `KVCache` for the adapter to hand to the
   drafter. This is the biggest new piece of plumbing.

## Updated work breakdown

## Goal

Classical drafter+verifier speculative decoding for Gemma4 — distinct from
the MTP-head path. Target = the existing 26B/27B Gemma4 model; drafter = a
small `Gemma4AssistantForCausalLM` shipped alongside the target in
`models/Gemma4-27B-MTPLX-Optimized-Speed/`.

## Why this is not "alias the model_type"

The drafter advertises `model_type=gemma4_assistant` and is **architecturally
distinct** from the target. Inspecting
`models/Gemma4-27B-MTPLX-Optimized-Speed/assistant/config.json`:

| dim | target gemma4 | gemma4_assistant |
|---|---|---|
| `hidden_size` | 5376 | **1024** |
| `num_hidden_layers` | ~40+ | **4** |
| `num_kv_shared_layers` | partial | **4 (all)** |
| `attention_k_eq_v` | false | **true** (K and V tied) |
| layer pattern | mixed | `[sliding × 3, full × 1]` |
| `vocab_size` | 262144 | 262144 (shared) |
| `tie_word_embeddings` | true | true |
| `backbone_hidden_size` | n/a | **5376** (input from target) |
| `num_centroids` | n/a | **2048** |
| `centroid_intermediate_top_k` | n/a | **32** |

The drafter is a ~50M-param model that:

1. Reads the target's `[B, T, 5376]` hidden state as input (the "backbone"
   signal). This requires a projection or centroid lookup down to its own
   1024-dim hidden space — that's what `num_centroids`/
   `centroid_intermediate_top_k` controls (centroid-based lookup with a top-k
   sparse mix, not a plain matmul).
2. Runs 4 small transformer layers with K=V-tied attention and 3 sliding-
   window layers + 1 full attention. Sliding window is 1024.
3. Re-uses the target's embedding matrix (vocab 262144) for the LM head via
   tied weights, but its own 1024-dim hidden state must be projected to
   vocab-space at output — likely via the same centroid table run in reverse
   or a separate proj head.

### 1. `gemma4-mlx/src/assistant.rs` (new) — ~1-2 days

Port `Gemma4AssistantForCausalLM` from the HF transformers reference.
Need to find the source — likely in:

- the assistant's `model.safetensors` weight layout (inspect via safetensors
  index to see what tensor names exist — that reveals architecture)
- the upstream `transformers` repo: `Gemma4AssistantForCausalLM` class.
  Was new as of `transformers_version: 5.7.0.dev0` per config. Likely
  unreleased — check google/gemma-4-31B-it-assistant model card and the
  `modeling_gemma4.py` source.
- the `mtplx_artifact.json` in the assistant dir may carry construction hints

Components to implement:
- Centroid embedding lookup module (`num_centroids=2048`, `top_k=32` —
  probably gathers top-32 centroid vectors weighted by some routing scalar
  computed from the input)
- 4-layer transformer block with `attention_k_eq_v` (one projection that
  serves as both K and V — saves ~25% of attention proj weight)
- Sliding-window mask handling (already exists in target gemma4 — reuse)
- LM head: either tied-embed projection back, or a separate centroid-driven
  decode path
- `load_assistant(path)` that reads the `assistant/` subdirectory

### 2. `dflash-mlx/src/engine/gemma4_pair_adapter.rs` (new) — ~0.5 day

Mirror `Gemma4TargetAdapter` but with a drafter side. The target adapter
already does:
- `prefill` → returns logits, captures backbone hidden into segments
- `verify(drafted)` → forward, capture hidden, return logits
- `rollback_kv(n_keep)` via `KVCache::trim`

The pair adapter needs:
- A `Gemma4AssistantAdapter` that takes `[B, T_seen, 5376]` backbone hidden
  (from the target's `last_target_hidden` accessor) plus the most recent
  accepted token, and emits draft tokens autoregressively until block_len
- Its own KV cache for the 4-layer drafter (trim-on-rollback)
- The shared spec loop (already in `dflash-mlx/src/engine/spec_epoch.rs`)
  consumes both via the `TargetModel` + draft-model traits

### 3. `dflash-mlx/examples/bench_gemma4_pair.rs` (new) — ~0.25 day

Following the pattern of `bench_gemma4` (target-only) and the existing pair
benches. Args: `<target_dir> <assistant_dir> <max_tokens> <prompt>`.
Bench against `ar_bench` (AR) and `bench_gemma4` (DFlash on the gemma4 path,
if applicable) on `hermes-gateway` fixtures.

### 4. Acceptance-ratio investigation

The benchmark JSON in `mtplx_pair.json` claims `ratio: 0.981` at `block_size=6`
and `observed_mtp_tok_s ≈ 44 tok/s`. That's the *Python* reference number on
this exact pair. Validating against our Rust port is the parity gate.

## Open questions before starting

- **Centroid lookup math** — is it weighted gather (top-k indices + weights
  → weighted sum of centroid vectors) or learned routing? The 32×1024 = 32k
  scalar matmul per token is cheap; the centroid table is 2048×1024×2 bytes
  = 4 MiB, fits easily.
- **Backbone hidden shape** — confirm the assistant takes target's hidden
  *before* the LM head (so 5376-dim, pre-norm-ish) or *after* the LM head
  projection. Pre-LM-head is what the existing
  `Gemma4TargetAdapter::last_target_hidden()` returns.
- **Decode side** — does the assistant produce its own logits via a private
  head, or does it produce a hidden state that gets routed back through the
  target's LM head? The latter would explain the centroid table's role at
  output time too.
- **Quantization** — assistant ships at 6-bit affine (group 64), target at
  4-bit. The existing gemma4-mlx quantized loader handles 4-bit; verify it
  handles 6-bit too (likely yes — `QuantizedLinear` is bits-agnostic).

## Recommended kick-off

1. Spend 30 min with `safetensors-cli show` (or python +
   `safetensors.safe_open`) on `assistant/model.safetensors` to dump tensor
   names + shapes. That gives the architecture skeleton without speculation.
2. Open the upstream `Gemma4AssistantForCausalLM` source side-by-side with
   `gemma4-mlx/src/model.rs` (the target).
3. Implement assistant + adapter in one branch. Target acceptance: parity
   against the Python `0.981` ratio on the same prompt suite.

## Pointers

- Target adapter: `dflash-mlx/src/engine/gemma4_adapter.rs`
- Target model: `gemma4-mlx/src/model.rs`
- Spec loop: `dflash-mlx/src/engine/spec_epoch.rs`
- Existing pair example for shape reference: search for any non-MTP
  drafter+verifier pair in `dflash-mlx/examples/` (Qwen3.6 DFlash uses the
  block-diffusion drafter, which is structurally different from this)
- Pair manifest: `models/Gemma4-27B-MTPLX-Optimized-Speed/mtplx_pair.json`
- Python reference performance: 44 tok/s @ ratio 0.981 on the
  `flappy` prompt suite, block_size=6

## Estimate

~2 days for the architectural port, ~0.5 day adapter + bench, ~1 day parity
investigation. **Total: 3-4 days of focused work.**
