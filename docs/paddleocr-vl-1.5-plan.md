# Inference support for `PaddleOCR-VL-1.5` — implementation plan

**Status:** design / exploration. No code committed yet.

**Reference:** `PaddlePaddle/PaddleOCR-VL-1.5` on HF
(`modeling_paddleocr_vl.py`, `image_processing_paddleocr_vl.py`,
`processing_paddleocr_vl.py`).

**Pipeline tag:** `image-text-to-text` — OCR / document-parse VLM with the
chat-completion shape.

---

## 1. Architecture summary

A small (~0.9 B total params) two-tower multimodal model. The text decoder is
labelled `Ernie4_5ForCausalLM` in the HF code but is structurally a stock
GQA Llama-family transformer with **multimodal RoPE (MROPE)**. The vision
side is a SigLIP-style ViT plus a 2×2 spatial-merge `Projector`.

### 1.1 Text decoder (`Ernie4_5ForCausalLM`)

| Property | Value |
|---|---|
| layers | 18 |
| hidden_size | 1024 |
| heads | 16 |
| kv_heads | **2** (4× GQA) |
| head_dim | 128 |
| intermediate_size | 3072 |
| vocab_size | **103,424** (custom tokenizer; not Llama/Mistral) |
| context | 131,072 |
| rope_theta | 500,000 |
| rope_scaling | `{ mrope_section: [16, 24, 24], rope_type: "default" }` |
| rms_norm_eps | 1e-5 |
| sliding_window | none |
| tie_word_embeddings | **false** (separate `lm_head`) |
| PLE / final-logit softcap / extra heads | none |

The big departure from anything currently in our tree is **MROPE**: position
encoding is 3-channel `(t, h, w)` rather than 1-channel `position_id`. For
text-only tokens all three channels carry the linear position; for image
tokens the `h, w` channels carry spatial grid coords within the image so
attention can reason about 2-D layout. `mrope_section: [16, 24, 24]` partitions
each `head_dim = 128`-wide rotation across the three position dims as
`16 + 24 + 24 + 64 (raw_left)` — first 32 dims get `t`-rotated, next 48 get
`h`-rotated, next 48 get `w`-rotated, remainder untouched.

### 1.2 Vision encoder (`PaddleOCRVisionModel`)

A SigLIP-style ViT with variable-resolution patch_embedding:

| Property | Value |
|---|---|
| layers | 27 |
| hidden_size | 1152 |
| heads | 16 |
| intermediate_size | 4304 |
| patch_size | **14** |
| image_size (canonical) | 384 |
| spatial_merge_size | 2 (post-encoder 2×2 token merge in `Projector`) |
| min_pixels | 112,896 (~336×336) |
| max_pixels | 1,003,520 (~1000×1000) |
| activation | `gelu_pytorch_tanh` |

Variable-resolution input: each image is bicubic-resized so its pixel area
lands in `[min_pixels, max_pixels]` with both H and W as multiples of
`patch_size × spatial_merge_size = 28`. Patchification is done via a
`Conv2d(stride=patch_size)` on the resized image, producing
`(num_patches, 1152)` features plus an `image_grid_thw = (1, H/patch, W/patch)`
descriptor.

### 1.3 `Projector` (vision → text)

```
image_features ∈ ℝ^[N, 1152]      where N = T·H·W (token count from the encoder)
image_grid_thw = (T, H, W)
m1 = m2 = 2

1. pre_norm   = LayerNorm(1152, ε=1e-5)
2. rearrange  ("(t h p1 w p2) d → (t h w) (p1 p2 d)")
                                          ↑ collapse spatial 2×2 blocks into single tokens
3. linear_1   : ℝ^[N/(m1·m2), 4608] → ℝ^[N/4, 4608]
4. GELUActivation (tanh approx)
5. linear_2   : ℝ^[N/4, 4608] → ℝ^[N/4, 1024]  (text hidden size)
```

After the projector, each 2×2 spatial block of the vision encoder becomes
**one** soft token in the text-space embedding. For a 336×336 image that's
24×24 = 576 → 144 soft tokens. Variable-resolution images therefore produce
variable soft-token counts, which the chat template prefixes with `<im_start>`
/ `<im_end>` markers around the right number of placeholder `image_token_id`s.

### 1.4 Image preprocessing (`PaddleOCRVLImageProcessor`)

```
1. Convert RGB
2. Resize: pick (h_resized, w_resized) such that
     h_resized × w_resized ∈ [min_pixels, max_pixels]
     aspect_ratio preserved
     both dims multiples of patch_size × spatial_merge_size = 28
   Use bicubic resample.
3. Rescale: pixel /= 255  → [0, 1]
4. Normalize: (pixel − 0.5) / 0.5 → [−1, 1]
5. Output shape (T, C, H, W) = (1, 3, h_resized, w_resized)
   image_grid_thw = (1, h_resized/14, w_resized/14)
```

### 1.5 End-to-end inference flow

```
1. Preprocess each image → (pixel_values, image_grid_thw)
2. Vision encoder forward → image_features ∈ ℝ^[N, 1152]
3. Projector(image_features, image_grid_thw) → soft_tokens ∈ ℝ^[N/4, 1024]
4. Tokenize prompt with image_token_id × (N/4) per image
5. inputs_embeds = embed_tokens(input_ids)
   Splice soft_tokens into inputs_embeds at every image_token_id position
6. Build 3-D MROPE position_ids:
     - text tokens: (pos, pos, pos)
     - image tokens: (segment_idx, h_grid, w_grid)
7. Forward Ernie4.5 decoder with MROPE → logits
8. Sample → next token
```

The decode loop is otherwise standard single-token AR. MROPE for text-only
post-image generation simply continues the time-channel and replicates it
into h, w.

---

## 2. Implementation roadmap

Eight phases, separable so we can pause / benchmark between them.

### Phase 1 — Configs + tokenizer wiring

Files: `paddleocr-vl-mlx/src/config.rs` (new crate), `paddleocr-vl-mlx/src/lib.rs`.

- `PaddleOcrVlConfig` deserializer for the top-level config (text fields are
  on the root, vision under `vision_config`).
- Tokenizer load: stock `tokenizers` crate against `tokenizer.json`; verify
  `image_token_id`, `vision_start_token_id`, `vision_end_token_id` resolve.
- New crate `paddleocr-vl-mlx` (mirrors gemma4-mlx / qwen3-vl-mlx structure).

### Phase 2 — Ernie 4.5 text decoder

Files: `paddleocr-vl-mlx/src/text_model.rs`.

Implements:
- `Ernie45Config` (text-side fields)
- Standard GQA Llama-style attention with **MROPE** instead of 1-D RoPE
- 18 `DecoderLayer`s (RMSNorm → Attention → RMSNorm → SwiGLU MLP)
- `Model::forward_from_embeds(embeds, position_ids_3d, cache)` returning
  logits at all positions (we need the per-position variant from the start
  for image-token splicing; the canonical pattern is in gemma4-mlx's
  `forward_all_logits_from_embeds`).

Notable: **MROPE** requires a small kernel change vs. our existing RoPE:
instead of `cos[pos] / sin[pos]` indexed by a 1-D position, we precompute
three cos/sin tables (one per channel), and gather each rotation according
to `mrope_section`. Reference: `apply_multimodal_rotary_pos_emb` in
`modeling_paddleocr_vl.py:313`.

**Risk**: Easy if we adapt our existing RoPE code. There's no in-tree MROPE
yet; closest is `qwen3-vl-mlx` which does 2-D RoPE for vision tokens but not
the 3-D variant here.

### Phase 3 — SigLIP-style vision encoder

Files: `paddleocr-vl-mlx/src/vision_model.rs`.

- `Conv2d` patch embedding (stride = patch_size = 14)
- Learned absolute position embeddings on a fixed `(image_size/patch_size)²`
  grid, **interpolated** to the variable-resolution input grid at runtime
- 27 transformer layers (LayerNorm pre-attn, MHA with full attention, MLP
  with `gelu_pytorch_tanh`)
- Variable-resolution attention: no padding mask needed (each image is
  processed individually), but Q/K rotation uses
  `apply_rotary_pos_emb_vision` over a 2-D position grid

**Risk**: Vision encoder is the biggest piece (~27 × full transformer block).
Most of the structure is shared with `qwen3-vl-mlx`'s vision tower. Sliding
window not used. Reusable: attention, MLP, LayerNorm.

### Phase 4 — Projector

Files: `paddleocr-vl-mlx/src/projector.rs`.

```rust
pub struct Projector {
    pre_norm: nn::LayerNorm,                 // dim = 1152
    linear_1: nn::Linear,                    // 4608 → 4608, bias
    linear_2: nn::Linear,                    // 4608 → 1024, bias
}

impl Projector {
    /// Returns soft tokens in text-hidden space.
    pub fn forward(
        &self,
        image_features: &Array,    // (N, 1152)
        grid_thw: (i32, i32, i32), // (T, H, W) in PATCH coordinates
    ) -> Result<Array> { ... }
}
```

The `einops` rearrange is two reshapes + a transpose in MLX. Small, low risk.

### Phase 5 — Image preprocessor

Files: `paddleocr-vl-mlx/src/preprocess.rs`.

- Decode (use `image` crate, already a dep)
- Aspect-preserving resize to multiple-of-28 dims within
  `[min_pixels, max_pixels]` (bicubic via `image::imageops::resize` with
  `FilterType::CatmullRom`)
- Rescale + normalize → `[-1, 1]`
- Patchify in CHW order to match the Conv2d patch_embedding kernel layout
- Return `(pixel_values, image_grid_thw)`

Almost identical structure to the gemma4 / unified preprocessors we already
have; the only new wrinkle is the variable-resolution sizing math.

### Phase 6 — MROPE position-id builder

Files: `paddleocr-vl-mlx/src/mrope.rs`.

```rust
/// Build 3-D position_ids for a sequence containing N image regions.
///   - Text-only tokens get (i, i, i)
///   - The k-th image region's tokens get (segment_k, h_in_grid, w_in_grid)
/// where segment_k counts image regions, and h_in_grid / w_in_grid index
/// into the corresponding image_grid_thw[k].
///
/// Reference: PaddleOCRVLForConditionalGeneration.get_rope_index in
/// modeling_paddleocr_vl.py:1963.
pub fn build_position_ids_3d(
    input_ids: &[i32],
    image_token_id: i32,
    image_grids: &[(i32, i32, i32)],  // (T, H, W) per image
    spatial_merge_size: i32,           // 2
) -> Array;  // shape (3, T_seq)
```

Direct port of `get_rope_index`. Risk: low — pure Rust math.

### Phase 7 — End-to-end model + API wire-up

Files: `paddleocr-vl-mlx/src/model.rs` (combining text + vision + projector),
`OminiX-API/src/engines/llm.rs` (new `ModelBackend::PaddleOcrVl` variant).

```rust
pub struct PaddleOcrVlModel {
    pub text:      Ernie45Model,
    pub vision:    PaddleOcrVisionModel,
    pub projector: Projector,
    pub image_token_id:   i32,
    pub vision_start_id:  i32,
    pub vision_end_id:    i32,
}

impl PaddleOcrVlModel {
    pub fn encode_image_bytes(&mut self, bytes: &[u8])
        -> Result<(Array /* soft_tokens (N/4, 1024) */, (i32, i32, i32) /* grid */)>;
    pub fn prefill_multimodal<C>(
        &mut self,
        input_ids: &[i32],
        image_features: &[(Array, (i32, i32, i32))],
        cache: &mut Vec<C>,
    ) -> Result<Array /* last-token logits */>;
    pub fn decode_token<C>(&mut self, token_id: i32, position_ids: &Array, cache: &mut Vec<C>)
        -> Result<Array /* last logits */>;
    pub fn new_cache(&self) -> Vec<KVCache>;
}
```

API match-arm wiring follows the unified VL pattern from `b5ac749` / `cfca98f`
(13 sites — `is_vl_backend`, `build_vl_messages`, paged-KV gate, template
routing, `generate_*_vl_multimodal`, streaming wrapper, swap-suite glue).

### Phase 8 — Validation

- Smoke test: load + `encode_image_bytes` + `prefill_multimodal` + decode
  loop returns coherent OCR output for a known image (one of the README
  examples).
- e2e: add the model to `tests/run_all_e2e.sh` matrix; verify curl + goose +
  hermes-gateway + swap suites pass.
- Bench: `decode_tps` baseline; PLD compatibility check (this model is a
  prime PLD target — OCR output literally copies from the image's text,
  which after a few tokens will mirror the prompt's RAG-style ground-truth).

---

## 3. Effort estimate

| Phase | LOC | Wall-clock | Risk |
|---|---|---|---|
| 1. Config + tokenizer | 150 | 0.25 day | low |
| 2. Ernie 4.5 text decoder | 600 | **1.5 days** | medium — MROPE kernel is new |
| 3. SigLIP vision encoder | 700 | **1.5 days** | medium — adapt from qwen3-vl |
| 4. Projector | 80 | 0.25 day | low |
| 5. Image preprocessor | 150 | 0.5 day | low |
| 6. MROPE position-id builder | 100 | 0.25 day | low |
| 7. E2E model + API wire-up | 300 | 0.5 day | low (pattern established) |
| 8. Smoke + e2e + bench | 100 | 0.5 day | low |
| **Total** | **~2180** | **~5 days** | |

Risk concentration: **MROPE** (phase 2) and **variable-resolution attention**
(phase 3). The text decoder is otherwise a stock GQA Llama lookalike; the
vision tower is otherwise stock SigLIP. The split is similar to how
qwen3-vl-mlx was structured.

---

## 4. Reuse opportunities from existing crates

- **`qwen3-vl-mlx`**: vision encoder structure is the closest match
  (SigLIP-style ViT, variable resolution, 2-D RoPE on Q/K). Patch embedding
  + transformer block can be lifted with minor changes. Projector pattern
  (Linear → activation → Linear) is also similar.
- **`qwen3.6-mlx`**: text decoder GQA attention + RMSNorm + SwiGLU MLP are
  byte-for-byte the same except for MROPE. Can lift the attention shell and
  swap `rope_apply` for `mrope_apply`.
- **`gemma4-mlx/src/unified_vision.rs`** (recent work): the
  `encode_image_bytes` → `prefill_multimodal` → `decode_token` →
  `new_cache(_paged)` method shape is the contract the API expects. Use the
  same surface for `PaddleOcrVlModel` so the API wire-up is the established
  13-site pattern.

---

## 5. Open questions

1. **MROPE caching.** The text decoder needs the full 3-D position_id
   sequence at prefill time; during AR decode the new position is just one
   step's worth. We need a cheap incremental "next position" computation
   that mirrors HF's `get_rope_index` without rebuilding the full table.
   Likely a small helper that takes the previous-step's `(t, h, w)` and
   bumps `t` by 1, keeping `h, w` mirrored.

2. **`vocab_size = 103,424`.** Confirm the tokenizer has full coverage of
   English + the digits / punctuation / Latin extended typical of OCR
   targets (esp. for non-ASCII document parsing). The tokenizer is custom
   per HF's `tokenizer.json` — no surprises expected, but should validate
   on a sample doc image.

3. **PaddleOCR-VL-1.6 vs 1.5.** The 1.6 release is also up on HF (and a 1.6
   GGUF). 1.6 likely supersedes 1.5; worth confirming whether the API
   contract is identical (just a checkpoint diff) or whether 1.6 introduced
   architectural changes. If 1.6 is a drop-in, target it first.

4. **OCR-specific decoding tricks.** The model card mentions "document
   parsing" — common practice in OCR pipelines is to constrain the model
   to a strict JSON / Markdown grammar. Our existing JSON-grammar gate
   (`bc7148c`, recently extended for Gemma 4 tool-calling) should drop in
   unchanged.

---

## 6. Recommended next action

Phase 1 (config + tokenizer load) is the natural first commit — standalone,
~150 LOC, unblocks phase 2. The single most de-risking step before
committing to the full ~5-day path is **drafting phase 2's MROPE kernel
against a tiny synthetic input** (no checkpoint needed) and asserting the
output matches a 5-line PyTorch reference. If MROPE comes out clean in a
half-day spike, the rest is mechanical adaptation of crates we already
have.
