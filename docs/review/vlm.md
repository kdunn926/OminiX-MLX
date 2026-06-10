# Code review: minicpm-sala-mlx, moxin-vlm-mlx, qwen3-vl-mlx

Scope: `src/` and `examples/` of the three crates, read-only. Findings verified by reading
the code against the reference implementations (HF Qwen3-VL processor/vision model,
InfLLMv2/Lightning-attention papers + MiniCPM-SALA reference, Prismatic/SigLIP).

## HIGH

1. **[bug / preprocessing] qwen3-vl-mlx/src/lib.rs:841-844** — `preprocess_image` only
   rescales pixels to `[0, 1]`; the Qwen3-VL processor additionally normalizes with
   `image_mean/image_std = [0.5, 0.5, 0.5]`, so the ViT was trained on inputs in `[-1, 1]`.
   Fix: `(v/255.0 - 0.5) / 0.5` per channel.

2. **[bug / vision RoPE] qwen3-vl-mlx/src/lib.rs:300-332** — `VisionAttention::forward`
   applies no positional rotation at all, but the reference Qwen3-VL ViT applies 2-D rotary
   embeddings (from each patch's `(row, col)` coordinates) to Q/K in every block, in addition
   to the learned `pos_embed`. Fix: build per-patch 2-D RoPE cos/sin from `(h_patches, w_patches)`
   and apply to q/k (same pattern as the recent paddleocr-vl-mlx fix, commit f918e3d).

3. **[bug / pos-embed math] qwen3-vl-mlx/src/lib.rs:434-438** — positional embeddings are
   looked up with linear ids `0..h*w` out of the learned 48x48 grid (`num_position_embeddings`
   = 2304), but the reference bilinearly interpolates the 48x48 grid down to the actual
   `(h_patches, w_patches)` grid (`fast_pos_embed_interpolate`). With the hardcoded 28x28 grid
   this reads only the first ~16 rows of the table with wrong geometry. Fix: implement bilinear
   interpolation of the 48x48 table to the target grid.

4. **[bug / merge-window math] qwen3-vl-mlx/src/lib.rs:384,406** — `FinalMerger` /
   `DeepStackMerger` reshape `[N, d] -> [N/4, 4d]` on row-major patch order, merging 4
   horizontally adjacent patches in the same row; the reference processor pre-permutes patches
   into merge-grouped order so each group of 4 is a 2x2 spatial block. Fix: permute hidden
   states `[h/2, 2, w/2, 2]` -> `[h/2, w/2, 2, 2]` before the merger reshape (and keep
   pos-embed assignment consistent).

5. **[bug / decay math] minicpm-sala-mlx/src/attention/lightning.rs:535-566** — the partial
   last chunk is zero-padded to `C=64` but the state update reuses `reverse_decay`/`chunk_decay`
   built for a full chunk, so after a prefill of length `L % 64 = n != 0` the recurrent state
   carries a spurious extra decay `exp(slope*(C-n))` on every head (up to `e^{-44}` for the
   steepest slope) — long-range memory is suppressed for essentially every prompt length.
   Fix: build decay tensors for `actual_c` for the final chunk (or rescale state by
   `exp(-slope*(C-actual_c))` after the fused update).

## MEDIUM

6. **[bug / multi-image splice] qwen3-vl-mlx/src/lib.rs:726-738** — the prefill splice
   handles exactly one contiguous image block: after `image_inserted` is set, any further
   `image_token_id` falls into the `image_inserted` branch and is embedded as an ordinary
   text token, silently corrupting a second image; there is also no check that the run length
   of pad tokens equals `n_visual`. Fix: support a list of `(features, span)` per image and
   error on count mismatch.

7. **[bug / sparse-attention masking] minicpm-sala-mlx/src/attention/sparse.rs:233-297** —
   `infllmv2_attention` selects top-k blocks using only the last query position
   (`q_pos = L_q - 1`) and runs the final SDPA with no causal mask, so for multi-token inputs
   past `dense_len` (long prefill, speculative verify) earlier queries attend to future tokens
   inside the gathered window region and reuse the last token's block selection. Fix: pass a
   causal mask over the gathered window segment and select blocks per query position (or
   restrict the sparse path to `L_q == 1`).

8. **[bug / kernel-stride] minicpm-sala-mlx/src/attention/sparse.rs:115-129** —
   `compress_keys` mean-pools non-overlapping `kernel_size=32` windows and uses those pools as
   the selection unit; reference InfLLMv2 pools with overlapping `kernel_stride=16` windows and
   aggregates scores onto `block_size=64` token blocks before top-k. `kernel_stride`/`block_size`
   from `SparseConfig` are effectively unused, so block scoring/granularity diverges from the
   trained model. Fix: implement strided pooling and score aggregation per 64-token block.

9. **[bug / preprocessing] qwen3-vl-mlx/src/lib.rs:831-832** — every image is
   `resize_exact(448, 448)`, destroying aspect ratio; the reference `smart_resize` keeps aspect
   ratio while rounding H/W to multiples of `patch_size*merge_size` within min/max pixel budgets.
   Fix: implement smart_resize and pass the resulting per-image grid dims through.

10. **[bug / activation] moxin-vlm-mlx/src/vision.rs:157-161** — `ViTMlp` uses exact
    `nn::gelu` for both backbones; SigLIP-SO400M's reference act is `gelu_pytorch_tanh`
    (DINOv2's exact GELU is correct). Small but systematic divergence across 27 SigLIP layers.
    Fix: add an act-fn flag to `ViTConfig` and use `gelu_approximate` for SigLIP.

11. **[silent loader skips] minicpm-sala-mlx/src/model.rs:341-358 (via
    mlx-rs/src/module/module.rs:267-282)** — the non-quantized path uses
    `ModuleParametersExt::load_safetensors`, which silently drops file keys that don't match a
    param and leaves unmatched params at random init; a naming drift loads a garbage model with
    no error (the quantized path and the other two crates fail loudly via `get_weight`).
    Fix: after loading, verify all params were hit (or diff file keys vs param keys).

12. **[perf] qwen3-vl-mlx/src/lib.rs:323-329** — vision attention is manual
    `matmul -> softmax -> matmul`, materializing the full `[16, N, N]` score matrix per block
    (24 blocks, N=784); moxin's ViT (vision.rs:123) already uses
    `mlx_rs::fast::scaled_dot_product_attention`. Fix: switch to fused SDPA (also needed
    anyway once 2-D RoPE is added).

## LOW

13. **[dead code] qwen3-vl-mlx/src/lib.rs:697-722** — the first image-splice loop builds
    `result_embeds` and is then unconditionally discarded by `result_embeds.clear()` ("Will
    re-do via positional approach"); pure dead work plus a misleading comment. Fix: delete the
    first loop.

14. **[dead code] moxin-vlm-mlx/src/vision.rs:252,432-444** — `ViTEncoder.norm` is loaded but
    never applied (correct for Prismatic's pre-norm intermediate features, but it's a dead
    parameter); relatedly `ViTBlock::training_mode` (vision.rs:232-237) doesn't propagate to
    `ls1`/`ls2`. Fix: drop the field or document why it's retained.

15. **[bug / example] moxin-vlm-mlx/examples/generate.rs:139-142** — streaming print slices
    the re-decoded string at the previous byte length (`&text[prev_text_len..]`); when an
    earlier partial UTF-8 sequence is re-decoded differently the index can land mid-codepoint
    and panic. minicpm's `ThinkFilter` (minicpm-sala-mlx/src/lib.rs:82-88) has the same
    byte-length assumption. Fix: use `char_indices`-safe slicing or `get(prev..)`.

Not counted against the cap (noted): qwen3-vl `load_model` re-reads `config.json`
unconditionally at lib.rs:940 and errors if it's absent despite the default fallback at 928;
minicpm's `fused_gla_decode` is intentionally retained dead code (documented,
metal_kernels.rs:148-152); speculative.rs:85-99 verifies with independent sampling, which is
only distribution-correct at temperature 0 (lightning state non-rollback is documented).

## Test suites

**qwen3-vl-mlx** — the three unit tests (src/lib.rs:1352-1382) are vacuous: they assert that
struct literals contain the values just assigned and that `Config` is `Clone`. None of the
risky logic — image-token splice, chat-token construction, preprocessing shapes, merger
reshapes — is tested, so all four HIGH findings above would pass `cargo test`. At minimum,
`build_chat_tokens`/`prefill` splice tests with synthetic ids (including two images) and a
shape/golden-value test for `preprocess_image` are warranted.

**moxin-vlm-mlx** — no tests at all (no `#[cfg(test)]` modules, no `tests/`). The pos-embed
before/after-CLS branch (vision.rs:269-286), the qkv-split loader (vision.rs:529-551), and the
projector key fallbacks (projector.rs:71-101) are all branchy weight-format logic that would
benefit from small synthetic-weight tests.

**minicpm-sala-mlx** — one real test (tests/test_model.rs) covering config parsing and derived
scalars; reasonable as far as it goes, but nothing exercises the numerics: the custom Metal
kernels (`fused_intra_chunk_attn`, `fused_state_update`) have no parity tests against the
plain-MLX composition they replaced, chunked-vs-recurrent GLA equivalence is untested (which
would have caught HIGH finding 5 for non-multiple-of-64 lengths), and the sparse gather path
is untested. `examples/test_rope_batch.rs` is a manual harness, not CI coverage.
