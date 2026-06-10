# Code Review: flux-klein-mlx & zimage-mlx

Scope: all `src/` and `examples/` in both crates, read-only. Focus order: scheduler/CFG/VAE/RoPE/loader bugs → perf → tests → duplication/dead code. 15 findings.

Note: neither crate applies CFG (both models are distilled few-step models; `apply_cfg` in `sampler.rs` is unused — correct, not a bug). Both examples correctly `eval()` once per denoise step and pre-compute RoPE once outside the loop.

---

## HIGH

### H1. Debug reference-file hijack baked into the Z-Image example
- **Category:** bug / debug code  |  **Crate:** zimage-mlx
- **Where:** `zimage-mlx/examples/generate_zimage.rs:472-480` (text embeddings), `:641-659` (initial latents), `:578-626` (Step 8 parity probe with hard-coded Python sums)
- If stale `/tmp/ref_text_embed.bin` or `/tmp/ref_initial_latents.bin` files exist, the example silently replaces the real prompt embedding and the seeded noise with binary dumps (shape hard-coded to `[1, 512, 2560]` / 512×512), so the generated image ignores the user's prompt with only a log line as evidence.
- **Fix:** gate all `/tmp/ref_*.bin` loading behind an explicit `--parity` flag (or delete the parity harness from the example).

### H2. Manual attention instead of fused SDPA in all four DiT model files
- **Category:** perf
- **Where:** `flux-klein-mlx/src/klein_model.rs:474-483, 656-659`; `flux-klein-mlx/src/klein_quantized.rs:261-270, 413-417`; `zimage-mlx/src/zimage_model.rs:369-384`; `zimage-mlx/src/zimage_model_quantized.rs:167-181`
- Every transformer block does `matmul → divide → softmax_axis → matmul`, materializing the full attention matrix (klein single blocks: 24 heads × 1536 × 1536 f32 ≈ 226 MB per block × 20 blocks per step; Z-Image: 30 heads × ~1550² × 32 blocks) — while the same workspace's Qwen3 encoder (`qwen3_encoder.rs:203-219`) already uses `mlx_rs::fast::scaled_dot_product_attention`.
- **Fix:** replace the manual path with `fast::scaled_dot_product_attention(q, k, v, scale, None)` (no mask is ever passed in these blocks).

---

## MEDIUM

### M1. `final_norm` is RmsNorm but is documented (and expected) to be LayerNorm
- **Category:** bug (parity risk)  |  **Crate:** flux-klein-mlx
- **Where:** `flux-klein-mlx/src/klein_model.rs:720` (`pub final_norm: RmsNorm,  // LayerNorm before final modulation`), applied at `:851`; same in `klein_quantized.rs:479, 620, 806`
- The file consistently uses `LayerNorm(affine=false)` for pre-modulation norms "matching diffusers", but the final AdaLN-continuous norm is an RmsNorm whose weight is never present in the checkpoint (loader comment at `klein_quantized.rs:806`: "default-initialized — not present in diffusers weights"). RmsNorm skips mean subtraction, so the final projection input differs from a diffusers-style LayerNorm.
- **Fix:** make `final_norm` a `LayerNormBuilder::new(hidden).affine(false).eps(1e-6)` like the block norms (and validate output parity).

### M2. FLUX.2 VAE config reuses FLUX.1 scale/shift constants
- **Category:** bug (VAE scaling)  |  **Crate:** flux-klein-mlx
- **Where:** `flux-klein-mlx/src/autoencoder.rs:64-76` (`flux2()` sets `scale_factor: 0.3611, shift_factor: 0.1159` — the well-known FLUX.1 values), applied in `Decoder::forward` at `:376-378`
- The 32-channel FLUX.2 VAE gets the 16-channel FLUX.1 latent statistics with no source comment; if the FLUX.2 checkpoint's `vae/config.json` specifies different values (or per-channel stats), every decode is mis-scaled.
- **Fix:** read `scaling_factor`/`shift_factor` from the downloaded `vae/config.json` instead of hard-coding, or document the verified FLUX.2 values.

### M3. Silent key drops in `sanitize_klein_model_weights` + unvalidated `update_flattened`
- **Category:** bug (weight loader)  |  **Crate:** flux-klein-mlx
- **Where:** `flux-klein-mlx/src/weights.rs:474-597`; consumed at `examples/generate_klein.rs:286-303`
- The sanitizer is an if/else-if cascade that silently discards any checkpoint key it doesn't recognize (no warning, no count check), and `update_flattened` leaves unmatched model params at their random init — a single upstream key rename produces garbage images with only a "Loaded N weights" print.
- **Fix:** collect unmatched keys and warn/error, and assert the sanitized count equals the model's expected linear/norm parameter count.

### M4. Z-Image quantized loader silently skips missing `norm_q`/`norm_k` weights
- **Category:** bug (weight loader)  |  **Crate:** zimage-mlx
- **Where:** `zimage-mlx/src/zimage_model_quantized.rs:584-593`
- Every other tensor in `create_quantized_block` / `create_quantized_linear` panics with a named key when absent, but the attention QK-norm weights use `if let Some(w) = weights.get(..)` and silently fall back to default all-ones RmsNorm — a key-naming drift corrupts attention with no signal.
- **Fix:** use the same panic-on-missing pattern as `load_rms_norm` two lines above (`:573-581`).

### M5. Dead legacy architecture: `layers.rs` blocks + two sanitizers for a model that no longer exists
- **Category:** dead code / duplication  |  **Crate:** flux-klein-mlx
- **Where:** `flux-klein-mlx/src/layers.rs:46-848` (`Modulation`, `QKNorm`, `SelfAttention`, `DoubleStreamBlock`, `SingleStreamBlock`, `FinalLayer`, `MlpEmbedder`, `apply_rope`, `get_2d_rope_freqs` — only `timestep_embedding` is used by `klein_model.rs:39` / `klein_quantized.rs:37`); `weights.rs:53-125` (`sanitize_flux_weights`, incl. no-op self-replaces at `:86-89`) and `:284-466` (`sanitize_flux2_klein_weights`) map to key names (`img_in`, `txt_in`, `time_in`, `img_attn.qkv`, per-block `mod_layer`) that match nothing in `klein_model.rs`
- ~900 lines of an abandoned implementation, with `sanitize_flux2_klein_weights` still publicly exported from `lib.rs:44`, inviting misuse.
- **Fix:** delete the dead blocks/sanitizers; move `timestep_embedding` into `klein_model.rs`.

### M6. Schedule code duplicated verbatim between `sampler.rs` and the example
- **Category:** duplication / dead code  |  **Crate:** flux-klein-mlx
- **Where:** `examples/generate_klein.rs:560-602` duplicates `src/sampler.rs:255-300` (`compute_empirical_mu`, `generalized_time_snr_shift`, `official_schedule`); `FluxSampler` itself is used by no example
- Also latent bugs in the unused copy: `FluxSamplerConfig::dev()` uses a fixed `shift: 1.0` instead of the resolution-dependent shift, and `denoise_loop` (`sampler.rs:198-223`) never `eval()`s inside the loop, so any future caller builds an unbounded lazy graph.
- **Fix:** have the example call `flux_klein_mlx::sampler::official_schedule` and delete the copy; either fix or remove `FluxSampler`.

### M7. Z-Image image-position grid is a known-imperfect "blurry" variant
- **Category:** bug (patch RoPE)  |  **Crate:** zimage-mlx
- **Where:** `zimage-mlx/examples/generate_zimage.rs:558-561` — "Image positions: original version that produced blurry-but-recognizable images", grid `(1, h_tok, w_tok)` at start `(cap_len + 1, 0, 0)`; same layout copied into `examples/generate_zimage_quantized.rs:57-60`
- The comment is a self-admission that the positional encoding has not been verified against the reference (`MLX_z-image`); image quality issues will trace back here.
- **Fix:** dump and diff `x_pos`/`cap_pos` (and the resulting cos/sin) against the Python reference, then delete the apologetic comment.

---

## LOW

### L1. Stale stability docs + four dead `#[param]` norm layers in the klein model
- **Category:** dead code / misleading docs  |  **Crate:** flux-klein-mlx
- **Where:** `klein_model.rs:9-25` (module doc describes tanh gate clamping, 0.5× txt residual scaling, ±65504 clipping — none exist in the code), `:296-302` (`txt_post_attn_norm`/`txt_post_mlp_norm` never called), `:570-572` (`post_norm` never called), `:693-696` (`txt_norm` "critical for stability" — never called); unused helpers `clip_values`/`has_nan`/`get_range` at `:873-949` (and a second `get_range`/`has_nan_arr` copy in `layers.rs:871-897`)
- **Fix:** delete the unused params/helpers and rewrite the module doc to describe the actual forward pass.

### L2. Inverted `x_mask`/`cap_mask` semantics in Z-Image transformer
- **Category:** bug (latent footgun)  |  **Crate:** zimage-mlx
- **Where:** `zimage-mlx/src/zimage_model.rs:781-792`, `zimage_model_quantized.rs:411-421` — `ops::where(&mask, &pad_token, &x)` replaces tokens where mask is **true**
- Every existing caller passes `None`, but any caller using the conventional 1=valid mask (as the Qwen3 encoder in the same pipeline does) would zero out all real tokens.
- **Fix:** invert the predicate (or document that the mask means "is padding") — and note `cap_seq` is computed and unused in both files (`zimage_model.rs:773`, `zimage_model_quantized.rs:403`).

### L3. klein example up-casts the whole model bf16 → f32
- **Category:** perf  |  **Crate:** flux-klein-mlx
- **Where:** `examples/generate_klein.rs:237-243` (text encoder) and `:290-296` (transformer): `v.as_type::<f32>()` on every weight
- Doubles resident memory (~8 GB → ~16 GB for the transformer + encoder) and memory bandwidth per step versus running bf16, which MLX supports natively (the zimage quantized example correctly runs bf16).
- **Fix:** keep weights/activations in bf16; cast only the final image to f32/u8.

### L4. `create_img_ids` doc contradicts the code (4-axis RoPE layout)
- **Category:** docs / parity confusion  |  **Crate:** flux-klein-mlx
- **Where:** `examples/generate_klein.rs:512-534` — doc says dims are `(T, H1, H2, W)` but the code writes `(0, y, x, 0)`; text ids put the sequence index in dim 3 (`:543-556`)
- The implementation may match flux.c, but the comment describes a different layout, which is exactly the kind of thing that makes RoPE parity bugs invisible in review.
- **Fix:** correct the comment to `(T=0, H=y, W=x, L=0)` for images and `(0,0,0,L=s)` for text, citing the flux.c source.

### L5. Hard-coded `/tmp` model path and per-step host syncs in benchmark/example loops
- **Category:** debug code / perf  |  **Crate:** zimage-mlx
- **Where:** `examples/generate_zimage_quantized.rs:24` (`/tmp/zimage_mlx_test/Z-Image-Turbo-MLX/transformer` hard-coded, no CLI arg); `examples/generate_zimage.rs:719-721, 725` (per-step `min/max/sum` + `.item::<f32>()` forces extra reductions and GPU→CPU syncs inside the denoise loop, plus hard-coded "Python reference: -8182.2139" prints)
- **Fix:** take the model dir as an argument; move latent-stat printing behind a `--debug` flag.

### L6. `load_quantized_flux_klein` ignores any model config
- **Category:** bug (latent footgun)  |  **Crate:** flux-klein-mlx
- **Where:** `flux-klein-mlx/src/klein_quantized.rs:711` — `let params = FluxKleinParams::default();` with no params argument
- Works only for the exact 5/20-block klein-4B layout; any variant (or group_size/bits mismatch with the saved file from `quantize_and_save_flux_klein`, which also doesn't record its quantization metadata) panics deep in `create_quantized_linear` or silently mis-decodes.
- **Fix:** accept `FluxKleinParams` (and group_size/bits) as arguments, or persist them alongside the quantized safetensors.

---

## Test suite assessment

**flux-klein-mlx.** 15 in-module `#[test]`s, no `tests/` directory. All are shape/smoke tests: `sampler.rs` checks config fields, timestep endpoints, and one hand-computed Euler step (the only numeric assertion in the crate); `autoencoder.rs` checks ResNet/encoder output shapes with random weights; `layers.rs` tests only the *dead* `Modulation`/`QKNorm` code (M5); `weights.rs::test_sanitize_keys` exercises `sanitize_flux_weights` — also dead — while the sanitizer actually used in production (`sanitize_klein_model_weights`, M3) has zero coverage. Nothing tests RoPE numerics (`compute_rope_freqs`/`apply_rope`), the modulate/gate helpers, the final-layer scale/shift ordering, or quantized-vs-f32 forward agreement — i.e., every area where the real parity bugs (M1, M2, L4) would hide. Overly permissive overall: tests assert shapes, never values.

**zimage-mlx.** 5 `#[test]`s total, no `tests/` directory: config defaults, coordinate-grid shape, RoPE cos/sin *shape* (positions all zeros, so values are trivially cos=1/sin=0 and never checked), block creation `is_ok()`, and quantized-encoder creation `is_ok()`. No test covers `apply_rope_3axis` rotation values, either sanitizer (`sanitize_zimage_weights` with its regex patch-size rewrites is completely untested), the patchify/unpatchify round-trip in the examples, or the sigma schedule. The crate's de facto regression test is the Step-8 `/tmp` parity probe inside the example binary (H1), which is exactly where that logic should not live — porting it into `tests/` with checked-in fixture tensors would convert the crate's biggest liability into its missing test coverage.
