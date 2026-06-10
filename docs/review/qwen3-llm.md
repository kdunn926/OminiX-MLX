# Code review: qwen3-mlx, qwen3.5-35B-mlx, qwen3.6-mlx

Date: 2026-06-10. Scope: src/ + examples of the three LLM crates (read-only; no cargo runs).
Priority order: correctness (GDN state, rollback, RoPE/KV offsets, MoE, loaders) > perf > tests > duplication > dead code.

## HIGH

### H1. Prefix-cache load double-processes the last cached token (KV + GDN desync)
- **Category:** bug — KV-offset / GDN staleness. **File:** `qwen3.6-mlx/src/cache.rs:164-211` (`HybridCache::try_load_hybrid_caches`)
- When the cached token sequence exactly equals the new prompt, `n` is capped to `prompt_tokens.len() - 1` (line 189) but the loaded caches are returned **untrimmed**: KV layers still hold all `len` positions and recurrent layers' saved state already includes the final token. The caller then re-prefills `prompt[n..]`, so the last token enters the KV cache twice (at RoPE offsets `len-1` and `len`) and flows through the GDN recurrence a second time — silent corruption on identical-prompt regeneration (a common API pattern; qwen3.6 is the production model). Compare `mlx-rs-core/src/cache.rs:468-472`, where `try_load_kv_caches` explicitly trims `cache_offset - n`.
- **Fix:** trim KV/Quantized layers by `offset() - n` after load, and return `Ok(None)` when any `Recurrent` layer's `step > n` (recurrent state cannot be rolled back without a tape).

### H2. qwen3.5 `conv1d_prefill` drops existing conv state and stores a malformed window for short prompts
- **Category:** bug — recurrent state staleness. **File:** `qwen3.5-35B-mlx/src/deltanet.rs:304-335`
- Unlike the qwen3.6 version (`qwen3.6-mlx/src/deltanet.rs:693-702`), prefill always zero-pads the left context (`zeros` at line 314, ignoring `cache.conv_state`) — any second multi-token forward on a warm cache (multi-turn continuation through the public `Model::forward`) silently zeroes the conv window of the prior context. Additionally `cache.conv_state = qkv_cf[..., -(k-1)..]` (line 321) clamps to `L` elements when `L < kernel_size-1`, leaving a too-short window that breaks the `concat + multiply` in `conv1d_step` on the first decode after a 2-token prompt. In-crate `Generate` (fresh cache, single prefill, `L=1` decode) doesn't hit it, but the public API does.
- **Fix:** backport the qwen3.6 logic: use `cache.conv_state.take()` as the left pad and save `padded[..., -(k-1)..]`.

## MED

### M1. Per-token env-var reads in the qwen3.6 decode loop
- **Category:** perf. **File:** `qwen3.6-mlx/src/lib.rs:294, 311-314`
- Every `Generate::next` decode step calls `std::env::var("QWEN36_PROFILE_DECODE")` and parses `QWEN36_CACHE_CLEAR_INTERVAL` — two lookups per generated token (the crate's own `verify_hook.rs:70-81` deliberately caches its env flag in a `OnceLock` for exactly this reason).
- **Fix:** read both once in `Generate::new*` (struct fields) or via `OnceLock`.

### M2. qwen3.5 prefill materializes full `[B, T, vocab]` logits
- **Category:** perf. **Files:** `qwen3.5-35B-mlx/src/lib.rs:88`, `qwen3.5-35B-mlx/src/model.rs:135-145`
- The prefill arm calls `model.forward(prompt, ...)`, which applies the LM head to **every** prompt position; with vocab 248,320 that's ~1 GB of bf16 activations for a 2k-token prompt, plus the wasted matmul. There is also no chunked prefill. qwen3-mlx (`forward_last_logits`) and qwen3.6 (chunked `forward_last_logits`) both avoid this.
- **Fix:** add a `forward_last_logits` that slices the hidden state to the last position before the LM head.

### M3. Lazy cache init ignores the configured `KVCacheMode`
- **Category:** bug (footgun, silent divergence). **File:** `qwen3.6-mlx/src/model.rs:352-360, 388-395, 443-451, 544-552`
- Four forward paths lazily populate an empty cache vector with hard-coded `HybridCache::KV(KVCache::new())`. A caller that builds caches via `new_cache(KVCacheMode::Quantized/Paged/TurboQuant)` is fine, but anyone passing a fresh `Vec` to `forward_verify_with_snapshots` / `forward_from_embeds` / `forward_pre_norm_hidden` silently gets the standard fp16 cache regardless of intended mode — and the logic is copy-pasted four times.
- **Fix:** factor a single `ensure_cache(&mut Vec<HybridCache>)` (or take the mode as a parameter / error on empty cache).

### M4. Vision tower position embeddings indexed sequentially, not spatially
- **Category:** bug — VL parity. **File:** `qwen3.6-mlx/src/vision.rs:183-187` (with `config.rs:212-214`)
- The pos-embed table has `num_position_embeddings = 2304` (a 48×48 grid), but `forward` looks up raw ids `0..h_patches*w_patches` (784 for a 448×448 image, 28 patches wide). Raster position `k` in a 48-wide trained grid does not correspond to raster position `k` in the 28-wide image grid, so rows beyond the first get spatially wrong embeddings; the reference implementation 2-D-interpolates the 48×48 table down to the actual patch grid.
- **Fix:** build ids by mapping each `(row, col)` of the image grid into the 48×48 table (or bilinear-interpolate the table) before lookup.

### M5. Deepstack merger outputs summed into the final visual features
- **Category:** bug — VL parity. **File:** `qwen3.6-mlx/src/vision.rs:205-210`
- The three deepstack merger outputs are simply added onto the final merged features before splicing into the prompt. Qwen3-VL's deepstack design injects each deepstack tensor into a *different early decoder layer's* hidden state at the image-token positions; collapsing them into the input embedding changes where the information enters the residual stream. (`deepstack_visual_indexes` selects the ViT capture layers, but nothing consumes them per-decoder-layer on the text side.)
- **Fix:** plumb `deepstack_outputs` into the text forward and add each at its configured decoder layer, or document the approximation.

### M6. Multi-image prompts silently mis-embedded
- **Category:** bug. **File:** `qwen3.6-mlx/src/model.rs:1000-1043` (`VlModel::prefill_multimodal`)
- Only the **first** contiguous run of `image_token_id` is replaced with visual features; any further image-token block lands in `post_ids` and is embedded as raw placeholder-token embeddings with no error. One image works; two images silently corrupt the prompt.
- **Fix:** loop over all placeholder blocks (erroring if block count != provided feature sets), or reject prompts containing a second block.

### M7. qwen3.5 GDN q/k normalization uses the drift-prone l2-normalize variant
- **Category:** bug — numerical parity. **File:** `qwen3.5-35B-mlx/src/deltanet.rs:53-58, 96-102, 170-176`
- `l2_normalize` adds `eps` to the **sum** of squares (not the mean), and the scale layout (`q *= d^-1/2`, `k` unscaled) is the formulation qwen3.6's deltanet explicitly replaced because eps placement "differed by N, causing ~3.9% relmax drift on k_norm at small magnitudes" (`qwen3.6-mlx/src/deltanet.rs:228-233`). The fix (`mx.fast.rms_norm(x, None, eps)` + `inv_scale²`/`inv_scale`) was never backported.
- **Fix:** port `rms_norm_no_weight` + the qwen3.6 scaling into qwen3.5's `forward_step`/`forward_prefill`.

### M8. Integration tests silently pass when models are absent; tautological assert
- **Category:** overly-permissive tests. **File:** `qwen3.6-mlx/tests/model_load.rs:15-96`
- All four tests `return` (green) when the model dir/config is missing, so CI without weights reports success with zero coverage; there is no way to distinguish "passed" from "skipped". Also `assert!(model.lm_head.is_some() || !model.args.tie_word_embeddings)` (line 71) is satisfied by every model the loader can construct — it asserts nothing.
- **Fix:** mark weight-dependent tests `#[ignore]` (run explicitly) and assert a real post-load invariant (e.g. layer count and per-layer attention kind match `layer_types`).

## LOW

### L1. GDN rollback unit test uses all-zero tensors
- **Category:** overly-permissive test. **File:** `qwen3.6-mlx/src/cache.rs:352-389`
- With zero state, tape, k, decay, and conv input, `trim_gdn`'s replay and conv-window reconstruction are trivially zero — only the `step` bookkeeping is exercised. A wrong `replay_prefix` or `rolled_conv_state` would still pass.
- **Fix:** seed random state/tape, roll forward N steps manually, and compare the replayed state numerically.

### L2. Dead padding knobs and dead helper in qwen3.6 deltanet
- **Category:** dead code. **File:** `qwen3.6-mlx/src/deltanet.rs:28-30, 140-145, 170-190`
- `EXACT_SMALL_PROJ_{AB,QKV,Z}_PAD_M` are all 0, so `exact_small_proj_with_pad_m`'s pad branch is unreachable yet still wraps every projection call (shape probe per call); `l2_normalize` survives only under `#[allow(dead_code)]`.
- **Fix:** delete the constants/helper and call `verify_hook::quantized_linear_forward` directly.

### L3. `compute_decay` / `compute_decay_batched` are byte-identical
- **Category:** dead code / duplication. **Files:** `qwen3.6-mlx/src/deltanet.rs:789-814`, `qwen3.5-35B-mlx/src/deltanet.rs:401-416`
- Both bodies are the same broadcasting expression; the "batched" copy exists only for the doc comment — in both crates.
- **Fix:** keep one function.

### L4. Heavy cross-crate duplication
- **Category:** duplication. **Files:** `qwen3.5-35B-mlx/src/{attention,deltanet,cache}.rs` vs `qwen3.6-mlx/src/{attention,deltanet,cache}.rs`; `qwen3-mlx/src/qwen3_moe.rs:322-477` vs `qwen3.6-mlx/src/moe.rs:26-168`; `make_quantized_linear`/`make_quantized_embedding` copied in 4 files; the `Generate` iterator copied in 3.
- qwen3.5's GDN/attention is an older snapshot of qwen3.6's (H2 and M7 are exactly the divergences that this duplication let rot). `SwitchGLU`/`gather_sort`/`scatter_unsort`/`QuantizedSwitchLinear` are line-for-line duplicates.
- **Fix:** move SwitchGLU + quantized-loader helpers into `mlx-rs-core`; consider retiring qwen3.5-35B-mlx in favor of qwen3.6 (same architecture family) or re-exporting from it.

### L5. qwen3-mlx MoE keeps the reverse-order top-k that qwen3.6 documents as drift
- **Category:** bug — numerical parity (minor). **File:** `qwen3-mlx/src/qwen3_moe.rs:503-506`
- `argpartition(−gates, k−1)` + leading-k yields the right expert set but reversed within-top-k order; `qwen3.6-mlx/src/moe.rs:251-265` documents that this exact pattern produced ~1% relmax drift per MoE layer vs Python because the BF16 weighted sum is order-dependent.
- **Fix:** mirror Python's `argpartition(gates, kth=N−k)[..., −k:]` as qwen3.6 does.

## Test suites

**qwen3-mlx** has no tests at all — zero `#[test]` across `model.rs`, `qwen2.rs`, `qwen3_moe.rs`. Coverage is entirely manual via the three examples (`generate_qwen3`, `chat_qwen3`, `bench`), which need local model weights. Even cheap weight-free units (config parsing, `is_moe_layer` boundary cases, `gather_sort`/`scatter_unsort` round-trip) are untested.

**qwen3.5-35B-mlx** also has no tests. Given H2 and M7 live here, and the crate duplicates an older copy of qwen3.6 logic, the lack of even a conv1d-prefill-vs-step consistency test means regressions in the recurrence are invisible. The single example (`generate.rs`) requires weights.

**qwen3.6-mlx** is the best-covered of the three but still thin relative to its production role: `config.rs` tests are solid (MoE detection, quant precedence); `mtp.rs` has a genuinely good synthetic end-to-end forward plus an `#[ignore]`d real-checkpoint test; but the only rollback test uses all-zero tensors (L1) and the four integration tests silently skip without weights (M8). Nothing tests the GDN tape-replay math, `try_load_hybrid_caches` round-trips (H1 would have been caught by a save→load→exact-prompt test), chunked-vs-single prefill equivalence, or the verify/trim_gdn path end-to-end; `examples/deltanet_consistency.rs` covers some of this but only as a manual harness, and the five `target_*` debug examples are leftover repro tooling rather than tests.
