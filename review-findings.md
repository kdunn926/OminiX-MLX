# Workspace code review — task list (2026-06-10)

Full-workspace review for bugs, performance, overly-permissive tests, bloat, and
non-idiomatic Rust. Constraints: public APIs and observable behavior stay stable;
changes left uncommitted; verify with `cargo build/test/clippy` plus examples where
local models exist.

Severity: **HIGH** = correctness bug or major perf hit · **MED** = real but bounded ·
**LOW** = tidiness. Each item is checked off when addressed (fixed, or explicitly
documented as a known limitation when a real fix is out of scope for "tidy, keep APIs").

Findings below were produced by parallel review agents over wave 1
(gemma4-mlx, mlx-rs-core, mlx-rs fork, gpt-sovits-mlx, step-audio2-mlx,
qwen-image-mlx, dflash-mlx, mtplx-mlx, xtask, ane-vit/coreml-bridge) and verified
against the code before fixing. Crates in the "pending inline review" section at the
bottom are reviewed directly as part of working the list.

---

## P0 — current spike branch (gemma4 sliding cache / async vision prefill)

- [ ] **HIGH bug (2026-06-11)** `gemma4-mlx/examples/chat_gemma4_ud.rs` — the flat
  `Vec<KVCache>` path degenerates on the 26B-A4B UD checkpoint (instant repetition
  loops: "enough to stability." / "..."), under EVERY expert activation variant —
  so it is NOT the GeGLU experts fix. The same checkpoint + loader through
  `chat_text` (init_layered_cache: SlidingKVCache for sliding slots) answers
  perfectly at ~25 tok/s with proper EOS. Suspect: sliding-window masking vs an
  unbounded flat KVCache in `Attention::forward` (mask offset / window math), or
  the prompt-cache plumbing. The example's own header notes it stays on
  Vec<KVCache> only because the prompt-cache API is typed to KVCache — either fix
  the flat path or port the example to layered caches + extend the save/load API.
- [x] **HIGH bug** `mlx-rs-core/src/cache.rs:111-138` — blanket `impl KeyValueCache for &mut T`
  forwards only 6 of 10 methods; `trim_kv`, `compact_kv`, `try_fused_attention`,
  `compact_to_last_n`, `physical_offset` fall through to trait defaults: sliding
  compaction silently no-ops (unbounded KV growth), spec-decode rollback errors,
  fused attention silently disabled through `&mut C`. Forward all methods.
- [x] **HIGH bug** `gemma4-mlx/src/mixed_cache.rs:213` + `model.rs:601-605,2462-2467` —
  `init_layered_cache` assigns `SlidingKVCache` to KV-store slots on shared-KV models
  (e4b): shared snapshot taken post-compaction and reader RoPE offset derived from
  physical length, so once logical pos > window, readers attend with garbage offsets.
  Force `kv_store_layers` slots to `MixedKvCache::Kv` (or carry logical offset).
- [x] **HIGH bug** `gemma4-mlx/src/model.rs:4080-4083` + `unified_vision.rs:453-456` —
  both `new_cache_paged` impls build all-default (`Kv`) caches; zero paged slots, so
  `OMINIX_PAGED_ATTENTION=1` VL paths silently run unpaged. Use `init_mixed_paged_cache`.
- [x] **MED bug** `gemma4-mlx/src/chat.rs:987-1009` — `extract_bare_call_payloads`
  matches `call:` without word boundary and splices "call:" out of prose on
  non-matches (mutates user-visible content; "Recall: budget {2024}" → spurious tool
  call). Require word boundary; advance cursor instead of splicing on invalid names.
- [x] **MED bug** `gemma4-mlx/src/model.rs:4169-4175` + `unified_vision.rs:335-341` —
  `src_row = out_idx % vis_len` silently wraps on multi-image token-count mismatch
  (image 2 reuses image 1 features). Error on mismatch like the async path does.
- [x] **MED bug** `gemma4-mlx/src/lib.rs:120` — VL chat tokens still emit role
  `assistant`; Gemma4 expects `model` (text path already fixed in chat.rs). Emit
  `model` and map incoming `assistant` roles.
- [x] **MED perf** `gemma4-mlx/src/vision.rs:107-174,271` — 2-D RoPE called twice per
  encoder layer; each call does `eval(positions)` (GPU sync) + rebuilds cos/sin tables
  on CPU (~54 syncs/table builds per image). Precompute once per forward.
- [x] **MED perf** `gemma4-mlx/src/vision.rs:332-336` — debug-era `eval` every 5 ViT
  layers defeats the async-prefill overlap. Remove or env-gate.
- [x] **MED perf** `gemma4-mlx/src/unified_vision.rs:604-608` — `load_unified_4bit_vl`
  reads the multi-GB checkpoint twice. Load raw map once, feed both loaders.
- [x] **MED bloat** `gemma4-mlx/src/unified_vision.rs:283-386` — ~100-line near-copy of
  `Gemma4VlModel::prefill_multimodal` (model.rs:4127-4229), already drifting. Extract
  shared helper.
- [x] **MED test** `gemma4-mlx/src/sliding_cache.rs:207-336` — all 8 tests use zero
  tensors and assert lengths only; wrong-end compaction would pass. Stamp positions
  into values and assert tail contents.
- [x] **MED test** `gemma4-mlx/src/chat.rs:1616-1674` — real-template smoke test
  silently no-ops without models and hardcodes `/Users/kyle/...`. Env-var the model
  root; `#[ignore]` when absent.
- [x] **LOW perf** `gemma4-mlx/src/model.rs:4606-4607` — dead `async_eval` immediately
  followed by `eval`.
- [x] **LOW perf** `gemma4-mlx/src/model.rs:4620` — `GEMMA4_PROFILE_DECODE` (+ cache
  clear interval) env looked up every decode step; read once.
- [x] **LOW bloat** `gemma4-mlx/src/mtplx_target.rs:25-54` duplicates
  `ud_loader.rs:38-71`; `chat.rs:1366` unused `_template` param; `let _ = NewAxis;`
  hacks (assistant.rs:190, sliding_cache.rs:328).
- [x] **LOW bug(doc)** `gemma4-mlx/examples/sliding_bench.rs:87-96` — comment claims
  `model.forward` chunks prefill via `GEMMA4_PREFILL_CHUNK`; it doesn't. Fix comment
  (or chunk explicitly in the bench).
- [x] **LOW idiom** `gemma4-mlx/src/chat.rs:476` — `eprintln!` in render fallback path.
  *Disposition: kept — single warning on a degraded-mode fallback; the crate has no
  logging facade and adding one is out of scope.*
- [ ] **MED perf (deferred)** `gemma4-mlx/src/chat.rs` tool loop — each tool
  iteration re-renders the conversation and re-prefills from an empty cache
  (O(n²) prefill across iterations). Real fix is prefix-cache reuse across
  iterations — a feature change beyond tidy scope; documented here as known cost.

## P1 — mlx-rs-core (shared hot path)

- [x] **HIGH bug** `metal_kernels.rs:2932-2982` (`fused_modulate`) — mean/var reduction
  accumulates in template type `T`: bf16 loses mantissa, f16 sum-of-squares can
  overflow → NaN; also uses cancellation-prone `E[x²]−mean²`. Accumulate in `float`.
- [x] **HIGH bug** `cache.rs:1080-1084` — TurboQuant fused path A requires
  `n_bulk >= 8192 && n_bulk <= TQ_SDPA_MAX_KV(=2048)`: unsatisfiable under defaults,
  dead kernel path + stale comment. Give path A a sane threshold or delete it.
- [x] **MED perf** `cache.rs:1056` — `TurboQuantKVCache::try_fused_attention` appends
  via `update_and_fetch`, building (and discarding) full-cache reconstruct graphs
  every token. Factor an append-only method (cf. `QuantizedKVCache::append`).
- [x] **MED perf** `cache.rs:1670-1729` — QuantizedKVCache GQA expand materializes
  `kv_repeat` copies of the entire quantized store per step (broadcast+reshape forces
  copy). Reshape Q and let `quantized_matmul` broadcast instead.
- [x] **MED perf** `generate/mod.rs:243-244` — decode loop blocks on `try_item` with no
  async_eval pipelining (the known ~27% gap; see memory note). Document/track; full
  pipelining is the gemma4-migration work — out of scope to land here, but leave a
  doc comment.
- [x] **MED test** `cache.rs:1847-1864` — test claims numeric parity, asserts shapes
  only. Compare values.
- [x] **MED bug** `metal_kernels.rs:3379-3403` (`fused_swiglu`) — no shape/dtype
  validation of `x` vs `gate`; mismatch is a silent OOB GPU read. Validate.
- [x] **MED test** `metal_kernels.rs` — no parity tests for `fused_swiglu`,
  `fused_modulate`, `per_position_rope`, `kv_compact`. Add reference-comparison tests
  (bf16 included — would have caught the fused_modulate bug).
- [x] **LOW bug** `cache.rs:282-287` — `KVCache::compact` identity fast-path trusts
  caller. Document precondition + debug_assert.
- [x] **LOW perf** `cache.rs:1076-1079,1109-1111,1133-1135` — TURBOQUANT_* env vars
  read/parsed every step per layer. Cache in OnceLock/fields.
- [x] **LOW perf** `turboquant.rs:68-78` + cache.rs call sites — `cached_signs` clones
  host Vec from mutex and rebuilds device Arrays every update. Cache the device Array.
- [x] **LOW bloat** `lib.rs:15` — docs advertise nonexistent `speculative` module.
- [x] **LOW bloat** `utils.rs:191-209` — unused `_cache` generic param on
  `scaled_dot_product_attention` (forces turbofish); `SdpaMask` duplicates
  `AttentionMask`. (API-affecting — only tidy what doesn't break callers.)
  *Disposition: documented — removing the param or unifying the enums breaks
  every model crate's call sites; the param is now doc'd as compat-only.*
- [x] **LOW bloat** `metal_kernels.rs` — ~10 kernel wrappers repeat ~100 lines of FFI
  boilerplate each; `let _ = scale;` (1676), `let _ = d;` (cache.rs:1731), dead
  `offset == 0` check (cache.rs:1662-1664). Factor a shared dispatch helper where it
  doesn't churn the API.
  *Disposition: debris removed (`let _ = scale`, `let _ = d` now used by the GQA-fold
  rewrite; the `offset == 0` guard kept — it's a legitimate empty-append check). The
  generic dispatch-helper refactor is deferred: ~10 hand-tuned unsafe wrappers with
  per-kernel grid math; consolidating them wholesale risks subtle dispatch bugs for
  a tidiness win — better done incrementally as kernels are next touched.*
- [x] **LOW doc** `utils.rs:169-176` — `create_attention_mask` derives offset/window
  from `cache.first()` only; wrong for mixed per-layer windows. Document.

## P1 — mlx-rs (vendored fork; conservative fixes only)

- [x] **HIGH bug (LOCAL)** `mlx-rs/src/array/mod.rs:334-362` — `raw_data_bytes()` is a
  safe fn that can segfault on unevaluated arrays and ignores strides. Eval first +
  contiguity check (or mark unsafe).
- [x] **MED perf (UPSTREAM)** `mlx-rs/src/transforms/compile/compile.rs:24-29` —
  returned closure re-compiles + erases the compile cache every call. Hoist
  `f.compile()` out of the closure.
- [x] **MED bug (LOCAL)** `mlx-rs/src/fast.rs:95-102` — `Mask::Arrays` silently uses
  only `masks[0]`. Error on len > 1.
- [x] **MED bug (LOCAL)** `mlx-rs/src/ops/quantization.rs:64-74` — real MLX error
  message dropped (+ leaked) on `mlx_quantize` failure; vector-get statuses ignored.
  Route through standard error retrieval.
- [x] **MED bug (UPSTREAM)** `mlx-rs/mlx-lm-utils/src/tokenizer.rs:518-528` —
  `final_msg_len` from untrimmed string can slice past end / split UTF-8 → panic. Use
  `.get(range)` with fallback.
- [x] **MED bug (UPSTREAM)** `mlx-rs/src/array/mod.rs:575-603` — `deep_clone()` same
  unevaluated-read class + wrong order for strided views. Eval + contiguity guard.
- [x] **MED bug (UPSTREAM)** `compile_with_state.rs:433-442` — positional zip of state
  write-back is wrong if compiler prunes a middle state array. Assert equal lengths.
- [x] **LOW bug** `array/mod.rs:542-544` — `try_as_slice` errors on legitimately empty
  arrays.
- [x] **LOW doc** `array/mod.rs:90-96` — `from_ptr` safety doc references removed
  `mlx_retain`.
- [x] **LOW idiom (LOCAL)** `array/mod.rs:368-370` — `metal_buffer_ptr` blind u32 cast,
  misleading name. Document/validate.
- [x] **LOW bloat (LOCAL)** `transforms/compile_dyn.rs:33-57` — `CompiledFn` never
  compiles; name oversells (used by mtplx). Fix docs.
- [x] **LOW bloat (UPSTREAM)** `mlx-lm-utils/src/tokenizer.rs` — ~230 lines of pasted
  Python comments; fn typo `render_jinja_tempalte` (pub API — alias rather than break).
- [x] **LOW bloat (UPSTREAM)** `compile.rs:353-357` — no-op `take(result_len)`.
- [x] **LOW bug (UPSTREAM)** `compile/mod.rs:184-204` — `Clone` + `Drop`-erases-cache
  interaction forces recompiles. Document/guard.
- [x] **LOW test (UPSTREAM)** `fast.rs:301-331` — SDPA test asserts shape/dtype only.
  Add numeric check.

### Discovered while working mlx-rs (pre-existing, fixed or documented)

- [x] **HIGH test** `cargo test -p mlx-rs` did not even compile: quantize/dequantize
  tests called pre-`mode`-parameter signatures (ops/quantization.rs:275,280). Fixed —
  this had been masking every other mlx-rs lib-test failure below.
- [x] **MED test** `mlx-lm-utils` fixture `tests/fixtures/qwen3/tokenizer_config.json`
  was never committed; two tokenizer tests failed on any fresh checkout. Fixture now
  bundled (copied from the local Qwen3-4B checkpoint, 9.7 KB).
- [x] **MED test** fft tests + `test_full_array` read multi-axis FFT / broadcast
  outputs via `as_slice` — those are legitimately non-row-contiguous, so the fork's
  (correct) contiguity check in `try_as_slice` rejects them. Tests now materialize a
  `.contiguous()` copy first.
- [ ] **HIGH bug (pre-existing, documented)** `compile`/`compile_with_state` and parts
  of the optimizer test-suite fail at runtime with
  `"There is no Stream(gpu, N) in current thread"` thrown from mlx core during eval
  of trampoline-traced graphs (mlx-tests test_compile_with_state: 11/12 fail;
  test_optimizers: 2 fail single-threaded, 12 multi-threaded;
  mlx-rs nn::value_and_grad/nn::container: 6 fail). Verified identical on the
  pristine tree — NOT introduced by this review's changes; likely fallout from an
  mlx/mlx-c version bump that made default streams thread-local. No model crate uses
  `compile` or non-default streams at runtime (inference is unaffected; mtplx's
  `CompiledFn` deliberately does not compile). Needs a dedicated fork investigation.

## P2 — dflash-mlx / mtplx-mlx / xtask / coreml-bridge

- [x] **HIGH bug** `ane-vit/coreml-bridge/src/ane_runner.swift:175-177` —
  `coreml_predict` copies `pixelCount` floats into a model-shaped MLMultiArray
  unvalidated: heap overflow on oversized input. Validate `pixelCount == shape
  product`.
- [x] **HIGH bug** `ane_runner.swift:96-112` — zero-copy variant ignores `pixelCount`:
  OOB read of Rust memory on undersized buffer. Validate.
- [x] **HIGH bug** `dflash-mlx/src/engine/gqa_sdpa.rs:52-67` — GQA fold is g-major but
  mask grown with `repeat_axis` (np.repeat semantics): mask rows misaligned whenever
  `gqa > 1 && q_len > 1` (27B draft SWA path). Tile instead of repeat.
- [x] **HIGH bug** `xtask/src/main.rs:8,55,455,506` — submodule path `mlx-sys/src/mlx-c`
  should be `mlx-rs/mlx-sys/src/mlx-c`; tool panics on startup.
- [x] **MED bug** `ane_runner.swift:18-22` — compute-units code 4 returns `.all` on
  macOS 14+, contradicting docs ("cpuAndNeuralEngine"). Fix branch.
- [x] **MED perf** `ane_runner.swift:53-55` — recompiles `.mlpackage` on every load,
  leaks temp `.mlmodelc`. Cache/delete.
- [x] **MED bug** `dflash-mlx/src/model.rs:372-434` — uncached draft forward never
  applies the SWA mask (cached path does); parity harnesses diverge from runtime. Pass
  the mask.
- [x] **MED bug** `dflash-mlx/src/engine/gemma4_adapter.rs:110-122` —
  `DFLASH_MAX_HIDDEN_SEGS` cap drops oldest segments, silently desyncing positions in
  the projected-context cache. Rebase positions or reset caches on drop; document.
- [ ] **MED perf (deferred)** `dflash-mlx/src/engine/acceptance.rs:40-63` +
  `mtplx-mlx/src/acceptance.rs:136-141` — T>0 acceptance copies full `[K+1, vocab]`
  fp32 probs to host every cycle (~10 MB). Gather per-draft-token probs on device.
- [x] **MED bloat** dflash-mlx dead modules: `cache/{prefix_l1,prefix_l2,snapshot}.rs`,
  `runtime/{bundle,registry}.rs` + `load_runtime_bundle` (empty `MODEL_SUPPORT_SPECS`),
  `rollback.rs` `RecurrentRollbackCache`; unused deps `safetensors`, `memmap2`
  (+ `fnv`/`lru` only serve dead code). Delete.
- [x] **LOW test** `dflash-mlx/src/engine/copyspec.rs:228-243` — FNV test computes its
  own reference with the same loop. Hardcode literal.
- [x] **LOW bloat** `spec_epoch.rs:303-306` empty else; `:698` `let _ = from_copyspec;`;
  `:346/360` drafted tokens eval'd+copied twice per cycle.
- [ ] **LOW bug** `dflash-mlx/src/model.rs:57-73` — `block_size()`/`mask_token_id()`
  `.expect()` panic on configs without dflash keys. Return Result/Option.
- [x] **LOW bug** `spec_epoch.rs:631-634` — ddtree path emits at most max_tokens-1 (no
  1-token fallback) and never updates `total_tokens`.
- [x] **LOW idiom** `gemma4_adapter.rs:478` — real error collapsed via `.ok().flatten()`.
- [x] **LOW bloat** `mtplx-mlx/src/graph_bank.rs` observation-only (hits impossible);
  session.rs:146/444 cache-mode env selection copy-pasted; Cargo.toml `mlx-sys` unused.
- [x] **LOW bug** `xtask/src/main.rs:11-52` — git invocations never check exit status;
  silent self-diff reports "No API changes". Check status.
- [x] **LOW bloat** `ane-vit/coreml-bridge/build.rs:84-90` — hardcoded CLT swift lib
  path breaks under Xcode.app toolchains. Derive via xcrun.

## P2 — step-audio2-mlx

- [x] **HIGH bug** `src/model.rs:913` — think mode prefills bare audio embeddings with
  no chat template (ASR path wraps correctly). Build the same prompt.
- [x] **HIGH bug** `src/model.rs:975-991` + `src/think.rs:205-228` —
  `decode_single_token` placeholder returns `"[id]"` strings so `<think>` detection
  never fires; Initial-state tokens dropped until 50-char buffer. Decode via tokenizer
  and fix buffering.
- [x] **HIGH perf** `src/llm.rs:433-464` — repetition penalty: eval + full-vocab
  GPU→CPU copy + dead clone + re-upload per token. Penalize on device or minimize
  transfer.
- [x] **HIGH bug** `src/model.rs:677,712` — TTS decoder loaded from hardcoded
  `./Step-Audio-2-mini` CWD path on every call. Store model dir; cache decoder.
- [x] **HIGH bug** `src/tts/s3tokenizer_mlx.rs:481-523` — FSQ quantize uses 2 of 8 dims
  ("for simplicity"): structurally wrong codes. Implement full FSQ or document as
  known-broken.
- [x] **MED bug** `src/pipeline.rs:598-620` — tool-call loop re-executes the same calls
  up to 5×; results never fed back. Strip processed spans.
- [ ] **MED perf** `src/model.rs:245,294` + `src/llm.rs:391` — every shard deserialized
  3× at load. Load once, dispatch by prefix.
- [x] **MED bug** loaders silently skip unmatched keys (model.rs:262,317,
  llm.rs:394-399). Count loaded vs expected; warn/fail.
- [x] **MED bug** `src/audio.rs:295-302` — symmetric Hann; reference is periodic.
  Fix denominator.
- [ ] **MED bloat** `src/pipeline.rs` — `SamplingConfig` dead (values hardcoded
  elsewhere), `synthesize`/`chat_text` stubs always Err, Conversation history unused.
  Wire or trim facade.
- [x] **MED bug** `src/model.rs:194` + `src/llm.rs:294-297` — `tie_word_embeddings`
  parsed, never honored. Honor it.
- [ ] **MED bug** `src/tts/flow.rs:166-207` + `hifigan.rs:122-124` — `w()` panics on
  missing weight mid-synthesis; `validate_weights` covers a subset. Validate full set
  or return Result.
- [x] **MED doc** `src/tts/hifigan.rs:1-7,240-260` — header claims HiFT source
  modeling; implementation is a tanh-mean approximation. Document honestly (real fix
  out of scope).
- [x] **LOW bloat** `src/tts/s3tokenizer.rs` superseded; `flow.rs:540` empty Codebook
  stub; `llm.rs:404-416` no-op replaces; mock `WebSearchTool` registered by default
  (tools.rs:203-217,426-431) — make opt-in.
- [x] **LOW perf/bug** `src/audio.rs:373-421` — naive O(n²) CPU DFT fallback; silent
  single-zero-frame on GPU error. Propagate error.

## P2 — qwen-image-mlx

- [x] **HIGH bug** `src/pipeline.rs:144-156,202-227` — pipeline conditions transformer
  on unshifted linear `t` while stepping with shifted sigmas (example does it right).
  Feed `sigmas[idx]`.
- [x] **HIGH bloat** `src/qwen_quantized_debug.rs` (990 lines) — orphaned module, never
  compiled. Delete.
- [x] **HIGH bloat/perf** `src/qwen_quantized.rs` — ~44 debug print/eval sites and nine
  `if false` blocks in the production transformer (forced syncs during inference).
  Strip/gate.
- [x] **MED bug** loaders silently skip unmatched keys (`text_encoder.rs:426-495`,
  `vae/weights.rs:33-52`; zero-init at text_encoder.rs:81). Count + warn/fail.
- [x] **MED bloat** three parallel transformer impls; unquantized path unused by any
  example and carries the scheduler bug. Consolidate/remove unused path (keep pub API).
- [x] **MED bug** `src/transformer/transformer.rs:92-95` — patch_embed built for 256
  inputs vs patchify's 64; works only because weights overwrite shape. Fix dims.
- [x] **MED perf** `examples/generate_qwen_image.rs:780-842` — no eval per denoise
  step; lazy graph for 100 forwards accumulates. Eval per step.
- [x] **MED perf** `src/qwen_quantized.rs:300-319`, `text_encoder.rs:243-261` — manual
  attention materializing huge score tensors. Use fused SDPA.
- [x] **MED bug** `src/pipeline.rs:145,203` — `Array::from_slice(&[t], &[batch])` OOB
  for batch > 1. Use `Array::full`.
- [x] **LOW bloat** `pipeline.rs:236-246` dead `build_attention_mask` (wrong comment);
  `qwen_quantized.rs:918-936` dead `clip_values` + contradictory comments.

## P2 — gpt-sovits-mlx

- [x] **HIGH bug** `src/sampling.rs:192-201` — top-p removes the first token crossing
  the threshold: top token prob > top_p ⇒ all -inf ⇒ NaN ⇒ silent token 0. Keep first
  sorted index.
- [x] **HIGH bug** `src/audio/stft_gpu.rs:39-46` — GPU mel pads 1024 zeros vs Python's
  704 reflect; frame count and edge content wrong vs reference. Match
  `spectrogram_torch`.
- [x] **HIGH test** `src/models/t2s.rs:1186-1190` — forward test asserts seq len 11 but
  model adds BERT (len 6); cannot pass. Fix expectation (proves tests aren't run).
- [x] **HIGH perf** `src/audio/mel.rs:188-211` — O(N²) CPU DFT recomputing Hann per
  frame in the training loss; dead `audio_1d`/`window`. Route through rfft; hoist.
- [x] **HIGH bloat** `src/models/sovits.rs` (902 lines) — orphaned, never compiled.
  Delete.
- [x] **MED bug** `src/inference.rs:135-149,207-242` — `generate_semantic_tokens`
  ignores its own top_k/top_p/rep-penalty config; EOS before min_tokens on first
  token. Use `sampling::Sampler` or remove duplicate loop.
- [x] **MED bug** `src/voice_clone.rs:1642-1707` — prompt tokens never seeded into
  sampler (comment claims Python parity); first token exempt from rep penalty. Seed
  history.
- [x] **MED bug** `mel.rs:51`, `stft_gpu.rs:18` — symmetric Hann vs torch periodic. Fix.
- [x] **MED bug** `src/voice_clone.rs:695-732` — cancellation/timeout only checked
  before/after full synthesis. Thread into chunk loop.
- [x] **MED perf** `src/models/vits.rs:2053-2058` — reference-mel style embedding
  recomputed per chunk. Cache in `set_reference_audio`.
- [x] **MED bloat** `src/voice_clone.rs` ~600 lines dead/duplicated (zero/few-shot
  paths, 4× punctuation block, dead helpers); `lingua` dep only serves dead code.
  Delete + drop dep.
- [x] **MED bloat** ~155 println/eprintln in src/ hot paths; crate already uses
  tracing. Convert key paths to debug!/trace!.
- [x] **MED test** `tests/integration.rs:14-23,135-210` + `src/text/bert_features.rs:215,235`
  — silent green without models; literal `~` path never expands ⇒ permanently
  skipped; hardcoded `/Users/yuechen/...`. Env-var root + `#[ignore]`.
- [x] **LOW idiom** `src/text/preprocessor.rs` — regexes recompiled per call; dead
  duplicate normalizer fns. LazyLock + delete.
- [x] **LOW bug** `src/text/symbols.rs:366` — unknown phoneme → "!" (id 0) not UNK(86).
- [x] **LOW bug** `src/audio/mel.rs:356-367` — clamped slice then fixed reshape can
  panic. Pad or validate.
- [x] **LOW bloat** `src/models/t2s.rs:1020-1130` unused T2SGenerate iterator; loader
  duplication (f16→f32 conversion only in unused path). Delete/unify.
- [x] **LOW bloat** committed debug artifacts: `gpt-sovits-mlx/2/`,
  `CRITICAL_CODE_REVIEW.md`, `CODE_REVIEW_COMPARISON.md`; 16/21 examples are one-off
  debug scripts. Remove/relocate.
- [x] **LOW idiom** `Error::Message(e.to_string())` erases typed variants (100+ sites)
  — fix opportunistically where touched.

*All gpt-sovits items fixed by a delegated agent: top-p HF semantics (+test),
reflect padding, periodic Hann, sampler config wiring + prompt seeding, per-chunk
cancellation, cached reference embedding, ~845 lines of dead code removed +
`lingua` dep dropped, hot-path prints → tracing, env-var'd test roots, LazyLock
regexes, UNK mapping, mel-slice padding, debug artifacts deleted, clippy 159 → 0.
Note: 17 test failures remain in the full run — 16 are the pre-existing MLX
"no Stream in current thread" environment issue (see mlx-rs section), and
`test_full_number_normalization` was A/B-verified failing on pristine HEAD.*

## P2 — qwen3-tts-mlx / qwen3-tts-core / qwen3-asr-mlx (wave-2 review)

- [x] **HIGH bug** `qwen3-asr-mlx/src/encoder.rs:76-80` — Python floor division ported
  as Rust truncating division in `get_feat_extract_output_lengths`: lengths that are
  multiples of 100 give 14 instead of 13, skewing block-attention windows (112 vs
  104) for all audio > 1 chunk. Use `div_euclid`.
- [ ] **HIGH bug** `qwen3-tts-mlx/src/lib.rs:1051-1061` + `speech_tokenizer.rs:309-363`
  — streaming decode is stateless per chunk (conv zero-pads from scratch, positions
  restart at 0): chunk-boundary artifacts. Carry decoder state or overlap+trim.
- [x] **HIGH bug** `qwen3-asr-mlx/src/model.rs:500-534` — fallback tokenizer installs
  ByteLevel as decoder only (no pre-tokenizer): mis-tokenizes everything and caches
  the broken tokenizer.json persistently. Add the ByteLevel pre-tokenizer.
- [x] **MED bug** `qwen3-asr-mlx/src/audio.rs:88-99` — STFT lacks Whisper center
  (reflect) padding; mel shifted 12.5 ms vs reference.
- [x] **MED bug** `qwen3-tts-mlx/src/speaker_encoder.rs:377-383` — symmetric vs
  periodic Hann.
- [x] **MED bug** `qwen3-asr-mlx/src/audio.rs:164` — `1 << 31` in i32 → negative
  max_val for 32-bit WAVs flips every sample. Use i64.
- [x] **MED bug** `qwen3-tts-mlx/src/lib.rs:1085-1116` — BPE loader never registers
  special tokens; ChatML markers get BPE'd. Load added_tokens_decoder like the ASR
  crate.
- [x] **MED bug** `qwen3-tts-mlx/src/lib.rs:1003-1005` — `start_streaming` forwards
  speed_factor>1 into EOS steering (premature truncation); clamp to 1.0.
- [ ] **MED perf** `qwen3-tts-mlx/src/lib.rs:130,235` — ~2.3 GB weights loaded twice
  for Base models; Mimi encoder loaded though ICL disabled.
- [ ] **MED bloat** `qwen3-tts-core` — ~550 LOC (generate/sampling/backend/
  codec_prefix/text) have zero callers; qwen3-tts-mlx reimplements with diverging
  constants. Migrate or delete.
- [ ] **MED bloat** `qwen3-tts-mlx/src/lib.rs` — ~60-line override/decode block
  copy-pasted 5×; extract helpers.
- [ ] **MED idiom** `qwen3-asr-mlx/src/model.rs:293-687` — unconditional eprintln in
  transcription hot path; use tracing.
- [x] **MED bloat** `qwen3-tts-mlx/src/mrope.rs` dead module; dead rope_speed_factor
  machinery; unused sample_logits wrapper.
- [x] **LOW bug** `qwen3-tts-core/src/generate.rs:151-161` — panic! in library loop +
  per-step logit dumps.
- [x] **LOW bug** `qwen3-tts-mlx/src/speech_tokenizer.rs:411` — window+1 attendable
  positions (off-by-one vs HF convention).
- [x] **LOW bug** `qwen3-tts-mlx/src/metal_kernels.rs:68` — hardcoded eps 1e-6 vs
  config.rms_norm_eps.
- [x] **LOW bug** `qwen3-asr-mlx/src/audio.rs:174-181` — >2-channel WAVs processed as
  interleaved mono.
- [ ] **LOW perf** `qwen3-tts-mlx/src/talker.rs:554-567` — GPU→CPU→GPU roundtrip to
  zero 3 positions; use a mask multiply.
- [x] **LOW bloat** `qwen3-tts-mlx/src/lib.rs` — 5× no-op `eval(std::iter::empty())`.
- [ ] **LOW test** `qwen3-tts-mlx/examples/benchmark.rs:9-11,248-291` — hardcoded
  relative paths; silent skips.
- [ ] **Test suites**: all three crates have ZERO tests. Highest-value additions:
  `get_feat_extract_output_lengths(100) == 13` (catches the HIGH), golden-mel vs HF
  extractor, WSOLA/EOS-bias unit tests.


### Deferred with rationale (P2 carry-overs)

- step-audio2: per-shard triple deserialization at load (#7), full-key
  `validate_weights` for flow/hifigan (#11), `SamplingConfig` wiring (#13) —
  all need loader/API restructuring; tracked here, not blocking correctness.
- dflash/mtplx T>0 acceptance: full-vocab fp32 host copy per cycle — needs an
  on-device gather (take_along_axis) + small-row transfer; both crates have
  seeded statistical acceptance tests to validate against when done.
- qwen3-tts: stateful streaming decode (HIGH — chunk-boundary artifacts),
  double weight load for Base models, special-token registration in the BPE
  fallback loader, 5× copy-pasted synthesize block, qwen3-tts-core dead
  modules (~550 LOC reimplemented divergently in qwen3-tts-mlx — needs a
  migrate-or-delete decision), eps template arg in fused_residual_rmsnorm,
  Whisper center padding in qwen3-asr mel. Also note: users who already ran
  qwen3-asr with the broken fallback tokenizer have a corrupted cached
  tokenizer.json in their model dir — delete it to regenerate.
- mtplx graph_bank: left observation-only (doc'd); wiring `invoke` into the
  verify forward is a feature change.

## P3 — clippy backlog

- [x] Drive `cargo clippy --workspace --all-targets` warnings down (159 in
  gpt-sovits-mlx lib alone; full inventory in /tmp/clippy-full.txt). Apply safe
  mechanical fixes (`cargo clippy --fix` + review), crate by crate.
  *Done for every crate touched in this pass: gpt-sovits 159→0, qwen-image,
  step-audio2, mlx-rs-core, dflash, mtplx, gemma4 all clippy-clean
  (--all-targets). Crates pending review below still carry their warnings.*

## P4 — pending crate reviews

Full reports landed for three groups — see `docs/review/qwen3-llm.md`,
`docs/review/image-gen.md`, `docs/review/vlm.md`. Their headline items:

- [x] **HIGH bug** `qwen3.6-mlx/src/cache.rs:164-211` — prefix-cache load on an
  identical prompt re-prefills the last token without trimming the loaded
  caches: KV holds it twice and GDN recurrence runs it twice (silent corruption
  on regeneration — production API pattern). *Fixed: trim KV by `offset-n`;
  reject snapshots whose recurrent step > n.*
- [x] **MED perf** `qwen3.6-mlx/src/lib.rs:294,311-314` — env vars read per
  decoded token. *Fixed: read once at Generate construction.*
- [x] **MED bug** `qwen3.6-mlx/src/model.rs` ×4 — lazy cache init hard-codes
  `HybridCache::KV`, ignoring the configured KVCacheMode; copy-pasted 4×.
  *Fixed: factored `ensure_cache` helper (keeps standard-mode default, now in
  one place with a doc note).*
- [x] **HIGH bug** `qwen3.5-35B-mlx/src/deltanet.rs:304-335` — conv1d_prefill
  drops warm conv state + malformed short-prompt window (qwen3.6 fix never
  backported). *Fixed: backported `conv_state.take()` + padded-slice save.*
- [ ] **MED bugs** qwen3.6 VL parity: vision pos-embed indexed sequentially not
  spatially (vision.rs:183), deepstack outputs summed into input embeds instead
  of per-decoder-layer injection (vision.rs:205), multi-image prompts silently
  mis-embedded (model.rs:1000-1043).
- [ ] **MED bug** `qwen3.5-35B-mlx/src/deltanet.rs` l2-normalize eps drift
  (~3.9% relmax, qwen3.6 documented + fixed it; backport).
- [ ] qwen3-llm LOW items: zero-tensor GDN rollback test, dead padding knobs,
  compute_decay duplicate, cross-crate duplication (SwitchGLU etc. → core),
  qwen3-mlx reverse-order top-k drift. See report.
- [x] **HIGH bug** `zimage-mlx/examples/generate_zimage.rs:472-659` — stale
  `/tmp/ref_*.bin` files silently hijack prompt embeddings/latents. *Fixed:
  gated behind `ZIMAGE_PARITY=1`.*
- [x] **HIGH perf** flux-klein + zimage: manual attention in all four DiT model
  files (~226 MB score tensors per block); use fused SDPA. *Fixed: all six
  manual chains (klein_model ×2, klein_quantized ×2, zimage ×2) now use
  `fast::scaled_dot_product_attention`; also fixed the pre-existing
  `test_rope_3axis` wrong expectation (half-dim angles) + added value asserts.*
  Remaining MED items:
  final_norm RmsNorm-vs-LayerNorm parity, FLUX.1 VAE constants reused for
  FLUX.2, silent loader key drops, ~900 lines dead legacy architecture. See
  report.
- [ ] **HIGH bugs** qwen3-vl-mlx: preprocessing missing mean/std normalization
  (trained on [-1,1]), ViT applies NO 2-D RoPE at all, pos-embed table read
  with linear ids instead of bilinear interpolation, merger groups 4 row-major
  patches instead of 2×2 blocks. (All four mean the "validated" describe_image
  output is far off reference; needs a parity pass against HF.) Plus MED:
  single-image-only splice.
- [ ] **HIGH bug** `minicpm-sala-mlx/src/attention/lightning.rs:535-566` —
  partial-chunk decay bug suppresses long-range memory for almost every prompt
  length. Plus MED: sparse-attention top-k uses last query position only + no
  causal mask in gathered window. See report.

Still pending (agents hit session caps twice; reports not written):

- [ ] funasr-mlx, funasr-nano-mlx, funasr-qwen4b-mlx (quantify cross-crate duplication)
- [ ] paddleocr-vl-mlx, glm-ocr-mlx, deepseek-ocr2-mlx
- [ ] glm4-mlx, glm4-moe-mlx, glm-4.7-flash-mlx, mistral-mlx, mixtral-mlx
  (copy-paste divergence)

## Verification matrix

- Per-crate: `cargo build --release -p <crate>` + `cargo test -p <crate>` + clippy.
- Examples with local models (./models/): gemma4 (gemma-4-12B-it-4bit, e4b, 26B UD),
  qwen3 (Qwen3-4B), qwen3-vl (Qwen3-VL-4B-Instruct-4bit), qwen3.5/3.6, dflash
  (Qwen3.6-27B-DFlash, gemma-4-26B-A4B-it-DFlash), mtplx (MTPLX-Optimized-Speed),
  glm-ocr (GLM-OCR), paddleocr (PaddleOCR-VL-1.5/1.6), moxin (moxin-llm-7b),
  qwen3-asr (qwen3-asr-1.7b). No local models for: image gen, TTS, funasr,
  step-audio2 — build/test only.
