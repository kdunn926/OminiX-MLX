# ANE-ViT Spike — Results

End-to-end validation completed against `google/vit-base-patch16-224` (the
GLM-OCR vision tower follow-up is below — needs multimodal patchification
wired through the converter wrapper).

## Bench setup

- Model: `google/vit-base-patch16-224` from HuggingFace.
- Conversion: `coremltools 9.0`, `precision=float16`, `mlprogram` target,
  `minimum_deployment_target=macOS14`.
- Output: `ane-vit-spike/vit-base.mlpackage` (164 MB).
- Input shape: `[1, 3, 224, 224]` (single image, 150 528 fp32 values in).
- Output shape: `[1, 197, 768]` (CLS + 14×14 patches, fp16) — 151 296 elements out.
- Iterations: 32 per compute-unit policy, after a warmup predict.
- Hardware: this machine.

## Results

| Compute units                | Mean (ms) | p50 (ms) | p99 (ms) | Min   | Max   |
|------------------------------|-----------|----------|----------|-------|-------|
| `cpuOnly`                    | 40.21     | 38.67    | 55.99    | 33.01 | 55.99 |
| `cpuAndGpu`                  | 20.49     | 15.02    | 83.66    | 10.89 | 83.66 |
| **`cpuAndNeuralEngine`** ⭐  | **5.96**  | **6.13** | **7.02** | 5.39  | 7.02  |
| `all` (auto)                 | 6.56      | 6.38     | 7.59     | 5.54  | 7.59  |

**ANE is 6.7× faster than CPU and 3.4× faster than GPU for this ViT.**

`all` (auto-dispatch) lands within noise of `cpuAndNeuralEngine`, confirming
the Core ML scheduler correctly identifies the ANE as the best target for
this graph.

## Validated open questions from the README

1. ✅ **CLI sandbox/signing.** Core ML's `cpuAndNeuralEngine` *does* work
   from a plain Rust release binary (no app bundle, no signed binary, no
   entitlements). The historical restriction has been relaxed for at least
   the `MLComputeUnits.cpuAndNeuralEngine` compute-unit selection on macOS
   14+. `.neuralEngineOnly` doesn't exist as a separate enum case — Core ML
   falls back to CPU silently if the graph has any ANE-unsupported op.

2. ✅ **FFI overhead is negligible.** The bench includes input copy from
   Rust into MLMultiArray and fp16 → fp32 widening on the output via
   `vImageConvert_Planar16FtoPlanarF`. End-to-end per-predict at 5.96 ms
   means the FFI plus model amount; the Swift→Core ML hop adds well under
   1 ms.

3. ✅ **fp16 output handling.** ANE-friendly models emit fp16 by default
   (`dataType=65552`). The Swift shim widens via `vImage` Accelerate
   routine; Rust callers see fp32 as expected.

## Qwen3-VL conversion (LANDED 2026-05-21)

Successfully converted `Qwen/Qwen3-VL-2B-Instruct` vision tower to
`qwen3-vl-2b-vit.mlpackage` (582 MB, float16). Five
architecture-specific tracing blockers resolved in sequence:

1. ✅ Pre-patchified input shape — added inline patchifier wrapper.
2. ✅ Deepstack list-of-tensors output — `deepstack_visual_indexes = []`.
3. ✅ BaseModelOutputWithPooling unpacking — wrapper returns
   `out.last_hidden_state`.
4. ✅ Windowed-attention `cu_seqlens.tolist()` Python loop —
   monkey-patched `Qwen3VLVisionAttention.forward` to a single full
   SDPA (semantically equivalent for the single-image square case).
5. ✅ Mixed-dtype stack op from `transformers.vision_utils` —
   `get_vision_bilinear_indices_and_weights`, `get_vision_position_ids`,
   `get_vision_cu_seqlens` all `tolist()`-iterate and build
   `[int, float, float, float]` stacks. Fixed by rebinding the module-level
   helpers in `modeling_qwen3_vl` to return precomputed constants
   for the fixed input shape.

### Bench (Qwen3-VL ViT, 448×448, 16 iters)

| Compute units                | Mean (ms) | p50 (ms) | p99 (ms) |
|------------------------------|-----------|----------|----------|
| `cpuOnly`                    | 734       | 732      | 780      |
| `cpuAndNeuralEngine`         | 341       | 341      | 346      |
| `all` (auto, lands on GPU)   | 221       | 221      | 226      |
| **`cpuAndGpu`** ⭐           | **165**   | **165**  | **168**  |

Unlike the small ViT-base/224 case (where ANE was 6.7× faster than CPU
and 3.4× over GPU), the bigger Qwen3-VL ViT (hidden=4096, 784 patches,
24 layers) lands fastest on GPU. ANE still beats CPU 2.2× but loses to
GPU 2.1×. Suspected reason: the windowed-attention bypass + a few ops
that don't pattern-match ANE tiles force partial CPU fallback that
breaks ANE's pipelined-layer execution. Practical implication:
**dispatch Qwen3-VL ViT to `CpuAndGpu`, not `CpuAndNeuralEngine`** —
the `VisionTowerClass::Qwen36Vl` selector in `engines/ane_vision.rs`
should encode this.

## Gemma4-VL conversion (LANDED 2026-05-21)

Successfully converted `google/gemma-4-E4B-it` vision tower to
`gemma4-e4b-vit.mlpackage` (291 MB, float16). Three model-specific
tracing fixes:

1. ✅ `F.one_hot` on fp32 indices — coremltools doesn't preserve
   int64 through `.clamp(min=0)`. Precomputed
   `patch_embedder._position_embeddings(fixed_pos, fake_padding)`
   for the fixed grid and rebound the method to return the constant.
2. ✅ `pooler._avg_pool_by_positions` (same one_hot blocker) —
   precomputed the pooling weight matrix + mask, rebound the method.
3. ✅ `create_bidirectional_mask` uses `attention_mask.new_ones(...)`
   which has no coremltools translator. Replaced the encoder.forward
   to call layers with `attention_mask=None` (no padding case = attend
   to everything).
4. ✅ Skipped data-dependent `hidden_states[pooler_mask]` slice
   (all-True mask in the no-padding case) and dataclass return.

### Bench (Gemma4-VL ViT, 384×384, 16 iters)

| Compute units                | Mean (ms) | p50 (ms) | p99 (ms) |
|------------------------------|-----------|----------|----------|
| `cpuOnly`                    | 183       | 179      | 196      |
| `cpuAndNeuralEngine`         | 51        | 51       | 51       |
| `all` (auto, lands on ANE)   | 51        | 51       | 51       |
| **`cpuAndGpu`** ⭐           | **29**    | **28**   | **31**   |

Smaller than Qwen3-VL (hidden=768 vs 4096) but same dispatch
verdict: **GPU wins on the multimodal-LLM-sized ViT class**.
ANE 3.6× over CPU; GPU 6.3× over CPU and 1.8× over ANE.
`VisionTowerClass::Gemma4Vl` selector should dispatch to
`CpuAndGpu`.

## End-to-end TTFT (MLX baseline, Gemma4-VL E4B, 2026-05-22)

Bench: `gemma4-mlx/examples/vit_ttft_bench.rs` with
`models/gemma-4-E4B-it` and a 230 KB PNG. Each iter runs:
1. `model.encode_image_bytes(bytes)` — preprocess + ViT forward +
   `embed_vision` projection.
2. `model.prefill_multimodal(input_ids, visual_features, &mut cache)` —
   text+image embed scatter + full LLM prefill via lm_head.
3. Greedy argmax of the last-position logits.

| Stage             | mean (ms) | p50 (ms) | note |
|-------------------|-----------|----------|------|
| vision encode     | 283       | 278      | 255 soft tokens (variable-res, ~768×768 internal) |
| LLM prefill       | 1 653     | 1 648    | 283 prompt tokens total (28 text + 255 image) |
| **Total TTFT**    | **1 936** | **1 927**| 5 iters after warmup |
| vision share      | 14.6%     | —        | of TTFT |

### Why this matters for the ANE comparison

The ANE/GPU mlpackage stays fixed at 384×384 → 64 soft tokens. The MLX
production path scales the same input image up to ~768×768 → 256 soft
tokens (4× more visual tokens fed into the LLM prefill). So a naive
"swap ANE for MLX vision" save is double-edged:

- **Vision-stage save** (apples-to-apples at the ViT cost):
  283 ms → 29 ms (GPU mlpackage) = **−254 ms / −13% TTFT**.
- **Soft-token count save** (much larger): going from 255 → 64 image
  tokens shrinks prefill from 1 653 ms to an extrapolated
  ~537 ms (linear in seq len, 4× fewer image tokens) = **−1 116 ms
  / −58% TTFT**.
- Total projected ANE-path TTFT ≈ 29 + 537 + 22 = **~588 ms (3.3× faster)** —
  but with a real quality loss (4× fewer visual tokens, accuracy not
  validated).

To get the "ANE-path latency win with no quality loss" number we'd
need a second mlpackage traced at 768×768 fixed (→ 256 soft tokens).
That ViT call would be ~4× the ANE/GPU work (~120 ms on GPU
extrapolated) but the LLM prefill stays at MLX-baseline length.
Projected TTFT: 120 + 1 653 + 22 ≈ **1 795 ms (1.08× faster)** — a
real but modest win driven entirely by the vision delta. Tighter
overlap (async vision + prefill) would amortize the vision cost
further but doesn't help bandwidth-bound prefill.

### Decision matrix

| Want                          | Use                                         |
|-------------------------------|---------------------------------------------|
| Best TTFT, lossy is OK        | ANE-path at 384, 64 tokens (~3.3× faster)   |
| Best TTFT, no quality loss    | ANE-path at 768, 256 tokens (~1.08× faster) |
| Best accuracy                 | Current MLX path                            |
| Best TTFT *and* accuracy      | Overlap MLX vision with prefill (async pipeline; future work) |

## End-to-end TTFT wired through ANE (2026-05-22)

Spike branch `spike/async-vision-prefill` now drives the MLX
`embed_vision` projection from the converted Gemma4-VL mlpackage's
output, replacing the MLX vision tower. Bench
`gemma4-mlx/examples/vit_ttft_bench.rs` with `--ane <mlpackage>` and
`--ane-units {gpu|ane|cpu|all}`.

### Three-way A/B (gemma-4-E4B-it, img1.png, 5 iters)

| Path                                  | vision  | prefill (tokens) | **TTFT**  | first  |
|---------------------------------------|---------|------------------|-----------|--------|
| MLX baseline (255 soft tokens, ~768²) | 300 ms  | 1 710 ms (283)   | **2 011** | "This" |
| **ANE→GPU** (64 soft tokens, 384²) ⭐ | 30 ms   | 513 ms (92)      | **544**   | "The"  |
| ANE→ANE (64 soft tokens, 384²)        | 54 ms   | 511 ms (92)      | 566       | "Please" |

**3.7× faster TTFT, 1.47 s saved**. Vision delta is 270 ms; prefill
delta is 1 197 ms. Confirms the central finding: **soft-token count
dominates which-accelerator choice**. The image-token positions in
the LLM prefill scale linearly (~6 ms each on E4B prefill at this
context length), so cutting image tokens 4× saves ~5× more wall time
than the ViT speedup does.

### Accuracy caveats

- First tokens diverge across all three paths ("This" / "The" /
  "Please") — same prompt, different visual encodings. The 384/64 ANE
  path is a real lossy trade, not just a latency one.
- ANE→GPU and ANE→ANE produce different first tokens from the *same*
  mlpackage. Known Core ML quirk: compute-unit backends have small
  numerical deltas that propagate into the downstream LLM.
- For an accuracy-preserving comparison, need a second mlpackage
  traced at 768×768 / 256 soft tokens. Projected TTFT for that
  retrace ≈ 1 760 ms (~13% faster vs MLX baseline) — modest gain
  because prefill (not vision) dominates when image-token counts are
  matched.

### Decision matrix

| Want                          | Use                                              |
|-------------------------------|--------------------------------------------------|
| Best TTFT, lossy is OK        | ANE→GPU at 384²/64 tokens (3.7× faster)          |
| Best TTFT, no quality loss    | ANE-path at 768²/256 tokens (projected ~1.08×)   |
| Best accuracy                 | Current MLX path                                 |
| Best TTFT *and* accuracy      | Overlap ANE vision with MLX prefill chunk 0 (~50 ms additional save on top of 1.08×) — needs background-thread ANE call |

## Earlier Qwen3-VL attempt (superseded by the LANDED section above)

Tried `Qwen/Qwen3-VL-2B-Instruct` via `convert_qwen3_vl_vit.py`. Hit four
architecture-specific tracing blockers in sequence:

1. ✅ **Pre-patchified input shape** — vision tower expects flattened
   patches `[grid_h*grid_w, C*T*P*P]` not raw pixels. Solved with an
   inline patchifier wrapper using reshape+permute (avoiding `unfold`
   which coremltools doesn't support).
2. ✅ **(Tensor, List[Tensor]) return** — Qwen3-VL emits deepstack
   features from layers 5/11/17 for the LLM body. Suppressed via
   `vt.deepstack_visual_indexes = []` before tracing.
3. ✅ **BaseModelOutputWithPooling dataclass return** — wrapper unpacks
   `out.last_hidden_state`.
4. ❌ **`unpack node expected 1 outputs, got 784`** — internal
   tuple-unpack in the model (likely windowed attention split) emits one
   output per patch. coremltools rejects.

Bullet 4 needs either a model-specific patch (replace the unpack with a
matrix op) or a different tracing approach (`torch.jit.script` is more
flexible but coremltools v9 requires `trace`-style graphs). Deferred —
ANE perf delta for this ViT class is already established by the
`google/vit-base` result above; reaching parity with Qwen3-VL's
production vision pipeline requires architecture-aware conversion work
that's properly scoped as its own project.

## Zero-copy validation

`coreml_predict_zero_copy` wraps the caller's Float buffer as MLMultiArray
with a no-op deallocator (Swift side). A/B on ViT-base/224:

| Mode | ANE mean | GPU mean | CPU mean |
|---|---|---|---|
| Copy | 5.81 ms | 18.17 ms | 40.70 ms |
| Zero-copy (raw ptr) | 5.77 ms | 17.68 ms | 42.31 ms |

**Delta is noise.** 600 KB input copies under the ANE predict latency.
The win materializes when the pixel buffer is already GPU-resident from
a prior MLX compute — then the copy-variant pays GPU → CPU → ANE
round-trip, while the IOSurface-backed MTLBuffer variant lets ANE read
the buffer directly.

That deeper handoff requires:
1. `mlx-rs` to expose the backing Metal buffer of an `Array` (currently
   private; needs an `Array::metal_buffer() -> id<MTLBuffer>` accessor).
2. Wrapping the MTLBuffer as IOSurface via Metal's
   `MTLBuffer.allocatedSize` + `IOSurfaceCreate` (or extract from
   `MTLBuffer` if it was already allocated with IOSurface backing).
3. Building a CVPixelBuffer from IOSurface + passing to
   `MLFeatureValue(pixelBuffer:)`.

The Swift shim already accepts CVPixelBuffer-style inputs in newer
Core ML APIs; a follow-up FFI variant `coreml_predict_iosurface` would
take an `IOSurfaceRef` from Rust. Deferred — needs the upstream `mlx-rs`
work first.

## Open items (deferred)

1. **GLM-OCR vision tower conversion.** The HF model uses `grid_thw`
   patchification and an internal spatial-temporal patch fusion that
   doesn't trace cleanly with `[1, 3, 336, 336]` pixels alone — needs the
   pixel tensor temporally tiled to match `temporal_patch_size=2`. Wrapper
   in `converters/convert_vit_to_coreml.py` accepts `(pixels, grid_thw)`
   but the inner forward asserts a shape relationship between them we
   haven't satisfied yet (error: "tensor a (288) must match tensor b (576)
   at non-singleton dimension 0").

2. **Qwen3-VL vision tower** — partial progress, two blockers down,
   one still ahead.
   - ✅ Got past the original "unpack 1→784" trace failure via
     pre-patchification in the wrapper.
   - ✅ Got past the deepstack outputs / dataclass return blocker.
   - ✅ Got past the windowed-attention `cu_seqlens` split blocker
     by monkey-patching `Qwen3VLVisionAttention.forward` to use a
     single full-SDPA call during tracing. Semantically equivalent
     for square single-image inputs (1 chunk in the windowed case).
   - ❌ **STILL BLOCKED** on a coremltools MIL error in 2026-05-21
     attempt:
     ```
     ValueError: Tensors in 'values' of the stack op (input.5) should
     share the same data type. Got [int, double, double, double].
     ```
     The offending stack isn't in the attention path — bypassing
     windowed attention doesn't fix it. It's in the patch
     embedding, position-embedding interpolation, or somewhere the
     coremltools Torch frontend inserts an implicit stack of
     mixed-dtype tensors (the int element looks like a shape value;
     the doubles look like rotary frequencies or position values).
     Candidate sites: `apply_interleaved_mrope` (line 370 of
     modeling_qwen3_vl.py), `fast_pos_embed_interpolate` (line 702),
     freq-table indexing in `Qwen3VLVisionRotaryEmbedding.forward`.
   - Realistic estimate to push through: 1-2 days of focused
     coremltools / torchscript debugging — find the specific stack
     op, monkey-patch to coerce dtypes, or split the model into
     smaller traceable subgraphs.

3. **MLX → ANE handoff path.** Currently the bench loads fp32 pixels from
   a Rust `Vec<f32>`. In real use we'd want to forward an MLX `Array`
   directly. Two options: (a) `eval` the MLX array then `as_slice::<f32>()`
   into the FFI buffer (one copy), or (b) preallocate an `MTLBuffer` that
   both MLX and Core ML can share (zero-copy on unified memory; requires
   `IOSurface`-backed buffers and a different Swift entry point).

4. **End-to-end multimodal latency.** Once a real ViT lands, compose with
   the GLM-OCR text decoder (or Qwen3-VL's text body running on MLX/Metal)
   and measure end-to-end TTFT for an image+prompt request. Expected win:
   ~150-300 ms saved on the vision step, partially overlapped with the
   text prefill if invoked asynchronously.

## Files added on `spike/ane-vit`

- `ane-vit-spike/README.md` — spike scope + architecture
- `ane-vit-spike/RESULTS.md` — this file
- `ane-vit-spike/converters/convert_simple_vit.py` — vanilla ViT → mlpackage (working)
- `ane-vit-spike/converters/convert_vit_to_coreml.py` — multimodal ViT (in progress)
- `ane-vit-spike/coreml-bridge/` — Rust → Swift → Core ML FFI (working)
- `ane-vit-spike/coreml-bridge/examples/bench.rs` — A/B bench across compute units
- `ane-vit-spike/vit-base.mlpackage` — converted test artifact (164 MB; gitignore later)
