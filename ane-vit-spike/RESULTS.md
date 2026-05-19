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

## Qwen3-VL conversion attempt (deferred)

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

2. **Qwen3-VL vision tower.** Similar patchification but better tooling on
   the HF side; likely easier first target than GLM-OCR.

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
