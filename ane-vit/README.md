# ANE Vision-ViT Spike

Exploration: route the vision encoder (ViT) of multimodal models in this repo
through the Apple Neural Engine (ANE) via Core ML, while keeping the LLM body
on MLX/Metal.

## Why a vision-only carve-out

ANE is convolution + small-matmul-optimized. It is *not* competitive with
Metal GPU for large LLM matmuls or long-context attention at this scale
(see `docs/coreml-tradeoffs.md` — TL;DR is in the parent design note). But
ViTs are exactly the workload ANE was built for:

- Fixed input shape (patches of `336x336x3` or `224x224x3`).
- 24-32 transformer blocks of modest hidden size (1024-1280).
- Patch embed = a Conv2d — ANE-native.
- No KV cache, no dynamic shapes, no mutable state — eliminates Core ML's
  worst pain points.

Target end-state: small TTFT reduction on multimodal requests by offloading
~10-15% of model compute (the vision tower) onto otherwise-idle hardware,
while the GPU prefills the text portion in parallel.

## Candidate targets (in this repo)

| Model | Vision config | Notes |
|---|---|---|
| `glm-ocr-mlx` (scaffolded) | 24 layers, hidden 1024, 16 heads, image 336, patch 14, spatial_merge=2 | Cleanest target — just scaffolded; no existing perf to regress |
| `gemma4-mlx` vision tower | depends on `Gemma4VlModel` variant | Existing E4B VL path |
| `qwen3-vl-mlx` | Qwen3-VL ViT | Image_size=560, patch=14, more layers |
| `deepseek-ocr2-mlx` | Custom OCR vision tower | High complexity |

Start with **GLM-OCR**: it's the simplest, doesn't yet exist as an MLX
implementation that we'd regress, and the spike can wire the Core ML
side-channel without competing with an MLX baseline.

## Architecture

```
                            ┌─────────────────┐
   image bytes ──[PIL]────► │ Core ML ViT     │  (ANE / mlpackage)
                            │  336×336×3 →    │
                            │  144×1536       │
                            └────────┬────────┘
                                     │ image_tokens [144, 1536]
                                     ▼
   text prompt ──[tokenize]── ▶┌─────────────────┐
                              │ MLX text decoder│  (GPU / MLX)
                              │ glm_ocr_text    │
                              │ + mrope         │
                              └─────────────────┘
                                     │
                                     ▼
                                  output tokens
```

Process model:
1. **Python conversion (one-time)** — load HF/PyTorch ViT weights, trace +
   convert to Core ML `.mlpackage` with `MLComputeUnits.all`. ANE-eligible
   ops dispatch automatically; CPU fallbacks logged.
2. **Rust ⟶ Swift FFI** — small Swift `.dylib` exposes `predict(input_pixels) -> output_tensor`
   via C ABI; Rust `coreml-bridge` crate calls it.
3. **MLX integration** — `glm_ocr_mlx::VisionEncoder::forward` becomes a thin
   dispatcher: if Core ML backend available and `ANE_VISION=1`, route through
   the FFI; otherwise fall back to (future) MLX implementation.

## Components in this directory

- `converters/convert_vit_to_coreml.py` — coremltools-based ViT → mlpackage converter.
- `coreml-bridge/` — Rust crate with C FFI to a Swift Core ML helper.
- `coreml-bridge/src/ane_runner.swift` — Swift glue: load mlpackage, predict, return raw output.
- `bench/ane_vit_bench.rs` — micro-bench comparing ANE-routed ViT vs MLX-routed ViT (when the latter lands).

## Expected outcomes (going in)

- **Best case (15-25% smaller TTFT on multimodal requests):** ANE handles the
  vision tower in 30-80 ms while GPU prefills the text portion in parallel;
  total visible TTFT improves by the overlap.
- **Likely case (5-15% TTFT win):** ANE marginally faster than GPU on the ViT
  alone; overlap with GPU prefill yields a modest but real reduction.
- **Worst case (no win):** ANE compiler fails to map enough of the ViT
  graph to the Neural Engine; ops fall back to CPU/GPU; net wash or
  regression vs running everything on GPU.

Validating which regime we land in is the whole point of the spike — we
should know after the first end-to-end run with `MLComputeUnits.all`
profiling.

## Open questions

1. Will the GLM-OCR ViT's `attention_bias=true` cause ANE to fall back? Some
   bias patterns the ANE compiler doesn't accept (it prefers no-bias variants
   in linear layers).
2. `spatial_merge_size=2` is a custom reshape; coremltools may or may not
   trace it cleanly.
3. Sandbox / signing concerns when invoking ANE from a non-app-bundle Rust
   binary — historically `MLComputeUnits.cpuAndNeuralEngine` *requires* a
   signed app bundle to access ANE; CLI binaries are often restricted to
   GPU/CPU. Verify before measuring.
4. Bridging cost: copying pixels in → tensor out across FFI. For 336×336×3 =
   ~340 KB this is negligible per request, but worth measuring.

## Status

Spike only. Nothing here is wired into mainline yet.
