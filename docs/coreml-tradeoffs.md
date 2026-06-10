# Core ML / ANE Tradeoffs for OminiX-MLX Vision Encoders

Spike branch: `spike/async-vision-prefill`  
Dates: 2026-05-21 – 2026-06-10

## TL;DR decision matrix

| Goal | Approach | Expected TTFT |
|------|----------|---------------|
| Best TTFT, lossy (4× fewer image tokens) | ANE-384→CpuAndGpu (gemma4-e4b) | ~544 ms |
| Best TTFT, matched image tokens | ANE-768→CpuAndGpu (gemma4-e4b) | ~1 891 ms |
| Best accuracy | MLX baseline | ~2 011 ms |
| Streaming ASR overlap | Core ML CpuAndGpu chunk encoder | fully hidden |

**Core ML `CpuAndGpu` is the correct compute-unit choice for every model
currently in this workspace.**  `CpuAndNeuralEngine` is competitive only for
very small vision encoders (≤200 patches, ≤768 hidden), which none of the
shipping VLMs or ASR models reach.

---

## 1. ANE eligibility heuristic

Derived from empirical benchmarks on M-series hardware:

```
fn recommended_for(patch_count: usize, hidden_size: usize) -> ComputeUnits {
    if patch_count <= 200 && hidden_size <= 768 {
        CpuAndNeuralEngine   // ANE is 3–7× faster than GPU here
    } else {
        CpuAndGpu            // GPU wins everywhere else
    }
}
```

| Model | Patches | Hidden | Best units | Ratio |
|-------|---------|--------|------------|-------|
| vit-base-patch16-224 | 196 | 768 | ANE ⭐ | 3.4× over GPU |
| Gemma4 E4B ViT (384²) | 576 | 768 | CpuAndGpu ⭐ | 1.8× over ANE |
| Qwen3-VL ViT (448²) | 784 | 2560 | CpuAndGpu ⭐ | 2.1× over ANE |
| Qwen3-ASR encoder (13 tok) | 13 | 1024 | CpuAndGpu ⭐ | 1.4× over ANE |

The 200-patch boundary is a hard cliff: below it ANE dominates; above it,
partial CPU fallback in the compiled graph breaks ANE's pipelined-layer
execution and GPU takes over.

---

## 2. End-to-end TTFT measurements (Gemma4 E4B, 230 KB PNG)

### 3-way comparison at 384²/64 soft tokens

| Path | Vision | Prefill (tokens) | TTFT | First token |
|------|--------|------------------|------|-------------|
| MLX baseline (255 tok, ~768²) | 300 ms | 1 710 ms (283) | **2 011 ms** | "This" |
| **ANE-384→CpuAndGpu** ⭐ | 30 ms | 513 ms (92) | **544 ms** | "The" |
| ANE-384→CpuAndNE | 54 ms | 511 ms (92) | 566 ms | "Please" |

3.7× faster end-to-end, 1.47 s saved — but at a real accuracy cost (4× fewer image tokens, different first token).

### Accuracy-preserving retrace at 768²/256 soft tokens

| Path | Vision | Prefill | TTFT | First token |
|------|--------|---------|------|-------------|
| ANE-768→**CpuAndGpu** ⭐ | 138 ms | 1 753 ms | **1 891 ms** | "The" |
| ANE-768→CpuAndNE | 578 ms | 1 753 ms | 2 330 ms | "I" |

1.06× faster, 120 ms saved. Modest because prefill (1.7 s) dominates when image-token counts are matched.

### Why soft-token count dominates

Each image token costs ~6 ms in LLM prefill on E4B. Cutting image tokens 4×
(255 → 64) saves ~1 160 ms in prefill — 5× more than the ViT speedup itself.
The ANE/GPU dispatch decision matters for the vision stage, but **reducing
image-token count is the dominant lever for TTFT**.

---

## 3. Async-overlap experiment (split prefill)

### What was tested

Background-thread ANE predict + simultaneous GPU `prefill_text(prefix_ids)`,
then join before the image span and continue with `prefill_multimodal`.

### Results

| Run | Prefix | ANE | Prefill | TTFT |
|-----|--------|-----|---------|------|
| Sync ANE baseline (short prompt) | — | 28 ms | 160 ms | **188 ms** |
| Overlap (short prompt, ~15 prefix tokens) | 2.2 ms | 26 ms | 182 ms | **210 ms** |
| Overlap (long prompt, ~140 prefix tokens) | 2.2 ms | 26 ms | 261 ms | **289 ms** |

**Overlap was 22 ms slower than sync** for the short prompt case.

### Why it didn't work

1. **Benchmark bug**: `eval([] as [&Array; 0])` is a no-op in MLX — calling
   `mlx_eval` with an empty `VectorArray` doesn't flush GPU work. The
   `prefix=2.2ms` reading measured only CPU Metal command-queue latency, not
   GPU execution. The real prefix GPU time leaked into the continuation prefill
   measurement.

2. **Wrong accelerator**: The E4B ViT at 576 patches runs in 28 ms on
   `CpuAndGpu`. There is no cheaper alternative to hide against.

3. **Split-path overhead**: `prefill_text` calls `forward_last_logits`, which
   runs the lm_head matmul on prefix tokens unnecessarily. The sync path skips
   this intermediate step.

### What would make it viable

- ViT ≤200 patches (ANE ~6 ms) + prefix ≥ 4 text tokens → ANE fully hidden
- Fix the eval call: `eval([&prefix_logits])?` instead of `eval([])?`
- Add `forward_no_logits` to skip the lm_head on prefix tokens

**Decision: vision carve-out closed for E4B. Infrastructure retained.**

---

## 4. Qwen3-ASR encoder bench

The audio encoder processes mel spectrograms in 100-frame chunks:
`[1, 128, 100]` → 3× Conv2d frontend → 24 Transformer layers (13 tokens) → `[13, 2048]`.

### Results (16 iters, models/qwen3-asr-1.7b)

| Path | Mean | p50 | min | max |
|------|------|-----|-----|-----|
| MLX Metal/GPU | 14.02 ms | 13.77 | 13.56 | 16.02 |
| Core ML CpuAndNE | 10.89 ms | 10.88 | 10.77 | 11.25 |
| **Core ML CpuAndGpu** ⭐ | **7.91 ms** | 7.62 | 7.36 | 10.57 |
| Core ML CpuOnly | 75.44 ms | 72.91 | 70.86 | 111.28 |

GPU wins (1.8× over MLX, 1.4× over ANE). The encoder's hidden_size=1024
is above the ANE eligibility threshold.

### Streaming overlap verdict: unconditionally free

```
Core ML CpuAndGpu per chunk:  7.91 ms
Text decode at 60 tok/s:     16.7 ms/tok
Typical text/chunk:           ~4 tokens  →  66.7 ms decode window
→ ANE FULLY HIDDEN (7.9 ms << 66.7 ms)
Min speech density:           0.5 text tok/chunk (~28 tok/s speech)
```

Encoding chunk N+1 on Core ML while decoding chunk N's text tokens on GPU
is always free at any real speech density. No split-prefill tricks needed —
the streaming chunk boundary is the natural sync point.

---

## 5. What Core ML actually compiles for `CpuAndGpu`

When compiled with `compute_units=cpuAndGPU`, Core ML's MIL compiler targets
Metal Performance Shaders for the dense matmuls and attention layers, and uses
CPU for scalar ops, conditionals, and anything that doesn't pattern-match GPU
tiles. The runtime load is fast (~87 ms warmup for the ASR encoder); subsequent
predicts hit the compiled Metal program directly.

When compiled with `cpuAndNeuralEngine`, the compiler attempts to tile ops onto
ANE's fixed-function tiles. For sequences >200 tokens or hidden >768, partial
CPU fallback breaks the pipelined tile execution and the latency climbs above
the GPU baseline.

---

## 6. Infrastructure shipped on this spike

| Artifact | Location | Purpose |
|----------|----------|---------|
| `ComputeUnits::recommended_for(patches, hidden)` | `ane-vit/coreml-bridge/src/lib.rs` | Empirical dispatch policy |
| Manifest `recommended_units` field + reader | manifests + `vit_ttft_bench.rs` | Runtime unit selection from sidecar |
| `--overlap` bench mode | `gemma4-mlx/examples/vit_ttft_bench.rs` | Scaffold for future overlap experiments |
| `--parity-check` bench mode | same | Cosine-sim ANE vs MLX ViT validation |
| `--system-prompt` / `--chunks` flags | same | Overlap coverage + multi-chunk timing |
| `--enumerated-sizes` converter flag | `ane-vit/converters/convert_gemma4_vl_vit.py` | Multi-res per-size mlpackages |
| Standalone ASR encoder converter | `ane-vit/converters/convert_qwen3_asr_encoder.py` | No transformers runtime needed |
| `asr_ane_bench` example | `qwen3-asr-mlx/examples/asr_ane_bench.rs` | ANE vs MLX per-chunk timing |
| Gemma4-VL weight key normalization | `gemma4-mlx/src/model.rs` | Fixes E4B/12B quantized VL load |
| `MaybeQuantized<Linear>` embedding_projection | `gemma4-mlx/src/vision.rs` | Fixes 4-bit matmul for embed_vision |
| `qwen3-asr-encoder.mlpackage` + manifest | `ane-vit/` | Converted ASR encoder (gitignored) |

---

## 7. Open items

1. **ASR encoder parity check**: cosine-similarity between MLX and Core ML
   encode outputs. The converter was built from the Rust architecture without
   a Python reference; a numerical validation pass is needed before using
   Core ML output in a production transcription pipeline.

2. **Small-ViT models**: if a future model ships with a ViT ≤200 patches /
   ≤768 hidden (e.g. a distilled vision encoder), revisit ANE dispatch and the
   async-overlap path. The bench scaffold and `recommended_for` heuristic are
   ready.

3. **Qwen3-ASR streaming integration**: wire `asr_ane_bench`'s overlap
   finding into the actual `transcribe_samples_chunked` loop — spawn a Core
   ML thread for chunk N+1 while decoding chunk N, using `Arc<Mutex<CoreMlModel>>`.
