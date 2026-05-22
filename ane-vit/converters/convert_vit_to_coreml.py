#!/usr/bin/env python3
"""Convert a Hugging Face vision encoder to Core ML for ANE.

Targets the GLM-OCR vision tower (`models/GLM-OCR/`) by default, but the
script accepts any HF model directory containing a `vision_config` block.
Output: a `.mlpackage` directory plus a JSON manifest with shape metadata
the Rust runtime needs to call into the model.

Usage:
    pip install torch transformers coremltools pillow
    python convert_vit_to_coreml.py \
        --model-dir ../../models/GLM-OCR \
        --output     ../glm-ocr-vit.mlpackage \
        --compute-units all     # 'all' | 'cpuAndNeuralEngine' | 'cpuAndGPU'

Notes:
- `compute-units=all` lets Core ML's scheduler pick CPU / GPU / ANE per op.
  Use `cpuAndNeuralEngine` to FORCE the ANE attempt (will fall back to CPU
  on unsupported ops; never to GPU). Use this for spike measurement.
- ANE access from a non-bundled CLI may be restricted. The first run logs
  which ops landed on which compute unit — check the `compute_unit_report`.
- The conversion may take 1-3 minutes for a 24-layer ViT.
"""

import argparse
import json
import os
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True, type=Path,
                   help="HF model directory containing vision_config + safetensors")
    p.add_argument("--output", required=True, type=Path,
                   help="Destination .mlpackage path")
    p.add_argument("--compute-units", default="all",
                   choices=["all", "cpuAndNeuralEngine", "cpuAndGPU", "cpuOnly"],
                   help="Core ML compute-unit dispatch policy")
    p.add_argument("--precision", default="float16",
                   choices=["float16", "float32"],
                   help="ANE prefers float16")
    args = p.parse_args()

    try:
        import torch
        import coremltools as ct
        from transformers import AutoModel, AutoConfig
    except ImportError as e:
        sys.stderr.write(
            f"Missing dependency: {e}\n"
            "Install with: pip install torch transformers coremltools pillow\n"
        )
        sys.exit(1)

    cfg_path = args.model_dir / "config.json"
    if not cfg_path.exists():
        sys.exit(f"config.json not found in {args.model_dir}")
    cfg = json.loads(cfg_path.read_text())
    vcfg = cfg.get("vision_config")
    if vcfg is None:
        sys.exit("config.json has no vision_config block")

    image_size = int(vcfg["image_size"])
    patch_size = int(vcfg["patch_size"])
    hidden = int(vcfg["hidden_size"])
    out_hidden = int(vcfg.get("out_hidden_size", hidden))

    print(f"[convert] vision_config: image={image_size} patch={patch_size} "
          f"hidden={hidden} out_hidden={out_hidden}")

    # Load full multimodal model, then pull just the vision tower. The exact
    # attribute path depends on the architecture; coremltools wants a single
    # `nn.Module` whose forward takes the raw pixel tensor.
    print(f"[convert] loading {args.model_dir} ...")
    model = AutoModel.from_pretrained(args.model_dir, trust_remote_code=True)
    model.eval()

    vision_raw = find_vision_tower(model)
    if vision_raw is None:
        sys.exit("Could not locate vision tower module on loaded model. "
                 "Inspect the model and adjust `find_vision_tower`.")

    # Many multimodal ViT towers (Qwen-VL, GLM-OCR) take extra args alongside
    # pixels — typically `grid_thw` (a 3-vector per image describing temporal
    # / height / width patch counts). Wrap the tower so the traced graph
    # accepts only the pixel tensor; grid_thw is hardcoded to a single 1xHxW
    # image at the model's native image_size / patch_size.
    grid_h = image_size // patch_size
    grid_w = image_size // patch_size

    class SinglePatchWrap(torch.nn.Module):
        def __init__(self, inner, grid_h, grid_w):
            super().__init__()
            self.inner = inner
            self.register_buffer(
                "grid_thw",
                torch.tensor([[1, grid_h, grid_w]], dtype=torch.long),
            )

        def forward(self, pixels):
            try:
                return self.inner(pixels, self.grid_thw)
            except TypeError:
                return self.inner(pixels)

    vision = SinglePatchWrap(vision_raw, grid_h, grid_w)
    vision.eval()

    # Trace with a dummy pixel input.
    dummy = torch.zeros(1, 3, image_size, image_size, dtype=torch.float32)
    print(f"[convert] tracing with input shape {tuple(dummy.shape)} grid_thw=[1,{grid_h},{grid_w}] ...")
    traced = torch.jit.trace(vision, dummy, strict=False)

    print("[convert] converting traced graph to Core ML ...")
    cu_map = {
        "all": ct.ComputeUnit.ALL,
        "cpuAndNeuralEngine": ct.ComputeUnit.CPU_AND_NE,
        "cpuAndGPU": ct.ComputeUnit.CPU_AND_GPU,
        "cpuOnly": ct.ComputeUnit.CPU_ONLY,
    }
    precision_map = {
        "float16": ct.precision.FLOAT16,
        "float32": ct.precision.FLOAT32,
    }
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="pixels",
                               shape=(1, 3, image_size, image_size),
                               dtype=float)],
        compute_units=cu_map[args.compute_units],
        compute_precision=precision_map[args.precision],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS14,
    )

    mlmodel.short_description = f"Vision encoder ({args.model_dir.name})"
    mlmodel.save(str(args.output))
    print(f"[convert] saved {args.output}")

    manifest = {
        "model_dir": str(args.model_dir),
        "output": str(args.output),
        "image_size": image_size,
        "patch_size": patch_size,
        "hidden": hidden,
        "out_hidden": out_hidden,
        "compute_units": args.compute_units,
        "precision": args.precision,
        "input_name": "pixels",
        "input_shape": [1, 3, image_size, image_size],
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[convert] manifest at {manifest_path}")


def find_vision_tower(model):
    """Best-effort lookup of the vision encoder submodule on a multimodal HF model."""
    candidates = [
        "vision_tower",
        "vision_model",
        "model.vision_tower",
        "model.vision_model",
        "visual",
        "vision",
    ]
    for path in candidates:
        obj = model
        ok = True
        for part in path.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok:
            print(f"[convert] using vision module path: {path}")
            return obj
    # Last resort: dump module structure for inspection.
    print("[convert] vision tower path not found. Module tree:")
    for name, _ in model.named_modules():
        print(" ", name)
    return None


if __name__ == "__main__":
    main()
