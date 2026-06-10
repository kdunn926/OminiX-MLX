#!/usr/bin/env python3
"""Convert a vanilla HF ViT to Core ML for the ANE spike validation.

Used to prove the end-to-end FFI + ANE dispatch path before tackling the
multimodal-specific patchification of GLM-OCR / Qwen-VL towers.

Usage:
    uv run --isolated --with torch==2.7.0 --with transformers --with coremltools \
        python ane-vit-spike/converters/convert_simple_vit.py \
            --output ane-vit-spike/vit-base.mlpackage
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/vit-base-patch16-224",
                   help="HF model id; cached to ~/.cache/huggingface")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--compute-units", default="all",
                   choices=["all", "cpuAndNeuralEngine", "cpuAndGPU", "cpuOnly"])
    p.add_argument("--precision", default="float16",
                   choices=["float16", "float32"])
    args = p.parse_args()

    import torch
    import coremltools as ct
    from transformers import ViTModel

    print(f"[convert] loading {args.model} ...")
    model = ViTModel.from_pretrained(args.model, add_pooling_layer=False)
    model.eval()

    # Wrap so the traced module returns just the hidden tensor (HF
    # returns a BaseModelOutput dataclass which Core ML can't represent).
    class Wrap(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, pixels):
            out = self.inner(pixel_values=pixels)
            return out.last_hidden_state

    wrapped = Wrap(model).eval()

    # google/vit-base-patch16-224: input is 224x224x3.
    dummy = torch.zeros(1, 3, 224, 224, dtype=torch.float32)
    print("[convert] tracing ...")
    traced = torch.jit.trace(wrapped, dummy, strict=False)

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

    print(f"[convert] converting (compute_units={args.compute_units}, precision={args.precision}) ...")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="pixels", shape=(1, 3, 224, 224), dtype=float)],
        compute_units=cu_map[args.compute_units],
        compute_precision=precision_map[args.precision],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS14,
    )
    mlmodel.short_description = f"ViT-base (224, patch16) — ANE spike test"
    mlmodel.save(str(args.output))
    print(f"[convert] saved {args.output}")

    manifest = {
        "model": args.model,
        "output": str(args.output),
        "image_size": 224,
        "patch_size": 16,
        "input_name": "pixels",
        "input_shape": [1, 3, 224, 224],
        "compute_units": args.compute_units,
        "precision": args.precision,
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
