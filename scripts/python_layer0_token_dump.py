#!/usr/bin/env python3
"""
Python mirror of `qwen3.6-mlx/examples/target_layer0_token_dump.rs`.

Dumps layer-0 internals for a fixed token window so the Rust and Python
quantized projection paths can be compared byte-for-byte on identical inputs.

Usage:
    /Users/kyle/miniforge3/bin/python scripts/python_layer0_token_dump.py \
        --target models/Qwen3.6-35B-A3B-4bit \
        --tokens 785,5754,315,4586,1351 \
        --out /tmp/qwen_target_layer0_token_dump_py.safetensors
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load_model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--cpu", action="store_true", help="force CPU device")
    parser.add_argument(
        "--tokens",
        required=True,
        help="comma-separated u32 token IDs",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="output safetensors path",
    )
    args = parser.parse_args()
    if args.cpu:
        mx.set_default_device(mx.cpu)

    token_ids = [int(x) for x in args.tokens.split(",") if x.strip()]
    if not token_ids:
        print("--tokens must not be empty", file=sys.stderr)
        return 2

    tokens = mx.array([token_ids], dtype=mx.uint32)

    model, _config = load_model(args.target)
    text_model = model.language_model.model
    layer0 = text_model.layers[0]

    embeddings = text_model.embed_tokens(tokens)
    layer0_input_norm = layer0.input_layernorm(embeddings)

    linear_attn = layer0.linear_attn
    qkv_direct = linear_attn.in_proj_qkv(layer0_input_norm)
    z_direct = linear_attn.in_proj_z(layer0_input_norm)
    a_direct = linear_attn.in_proj_a(layer0_input_norm)
    b_direct = linear_attn.in_proj_b(layer0_input_norm)

    tensors = {
        "tokens": tokens.astype(mx.int32),
        "embeddings": embeddings.astype(mx.float32),
        "layer0_input_norm": layer0_input_norm.astype(mx.float32),
        "qkv_direct": qkv_direct.astype(mx.float32),
        "z_direct": z_direct.astype(mx.float32),
        "a_direct": a_direct.astype(mx.float32),
        "b_direct": b_direct.astype(mx.float32),
    }

    for t in tensors.values():
        mx.eval(t)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(args.out), tensors, metadata={"target": str(args.target)})
    print(f"saved={args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
