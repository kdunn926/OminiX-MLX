#!/usr/bin/env python3
"""Python mirror of qwen3.6-mlx/examples/target_moe_dump.rs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load_model, load_tokenizer

DEFAULT_PROMPT = (
    "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.target)
    prompt_ids_list = tokenizer.encode(args.prompt, add_special_tokens=False)
    prompt_ids = mx.array([prompt_ids_list], dtype=mx.uint32)

    target_model, _ = load_model(args.target)
    text_model = target_model.language_model.model
    layer0 = text_model.layers[0]
    layer_k = text_model.layers[args.layer]
    moe = layer_k.mlp

    embeddings = text_model.embed_tokens(prompt_ids)
    mlp_in = layer0.input_layernorm(embeddings)  # bit-exact across Rust/Python

    # MoE forward — manually capture intermediates with the same names as Rust.
    gates_raw = moe.gate(mlp_in)
    gates = mx.softmax(gates_raw, axis=-1, precise=True)
    k = moe.top_k
    top_k_indices = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    top_k_scores_raw = mx.take_along_axis(gates, top_k_indices, axis=-1)
    top_k_scores = top_k_scores_raw / top_k_scores_raw.sum(axis=-1, keepdims=True)
    moe_out = moe(mlp_in)

    tensors = {
        "mlp_in": mlp_in.astype(mx.float32),
        "gates_raw": gates_raw.astype(mx.float32),
        "gates": gates.astype(mx.float32),
        "top_k_indices": top_k_indices.astype(mx.int32),
        "top_k_scores": top_k_scores.astype(mx.float32),
        "moe_out": moe_out.astype(mx.float32),
    }
    for t in tensors.values():
        mx.eval(t)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(args.out),
        tensors,
        metadata={"target": str(args.target), "layer": str(args.layer)},
    )
    print(f"saved={args.out} layer={args.layer}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
