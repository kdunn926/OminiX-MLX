#!/usr/bin/env python3
"""
Python mirror of qwen3.6-mlx/examples/target_layer0_debug.rs.

Captures the same DeltaNet (linear_attn) layer-0 intermediates that the
Rust example dumps, so we can locate where exactly the per-step drift
starts. Uses the same access path as mlx_lm but extracts manually so we
can name each sub-tensor identically to the Rust dump.

Usage:
    /Users/kyle/miniforge3/bin/python scripts/python_layer0_debug.py \
        --target models/Qwen3.6-35B-A3B-4bit \
        --out /tmp/qwen_l0_debug_py.safetensors
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import load_model, load_tokenizer

DEFAULT_PROMPT = (
    "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if args.cpu:
        mx.set_default_device(mx.cpu)

    tokenizer = load_tokenizer(args.target)
    prompt_ids_list = tokenizer.encode(args.prompt, add_special_tokens=False)
    prompt_ids = mx.array([prompt_ids_list], dtype=mx.uint32)

    target_model, _config = load_model(args.target)
    text_model = target_model.language_model.model
    layer0 = text_model.layers[0]
    attn = layer0.linear_attn  # GatedDeltaNet

    embeddings = text_model.embed_tokens(prompt_ids)
    layer0_input_norm = layer0.input_layernorm(embeddings)
    inputs = layer0_input_norm
    B, S, _ = inputs.shape

    qkv = attn.in_proj_qkv(inputs)
    z_flat = attn.in_proj_z(inputs)
    z = z_flat.reshape(B, S, attn.num_v_heads, attn.head_v_dim)
    b = attn.in_proj_b(inputs)
    a = attn.in_proj_a(inputs)

    conv_state = mx.zeros(
        (B, attn.conv_kernel_size - 1, attn.conv_dim),
        dtype=inputs.dtype,
    )
    conv_input = mx.concatenate([conv_state, qkv], axis=1)
    conv_out_pre = attn.conv1d(conv_input)
    qkv_after_conv = nn.silu(conv_out_pre)

    q_flat, k_flat, v_flat = mx.split(
        qkv_after_conv, [attn.key_dim, 2 * attn.key_dim], axis=-1
    )
    q_heads = q_flat.reshape(B, S, attn.num_k_heads, attn.head_k_dim)
    k_heads = k_flat.reshape(B, S, attn.num_k_heads, attn.head_k_dim)
    v_heads = v_flat.reshape(B, S, attn.num_v_heads, attn.head_v_dim)

    inv_scale = attn.head_k_dim ** -0.5
    q_norm = (inv_scale ** 2) * mx.fast.rms_norm(q_heads, None, 1e-6)
    k_norm = inv_scale * mx.fast.rms_norm(k_heads, None, 1e-6)

    # gated_delta_update is opaque; capture its output but not internals.
    from mlx_lm.models.qwen3_5 import gated_delta_update
    out_lhd, _state = gated_delta_update(
        q_norm,
        k_norm,
        v_heads,
        a,
        b,
        attn.A_log,
        attn.dt_bias,
        None,
        None,
        use_kernel=True,
    )
    # out_lhd shape: (B, S, num_v_heads, head_v_dim)

    # norm + z gate inline (RMSNormGated semantics)
    normed = attn.norm(out_lhd, z)
    out_flat = normed.reshape(B, S, -1)
    out_proj = attn.out_proj(out_flat)

    # beta / decay equivalents (Rust dumps these separately)
    beta = mx.sigmoid(b)

    tensors = {
        "prompt_ids": prompt_ids,
        "embeddings": embeddings,
        "layer0_input_norm": layer0_input_norm,
        "qkv": qkv,
        "qkv_after_conv": qkv_after_conv,
        "z": z_flat,
        "a": a,
        "b": b,
        "beta": beta,
        "q_heads": q_heads,
        "k_heads": k_heads,
        "v_heads": v_heads,
        "q_norm": q_norm,
        "k_norm": k_norm,
        "output": out_lhd,
        "normed": normed,
        "out_proj": out_proj,
    }
    for t in tensors.values():
        mx.eval(t)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(args.out), tensors, metadata={"target": str(args.target)})
    print(f"saved={args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
