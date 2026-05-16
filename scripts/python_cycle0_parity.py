#!/usr/bin/env python3
"""
Python mirror of dflash-mlx/examples/cycle0_parity.rs.

Runs prefill on the target, captures hidden at the DFlash target_layer_ids,
runs the DFlash draft model on the staged-token noise embedding, and dumps
every intermediate so it can be diffed against the Rust dump.

Usage:
    /Users/kyle/miniforge3/bin/python scripts/python_cycle0_parity.py \
        --target models/Qwen3.6-35B-A3B-4bit \
        --draft  models/Qwen3.6-35B-A3B-DFlash \
        --out    /tmp/dflash_cycle0_py.safetensors
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load_model, load_tokenizer

from dflash_mlx.engine.target_qwen_gdn import QwenGdnTargetOps
from dflash_mlx.model import DFlashDraftModel, DFlashDraftModelArgs

DEFAULT_PROMPT = (
    "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def load_draft(draft_dir: Path) -> DFlashDraftModel:
    with (draft_dir / "config.json").open() as fh:
        cfg = json.load(fh)
    args = DFlashDraftModelArgs.from_dict(cfg)
    model = DFlashDraftModel(args)
    weights = mx.load(str(draft_dir / "model.safetensors"))
    sanitized = {k: v for k, v in weights.items()}
    if hasattr(model, "sanitize"):
        sanitized = model.sanitize(sanitized)
    model.load_weights(list(sanitized.items()), strict=False)
    mx.eval(model.parameters())
    return model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--draft", required=True, type=Path)
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
    draft_model = load_draft(args.draft)

    target_ops = QwenGdnTargetOps()
    target_layer_ids = draft_model.target_layer_ids
    block_len = int(draft_model.block_size)
    mask_token_id = int(draft_model.mask_token_id)

    # capture hidden at config.dflash_config.target_layer_ids
    # (the +1 offset is handled inside forward_with_hidden_capture)
    capture_layer_set = {lid + 1 for lid in target_layer_ids}
    capture_layer_set.add(0)  # ensure we have layer-0 input too for sanity
    cache = [None] * len(target_ops.text_model(target_model).layers)
    prefill_logits_full, captured = target_ops.forward_with_hidden_capture(
        target_model,
        input_ids=prompt_ids,
        cache=cache,
        capture_layer_ids=capture_layer_set,
        logits_last_only=False,
    )
    mx.eval(prefill_logits_full)
    prefill_logits = prefill_logits_full[:, -1, :]

    # Rust harness samples the staged token from prefill_logits (greedy at temp=0)
    staged_token = int(mx.argmax(prefill_logits, axis=-1).item())

    # Rebuild raw_target_hidden = concat of captures at target_layer_ids
    raw_target_hidden = target_ops.extract_context_feature(captured, target_layer_ids)
    mx.eval(raw_target_hidden)

    # Embed mask + staged tokens
    embed_tokens = target_ops.embed_tokens(target_model)
    mask_emb = embed_tokens(mx.array([[mask_token_id]], dtype=mx.uint32))
    staged_emb = embed_tokens(mx.array([[staged_token]], dtype=mx.uint32))

    # noise_emb = [staged_emb, mask × (block_len-1)]
    mask_tail = mx.broadcast_to(mask_emb, (1, block_len - 1, mask_emb.shape[-1]))
    noise_emb = mx.concatenate([staged_emb, mask_tail], axis=1)

    # Draft forward — uses the projected target context.
    projected_target_hidden = draft_model.project_target_hidden(raw_target_hidden)
    draft_hidden = draft_model.forward_projected_context(
        noise_embedding=noise_emb,
        draft_context=projected_target_hidden,
        cache=None,
    )
    # Predictions are at draft_hidden[..., 1:, :]; matmul against the target's
    # embed_tokens.as_linear (tied embedding) to get logits.
    prediction_hidden = draft_hidden[:, 1:, :]
    draft_logits = target_ops.logits_from_hidden(target_model, prediction_hidden)
    draft_tokens = mx.argmax(draft_logits, axis=-1).astype(mx.uint32)

    tensors = {
        "prompt_ids": prompt_ids,
        "block_len": mx.array([block_len], dtype=mx.int32),
        "staged_token": mx.array([staged_token], dtype=mx.uint32),
        "prefill_logits": prefill_logits,
        "raw_target_hidden": raw_target_hidden,
        "projected_target_hidden": projected_target_hidden,
        "noise_emb": noise_emb,
        "draft_hidden": draft_hidden,
        "draft_logits": draft_logits,
        "draft_tokens": draft_tokens,
    }
    for t in tensors.values():
        mx.eval(t)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(args.out),
        tensors,
        metadata={
            "target": str(args.target),
            "draft": str(args.draft),
            "prompt": args.prompt,
        },
    )
    print(f"Saved Python cycle-0 parity tensors to {args.out}")
    print(f"staged_token={staged_token} block_len={block_len}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
