#!/usr/bin/env python3
"""
Run Python dflash_mlx on a fixed prompt with deterministic sampling and
dump per-cycle stats (block_len, acceptance_len, commit_count) as JSON.

Counterpart to dflash-mlx/examples/trace_dflash_verify.rs — lets us
diff cycle-level behavior between Python and Rust on the same prompt.

Usage:
    /Users/kyle/miniforge3/bin/python scripts/python_dflash_trace.py \
        --model models/Qwen3.6-35B-A3B-4bit \
        --draft models/Qwen3.6-35B-A3B-DFlash \
        --prompt "The theory of general relativity" \
        --max-tokens 80 \
        --out /tmp/dflash_trace_py.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx

from dflash_mlx.engine.events import (
    CycleCompleteEvent,
    SummaryEvent,
    TokenEvent,
)
from dataclasses import replace

from dflash_mlx.diagnostics import DiagnosticsConfig, TraceConfig
from dflash_mlx.runtime import get_stop_token_ids, stream_dflash_generate
from dflash_mlx.runtime.bundle import load_runtime_bundle
from dflash_mlx.runtime.context import build_offline_runtime_context


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Target model dir")
    parser.add_argument("--draft", required=True, help="Draft model dir")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=80)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    runtime_context = build_offline_runtime_context()
    runtime_context = replace(
        runtime_context,
        diagnostics=DiagnosticsConfig(
            mode="full", trace=TraceConfig(cycle_events=True)
        ),
    )
    bundle = load_runtime_bundle(
        model_ref=args.model,
        draft_ref=args.draft,
        draft_quant=None,
        verify_config=runtime_context.verify,
    )

    stop_tokens = get_stop_token_ids(bundle.tokenizer)
    stream = stream_dflash_generate(
        target_model=bundle.target_model,
        target_ops=bundle.target_ops,
        tokenizer=bundle.tokenizer,
        draft_model=bundle.draft_model,
        draft_backend=bundle.draft_backend,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        use_chat_template=False,
        stop_token_ids=stop_tokens,
        runtime_context=runtime_context,
    )

    cycles: list[dict] = []
    tokens: list[int] = []
    summary: dict | None = None

    event_types: dict[str, int] = {}
    try:
        for event in stream:
            event_types[type(event).__name__] = event_types.get(type(event).__name__, 0) + 1
            if isinstance(event, CycleCompleteEvent):
                cycles.append(event.to_payload())
            elif isinstance(event, TokenEvent):
                tokens.append(int(event.token_id))
            elif isinstance(event, SummaryEvent):
                summary = {
                    "generation_tokens": int(event.generation_tokens),
                    "acceptance_ratio": float(event.acceptance_ratio),
                    "total_cycles": int(getattr(event, "total_cycles", -1)),
                }
    finally:
        close = getattr(stream, "close", None)
        if close is not None:
            close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {"cycles": cycles, "tokens": tokens, "summary": summary},
            indent=2,
        )
    )
    print(f"saved={args.out}")
    print(f"event_types={event_types}")
    print(f"cycles={len(cycles)} tokens={len(tokens)}")
    if summary is not None:
        print(
            f"acceptance_ratio={summary['acceptance_ratio']:.4f} "
            f"generation_tokens={summary['generation_tokens']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
