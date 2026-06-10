#!/usr/bin/env python3
"""Compare Python vs Rust layer-0 token dumps for DFlash target-parity analysis."""

from __future__ import annotations

import argparse
import sys

import mlx.core as mx


def stats(name: str, py: mx.array, rs: mx.array) -> None:
    if py.shape != rs.shape:
        print(f"{name:24} SHAPE MISMATCH py={py.shape} rs={rs.shape}")
        return
    diff = (py - rs).astype(mx.float32)
    abs_diff = mx.abs(diff)
    max_abs = float(mx.max(abs_diff))
    mean_abs = float(mx.mean(abs_diff))
    py_max = float(mx.max(mx.abs(py.astype(mx.float32))))
    rel = max_abs / py_max if py_max > 0 else 0.0
    print(
        f"{name:24} shape={tuple(py.shape)} max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} "
        f"py_absmax={py_max:.3e} relmax={rel:.3e}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--py", required=True)
    parser.add_argument("--rs", required=True)
    parser.add_argument(
        "--keys",
        default="embeddings,layer0_input_norm,qkv_direct,z_direct,a_direct,b_direct",
        help="comma-separated tensor keys to compare",
    )
    args = parser.parse_args()

    py = mx.load(args.py)
    rs = mx.load(args.rs)

    for key in args.keys.split(","):
        key = key.strip()
        if not key:
            continue
        if key not in py:
            print(f"{key:24} MISSING in --py file")
            continue
        if key not in rs:
            print(f"{key:24} MISSING in --rs file")
            continue
        stats(key, py[key], rs[key])

    return 0


if __name__ == "__main__":
    sys.exit(main())
