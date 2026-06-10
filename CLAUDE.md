# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

OminiX-MLX is a Rust Cargo workspace for ML inference on Apple Silicon, built on
Apple's MLX framework. It contains ~30 member crates: the MLX Rust bindings
(`mlx-rs` and below), one shared inference library (`mlx-rs-core`), and a
dedicated crate per model family (LLMs, VLMs, ASR, TTS, image gen). Inference is
pure Rust with no Python runtime dependency.

## Prerequisites & build

- macOS 14+, Apple Silicon (M1–M4), Rust 1.85+, Xcode Command Line Tools.
- The MLX C library is a git submodule at `mlx-rs/mlx-sys/src/mlx-c`. After clone:
  `git submodule update --init --recursive`. `mlx-sys` builds it via CMake in its
  build script.
- `mlx-rs` is consumed by model crates with `features = ["metal", "accelerate"]`
  (Metal GPU backend + Apple Accelerate). This is the default everywhere.

```bash
cargo build --release                  # all crates (first build compiles mlx-c via CMake — slow)
cargo build --release -p qwen3-mlx     # one crate
cargo test -p mlx-rs-core              # test a single crate
cargo test -p qwen3-mlx <test_name>    # single test by name filter
```

Most crates ship runnable examples rather than binaries. Run them with:

```bash
cargo run --release -p <crate> --example <name> -- <args>
# e.g.
cargo run --release -p qwen3-mlx --example generate_qwen3 -- ./models/Qwen3-4B "Hello"
```

Always build/run with `--release`; debug MLX inference is unusably slow. Models
live under `./models/` (gitignored); examples take a model directory path as an
argument.

## Architecture: the layer stack

Crates depend strictly downward through these layers:

1. **`mlx-rs/mlx-sys`** — bindgen FFI to the `mlx-c` submodule.
2. **`mlx-rs`** — safe Rust API: `Array`, ops, NN layers, transforms (`eval`, lazy
   evaluation). Also `mlx-rs/mlx-macros`, `mlx-internal-macros`, `mlx-lm-utils`.
3. **`mlx-rs-core`** — the shared inference layer that every model crate builds on.
   Read this crate first; it defines the contracts model crates implement.
4. **Model crates** (`qwen3-mlx`, `glm4-moe-mlx`, `qwen3-tts-mlx`, `flux-klein-mlx`,
   …) — model definition + weight loading + examples. They depend on
   `mlx-rs-core`, never on each other.

### mlx-rs-core is the integration point

`mlx-rs-core/src/lib.rs` defines the traits and generic generation loop that unify
the LLM crates. Key pieces:

- **`ModelInput` / `ModelOutput` traits** + `ModelInputBuilder` — a model crate
  implements these so its forward pass plugs into the shared generator.
- **`generate::Generate`** (`generate/mod.rs`) — a builder-pattern, generic
  autoregressive generation loop parameterized over model `M`, input `I`, sampler
  `S`, cache `C`, and state `T`. New LLM crates should reuse this rather than
  hand-rolling a decode loop. (See memory note: gemma4 is being migrated onto this.)
- **`cache`** — KV cache variants: `ConcatKeyValueCache` (default), `KVCache`,
  `QuantizedKVCache`, `TurboQuantKVCache`.
- **`metal_kernels`** — custom Metal kernels shared across models: `fused_swiglu`
  (MoE), `fused_modulate` (DiT), TurboQuant SDPA (`tq_sdpa_4bit*`), DeltaNet
  recurrence, MoE dense matmul.
- **`utils`** — `scaled_dot_product_attention`, `create_attention_mask`,
  `initialize_rope`. **`sampler`**, **`speculative`** (spec decoding), **`audio`**
  (mel/STFT), **`turboquant`**, **`memory`** (cache/wired-memory limits).
- `convert` module is behind the `convert` feature flag.

A typical model crate has `lib.rs` (loading + public API), `model.rs` (the
transformer), and per-variant files (e.g. `qwen3-mlx/src/{qwen2.rs, qwen3_moe.rs}`).

## ane-vit (current spike)

`ane-vit/` explores running vision encoders (ViT) on the Apple Neural Engine via
Core ML while the LLM stays on MLX/Metal. `ane-vit/coreml-bridge` is a workspace
member (FFI bridge to Core ML); `.mlpackage` directories are converted models with
`.manifest.json` sidecars. The active branch `spike/async-vision-prefill` works on
overlapping ANE vision prefill with GPU text prefill. See `ane-vit/README.md` and
`docs/coreml-tradeoffs.md` for rationale.

## Notes

- Commit messages in this repo do **not** include a `Co-Authored-By: Claude` trailer.
- `docs/` holds design notes and code reviews; several `*-wip.md` files at the repo
  root track in-progress work (dflash, gemma4 pair adapter). Check these before
  large changes to those areas.
- `ominix-api` is referenced in the README as the unified OpenAI-compatible server
  but is not currently a workspace member (not present in `Cargo.toml`).
- `xtask` is the workspace's build-task helper crate (run via `cargo run -p xtask`).
- Python scripts in `scripts/` are parity/debugging tools that dump and compare
  intermediate tensors against reference implementations — not part of the build.
</content>
</invoke>
