# Prefill / Decode Performance Sweep Plan

Goal: measure how the knobs touched in the recent fix session affect **prefill
TTFT** (separated from **decode tok/s**) and, where applicable, **speculative
acceptance**, across the available Qwen3.6 and Gemma4 checkpoints.

Do **not** run blindly — model loads are expensive (tens of seconds each, 4–35B
weights). The companion script `scripts/bench_sweep.sh` defaults to
`DRY_RUN=1`, which only prints the commands it *would* run. Flip to
`DRY_RUN=0` to execute.

All facts below were verified against source on branch
`spike/async-vision-prefill`; flags and env-var names are quoted from the
example sources, not guessed.

---

## 1. Benchmark entry points (verified)

### 1a. `bench_mtplx` — MTPLX speculative decoding (Qwen3.6 + Gemma4 AR)
- Source: `mtplx-mlx/examples/bench_mtplx.rs`
- Package: `mtplx-mlx`. Binary: `./target/release/examples/bench_mtplx`
- Build: `cargo build --release -p mtplx-mlx --example bench_mtplx`
- Invocation:
  ```
  ./target/release/examples/bench_mtplx \
    --target <model_dir> [--prompt "..."|<fixture.json>] \
    [--max-tokens 100] [--temp 0.0] [--greedy|--speculative]
  ```
- Flags (exact): `--target`, `--prompt` (string, or a path ending in `.json`
  rendered as an OpenAI chat fixture), `--max-tokens` (default 100), `--temp`
  (default 0.0), `--greedy` / `--speculative` (mutually exclusive; if neither
  given, auto-picks Speculative when temp>0 else Greedy).
- Env (exact): `MTPLX_BLOCK_LEN` (default 4) → `SpeculativeConfig.block_len`.
- Routing: reads `config.json` `model_type`; `gemma*` → AR-only Gemma4 path
  (no MTP head), otherwise the Qwen3.6 `MtplxSession`.
- Metric lines printed to **stdout**:
  ```
  prefill_s=<f> decode_s=<f> decode_tok_per_s=<f>
  total_tokens=<n> mtp_cycles=<n> ar_fallback_steps=<n> mtp_drafted=<n> mtp_accepted=<n> acceptance_rate=<f>
  ```
  (Gemma4 AR path prints the same `prefill_s=… decode_s=… decode_tok_per_s=…`
  line, then `total_tokens=… mtp_cycles=0 ar_fallback_steps=…`.)
- KV-cache env (`TURBO_KV` / `QUANTIZE_KV`) is **not** read by this example —
  it goes through `MtplxSession`, which selects its own cache. Treat KV-mode as
  fixed (session default) for the MTPLX cells.

### 1b. `bench_dflash` — DFlash speculative decoding vs AR (Qwen3.6 + Gemma4)
- Source: `dflash-mlx/examples/bench_dflash.rs`
- Package: `dflash-mlx`. Binary: `./target/release/examples/bench_dflash`
- Build: `cargo build --release -p dflash-mlx --example bench_dflash`
- Invocation:
  ```
  ./target/release/examples/bench_dflash \
    --target <model_dir> [--draft <draft_dir>] \
    [--prompt "..."] [--fixture <chat.json>] \
    [--max-tokens 200] [--temp 0.7] [--cpu] \
    [--ddtree --tree-budget 16 --tree-topk 4 --tree-naive]
  ```
- Flags (exact): `--target`, `--draft`, `--prompt`, `--fixture`,
  `--max-tokens` (default 200), `--temp` (default 0.7), `--cpu`, `--ddtree`,
  `--tree-budget` (default 16), `--tree-topk` (default 4), `--tree-naive`.
- Block length is **not** a CLI flag for DFlash — it comes from the draft
  checkpoint (`draft_model.args.block_size()`); the adaptive policy
  (`SpeculativeCycleConfig`) reduces it per cycle. `--ddtree`/`--tree-budget`/
  `--tree-topk` are the tunable tree knobs.
- Env (exact): `TURBO_KV` (TurboQuant KV) and `QUANTIZE_KV` (K=q8/V=q4) are
  read by the AR baseline (`run_autoregressive`) and by the Gemma4 DFlash path.
  `TURBO_KV` + `--ddtree` is rejected (TurboQuantKVCache lacks `compact()`).
- Draft resolution: `--draft`, else sibling `<target>-DFlash` dir, else
  `dflash_draft_path` in config; missing → AR-only line + notice; present but
  unsupported arch → MockDraftAdapter.
- Metric lines printed to **stdout**:
  ```
  Autoregressive: prefill_s=<f> decode_tok_s=<f> total_tokens=<n>
  DFlash: prefill_s=<f> decode_tok_s=<f> acceptance_ratio=<f> avg_block_len=<f> total_tokens=<n> | adaptive: Large=<n> Reduced=<n> Probe=<n>
  Speedup ratio: <f>x
  ```
  (Gemma4 lines are prefixed `DFlash(gemma4):` / `DFlash(gemma4+TQ):`; no AR
  baseline for Gemma4, so no Speedup line there.)

### 1c. `qwen3-6 generate` — plain AR with KV-mode toggle (TurboQuant driver)
- Source: `qwen3.6-mlx/examples/generate.rs`
- Package: `qwen3-6-mlx` (dir `qwen3.6-mlx`). Binary:
  `./target/release/examples/generate`
- Build: `cargo build --release -p qwen3-6-mlx --example generate`
- Invocation: `./target/release/examples/generate <model_dir> "<prompt>" <max_tokens> [--cpu]`
  (positional: arg1 model dir, arg2 prompt string, arg3 max tokens; no
  `--prompt`/`--max-tokens` flags here.)
- Env (exact): `TURBO_KV` → TurboQuant KV; `QUANTIZE_KV` → quantized (K=q8/V=q4);
  unset → standard fp16 (`KVCache`). Also `QWEN36_PROMPT_CACHE_DIR`.
- **This is the only LLM example that exposes the full KV-mode + TurboQuant env
  matrix on Qwen3.6.** It is therefore the harness used for the
  TURBOQUANT_SINK_TOKENS / SIMD_MATMUL / FUSED_KV_MIN cells.
- Metrics: printed to **stderr** as a prose line (no fused machine line):
  ```
  TTFT: <f>s | Decode: <n> tok in <f>s (<f> tok/s) | Total: <n> tok in <f>s
  ```
  The script parses TTFT → prefill_s and the `(<f> tok/s)` group → decode_tps.
  `kv_backend: …` is also emitted on stderr for provenance.

### 1d. `qwen3 bench` — clean machine-readable AR line (qwen3, not 3.6)
- Source: `qwen3-mlx/examples/bench.rs`. Package `qwen3-mlx`.
- Build: `cargo build --release -p qwen3-mlx --example bench`
- Invocation: `./target/release/examples/bench bench <model_dir> <prompt_len> <max_new>`
  (`prompt_len` is a synthesized **token count** — best harness for a clean
  short-vs-long prefill sweep, but it is the *old qwen3* model and uses
  `KVCache` only; no TurboQuant env.) Listed here for reference; not in the
  default sweep since the session's knobs live on the 3.6/Gemma4 path.

### 1e. `bench_kv_quant` — fixed fp16-vs-quantized table (reference only)
- Source: `qwen3.6-mlx/examples/bench_kv_quant.rs`. No env knobs, no TurboQuant;
  iterates ctx ∈ {256,1024,4096} × {fp16, q8k/q4v} internally. Not driven by the
  sweep (it can't vary the session's new knobs), but useful as a sanity cross-check.

---

## 2. Model inventory (verified on disk)

`models/` dir; `model_type` from each `config.json`; artifacts listed by name.

| Model dir | model_type | MTP head | DFlash | Drives via |
|---|---|---|---|---|
| `Qwen3.6-27B-4bit` | qwen3_5 | no | no | generate (KV-mode sweep) |
| `Qwen3.6-35B-A3B-4bit` | qwen3_5_moe | no | no | generate (KV-mode sweep), bench_dflash (AR + Mock/Missing draft) |
| `Qwen3.6-27B-MTPLX-Optimized-Speed` | qwen3_5 | **yes** (`mtp.safetensors`, `mtplx_runtime.json`) | no | **bench_mtplx** |
| `Qwen3.6-27B-DFlash` | qwen3 | no | (draft repo) | bench_dflash `--target Qwen3.6-27B-4bit --draft this` (see note) |
| `Qwen3.6-35B-A3B-DFlash` | qwen3 | no | `dflash.py` (Python draft) | bench_dflash draft; likely **Mock** fallback (non-native arch) |
| `Gemma4-27B-MTPLX-Optimized-Speed` | (mtplx_pair.json) | paired | no | bench_mtplx (Gemma4 AR path; paired-MTP TBD) |
| `gemma-4-26B-A4B-it` | gemma4 | no | no | bench_mtplx (Gemma4 AR), bench_dflash (Gemma4 + TURBO_KV), ar_bench |
| `gemma-4-26B-A4B-it-DFlash` | qwen3 | no | DFlash draft for gemma | bench_dflash `--target gemma-4-26B-A4B-it --draft this` |
| `gemma-4-E4B-it` | gemma4_audio | no | no | (audio variant — skip for text perf) |

Notes:
- The `-DFlash` dirs are **draft** checkpoints (`model_type: qwen3`,
  `dflash.py`), passed via `--draft`, *not* as `--target`. Several are
  Python-arch and will trigger the `MockDraftAdapter` fallback (still useful for
  pipeline TTFT/decode timing, but acceptance is synthetic).
- `Gemma4-27B-MTPLX-Optimized-Speed` carries `mtplx_pair.json` (paired-model
  layout); `bench_mtplx` currently runs Gemma4 AR-only (paired MTP is TBD), so
  it measures AR throughput, not speculation.
- The script skips any model dir that is missing (resumable / skip-on-missing).

---

## 3. Sweep matrix

Two prompt regimes to separate prefill from decode:
- **short**: `"The theory of general relativity"` (~6 tokens) → decode-dominated.
- **long**: a ~1–2k-token synthetic prompt (env `LONG_PROMPT`, default is a long
  repeated paragraph passed via `--prompt`) → prefill/TTFT-dominated.

`max-tokens`: `64` (quick) and `256` (full) — enough decode steps for a stable
tok/s without dominating wall-clock.

### Cell A — TurboQuant KV + new SDPA knobs (driver: `generate`, Qwen3.6)
Hold: model = `Qwen3.6-27B-4bit` (quick) / + `Qwen3.6-35B-A3B-4bit` (full),
temp=0, max-tokens as above.
Vary:
- KV mode ∈ { *unset*=fp16(KVCache), `QUANTIZE_KV=1`, `TURBO_KV=1` }
- For `TURBO_KV=1` only, sub-sweep:
  - `TURBOQUANT_SINK_TOKENS` ∈ {4 (default), 0 (enable fully-fused single-dispatch SDPA)}
  - `TURBOQUANT_SIMD_MATMUL` ∈ { *unset*, `=1` }
  - `TURBOQUANT_FUSED_KV_MIN`: default 8192; for the **long** prompt also try
    `=512` to actually engage the fused/online path at moderate context
    (default 8192 won't trigger on a ~1–2k prompt).
- prompt ∈ {short, long}
This is the cell that directly exercises the session's TurboQuant fixes.
fully-fused path requires `SINK_TOKENS=0` **and** kv_len ≥ FUSED_KV_MIN.

### Cell B — MTPLX speculative (driver: `bench_mtplx`)
Model = `Qwen3.6-27B-MTPLX-Optimized-Speed` (only local checkpoint with an MTP head).
Vary:
- `MTPLX_BLOCK_LEN` ∈ {1, 4, 8}
- acceptance/temp ∈ { `--greedy --temp 0.0` (lossless greedy accept),
  `--speculative --temp 0.7` (distribution-preserving rejection sampling) }
- prompt ∈ {short, long}; max-tokens as above.
Captures `acceptance_rate`. block_len=1 ≈ AR baseline for speedup reference.

### Cell C — DFlash (driver: `bench_dflash`)
Targets: `Qwen3.6-35B-A3B-4bit` (+ draft `Qwen3.6-35B-A3B-DFlash`),
`gemma-4-26B-A4B-it` (+ draft `gemma-4-26B-A4B-it-DFlash`).
Vary:
- temp ∈ {0.0 (greedy accept), 0.7 (default)}
- DDTree ∈ { off, `--ddtree --tree-budget 16 --tree-topk 4` }
- KV mode for the Gemma4 target: { *unset*, `TURBO_KV=1` } (note: `TURBO_KV`
  + `--ddtree` is rejected, so that combo is skipped automatically).
- prompt ∈ {short, long}; max-tokens as above.
Captures `acceptance_ratio`, `avg_block_len`, adaptive `Large/Reduced/Probe`,
and the AR-vs-DFlash `Speedup ratio` (Qwen3.6 only).

### Quick vs Full
- **quick** (`SWEEP=quick`, default): one small model per cell, short prompt
  only, max-tokens=64, KV modes {unset, TURBO_KV}, SINK_TOKENS {0,4},
  MTPLX_BLOCK_LEN {1,4}, temp {0.0}. ~20–30 runs.
- **full** (`SWEEP=full`): both prompts, both max-tokens, all KV/SIMD/KV_MIN
  combos, MTPLX_BLOCK_LEN {1,4,8}, both temps, DDTree on/off. Order ~hundreds
  of cells — run overnight; the script is resumable so partial CSVs are safe.

---

## 4. Metrics captured per run

Into a CSV (`results/bench_sweep.csv`) and an echo'd JSONL
(`results/bench_sweep.jsonl`), one row per cell:

`timestamp,sweep,model,harness,prompt,max_tokens,kv_mode,sink_tokens,simd_matmul,fused_kv_min,block_len,accept_mode,temp,ddtree,prefill_s,decode_tps,total_tokens,accept_rate,extra`

- `prefill_s` — from `prefill_s=` (mtplx/dflash) or `TTFT: <f>s` (generate).
- `decode_tps` — from `decode_tok_per_s=` (mtplx) / `decode_tok_s=` (dflash) /
  `(<f> tok/s)` (generate).
- `total_tokens` — from `total_tokens=`.
- `accept_rate` — `acceptance_rate=` (mtplx) / `acceptance_ratio=` (dflash);
  blank for plain AR cells.
- `extra` — free-form (avg_block_len, adaptive counts, speedup, kv_backend
  provenance) when present.
- Peak memory: not printed by any of these examples today, so it is not
  captured. If a future build adds a `peak_mem` line the parser's `extra`
  field will pick up a `peak_mem=` token (the script greps for it defensively).

Each cell is run **twice** by default (`REPS=2`): first run warms allocator /
Metal pipeline caches; the script records both and you keep the 2nd.

---

## 5. Operational notes

- Build all needed examples first (the script does this unless `SKIP_BUILD=1`):
  ```
  cargo build --release -p mtplx-mlx   --example bench_mtplx
  cargo build --release -p dflash-mlx  --example bench_dflash
  cargo build --release -p qwen3-6-mlx --example generate
  ```
- Binaries live at `./target/release/examples/<name>` (workspace shared target).
- `DRY_RUN=1` (default) prints commands only — safe to inspect before paying for
  loads. `DRY_RUN=0` executes.
- Resumable: a cell whose row (keyed by its full param tuple) already exists in
  the CSV is skipped. Delete the CSV to start fresh.
- Skip-on-missing: any `--target`/`--draft` dir that does not exist is logged and
  skipped, no failure.
