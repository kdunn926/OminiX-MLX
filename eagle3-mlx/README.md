# eagle3-mlx

EAGLE-3 speculative decoding for MLX Rust, loading [speculators-format](https://github.com/vllm-project/speculators)
draft checkpoints. Currently wired for the Gemma4 family via
`RedHatAI/gemma-4-26B-A4B-it-speculator.eagle3` (a 0.9B single-layer draft head
trained against `google/gemma-4-26B-A4B-it`).

The whole feature is **env-gated**: nothing engages unless `OMINIX_EAGLE3=1`.

## How it works

EAGLE-3 drafts in the target's feature space instead of running a separate
small LM:

1. During every target forward (prefill + verify), the residual stream
   entering target layers `[2, 15, 27]` is captured (reusing the DFlash
   hidden-capture path in `Gemma4TargetAdapter`; note the capture API hooks
   layer *outputs*, so it is fed ids `[1, 14, 26]`).
2. The draft fuses the three captures through `fc` (`[3H] -> [H]`) and runs a
   single llama-style decoder layer whose input is
   `concat(norm(embed(token_{i+1})), norm(fused_i))` — token/hidden pairs are
   shifted by one, RoPE position = the hidden's position.
3. Chain steps feed the layer's own pre-norm output back as the next step's
   feature (no `fc`), drafting greedily over a reduced 32k vocab that maps to
   target ids via `d2t` offsets.
4. The shared `dflash_mlx::DFlashSession` cycle verifies each block on the
   target, with greedy longest-prefix acceptance at `T=0` and
   distribution-preserving acceptance at `T>0`.

The draft KV cache persists across cycles: speculative chain rows are trimmed
on rollback and the newly committed span is re-ingested from *target-derived*
hiddens (the llama.cpp `seq_rm` + re-seed strategy). Reference
implementations: vLLM `llama_eagle3.py`, llama.cpp commit `88a3927`.

## Usage

```bash
# one-time: fetch the draft head (~1.7 GB)
hf download RedHatAI/gemma-4-26B-A4B-it-speculator.eagle3 \
    --local-dir ./models/gemma-4-26B-A4B-it-eagle3

OMINIX_EAGLE3=1 cargo run --release -p eagle3-mlx --example eagle3_generate -- \
    ./models/gemma4-26B-a4b-it-UD-MLX-4bit \
    ./models/gemma-4-26B-A4B-it-eagle3 \
    "Explain speculative decoding in two sentences."
```

### Env knobs

| var | effect |
|-----|--------|
| `OMINIX_EAGLE3=1` | master gate; `Eagle3Session::load` refuses without it |
| `EAGLE3_DRAFT_DIR` | default draft checkpoint dir for the example |
| `EAGLE3_BLOCK` | draft chain length (default: checkpoint's `speculative_tokens` = 3; the head was trained with 3 TTT steps, so much deeper chains exceed the training horizon) |
| `EAGLE3_MAX_TOKENS` / `EAGLE3_TEMP` | example-only generation knobs |

## Model gating

`Eagle3Session::load` validates the pairing before anything runs: the draft's
`speculators_config.verifier` must name a Gemma4 model, aux layer ids must fit
the target's layer count, and hidden sizes must match. Other model families
fail at load with a clear error — wiring a new family means adding its target
adapter (the draft model + session loop are family-agnostic).

## Results (M-series, gemma4-26B-A4B UD-MLX-4bit target, greedy)

300-token explanation prompt:

| path | decode tok/s | acceptance |
|------|--------------|------------|
| autoregressive (`ar_bench`, layered KV) | 32.0 | — |
| autoregressive (`ar_bench`, flat KV) | 32.2 | — |
| EAGLE-3, block 2 | 35.4 | 0.48 |
| EAGLE-3, block 3 (default) | 36.3 | 0.48 |
| EAGLE-3, block 4 | 35.5 | 0.47 |
| EAGLE-3, block 5 | 34.9 | 0.45 |

A 1.13x speedup over AR. Short factual answers reach 0.83 acceptance. The
modest ratio is the same MoE economics dflash hit: with only 3.8B active
params the AR step is already cheap, while each verify block pays the expert
gather for `block+1` positions and the draft's bf16 lm_head/embedding adds
~0.5 GB of reads per cycle (quantizing the draft head is the obvious next
optimization). Greedy speculative decoding is output-identical to greedy AR
by construction (every committed token is argmax'd from target logits).

## Known limits

- Linear chains only (tree drafting was evaluated for this codebase and closed
  as a regression; see `gemma4-pair-adapter-wip.md`).
- The target adapter uses the flat `KVCache`; contexts past the 1024-token
  sliding window inherit the open `chat_gemma4_ud` flat-cache finding in
  `review-findings.md` until that bug is resolved.
- `Eagle3Session` disables the DFlash hidden-segment cap
  (`DFLASH_MAX_HIDDEN_SEGS=0`) unless overridden, so the hidden accumulator
  grows with generation length (~17 KB/token at 3x2816 bf16).
