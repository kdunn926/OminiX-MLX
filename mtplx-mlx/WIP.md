# mtplx-mlx — WIP

Multi-Token Prediction speculative decoding for MLX Rust.

## Status (2026-05-15)

**The forward path is implemented; what's missing is the checkpoint
that ships MTP weights.** Once a `mlx-community` (or upstream) release
includes the `mtp.*` block, this crate will start drafting tokens with
zero code change — the loader picks them up automatically.

* Crate bootstrapped: `Cargo.toml`, `src/lib.rs`, `src/session.rs`,
  `examples/bench_mtplx.rs`.
* `qwen3.6-mlx` parses `mtp_num_hidden_layers` /
  `mtp_use_dedicated_embeddings` from both `text_config` and top-level
  config.
* `qwen3.6-mlx::mtp::MtpHead` now has a real `forward()` implementing
  the DeepSeek-V3 / Qwen3-Next layout: `enorm`/`hnorm` →
  `eh_proj(2H→H)` → one or more `TransformerBlock`s (reusing the host
  model's full attention + MoE/dense MLP loaders) →
  `shared_head.norm` → matmul against `shared_head.head` (quantized or
  plain).
* `qwen3.6-mlx::mtp::load_mtp_head` tries five prefix conventions
  (`mtp.layers`, `model.mtp.layers`, `model.mtp_layers`,
  `language_model.model.mtp_layers`,
  `language_model.model.mtp.layers`) and returns `Ok(None)` gracefully
  on any missing required key.
* `qwen3.6-mlx::Model::forward_last_hidden_and_logits` exposes the
  last-position post-norm hidden state alongside logits — needed by
  the MTP cycle.
* A synthetic-weights unit test
  (`mtp::tests::synthetic_forward_runs_end_to_end`) constructs a tiny
  weight map and verifies `MtpHead::forward` returns `[1, 1, vocab]`
  logits. This proves the wiring even though no real checkpoint ships
  MTP weights yet.
* `MtplxSession::mtp_cycle` is now a real K=1 greedy draft+verify
  cycle (no longer a stub):
    1. Target forward on the last committed token; capture hidden +
       logits → AR-next.
    2. Embed last token via host `embed_tokens`.
    3. `mtp_head.forward(hidden, prev_emb)` → drafted token.
    4. Target forward on AR-next to verify; if its argmax matches the
       drafted token we accept both, otherwise we still commit
       AR-next + the target's verified continuation.
* `target_supports_mtp(&mut Model)` is a runtime check (does the
  loaded head have real weights, not just a stub).
* When weights are absent the session still falls back to greedy AR —
  the bench example exercises this path.

## What the target accessor looks like

```rust
let mut model = qwen3_6_mlx::load_model(&target_dir)?;
match model.mtp_head() {
    Some(head) if !head.is_stub() => { /* draft K tokens */ }
    _ => { /* AR fallback */ }
}
```

## Expected weight layout

For reference: the loader looks for these keys under any of the
supported prefixes (e.g. `model.mtp_layers.0.`, `mtp.layers.0.`, etc.):

```
<prefix>.0.enorm.weight
<prefix>.0.hnorm.weight
<prefix>.0.eh_proj.{weight,scales,biases}
<prefix>.0.self_attn.{q,k,v,o}_proj.{weight,scales,biases}
<prefix>.0.self_attn.{q,k}_norm.weight
<prefix>.0.input_layernorm.weight
<prefix>.0.post_attention_layernorm.weight
<prefix>.0.mlp.{gate,up,down}_proj.{weight,scales,biases}      # dense
<prefix>.0.mlp.gate.{weight,scales,biases}                     # MoE
<prefix>.0.mlp.switch_mlp.{gate,up,down}_proj.{weight,scales,biases}
<prefix>.0.mlp.shared_expert.{gate,up,down}_proj.{weight,scales,biases}
<prefix>.0.mlp.shared_expert_gate.{weight,scales,biases}
<prefix>.0.shared_head.norm.weight
<prefix>.0.shared_head.head.weight (+ optional .scales/.biases)
```

## Out of scope (TODOs)

- [ ] K > 1 MTP layers: today the loader supports `num_layers > 1` by
      composing layers iteratively, but Qwen3.6 ships with K=1 so this
      path is untested.
- [ ] Leviathan-Chen probability ratio acceptance + residual `(p-q)+`
      sampling (currently only greedy at T=0).
- [ ] Deterministic GDN-replay Metal kernel for the linear-attention
      layers so we can rewind state on rejection without recomputing
      from scratch.
- [ ] GraphBank: cache compiled MLX graphs across cycles to amortize
      JIT cost.
- [ ] Per-position logits in `forward_last_logits` so we can verify K
      candidates in one target forward (today it returns last-position
      only — adequate for K=1 but not K>1).
- [ ] DeepSeek / GLM / MiMo / Nemotron MTP support (separate crates).

## Bench smoke test

```
cargo build --release -p mtplx-mlx --example bench_mtplx
./target/release/examples/bench_mtplx \
    --target models/Qwen3.6-35B-A3B-4bit \
    --max-tokens 100 --temp 0.0 \
    --prompt "The theory of general relativity"
```

Expected output: a `[mtplx] target.mtp_head() is None — falling back to
autoregressive` notice, then standard AR generation.
