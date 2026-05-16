# mtplx-mlx — WIP

Multi-Token Prediction speculative decoding for MLX Rust.

## Status (2026-05-15)

* Crate bootstrapped: `Cargo.toml`, `src/lib.rs`, `src/session.rs`,
  `examples/bench_mtplx.rs`.
* `qwen3.6-mlx` learned to parse `mtp_num_hidden_layers` /
  `mtp_use_dedicated_embeddings` from both `text_config` and top-level
  config.
* `qwen3.6-mlx::Model` gained `pub mtp_head: Option<MtpHead>` and a
  `Model::mtp_head() -> Option<&mut MtpHead>` accessor.
* `qwen3.6-mlx::mtp::load_mtp_head` scans the weight map for
  `mtp.*` / `model.mtp_layers.*` keys. The stock
  `mlx-community/Qwen3.6-35B-A3B-4bit` checkpoint strips these (see
  `mlx_lm.models.qwen3_5.Qwen3_5MoEModel.sanitize`, which deletes every
  key containing `"mtp."`), so the loader returns `Ok(None)` in
  practice.
* `target_supports_mtp(&mut Model)` is a **runtime** check, replacing
  the old `MtpError::NoMtpHead` compile-time capability gate.
* `MtplxSession::generate` runs greedy AR when `has_mtp == false` —
  the bench example demonstrates the graceful fallback path.

## What the target accessor looks like

```rust
let mut model = qwen3_6_mlx::load_model(&target_dir)?;
match model.mtp_head() {
    Some(head) => { /* draft K tokens */ }
    None      => { /* AR fallback */ }
}
```

## What still needs the actual weights

Whenever a Qwen3.6 checkpoint ships the MTP block, fill in
`qwen3.6-mlx/src/mtp.rs::load_mtp_head` to construct a concrete
transformer block. The expected key layout (inferred from DeepSeek-V3 /
Qwen3-Next references; not verified for Qwen3.6):

```
model.mtp_layers.0.enorm.weight
model.mtp_layers.0.hnorm.weight
model.mtp_layers.0.eh_proj.{weight,scales,biases}
model.mtp_layers.0.self_attn.{q,k,v,o}_proj.{weight,scales,biases}
model.mtp_layers.0.self_attn.{q,k}_norm.weight
model.mtp_layers.0.input_layernorm.weight
model.mtp_layers.0.post_attention_layernorm.weight
model.mtp_layers.0.mlp.{gate,up,down}_proj.{weight,scales,biases}
model.mtp_layers.0.shared_head.norm.weight
model.mtp_layers.0.shared_head.head.{weight,scales,biases}   # may be tied
```

Then implement `MtpHead::forward(hidden, prev_token_emb) -> Array` to
return logits for the +1 token. For K>1 drafting, unroll the head K
times, feeding each predicted token's embedding back in.

## Out of scope (TODOs)

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
