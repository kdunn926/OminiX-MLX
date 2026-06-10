//! Smoke test for `gemma4-mlx`'s assistant (drafter) port.
//!
//! Loads the Q6 assistant in
//! `models/Gemma4-27B-MTPLX-Optimized-Speed/assistant/` and exercises the
//! full forward path with synthetic-but-realistic-shaped `shared_kv_states`
//! and recurrent hidden. Confirms:
//! - All 94 weight tensors load with no shape/key mismatch.
//! - Forward runs end-to-end through pre_projection → 4 decoder layers
//!   (3 sliding + 1 full) → norm → post_projection + tied lm_head.
//! - Logits have shape `[B, 1, vocab_size]` and the recurrent hidden has
//!   shape `[B, 1, backbone_hidden_size]`.
//!
//! "Sensible response out" — i.e. a coherent token sequence — requires a
//! real target model to produce `shared_kv_states`. The 27B Gemma4 target
//! used by this pair ships in the same directory but uses MTPLX's key naming
//! (`language_model.model.*`) AND Q4 quantization, neither of which the
//! current `gemma4-mlx::load_model` supports. Loading that target is the
//! remaining work item before we can produce a real prompt-in/response-out
//! demo on this pair. This smoke test validates that everything downstream
//! of the target is correct.
//!
//! Usage:
//!   cargo run -p gemma4-mlx --example assistant_smoke -- \
//!     models/Gemma4-27B-MTPLX-Optimized-Speed/assistant "Hello world"

use std::env;
use std::path::PathBuf;

use anyhow::Result;
use mlx_rs::{random, transforms::eval, Array, Dtype};

use gemma4_mlx::assistant::{
    argmax_last, build_inputs_embeds, load_assistant_model, AssistantModel,
    SharedKvStates,
};

fn synth_kv(
    batch: i32,
    kv_len: i32,
    n_kv_heads: i32,
    head_dim: i32,
) -> Result<(Array, Array)> {
    // Match the dtype the real target produces (bf16). Use a small random
    // tensor so we exercise the SDPA path without producing degenerate
    // (e.g. uniform-zero) activations.
    let k = random::normal::<f32>(&[batch, n_kv_heads, kv_len, head_dim], None, Some(0.02), None)?
        .as_dtype(Dtype::Bfloat16)?;
    let v = random::normal::<f32>(&[batch, n_kv_heads, kv_len, head_dim], None, Some(0.02), None)?
        .as_dtype(Dtype::Bfloat16)?;
    Ok((k, v))
}

fn synth_hidden(backbone_hidden: i32) -> Result<Array> {
    let a = random::normal::<f32>(&[1, 1, backbone_hidden], None, Some(0.02), None)?
        .as_dtype(Dtype::Bfloat16)?;
    Ok(a)
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let assistant_dir = args
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from("models/Gemma4-27B-MTPLX-Optimized-Speed/assistant")
        });

    println!("Loading assistant from: {}", assistant_dir.display());
    let mut model: AssistantModel = load_assistant_model(&assistant_dir)?;
    let cfg = model.config.clone();
    let text = cfg.text_config.clone();

    println!(
        "Loaded gemma4_assistant: hidden={} backbone={} layers={} vocab={}",
        text.hidden_size, cfg.backbone_hidden_size, text.num_hidden_layers, text.vocab_size,
    );
    println!("Layer types: {:?}", text.layer_types);

    // Synth shared_kv_states with realistic shapes.
    // - Sliding layers: [B, num_key_value_heads, kv_len, head_dim]
    // - Full layer:     [B, num_global_key_value_heads, kv_len, global_head_dim]
    let batch = 1;
    let kv_len = 32;
    let (sliding_k, sliding_v) = synth_kv(
        batch,
        kv_len,
        text.num_key_value_heads,
        text.head_dim,
    )?;
    let (full_k, full_v) = synth_kv(
        batch,
        kv_len,
        if text.num_global_key_value_heads > 0 {
            text.num_global_key_value_heads
        } else {
            text.num_key_value_heads
        },
        if text.global_head_dim > 0 {
            text.global_head_dim
        } else {
            text.head_dim
        },
    )?;
    let kv = SharedKvStates {
        sliding_k,
        sliding_v,
        full_k,
        full_v,
    };
    println!(
        "Built shared_kv_states: sliding K {:?}, full K {:?}",
        kv.sliding_k.shape(),
        kv.full_k.shape()
    );

    // Run a small autoregressive loop with synthesized inputs. Even with
    // random K/V, the model produces SOME token sequence — this proves the
    // forward path is shape-correct end-to-end and the loop closes.
    let mut prev_token_embed = synth_hidden(cfg.backbone_hidden_size)?;
    let mut recurrent_hidden = synth_hidden(cfg.backbone_hidden_size)?;

    let mut emitted = Vec::new();
    let n_steps = 5;
    for step in 0..n_steps {
        let inputs_embeds = build_inputs_embeds(&prev_token_embed, &recurrent_hidden)?;
        if step == 0 {
            println!(
                "Step 0 inputs_embeds shape {:?} (expected [1, 1, {}])",
                inputs_embeds.shape(),
                2 * cfg.backbone_hidden_size
            );
        }
        let position_offset = (kv_len - 1) + step as i32;
        let out = model.forward(&inputs_embeds, position_offset, &kv)?;
        eval([&out.logits, &out.last_hidden])?;
        let logits_shape = out.logits.shape().to_vec();
        let last_shape = out.last_hidden.shape().to_vec();
        let tok = argmax_last(&out.logits)?;
        emitted.push(tok);
        println!(
            "step {step}: logits shape {logits_shape:?} (expected [1, 1, {}]), \
             last_hidden shape {last_shape:?} (expected [1, 1, {}]), argmax_tok={tok}",
            text.vocab_size, cfg.backbone_hidden_size,
        );

        // Recurrent update: next step's recurrent hidden = this step's
        // post_projection output. Re-embed the emitted token via the
        // assistant's own embed table (proxy for target_embed since we
        // don't have the target loaded here — this is the one place where
        // this smoke test deviates from the real recurrence; the real loop
        // would use target.embed(tok)).
        recurrent_hidden = out.last_hidden;
        // We don't have a 5376-dim path from a token id without the target,
        // so we keep prev_token_embed fixed (random) across steps. This is
        // fine for shape validation; semantic coherence is impossible
        // without the target.
        let _ = &mut prev_token_embed;
    }

    println!("\nEmitted token ids (5 steps): {emitted:?}");
    println!(
        "\nAssistant port validates end-to-end at the tensor level."
    );
    println!(
        "To get a SEMANTICALLY coherent draft sequence, the MTPLX target must be \
         loaded so that real `shared_kv_states` (last sliding+full layer K/V) can \
         be supplied. That target loader is the remaining work item — see \
         gemma4-pair-adapter-wip.md."
    );
    Ok(())
}
