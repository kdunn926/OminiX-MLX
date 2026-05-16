//! End-to-end pair smoke test: MTPLX 27B target + Q6 assistant drafter.
//!
//! Loads the MTPLX target (Q4, dequantized on the fly) and the assistant
//! (Q6 native), runs the recurrent drafter loop, and prints the
//! drafter-emitted continuation. The drafter is NOT verified by the
//! target; the target's role here is only to supply `shared_kv_states`
//! and the seed recurrent hidden state. Output should be a coherent
//! continuation of the prompt — the assistant was trained to mimic the
//! target's distribution.
//!
//! Usage:
//!   cargo run --release -p gemma4-mlx --example pair_smoke -- \
//!     models/Gemma4-27B-MTPLX-Optimized-Speed "Hello, how are you?" 32
//!
//! Memory: the MTPLX target is dequantized to BF16 at load time so it can
//! reuse the existing gemma4-mlx Model path. Expect ~54GB resident.

use std::{env, path::PathBuf};

use anyhow::{anyhow, Result};
use mlx_rs::{
    module::Module,
    ops::indexing::{IndexOp, NewAxis},
    transforms::eval,
    Array, Dtype,
};

use gemma4_mlx::{
    assistant::{
        argmax_last, build_inputs_embeds, load_assistant_model, AssistantModel, SharedKvStates,
    },
    init_cache, load_tokenizer,
    model::ModelInput,
    mtplx_target::load_mtplx_target,
    KVCache,
};

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let model_root = args
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/Gemma4-27B-MTPLX-Optimized-Speed"));
    let prompt = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "Hello world. The quick brown fox".to_string());
    let n_draft: usize = args
        .get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);

    let target_dir = model_root.join("target");
    let assistant_dir = model_root.join("assistant");
    println!("target_dir   : {}", target_dir.display());
    println!("assistant_dir: {}", assistant_dir.display());
    println!("prompt       : {:?}", prompt);
    println!("n_draft      : {n_draft}");

    println!("\nLoading tokenizer (from target dir)…");
    let tokenizer = load_tokenizer(&target_dir)?;

    println!("Loading MTPLX target (native Q4)…");
    let mut target = load_mtplx_target(&target_dir)?;
    println!(
        "  target loaded: {} layers, hidden={}, vocab={}",
        target.args.num_hidden_layers, target.args.hidden_size, target.args.vocab_size
    );
    let n_target_layers = target.args.num_hidden_layers as usize;
    let layer_types: Vec<String> = target.args.layer_types.clone();
    let last_sliding = layer_types
        .iter()
        .rposition(|t| t == "sliding_attention")
        .ok_or_else(|| anyhow!("target has no sliding_attention layers"))?;
    let last_full = layer_types
        .iter()
        .rposition(|t| t == "full_attention")
        .ok_or_else(|| anyhow!("target has no full_attention layers"))?;
    println!(
        "  KV-borrow layers: sliding={last_sliding}, full={last_full}"
    );

    println!("Loading assistant…");
    let mut assistant: AssistantModel = load_assistant_model(&assistant_dir)?;
    let backbone_h = assistant.config.backbone_hidden_size;
    let hidden_h = target.args.hidden_size;
    if backbone_h != hidden_h {
        return Err(anyhow!(
            "assistant.backbone_hidden_size={backbone_h} != target.hidden_size={hidden_h}"
        ));
    }

    // Tokenize prompt (no BOS — let the tokenizer handle).
    let enc = tokenizer
        .encode(prompt.as_str(), true)
        .map_err(|e| anyhow!("tokenizer.encode: {e}"))?;
    let prompt_ids: Vec<i32> = enc.get_ids().iter().map(|&i| i as i32).collect();
    println!(
        "Prompt tokens ({} ids): {:?}",
        prompt_ids.len(),
        &prompt_ids[..prompt_ids.len().min(16)]
    );
    let prompt_arr = Array::from(prompt_ids.as_slice()).index(NewAxis); // [1, T]

    // Prefill target — populates the KV cache and produces the last hidden.
    let n_cache_slots = n_target_layers;
    let mut cache: Vec<KVCache> = init_cache::<KVCache>(n_cache_slots);
    println!("Running target prefill on {} tokens…", prompt_ids.len());
    let t0 = std::time::Instant::now();
    let last_hidden_full: Array = target.model.forward(ModelInput {
        inputs: &prompt_arr,
        mask: None,
        cache: &mut cache,
    })?;
    eval([&last_hidden_full])?;
    println!(
        "  prefill done in {:.2}s. last_hidden shape {:?}",
        t0.elapsed().as_secs_f32(),
        last_hidden_full.shape()
    );

    // Slice last-position hidden as the seed recurrent hidden for step 0.
    let mut recurrent_hidden = last_hidden_full
        .index((.., -1, ..))
        .reshape(&[1, 1, hidden_h])?;
    println!(
        "Seed recurrent_hidden shape {:?} (dtype {:?})",
        recurrent_hidden.shape(),
        recurrent_hidden.dtype()
    );

    // Extract shared_kv_states from the target's last sliding and last full layer.
    let (sliding_k, sliding_v) = cache[last_sliding]
        .current_kv()
        .ok_or_else(|| anyhow!("target cache[last_sliding] empty after prefill"))?;
    let (full_k, full_v) = cache[last_full]
        .current_kv()
        .ok_or_else(|| anyhow!("target cache[last_full] empty after prefill"))?;
    eval([&sliding_k, &sliding_v, &full_k, &full_v])?;
    println!(
        "Borrowed KV shapes: sliding K {:?}, full K {:?}",
        sliding_k.shape(),
        full_k.shape()
    );
    let kv = SharedKvStates {
        sliding_k,
        sliding_v,
        full_k,
        full_v,
    };

    // Initial prev_token = last prompt token. Embed via target.embed_tokens
    // and multiply by Gemma's sqrt(H) embed_scale, matching what the
    // candidate generator passes as `target_model_input_embeddings`.
    let last_prompt_token = *prompt_ids.last().unwrap();
    let prev_arr = Array::from(&[last_prompt_token][..]).index(NewAxis);
    let mut prev_token_embed = scaled_token_embed(&mut target, &prev_arr)?;
    println!(
        "Initial prev_token_embed shape {:?} (token id {})",
        prev_token_embed.shape(),
        last_prompt_token
    );

    // Recurrent drafter loop.
    let kv_len = kv.sliding_k.shape()[2];
    let mut draft_ids = Vec::with_capacity(n_draft);
    println!("\nDrafter loop:");
    for step in 0..n_draft {
        let inputs_embeds = build_inputs_embeds(&prev_token_embed, &recurrent_hidden)?;
        let position_offset = (kv_len - 1) + step as i32;
        let out = assistant.forward(&inputs_embeds, position_offset, &kv)?;
        eval([&out.logits, &out.last_hidden])?;

        let tok = argmax_last(&out.logits)?;
        draft_ids.push(tok);

        // Recurrence: next prev_token = sampled token; next recurrent_hidden = post_projection output.
        let tok_arr = Array::from(&[tok][..]).index(NewAxis);
        prev_token_embed = scaled_token_embed(&mut target, &tok_arr)?;
        recurrent_hidden = out.last_hidden;

        if step < 8 || step + 1 == n_draft {
            let s = tokenizer
                .decode(&[tok as u32], false)
                .unwrap_or_else(|_| "?".to_string());
            println!("  step {step:2}: tok={tok:6}  text={:?}", s);
        }
    }

    // Final decode.
    let u32_ids: Vec<u32> = draft_ids.iter().map(|&i| i as u32).collect();
    let drafted_text = tokenizer
        .decode(&u32_ids, false)
        .unwrap_or_else(|_| "<decode error>".to_string());
    println!("\n=== Prompt ===\n{prompt}");
    println!("\n=== Drafter continuation ({} tokens) ===\n{drafted_text}", draft_ids.len());
    Ok(())
}

/// Run `embed_tokens(ids) * sqrt(hidden_size)` — matches what gemma4's
/// `LanguageModel::forward` does at the input and what HF's
/// `target_model_input_embeddings(last_token_id)` returns.
fn scaled_token_embed(
    model: &mut gemma4_mlx::Model,
    ids: &Array,
) -> Result<Array> {
    let embed = model.model.embed_tokens.forward(ids)?;
    let scale = if model.model.embed_scale.dtype() == embed.dtype() {
        model.model.embed_scale.clone()
    } else {
        model.model.embed_scale.as_dtype(embed.dtype())?
    };
    Ok(embed.multiply(&scale)?.as_dtype(Dtype::Bfloat16)?)
}
