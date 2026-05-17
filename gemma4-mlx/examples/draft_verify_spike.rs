//! Spike: drafter+verifier on the Gemma4 27B pair. Compares LINEAR drafting
//! against a 2-BRANCH TREE drafting topology to see if tree drafting yields
//! higher target tokens/s.
//!
//! Both modes share the same outer loop:
//!   - Prefill target on prompt; record `last_hidden` (post-norm) and
//!     `last_logits` (the target's prediction for position P).
//!   - Repeat until max_new tokens emitted:
//!       1. Build `shared_kv_states` from target.cache[last_sliding] /
//!          [last_full].
//!       2. DRAFT a sequence (linear) or a forest (tree) of candidate
//!          continuations.
//!       3. VERIFY each chain with `target.forward(chain_tokens)` at
//!          positions P..P+N. For the tree mode the two branches are
//!          verified as two separate forwards with KV snapshot/restore
//!          (no fused tree-mask kernel — keep the spike minimal).
//!       4. Accept the longest matching prefix; trim the rejected tail
//!          from the target's KV; run one extra 1-token forward on the
//!          target's "bonus" correction to (a) install its K/V and (b)
//!          capture the recurrent hidden for the next cycle.
//!
//! Tree benefit hypothesis (greedy decode): a 2-way fork at the root costs
//! ~2× verify work but wins if drafter top-1 is wrong AND top-2 is right
//! often enough to make up for it. Expect modest gains at temp=0.
//!
//! ============================================================
//! RESULTS (on 27B Q4 native, prompt = sky-blue question, chat
//! template, max_new=64, block=8, temp=0):
//!
//!   Plain AR (no drafter)   : 11.7 tok/s   (mtplx_chat baseline)
//!   LINEAR drafter+verify   :  0.55 tok/s  — 21× SLOWER than AR
//!   TREE2 drafter+verify    :  0.30 tok/s  — 35× SLOWER than AR
//!   Tree/Lin ratio          :  0.54×       — tree is WORSE here
//!
//!   Linear acceptance       :  9 / 304 drafted tokens  = 3.0%
//!   Tree2  acceptance       : 10 / 296 drafted tokens  = 3.4%
//!
//! The drafter agrees with the target ~3% of the time on chat-formatted
//! prompts; Python reference (`mtplx_pair.json::benchmark`) claims 98%
//! on its `flappy` long-form code suite. Difference suggests either a
//! subtle drafter-implementation bug, or that drafter quality depends
//! heavily on prompt regime (long predictable code-completion vs short
//! reasoning-mode answers).
//!
//! Conclusion for the tree-drafting hypothesis: with this drafter, tree
//! offers NO benefit — it costs 2× verify work for nearly the same
//! acceptance rate. The bottleneck is drafter quality (drafter agreement
//! with target), not the drafting topology. Tree drafting only helps
//! when drafter top-2 frequently captures the target's pick — that's
//! not the case here.
//! ============================================================
//!
//! Usage:
//!   cargo run --release -p gemma4-mlx --example draft_verify_spike -- \
//!     models/Gemma4-27B-MTPLX-Optimized-Speed \
//!     "Explain why the sky is blue." 96 8

use std::{env, path::PathBuf};

use anyhow::{anyhow, Result};
use mlx_rs::{
    argmax_axis,
    module::Module,
    ops::indexing::{IndexOp, NewAxis},
    transforms::eval,
    Array, Dtype,
};

use gemma4_mlx::{
    assistant::{build_inputs_embeds, load_assistant_model, AssistantModel, SharedKvStates},
    init_cache, load_tokenizer,
    model::{restore_cache, snapshot_cache, ModelInput},
    mtplx_target::load_mtplx_target,
    Gemma4ChatTemplate, Gemma4Message, KVCache, Model,
};

const EOS: &[i32] = &[1, 106, 50];

struct Pair {
    target: Model,
    assistant: AssistantModel,
    last_sliding: usize,
    last_full: usize,
}

fn scaled_token_embed(model: &mut Model, ids: &Array) -> Result<Array> {
    let embed = model.model.embed_tokens.forward(ids)?;
    let scale = if model.model.embed_scale.dtype() == embed.dtype() {
        model.model.embed_scale.clone()
    } else {
        model.model.embed_scale.as_dtype(embed.dtype())?
    };
    Ok(embed.multiply(&scale)?.as_dtype(Dtype::Bfloat16)?)
}

fn build_shared_kv(
    cache: &[KVCache],
    last_sliding: usize,
    last_full: usize,
) -> Result<SharedKvStates> {
    let (sk, sv) = cache[last_sliding]
        .current_kv()
        .ok_or_else(|| anyhow!("sliding cache empty"))?;
    let (fk, fv) = cache[last_full]
        .current_kv()
        .ok_or_else(|| anyhow!("full cache empty"))?;
    eval([&sk, &sv, &fk, &fv])?;
    Ok(SharedKvStates {
        sliding_k: sk,
        sliding_v: sv,
        full_k: fk,
        full_v: fv,
    })
}

fn argmax_i32(a: &Array) -> Result<i32> {
    Ok(argmax_axis!(a, -1)?.as_dtype(Dtype::Int32)?.item::<i32>())
}

/// Top-k token ids from a 1-D logits row. For k=2 this is just argmax then
/// partial-sort one more, done in Rust to avoid materialising vocab-sized
/// MLX masking tensors at vocab=262144.
fn topk_ids(row_logits: &Array, k: usize) -> Result<Vec<i32>> {
    let f32_row = row_logits.as_dtype(Dtype::Float32)?;
    eval([&f32_row])?;
    let slice = f32_row.as_slice::<f32>();
    let mut indexed: Vec<(usize, f32)> = slice.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    Ok(indexed.into_iter().take(k).map(|(i, _)| i as i32).collect())
}

/// Run the recurrent drafter for `n` steps. Returns the drafted token ids.
fn draft_linear(
    pair: &mut Pair,
    kv: &SharedKvStates,
    kv_offset: i32,
    seed_token: i32,
    seed_hidden: Array,
    n: usize,
) -> Result<Vec<i32>> {
    let mut prev_arr = Array::from(&[seed_token][..]).index(NewAxis);
    let mut prev_embed = scaled_token_embed(&mut pair.target, &prev_arr)?;
    let mut recurrent = seed_hidden;
    let mut out = Vec::with_capacity(n);
    for step in 0..n {
        let inputs = build_inputs_embeds(&prev_embed, &recurrent)?;
        // The HF Gemma4 assistant uses a SINGLE position_id (= kv_len - 1)
        // for the entire draft loop; the recurrent hidden carries the "step"
        // information, RoPE doesn't advance. See `candidate_generator.py::
        // Gemma4AssistantCandidateGenerator.get_candidates`.
        let _ = step;
        let position_offset = kv_offset - 1;
        let aout = pair.assistant.forward(&inputs, position_offset, kv)?;
        eval([&aout.logits, &aout.last_hidden])?;
        let tok = argmax_i32(&aout.logits.index((.., -1, ..)).reshape(&[-1])?)?;
        out.push(tok);
        prev_arr = Array::from(&[tok][..]).index(NewAxis);
        prev_embed = scaled_token_embed(&mut pair.target, &prev_arr)?;
        recurrent = aout.last_hidden;
    }
    Ok(out)
}

/// Same as `draft_linear`, but at step 0 takes the SECOND best token from
/// the drafter (top-2 instead of top-1) and then continues linearly.
fn draft_linear_top2(
    pair: &mut Pair,
    kv: &SharedKvStates,
    kv_offset: i32,
    seed_token: i32,
    seed_hidden: Array,
    n: usize,
) -> Result<Vec<i32>> {
    let mut prev_arr = Array::from(&[seed_token][..]).index(NewAxis);
    let mut prev_embed = scaled_token_embed(&mut pair.target, &prev_arr)?;
    let mut recurrent = seed_hidden;
    let mut out = Vec::with_capacity(n);
    for step in 0..n {
        let inputs = build_inputs_embeds(&prev_embed, &recurrent)?;
        // The HF Gemma4 assistant uses a SINGLE position_id (= kv_len - 1)
        // for the entire draft loop; the recurrent hidden carries the "step"
        // information, RoPE doesn't advance. See `candidate_generator.py::
        // Gemma4AssistantCandidateGenerator.get_candidates`.
        let _ = step;
        let position_offset = kv_offset - 1;
        let aout = pair.assistant.forward(&inputs, position_offset, kv)?;
        eval([&aout.logits, &aout.last_hidden])?;
        let row = aout.logits.index((.., -1, ..)).reshape(&[-1])?;
        let tok = if step == 0 {
            let topk = topk_ids(&row, 2)?;
            topk[1]
        } else {
            argmax_i32(&row)?
        };
        out.push(tok);
        prev_arr = Array::from(&[tok][..]).index(NewAxis);
        prev_embed = scaled_token_embed(&mut pair.target, &prev_arr)?;
        recurrent = aout.last_hidden;
    }
    Ok(out)
}

/// Verify a candidate chain on the target. Inputs:
///   `prev_logits_last`: target's logits at position P (predicts what
///       drafted[0] should be). Captured from the previous cycle.
///   `drafted`: candidate continuation [d0, d1, ..., dN-1].
/// Returns (accepted_count, bonus_token, last_hidden_for_bonus).
///
/// Side effect: target's KV cache grows by `accepted + 1` positions (the
/// accepted drafted prefix + the bonus). On entry the cache must be at
/// position P; on exit it is at P + accepted + 1.
fn verify_chain(
    pair: &mut Pair,
    cache: &mut Vec<KVCache>,
    prev_logits_last: &Array,
    drafted: &[i32],
) -> Result<(usize, i32, Array)> {
    let n = drafted.len();
    let in_arr = Array::from(drafted).index(NewAxis); // [1, N]

    // Run target forward on the N drafted tokens; get per-position logits
    // AND post-norm hidden (we need the latter for the next cycle's
    // drafter seed). Use the LanguageModel forward (returns hidden), then
    // apply lm_head manually.
    let hidden = pair.target.model.forward(ModelInput {
        inputs: &in_arr,
        mask: None,
        cache,
    })?;
    eval([&hidden])?;
    // Project hidden through lm_head (or tied embed) for per-position logits.
    let logits = pair.target.forward_via_hidden(&hidden)?;
    eval([&logits])?;

    // Acceptance loop. drafted[0] must match argmax(prev_logits_last).
    // drafted[i] (i>=1) must match argmax(logits[i-1]).
    let prev_pred = argmax_i32(&prev_logits_last.reshape(&[-1])?)?;
    let mut accepted = 0usize;
    if !drafted.is_empty() && drafted[0] == prev_pred {
        accepted = 1;
        for i in 1..n {
            let pred = argmax_i32(&logits.index((.., (i - 1) as i32, ..)).reshape(&[-1])?)?;
            if drafted[i] == pred {
                accepted += 1;
            } else {
                break;
            }
        }
    }

    // Bonus token: target's prediction for the position AFTER the last
    // accepted token. If accepted == 0, bonus = prev_pred (we never even
    // matched drafted[0], so the target says position P should be
    // prev_pred). If accepted >= 1, bonus = argmax(logits[accepted-1]).
    let bonus = if accepted == 0 {
        prev_pred
    } else {
        argmax_i32(
            &logits
                .index((.., (accepted - 1) as i32, ..))
                .reshape(&[-1])?,
        )?
    };

    // Cache accounting. We fed N tokens; cache grew by N. Of those, the
    // first `accepted` correspond to positions P..P+accepted-1 — keep
    // them. The remaining (N - accepted) entries are wrong and must be
    // trimmed. Then we need to install the bonus token's K/V at position
    // P+accepted; do that via a 1-token target forward on [bonus]. That
    // forward also produces the post-norm hidden we'll use as the next
    // drafter's `seed_hidden`.
    let to_drop = (n as i32) - (accepted as i32);
    if to_drop > 0 {
        for c in cache.iter_mut() {
            c.trim(to_drop);
        }
    }
    let bonus_arr = Array::from(&[bonus][..]).index(NewAxis);
    let bonus_hidden = pair.target.model.forward(ModelInput {
        inputs: &bonus_arr,
        mask: None,
        cache,
    })?;
    eval([&bonus_hidden])?;
    let h_b = bonus_hidden
        .index((.., -1, ..))
        .reshape(&[1, 1, pair.target.args.hidden_size])?;
    Ok((accepted, bonus, h_b))
}

#[derive(Debug, Clone, Copy)]
struct CycleStats {
    drafted: usize,
    accepted: usize,
}

#[derive(Debug, Clone, Copy)]
enum Mode {
    Linear,
    Tree2,
}

fn run_mode(
    pair: &mut Pair,
    mode: Mode,
    prompt_ids: &[i32],
    max_new: usize,
    block: usize,
    verbose: bool,
) -> Result<(Vec<i32>, Vec<CycleStats>, f32)> {
    let n_layers = pair.target.args.num_hidden_layers as usize;
    let mut cache: Vec<KVCache> = init_cache::<KVCache>(n_layers);
    let hidden_h = pair.target.args.hidden_size;

    // Prefill.
    let prompt_arr = Array::from(prompt_ids).index(NewAxis);
    let prefill_hidden = pair.target.model.forward(ModelInput {
        inputs: &prompt_arr,
        mask: None,
        cache: &mut cache,
    })?;
    eval([&prefill_hidden])?;
    let last_h = prefill_hidden
        .index((.., -1, ..))
        .reshape(&[1, 1, hidden_h])?;
    // Initial "prev_logits": target's prediction for the first new token.
    let last_h_for_lm = prefill_hidden.index((.., -1, ..)).reshape(&[1, 1, hidden_h])?;
    let mut prev_logits = pair.target.forward_via_hidden(&last_h_for_lm)?;
    eval([&prev_logits])?;
    let mut seed_token = *prompt_ids.last().unwrap();
    let mut seed_hidden = last_h;

    let mut emitted: Vec<i32> = Vec::new();
    let mut stats: Vec<CycleStats> = Vec::new();
    let t0 = std::time::Instant::now();

    while emitted.len() < max_new {
        let kv = build_shared_kv(&cache, pair.last_sliding, pair.last_full)?;
        let kv_offset = kv.sliding_k.shape()[2];

        let (drafted, accepted, bonus, new_seed_hidden) = match mode {
            Mode::Linear => {
                let d = draft_linear(pair, &kv, kv_offset, seed_token, seed_hidden.clone(), block)?;
                if std::env::var("SPIKE_DEBUG").is_ok() && stats.is_empty() {
                    let pred0 = argmax_i32(&prev_logits.reshape(&[-1])?)?;
                    eprintln!(
                        "  [debug cycle 0] kv_len={kv_offset} seed_token={seed_token} \
                         target_pred(P)={pred0} drafted={:?}",
                        d
                    );
                }
                let (acc, bonus, h) = verify_chain(pair, &mut cache, &prev_logits, &d)?;
                (d, acc, bonus, h)
            }
            Mode::Tree2 => {
                // Snapshot cache for branch rollback.
                let snap = snapshot_cache(&cache);

                // Branch A: top-1 drafter chain.
                let d_a =
                    draft_linear(pair, &kv, kv_offset, seed_token, seed_hidden.clone(), block)?;
                let (acc_a, bonus_a, h_a) = verify_chain(pair, &mut cache, &prev_logits, &d_a)?;

                // Snapshot result of branch A so we can compare.
                let snap_a = snapshot_cache(&cache);
                // Restore to pre-verify state for branch B.
                restore_cache(&mut cache, snap);

                // Branch B: top-2 first token + linear continuation.
                let d_b = draft_linear_top2(
                    pair,
                    &kv,
                    kv_offset,
                    seed_token,
                    seed_hidden.clone(),
                    block,
                )?;
                let (acc_b, bonus_b, h_b) = verify_chain(pair, &mut cache, &prev_logits, &d_b)?;

                // Pick the winner. Greedy: branch A's accepted_0 == prev_pred
                // is mandatory for A; if A.accepted >= 1 it wins by default
                // (since B.drafted[0] is by construction NOT prev_pred and
                // therefore B.accepted == 0). Tree gain materialises only
                // when A.accepted == 0 and B.accepted >= 1.
                let (use_a, d, acc, bonus, h) = if acc_a >= acc_b {
                    (true, d_a, acc_a, bonus_a, h_a)
                } else {
                    (false, d_b, acc_b, bonus_b, h_b)
                };
                if use_a {
                    restore_cache(&mut cache, snap_a);
                }
                // else: cache is already in branch-B state.
                (d, acc, bonus, h)
            }
        };

        stats.push(CycleStats {
            drafted: block,
            accepted,
        });

        let mut committed_any = false;
        for i in 0..accepted {
            emitted.push(drafted[i]);
            committed_any = true;
            if EOS.contains(&drafted[i]) || emitted.len() >= max_new {
                break;
            }
        }
        // Always commit bonus too (target's correction / next-token).
        if (emitted.last().map_or(true, |&t| !EOS.contains(&t))) && emitted.len() < max_new {
            emitted.push(bonus);
            committed_any = true;
        }

        if verbose {
            eprintln!(
                "  [{}] cycle: drafted={block} accepted={accepted} \
                 emitted_total={}",
                match mode {
                    Mode::Linear => "linear",
                    Mode::Tree2 => "tree2 ",
                },
                emitted.len()
            );
        }

        if !committed_any || emitted.last().map_or(false, |&t| EOS.contains(&t)) {
            break;
        }

        // Set up next cycle.
        seed_token = *emitted.last().unwrap();
        seed_hidden = new_seed_hidden;
        // prev_logits = target's prediction for the position AFTER bonus.
        // Bonus's K/V is already in the cache; bonus_hidden is the last
        // hidden we just computed. So next cycle's prev_logits =
        // lm_head(bonus_hidden) = lm_head(new_seed_hidden).
        prev_logits = pair.target.forward_via_hidden(&seed_hidden)?;
        eval([&prev_logits])?;
    }

    Ok((emitted, stats, t0.elapsed().as_secs_f32()))
}

fn report(mode: &str, emitted: &[i32], stats: &[CycleStats], secs: f32, tokenizer: &tokenizers::Tokenizer) {
    let total_drafted: usize = stats.iter().map(|s| s.drafted).sum();
    let total_accepted: usize = stats.iter().map(|s| s.accepted).sum();
    let acc_rate = if total_drafted > 0 {
        total_accepted as f32 / total_drafted as f32
    } else {
        0.0
    };
    let tps = emitted.len() as f32 / secs.max(1e-6);
    let text = tokenizer
        .decode(&emitted.iter().map(|&i| i as u32).collect::<Vec<_>>(), false)
        .unwrap_or_default();
    println!("\n=== {mode} ===");
    println!(
        "  Emitted   : {} tok in {:.2}s ({:.1} tok/s)",
        emitted.len(),
        secs,
        tps
    );
    println!(
        "  Cycles    : {}  acceptance: {}/{} = {:.2}",
        stats.len(),
        total_accepted,
        total_drafted,
        acc_rate
    );
    println!("  Text      : {text}");
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let root = args
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/Gemma4-27B-MTPLX-Optimized-Speed"));
    let prompt = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "Explain in two sentences why the sky appears blue.".to_string());
    let max_new: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(96);
    let block: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(8);

    println!("target_dir : {}", root.display());
    println!("prompt     : {prompt:?}");
    println!("max_new    : {max_new}");
    println!("block      : {block}");

    let target_dir = root.join("target");
    let assistant_dir = root.join("assistant");

    let tokenizer = load_tokenizer(&target_dir)?;
    // Whether to wrap the prompt in the Gemma4 chat template. Disabled by
    // default for the spike: the drafter was trained on plain text
    // continuations (mtplx_pair.json benchmark prompt is the "flappy"
    // long-form code suite), and chat-formatted prompts that trigger
    // channel/thought reasoning hurt acceptance dramatically.
    let use_chat = std::env::var("CHAT").is_ok();
    let to_encode = if use_chat {
        let template = Gemma4ChatTemplate::load(&target_dir)?;
        template.render_prompt(&[Gemma4Message::user(&prompt)], &[], true)?
    } else {
        prompt.clone()
    };
    // Chat template already inserts BOS / turn markers — don't double-add.
    // Raw prompts get add_special_tokens=true to insert BOS.
    let add_special = !use_chat;
    let enc = tokenizer
        .encode(to_encode.as_str(), add_special)
        .map_err(|e| anyhow!("encode: {e}"))?;
    let prompt_ids: Vec<i32> = enc.get_ids().iter().map(|&i| i as i32).collect();
    println!("chat       : {use_chat}  ({} prompt tokens)", prompt_ids.len());

    println!("Loading target + assistant (native Q4)…");
    let target = load_mtplx_target(&target_dir)?;
    let assistant = load_assistant_model(&assistant_dir)?;
    let layer_types = target.args.layer_types.clone();
    let last_sliding = layer_types
        .iter()
        .rposition(|t| t == "sliding_attention")
        .unwrap();
    let last_full = layer_types
        .iter()
        .rposition(|t| t == "full_attention")
        .unwrap();
    let mut pair = Pair {
        target,
        assistant,
        last_sliding,
        last_full,
    };

    let (e_lin, s_lin, t_lin) = run_mode(&mut pair, Mode::Linear, &prompt_ids, max_new, block, false)?;
    report("LINEAR", &e_lin, &s_lin, t_lin, &tokenizer);

    let (e_tree, s_tree, t_tree) =
        run_mode(&mut pair, Mode::Tree2, &prompt_ids, max_new, block, false)?;
    report("TREE2 (top-2 fork @ root)", &e_tree, &s_tree, t_tree, &tokenizer);

    let lin_tps = e_lin.len() as f32 / t_lin.max(1e-6);
    let tree_tps = e_tree.len() as f32 / t_tree.max(1e-6);
    println!("\n=== Comparison ===");
    println!("  Linear : {:.2} tok/s", lin_tps);
    println!("  Tree2  : {:.2} tok/s", tree_tps);
    println!("  Tree/Lin: {:.2}×", tree_tps / lin_tps.max(1e-6));
    Ok(())
}
