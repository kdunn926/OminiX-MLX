//! Standalone autoregressive chat on the MTPLX 27B Gemma4 target.
//!
//! Loads `models/Gemma4-27B-MTPLX-Optimized-Speed/target/` via
//! `mtplx_target::load_mtplx_target` (Q4 → BF16 dequant) and runs the
//! standard `Generate` iterator. No drafter, no speculation — this is the
//! sanity check that the target model alone produces a coherent
//! continuation under gemma4-mlx's existing AR path.
//!
//! Usage:
//!   cargo run --release -p gemma4-mlx --example mtplx_chat -- \
//!     models/Gemma4-27B-MTPLX-Optimized-Speed/target \
//!     "The quick brown fox" 64

use std::{env, path::PathBuf};

use anyhow::{anyhow, Result};
use mlx_rs::{
    ops::indexing::{IndexOp, NewAxis},
    transforms::eval,
    Array,
};

use gemma4_mlx::{
    load_tokenizer, mtplx_target::load_mtplx_target, Gemma4ChatTemplate, Gemma4Message, Generate,
    KVCache,
};

/// EOS token IDs from generation_config.json.
const EOS_TOKEN_IDS: &[i32] = &[1, 106, 50];
fn temperature() -> f32 {
    std::env::var("TEMP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0)
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let target_dir = args
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from("models/Gemma4-27B-MTPLX-Optimized-Speed/target")
        });
    let prompt = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "The quick brown fox".to_string());
    let max_new: usize = args
        .get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);

    println!("target_dir : {}", target_dir.display());
    println!("prompt     : {:?}", prompt);
    println!("max_new    : {max_new}");
    let temp = temperature();
    println!("temperature: {temp}");
    println!("eos        : {EOS_TOKEN_IDS:?}");
    println!();

    println!("Loading tokenizer + chat template…");
    let tokenizer = load_tokenizer(&target_dir)?;
    let template = Gemma4ChatTemplate::load(&target_dir)?;
    let rendered = template.render_prompt(&[Gemma4Message::user(&prompt)], &[], true)?;
    println!("Rendered prompt ({} chars).", rendered.len());

    println!("Loading MTPLX target (native Q4; ~13GB resident)…");
    let t0 = std::time::Instant::now();
    let mut model = load_mtplx_target(&target_dir)?;
    println!("  target loaded in {:.1}s", t0.elapsed().as_secs_f32());

    let enc = tokenizer
        .encode(rendered.as_str(), false)
        .map_err(|e| anyhow!("encode: {e}"))?;
    let prompt_ids: Vec<i32> = enc.get_ids().iter().map(|&i| i as i32).collect();
    println!(
        "Prompt tokens ({} ids): {:?}",
        prompt_ids.len(),
        &prompt_ids[..prompt_ids.len().min(16)]
    );
    let prompt_arr = Array::from(prompt_ids.as_slice()).index(NewAxis);

    let mut cache: Vec<KVCache> = Vec::new();
    let generator = Generate::new(&mut model, &mut cache, temp, &prompt_arr);

    print!("Generating: ");
    let mut emitted: Vec<i32> = Vec::with_capacity(max_new);
    let t_gen = std::time::Instant::now();
    let mut t_first: Option<std::time::Duration> = None;
    for (i, tok) in generator.enumerate() {
        if i >= max_new {
            break;
        }
        let tok_arr = tok.map_err(|e| anyhow!("gen step: {e:?}"))?;
        eval([&tok_arr])?;
        if t_first.is_none() {
            t_first = Some(t_gen.elapsed());
        }
        let id: i32 = tok_arr.reshape(&[-1])?.item::<i32>();
        if EOS_TOKEN_IDS.contains(&id) {
            println!("\n[hit eos at step {i}: {id}]");
            break;
        }
        emitted.push(id);
    }
    let total = t_gen.elapsed();
    let ttft = t_first.unwrap_or(total);
    let decode_secs = (total - ttft).as_secs_f32();
    let decoded_tokens = emitted.len().saturating_sub(1) as f32; // exclude first
    let decode_tps = if decoded_tokens > 0.0 {
        decoded_tokens / decode_secs.max(1e-6)
    } else {
        0.0
    };
    let prompt_tps = prompt_ids.len() as f32 / ttft.as_secs_f32().max(1e-6);

    let u32_ids: Vec<u32> = emitted.iter().map(|&i| i as u32).collect();
    let text = tokenizer
        .decode(&u32_ids, false)
        .unwrap_or_else(|_| "<decode error>".to_string());
    println!();
    println!("\n=== Timing ===");
    println!("  TTFT (prefill + 1st token): {:.2}s", ttft.as_secs_f32());
    println!(
        "  Prefill throughput:         {:.1} prompt-tok/s",
        prompt_tps
    );
    println!(
        "  Decode throughput:          {:.1} tok/s  ({} tokens in {:.2}s)",
        decode_tps, decoded_tokens as usize, decode_secs
    );
    println!("  Total:                      {:.2}s", total.as_secs_f32());
    println!("\n=== Output ===");
    println!("{prompt}{text}");
    Ok(())
}
