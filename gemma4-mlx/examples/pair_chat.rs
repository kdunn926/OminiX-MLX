//! Chat with the Gemma4 MTPLX pair (target + assistant speculative
//! decoding) through the production `Gemma4PairSession` API.
//!
//! Usage:
//!   cargo run --release -p gemma4-mlx --example pair_chat -- \
//!     models/Gemma4-27B-MTPLX-Optimized-Speed "Why is the sky blue?" [max_tokens]
//!
//! Env:
//!   PAIR_TEMP / PAIR_TOP_K / PAIR_TOP_P  stochastic sampling + LC acceptance
//!   PAIR_BLOCK                            draft block (default 4)
//!   PAIR_RAW=1                            skip the chat template

use std::{env, path::PathBuf};

use anyhow::{anyhow, Result};
use gemma4_mlx::{
    load_tokenizer,
    pair_session::{Gemma4PairSession, PairGenerateOptions, PAIR_EOS_TOKEN_IDS},
    Gemma4ChatTemplate, Gemma4Message,
};

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let pair_root = args
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/Gemma4-27B-MTPLX-Optimized-Speed"));
    let prompt = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "Why is the sky blue?".to_string());
    let max_tokens: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(256);

    let target_dir = pair_root.join("target");
    let tokenizer = load_tokenizer(&target_dir)?;
    let use_chat = env::var("PAIR_RAW").is_err();
    let rendered = if use_chat {
        let template = Gemma4ChatTemplate::load(&target_dir)?;
        template.render_prompt(&[Gemma4Message::user(&prompt)], &[], true)?
    } else {
        prompt.clone()
    };
    let enc = tokenizer
        .encode(rendered.as_str(), !use_chat)
        .map_err(|e| anyhow!("encode: {e}"))?;
    let prompt_ids: Vec<i32> = enc.get_ids().iter().map(|&i| i as i32).collect();

    eprintln!("[pair] loading {} ...", pair_root.display());
    let t0 = std::time::Instant::now();
    let mut session = Gemma4PairSession::load(&pair_root)?;
    eprintln!("[pair] loaded in {:.1}s", t0.elapsed().as_secs_f64());

    let mut opts = PairGenerateOptions {
        max_tokens,
        ..Default::default()
    };
    if let Some(t) = env::var("PAIR_TEMP").ok().and_then(|s| s.parse().ok()) {
        opts.temp = t;
    }
    if let Some(k) = env::var("PAIR_TOP_K").ok().and_then(|s| s.parse().ok()) {
        opts.top_k = k;
    }
    if let Some(p) = env::var("PAIR_TOP_P").ok().and_then(|s| s.parse().ok()) {
        opts.top_p = p;
    }
    if let Some(b) = env::var("PAIR_BLOCK").ok().and_then(|s| s.parse().ok()) {
        opts.block = b;
    }

    // Stream by decoding the accumulated ids and printing the new suffix.
    let mut acc_ids: Vec<u32> = Vec::new();
    let mut printed = 0usize;
    let tok_stream = tokenizer.clone();
    let (emitted, metrics) =
        session.generate(&prompt_ids, &opts, PAIR_EOS_TOKEN_IDS, |tok| {
            acc_ids.push(tok as u32);
            if let Ok(text) = tok_stream.decode(&acc_ids, true) {
                if text.len() > printed && text.is_char_boundary(printed) {
                    print!("{}", &text[printed..]);
                    use std::io::Write;
                    let _ = std::io::stdout().flush();
                    printed = text.len();
                }
            }
        })?;
    println!();

    eprintln!(
        "\n[pair] prefill {:.2}s | {} tok in {:.2}s = {:.2} tok/s | {} cycles, acceptance {}/{} = {:.2}",
        metrics.prefill_s,
        emitted.len(),
        metrics.decode_s,
        metrics.decode_tok_per_s(),
        metrics.cycles,
        metrics.accepted,
        metrics.drafted,
        metrics.acceptance_rate(),
    );
    Ok(())
}
