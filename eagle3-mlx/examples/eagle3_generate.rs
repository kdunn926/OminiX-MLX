//! EAGLE-3 speculative generation against a Gemma4 target.
//!
//! Env-gated: refuses to run without `OMINIX_EAGLE3=1`.
//!
//! ```bash
//! OMINIX_EAGLE3=1 cargo run --release -p eagle3-mlx --example eagle3_generate -- \
//!     ./models/gemma4-26B-a4b-it-UD-MLX-4bit \
//!     ./models/gemma-4-26B-A4B-it-eagle3 \
//!     "Write a haiku about speculative decoding."
//! ```
//!
//! The draft dir may be omitted if `EAGLE3_DRAFT_DIR` is set. Other knobs:
//! `EAGLE3_BLOCK` (chain length, default 3), `EAGLE3_MAX_TOKENS` (default
//! 256), `EAGLE3_TEMP` (default 0 = greedy).

use std::io::Write as _;
use std::time::Instant;

use anyhow::{anyhow, bail, Result};

use eagle3_mlx::Eagle3Session;
use gemma4_mlx::{load_tokenizer, Gemma4ChatTemplate, Gemma4Message, EOS_TOKEN_IDS};

fn main() -> Result<()> {
    if !eagle3_mlx::env_enabled() {
        bail!(
            "EAGLE-3 is env-gated. Re-run with OMINIX_EAGLE3=1 (and optionally \
             EAGLE3_DRAFT_DIR=<path-to-speculator-checkpoint>)."
        );
    }

    let mut args = std::env::args().skip(1);
    let target_dir = args
        .next()
        .ok_or_else(|| anyhow!("usage: eagle3_generate <target_dir> [draft_dir] [prompt]"))?;
    // Second positional arg is the draft dir if it exists on disk, else it's
    // the prompt and the draft dir comes from EAGLE3_DRAFT_DIR.
    let (draft_dir, prompt) = match args.next() {
        Some(a) if std::path::Path::new(&a).is_dir() => (a, args.next()),
        other => (
            std::env::var("EAGLE3_DRAFT_DIR")
                .map_err(|_| anyhow!("no draft dir argument and EAGLE3_DRAFT_DIR unset"))?,
            other,
        ),
    };
    let prompt = prompt.unwrap_or_else(|| "Explain speculative decoding in two sentences.".into());

    let max_tokens: usize = std::env::var("EAGLE3_MAX_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(256);
    let temp: f32 = std::env::var("EAGLE3_TEMP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0);

    let tokenizer = load_tokenizer(&target_dir)?;
    let template = Gemma4ChatTemplate::load(&target_dir)?;
    let rendered = template.render_prompt(&[Gemma4Message::user(&prompt)], &[], true)?;
    let enc = tokenizer
        .encode(rendered.as_str(), true)
        .map_err(|e| anyhow!("encode: {e}"))?;
    let prompt_ids: Vec<u32> = enc.get_ids().to_vec();
    eprintln!("prompt: {} tokens", prompt_ids.len());

    let load_start = Instant::now();
    let mut session = Eagle3Session::load(&target_dir, &draft_dir, temp)?;
    eprintln!(
        "loaded target + EAGLE-3 draft in {:.1}s (block_len={})",
        load_start.elapsed().as_secs_f64(),
        session.block_len
    );

    let gen_start = Instant::now();
    let mut first_token_at: Option<Instant> = None;
    let mut emitted = 0usize;
    let mut pending_text = String::new();
    for item in session.generate(prompt_ids, max_tokens, temp, EOS_TOKEN_IDS) {
        let token = item?;
        if first_token_at.is_none() {
            first_token_at = Some(Instant::now());
        }
        emitted += 1;
        // Decode incrementally; tokenizer.decode on the running tail keeps
        // multi-byte sequences intact without a streaming decoder.
        pending_text.clear();
        if let Ok(s) = tokenizer.decode(&[token], true) {
            pending_text.push_str(&s);
        }
        print!("{pending_text}");
        std::io::stdout().flush().ok();
    }
    println!();

    let total_s = gen_start.elapsed().as_secs_f64();
    let prefill_s = first_token_at
        .map(|t| (t - gen_start).as_secs_f64())
        .unwrap_or(total_s);
    let decode_s = (total_s - prefill_s).max(1e-9);
    let m = session.metrics();
    eprintln!(
        "eagle3: tokens={emitted} prefill_s={prefill_s:.2} decode_tok_s={:.2} \
         acceptance_ratio={:.3} avg_block_len={:.2} cycles={}",
        (emitted.saturating_sub(1)) as f64 / decode_s,
        m.acceptance_ratio,
        m.avg_block_len,
        m.total_cycles,
    );
    Ok(())
}
