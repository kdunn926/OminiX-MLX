//! Gemma4 generation with EAGLE-3 speculative decoding **on by default**.
//!
//! Resolution: an explicit draft-dir argument or `EAGLE3_DRAFT_DIR` wins;
//! otherwise the draft is auto-discovered next to the target (any sibling
//! dir named like the upstream `*eagle3*` checkpoints). If no draft is
//! found, validation fails, or `OMINIX_EAGLE3=0` is set, generation falls
//! back to plain autoregressive decoding on the same model.
//!
//! ```bash
//! cargo run --release -p eagle3-mlx --example eagle3_generate -- \
//!     ./models/gemma4-26B-a4b-it-UD-MLX-4bit \
//!     "Write a haiku about speculative decoding."
//! ```
//!
//! Knobs: `EAGLE3_BLOCK` (chain length, default 3), `EAGLE3_QUANT_DRAFT`
//! (8/4/off), `EAGLE3_MAX_TOKENS` (default 256), `EAGLE3_TEMP` (default 0).

use std::io::Write as _;
use std::time::Instant;

use anyhow::{anyhow, Result};

use eagle3_mlx::Eagle3Session;
use gemma4_mlx::{
    load_tokenizer, Gemma4ChatTemplate, Gemma4Message, Generate, KVCache, EOS_TOKEN_IDS,
};
use mlx_rs::{
    ops::indexing::{IndexOp, NewAxis},
    Array,
};
use tokenizers::Tokenizer;

struct GenArgs {
    max_tokens: usize,
    temp: f32,
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let target_dir = args
        .next()
        .ok_or_else(|| anyhow!("usage: eagle3_generate <target_dir> [draft_dir] [prompt]"))?;
    // Second positional arg is the draft dir if it exists on disk, else it's
    // the prompt.
    let (draft_arg, prompt) = match args.next() {
        Some(a) if std::path::Path::new(&a).join("config.json").is_file() => (Some(a), args.next()),
        other => (None, other),
    };
    let prompt = prompt.unwrap_or_else(|| "Explain speculative decoding in two sentences.".into());

    let gen_args = GenArgs {
        max_tokens: std::env::var("EAGLE3_MAX_TOKENS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(256),
        temp: std::env::var("EAGLE3_TEMP")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.0),
    };

    let tokenizer = load_tokenizer(&target_dir)?;
    let template = Gemma4ChatTemplate::load(&target_dir)?;
    let rendered = template.render_prompt(&[Gemma4Message::user(&prompt)], &[], true)?;
    let enc = tokenizer
        .encode(rendered.as_str(), true)
        .map_err(|e| anyhow!("encode: {e}"))?;
    let prompt_ids: Vec<u32> = enc.get_ids().to_vec();
    eprintln!("prompt: {} tokens", prompt_ids.len());

    // EAGLE-3 is the default: explicit draft arg > EAGLE3_DRAFT_DIR >
    // auto-discovery, unless opted out via OMINIX_EAGLE3=0.
    let draft_dir = if eagle3_mlx::env_enabled() {
        draft_arg
            .map(std::path::PathBuf::from)
            .or_else(|| eagle3_mlx::discover_draft(&target_dir))
    } else {
        eprintln!("[eagle3] disabled via OMINIX_EAGLE3=0 — autoregressive decode");
        None
    };

    if let Some(d) = draft_dir {
        eprintln!("[eagle3] draft: {}", d.display());
        let load_start = Instant::now();
        match Eagle3Session::load(&target_dir, &d, gen_args.temp) {
            Ok(mut session) => {
                eprintln!(
                    "loaded target + EAGLE-3 draft in {:.1}s (block_len={})",
                    load_start.elapsed().as_secs_f64(),
                    session.block_len
                );
                return run_eagle(&mut session, &tokenizer, prompt_ids, &gen_args);
            }
            Err(e) => {
                eprintln!("[eagle3] draft pairing failed ({e}) — falling back to AR");
            }
        }
    } else if eagle3_mlx::env_enabled() {
        eprintln!("[eagle3] no draft checkpoint found for this target — autoregressive decode");
    }

    run_ar(&target_dir, &tokenizer, prompt_ids, &gen_args)
}

fn run_eagle(
    session: &mut Eagle3Session,
    tokenizer: &Tokenizer,
    prompt_ids: Vec<u32>,
    gen_args: &GenArgs,
) -> Result<()> {
    let gen_start = Instant::now();
    let mut first_token_at: Option<Instant> = None;
    let mut emitted = 0usize;
    for item in session.generate(prompt_ids, gen_args.max_tokens, gen_args.temp, EOS_TOKEN_IDS) {
        let token = item?;
        if first_token_at.is_none() {
            first_token_at = Some(Instant::now());
        }
        emitted += 1;
        if let Ok(s) = tokenizer.decode(&[token], true) {
            print!("{s}");
        }
        std::io::stdout().flush().ok();
    }
    println!();

    let (prefill_s, decode_s) = split_times(gen_start, first_token_at);
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

fn run_ar(
    target_dir: &str,
    tokenizer: &Tokenizer,
    prompt_ids: Vec<u32>,
    gen_args: &GenArgs,
) -> Result<()> {
    let load_start = Instant::now();
    let mut model = gemma4_mlx::load_model(target_dir)?;
    eprintln!("loaded target in {:.1}s (AR)", load_start.elapsed().as_secs_f64());

    let ids: Vec<i32> = prompt_ids.iter().map(|&i| i as i32).collect();
    let prompt = Array::from_slice(&ids, &[ids.len() as i32]).index(NewAxis);
    let mut cache: Vec<KVCache> = Vec::new();

    let gen_start = Instant::now();
    let mut first_token_at: Option<Instant> = None;
    let mut emitted = 0usize;
    for token in
        Generate::new(&mut model, &mut cache, gen_args.temp, &prompt).take(gen_args.max_tokens)
    {
        let token = token.map_err(|e| anyhow!(e.to_string()))?;
        let token_id = token.item::<u32>();
        if EOS_TOKEN_IDS.contains(&token_id) {
            break;
        }
        if first_token_at.is_none() {
            first_token_at = Some(Instant::now());
        }
        emitted += 1;
        if let Ok(s) = tokenizer.decode(&[token_id], true) {
            print!("{s}");
        }
        std::io::stdout().flush().ok();
    }
    println!();

    let (prefill_s, decode_s) = split_times(gen_start, first_token_at);
    eprintln!(
        "ar: tokens={emitted} prefill_s={prefill_s:.2} decode_tok_s={:.2}",
        (emitted.saturating_sub(1)) as f64 / decode_s,
    );
    Ok(())
}

fn split_times(gen_start: Instant, first_token_at: Option<Instant>) -> (f64, f64) {
    let total_s = gen_start.elapsed().as_secs_f64();
    let prefill_s = first_token_at
        .map(|t| (t - gen_start).as_secs_f64())
        .unwrap_or(total_s);
    ((prefill_s), (total_s - prefill_s).max(1e-9))
}
