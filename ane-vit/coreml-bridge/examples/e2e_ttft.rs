//! End-to-end multimodal TTFT bench scaffold.
//!
//! Composes: ANE-routed vision encoder (Core ML) + MLX/Metal text body.
//! Reports TTFT decomposed into (vision_ms, text_prefill_ms, sample_ms)
//! for an image+prompt input.
//!
//! Status: scaffold only. The vision encoder side is wired (use the same
//! `CoreMlModel` path as `bench.rs`). The text body side is a STUB that
//! prints how it would integrate — wiring it to an MLX `Model` requires
//! deciding which multimodal target to run (Qwen3-VL via qwen3-vl-mlx,
//! Gemma4-VL via gemma4_mlx::VlModel, or another) and pasting the image
//! token embeddings into the text decoder's input stream at the
//! `<|vision_start|>...<|vision_end|>` placeholder positions.
//!
//! Run plan:
//!   1. Pre-converted vision encoder mlpackage (currently ViT-base; would
//!      become Qwen3-VL or similar when conversion lands).
//!   2. Load MLX target text body.
//!   3. For a fixed test prompt + 224×224 image:
//!      a. ANE vision encode → image_tokens
//!      b. MLX prefill on `[text_tokens, image_tokens, text_tokens]`
//!      c. Sample first token
//!   4. Time each phase, sum to TTFT.
//!
//! Compare:
//!   - all-GPU (image through MLX vision tower + MLX text body)
//!   - hybrid (image through ANE Core ML + MLX text body)

use coreml_bridge::{ComputeUnits, CoreMlModel};
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mlpackage = arg(&args, "--mlpackage")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("../vit-base.mlpackage"));
    let iters: usize = arg(&args, "--iters").and_then(|s| s.parse().ok()).unwrap_or(8);
    let units = match arg(&args, "--compute-units").as_deref() {
        Some("cpu") => ComputeUnits::CpuOnly,
        Some("gpu") => ComputeUnits::CpuAndGpu,
        Some("ane") => ComputeUnits::CpuAndNeuralEngine,
        _ => ComputeUnits::All,
    };

    eprintln!("[e2e] vision mlpackage = {}", mlpackage.display());
    eprintln!("[e2e] vision compute units = {:?}", units);

    let vision = match CoreMlModel::load(&mlpackage, units) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("vision load failed: {e}");
            std::process::exit(1);
        }
    };

    let mut pixels = vec![0.0f32; 1 * 3 * 224 * 224];
    let mut img_tokens = vec![0.0f32; 197 * 768];

    // Phase A: vision encode.
    // Phase B: TEXT BODY STUB — measured as constant 0 until wired.
    // Phase C: SAMPLE STUB — also 0 until wired.
    //
    // Once the multimodal pipeline lands, replace these stubs with the
    // actual MLX prefill + sample calls.

    let mut totals = (0.0f64, 0.0f64, 0.0f64); // (vision, text, sample)
    let mut samples_vision = Vec::with_capacity(iters);

    // Warmup.
    let _ = vision.predict(&pixels, &mut img_tokens);

    for _ in 0..iters {
        // A: vision encode
        let ta = Instant::now();
        if let Err(e) = vision.predict(&pixels, &mut img_tokens) {
            eprintln!("vision predict failed: {e}");
            std::process::exit(1);
        }
        let vision_ms = ta.elapsed().as_secs_f64() * 1000.0;
        samples_vision.push(vision_ms);

        // B: TODO — feed img_tokens into MLX text body, prefill text+image.
        // Stub: pretend zero-cost.
        let text_ms = 0.0;

        // C: TODO — sample first token from text body.
        let sample_ms = 0.0;

        totals.0 += vision_ms;
        totals.1 += text_ms;
        totals.2 += sample_ms;
    }
    let _ = (pixels.first(), img_tokens.first()); // suppress unused

    samples_vision.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = samples_vision.len();
    let mean = totals.0 / n as f64;
    let p50 = samples_vision[n / 2];
    let p99 = samples_vision[(n * 99) / 100];
    println!(
        "vision_only ({:?}): mean={:.2}ms p50={:.2}ms p99={:.2}ms ({} iters)",
        units, mean, p50, p99, n
    );
    println!("text_body STUB — wire to MLX::Model::forward to complete e2e measurement");
    println!("TTFT (vision+text+sample) currently: vision={:.2}ms text=0ms sample=0ms",
             mean);
}

fn arg(args: &[String], key: &str) -> Option<String> {
    args.windows(2)
        .find(|w| w[0] == key)
        .map(|w| w[1].clone())
}
