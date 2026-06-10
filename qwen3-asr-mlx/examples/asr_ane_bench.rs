//! ANE vs MLX latency bench for the Qwen3-ASR audio encoder.
//!
//! Measures per-chunk encode latency across three paths:
//!   MLX  — `audio_tower.forward_encoder()` on Metal/GPU (baseline)
//!   ANE  — Core ML predict via the converted mlpackage
//!   CPU  — Core ML with cpuOnly compute units (lower bound)
//!
//! Each 100-frame mel chunk encodes to 13 audio tokens.  The headline
//! question is whether ANE predict time < GPU decode time for 13 text
//! tokens — if so, ANE encodes are fully hidden in a streaming pipeline.
//!
//! Usage:
//!   cargo run --release -p qwen3-asr-mlx --example asr_ane_bench -- \
//!     --model models/qwen3-asr-1.7b \
//!     --ane ane-vit/qwen3-asr-encoder.mlpackage \
//!     --iters 16
//!
//!   # build the mlpackage first (once):
//!   uv run --isolated --python 3.10 \
//!     --with torch --with transformers --with coremltools==9.0 \
//!     --with "numpy<2" --with scipy --with accelerate \
//!     python ane-vit/converters/convert_qwen3_asr_encoder.py \
//!       --model models/qwen3-asr-1.7b \
//!       --output ane-vit/qwen3-asr-encoder.mlpackage

use coreml_bridge::{ComputeUnits, CoreMlModel};
use mlx_rs::transforms::eval;
use mlx_rs::Array;
use qwen3_asr_mlx::model::Qwen3ASR;
use std::path::PathBuf;
use std::time::Instant;

fn arg(args: &[String], key: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == key).map(|w| w[1].clone())
}

fn stats(samples: &mut [f64]) -> (f64, f64, f64, f64) {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = samples.len();
    let mean = samples.iter().sum::<f64>() / n as f64;
    let p50 = samples[n / 2];
    (mean, p50, samples[0], samples[n - 1])
}

/// Build a synthetic mel chunk: uniform noise in [-1, 1].
fn synthetic_mel(n_mels: usize, n_frames: usize) -> Vec<f32> {
    // Use a simple LCG so results are reproducible across runs.
    let mut state: u64 = 0x5851f42d4c957f2d;
    let mut out = vec![0.0f32; n_mels * n_frames];
    for v in &mut out {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *v = ((state >> 33) as f32 / (u32::MAX as f32)) * 2.0 - 1.0;
    }
    out
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let model_dir = arg(&args, "--model")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/qwen3-asr-1.7b"));
    let ane_path = arg(&args, "--ane").map(PathBuf::from);
    let iters: usize = arg(&args, "--iters").and_then(|s| s.parse().ok()).unwrap_or(16);
    let n_chunks: usize = arg(&args, "--chunks").and_then(|s| s.parse().ok()).unwrap_or(1);

    // Load MLX model (audio encoder + text decoder).
    println!("[bench] loading {} ...", model_dir.display());
    let t0 = Instant::now();
    let mut model = Qwen3ASR::load(&model_dir)?;
    println!("[bench] loaded in {:.2}s", t0.elapsed().as_secs_f64());

    let cfg = &model.config.audio_config;
    let n_mels = cfg.num_mel_bins as usize;
    let n_frames = (cfg.n_window * 2) as usize;   // 100 frames per chunk
    let n_out_tokens = 13usize;                    // 100-frame chunk → 13 audio tokens
    let output_dim = cfg.output_dim as usize;      // 2048

    println!(
        "[bench] encoder: d_model={} layers={} output_dim={} \
         chunk={}f→{}tok",
        cfg.d_model, cfg.encoder_layers, output_dim, n_frames, n_out_tokens
    );
    println!("[bench] bench config: {} iters, {} chunk(s) per call", iters, n_chunks);

    // Build synthetic mel: [n_mels, n_frames * n_chunks]
    let raw = synthetic_mel(n_mels, n_frames * n_chunks);
    let mel = Array::from_slice(&raw, &[n_mels as i32, (n_frames * n_chunks) as i32]);

    // ── MLX path ──────────────────────────────────────────────────────────────
    println!("\n[bench] === MLX (Metal/GPU) encoder ===");
    let mut mlx_samples = Vec::with_capacity(iters);

    for it in 0..(iters + 1) {
        let t = Instant::now();
        let out = model.audio_tower.forward_encoder(&mel)
            .map_err(|e| anyhow::anyhow!("MLX encode: {e}"))?;
        eval([&out])?;
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        if it == 0 {
            println!("  warmup: {:.1}ms  output shape {:?}", ms, out.shape());
        } else {
            mlx_samples.push(ms);
        }
    }

    let (mm, mp50, mmin, mmax) = stats(&mut mlx_samples);
    println!(
        "  mean={:.2}ms  p50={:.2}ms  min={:.2}  max={:.2}  \
         per-chunk={:.2}ms",
        mm, mp50, mmin, mmax, mm / n_chunks as f64
    );

    // ── ANE / Core ML path ────────────────────────────────────────────────────
    if let Some(ref p) = ane_path {
        println!("\n[bench] === Core ML ({}) ===", p.display());

        // Read recommended units from manifest sidecar if present.
        let manifest_units: Option<ComputeUnits> = (|| {
            let mp = p.with_extension("manifest.json");
            let text = std::fs::read_to_string(&mp).ok()?;
            let v: serde_json::Value = serde_json::from_str(&text).ok()?;
            match v["recommended_units"].as_str()? {
                "cpuAndNeuralEngine" => Some(ComputeUnits::CpuAndNeuralEngine),
                "cpuAndGpu" | "cpuAndGPU" => Some(ComputeUnits::CpuAndGpu),
                "cpuOnly" => Some(ComputeUnits::CpuOnly),
                _ => None,
            }
        })();

        // Input for a single chunk (ANE mlpackage is traced on 1 chunk).
        let single_raw = synthetic_mel(n_mels, n_frames);
        let input_cap = n_mels * n_frames;         // 12 800 f32
        let output_cap = n_out_tokens * output_dim; // 26 624 f32

        // Bench each compute-unit variant.
        let variants: &[(&str, ComputeUnits)] = &[
            ("cpuAndNeuralEngine", ComputeUnits::CpuAndNeuralEngine),
            ("cpuAndGpu",         ComputeUnits::CpuAndGpu),
            ("cpuOnly",           ComputeUnits::CpuOnly),
        ];

        let mut best_ms = f64::MAX;
        let mut best_label = "";

        for (label, units) in variants {
            let rec_marker = if manifest_units.as_ref().map(|u| u == units).unwrap_or(false) {
                " ★ manifest recommendation"
            } else {
                ""
            };
            print!("  {label}{rec_marker}: ");

            let m = match CoreMlModel::load(p, *units) {
                Ok(m) => m,
                Err(e) => {
                    println!("load error: {e}");
                    continue;
                }
            };

            let mut samples = Vec::with_capacity(iters);
            let mut warmup_ms = 0.0;
            for it in 0..(iters + 1) {
                let mut out_buf = vec![0.0f32; output_cap];
                let t = Instant::now();
                m.predict(&single_raw[..input_cap], &mut out_buf)
                    .map_err(|e| anyhow::anyhow!("ANE predict: {e}"))?;
                let ms = t.elapsed().as_secs_f64() * 1000.0;
                if it == 0 { warmup_ms = ms; } else { samples.push(ms); }
            }

            let (am, ap50, amin, amax) = stats(&mut samples);
            println!(
                "mean={:.2}ms  p50={:.2}ms  min={:.2}  max={:.2}  (warmup={:.1}ms)",
                am, ap50, amin, amax, warmup_ms
            );
            if am < best_ms {
                best_ms = am;
                best_label = label;
            }
        }

        // ── Overlap assessment ───────────────────────────────────────────────
        //
        // In a streaming ASR pipeline:
        //   - Each 100-frame chunk (~0.6 s of 16kHz audio) → 13 audio tokens
        //     → autoregressive decode → T_text text tokens
        //   - While decoding T_text tokens from chunk N, encode chunk N+1 on ANE
        //   - If ANE_encode_ms < T_text × ms_per_text_token → encode is free
        //
        // The ANE time is small (~8–11ms); speech at ~3 words/s ≈ 4 tokens/s
        // per chunk means T_text ≈ 2–5 text tokens per chunk.  At 50 tok/s
        // decode speed that's 40–100ms of decode time — ANE is fully hidden.
        println!("\n[bench] === Overlap assessment ===");
        let mlx_single = mm / n_chunks as f64;
        println!("  MLX encoder per chunk:      {mlx_single:.2}ms");
        println!("  Best Core ML per chunk ({best_label}): {best_ms:.2}ms");

        // How much faster is Core ML vs MLX?
        if best_ms < mlx_single {
            println!(
                "  Core ML speedup vs MLX:     {:.1}×",
                mlx_single / best_ms
            );
        } else {
            println!(
                "  MLX is {:.1}× faster than best Core ML",
                best_ms / mlx_single
            );
        }

        // Estimate: Qwen3-ASR 1.7B text decoder on M-series.
        // 8-bit quantized, ~1.7B params, typical ~60-80 tok/s.
        let tokens_per_sec_estimate = 60.0f64;
        let ms_per_text_token = 1000.0 / tokens_per_sec_estimate;
        // Typical speech density: ~4 text tokens per 100-frame chunk
        // (roughly 3 words/s × ~1.3 tokens/word ÷ ~1 chunk/s)
        let typical_text_per_chunk = 4.0f64;
        let decode_window_ms = typical_text_per_chunk * ms_per_text_token;
        println!(
            "\n  Streaming overlap (encode chunk N+1 while decoding chunk N text):"
        );
        println!(
            "    Text decode speed:          ~{tokens_per_sec_estimate:.0} tok/s → {ms_per_text_token:.1}ms/tok"
        );
        println!(
            "    Typical text tokens/chunk:  ~{typical_text_per_chunk:.0}  → {decode_window_ms:.1}ms decode window"
        );
        println!("    ANE encode time:            {best_ms:.2}ms");
        if best_ms < decode_window_ms {
            println!(
                "    → ANE FULLY HIDDEN ✓  ({best_ms:.1}ms < {decode_window_ms:.1}ms decode window)"
            );
        } else {
            let min_text = (best_ms / ms_per_text_token).ceil();
            println!(
                "    → Need ≥{min_text:.0} text tokens/chunk for full hide"
            );
        }
        println!(
            "\n  Min speech density for full hide: {:.1} text tok/chunk  ({:.0} tok/s speech)",
            best_ms / ms_per_text_token,
            (best_ms / ms_per_text_token) / (1.0 / tokens_per_sec_estimate * 1000.0 / 1000.0)
        );
    } else {
        println!("\n[bench] (no --ane path given; skipping Core ML comparison)");
        println!("[bench] to convert: uv run --isolated --python 3.10 \\");
        println!("          --with torch --with transformers --with coremltools==9.0 \\");
        println!("          --with 'numpy<2' --with scipy --with accelerate \\");
        println!("          python ane-vit/converters/convert_qwen3_asr_encoder.py \\");
        println!("            --model {} \\", model_dir.display());
        println!("            --output ane-vit/qwen3-asr-encoder.mlpackage");
    }

    Ok(())
}
