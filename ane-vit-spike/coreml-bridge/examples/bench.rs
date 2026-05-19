//! ANE vs CPU vs GPU ViT predict bench.
//!
//! Reuses the same model loaded with different `compute_units` so each
//! variant exercises a different dispatch policy in Core ML.

use coreml_bridge::{ComputeUnits, CoreMlModel};
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mlpackage = arg(&args, "--mlpackage")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("../vit-base.mlpackage"));
    let iters: usize = arg(&args, "--iters").and_then(|s| s.parse().ok()).unwrap_or(32);

    let zero_copy = std::env::args().any(|a| a == "--zero-copy");
    // ViT-base/224 input: 1 × 3 × 224 × 224 = 150_528 f32
    let mut pixels = vec![0.0f32; 1 * 3 * 224 * 224];
    // 197 = 1 cls + 14×14 patches; hidden 768. Output capacity buffer.
    let mut out = vec![0.0f32; 197 * 768];

    for (label, units) in [
        ("all", ComputeUnits::All),
        ("cpuOnly", ComputeUnits::CpuOnly),
        ("cpuAndGpu", ComputeUnits::CpuAndGpu),
        ("cpuAndAne", ComputeUnits::CpuAndNeuralEngine),
    ] {
        eprintln!("\n[bench] loading {} ({:?})", mlpackage.display(), units);
        let model = match CoreMlModel::load(&mlpackage, units) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("  load failed: {e}");
                continue;
            }
        };
        // Warmup.
        let warmup = if zero_copy {
            model.predict_zero_copy(&mut pixels, &mut out)
        } else {
            model.predict(&pixels, &mut out)
        };
        if let Err(e) = warmup {
            eprintln!("  warmup failed: {e}");
            continue;
        }
        let mut samples = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t = Instant::now();
            let r = if zero_copy {
                model.predict_zero_copy(&mut pixels, &mut out)
            } else {
                model.predict(&pixels, &mut out)
            };
            if let Err(e) = r {
                eprintln!("  predict failed: {e}");
                samples.clear();
                break;
            }
            samples.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        if samples.is_empty() {
            continue;
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let p50 = samples[samples.len() / 2];
        let p99 = samples[(samples.len() * 99) / 100];
        let min = *samples.first().unwrap();
        let max = *samples.last().unwrap();
        println!(
            "{:12} mean={:.2}ms p50={:.2}ms p99={:.2}ms min={:.2}ms max={:.2}ms ({} iters)",
            label, mean, p50, p99, min, max, samples.len()
        );
    }
}

fn arg(args: &[String], key: &str) -> Option<String> {
    args.windows(2)
        .find(|w| w[0] == key)
        .map(|w| w[1].clone())
}
