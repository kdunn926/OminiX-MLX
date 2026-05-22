//! Micro-bench: ANE-routed ViT predict vs (placeholder) MLX-routed ViT.
//!
//! Standalone example for the spike. Pulls the converted GLM-OCR vision
//! tower from `ane-vit-spike/glm-ocr-vit.mlpackage` (created by
//! `converters/convert_vit_to_coreml.py`) and times N forward passes.
//!
//! Run from the workspace root:
//!   cargo run --release \
//!     --manifest-path ane-vit-spike/coreml-bridge/Cargo.toml \
//!     --example ane_vit_bench -- \
//!     --mlpackage ane-vit-spike/glm-ocr-vit.mlpackage \
//!     --iters 32
//!
//! NOTE: this example file currently lives outside the crate's [[example]]
//! table — wire it into `coreml-bridge/Cargo.toml` once the bridge crate
//! builds end-to-end. Until then it serves as a reference harness.

use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mlpackage = arg_value(&args, "--mlpackage")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("ane-vit-spike/glm-ocr-vit.mlpackage"));
    let iters: usize = arg_value(&args, "--iters")
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);
    let compute_units = arg_value(&args, "--compute-units")
        .as_deref()
        .map(|s| match s {
            "all" => coreml_bridge::ComputeUnits::All,
            "cpu" => coreml_bridge::ComputeUnits::CpuOnly,
            "gpu" => coreml_bridge::ComputeUnits::CpuAndGpu,
            "ane" => coreml_bridge::ComputeUnits::CpuAndNeuralEngine,
            _ => coreml_bridge::ComputeUnits::All,
        })
        .unwrap_or(coreml_bridge::ComputeUnits::All);

    eprintln!(
        "[ane-bench] mlpackage={} iters={} compute_units={:?}",
        mlpackage.display(),
        iters,
        compute_units
    );

    let model = match coreml_bridge::CoreMlModel::load(&mlpackage, compute_units) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("load failed: {e}");
            std::process::exit(1);
        }
    };

    // GLM-OCR ViT input: [1, 3, 336, 336] = 338688 f32
    let pixels = vec![0.0f32; 1 * 3 * 336 * 336];
    let mut out = vec![0.0f32; 144 * 1536]; // expected merged-grid×out_hidden

    // Warm-up.
    if let Err(e) = model.predict(&pixels, &mut out) {
        eprintln!("warmup predict failed: {e}");
        std::process::exit(1);
    }

    let mut ms_samples = Vec::with_capacity(iters);
    for i in 0..iters {
        let t0 = Instant::now();
        if let Err(e) = model.predict(&pixels, &mut out) {
            eprintln!("predict {i} failed: {e}");
            std::process::exit(1);
        }
        ms_samples.push(t0.elapsed().as_secs_f64() * 1000.0);
    }

    ms_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = ms_samples.iter().sum::<f64>() / iters as f64;
    let p50 = ms_samples[iters / 2];
    let p99 = ms_samples[(iters * 99) / 100];
    println!(
        "{:?}: mean={:.2}ms p50={:.2}ms p99={:.2}ms min={:.2}ms max={:.2}ms",
        compute_units, mean, p50, p99,
        ms_samples.first().copied().unwrap_or(0.0),
        ms_samples.last().copied().unwrap_or(0.0),
    );
}

fn arg_value(args: &[String], key: &str) -> Option<String> {
    args.windows(2)
        .find(|w| w[0] == key)
        .map(|w| w[1].clone())
}
