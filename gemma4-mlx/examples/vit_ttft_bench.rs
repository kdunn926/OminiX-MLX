//! End-to-end TTFT-with-image bench for the MLX Gemma4-VL path.
//!
//! Measures the full pipeline a real chat request walks: image bytes →
//! preprocess → vision tower → embed_vision → prefill_multimodal →
//! sample first token. Reports per-stage timings and total TTFT.
//!
//!   cargo run --release -p gemma4-mlx --example vit_ttft_bench -- \
//!     --model models/gemma-4-E4B-it \
//!     --image ane-spike-work/benchmarks/bench_results/img1.png \
//!     --prompt "Describe this image briefly." --iters 5

use coreml_bridge::{ComputeUnits, CoreMlModel};
use gemma4_mlx::{
    build_gemma4_vl_chat_tokens, load_tokenizer, load_vl_model, GemmaVlChatMessage,
};
use image::imageops::FilterType;
use mlx_rs::ops::indexing::{argmax, IndexOp};
use mlx_rs::transforms::eval;
use mlx_rs::Array;
use std::path::PathBuf;
use std::time::Instant;

/// Resize image to a fixed square and normalize to a `[3, H, W]` fp32
/// buffer in [0, 1] — the layout the converted ANE mlpackage expects
/// (the patch_embedder's `2*(x-0.5)` runs inside the traced graph).
fn preprocess_ane_pixels(bytes: &[u8], side: u32) -> anyhow::Result<Vec<f32>> {
    let img = image::load_from_memory(bytes)?;
    let resized = img.resize_exact(side, side, FilterType::CatmullRom);
    let rgb = resized.to_rgb8();
    let n = (side * side) as usize;
    let mut pixels = vec![0f32; 3 * n];
    for y in 0..side {
        for x in 0..side {
            let px = rgb.get_pixel(x, y).0;
            let idx = (y * side + x) as usize;
            pixels[idx] = px[0] as f32 / 255.0;
            pixels[n + idx] = px[1] as f32 / 255.0;
            pixels[2 * n + idx] = px[2] as f32 / 255.0;
        }
    }
    Ok(pixels)
}

fn arg(args: &[String], key: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == key).map(|w| w[1].clone())
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let model_dir = arg(&args, "--model")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/gemma-4-E4B-it"));
    let image_path = arg(&args, "--image")
        .map(PathBuf::from)
        .ok_or_else(|| anyhow::anyhow!("--image PATH required"))?;
    let prompt = arg(&args, "--prompt").unwrap_or_else(|| "Describe this image briefly.".into());
    let iters: usize = arg(&args, "--iters").and_then(|s| s.parse().ok()).unwrap_or(5);
    let async_mode = std::env::args().any(|a| a == "--async");
    let ane_mlpackage = arg(&args, "--ane").map(PathBuf::from);
    let ane_side: u32 = arg(&args, "--ane-side")
        .and_then(|s| s.parse().ok())
        .unwrap_or(384);
    let ane_units = match arg(&args, "--ane-units").as_deref() {
        Some("gpu") | Some("cpuAndGpu") => ComputeUnits::CpuAndGpu,
        Some("ane") | Some("cpuAndAne") => ComputeUnits::CpuAndNeuralEngine,
        Some("cpu") | Some("cpuOnly") => ComputeUnits::CpuOnly,
        _ => ComputeUnits::All,
    };

    println!("[bench] loading {} ...", model_dir.display());
    let load_t = Instant::now();
    let mut model = load_vl_model(&model_dir)?;
    let tokenizer = load_tokenizer(&model_dir)?;
    println!("[bench] loaded in {:.2}s", load_t.elapsed().as_secs_f64());

    let image_bytes = std::fs::read(&image_path)?;
    println!(
        "[bench] image: {} ({} bytes)",
        image_path.display(),
        image_bytes.len()
    );

    // Optional ANE vision tower (Core ML mlpackage).
    let (ane_model, ane_pixels, ane_out_cap, ane_hidden, ane_n_tokens) =
        if let Some(p) = &ane_mlpackage {
            let m = CoreMlModel::load(p, ane_units)?;
            let pixels = preprocess_ane_pixels(&image_bytes, ane_side)?;
            // Output is [N_soft_tokens, hidden]. For the 384-side
            // mlpackage that's [64, 768]; for a 768-retrace it would
            // be [256, 768]. Caller picks via --ane-hidden / --ane-soft.
            let hidden: i32 = arg(&args, "--ane-hidden")
                .and_then(|s| s.parse().ok())
                .unwrap_or(768);
            let soft: i32 = arg(&args, "--ane-soft")
                .and_then(|s| s.parse().ok())
                .unwrap_or(64);
            let cap = (soft * hidden) as usize;
            println!(
                "[bench] ANE mlpackage: {} (side={}, units={:?}, expect {} soft tokens × hidden {})",
                p.display(),
                ane_side,
                ane_units,
                soft,
                hidden
            );
            (Some(m), pixels, cap, hidden, soft)
        } else {
            (None, Vec::new(), 0, 0, 0)
        };

    let image_token_id = model.image_token_id;
    let boi_token_id = model.boi_token_id;
    let eoi_token_id = model.eoi_token_id;

    // Build messages (image attaches to first user turn).
    let make_tokens = |n_vis: usize| -> anyhow::Result<Vec<i32>> {
        let msgs = vec![GemmaVlChatMessage {
            role: "user".into(),
            content: prompt.clone(),
            n_vision_tokens: Some(n_vis),
            has_image: true,
        }];
        Ok(build_gemma4_vl_chat_tokens(
            &tokenizer,
            &msgs,
            image_token_id,
            boi_token_id,
            eoi_token_id,
        )?)
    };

    let mut total_samples = Vec::with_capacity(iters);
    let mut vision_samples = Vec::with_capacity(iters);
    let mut prefill_samples = Vec::with_capacity(iters);

    for it in 0..(iters + 1) {
        let warm = it == 0;
        let t0 = Instant::now();

        // (1) Vision encode. In async mode we DON'T eval here — the
        // visual features stay lazy so MLX can schedule the ViT graph
        // alongside the prefill graph kernels.
        let vt = Instant::now();
        let visual_features = if let Some(ane) = ane_model.as_ref() {
            // ANE path: predict on fixed-side pixels → wrap as MLX Array
            // → run through embed_vision projection on Metal.
            let mut out = vec![0f32; ane_out_cap];
            ane.predict(&ane_pixels, &mut out)
                .map_err(|e| anyhow::anyhow!("ANE predict: {e}"))?;
            let hidden = Array::from_slice(&out, &[1, ane_n_tokens, ane_hidden]);
            let embeds = model
                .embed_vision
                .forward(&hidden)
                .map_err(|e| anyhow::anyhow!("embed_vision: {e}"))?;
            let s1 = embeds.shape()[1];
            let s2 = embeds.shape()[2];
            embeds
                .reshape(&[s1, s2])
                .map_err(|e| anyhow::anyhow!("reshape: {e}"))?
        } else if async_mode {
            model
                .encode_image_bytes_async(&image_bytes)
                .map_err(|e| anyhow::anyhow!("encode_image_bytes_async: {e}"))?
        } else {
            model
                .encode_image_bytes(&image_bytes)
                .map_err(|e| anyhow::anyhow!("encode_image_bytes: {e}"))?
        };
        let n_vis = visual_features.shape()[0] as usize; // lazy shape — no eval
        let vision_ms = if async_mode {
            // Build-only time; the actual GPU work folds into prefill.
            vt.elapsed().as_secs_f64() * 1000.0
        } else {
            eval([&visual_features])?;
            vt.elapsed().as_secs_f64() * 1000.0
        };

        // (2) Build prompt tokens now that we know vision token count.
        let input_ids = make_tokens(n_vis)?;

        // (3) Prefill multimodal.
        let pt = Instant::now();
        let mut cache = model.new_cache();
        let logits = if async_mode {
            model
                .prefill_multimodal_async(&input_ids, &visual_features, &mut cache)
                .map_err(|e| anyhow::anyhow!("prefill_multimodal_async: {e}"))?
        } else {
            model
                .prefill_multimodal(&input_ids, &visual_features, &mut cache)
                .map_err(|e| anyhow::anyhow!("prefill_multimodal: {e}"))?
        };
        // (4) Greedy first-token sample (argmax).
        // `forward_from_embeds` already returns logits of shape [1, V] for the last position.
        let last_logits = logits.index((0i32, ..));
        let next_token = argmax(&last_logits, None)?;
        eval([&next_token])?;
        let prefill_ms = pt.elapsed().as_secs_f64() * 1000.0;

        let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let tok_id = next_token.item::<i32>();
        let tok_str = tokenizer
            .decode(&[tok_id as u32], false)
            .unwrap_or_else(|_| format!("<id={tok_id}>"));
        println!(
            "[bench] iter {}{}: vision={:.1}ms ({} soft tokens) prefill={:.1}ms ({} prompt toks) total={:.1}ms first={:?}",
            it,
            if warm { " (warmup)" } else { "" },
            vision_ms,
            n_vis,
            prefill_ms,
            input_ids.len(),
            total_ms,
            tok_str
        );
        if !warm {
            total_samples.push(total_ms);
            vision_samples.push(vision_ms);
            prefill_samples.push(prefill_ms);
        }
    }

    fn stats(samples: &mut [f64]) -> (f64, f64, f64, f64) {
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = samples.len();
        let mean = samples.iter().sum::<f64>() / n as f64;
        let p50 = samples[n / 2];
        (mean, p50, samples[0], samples[n - 1])
    }
    let (vm, vp50, vmin, vmax) = stats(&mut vision_samples);
    let (pm, pp50, pmin, pmax) = stats(&mut prefill_samples);
    let (tm, tp50, tmin, tmax) = stats(&mut total_samples);
    println!(
        "\n[bench] summary over {} iters (mode: {})",
        iters,
        if async_mode { "async-overlap" } else { "sync" }
    );
    println!(
        "  vision   mean={:.1}ms p50={:.1}ms min={:.1} max={:.1}",
        vm, vp50, vmin, vmax
    );
    println!(
        "  prefill  mean={:.1}ms p50={:.1}ms min={:.1} max={:.1}",
        pm, pp50, pmin, pmax
    );
    println!(
        "  TTFT     mean={:.1}ms p50={:.1}ms min={:.1} max={:.1}",
        tm, tp50, tmin, tmax
    );
    println!(
        "  vision share: {:.1}% of TTFT",
        100.0 * vm / tm
    );
    Ok(())
}
