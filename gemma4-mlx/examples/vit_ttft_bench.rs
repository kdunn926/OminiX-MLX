//! End-to-end TTFT-with-image bench for the MLX Gemma4-VL path.
//!
//! Measures the full pipeline a real chat request walks: image bytes →
//! preprocess → vision tower → embed_vision → prefill_multimodal →
//! sample first token. Reports per-stage timings and total TTFT.
//!
//! Modes
//! ------
//! Default (sync):  sequential ANE predict → GPU prefill.
//! --async:         MLX lazy-graph vision tower overlapped with prefill.
//! --overlap:       ANE predict in background thread; GPU prefills the
//!                  text prefix concurrently; join before the image span.
//!                  Requires --ane.
//! --parity-check:  Run both ANE and MLX ViT on the same image; report
//!                  mean and min cosine similarity of the projected soft
//!                  tokens.  Requires --ane.
//!
//! Usage
//! -----
//!   cargo run --release -p gemma4-mlx --example vit_ttft_bench -- \
//!     --model models/gemma-4-E4B-it \
//!     --image ane-spike-work/benchmarks/bench_results/img1.png \
//!     --prompt "Describe this image briefly." --iters 5
//!
//!   # background-thread overlap experiment
//!   cargo run --release -p gemma4-mlx --example vit_ttft_bench -- \
//!     --model models/gemma-4-E4B-it \
//!     --image img.png \
//!     --ane ane-vit/gemma4-e4b-vit.mlpackage --overlap --iters 5
//!
//!   # long-system-prompt case (shows more overlap benefit)
//!   cargo run --release -p gemma4-mlx --example vit_ttft_bench -- \
//!     --model models/gemma-4-E4B-it --image img.png \
//!     --ane ane-vit/gemma4-e4b-vit.mlpackage --overlap \
//!     --system-prompt "You are a helpful assistant. ..." --iters 5

use coreml_bridge::{ComputeUnits, CoreMlModel};
use gemma4_mlx::{
    build_gemma4_vl_chat_tokens, load_tokenizer, load_vl_model, GemmaVlChatMessage,
};
use image::imageops::FilterType;
use mlx_rs::ops::indexing::{argmax, IndexOp};
use mlx_rs::transforms::eval;
use mlx_rs::Array;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
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

/// Parse a sidecar manifest JSON and return `(num_patches, hidden_size)` if
/// present — used to drive `ComputeUnits::recommended_for` when no explicit
/// `--ane-units` flag is given.
fn read_manifest_recommendation(mlpackage: &std::path::Path) -> Option<ComputeUnits> {
    let manifest_path = mlpackage.with_extension("manifest.json");
    let text = std::fs::read_to_string(&manifest_path).ok()?;
    let v: serde_json::Value = serde_json::from_str(&text).ok()?;
    // Prefer an explicit recommended_units field written by the converter.
    if let Some(rec) = v["recommended_units"].as_str() {
        return match rec {
            "cpuAndNeuralEngine" => Some(ComputeUnits::CpuAndNeuralEngine),
            "cpuAndGpu" | "cpuAndGPU" => Some(ComputeUnits::CpuAndGpu),
            "cpuOnly" => Some(ComputeUnits::CpuOnly),
            _ => None,
        };
    }
    // Fall back to computing it from patch_count + hidden_size.
    let patches = v["num_patches"].as_u64()? as usize;
    let hidden = v["hidden_size"].as_u64()? as usize;
    Some(ComputeUnits::recommended_for(patches, hidden))
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
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
    let system_prompt = arg(&args, "--system-prompt")
        .unwrap_or_else(|| "You are a helpful assistant.".into());
    let iters: usize = arg(&args, "--iters").and_then(|s| s.parse().ok()).unwrap_or(5);
    let async_mode = std::env::args().any(|a| a == "--async");
    let overlap_mode = std::env::args().any(|a| a == "--overlap");
    let parity_check = std::env::args().any(|a| a == "--parity-check");
    let ane_mlpackage = arg(&args, "--ane").map(PathBuf::from);
    let ane_side: u32 = arg(&args, "--ane-side")
        .and_then(|s| s.parse().ok())
        .unwrap_or(384);
    let ane_units = match arg(&args, "--ane-units").as_deref() {
        Some("gpu") | Some("cpuAndGpu") | Some("cpuAndGPU") => Some(ComputeUnits::CpuAndGpu),
        Some("ane") | Some("cpuAndAne") | Some("cpuAndNeuralEngine") => {
            Some(ComputeUnits::CpuAndNeuralEngine)
        }
        Some("cpu") | Some("cpuOnly") => Some(ComputeUnits::CpuOnly),
        Some("all") => Some(ComputeUnits::All),
        None => None,
        _ => None,
    };

    if (overlap_mode || parity_check) && ane_mlpackage.is_none() {
        anyhow::bail!("--overlap and --parity-check require --ane <path>");
    }

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
    let (ane_model_opt, ane_pixels, ane_out_cap, ane_hidden, ane_n_tokens) =
        if let Some(p) = &ane_mlpackage {
            // Resolve compute units: explicit flag → manifest recommendation → All.
            let units = ane_units
                .or_else(|| read_manifest_recommendation(p))
                .unwrap_or(ComputeUnits::All);
            let m = CoreMlModel::load(p, units)?;
            let pixels = preprocess_ane_pixels(&image_bytes, ane_side)?;
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
                units,
                soft,
                hidden
            );
            (Some(Arc::new(Mutex::new(m))), pixels, cap, hidden, soft)
        } else {
            (None, Vec::new(), 0, 0, 0)
        };

    let image_token_id = model.image_token_id;
    let boi_token_id = model.boi_token_id;
    let eoi_token_id = model.eoi_token_id;

    // Build messages (system prompt + user turn with image).
    let make_tokens = |n_vis: usize| -> anyhow::Result<Vec<i32>> {
        let msgs = vec![
            GemmaVlChatMessage {
                role: "system".into(),
                content: system_prompt.clone(),
                n_vision_tokens: None,
                has_image: false,
            },
            GemmaVlChatMessage {
                role: "user".into(),
                content: prompt.clone(),
                n_vision_tokens: Some(n_vis),
                has_image: true,
            },
        ];
        Ok(build_gemma4_vl_chat_tokens(
            &tokenizer,
            &msgs,
            image_token_id,
            boi_token_id,
            eoi_token_id,
        )?)
    };

    // ── parity check ─────────────────────────────────────────────────────────
    if parity_check {
        println!("\n[parity] comparing ANE and MLX ViT projected soft tokens ...");
        let ane_arc = ane_model_opt.as_ref().unwrap();

        // ANE path.
        let mut ane_out = vec![0f32; ane_out_cap];
        ane_arc.lock().unwrap().predict(&ane_pixels, &mut ane_out)
            .map_err(|e| anyhow::anyhow!("ANE predict: {e}"))?;
        let ane_hidden_arr = Array::from_slice(&ane_out, &[1, ane_n_tokens, ane_hidden]);
        let ane_proj = model.embed_vision.forward(&ane_hidden_arr)
            .map_err(|e| anyhow::anyhow!("embed_vision (ANE): {e}"))?;
        let ane_proj = ane_proj.reshape(&[ane_n_tokens, -1])
            .map_err(|e| anyhow::anyhow!("reshape ANE: {e}"))?;
        eval([&ane_proj])?;
        let ane_proj_f32 = ane_proj.as_dtype(mlx_rs::Dtype::Float32)?;
        eval([&ane_proj_f32])?;

        // MLX ViT path.
        let mlx_features = model.encode_image_bytes(&image_bytes)
            .map_err(|e| anyhow::anyhow!("encode_image_bytes (MLX): {e}"))?;
        eval([&mlx_features])?;
        let mlx_f32 = mlx_features.as_dtype(mlx_rs::Dtype::Float32)?;
        eval([&mlx_f32])?;

        let ane_slice = ane_proj_f32.try_as_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("ANE slice: {e}"))?;
        let mlx_slice = mlx_f32.try_as_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("MLX slice: {e}"))?;

        let ane_toks = ane_proj_f32.shape()[0] as usize;
        let mlx_toks = mlx_f32.shape()[0] as usize;
        let hidden_dim = ane_proj_f32.shape()[1] as usize;
        println!(
            "[parity] ANE tokens={} MLX tokens={} hidden={}",
            ane_toks, mlx_toks, hidden_dim
        );

        // Mean-pool both and compare (works even if token counts differ).
        let mean_pool = |slice: &[f32], n_toks: usize, dim: usize| -> Vec<f32> {
            let mut out = vec![0f32; dim];
            for row in slice.chunks_exact(dim) {
                for (o, v) in out.iter_mut().zip(row) {
                    *o += v;
                }
            }
            for o in &mut out {
                *o /= n_toks as f32;
            }
            out
        };
        let ane_mean = mean_pool(ane_slice, ane_toks, hidden_dim);
        let mlx_mean = mean_pool(mlx_slice, mlx_toks, hidden_dim);
        let mean_cos = cosine_similarity(&ane_mean, &mlx_mean);
        println!("[parity] mean-pooled cosine similarity: {:.4}", mean_cos);

        // Per-token cosine sim when counts match (same preprocessing shape).
        if ane_toks == mlx_toks {
            let mut sims: Vec<f32> = (0..ane_toks)
                .map(|i| {
                    let a = &ane_slice[i * hidden_dim..(i + 1) * hidden_dim];
                    let b = &mlx_slice[i * hidden_dim..(i + 1) * hidden_dim];
                    cosine_similarity(a, b)
                })
                .collect();
            sims.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mean_tok: f32 = sims.iter().sum::<f32>() / sims.len() as f32;
            println!(
                "[parity] per-token cosine: mean={:.4} min={:.4} p25={:.4}",
                mean_tok,
                sims[0],
                sims[sims.len() / 4]
            );
        } else {
            println!(
                "[parity] token counts differ ({} vs {}) — per-token comparison skipped.\n\
                 Note: ANE path squashes to {}×{} (aspect-distorting); MLX path uses\n\
                 aspect-preserving resize — preprocessing mismatch is expected.",
                ane_toks, mlx_toks, ane_side, ane_side
            );
        }
        println!(
            "[parity] interpretation: cosine ≥ 0.99 → essentially equivalent;\n\
             0.95–0.99 → minor fp16/bf16 delta; < 0.95 → meaningful divergence"
        );
        return Ok(());
    }

    // ── timing loop ──────────────────────────────────────────────────────────
    let mode_str = if overlap_mode {
        "overlap (ANE-thread + GPU-prefix)"
    } else if async_mode {
        "async-overlap (MLX lazy graph)"
    } else if ane_model_opt.is_some() {
        "sync-ane"
    } else {
        "sync-mlx"
    };

    let mut total_samples = Vec::with_capacity(iters);
    let mut vision_samples = Vec::with_capacity(iters);
    let mut prefill_samples = Vec::with_capacity(iters);
    // Overlap mode: track how much of ANE was hidden behind prefix prefill.
    let mut prefix_samples: Vec<f64> = Vec::with_capacity(iters);
    let mut ane_samples: Vec<f64> = Vec::with_capacity(iters);

    for it in 0..(iters + 1) {
        let warm = it == 0;
        let t0 = Instant::now();

        let vt = Instant::now();

        if overlap_mode {
            // ── overlap path ──────────────────────────────────────────────
            // 1. Build token sequence using known ANE soft-token count.
            let input_ids = make_tokens(ane_n_tokens as usize)?;
            let prefix_len = input_ids
                .iter()
                .position(|&id| id as u32 == image_token_id)
                .unwrap_or(input_ids.len());
            let prefix_ids = input_ids[..prefix_len].to_vec();
            let continuation_ids = input_ids[prefix_len..].to_vec();

            // 2. Spawn ANE predict thread. CoreMlModel is Send; Arc<Mutex<>>
            //    lets us clone the handle into the thread each iteration.
            let ane_arc = ane_model_opt.as_ref().unwrap().clone();
            let pixels_clone = ane_pixels.clone();
            let cap = ane_out_cap;
            let (tx, rx) = std::sync::mpsc::channel::<anyhow::Result<Vec<f32>>>();
            std::thread::spawn(move || {
                let mut out = vec![0f32; cap];
                let res = ane_arc
                    .lock()
                    .unwrap()
                    .predict(&pixels_clone, &mut out)
                    .map(|_| out)
                    .map_err(|e| anyhow::anyhow!("ANE predict: {e}"));
                let _ = tx.send(res);
            });

            // 3. GPU: prefill text prefix while ANE runs.
            let pt0 = Instant::now();
            let mut cache = model.new_cache();
            if !prefix_ids.is_empty() {
                model
                    .prefill_text(&prefix_ids, &mut cache)
                    .map_err(|e| anyhow::anyhow!("prefill_text prefix: {e}"))?;
                eval([] as [&Array; 0])?;
            }
            let prefix_ms = pt0.elapsed().as_secs_f64() * 1000.0;

            // 4. Join ANE thread.
            let ane_out = rx.recv().unwrap()?;
            let ane_elapsed_ms = vt.elapsed().as_secs_f64() * 1000.0 - prefix_ms;
            let vision_ms = vt.elapsed().as_secs_f64() * 1000.0;

            // 5. Wrap ANE output as MLX Array and project to LM hidden size.
            let hidden = Array::from_slice(&ane_out, &[1, ane_n_tokens, ane_hidden]);
            let embeds = model
                .embed_vision
                .forward(&hidden)
                .map_err(|e| anyhow::anyhow!("embed_vision: {e}"))?;
            let s1 = embeds.shape()[1];
            let s2 = embeds.shape()[2];
            let visual_features = embeds
                .reshape(&[s1, s2])
                .map_err(|e| anyhow::anyhow!("reshape: {e}"))?;

            // 6. GPU: prefill image span + suffix from existing cache state.
            let pt = Instant::now();
            let logits = model
                .prefill_multimodal(&continuation_ids, &visual_features, &mut cache)
                .map_err(|e| anyhow::anyhow!("prefill_multimodal continuation: {e}"))?;
            let last_logits = logits.index((0i32, ..));
            let next_token = argmax(&last_logits, None)?;
            eval([&next_token])?;
            let prefill_ms = pt.elapsed().as_secs_f64() * 1000.0;
            let total_ms = t0.elapsed().as_secs_f64() * 1000.0;

            let tok_id = next_token.item::<i32>();
            let tok_str = tokenizer
                .decode(&[tok_id as u32], false)
                .unwrap_or_else(|_| format!("<id={tok_id}>"));
            let hidden_ms = (prefix_ms - ane_elapsed_ms.max(0.0)).max(0.0);
            println!(
                "[bench] iter {}{}: prefix={:.1}ms ane={:.1}ms (hidden={:.1}ms) \
                 prefill={:.1}ms total={:.1}ms first={:?}",
                it,
                if warm { " (warmup)" } else { "" },
                prefix_ms,
                ane_elapsed_ms,
                hidden_ms,
                prefill_ms,
                total_ms,
                tok_str,
            );
            if !warm {
                total_samples.push(total_ms);
                vision_samples.push(vision_ms);
                prefill_samples.push(prefill_ms);
                prefix_samples.push(prefix_ms);
                ane_samples.push(ane_elapsed_ms);
            }
        } else {
            // ── sync / async-MLX path (original behaviour) ────────────────
            let visual_features = if let Some(ane) = ane_model_opt.as_ref() {
                let mut out = vec![0f32; ane_out_cap];
                ane.lock()
                    .unwrap()
                    .predict(&ane_pixels, &mut out)
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
            let n_vis = visual_features.shape()[0] as usize;
            let vision_ms = if async_mode {
                vt.elapsed().as_secs_f64() * 1000.0
            } else {
                eval([&visual_features])?;
                vt.elapsed().as_secs_f64() * 1000.0
            };

            let input_ids = make_tokens(n_vis)?;

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
                "[bench] iter {}{}: vision={:.1}ms ({} soft tokens) prefill={:.1}ms \
                 ({} prompt toks) total={:.1}ms first={:?}",
                it,
                if warm { " (warmup)" } else { "" },
                vision_ms,
                n_vis,
                prefill_ms,
                input_ids.len(),
                total_ms,
                tok_str,
            );
            if !warm {
                total_samples.push(total_ms);
                vision_samples.push(vision_ms);
                prefill_samples.push(prefill_ms);
            }
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
    println!("\n[bench] summary over {} iters (mode: {})", iters, mode_str);
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
    println!("  vision share: {:.1}% of TTFT", 100.0 * vm / tm);

    if overlap_mode && !ane_samples.is_empty() {
        let (am, _, _, _) = stats(&mut ane_samples.clone());
        let (prm, _, _, _) = stats(&mut prefix_samples.clone());
        let theoretical_hidden = prm.min(am);
        println!(
            "\n[overlap] prefix prefill mean={:.1}ms  ANE (net) mean={:.1}ms",
            prm, am
        );
        println!(
            "  theoretical hidden: {:.1}ms ({:.1}% of ANE time)",
            theoretical_hidden,
            100.0 * theoretical_hidden / am.max(0.001)
        );
        println!(
            "  interpretation: if prefix_prefill > ANE time, ANE is fully hidden."
        );
    }
    Ok(())
}
