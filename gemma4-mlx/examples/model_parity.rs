//! Full-model logits parity vs an mlx-vlm reference dump.
//!
//! Runs the complete Gemma 4 forward pass on the recorded `tokens.npy` and
//! compares logits at the last position to `logits_last.npy`. Both sides use
//! the same 4-bit weights; bf16 accumulation across many layers means a
//! handful of ULPs of difference is expected, but the argmax must match.
//!
//! Generate the reference first with:
//!
//!   python gemma4-mlx/scripts/dump_gemma4_logits.py <MODEL_DIR> <GOLDEN_DIR>
//!
//! Then run:
//!
//!   cargo run --release -p gemma4-mlx --example model_parity -- \
//!     <MODEL_DIR> <GOLDEN_DIR>
//!
//! Auto-detects heterogeneous-quantized checkpoints (`config.json` has a
//! `quantization` block) and routes through `load_ud_mlx_4bit`; otherwise
//! falls back to `load_model`.

use std::path::Path;

use anyhow::{anyhow, Result};
use gemma4_mlx::{init_cache, load_model, ud_loader::load_ud_mlx_4bit, KVCache, ModelInput};
use mlx_rs::{Array, Dtype};

fn read_npy_f32(path: &Path) -> Result<(Vec<f32>, Vec<i32>)> {
    let bytes = std::fs::read(path)?;
    let npy = npyz::NpyFile::new(&bytes[..])?;
    let shape: Vec<i32> = npy.shape().iter().map(|&d| d as i32).collect();
    let data: Vec<f32> = npy.into_vec::<f32>()?;
    Ok((data, shape))
}

fn read_npy_i32(path: &Path) -> Result<(Vec<i32>, Vec<i32>)> {
    let bytes = std::fs::read(path)?;
    let npy = npyz::NpyFile::new(&bytes[..])?;
    let shape: Vec<i32> = npy.shape().iter().map(|&d| d as i32).collect();
    let data: Vec<i32> = npy.into_vec::<i32>()?;
    Ok((data, shape))
}

fn diffs(a: &[f32], b: &[f32]) -> (f32, f32) {
    assert_eq!(a.len(), b.len(), "length mismatch {} vs {}", a.len(), b.len());
    let mut max = 0.0f32;
    let mut sum = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (x - y).abs();
        if d > max {
            max = d;
        }
        sum += d as f64;
    }
    (max, (sum / a.len() as f64) as f32)
}

fn top_k(values: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..values.len()).collect();
    idx.sort_unstable_by(|&a, &b| values[b].partial_cmp(&values[a]).unwrap());
    idx.truncate(k);
    idx
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}

fn is_quantized(model_dir: &Path) -> Result<bool> {
    let raw: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(model_dir.join("config.json"))?)?;
    Ok(raw.get("quantization").map(|q| q.is_object()).unwrap_or(false))
}

fn main() -> Result<()> {
    let mut argv = std::env::args().skip(1);
    let model_dir = argv
        .next()
        .ok_or_else(|| anyhow!("usage: model_parity <MODEL_DIR> <GOLDEN_DIR>"))?;
    let golden_dir = argv
        .next()
        .ok_or_else(|| anyhow!("usage: model_parity <MODEL_DIR> <GOLDEN_DIR>"))?;
    let model_dir = Path::new(&model_dir);
    let golden_dir = Path::new(&golden_dir);

    println!("model_dir  : {}", model_dir.display());
    println!("golden_dir : {}", golden_dir.display());

    let (tok_data, tok_shape) = read_npy_i32(&golden_dir.join("tokens.npy"))?;
    println!("tokens shape {:?}  values {:?}", tok_shape, &tok_data);
    let tokens = Array::from_slice(&tok_data, &tok_shape);

    let (ref_logits, ref_shape) = read_npy_f32(&golden_dir.join("logits_last.npy"))?;
    println!("ref logits shape {:?}  len {}", ref_shape, ref_logits.len());
    let ref_argmax = argmax(&ref_logits);
    let ref_top5 = top_k(&ref_logits, 5);
    println!("ref argmax : {ref_argmax}  top5 : {ref_top5:?}");

    let quant = is_quantized(model_dir)?;
    println!(
        "loader     : {}",
        if quant { "load_ud_mlx_4bit (quantized)" } else { "load_model (bf16)" }
    );
    let mut model = if quant { load_ud_mlx_4bit(model_dir)? } else { load_model(model_dir)? };
    let num_slots = *model.model.kv_cache_map.iter().max().unwrap_or(&0) + 1;
    let mut cache: Vec<KVCache> = init_cache(num_slots);

    let logits = model.forward_last_logits(ModelInput {
        inputs: &tokens,
        mask: None,
        cache: &mut cache,
    })?;
    logits.eval()?;
    println!("logits shape : {:?}", logits.shape());

    let last = logits.as_dtype(Dtype::Float32)?;
    last.eval()?;
    let ours: Vec<f32> = last.as_slice::<f32>().to_vec();

    let our_argmax = argmax(&ours);
    let our_top5 = top_k(&ours, 5);
    let (max_abs_diff, mean_abs_diff) = diffs(&ours, &ref_logits);
    let our_absmax = ours.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    let ref_absmax = ref_logits.iter().fold(0.0f32, |m, &v| m.max(v.abs()));

    println!("\n========== MODEL PARITY RESULTS ==========");
    println!("  rust_argmax    : {our_argmax}");
    println!("  ref_argmax     : {ref_argmax}");
    println!("  MATCH          : {}", our_argmax == ref_argmax);
    println!("  our top5       : {our_top5:?}");
    println!("  ref top5       : {ref_top5:?}");
    println!("  our absmax     : {our_absmax:.4}");
    println!("  ref absmax     : {ref_absmax:.4}");
    println!("  max-abs-diff   : {max_abs_diff:.6e}");
    println!("  mean-abs-diff  : {mean_abs_diff:.6e}");
    println!("==========================================");

    if our_argmax == ref_argmax {
        println!("PASS — argmax matches");
        Ok(())
    } else {
        eprintln!(
            "FAIL — rust_argmax ({our_argmax}) != ref_argmax ({ref_argmax}); \
             our top5 {our_top5:?} vs ref top5 {ref_top5:?}"
        );
        std::process::exit(1);
    }
}
