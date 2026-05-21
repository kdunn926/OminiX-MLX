//! UD-MLX-4bit Gemma4 loader.
//!
//! The Unsloth "UD" (Unsloth Dynamic) MLX 4-bit quantization for Gemma4
//! ships with three differences from the canonical mlx-community
//! gemma-4-26B-A4B-it format:
//!
//! 1. **Key prefix**: tensors are named `language_model.model.*` rather
//!    than `model.language_model.*` (same swap as MTPLX target).
//! 2. **MoE layout**: experts use the `switch_glu` submodule with three
//!    separate per-expert projections (`gate_proj`, `up_proj`, `down_proj`)
//!    rather than the canonical fused `gate_up_proj` + `down_proj`.
//! 3. **Heterogeneous quantization**: the `quantization` config block
//!    lists per-tensor `{bits, group_size}` overrides — e.g. attention
//!    projections at Q8, MLP projections at Q4, embed_tokens at Q6. The
//!    canonical loader assumes a single global `(bits, group_size)`.
//!
//! This loader dequantizes every weight to bf16 at load time using
//! `mlx_rs::ops::dequantize` with the correct per-tensor `(bits,
//! group_size)`. The MoE switch_glu `gate_proj` and `up_proj` per-expert
//! tensors are concatenated along the output axis to synthesize the
//! canonical `[E, 2I, H]` `gate_up_proj` tensor. The dequantized weights
//! are then handed to the canonical `build_model_from_weights` with
//! `quantization=None`, so inference runs through the existing fp16/bf16
//! MoE forward path (forward_topk, v6, etc.) unchanged.
//!
//! Memory: post-dequant resident is ~50 GB on Gemma4-26B-A4B-it (same as
//! the canonical bf16 model). Disk savings (~15 GB packed vs ~48 GB
//! bf16) only apply to load-time bandwidth. The UD format's value is
//! accuracy on disk — heterogeneous bit-widths preserve attention
//! precision while compressing the bulky MLP weights.

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::{ops::dequantize, Array, Dtype};
use mlx_rs_core::Error;
use serde_json::Value;

use crate::model::{build_model_from_weights, get_model_args, load_all_weights_unfiltered, Model};

/// Default quant config (used when no per-tensor override is present).
struct DefaultQuant {
    bits: i32,
    group_size: i32,
}

/// Per-tensor quantization override.
struct PerTensorQuant {
    bits: i32,
    group_size: i32,
}

/// Parse the `quantization` block from config.json. Returns the default
/// (bits, group_size) plus a map of per-tensor overrides keyed by the
/// original `language_model.model.*` path (or other top-level paths used
/// in the config block).
fn parse_quant_config(
    config_json: &Value,
) -> Result<(DefaultQuant, HashMap<String, PerTensorQuant>), Error> {
    let q = match config_json.get("quantization").and_then(Value::as_object) {
        Some(o) => o,
        None => {
            return Ok((
                DefaultQuant { bits: 4, group_size: 64 },
                HashMap::new(),
            ));
        }
    };
    let default_bits = q
        .get("bits")
        .and_then(Value::as_i64)
        .map(|v| v as i32)
        .unwrap_or(4);
    let default_gs = q
        .get("group_size")
        .and_then(Value::as_i64)
        .map(|v| v as i32)
        .unwrap_or(64);

    let mut overrides: HashMap<String, PerTensorQuant> = HashMap::new();
    for (k, v) in q.iter() {
        if k == "bits" || k == "group_size" || k == "mode" {
            continue;
        }
        if let Some(obj) = v.as_object() {
            let bits = obj
                .get("bits")
                .and_then(Value::as_i64)
                .map(|x| x as i32)
                .unwrap_or(default_bits);
            let gs = obj
                .get("group_size")
                .and_then(Value::as_i64)
                .map(|x| x as i32)
                .unwrap_or(default_gs);
            overrides.insert(k.clone(), PerTensorQuant { bits, group_size: gs });
        }
    }
    Ok((
        DefaultQuant { bits: default_bits, group_size: default_gs },
        overrides,
    ))
}

/// Per-tensor lookup helper: returns the (bits, group_size) for a given
/// tensor prefix (without `.weight` / `.scales` / `.biases` suffix).
fn lookup_quant(
    prefix: &str,
    default: &DefaultQuant,
    overrides: &HashMap<String, PerTensorQuant>,
) -> (i32, i32) {
    if let Some(p) = overrides.get(prefix) {
        (p.bits, p.group_size)
    } else {
        (default.bits, default.group_size)
    }
}

/// MTPLX-style prefix swap so downstream code sees canonical
/// `model.language_model.*` names.
fn canonical(key: &str) -> String {
    if let Some(rest) = key.strip_prefix("language_model.model.") {
        format!("model.language_model.{rest}")
    } else if let Some(rest) = key.strip_prefix("language_model.lm_head.") {
        format!("lm_head.{rest}")
    } else {
        key.to_string()
    }
}

/// Dequantize a `(weight, scales, biases)` triple to bf16.
fn dequant_triple(
    weight: &Array,
    scales: &Array,
    biases: &Array,
    group_size: i32,
    bits: i32,
) -> Result<Array, Error> {
    let out = dequantize(weight, scales, biases, group_size, bits, None::<&str>)
        .map_err(|e| Error::Model(format!("dequantize failed: {e:?}")))?;
    out.as_dtype(Dtype::Bfloat16)
        .map_err(|e| Error::Model(format!("dequant → bf16 cast: {e:?}")))
}

/// Walk the raw safetensors map and produce a canonical bf16 weights map:
///   - dequantize every `(.weight, .scales, .biases)` triple using the
///     correct per-tensor `(bits, group_size)`
///   - fuse switch_glu's `gate_proj` + `up_proj` into `gate_up_proj`
///   - rename keys to the canonical `model.language_model.*` prefix
fn synthesize_canonical_weights(
    raw: HashMap<String, Array>,
    default_q: &DefaultQuant,
    overrides: &HashMap<String, PerTensorQuant>,
) -> Result<HashMap<String, Array>, Error> {
    // First pass: collect (.weight, .scales, .biases) triples by prefix.
    // Any tensor that doesn't have a .scales sibling is treated as plain
    // (unquantized) and carried through as-is.
    let mut by_prefix: HashMap<String, HashMap<&'static str, Array>> = HashMap::new();
    let mut plain: HashMap<String, Array> = HashMap::new();

    for (k, v) in raw.into_iter() {
        let (prefix, suffix) = if let Some(p) = k.strip_suffix(".weight") {
            (p.to_string(), "weight")
        } else if let Some(p) = k.strip_suffix(".scales") {
            (p.to_string(), "scales")
        } else if let Some(p) = k.strip_suffix(".biases") {
            (p.to_string(), "biases")
        } else {
            // No quant suffix — carry through as plain.
            plain.insert(k, v);
            continue;
        };
        by_prefix.entry(prefix).or_default().insert(suffix, v);
    }

    let mut out: HashMap<String, Array> = HashMap::new();
    // Stage 1: dequantize any triple that has all three components;
    // promote singletons (`.weight` only) to plain.
    let mut dequant: HashMap<String, Array> = HashMap::new();
    for (prefix, parts) in by_prefix.into_iter() {
        let weight = parts.get("weight").cloned();
        let scales = parts.get("scales").cloned();
        let biases = parts.get("biases").cloned();
        match (weight, scales, biases) {
            (Some(w), Some(s), Some(b)) => {
                let (bits, gs) = lookup_quant(&prefix, default_q, overrides);
                let deq = dequant_triple(&w, &s, &b, gs, bits)?;
                dequant.insert(prefix, deq);
            }
            (Some(w), None, None) => {
                plain.insert(format!("{prefix}.weight"), w);
            }
            _ => {
                return Err(Error::Model(format!(
                    "ud_loader: incomplete quant triple at {prefix}"
                )));
            }
        }
    }

    // Stage 2: for each layer, fuse switch_glu.{gate_proj, up_proj} into
    // a single gate_up_proj along the output axis. switch_glu tensors
    // have shape [E, I, H] after dequant; concat axis=1 yields [E, 2I, H].
    let mut switch_glu_layers: HashMap<i32, [Option<Array>; 3]> = HashMap::new();
    // index 0 = gate, 1 = up, 2 = down
    let mut leftover: HashMap<String, Array> = HashMap::new();

    for (prefix, arr) in dequant.into_iter() {
        // Match pattern: language_model.model.layers.{N}.experts.switch_glu.{which}_proj
        if let Some(rest) = prefix.strip_prefix("language_model.model.layers.") {
            if let Some(dot) = rest.find('.') {
                let layer_str = &rest[..dot];
                let tail = &rest[dot + 1..];
                if let Ok(layer) = layer_str.parse::<i32>() {
                    let which = match tail {
                        "experts.switch_glu.gate_proj" => Some(0_usize),
                        "experts.switch_glu.up_proj" => Some(1_usize),
                        "experts.switch_glu.down_proj" => Some(2_usize),
                        _ => None,
                    };
                    if let Some(idx) = which {
                        let slot = switch_glu_layers.entry(layer).or_insert([None, None, None]);
                        slot[idx] = Some(arr);
                        continue;
                    }
                }
            }
        }
        leftover.insert(prefix, arr);
    }

    // Emit fused gate_up_proj + down_proj per layer with canonical names.
    for (layer, parts) in switch_glu_layers.into_iter() {
        let [gate, up, down] = parts;
        let (gate, up, down) = match (gate, up, down) {
            (Some(g), Some(u), Some(d)) => (g, u, d),
            _ => {
                return Err(Error::Model(format!(
                    "ud_loader: layer {layer} missing one of switch_glu.{{gate,up,down}}_proj"
                )));
            }
        };
        let gate_up = mlx_rs::ops::concatenate_axis(&[&gate, &up], 1)
            .map_err(|e| Error::Model(format!("layer {layer} concat gate+up: {e:?}")))?;
        out.insert(
            format!("model.language_model.layers.{layer}.experts.gate_up_proj"),
            gate_up,
        );
        out.insert(
            format!("model.language_model.layers.{layer}.experts.down_proj"),
            down,
        );
    }

    // Stage 3: emit remaining dequantized tensors with `.weight` suffix
    // and canonical prefix.
    for (prefix, arr) in leftover.into_iter() {
        out.insert(format!("{}.weight", canonical(&prefix)), arr);
    }

    // Stage 4: plain (non-quantized) tensors — translate prefix, carry through.
    for (k, v) in plain.into_iter() {
        out.insert(canonical(&k), v);
    }

    Ok(out)
}

/// Load `models/gemma4-26B-a4b-it-UD-MLX-4bit/` (or any UD-MLX-4bit
/// Gemma4 variant) and return a canonical `Model`. Dequantizes per-tensor
/// using the heterogeneous bit widths declared in the model's
/// `config.json::quantization` block.
pub fn load_ud_mlx_4bit(model_dir: impl AsRef<Path>) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    let mut config = get_model_args(model_dir)?;
    let args = config.text_config.clone();

    // Parse heterogeneous quant config directly from the JSON so we get
    // the full per-tensor override map.
    let cfg_path = model_dir.join("config.json");
    let cfg_json: Value = serde_json::from_reader(std::fs::File::open(&cfg_path)?)?;
    let (default_q, overrides) = parse_quant_config(&cfg_json)?;

    let raw = load_all_weights_unfiltered(model_dir)?;
    let weights = synthesize_canonical_weights(raw, &default_q, &overrides)?;

    // The synthesized weights are bf16 (post-dequant). Tell the builder to
    // expect an unquantized model so it routes through the canonical
    // bf16 path (forward_topk + v6 / MaybeQuantized::Original).
    config.quantization = None;
    build_model_from_weights(&config, args, &weights)
}
