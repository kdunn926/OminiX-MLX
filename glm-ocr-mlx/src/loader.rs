//! Safetensors loader for GLM-OCR-bf16 (mlx-community conversion).
//!
//! Builds a fully-loaded [`GlmOcrModel`] (vision tower + downsample +
//! merger + Glm4 text decoder + tokenizer + special tokens + image
//! preprocessor) from a HF-format checkpoint directory.
//!
//! Weight names mirror the layout that ships in
//! `mlx-community/GLM-OCR-bf16`:
//!
//! ```text
//!   vision_tower.patch_embed.proj.{weight,bias}        // (out, kT, kH, kW, in)
//!   vision_tower.blocks.{i}.norm1.weight               // RMS
//!   vision_tower.blocks.{i}.norm2.weight
//!   vision_tower.blocks.{i}.attn.qkv.{weight,bias}     // fused
//!   vision_tower.blocks.{i}.attn.q_norm.weight         // (head_dim,)
//!   vision_tower.blocks.{i}.attn.k_norm.weight
//!   vision_tower.blocks.{i}.attn.proj.{weight,bias}
//!   vision_tower.blocks.{i}.mlp.{gate,up,down}_proj.{weight,bias}
//!   vision_tower.post_layernorm.weight
//!   vision_tower.downsample.{weight,bias}              // (out, m, m, in)
//!   vision_tower.merger.proj.weight
//!   vision_tower.merger.post_projection_norm.{weight,bias}  // LayerNorm
//!   vision_tower.merger.{gate,up,down}_proj.weight
//!
//!   language_model.model.embed_tokens.weight
//!   language_model.model.layers.{i}.input_layernorm.weight
//!   language_model.model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
//!   language_model.model.layers.{i}.post_self_attn_layernorm.weight
//!   language_model.model.layers.{i}.post_attention_layernorm.weight
//!   language_model.model.layers.{i}.mlp.gate_up_proj.weight   // fused
//!   language_model.model.layers.{i}.mlp.down_proj.weight
//!   language_model.model.layers.{i}.post_mlp_layernorm.weight
//!   language_model.model.norm.weight
//!   language_model.lm_head.weight
//! ```
//!
//! Patch-embed weight ships as `(out, kT, kH, kW, in)` — channels-last,
//! matching the (t, h, w, c) row order the phase-1 preprocessor emits;
//! we flatten it to `(out, kT*kH*kW*in)` for a Linear consumer.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use mlx_rs::{module::Param, nn, Array, Dtype};
use mlx_rs_core::cache::KVCache;
use mlx_rs_core::error::{Error as MlxError, Result as MlxResult};
use serde::Deserialize;

use crate::config::{GlmOcrConfig, GlmOcrTextConfig, GlmOcrVisionConfig, load_config};
use crate::error::Error;
use crate::mrope::{self, MropePartition};
use crate::position_ids::{build_position_ids_3d, ImageGrid};
use crate::preprocessor::{
    load_preprocessor_config, preprocess_image_bytes, PreprocessorConfig,
};
use crate::text_decoder::{
    Glm4OcrAttention, Glm4OcrDecoderLayer, Glm4OcrForCausalLM, Glm4OcrMlp, Glm4OcrModel,
};
use crate::vision::{
    Downsample, RmsNorm, VisionAttention, VisionBlock, VisionEncoder, VisionMerger, VisionMlp,
    VisionRotary,
};

#[derive(Debug, Clone, Deserialize)]
struct WeightMap {
    #[serde(default)]
    weight_map: HashMap<String, String>,
}

/// Resolved special-token ids the multimodal path needs.
#[derive(Debug, Clone, Copy)]
pub struct SpecialTokens {
    pub image_start_id: i32,
    pub image_end_id: i32,
    pub image_token_id: i32,
}

/// Fully-loaded GLM-OCR model.
pub struct GlmOcrModel {
    pub config: GlmOcrConfig,
    pub preprocessor: PreprocessorConfig,
    pub tokenizer: tokenizers::Tokenizer,
    pub special_tokens: SpecialTokens,
    pub vision: VisionEncoder,
    pub llm: Glm4OcrForCausalLM,
}

impl GlmOcrModel {
    /// One KVCache slot per text-decoder layer.
    pub fn new_cache(&self) -> Vec<KVCache> {
        (0..self.config.text_config.num_hidden_layers)
            .map(|_| KVCache::default())
            .collect()
    }

    /// Image bytes → (soft_tokens, image_grid_thw). Same shape contract
    /// as paddleocr-vl-mlx so the API wire-up can mirror it directly.
    pub fn encode_image_bytes(&mut self, bytes: &[u8]) -> Result<(Array, ImageGrid), Error> {
        let prep = preprocess_image_bytes(bytes, &self.preprocessor)?;
        let feats = self.vision.forward(&prep.patches, prep.image_grid_thw)?;
        Ok((feats, prep.image_grid_thw))
    }

    /// Build the 3-channel MROPE position-id tensor for a prompt's
    /// `input_ids` + `image_grids` list. Returns shape `(3, 1, T_seq)`.
    pub fn build_position_ids(
        &self,
        input_ids: &[i32],
        image_grids: &[ImageGrid],
    ) -> Result<Array, Error> {
        let merge = self.config.vision_config.spatial_merge_size;
        let temporal = self.config.vision_config.temporal_patch_size;
        let flat = build_position_ids_3d(
            input_ids,
            self.special_tokens.image_token_id,
            image_grids,
            merge,
            temporal,
        )
        .map_err(map_core_err)?;
        flat.reshape(&[3, 1, input_ids.len() as i32])
            .map_err(|e| Error::Mlx(e))
    }
}

fn map_core_err(e: MlxError) -> Error {
    Error::MlxCore(e)
}

/// Top-level loader.
pub fn load_from_path(model_dir: impl AsRef<Path>) -> Result<GlmOcrModel, Error> {
    let dir = model_dir.as_ref();
    let config = load_config(dir)?;
    if config.model_type != "glm_ocr" {
        return Err(Error::Config(format!(
            "expected model_type=glm_ocr, got {}",
            config.model_type
        )));
    }
    let tokenizer = crate::load_tokenizer(dir)?;
    let preprocessor = load_preprocessor_config(dir)?;
    let special_tokens = SpecialTokens {
        image_start_id: config.image_start_token_id.ok_or_else(|| {
            Error::Config("config.image_start_token_id missing".to_string())
        })?,
        image_end_id: config.image_end_token_id.ok_or_else(|| {
            Error::Config("config.image_end_token_id missing".to_string())
        })?,
        image_token_id: config
            .image_token_id
            .ok_or_else(|| Error::Config("config.image_token_id missing".to_string()))?,
    };

    let files = enumerate_weight_files(dir)?;
    let mut weights: HashMap<String, Array> = HashMap::new();
    for f in &files {
        let loaded = Array::load_safetensors(f)
            .map_err(|e| Error::Safetensors(format!("{}: {e}", f.display())))?;
        for (k, v) in loaded {
            weights.insert(k, v);
        }
    }

    let mut used: HashSet<String> = HashSet::new();
    let vision = load_vision(&weights, &config.vision_config, &mut used)?;
    let llm = load_text_decoder(&weights, &config.text_config, &mut used)?;

    let skipped: Vec<&String> = weights.keys().filter(|k| !used.contains(*k)).collect();
    if !skipped.is_empty() {
        let sample: Vec<&str> = skipped.iter().take(5).map(|s| s.as_str()).collect();
        eprintln!(
            "glm-ocr-mlx loader: ignored {} unused weight key(s) (sample: {:?})",
            skipped.len(),
            sample
        );
    }

    Ok(GlmOcrModel {
        config,
        preprocessor,
        tokenizer,
        special_tokens,
        vision,
        llm,
    })
}

fn enumerate_weight_files(dir: &Path) -> Result<Vec<PathBuf>, Error> {
    let index = dir.join("model.safetensors.index.json");
    let single = dir.join("model.safetensors");
    if index.exists() {
        let txt = std::fs::read_to_string(&index)
            .map_err(|e| Error::Io(format!("read {}: {e}", index.display())))?;
        let map: WeightMap = serde_json::from_str(&txt)
            .map_err(|e| Error::Config(format!("decode {}: {e}", index.display())))?;
        let files: HashSet<_> = map.weight_map.values().map(|s| dir.join(s)).collect();
        Ok(files.into_iter().collect())
    } else if single.exists() {
        Ok(vec![single])
    } else {
        Err(Error::Io(format!(
            "{}: no model.safetensors or index.json",
            dir.display()
        )))
    }
}

// ── Component loaders ─────────────────────────────────────────────────────

fn take(
    weights: &HashMap<String, Array>,
    used: &mut HashSet<String>,
    key: &str,
) -> Result<Array, Error> {
    let v = weights
        .get(key)
        .ok_or_else(|| Error::Model(format!("missing weight: {key}")))?;
    used.insert(key.to_string());
    Ok(v.clone().as_dtype(Dtype::Bfloat16).map_err(Error::Mlx)?)
}

fn take_linear(
    weights: &HashMap<String, Array>,
    used: &mut HashSet<String>,
    prefix: &str,
    with_bias: bool,
) -> Result<nn::Linear, Error> {
    let weight = take(weights, used, &format!("{prefix}.weight"))?;
    let bias = if with_bias {
        Some(take(weights, used, &format!("{prefix}.bias"))?)
    } else {
        let bk = format!("{prefix}.bias");
        if weights.contains_key(&bk) {
            Some(take(weights, used, &bk)?)
        } else {
            None
        }
    };
    Ok(nn::Linear {
        weight: Param::new(weight),
        bias: Param::new(bias),
    })
}

fn take_rmsnorm(
    weights: &HashMap<String, Array>,
    used: &mut HashSet<String>,
    key: &str,
    eps: f32,
) -> Result<nn::RmsNorm, Error> {
    Ok(nn::RmsNorm {
        weight: Param::new(take(weights, used, key)?),
        eps,
    })
}

fn take_layernorm(
    weights: &HashMap<String, Array>,
    used: &mut HashSet<String>,
    prefix: &str,
    eps: f32,
) -> Result<nn::LayerNorm, Error> {
    let w = take(weights, used, &format!("{prefix}.weight"))?;
    let b = if weights.contains_key(&format!("{prefix}.bias")) {
        Some(take(weights, used, &format!("{prefix}.bias"))?)
    } else {
        None
    };
    Ok(nn::LayerNorm {
        weight: Param::new(Some(w)),
        bias: Param::new(b),
        eps,
        dimensions: 0, // unused at forward time
    })
}

fn load_vision(
    weights: &HashMap<String, Array>,
    vc: &GlmOcrVisionConfig,
    used: &mut HashSet<String>,
) -> Result<VisionEncoder, Error> {
    // ── Patch embed: Conv3d weight (out, kT, kH, kW, in) → Linear ──
    let patch_w = take(weights, used, "vision_tower.patch_embed.proj.weight")?;
    let pw_shape = patch_w.shape().to_vec();
    if pw_shape.len() != 5 {
        return Err(Error::Vision(format!(
            "patch_embed.proj.weight expected 5-D (out, kT, kH, kW, in), got {:?}",
            pw_shape
        )));
    }
    let out_c = pw_shape[0];
    let in_per_patch: i32 = pw_shape[1..].iter().product();
    let patch_w_flat = patch_w
        .reshape(&[out_c, in_per_patch])
        .map_err(Error::Mlx)?;
    let patch_b = take(weights, used, "vision_tower.patch_embed.proj.bias")?;
    let patch_embed = nn::Linear {
        weight: Param::new(patch_w_flat),
        bias: Param::new(Some(patch_b)),
    };

    // Blocks.
    let head_dim = vc.hidden_size / vc.num_heads;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let mut blocks = Vec::with_capacity(vc.depth as usize);
    for i in 0..vc.depth {
        let lp = format!("vision_tower.blocks.{i}");
        let norm1 = take_rmsnorm(weights, used, &format!("{lp}.norm1.weight"), vc.rms_norm_eps)?;
        let norm2 = take_rmsnorm(weights, used, &format!("{lp}.norm2.weight"), vc.rms_norm_eps)?;
        let qkv = take_linear(weights, used, &format!("{lp}.attn.qkv"), vc.attention_bias)?;
        let q_norm = take_rmsnorm(
            weights,
            used,
            &format!("{lp}.attn.q_norm.weight"),
            vc.rms_norm_eps,
        )?;
        let k_norm = take_rmsnorm(
            weights,
            used,
            &format!("{lp}.attn.k_norm.weight"),
            vc.rms_norm_eps,
        )?;
        let proj = take_linear(weights, used, &format!("{lp}.attn.proj"), true)?;
        let attn = VisionAttention {
            num_heads: vc.num_heads,
            head_dim,
            scale,
            qkv,
            q_norm,
            k_norm,
            proj,
        };
        let mlp = VisionMlp {
            gate_proj: take_linear(weights, used, &format!("{lp}.mlp.gate_proj"), true)?,
            up_proj: take_linear(weights, used, &format!("{lp}.mlp.up_proj"), true)?,
            down_proj: take_linear(weights, used, &format!("{lp}.mlp.down_proj"), true)?,
        };
        blocks.push(VisionBlock { norm1, attn, norm2, mlp });
    }

    let post_layernorm = take_rmsnorm(
        weights,
        used,
        "vision_tower.post_layernorm.weight",
        vc.rms_norm_eps,
    )?;

    // Downsample: Conv2d weight (out, kH, kW, in) → Linear over (m*m*in).
    let dw = take(weights, used, "vision_tower.downsample.weight")?;
    let dws = dw.shape().to_vec();
    if dws.len() != 4 {
        return Err(Error::Vision(format!(
            "downsample.weight expected 4-D (out, kH, kW, in), got {:?}",
            dws
        )));
    }
    let down_out = dws[0];
    let down_in_per_patch: i32 = dws[1..].iter().product();
    let dw_flat = dw.reshape(&[down_out, down_in_per_patch]).map_err(Error::Mlx)?;
    let db = take(weights, used, "vision_tower.downsample.bias")?;
    let downsample = Downsample {
        linear: nn::Linear {
            weight: Param::new(dw_flat),
            bias: Param::new(Some(db)),
        },
        merge_size: vc.spatial_merge_size,
        in_hidden: vc.hidden_size,
        out_hidden: vc.out_hidden_size,
    };

    // Merger.
    let merger = VisionMerger {
        proj: take_linear(weights, used, "vision_tower.merger.proj", false)?,
        post_projection_norm: take_layernorm(
            weights,
            used,
            "vision_tower.merger.post_projection_norm",
            1e-5,
        )?,
        gate_proj: take_linear(weights, used, "vision_tower.merger.gate_proj", false)?,
        up_proj: take_linear(weights, used, "vision_tower.merger.up_proj", false)?,
        down_proj: take_linear(weights, used, "vision_tower.merger.down_proj", false)?,
    };

    let rotary = VisionRotary::new(head_dim, 10_000.0);

    Ok(VisionEncoder {
        config: vc.clone(),
        patch_embed,
        rotary,
        blocks,
        post_layernorm,
        downsample,
        merger,
    })
}

fn load_text_decoder(
    weights: &HashMap<String, Array>,
    tc: &GlmOcrTextConfig,
    used: &mut HashSet<String>,
) -> Result<Glm4OcrForCausalLM, Error> {
    let h = tc.hidden_size;
    let n_h = tc.num_attention_heads;
    let n_kv = tc.num_key_value_heads;
    let d = tc.head_dim;
    let scale = 1.0_f32 / (d as f32).sqrt();

    let embed_w = take(weights, used, "language_model.model.embed_tokens.weight")?;
    let embed_tokens = nn::Embedding {
        weight: Param::new(embed_w),
    };

    let mut layers = Vec::with_capacity(tc.num_hidden_layers as usize);
    for i in 0..tc.num_hidden_layers {
        let lp = format!("language_model.model.layers.{i}");
        let attn = Glm4OcrAttention {
            n_heads: n_h,
            n_kv_heads: n_kv,
            head_dim: d,
            scale,
            q_proj: take_linear(weights, used, &format!("{lp}.self_attn.q_proj"), tc.attention_bias)?,
            k_proj: take_linear(weights, used, &format!("{lp}.self_attn.k_proj"), tc.attention_bias)?,
            v_proj: take_linear(weights, used, &format!("{lp}.self_attn.v_proj"), tc.attention_bias)?,
            o_proj: take_linear(weights, used, &format!("{lp}.self_attn.o_proj"), false)?,
        };
        let mlp = Glm4OcrMlp {
            gate_up_proj: take_linear(weights, used, &format!("{lp}.mlp.gate_up_proj"), false)?,
            down_proj: take_linear(weights, used, &format!("{lp}.mlp.down_proj"), false)?,
        };
        layers.push(Glm4OcrDecoderLayer {
            input_layernorm: take_rmsnorm(
                weights,
                used,
                &format!("{lp}.input_layernorm.weight"),
                tc.rms_norm_eps,
            )?,
            self_attn: attn,
            post_self_attn_layernorm: take_rmsnorm(
                weights,
                used,
                &format!("{lp}.post_self_attn_layernorm.weight"),
                tc.rms_norm_eps,
            )?,
            post_attention_layernorm: take_rmsnorm(
                weights,
                used,
                &format!("{lp}.post_attention_layernorm.weight"),
                tc.rms_norm_eps,
            )?,
            mlp,
            post_mlp_layernorm: take_rmsnorm(
                weights,
                used,
                &format!("{lp}.post_mlp_layernorm.weight"),
                tc.rms_norm_eps,
            )?,
        });
    }

    let norm = take_rmsnorm(
        weights,
        used,
        "language_model.model.norm.weight",
        tc.rms_norm_eps,
    )?;
    let rope_theta = tc.rope_parameters.rope_theta.unwrap_or(10_000.0);
    let mrope_section = tc
        .rope_parameters
        .mrope_section
        .clone()
        .ok_or_else(|| Error::Config("rope_parameters.mrope_section missing".to_string()))?;
    let inv_freq = mrope::inv_freq(d, rope_theta);
    let partition = MropePartition::new(&mrope_section, d).map_err(map_core_err)?;
    let lm_head = take_linear(weights, used, "language_model.lm_head", false)?;

    Ok(Glm4OcrForCausalLM {
        model: Glm4OcrModel {
            embed_tokens,
            layers,
            norm,
            inv_freq,
            partition,
        },
        lm_head,
        config: tc.clone(),
    })
}

// ── Splice helper for prefill ─────────────────────────────────────────────

/// Replace `image_token_id` slots in `text_embeds` with rows from
/// `soft_tokens`. Same contract as paddleocr-vl-mlx's
/// `splice_image_tokens` — see that crate for invariants. The number of
/// `image_token_id` occurrences in `input_ids` must equal `soft_tokens`'
/// row count, and hidden dims must match.
pub fn splice_image_tokens(
    text_embeds: &Array,
    input_ids: &[i32],
    soft_tokens: &Array,
    image_token_id: i32,
) -> Result<Array, Error> {
    let n_img = input_ids.iter().filter(|&&id| id == image_token_id).count();
    let s = soft_tokens.shape();
    if s.len() != 2 {
        return Err(Error::Model(format!(
            "splice_image_tokens: soft_tokens shape {:?} (expected 2-D)",
            s
        )));
    }
    if (s[0] as usize) != n_img {
        return Err(Error::Model(format!(
            "splice_image_tokens: {n_img} image_token_id occurrences vs {} soft tokens",
            s[0]
        )));
    }
    let te_shape = text_embeds.shape();
    if te_shape.len() != 3 || te_shape[0] != 1 || (te_shape[1] as usize) != input_ids.len() {
        return Err(Error::Model(format!(
            "splice_image_tokens: text_embeds shape {:?} vs input_ids len {}",
            te_shape,
            input_ids.len()
        )));
    }
    let hidden = te_shape[2];
    if s[1] != hidden {
        return Err(Error::Model(format!(
            "splice_image_tokens: soft hidden {} vs text hidden {hidden}",
            s[1]
        )));
    }
    // Host-side scatter via f32 round-trip (mirrors paddleocr-vl-mlx).
    let te_f32 = text_embeds.as_dtype(Dtype::Float32).map_err(Error::Mlx)?;
    let so_f32 = soft_tokens.as_dtype(Dtype::Float32).map_err(Error::Mlx)?;
    mlx_rs::transforms::eval([&te_f32, &so_f32])
        .map_err(|e| Error::Mlx(e))?;
    let te_slice = te_f32
        .try_as_slice::<f32>()
        .map_err(|e| Error::Model(format!("text_embeds slice: {e}")))?;
    let so_slice = so_f32
        .try_as_slice::<f32>()
        .map_err(|e| Error::Model(format!("soft_tokens slice: {e}")))?;
    let h = hidden as usize;
    let mut combined = te_slice.to_vec();
    let mut img_row = 0usize;
    for (i, &tok) in input_ids.iter().enumerate() {
        if tok == image_token_id {
            let dst = i * h;
            let src = img_row * h;
            combined[dst..dst + h].copy_from_slice(&so_slice[src..src + h]);
            img_row += 1;
        }
    }
    let arr = Array::from_slice(&combined, &[1, input_ids.len() as i32, hidden]);
    arr.as_dtype(text_embeds.dtype()).map_err(Error::Mlx)
}
