//! Safetensors loader for PaddleOCR-VL-1.5.
//!
//! Reads `model.safetensors` (or the sharded `model.safetensors.index.json`
//! manifest) from a checkpoint directory and constructs a fully-loaded
//! [`PaddleOcrVlModel`] — vision tower + Projector + Ernie 4.5 text
//! decoder + tokenizer + special-tokens, all weights wired to the modules
//! built in phases 1–4.
//!
//! Weight names mirror the HF layout that ships in
//! `PaddlePaddle/PaddleOCR-VL-1.5`:
//!
//! ```text
//!   model.embed_tokens.weight
//!   model.layers.{i}.input_layernorm.weight
//!   model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
//!   model.layers.{i}.post_attention_layernorm.weight
//!   model.layers.{i}.mlp.{gate,up,down}_proj.weight
//!   model.norm.weight
//!   lm_head.weight
//!
//!   visual.vision_model.embeddings.patch_embedding.{weight,bias}
//!   visual.vision_model.embeddings.position_embedding.weight
//!   visual.vision_model.encoder.layers.{i}.layer_norm{1,2}.{weight,bias}
//!   visual.vision_model.encoder.layers.{i}.self_attn.{q,k,v}_proj.{weight,bias}
//!   visual.vision_model.encoder.layers.{i}.self_attn.out_proj.{weight,bias}
//!   visual.vision_model.encoder.layers.{i}.mlp.fc{1,2}.{weight,bias}
//!   visual.vision_model.post_layernorm.{weight,bias}
//!
//!   mlp_AR.pre_norm.{weight,bias}
//!   mlp_AR.linear_1.{weight,bias}
//!   mlp_AR.linear_2.{weight,bias}
//! ```
//!
//! Unused weights from the checkpoint (the SigLIP pooler `head`, the
//! `packing_position_embedding` for the batched-inference path) are
//! tolerated — the loader logs the count of skipped keys instead of
//! erroring so we can still load 1.5 checkpoints whose shapes the
//! simplified inference graph doesn't need.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use mlx_rs::{
    module::Param,
    nn,
    ops::{self, indexing::IndexOp},
    Array, Dtype,
};
use mlx_rs_core::cache::KVCache;
use mlx_rs_core::error::{Error, Result};
use serde::Deserialize;

use crate::config::{PaddleOcrVlConfig, SpecialTokens};
use crate::mrope::{self, MropePartition};
use crate::position_ids::{build_position_ids_3d, ImageGrid};
use crate::preprocess::{preprocess_image_bytes, PreprocessParams};
use crate::projector::Projector;
use crate::text_model::{
    Ernie45Attention, Ernie45DecoderLayer, Ernie45ForCausalLM, Ernie45Mlp, Ernie45Model,
};
use crate::vision_model::{
    VisionAttention, VisionEmbeddings, VisionEncoderLayer, VisionLayerNorm, VisionMlp,
    VisionTransformer,
};

#[derive(Debug, Clone, Deserialize)]
struct WeightMap {
    #[serde(default)]
    weight_map: HashMap<String, String>,
}

/// Fully-loaded PaddleOCR-VL-1.5 model.
///
/// Combines every phase: image preprocessor, vision tower, Projector, text
/// decoder, tokenizer, and special-token resolution. `from_path` builds
/// one from a HF-format checkpoint directory.
pub struct PaddleOcrVlModel {
    pub config: PaddleOcrVlConfig,
    pub preprocess_params: PreprocessParams,
    pub tokenizer: tokenizers::Tokenizer,
    pub special_tokens: SpecialTokens,
    pub vit: VisionTransformer,
    pub projector: Projector,
    pub llm: Ernie45ForCausalLM,
    /// `SigLIPRotaryEmbedding` inv_freq table for vision RoPE — cached at
    /// load so we don't recompute the tiny `(head_dim/4,)` vector every
    /// image. Drives [`crate::vision_model::vision_rope_cos_sin`].
    pub vision_rope_inv_freq: Array,
    /// Vision head_dim = `vision_hidden / num_vision_heads`. Cached so the
    /// `encode_image_bytes` call site doesn't have to walk the config.
    pub vision_head_dim: i32,
}

impl PaddleOcrVlModel {
    /// Build a fresh KV cache sized for `self.llm` (one slot per text
    /// decoder layer).
    pub fn new_cache(&self) -> Vec<KVCache> {
        (0..self.config.num_hidden_layers).map(|_| KVCache::default()).collect()
    }

    /// One-shot image bytes → soft tokens. Mirrors the `encode_image_bytes`
    /// contract on `Gemma4UnifiedVlModel` / `Gemma4VlModel` so the API
    /// wire-up can hand off identically.
    pub fn encode_image_bytes(&mut self, bytes: &[u8]) -> Result<(Array, ImageGrid)> {
        let prep = preprocess_image_bytes(bytes, &self.preprocess_params)?;
        let (_t, gh, gw) = prep.image_grid_thw;
        // Production vision path: feed the patch-grid `(grid_h, grid_w)`
        // into the encoder so each layer applies 2-D vision RoPE on Q/K.
        // The HF reference *always* calls `self.visual(..., use_rope=True)`
        // for inference; skipping the rotary collapses the encoder to OOD.
        let feats = self.vit.forward_with_grid(
            &prep.pixel_values,
            gh,
            gw,
            &self.vision_rope_inv_freq,
            self.vision_head_dim,
        )?;
        let soft = self.projector.forward_batched(&feats, prep.image_grid_thw)?;
        Ok((soft, prep.image_grid_thw))
    }

    /// Build the 3-D MROPE position-id tensor for a prompt's `input_ids` +
    /// `image_grids` list. Pure host-side; cheap to recompute per request.
    /// Output shape: `(3, 1, T_seq)` — the batch dim the text decoder's
    /// `forward_*_from_embeds` expects, with B=1.
    pub fn build_position_ids(
        &self,
        input_ids: &[i32],
        image_grids: &[ImageGrid],
    ) -> Result<Array> {
        let flat = build_position_ids_3d(
            input_ids,
            self.config.image_token_id,
            image_grids,
            self.config.vision_config.spatial_merge_size,
        )?;
        // (3, T) → (3, 1, T) for the (3, B, T) shape that MROPE expects.
        flat.reshape(&[3, 1, input_ids.len() as i32])
            .map_err(Error::from)
    }
}

/// Top-level loader.
pub fn load_from_path(model_dir: impl AsRef<Path>) -> Result<PaddleOcrVlModel> {
    let dir = model_dir.as_ref();
    let config = PaddleOcrVlConfig::from_path(dir)?;
    let tokenizer = crate::config::load_tokenizer(dir)?;
    let special_tokens = SpecialTokens::resolve(&tokenizer, config.image_token_id)?;
    let preprocess_params = PreprocessParams::from_vision_config(&config.vision_config);

    // Load every safetensors shard into one big name→Array map.
    let weight_files = enumerate_weight_files(dir)?;
    let mut weights: HashMap<String, Array> = HashMap::new();
    for f in &weight_files {
        let loaded = Array::load_safetensors(f).map_err(|e| {
            Error::Model(format!("load_safetensors({}): {e}", f.display()))
        })?;
        for (k, v) in loaded {
            weights.insert(k, v);
        }
    }

    let mut used: HashSet<String> = HashSet::with_capacity(weights.len());
    let vit = load_vision(&weights, &config, &mut used)?;
    let projector = load_projector(&weights, &config, &mut used)?;
    let llm = load_text_decoder(&weights, &config, &mut used)?;

    // Surface keys we ignored so the user can see e.g. the pooler /
    // packing-position-embedding weights aren't being used by the
    // simplified inference graph. Not an error.
    let skipped: Vec<&String> = weights.keys().filter(|k| !used.contains(*k)).collect();
    if !skipped.is_empty() {
        let sample: Vec<&str> = skipped.iter().take(5).map(|s| s.as_str()).collect();
        eprintln!(
            "paddleocr-vl-mlx loader: ignored {} unused weight key(s) (sample: {:?})",
            skipped.len(),
            sample
        );
    }

    // Vision RoPE inv_freq: cached once at load. PaddleOCR-VL's
    // SigLIPRotaryEmbedding uses `theta = 10000.0` and `dim = head_dim/2`.
    let vision_head_dim = {
        let vh = config.vision_config.hidden_size;
        let nh = config.vision_config.num_attention_heads;
        if nh == 0 || vh % nh != 0 {
            return Err(Error::InvalidConfig(format!(
                "vision: hidden_size {vh} not divisible by num_attention_heads {nh}"
            )));
        }
        vh / nh
    };
    let vision_rope_inv_freq =
        crate::vision_model::vision_inv_freq(vision_head_dim, 10_000.0);

    Ok(PaddleOcrVlModel {
        config,
        preprocess_params,
        tokenizer,
        special_tokens,
        vit,
        projector,
        llm,
        vision_rope_inv_freq,
        vision_head_dim,
    })
}

fn enumerate_weight_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let index_path = dir.join("model.safetensors.index.json");
    let single_path = dir.join("model.safetensors");
    if index_path.exists() {
        let txt = std::fs::read_to_string(&index_path)?;
        let map: WeightMap = serde_json::from_str(&txt)
            .map_err(|e| Error::Model(format!("decode {}: {e}", index_path.display())))?;
        let files: HashSet<_> = map.weight_map.values().map(|s| dir.join(s)).collect();
        Ok(files.into_iter().collect())
    } else if single_path.exists() {
        Ok(vec![single_path])
    } else {
        Err(Error::Model(format!(
            "load_from_path({}): no model.safetensors or model.safetensors.index.json",
            dir.display()
        )))
    }
}

// ── Per-component loaders ─────────────────────────────────────────────────

fn take(
    weights: &HashMap<String, Array>,
    used: &mut HashSet<String>,
    key: &str,
) -> Result<Array> {
    let v = weights
        .get(key)
        .ok_or_else(|| Error::Model(format!("missing weight: {key}")))?;
    used.insert(key.to_string());
    let arr = v.clone();
    // Most weights ship in BF16. Cast to BF16 explicitly so downstream
    // ops don't have to coerce; this is a no-op when the tensor already
    // is BF16 and a quick cast otherwise.
    arr.as_dtype(Dtype::Bfloat16).map_err(Error::from)
}

fn take_linear(
    weights: &HashMap<String, Array>,
    used: &mut HashSet<String>,
    prefix: &str,
    with_bias: bool,
) -> Result<nn::Linear> {
    let weight = take(weights, used, &format!("{prefix}.weight"))?;
    let bias = if with_bias {
        Some(take(weights, used, &format!("{prefix}.bias"))?)
    } else {
        // Linear with `bias=false` may still have a bias tensor in some
        // checkpoints; tolerate by checking + consuming.
        let key = format!("{prefix}.bias");
        if weights.contains_key(&key) {
            Some(take(weights, used, &key)?)
        } else {
            None
        }
    };
    Ok(nn::Linear {
        weight: Param::new(weight),
        bias: Param::new(bias),
    })
}

fn take_layernorm(
    weights: &HashMap<String, Array>,
    used: &mut HashSet<String>,
    prefix: &str,
    eps: f32,
) -> Result<VisionLayerNorm> {
    Ok(VisionLayerNorm {
        weight: take(weights, used, &format!("{prefix}.weight"))?,
        bias: take(weights, used, &format!("{prefix}.bias"))?,
        eps,
    })
}

fn take_rmsnorm(
    weights: &HashMap<String, Array>,
    used: &mut HashSet<String>,
    key: &str,
    eps: f32,
) -> Result<nn::RmsNorm> {
    Ok(nn::RmsNorm {
        weight: Param::new(take(weights, used, key)?),
        eps,
    })
}

fn load_vision(
    weights: &HashMap<String, Array>,
    config: &PaddleOcrVlConfig,
    used: &mut HashSet<String>,
) -> Result<VisionTransformer> {
    let vc = &config.vision_config;
    let prefix = "visual.vision_model";

    // Conv2d patch_embedding: HF stores weight as
    //   (out_channels=1152, in_channels=3, kH=14, kW=14)
    // MLX nn::Conv2d expects channels-last:
    //   (out_channels, kH, kW, in_channels)
    // Transpose (0, 2, 3, 1).
    let raw_conv = take(weights, used, &format!("{prefix}.embeddings.patch_embedding.weight"))?;
    let conv_w = raw_conv
        .transpose_axes(&[0, 2, 3, 1])
        .map_err(Error::from)?;
    let conv_b = take(weights, used, &format!("{prefix}.embeddings.patch_embedding.bias"))?;
    let patch_embedding = nn::Conv2d {
        weight: Param::new(conv_w),
        bias: Param::new(Some(conv_b)),
        stride: (vc.patch_size, vc.patch_size),
        padding: (0, 0),
        dilation: (1, 1),
        groups: 1,
    };

    let pos_w = take(weights, used, &format!("{prefix}.embeddings.position_embedding.weight"))?;
    let pos_shape = pos_w.shape();
    if pos_shape.len() != 2 || pos_shape[1] != vc.hidden_size {
        return Err(Error::Model(format!(
            "vision.position_embedding.weight shape {:?} doesn't match vision_hidden={}",
            pos_shape, vc.hidden_size
        )));
    }
    let num_positions = pos_shape[0];
    let position_embedding = nn::Embedding {
        weight: Param::new(pos_w),
    };
    let position_ids: Vec<i32> = (0..num_positions).collect();
    let position_ids = Array::from_slice(&position_ids, &[num_positions]);

    // Tolerate (but ignore) the packing position embedding — used by the
    // HF batched-inference path we don't ship.
    let packing_key = format!("{prefix}.embeddings.packing_position_embedding.weight");
    if weights.contains_key(&packing_key) {
        used.insert(packing_key);
    }

    let embeddings = VisionEmbeddings {
        patch_embedding,
        position_embedding,
        position_ids,
        num_positions,
        embed_dim: vc.hidden_size,
    };

    let mut layers = Vec::with_capacity(vc.num_hidden_layers as usize);
    let scale = 1.0_f32 / ((vc.hidden_size / vc.num_attention_heads) as f32).sqrt();
    for i in 0..vc.num_hidden_layers {
        let lp = format!("{prefix}.encoder.layers.{i}");
        let attn = VisionAttention {
            num_heads: vc.num_attention_heads,
            head_dim: vc.hidden_size / vc.num_attention_heads,
            scale,
            q_proj: take_linear(weights, used, &format!("{lp}.self_attn.q_proj"), true)?,
            k_proj: take_linear(weights, used, &format!("{lp}.self_attn.k_proj"), true)?,
            v_proj: take_linear(weights, used, &format!("{lp}.self_attn.v_proj"), true)?,
            out_proj: take_linear(weights, used, &format!("{lp}.self_attn.out_proj"), true)?,
        };
        let mlp = VisionMlp {
            fc1: take_linear(weights, used, &format!("{lp}.mlp.fc1"), true)?,
            fc2: take_linear(weights, used, &format!("{lp}.mlp.fc2"), true)?,
        };
        layers.push(VisionEncoderLayer {
            layer_norm1: take_layernorm(weights, used, &format!("{lp}.layer_norm1"), vc.layer_norm_eps)?,
            self_attn: attn,
            layer_norm2: take_layernorm(weights, used, &format!("{lp}.layer_norm2"), vc.layer_norm_eps)?,
            mlp,
        });
    }

    let post_layernorm =
        take_layernorm(weights, used, &format!("{prefix}.post_layernorm"), vc.layer_norm_eps)?;

    // The SigLIP pooler `head` is in the checkpoint but our simplified
    // forward doesn't use it. Consume its keys so they don't show up in
    // the "ignored" list.
    let head_prefix = format!("{prefix}.head.");
    let head_keys: Vec<String> = weights
        .keys()
        .filter(|k| k.starts_with(&head_prefix))
        .cloned()
        .collect();
    for k in head_keys {
        used.insert(k);
    }

    Ok(VisionTransformer {
        embeddings,
        layers,
        post_layernorm,
    })
}

fn load_projector(
    weights: &HashMap<String, Array>,
    config: &PaddleOcrVlConfig,
    used: &mut HashSet<String>,
) -> Result<Projector> {
    let vc = &config.vision_config;
    let m1 = vc.spatial_merge_size;
    let m2 = vc.spatial_merge_size;
    let vision_hidden = vc.hidden_size;
    let merged_hidden = vision_hidden * m1 * m2;
    let text_hidden = config.hidden_size;

    // Projector pre_norm uses eps=1e-5 in the reference (distinct from the
    // vision encoder's 1e-6).
    let pre_norm = take_layernorm(weights, used, "mlp_AR.pre_norm", 1e-5)?;
    let linear_1 = take_linear(weights, used, "mlp_AR.linear_1", true)?;
    let linear_2 = take_linear(weights, used, "mlp_AR.linear_2", true)?;

    Ok(Projector {
        merge_kernel: (m1, m2),
        vision_hidden,
        merged_hidden,
        text_hidden,
        pre_norm,
        linear_1,
        linear_2,
    })
}

fn load_text_decoder(
    weights: &HashMap<String, Array>,
    config: &PaddleOcrVlConfig,
    used: &mut HashSet<String>,
) -> Result<Ernie45ForCausalLM> {
    let h = config.hidden_size;
    let n_h = config.num_attention_heads;
    let n_kv = config.num_key_value_heads;
    let d = config.head_dim;
    let scale = 1.0_f32 / (d as f32).sqrt();

    let embed_w = take(weights, used, "model.embed_tokens.weight")?;
    let embed_tokens = nn::Embedding {
        weight: Param::new(embed_w),
    };

    let mut layers = Vec::with_capacity(config.num_hidden_layers as usize);
    for i in 0..config.num_hidden_layers {
        let lp = format!("model.layers.{i}");
        let attn = Ernie45Attention {
            n_heads: n_h,
            n_kv_heads: n_kv,
            head_dim: d,
            scale,
            q_proj: take_linear(weights, used, &format!("{lp}.self_attn.q_proj"), false)?,
            k_proj: take_linear(weights, used, &format!("{lp}.self_attn.k_proj"), false)?,
            v_proj: take_linear(weights, used, &format!("{lp}.self_attn.v_proj"), false)?,
            o_proj: take_linear(weights, used, &format!("{lp}.self_attn.o_proj"), false)?,
        };
        let mlp = Ernie45Mlp {
            gate_proj: take_linear(weights, used, &format!("{lp}.mlp.gate_proj"), false)?,
            up_proj: take_linear(weights, used, &format!("{lp}.mlp.up_proj"), false)?,
            down_proj: take_linear(weights, used, &format!("{lp}.mlp.down_proj"), false)?,
        };
        layers.push(Ernie45DecoderLayer {
            input_layernorm: take_rmsnorm(
                weights,
                used,
                &format!("{lp}.input_layernorm.weight"),
                config.rms_norm_eps,
            )?,
            self_attn: attn,
            post_attention_layernorm: take_rmsnorm(
                weights,
                used,
                &format!("{lp}.post_attention_layernorm.weight"),
                config.rms_norm_eps,
            )?,
            mlp,
        });
    }

    let norm = take_rmsnorm(weights, used, "model.norm.weight", config.rms_norm_eps)?;
    let inv_freq = mrope::inv_freq(d, config.rope_theta);
    let partition = MropePartition::new(&config.rope_scaling.mrope_section, d)?;

    let model = Ernie45Model {
        embed_tokens,
        layers,
        norm,
        inv_freq,
        partition,
    };

    // lm_head is its own linear since tie_word_embeddings = false on the
    // canonical 1.5 checkpoint. Verify the shape matches vocab × hidden.
    let lm_head = take_linear(weights, used, "lm_head", false)?;
    let lm_shape = lm_head.weight.shape();
    let _ = h;
    if lm_shape.len() != 2 {
        return Err(Error::Model(format!(
            "lm_head.weight shape {:?} (expected 2-D)",
            lm_shape
        )));
    }
    Ok(Ernie45ForCausalLM {
        model,
        lm_head,
        config: config.clone(),
    })
}

// ── Helper for image-token-splice prefill ─────────────────────────────────

/// Splice `soft_tokens` into `text_embeds` at every `image_token_id`
/// position in `input_ids`. Returns a `(1, T_seq, hidden)` array ready to
/// pass to `Ernie45ForCausalLM::forward_*_from_embeds`.
///
/// Validates that the number of `image_token_id` placeholders equals the
/// number of `soft_tokens` rows so loader / preprocessor / tokenizer
/// drift fails loudly.
pub fn splice_image_tokens(
    text_embeds: &Array,
    input_ids: &[i32],
    soft_tokens: &Array,
    image_token_id: i32,
) -> Result<Array> {
    let n_img: usize = input_ids.iter().filter(|&&id| id == image_token_id).count();
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
            "splice_image_tokens: soft_tokens hidden {} vs text_embeds hidden {hidden}",
            s[1]
        )));
    }

    // Host-side scatter. The text_embeds + soft_tokens tensors are both
    // BF16; round-trip via f32 to avoid lossy reductions.
    let te_f32 = text_embeds
        .as_dtype(Dtype::Float32)
        .map_err(Error::from)?;
    let so_f32 = soft_tokens
        .as_dtype(Dtype::Float32)
        .map_err(Error::from)?;
    mlx_rs::transforms::eval([&te_f32, &so_f32])?;
    let te_slice = te_f32
        .try_as_slice::<f32>()
        .map_err(|e| Error::Model(format!("text_embeds → slice: {e}")))?;
    let so_slice = so_f32
        .try_as_slice::<f32>()
        .map_err(|e| Error::Model(format!("soft_tokens → slice: {e}")))?;
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
    let arr = Array::from_slice(
        &combined,
        &[1, input_ids.len() as i32, hidden],
    );
    arr.as_dtype(text_embeds.dtype()).map_err(Error::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::random::uniform;

    #[test]
    fn splice_image_tokens_replaces_placeholders() {
        let hidden = 4;
        let img_tok = 99_i32;
        let input_ids = vec![1, 2, img_tok, img_tok, 3];
        // text_embeds: (1, 5, 4), filled with marker values per position.
        let te: Vec<f32> = (0..5 * 4).map(|i| (i as f32) * 0.1).collect();
        let text_embeds = Array::from_slice(&te, &[1, 5, hidden]);
        // soft_tokens: (2, 4), distinct marker values.
        let so: Vec<f32> = (0..2 * 4).map(|i| 100.0 + i as f32).collect();
        let soft = Array::from_slice(&so, &[2, hidden]);

        let out = splice_image_tokens(&text_embeds, &input_ids, &soft, img_tok).unwrap();
        assert_eq!(out.shape(), &[1, 5, hidden]);
        let f32_out = out.as_dtype(Dtype::Float32).unwrap();
        let bytes = f32_out.try_as_slice::<f32>().unwrap();
        // Positions 0, 1, 4 must equal the original text_embeds rows;
        // positions 2, 3 must equal soft_tokens rows 0, 1.
        for h in 0..4 {
            assert!((bytes[0 * 4 + h] - te[0 * 4 + h]).abs() < 1e-5);
            assert!((bytes[1 * 4 + h] - te[1 * 4 + h]).abs() < 1e-5);
            assert!((bytes[2 * 4 + h] - so[0 * 4 + h]).abs() < 1e-5);
            assert!((bytes[3 * 4 + h] - so[1 * 4 + h]).abs() < 1e-5);
            assert!((bytes[4 * 4 + h] - te[4 * 4 + h]).abs() < 1e-5);
        }
    }

    #[test]
    fn splice_image_tokens_rejects_count_mismatch() {
        let hidden = 4;
        let img_tok = 99_i32;
        let input_ids = vec![1, img_tok, 2]; // 1 placeholder
        let text_embeds = uniform::<_, f32>(-1.0, 1.0, &[1, 3, hidden], None).unwrap();
        let soft = uniform::<_, f32>(-1.0, 1.0, &[2, hidden], None).unwrap(); // 2 rows
        let err = splice_image_tokens(&text_embeds, &input_ids, &soft, img_tok);
        assert!(err.is_err());
    }

    #[test]
    fn splice_image_tokens_rejects_hidden_mismatch() {
        let img_tok = 99_i32;
        let input_ids = vec![img_tok];
        let text_embeds = uniform::<_, f32>(-1.0, 1.0, &[1, 1, 4], None).unwrap();
        let soft = uniform::<_, f32>(-1.0, 1.0, &[1, 8], None).unwrap();
        let err = splice_image_tokens(&text_embeds, &input_ids, &soft, img_tok);
        assert!(err.is_err());
    }
}
