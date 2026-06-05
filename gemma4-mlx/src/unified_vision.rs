//! Encoder-free vision embedder for Gemma 4 Unified (12B+).
//!
//! Ports the merged sgl-project/sglang#27167 reference. The "unified" Gemma 4
//! variant ships no SigLIP-style ViT encoder; each raw
//! `model_patch_size² × 3` pixel patch (48×48×3 = 6912 for the 12B) is run
//! through
//!
//!     patch_ln1 → patch_dense → patch_ln2 → + factorized 2D pos embed → pos_norm
//!
//! producing a `(P, mm_embed_dim)` tensor. The shared
//! [`crate::vision::EmbedVision`] (RMSNorm-no-scale + Linear) then projects to
//! the LM hidden space.
//!
//! Phase 1: load + forward only. Image preprocessing (HF
//! `Gemma4UnifiedImageProcessor` equivalent) and API splicing of vision soft
//! tokens into the LM input embedding are deferred to Phase 2.

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::{
    array,
    module::{Module, Param},
    nn,
    ops::{self, indexing::IndexOp},
    transforms::eval,
    Array,
};
use mlx_rs_core::error::{Error, Result};
use serde::Deserialize;

use crate::model::{get_model_args, load_all_weights_unfiltered, Model};
use crate::ud_loader::load_ud_mlx_4bit;
use crate::vision::{EmbedVision, VisionRmsNormNoScale};

/// Vision config for the `gemma4_unified_vision` schema (12B+).
#[derive(Debug, Clone, Deserialize)]
pub struct Gemma4UnifiedVisionConfig {
    pub model_type: String,
    pub mm_embed_dim: i32,
    pub mm_posemb_size: i32,
    pub model_patch_size: i32,
    pub patch_size: i32,
    #[serde(default)]
    pub pooling_kernel_size: i32,
    #[serde(default, alias = "max_soft_tokens")]
    pub num_soft_tokens: i32,
    pub output_proj_dims: i32,
    #[serde(default = "default_eps")]
    pub rms_norm_eps: f32,
}

fn default_eps() -> f32 {
    1e-6
}

/// `nn::LayerNorm` uses 1e-5 by default; the unified checkpoint's LN epsilons
/// are not in the config (HF's `LayerNorm` constructor default is 1e-5 too).
const LAYER_NORM_EPS: f32 = 1e-5;

fn get_weight(weights: &HashMap<String, Array>, key: &str) -> Result<Array> {
    weights
        .get(key)
        .cloned()
        .ok_or_else(|| Error::WeightNotFound(key.to_string()))
}

fn build_layer_norm(weight: Array, bias: Option<Array>, dim: i32) -> nn::LayerNorm {
    nn::LayerNorm {
        dimensions: dim,
        eps: LAYER_NORM_EPS,
        weight: Param::new(Some(weight)),
        bias: Param::new(bias),
    }
}

/// Dequantize an MLX-packed `(weight U32, scales BF16, biases BF16)` triplet to
/// a plain BF16 tensor — keeps the forward `nn::Linear`-shaped and matches the
/// canonical [`EmbedVision`] which expects a non-quantized projection.
fn dequant_to_linear(
    weights: &HashMap<String, Array>,
    prefix: &str,
    bits: i32,
    group_size: i32,
    bias_present: bool,
) -> Result<nn::Linear> {
    let w = get_weight(weights, &format!("{prefix}.weight"))?;
    let s = get_weight(weights, &format!("{prefix}.scales"))?;
    let b = get_weight(weights, &format!("{prefix}.biases"))?;
    let dequantized =
        ops::dequantize(&w, &s, &b, group_size, bits, None::<&str>).map_err(Error::from)?;
    let bias = if bias_present {
        Some(get_weight(weights, &format!("{prefix}.bias"))?)
    } else {
        None
    };
    Ok(nn::Linear {
        weight: Param::new(dequantized),
        bias: Param::new(bias),
    })
}

/// Encoder-free vision embedder. See module docs.
pub struct UnifiedVisionEmbedder {
    pub patch_ln1: nn::LayerNorm,
    pub patch_dense: nn::Linear,
    pub patch_ln2: nn::LayerNorm,
    /// `(mm_posemb_size, 2, mm_embed_dim)` factorized 2D positional table.
    /// Sliced into X (`[:, 0, :]`) and Y (`[:, 1, :]`) lookups during forward.
    pub pos_embedding: Array,
    pub pos_norm: nn::LayerNorm,
}

impl UnifiedVisionEmbedder {
    /// Embed a batch of raw pixel patches.
    ///
    /// * `pixel_values` — `(B, P, model_patch_size² × 3)` raw patches, scaled
    ///   to `[0, 1]` (HF processor does the `1/255` rescale upstream).
    /// * `position_ids` — `(B, P, 2)` int32 (X, Y) grid coords; `-1` marks
    ///   padding patches, which are zeroed out before the pos-norm.
    ///
    /// Output: `(B, P, mm_embed_dim)`.
    pub fn forward(&mut self, pixel_values: &Array, position_ids: &Array) -> Result<Array> {
        let dtype = self.patch_dense.weight.dtype();
        let x = pixel_values.as_dtype(dtype)?;
        let x = self.patch_ln1.forward(&x)?;
        let x = self.patch_dense.forward(&x)?;
        let x = self.patch_ln2.forward(&x)?;
        let pos = self.factorized_pos_emb(position_ids)?;
        let x = x.add(&pos.as_dtype(dtype)?)?;
        self.pos_norm.forward(&x).map_err(Into::into)
    }

    /// Factorized 2D positional embedding lookup.
    ///
    /// Mirrors the merged sglang reference:
    ///   `clamped = max(position_ids, 0)`
    ///   `valid = (position_ids != -1).unsqueeze(-1)`
    ///   `(pos_embedding[clamped, axes] * valid).sum(-2)`
    /// implemented as two independent `take_axis` gathers (X and Y) summed
    /// after masking — padding rows have both axes set to -1 so a single
    /// validity flag (from X) suffices.
    fn factorized_pos_emb(&self, position_ids: &Array) -> Result<Array> {
        let x_table = self.pos_embedding.index((.., 0, ..));
        let y_table = self.pos_embedding.index((.., 1, ..));
        let raw_x = position_ids.index((.., .., 0));
        let raw_y = position_ids.index((.., .., 1));
        // Lower-bound at 0 via `maximum`: `ops::clip` doesn't support a
        // one-sided bound shape directly, and we don't need an upper clamp
        // (the valid mask zeros padding contributions regardless of which
        // in-range row gets gathered).
        let zero = array!(0_i32);
        let x_idx = ops::maximum(&raw_x, &zero)?;
        let y_idx = ops::maximum(&raw_y, &zero)?;
        eval([&x_idx, &y_idx]).map_err(|e| Error::Model(format!("eval pos idx: {e}")))?;
        let x_emb = ops::indexing::take_axis(&x_table, &x_idx, 0)?;
        let y_emb = ops::indexing::take_axis(&y_table, &y_idx, 0)?;
        let minus_one = array!(-1_i32);
        let valid = raw_x.ne(&minus_one)?;
        let valid_f = valid.as_dtype(x_emb.dtype())?.expand_dims(-1)?;
        x_emb.add(&y_emb)?.multiply(&valid_f).map_err(Into::into)
    }
}

/// Build the unified-vision embedder from a flattened weight map.
///
/// Expects the checkpoint to ship the patch_dense + embedding_projection
/// linears as packed MLX-4bit triplets (`U32 weight + BF16 scales + BF16
/// biases`, `group_size = 64`). Both are dequantized to BF16 here so the
/// runtime forward stays on plain `nn::Linear`.
pub fn load_unified_embedder(
    weights: &HashMap<String, Array>,
    config: &Gemma4UnifiedVisionConfig,
) -> Result<UnifiedVisionEmbedder> {
    let patch_dim = config.model_patch_size * config.model_patch_size * 3;
    let mm_embed_dim = config.mm_embed_dim;

    let patch_ln1 = build_layer_norm(
        get_weight(weights, "vision_embedder.patch_ln1.weight")?,
        Some(get_weight(weights, "vision_embedder.patch_ln1.bias")?),
        patch_dim,
    );
    let patch_dense = dequant_to_linear(
        weights,
        "vision_embedder.patch_dense",
        /* bits */ 4,
        /* group_size */ 64,
        /* bias_present */ true,
    )?;
    let patch_ln2 = build_layer_norm(
        get_weight(weights, "vision_embedder.patch_ln2.weight")?,
        Some(get_weight(weights, "vision_embedder.patch_ln2.bias")?),
        mm_embed_dim,
    );
    let pos_embedding = get_weight(weights, "vision_embedder.pos_embedding")?;
    let pos_norm = build_layer_norm(
        get_weight(weights, "vision_embedder.pos_norm.weight")?,
        Some(get_weight(weights, "vision_embedder.pos_norm.bias")?),
        mm_embed_dim,
    );

    Ok(UnifiedVisionEmbedder {
        patch_ln1,
        patch_dense,
        patch_ln2,
        pos_embedding,
        pos_norm,
    })
}

/// Build the [`EmbedVision`] (RMSNorm-no-scale + Linear projection) used by
/// the unified path. The projection weight is also packed 4-bit; dequantize.
pub fn load_unified_embed_vision(
    weights: &HashMap<String, Array>,
    config: &Gemma4UnifiedVisionConfig,
) -> Result<EmbedVision> {
    let projection = dequant_to_linear(
        weights,
        "embed_vision.embedding_projection",
        4,
        64,
        /* bias_present */ false,
    )?;
    Ok(EmbedVision {
        embedding_pre_projection_norm: VisionRmsNormNoScale {
            eps: config.rms_norm_eps,
        },
        embedding_projection: projection,
    })
}

/// Combined text + unified vision model.
///
/// Mirrors the canonical [`crate::Gemma4VlModel`] for the unified schema —
/// text is loaded through the existing UD text-only loader (`load_ud_mlx_4bit`),
/// vision is the encoder-free embedder above.
pub struct Gemma4UnifiedVlModel {
    pub text: Model,
    pub vision: UnifiedVisionEmbedder,
    pub embed_vision: EmbedVision,
    pub image_token_id: u32,
    pub boi_token_id: u32,
    pub eoi_token_id: u32,
    /// Number of vision soft tokens per image; HF surfaces this as
    /// `vision_config.num_soft_tokens` (alias `max_soft_tokens`).
    pub n_vision_tokens: usize,
}

/// Top-level loader for a UD-MLX-4bit unified checkpoint
/// (e.g. `mlx-community/gemma-4-12B-it-4bit`).
///
/// Skips when the checkpoint's `vision_config.model_type` is not
/// `gemma4_unified_vision` so the canonical Gemma4-VL loader can claim the
/// e4b/26B path.
pub fn load_unified_4bit_vl(model_dir: impl AsRef<Path>) -> Result<Gemma4UnifiedVlModel> {
    let model_dir = model_dir.as_ref();
    let cfg_path = model_dir.join("config.json");
    let cfg_json: serde_json::Value = serde_json::from_reader(std::fs::File::open(&cfg_path)?)?;
    let vision_value = cfg_json
        .get("vision_config")
        .cloned()
        .ok_or_else(|| Error::Model("missing vision_config".to_string()))?;
    let vision_config: Gemma4UnifiedVisionConfig = serde_json::from_value(vision_value)
        .map_err(|e| Error::Model(format!("vision_config decode (unified schema): {e}")))?;
    if vision_config.model_type != "gemma4_unified_vision" {
        return Err(Error::Model(format!(
            "load_unified_4bit_vl: vision_config.model_type='{}' (expected 'gemma4_unified_vision')",
            vision_config.model_type
        )));
    }

    // Text uses the same UD layout as gemma-4-26B-A4B-it-UD-MLX-4bit:
    // `language_model.model.*` prefix, heterogeneous quantization. Delegate.
    let text = load_ud_mlx_4bit(model_dir)?;

    // Vision-side weights aren't translated by the text loader; reload the
    // raw weight map for the unified vision modules.
    let raw = load_all_weights_unfiltered(model_dir)?;
    let vision = load_unified_embedder(&raw, &vision_config)?;
    let embed_vision = load_unified_embed_vision(&raw, &vision_config)?;

    let full_cfg = get_model_args(model_dir)?;
    let n_vision_tokens = if vision_config.num_soft_tokens > 0 {
        vision_config.num_soft_tokens as usize
    } else {
        full_cfg.vision_soft_tokens_per_image.max(256)
    };

    Ok(Gemma4UnifiedVlModel {
        text,
        vision,
        embed_vision,
        image_token_id: full_cfg.image_token_id.unwrap_or(258880),
        boi_token_id: full_cfg.boi_token_id.unwrap_or(255999),
        eoi_token_id: full_cfg.eoi_token_id.unwrap_or(258882),
        n_vision_tokens,
    })
}
