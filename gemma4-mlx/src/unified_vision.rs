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

use image::imageops::FilterType;
use mlx_rs::{
    array,
    module::{Module, Param},
    nn,
    ops::{self, indexing::IndexOp},
    quantization::MaybeQuantized,
    transforms::eval,
    Array,
};
use mlx_rs_core::error::{Error, Result};
use serde::Deserialize;

use crate::model::{get_model_args, load_all_weights_unfiltered, Model};
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
        embedding_projection: MaybeQuantized::Original(projection),
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

impl Gemma4UnifiedVlModel {
    /// One-shot image → soft-tokens encode. Mirrors the canonical
    /// `Gemma4VlModel::encode_image_bytes` contract used by the API's
    /// `generate_gemma4_vl_multimodal` so the same splice path works:
    ///
    ///   1. Aspect-preserving resize + patchify to `(1, P, model_patch_size² × 3)`.
    ///   2. `UnifiedVisionEmbedder` forward (LN → Dense → LN → +pos → pos_norm).
    ///   3. `EmbedVision` projection to LM hidden size.
    ///   4. Drop padding rows so the returned tensor has shape
    ///      `(num_real_patches, text_hidden)` — exactly what the LM
    ///      embedding-splice expects at the `image_token_id` placeholder
    ///      positions.
    pub fn encode_image_bytes(&mut self, bytes: &[u8]) -> Result<Array> {
        let (pixel_values, pos_ids, n_real) =
            preprocess_image_unified(bytes, 48, self.n_vision_tokens as i32)?;
        let hidden = self.vision.forward(&pixel_values, &pos_ids)?;
        let projected = self.embed_vision.forward(&hidden)?;
        // (1, max_soft_tokens, hidden) → drop the leading batch + padding tail.
        let valid = projected.index((0, ..n_real, ..));
        Ok(valid)
    }

    /// Multimodal prefill: embed `input_ids`, splice `visual_features` rows
    /// into the embeddings at every `image_token_id` position, run a chunked
    /// PLE-aware prefill, and return the last-position logits.
    ///
    /// Mirrors `Gemma4VlModel::prefill_multimodal` byte-for-byte (same
    /// embedding-scale, host-side scatter via `try_as_slice`, chunk size 32,
    /// PLE precompute) so the API's existing `generate_gemma4_vl_multimodal`
    /// can drive this path identically.
    ///
    /// If no `image_token_id` positions exist in `input_ids` the call
    /// degrades to a plain text prefill.
    pub fn prefill_multimodal<C>(
        &mut self,
        input_ids: &[i32],
        visual_features: &Array,
        cache: &mut Vec<C>,
    ) -> Result<Array>
    where
        C: mlx_rs_core::cache::KeyValueCache + Default,
    {
        if !input_ids.iter().any(|&id| id as u32 == self.image_token_id) {
            // No image tokens — fall back to a plain text prefill via Model.
            let chunk = Array::from_slice(input_ids, &[1, input_ids.len() as i32]);
            let input = crate::ModelInput {
                inputs: &chunk,
                mask: None,
                cache,
            };
            return self.text.forward_last_logits(input).map_err(Error::from);
        }
        crate::Gemma4VlModel::prefill_multimodal_scatter(
            &mut self.text,
            self.image_token_id,
            input_ids,
            visual_features,
            cache,
        )
    }

    /// Decode a single token (post-prefill) — same contract as the canonical
    /// `Gemma4VlModel::decode_token`. The API's decode loop can call this
    /// uniformly across the canonical and unified VL backends.
    pub fn decode_token<C>(&mut self, token_id: u32, cache: &mut Vec<C>) -> Result<Array>
    where
        C: mlx_rs_core::cache::KeyValueCache + Default,
    {
        let chunk = Array::from_slice(&[token_id as i32], &[1, 1]);
        let input = crate::ModelInput {
            inputs: &chunk,
            mask: None,
            cache,
        };
        self.text.forward_last_logits(input).map_err(Error::from)
    }

    /// Run a verify forward for prompt-lookup speculative decoding (PLD).
    ///
    /// Given the trailing committed token `committed` and a `draft` of K
    /// guessed token IDs, runs a sequence forward of length `K + 1` through
    /// `self.text` and returns the lm-head logits at each position
    /// (`shape = [K + 1, vocab]`). The caller compares
    /// `argmax(logits[i])` to `draft[i]` to accept the leading prefix of
    /// the draft, then [`KeyValueCache::trim_kv`]s the cache by
    /// `K - accepted` to discard the rejected tail.
    ///
    /// Returns the per-position logits (shape `(K + 1, vocab)`).
    pub fn verify_draft<C>(
        &mut self,
        committed: i32,
        draft: &[i32],
        cache: &mut Vec<C>,
    ) -> Result<Array>
    where
        C: mlx_rs_core::cache::KeyValueCache + Default,
    {
        let mut seq = Vec::with_capacity(1 + draft.len());
        seq.push(committed);
        seq.extend_from_slice(draft);
        // Compute embeddings first so the second borrow of `self.text` is
        // strictly sequential with the first (Rust's borrow checker rejects
        // back-to-back `&mut` reborrows across argument positions).
        let embeds = self.text.embed_tokens(&seq)?;
        let logits = self
            .text
            .forward_all_logits_from_embeds(&embeds, cache, None)
            .map_err(Error::from)?;
        // Drop the batch dim → (K+1, vocab).
        logits
            .index((0, .., ..))
            .reshape(&[seq.len() as i32, -1])
            .map_err(Error::from)
    }

    /// Build a fresh contiguous KV cache sized for `self.text`.
    pub fn new_cache(&self) -> Vec<crate::KVCache> {
        let num_slots = *self.text.model.kv_cache_map.iter().max().unwrap_or(&0) + 1;
        crate::init_cache::<crate::KVCache>(num_slots)
    }

    /// Paged variant of [`Self::new_cache`] — parallel to
    /// [`crate::Gemma4VlModel::new_cache_paged`]. Full-attention layers draw
    /// from the shared paged pool; sliding-window layers stay contiguous (the
    /// "mixed" of `MixedKvCache`). Used by the API when
    /// `OMINIX_PAGED_ATTENTION=1` is set.
    pub fn new_cache_paged(&self) -> Vec<crate::mixed_cache::MixedKvCache> {
        crate::mixed_cache::init_mixed_paged_cache(&self.text)
    }
}

/// Image preprocessor for the unified vision path.
///
/// Mirrors the area-preserving aspect-ratio resize the canonical
/// [`crate::preprocess_image_gemma4`] does — pick the largest target whose
/// width and height are multiples of `model_patch_size` and whose total
/// patch count is `≤ max_soft_tokens`, then bicubic-resize, `1/255`-rescale,
/// and partition into `model_patch_size × model_patch_size × 3` patches.
///
/// Returns
///   * `pixel_values` — `(1, max_soft_tokens, model_patch_size² × 3)` BF16
///     tensor; rows past `num_real_patches` are zero-padded.
///   * `position_ids` — `(1, max_soft_tokens, 2)` int32 (X, Y) grid coords;
///     rows past `num_real_patches` are `(-1, -1)` per the embedder's
///     padding convention.
///   * `num_real_patches` — count of non-padded patches in the leading
///     prefix of `pixel_values` / `position_ids`.
pub fn preprocess_image_unified(
    bytes: &[u8],
    model_patch_size: i32,
    max_soft_tokens: i32,
) -> Result<(Array, Array, i32)> {
    let image = image::load_from_memory(bytes).map_err(|e| {
        Error::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            e.to_string(),
        ))
    })?;
    let rgb = image.to_rgb8();
    let (width, height) = rgb.dimensions();
    let width = width as i32;
    let height = height as i32;
    if width <= 0 || height <= 0 {
        return Err(Error::InvalidConfig(
            "Attempting to resize to a 0x0 image".to_string(),
        ));
    }

    // Aspect-preserving fit to ≤ max_soft_tokens patches at model_patch_size².
    let target_px = max_soft_tokens * model_patch_size * model_patch_size;
    let factor = ((target_px as f32) / ((height * width) as f32)).sqrt();
    let mps = model_patch_size;
    let mut target_h = ((factor * height as f32) / mps as f32).floor() as i32 * mps;
    let mut target_w = ((factor * width as f32) / mps as f32).floor() as i32 * mps;
    // Bound by the maximum side length the position table can address (one row
    // or one column of patches must still fit into `max_soft_tokens`).
    let max_side = max_soft_tokens * mps;
    if target_h == 0 && target_w == 0 {
        return Err(Error::InvalidConfig(
            "Attempting to resize to a 0x0 image".to_string(),
        ));
    }
    if target_h == 0 {
        target_h = mps;
        target_w = ((width as f32 / height as f32).floor() as i32).max(1) * mps;
        target_w = target_w.min(max_side);
    } else if target_w == 0 {
        target_w = mps;
        target_h = ((height as f32 / width as f32).floor() as i32).max(1) * mps;
        target_h = target_h.min(max_side);
    }

    let resized = image
        .resize_exact(target_w as u32, target_h as u32, FilterType::CatmullRom)
        .to_rgb8();

    // Partition into (P_y × P_x) macro-patches of model_patch_size × model_patch_size × 3
    // pixels each, channel-last flattened into a single 6912-element row.
    let p_h = target_h / mps;
    let p_w = target_w / mps;
    let n_real = (p_h * p_w).min(max_soft_tokens);
    let patch_pixels = (mps * mps * 3) as usize;
    let mut pixel_values = vec![0f32; max_soft_tokens as usize * patch_pixels];
    for py in 0..p_h {
        for px in 0..p_w {
            let patch_idx = (py * p_w + px) as usize;
            if patch_idx >= max_soft_tokens as usize {
                break;
            }
            let dst_offset = patch_idx * patch_pixels;
            for dy in 0..mps {
                for dx in 0..mps {
                    let sx = px * mps + dx;
                    let sy = py * mps + dy;
                    let pix = resized.get_pixel(sx as u32, sy as u32).0;
                    let row_offset = (dy * mps + dx) as usize * 3;
                    pixel_values[dst_offset + row_offset] = pix[0] as f32 / 255.0;
                    pixel_values[dst_offset + row_offset + 1] = pix[1] as f32 / 255.0;
                    pixel_values[dst_offset + row_offset + 2] = pix[2] as f32 / 255.0;
                }
            }
        }
    }
    let pixel_array = Array::from_slice(
        &pixel_values,
        &[1, max_soft_tokens, patch_pixels as i32],
    );

    // Position IDs: (X, Y) for every real patch, then (-1, -1) padding.
    let mut pos_ids = Vec::with_capacity(max_soft_tokens as usize * 2);
    let mut emitted = 0;
    for py in 0..p_h {
        for px in 0..p_w {
            if emitted >= max_soft_tokens {
                break;
            }
            pos_ids.push(px);
            pos_ids.push(py);
            emitted += 1;
        }
    }
    while emitted < max_soft_tokens {
        pos_ids.push(-1);
        pos_ids.push(-1);
        emitted += 1;
    }
    let pos_array = Array::from_slice(&pos_ids, &[1, max_soft_tokens, 2]);

    Ok((pixel_array, pos_array, n_real))
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

    // Load the multi-GB weight map once: the unified vision modules read it
    // under the raw (untranslated) names, then the text side consumes it via
    // the same UD-prefix translation `load_ud_mlx_4bit` uses.
    let raw = load_all_weights_unfiltered(model_dir)?;
    let vision = load_unified_embedder(&raw, &vision_config)?;
    let embed_vision = load_unified_embed_vision(&raw, &vision_config)?;

    let full_cfg = get_model_args(model_dir)?;
    let weights = crate::ud_loader::translate_keys(raw);
    let text = crate::model::build_model_from_weights(
        &full_cfg,
        full_cfg.text_config.clone(),
        &weights,
    )?;
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
