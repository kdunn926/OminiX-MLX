use std::collections::{HashMap, HashSet};
use std::path::Path;

use mlx_rs::{
    error::Exception,
    module::{Module, ModuleParameters, Param},
    nn,
    ops::{concatenate_axis, dequantize, indexing::IndexOp},
    quantization::MaybeQuantized,
    Array, Dtype,
};

use mlx_rs_core::{
    cache::{KVCache, QuantizedKVCache},
    error::Error,
    utils::initialize_rope,
};

use crate::attention::{GatedAttention, GatedAttentionInput};
use crate::cache::{HybridCache, RecurrentState};
use crate::config::{ModelArgs, TextConfig};
use crate::deltanet::GatedDeltaNet;
use crate::moe::{DenseMlp, MoeBlock, QuantizedSwitchLinear, SharedExpert, SwitchGLU};
use crate::vision::VisionTower;

// ============================================================================
// Layer Types
// ============================================================================

pub enum AttentionLayer {
    FullAttention(GatedAttention),
    LinearAttention(GatedDeltaNet),
}

/// FFN for a transformer block — either MoE (35B) or dense MLP (27B).
pub enum FfnBlock {
    Moe(MoeBlock),
    Dense(DenseMlp),
}

pub struct TransformerBlock {
    pub attention: AttentionLayer,
    pub ffn: FfnBlock,
    pub input_layernorm: nn::RmsNorm,
    pub post_attention_layernorm: nn::RmsNorm,
}

impl TransformerBlock {
    #[allow(non_snake_case)]
    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&mlx_rs_core::utils::AttentionMask>,
        cache: &mut HybridCache,
    ) -> Result<Array, Exception> {
        let normed = self.input_layernorm.forward(x)?;

        let attn_out = match (&mut self.attention, cache) {
            (AttentionLayer::FullAttention(attn), HybridCache::KV(kv_cache)) => {
                attn.forward(GatedAttentionInput {
                    x: &normed,
                    mask,
                    cache: Some(kv_cache),
                })?
            }
            (AttentionLayer::FullAttention(attn), HybridCache::QuantizedKV(qkv_cache)) => attn
                .forward(GatedAttentionInput {
                    x: &normed,
                    mask,
                    cache: Some(qkv_cache),
                })?,
            (AttentionLayer::LinearAttention(delta), HybridCache::Recurrent(rec_cache)) => {
                let L = normed.shape()[1];
                if L > 1 {
                    delta.forward_prefill(&normed, rec_cache)?
                } else {
                    delta.forward_step(&normed, rec_cache)?
                }
            }
            _ => return Err(Exception::custom("Cache type mismatch with layer type")),
        };

        let h = x.add(attn_out)?;
        let normed_h = self.post_attention_layernorm.forward(&h)?;
        let mlp_out = match &mut self.ffn {
            FfnBlock::Moe(m) => m.forward(&normed_h)?,
            FfnBlock::Dense(d) => d.forward(&normed_h)?,
        };
        h.add(mlp_out)
    }
}

// ============================================================================
// Full Model
// ============================================================================

/// Selects which KV cache implementation to use for full-attention layers.
#[derive(Debug, Clone, Copy, Default)]
pub enum KVCacheMode {
    /// Standard fp16 KV cache (default).
    #[default]
    Standard,
    /// Mixed-precision cache: K=q8, V=q4.
    Quantized,
}

pub struct Qwen36TextModel {
    pub embed_tokens: MaybeQuantized<nn::Embedding>,
    pub layers: Vec<TransformerBlock>,
    pub norm: nn::RmsNorm,
    pub layer_types: Vec<String>,
}

pub struct Model {
    pub args: ModelArgs,
    pub text_model: Qwen36TextModel,
    pub lm_head: Option<MaybeQuantized<nn::Linear>>,
}

impl Model {
    /// Allocate a fresh cache vector appropriate for this model.
    ///
    /// - `mode = Standard`   → full-attention layers get `KVCache`
    /// - `mode = Quantized`  → full-attention layers get `QuantizedKVCache` (K=q8, V=q4)
    pub fn new_cache(&self, mode: KVCacheMode) -> Vec<HybridCache> {
        self.text_model
            .layer_types
            .iter()
            .map(|lt| {
                if lt == "full_attention" {
                    match mode {
                        KVCacheMode::Standard => HybridCache::KV(KVCache::new()),
                        KVCacheMode::Quantized => {
                            HybridCache::QuantizedKV(QuantizedKVCache::default())
                        }
                    }
                } else {
                    HybridCache::Recurrent(RecurrentState::new())
                }
            })
            .collect()
    }

    /// Run the transformer and project only the last sequence position through
    /// the LM head. Avoids a `[B, T, vocab]` matmul during prefill.
    pub fn forward_last_logits(
        &mut self,
        inputs: &Array,
        cache: &mut Vec<HybridCache>,
    ) -> Result<Array, Exception> {
        let h = self.forward_hidden(inputs, cache)?;
        let last = h.index((.., -1, ..));
        self.apply_lm_head(&last)
    }

    #[allow(non_snake_case)]
    pub fn forward(
        &mut self,
        inputs: &Array,
        cache: &mut Vec<HybridCache>,
    ) -> Result<Array, Exception> {
        let h = self.forward_hidden(inputs, cache)?;
        self.apply_lm_head(&h)
    }

    /// Forward pass starting from pre-built embeddings instead of token IDs.
    /// Used for multimodal generation where visual features are spliced in.
    pub fn forward_from_embeds(
        &mut self,
        embeddings: &Array,
        cache: &mut Vec<HybridCache>,
    ) -> Result<Array, Exception> {
        let mut h = embeddings.clone();
        let t = h.shape()[1];
        let mask = if t > 1 {
            Some(mlx_rs_core::utils::AttentionMask::Causal)
        } else {
            None
        };
        if cache.is_empty() {
            for layer_type in &self.text_model.layer_types {
                if layer_type == "full_attention" {
                    cache.push(HybridCache::KV(KVCache::new()));
                } else {
                    cache.push(HybridCache::Recurrent(RecurrentState::new()));
                }
            }
        }
        for (block, c) in self.text_model.layers.iter_mut().zip(cache.iter_mut()) {
            h = block.forward(&h, mask.as_ref(), c)?;
        }
        h = self.text_model.norm.forward(&h)?;
        let last = h.index((.., -1i32, ..));
        self.apply_lm_head(&last)
    }

    /// Embed a slice of token IDs.
    pub fn embed_tokens(&mut self, ids: &[i32]) -> Result<Array, Exception> {
        let arr = Array::from_slice(ids, &[1, ids.len() as i32]);
        self.text_model.embed_tokens.forward(&arr)
    }

    /// Run all layers, capture hidden states at specified layer indices, return
    /// last-position logits AND concatenated captures [B, T, num_captures * hidden].
    /// `layer_ids` must be valid 0-based layer indices. Captures the hidden state
    /// AFTER the block forward for that layer (before the final norm).
    /// Only projects the last sequence position through lm_head to avoid [B, T, vocab] OOM.
    pub fn forward_last_logits_with_hidden_capture(
        &mut self,
        inputs: &Array,
        cache: &mut Vec<HybridCache>,
        capture_layer_ids: &[usize],
    ) -> Result<(Array, Array), Exception> {
        if capture_layer_ids.is_empty() {
            return Ok((self.forward_last_logits(inputs, cache)?, Array::zeros::<f32>(&[0])?));
        }
        if let Some(&layer_id) = capture_layer_ids
            .iter()
            .find(|&&layer_id| layer_id >= self.text_model.layers.len())
        {
            return Err(Exception::custom(format!(
                "capture layer index {layer_id} out of range for {} layers",
                self.text_model.layers.len()
            )));
        }

        let mut h = self.text_model.embed_tokens.forward(inputs)?;
        let t = h.shape()[1];
        let mask = if t > 1 {
            Some(mlx_rs_core::utils::AttentionMask::Causal)
        } else {
            None
        };

        if cache.is_empty() {
            for layer_type in &self.text_model.layer_types {
                if layer_type == "full_attention" {
                    cache.push(HybridCache::KV(KVCache::new()));
                } else {
                    cache.push(HybridCache::Recurrent(RecurrentState::new()));
                }
            }
        }

        let mut captures = Vec::with_capacity(capture_layer_ids.len());
        for (layer_idx, (layer, c)) in self
            .text_model
            .layers
            .iter_mut()
            .zip(cache.iter_mut())
            .enumerate()
        {
            h = layer.forward(&h, mask.as_ref(), c)?;
            if capture_layer_ids.iter().any(|&id| id == layer_idx) {
                captures.push(h.clone());
            }
        }

        h = self.text_model.norm.forward(&h)?;
        let logits = self.apply_lm_head(&h.index((.., -1i32, ..)))?;
        let capture_refs = captures.iter().collect::<Vec<_>>();
        let captures = concatenate_axis(&capture_refs, 2)?;
        Ok((logits, captures))
    }

    /// Apply lm_head (or embed_tokens.T if no lm_head) to arbitrary hidden states.
    /// `h` can be any shape ending in hidden_size — result replaces last dim with vocab_size.
    pub fn apply_lm_head(&mut self, h: &Array) -> Result<Array, Exception> {
        match self.lm_head.as_mut() {
            Some(lm_head) => lm_head.forward(h),
            None => match &mut self.text_model.embed_tokens {
                MaybeQuantized::Original(e) => e.as_linear(h),
                MaybeQuantized::Quantized(qe) => qe.as_linear(h),
            },
        }
    }

    /// Extract the lm_head weight matrix [vocab_size, hidden_size] as a contiguous BF16 Array.
    /// For tied weights (no separate lm_head), this dequantizes the embedding weight.
    /// Used to project draft model hidden states through the target's vocabulary projection.
    pub fn get_lm_head_weight(&mut self) -> Result<Array, Exception> {
        let weight = match self.lm_head.as_mut() {
            Some(MaybeQuantized::Original(l)) => l.weight.as_ref().clone(),
            Some(MaybeQuantized::Quantized(ql)) => dequantize(
                &ql.inner.weight,
                &ql.scales,
                &ql.biases,
                ql.group_size,
                ql.bits,
                None::<&str>,
            )?,
            None => match &mut self.text_model.embed_tokens {
                MaybeQuantized::Original(e) => e.weight.as_ref().clone(),
                MaybeQuantized::Quantized(qe) => dequantize(
                    &qe.inner.weight,
                    &qe.scales,
                    &qe.biases,
                    qe.group_size,
                    qe.bits,
                    None::<&str>,
                )?,
            },
        };
        weight.as_dtype(Dtype::Bfloat16)?.contiguous()
    }

    #[allow(non_snake_case)]
    fn forward_hidden(
        &mut self,
        inputs: &Array,
        cache: &mut Vec<HybridCache>,
    ) -> Result<Array, Exception> {
        let mut h = self.text_model.embed_tokens.forward(inputs)?;

        let T = h.shape()[1];
        // Causal-only prefill: pass the SDPA causal-mode marker through to attention
        // instead of materializing an O(T^2) explicit mask array. The KV-offset is
        // handled inside MLX's fused SDPA when paired with the cache offset.
        let mask = if T > 1 {
            Some(mlx_rs_core::utils::AttentionMask::Causal)
        } else {
            None
        };

        // Lazily allocate standard caches; callers that want quantized caches
        // should pre-populate via `model.new_cache(KVCacheMode::Quantized)`.
        if cache.is_empty() {
            for layer_type in &self.text_model.layer_types {
                if layer_type == "full_attention" {
                    cache.push(HybridCache::KV(KVCache::new()));
                } else {
                    cache.push(HybridCache::Recurrent(RecurrentState::new()));
                }
            }
        }

        for (layer, c) in self.text_model.layers.iter_mut().zip(cache.iter_mut()) {
            h = layer.forward(&h, mask.as_ref(), c)?;
        }

        self.text_model.norm.forward(&h)
    }
}

// ============================================================================
// Weight Loading
// ============================================================================

#[derive(Debug, Clone, serde::Deserialize)]
pub struct WeightMap {
    pub weight_map: HashMap<String, String>,
}

fn load_all_weights(model_dir: &Path) -> Result<HashMap<String, Array>, Error> {
    let weights_index = model_dir.join("model.safetensors.index.json");

    if weights_index.exists() {
        let json = std::fs::read_to_string(weights_index)?;
        let weight_map: WeightMap = serde_json::from_str(&json)?;
        let weight_files: HashSet<&String> = weight_map.weight_map.values().collect();

        let mut all_weights: HashMap<String, Array> = HashMap::new();
        for weight_file in weight_files {
            let path = model_dir.join(weight_file);
            let loaded = Array::load_safetensors(&path)?;
            all_weights.extend(loaded);
        }
        Ok(all_weights)
    } else {
        let path = model_dir.join("model.safetensors");
        let loaded = Array::load_safetensors(&path)?;
        Ok(loaded)
    }
}

fn get_weight(weights: &HashMap<String, Array>, key: &str) -> Result<Array, Error> {
    weights
        .get(key)
        .cloned()
        .ok_or_else(|| Error::WeightNotFound(key.to_string()))
}

fn make_quantized_linear(
    weights: &HashMap<String, Array>,
    prefix: &str,
    group_size: i32,
    bits: i32,
) -> Result<nn::QuantizedLinear, Error> {
    let weight = get_weight(weights, &format!("{}.weight", prefix))?;
    let scales = get_weight(weights, &format!("{}.scales", prefix))?;
    let biases = get_weight(weights, &format!("{}.biases", prefix))?;

    let inner = nn::Linear {
        weight: Param::new(weight),
        bias: Param::new(None),
    };

    let mut ql = nn::QuantizedLinear {
        group_size,
        bits,
        scales: Param::new(scales),
        biases: Param::new(biases),
        inner,
    };
    ql.freeze_parameters(true);
    Ok(ql)
}

fn make_quantized_embedding(
    weights: &HashMap<String, Array>,
    prefix: &str,
    group_size: i32,
    bits: i32,
) -> Result<nn::QuantizedEmbedding, Error> {
    let weight = get_weight(weights, &format!("{}.weight", prefix))?;
    let scales = get_weight(weights, &format!("{}.scales", prefix))?;
    let biases = get_weight(weights, &format!("{}.biases", prefix))?;

    let inner = nn::Embedding {
        weight: Param::new(weight),
    };

    let mut qe = nn::QuantizedEmbedding {
        group_size,
        bits,
        scales: Param::new(scales),
        biases: Param::new(biases),
        inner,
    };
    qe.freeze_parameters(true);
    Ok(qe)
}

fn load_rms_norm(
    weights: &HashMap<String, Array>,
    key: &str,
    eps: f32,
) -> Result<nn::RmsNorm, Error> {
    Ok(nn::RmsNorm {
        weight: Param::new(get_weight(weights, key)?),
        eps,
    })
}

fn make_quantized_switch_linear(
    weights: &HashMap<String, Array>,
    prefix: &str,
    group_size: i32,
    bits: i32,
) -> Result<QuantizedSwitchLinear, Error> {
    let weight = get_weight(weights, &format!("{}.weight", prefix))?;
    let scales = get_weight(weights, &format!("{}.scales", prefix))?;
    let biases = get_weight(weights, &format!("{}.biases", prefix))?;

    let shape = weight.shape();
    let num_experts = shape[0] as i32;
    let output_dims = shape[1] as i32;
    let scales_shape = scales.shape();
    let input_dims = (scales_shape[2] as i32) * group_size;

    Ok(QuantizedSwitchLinear {
        num_experts,
        input_dims,
        output_dims,
        group_size,
        bits,
        weight: Param::new(weight),
        scales: Param::new(scales),
        biases: Param::new(biases),
    })
}

/// Detect the weight key prefix (VLM vs standalone text model).
fn detect_prefix(weights: &HashMap<String, Array>) -> &'static str {
    if weights.keys().any(|k| k.starts_with("language_model.")) {
        "language_model.model"
    } else {
        "model"
    }
}

fn detect_lm_head_prefix(weights: &HashMap<String, Array>) -> &'static str {
    if weights.contains_key("language_model.lm_head.weight") {
        "language_model.lm_head"
    } else {
        "lm_head"
    }
}

fn build_model_from_weights(
    weights: &HashMap<String, Array>,
    args: &ModelArgs,
) -> Result<Model, Error> {
    let tc = &args.text_config;

    let quant = args.quantization();
    let (group_size, bits) = match quant {
        Some(q) => (q.group_size, q.bits),
        None => {
            return Err(Error::Model(
                "Only quantized models are supported".to_string(),
            ))
        }
    };

    if tc.layer_types.len() != tc.num_hidden_layers as usize {
        return Err(Error::Model(format!(
            "layer_types length ({}) != num_hidden_layers ({})",
            tc.layer_types.len(),
            tc.num_hidden_layers
        )));
    }

    let prefix = detect_prefix(weights);
    let lm_head_prefix = detect_lm_head_prefix(weights);

    let mut layers = Vec::with_capacity(tc.num_hidden_layers as usize);
    for i in 0..tc.num_hidden_layers {
        let layer_prefix = format!("{}.layers.{}", prefix, i);
        let layer_type = &tc.layer_types[i as usize];

        let attention = if layer_type == "full_attention" {
            AttentionLayer::FullAttention(load_gated_attention(
                weights,
                &layer_prefix,
                tc,
                group_size,
                bits,
            )?)
        } else {
            AttentionLayer::LinearAttention(load_gated_deltanet(
                weights,
                &layer_prefix,
                tc,
                group_size,
                bits,
            )?)
        };

        let ffn = if tc.is_moe() {
            let num_experts = tc
                .num_experts
                .ok_or_else(|| Error::Model("MoE config missing num_experts".to_string()))?;
            let top_k = tc.num_experts_per_tok.ok_or_else(|| {
                Error::Model("MoE config missing num_experts_per_tok".to_string())
            })?;

            let gate = MaybeQuantized::Quantized(make_quantized_linear(
                weights,
                &format!("{}.mlp.gate", layer_prefix),
                group_size,
                8,
            )?);

            let switch_mlp = SwitchGLU {
                gate_proj: make_quantized_switch_linear(
                    weights,
                    &format!("{}.mlp.switch_mlp.gate_proj", layer_prefix),
                    group_size,
                    bits,
                )?,
                up_proj: make_quantized_switch_linear(
                    weights,
                    &format!("{}.mlp.switch_mlp.up_proj", layer_prefix),
                    group_size,
                    bits,
                )?,
                down_proj: make_quantized_switch_linear(
                    weights,
                    &format!("{}.mlp.switch_mlp.down_proj", layer_prefix),
                    group_size,
                    bits,
                )?,
            };

            let shared_expert = SharedExpert {
                gate_proj: MaybeQuantized::Quantized(make_quantized_linear(
                    weights,
                    &format!("{}.mlp.shared_expert.gate_proj", layer_prefix),
                    group_size,
                    bits,
                )?),
                up_proj: MaybeQuantized::Quantized(make_quantized_linear(
                    weights,
                    &format!("{}.mlp.shared_expert.up_proj", layer_prefix),
                    group_size,
                    bits,
                )?),
                down_proj: MaybeQuantized::Quantized(make_quantized_linear(
                    weights,
                    &format!("{}.mlp.shared_expert.down_proj", layer_prefix),
                    group_size,
                    bits,
                )?),
            };

            let shared_expert_gate = MaybeQuantized::Quantized(make_quantized_linear(
                weights,
                &format!("{}.mlp.shared_expert_gate", layer_prefix),
                group_size,
                8,
            )?);

            FfnBlock::Moe(MoeBlock {
                num_experts,
                top_k,
                gate,
                switch_mlp,
                shared_expert,
                shared_expert_gate,
            })
        } else {
            FfnBlock::Dense(DenseMlp {
                gate_proj: MaybeQuantized::Quantized(make_quantized_linear(
                    weights,
                    &format!("{}.mlp.gate_proj", layer_prefix),
                    group_size,
                    bits,
                )?),
                up_proj: MaybeQuantized::Quantized(make_quantized_linear(
                    weights,
                    &format!("{}.mlp.up_proj", layer_prefix),
                    group_size,
                    bits,
                )?),
                down_proj: MaybeQuantized::Quantized(make_quantized_linear(
                    weights,
                    &format!("{}.mlp.down_proj", layer_prefix),
                    group_size,
                    bits,
                )?),
            })
        };

        let block = TransformerBlock {
            attention,
            ffn,
            input_layernorm: load_rms_norm(
                weights,
                &format!("{}.input_layernorm.weight", layer_prefix),
                tc.rms_norm_eps,
            )?,
            post_attention_layernorm: load_rms_norm(
                weights,
                &format!("{}.post_attention_layernorm.weight", layer_prefix),
                tc.rms_norm_eps,
            )?,
        };

        layers.push(block);
    }

    let embed_tokens = MaybeQuantized::Quantized(make_quantized_embedding(
        weights,
        &format!("{}.embed_tokens", prefix),
        group_size,
        bits,
    )?);

    let norm = load_rms_norm(weights, &format!("{}.norm.weight", prefix), tc.rms_norm_eps)?;

    let lm_head = if !args.tie_word_embeddings {
        Some(MaybeQuantized::Quantized(make_quantized_linear(
            weights,
            lm_head_prefix,
            group_size,
            bits,
        )?))
    } else {
        None
    };

    let text_model = Qwen36TextModel {
        embed_tokens,
        layers,
        norm,
        layer_types: tc.layer_types.clone(),
    };

    Ok(Model {
        args: args.clone(),
        text_model,
        lm_head,
    })
}

pub fn load_model(model_dir: impl AsRef<Path>) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();

    let config_file = std::fs::File::open(model_dir.join("config.json"))?;
    let args: ModelArgs = serde_json::from_reader(config_file)?;
    let tc = &args.text_config;

    let quant = args.quantization();
    let (_, bits) = match quant {
        Some(q) => (q.group_size, q.bits),
        None => {
            return Err(Error::Model(
                "Only quantized models are supported".to_string(),
            ))
        }
    };

    eprintln!(
        "Loading {}-bit quantized Qwen3.6 ({} layers, {})...",
        bits,
        tc.num_hidden_layers,
        if tc.is_moe() {
            format!(
                "MoE {} experts top-{}",
                tc.num_experts.unwrap_or(0),
                tc.num_experts_per_tok.unwrap_or(0)
            )
        } else {
            "dense MLP".to_string()
        }
    );

    let weights = load_all_weights(model_dir)?;
    build_model_from_weights(&weights, &args)
}

pub struct VlModel {
    pub text: Model,
    pub vision: VisionTower,
    pub image_token_id: i32,
    pub vision_start_token_id: i32,
    pub vision_end_token_id: i32,
}

impl VlModel {
    pub fn new_cache(&self, mode: KVCacheMode) -> Vec<HybridCache> {
        self.text.new_cache(mode)
    }

    /// Encode image bytes (JPEG/PNG/etc) into visual feature tokens.
    /// Returns `(visual_features: Array [N_vis, hidden], h_patches, w_patches)`.
    pub fn encode_image_bytes(
        &mut self,
        bytes: &[u8],
    ) -> Result<(Array, i32, i32), mlx_rs_core::error::Error> {
        let img = image::load_from_memory(bytes).map_err(|e| {
            mlx_rs_core::error::Error::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                e.to_string(),
            ))
        })?;
        let patch_size = self
            .text
            .args
            .vision_config
            .as_ref()
            .map(|vc| vc.patch_size)
            .unwrap_or(16);
        let temporal_patch_size = self
            .text
            .args
            .vision_config
            .as_ref()
            .map(|vc| vc.temporal_patch_size)
            .unwrap_or(2);
        let (pixel_array, h_patches, w_patches) =
            crate::vision::preprocess_image(&img, patch_size, temporal_patch_size)?;
        let visual_features = self.vision.forward(&pixel_array, h_patches, w_patches)?;
        Ok((visual_features, h_patches, w_patches))
    }

    /// Prefill with mixed text+image content.
    pub fn prefill_multimodal(
        &mut self,
        input_ids: &[i32],
        visual_features: &Array,
        cache: &mut Vec<HybridCache>,
    ) -> Result<Array, mlx_rs_core::error::Error> {
        let n_visual = visual_features.shape()[0] as usize;
        let hidden_size = visual_features.shape()[1];

        let image_token_id = self.image_token_id;
        let block_start = input_ids.iter().position(|&t| t == image_token_id);

        let logits = if let Some(start) = block_start {
            let block_len = input_ids[start..]
                .iter()
                .take_while(|&&t| t == image_token_id)
                .count();
            if block_len != n_visual {
                return Err(mlx_rs_core::error::Error::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!(
                        "Image token count mismatch: {} placeholders vs {} visual tokens",
                        block_len, n_visual
                    ),
                )));
            }

            let pre_ids = &input_ids[..start];
            let post_ids = &input_ids[start + block_len..];

            let mut parts: Vec<Array> = Vec::new();

            if !pre_ids.is_empty() {
                let pre_emb = self.text.embed_tokens(pre_ids).map_err(|e| {
                    mlx_rs_core::error::Error::Io(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        e.to_string(),
                    ))
                })?;
                parts.push(pre_emb);
            }

            let vis = visual_features.reshape(&[1, n_visual as i32, hidden_size])?;
            parts.push(vis);

            if !post_ids.is_empty() {
                let post_emb = self.text.embed_tokens(post_ids).map_err(|e| {
                    mlx_rs_core::error::Error::Io(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        e.to_string(),
                    ))
                })?;
                parts.push(post_emb);
            }

            let combined = if parts.len() == 1 {
                parts.remove(0)
            } else {
                let refs: Vec<&Array> = parts.iter().collect();
                mlx_rs::ops::concatenate_axis(&refs, 1).map_err(|e| {
                    mlx_rs_core::error::Error::Io(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        e.to_string(),
                    ))
                })?
            };

            self.text
                .forward_from_embeds(&combined, cache)
                .map_err(|e| {
                    mlx_rs_core::error::Error::Io(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        e.to_string(),
                    ))
                })?
        } else {
            let arr = Array::from_slice(input_ids, &[1, input_ids.len() as i32]);
            self.text.forward_last_logits(&arr, cache).map_err(|e| {
                mlx_rs_core::error::Error::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    e.to_string(),
                ))
            })?
        };

        Ok(logits)
    }

    pub fn prefill_text(
        &mut self,
        input_ids: &[i32],
        cache: &mut Vec<HybridCache>,
    ) -> Result<Array, mlx_rs_core::error::Error> {
        let arr = Array::from_slice(input_ids, &[1, input_ids.len() as i32]);
        self.text.forward_last_logits(&arr, cache).map_err(|e| {
            mlx_rs_core::error::Error::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                e.to_string(),
            ))
        })
    }

    /// Single-token decode step. `token_id` is the previously sampled token.
    pub fn decode_token(
        &mut self,
        token_id: i32,
        cache: &mut Vec<HybridCache>,
    ) -> Result<Array, mlx_rs_core::error::Error> {
        let arr = Array::from_slice(&[token_id], &[1, 1]);
        self.text.forward_last_logits(&arr, cache).map_err(|e| {
            mlx_rs_core::error::Error::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                e.to_string(),
            ))
        })
    }
}

pub fn load_vl_model(model_dir: impl AsRef<Path>) -> Result<VlModel, Error> {
    let model_dir = model_dir.as_ref();
    let config_file = std::fs::File::open(model_dir.join("config.json"))?;
    let args: ModelArgs = serde_json::from_reader(config_file)?;

    let vc = args
        .vision_config
        .as_ref()
        .ok_or_else(|| Error::Model("Not a VL model: missing vision_config".to_string()))?;
    let image_token_id = args.image_token_id.unwrap_or(248056);
    let vision_start_token_id = args.vision_start_token_id.unwrap_or(248053);
    let vision_end_token_id = args.vision_end_token_id.unwrap_or(248054);

    if vc.out_hidden_size != args.text_config.hidden_size {
        return Err(Error::Model(format!(
            "Vision out_hidden_size ({}) != text hidden_size ({})",
            vc.out_hidden_size, args.text_config.hidden_size
        )));
    }

    let weights = load_all_weights(model_dir)?;
    let text = build_model_from_weights(&weights, &args)?;
    let vision = crate::vision::load_vision_tower(&weights, vc)?;

    Ok(VlModel {
        text,
        vision,
        image_token_id,
        vision_start_token_id,
        vision_end_token_id,
    })
}

fn load_gated_attention(
    weights: &HashMap<String, Array>,
    layer_prefix: &str,
    tc: &TextConfig,
    group_size: i32,
    bits: i32,
) -> Result<GatedAttention, Error> {
    let attn_prefix = format!("{}.self_attn", layer_prefix);

    let rope_dims = (tc.head_dim as f32 * tc.rope_parameters.partial_rotary_factor) as i32;
    let rope = initialize_rope(
        rope_dims,
        tc.rope_parameters.rope_theta,
        false,
        &None,
        tc.max_position_embeddings,
    )?;

    Ok(GatedAttention {
        n_heads: tc.num_attention_heads,
        n_kv_heads: tc.num_key_value_heads,
        head_dim: tc.head_dim,
        scale: (tc.head_dim as f32).sqrt().recip(),
        q_proj: MaybeQuantized::Quantized(make_quantized_linear(
            weights,
            &format!("{}.q_proj", attn_prefix),
            group_size,
            bits,
        )?),
        k_proj: MaybeQuantized::Quantized(make_quantized_linear(
            weights,
            &format!("{}.k_proj", attn_prefix),
            group_size,
            bits,
        )?),
        v_proj: MaybeQuantized::Quantized(make_quantized_linear(
            weights,
            &format!("{}.v_proj", attn_prefix),
            group_size,
            bits,
        )?),
        o_proj: MaybeQuantized::Quantized(make_quantized_linear(
            weights,
            &format!("{}.o_proj", attn_prefix),
            group_size,
            bits,
        )?),
        q_norm: load_rms_norm(
            weights,
            &format!("{}.q_norm.weight", attn_prefix),
            tc.rms_norm_eps,
        )?,
        k_norm: load_rms_norm(
            weights,
            &format!("{}.k_norm.weight", attn_prefix),
            tc.rms_norm_eps,
        )?,
        rope,
    })
}

fn load_gated_deltanet(
    weights: &HashMap<String, Array>,
    layer_prefix: &str,
    tc: &TextConfig,
    group_size: i32,
    bits: i32,
) -> Result<GatedDeltaNet, Error> {
    let attn_prefix = format!("{}.linear_attn", layer_prefix);

    let num_k_heads = tc.linear_num_key_heads;
    let num_v_heads = tc.linear_num_value_heads;
    if num_v_heads % num_k_heads != 0 {
        return Err(Error::Model(format!(
            "linear_num_value_heads ({}) must be divisible by linear_num_key_heads ({})",
            num_v_heads, num_k_heads
        )));
    }
    let key_head_dim = tc.linear_key_head_dim;
    let value_head_dim = tc.linear_value_head_dim;
    let key_dim = num_k_heads * key_head_dim;
    let value_dim = num_v_heads * value_head_dim;
    let conv_dim = key_dim * 2 + value_dim;
    let conv_kernel_size = tc.linear_conv_kernel_dim;

    Ok(GatedDeltaNet {
        in_proj_qkv: MaybeQuantized::Quantized(make_quantized_linear(
            weights,
            &format!("{}.in_proj_qkv", attn_prefix),
            group_size,
            bits,
        )?),
        in_proj_z: MaybeQuantized::Quantized(make_quantized_linear(
            weights,
            &format!("{}.in_proj_z", attn_prefix),
            group_size,
            bits,
        )?),
        in_proj_a: MaybeQuantized::Quantized(make_quantized_linear(
            weights,
            &format!("{}.in_proj_a", attn_prefix),
            group_size,
            bits,
        )?),
        in_proj_b: MaybeQuantized::Quantized(make_quantized_linear(
            weights,
            &format!("{}.in_proj_b", attn_prefix),
            group_size,
            bits,
        )?),
        conv1d_weight: Param::new(get_weight(
            weights,
            &format!("{}.conv1d.weight", attn_prefix),
        )?),
        a_log: Param::new(get_weight(weights, &format!("{}.A_log", attn_prefix))?),
        dt_bias: Param::new(get_weight(weights, &format!("{}.dt_bias", attn_prefix))?),
        norm: load_rms_norm(
            weights,
            &format!("{}.norm.weight", attn_prefix),
            tc.rms_norm_eps,
        )?,
        out_proj: MaybeQuantized::Quantized(make_quantized_linear(
            weights,
            &format!("{}.out_proj", attn_prefix),
            group_size,
            bits,
        )?),
        num_k_heads,
        num_v_heads,
        key_head_dim,
        value_head_dim,
        key_dim,
        value_dim,
        conv_dim,
        conv_kernel_size,
    })
}
