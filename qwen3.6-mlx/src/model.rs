use std::collections::{HashMap, HashSet};
use std::path::Path;

use mlx_rs::{
    error::Exception,
    module::{Module, ModuleParameters, Param},
    nn,
    ops::indexing::IndexOp,
    quantization::MaybeQuantized,
    Array,
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
use crate::moe::{
    DenseMlp, MoeBlock, QuantizedSwitchLinear, SharedExpert, SwitchGLU,
};

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
            (AttentionLayer::FullAttention(attn), HybridCache::QuantizedKV(qkv_cache)) => {
                attn.forward(GatedAttentionInput {
                    x: &normed,
                    mask,
                    cache: Some(qkv_cache),
                })?
            }
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
        self.text_model.layer_types.iter().map(|lt| {
            if lt == "full_attention" {
                match mode {
                    KVCacheMode::Standard => HybridCache::KV(KVCache::new()),
                    KVCacheMode::Quantized => HybridCache::QuantizedKV(QuantizedKVCache::default()),
                }
            } else {
                HybridCache::Recurrent(RecurrentState::new())
            }
        }).collect()
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
        match self.lm_head.as_mut() {
            Some(lm_head) => lm_head.forward(&last),
            None => match &mut self.text_model.embed_tokens {
                MaybeQuantized::Original(e) => e.as_linear(&last),
                MaybeQuantized::Quantized(qe) => qe.as_linear(&last),
            },
        }
    }

    #[allow(non_snake_case)]
    pub fn forward(
        &mut self,
        inputs: &Array,
        cache: &mut Vec<HybridCache>,
    ) -> Result<Array, Exception> {
        let h = self.forward_hidden(inputs, cache)?;
        match self.lm_head.as_mut() {
            Some(lm_head) => lm_head.forward(&h),
            None => match &mut self.text_model.embed_tokens {
                MaybeQuantized::Original(e) => e.as_linear(&h),
                MaybeQuantized::Quantized(qe) => qe.as_linear(&h),
            },
        }
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

pub fn load_model(model_dir: impl AsRef<Path>) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();

    let config_file = std::fs::File::open(model_dir.join("config.json"))?;
    let args: ModelArgs = serde_json::from_reader(config_file)?;
    let tc = &args.text_config;

    let quant = args.quantization();
    let (group_size, bits) = match quant {
        Some(q) => (q.group_size, q.bits),
        None => return Err(Error::Model("Only quantized models are supported".to_string())),
    };

    if tc.layer_types.len() != tc.num_hidden_layers as usize {
        return Err(Error::Model(format!(
            "layer_types length ({}) != num_hidden_layers ({})",
            tc.layer_types.len(),
            tc.num_hidden_layers
        )));
    }

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
    let prefix = detect_prefix(&weights);
    let lm_head_prefix = detect_lm_head_prefix(&weights);

    let mut layers = Vec::with_capacity(tc.num_hidden_layers as usize);
    for i in 0..tc.num_hidden_layers {
        let layer_prefix = format!("{}.layers.{}", prefix, i);
        let layer_type = &tc.layer_types[i as usize];

        // Attention: gated full attention or DeltaNet
        let attention = if layer_type == "full_attention" {
            AttentionLayer::FullAttention(load_gated_attention(
                &weights,
                &layer_prefix,
                tc,
                group_size,
                bits,
            )?)
        } else {
            AttentionLayer::LinearAttention(load_gated_deltanet(
                &weights,
                &layer_prefix,
                tc,
                group_size,
                bits,
            )?)
        };

        // FFN: MoE for 35B, dense MLP for 27B
        let ffn = if tc.is_moe() {
            let num_experts = tc.num_experts.ok_or_else(|| {
                Error::Model("MoE config missing num_experts".to_string())
            })?;
            let top_k = tc.num_experts_per_tok.ok_or_else(|| {
                Error::Model("MoE config missing num_experts_per_tok".to_string())
            })?;

            let gate = MaybeQuantized::Quantized(make_quantized_linear(
                &weights,
                &format!("{}.mlp.gate", layer_prefix),
                group_size,
                8, // 8-bit for routing accuracy
            )?);

            let switch_mlp = SwitchGLU {
                gate_proj: make_quantized_switch_linear(
                    &weights,
                    &format!("{}.mlp.switch_mlp.gate_proj", layer_prefix),
                    group_size,
                    bits,
                )?,
                up_proj: make_quantized_switch_linear(
                    &weights,
                    &format!("{}.mlp.switch_mlp.up_proj", layer_prefix),
                    group_size,
                    bits,
                )?,
                down_proj: make_quantized_switch_linear(
                    &weights,
                    &format!("{}.mlp.switch_mlp.down_proj", layer_prefix),
                    group_size,
                    bits,
                )?,
            };

            let shared_expert = SharedExpert {
                gate_proj: MaybeQuantized::Quantized(make_quantized_linear(
                    &weights,
                    &format!("{}.mlp.shared_expert.gate_proj", layer_prefix),
                    group_size,
                    bits,
                )?),
                up_proj: MaybeQuantized::Quantized(make_quantized_linear(
                    &weights,
                    &format!("{}.mlp.shared_expert.up_proj", layer_prefix),
                    group_size,
                    bits,
                )?),
                down_proj: MaybeQuantized::Quantized(make_quantized_linear(
                    &weights,
                    &format!("{}.mlp.shared_expert.down_proj", layer_prefix),
                    group_size,
                    bits,
                )?),
            };

            let shared_expert_gate = MaybeQuantized::Quantized(make_quantized_linear(
                &weights,
                &format!("{}.mlp.shared_expert_gate", layer_prefix),
                group_size,
                8, // 8-bit for gating accuracy
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
                    &weights,
                    &format!("{}.mlp.gate_proj", layer_prefix),
                    group_size,
                    bits,
                )?),
                up_proj: MaybeQuantized::Quantized(make_quantized_linear(
                    &weights,
                    &format!("{}.mlp.up_proj", layer_prefix),
                    group_size,
                    bits,
                )?),
                down_proj: MaybeQuantized::Quantized(make_quantized_linear(
                    &weights,
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
                &weights,
                &format!("{}.input_layernorm.weight", layer_prefix),
                tc.rms_norm_eps,
            )?,
            post_attention_layernorm: load_rms_norm(
                &weights,
                &format!("{}.post_attention_layernorm.weight", layer_prefix),
                tc.rms_norm_eps,
            )?,
        };

        layers.push(block);
    }

    let embed_tokens = MaybeQuantized::Quantized(make_quantized_embedding(
        &weights,
        &format!("{}.embed_tokens", prefix),
        group_size,
        bits,
    )?);

    let norm = load_rms_norm(
        &weights,
        &format!("{}.norm.weight", prefix),
        tc.rms_norm_eps,
    )?;

    let lm_head = if !args.tie_word_embeddings {
        Some(MaybeQuantized::Quantized(make_quantized_linear(
            &weights,
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
        args,
        text_model,
        lm_head,
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
        a_log: Param::new(get_weight(
            weights,
            &format!("{}.A_log", attn_prefix),
        )?),
        dt_bias: Param::new(get_weight(
            weights,
            &format!("{}.dt_bias", attn_prefix),
        )?),
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
