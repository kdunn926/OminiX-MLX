//! Glm46V text decoder — `Glm4vTextDecoderLayer` port.
//!
//! 16-layer Glm4 family decoder with the **sandwich-norm** pattern (four
//! RMSNorms per layer, two around attention and two around the MLP) and
//! a **fused** `gate_up_proj` for the SwiGLU MLP:
//!
//! ```text
//!   x_in = x
//!   x1 = input_layernorm(x_in)
//!   a  = self_attn(x1, mrope-cos/sin, mask, cache)
//!   a  = post_self_attn_layernorm(a)
//!   x  = x_in + a
//!
//!   x_in = x
//!   x1 = post_attention_layernorm(x_in)
//!   m  = mlp(x1)                                   # gate_up_proj split,
//!                                                  # silu(gate)*up → down
//!   m  = post_mlp_layernorm(m)
//!   x  = x_in + m
//! ```
//!
//! Attention is standard GQA + MROPE (16 Q heads, 8 KV heads, head_dim
//! 128, attention_bias=false). The MROPE cos/sin tables are built by
//! [`crate::mrope`] from the 3-channel position-id tensor produced by
//! [`crate::position_ids`].

use mlx_rs::{
    module::{Module, Param},
    nn,
    ops::indexing::IndexOp,
    Array,
};
use mlx_rs_core::{
    cache::KeyValueCache,
    error::{Error, Result},
    utils::{scaled_dot_product_attention, SdpaMask},
};

use crate::config::GlmOcrTextConfig;
use crate::mrope::{self, MropePartition};

// ── Building blocks ────────────────────────────────────────────────────────

/// SwiGLU MLP with a **fused** gate+up projection:
///   `gate_up_proj.weight` ships shape `(2 * intermediate, hidden)`.
/// We split along the last axis after projecting, multiply
/// `silu(gate) * up`, then down-project.
pub struct Glm4OcrMlp {
    pub gate_up_proj: nn::Linear,
    pub down_proj: nn::Linear,
}

impl Glm4OcrMlp {
    pub fn forward(&mut self, x: &Array) -> Result<Array> {
        let gu = self.gate_up_proj.forward(x).map_err(Error::from)?;
        let last = gu.shape()[gu.ndim() - 1];
        let half = last / 2;
        let gate = gu.index((.., .., ..half));
        let up = gu.index((.., .., half..));
        let act = mlx_rs::nn::silu(&gate).map_err(Error::from)?;
        let prod = act.multiply(&up).map_err(Error::from)?;
        self.down_proj.forward(&prod).map_err(Error::from)
    }
}

/// GQA attention with MROPE. Separate q/k/v/o projections (no bias).
pub struct Glm4OcrAttention {
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub q_proj: nn::Linear,
    pub k_proj: nn::Linear,
    pub v_proj: nn::Linear,
    pub o_proj: nn::Linear,
}

pub struct AttentionInput<'a, C: KeyValueCache + Default> {
    pub x: &'a Array,
    pub cos: &'a Array,
    pub sin: &'a Array,
    pub mask: Option<SdpaMask<'a>>,
    pub cache: &'a mut C,
}

impl Glm4OcrAttention {
    pub fn forward<C>(&mut self, input: AttentionInput<'_, C>) -> Result<Array>
    where
        C: KeyValueCache + Default,
    {
        let AttentionInput { x, cos, sin, mask, cache } = input;
        let s = x.shape();
        let b = s[0];
        let t = s[1];
        let q = self.q_proj.forward(x).map_err(Error::from)?;
        let k = self.k_proj.forward(x).map_err(Error::from)?;
        let v = self.v_proj.forward(x).map_err(Error::from)?;
        let q = q
            .reshape(&[b, t, self.n_heads, self.head_dim])
            .map_err(Error::from)?
            .transpose_axes(&[0, 2, 1, 3])
            .map_err(Error::from)?;
        let k = k
            .reshape(&[b, t, self.n_kv_heads, self.head_dim])
            .map_err(Error::from)?
            .transpose_axes(&[0, 2, 1, 3])
            .map_err(Error::from)?;
        let v = v
            .reshape(&[b, t, self.n_kv_heads, self.head_dim])
            .map_err(Error::from)?
            .transpose_axes(&[0, 2, 1, 3])
            .map_err(Error::from)?;
        let (q, k) = mrope::apply_rotary_qk(&q, &k, cos, sin)?;
        let (k_full, v_full) = cache.update_and_fetch(k, v).map_err(Error::from)?;
        let attn = scaled_dot_product_attention::<C>(q, k_full, v_full, None, self.scale, mask)
            .map_err(Error::from)?;
        let attn = attn
            .transpose_axes(&[0, 2, 1, 3])
            .map_err(Error::from)?
            .reshape(&[b, t, self.n_heads * self.head_dim])
            .map_err(Error::from)?;
        self.o_proj.forward(&attn).map_err(Error::from)
    }
}

/// One sandwich-norm decoder block.
pub struct Glm4OcrDecoderLayer {
    pub input_layernorm: nn::RmsNorm,
    pub self_attn: Glm4OcrAttention,
    pub post_self_attn_layernorm: nn::RmsNorm,
    pub post_attention_layernorm: nn::RmsNorm,
    pub mlp: Glm4OcrMlp,
    pub post_mlp_layernorm: nn::RmsNorm,
}

pub struct LayerInput<'a, C: KeyValueCache + Default> {
    pub x: &'a Array,
    pub cos: &'a Array,
    pub sin: &'a Array,
    pub mask: Option<SdpaMask<'a>>,
    pub cache: &'a mut C,
}

impl Glm4OcrDecoderLayer {
    pub fn forward<C>(&mut self, input: LayerInput<'_, C>) -> Result<Array>
    where
        C: KeyValueCache + Default,
    {
        let LayerInput { x, cos, sin, mask, cache } = input;
        // Attention sub-block (sandwich norm).
        let normed = self.input_layernorm.forward(x).map_err(Error::from)?;
        let a = self.self_attn.forward(AttentionInput {
            x: &normed,
            cos,
            sin,
            mask,
            cache,
        })?;
        let a = self
            .post_self_attn_layernorm
            .forward(&a)
            .map_err(Error::from)?;
        let x = x.add(&a).map_err(Error::from)?;

        // MLP sub-block (sandwich norm).
        let normed = self
            .post_attention_layernorm
            .forward(&x)
            .map_err(Error::from)?;
        let m = self.mlp.forward(&normed)?;
        let m = self.post_mlp_layernorm.forward(&m).map_err(Error::from)?;
        x.add(&m).map_err(Error::from)
    }
}

// ── Top-level model ───────────────────────────────────────────────────────

pub struct Glm4OcrModel {
    pub embed_tokens: nn::Embedding,
    pub layers: Vec<Glm4OcrDecoderLayer>,
    pub norm: nn::RmsNorm,
    pub inv_freq: Array,
    pub partition: MropePartition,
}

impl Glm4OcrModel {
    pub fn embed(&mut self, input_ids: &Array) -> Result<Array> {
        self.embed_tokens.forward(input_ids).map_err(Error::from)
    }

    pub fn forward_from_embeds<C>(
        &mut self,
        embeds: &Array,
        position_ids_3d: &Array,
        cache: &mut [C],
    ) -> Result<Array>
    where
        C: KeyValueCache + Default,
    {
        let (cos, sin) = mrope::build_cos_sin(&self.inv_freq, position_ids_3d, &self.partition)?;
        if cache.len() != self.layers.len() {
            return Err(Error::InvalidConfig(format!(
                "Glm4OcrModel::forward_from_embeds: cache len {} != num_layers {}",
                cache.len(),
                self.layers.len()
            )));
        }
        let mut h = embeds.clone();
        for (layer, c) in self.layers.iter_mut().zip(cache.iter_mut()) {
            h = layer.forward(LayerInput {
                x: &h,
                cos: &cos,
                sin: &sin,
                mask: Some(SdpaMask::Causal),
                cache: c,
            })?;
        }
        self.norm.forward(&h).map_err(Error::from)
    }
}

pub struct Glm4OcrForCausalLM {
    pub model: Glm4OcrModel,
    pub lm_head: nn::Linear,
    pub config: GlmOcrTextConfig,
}

impl Glm4OcrForCausalLM {
    pub fn forward_all_logits_from_embeds<C>(
        &mut self,
        embeds: &Array,
        position_ids_3d: &Array,
        cache: &mut [C],
    ) -> Result<Array>
    where
        C: KeyValueCache + Default,
    {
        let hidden = self
            .model
            .forward_from_embeds(embeds, position_ids_3d, cache)?;
        self.lm_head.forward(&hidden).map_err(Error::from)
    }

    pub fn forward_last_logits_from_embeds<C>(
        &mut self,
        embeds: &Array,
        position_ids_3d: &Array,
        cache: &mut [C],
    ) -> Result<Array>
    where
        C: KeyValueCache + Default,
    {
        let hidden = self
            .model
            .forward_from_embeds(embeds, position_ids_3d, cache)?;
        let last = hidden.index((.., -1, ..));
        self.lm_head.forward(&last).map_err(Error::from)
    }
}

// ── Random-weight smoke (no safetensors needed) ────────────────────────────

pub fn build_with_random_weights(config: &GlmOcrTextConfig) -> Result<Glm4OcrForCausalLM> {
    use mlx_rs::random::uniform;

    let h = config.hidden_size;
    let n_h = config.num_attention_heads;
    let n_kv = config.num_key_value_heads;
    let d = config.head_dim;
    let v = config.vocab_size;
    let inter = config.intermediate_size;
    let scale = 1.0_f32 / (d as f32).sqrt();

    let lin = |in_d: i32, out_d: i32, bias: bool| -> Result<nn::Linear> {
        let lo = -1.0 / (in_d as f32).sqrt();
        let hi = 1.0 / (in_d as f32).sqrt();
        let w = uniform::<_, f32>(lo, hi, &[out_d, in_d], None).map_err(Error::from)?;
        let b = if bias {
            Some(uniform::<_, f32>(lo, hi, &[out_d], None).map_err(Error::from)?)
        } else {
            None
        };
        Ok(nn::Linear {
            weight: Param::new(w),
            bias: Param::new(b),
        })
    };
    let rms = |dim: i32| -> Result<nn::RmsNorm> {
        let w = mlx_rs::ops::ones::<f32>(&[dim]).map_err(Error::from)?;
        Ok(nn::RmsNorm {
            weight: Param::new(w),
            eps: config.rms_norm_eps,
        })
    };

    let mut layers = Vec::with_capacity(config.num_hidden_layers as usize);
    for _ in 0..config.num_hidden_layers {
        let attn = Glm4OcrAttention {
            n_heads: n_h,
            n_kv_heads: n_kv,
            head_dim: d,
            scale,
            q_proj: lin(h, n_h * d, false)?,
            k_proj: lin(h, n_kv * d, false)?,
            v_proj: lin(h, n_kv * d, false)?,
            o_proj: lin(n_h * d, h, false)?,
        };
        let mlp = Glm4OcrMlp {
            gate_up_proj: lin(h, 2 * inter, false)?,
            down_proj: lin(inter, h, false)?,
        };
        layers.push(Glm4OcrDecoderLayer {
            input_layernorm: rms(h)?,
            self_attn: attn,
            post_self_attn_layernorm: rms(h)?,
            post_attention_layernorm: rms(h)?,
            mlp,
            post_mlp_layernorm: rms(h)?,
        });
    }
    let embed_w = uniform::<_, f32>(-0.02, 0.02, &[v, h], None).map_err(Error::from)?;
    let embed_tokens = nn::Embedding {
        weight: Param::new(embed_w),
    };
    let norm = rms(h)?;
    let rope_theta = config.rope_parameters.rope_theta.unwrap_or(10_000.0);
    let mrope_section = config
        .rope_parameters
        .mrope_section
        .clone()
        .ok_or_else(|| Error::InvalidConfig("rope_parameters.mrope_section missing".to_string()))?;
    let inv_freq = mrope::inv_freq(d, rope_theta);
    let partition = MropePartition::new(&mrope_section, d)?;
    let model = Glm4OcrModel {
        embed_tokens,
        layers,
        norm,
        inv_freq,
        partition,
    };
    let lm_head = lin(h, v, false)?;
    Ok(Glm4OcrForCausalLM {
        model,
        lm_head,
        config: config.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs_core::cache::KVCache;

    fn tiny_config() -> GlmOcrTextConfig {
        serde_json::from_str::<GlmOcrTextConfig>(
            r#"{
              "model_type": "glm_ocr_text",
              "hidden_size": 16,
              "num_hidden_layers": 2,
              "num_attention_heads": 2,
              "num_key_value_heads": 1,
              "head_dim": 8,
              "intermediate_size": 32,
              "vocab_size": 20,
              "max_position_embeddings": 64,
              "rope_parameters": {
                "rope_type": "default",
                "mrope_section": [1, 1, 2],
                "rope_theta": 10000
              },
              "rms_norm_eps": 1e-5,
              "hidden_act": "silu",
              "tie_word_embeddings": false
            }"#,
        )
        .unwrap()
    }

    fn build_cache(n: usize) -> Vec<KVCache> {
        (0..n).map(|_| KVCache::default()).collect()
    }

    #[test]
    fn forward_all_logits_shape() {
        let cfg = tiny_config();
        let mut model = build_with_random_weights(&cfg).unwrap();
        let mut cache = build_cache(cfg.num_hidden_layers as usize);
        let ids = Array::from_slice(&[0_i32, 1, 2, 3], &[1, 4]);
        let mut pos = vec![0_i32; 3 * 4];
        for t in 0..4 {
            pos[t] = t as i32;
            pos[4 + t] = t as i32;
            pos[8 + t] = t as i32;
        }
        let pos = Array::from_slice(&pos, &[3, 1, 4]);
        let embeds = model.model.embed(&ids).unwrap();
        let logits = model.forward_all_logits_from_embeds(&embeds, &pos, &mut cache).unwrap();
        assert_eq!(logits.shape(), &[1, 4, cfg.vocab_size]);
    }

    #[test]
    fn forward_last_logits_shape() {
        let cfg = tiny_config();
        let mut model = build_with_random_weights(&cfg).unwrap();
        let mut cache = build_cache(cfg.num_hidden_layers as usize);
        let ids = Array::from_slice(&[5_i32, 7, 11], &[1, 3]);
        let mut pos = vec![0_i32; 3 * 3];
        for t in 0..3 {
            pos[t] = t as i32;
            pos[3 + t] = t as i32;
            pos[6 + t] = t as i32;
        }
        let pos = Array::from_slice(&pos, &[3, 1, 3]);
        let embeds = model.model.embed(&ids).unwrap();
        let last = model.forward_last_logits_from_embeds(&embeds, &pos, &mut cache).unwrap();
        assert_eq!(last.shape(), &[1, cfg.vocab_size]);
    }
}
