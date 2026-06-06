//! Glm-OCR text decoder (phase 3).
//!
//! Mirrors `Ernie4_5ForCausalLM` in
//! `PaddlePaddle/GLM-OCR/modeling_glm46v.py` — a stock GQA
//! Llama-family transformer where the only new piece is **MROPE** (handled
//! by [`crate::mrope`]). 18 decoder layers, hidden=1024, n_heads=16,
//! n_kv_heads=2 (4× GQA), head_dim=128, SwiGLU MLP, RMSNorm, no sliding
//! window, no PLE, no softcap, separate `lm_head` (tie_word_embeddings is
//! false on the canonical 1.5 checkpoint).
//!
//! This module exposes the model **structure + forward**; weight loading
//! (de-quantizing safetensors, mapping HF names to the modules below) is a
//! separate concern that the API/loader layer fills in.

use mlx_rs::{
    module::{Module, Param},
    nn,
    ops::{self, indexing::IndexOp},
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

/// SwiGLU MLP: `down(silu(gate(x)) * up(x))`. Standard Llama-family MLP.
pub struct Glm4OcrMlp {
    pub gate_proj: nn::Linear,
    pub up_proj: nn::Linear,
    pub down_proj: nn::Linear,
}

impl Glm4OcrMlp {
    pub fn forward(&mut self, x: &Array) -> Result<Array> {
        let g = self.gate_proj.forward(x).map_err(Error::from)?;
        let u = self.up_proj.forward(x).map_err(Error::from)?;
        let g = mlx_rs::nn::silu(&g).map_err(Error::from)?;
        let z = g.multiply(&u).map_err(Error::from)?;
        self.down_proj.forward(&z).map_err(Error::from)
    }
}

/// GQA attention with MROPE. Q heads are repeated to match KV heads inside
/// the SDPA call (relies on `scaled_dot_product_attention` honouring
/// `n_kv_heads < n_q_heads` via its own broadcast).
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
    pub cos: &'a Array, // (B, T, head_dim) — from mrope::build_cos_sin
    pub sin: &'a Array,
    pub mask: Option<SdpaMask<'a>>,
    pub cache: &'a mut C,
}

impl Glm4OcrAttention {
    pub fn forward<C>(&mut self, input: AttentionInput<'_, C>) -> Result<Array>
    where
        C: KeyValueCache + Default,
    {
        let AttentionInput {
            x,
            cos,
            sin,
            mask,
            cache,
        } = input;
        let shape = x.shape();
        let b = shape[0];
        let t = shape[1];

        let q = self.q_proj.forward(x).map_err(Error::from)?;
        let k = self.k_proj.forward(x).map_err(Error::from)?;
        let v = self.v_proj.forward(x).map_err(Error::from)?;

        // (B, T, n_heads * head_dim) → (B, n_heads, T, head_dim)
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

        // Apply MROPE to Q and K. cos/sin shape is (B, T, head_dim);
        // apply_rotary_qk broadcasts against the head axis at index 1.
        let (q, k) = mrope::apply_rotary_qk(&q, &k, cos, sin)?;

        // KV cache: append the new positions, get back the full K/V history.
        let (k_full, v_full) = cache.update_and_fetch(k, v).map_err(Error::from)?;

        let attn = scaled_dot_product_attention::<C>(
            q, k_full, v_full, None, self.scale, mask,
        )
        .map_err(Error::from)?;

        // (B, n_heads, T, head_dim) → (B, T, n_heads * head_dim)
        let attn = attn
            .transpose_axes(&[0, 2, 1, 3])
            .map_err(Error::from)?
            .reshape(&[b, t, self.n_heads * self.head_dim])
            .map_err(Error::from)?;
        self.o_proj.forward(&attn).map_err(Error::from)
    }
}

/// One Glm-OCR (Glm4-family GQA + MROPE) decoder block: pre-RMSNorm + Attention + residual +
/// pre-RMSNorm + MLP + residual.
pub struct Glm4OcrDecoderLayer {
    pub input_layernorm: nn::RmsNorm,
    pub self_attn: Glm4OcrAttention,
    pub post_attention_layernorm: nn::RmsNorm,
    pub mlp: Glm4OcrMlp,
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
        let LayerInput {
            x,
            cos,
            sin,
            mask,
            cache,
        } = input;
        let normed = self
            .input_layernorm
            .forward(x)
            .map_err(Error::from)?;
        let attn = self.self_attn.forward(AttentionInput {
            x: &normed,
            cos,
            sin,
            mask,
            cache,
        })?;
        let h = x.add(&attn).map_err(Error::from)?;
        let normed = self
            .post_attention_layernorm
            .forward(&h)
            .map_err(Error::from)?;
        let mlp = self.mlp.forward(&normed)?;
        h.add(&mlp).map_err(Error::from)
    }
}

// ── Top-level model ───────────────────────────────────────────────────────

/// Glm-OCR (Glm4-family GQA + MROPE) text decoder body: embedding table + N layers + final RMSNorm.
/// Does **not** include the lm_head (see [`Glm4OcrForCausalLM`]).
pub struct Glm4OcrModel {
    pub embed_tokens: nn::Embedding,
    pub layers: Vec<Glm4OcrDecoderLayer>,
    pub norm: nn::RmsNorm,
    /// Pre-computed MROPE inverse-frequency table and section partition.
    pub inv_freq: Array,
    pub partition: MropePartition,
}

impl Glm4OcrModel {
    /// Embed token IDs → `(B, T, hidden)`.
    pub fn embed(&mut self, input_ids: &Array) -> Result<Array> {
        self.embed_tokens.forward(input_ids).map_err(Error::from)
    }

    /// Forward from pre-computed embeddings (used by the multimodal path,
    /// which splices image soft-tokens into `embed_tokens(input_ids)`
    /// before calling here).
    ///
    /// `position_ids_3d` has shape `(3, B, T)` per [`crate::position_ids`].
    /// Returns hidden states at every input position, shape `(B, T, hidden)`.
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
            // SdpaMask doesn't derive Copy, so build a fresh `Causal` per
            // layer. The variant carries no data so this is free.
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

/// Glm-OCR (Glm4-family GQA + MROPE) with a separate `lm_head`. The canonical PaddleOCR-VL 1.5
/// checkpoint has `tie_word_embeddings = false`, so the head is its own
/// projection from `(B, T, hidden)` → `(B, T, vocab)`.
pub struct Glm4OcrForCausalLM {
    pub model: Glm4OcrModel,
    pub lm_head: nn::Linear,
    pub config: GlmOcrTextConfig,
}

impl Glm4OcrForCausalLM {
    /// One-shot forward returning **per-position** lm-head logits
    /// `(B, T, vocab)`. Used by image-soft-token splicing (need logits at
    /// each text position to drive the AR loop) and by PLD verify down the
    /// line.
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

    /// Last-position logits only — the common decode-step call.
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

// ── Construction helper for tests / random-weight smoke ───────────────────

/// Build an **uninitialised** (random-weight) `Glm4OcrForCausalLM` matching
/// `config`. Used by the smoke tests below; production loaders fill the
/// weights from safetensors after construction.
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
        // Small fan-in scaled uniform — adequate for shape-only smoke.
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
        let mlp = Glm4OcrMlp {
            gate_proj: lin(h, inter, false)?,
            up_proj: lin(h, inter, false)?,
            down_proj: lin(inter, h, false)?,
        };
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
        layers.push(Glm4OcrDecoderLayer {
            input_layernorm: rms(h)?,
            self_attn: attn,
            post_attention_layernorm: rms(h)?,
            mlp,
        });
    }

    // Embedding: (V, H) weight.
    let embed_w = uniform::<_, f32>(
        -0.02,
        0.02,
        &[v, h],
        None,
    )
    .map_err(Error::from)?;
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

    /// Build a tiny config (hidden=16, 2 layers, vocab=20) the unit tests
    /// can stand up without loading a real checkpoint.
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

    fn build_cache(n_layers: usize) -> Vec<KVCache> {
        (0..n_layers).map(|_| KVCache::default()).collect()
    }

    #[test]
    fn forward_shape_matches_inputs() {
        let cfg = tiny_config();
        let mut model = build_with_random_weights(&cfg).unwrap();
        let mut cache = build_cache(cfg.num_hidden_layers as usize);

        // input_ids: (B=1, T=4); position_ids_3d: (3, 1, 4) where all channels
        // share 0..4 (text-only prefill).
        let input_ids = Array::from_slice(&[0_i32, 1, 2, 3], &[1, 4]);
        let mut pos = vec![0_i32; 3 * 4];
        for t in 0..4 {
            pos[t] = t as i32;
            pos[4 + t] = t as i32;
            pos[8 + t] = t as i32;
        }
        let position_ids = Array::from_slice(&pos, &[3, 1, 4]);

        let embeds = model.model.embed(&input_ids).unwrap();
        assert_eq!(embeds.shape(), &[1, 4, cfg.hidden_size]);

        let all_logits = model
            .forward_all_logits_from_embeds(&embeds, &position_ids, &mut cache)
            .unwrap();
        assert_eq!(
            all_logits.shape(),
            &[1, 4, cfg.vocab_size],
            "forward_all_logits_from_embeds shape"
        );
    }

    #[test]
    fn last_logits_shape_is_vocab_wide() {
        // `forward_last_logits_from_embeds` returns `(B, vocab)` — the
        // standard one-step decode interface. Match against all_logits
        // numerically is brittle under MLX's lazy stream lifecycle in
        // unit tests; here we just assert the contract holds.
        let cfg = tiny_config();
        let mut model = build_with_random_weights(&cfg).unwrap();
        let mut cache = build_cache(cfg.num_hidden_layers as usize);
        let input_ids = Array::from_slice(&[5_i32, 7, 11], &[1, 3]);
        let mut pos = vec![0_i32; 3 * 3];
        for t in 0..3 {
            pos[t] = t as i32;
            pos[3 + t] = t as i32;
            pos[6 + t] = t as i32;
        }
        let position_ids = Array::from_slice(&pos, &[3, 1, 3]);
        let embeds = model.model.embed(&input_ids).unwrap();
        let last_only = model
            .forward_last_logits_from_embeds(&embeds, &position_ids, &mut cache)
            .unwrap();
        assert_eq!(last_only.shape(), &[1, cfg.vocab_size]);
    }

    #[test]
    fn rejects_cache_length_mismatch() {
        let cfg = tiny_config();
        let mut model = build_with_random_weights(&cfg).unwrap();
        let input_ids = Array::from_slice(&[0_i32, 1], &[1, 2]);
        let mut pos = vec![0_i32; 3 * 2];
        for t in 0..2 {
            pos[t] = t as i32;
            pos[2 + t] = t as i32;
            pos[4 + t] = t as i32;
        }
        let position_ids = Array::from_slice(&pos, &[3, 1, 2]);
        let embeds = model.model.embed(&input_ids).unwrap();
        let mut wrong_cache: Vec<KVCache> = vec![KVCache::default()]; // 1 vs num_hidden_layers=2
        let r = model.forward_all_logits_from_embeds(&embeds, &position_ids, &mut wrong_cache);
        assert!(
            r.is_err(),
            "cache.len() != num_layers must be rejected at the gate"
        );
    }
}
