use std::{collections::HashMap, path::Path};

use mlx_rs::{
    array,
    error::Exception,
    module::{Module, Param},
    nn,
    ops::{arange, clip, concatenate_axis, exp},
    quantization::MaybeQuantized,
    Array, Dtype,
};
use mlx_rs_core::fused_swiglu;
use serde::Deserialize;

use crate::cache::ProjectedContextCache;
use crate::engine::gqa_sdpa::grouped_gqa_sdpa;

#[derive(Debug, Clone, Deserialize)]
pub struct DFlashDraftModelArgs {
    #[serde(default)]
    pub architectures: Vec<String>,
    #[serde(default)]
    pub model_type: String,
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub head_dim: i32,
    pub intermediate_size: i32,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    #[serde(default)]
    pub block_size: Option<usize>,
    #[serde(default)]
    pub mask_token_id: Option<u32>,
    #[serde(default)]
    pub target_layer_ids: Option<Vec<usize>>,
    #[serde(default)]
    pub dflash_config: Option<serde_json::Value>,
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,
    /// Per-layer attention type — `"sliding_attention"` or `"full_attention"`.
    /// Defaults to all-full when missing (matches the 35B-A3B-DFlash draft).
    /// The 27B-DFlash draft uses sliding attention in 4 of 5 layers.
    #[serde(default)]
    pub layer_types: Vec<String>,
    /// Sliding-window size in tokens. Only applied to layers where
    /// `layer_types[layer_idx] == "sliding_attention"`.
    #[serde(default)]
    pub sliding_window: Option<i32>,
}

impl DFlashDraftModelArgs {
    fn dflash_value(&self, key: &str) -> Option<&serde_json::Value> {
        self.dflash_config.as_ref()?.get(key)
    }

    pub fn block_size(&self) -> usize {
        self.block_size
            .or_else(|| {
                self.dflash_value("block_size")
                    .and_then(|v| v.as_u64().map(|x| x as usize))
            })
            .expect("DFlashDraftModelArgs missing dflash_config.block_size")
    }

    pub fn mask_token_id(&self) -> u32 {
        self.mask_token_id
            .or_else(|| {
                self.dflash_value("mask_token_id")
                    .and_then(|v| v.as_u64().map(|x| x as u32))
            })
            .expect("DFlashDraftModelArgs missing dflash_config.mask_token_id")
    }

    pub fn target_layer_ids(&self) -> Vec<usize> {
        self.target_layer_ids.clone().or_else(|| {
            self.dflash_value("target_layer_ids").and_then(|v| {
                v.as_array().map(|ids| {
                    ids.iter()
                        .filter_map(|id| id.as_u64().map(|x| x as usize))
                        .collect()
                })
            })
        })
        .unwrap_or_default()
    }
}

/// RoPE with optional YaRN (Yet another RoPE extensioN) frequency scaling.
///
/// Precomputes per-frequency scaling so that `fast::rope` can be called with
/// custom `freqs` directly, extending the effective context window to
/// `factor * original_max_position_embeddings` without retraining.
#[derive(Debug, Clone)]
pub struct YarnRope {
    /// Precomputed frequencies, shape [dims/2]. For plain RoPE these are
    /// `1/base^(i/half_dims)`; for YaRN they are additionally scaled by the
    /// smooth interpolation factor.
    freqs: Array,
    /// Position scale (mscale). Plain RoPE: 1.0. YaRN: ~1.042 for factor=64.
    scale: f32,
    dims: i32,
}

impl YarnRope {
    /// Build YaRN-scaled frequencies from the `rope_scaling` config block.
    ///
    /// `fast::rope` with custom `freqs` computes `theta_j = position / freqs[j]`, so we
    /// must pass **periods** (`base^(2j/dims)`, growing from 1) not angular frequencies
    /// (`1/base^(2j/dims)`, shrinking).  The YaRN blending scales low-frequency dims'
    /// periods by `factor` (slower rotation = longer period).  The mscale compensation
    /// is applied by pre-multiplying the input `x` in `forward`, not via the `scale`
    /// parameter of `fast::rope` (which would scale rotation angles, not input magnitudes).
    pub fn new_yarn(
        dims: i32,
        base: f32,
        factor: f32,
        beta_fast: f32,
        beta_slow: f32,
        orig_max_pe: i32,
    ) -> Result<Self, Exception> {
        let half_dims = dims / 2;
        let log_base = (base as f64).ln();

        // periods[j] = base^(2j/dims) = exp(+ln(base) * j / half_dims)
        // fast::rope divides position by freqs[j] to get angle:
        //   theta_j = position / periods[j] = position * base^(-2j/dims)
        // which is the standard RoPE angular frequency.
        let indices = arange::<_, f32>(0, half_dims, None)?;
        let freq_scale = (log_base as f32) / half_dims as f32;
        let periods = exp(&indices.multiply(array!(freq_scale))?)?;

        // YaRN boundary indices.
        // Dim j is "high-frequency" (fast rotation) when j < low — keep period unchanged.
        // Dim j is "low-frequency"  (slow rotation) when j > high — scale period by factor
        // (slower rotation = longer period).
        let low = ((dims as f64)
            * ((orig_max_pe as f64) / (2.0 * std::f64::consts::PI * beta_fast as f64)).ln()
            / (2.0 * log_base))
            .floor() as i32;
        let high = ((dims as f64)
            * ((orig_max_pe as f64) / (2.0 * std::f64::consts::PI * beta_slow as f64)).ln()
            / (2.0 * log_base))
            .ceil() as i32;
        let denom = (high - low).max(1) as f32;

        // ramp[j] = clip((j - low) / denom, 0, 1)
        //   ramp=0 → high-freq dim (period unchanged)
        //   ramp=1 → low-freq dim  (period × factor)
        let ramp_pre = indices
            .subtract(array!(low as f32))?
            .multiply(array!(1.0_f32 / denom))?;
        let ramp = clip(&ramp_pre, (array!(0.0_f32), array!(1.0_f32)))?;
        let one_minus_ramp = array!(1.0_f32).subtract(&ramp)?;

        // Harmonic-blended periods matching the Python YarnRoPE formula:
        //   freqs[j] = factor * periods[j] / (factor * (1-ramp[j]) + ramp[j])
        //
        // ramp=0 (high-freq): freqs = factor*period/factor = period    (unchanged)
        // ramp=1 (low-freq):  freqs = factor*period/1     = factor*period (extended)
        let freqs = array!(factor)
            .multiply(&periods)?
            .divide(&array!(factor).multiply(&one_minus_ramp)?.add(&ramp)?)?;

        // mscale = 0.1 * ln(factor) + 1.0  (YaRN attention scale compensation).
        // Applied to input x in forward(), NOT as the scale arg to fast::rope.
        let scale = if factor <= 1.0 {
            1.0_f32
        } else {
            0.1_f32 * factor.ln() + 1.0
        };

        Ok(YarnRope { freqs, scale, dims })
    }

    /// Build plain RoPE frequencies (no YaRN scaling). Used as fallback when
    /// no `rope_scaling` config is present.
    pub fn new_plain(dims: i32, base: f32) -> Result<Self, Exception> {
        let half_dims = dims / 2;
        let log_base = (base as f64).ln();
        let indices = arange::<_, f32>(0, half_dims, None)?;
        // periods = base^(2j/dims); fast::rope divides position by these → standard RoPE
        let freq_scale = (log_base as f32) / half_dims as f32;
        let freqs = exp(&indices.multiply(array!(freq_scale))?)?;
        Ok(YarnRope { freqs, scale: 1.0, dims })
    }

    fn scaled_input(&self, x: &Array) -> Result<Array, Exception> {
        let x = x.contiguous()?;
        if (self.scale - 1.0).abs() > 1e-6 {
            x.multiply(array!(self.scale))?.as_dtype(x.dtype())
        } else {
            Ok(x)
        }
    }

    pub fn forward(&self, x: &Array, offset: i32) -> Result<Array, Exception> {
        // Pre-multiply input by mscale (YaRN attention compensation) before rotation.
        // Using scale=1.0 in fast::rope keeps rotation angles correct; mscale only
        // affects input magnitudes (not rotation speed).
        let x_in = self.scaled_input(x)?;
        mlx_rs::fast::rope(&x_in, self.dims, false, None::<f32>, 1.0, offset, Some(&self.freqs))
    }
}

fn build_rope(args: &DFlashDraftModelArgs) -> Result<YarnRope, Exception> {
    let rope_type = args
        .rope_scaling
        .as_ref()
        .and_then(|rs| rs.get("type").or_else(|| rs.get("rope_type")))
        .and_then(|v| v.as_str())
        .unwrap_or("default");

    if rope_type == "yarn" {
        let rs = args.rope_scaling.as_ref().unwrap();
        let factor = rs.get("factor").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
        let beta_fast = rs.get("beta_fast").and_then(|v| v.as_f64()).unwrap_or(32.0) as f32;
        let beta_slow = rs.get("beta_slow").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
        let orig_max_pe = rs
            .get("original_max_position_embeddings")
            .and_then(|v| v.as_i64())
            .unwrap_or(4096) as i32;
        YarnRope::new_yarn(args.head_dim, args.rope_theta as f32, factor, beta_fast, beta_slow, orig_max_pe)
    } else {
        YarnRope::new_plain(args.head_dim, args.rope_theta as f32)
    }
}


#[derive(Debug, Clone)]
pub struct DFlashDraftMlp {
    gate_proj: MaybeQuantized<nn::Linear>,
    up_proj: MaybeQuantized<nn::Linear>,
    down_proj: MaybeQuantized<nn::Linear>,
}

impl DFlashDraftMlp {
    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let activated = fused_swiglu(&up, &gate)?;
        self.down_proj.forward(&activated)
    }
}

#[derive(Debug, Clone)]
pub struct DFlashDraftAttention {
    q_proj: MaybeQuantized<nn::Linear>,
    k_proj: MaybeQuantized<nn::Linear>,
    v_proj: MaybeQuantized<nn::Linear>,
    o_proj: MaybeQuantized<nn::Linear>,
    q_norm: nn::RmsNorm,
    k_norm: nn::RmsNorm,
    rope: YarnRope,
    n_heads: i32,
    n_kv_heads: i32,
    head_dim: i32,
    scale: f32,
    /// Sliding window size for this layer, or None for full attention.
    /// Matches Python's `Qwen3_5.DFlashAttention.sliding_window`: when set,
    /// the noise query at position q attends only to keys k where
    /// `q >= k && q < k + sliding_window` (causal + windowed).
    sliding_window: Option<i32>,
}

impl DFlashDraftAttention {
    /// Sliding-window + causal mask, matching Python's `_attention_mask`.
    /// Returns `None` for full-attention layers (no mask = full bidirectional).
    /// Otherwise returns a `[block_len, total_key_len]` boolean array where
    /// `mask[q, k] = (qpos >= kpos) && (qpos < kpos + sliding_window)`.
    /// Cached keys live at positions `[0..cache_offset)`; noise keys at
    /// `[cache_offset..cache_offset+block_len)`.
    fn build_swa_mask(
        &self,
        block_len: i32,
        cache_offset: i32,
    ) -> Result<Option<Array>, Exception> {
        let Some(window) = self.sliding_window else {
            return Ok(None);
        };
        let total_key_len = cache_offset + block_len;
        let query_positions = mlx_rs::ops::arange::<_, i32>(
            cache_offset,
            cache_offset + block_len,
            None,
        )?
        .reshape(&[block_len, 1])?;
        let key_positions = mlx_rs::ops::arange::<_, i32>(0, total_key_len, None)?
            .reshape(&[1, total_key_len])?;
        let causal = query_positions.ge(&key_positions)?;
        let within = query_positions.lt(&key_positions.add(array!(window))?)?;
        Ok(Some(causal.logical_and(&within)?))
    }

    pub fn debug_tensors(
        &mut self,
        noise: &Array,
        target: &Array,
        ctx_offset: usize,
    ) -> Result<HashMap<String, Array>, Exception> {
        let b = noise.shape()[0];
        let q_len = noise.shape()[1];
        let ctx_len = target.shape()[1];

        let queries = self
            .q_proj
            .forward(noise)?
            .reshape(&[b, q_len, self.n_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let queries_norm = self.q_norm.forward(&queries)?;

        let ctx_keys = self
            .k_proj
            .forward(target)?
            .reshape(&[b, ctx_len, self.n_kv_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let ctx_keys_norm = self.k_norm.forward(&ctx_keys)?;
        let ctx_values = self
            .v_proj
            .forward(target)?
            .reshape(&[b, ctx_len, self.n_kv_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let noise_keys = self
            .k_proj
            .forward(noise)?
            .reshape(&[b, q_len, self.n_kv_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let noise_keys_norm = self.k_norm.forward(&noise_keys)?;
        let noise_values = self
            .v_proj
            .forward(noise)?
            .reshape(&[b, q_len, self.n_kv_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let queries_rope_input = self.rope.scaled_input(&queries_norm)?;
        let ctx_keys_rope_input = self.rope.scaled_input(&ctx_keys_norm)?;
        let noise_keys_rope_input = self.rope.scaled_input(&noise_keys_norm)?;
        let q_t = self.rope.forward(&queries_norm, ctx_offset as i32)?;
        let ctx_keys_t = self.rope.forward(&ctx_keys_norm, 0)?;
        let noise_keys_t = self.rope.forward(&noise_keys_norm, ctx_offset as i32)?;

        let keys = concatenate_axis(&[&ctx_keys_t, &noise_keys_t], 2)?;
        let values = concatenate_axis(&[&ctx_values, &noise_values], 2)?;
        let attn_heads = grouped_gqa_sdpa(&q_t, &keys, &values, self.scale, None)?;
        let attn_merged = attn_heads
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, q_len, self.n_heads * self.head_dim])?;
        let o_proj_output = self.o_proj.forward(&attn_merged)?;

        Ok(HashMap::from([
            ("queries_norm".to_string(), queries_norm),
            ("ctx_keys_norm".to_string(), ctx_keys_norm),
            ("ctx_values".to_string(), ctx_values),
            ("noise_keys_norm".to_string(), noise_keys_norm),
            ("noise_values".to_string(), noise_values),
            ("queries_rope_input".to_string(), queries_rope_input),
            ("ctx_keys_rope_input".to_string(), ctx_keys_rope_input),
            ("noise_keys_rope_input".to_string(), noise_keys_rope_input),
            ("rope_freqs".to_string(), self.rope.freqs.clone()),
            ("q_rope".to_string(), q_t),
            ("ctx_keys_rope".to_string(), ctx_keys_t),
            ("noise_keys_rope".to_string(), noise_keys_t),
            ("keys".to_string(), keys),
            ("values".to_string(), values),
            ("attn_heads".to_string(), attn_heads),
            ("attn_merged".to_string(), attn_merged),
            ("o_proj_output".to_string(), o_proj_output),
        ]))
    }

    pub fn forward(
        &mut self,
        noise: &Array,
        target: &Array,
        ctx_offset: usize,
    ) -> Result<Array, Exception> {
        let b = noise.shape()[0];
        let q_len = noise.shape()[1];
        let ctx_len = target.shape()[1];

        // Q from draft hidden states (noise).
        let queries = self
            .q_proj
            .forward(noise)?
            .reshape(&[b, q_len, self.n_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let queries = self.q_norm.forward(&queries)?;

        // Context K/V: projected from target hidden context.
        let ctx_keys = self
            .k_proj
            .forward(target)?
            .reshape(&[b, ctx_len, self.n_kv_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let ctx_keys = self.k_norm.forward(&ctx_keys)?;
        let ctx_values = self
            .v_proj
            .forward(target)?
            .reshape(&[b, ctx_len, self.n_kv_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        // Noise K/V: projected from the SAME noise hidden states as Q.
        // This enables bidirectional self-attention among draft positions — each
        // draft token sees all others in the block (non-autoregressive generation).
        let noise_keys = self
            .k_proj
            .forward(noise)?
            .reshape(&[b, q_len, self.n_kv_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let noise_keys = self.k_norm.forward(&noise_keys)?;
        let noise_values = self
            .v_proj
            .forward(noise)?
            .reshape(&[b, q_len, self.n_kv_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        // RoPE: Q and noise K both at draft positions (ctx_offset..ctx_offset+block_len).
        // Context K at positions 0..ctx_len-1 (no offset).
        let q_t = self.rope.forward(&queries, ctx_offset as i32)?;
        let ctx_keys_t = self.rope.forward(&ctx_keys, 0)?;
        let noise_keys_t = self.rope.forward(&noise_keys, ctx_offset as i32)?;

        // K = [context_keys || noise_keys], V = [context_values || noise_values].
        // Draft positions attend bidirectionally within the block; sliding
        // layers additionally need the same SWA mask the cached path
        // (`forward_with_cache`) applies — without it this uncached forward
        // silently ran full attention and diverged from the runtime path
        // (including the parity harnesses built on it).
        let keys = concatenate_axis(&[&ctx_keys_t, &noise_keys_t], 2)?;
        let values = concatenate_axis(&[&ctx_values, &noise_values], 2)?;

        let mask_arr = if self.sliding_window.is_some() {
            // `build_swa_mask` lays keys out as [0..offset) ++ block — that
            // matches only when the draft block directly follows the context.
            if ctx_offset as i32 != ctx_len {
                return Err(Exception::custom(format!(
                    "DFlashDraftAttention::forward: SWA layer requires \
                     ctx_offset ({ctx_offset}) == ctx_len ({ctx_len})"
                )));
            }
            self.build_swa_mask(q_len, ctx_len)?
        } else {
            None
        };
        let mask = mask_arr.as_ref().map(mlx_rs_core::SdpaMask::Array);
        let attn = grouped_gqa_sdpa(&q_t, &keys, &values, self.scale, mask)?;
        let attn = attn
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, q_len, self.n_heads * self.head_dim])?;
        self.o_proj.forward(&attn)
    }

    /// Cached variant of `forward` for cross-attention to a long target
    /// context. The `target_delta` argument is the **new committed
    /// positions only** (post fc+hidden_norm projection); their k/v is
    /// computed once, RoPE'd at the right offset, and appended to `cache`.
    /// Attention uses `concat(cache.keys, noise_keys)` so prior committed
    /// positions are reused without re-projection.
    pub fn forward_with_cache(
        &mut self,
        noise: &Array,
        target_delta: &Array,
        cache: &mut ProjectedContextCache,
    ) -> Result<Array, Exception> {
        let b = noise.shape()[0];
        let q_len = noise.shape()[1];
        let delta_len = target_delta.shape()[1];

        // Q from noise.
        let queries = self
            .q_proj
            .forward(noise)?
            .reshape(&[b, q_len, self.n_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let queries = self.q_norm.forward(&queries)?;

        // Append the delta to the cache (if non-empty). RoPE offset for the
        // delta is the cache's current offset (= number of positions cached
        // before this append).
        if delta_len > 0 {
            let cache_offset = cache.offset() as i32;
            let new_keys = self
                .k_proj
                .forward(target_delta)?
                .reshape(&[b, delta_len, self.n_kv_heads, self.head_dim])?
                .transpose_axes(&[0, 2, 1, 3])?;
            let new_keys = self.k_norm.forward(&new_keys)?;
            let new_keys = self.rope.forward(&new_keys, cache_offset)?;
            let new_values = self
                .v_proj
                .forward(target_delta)?
                .reshape(&[b, delta_len, self.n_kv_heads, self.head_dim])?
                .transpose_axes(&[0, 2, 1, 3])?;
            cache.append(new_keys, new_values, delta_len as usize)?;
        }

        let ctx_offset = cache.offset() as i32;

        // Noise K/V.
        let noise_keys = self
            .k_proj
            .forward(noise)?
            .reshape(&[b, q_len, self.n_kv_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let noise_keys = self.k_norm.forward(&noise_keys)?;
        let noise_values = self
            .v_proj
            .forward(noise)?
            .reshape(&[b, q_len, self.n_kv_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let q_t = self.rope.forward(&queries, ctx_offset)?;
        let noise_keys_t = self.rope.forward(&noise_keys, ctx_offset)?;

        let (keys, values) = match (cache.keys(), cache.values()) {
            (Some(ck), Some(cv)) => (
                concatenate_axis(&[ck, &noise_keys_t], 2)?,
                concatenate_axis(&[cv, &noise_values], 2)?,
            ),
            _ => (noise_keys_t, noise_values),
        };

        let mask_arr = self.build_swa_mask(q_len, ctx_offset)?;
        let mask = mask_arr.as_ref().map(mlx_rs_core::SdpaMask::Array);
        let attn = grouped_gqa_sdpa(&q_t, &keys, &values, self.scale, mask)?;
        let attn = attn
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, q_len, self.n_heads * self.head_dim])?;
        self.o_proj.forward(&attn)
    }
}

#[derive(Debug, Clone)]
pub struct DFlashDraftLayer {
    pub input_layernorm: nn::RmsNorm,
    pub self_attn: DFlashDraftAttention,
    pub post_attention_layernorm: nn::RmsNorm,
    pub mlp: DFlashDraftMlp,
}

impl DFlashDraftLayer {
    pub fn forward(
        &mut self,
        hidden_states: &Array,
        target_hidden: &Array,
        ctx_offset: usize,
    ) -> Result<Array, Exception> {
        let residual = hidden_states.clone();
        let h = self.input_layernorm.forward(hidden_states)?;
        let h = self.self_attn.forward(&h, target_hidden, ctx_offset)?;
        let h = h.add(residual)?;
        let residual2 = h.clone();
        let h = self.post_attention_layernorm.forward(&h)?;
        let h = self.mlp.forward(&h)?;
        h.add(residual2)
    }

    pub fn forward_with_cache(
        &mut self,
        hidden_states: &Array,
        target_delta: &Array,
        cache: &mut ProjectedContextCache,
    ) -> Result<Array, Exception> {
        let residual = hidden_states.clone();
        let h = self.input_layernorm.forward(hidden_states)?;
        let h = self.self_attn.forward_with_cache(&h, target_delta, cache)?;
        let h = h.add(residual)?;
        let residual2 = h.clone();
        let h = self.post_attention_layernorm.forward(&h)?;
        let h = self.mlp.forward(&h)?;
        h.add(residual2)
    }
}

#[derive(Debug, Clone)]
pub struct DFlashDraftModel {
    pub args: DFlashDraftModelArgs,
    pub layers: Vec<DFlashDraftLayer>,
    pub fc: MaybeQuantized<nn::Linear>,
    pub hidden_norm: nn::RmsNorm,
    pub norm: nn::RmsNorm,
}

impl DFlashDraftModel {
    pub fn project_target_hidden(&mut self, target_hidden: &Array) -> Result<Array, Exception> {
        // Target hidden captures may be Float32 depending on the target model's
        // intermediate dtype. Cast to Bfloat16 to match the draft model's weight dtype.
        let target_hidden = target_hidden.as_dtype(Dtype::Bfloat16)?;
        let target_h_pre = self.fc.forward(&target_hidden)?;
        self.hidden_norm.forward(&target_h_pre)
    }

    pub fn forward_projected_context(
        &mut self,
        noise_emb: &Array,
        projected_target_hidden: &Array,
        ctx_offset: usize,
    ) -> Result<Array, Exception> {
        let mut h = noise_emb.clone();
        for layer in &mut self.layers {
            h = layer.forward(&h, projected_target_hidden, ctx_offset)?;
        }
        self.norm.forward(&h)
    }

    pub fn forward(
        &mut self,
        noise_emb: &Array,
        target_hidden: &Array,
        ctx_offset: usize,
    ) -> Result<Array, Exception> {
        let target_h = self.project_target_hidden(target_hidden)?;
        self.forward_projected_context(noise_emb, &target_h, ctx_offset)
    }

    /// Cached forward: `raw_target_delta` is the **new** raw target hidden
    /// (positions not yet cached). It is projected through `fc + hidden_norm`
    /// here and then each attention layer extends its own `ProjectedContextCache`.
    /// `caches.len()` must equal `self.layers.len()`.
    pub fn forward_with_caches(
        &mut self,
        noise_emb: &Array,
        raw_target_delta: &Array,
        caches: &mut [ProjectedContextCache],
    ) -> Result<Array, Exception> {
        if caches.len() != self.layers.len() {
            return Err(Exception::custom(format!(
                "forward_with_caches: expected {} caches, got {}",
                self.layers.len(),
                caches.len()
            )));
        }
        let delta_len = raw_target_delta.shape()[1];
        let projected_delta = if delta_len > 0 {
            self.project_target_hidden(raw_target_delta)?
        } else {
            raw_target_delta.as_dtype(Dtype::Bfloat16)?
        };
        let mut h = noise_emb.clone();
        for (layer, cache) in self.layers.iter_mut().zip(caches.iter_mut()) {
            h = layer.forward_with_cache(&h, &projected_delta, cache)?;
        }
        self.norm.forward(&h)
    }

    pub fn debug_first_layer(
        &mut self,
        noise_emb: &Array,
        projected_target_hidden: &Array,
        ctx_offset: usize,
    ) -> Result<HashMap<String, Array>, Exception> {
        let layer0 = self
            .layers
            .get_mut(0)
            .ok_or_else(|| Exception::custom("draft model has no layers"))?;
        let residual = noise_emb.clone();
        let input_norm = layer0.input_layernorm.forward(noise_emb)?;
        let attn = layer0
            .self_attn
            .debug_tensors(&input_norm, projected_target_hidden, ctx_offset)?;
        let attn_out = attn
            .get("o_proj_output")
            .ok_or_else(|| Exception::custom("attention debug missing o_proj_output"))?
            .clone();
        let post_attn = attn_out.add(residual)?;
        let post_attn_norm = layer0.post_attention_layernorm.forward(&post_attn)?;
        let gate = layer0.mlp.gate_proj.forward(&post_attn_norm)?;
        let up = layer0.mlp.up_proj.forward(&post_attn_norm)?;
        let activated = fused_swiglu(&up, &gate)?;
        let mlp_out = layer0.mlp.down_proj.forward(&activated)?;
        let layer_output = mlp_out.add(post_attn.clone())?;

        let mut tensors = HashMap::from([
            ("layer0_input_norm".to_string(), input_norm),
            ("layer0_post_attn".to_string(), post_attn),
            ("layer0_post_attn_norm".to_string(), post_attn_norm),
            ("layer0_mlp_gate".to_string(), gate),
            ("layer0_mlp_up".to_string(), up),
            ("layer0_mlp_activated".to_string(), activated),
            ("layer0_mlp_out".to_string(), mlp_out),
            ("layer0_output".to_string(), layer_output),
        ]);
        for (name, value) in attn {
            tensors.insert(format!("layer0_attn_{name}"), value);
        }
        Ok(tensors)
    }

    pub fn load_from_path(path: &Path) -> Result<Self, Exception> {
        let config_path = path.join("config.json");
        let config_text = std::fs::read_to_string(&config_path).map_err(|e| {
            Exception::custom(format!("failed to read {}: {e}", config_path.display()))
        })?;
        let args: DFlashDraftModelArgs = serde_json::from_str(&config_text).map_err(|e| {
            Exception::custom(format!("failed to parse {}: {e}", config_path.display()))
        })?;

        let weights_path = path.join("model.safetensors");
        let weights = Array::load_safetensors(&weights_path).map_err(|e| {
            Exception::custom(format!("failed to load {}: {e}", weights_path.display()))
        })?;

        let rope = build_rope(&args)?;
        let eps = args.rms_norm_eps as f32;
        let scale = (args.head_dim as f32).sqrt().recip();

        // Optional on-load quantization of the draft's linears. The DFlash
        // checkpoints ship BF16 (0.95 GB on 35B-A3B, 3.46 GB on dense 27B)
        // and decode is bandwidth-bound, so every cycle re-reads the full
        // draft — quantizing to 4-bit cuts that ~4x.
        //   DFLASH_QUANT_DRAFT unset/0 → BF16 (checkpoint as-is)
        //   DFLASH_QUANT_DRAFT=4|1|true → 4-bit, group 64 (matches target)
        //   DFLASH_QUANT_DRAFT=8 → 8-bit, group 64
        let quant = draft_quant_from_env();
        if let Some((group_size, bits)) = quant {
            eprintln!("DFlash draft: quantizing linears to {bits}-bit (group {group_size})");
        }

        let mut layers = Vec::with_capacity(args.num_hidden_layers as usize);
        for i in 0..args.num_hidden_layers as usize {
            let prefix = format!("layers.{i}");
            // Per-layer SWA: matches Python's
            //   self.sliding_window = (
            //       config.sliding_window
            //       if config.layer_types[layer_idx] == "sliding_attention"
            //       else None
            //   )
            // Falls back to None (full attention) when layer_types is empty
            // (e.g. 35B-A3B-DFlash draft) or sliding_window is unset.
            let layer_sliding = args
                .layer_types
                .get(i)
                .map(|t| t == "sliding_attention")
                .unwrap_or(false)
                .then_some(args.sliding_window)
                .flatten();
            layers.push(DFlashDraftLayer {
                input_layernorm: rms_norm(&weights, &format!("{prefix}.input_layernorm.weight"), eps)?,
                self_attn: DFlashDraftAttention {
                    q_proj: linear(&weights, &format!("{prefix}.self_attn.q_proj.weight"), quant)?,
                    k_proj: linear(&weights, &format!("{prefix}.self_attn.k_proj.weight"), quant)?,
                    v_proj: linear(&weights, &format!("{prefix}.self_attn.v_proj.weight"), quant)?,
                    o_proj: linear(&weights, &format!("{prefix}.self_attn.o_proj.weight"), quant)?,
                    q_norm: rms_norm(&weights, &format!("{prefix}.self_attn.q_norm.weight"), eps)?,
                    k_norm: rms_norm(&weights, &format!("{prefix}.self_attn.k_norm.weight"), eps)?,
                    rope: rope.clone(),
                    n_heads: args.num_attention_heads,
                    n_kv_heads: args.num_key_value_heads,
                    head_dim: args.head_dim,
                    scale,
                    sliding_window: layer_sliding,
                },
                post_attention_layernorm: rms_norm(
                    &weights,
                    &format!("{prefix}.post_attention_layernorm.weight"),
                    eps,
                )?,
                mlp: DFlashDraftMlp {
                    gate_proj: linear(&weights, &format!("{prefix}.mlp.gate_proj.weight"), quant)?,
                    up_proj: linear(&weights, &format!("{prefix}.mlp.up_proj.weight"), quant)?,
                    down_proj: linear(&weights, &format!("{prefix}.mlp.down_proj.weight"), quant)?,
                },
            });
        }

        Ok(Self {
            fc: linear(&weights, "fc.weight", quant)?,
            hidden_norm: rms_norm(&weights, "hidden_norm.weight", eps)?,
            norm: rms_norm(&weights, "norm.weight", eps)?,
            args,
            layers,
        })
    }
}

/// Parse `DFLASH_QUANT_DRAFT` into `(group_size, bits)`. See `load_from_path`.
fn draft_quant_from_env() -> Option<(i32, i32)> {
    let v = std::env::var("DFLASH_QUANT_DRAFT").ok()?;
    match v.trim() {
        "" | "0" | "false" => None,
        "8" => Some((64, 8)),
        _ => Some((64, 4)),
    }
}

fn get_weight(weights: &HashMap<String, Array>, key: &str) -> Result<Array, Exception> {
    weights
        .get(key)
        .cloned()
        .ok_or_else(|| Exception::custom(format!("missing draft weight: {key}")))
}

fn linear(
    weights: &HashMap<String, Array>,
    key: &str,
    quant: Option<(i32, i32)>,
) -> Result<MaybeQuantized<nn::Linear>, Exception> {
    let dense = nn::Linear {
        weight: Param::new(get_weight(weights, key)?),
        bias: Param::new(None),
    };
    match quant {
        Some((group_size, bits)) => Ok(MaybeQuantized::Quantized(
            nn::QuantizedLinear::try_from_linear(dense, group_size, bits)?,
        )),
        None => Ok(MaybeQuantized::Original(dense)),
    }
}

fn rms_norm(weights: &HashMap<String, Array>, key: &str, eps: f32) -> Result<nn::RmsNorm, Exception> {
    Ok(nn::RmsNorm {
        weight: Param::new(get_weight(weights, key)?),
        eps,
    })
}
