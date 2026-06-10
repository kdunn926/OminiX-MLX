use std::collections::HashMap;

use image::imageops::FilterType;
use mlx_rs::{
    array,
    module::{Module, Param},
    nn,
    ops::{self, indexing::IndexOp},
    quantization::MaybeQuantized,
    transforms::eval,
    Array, Dtype,
};
use mlx_rs_core::error::{Error, Result};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Gemma4VisionRopeParameters {
    pub rope_theta: f32,
    pub rope_type: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Gemma4VisionConfig {
    pub model_type: String,
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub head_dim: i32,
    pub patch_size: i32,
    pub intermediate_size: i32,
    pub rms_norm_eps: f32,
    pub rope_parameters: Gemma4VisionRopeParameters,
    pub position_embedding_size: i32,
    pub pooling_kernel_size: i32,
    pub default_output_length: i32,
    pub standardize: bool,
    #[serde(default)]
    pub use_clipped_linears: bool,
}

fn get_weight(weights: &HashMap<String, Array>, key: &str) -> Result<Array> {
    weights
        .get(key)
        .cloned()
        .ok_or_else(|| Error::WeightNotFound(key.to_string()))
}

fn make_linear(weight: Array) -> nn::Linear {
    nn::Linear {
        weight: Param::new(weight),
        bias: Param::new(None::<Array>),
    }
}

pub struct VisionRmsNorm {
    pub weight: Array,
    pub eps: f32,
}

impl VisionRmsNorm {
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let x_f32 = x.as_dtype(Dtype::Float32)?;
        let variance = x_f32.square()?.mean_axis(-1, true)?;
        let normed = x_f32.multiply(&variance.add(&array!(self.eps))?.rsqrt()?)?;
        let weight = self.weight.as_dtype(Dtype::Float32)?;
        normed.multiply(&weight)?.as_dtype(x.dtype()).map_err(Into::into)
    }
}

pub struct VisionRmsNormNoScale {
    pub eps: f32,
}

impl VisionRmsNormNoScale {
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let x_f32 = x.as_dtype(Dtype::Float32)?;
        let variance = x_f32.square()?.mean_axis(-1, true)?;
        x_f32
            .multiply(&variance.add(&array!(self.eps))?.rsqrt()?)?
            .as_dtype(x.dtype())
            .map_err(Into::into)
    }
}

pub fn rotate_half(x: &Array) -> Result<Array> {
    let half_dim = x.shape()[x.shape().len() - 1] / 2;
    let rotated = match x.shape().len() {
        3 => {
            let x1 = x.index((.., .., ..half_dim));
            let x2 = x.index((.., .., half_dim..));
            ops::concatenate_axis(&[&x2.negative()?, &x1], -1)?
        }
        4 => {
            let x1 = x.index((.., .., .., ..half_dim));
            let x2 = x.index((.., .., .., half_dim..));
            ops::concatenate_axis(&[&x2.negative()?, &x1], -1)?
        }
        ndim => {
            return Err(Error::Model(format!(
                "rotate_half expects 3D or 4D input, got {ndim}D"
            )))
        }
    };
    Ok(rotated)
}

/// Precomputed 2-D RoPE rotation tables: per spatial dim, `(cos, sin)`
/// shaped `[B, L, 1, channels_per_dim]`. The tables depend only on the
/// patch positions and `rope_theta`, so the encoder computes them once per
/// forward and every layer reuses them for both `q` and `k` — instead of
/// rebuilding identical tables (and syncing on `positions`) twice per
/// layer. Built with lazy MLX ops only, so `VisionModel::forward_lazy`
/// stays genuinely eval-free.
pub struct Rope2dTables {
    channels_per_dim: i32,
    dims: [(Array, Array); 2],
}

pub fn compute_rope_2d_tables(
    positions: &Array,
    head_dim: i32,
    rope_theta: f32,
    dtype: mlx_rs::Dtype,
) -> Result<Rope2dTables> {
    let pos_shape = positions.shape();
    if pos_shape.len() != 3 || pos_shape[2] != 2 {
        return Err(Error::Model(format!(
            "2D RoPE expects positions [B, L, 2], got {:?}",
            pos_shape
        )));
    }
    let (b, l) = (pos_shape[0], pos_shape[1]);
    let ndim = 2;
    let channels_per_dim = 2 * (head_dim / (2 * ndim));
    let half_per_dim = channels_per_dim / 2;
    let timescales: Vec<f32> = (0..half_per_dim)
        .map(|i| rope_theta.powf((2.0 / channels_per_dim as f32) * i as f32))
        .collect();
    let timescales = Array::from_slice(&timescales, &[1, 1, half_per_dim]);

    let mut dims = Vec::with_capacity(ndim as usize);
    for d in 0..ndim {
        let pos_d = positions
            .index((.., .., d))
            .as_dtype(mlx_rs::Dtype::Float32)?
            .reshape(&[b, l, 1])?;
        let angles = pos_d.divide(&timescales)?; // [B, L, half]
        let cos = ops::cos(&angles)?;
        let sin = ops::sin(&angles)?;
        // Duplicate halves: [cos, cos] / [sin, sin] → [B, L, 1, channels],
        // matching the rotate_half pairing below.
        let cos = ops::concatenate_axis(&[&cos, &cos], -1)?
            .reshape(&[b, l, 1, channels_per_dim])?
            .as_dtype(dtype)?;
        let sin = ops::concatenate_axis(&[&sin, &sin], -1)?
            .reshape(&[b, l, 1, channels_per_dim])?
            .as_dtype(dtype)?;
        dims.push((cos, sin));
    }
    let d1 = dims.pop().expect("two dims");
    let d0 = dims.pop().expect("two dims");
    Ok(Rope2dTables {
        channels_per_dim,
        dims: [d0, d1],
    })
}

/// Apply precomputed 2-D RoPE tables to `q`/`k` of shape `[B, L, H, D]`.
pub fn apply_rope_2d_tables(x: &Array, tables: &Rope2dTables) -> Result<Array> {
    let shape = x.shape();
    if shape.len() != 4 {
        return Err(Error::Model(format!(
            "2D RoPE expects [B, L, H, D], got {:?}",
            shape
        )));
    }
    let c = tables.channels_per_dim;
    let mut parts = Vec::with_capacity(tables.dims.len());
    for (d, (cos, sin)) in tables.dims.iter().enumerate() {
        let d = d as i32;
        let x_part = x.index((.., .., .., (d * c)..((d + 1) * c)));
        let rotated = rotate_half(&x_part)?;
        parts.push(x_part.multiply(cos)?.add(&rotated.multiply(sin)?)?);
    }
    let refs: Vec<&Array> = parts.iter().collect();
    ops::concatenate_axis(&refs, -1).map_err(Into::into)
}

/// One-shot convenience wrapper: build tables for this call only and apply
/// them. Prefer [`compute_rope_2d_tables`] + [`apply_rope_2d_tables`] when
/// applying RoPE more than once for the same positions (the encoder path).
pub fn apply_multidimensional_rope_2d(
    q: &Array,
    positions: &Array,
    rope_theta: f32,
) -> Result<Array> {
    let shape = q.shape();
    if shape.len() != 4 {
        return Err(Error::Model(format!(
            "2D RoPE expects [B, L, H, D], got {:?}",
            shape
        )));
    }
    let tables = compute_rope_2d_tables(positions, shape[3], rope_theta, q.dtype())?;
    apply_rope_2d_tables(q, &tables)
}

pub struct PatchEmbedder {
    pub hidden_size: i32,
    pub patch_size: i32,
    pub position_embedding_size: i32,
    pub input_proj: nn::Linear,
    pub position_embedding_table: Array,
}

impl PatchEmbedder {
    fn position_embeddings(
        &self,
        patch_positions: &Array,
        padding_positions: &Array,
    ) -> Result<Array> {
        let row_ids = patch_positions.index((.., .., 0));
        let col_ids = patch_positions.index((.., .., 1));
        let row_table = self.position_embedding_table.index((0, .., ..));
        let col_table = self.position_embedding_table.index((1, .., ..));
        eval([&row_ids, &col_ids]).map_err(|e| Error::Model(format!("eval row/col_ids: {e}")))?;
        let row_emb = mlx_rs::ops::indexing::take_axis(&row_table, &row_ids, 0)?;
        let col_emb = mlx_rs::ops::indexing::take_axis(&col_table, &col_ids, 0)?;
        let pos_emb = row_emb.add(&col_emb)?;
        let zeros = ops::zeros_like(&pos_emb)?;
        let result = ops::which(&padding_positions.expand_dims(-1)?, &zeros, &pos_emb)?;
        Ok(result)
    }

    pub fn forward(
        &mut self,
        pixel_values: &Array,
        patch_positions: &Array,
        padding_positions: &Array,
    ) -> Result<Array> {
        let shape = pixel_values.shape();
        let b = shape[0];
        let c = shape[1];
        let h = shape[2];
        let w = shape[3];
        let p = self.patch_size;
        let p_h = h / p;
        let p_w = w / p;

        let patches = pixel_values
            .reshape(&[b, c, p_h, p, p_w, p])?
            .transpose_axes(&[0, 2, 4, 3, 5, 1])?
            .reshape(&[b, p_h * p_w, c * p * p])?;
        let patches = patches.subtract(&array!(0.5))?.multiply(&array!(2.0))?;
        let patches_typed = patches.as_dtype(self.input_proj.weight.dtype())?;
        let hidden = self.input_proj.forward(&patches_typed)?;
        let pos = self.position_embeddings(patch_positions, padding_positions)?;
        hidden.add(&pos).map_err(Into::into)
    }
}

pub struct VisionAttention {
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub rope_theta: f32,
    pub q_proj: nn::Linear,
    pub k_proj: nn::Linear,
    pub v_proj: nn::Linear,
    pub o_proj: nn::Linear,
    pub q_norm: VisionRmsNorm,
    pub k_norm: VisionRmsNorm,
    pub v_norm: VisionRmsNormNoScale,
}

impl VisionAttention {
    pub fn forward(
        &mut self,
        x: &Array,
        rope: &Rope2dTables,
        mask: Option<&Array>,
    ) -> Result<Array> {
        let shape = x.shape();
        let b = shape[0];
        let l = shape[1];
        let hidden = shape[2];

        let mut q = self
            .q_proj
            .forward(x)?
            .reshape(&[b, l, self.num_heads, self.head_dim])?;
        let mut k = self
            .k_proj
            .forward(x)?
            .reshape(&[b, l, self.num_kv_heads, self.head_dim])?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape(&[b, l, self.num_kv_heads, self.head_dim])?;

        q = self.q_norm.forward(&q)?;
        k = self.k_norm.forward(&k)?;
        let v = self.v_norm.forward(&v)?;
        q = apply_rope_2d_tables(&q, rope)?;
        k = apply_rope_2d_tables(&k, rope)?;
        let q = q.transpose_axes(&[0, 2, 1, 3])?;
        let k = k.transpose_axes(&[0, 2, 1, 3])?;
        let v = v.transpose_axes(&[0, 2, 1, 3])?;

        let mut scores = q.matmul(&k.transpose_axes(&[0, 1, 3, 2])?)?;
        if let Some(mask) = mask {
            scores = scores.add(mask)?;
        }
        let attn = mlx_rs::ops::softmax_axis(&scores, -1, None)?;
        let out = attn
            .matmul(&v)?
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, l, hidden])?;
        self.o_proj.forward(&out).map_err(Into::into)
    }
}

pub struct VisionMLP {
    pub gate_proj: nn::Linear,
    pub up_proj: nn::Linear,
    pub down_proj: nn::Linear,
}

impl VisionMLP {
    pub fn forward(&mut self, x: &Array) -> Result<Array> {
        let gate = nn::gelu_approximate(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&gate.multiply(&up)?).map_err(Into::into)
    }
}

pub struct VisionEncoderLayer {
    pub self_attn: VisionAttention,
    pub mlp: VisionMLP,
    pub input_layernorm: VisionRmsNorm,
    pub post_attention_layernorm: VisionRmsNorm,
    pub pre_feedforward_layernorm: VisionRmsNorm,
    pub post_feedforward_layernorm: VisionRmsNorm,
}

impl VisionEncoderLayer {
    pub fn forward(&mut self, x: &Array, rope: &Rope2dTables, mask: Option<&Array>) -> Result<Array> {
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = self.self_attn.forward(&normed, rope, mask)?;
        let attn_out = self.post_attention_layernorm.forward(&attn_out)?;
        let h = x.add(&attn_out)?;
        let normed_h = self.pre_feedforward_layernorm.forward(&h)?;
        let ff_out = self.mlp.forward(&normed_h)?;
        let ff_out = self.post_feedforward_layernorm.forward(&ff_out)?;
        h.add(&ff_out).map_err(Into::into)
    }
}

pub struct VisionEncoder {
    pub layers: Vec<VisionEncoderLayer>,
}

impl VisionEncoder {
    pub fn forward(&mut self, hidden_states: &Array, positions: &Array, mask: &Array) -> Result<Array> {
        let Some(first) = self.layers.first() else {
            return Ok(hidden_states.clone());
        };
        // All layers share rope_theta/head_dim, so the cos/sin tables are
        // computed once here and reused by every layer's q and k.
        let rope = compute_rope_2d_tables(
            positions,
            first.self_attn.head_dim,
            first.self_attn.rope_theta,
            hidden_states.dtype(),
        )?;
        let mut hidden = hidden_states.clone();
        for layer in self.layers.iter_mut() {
            hidden = layer.forward(&hidden, &rope, Some(mask))?;
        }
        Ok(hidden)
    }
}

pub struct VisionPooler {
    pub hidden_size: i32,
    pub default_output_length: i32,
    pub root_hidden_size: f32,
}

impl VisionPooler {
    fn avg_pool_by_positions(
        &self,
        x: &Array,
        patch_positions: &Array,
        length: i32,
    ) -> Result<(Array, Array)> {
        let b = x.shape()[0] as usize;
        let l = x.shape()[1] as usize;
        let length_usize = length as usize;
        let input_seq_len = x.shape()[1];
        let k = ((input_seq_len / length).max(1) as f32).sqrt() as i32;
        let k = k.max(1);
        let k_squared = (k * k) as f32;
        eval(std::slice::from_ref(patch_positions)).map_err(|e| Error::Model(format!("eval patch_positions: {e}")))?;
        let pos = patch_positions.try_as_slice::<i32>().map_err(|e| {
            Error::Model(format!("patch_positions must be contiguous int32 for pooling: {e}"))
        })?;

        let mut max_x = vec![0i32; b];
        for batch in 0..b {
            let mut batch_max = 0i32;
            for token in 0..l {
                let base = (batch * l + token) * 2;
                let x = pos[base].max(0);
                if x > batch_max {
                    batch_max = x;
                }
            }
            max_x[batch] = batch_max + 1;
        }

        let mut weights = vec![0f32; b * l * length_usize];
        let mut mask = vec![false; b * length_usize];
        for batch in 0..b {
            let stride_x = (max_x[batch] / k).max(1);
            for token in 0..l {
                let base = (batch * l + token) * 2;
                let x_pos = pos[base];
                let y_pos = pos[base + 1];
                if x_pos < 0 || y_pos < 0 {
                    continue;
                }
                let kernel_x = x_pos / k;
                let kernel_y = y_pos / k;
                let kernel_idx = kernel_x + stride_x * kernel_y;
                if kernel_idx >= 0 && (kernel_idx as usize) < length_usize {
                    weights[((batch * l + token) * length_usize) + kernel_idx as usize] =
                        1.0 / k_squared;
                    mask[batch * length_usize + kernel_idx as usize] = true;
                }
            }
        }

        // [b, l, L] @ [b, L, d] -> [b, l, d]  (equivalent to einsum "bLl,bLd->bld")
        // Cast weights to match x dtype to avoid mixed-precision matmul error in MLX.
        let weights_arr = Array::from_slice(&weights, &[b as i32, l as i32, length])
            .as_dtype(x.dtype())?;
        let wt = weights_arr.transpose_axes(&[0, 2, 1])?; // [b, length, l]
        eval([&wt]).map_err(|e| Error::Model(format!("eval wt transpose: {e}")))?;
        let output = wt.matmul(x)?;
        eval([&output]).map_err(|e| Error::Model(format!("pool matmul failed: {e}")))?;
        let mask = Array::from_slice(&mask, &[b as i32, length]);
        Ok((output, mask))
    }

    pub fn forward(
        &self,
        hidden_states: &Array,
        patch_positions: &Array,
        padding_positions: &Array,
        output_length: Option<i32>,
    ) -> Result<(Array, Array)> {
        let zeros = ops::zeros_like(hidden_states)?;
        let hidden_states = ops::which(
            &padding_positions.expand_dims(-1)?,
            &zeros,
            hidden_states,
        )?;

        let length = output_length.unwrap_or(self.default_output_length);
        let (hidden_states, valid_mask) = if hidden_states.shape()[1] == length {
            (hidden_states, padding_positions.logical_not()?)
        } else {
            self.avg_pool_by_positions(&hidden_states, patch_positions, length)?
        };
        Ok((
            // Scale by 1/sqrt(hidden_size); cast scalar to match hidden_states dtype.
            hidden_states.multiply(&Array::from(self.root_hidden_size).as_dtype(hidden_states.dtype())?)?,
            valid_mask,
        ))
    }
}

pub struct VisionModel {
    pub config: Gemma4VisionConfig,
    pub patch_size: i32,
    pub pooling_kernel_size: i32,
    pub default_output_length: i32,
    pub max_patches: i32,
    pub patch_embedder: PatchEmbedder,
    pub encoder: VisionEncoder,
    pub pooler: VisionPooler,
    pub std_bias: Option<Array>,
    pub std_scale: Option<Array>,
}

impl VisionModel {
    /// Original eval-and-sync vision forward. Preserved verbatim
    /// (modulo formatting) so the sync code path's outputs and timing
    /// are unchanged for callers that haven't migrated. New code
    /// should call `forward_lazy` with caller-known counts to keep
    /// the entire pipeline lazy.
    pub fn forward(
        &mut self,
        pixel_values: &Array,
        patch_positions: &Array,
        padding_positions: &Array,
    ) -> Result<Array> {
        eval(std::slice::from_ref(padding_positions)).map_err(|e| {
            Error::Model(format!("eval padding_positions: {e}"))
        })?;
        let padding_slice = padding_positions
            .try_as_slice::<bool>()
            .map_err(|e| Error::Model(format!("padding_positions must be contiguous bool: {e}")))?;
        let total = padding_positions.shape()[1] as usize;
        let num_real = padding_slice
            .iter()
            .take(total)
            .position(|&is_pad| is_pad)
            .unwrap_or(total) as i32;
        let mut inputs_embeds = self.patch_embedder.forward(
            pixel_values,
            &patch_positions.index((.., ..num_real, ..)),
            &padding_positions.index((.., ..num_real)),
        )?;

        let num_padding = self.max_patches - num_real;
        if num_padding > 0 {
            let pad_embeds = ops::zeros_dtype(
                &[pixel_values.shape()[0], num_padding, inputs_embeds.shape()[2]],
                inputs_embeds.dtype(),
            )?;
            inputs_embeds = ops::concatenate_axis(&[&inputs_embeds, &pad_embeds], 1)?;
        }

        let b = padding_positions.shape()[0] as usize;
        let l = padding_positions.shape()[1] as usize;
        let mut attn_mask = vec![0f32; b * l * l];
        for batch in 0..b {
            for i in 0..l {
                let valid_i = !padding_slice[batch * l + i];
                for j in 0..l {
                    let valid_j = !padding_slice[batch * l + j];
                    attn_mask[(batch * l + i) * l + j] =
                        if valid_i && valid_j { 0.0 } else { -1e4 };
                }
            }
        }
        let attn_mask =
            Array::from_slice(&attn_mask, &[b as i32, 1, l as i32, l as i32])
                .as_dtype(inputs_embeds.dtype())?;
        let hidden_states =
            self.encoder.forward(&inputs_embeds, patch_positions, &attn_mask)?;
        let (pooled, valid_mask) = self
            .pooler
            .forward(&hidden_states, patch_positions, padding_positions, None)?;

        eval(std::slice::from_ref(&valid_mask)).map_err(|e| {
            Error::Model(format!("eval valid_mask: {e}"))
        })?;
        let valid = valid_mask.try_as_slice::<bool>().map_err(|e| {
            Error::Model(format!("valid pool mask must be contiguous bool: {e}"))
        })?;
        let mut outputs = Vec::with_capacity(b);
        for batch in 0..b {
            let n_valid = valid[batch * valid_mask.shape()[1] as usize
                ..(batch + 1) * valid_mask.shape()[1] as usize]
                .iter()
                .filter(|&&v| v)
                .count() as i32;
            outputs.push(pooled.index((batch as i32, ..n_valid, ..)));
        }
        let refs: Vec<&Array> = outputs.iter().collect();
        let mut hidden_states = if refs.len() == 1 {
            refs[0].reshape(&[1, refs[0].shape()[0], refs[0].shape()[1]])?
        } else {
            ops::concatenate_axis(&refs, 0)?
                .reshape(&[1, -1, pooled.shape()[2]])?
        };

        eval([&hidden_states]).map_err(|e| {
            Error::Model(format!("eval hidden_states after reshape: {e}"))
        })?;
        if let (Some(std_bias), Some(std_scale)) = (&self.std_bias, &self.std_scale) {
            let std_bias = std_bias.as_dtype(hidden_states.dtype())?;
            let std_scale = std_scale.as_dtype(hidden_states.dtype())?;
            hidden_states = hidden_states.subtract(&std_bias)?.multiply(&std_scale)?;
            eval([&hidden_states]).map_err(|e| {
                Error::Model(format!("eval after std norm: {e}"))
            })?;
        }
        Ok(hidden_states)
    }

    /// Fully-lazy vision forward.
    ///
    /// Callers must supply:
    /// - `num_real`: number of real (non-padded) patches at the head of
    ///   `patch_positions` / `padding_positions`.
    /// - `n_valid`: number of valid post-pooler positions for the
    ///   (single) image. Equals `(num_real_per_side / pool)^2`.
    ///
    /// Both are derivable in `preprocess_image_gemma4` without any GPU
    /// readback — this signature exists so the entire vision graph
    /// stays unevaluated until the caller (typically
    /// `prefill_multimodal_async`) issues a single eval at the end of
    /// the first prefill chunk.
    pub fn forward_lazy(
        &mut self,
        pixel_values: &Array,
        patch_positions: &Array,
        padding_positions: &Array,
        num_real: i32,
        n_valid: i32,
    ) -> Result<Array> {
        let mut inputs_embeds = self.patch_embedder.forward(
            pixel_values,
            &patch_positions.index((.., ..num_real, ..)),
            &padding_positions.index((.., ..num_real)),
        )?;

        let num_padding = self.max_patches - num_real;
        if num_padding > 0 {
            let pad_embeds = ops::zeros_dtype(
                &[pixel_values.shape()[0], num_padding, inputs_embeds.shape()[2]],
                inputs_embeds.dtype(),
            )?;
            inputs_embeds = ops::concatenate_axis(&[&inputs_embeds, &pad_embeds], 1)?;
        }

        // Lazy bidirectional attention mask: -1e4 where either row or
        // column is a padded patch, 0 elsewhere. Built from
        // `padding_positions` (shape [B, L], bool) via broadcast
        // reductions — no CPU loop, no eval/sync.
        let b = padding_positions.shape()[0];
        let l = padding_positions.shape()[1];
        let not_pad = padding_positions.logical_not()?; // [B, L] bool
        let row = not_pad.reshape(&[b, 1, l, 1])?;
        let col = not_pad.reshape(&[b, 1, 1, l])?;
        let valid = row.logical_and(&col)?; // [B, 1, L, L] bool
        let zero =
            Array::from(0.0_f32).as_dtype(inputs_embeds.dtype())?;
        let neg_inf =
            Array::from(-1.0e4_f32).as_dtype(inputs_embeds.dtype())?;
        let attn_mask = ops::r#where(&valid, &zero, &neg_inf)?;

        let hidden_states =
            self.encoder.forward(&inputs_embeds, patch_positions, &attn_mask)?;
        let (pooled, _valid_mask) = self
            .pooler
            .forward(&hidden_states, patch_positions, padding_positions, None)?;

        // Crop trailing invalid pool positions using the caller-provided
        // count — same as the historical CPU-readback version, but
        // without the eval. Single-batch only (the only call shape in
        // `encode_image_bytes`).
        let mut hidden_states = pooled
            .index((0i32, ..n_valid, ..))
            .reshape(&[1, n_valid, pooled.shape()[2]])?;

        if let (Some(std_bias), Some(std_scale)) =
            (&self.std_bias, &self.std_scale)
        {
            let std_bias = std_bias.as_dtype(hidden_states.dtype())?;
            let std_scale = std_scale.as_dtype(hidden_states.dtype())?;
            hidden_states = hidden_states
                .subtract(&std_bias)?
                .multiply(&std_scale)?;
        }
        Ok(hidden_states)
    }
}

pub struct EmbedVision {
    pub embedding_pre_projection_norm: VisionRmsNormNoScale,
    pub embedding_projection: MaybeQuantized<nn::Linear>,
}

impl EmbedVision {
    pub fn forward(&mut self, hidden_states: &Array) -> Result<Array> {
        // The vision encoder may run in a different dtype (e.g. Float32) than the
        // embedding projection weights (e.g. Bfloat16). Cast here so the linear op
        // sees consistent dtypes.
        let weight_dtype = match &self.embedding_projection {
            MaybeQuantized::Original(l) => l.weight.dtype(),
            MaybeQuantized::Quantized(q) => q.inner.weight.dtype(),
        };
        let hidden_states = hidden_states.as_dtype(weight_dtype)?;
        let normed = self.embedding_pre_projection_norm.forward(&hidden_states)?;
        // No eval here: `encode_image_bytes_async` relies on this graph
        // staying lazy so vision kernels overlap with the first prefill chunk.
        self.embedding_projection
            .forward(&normed)
            .map_err(|e| Error::Model(e.to_string()))
    }
}

/// Same as `preprocess_image_gemma4` but also returns the number of
/// non-padding patches (`num_real`) and the resulting valid post-pooler
/// token count (`n_valid`) so callers can drive `forward_lazy` without
/// any GPU readback. The two extra ints are derived purely from the
/// chosen target resolution on the host side.
pub fn preprocess_image_gemma4_with_counts(
    bytes: &[u8],
) -> Result<(Array, Vec<i32>, Vec<i32>, i32, i32)> {
    let (pixel_values, patch_positions, padding_positions) =
        preprocess_image_gemma4(bytes)?;
    let pooling_kernel_size = 3i32;
    let max_soft_tokens = 280i32;
    let max_patches = max_soft_tokens * pooling_kernel_size * pooling_kernel_size;
    let n_pad = padding_positions.iter().filter(|&&v| v != 0).count() as i32;
    let num_real = max_patches - n_pad;
    // Pooler emits one soft token per pool^2 kernel over the valid
    // patch grid; for the canonical square-resize path that the
    // preprocess function uses, `num_real` is always a perfect
    // multiple of pool^2, so this is exact.
    let n_valid = num_real / (pooling_kernel_size * pooling_kernel_size);
    Ok((pixel_values, patch_positions, padding_positions, num_real, n_valid))
}

pub fn preprocess_image_gemma4(bytes: &[u8]) -> Result<(Array, Vec<i32>, Vec<i32>)> {
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

    let patch_size = 16i32;
    let pooling_kernel_size = 3i32;
    let max_soft_tokens = 280i32;
    let max_patches = max_soft_tokens * pooling_kernel_size * pooling_kernel_size;
    let side_mult = patch_size * pooling_kernel_size;
    let target_px = max_patches * patch_size * patch_size;
    let factor = ((target_px as f32) / ((height * width) as f32)).sqrt();

    let mut target_height = ((factor * height as f32) / side_mult as f32).floor() as i32 * side_mult;
    let mut target_width = ((factor * width as f32) / side_mult as f32).floor() as i32 * side_mult;
    if target_height == 0 && target_width == 0 {
        return Err(Error::InvalidConfig(
            "Attempting to resize to a 0x0 image".to_string(),
        ));
    }
    let max_side_length = (max_patches / (pooling_kernel_size * pooling_kernel_size)) * side_mult;
    if target_height == 0 {
        target_height = side_mult;
        target_width = (((width as f32 / height as f32).floor() as i32).max(1) * side_mult)
            .min(max_side_length);
    } else if target_width == 0 {
        target_width = side_mult;
        target_height = (((height as f32 / width as f32).floor() as i32).max(1) * side_mult)
            .min(max_side_length);
    }

    let resized = image.resize_exact(target_width as u32, target_height as u32, FilterType::CatmullRom);
    let rgb = resized.to_rgb8();
    let h = target_height as usize;
    let w = target_width as usize;
    let mut pixels = vec![0f32; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            let px = rgb.get_pixel(x as u32, y as u32).0;
            let idx = y * w + x;
            pixels[idx] = px[0] as f32 / 255.0;
            pixels[h * w + idx] = px[1] as f32 / 255.0;
            pixels[2 * h * w + idx] = px[2] as f32 / 255.0;
        }
    }
    let pixel_values = Array::from_slice(&pixels, &[1, 3, target_height, target_width]);

    let p_h = target_height / patch_size;
    let p_w = target_width / patch_size;
    let num_patches = (p_h * p_w).min(max_patches) as usize;
    let mut patch_positions = Vec::with_capacity(max_patches as usize * 2);
    for y in 0..p_h {
        for x in 0..p_w {
            if (patch_positions.len() / 2) >= max_patches as usize {
                break;
            }
            patch_positions.push(x);
            patch_positions.push(y);
        }
    }
    while (patch_positions.len() / 2) < max_patches as usize {
        patch_positions.push(-1);
        patch_positions.push(-1);
    }

    let mut padding_positions = vec![0i32; max_patches as usize];
    for idx in num_patches..max_patches as usize {
        padding_positions[idx] = 1;
    }

    Ok((pixel_values, patch_positions, padding_positions))
}

pub fn load_vision_model(
    weights: &HashMap<String, Array>,
    config: &Gemma4VisionConfig,
) -> Result<VisionModel> {
    let prefix = "model.vision_tower";
    let patch_embedder = PatchEmbedder {
        hidden_size: config.hidden_size,
        patch_size: config.patch_size,
        position_embedding_size: config.position_embedding_size,
        input_proj: make_linear(get_weight(weights, &format!("{prefix}.patch_embedder.input_proj.weight"))?),
        position_embedding_table: get_weight(
            weights,
            &format!("{prefix}.patch_embedder.position_embedding_table"),
        )?,
    };

    let mut layers = Vec::with_capacity(config.num_hidden_layers as usize);
    for i in 0..config.num_hidden_layers {
        let layer_prefix = format!("{prefix}.encoder.layers.{i}");
        layers.push(VisionEncoderLayer {
            self_attn: VisionAttention {
                num_heads: config.num_attention_heads,
                num_kv_heads: config.num_key_value_heads,
                head_dim: config.head_dim,
                rope_theta: config.rope_parameters.rope_theta,
                q_proj: make_linear(get_weight(weights, &format!("{layer_prefix}.self_attn.q_proj.linear.weight"))?),
                k_proj: make_linear(get_weight(weights, &format!("{layer_prefix}.self_attn.k_proj.linear.weight"))?),
                v_proj: make_linear(get_weight(weights, &format!("{layer_prefix}.self_attn.v_proj.linear.weight"))?),
                o_proj: make_linear(get_weight(weights, &format!("{layer_prefix}.self_attn.o_proj.linear.weight"))?),
                q_norm: VisionRmsNorm {
                    weight: get_weight(weights, &format!("{layer_prefix}.self_attn.q_norm.weight"))?,
                    eps: config.rms_norm_eps,
                },
                k_norm: VisionRmsNorm {
                    weight: get_weight(weights, &format!("{layer_prefix}.self_attn.k_norm.weight"))?,
                    eps: config.rms_norm_eps,
                },
                v_norm: VisionRmsNormNoScale {
                    eps: config.rms_norm_eps,
                },
            },
            mlp: VisionMLP {
                gate_proj: make_linear(get_weight(weights, &format!("{layer_prefix}.mlp.gate_proj.linear.weight"))?),
                up_proj: make_linear(get_weight(weights, &format!("{layer_prefix}.mlp.up_proj.linear.weight"))?),
                down_proj: make_linear(get_weight(weights, &format!("{layer_prefix}.mlp.down_proj.linear.weight"))?),
            },
            input_layernorm: VisionRmsNorm {
                weight: get_weight(weights, &format!("{layer_prefix}.input_layernorm.weight"))?,
                eps: config.rms_norm_eps,
            },
            post_attention_layernorm: VisionRmsNorm {
                weight: get_weight(weights, &format!("{layer_prefix}.post_attention_layernorm.weight"))?,
                eps: config.rms_norm_eps,
            },
            pre_feedforward_layernorm: VisionRmsNorm {
                weight: get_weight(weights, &format!("{layer_prefix}.pre_feedforward_layernorm.weight"))?,
                eps: config.rms_norm_eps,
            },
            post_feedforward_layernorm: VisionRmsNorm {
                weight: get_weight(weights, &format!("{layer_prefix}.post_feedforward_layernorm.weight"))?,
                eps: config.rms_norm_eps,
            },
        });
    }

    let std_bias = if config.standardize {
        Some(get_weight(weights, &format!("{prefix}.std_bias"))?)
    } else {
        None
    };
    let std_scale = if config.standardize {
        Some(get_weight(weights, &format!("{prefix}.std_scale"))?)
    } else {
        None
    };

    Ok(VisionModel {
        config: config.clone(),
        patch_size: config.patch_size,
        pooling_kernel_size: config.pooling_kernel_size,
        default_output_length: config.default_output_length,
        max_patches: config.default_output_length * config.pooling_kernel_size * config.pooling_kernel_size,
        patch_embedder,
        encoder: VisionEncoder { layers },
        pooler: VisionPooler {
            hidden_size: config.hidden_size,
            default_output_length: config.default_output_length,
            root_hidden_size: (config.hidden_size as f32).sqrt(),
        },
        std_bias,
        std_scale,
    })
}

pub fn load_embed_vision(
    weights: &HashMap<String, Array>,
    rms_eps: f32,
    quant: Option<&crate::model::QuantizationConfig>,
) -> Result<EmbedVision> {
    let projection = crate::model::make_mq_linear(
        weights,
        "model.embed_vision.embedding_projection",
        quant,
    )
    .map_err(|e| Error::Model(e.to_string()))?;
    Ok(EmbedVision {
        embedding_pre_projection_norm: VisionRmsNormNoScale { eps: rms_eps },
        embedding_projection: projection,
    })
}
