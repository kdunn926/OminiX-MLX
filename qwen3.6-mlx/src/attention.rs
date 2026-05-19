use mlx_rs::{
    builder::Builder, error::Exception, macros::ModuleParameters, module::Module, nn,
    ops::indexing::IndexOp, quantization::MaybeQuantized, Array,
};
use mlx_rs_core::{
    cache::{KVCache, KeyValueCache},
    utils::{initialize_rope, scaled_dot_product_attention, AttentionMask, SdpaMask},
};

use crate::config::TextConfig;

/// Full attention with output gate (Qwen3.5 GatedAttention).
///
/// q_proj outputs double width: [query_dims | gate_dims] per head.
/// After attention: output = attn_output * sigmoid(gate), then o_proj.
/// RoPE is partial (only partial_rotary_factor * head_dim dimensions).
#[derive(Debug, ModuleParameters)]
pub struct GatedAttention {
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,

    #[param]
    pub q_proj: MaybeQuantized<nn::Linear>,
    #[param]
    pub k_proj: MaybeQuantized<nn::Linear>,
    #[param]
    pub v_proj: MaybeQuantized<nn::Linear>,
    #[param]
    pub o_proj: MaybeQuantized<nn::Linear>,
    #[param]
    pub q_norm: nn::RmsNorm,
    #[param]
    pub k_norm: nn::RmsNorm,
    #[param]
    pub rope: nn::Rope,
}

pub struct GatedAttentionInput<'a, C: KeyValueCache> {
    pub x: &'a Array,
    pub mask: Option<&'a AttentionMask>,
    pub cache: Option<&'a mut C>,
}

impl<C: KeyValueCache> Module<GatedAttentionInput<'_, C>> for GatedAttention {
    type Output = Array;
    type Error = Exception;

    #[allow(non_snake_case)]
    fn forward(&mut self, input: GatedAttentionInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let GatedAttentionInput { x, mask, mut cache } = input;

        let shape = x.shape();
        let B = shape[0];
        let L = shape[1];

        // Q projection: outputs [B, L, n_heads * head_dim * 2] (query + gate)
        let q_and_gate = crate::verify_hook::quantized_linear_forward(&mut self.q_proj, x)?;
        // Reshape to [B, L, n_heads, head_dim * 2], then split
        let q_and_gate = q_and_gate.reshape(&[B, L, self.n_heads, self.head_dim * 2])?;
        let queries = q_and_gate.index((.., .., .., ..self.head_dim));
        let gate = q_and_gate.index((.., .., .., self.head_dim..));
        let gate = gate.reshape(&[B, L, -1])?; // [B, L, n_heads * head_dim]

        // K, V projections
        let keys = crate::verify_hook::quantized_linear_forward(&mut self.k_proj, x)?;
        let values = crate::verify_hook::quantized_linear_forward(&mut self.v_proj, x)?;

        // Reshape and transpose to [B, heads, L, head_dim]
        let mut queries = self
            .q_norm
            .forward(&queries.transpose_axes(&[0, 2, 1, 3])?)?;
        let mut keys = self.k_norm.forward(
            &keys
                .reshape(&[B, L, self.n_kv_heads, -1])?
                .transpose_axes(&[0, 2, 1, 3])?,
        )?;
        let mut values = values
            .reshape(&[B, L, self.n_kv_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;

        // Apply RoPE with cache offset
        let mut fused_attn_out: Option<Array> = None;
        if let Some(cache) = cache.as_mut() {
            let q_input = nn::RopeInputBuilder::new(&queries)
                .offset(cache.offset())
                .build()?;
            queries = self.rope.forward(q_input)?;
            let k_input = nn::RopeInputBuilder::new(&keys)
                .offset(cache.offset())
                .build()?;
            keys = self.rope.forward(k_input)?;

            // Spike: try the cache's fused TurboQuant attention path
            // for decode steps (L=1). On hit we get attn_out directly
            // and skip update_and_fetch + SDPA. q is [B, Hq, 1, D];
            // GatedAttention stores queries as [B, Hq, L, head_dim]
            // already so the shape matches the trait contract.
            let kv_repeat = self.n_heads as i32 / self.n_kv_heads as i32;
            if L == 1 {
                let mask_arr = match mask {
                    Some(AttentionMask::Array(m)) => Some(m),
                    _ => None,
                };
                if let Some(out) = cache.try_fused_attention(
                    &queries,
                    keys.clone(),
                    values.clone(),
                    self.scale,
                    mask_arr,
                    kv_repeat,
                )? {
                    fused_attn_out = Some(out);
                }
            }

            if fused_attn_out.is_none() {
                let (k, v) = cache.update_and_fetch(keys, values)?;
                keys = k;
                values = v;
            }
        } else {
            queries = self.rope.forward(nn::RopeInput::new(&queries))?;
            keys = self.rope.forward(nn::RopeInput::new(&keys))?;
        }

        // Scaled dot-product attention
        let sdpa_mask = match mask {
            Some(AttentionMask::Array(m)) => Some(SdpaMask::Array(m)),
            Some(AttentionMask::Causal) => Some(SdpaMask::Causal),
            None if L > 1 => Some(SdpaMask::Causal),
            None => None,
        };

        let attn_output = if let Some(out) = fused_attn_out {
            out.transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[B, L, -1])?
        } else {
            scaled_dot_product_attention::<KVCache>(
                queries, keys, values, None, self.scale, sdpa_mask,
            )?
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[B, L, -1])?
        }; // [B, L, n_heads * head_dim]

        // Apply output gate: output * sigmoid(gate)
        let gated = attn_output.multiply(nn::sigmoid(gate)?)?;

        crate::verify_hook::quantized_linear_forward(&mut self.o_proj, &gated)
    }

    fn training_mode(&mut self, _mode: bool) {}
}

impl GatedAttention {
    pub fn new_from_config(config: &TextConfig) -> Result<Self, Exception> {
        let n_heads = config.num_attention_heads;
        let n_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let hidden = config.hidden_size;
        let scale = (head_dim as f32).sqrt().recip();

        let q_proj = nn::LinearBuilder::new(hidden, n_heads * head_dim * 2)
            .bias(false)
            .build()?;
        let k_proj = nn::LinearBuilder::new(hidden, n_kv_heads * head_dim)
            .bias(false)
            .build()?;
        let v_proj = nn::LinearBuilder::new(hidden, n_kv_heads * head_dim)
            .bias(false)
            .build()?;
        let o_proj = nn::LinearBuilder::new(n_heads * head_dim, hidden)
            .bias(false)
            .build()?;

        let q_norm = nn::RmsNormBuilder::new(head_dim)
            .eps(config.rms_norm_eps)
            .build()?;
        let k_norm = nn::RmsNormBuilder::new(head_dim)
            .eps(config.rms_norm_eps)
            .build()?;

        let rope_dims = (head_dim as f32 * config.rope_parameters.partial_rotary_factor) as i32;
        let rope = initialize_rope(
            rope_dims,
            config.rope_parameters.rope_theta,
            false,
            &None,
            config.max_position_embeddings,
        )?;

        Ok(Self {
            n_heads,
            n_kv_heads,
            head_dim,
            scale,
            q_proj: MaybeQuantized::Original(q_proj),
            k_proj: MaybeQuantized::Original(k_proj),
            v_proj: MaybeQuantized::Original(v_proj),
            o_proj: MaybeQuantized::Original(o_proj),
            q_norm,
            k_norm,
            rope,
        })
    }
}
