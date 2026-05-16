use mlx_rs::{error::Exception, Array};
use mlx_rs_core::{scaled_dot_product_attention, KVCache, SdpaMask};

/// GQA-aware scaled dot-product attention.
///
/// mlx's fast SDPA handles GQA natively (K/V heads < Q heads) — do NOT
/// pre-tile K/V. Pass them at their native head count and let the kernel expand.
pub fn grouped_gqa_sdpa<'a>(
    q: &Array,
    k: &Array,
    v: &Array,
    scale: f32,
    mask: Option<SdpaMask<'a>>,
) -> Result<Array, Exception> {
    let q_shape = q.shape();
    let k_shape = k.shape();
    let v_shape = v.shape();
    if q_shape.len() != 4 || k_shape.len() != 4 || v_shape.len() != 4 {
        return Err(Exception::custom(format!(
            "grouped_gqa_sdpa expects rank-4 tensors, got q={q_shape:?} k={k_shape:?} v={v_shape:?}"
        )));
    }

    let batch = q_shape[0];
    let q_heads = q_shape[1];
    let q_len = q_shape[2];
    let head_dim = q_shape[3];
    let kv_heads = k_shape[1];

    if kv_heads <= 0 || q_heads == kv_heads || q_heads % kv_heads != 0 {
        return scaled_dot_product_attention(
            q.clone(),
            k.clone(),
            v.clone(),
            None::<&mut KVCache>,
            scale,
            mask,
        );
    }

    // Match the Python reference implementation: fold each GQA group of query heads into
    // the sequence dimension before SDPA, then reshape back. MLX's SDPA cannot be assumed
    // to expand K/V heads for us across all code paths.
    let gqa = q_heads / kv_heads;
    let grouped_queries = q
        .reshape(&[batch, kv_heads, gqa, q_len, head_dim])?
        .reshape(&[batch, kv_heads, gqa * q_len, head_dim])?;
    let output = scaled_dot_product_attention(
        grouped_queries,
        k.clone(),
        v.clone(),
        None::<&mut KVCache>,
        scale,
        mask,
    )?;
    output
        .reshape(&[batch, kv_heads, gqa, q_len, head_dim])?
        .reshape(&[batch, q_heads, q_len, head_dim])
}
