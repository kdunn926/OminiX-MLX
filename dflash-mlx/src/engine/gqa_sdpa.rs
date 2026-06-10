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
    // The fold is g-major: folded query row `r` is original row `r % q_len`
    // (group `r / q_len`). A pre-fold `[.., q_len, kv]` mask therefore has
    // to be TILED along the q axis (`[m; m; ...]`, gqa copies) so row `r`
    // reads `m[r % q_len]`. `repeat_axis` has np.repeat semantics
    // (row `r` ← `m[r / gqa]`) and silently misaligned the mask whenever
    // `gqa > 1 && q_len > 1` — exactly the draft SWA path on 27B.
    let tile_q_axis = |m: &Array| -> Result<Array, Exception> {
        let q_axis = m.shape().len().saturating_sub(2) as i32;
        let copies: Vec<&Array> = std::iter::repeat(m).take(gqa as usize).collect();
        mlx_rs::ops::concatenate_axis(&copies, q_axis)
    };
    let folded_mask_owned = match &mask {
        Some(SdpaMask::Array(m)) => {
            let m_shape = m.shape();
            let q_axis = m_shape.len().saturating_sub(2);
            if m_shape.get(q_axis).copied() == Some(q_len) && gqa > 1 {
                Some(tile_q_axis(m)?)
            } else {
                None
            }
        }
        // Causal over the folded rows would compare the folded row index
        // (g·q_len + t) against kv positions — wrong for every group g > 0
        // when q_len > 1. Materialize the true [q_len, kv] causal mask and
        // tile it instead.
        Some(SdpaMask::Causal) if q_len > 1 && gqa > 1 => {
            let kv_len = k_shape[2];
            let causal = mlx_rs_core::utils::create_causal_mask(
                q_len,
                Some(kv_len - q_len),
                None,
                None,
            )?;
            Some(tile_q_axis(&causal)?)
        }
        _ => None,
    };
    let folded_mask = match (&mask, &folded_mask_owned) {
        (_, Some(arr)) => Some(SdpaMask::Array(arr)),
        (Some(SdpaMask::Causal), _) => Some(SdpaMask::Causal),
        (Some(SdpaMask::Array(m)), _) => Some(SdpaMask::Array(*m)),
        (None, _) => None,
    };
    let output = scaled_dot_product_attention(
        grouped_queries,
        k.clone(),
        v.clone(),
        None::<&mut KVCache>,
        scale,
        folded_mask,
    )?;
    output
        .reshape(&[batch, kv_heads, gqa, q_len, head_dim])?
        .reshape(&[batch, q_heads, q_len, head_dim])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(n: i32, f: f32) -> Vec<f32> {
        (0..n).map(|i| ((i as f32) * f).sin()).collect()
    }

    /// Regression test for the folded-mask alignment: with gqa > 1 and
    /// q_len > 1 the pre-fold mask must be tiled (row r % q_len), not
    /// row-repeated (row r / gqa). Compares against a per-head expanded
    /// reference with a sliding-window additive mask — the draft SWA
    /// configuration that hit the bug.
    #[test]
    fn folded_gqa_sdpa_matches_expanded_reference_with_swa_mask() {
        let (b, hq, hkv, ql, kv, d) = (1_i32, 4, 2, 3, 5, 8);
        let gqa = hq / hkv;
        let scale = 1.0 / (d as f32).sqrt();
        let q = Array::from_slice(&ramp(b * hq * ql * d, 0.07), &[b, hq, ql, d]);
        let k = Array::from_slice(&ramp(b * hkv * kv * d, 0.11), &[b, hkv, kv, d]);
        let v = Array::from_slice(&ramp(b * hkv * kv * d, 0.13), &[b, hkv, kv, d]);

        // Sliding-window causal mask [ql, kv], additive -inf form, window 3.
        // Deliberately asymmetric across rows so any row misalignment shows.
        let offset = kv - ql;
        let window = 3;
        let mut mdata = vec![0.0_f32; (ql * kv) as usize];
        for t in 0..ql {
            for s in 0..kv {
                let pos = offset + t;
                let allowed = s <= pos && s + window > pos;
                if !allowed {
                    mdata[(t * kv + s) as usize] = f32::NEG_INFINITY;
                }
            }
        }
        let mask = Array::from_slice(&mdata, &[ql, kv]);

        let out = grouped_gqa_sdpa(&q, &k, &v, scale, Some(SdpaMask::Array(&mask))).unwrap();

        // Reference: expand K/V to Hq heads (query head h reads kv head
        // h / gqa) and run the unfused op chain.
        let expand = |x: &Array| -> Array {
            let xs = x.shape().to_vec();
            let x5 = x.reshape(&[b, hkv, 1, xs[2], d]).unwrap();
            mlx_rs::ops::broadcast_to(&x5, &[b, hkv, gqa, xs[2], d])
                .unwrap()
                .reshape(&[b, hq, xs[2], d])
                .unwrap()
        };
        let k_exp = expand(&k);
        let v_exp = expand(&v);
        let scores = q
            .matmul(&k_exp.transpose_axes(&[0, 1, 3, 2]).unwrap())
            .unwrap()
            .multiply(mlx_rs::array!(scale))
            .unwrap()
            .add(&mask)
            .unwrap();
        let probs = mlx_rs::ops::softmax_axis(&scores, -1, None).unwrap();
        let reference = probs.matmul(&v_exp).unwrap();

        let diff = out
            .subtract(&reference)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item::<f32>();
        assert!(diff < 1e-5, "folded GQA SDPA diverges from reference: {diff}");
    }
}
