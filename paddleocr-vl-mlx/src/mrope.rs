//! Multimodal Rotary Position Embedding (MROPE).
//!
//! Reference: `apply_multimodal_rotary_pos_emb` in
//! `PaddlePaddle/PaddleOCR-VL-1.5/modeling_paddleocr_vl.py:313` (originally
//! introduced for Qwen 2-VL — <https://qwenlm.github.io/blog/qwen2-vl/>).
//!
//! ## Why MROPE
//!
//! Standard 1-D RoPE rotates each `(head_dim/2)`-pair of channels by an
//! angle derived from a single position index `pos ∈ [0, T)`. For text-only
//! sequences this captures linear order.
//!
//! Vision tokens have 2-D spatial layout (a token at grid `(h, w)` should
//! attend differently to other tokens depending on row/column distance).
//! MROPE handles this by carrying **three** position channels — temporal
//! `t`, height `h`, width `w` — and partitioning the rotation across
//! channels via `mrope_section`. For PaddleOCR-VL the partition is
//! `[16, 24, 24]`, doubled to `[16, 24, 24, 16, 24, 24]` so that
//! [`rotate_half`]'s `dim ↔ dim + head_dim/2` pairing always rotates
//! within the same channel.
//!
//! For text-only tokens, all three channels share the same position index;
//! MROPE collapses to 1-D RoPE in that case.

use std::sync::Arc;

use mlx_rs::{
    ops::{self, indexing::IndexOp},
    Array, Dtype,
};
use mlx_rs_core::error::{Error, Result};

/// Precomputed per-position channel-to-output-dim mapping. Computed once
/// from `mrope_section` and reused at every layer.
#[derive(Debug, Clone)]
pub struct MropePartition {
    /// `mrope_section * 2` — e.g. `[16, 24, 24, 16, 24, 24]` for the
    /// PaddleOCR-VL `[16, 24, 24]` section.
    sections: Vec<i32>,
    /// `head_dim`, the sum of `sections`.
    head_dim: i32,
}

impl MropePartition {
    /// Build from a raw `mrope_section` list (e.g. `[16, 24, 24]`).
    /// `head_dim` must equal `2 * sum(mrope_section)`.
    pub fn new(mrope_section: &[i32], head_dim: i32) -> Result<Self> {
        let sum: i32 = mrope_section.iter().sum();
        if sum * 2 != head_dim {
            return Err(Error::InvalidConfig(format!(
                "MropePartition: 2 * sum(mrope_section) = {} != head_dim = {}",
                sum * 2,
                head_dim
            )));
        }
        let mut sections = Vec::with_capacity(mrope_section.len() * 2);
        sections.extend_from_slice(mrope_section);
        sections.extend_from_slice(mrope_section);
        Ok(Self { sections, head_dim })
    }
}

/// Inverse-frequency table used to build cos/sin. Standard RoPE formula:
/// `inv_freq[i] = 1 / (rope_theta ^ (2i / head_dim))` for `i ∈ [0, head_dim/2)`.
pub fn inv_freq(head_dim: i32, rope_theta: f32) -> Array {
    let half = (head_dim / 2) as usize;
    let mut data = Vec::with_capacity(half);
    for i in 0..half {
        let exp = (2 * i) as f32 / head_dim as f32;
        data.push(1.0_f32 / rope_theta.powf(exp));
    }
    Array::from_slice(&data, &[half as i32])
}

/// Inner helper: build `cos`/`sin` of shape `(3, B, T, head_dim)` from
/// `position_ids` `(3, B, T)`.
fn cos_sin_per_channel(
    inv_freq: &Array,
    position_ids: &Array,
) -> Result<(Array, Array)> {
    let shape = position_ids.shape();
    if shape.len() != 3 || shape[0] != 3 {
        return Err(Error::InvalidConfig(format!(
            "MROPE position_ids must have shape (3, B, T); got {:?}",
            shape
        )));
    }
    let _b = shape[1];
    let _t = shape[2];
    // freqs = position_ids[..., None] * inv_freq[None, None, None, :]
    // → (3, B, T, head_dim/2)
    let pos_f = position_ids
        .as_dtype(Dtype::Float32)
        .map_err(Error::from)?
        .expand_dims(-1)
        .map_err(Error::from)?;
    let inv = inv_freq
        .as_dtype(Dtype::Float32)
        .map_err(Error::from)?
        .reshape(&[1, 1, 1, -1])
        .map_err(Error::from)?;
    let freqs = pos_f.multiply(&inv).map_err(Error::from)?;
    // emb = cat([freqs, freqs], dim=-1) → (3, B, T, head_dim)
    let emb = ops::concatenate_axis(&[&freqs, &freqs], -1).map_err(Error::from)?;
    let cos = emb.cos().map_err(Error::from)?;
    let sin = emb.sin().map_err(Error::from)?;
    Ok((cos, sin))
}

/// Merge `(3, B, T, head_dim)` per-channel cos/sin into `(B, T, head_dim)`
/// by selecting the right channel for each `mrope_section`-aligned slice.
///
/// Layout of the merged tensor's last dim, for `mrope_section = [a, b, c]`
/// (head_dim = 2(a+b+c)):
/// `[t × a | h × b | w × c | t × a | h × b | w × c]`
fn merge_channels(
    per_channel: &Array,
    partition: &MropePartition,
) -> Result<Array> {
    let shape = per_channel.shape();
    if shape.len() != 4 || shape[0] != 3 || shape[3] != partition.head_dim {
        return Err(Error::InvalidConfig(format!(
            "merge_channels expects (3, B, T, head_dim={}); got {:?}",
            partition.head_dim, shape
        )));
    }
    let mut offset = 0i32;
    let mut pieces: Vec<Array> = Vec::with_capacity(partition.sections.len());
    for (i, &sec) in partition.sections.iter().enumerate() {
        let chan = (i % 3) as i32;
        // per_channel[chan, :, :, offset:offset+sec] — shape (B, T, sec)
        let piece = per_channel
            .index((chan, .., .., offset..offset + sec));
        pieces.push(piece);
        offset += sec;
    }
    let refs: Vec<&Array> = pieces.iter().collect();
    ops::concatenate_axis(&refs, -1).map_err(Error::from)
}

/// rotate_half: `[x0, x1] → [-x1, x0]` along the last dim, where `x0 = x[..., :H/2]`
/// and `x1 = x[..., H/2:]`. Pair-with-itself rotation used by RoPE / MROPE.
pub fn rotate_half(x: &Array) -> Result<Array> {
    let dim = *x.shape().last().unwrap();
    let half = dim / 2;
    let nd = x.shape().len();
    let rotated = match nd {
        3 => {
            let x1 = x.index((.., .., ..half));
            let x2 = x.index((.., .., half..));
            ops::concatenate_axis(&[&x2.negative().map_err(Error::from)?, &x1], -1)
                .map_err(Error::from)?
        }
        4 => {
            let x1 = x.index((.., .., .., ..half));
            let x2 = x.index((.., .., .., half..));
            ops::concatenate_axis(&[&x2.negative().map_err(Error::from)?, &x1], -1)
                .map_err(Error::from)?
        }
        n => {
            return Err(Error::InvalidConfig(format!(
                "rotate_half expects 3- or 4-D input, got {n}-D"
            )))
        }
    };
    Ok(rotated)
}

/// Build per-token cos/sin tensors suitable for [`apply_rotary_qk`].
///
/// Returns `(cos, sin)` each of shape `(B, T, head_dim)`. For a 4-D
/// `(B, H, T, head_dim)` Q/K layout, broadcast against the head axis at
/// call time (or call this with `position_ids` reshaped to `(3, B, T)`).
pub fn build_cos_sin(
    inv_freq: &Array,
    position_ids: &Array,
    partition: &MropePartition,
) -> Result<(Array, Array)> {
    let (cos_pc, sin_pc) = cos_sin_per_channel(inv_freq, position_ids)?;
    let cos = merge_channels(&cos_pc, partition)?;
    let sin = merge_channels(&sin_pc, partition)?;
    Ok((cos, sin))
}

/// Apply MROPE to a Q/K pair. Inputs:
///   - `q`, `k`: shape `(B, H_q, T, head_dim)` and `(B, H_kv, T, head_dim)`.
///   - `cos`, `sin`: shape `(B, T, head_dim)` from [`build_cos_sin`].
///
/// Returns `(q_rot, k_rot)` of the same shapes as `q, k`.
pub fn apply_rotary_qk(
    q: &Array,
    k: &Array,
    cos: &Array,
    sin: &Array,
) -> Result<(Array, Array)> {
    // Broadcast cos/sin against the head axis (H).
    let cos_b = cos.expand_dims(1).map_err(Error::from)?;
    let sin_b = sin.expand_dims(1).map_err(Error::from)?;
    let q_rot = q
        .multiply(&cos_b)
        .map_err(Error::from)?
        .add(&rotate_half(q)?.multiply(&sin_b).map_err(Error::from)?)
        .map_err(Error::from)?;
    let k_rot = k
        .multiply(&cos_b)
        .map_err(Error::from)?
        .add(&rotate_half(k)?.multiply(&sin_b).map_err(Error::from)?)
        .map_err(Error::from)?;
    Ok((q_rot, k_rot))
}

/// Convenience: shared `Arc<MropePartition>` for cheap clones across layers.
pub fn shared_partition(
    mrope_section: &[i32],
    head_dim: i32,
) -> Result<Arc<MropePartition>> {
    Ok(Arc::new(MropePartition::new(mrope_section, head_dim)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Standard 1-D RoPE: each channel of `position_ids` is the same `pos`.
    /// Result must equal the canonical 1-D rotation (Llama/Gemma style).
    #[test]
    fn text_only_collapses_to_1d_rope() {
        let head_dim = 8_i32; // section = [1, 1, 2] doubles to [1, 1, 2, 1, 1, 2] = 8
        let section = vec![1_i32, 1, 2];
        let part = MropePartition::new(&section, head_dim).unwrap();
        let inv = inv_freq(head_dim, 10_000.0);
        // Channel-major (3, 1, 4): rows are [t-positions, h-positions, w-positions].
        // For "text-only" we set all three to the same 0..4.
        let mut chan_major = vec![0_i32; 3 * 4];
        for t in 0..4 {
            chan_major[t] = t as i32; // t-channel
            chan_major[4 + t] = t as i32; // h
            chan_major[8 + t] = t as i32; // w
        }
        let pos = Array::from_slice(&chan_major, &[3, 1, 4]);

        let (cos, sin) = build_cos_sin(&inv, &pos, &part).unwrap();
        assert_eq!(cos.shape(), &[1, 4, head_dim]);
        assert_eq!(sin.shape(), &[1, 4, head_dim]);

        // For text-only positions, MROPE output must equal 1-D RoPE output.
        // 1-D reference: freq = inv_freq * pos; emb = cat([freq, freq]); cos = cos(emb).
        // The merged cos should equal that 1-D cos exactly.
        let pos_1d = Array::from_slice(&[0_f32, 1.0, 2.0, 3.0], &[4]);
        let inv_1d = inv.as_dtype(Dtype::Float32).unwrap();
        let pos_1d_col = pos_1d.expand_dims(-1).unwrap();
        let inv_1d_row = inv_1d.reshape(&[1, -1]).unwrap();
        let freqs_1d = pos_1d_col.multiply(&inv_1d_row).unwrap();
        let emb_1d = ops::concatenate_axis(&[&freqs_1d, &freqs_1d], -1).unwrap();
        let cos_1d_ref = emb_1d.cos().unwrap().expand_dims(0).unwrap(); // (1, 4, head_dim)

        let diff = cos.subtract(&cos_1d_ref).unwrap().abs().unwrap();
        let max_abs = diff.max(None).unwrap();
        let max_abs_v = max_abs.as_dtype(Dtype::Float32).unwrap().item::<f32>();
        assert!(
            max_abs_v < 1e-5,
            "MROPE on text-only positions must equal 1-D RoPE; got max abs diff {max_abs_v}"
        );
    }

    /// Pure rotation preserves vector norm — apply MROPE to a random Q and
    /// the per-token norm must equal the input norm (within FP tolerance).
    #[test]
    fn preserves_norm() {
        let head_dim = 8_i32;
        let section = vec![1_i32, 1, 2];
        let part = MropePartition::new(&section, head_dim).unwrap();
        let inv = inv_freq(head_dim, 10_000.0);
        let mut chan_major = vec![0_i32; 3 * 5];
        for t in 0..5 {
            chan_major[t] = (t * 2) as i32;
            chan_major[5 + t] = t as i32;
            chan_major[10 + t] = (t + 1) as i32;
        }
        let pos = Array::from_slice(&chan_major, &[3, 1, 5]);
        let (cos, sin) = build_cos_sin(&inv, &pos, &part).unwrap();

        // Random q of shape (B=1, H=2, T=5, head_dim=8). Use a deterministic
        // fill so the test is reproducible.
        let q_data: Vec<f32> = (0..(2 * 5 * 8)).map(|i| (i as f32) * 0.1 - 1.5).collect();
        let q = Array::from_slice(&q_data, &[1, 2, 5, head_dim]);
        let k = q.clone();
        let (q_rot, _) = apply_rotary_qk(&q, &k, &cos, &sin).unwrap();

        // Per-token L2 norm: sum over (head_dim) then sqrt; compare against q's.
        let q2 = q.multiply(&q).unwrap();
        let q2_sum = q2.sum_axis(-1, false).unwrap();
        let qr2 = q_rot.multiply(&q_rot).unwrap();
        let qr2_sum = qr2.sum_axis(-1, false).unwrap();
        let diff = q2_sum.subtract(&qr2_sum).unwrap().abs().unwrap();
        let max_abs_v = diff
            .max(None)
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap()
            .item::<f32>();
        assert!(
            max_abs_v < 1e-3,
            "MROPE must preserve per-token L2 norm; got max abs ‖q‖² - ‖q_rot‖² = {max_abs_v}"
        );
    }

    /// Different image-token positions (h ≠ h' or w ≠ w', same t) must
    /// produce different rotations — i.e. spatial information is reaching
    /// the output. (Sanity check: a bug that ignored h/w channels would
    /// pass `text_only_collapses_to_1d_rope` but fail this one.)
    #[test]
    fn spatial_channels_matter() {
        let head_dim = 8_i32;
        let section = vec![1_i32, 1, 2];
        let part = MropePartition::new(&section, head_dim).unwrap();
        let inv = inv_freq(head_dim, 10_000.0);
        // Two SAME positions: both tokens at (t=5, h=0, w=0). Result for
        // both tokens should be identical along the T axis.
        let pos_same = {
            let buf = vec![5_i32, 5, 0, 0, 0, 0];
            Array::from_slice(&buf, &[3, 1, 2])
        };
        let (cos_same, _) = build_cos_sin(&inv, &pos_same, &part).unwrap();
        let row0 = cos_same.index((0, 0, ..));
        let row1 = cos_same.index((0, 1, ..));
        let same_diff = row0
            .subtract(&row1)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap()
            .item::<f32>();
        assert!(
            same_diff < 1e-6,
            "identical (t,h,w) must produce identical cos rows; got {same_diff}"
        );

        // Different h/w (same t): rows must differ.
        let pos_diff = {
            let buf = vec![5_i32, 5, 0, 1, 0, 2];
            Array::from_slice(&buf, &[3, 1, 2])
        };
        let (cos_diff, _) = build_cos_sin(&inv, &pos_diff, &part).unwrap();
        let r0 = cos_diff.index((0, 0, ..));
        let r1 = cos_diff.index((0, 1, ..));
        let diff = r0
            .subtract(&r1)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap()
            .item::<f32>();
        assert!(
            diff > 1e-3,
            "differing h/w must produce different cos rows (sanity); got max diff {diff}"
        );
    }

    /// Partition arithmetic: `2 * sum(section) == head_dim` required.
    #[test]
    fn partition_rejects_size_mismatch() {
        // sum * 2 = 12, head_dim = 16 → reject.
        let r = MropePartition::new(&[1, 1, 4], 16);
        assert!(r.is_err(), "must reject section/head_dim mismatch");
    }

    /// Tight TDD check against an explicit brute-force re-implementation of
    /// HF `apply_multimodal_rotary_pos_emb` for PaddleOCR-VL's actual config
    /// (`mrope_section=[16,24,24]`, `head_dim=128`, `rope_theta=500_000`).
    ///
    /// Brute-force formula (mirrors the HF Python line-for-line):
    ///   sections2 = [16, 24, 24, 16, 24, 24]   # mrope_section * 2
    ///   for each output dim `d` in [0, head_dim):
    ///       walk sections2 until `d` lands in chunk `j` (cumulative offset)
    ///       chan = j % 3                          # 0=t, 1=h, 2=w
    ///       pos  = position_ids[chan, b, t]
    ///       freq = pos * inv_freq[d % (head_dim/2)]
    ///       cos[b, t, d] = cos(freq) ; sin[b, t, d] = sin(freq)
    ///
    /// Tests a mix of text-only and image-grid positions to exercise every
    /// branch of the channel selector.
    #[test]
    fn matches_hf_brute_force_for_paddleocr_partition() {
        let head_dim = 128_i32;
        let section = vec![16_i32, 24, 24];
        let rope_theta = 500_000.0_f32;
        let part = MropePartition::new(&section, head_dim).unwrap();
        let inv = inv_freq(head_dim, rope_theta);

        // 6 tokens: 2 text-before, 3 image-grid (t=2, h=0..1, w=0..1 slice),
        // 1 text-after. Channel-major (3, B=1, T=6).
        // text-before: positions (0,0,0), (1,1,1)
        // image t=2: (2,2,2), (2,2,3), (2,3,2)
        // text-after: positions (4,4,4)
        let pos_data: Vec<i32> = vec![
            // t-channel
            0, 1, 2, 2, 2, 4,
            // h-channel
            0, 1, 2, 2, 3, 4,
            // w-channel
            0, 1, 2, 3, 2, 4,
        ];
        let position_ids = Array::from_slice(&pos_data, &[3, 1, 6]);

        let (cos_actual, sin_actual) =
            build_cos_sin(&inv, &position_ids, &part).unwrap();
        assert_eq!(cos_actual.shape(), &[1, 6, head_dim]);

        // Brute-force reference, host-side.
        let inv_f32 = inv
            .as_dtype(Dtype::Float32)
            .unwrap()
            .try_as_slice::<f32>()
            .unwrap()
            .to_vec();
        let half = (head_dim / 2) as usize;
        assert_eq!(inv_f32.len(), half);

        // For each output dim d, precompute which sections2 chunk it lives in.
        let sections2: Vec<i32> = section
            .iter()
            .copied()
            .chain(section.iter().copied())
            .collect();
        let mut dim_to_chan = vec![0i32; head_dim as usize];
        {
            let mut cum = 0i32;
            for (j, &sec) in sections2.iter().enumerate() {
                for d in cum..cum + sec {
                    dim_to_chan[d as usize] = (j % 3) as i32;
                }
                cum += sec;
            }
            assert_eq!(cum, head_dim);
        }

        let t_seq = 6usize;
        let mut cos_ref = vec![0_f32; t_seq * head_dim as usize];
        let mut sin_ref = vec![0_f32; t_seq * head_dim as usize];
        for t in 0..t_seq {
            for d in 0..head_dim as usize {
                let chan = dim_to_chan[d] as usize;
                let pos = pos_data[chan * t_seq + t] as f32;
                let freq = pos * inv_f32[d % half];
                let idx = t * head_dim as usize + d;
                cos_ref[idx] = freq.cos();
                sin_ref[idx] = freq.sin();
            }
        }
        let cos_ref_arr = Array::from_slice(&cos_ref, &[1, t_seq as i32, head_dim]);
        let sin_ref_arr = Array::from_slice(&sin_ref, &[1, t_seq as i32, head_dim]);

        let cos_max_abs = cos_actual
            .subtract(&cos_ref_arr)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap()
            .item::<f32>();
        let sin_max_abs = sin_actual
            .subtract(&sin_ref_arr)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap()
            .item::<f32>();
        assert!(
            cos_max_abs < 1e-5,
            "build_cos_sin must match HF brute-force on PaddleOCR partition; cos max abs diff {cos_max_abs}"
        );
        assert!(
            sin_max_abs < 1e-5,
            "build_cos_sin must match HF brute-force on PaddleOCR partition; sin max abs diff {sin_max_abs}"
        );
    }
}
