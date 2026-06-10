//! 3-D position-id construction for MROPE (GLM-OCR variant).
//!
//! Mirrors the paddleocr-vl-mlx port of `get_rope_index` but threads the
//! `temporal_patch_size` from the vision config: GLM-OCR ships
//! `temporal_patch_size = 2`, which means the image processor packs each
//! still image into a `T = 2` patch grid (the same patch is duplicated
//! across the temporal axis). After the spatial-merge projector that
//! collapses to `llm_t = T / temporal_patch_size = 1` time slot, so the
//! actual soft-token count per image becomes
//! `(T / temporal_patch_size) × (H / spatial_merge) × (W / spatial_merge)`.
//!
//! Token-stream walk:
//!   1. Text run → `(p, p, p)` for `p ∈ [next_pos, next_pos + text_len)`.
//!   2. Image run (the next `n_soft` `image_token_id`s) → `(t, h, w)`
//!      offset by `next_pos`.
//!   3. Any trailing text run resumes from `max(prior) + 1`.
//!
//! Output: `Array(int32, (3, T_seq))` ready to pass to
//! [`crate::mrope::build_cos_sin`].

use mlx_rs::Array;
use mlx_rs_core::error::{Error, Result};

/// `(T, H, W)` patch-grid (BEFORE spatial-merge / temporal collapse) for
/// one image. Matches `image_grid_thw` from the GLM-OCR processor.
pub type ImageGrid = (i32, i32, i32);

/// Build a `(3, T_seq)` position-id tensor.
///
/// `temporal_patch_size` and `spatial_merge_size` must match the vision
/// config — they determine how many soft tokens each image consumes.
pub fn build_position_ids_3d(
    input_ids: &[i32],
    image_token_id: i32,
    image_grids: &[ImageGrid],
    spatial_merge_size: i32,
    temporal_patch_size: i32,
) -> Result<Array> {
    if spatial_merge_size <= 0 || temporal_patch_size <= 0 {
        return Err(Error::InvalidConfig(format!(
            "spatial_merge_size / temporal_patch_size must be positive \
             (got merge={spatial_merge_size}, temporal={temporal_patch_size})"
        )));
    }
    let t_seq = input_ids.len();
    let mut t_chan = vec![0_i32; t_seq];
    let mut h_chan = vec![0_i32; t_seq];
    let mut w_chan = vec![0_i32; t_seq];

    let mut cur = 0usize;
    let mut next_pos = 0_i32;
    let mut img_idx = 0usize;

    while cur < t_seq {
        if input_ids[cur] != image_token_id {
            let mut end = cur;
            while end < t_seq && input_ids[end] != image_token_id {
                t_chan[end] = next_pos + (end - cur) as i32;
                h_chan[end] = t_chan[end];
                w_chan[end] = t_chan[end];
                end += 1;
            }
            next_pos += (end - cur) as i32;
            cur = end;
            continue;
        }

        if img_idx >= image_grids.len() {
            return Err(Error::InvalidConfig(format!(
                "build_position_ids_3d: ran out of image grids at input position \
                 {cur} (saw image_token_id but image_grids has {} entries)",
                image_grids.len()
            )));
        }
        let (gt, gh, gw) = image_grids[img_idx];
        if gt <= 0 || gh <= 0 || gw <= 0 {
            return Err(Error::InvalidConfig(format!(
                "build_position_ids_3d: image_grids[{img_idx}] = ({gt}, {gh}, {gw}) \
                 has non-positive components"
            )));
        }
        if gt % temporal_patch_size != 0 {
            return Err(Error::InvalidConfig(format!(
                "build_position_ids_3d: image_grids[{img_idx}].T = {gt} must be a \
                 multiple of temporal_patch_size = {temporal_patch_size}"
            )));
        }
        if gh % spatial_merge_size != 0 || gw % spatial_merge_size != 0 {
            return Err(Error::InvalidConfig(format!(
                "build_position_ids_3d: image_grids[{img_idx}] = ({gt}, {gh}, {gw}) \
                 — H and W must be multiples of spatial_merge_size = {spatial_merge_size}"
            )));
        }
        let llm_t = gt / temporal_patch_size;
        let llm_h = gh / spatial_merge_size;
        let llm_w = gw / spatial_merge_size;
        let n_soft = (llm_t * llm_h * llm_w) as usize;
        if cur + n_soft > t_seq {
            return Err(Error::InvalidConfig(format!(
                "build_position_ids_3d: image at input position {cur} expects \
                 {n_soft} soft tokens but only {} remain",
                t_seq - cur
            )));
        }
        for off in 0..n_soft {
            if input_ids[cur + off] != image_token_id {
                return Err(Error::InvalidConfig(format!(
                    "build_position_ids_3d: image at input position {cur} expects \
                     {n_soft} consecutive image_token_ids but input_ids[{}] = {} \
                     interrupts the run",
                    cur + off,
                    input_ids[cur + off]
                )));
            }
        }
        let mut p = 0usize;
        for t in 0..llm_t {
            for h in 0..llm_h {
                for w in 0..llm_w {
                    let slot = cur + p;
                    t_chan[slot] = next_pos + t;
                    h_chan[slot] = next_pos + h;
                    w_chan[slot] = next_pos + w;
                    p += 1;
                }
            }
        }
        let block_max = llm_t.max(llm_h).max(llm_w) - 1;
        next_pos += block_max + 1;
        cur += n_soft;
        img_idx += 1;
    }

    if img_idx != image_grids.len() {
        return Err(Error::InvalidConfig(format!(
            "build_position_ids_3d: consumed {img_idx} image grids but {} were supplied",
            image_grids.len()
        )));
    }

    let mut packed = Vec::with_capacity(3 * t_seq);
    packed.extend_from_slice(&t_chan);
    packed.extend_from_slice(&h_chan);
    packed.extend_from_slice(&w_chan);
    Ok(Array::from_slice(&packed, &[3, t_seq as i32]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::Dtype;

    fn to_vec(a: &Array) -> Vec<i32> {
        a.as_dtype(Dtype::Int32)
            .unwrap()
            .try_as_slice::<i32>()
            .unwrap()
            .to_vec()
    }

    #[test]
    fn text_only_sequence_yields_monotonic_positions() {
        let ids = vec![10, 11, 12, 13];
        let out = build_position_ids_3d(&ids, 99, &[], 2, 2).unwrap();
        assert_eq!(out.shape(), &[3, 4]);
        let v = to_vec(&out);
        assert_eq!(&v[0..4], &[0, 1, 2, 3]);
        assert_eq!(&v[4..8], &[0, 1, 2, 3]);
        assert_eq!(&v[8..12], &[0, 1, 2, 3]);
    }

    #[test]
    fn image_with_temporal_patch_size_two() {
        // GLM-OCR canonical: temporal_patch_size = 2, spatial_merge = 2.
        // Image grid (2, 4, 4): llm_t = 1, llm_h = 2, llm_w = 2 → 4 soft tokens.
        let img_tok = 99;
        let ids = vec![1, 2, img_tok, img_tok, img_tok, img_tok, 3];
        let grids = vec![(2_i32, 4_i32, 4_i32)];
        let out = build_position_ids_3d(&ids, img_tok, &grids, 2, 2).unwrap();
        let v = to_vec(&out);
        // text [0,1], image [2,2,2,2 / 2,2,3,3 / 2,3,2,3], trailing [4]
        // block_max = max(1,2,2) - 1 = 1, next_pos after image = 2 + 1 + 1 = 4
        assert_eq!(&v[0..7], &[0, 1, 2, 2, 2, 2, 4]);
        assert_eq!(&v[7..14], &[0, 1, 2, 2, 3, 3, 4]);
        assert_eq!(&v[14..21], &[0, 1, 2, 3, 2, 3, 4]);
    }

    #[test]
    fn rejects_grid_T_not_multiple_of_temporal_patch_size() {
        // temporal_patch_size=2 but grid_T=1 should fail.
        let img_tok = 99;
        let ids = vec![img_tok];
        let grids = vec![(1_i32, 2_i32, 2_i32)];
        let err = build_position_ids_3d(&ids, img_tok, &grids, 2, 2);
        assert!(err.is_err());
    }

    #[test]
    fn rejects_wrong_soft_token_count() {
        let img_tok = 99;
        // grid (2,4,4) expects 4 soft tokens but supplied 3.
        let ids = vec![10, img_tok, img_tok, img_tok, 11];
        let grids = vec![(2_i32, 4_i32, 4_i32)];
        let err = build_position_ids_3d(&ids, img_tok, &grids, 2, 2);
        assert!(err.is_err());
    }
}
