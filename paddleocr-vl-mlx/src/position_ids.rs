//! 3-D position-id construction for MROPE.
//!
//! Port of `PaddleOCRVLForConditionalGeneration.get_rope_index`
//! (`modeling_paddleocr_vl.py:1960`) — images-only, single-batch slice
//! that's sufficient for the chat path. Walks the token stream:
//!
//!   1. Text run → `(p, p, p)` for `p ∈ [st_idx, st_idx + text_len)`.
//!   2. Image run (the next `(H/m) × (W/m) × T` `image_token_id`s, for an
//!      image with patch-grid `(T, H, W)` and `m = spatial_merge_size`) →
//!      `(0, h, w)` per soft-token position, offset so the image starts
//!      right after the preceding text.
//!   3. Any trailing text run → resumes from `max(prior) + 1`.
//!
//! Output: `Array(int32, (3, T))` ready to pass to
//! [`crate::mrope::build_cos_sin`].

use mlx_rs::Array;
use mlx_rs_core::error::{Error, Result};

/// `(T, H, W)` patch-grid (BEFORE spatial-merge collapse) for one image.
/// Matches the image processor's `image_grid_thw` output and the HF
/// reference's `image_grid_thw[i]` indexing.
pub type ImageGrid = (i32, i32, i32);

/// Build a `(3, T_seq)` position-id tensor for an `input_ids` sequence that
/// contains `image_grids.len()` images.
///
/// Each image must be represented by exactly `(T·H·W)/m²` consecutive
/// `image_token_id` tokens in the stream (one per soft token after the
/// projector's 2x2 spatial merge). If the counts don't line up the function
/// returns `Error::InvalidConfig`.
pub fn build_position_ids_3d(
    input_ids: &[i32],
    image_token_id: i32,
    image_grids: &[ImageGrid],
    spatial_merge_size: i32,
) -> Result<Array> {
    if spatial_merge_size <= 0 {
        return Err(Error::InvalidConfig(format!(
            "spatial_merge_size must be positive, got {spatial_merge_size}"
        )));
    }
    let t_seq = input_ids.len();
    let mut t_chan = vec![0_i32; t_seq];
    let mut h_chan = vec![0_i32; t_seq];
    let mut w_chan = vec![0_i32; t_seq];

    let mut cur = 0usize; // next slot to fill in input_ids
    let mut next_pos = 0_i32; // next free `p` for text runs / image t-offset
    let mut img_idx = 0usize;

    while cur < t_seq {
        if input_ids[cur] != image_token_id {
            // Walk the contiguous text run.
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

        // Image run starts at `cur`. Compute its soft-token count from the
        // grid; we expect `cur..cur+expected` to all be `image_token_id`.
        if img_idx >= image_grids.len() {
            return Err(Error::InvalidConfig(format!(
                "build_position_ids_3d: ran out of image grids at input position {cur} \
                 (saw image_token_id but image_grids has {} entries)",
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
        if gh % spatial_merge_size != 0 || gw % spatial_merge_size != 0 {
            return Err(Error::InvalidConfig(format!(
                "build_position_ids_3d: image_grids[{img_idx}] = ({gt}, {gh}, {gw}) — \
                 H and W must be multiples of spatial_merge_size = {spatial_merge_size}"
            )));
        }
        let llm_t = gt;
        let llm_h = gh / spatial_merge_size;
        let llm_w = gw / spatial_merge_size;
        let n_soft = (llm_t * llm_h * llm_w) as usize;
        if cur + n_soft > t_seq {
            return Err(Error::InvalidConfig(format!(
                "build_position_ids_3d: image at input position {cur} expects {n_soft} \
                 soft tokens but only {} remain",
                t_seq - cur
            )));
        }
        for off in 0..n_soft {
            if input_ids[cur + off] != image_token_id {
                return Err(Error::InvalidConfig(format!(
                    "build_position_ids_3d: image at input position {cur} expects {n_soft} \
                     consecutive image_token_ids but input_ids[{}] = {} interrupts the run",
                    cur + off,
                    input_ids[cur + off]
                )));
            }
        }

        // Fill (t, h, w) positions. For PaddleOCR-VL with T=1 and no time
        // coordinate (`tokens_per_second = 0`), `t_index` is all zeros;
        // we still write it for shape symmetry (T>1 future-proofing).
        // Layout matches the HF reference's torch.arange.view/expand:
        //   t_index : 0,0,…,0 (each value repeated llm_h * llm_w times)
        //   h_index : 0,0,…,0,1,1,…,1,…,llm_h-1,llm_h-1,…,llm_h-1  (each h × llm_w)
        //   w_index : 0,1,…,llm_w-1, 0,1,…,llm_w-1, … (cycling llm_w, repeating llm_h × llm_t times)
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

    // Pack into (3, T) channel-major layout: [t-channel, h-channel, w-channel].
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
        // No images → all three channels carry the same 0,1,2,…
        let ids = vec![10, 11, 12, 13];
        let out = build_position_ids_3d(&ids, 99, &[], 2).unwrap();
        assert_eq!(out.shape(), &[3, 4]);
        let v = to_vec(&out);
        // (3, 4) channel-major: [t0 t1 t2 t3 | h0 … | w0 …]
        assert_eq!(&v[0..4], &[0, 1, 2, 3]);
        assert_eq!(&v[4..8], &[0, 1, 2, 3]);
        assert_eq!(&v[8..12], &[0, 1, 2, 3]);
    }

    #[test]
    fn one_image_in_the_middle_offsets_trailing_text() {
        // 2 text tokens, 1 image (grid 1×4×4 → llm 1×2×2 = 4 soft tokens),
        // 1 trailing text token. The trailing text resumes from
        // max(prior) + 1 = max(2-pos prefix max=1, image max=1) + 1 = 2.
        let img_tok = 99;
        let ids = vec![10, 11, img_tok, img_tok, img_tok, img_tok, 12];
        let grids = vec![(1_i32, 4_i32, 4_i32)];
        let out = build_position_ids_3d(&ids, img_tok, &grids, 2).unwrap();
        let v = to_vec(&out);
        // T axis = 0,1, (image: all 0+text_len=2 → 2,2,2,2), (trailing: max+1)
        // text_len = 2, st_idx = 0
        // text run: positions 0, 1 → t=h=w = 0, 1
        // image run: llm_t=1, llm_h=2, llm_w=2, offset = text_len + st_idx = 2
        //   t_index: [0,0,0,0] + 2 → [2,2,2,2]
        //   h_index: [0,0,1,1] + 2 → [2,2,3,3]
        //   w_index: [0,1,0,1] + 2 → [2,3,2,3]
        // After image: max position = max(2,3,3) = 3, next_pos = 3 + 1 = 4
        //   wait — code uses next_pos += max(llm_t, llm_h, llm_w). For llm=(1,2,2) max=2.
        //   So next_pos = 2 + 2 = 4.
        // trailing text: 1 token at position 4 → t=h=w=4
        assert_eq!(&v[0..7],  &[0, 1, 2, 2, 2, 2, 4]); // t-channel
        assert_eq!(&v[7..14], &[0, 1, 2, 2, 3, 3, 4]); // h-channel
        assert_eq!(&v[14..21], &[0, 1, 2, 3, 2, 3, 4]); // w-channel
    }

    #[test]
    fn rejects_wrong_image_token_count() {
        // Grid 1×4×4 ⇒ expects 4 soft tokens but we only supply 3.
        let img_tok = 99;
        let ids = vec![10, img_tok, img_tok, img_tok, 11];
        let grids = vec![(1_i32, 4_i32, 4_i32)];
        let err = build_position_ids_3d(&ids, img_tok, &grids, 2);
        assert!(err.is_err(), "wrong soft-token count must be rejected");
    }

    #[test]
    fn rejects_grid_not_multiple_of_merge_size() {
        // H=3 with merge_size=2 is malformed.
        let img_tok = 99;
        let ids = vec![img_tok, img_tok, img_tok, img_tok, img_tok, img_tok];
        let grids = vec![(1_i32, 3_i32, 4_i32)];
        let err = build_position_ids_3d(&ids, img_tok, &grids, 2);
        assert!(err.is_err(), "grid not divisible by merge_size must be rejected");
    }

    #[test]
    fn rejects_unused_or_extra_image_grids() {
        // Supplied 2 grids but only 1 image in stream → leftover grid.
        let img_tok = 99;
        let ids = vec![img_tok, img_tok, img_tok, img_tok]; // 1 image of grid 1×2×2 → 1 soft tok?
        // Actually 1 image of grid (1, 2, 2)/m=2 → llm 1×1×1 = 1 soft tok. So 4 image tokens
        // would imply 4 separate (1, 2, 2) images. Use 2 grids for 4 tokens.
        let grids = vec![(1, 2, 2), (1, 2, 2), (1, 2, 2)]; // 3 grids, only 2 fit
        let err = build_position_ids_3d(&ids, img_tok, &grids, 2);
        assert!(err.is_err(), "extra image grids must be rejected");
    }

    #[test]
    fn two_text_runs_around_image_are_continuous() {
        // 3 text tokens, image (1×2×2 → 1 soft), 2 text tokens.
        // Positions: text=[0,1,2], image=[3,3,3], text=[4,5]
        let img_tok = 99;
        let ids = vec![1, 2, 3, img_tok, 4, 5];
        let grids = vec![(1, 2, 2)];
        let out = build_position_ids_3d(&ids, img_tok, &grids, 2).unwrap();
        let v = to_vec(&out);
        // All three channels identical here since llm_grid is 1×1×1
        // (the image's t/h/w are all 0; offset = 3 → all (3,3,3)).
        assert_eq!(&v[0..6], &[0, 1, 2, 3, 4, 5]);
        assert_eq!(&v[6..12], &[0, 1, 2, 3, 4, 5]);
        assert_eq!(&v[12..18], &[0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn image_at_sequence_start() {
        // No leading text. Image positions start at next_pos = 0.
        let img_tok = 99;
        let ids = vec![img_tok, img_tok, img_tok, img_tok, 1, 2];
        let grids = vec![(1, 4, 4)]; // llm 2×2 → 4 soft tokens
        let out = build_position_ids_3d(&ids, img_tok, &grids, 2).unwrap();
        let v = to_vec(&out);
        // image: t=[0,0,0,0], h=[0,0,1,1], w=[0,1,0,1]
        // trailing text: positions 2, 3
        assert_eq!(&v[0..6], &[0, 0, 0, 0, 2, 3]);
        assert_eq!(&v[6..12], &[0, 0, 1, 1, 2, 3]);
        assert_eq!(&v[12..18], &[0, 1, 0, 1, 2, 3]);
    }
}
