use std::cell::Cell;

use mlx_rs::{error::Exception, ops::indexing::IndexOp, ops::softmax_axis, Array, Dtype};

/// Outcome of distribution-preserving (speculative) acceptance.
#[derive(Debug, Clone, Copy)]
pub struct SpecAccept {
    /// Number of leading drafted tokens accepted (in `[0, drafted_count]`).
    pub n_accepted: usize,
    /// Token to emit after the accepted prefix: the residual `(p−δ)+` sample on
    /// a rejection, or a fresh bonus sample from the target distribution on full
    /// acceptance. Either way it is a draw from the target distribution, so the
    /// committed sequence is distributed exactly as the target would sample.
    pub correction: u32,
}

/// Distribution-preserving acceptance for a **deterministic (argmax) draft
/// proposal** — the regime DFlash's drafter operates in.
///
/// Standard speculative sampling proposes `x ~ q` and accepts with probability
/// `min(1, p(x)/q(x))`. When the proposal is deterministic (`q` is a point mass
/// at the drafted token `t`), this reduces to: accept `t` with probability
/// `p(t)`; on rejection sample the correction from the residual `p` with `t`'s
/// mass removed and renormalized. On full acceptance draw a bonus from `p` at
/// the position past the last draft. Each emitted token is then distributed
/// exactly as `p` (the target softmax at `temp`), unlike greedy acceptance
/// which biases accepted tokens toward the argmax.
///
/// `verify_logits` is `[1, R, vocab]` with `R >= drafted.len() + 1`. `temp`
/// must be > 0 (callers route `temp == 0` through the greedy path).
pub fn speculative_accept<R: FnMut() -> f32>(
    verify_logits: &Array,
    drafted: &[u32],
    temp: f32,
    rng: &mut R,
) -> Result<SpecAccept, Exception> {
    let dc = drafted.len();
    let vocab = *verify_logits.shape().last().unwrap_or(&0) as usize;
    // Rows 0..dc cover every accept test plus the bonus row at index dc.
    let sub = verify_logits.index((0, ..(dc as i32 + 1), ..)); // [dc+1, vocab]
    let probs = softmax_rows_fp32(&sub, temp)?;
    let flat = probs.as_slice::<f32>();

    for (i, &tok) in drafted.iter().enumerate() {
        let row = i * vocab;
        let t = tok as usize;
        if t >= vocab {
            return Err(Exception::custom(format!(
                "speculative_accept: draft token {t} out of vocab {vocab}"
            )));
        }
        let p_t = flat[row + t];
        if rng() < p_t {
            continue; // accept (q is a point mass at t, so ratio = p_t)
        }
        // Reject: residual = p with t removed, renormalized.
        let mut residual = Vec::with_capacity(vocab);
        let mut s = 0.0f32;
        for j in 0..vocab {
            let r = if j == t { 0.0 } else { flat[row + j] };
            residual.push(r);
            s += r;
        }
        let correction = if s > 0.0 {
            sample_from_unnormalized(&residual, s, rng)
        } else {
            // Degenerate (p was a point mass at t): emit t.
            t
        };
        return Ok(SpecAccept {
            n_accepted: i,
            correction: correction as u32,
        });
    }

    // Full acceptance: bonus ~ p at the position past the last draft (row dc).
    let row = dc * vocab;
    let row_p = &flat[row..row + vocab];
    let s: f32 = row_p.iter().sum();
    let bonus = sample_from_unnormalized(row_p, s, rng);
    Ok(SpecAccept {
        n_accepted: dc,
        correction: bonus as u32,
    })
}

/// Row-wise softmax in fp32 (`[R, vocab]` → fp32), applying `logits / temp`.
fn softmax_rows_fp32(logits: &Array, temp: f32) -> Result<Array, Exception> {
    let l32 = if logits.dtype() == Dtype::Float32 {
        logits.clone()
    } else {
        logits.as_dtype(Dtype::Float32)?
    };
    let scaled = if (temp - 1.0).abs() < f32::EPSILON {
        l32
    } else {
        let inv = Array::from_slice(&[1.0f32 / temp], &[1]);
        mlx_rs::ops::multiply(&l32, &inv)?
    };
    softmax_axis(&scaled, -1, true)
}

fn sample_from_unnormalized<R: FnMut() -> f32>(weights: &[f32], sum: f32, rng: &mut R) -> usize {
    let u = rng() * sum;
    let mut acc = 0.0f32;
    for (i, &w) in weights.iter().enumerate() {
        acc += w;
        if u < acc {
            return i;
        }
    }
    weights.len().saturating_sub(1)
}

thread_local! {
    static LCG_STATE: Cell<u64> = Cell::new(0x5EED_5EED_5EED_5EED);
}

/// Thread-local splitmix64 RNG returning `[0, 1)`. Deterministic per-thread
/// unless reseeded via [`seed_default_rng`].
pub fn default_rng() -> f32 {
    LCG_STATE.with(|cell| {
        let mut x = cell.get().wrapping_add(0x9E37_79B9_7F4A_7C15);
        cell.set(x);
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^= x >> 31;
        ((x >> 40) as f32) / ((1u64 << 24) as f32)
    })
}

/// Seed the thread-local RNG (tests rely on this for determinism).
pub fn seed_default_rng(seed: u64) {
    LCG_STATE.with(|cell| cell.set(seed.wrapping_add(1)));
}

/// Returns the count of the longest matching prefix between drafted and posterior tokens.
/// Uses cumprod(equal(drafted, posterior)) to find the longest prefix of 1s.
pub fn match_acceptance_length(drafted: &Array, posterior: &Array) -> Result<usize, Exception> {
    if drafted.shape().len() != 1 || posterior.shape().len() != 1 {
        return Err(Exception::custom(format!(
            "match_acceptance_length expects 1D arrays, got drafted={:?} posterior={:?}",
            drafted.shape(),
            posterior.shape()
        )));
    }
    if drafted.shape() != posterior.shape() {
        return Err(Exception::custom(format!(
            "match_acceptance_length shape mismatch: drafted={:?} posterior={:?}",
            drafted.shape(),
            posterior.shape()
        )));
    }
    if drafted.dtype() != Dtype::Uint32 || posterior.dtype() != Dtype::Uint32 {
        return Err(Exception::custom(format!(
            "match_acceptance_length expects u32 arrays, got drafted={:?} posterior={:?}",
            drafted.dtype(),
            posterior.dtype()
        )));
    }

    let matches = drafted.eq(posterior)?.as_dtype(Dtype::Uint32)?;
    let prefix = matches.cumprod(0, None, None)?;
    Ok(prefix.sum(false)?.item::<u32>() as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::Array;

    #[test]
    fn test_match_acceptance_length_exact_prefix() {
        let _guard = crate::mlx_test_guard();
        let drafted = Array::from_slice(&[11u32, 22, 33, 44], &[4]);
        let posterior = Array::from_slice(&[11u32, 22, 99, 44], &[4]);
        assert_eq!(match_acceptance_length(&drafted, &posterior).unwrap(), 2);
    }

    #[test]
    fn test_match_acceptance_length_all_match() {
        let _guard = crate::mlx_test_guard();
        let drafted = Array::from_slice(&[1u32, 2, 3], &[3]);
        let posterior = Array::from_slice(&[1u32, 2, 3], &[3]);
        assert_eq!(match_acceptance_length(&drafted, &posterior).unwrap(), 3);
    }

    // verify_logits laid out [1, R, vocab].
    fn verify_logits(rows: &[&[f32]]) -> Array {
        let r = rows.len();
        let v = rows[0].len();
        let mut buf = Vec::with_capacity(r * v);
        for row in rows {
            assert_eq!(row.len(), v);
            buf.extend_from_slice(row);
        }
        Array::from_slice(&buf, &[1, r as i32, v as i32])
    }

    #[test]
    fn speculative_accepts_when_target_favours_draft() {
        let _guard = crate::mlx_test_guard();
        // Row 0 puts ~all mass on id=1 (the draft) → accept. Row 1 = bonus.
        let vl = verify_logits(&[&[0.0, 20.0, 0.0], &[0.0, 0.0, 20.0]]);
        for s in 0..20u64 {
            seed_default_rng(s);
            let mut rng = default_rng;
            let r = speculative_accept(&vl, &[1], 1.0, &mut rng).unwrap();
            assert_eq!(r.n_accepted, 1, "p(draft)≈1 must accept");
            assert_eq!(r.correction, 2, "bonus must come from row dc (id=2)");
        }
    }

    #[test]
    fn speculative_rejects_and_corrects_from_residual() {
        let _guard = crate::mlx_test_guard();
        // Row 0 target favors id=2, draft proposed id=0 → p(0)≈0 → reject,
        // residual concentrates on id=2.
        let vl = verify_logits(&[&[0.0, 0.0, 20.0], &[0.0, 0.0, 0.0]]);
        let mut rejects = 0;
        let mut corr_two = 0;
        for s in 0..40u64 {
            seed_default_rng(s);
            let mut rng = default_rng;
            let r = speculative_accept(&vl, &[0], 1.0, &mut rng).unwrap();
            if r.n_accepted == 0 {
                rejects += 1;
                if r.correction == 2 {
                    corr_two += 1;
                }
            }
        }
        assert!(rejects >= 38, "p(draft)≈0 must mostly reject, got {rejects}");
        assert!(corr_two >= rejects - 1, "residual must concentrate on id=2");
    }
}
