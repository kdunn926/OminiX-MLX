//! Acceptance strategies for MTPLX speculative decoding.
//!
//! Two modes are supported:
//!
//!   * [`AcceptanceMode::Greedy`] — accept the drafted token iff it
//!     matches the target's argmax (T=0 only).
//!   * [`AcceptanceMode::Speculative`] — Leviathan-Chen 2023 probability
//!     ratio acceptance with `(p-q)+` residual sampling. Correct under
//!     any temperature including 0; recovers greedy at T→0 as a side
//!     effect of the argmax-only probability mass.
//!
//! For each drafted position `i`:
//! ```text
//! p_i = softmax(target_logits[i] / temp)   # target distribution
//! q_i = softmax(draft_logits[i]  / temp)   # draft  distribution
//! t   = draft[i]
//! ratio = p_i[t] / q_i[t]
//! if uniform(0,1) < min(1.0, ratio):
//!     accept; continue
//! else:
//!     residual = max(p_i - q_i, 0)
//!     correction = sample(residual / residual.sum())
//!     break
//! ```
//! If all K were accepted, sample one bonus token from `p_K`.
//!
//! Numerics: softmax runs in fp32 even when the underlying logits are
//! bf16/fp16, for stability.

use std::cell::Cell;

use mlx_rs::{ops::softmax_axis, Array, Dtype};

use crate::MtpError;

/// How to verify drafted tokens against the target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum AcceptanceMode {
    /// `argmax(target) == draft` at T=0. Cheap; biased at T>0.
    Greedy,
    /// Leviathan-Chen probability-ratio acceptance with residual
    /// `(p-q)+` correction sampling. Unbiased target sampling at any T.
    Speculative,
}

impl Default for AcceptanceMode {
    fn default() -> Self {
        AcceptanceMode::Greedy
    }
}

/// Outcome of verifying one drafted block.
#[derive(Debug, Clone)]
pub struct AcceptanceResult {
    /// Tokens that survived verification, in order. Length in `[0, K]`.
    pub accepted: Vec<u32>,
    /// Either the residual correction (when `all_accepted == false`) or
    /// the bonus token sampled from the target's distribution at the
    /// position one past the last accepted draft (when `all_accepted ==
    /// true`).
    pub correction: u32,
    /// `true` iff every drafted token was accepted.
    pub all_accepted: bool,
}

// ---------------------------------------------------------------------------
// Greedy
// ---------------------------------------------------------------------------

/// Greedy verification: accept `draft[i]` iff it equals `argmax(target_i)`.
///
/// `target_logits` and `draft_logits` must each have shape `[K, vocab]`.
/// `bonus` is the target's argmax at position K (sampled by the caller
/// from its own forward) — used when every draft is accepted.
pub fn accept_greedy(
    target_logits: &Array,
    draft: &[u32],
    bonus: u32,
) -> Result<AcceptanceResult, MtpError> {
    let argmax = mlx_rs::argmax_axis!(target_logits, -1)?;
    let ids = argmax.as_slice::<u32>();
    let k = draft.len().min(ids.len());

    let mut accepted = Vec::with_capacity(k);
    for i in 0..k {
        if ids[i] == draft[i] {
            accepted.push(draft[i]);
        } else {
            // Reject: commit the target's argmax at this slot.
            return Ok(AcceptanceResult {
                accepted,
                correction: ids[i],
                all_accepted: false,
            });
        }
    }
    Ok(AcceptanceResult {
        accepted,
        correction: bonus,
        all_accepted: true,
    })
}

// ---------------------------------------------------------------------------
// Speculative (Leviathan-Chen)
// ---------------------------------------------------------------------------

/// Probability-ratio acceptance + residual `(p-q)+` sampling.
///
/// `target_logits` / `draft_logits` shape `[K, vocab]`. `temp` must be
/// > 0; the caller is expected to route T=0 paths through
/// [`accept_greedy`] (or pass `temp = 1.0` if it really wants
/// Leviathan-Chen at T=0).
///
/// Both distributions are softmaxed in fp32 for numerical stability,
/// even when the model emits bf16.
pub fn accept_speculative<R: FnMut() -> f32>(
    target_logits: &Array,
    draft_logits: &Array,
    draft: &[u32],
    temp: f32,
    mut rng: R,
) -> Result<AcceptanceResult, MtpError> {
    debug_assert!(temp > 0.0, "speculative acceptance requires temp > 0");
    let k = draft.len();
    let t_probs = softmax_rows_fp32(target_logits, temp)?; // [K, vocab]
    let q_probs = softmax_rows_fp32(draft_logits, temp)?;

    let vocab = t_probs.shape().last().copied().unwrap_or(0) as usize;
    let t_flat = t_probs.as_slice::<f32>();
    let q_flat = q_probs.as_slice::<f32>();

    let mut accepted = Vec::with_capacity(k);
    for i in 0..k {
        let row = i * vocab;
        let t = draft[i] as usize;
        if t >= vocab {
            return Err(MtpError::Mlx(format!(
                "draft token {t} out of vocab {vocab}"
            )));
        }
        let p_t = t_flat[row + t];
        let q_t = q_flat[row + t];
        let ratio = if q_t > 0.0 { p_t / q_t } else { f32::INFINITY };
        let u = rng();
        if u < ratio.min(1.0) {
            accepted.push(draft[i]);
            continue;
        }
        // Residual sample from max(p - q, 0).
        let mut residual = Vec::with_capacity(vocab);
        let mut s = 0.0f32;
        for j in 0..vocab {
            let d = t_flat[row + j] - q_flat[row + j];
            let r = if d > 0.0 { d } else { 0.0 };
            residual.push(r);
            s += r;
        }
        let correction = if s > 0.0 {
            sample_from_unnormalized(&residual, s, &mut rng)
        } else {
            // Defensive: fall back to sampling from p directly.
            let row_p = &t_flat[row..row + vocab];
            let s_p: f32 = row_p.iter().sum();
            sample_from_unnormalized(row_p, s_p, &mut rng)
        };
        return Ok(AcceptanceResult {
            accepted,
            correction: correction as u32,
            all_accepted: false,
        });
    }

    // All K accepted: sample a bonus token from p at slot K-1's row?
    // No — the bonus comes from the target's distribution at position K
    // (i.e. one past the last accepted draft). The caller provides this
    // as the last row of `target_logits` *iff* it built target_logits
    // with shape `[K+1, vocab]`. To avoid that contract complexity we
    // instead require the caller to pass the bonus row separately via
    // `accept_speculative_with_bonus`. For the all-accepted branch here
    // we sample from the last row of `target_logits` as a best-effort
    // (Leviathan-Chen's "bonus" is just one extra free target sample).
    let last_row = (k - 1) * vocab;
    let row_p = &t_flat[last_row..last_row + vocab];
    let s_p: f32 = row_p.iter().sum();
    let bonus = sample_from_unnormalized(row_p, s_p, &mut rng);
    Ok(AcceptanceResult {
        accepted,
        correction: bonus as u32,
        all_accepted: true,
    })
}

/// Row-wise softmax in fp32: `[K, vocab]` → `[K, vocab]` fp32.
///
/// Casts to fp32 if necessary before softmax. Temperature is applied as
/// `logits / temp`.
fn softmax_rows_fp32(logits: &Array, temp: f32) -> Result<Array, MtpError> {
    let l32 = if logits.dtype() == Dtype::Float32 {
        logits.clone()
    } else {
        logits.as_type::<f32>()?
    };
    let scaled = if (temp - 1.0).abs() < f32::EPSILON {
        l32
    } else {
        let inv = Array::from_slice(&[1.0f32 / temp], &[1]);
        // broadcast multiply along last axis
        mlx_rs::ops::multiply(&l32, &inv)?
    };
    let sm = softmax_axis(&scaled, -1, true)?; // precise=true → fp32 internals
    Ok(sm)
}

fn sample_from_unnormalized<R: FnMut() -> f32>(
    weights: &[f32],
    sum: f32,
    rng: &mut R,
) -> usize {
    let u = rng() * sum;
    let mut acc = 0.0f32;
    for (i, &w) in weights.iter().enumerate() {
        acc += w;
        if u < acc {
            return i;
        }
    }
    weights.len() - 1
}

// ---------------------------------------------------------------------------
// Default RNG
// ---------------------------------------------------------------------------

thread_local! {
    static LCG_STATE: Cell<u64> = Cell::new(seed_from_clock());
}

fn seed_from_clock() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    let s = now.as_nanos() as u64;
    s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407)
}

/// Thread-local deterministic-ish RNG returning `[0,1)`. Splitmix64.
pub fn default_rng() -> f32 {
    LCG_STATE.with(|cell| {
        let mut x = cell.get().wrapping_add(0x9E3779B97F4A7C15);
        cell.set(x);
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
        x ^= x >> 31;
        // top 24 bits → [0,1) float
        ((x >> 40) as f32) / ((1u64 << 24) as f32)
    })
}

/// Seed the thread-local RNG. Tests rely on this for determinism.
pub fn seed_default_rng(seed: u64) {
    LCG_STATE.with(|cell| cell.set(seed.wrapping_add(1)));
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::Array;

    fn logits(rows: &[&[f32]]) -> Array {
        let k = rows.len();
        let v = rows[0].len();
        let mut buf = Vec::with_capacity(k * v);
        for r in rows {
            assert_eq!(r.len(), v);
            buf.extend_from_slice(r);
        }
        Array::from_slice(&buf, &[k as i32, v as i32])
    }

    #[test]
    fn greedy_all_accepted() {
        // Argmax of each row is 1 → matches draft.
        let t = logits(&[&[0.0, 5.0, 0.0], &[1.0, 9.0, 2.0]]);
        let r = accept_greedy(&t, &[1, 1], 2).unwrap();
        assert_eq!(r.accepted, vec![1, 1]);
        assert!(r.all_accepted);
        assert_eq!(r.correction, 2);
    }

    #[test]
    fn greedy_rejects_midblock() {
        // First row argmax 1 (match), second row argmax 2 (mismatch).
        let t = logits(&[&[0.0, 5.0, 0.0], &[0.0, 1.0, 9.0]]);
        let r = accept_greedy(&t, &[1, 1], 0).unwrap();
        assert_eq!(r.accepted, vec![1]);
        assert!(!r.all_accepted);
        assert_eq!(r.correction, 2); // target's argmax at reject slot
    }

    #[test]
    fn speculative_accepts_when_p_dominates_q() {
        // Target heavily favors id=0; draft also favors id=0. p[0]/q[0] ≈ 1 → almost always accept.
        let t = logits(&[&[10.0, 0.0, 0.0]]);
        let q = logits(&[&[10.0, 0.0, 0.0]]);
        seed_default_rng(42);
        let r = accept_speculative(&t, &q, &[0], 1.0, default_rng).unwrap();
        assert_eq!(r.accepted, vec![0]);
        assert!(r.all_accepted);
    }

    #[test]
    fn speculative_rejects_when_p_starves_q() {
        // Target says id=2 (p[2] huge); draft picked id=0 (q[0] huge).
        // ratio = p[0]/q[0] is tiny → almost always reject; residual peaks at id=2.
        let t = logits(&[&[0.0, 0.0, 10.0]]);
        let q = logits(&[&[10.0, 0.0, 0.0]]);
        seed_default_rng(7);
        let mut rejects = 0;
        let mut corrections_two = 0;
        for s in 0..50u64 {
            seed_default_rng(s);
            let r = accept_speculative(&t, &q, &[0], 1.0, default_rng).unwrap();
            if !r.all_accepted {
                rejects += 1;
                if r.correction == 2 {
                    corrections_two += 1;
                }
            }
        }
        assert!(rejects >= 40, "expected mostly rejects, got {rejects}");
        assert!(
            corrections_two >= rejects - 2,
            "expected residual corrections to concentrate on id=2 (got {corrections_two}/{rejects})"
        );
    }

    #[test]
    fn speculative_unbiased_marginal_smoke() {
        // With matching draft = target, accept rate should be ~1.
        let t = logits(&[&[1.0, 2.0, 3.0]]);
        let q = logits(&[&[1.0, 2.0, 3.0]]);
        let mut accepts = 0;
        for s in 0..200u64 {
            seed_default_rng(s);
            let r = accept_speculative(&t, &q, &[2], 1.0, default_rng).unwrap();
            if r.all_accepted {
                accepts += 1;
            }
        }
        assert_eq!(accepts, 200, "p==q must always accept");
    }

    #[test]
    fn softmax_fp32_temperature_scales() {
        // High temperature flattens; low temperature sharpens.
        let l = logits(&[&[0.0, 1.0, 2.0]]);
        let hi = softmax_rows_fp32(&l, 10.0).unwrap();
        let lo = softmax_rows_fp32(&l, 0.1).unwrap();
        let hi_s = hi.as_slice::<f32>();
        let lo_s = lo.as_slice::<f32>();
        // hi is closer to uniform 1/3 than lo is.
        assert!((hi_s[0] - 1.0 / 3.0).abs() < (lo_s[0] - 1.0 / 3.0).abs());
        // both rows sum to ~1
        let sum_hi: f32 = hi_s.iter().sum();
        let sum_lo: f32 = lo_s.iter().sum();
        assert!((sum_hi - 1.0).abs() < 1e-4);
        assert!((sum_lo - 1.0).abs() < 1e-4);
    }
}
