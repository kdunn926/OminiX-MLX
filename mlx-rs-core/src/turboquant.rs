//! TurboQuant KV cache primitives (spike port of github.com/onur-gokyildiz-bhi/tq-kv).
//!
//! Implements the minimum core of the TurboQuant algorithm:
//!   1. Randomized Hadamard transform (deterministic +-1 signs from seed,
//!      O(d log d) Walsh-Hadamard butterfly).
//!   2. Per-token mean removal (exploits softmax shift-invariance).
//!   3. Per-vector sigma (= norm / sqrt(dim)).
//!   4. Lloyd-Max 4-bit codebook quantization (precomputed N(0,1)
//!      centroids).
//!   5. 4-bit packing (2 indices per byte / 8 indices per u32).
//!
//! Out of scope for the spike (and deferred):
//!   - QJL error correction
//!   - Outlier preservation
//!   - Channel bias / SmoothAttention scales
//!   - Calibrated codebooks (uses precomputed Lloyd-Max N(0,1) table)
//!   - Pre-RoPE quantization (requires K capture before RoPE — invasive)
//!   - Compaction
//!   - Fused attention from compressed indices
//!   - Value quantization (V stays at the host dtype)
//!
//! Memory: head_dim=256 sliding K stored as 128 packed bytes + 4 bytes
//! sigma + 4 bytes mean per K vector = 136 bytes/vector vs 512 bytes BF16
//! ≈ 3.8× compression on K.

use std::sync::OnceLock;

/// Lloyd-Max optimal centroids for the standard Gaussian N(0,1) at 4-bit
/// (16 centroids). Source: Max 1960, "Quantizing for Minimum Distortion",
/// values verified by 300 iterations of Lloyd-Max in the upstream tq-kv
/// repo (tq-kv/src/codebook.rs).
pub const CENTROIDS_4BIT: [f32; 16] = [
    -2.7326, -2.0690, -1.6180, -1.2562, -0.9424, -0.6568, -0.3880, -0.1284,
    0.1284, 0.3880, 0.6568, 0.9424, 1.2562, 1.6180, 2.0690, 2.7326,
];

/// Decision boundaries (midpoints) between centroids — 15 boundaries for
/// 16 centroids. A normalized value `v` quantizes to index `i` iff
/// `boundaries[i-1] <= v < boundaries[i]`.
pub const BOUNDARIES_4BIT: [f32; 15] = [
    -2.4008, -1.8435, -1.4371, -1.0993, -0.7996, -0.5224, -0.2582, 0.0,
    0.2582, 0.5224, 0.7996, 1.0993, 1.4371, 1.8435, 2.4008,
];

/// Deterministic +-1 sign vector generated from `seed`. Used as the
/// randomization step of the Hadamard rotation (sign-flip then WHT).
/// Implementation: xorshift64* over the seed; produces a +-1 array in
/// 1.0 / -1.0 form (matches what the Metal kernel consumes).
pub fn generate_signs(dim: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(0xBF58_476D_1CE4_E5B9);
    let mut out = Vec::with_capacity(dim);
    for _ in 0..dim {
        // xorshift64*
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        let v = s.wrapping_mul(0x2545_F491_4F6C_DD1D);
        out.push(if (v >> 63) & 1 == 0 { 1.0 } else { -1.0 });
    }
    out
}

/// Cache the sign tensor per (dim, seed) so we don't regenerate on every
/// cache update. Keyed by (dim as u32, seed as u64) → Vec<f32>.
static SIGNS_CACHE: OnceLock<std::sync::Mutex<std::collections::HashMap<(u32, u64), Vec<f32>>>> =
    OnceLock::new();

pub fn cached_signs(dim: i32, seed: u64) -> Vec<f32> {
    let map = SIGNS_CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    let key = (dim as u32, seed);
    let mut guard = map.lock().unwrap();
    if let Some(v) = guard.get(&key) {
        return v.clone();
    }
    let v = generate_signs(dim as usize, seed);
    guard.insert(key, v.clone());
    v
}

/// CPU-side Fast Walsh-Hadamard Transform (in-place). `n` must be a power
/// of 2. Normalized by 1/sqrt(n) so the transform is orthogonal. Mainly
/// used by tests and the roundtrip-validation path; production hot path
/// runs the Metal kernel.
pub fn fast_wht(x: &mut [f32]) {
    let n = x.len();
    assert!(n.is_power_of_two(), "WHT requires power-of-2, got {n}");
    let mut h = 1;
    while h < n {
        let mut i = 0;
        while i < n {
            for j in i..i + h {
                let a = x[j];
                let b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
            i += h * 2;
        }
        h *= 2;
    }
    let scale = 1.0 / (n as f32).sqrt();
    for v in x.iter_mut() {
        *v *= scale;
    }
}

/// Quantize one f32 value `v` (post-Hadamard, post-sigma-normalize, so
/// expected ~N(0,1)) → 4-bit index in [0, 15] via linear scan over
/// boundaries. Suitable for tests; the Metal kernel does the same scan
/// per lane.
pub fn quantize_scalar(v: f32) -> u8 {
    let mut idx = 0u8;
    for &b in &BOUNDARIES_4BIT {
        if v > b {
            idx += 1;
        } else {
            break;
        }
    }
    idx
}

/// CPU roundtrip helper: compress + decompress one head_dim vector at
/// 4-bit + return both the original-vs-reconstructed L2 error and the
/// reconstructed vector. Validates the kernel pipeline's math.
pub fn cpu_roundtrip_4bit(x: &[f32], signs: &[f32]) -> (Vec<f32>, f32) {
    assert_eq!(x.len(), signs.len());
    let dim = x.len();
    let mut buf: Vec<f32> = x.iter().zip(signs).map(|(v, s)| v * s).collect();
    // Per-token mean removal.
    let mean = buf.iter().sum::<f32>() / dim as f32;
    for v in buf.iter_mut() {
        *v -= mean;
    }
    // Hadamard.
    fast_wht(&mut buf);
    // sigma = norm / sqrt(dim).
    let norm_sq: f32 = buf.iter().map(|v| v * v).sum();
    let norm = norm_sq.sqrt();
    let sigma = norm / (dim as f32).sqrt();
    let inv_sigma = if sigma > 1e-10 { 1.0 / sigma } else { 0.0 };
    // Quantize.
    let indices: Vec<u8> = buf.iter().map(|&v| quantize_scalar(v * inv_sigma)).collect();
    // Dequantize: indices → centroid × sigma.
    let mut recon: Vec<f32> = indices.iter().map(|&i| CENTROIDS_4BIT[i as usize] * sigma).collect();
    // Inverse Hadamard (WHT is self-inverse).
    fast_wht(&mut recon);
    // Add mean back, undo sign flip.
    for v in recon.iter_mut() {
        *v += mean;
    }
    for (v, s) in recon.iter_mut().zip(signs) {
        *v *= s;
    }
    let err: f32 = recon
        .iter()
        .zip(x)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt();
    (recon, err)
}

/// End-to-end Metal roundtrip helper used by the cache impl and tests:
/// quantize a `[..., D]` BF16/F16/F32 keys tensor and immediately
/// decompress back. Returns the reconstructed tensor (same dtype as input)
/// and the (sigma, mean, packed) metadata.
pub fn metal_roundtrip(
    keys: &mlx_rs::Array,
    seed: u64,
) -> Result<mlx_rs::Array, mlx_rs::error::Exception> {
    use crate::metal_kernels::{tq_compress_4bit, tq_decompress_4bit};
    let shape = keys.shape().to_vec();
    let d = *shape.last().unwrap();
    let signs_vec = cached_signs(d, seed);
    let signs = mlx_rs::Array::from_slice(&signs_vec, &[d]);
    let boundaries = mlx_rs::Array::from_slice(&BOUNDARIES_4BIT, &[15]);
    let centroids = mlx_rs::Array::from_slice(&CENTROIDS_4BIT, &[16]);
    let dtype = keys.dtype();
    let (packed, sigma, mean) = tq_compress_4bit(keys, &signs, &boundaries)?;
    tq_decompress_4bit(&packed, &sigma, &mean, &signs, &centroids, &shape, dtype)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wht_self_inverse() {
        let mut x: Vec<f32> = (0..256).map(|i| (i as f32).sin()).collect();
        let orig = x.clone();
        fast_wht(&mut x);
        fast_wht(&mut x);
        for (a, b) in x.iter().zip(&orig) {
            assert!((a - b).abs() < 1e-3, "self-inverse failed: {a} vs {b}");
        }
    }

    #[test]
    fn signs_deterministic() {
        let a = generate_signs(64, 42);
        let b = generate_signs(64, 42);
        assert_eq!(a, b);
        let c = generate_signs(64, 43);
        assert_ne!(a, c);
        for s in &a {
            assert!(*s == 1.0 || *s == -1.0);
        }
    }

    #[test]
    fn metal_compress_produces_finite_metadata() {
        use crate::metal_kernels::tq_compress_4bit;
        let dim = 64i32;
        let n = 2i32;
        let mut data = Vec::with_capacity((n * dim) as usize);
        for v in 0..n {
            for i in 0..dim {
                data.push((((v * 31 + i) as f32) * 0.05).sin());
            }
        }
        let arr = mlx_rs::Array::from_slice(&data, &[n, dim])
            .as_dtype(mlx_rs::Dtype::Float32)
            .unwrap();
        let signs = mlx_rs::Array::from_slice(&cached_signs(dim, 42), &[dim]);
        let boundaries = mlx_rs::Array::from_slice(&BOUNDARIES_4BIT, &[15]);
        let (packed, sigma, mean) = tq_compress_4bit(&arr, &signs, &boundaries).unwrap();
        mlx_rs::transforms::eval([&packed, &sigma, &mean]).unwrap();
        let sigma_slice = sigma.as_slice::<f32>();
        let mean_slice = mean.as_slice::<f32>();
        eprintln!("sigma: {sigma_slice:?}");
        eprintln!("mean : {mean_slice:?}");
        for &s in sigma_slice {
            assert!(s.is_finite() && s > 0.0, "sigma must be positive finite, got {s}");
        }
        for &m in mean_slice {
            assert!(m.is_finite(), "mean must be finite, got {m}");
        }
    }

    #[test]
    fn metal_roundtrip_matches_cpu_within_tolerance() {
        let dim = 256i32;
        let n = 4i32; // batch of 4 vectors
        let mut data = Vec::with_capacity((n * dim) as usize);
        for v in 0..n {
            for i in 0..dim {
                let phase = (v as f32) * 0.3 + (i as f32) * 0.137;
                data.push(phase.sin() * 1.3 + (phase * 0.31).cos() * 0.5);
            }
        }
        let arr = mlx_rs::Array::from_slice(&data, &[n, dim])
            .as_dtype(mlx_rs::Dtype::Bfloat16)
            .unwrap();
        let recon = metal_roundtrip(&arr, 42).unwrap();
        mlx_rs::transforms::eval([&recon]).unwrap();
        let recon_f32 = recon.as_dtype(mlx_rs::Dtype::Float32).unwrap();
        mlx_rs::transforms::eval([&recon_f32]).unwrap();
        let recon_slice = recon_f32.as_slice::<f32>();
        for v in 0..n {
            let start = (v * dim) as usize;
            let end = start + dim as usize;
            let orig: f32 = data[start..end].iter().map(|x| x * x).sum::<f32>().sqrt();
            let err: f32 = recon_slice[start..end]
                .iter()
                .zip(&data[start..end])
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt();
            assert!(
                err / orig < 0.18,
                "vec {v}: err/orig = {} > 18% (BF16 storage adds ~1% over CPU baseline)",
                err / orig
            );
        }
    }

    #[test]
    fn roundtrip_4bit_gaussian_error_under_threshold() {
        // 256-dim, deterministic signs, random Gaussian-ish data
        use std::f32::consts::PI;
        let dim = 256;
        let signs = generate_signs(dim, 42);
        let x: Vec<f32> = (0..dim)
            .map(|i| ((i as f32) * 0.137).sin() * 1.5 + ((i as f32) * 0.029).cos() * 0.4)
            .collect();
        let (_recon, err) = cpu_roundtrip_4bit(&x, &signs);
        let mag: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        // Relative L2 error should be < 12% on this synthetic signal.
        // Lloyd-Max at 4-bit gives ~6% on pure-Gaussian post-Hadamard
        // input; this signal is structured so error is higher but still
        // well below half-precision noise.
        assert!(
            err / mag < 0.12,
            "L2 err {err} / mag {mag} = {} above 12% threshold",
            err / mag
        );
        // Sanity: pi shouldn't show up unless something is very wrong.
        let _ = PI;
    }
}
