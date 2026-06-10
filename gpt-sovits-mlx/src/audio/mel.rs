//! MLX-native mel spectrogram computation for VITS training
//!
//! This module provides GPU-accelerated STFT and mel spectrogram computation
//! using MLX arrays, enabling efficient loss computation during training.

use mlx_rs::{
    array,
    error::Exception,
    ops::{concatenate_axis, indexing::IndexOp, matmul, maximum, zeros},
    Array,
};
use std::f32::consts::PI;

/// Configuration for mel spectrogram computation
#[derive(Debug, Clone)]
pub struct MelConfig {
    /// FFT size (default: 2048)
    pub n_fft: i32,
    /// Hop length in samples (default: 640)
    pub hop_length: i32,
    /// Window length (default: 2048)
    pub win_length: i32,
    /// Sample rate (default: 32000)
    pub sample_rate: i32,
    /// Number of mel bins (default: 128)
    pub n_mels: i32,
    /// Minimum frequency for mel filterbank (default: 0.0)
    pub fmin: f32,
    /// Maximum frequency for mel filterbank (default: None = sr/2)
    pub fmax: Option<f32>,
}

impl Default for MelConfig {
    fn default() -> Self {
        Self {
            n_fft: 2048,
            hop_length: 640,
            win_length: 2048,
            sample_rate: 32000,
            n_mels: 128,
            fmin: 0.0,
            fmax: None, // Will use sr/2
        }
    }
}

/// Create Hann window for STFT
///
/// Periodic Hann (denominator = size), matching `torch.hann_window(size)`.
fn hann_window(size: i32) -> Array {
    let window: Vec<f32> = (0..size as usize)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / size as f32).cos()))
        .collect();
    Array::from_slice(&window, &[size])
}

use mlx_rs_core::audio::{hz_to_mel, mel_to_hz};

/// Create mel filterbank matrix
///
/// Returns Array of shape [n_mels, n_fft/2 + 1]
pub fn create_mel_filterbank(config: &MelConfig) -> Array {
    let n_freqs = (config.n_fft / 2 + 1) as usize;
    let fmax = config.fmax.unwrap_or(config.sample_rate as f32 / 2.0);

    // Mel scale edges
    let mel_min = hz_to_mel(config.fmin);
    let mel_max = hz_to_mel(fmax);

    // Create mel points (n_mels + 2 for edges)
    let n_points = config.n_mels as usize + 2;
    let mel_points: Vec<f32> = (0..n_points)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_points - 1) as f32)
        .collect();

    // Convert to Hz
    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    // Convert to FFT bin indices
    let fft_freqs: Vec<f32> = (0..n_freqs)
        .map(|i| i as f32 * config.sample_rate as f32 / config.n_fft as f32)
        .collect();

    // Create filterbank
    let mut filterbank = vec![0.0f32; config.n_mels as usize * n_freqs];

    for m in 0..config.n_mels as usize {
        let f_m_minus = hz_points[m];
        let f_m = hz_points[m + 1];
        let f_m_plus = hz_points[m + 2];

        for k in 0..n_freqs {
            let freq = fft_freqs[k];
            if freq >= f_m_minus && freq <= f_m {
                // Rising edge
                filterbank[m * n_freqs + k] = (freq - f_m_minus) / (f_m - f_m_minus);
            } else if freq >= f_m && freq <= f_m_plus {
                // Falling edge
                filterbank[m * n_freqs + k] = (f_m_plus - freq) / (f_m_plus - f_m);
            }
        }
    }

    Array::from_slice(&filterbank, &[config.n_mels, n_freqs as i32])
}

/// Compute STFT magnitude using MLX
///
/// Input: audio [batch, samples] or [samples]
/// Output: magnitude [batch, n_fft/2+1, frames]
///
/// Uses batched `fft::rfft` (O(N log N)) over windowed frames instead of a
/// per-frame DFT-matrix multiply.
///
/// PERF/PARITY note: padding is zero padding of (n_fft - hop) / 2 per side;
/// GPT-SoVITS's `spectrogram_torch` uses REFLECT padding of the same width
/// (see `stft_gpu::stft_rfft` for the reflect-padded inference path).
pub fn stft_mlx(
    audio: &Array,
    n_fft: i32,
    hop_length: i32,
    win_length: i32,
) -> Result<Array, Exception> {
    let shape = audio.shape();
    let is_batched = shape.len() == 2;

    let batch_size = if is_batched { shape[0] } else { 1 };
    let num_samples = if is_batched { shape[1] } else { shape[0] };

    // Pad audio
    let pad_left = (n_fft - hop_length) / 2;
    let pad_right = (n_fft - hop_length) / 2;
    let padded_len = num_samples + pad_left + pad_right;

    // Number of frames
    let n_frames = (padded_len - n_fft) / hop_length + 1;
    let n_freqs = n_fft / 2 + 1;
    if n_frames <= 0 {
        return Err(Exception::from("No frames computed in STFT"));
    }

    // Create window once (hoisted out of the per-frame loop)
    let window = hann_window(win_length);
    let window_data: Vec<f32> = window.as_slice().to_vec();

    // Get audio data for frame extraction (zero padding applied below)
    let audio_flat = audio.flatten(None, None)?;
    let audio_data: Vec<f32> = audio_flat.as_slice().to_vec();

    // Build windowed frames: [batch * n_frames, n_fft]
    let mut frames_data = vec![0.0f32; (batch_size * n_frames * n_fft) as usize];
    for b in 0..batch_size as usize {
        let sample_base = b * num_samples as usize;
        for frame_idx in 0..n_frames as usize {
            let frame_base = (b * n_frames as usize + frame_idx) * n_fft as usize;
            // Position in the (virtually) padded signal
            let padded_start = frame_idx * hop_length as usize;
            for i in 0..win_length as usize {
                let padded_pos = padded_start + i;
                // Map back to the unpadded signal; outside = zero padding
                if padded_pos >= pad_left as usize {
                    let sample_pos = padded_pos - pad_left as usize;
                    if sample_pos < num_samples as usize {
                        frames_data[frame_base + i] =
                            audio_data[sample_base + sample_pos] * window_data[i];
                    }
                }
            }
        }
    }

    let frames = Array::from_slice(&frames_data, &[batch_size * n_frames, n_fft]);

    // Batched real FFT: [batch * n_frames, n_fft] -> [batch * n_frames, n_freqs]
    let fft_result = mlx_rs::fft::rfft(&frames, n_fft, -1)?;
    // Magnitude: |complex|
    let magnitude = mlx_rs::ops::abs(&fft_result)?;

    // [batch * n_frames, n_freqs] -> [batch, n_frames, n_freqs] -> [batch, n_freqs, n_frames]
    magnitude
        .reshape(&[batch_size, n_frames, n_freqs])?
        .transpose_axes(&[0, 2, 1])
}

/// Compute mel spectrogram from audio using MLX
///
/// Input: audio [batch, samples] or [samples]
/// Output: mel [batch, n_mels, frames] or [n_mels, frames]
///
/// Applies log compression: log(clamp(mel, 1e-5))
pub fn mel_spectrogram_mlx(
    audio: &Array,
    config: &MelConfig,
) -> Result<Array, Exception> {
    // Compute STFT magnitude
    let stft_mag = stft_mlx(audio, config.n_fft, config.hop_length, config.win_length)?;

    // Create mel filterbank
    let mel_basis = create_mel_filterbank(config);

    let shape = stft_mag.shape();
    let is_batched = shape.len() == 3;

    // Apply mel filterbank: [n_mels, n_freqs] @ [n_freqs, frames] = [n_mels, frames]
    let mel = if is_batched {
        // For batched: [batch, n_freqs, frames] -> process each batch
        let batch_size = shape[0];
        let n_frames = shape[2];

        let mut mels = Vec::new();
        for b in 0..batch_size as usize {
            // Extract batch: [n_freqs, frames]
            let batch_stft = stft_mag.index((b as i32, .., ..));
            let batch_mel = matmul(&mel_basis, &batch_stft)?;
            mels.push(batch_mel);
        }

        let refs: Vec<&Array> = mels.iter().collect();
        let stacked = concatenate_axis(&refs, 0)?;
        stacked.reshape(&[batch_size, config.n_mels, n_frames])?
    } else {
        // [n_freqs, frames] -> [n_mels, frames]
        matmul(&mel_basis, &stft_mag)?
    };

    // Log compression
    let mel_log = maximum(&mel, &array!(1e-5f32))?.log()?;

    Ok(mel_log)
}

/// Convert linear spectrogram to mel spectrogram
///
/// This matches Python's spec_to_mel_torch:
/// 1. Apply mel filterbank: mel = mel_basis @ spec
/// 2. Apply log compression: log(clamp(mel, 1e-5))
///
/// Input: spec [batch, n_fft/2+1, frames] or [n_fft/2+1, frames]
/// Output: mel [batch, n_mels, frames] or [n_mels, frames]
///
/// This is used for computing mel loss in training where we want to use
/// the original spectrogram (from data preprocessing) rather than recomputing
/// from audio.
pub fn spec_to_mel(
    spec: &Array,
    config: &MelConfig,
) -> Result<Array, Exception> {
    // Create mel filterbank
    let mel_basis = create_mel_filterbank(config);

    let shape = spec.shape();
    let is_batched = shape.len() == 3;

    // Apply mel filterbank: [n_mels, n_freqs] @ [n_freqs, frames] = [n_mels, frames]
    let mel = if is_batched {
        let batch_size = shape[0];
        let n_frames = shape[2];

        let mut mels = Vec::new();
        for b in 0..batch_size as usize {
            // Extract batch: [n_freqs, frames]
            let batch_spec = spec.index((b as i32, .., ..));
            let batch_mel = matmul(&mel_basis, &batch_spec)?;
            mels.push(batch_mel);
        }

        let refs: Vec<&Array> = mels.iter().collect();
        let stacked = concatenate_axis(&refs, 0)?;
        stacked.reshape(&[batch_size, config.n_mels, n_frames])?
    } else {
        // [n_freqs, frames] -> [n_mels, frames]
        matmul(&mel_basis, spec)?
    };

    // Log compression (matches Python's spectral_normalize_torch)
    let mel_log = maximum(&mel, &array!(1e-5f32))?.log()?;

    Ok(mel_log)
}

/// Slice segments from mel spectrogram at given frame indices
///
/// Input: mel [batch, n_mels, frames]
/// ids_slice: [batch] frame indices
/// segment_frames: number of frames to extract
/// Output: mel [batch, n_mels, segment_frames]
pub fn slice_mel_segments(
    mel: &Array,
    ids_slice: &Array,
    segment_frames: i32,
) -> Result<Array, Exception> {
    let batch_size = mel.dim(0);
    let n_mels = mel.dim(1);
    let total_frames = mel.dim(2);

    let mut slices = Vec::new();
    for b in 0..batch_size as usize {
        let start_frame: i32 = ids_slice.index(b as i32).item();
        let start_frame = start_frame.clamp(0, total_frames);
        let end_frame = (start_frame + segment_frames).min(total_frames);

        // Slice: [n_mels, available_frames]
        let slice = mel.index((b as i32, .., start_frame..end_frame));

        // Pad short slices with zeros so every slice is exactly segment_frames
        // (otherwise the reshape below panics on slices near the end of mel).
        let available = end_frame - start_frame;
        let slice = if available < segment_frames {
            let pad = zeros::<f32>(&[n_mels, segment_frames - available])?;
            concatenate_axis(&[&slice, &pad], 1)?
        } else {
            slice
        };
        slices.push(slice);
    }

    // Stack: [batch, n_mels, segment_frames]
    let refs: Vec<&Array> = slices.iter().collect();
    let stacked = concatenate_axis(&refs, 0)?;
    stacked.reshape(&[batch_size, n_mels, segment_frames])
}

/// Configuration for spectrogram computation
#[derive(Debug, Clone)]
pub struct SpectrogramConfig {
    /// FFT size (default: 2048)
    pub n_fft: i32,
    /// Hop length in samples (default: 640)
    pub hop_length: i32,
    /// Window length (default: 2048)
    pub win_length: i32,
}

impl Default for SpectrogramConfig {
    fn default() -> Self {
        Self {
            n_fft: 2048,
            hop_length: 640,
            win_length: 2048,
        }
    }
}

/// Compute spectrogram (linear STFT magnitude) from audio using MLX
///
/// Input: audio [batch, samples] or [samples]
/// Output: spectrogram [batch, n_fft/2+1, frames] or [n_fft/2+1, frames]
///
/// This is the linear spectrogram needed for VITS enc_q input.
pub fn spectrogram_mlx(
    audio: &Array,
    config: &SpectrogramConfig,
) -> Result<Array, Exception> {
    stft_mlx(audio, config.n_fft, config.hop_length, config.win_length)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mel_filterbank() {
        let config = MelConfig::default();
        let fb = create_mel_filterbank(&config);
        assert_eq!(fb.shape(), &[128, 1025]);
    }

    #[test]
    fn test_hann_window() {
        let win = hann_window(1024);
        assert_eq!(win.shape(), &[1024]);
        // Hann window should be 0 at edges, 1 at center
        let data: Vec<f32> = win.as_slice().to_vec();
        assert!(data[0].abs() < 0.01);
        assert!((data[512] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_stft_basic() {
        // Create a simple sine wave
        let sr = 32000;
        let freq = 440.0;
        let duration = 0.1;
        let n_samples = (sr as f32 * duration) as usize;

        let audio: Vec<f32> = (0..n_samples)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr as f32).sin())
            .collect();

        let audio_arr = Array::from_slice(&audio, &[n_samples as i32]);
        let config = MelConfig::default();

        let result = stft_mlx(&audio_arr, config.n_fft, config.hop_length, config.win_length);
        assert!(result.is_ok());

        let stft_mag = result.unwrap();
        // Should have shape [batch, n_fft/2+1, frames]
        assert_eq!(stft_mag.shape()[0], 1);
        assert_eq!(stft_mag.shape()[1], config.n_fft / 2 + 1);
    }
}
