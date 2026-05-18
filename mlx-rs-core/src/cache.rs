//! KV Cache implementations for LLM inference

use mlx_rs::{error::Exception, ops::concatenate_axis, ops::zeros_dtype, Array};
use mlx_rs::ops::indexing::{IndexMutOp, IndexOp, Ellipsis};
use mlx_rs::ops::{dequantize, quantize};

/// Trait for key-value caches used in attention
pub trait KeyValueCache {
    /// Returns the current offset (number of tokens in cache)
    fn offset(&self) -> i32;

    /// Returns the maximum cache size (for sliding window), if any
    fn max_size(&self) -> Option<i32>;

    /// Update cache with new keys/values and return full cache contents
    fn update_and_fetch(&mut self, keys: Array, values: Array) -> Result<(Array, Array), Exception>;

    /// Reset the cache offset to 0 without deallocating buffers.
    /// Default implementation does nothing (for caches that don't support reset).
    fn reset(&mut self) {}

    /// Optional fused-attention fast path. When a cache holds keys in a
    /// non-trivial compressed form (e.g. TurboQuant), it can fold the
    /// new-block append + Q @ K^T score + scale + mask + softmax + attn
    /// @ V steps into one operation that never materialises a full
    /// dequantized K tensor.
    ///
    /// Returns `Ok(Some(attn_out))` when handled; `Ok(None)` to fall
    /// back to the caller's standard `update_and_fetch` + SDPA path.
    ///
    /// `q` is `[B, Hq, q_len, D]`, `k_new`/`v_new` are
    /// `[B, Hkv, q_len_or_more, D]` (the just-projected new K/V block to
    /// append). `kv_repeat = Hq / Hkv` for GQA. `mask` is the same
    /// per-query mask the caller would have passed to SDPA (additive
    /// log-prob style, shape broadcastable to `[..., q_len, kv_len]`).
    ///
    /// Default impl: not fused.
    fn try_fused_attention(
        &mut self,
        _q: &Array,
        _k_new: Array,
        _v_new: Array,
        _scale: f32,
        _mask: Option<&Array>,
        _kv_repeat: i32,
    ) -> Result<Option<Array>, Exception> {
        Ok(None)
    }

    /// Returns sliced key/value tensors up to the current offset, if available.
    fn current_kv(&self) -> Option<(Array, Array)> {
        None
    }

    /// Materialize lazy computation graphs for cached arrays.
    fn eval(&self) -> Result<(), Exception> {
        Ok(())
    }
}

impl<T> KeyValueCache for &'_ mut T
where
    T: KeyValueCache,
{
    fn offset(&self) -> i32 {
        T::offset(self)
    }

    fn max_size(&self) -> Option<i32> {
        T::max_size(self)
    }

    fn update_and_fetch(&mut self, keys: Array, values: Array) -> Result<(Array, Array), Exception> {
        T::update_and_fetch(self, keys, values)
    }

    fn reset(&mut self) {
        T::reset(self)
    }

    fn current_kv(&self) -> Option<(Array, Array)> {
        T::current_kv(self)
    }

    fn eval(&self) -> Result<(), Exception> {
        T::eval(self)
    }
}

/// Simple concatenation-based KV cache
#[derive(Debug, Clone, Default)]
pub struct ConcatKeyValueCache {
    keys: Option<Array>,
    values: Option<Array>,
    offset: i32,
}

impl ConcatKeyValueCache {
    pub fn new() -> Self {
        Self::default()
    }
}

impl KeyValueCache for ConcatKeyValueCache {
    fn offset(&self) -> i32 {
        self.offset
    }

    fn max_size(&self) -> Option<i32> {
        None
    }

    fn update_and_fetch(&mut self, keys: Array, values: Array) -> Result<(Array, Array), Exception> {
        match (self.keys.take(), self.values.take()) {
            (Some(k), Some(v)) => {
                self.keys = Some(concatenate_axis(&[k, keys], -2)?);
                self.values = Some(concatenate_axis(&[v, values], -2)?);
            }
            _ => {
                self.keys = Some(keys);
                self.values = Some(values);
            }
        }
        let shape = self.keys.as_ref().expect("Keys cannot be None").shape();
        self.offset = shape[shape.len() - 2];

        Ok((
            self.keys.clone().expect("Keys cannot be None"),
            self.values.clone().expect("Values cannot be None"),
        ))
    }
}

/// Step-based KV Cache with pre-allocation (matches Python mlx-lm KVCache)
///
/// This cache pre-allocates buffers in steps of 256 tokens and uses in-place
/// slice updates, avoiding expensive concatenation on every token.
#[derive(Debug, Clone)]
pub struct KVCache {
    keys: Option<Array>,
    values: Option<Array>,
    offset: i32,
    step: i32,
    /// Soft reservation hint used by the first `update_and_fetch`: when set,
    /// the initial allocation rounds up to at least this many tokens so the
    /// cache doesn't need to regrow mid-decode for short generations.
    reserved: i32,
}

impl Default for KVCache {
    fn default() -> Self {
        Self::new()
    }
}

impl KVCache {
    pub fn new() -> Self {
        Self::with_step(256)
    }

    pub fn with_step(step: i32) -> Self {
        Self {
            keys: None,
            values: None,
            offset: 0,
            step,
            reserved: 0,
        }
    }

    /// Hint that the cache should pre-allocate room for at least `capacity`
    /// tokens on its first `update_and_fetch`. No-op once the buffers exist;
    /// safe to call before prefill with `prompt_len + max_new_tokens`.
    pub fn reserve(&mut self, capacity: i32) {
        if capacity > self.reserved {
            self.reserved = capacity;
        }
    }

    /// Current preallocated buffer capacity in tokens, if any.
    pub fn capacity(&self) -> Option<i32> {
        self.keys.as_ref().map(|k| k.shape()[2])
    }

    /// Returns sliced key/value tensors up to the current offset, if any.
    pub fn current_kv(&self) -> Option<(Array, Array)> {
        match (&self.keys, &self.values) {
            (Some(k), Some(v)) if self.offset > 0 => {
                Some((
                    k.index((Ellipsis, ..self.offset, ..)),
                    v.index((Ellipsis, ..self.offset, ..)),
                ))
            }
            _ => None,
        }
    }

    /// Borrow the underlying keys/values buffers (full preallocated length,
    /// not sliced to offset). Returns `None` until the first `update_and_fetch`.
    /// Useful for cache adapters that need to share storage.
    pub fn keys_buffer(&self) -> Option<&Array> {
        self.keys.as_ref()
    }

    pub fn values_buffer(&self) -> Option<&Array> {
        self.values.as_ref()
    }

    /// Compact the "appended window" `[past_length .. offset)` by keeping
    /// only the slots in `keep_indices` (interpreted as offsets within the
    /// window: `0` ⇒ position `past_length`, `1` ⇒ `past_length + 1`, …).
    /// Kept slots are written contiguously starting at `past_length` and
    /// `offset` becomes `past_length + keep_indices.len()`.
    ///
    /// Used by tree-based speculative decoding (e.g. DDTree) to drop
    /// rejected branches from the middle of a verify forward in-place,
    /// avoiding the cost of rolling back and re-feeding the accepted
    /// prefix. `keep_indices` must be a 1-D int32 tensor; its values must
    /// be in `[0, offset - past_length)`. Out-of-bounds indices are
    /// silently treated as if they were the last position (mlx semantics).
    pub fn compact(&mut self, past_length: i32, keep_indices: &Array) -> Result<(), Exception> {
        let past_length = past_length.max(0);
        if past_length >= self.offset {
            return Ok(());
        }
        let keep_count = keep_indices.shape()[0];
        if keep_count == 0 {
            self.offset = past_length;
            return Ok(());
        }
        let current_length = self.offset - past_length;
        if keep_count == current_length {
            // Fast path: nothing to compact iff indices == 0,1,...,n-1.
            // We trust the caller here — DDTree's accepted path is
            // generally a proper subset.
            return Ok(());
        }
        if let (Some(keys), Some(values)) = (self.keys.as_mut(), self.values.as_mut()) {
            // Single fused Metal dispatch per K/V: prefix [0..past_length)
            // copies straight through, suffix [past_length..past_length+keep_count)
            // gathers from input[past_length + keep_indices[i]]. Replaces the
            // prior two-dispatch take_axis + index_mut pattern.
            let new_keys = crate::metal_kernels::kv_compact(keys, past_length, keep_indices)?;
            let new_values = crate::metal_kernels::kv_compact(values, past_length, keep_indices)?;
            // The fused kernel produces a buffer of length past_length +
            // keep_count, which is smaller than the original preallocated
            // capacity; the next `update_and_fetch` will grow it as needed.
            *keys = new_keys;
            *values = new_values;
        }
        self.offset = past_length + keep_count;
        Ok(())
    }

    /// Drop the last `n_drop` cached positions by rewinding `offset`. The
    /// underlying preallocated buffer is unchanged; the next
    /// `update_and_fetch` will overwrite the rolled-back slots in place.
    ///
    /// Used by DFlash-style speculative-decoding rollback: after the target
    /// verifies a draft block, if N positions are accepted we trim the
    /// remaining `verify_len - N` positions in O(1) instead of cloning the
    /// pre-verify cache and re-running a forward pass.
    pub fn trim(&mut self, n_drop: i32) {
        if n_drop <= 0 {
            return;
        }
        let dropped = n_drop.min(self.offset);
        self.offset -= dropped;
    }

    /// Materialize lazy computation graphs for cached arrays.
    pub fn eval(&self) -> Result<(), Exception> {
        let mut arrays: Vec<&Array> = Vec::new();
        if let Some(k) = &self.keys {
            arrays.push(k);
        }
        if let Some(v) = &self.values {
            arrays.push(v);
        }
        if !arrays.is_empty() {
            mlx_rs::transforms::eval(arrays)?;
        }
        Ok(())
    }
}

impl KeyValueCache for KVCache {
    fn offset(&self) -> i32 {
        self.offset
    }

    fn max_size(&self) -> Option<i32> {
        None
    }

    fn reset(&mut self) {
        self.offset = 0;
    }

    fn update_and_fetch(&mut self, keys: Array, values: Array) -> Result<(Array, Array), Exception> {
        let prev = self.offset;
        let keys_shape = keys.shape();
        let values_shape = values.shape();
        let num_new = keys_shape[2];

        // Check if we need to grow the buffer
        let needs_grow = match &self.keys {
            None => true,
            Some(k) => (prev + num_new) > k.shape()[2],
        };

        if needs_grow {
            let b = keys_shape[0];
            let n_kv_heads = keys_shape[1];
            let k_head_dim = keys_shape[3];
            let v_head_dim = values_shape[3];

            let needed = prev + num_new;
            // Round needed up to the next step boundary, then honor any
            // reservation hint set before the first update_and_fetch.
            let n_steps = (needed + self.step - 1) / self.step;
            let mut new_size = n_steps * self.step;
            if self.keys.is_none() && self.reserved > new_size {
                let reserved_steps = (self.reserved + self.step - 1) / self.step;
                new_size = reserved_steps * self.step;
            }

            let k_shape = &[b, n_kv_heads, new_size, k_head_dim];
            let v_shape = &[b, n_kv_heads, new_size, v_head_dim];

            let k_dtype = keys.dtype();
            let v_dtype = values.dtype();
            let new_k = zeros_dtype(k_shape, k_dtype)?;
            let new_v = zeros_dtype(v_shape, v_dtype)?;

            match (self.keys.take(), self.values.take()) {
                (Some(old_k), Some(old_v)) => {
                    let (old_k, old_v) = if prev % self.step != 0 {
                        (
                            old_k.index((Ellipsis, ..prev, ..)),
                            old_v.index((Ellipsis, ..prev, ..)),
                        )
                    } else {
                        (old_k, old_v)
                    };
                    self.keys = Some(concatenate_axis(&[old_k, new_k], 2)?);
                    self.values = Some(concatenate_axis(&[old_v, new_v], 2)?);
                }
                _ => {
                    self.keys = Some(new_k);
                    self.values = Some(new_v);
                }
            }
        }

        self.offset += num_new;

        let k = self.keys.as_mut().unwrap();
        let v = self.values.as_mut().unwrap();
        k.index_mut((Ellipsis, prev..self.offset, ..), &keys);
        v.index_mut((Ellipsis, prev..self.offset, ..), &values);

        Ok((
            k.index((Ellipsis, ..self.offset, ..)),
            v.index((Ellipsis, ..self.offset, ..)),
        ))
    }

    fn current_kv(&self) -> Option<(Array, Array)> {
        KVCache::current_kv(self)
    }

    fn eval(&self) -> Result<(), Exception> {
        KVCache::eval(self)
    }
}

// ============================================================================
// TurboQuantKVCache — 4-bit Lloyd-Max on Hadamard-rotated keys (spike)
// ============================================================================

/// KV cache where keys are stored in TurboQuant 4-bit form (Hadamard-
/// rotated + per-vector sigma + Lloyd-Max N(0,1) codebook) and values
/// are stored in 8-bit per-group form (stock `mlx_rs::ops::quantize`
/// symmetric absmax, group_size=64). Sink tokens (configurable, default
/// 4) stay at the native dtype on both K and V — per the upstream
/// `tq-kv` 3-Fix ablation, these first few tokens carry the
/// disproportionate attention weight at decode time and quantizing them
/// is the largest single source of quality loss.
///
/// Memory (Gemma4-26B-A4B sliding head, head_dim=256, group_size=64):
///   - K: 512 bytes BF16  → 128 (packed) + 4 (sigma) + 4 (mean) = 136 B  ≈ 3.8×
///   - V: 512 bytes BF16  → 256 (u32 packed) + 16 (4 scales) + 16 (4 biases) ≈ 1.7×
///   - Combined KV: ≈ 2.6× compression including the sink-token overhead.
///   - Long contexts amortise the sink-token cost: 4 BF16-sink tokens on
///     a 4096-token cache add ~0.1% overhead vs the compressed bulk.
///
/// Spike caveats (matches `turboquant.rs`):
///   - No pre-RoPE quantization, no QJL, no calibrated codebooks.
///   - V uses mlx stock symmetric quantize; the upstream paper's
///     4-bit V path (+1.3% PPL) would need a Lloyd-Max V codebook.
///   - Cached signs reuse the same seed across all cycles; for safety
///     across a long process lifetime, use a unique seed per cache.
#[derive(Debug, Clone)]
pub struct TurboQuantKVCache {
    /// Packed 4-bit key indices, shape `[B, H, n_tokens, D/8]` u32.
    /// Holds only positions [sink_tokens..offset). Positions [0..sink)
    /// live in `sink_keys` at native dtype.
    packed_keys: Option<Array>,
    /// Per-vector sigma for compressed K, shape `[B, H, n_compressed]` f32.
    key_sigma: Option<Array>,
    /// Per-vector mean for compressed K, shape `[B, H, n_compressed]` f32.
    key_mean: Option<Array>,
    /// Sink tokens for K: first `sink_tokens` positions kept at native
    /// dtype, shape `[B, H, n_sink, D]`. None until the first update.
    sink_keys: Option<Array>,
    /// Compressed values: quantized `[B, H, n_compressed, packed_cols]`,
    /// scales/biases broadcast over groups.
    quant_v: Option<Array>,
    v_scales: Option<Array>,
    v_biases: Option<Array>,
    /// Sink tokens for V: first `sink_tokens` positions at native dtype.
    sink_values: Option<Array>,
    /// Number of sink tokens (capped at `offset` until the cache grows).
    sink_tokens: i32,
    /// Group size for V's per-group quantizer (must divide head_dim).
    v_group_size: i32,
    /// Bits for V (default 8; supported: 4 or 8).
    v_bits: i32,
    offset: i32,
    /// Sign tensor seed (deterministic across the lifetime of this cache).
    seed: u64,
}

impl TurboQuantKVCache {
    pub fn new() -> Self {
        // Default sink size from env (TURBOQUANT_SINK_TOKENS, defaults to 4).
        // Set to 0 to exclusively use the fully-fused SDPA path on the
        // entire compressed bulk.
        let sink_tokens: i32 = std::env::var("TURBOQUANT_SINK_TOKENS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4);
        Self {
            packed_keys: None,
            key_sigma: None,
            key_mean: None,
            sink_keys: None,
            quant_v: None,
            v_scales: None,
            v_biases: None,
            sink_values: None,
            sink_tokens,
            v_group_size: 64,
            v_bits: 8,
            offset: 0,
            seed: 0x5EED_5EED,
        }
    }

    pub fn with_seed(seed: u64) -> Self {
        Self { seed, ..Self::new() }
    }

    /// Override sink-token count. Set to 0 to disable.
    pub fn with_sink_tokens(mut self, n: i32) -> Self {
        self.sink_tokens = n.max(0);
        self
    }

    /// Override V quantization config. `bits` must be 4 or 8.
    pub fn with_v_quant(mut self, bits: i32, group_size: i32) -> Self {
        self.v_bits = bits;
        self.v_group_size = group_size;
        self
    }

    /// Reconstruct the full keys tensor `[B, H, offset, D]` at the
    /// caller-requested dtype: concatenates the native-dtype sink prefix
    /// with the TurboQuant-decompressed bulk.
    fn reconstruct_keys(&self, dtype: mlx_rs::Dtype, d: i32) -> Result<Array, Exception> {
        use crate::turboquant::{cached_signs, CENTROIDS_4BIT};
        let n_sink = self
            .sink_keys
            .as_ref()
            .map(|s| s.shape()[2])
            .unwrap_or(0);
        let n_compressed = self.offset - n_sink;
        if self.offset == 0 {
            return Err(Exception::custom("reconstruct_keys on empty cache"));
        }
        let bulk = if n_compressed > 0 {
            let packed = self
                .packed_keys
                .as_ref()
                .expect("packed_keys unset with compressed positions");
            let sigma = self.key_sigma.as_ref().expect("sigma unset");
            let mean = self.key_mean.as_ref().expect("mean unset");
            let shape = packed.shape().to_vec();
            let b = shape[0];
            let h = shape[1];
            let packed_w = packed.index((Ellipsis, ..n_compressed, ..));
            let sigma_w = sigma.index((Ellipsis, ..n_compressed));
            let mean_w = mean.index((Ellipsis, ..n_compressed));
            let signs_vec = cached_signs(d, self.seed);
            let signs = Array::from_slice(&signs_vec, &[d]);
            let centroids = Array::from_slice(&CENTROIDS_4BIT, &[16]);
            let out_shape = vec![b, h, n_compressed, d];
            Some(crate::metal_kernels::tq_decompress_4bit(
                &packed_w, &sigma_w, &mean_w, &signs, &centroids, &out_shape, dtype,
            )?)
        } else {
            None
        };
        match (self.sink_keys.as_ref(), bulk) {
            (Some(s), Some(b)) => concatenate_axis(&[s.clone(), b], 2),
            (Some(s), None) => Ok(s.clone()),
            (None, Some(b)) => Ok(b),
            (None, None) => Err(Exception::custom("reconstruct_keys: nothing to return")),
        }
    }

    /// Reconstruct the full values tensor `[B, H, offset, D]` by
    /// concatenating sink V with dequantized bulk V.
    fn reconstruct_values(&self, dtype: mlx_rs::Dtype) -> Result<Array, Exception> {
        let n_sink = self
            .sink_values
            .as_ref()
            .map(|s| s.shape()[2])
            .unwrap_or(0);
        let n_compressed = self.offset - n_sink;
        if self.offset == 0 {
            return Err(Exception::custom("reconstruct_values on empty cache"));
        }
        let bulk = if n_compressed > 0 {
            let q = self.quant_v.as_ref().expect("quant_v unset");
            let s = self.v_scales.as_ref().expect("v_scales unset");
            let bi = self.v_biases.as_ref().expect("v_biases unset");
            // q shape: [B, H, n_compressed, packed_cols]
            // s/bi shape: [B, H, n_compressed, n_groups]
            // We flatten to [B*H*n_compressed, packed_cols] for dequantize,
            // then reshape back.
            let qs = q.shape();
            let b = qs[0];
            let h = qs[1];
            let t = qs[2];
            let packed_cols = qs[3];
            let n_groups = s.shape()[3];
            let head_dim = n_groups * self.v_group_size;
            let q_flat = q.reshape(&[b * h * t, packed_cols])?;
            let s_flat = s.reshape(&[b * h * t, n_groups])?;
            let bi_flat = bi.reshape(&[b * h * t, n_groups])?;
            let deq = mlx_rs::ops::dequantize(
                &q_flat,
                &s_flat,
                &bi_flat,
                self.v_group_size,
                self.v_bits,
                None::<&str>,
            )?
            .as_dtype(dtype)?
            .reshape(&[b, h, t, head_dim])?;
            Some(deq)
        } else {
            None
        };
        match (self.sink_values.as_ref(), bulk) {
            (Some(s), Some(b)) => concatenate_axis(&[s.clone(), b], 2),
            (Some(s), None) => Ok(s.clone()),
            (None, Some(b)) => Ok(b),
            (None, None) => Err(Exception::custom("reconstruct_values: nothing to return")),
        }
    }
}

impl Default for TurboQuantKVCache {
    fn default() -> Self {
        Self::new()
    }
}

impl KeyValueCache for TurboQuantKVCache {
    fn offset(&self) -> i32 {
        self.offset
    }

    fn max_size(&self) -> Option<i32> {
        None
    }

    fn reset(&mut self) {
        self.packed_keys = None;
        self.key_sigma = None;
        self.key_mean = None;
        self.sink_keys = None;
        self.quant_v = None;
        self.v_scales = None;
        self.v_biases = None;
        self.sink_values = None;
        self.offset = 0;
    }

    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
    ) -> Result<(Array, Array), Exception> {
        use crate::turboquant::{cached_signs, BOUNDARIES_4BIT};
        let k_shape = keys.shape();
        let b = k_shape[0];
        let h = k_shape[1];
        let t_new = k_shape[2];
        let d = k_shape[3];
        let dtype = keys.dtype();

        // Determine how many of the new tokens go into the sink (kept at
        // native dtype) vs the compressed bulk. Sink positions are the
        // first `sink_tokens` positions of the whole cache; once the
        // sink is full all subsequent tokens go to the bulk.
        let already_sinked = self
            .sink_keys
            .as_ref()
            .map(|s| s.shape()[2])
            .unwrap_or(0);
        let sink_room = (self.sink_tokens - already_sinked).max(0);
        let n_to_sink = sink_room.min(t_new);
        let n_to_compress = t_new - n_to_sink;

        if n_to_sink > 0 {
            let k_sink = keys.index((Ellipsis, ..n_to_sink, ..));
            let v_sink = values.index((Ellipsis, ..n_to_sink, ..));
            self.sink_keys = Some(match self.sink_keys.take() {
                Some(prev) => concatenate_axis(&[prev, k_sink], 2)?,
                None => k_sink,
            });
            self.sink_values = Some(match self.sink_values.take() {
                Some(prev) => concatenate_axis(&[prev, v_sink], 2)?,
                None => v_sink,
            });
        }

        if n_to_compress > 0 {
            let k_bulk = keys.index((Ellipsis, n_to_sink.., ..));
            let v_bulk = values.index((Ellipsis, n_to_sink.., ..));

            // Compress K via TurboQuant Metal kernel.
            let signs_vec = cached_signs(d, self.seed);
            let signs = Array::from_slice(&signs_vec, &[d]);
            let boundaries = Array::from_slice(&BOUNDARIES_4BIT, &[15]);
            let (packed_new, sigma_new, mean_new) =
                crate::metal_kernels::tq_compress_4bit(&k_bulk, &signs, &boundaries)?;
            let packed_new = packed_new.reshape(&[b, h, n_to_compress, d / 8])?;
            let sigma_new = sigma_new.reshape(&[b, h, n_to_compress])?;
            let mean_new = mean_new.reshape(&[b, h, n_to_compress])?;

            // Compress V via stock symmetric per-group quantize. Reshape
            // [B, H, T, D] → [B*H*T, D] so quantize sees a 2D input and
            // produces grouped scales along the last axis.
            let v_flat = v_bulk.reshape(&[b * h * n_to_compress, d])?;
            let (vq, vs, vbi) = mlx_rs::ops::quantize(
                &v_flat,
                self.v_group_size,
                self.v_bits,
                None::<&str>,
            )?;
            let n_groups = d / self.v_group_size;
            let packed_v_cols = vq.shape()[1];
            let vq = vq.reshape(&[b, h, n_to_compress, packed_v_cols])?;
            let vs = vs.reshape(&[b, h, n_to_compress, n_groups])?;
            let vbi = vbi.reshape(&[b, h, n_to_compress, n_groups])?;

            self.packed_keys = Some(match self.packed_keys.take() {
                Some(prev) => concatenate_axis(&[prev, packed_new], 2)?,
                None => packed_new,
            });
            self.key_sigma = Some(match self.key_sigma.take() {
                Some(prev) => concatenate_axis(&[prev, sigma_new], 2)?,
                None => sigma_new,
            });
            self.key_mean = Some(match self.key_mean.take() {
                Some(prev) => concatenate_axis(&[prev, mean_new], 2)?,
                None => mean_new,
            });
            self.quant_v = Some(match self.quant_v.take() {
                Some(prev) => concatenate_axis(&[prev, vq], 2)?,
                None => vq,
            });
            self.v_scales = Some(match self.v_scales.take() {
                Some(prev) => concatenate_axis(&[prev, vs], 2)?,
                None => vs,
            });
            self.v_biases = Some(match self.v_biases.take() {
                Some(prev) => concatenate_axis(&[prev, vbi], 2)?,
                None => vbi,
            });
        }

        self.offset += t_new;

        let k_full = self.reconstruct_keys(dtype, d)?;
        let v_full = self.reconstruct_values(dtype)?;
        Ok((k_full, v_full))
    }

    fn current_kv(&self) -> Option<(Array, Array)> {
        if self.offset == 0 {
            return None;
        }
        // Infer head_dim + dtype from whichever side has the larger
        // surface (sink for very small caches, compressed otherwise).
        let (d, dtype) = if let Some(s) = self.sink_values.as_ref() {
            (s.shape()[3], s.dtype())
        } else if let Some(q) = self.quant_v.as_ref() {
            let packed_cols = q.shape()[3];
            // Recover D from packed_cols: packed_cols * 32 / v_bits.
            let d = (packed_cols * 32) / self.v_bits;
            // quant_v dtype is uint32; the original input was the model
            // dtype but we lost it. Default to bf16 since that's what
            // Gemma4 uses; downstream attention casts as needed.
            (d, mlx_rs::Dtype::Bfloat16)
        } else {
            return None;
        };
        let k = self.reconstruct_keys(dtype, d).ok()?;
        let v = self.reconstruct_values(dtype).ok()?;
        Some((k, v))
    }

    fn try_fused_attention(
        &mut self,
        q: &Array,
        k_new: Array,
        v_new: Array,
        scale: f32,
        mask: Option<&Array>,
        kv_repeat: i32,
    ) -> Result<Option<Array>, Exception> {
        use crate::turboquant::{cached_signs, CENTROIDS_4BIT};
        // Spike scope: q_len=1 only. Prefill / multi-token verify
        // (q_len > 1) takes the standard update_and_fetch + SDPA path.
        let qs = q.shape();
        if qs.len() != 4 || qs[2] != 1 {
            return Ok(None);
        }
        let b = qs[0];
        let h_q = qs[1];
        let d = qs[3];
        let dtype = q.dtype();

        // Append the new block first so subsequent reasoning sees the
        // updated offset and the right sink/bulk split.
        let _ = self.update_and_fetch(k_new, v_new)?;

        let n_sink = self
            .sink_keys
            .as_ref()
            .map(|s| s.shape()[2])
            .unwrap_or(0);
        let n_bulk = self.offset - n_sink;

        // FAST PATH: fully-fused single-dispatch SDPA. Requires:
        //   - no sink tokens (set TURBOQUANT_SINK_TOKENS=0)
        //   - bulk fits in the kernel's scratch buffer (TQ_SDPA_MAX_KV)
        //   - kv_len ≥ TURBOQUANT_FUSED_KV_MIN.
        //
        // Default kv_min is high (8192) because at small kv_len the
        // single-threadgroup-per-head design (8 thread-groups × 64
        // threads = 512 threads total) underutilises the GPU vs stock
        // mlx SDPA which schedules thousands of threadgroups. The fused
        // kernel becomes a win only at long contexts where K-decompress
        // bytes dominate kernel-launch + V-matmul costs.
        let fused_min: i32 = std::env::var("TURBOQUANT_FUSED_KV_MIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8192);
        let bulk_ok = n_sink == 0
            && n_bulk >= fused_min
            && n_bulk <= crate::metal_kernels::TQ_SDPA_MAX_KV;
        if bulk_ok {
            let packed = self.packed_keys.as_ref().unwrap();
            let sigma = self.key_sigma.as_ref().unwrap();
            let mean = self.key_mean.as_ref().unwrap();
            // V is now dequantised INSIDE the kernel; pass the packed
            // tensors directly instead of materialising a BF16 V tensor
            // first. Saves the v-dequant dispatch + the temporary
            // [B, Hkv, KV, D] BF16 buffer.
            let pv = self.quant_v.as_ref().unwrap();
            let vs = self.v_scales.as_ref().unwrap();
            let vb = self.v_biases.as_ref().unwrap();
            let signs_vec = cached_signs(d, self.seed);
            let signs = Array::from_slice(&signs_vec, &[d]);
            let centroids = Array::from_slice(&CENTROIDS_4BIT, &[16]);
            let out = crate::metal_kernels::tq_sdpa_4bit(
                q, packed, sigma, mean, pv, vs, vb,
                &signs, &centroids, mask, scale, kv_repeat, self.v_group_size,
            )?;
            return Ok(Some(out));
        }

        // SLOW PATH: partially-fused (QK kernel + standalone softmax + V
        // matmul) handles sink-token mixtures and oversized contexts.
        // The cache was already updated at the top of this method.

        // 2. Score the sink (BF16) prefix via a standard matmul.
        // Output shape: [B, Hq, 1, n_sink].
        let sink_scores = if n_sink > 0 {
            let sk = self.sink_keys.as_ref().unwrap(); // [B, Hkv, n_sink, D]
            let h_kv = sk.shape()[1];
            // Expand sink K to GQA layout by repeating along head axis.
            // Cheapest: matmul Q with sink K^T directly via per-head loop
            // would be slow; instead reshape Q to [B, Hkv, kv_repeat, 1, D]
            // and broadcast against sink K [B, Hkv, 1, n_sink, D]^T.
            // Simpler: reshape Q to [B*Hq, 1, D] and tile sink K to
            // [B*Hq, n_sink, D].
            let sk_dt = sk.as_dtype(dtype)?;
            // Tile sink K along head axis by kv_repeat.
            // Reshape to [B, Hkv, 1, n_sink, D] → broadcast to
            // [B, Hkv, kv_repeat, n_sink, D] → [B, Hq, n_sink, D].
            let sk_5d = sk_dt
                .reshape(&[b, h_kv, 1, n_sink, d])?;
            let sk_tiled = mlx_rs::ops::broadcast_to(
                &sk_5d,
                &[b, h_kv, kv_repeat, n_sink, d],
            )?
            .reshape(&[b, h_q, n_sink, d])?;
            // Q [B, Hq, 1, D] @ sk_tiled [B, Hq, D, n_sink].
            let sk_t = sk_tiled.transpose_axes(&[0, 1, 3, 2])?;
            Some(q.matmul(&sk_t)?.as_dtype(mlx_rs::Dtype::Float32)?)
        } else {
            None
        };

        // 3. Fused QK on the compressed bulk via tq_qk_score.
        // Output shape: [B, Hq, 1, n_bulk] f32.
        let bulk_scores = if n_bulk > 0 {
            let packed = self
                .packed_keys
                .as_ref()
                .ok_or_else(|| Exception::custom("bulk requested but packed_keys unset"))?;
            let sigma = self.key_sigma.as_ref().unwrap();
            let mean = self.key_mean.as_ref().unwrap();
            let signs_vec = cached_signs(d, self.seed);
            let signs = Array::from_slice(&signs_vec, &[d]);
            let centroids = Array::from_slice(&CENTROIDS_4BIT, &[16]);
            let scores = crate::metal_kernels::tq_qk_score(
                q,
                packed,
                sigma,
                mean,
                &signs,
                &centroids,
                kv_repeat,
            )?;
            Some(scores)
        } else {
            None
        };

        // 4. Concatenate scores along the kv axis (sink first, then bulk).
        let mut scores = match (sink_scores, bulk_scores) {
            (Some(s), Some(b)) => concatenate_axis(&[s, b], 3)?,
            (Some(s), None) => s,
            (None, Some(b)) => b,
            (None, None) => {
                return Err(Exception::custom("try_fused_attention: empty cache"));
            }
        };

        // 5. Scale + mask + softmax.
        scores = scores.multiply(mlx_rs::array!(scale))?;
        if let Some(m) = mask {
            let m_f = m.as_dtype(mlx_rs::Dtype::Float32)?;
            scores = scores.add(&m_f)?;
        }
        let attn = mlx_rs::ops::softmax_axis(&scores, -1, Some(true))?
            .as_dtype(dtype)?;

        // 6. V multiply on the reconstructed values (still dequantize V
        // for the spike — fused V-mul is the next deferred item).
        let v_full = self.reconstruct_values(dtype)?; // [B, Hkv, offset, D]
        let h_kv = v_full.shape()[1];
        let v_5d = v_full.reshape(&[b, h_kv, 1, self.offset, d])?;
        let v_tiled = mlx_rs::ops::broadcast_to(&v_5d, &[b, h_kv, kv_repeat, self.offset, d])?
            .reshape(&[b, h_q, self.offset, d])?;
        let out = attn.matmul(&v_tiled)?;
        Ok(Some(out))
    }

    fn eval(&self) -> Result<(), Exception> {
        let mut arrays: Vec<&Array> = Vec::new();
        if let Some(a) = &self.packed_keys  { arrays.push(a); }
        if let Some(a) = &self.key_sigma    { arrays.push(a); }
        if let Some(a) = &self.key_mean     { arrays.push(a); }
        if let Some(a) = &self.sink_keys    { arrays.push(a); }
        if let Some(a) = &self.quant_v      { arrays.push(a); }
        if let Some(a) = &self.v_scales     { arrays.push(a); }
        if let Some(a) = &self.v_biases     { arrays.push(a); }
        if let Some(a) = &self.sink_values  { arrays.push(a); }
        if !arrays.is_empty() {
            mlx_rs::transforms::eval(arrays)?;
        }
        Ok(())
    }
}

// ============================================================================
// QuantizedKVCache — K=q8, V=q4 mixed-precision KV cache
// ============================================================================

/// Mixed-precision KV cache: keys stored at q8, values at q4.
///
/// Tokens accumulate in an fp16 residual buffer until `step` tokens are
/// ready, then the full block is quantized and appended.  Every call to
/// `update_and_fetch` returns a fully dequantized view so attention
/// computation is unaffected.
///
/// Storage shape convention (B=batch, H=heads, T=tokens, D=head_dim):
/// - `k_q`      : `[B, H, n_quantized, D/(32/k_bits)]`
/// - `k_scales` : `[B, H, n_quantized, D/group_size]`
/// - `k_biases` : same shape as scales
/// - Residual   : `[B, H, r, D]`  where `r < step`
#[derive(Debug, Clone)]
pub struct QuantizedKVCache {
    // Quantized key storage [B, H, n_quantized, packed_cols]
    k_q: Option<Array>,
    k_scales: Option<Array>,
    k_biases: Option<Array>,
    // Quantized value storage [B, H, n_quantized, packed_cols]
    v_q: Option<Array>,
    v_scales: Option<Array>,
    v_biases: Option<Array>,
    // fp16 residual (tokens not yet quantized)
    k_residual: Option<Array>,
    v_residual: Option<Array>,
    /// Group size used for quantization (must divide head_dim).
    pub group_size: i32,
    /// Bits per element for keys (8 recommended).
    pub k_bits: i32,
    /// Bits per element for values (4 recommended).
    pub v_bits: i32,
    /// Accumulate this many tokens before quantizing.
    pub step: i32,
    // Dimensions (set on first update_and_fetch)
    batch: i32,
    n_kv_heads: i32,
    k_head_dim: i32,
    v_head_dim: i32,
    // Total tokens quantized (not counting residual)
    n_quantized: i32,
    offset: i32,
}

impl QuantizedKVCache {
    /// Create a cache with explicit configuration.
    ///
    /// # Params
    /// - `group_size` : quantization group size (must divide head_dim, typically 64)
    /// - `k_bits`     : bits for keys (8)
    /// - `v_bits`     : bits for values (4)
    /// - `step`       : tokens to buffer before quantizing (256 matches KVCache default)
    pub fn new(group_size: i32, k_bits: i32, v_bits: i32, step: i32) -> Self {
        Self {
            k_q: None,
            k_scales: None,
            k_biases: None,
            v_q: None,
            v_scales: None,
            v_biases: None,
            k_residual: None,
            v_residual: None,
            group_size,
            k_bits,
            v_bits,
            step,
            batch: 0,
            n_kv_heads: 0,
            k_head_dim: 0,
            v_head_dim: 0,
            n_quantized: 0,
            offset: 0,
        }
    }

    /// Default: K=q8, V=q4, group_size=64, step=256.
    pub fn default_config() -> Self {
        Self::new(64, 8, 4, 256)
    }

    /// Dequantize stored keys/values and concatenate with residual.
    fn reconstruct(&self) -> Result<(Array, Array), Exception> {
        let b = self.batch;
        let h = self.n_kv_heads;
        let kd = self.k_head_dim;
        let vd = self.v_head_dim;
        let n_q = self.n_quantized;

        let (quant_k, quant_v) = if n_q > 0 {
            // Flatten [B, H, n_q, packed] → [B*H*n_q, packed] for dequantize
            let kq = self.k_q.as_ref().unwrap().reshape(&[b * h * n_q, -1])?;
            let ks = self.k_scales.as_ref().unwrap().reshape(&[b * h * n_q, -1])?;
            let kb = self.k_biases.as_ref().unwrap().reshape(&[b * h * n_q, -1])?;
            let k_deq = dequantize(&kq, &ks, &kb, self.group_size, self.k_bits, None::<&str>)?
                .reshape(&[b, h, n_q, kd])?;

            let vq = self.v_q.as_ref().unwrap().reshape(&[b * h * n_q, -1])?;
            let vs = self.v_scales.as_ref().unwrap().reshape(&[b * h * n_q, -1])?;
            let vb = self.v_biases.as_ref().unwrap().reshape(&[b * h * n_q, -1])?;
            let v_deq = dequantize(&vq, &vs, &vb, self.group_size, self.v_bits, None::<&str>)?
                .reshape(&[b, h, n_q, vd])?;

            (Some(k_deq), Some(v_deq))
        } else {
            (None, None)
        };

        let full_k = match (quant_k, self.k_residual.as_ref()) {
            (Some(qk), Some(r)) => concatenate_axis(&[qk, r.clone()], 2)?,
            (Some(qk), None) => qk,
            (None, Some(r)) => r.clone(),
            (None, None) => return Err(Exception::custom("QuantizedKVCache: reconstruct on empty cache")),
        };
        let full_v = match (quant_v, self.v_residual.as_ref()) {
            (Some(qv), Some(r)) => concatenate_axis(&[qv, r.clone()], 2)?,
            (Some(qv), None) => qv,
            (None, Some(r)) => r.clone(),
            (None, None) => return Err(Exception::custom("QuantizedKVCache: reconstruct on empty cache")),
        };

        Ok((full_k, full_v))
    }

    /// Quantize a `[B, H, T, D]` block and store into `[B, H, T, packed_cols]`.
    fn quantize_block(
        arr: &Array,
        b: i32, h: i32, t: i32, d: i32,
        group_size: i32, bits: i32,
    ) -> Result<(Array, Array, Array), Exception> {
        // Flatten to 2D for the MLX quantize op
        let flat = arr.reshape(&[b * h * t, d])?;
        let (q, s, bv) = quantize(&flat, group_size, bits, None::<&str>)?;
        // Reshape back so tokens stay on axis-2
        let packed_cols = q.shape()[1];
        let scale_cols = s.shape()[1];
        Ok((
            q.reshape(&[b, h, t, packed_cols])?,
            s.reshape(&[b, h, t, scale_cols])?,
            bv.reshape(&[b, h, t, scale_cols])?,
        ))
    }
}

impl Default for QuantizedKVCache {
    fn default() -> Self {
        Self::default_config()
    }
}

impl KeyValueCache for QuantizedKVCache {
    fn offset(&self) -> i32 {
        self.offset
    }

    fn max_size(&self) -> Option<i32> {
        None
    }

    fn reset(&mut self) {
        *self = Self::new(self.group_size, self.k_bits, self.v_bits, self.step);
    }

    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
    ) -> Result<(Array, Array), Exception> {
        let ks = keys.shape();
        let b = ks[0] as i32;
        let h = ks[1] as i32;
        let t = ks[2] as i32;
        let kd = ks[3] as i32;
        let vd = values.shape()[3] as i32;

        // Validate dimensions on subsequent calls
        if self.batch == 0 {
            if kd % self.group_size != 0 {
                return Err(Exception::custom(format!(
                    "QuantizedKVCache: k_head_dim={kd} not divisible by group_size={}",
                    self.group_size
                )));
            }
            if vd % self.group_size != 0 {
                return Err(Exception::custom(format!(
                    "QuantizedKVCache: v_head_dim={vd} not divisible by group_size={}",
                    self.group_size
                )));
            }
            self.batch = b;
            self.n_kv_heads = h;
            self.k_head_dim = kd;
            self.v_head_dim = vd;
        } else if b != self.batch || h != self.n_kv_heads {
            return Err(Exception::custom(format!(
                "QuantizedKVCache: batch/heads mismatch: expected ({},{}) got ({b},{h})",
                self.batch, self.n_kv_heads
            )));
        }

        // Grow residual by appending new tokens along axis-2
        let k_res = match self.k_residual.take() {
            Some(prev) => concatenate_axis(&[prev, keys], 2)?,
            None => keys,
        };
        let v_res = match self.v_residual.take() {
            Some(prev) => concatenate_axis(&[prev, values], 2)?,
            None => values,
        };

        let res_len = k_res.shape()[2] as i32;
        let n_full_steps = res_len / self.step;

        if n_full_steps > 0 {
            let tokens_to_q = n_full_steps * self.step;
            let remaining = res_len - tokens_to_q;

            // Slice the tokens to quantize [B, H, tokens_to_q, D]
            let k_to_q = k_res.index((Ellipsis, ..tokens_to_q, ..));
            let v_to_q = v_res.index((Ellipsis, ..tokens_to_q, ..));

            // Quantize keeping [B, H, tokens_to_q, packed] shape
            let (new_kq, new_ks, new_kb) =
                Self::quantize_block(&k_to_q, b, h, tokens_to_q, kd, self.group_size, self.k_bits)?;
            let (new_vq, new_vs, new_vb) =
                Self::quantize_block(&v_to_q, b, h, tokens_to_q, vd, self.group_size, self.v_bits)?;

            // Append along token axis-2
            self.k_q = Some(match self.k_q.take() {
                Some(prev) => concatenate_axis(&[prev, new_kq], 2)?,
                None => new_kq,
            });
            self.k_scales = Some(match self.k_scales.take() {
                Some(prev) => concatenate_axis(&[prev, new_ks], 2)?,
                None => new_ks,
            });
            self.k_biases = Some(match self.k_biases.take() {
                Some(prev) => concatenate_axis(&[prev, new_kb], 2)?,
                None => new_kb,
            });
            self.v_q = Some(match self.v_q.take() {
                Some(prev) => concatenate_axis(&[prev, new_vq], 2)?,
                None => new_vq,
            });
            self.v_scales = Some(match self.v_scales.take() {
                Some(prev) => concatenate_axis(&[prev, new_vs], 2)?,
                None => new_vs,
            });
            self.v_biases = Some(match self.v_biases.take() {
                Some(prev) => concatenate_axis(&[prev, new_vb], 2)?,
                None => new_vb,
            });

            self.n_quantized += tokens_to_q;

            if remaining > 0 {
                self.k_residual = Some(k_res.index((Ellipsis, tokens_to_q.., ..)));
                self.v_residual = Some(v_res.index((Ellipsis, tokens_to_q.., ..)));
            }
            // else: residual is empty; leave as None
        } else {
            self.k_residual = Some(k_res);
            self.v_residual = Some(v_res);
        }

        self.offset += t;
        self.reconstruct()
    }

    fn current_kv(&self) -> Option<(Array, Array)> {
        if self.offset == 0 {
            return None;
        }
        self.reconstruct().ok()
    }

    fn eval(&self) -> Result<(), Exception> {
        let mut arrays: Vec<&Array> = Vec::new();
        macro_rules! push_opt {
            ($f:expr) => { if let Some(a) = &$f { arrays.push(a); } };
        }
        push_opt!(self.k_q);
        push_opt!(self.k_scales);
        push_opt!(self.k_biases);
        push_opt!(self.v_q);
        push_opt!(self.v_scales);
        push_opt!(self.v_biases);
        push_opt!(self.k_residual);
        push_opt!(self.v_residual);
        if !arrays.is_empty() {
            mlx_rs::transforms::eval(arrays)?;
        }
        Ok(())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::Array;

    fn make_kv(b: i32, h: i32, t: i32, d: i32, seed: f32) -> (Array, Array) {
        // Simple deterministic fill: value = (index * seed) % 1.0
        let k_size = (b * h * t * d) as usize;
        let k_data: Vec<f32> = (0..k_size).map(|i| ((i as f32) * seed) % 1.0).collect();
        let v_data: Vec<f32> = (0..k_size).map(|i| ((i as f32) * seed * 0.5) % 1.0).collect();
        (
            Array::from_slice(&k_data, &[b, h, t, d]),
            Array::from_slice(&v_data, &[b, h, t, d]),
        )
    }

    #[test]
    fn quantized_kv_cache_output_shape_single_call() {
        let mut cache = QuantizedKVCache::new(64, 8, 4, 256);
        let (k, v) = make_kv(1, 4, 10, 256, 0.01);
        let (k_out, v_out) = cache.update_and_fetch(k, v).unwrap();
        assert_eq!(k_out.shape(), &[1, 4, 10, 256]);
        assert_eq!(v_out.shape(), &[1, 4, 10, 256]);
        assert_eq!(cache.offset(), 10);
    }

    #[test]
    fn quantized_kv_cache_grows_across_decode_steps() {
        // step=8 so quantization triggers after 8 tokens
        let mut cache = QuantizedKVCache::new(64, 8, 4, 8);
        for i in 0..4_i32 {
            let (k, v) = make_kv(1, 2, 4, 256, 0.01);
            let (k_out, v_out) = cache.update_and_fetch(k, v).unwrap();
            let expected = ((i + 1) * 4) as i32;
            assert_eq!(k_out.shape()[2], expected);
            assert_eq!(v_out.shape()[2], expected);
            assert_eq!(cache.offset(), expected);
        }
    }

    #[test]
    fn quantized_kv_cache_step_triggers_quantization() {
        // After exactly `step` tokens the residual should be empty
        let step = 64_i32;
        let mut cache = QuantizedKVCache::new(64, 8, 4, step);
        let (k, v) = make_kv(1, 4, step, 256, 0.01);
        let _ = cache.update_and_fetch(k, v).unwrap();
        assert_eq!(cache.n_quantized, step);
        assert!(cache.k_residual.is_none(), "residual should be empty after full step");
    }

    #[test]
    fn quantized_kv_cache_reset_clears_state() {
        let mut cache = QuantizedKVCache::new(64, 8, 4, 8);
        let (k, v) = make_kv(1, 2, 8, 256, 0.01);
        cache.update_and_fetch(k, v).unwrap();
        assert_eq!(cache.offset(), 8);
        cache.reset();
        assert_eq!(cache.offset(), 0);
        assert_eq!(cache.n_quantized, 0);
        assert!(cache.k_q.is_none());
        assert!(cache.k_residual.is_none());
    }

    #[test]
    fn quantized_kv_cache_prefill_then_decode() {
        // Prefill 100 tokens, then decode 50 single-token steps
        let mut cache = QuantizedKVCache::new(64, 8, 4, 64);
        let (k_pre, v_pre) = make_kv(1, 4, 100, 256, 0.01);
        let (k_out, _) = cache.update_and_fetch(k_pre, v_pre).unwrap();
        assert_eq!(k_out.shape()[2], 100);

        for step in 1..=50_i32 {
            let (k, v) = make_kv(1, 4, 1, 256, 0.01 + step as f32 * 0.001);
            let (k_out, _) = cache.update_and_fetch(k, v).unwrap();
            assert_eq!(k_out.shape()[2], 100 + step);
            assert_eq!(cache.offset(), 100 + step);
        }
    }

    #[test]
    fn quantized_kv_cache_residual_matches_fp16_before_quantize() {
        // With step=256, first 255 tokens stay in residual; output should be
        // numerically identical to KVCache (no quantization applied yet).
        let mut fp16 = KVCache::new();
        let mut qkv = QuantizedKVCache::new(64, 8, 4, 256);

        for i in 0..10_i32 {
            let (k, v) = make_kv(1, 4, 1, 256, 0.01 + i as f32 * 0.001);
            let (fp_k, fp_v) = fp16.update_and_fetch(k.clone(), v.clone()).unwrap();
            let (q_k, q_v) = qkv.update_and_fetch(k, v).unwrap();
            // shapes must match
            assert_eq!(fp_k.shape(), q_k.shape(), "K shape mismatch at step {i}");
            assert_eq!(fp_v.shape(), q_v.shape(), "V shape mismatch at step {i}");
        }
        // No quantization triggered yet
        assert_eq!(qkv.n_quantized, 0);
        assert!(qkv.k_q.is_none());
    }

    #[test]
    fn quantized_kv_cache_invalid_group_size_errors() {
        let mut cache = QuantizedKVCache::new(64, 8, 4, 256);
        // head_dim=100 is not divisible by group_size=64
        let (k, v) = make_kv(1, 4, 1, 100, 0.01);
        let result = cache.update_and_fetch(k, v);
        assert!(result.is_err(), "Expected error for non-divisible head_dim");
    }
}
