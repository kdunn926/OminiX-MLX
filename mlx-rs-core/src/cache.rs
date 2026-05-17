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
        use mlx_rs::ops::indexing::take_axis;
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
            // No reordering needed iff indices are 0,1,...,current_length-1.
            // Fast path: trust the caller and skip — typical DDTree usage
            // accepts an arbitrary subset so this branch rarely fires, but
            // when it does it avoids an unnecessary index_select.
            return Ok(());
        }
        if let (Some(keys), Some(values)) = (self.keys.as_mut(), self.values.as_mut()) {
            // Slice the appended window, gather, and write back at the
            // contiguous prefix of the same window.
            let window_k = keys.index((Ellipsis, past_length..self.offset, ..));
            let window_v = values.index((Ellipsis, past_length..self.offset, ..));
            let kept_k = take_axis(&window_k, keep_indices, -2)?;
            let kept_v = take_axis(&window_v, keep_indices, -2)?;
            keys.index_mut(
                (Ellipsis, past_length..past_length + keep_count, ..),
                &kept_k,
            );
            values.index_mut(
                (Ellipsis, past_length..past_length + keep_count, ..),
                &kept_v,
            );
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
