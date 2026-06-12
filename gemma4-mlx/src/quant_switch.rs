//! Quantized expert MoE via MLX `gather_qmm` (ported from qwen3.6-mlx).
//!
//! Used for models that ship MoE weights in the "switch_glu" layout
//! (three separate per-expert projections, each quantized) — primarily
//! the UD-MLX-4bit variant of Gemma4-26B-A4B-it. Preserves the Q4/Q6/Q8
//! packed storage so resident memory stays at ~15 GB vs the ~50 GB of
//! the dequantized path.
//!
//! Mirrors the structure of `qwen3.6-mlx/src/moe.rs`. The arithmetic is
//! the same; the only difference is how `gemma4-mlx`'s outer MoE block
//! plumbs `(top_k_weights, top_k_index)` into the SwitchGLU.

use mlx_rs::{
    error::Exception,
    macros::ModuleParameters,
    module::Param,
    ops::indexing::take_axis,
    Array,
};

use mlx_rs_core::fused_swiglu;

/// Per-expert stacked-weight linear projection, quantized.
/// Weight shape (packed): `[E, output_dims, input_dims / packing]`.
#[derive(Debug, Clone, ModuleParameters)]
pub struct QuantizedSwitchLinear {
    pub num_experts: i32,
    pub input_dims: i32,
    pub output_dims: i32,
    pub group_size: i32,
    pub bits: i32,

    #[param]
    pub weight: Param<Array>,
    #[param]
    pub scales: Param<Array>,
    #[param]
    pub biases: Param<Array>,
}

impl QuantizedSwitchLinear {
    /// Apply gather_qmm with expert selection indices.
    /// `x` is `[batch_dims..., 1, 1, input_dims]`; `indices` is
    /// `[batch_dims..., k]` (or flattened to `[N]` when `sorted_indices`).
    pub fn apply(
        &self,
        x: &Array,
        indices: &Array,
        sorted_indices: bool,
    ) -> Result<Array, Exception> {
        mlx_rs::ops::gather_qmm(
            x,
            &*self.weight,
            &*self.scales,
            &*self.biases,
            None::<&Array>,
            Some(indices),
            true,
            self.group_size,
            self.bits,
            None::<&str>,
            sorted_indices,
        )
    }
}

/// Sort tokens by their assigned expert id so consecutive lanes share an
/// expert weight matrix. Returns `(x_sorted, indices_sorted, inv_order)`
/// where `inv_order` is the permutation that undoes the sort.
fn gather_sort(x: &Array, indices: &Array) -> Result<(Array, Array, Array), Exception> {
    let indices_shape = indices.shape();
    let m = *indices_shape.last().unwrap() as i32;

    let indices_flat = indices.flatten(None, None)?;
    let order = mlx_rs::ops::argsort(&indices_flat)?;
    let inv_order = mlx_rs::ops::argsort(&order)?;

    let x_shape = x.shape();
    let d = *x_shape.last().unwrap() as i32;
    let x_flat = x.reshape(&[-1, 1, d])?;

    let token_order = order.floor_divide(mlx_rs::array!(m))?;
    let x_sorted = take_axis(&x_flat, &token_order, 0)?;
    let indices_sorted = take_axis(&indices_flat, &order, 0)?;

    Ok((x_sorted, indices_sorted, inv_order))
}

/// Inverse permutation back to original (token, k_slot) order.
fn scatter_unsort(
    x: &Array,
    inv_order: &Array,
    original_shape: &[i32],
) -> Result<Array, Exception> {
    let x_shape = x.shape();
    let d = *x_shape.last().unwrap() as i32;

    let x_flat = x.reshape(&[-1, d])?;
    let x_unsorted = take_axis(&x_flat, inv_order, 0)?;

    let mut new_shape: Vec<i32> = original_shape.to_vec();
    new_shape.push(1);
    new_shape.push(d);
    x_unsorted.reshape(&new_shape)
}

/// Three-projection SwiGLU expert MLP with `gather_qmm` dispatch.
///
/// Equivalent to `gate*up → activation → down` per expert, where the
/// expert selection is encoded in `indices` of shape `[B, L, k]` and
/// each output position multiplies its k contributions by the routing
/// weights at the caller.
#[derive(Debug, Clone, ModuleParameters)]
pub struct SwitchGLU {
    #[param]
    pub gate_proj: QuantizedSwitchLinear,
    #[param]
    pub up_proj: QuantizedSwitchLinear,
    #[param]
    pub down_proj: QuantizedSwitchLinear,

    /// Expert activation. Gemma4 UD checkpoints use SwiGLU (the fused
    /// kernel); DiffusionGemma experts use geglu (gelu_approx(gate) * up).
    pub activation: crate::model::GemmaActivation,
}

impl SwitchGLU {
    fn activate(&self, gate: &Array, up: &Array) -> Result<Array, Exception> {
        match self.activation {
            crate::model::GemmaActivation::Silu => fused_swiglu(up, gate),
            other => other.apply(gate)?.multiply(up),
        }
    }
}

impl SwitchGLU {
    /// `x: [B, L, D]`, `indices: [B, L, k]` → output `[B, L, k, D]`.
    /// Caller multiplies by per-(token, k_slot) routing weights and
    /// reduces along the k axis.
    pub fn forward_experts(&mut self, x: &Array, indices: &Array) -> Result<Array, Exception> {
        let indices_shape = indices.shape();
        let b = indices_shape[0];
        let l = indices_shape[1];
        let k = indices_shape[2];

        let x_expanded = mlx_rs::ops::expand_dims(x, -2)?;
        let x_expanded = mlx_rs::ops::expand_dims(&x_expanded, -2)?;

        let indices_size = b * l * k;
        let do_sort = indices_size >= 64;

        if do_sort {
            let (x_sorted, indices_sorted, inv_order) = gather_sort(&x_expanded, indices)?;

            let gate = self.gate_proj.apply(&x_sorted, &indices_sorted, true)?;
            let up = self.up_proj.apply(&x_sorted, &indices_sorted, true)?;
            let activated = self.activate(&gate, &up)?;
            let output = self.down_proj.apply(&activated, &indices_sorted, true)?;

            let output_unsorted =
                scatter_unsort(&output, &inv_order, &[b as i32, l as i32, k as i32])?;
            let shape = output_unsorted.shape();
            output_unsorted.reshape(&[
                shape[0] as i32,
                shape[1] as i32,
                shape[2] as i32,
                shape[4] as i32,
            ])
        } else {
            let gate = self.gate_proj.apply(&x_expanded, indices, false)?;
            let up = self.up_proj.apply(&x_expanded, indices, false)?;
            let activated = self.activate(&gate, &up)?;
            let output = self.down_proj.apply(&activated, indices, false)?;

            let shape = output.shape();
            if shape.len() == 5 {
                output.reshape(&[
                    shape[0] as i32,
                    shape[1] as i32,
                    shape[2] as i32,
                    shape[4] as i32,
                ])
            } else {
                Ok(output)
            }
        }
    }
}

/// Gemma4 SwitchGLU experts wrapper. Drop-in alternative to
/// `crate::model::Experts` when MoE weights ship in quantized
/// switch_glu format. The router output (`top_k_index`, `top_k_weights`)
/// shape from `crate::model::Router` is `[n, k]` for our flattened
/// (B*L → n) inputs; this wrapper reshapes through `[n, 1, k]` to match
/// `SwitchGLU::forward_experts` and reduces the final routing
/// multiplication back to `[n, hidden]`.
#[derive(Debug, Clone, ModuleParameters)]
pub struct SwitchGluExperts {
    pub hidden_size: i32,
    pub intermediate_size: i32,

    #[param]
    pub switch_glu: SwitchGLU,
}

impl SwitchGluExperts {
    /// `hidden_states: [n, H]`, `top_k_index: [n, k]`, `top_k_weights: [n, k]`
    /// → output `[n, H]`. Mirrors `Experts::forward_topk`'s contract so the
    /// outer DecoderLayer can swap this in transparently.
    pub fn forward_topk(
        &mut self,
        hidden_states: &Array,
        top_k_index: &Array,
        top_k_weights: &Array,
    ) -> Result<Array, Exception> {
        let n = hidden_states.shape()[0];
        let h = hidden_states.shape()[1];
        let k = top_k_index.shape()[1];

        // SwitchGLU expects `[B, L, D]` for x and `[B, L, k]` for indices.
        // Wrap our flattened `[n, H]` as `[n, 1, H]`.
        let x = hidden_states.reshape(&[n, 1, h])?;
        let idx = top_k_index.reshape(&[n, 1, k])?;
        // Output shape: `[n, 1, k, H]`.
        let expert_out = self.switch_glu.forward_experts(&x, &idx)?;

        // Multiply by routing weights and sum over k.
        let weighted = expert_out
            .reshape(&[n, k, h])?
            .multiply(&top_k_weights.reshape(&[n, k, 1])?)?;
        weighted.sum_axis(1, false)
    }

    pub fn training_mode(&mut self, _mode: bool) {}
}
