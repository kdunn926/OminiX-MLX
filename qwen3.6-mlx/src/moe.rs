//! Mixture of Experts with shared expert for Qwen3.6.
//!
//! Each layer has 256 routed experts (top-8 selected) plus 1 shared expert
//! that always participates. The routing gate and shared expert gate use
//! 8-bit quantization for accuracy; expert weights use 4-bit.

use mlx_rs::{
    error::Exception,
    macros::ModuleParameters,
    module::{Module, Param},
    nn,
    ops::{
        self,
        indexing::{IndexOp, NewAxis, take_along_axis, take_axis},
    },
    quantization::MaybeQuantized,
    Array,
};

use mlx_rs_core::fused_swiglu;

// ============================================================================
// QuantizedSwitchLinear — stacked expert weights for gather_qmm
// ============================================================================

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
    pub fn apply(&self, x: &Array, indices: &Array, sorted_indices: bool) -> Result<Array, Exception> {
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

// ============================================================================
// SwitchGLU — efficient batched expert MLP via gather_qmm
// ============================================================================

/// Sort tokens by expert indices for coalesced memory access.
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

/// Unsort output back to original token order.
fn scatter_unsort(x: &Array, inv_order: &Array, original_shape: &[i32]) -> Result<Array, Exception> {
    let x_shape = x.shape();
    let d = *x_shape.last().unwrap() as i32;

    let x_flat = x.reshape(&[-1, d])?;
    let x_unsorted = take_axis(&x_flat, inv_order, 0)?;

    let mut new_shape: Vec<i32> = original_shape.to_vec();
    new_shape.push(1);
    new_shape.push(d);
    x_unsorted.reshape(&new_shape)
}

#[derive(Debug, Clone, ModuleParameters)]
pub struct SwitchGLU {
    #[param]
    pub gate_proj: QuantizedSwitchLinear,
    #[param]
    pub up_proj: QuantizedSwitchLinear,
    #[param]
    pub down_proj: QuantizedSwitchLinear,
}

impl SwitchGLU {
    /// x: [B, L, D], indices: [B, L, k] -> output: [B, L, k, D]
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
            let activated = fused_swiglu(&up, &gate)?;
            let output = self.down_proj.apply(&activated, &indices_sorted, true)?;

            let output_unsorted = scatter_unsort(&output, &inv_order, &[b as i32, l as i32, k as i32])?;
            let shape = output_unsorted.shape();
            output_unsorted.reshape(&[shape[0] as i32, shape[1] as i32, shape[2] as i32, shape[4] as i32])
        } else {
            let gate = self.gate_proj.apply(&x_expanded, indices, false)?;
            let up = self.up_proj.apply(&x_expanded, indices, false)?;
            let activated = fused_swiglu(&up, &gate)?;
            let output = self.down_proj.apply(&activated, indices, false)?;

            let shape = output.shape();
            if shape.len() == 5 {
                output.reshape(&[shape[0] as i32, shape[1] as i32, shape[2] as i32, shape[4] as i32])
            } else {
                Ok(output)
            }
        }
    }
}

// ============================================================================
// Shared Expert — dense MLP that always participates
// ============================================================================

#[derive(Debug, ModuleParameters)]
pub struct SharedExpert {
    #[param]
    pub gate_proj: MaybeQuantized<nn::Linear>,
    #[param]
    pub up_proj: MaybeQuantized<nn::Linear>,
    #[param]
    pub down_proj: MaybeQuantized<nn::Linear>,
}

impl Module<&Array> for SharedExpert {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Self::Output, Self::Error> {
        let gate = nn::silu(self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&gate.multiply(up)?)
    }

    fn training_mode(&mut self, _mode: bool) {}
}

// ============================================================================
// MoeBlock — top-k routing + shared expert
// ============================================================================

/// MoE block with top-k routed experts and a shared expert.
///
/// For each token:
///   output = sum_k(score_k * expert_k(x)) + shared_expert_gate(x) * shared_expert(x)
#[derive(Debug, ModuleParameters)]
pub struct MoeBlock {
    pub num_experts: i32,
    pub top_k: i32,

    #[param]
    pub gate: MaybeQuantized<nn::Linear>,
    #[param]
    pub switch_mlp: SwitchGLU,
    #[param]
    pub shared_expert: SharedExpert,
    #[param]
    pub shared_expert_gate: MaybeQuantized<nn::Linear>,
}

impl Module<&Array> for MoeBlock {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Self::Output, Self::Error> {
        // 1. Route to top-k experts
        let gates = self.gate.forward(x)?;
        let gates = ops::softmax_axis(&gates, -1, true)?;

        let neg_gates = gates.negative()?;
        let partitioned_inds = ops::argpartition_axis(&neg_gates, self.top_k - 1, -1)?;
        let top_k_indices = partitioned_inds.index((.., .., ..self.top_k));

        let top_k_scores = take_along_axis(&gates, &top_k_indices, -1)?;

        // Normalize scores
        let score_sum = top_k_scores.sum_axis(-1, true)?;
        let top_k_scores = top_k_scores.divide(&score_sum)?;

        // 2. Routed expert outputs
        let expert_out = self.switch_mlp.forward_experts(x, &top_k_indices)?;
        let scores_expanded = top_k_scores.index((.., .., .., NewAxis));
        let routed = expert_out.multiply(&scores_expanded)?.sum_axis(2, false)?;

        // 3. Shared expert output, gated by sigmoid
        let shared_out = self.shared_expert.forward(x)?;
        let shared_gate = nn::sigmoid(self.shared_expert_gate.forward(x)?)?;
        let shared = shared_out.multiply(&shared_gate)?;

        // 4. Combine
        routed.add(shared)
    }

    fn training_mode(&mut self, _mode: bool) {}
}
