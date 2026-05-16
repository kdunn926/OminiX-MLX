use mlx_rs::{error::Exception, Array, Dtype};
use std::ffi::CString;
use std::sync::OnceLock;

const GATED_DELTA_WITH_TAPE_SOURCE: &str = r#"
    constexpr uint THREADS = 32;
    constexpr uint DV_LANES = 4;
    constexpr uint DK_PER_THREAD = (DK + THREADS - 1) / THREADS;
    constexpr uint KV_REPEAT = Hv / Hk;

    uint tid_x = thread_position_in_threadgroup.x;
    uint tid_y = thread_position_in_threadgroup.y;
    uint dv = threadgroup_position_in_grid.y * DV_LANES + tid_y;
    if (dv >= Dv) {
        return;
    }

    uint bh = threadgroup_position_in_grid.z;
    uint b = bh / Hv;
    uint hv = bh % Hv;
    uint hk = hv / KV_REPEAT;

    uint state_base = (((b * Hv + hv) * Dv) + dv) * DK;
    T state_local[DK_PER_THREAD];
    for (uint kk = 0; kk < DK_PER_THREAD; ++kk) {
        uint dk_idx = tid_x + kk * THREADS;
        state_local[kk] = dk_idx < DK ? state_in[state_base + dk_idx] : T(0);
    }

    for (uint t = 0; t < T_LEN; ++t) {
        uint qk_base = (((b * T_LEN + t) * Hk) + hk) * DK;
        uint vh_base = (((b * T_LEN + t) * Hv) + hv) * Dv;
        uint gate_idx = ((b * T_LEN + t) * Hv) + hv;

        T local_kv = T(0);
        T local_q = T(0);
        for (uint kk = 0; kk < DK_PER_THREAD; ++kk) {
            uint dk_idx = tid_x + kk * THREADS;
            if (dk_idx < DK) {
                T k_val = k[qk_base + dk_idx];
                local_kv += state_local[kk] * k_val;
                local_q += state_local[kk] * q[qk_base + dk_idx];
            }
        }

        T innovation = (v[vh_base + dv] - simd_sum(local_kv)) * beta[gate_idx];
        T gate = g[gate_idx];

        T local_out = T(0);
        for (uint kk = 0; kk < DK_PER_THREAD; ++kk) {
            uint dk_idx = tid_x + kk * THREADS;
            if (dk_idx < DK) {
                T k_val = k[qk_base + dk_idx];
                T q_val = q[qk_base + dk_idx];
                state_local[kk] = state_local[kk] * gate + k_val * innovation;
                local_out += state_local[kk] * q_val;
            }
        }

        T out_val = simd_sum(local_out);
        if (tid_x == 0) {
            y[vh_base + dv] = out_val;
            innovation_tape[vh_base + dv] = innovation;
        }
    }

    for (uint kk = 0; kk < DK_PER_THREAD; ++kk) {
        uint dk_idx = tid_x + kk * THREADS;
        if (dk_idx < DK) {
            state_out[state_base + dk_idx] = state_local[kk];
        }
    }
"#;

const TAPE_REPLAY_SOURCE: &str = r#"
    constexpr uint THREADS = 32;
    constexpr uint DV_LANES = 4;
    constexpr uint DK_PER_THREAD = (DK + THREADS - 1) / THREADS;
    constexpr uint KV_REPEAT = Hv / Hk;

    uint tid_x = thread_position_in_threadgroup.x;
    uint tid_y = thread_position_in_threadgroup.y;
    uint dv = threadgroup_position_in_grid.y * DV_LANES + tid_y;
    if (dv >= Dv) {
        return;
    }

    uint bh = threadgroup_position_in_grid.z;
    uint b = bh / Hv;
    uint hv = bh % Hv;
    uint hk = hv / KV_REPEAT;

    uint state_base = (((b * Hv + hv) * Dv) + dv) * DK;
    T state_local[DK_PER_THREAD];
    for (uint kk = 0; kk < DK_PER_THREAD; ++kk) {
        uint dk_idx = tid_x + kk * THREADS;
        state_local[kk] = dk_idx < DK ? state_in[state_base + dk_idx] : T(0);
    }

    for (uint t = 0; t < T_LEN; ++t) {
        uint qk_base = (((b * T_LEN + t) * Hk) + hk) * DK;
        uint tape_base = (((b * T_LEN + t) * Hv) + hv) * Dv;
        uint gate_idx = ((b * T_LEN + t) * Hv) + hv;
        T gate = g[gate_idx];
        T innovation = tape[tape_base + dv];

        for (uint kk = 0; kk < DK_PER_THREAD; ++kk) {
            uint dk_idx = tid_x + kk * THREADS;
            if (dk_idx < DK) {
                T k_val = k[qk_base + dk_idx];
                state_local[kk] = state_local[kk] * gate + k_val * innovation;
            }
        }
    }

    for (uint kk = 0; kk < DK_PER_THREAD; ++kk) {
        uint dk_idx = tid_x + kk * THREADS;
        if (dk_idx < DK) {
            state_out[state_base + dk_idx] = state_local[kk];
        }
    }
"#;

static GATED_DELTA_WITH_TAPE_KERNEL: OnceLock<KernelState> = OnceLock::new();
static TAPE_REPLAY_KERNEL: OnceLock<KernelState> = OnceLock::new();

struct KernelState {
    kernel: mlx_sys::mlx_fast_metal_kernel,
    input_names: mlx_sys::mlx_vector_string,
    output_names: mlx_sys::mlx_vector_string,
}

unsafe impl Send for KernelState {}
unsafe impl Sync for KernelState {}

impl Drop for KernelState {
    fn drop(&mut self) {
        unsafe {
            mlx_sys::mlx_fast_metal_kernel_free(self.kernel);
            mlx_sys::mlx_vector_string_free(self.input_names);
            mlx_sys::mlx_vector_string_free(self.output_names);
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Dims {
    b: i32,
    t: i32,
    hk: i32,
    dk: i32,
    hv: i32,
    dv: i32,
}

fn create_kernel(name: &str, source: &str, inputs: &[&str], outputs: &[&str]) -> KernelState {
    unsafe {
        let input_names = mlx_sys::mlx_vector_string_new();
        for input in inputs {
            let input = CString::new(*input).unwrap();
            mlx_sys::mlx_vector_string_append_value(input_names, input.as_ptr());
        }

        let output_names = mlx_sys::mlx_vector_string_new();
        for output in outputs {
            let output = CString::new(*output).unwrap();
            mlx_sys::mlx_vector_string_append_value(output_names, output.as_ptr());
        }

        let source = CString::new(source).unwrap();
        let header = CString::new("").unwrap();
        let name = CString::new(name).unwrap();
        let kernel = mlx_sys::mlx_fast_metal_kernel_new(
            name.as_ptr(),
            input_names,
            output_names,
            source.as_ptr(),
            header.as_ptr(),
            true,
            false,
        );

        KernelState {
            kernel,
            input_names,
            output_names,
        }
    }
}

fn create_gated_delta_with_tape_kernel() -> KernelState {
    create_kernel(
        "gated_delta_with_tape",
        GATED_DELTA_WITH_TAPE_SOURCE,
        &["q", "k", "v", "g", "beta", "state_in"],
        &["y", "state_out", "innovation_tape"],
    )
}

fn create_tape_replay_kernel() -> KernelState {
    create_kernel(
        "tape_replay",
        TAPE_REPLAY_SOURCE,
        &["tape", "k", "g", "state_in"],
        &["state_out"],
    )
}

fn ensure_same_dtype(base: Dtype, arrays: &[(&str, &Array)]) -> Result<(), Exception> {
    for (name, array) in arrays {
        if array.dtype() != base {
            return Err(Exception::custom(format!(
                "dtype mismatch for {name}: expected {:?}, found {:?}",
                base,
                array.dtype()
            )));
        }
    }
    if !base.is_float() {
        return Err(Exception::custom(format!(
            "expected floating-point dtype, found {:?}",
            base
        )));
    }
    Ok(())
}

fn validate_gated_inputs(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state_in: &Array,
) -> Result<Dims, Exception> {
    let q_shape = q.shape();
    let k_shape = k.shape();
    let v_shape = v.shape();
    let g_shape = g.shape();
    let beta_shape = beta.shape();
    let state_shape = state_in.shape();

    if q_shape.len() != 4
        || k_shape.len() != 4
        || v_shape.len() != 4
        || g_shape.len() != 3
        || beta_shape.len() != 3
        || state_shape.len() != 4
    {
        return Err(Exception::custom(format!(
            "gated_delta_with_tape shape mismatch: q={q_shape:?} k={k_shape:?} v={v_shape:?} g={g_shape:?} beta={beta_shape:?} state={state_shape:?}"
        )));
    }

    let dims = Dims {
        b: q_shape[0],
        t: q_shape[1],
        hk: q_shape[2],
        dk: q_shape[3],
        hv: v_shape[2],
        dv: v_shape[3],
    };

    if k_shape != q_shape {
        return Err(Exception::custom(format!(
            "q/k shape mismatch: q={q_shape:?} k={k_shape:?}"
        )));
    }
    if g_shape != [dims.b, dims.t, dims.hv] || beta_shape != [dims.b, dims.t, dims.hv] {
        return Err(Exception::custom(format!(
            "g/beta shape mismatch: expected [{}, {}, {}], got g={g_shape:?} beta={beta_shape:?}",
            dims.b, dims.t, dims.hv
        )));
    }
    if state_shape != [dims.b, dims.hv, dims.dv, dims.dk] {
        return Err(Exception::custom(format!(
            "state_in shape mismatch: expected [{}, {}, {}, {}], got {state_shape:?}",
            dims.b, dims.hv, dims.dv, dims.dk
        )));
    }
    if dims.hk <= 0 || dims.hv <= 0 || dims.dk <= 0 || dims.dv <= 0 || dims.t <= 0 {
        return Err(Exception::custom("all kernel dimensions must be positive"));
    }
    if dims.hv % dims.hk != 0 {
        return Err(Exception::custom(format!(
            "Hv must be divisible by Hk, got Hv={} Hk={}",
            dims.hv, dims.hk
        )));
    }

    ensure_same_dtype(
        q.dtype(),
        &[
            ("k", k),
            ("v", v),
            ("g", g),
            ("beta", beta),
            ("state_in", state_in),
        ],
    )?;

    Ok(dims)
}

fn validate_replay_inputs(
    tape: &Array,
    k: &Array,
    g: &Array,
    state_in: &Array,
) -> Result<Dims, Exception> {
    let tape_shape = tape.shape();
    let k_shape = k.shape();
    let g_shape = g.shape();
    let state_shape = state_in.shape();

    if tape_shape.len() != 4 || k_shape.len() != 4 || g_shape.len() != 3 || state_shape.len() != 4 {
        return Err(Exception::custom(format!(
            "tape_replay shape mismatch: tape={tape_shape:?} k={k_shape:?} g={g_shape:?} state={state_shape:?}"
        )));
    }

    let dims = Dims {
        b: tape_shape[0],
        t: tape_shape[1],
        hk: k_shape[2],
        dk: k_shape[3],
        hv: tape_shape[2],
        dv: tape_shape[3],
    };

    if g_shape != [dims.b, dims.t, dims.hv] {
        return Err(Exception::custom(format!(
            "g shape mismatch: expected [{}, {}, {}], got {g_shape:?}",
            dims.b, dims.t, dims.hv
        )));
    }
    if state_shape != [dims.b, dims.hv, dims.dv, dims.dk] {
        return Err(Exception::custom(format!(
            "state_in shape mismatch: expected [{}, {}, {}, {}], got {state_shape:?}",
            dims.b, dims.hv, dims.dv, dims.dk
        )));
    }
    if dims.hv % dims.hk != 0 {
        return Err(Exception::custom(format!(
            "Hv must be divisible by Hk, got Hv={} Hk={}",
            dims.hv, dims.hk
        )));
    }

    ensure_same_dtype(tape.dtype(), &[("k", k), ("g", g), ("state_in", state_in)])?;

    Ok(dims)
}

fn configure_common(config: mlx_sys::mlx_fast_metal_kernel_config, dims: Dims, dtype: u32) {
    unsafe {
        let t_name = CString::new("T").unwrap();
        mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
            config,
            t_name.as_ptr(),
            dtype,
        );

        for (name, value) in [
            ("T_LEN", dims.t),
            ("Hv", dims.hv),
            ("Dv", dims.dv),
            ("DK", dims.dk),
            ("Hk", dims.hk),
        ] {
            let name = CString::new(name).unwrap();
            mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
                config,
                name.as_ptr(),
                value,
            );
        }

        let blocks_y = (dims.dv + 3) / 4;
        mlx_sys::mlx_fast_metal_kernel_config_set_grid(config, 32, blocks_y * 4, dims.b * dims.hv);
        mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(config, 32, 4, 1);
    }
}

pub fn gated_delta_with_tape(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state_in: &Array,
) -> Result<(Array, Array, Array), Exception> {
    let kernel = GATED_DELTA_WITH_TAPE_KERNEL.get_or_init(create_gated_delta_with_tape_kernel);
    let dims = validate_gated_inputs(q, k, v, g, beta, state_in)?;
    let dtype: u32 = q.dtype().into();

    unsafe {
        // SAFETY: raw MLX Metal-kernel APIs require FFI calls. Shapes, dtypes, and
        // output descriptors are validated above before dispatch.
        let stream = mlx_sys::mlx_default_gpu_stream_new();
        let config = mlx_sys::mlx_fast_metal_kernel_config_new();
        configure_common(config, dims, dtype);

        let y_shape = vec![dims.b, dims.t, dims.hv, dims.dv];
        let state_shape = vec![dims.b, dims.hv, dims.dv, dims.dk];
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            y_shape.as_ptr(),
            y_shape.len(),
            dtype,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            state_shape.as_ptr(),
            state_shape.len(),
            dtype,
        );
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            y_shape.as_ptr(),
            y_shape.len(),
            dtype,
        );

        let inputs = mlx_sys::mlx_vector_array_new();
        mlx_sys::mlx_vector_array_append_value(inputs, q.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, k.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, v.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, g.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, beta.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, state_in.as_ptr());

        let mut outputs = mlx_sys::mlx_vector_array_new();
        let ret = mlx_sys::mlx_fast_metal_kernel_apply(
            &mut outputs,
            kernel.kernel,
            inputs,
            config,
            stream,
        );
        if ret != 0 {
            mlx_sys::mlx_fast_metal_kernel_config_free(config);
            mlx_sys::mlx_vector_array_free(inputs);
            mlx_sys::mlx_vector_array_free(outputs);
            mlx_sys::mlx_stream_free(stream);
            return Err(Exception::custom(
                "gated_delta_with_tape Metal kernel execution failed",
            ));
        }

        mlx_sys::mlx_synchronize(stream);

        let mut y_ptr = mlx_sys::mlx_array_new();
        mlx_sys::mlx_vector_array_get(&mut y_ptr, outputs, 0);
        let mut state_ptr = mlx_sys::mlx_array_new();
        mlx_sys::mlx_vector_array_get(&mut state_ptr, outputs, 1);
        let mut tape_ptr = mlx_sys::mlx_array_new();
        mlx_sys::mlx_vector_array_get(&mut tape_ptr, outputs, 2);

        mlx_sys::mlx_fast_metal_kernel_config_free(config);
        mlx_sys::mlx_vector_array_free(inputs);
        mlx_sys::mlx_vector_array_free(outputs);
        mlx_sys::mlx_stream_free(stream);

        Ok((
            Array::from_ptr(y_ptr),
            Array::from_ptr(state_ptr),
            Array::from_ptr(tape_ptr),
        ))
    }
}

pub fn tape_replay(
    tape: &Array,
    k: &Array,
    g: &Array,
    state_in: &Array,
) -> Result<Array, Exception> {
    let kernel = TAPE_REPLAY_KERNEL.get_or_init(create_tape_replay_kernel);
    let dims = validate_replay_inputs(tape, k, g, state_in)?;
    let dtype: u32 = tape.dtype().into();

    unsafe {
        // SAFETY: raw MLX Metal-kernel APIs require FFI calls. Shapes, dtypes, and
        // output descriptors are validated above before dispatch.
        let stream = mlx_sys::mlx_default_gpu_stream_new();
        let config = mlx_sys::mlx_fast_metal_kernel_config_new();
        configure_common(config, dims, dtype);

        let state_shape = vec![dims.b, dims.hv, dims.dv, dims.dk];
        mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
            config,
            state_shape.as_ptr(),
            state_shape.len(),
            dtype,
        );

        let inputs = mlx_sys::mlx_vector_array_new();
        mlx_sys::mlx_vector_array_append_value(inputs, tape.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, k.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, g.as_ptr());
        mlx_sys::mlx_vector_array_append_value(inputs, state_in.as_ptr());

        let mut outputs = mlx_sys::mlx_vector_array_new();
        let ret = mlx_sys::mlx_fast_metal_kernel_apply(
            &mut outputs,
            kernel.kernel,
            inputs,
            config,
            stream,
        );
        if ret != 0 {
            mlx_sys::mlx_fast_metal_kernel_config_free(config);
            mlx_sys::mlx_vector_array_free(inputs);
            mlx_sys::mlx_vector_array_free(outputs);
            mlx_sys::mlx_stream_free(stream);
            return Err(Exception::custom(
                "tape_replay Metal kernel execution failed",
            ));
        }

        mlx_sys::mlx_synchronize(stream);

        let mut state_ptr = mlx_sys::mlx_array_new();
        mlx_sys::mlx_vector_array_get(&mut state_ptr, outputs, 0);

        mlx_sys::mlx_fast_metal_kernel_config_free(config);
        mlx_sys::mlx_vector_array_free(inputs);
        mlx_sys::mlx_vector_array_free(outputs);
        mlx_sys::mlx_stream_free(stream);

        Ok(Array::from_ptr(state_ptr))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::indexing::{IndexMutOp, IndexOp};
    use mlx_rs::Array;

    fn assert_arrays_close(a: &Array, b: &Array, tol: f32, label: &str) {
        mlx_rs::transforms::eval([a, b]).unwrap();
        let a = a.contiguous().unwrap();
        let b = b.contiguous().unwrap();
        mlx_rs::transforms::eval([&a, &b]).unwrap();
        let a_slice = a.as_slice::<f32>();
        let b_slice = b.as_slice::<f32>();
        let max_diff = a_slice
            .iter()
            .zip(b_slice.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < tol, "{label}: max diff {max_diff} >= {tol}");
    }

    fn gated_delta_reference(
        q: &Array,
        k: &Array,
        v: &Array,
        g: &Array,
        beta: &Array,
        state_in: &Array,
    ) -> Result<(Array, Array, Array), Exception> {
        let shape = q.shape();
        let (b, t, hk, dk) = (shape[0], shape[1], shape[2], shape[3]);
        let hv = v.shape()[2];
        let dv = v.shape()[3];
        let repeat = hv / hk;
        let mut state = state_in.clone();
        let mut outputs = Vec::with_capacity(t as usize);
        let mut tapes = Vec::with_capacity(t as usize);

        for step in 0..t {
            let mut step_out = Vec::with_capacity((b * hv * dv) as usize);
            let mut step_tape = Vec::with_capacity((b * hv * dv) as usize);
            for batch in 0..b {
                for head_v in 0..hv {
                    let head_k = head_v / repeat;
                    let gate = g.index((batch, step, head_v)).item::<f32>();
                    let beta_t = beta.index((batch, step, head_v)).item::<f32>();
                    for value_dim in 0..dv {
                        let mut kv_mem = 0.0f32;
                        for key_dim in 0..dk {
                            kv_mem += state
                                .index((batch, head_v, value_dim, key_dim))
                                .item::<f32>()
                                * k.index((batch, step, head_k, key_dim)).item::<f32>();
                        }
                        let innovation = (v.index((batch, step, head_v, value_dim)).item::<f32>()
                            - kv_mem)
                            * beta_t;
                        step_tape.push(innovation);

                        let mut out = 0.0f32;
                        for key_dim in 0..dk {
                            let updated = state
                                .index((batch, head_v, value_dim, key_dim))
                                .item::<f32>()
                                * gate
                                + k.index((batch, step, head_k, key_dim)).item::<f32>()
                                    * innovation;
                            state.index_mut(
                                (batch, head_v, value_dim, key_dim),
                                &Array::from_slice(&[updated], &[]),
                            );
                            out += updated * q.index((batch, step, head_k, key_dim)).item::<f32>();
                        }
                        step_out.push(out);
                    }
                }
            }
            outputs.push(Array::from_slice(&step_out, &[b, hv, dv]));
            tapes.push(Array::from_slice(&step_tape, &[b, hv, dv]));
        }

        Ok((
            mlx_rs::ops::stack_axis(&outputs, 1)?,
            state,
            mlx_rs::ops::stack_axis(&tapes, 1)?,
        ))
    }

    #[test]
    fn test_gated_delta_tape_and_replay() {
        let _guard = crate::mlx_test_guard();
        let q = Array::from_slice(
            &[
                0.10f32, 0.20, -0.10, 0.30, 0.15, -0.05, 0.25, 0.40, 0.05, 0.35, -0.20, 0.10,
                -0.10, 0.15, 0.20, 0.30,
            ],
            &[1, 2, 2, 4],
        );
        let k = Array::from_slice(
            &[
                0.20f32, 0.00, -0.10, 0.30, 0.10, 0.25, 0.35, -0.20, 0.30, -0.15, 0.05, 0.10, 0.20,
                0.10, -0.25, 0.15,
            ],
            &[1, 2, 2, 4],
        );
        let v = Array::from_slice(
            &[
                0.30f32, -0.10, 0.20, 0.40, 0.05, 0.25, -0.15, 0.35, 0.45, 0.10, -0.05, 0.20, 0.15,
                0.30, 0.10, -0.20,
            ],
            &[1, 2, 2, 4],
        );
        let g = Array::from_slice(&[0.90f32, 0.80, 0.85, 0.75], &[1, 2, 2]);
        let beta = Array::from_slice(&[0.70f32, 0.60, 0.65, 0.55], &[1, 2, 2]);
        let state_in = Array::from_slice(
            &[
                0.01f32, 0.02, 0.03, 0.04, -0.02, 0.01, 0.00, 0.05, 0.03, -0.01, 0.02, 0.01, 0.04,
                0.02, -0.03, 0.01, 0.02, -0.02, 0.01, 0.00, 0.05, 0.01, -0.01, 0.02, -0.03, 0.04,
                0.02, -0.01, 0.01, 0.03, 0.00, 0.02,
            ],
            &[1, 2, 4, 4],
        );

        let (ref_y, ref_state, ref_tape) =
            gated_delta_reference(&q, &k, &v, &g, &beta, &state_in).unwrap();
        let (kernel_y, kernel_state, kernel_tape) =
            gated_delta_with_tape(&q, &k, &v, &g, &beta, &state_in).unwrap();

        assert_arrays_close(&kernel_y, &ref_y, 1e-4, "gated_delta output");
        assert_arrays_close(&kernel_state, &ref_state, 1e-4, "gated_delta state");
        assert_arrays_close(&kernel_tape, &ref_tape, 1e-4, "innovation tape");

        let tape_first = kernel_tape.index((.., ..1, .., ..));
        let k_first = k.index((.., ..1, .., ..));
        let g_first = g.index((.., ..1, ..));
        let replay_state = tape_replay(&tape_first, &k_first, &g_first, &state_in).unwrap();
        let expected_first = ref_state.index((.., .., .., ..));
        let (_, manual_first_state, _) = gated_delta_reference(
            &q.index((.., ..1, .., ..)),
            &k.index((.., ..1, .., ..)),
            &v.index((.., ..1, .., ..)),
            &g.index((.., ..1, ..)),
            &beta.index((.., ..1, ..)),
            &state_in,
        )
        .unwrap();
        let _ = expected_first; // keep full-run state check above explicit while using manual first step below.
        assert_arrays_close(
            &replay_state,
            &manual_first_state,
            1e-4,
            "tape replay state",
        );
    }
}
