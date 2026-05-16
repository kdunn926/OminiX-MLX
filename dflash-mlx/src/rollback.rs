use crate::kernels::tape_replay;
use mlx_rs::{error::Exception, ops::indexing::IndexOp, Array};

/// Per-GDN-layer rollback cache for speculative-decoding verify passes.
///
/// Layout (matches `mlx_rs_core::deltanet_recurrence`):
///   state      : [B, H, K, V]
///   conv_state : [B, conv_dim, kernel_size - 1]
///   tape       : [B, H, L, V]   (recorded delta per verify step)
///   tape_k     : [B, H, L, K]
///   tape_decay : [B, H, L]      (= exp(g), the kernel decay tensor)
#[derive(Debug, Clone, Default)]
pub struct RecurrentRollbackCache {
    pub state: Option<Array>,
    pub conv_state: Option<Array>,
    snapshot_state: Option<Array>,
    snapshot_conv: Option<Array>,
    snapshot_step: i32,
    tape: Option<Array>,
    tape_k: Option<Array>,
    tape_decay: Option<Array>,
    pub step: i32,
}

impl RecurrentRollbackCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Call before running target verify to snapshot current state.
    pub fn arm_rollback(&mut self) {
        self.snapshot_state = self.state.clone();
        self.snapshot_conv = self.conv_state.clone();
        self.snapshot_step = self.step;
    }

    /// Record tape, keys, decay for the verify pass.
    pub fn record_tape(&mut self, tape: Array, k: Array, decay: Array) {
        self.tape = Some(tape);
        self.tape_k = Some(k);
        self.tape_decay = Some(decay);
    }

    /// Roll back to snapshot, then replay only the accepted steps via tape_replay kernel.
    pub fn rollback(&mut self, n_accepted: usize) -> Result<(), Exception> {
        self.state = self.snapshot_state.clone();
        self.conv_state = self.snapshot_conv.clone();
        self.step = self.snapshot_step;

        if n_accepted == 0 {
            return Ok(());
        }

        let state = self
            .state
            .clone()
            .ok_or_else(|| Exception::custom("rollback requested without a snapshotted state"))?;
        let tape = self
            .tape
            .as_ref()
            .ok_or_else(|| Exception::custom("rollback requested without recorded tape"))?;
        let tape_k = self
            .tape_k
            .as_ref()
            .ok_or_else(|| Exception::custom("rollback requested without recorded keys"))?;
        let tape_decay = self
            .tape_decay
            .as_ref()
            .ok_or_else(|| Exception::custom("rollback requested without recorded decay"))?;

        // Tape layout: [B, H, L, V] — accepted prefix is the first n_accepted
        // entries along the L axis.
        let total_steps = tape.shape()[2] as usize;
        if n_accepted > total_steps {
            return Err(Exception::custom(format!(
                "rollback requested {n_accepted} accepted steps, but only {total_steps} tape steps are available"
            )));
        }

        let end = n_accepted as i32;
        let replayed = tape_replay(
            &tape.index((.., .., ..end, ..)),
            &tape_k.index((.., .., ..end, ..)),
            &tape_decay.index((.., .., ..end)),
            &state,
        )?;
        self.state = Some(replayed);
        self.step = self.snapshot_step + n_accepted as i32;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::Array;

    fn assert_arrays_close(a: &Array, b: &Array, tol: f32) {
        mlx_rs::transforms::eval([a, b]).unwrap();
        let a = a.contiguous().unwrap();
        let b = b.contiguous().unwrap();
        mlx_rs::transforms::eval([&a, &b]).unwrap();
        let max_diff = a
            .as_slice::<f32>()
            .iter()
            .zip(b.as_slice::<f32>().iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < tol, "max diff {max_diff} >= {tol}");
    }

    /// Replay of an accepted prefix from a snapshotted state should match a
    /// `deltanet_recurrence` run over the same prefix from the same state.
    #[test]
    fn test_recurrent_rollback_replays_accepted_prefix() {
        let _guard = crate::mlx_test_guard();

        // Small but realistic shapes (K=32 is the minimum that satisfies the
        // K%32==0 requirement of the Metal kernels).
        let b = 1i32;
        let h = 2i32;
        let l = 3i32;
        let k_dim = 32i32;
        let v_dim = 4i32;

        let q_data: Vec<f32> = (0..(b * h * l * k_dim) as usize)
            .map(|i| ((i as f32) * 0.011 - 0.4).tanh())
            .collect();
        let k_data: Vec<f32> = (0..(b * h * l * k_dim) as usize)
            .map(|i| ((i as f32) * 0.013 - 0.3).tanh())
            .collect();
        let v_data: Vec<f32> = (0..(b * h * l * v_dim) as usize)
            .map(|i| ((i as f32) * 0.009).sin() * 0.4)
            .collect();
        let decay_data: Vec<f32> = (0..(b * h * l) as usize)
            .map(|i| (-((i as f32) * 0.1 + 0.1).exp()).exp())
            .collect();
        let beta_data: Vec<f32> = (0..(b * h * l) as usize)
            .map(|i| 1.0 / (1.0 + (-((i as f32) * 0.2)).exp()))
            .collect();
        let state_in = Array::zeros::<f32>(&[b, h, k_dim, v_dim]).unwrap();

        let q = Array::from_slice(&q_data, &[b, h, l, k_dim]);
        let k = Array::from_slice(&k_data, &[b, h, l, k_dim]);
        let v = Array::from_slice(&v_data, &[b, h, l, v_dim]);
        let decay = Array::from_slice(&decay_data, &[b, h, l]);
        let beta = Array::from_slice(&beta_data, &[b, h, l]);

        // Record a tape over all L steps from the snapshot state.
        let (_out, _full_state, tape) =
            mlx_rs_core::deltanet_with_tape(&q, &k, &v, &decay, &beta, &state_in)
                .expect("tape capture failed");

        let conv_snapshot = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 2, 2]);

        let mut cache = RecurrentRollbackCache::new();
        cache.state = Some(state_in.clone());
        cache.conv_state = Some(conv_snapshot.clone());
        cache.step = 7;
        cache.arm_rollback();
        // Pretend the verify pass moved state forward (will be discarded).
        cache.state = Some(Array::zeros::<f32>(&[b, h, k_dim, v_dim]).unwrap());
        cache.conv_state = Some(Array::zeros::<f32>(&[1, 2, 2]).unwrap());
        cache.step = 99;
        cache.record_tape(tape, k.clone(), decay.clone());

        // Roll back, keeping the first 2 verify steps.
        cache.rollback(2).unwrap();

        assert_eq!(cache.step, 9);
        assert_arrays_close(cache.conv_state.as_ref().unwrap(), &conv_snapshot, 1e-6);

        // Expected state: re-run deltanet_recurrence over the first 2 steps
        // starting from the snapshot state.
        let q_p = q.index((.., .., ..2, ..));
        let k_p = k.index((.., .., ..2, ..));
        let v_p = v.index((.., .., ..2, ..));
        let decay_p = decay.index((.., .., ..2));
        let beta_p = beta.index((.., .., ..2));
        let (_, expected) =
            mlx_rs_core::deltanet_recurrence(&q_p, &k_p, &v_p, &decay_p, &beta_p, &state_in)
                .unwrap();
        assert_arrays_close(cache.state.as_ref().unwrap(), &expected, 1e-4);
    }
}
