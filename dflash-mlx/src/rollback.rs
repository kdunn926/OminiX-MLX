use crate::kernels::tape_replay;
use mlx_rs::{error::Exception, ops::indexing::IndexOp, Array};

#[derive(Debug, Clone, Default)]
pub struct RecurrentRollbackCache {
    /// GDN recurrent state [B, Hv, Dv, Dk]
    pub state: Option<Array>,
    /// Conv1d sliding window [B, conv_dim, kernel_size-1]
    pub conv_state: Option<Array>,
    /// Pre-verify snapshot of state and conv_state
    snapshot_state: Option<Array>,
    snapshot_conv: Option<Array>,
    snapshot_step: i32,
    /// Recorded tape, keys, gates from verify pass
    tape: Option<Array>,
    tape_k: Option<Array>,
    tape_g: Option<Array>,
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

    /// Record tape, keys, gates for the verify pass.
    pub fn record_tape(&mut self, tape: Array, k: Array, g: Array) {
        self.tape = Some(tape);
        self.tape_k = Some(k);
        self.tape_g = Some(g);
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
        let tape_g = self
            .tape_g
            .as_ref()
            .ok_or_else(|| Exception::custom("rollback requested without recorded gates"))?;

        let total_steps = tape.shape()[1] as usize;
        if n_accepted > total_steps {
            return Err(Exception::custom(format!(
                "rollback requested {n_accepted} accepted steps, but only {total_steps} tape steps are available"
            )));
        }

        let end = n_accepted as i32;
        let replayed = tape_replay(
            &tape.index((.., ..end, .., ..)),
            &tape_k.index((.., ..end, .., ..)),
            &tape_g.index((.., ..end, ..)),
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

    #[test]
    fn test_recurrent_rollback_replays_accepted_prefix() {
        let _guard = crate::mlx_test_guard();
        let snapshot_state = Array::from_slice(
            &[
                0.10f32, 0.20, 0.30, 0.40, -0.10, 0.00, 0.20, 0.10, 0.05, 0.15, -0.05, 0.25, 0.30,
                -0.20, 0.10, 0.00, 0.20, 0.10, -0.10, 0.00, 0.05, 0.25, 0.15, -0.05, 0.10, -0.15,
                0.20, 0.30, 0.00, 0.10, 0.05, 0.15,
            ],
            &[1, 2, 4, 4],
        );
        let conv_snapshot = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 2, 2]);
        let tape = Array::from_slice(
            &[
                0.30f32, -0.10, 0.20, 0.40, 0.05, 0.25, -0.15, 0.35, 0.45, 0.10, -0.05, 0.20, 0.15,
                0.30, 0.10, -0.20,
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
        let g = Array::from_slice(&[0.90f32, 0.80, 0.85, 0.75], &[1, 2, 2]);

        let mut cache = RecurrentRollbackCache::new();
        cache.state = Some(snapshot_state.clone());
        cache.conv_state = Some(conv_snapshot.clone());
        cache.step = 7;
        cache.arm_rollback();
        cache.state = Some(Array::zeros::<f32>(&[1, 2, 4, 4]).unwrap());
        cache.conv_state = Some(Array::zeros::<f32>(&[1, 2, 2]).unwrap());
        cache.step = 99;
        cache.record_tape(tape.clone(), k.clone(), g.clone());
        cache.rollback(1).unwrap();

        assert_eq!(cache.step, 8);
        assert_arrays_close(cache.conv_state.as_ref().unwrap(), &conv_snapshot, 1e-6);

        let expected = crate::kernels::tape_replay(
            &tape.index((.., ..1, .., ..)),
            &k.index((.., ..1, .., ..)),
            &g.index((.., ..1, ..)),
            &snapshot_state,
        )
        .unwrap();
        assert_arrays_close(cache.state.as_ref().unwrap(), &expected, 1e-4);
    }
}
