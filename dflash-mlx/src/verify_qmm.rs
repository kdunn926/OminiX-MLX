use mlx_rs::{error::Exception, Array};

/// Verification fallback for the planned M=16 int4 matmul path.
///
/// This currently delegates to regular MLX matmul so the rest of the crate can
/// compile and integrate while the specialized kernel lands in a later step.
pub fn verify_qmm_m16(lhs: &Array, rhs: &Array) -> Result<Array, Exception> {
    if lhs.shape().len() < 2 || rhs.shape().len() < 2 {
        return Err(Exception::custom(
            "verify_qmm_m16 expects rank-2-or-higher lhs and rhs arrays",
        ));
    }
    lhs.matmul(rhs)
}
