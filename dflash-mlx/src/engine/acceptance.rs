use mlx_rs::{error::Exception, Array, Dtype};

/// Returns the count of the longest matching prefix between drafted and posterior tokens.
/// Uses cumprod(equal(drafted, posterior)) to find the longest prefix of 1s.
pub fn match_acceptance_length(drafted: &Array, posterior: &Array) -> Result<usize, Exception> {
    if drafted.shape().len() != 1 || posterior.shape().len() != 1 {
        return Err(Exception::custom(format!(
            "match_acceptance_length expects 1D arrays, got drafted={:?} posterior={:?}",
            drafted.shape(),
            posterior.shape()
        )));
    }
    if drafted.shape() != posterior.shape() {
        return Err(Exception::custom(format!(
            "match_acceptance_length shape mismatch: drafted={:?} posterior={:?}",
            drafted.shape(),
            posterior.shape()
        )));
    }
    if drafted.dtype() != Dtype::Uint32 || posterior.dtype() != Dtype::Uint32 {
        return Err(Exception::custom(format!(
            "match_acceptance_length expects u32 arrays, got drafted={:?} posterior={:?}",
            drafted.dtype(),
            posterior.dtype()
        )));
    }

    let matches = drafted.eq(posterior)?.as_dtype(Dtype::Uint32)?;
    let prefix = matches.cumprod(0, None, None)?;
    Ok(prefix.sum(false)?.item::<u32>() as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::Array;

    #[test]
    fn test_match_acceptance_length_exact_prefix() {
        let _guard = crate::mlx_test_guard();
        let drafted = Array::from_slice(&[11u32, 22, 33, 44], &[4]);
        let posterior = Array::from_slice(&[11u32, 22, 99, 44], &[4]);
        assert_eq!(match_acceptance_length(&drafted, &posterior).unwrap(), 2);
    }

    #[test]
    fn test_match_acceptance_length_all_match() {
        let _guard = crate::mlx_test_guard();
        let drafted = Array::from_slice(&[1u32, 2, 3], &[3]);
        let posterior = Array::from_slice(&[1u32, 2, 3], &[3]);
        assert_eq!(match_acceptance_length(&drafted, &posterior).unwrap(), 3);
    }
}
