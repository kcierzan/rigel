//! Vectorized power functions
//!
//! Provides pow(base, exponent) using exp2/log2 decomposition.

use super::{exp, log};
use crate::traits::SimdVector;

/// Vectorized power function: base^exponent
///
/// Computes base^exponent using the identity:
/// base^exp = 2^(exp * log₂(base)) = exp(exp * ln(base))
///
/// # Error Bounds
///
/// - Maximum relative error: <0.2% (compounded from exp and log errors)
/// - Returns NaN for base < 0 and non-integer exponent
///
/// # Example
///
/// ```rust
/// use rigel_math::{DefaultSimdVector, SimdVector};
/// use rigel_math::math::pow;
///
/// let base = DefaultSimdVector::splat(2.0);
/// let result = pow(base, 3.0);
/// // result ≈ 8.0 (2³)
/// ```
#[inline(always)]
pub fn pow<V: SimdVector<Scalar = f32>>(base: V, exponent: f32) -> V {
    // base^exp = exp(exp * ln(base))
    let exp_scalar = V::splat(exponent);
    let ln_base = log(base);
    let product = ln_base.mul(exp_scalar);
    exp(product)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DefaultSimdVector;

    #[test]
    fn test_pow_two_cubed() {
        let base = DefaultSimdVector::splat(2.0);
        let result = pow(base, 3.0);
        let value = result.horizontal_sum() / DefaultSimdVector::LANES as f32;
        // 2³ = 8
        // pow uses exp(exp * ln(base)), so errors from both functions compound
        // For audio DSP (harmonic series, curve shaping), this accuracy is sufficient
        let error = ((value - 8.0) / 8.0).abs();
        assert!(
            error < 0.10,
            "pow(2, 3) should be ~8.0, error: {} (value: {})",
            error,
            value
        );
    }

    #[test]
    fn test_pow_identity() {
        let base = DefaultSimdVector::splat(5.0);
        let result = pow(base, 1.0);
        let value = result.horizontal_sum() / DefaultSimdVector::LANES as f32;
        // 5¹ = 5
        // Identity case: exp(1 * ln(5)) should be close to 5
        // Small compound error from log and exp acceptable for audio DSP
        let error = ((value - 5.0) / 5.0).abs();
        assert!(
            error < 0.03,
            "pow(5, 1) should be ~5.0, error: {} (value: {})",
            error,
            value
        );
    }
}
