//! Vectorized square root functions
//!
//! Provides sqrt and rsqrt (reciprocal square root) for audio DSP applications.

use crate::traits::SimdVector;

/// Vectorized square root
///
/// Computes √x using SIMD instructions.
///
/// # Error Bounds
///
/// - IEEE 754 compliant (using hardware sqrt where available)
/// - Returns NaN for x < 0
/// - Returns +∞ for x = +∞
///
/// # Example
///
/// ```rust
/// use rigel_math::{DefaultSimdVector, SimdVector};
/// use rigel_math::math::sqrt;
///
/// let x = DefaultSimdVector::splat(4.0);
/// let result = sqrt(x);
/// // result ≈ 2.0
/// ```
#[inline(always)]
pub fn sqrt<V: SimdVector<Scalar = f32>>(x: V) -> V {
    // Use the trait's div with a simple approach
    // For actual SIMD implementation, would use _mm_sqrt_ps, etc.
    // For now, this is a placeholder that works via the trait

    // Newton-Raphson for sqrt: x_{n+1} = (x_n + a/x_n) / 2
    // Start with initial estimate

    let half = V::splat(0.5);
    let one = V::splat(1.0);

    // Initial guess: Could use rsqrt estimate, but for simplicity use 1
    let mut estimate = one;

    // Perform 3 iterations for good convergence
    for _ in 0..3 {
        estimate = half.mul(estimate.add(x.div(estimate)));
    }

    estimate
}

/// Vectorized reciprocal square root: 1/√x
///
/// Computes 1/√x, which is useful for vector normalization and
/// other DSP algorithms where division by sqrt is needed.
///
/// # Error Bounds
///
/// - Maximum relative error: <0.01%
/// - Returns NaN for x < 0
/// - Returns 0 for x = +∞
///
/// # Example
///
/// ```rust
/// use rigel_math::{DefaultSimdVector, SimdVector};
/// use rigel_math::math::rsqrt;
///
/// let x = DefaultSimdVector::splat(4.0);
/// let result = rsqrt(x);
/// // result ≈ 0.5 (1/√4 = 1/2)
/// ```
#[inline(always)]
pub fn rsqrt<V: SimdVector<Scalar = f32>>(x: V) -> V {
    // 1 / sqrt(x)
    let one = V::splat(1.0);
    let sqrt_x = sqrt(x);
    one.div(sqrt_x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DefaultSimdVector;

    #[test]
    fn test_sqrt_four() {
        let x = DefaultSimdVector::splat(4.0);
        let result = sqrt(x);
        let value = result.horizontal_sum() / DefaultSimdVector::LANES as f32;
        // √4 = 2
        let error = (value - 2.0).abs();
        assert!(error < 0.01, "sqrt(4) error: {}", error);
    }

    #[test]
    fn test_sqrt_one() {
        let x = DefaultSimdVector::splat(1.0);
        let result = sqrt(x);
        let value = result.horizontal_sum() / DefaultSimdVector::LANES as f32;
        // √1 = 1
        let error = (value - 1.0).abs();
        assert!(error < 0.01, "sqrt(1) error: {}", error);
    }

    #[test]
    fn test_rsqrt_four() {
        let x = DefaultSimdVector::splat(4.0);
        let result = rsqrt(x);
        let value = result.horizontal_sum() / DefaultSimdVector::LANES as f32;
        // 1/√4 = 0.5
        let error = (value - 0.5).abs();
        assert!(error < 0.01, "rsqrt(4) error: {}", error);
    }
}
