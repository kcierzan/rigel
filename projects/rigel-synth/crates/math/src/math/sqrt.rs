//! Vectorized square root functions
//!
//! Provides sqrt and rsqrt (reciprocal square root) for audio DSP applications.

use crate::traits::SimdVector;

/// Vectorized square root
///
/// Computes √x using hardware SIMD square root instructions.
///
/// # Error Bounds
///
/// - IEEE 754 compliant (uses hardware sqrt instructions)
/// - Returns NaN for x < 0
/// - Returns +∞ for x = +∞
///
/// # Performance
///
/// This function uses hardware SIMD instructions for optimal performance:
/// - AVX2: `_mm256_sqrt_ps` (~10-15 cycles)
/// - AVX-512: `_mm512_sqrt_ps` (~10-15 cycles)
/// - NEON: `vsqrtq_f32` (~10-15 cycles)
/// - Scalar: `libm::sqrtf` (optimal scalar performance)
///
/// Approximately 3-6x faster than software Newton-Raphson approximation.
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
    x.sqrt()
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
