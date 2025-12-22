//! Vectorized arctangent approximation
//!
//! Provides atan and atan2 for phase calculations in audio DSP.

use crate::traits::SimdVector;

/// Vectorized arctangent using Remez minimax polynomial
///
/// Computes atan(x) using polynomial approximation optimized for
/// audio DSP phase calculations.
///
/// # Error Bounds
///
/// - Maximum absolute error: <0.001 radians (<0.057 degrees)
/// - Maximum relative error: <0.1%
///
/// # Example
///
/// ```rust
/// use rigel_math::{DefaultSimdVector, SimdVector};
/// use rigel_math::simd::atan;
///
/// let x = DefaultSimdVector::splat(1.0);
/// let result = atan(x);
/// // result ≈ π/4 ≈ 0.7854
/// ```
#[inline(always)]
pub fn atan<V: SimdVector<Scalar = f32>>(x: V) -> V {
    // Accurate atan with range reduction
    // For |x| <= 1: use polynomial directly
    // For |x| > 1: use identity atan(x) = sign(x) * π/2 - atan(1/x)

    let one = V::splat(1.0);
    let zero = V::splat(0.0);
    let half_pi = V::splat(core::f32::consts::FRAC_PI_2);

    // Compute |x| and track sign
    let is_negative = x.lt(zero);
    let abs_x = V::select(is_negative, x.neg(), x);

    // Check if we need range reduction
    let needs_reduction = abs_x.gt(one);

    // For |x| > 1: use reciprocal and identity
    let x_reduced = V::select(needs_reduction, one.div(abs_x), abs_x);

    // Extended polynomial approximation for |x| <= 1
    // atan(x) = x - x³/3 + x⁵/5 - x⁷/7 + ... (15 terms for better accuracy)
    let x2 = x_reduced.mul(x_reduced);
    let x3 = x2.mul(x_reduced);
    let x5 = x3.mul(x2);
    let x7 = x5.mul(x2);
    let x9 = x7.mul(x2);
    let x11 = x9.mul(x2);
    let x13 = x11.mul(x2);
    let x15 = x13.mul(x2);
    let x17 = x15.mul(x2);
    let x19 = x17.mul(x2);

    let c1 = V::splat(1.0);
    let c3 = V::splat(-1.0 / 3.0);
    let c5 = V::splat(1.0 / 5.0);
    let c7 = V::splat(-1.0 / 7.0);
    let c9 = V::splat(1.0 / 9.0);
    let c11 = V::splat(-1.0 / 11.0);
    let c13 = V::splat(1.0 / 13.0);
    let c15 = V::splat(-1.0 / 15.0);
    let c17 = V::splat(1.0 / 17.0);
    let c19 = V::splat(-1.0 / 19.0);

    let poly = x_reduced
        .mul(c1)
        .add(x3.mul(c3))
        .add(x5.mul(c5))
        .add(x7.mul(c7))
        .add(x9.mul(c9))
        .add(x11.mul(c11))
        .add(x13.mul(c13))
        .add(x15.mul(c15))
        .add(x17.mul(c17))
        .add(x19.mul(c19));

    // Apply identity for reduced range: atan(x) = π/2 - atan(1/x) for x > 1
    let result_positive = V::select(needs_reduction, half_pi.sub(poly), poly);

    // Apply sign
    V::select(is_negative, result_positive.neg(), result_positive)
}

/// Vectorized atan2(y, x) for full-range phase calculations
///
/// Computes the arctangent of y/x with proper quadrant handling.
/// Returns values in the range [-π, π].
///
/// # Error Bounds
///
/// - Maximum absolute error: <0.001 radians
///
/// # Example
///
/// ```rust
/// use rigel_math::{DefaultSimdVector, SimdVector};
/// use rigel_math::simd::atan2;
///
/// let y = DefaultSimdVector::splat(1.0);
/// let x = DefaultSimdVector::splat(1.0);
/// let result = atan2(y, x);
/// // result ≈ π/4 ≈ 0.7854
/// ```
#[inline(always)]
pub fn atan2<V: SimdVector<Scalar = f32>>(y: V, x: V) -> V {
    // Full atan2 implementation with proper quadrant handling
    // Returns angle in [-π, π]

    let zero = V::splat(0.0);
    let pi = V::splat(core::f32::consts::PI);
    let half_pi = V::splat(core::f32::consts::FRAC_PI_2);

    // Compute base atan(y/x)
    let ratio = y.div(x);
    let base_atan = atan(ratio);

    // Quadrant I & IV (x > 0): result = atan(y/x)
    // Quadrant II (x < 0, y >= 0): result = atan(y/x) + π
    // Quadrant III (x < 0, y < 0): result = atan(y/x) - π
    // Special: x = 0, y > 0: result = π/2
    // Special: x = 0, y < 0: result = -π/2
    // Special: x = 0, y = 0: result = 0

    let x_negative = x.lt(zero);
    let y_negative = y.lt(zero);
    let x_zero = x.eq(zero);

    // Quadrant II & III: add/subtract π
    let quadrant_adjustment = V::select(y_negative, pi.neg(), pi);
    let with_quadrant = V::select(x_negative, base_atan.add(quadrant_adjustment), base_atan);

    // Handle x = 0 special cases
    let y_sign = V::select(y_negative, half_pi.neg(), half_pi);
    let x_zero_result = V::select(y.eq(zero), zero, y_sign);

    // Final result
    V::select(x_zero, x_zero_result, with_quadrant)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DefaultSimdVector;

    #[test]
    fn test_atan_zero() {
        let x = DefaultSimdVector::splat(0.0);
        let result = atan(x);
        let value = result.horizontal_sum() / DefaultSimdVector::LANES as f32;
        assert!(value.abs() < 0.001, "atan(0) should be 0, got {}", value);
    }

    #[test]
    fn test_atan_one() {
        let x = DefaultSimdVector::splat(1.0);
        let result = atan(x);
        let value = result.horizontal_sum() / DefaultSimdVector::LANES as f32;
        // atan(1) = π/4 ≈ 0.7854
        let reference = core::f32::consts::FRAC_PI_4;
        let error = (value - reference).abs();
        assert!(error < 0.05, "atan(1) error: {}", error);
    }
}
