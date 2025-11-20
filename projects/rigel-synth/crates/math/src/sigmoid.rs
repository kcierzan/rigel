//! Sigmoid curves with C1/C2 continuity
//!
//! Provides smooth sigmoid functions for parameter mapping and waveshaping.

use crate::traits::SimdVector;

/// Logistic sigmoid: 1 / (1 + e^(-x))
///
/// Classic S-shaped curve with C∞ continuity.
///
/// # Example
///
/// ```rust
/// use rigel_math::{DefaultSimdVector, SimdVector};
/// use rigel_math::sigmoid::logistic;
///
/// let x = DefaultSimdVector::splat(0.0);
/// let result = logistic(x);
/// // result ≈ 0.5 (sigmoid(0) = 0.5)
/// ```
#[inline(always)]
pub fn logistic<V: SimdVector<Scalar = f32>>(x: V) -> V {
    // sigmoid(x) = 1 / (1 + exp(-x))
    use crate::math::exp;

    let one = V::splat(1.0);
    let neg_x = x.neg();
    let exp_neg_x = exp(neg_x);
    let denominator = one.add(exp_neg_x);
    one.div(denominator)
}

/// Smoothstep function: 3x² - 2x³ (C1 continuity)
///
/// Maps [0, 1] to [0, 1] with smooth derivatives at endpoints.
///
/// # Example
///
/// ```rust
/// use rigel_math::{DefaultSimdVector, SimdVector};
/// use rigel_math::sigmoid::smoothstep;
///
/// let t = DefaultSimdVector::splat(0.5);
/// let result = smoothstep(t);
/// // result ≈ 0.5 (smoothstep(0.5) = 0.5)
/// ```
#[inline(always)]
pub fn smoothstep<V: SimdVector<Scalar = f32>>(t: V) -> V {
    // smoothstep(t) = 3t² - 2t³
    // Assumes t ∈ [0, 1]
    let t2 = t.mul(t);
    let t3 = t2.mul(t);

    let c2 = V::splat(3.0);
    let c3 = V::splat(-2.0);

    t2.mul(c2).add(t3.mul(c3))
}

/// Smootherstep function: 6x⁵ - 15x⁴ + 10x³ (C2 continuity)
///
/// Maps [0, 1] to [0, 1] with smooth first AND second derivatives at endpoints.
///
/// # Example
///
/// ```rust
/// use rigel_math::{DefaultSimdVector, SimdVector};
/// use rigel_math::sigmoid::smootherstep;
///
/// let t = DefaultSimdVector::splat(0.5);
/// let result = smootherstep(t);
/// ```
#[inline(always)]
pub fn smootherstep<V: SimdVector<Scalar = f32>>(t: V) -> V {
    // smootherstep(t) = 6t⁵ - 15t⁴ + 10t³
    let t2 = t.mul(t);
    let t3 = t2.mul(t);
    let t4 = t3.mul(t);
    let t5 = t4.mul(t);

    let c3 = V::splat(10.0);
    let c4 = V::splat(-15.0);
    let c5 = V::splat(6.0);

    t3.mul(c3).add(t4.mul(c4)).add(t5.mul(c5))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DefaultSimdVector;

    #[test]
    fn test_logistic_zero() {
        let x = DefaultSimdVector::splat(0.0);
        let result = logistic(x);
        let value = result.horizontal_sum() / DefaultSimdVector::LANES as f32;
        // sigmoid(0) = 0.5
        let error = (value - 0.5).abs();
        assert!(error < 0.01, "logistic(0) should be 0.5, got {}", value);
    }

    #[test]
    fn test_smoothstep_boundaries() {
        // smoothstep(0) = 0
        let t0 = DefaultSimdVector::splat(0.0);
        let result0 = smoothstep(t0);
        let value0 = result0.horizontal_sum() / DefaultSimdVector::LANES as f32;
        assert!(value0.abs() < 1e-5, "smoothstep(0) should be 0");

        // smoothstep(1) = 1
        let t1 = DefaultSimdVector::splat(1.0);
        let result1 = smoothstep(t1);
        let value1 = result1.horizontal_sum() / DefaultSimdVector::LANES as f32;
        assert!((value1 - 1.0).abs() < 1e-4, "smoothstep(1) should be 1");
    }
}
