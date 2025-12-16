//! Polynomial interpolation kernels
//!
//! Provides linear, cubic Hermite, and quintic interpolation for audio DSP.

use crate::traits::SimdVector;

/// Linear interpolation
///
/// Interpolates between a and b using parameter t ∈ [0, 1].
/// lerp(a, b, t) = a + t * (b - a) = (1 - t) * a + t * b
///
/// # Example
///
/// ```rust
/// use rigel_math::{DefaultSimdVector, SimdVector};
/// use rigel_math::interpolate::lerp;
///
/// let a = DefaultSimdVector::splat(0.0);
/// let b = DefaultSimdVector::splat(10.0);
/// let t = DefaultSimdVector::splat(0.5);
/// let result = lerp(a, b, t);
/// // result ≈ 5.0
/// ```
#[inline(always)]
pub fn lerp<V: SimdVector<Scalar = f32>>(a: V, b: V, t: V) -> V {
    // lerp(a, b, t) = a + t * (b - a)
    let diff = b.sub(a);
    a.add(t.mul(diff))
}

/// Cubic Hermite interpolation
///
/// Smooth interpolation with continuous first derivatives.
/// Uses the formula: (2t³ - 3t² + 1) * a + (t³ - 2t² + t) * ta +
///                   (-2t³ + 3t²) * b + (t³ - t²) * tb
///
/// # Example
///
/// ```rust
/// use rigel_math::{DefaultSimdVector, SimdVector};
/// use rigel_math::interpolate::cubic_hermite;
///
/// let a = DefaultSimdVector::splat(0.0);
/// let b = DefaultSimdVector::splat(10.0);
/// let tangent_a = DefaultSimdVector::splat(0.0);
/// let tangent_b = DefaultSimdVector::splat(0.0);
/// let t = DefaultSimdVector::splat(0.5);
/// let result = cubic_hermite(a, b, tangent_a, tangent_b, t);
/// ```
#[inline(always)]
pub fn cubic_hermite<V: SimdVector<Scalar = f32>>(
    a: V,
    b: V,
    tangent_a: V,
    tangent_b: V,
    t: V,
) -> V {
    let t2 = t.mul(t);
    let t3 = t2.mul(t);

    let two = V::splat(2.0);
    let three = V::splat(3.0);

    // Hermite basis functions
    let h00 = two.mul(t3).sub(three.mul(t2)).add(V::splat(1.0));
    let h10 = t3.sub(two.mul(t2)).add(t);
    let h01 = three.mul(t2).sub(two.mul(t3));
    let h11 = t3.sub(t2);

    h00.mul(a)
        .add(h10.mul(tangent_a))
        .add(h01.mul(b))
        .add(h11.mul(tangent_b))
}

/// Quintic interpolation (5th-order polynomial)
///
/// Even smoother interpolation with continuous second derivatives.
///
/// # Example
///
/// ```rust
/// use rigel_math::{DefaultSimdVector, SimdVector};
/// use rigel_math::interpolate::quintic;
///
/// let a = DefaultSimdVector::splat(0.0);
/// let b = DefaultSimdVector::splat(10.0);
/// let t = DefaultSimdVector::splat(0.5);
/// let result = quintic(a, b, t);
/// ```
#[inline(always)]
pub fn quintic<V: SimdVector<Scalar = f32>>(a: V, b: V, t: V) -> V {
    // Quintic ease: t³(10 + t(-15 + 6t))
    let t2 = t.mul(t);
    let t3 = t2.mul(t);
    let t4 = t3.mul(t);
    let t5 = t4.mul(t);

    // 6t⁵ - 15t⁴ + 10t³
    let c3 = V::splat(10.0);
    let c4 = V::splat(-15.0);
    let c5 = V::splat(6.0);

    let blend = t3.mul(c3).add(t4.mul(c4)).add(t5.mul(c5));

    // a + blend * (b - a)
    let diff = b.sub(a);
    a.add(blend.mul(diff))
}

/// Scalar cubic hermite interpolation
/// # Example
/// ```rust
/// use rigel_math::interpolate::hermite_scalar;
///
/// let a = 0.01;
/// let b = 0.1;
/// let t = 0.0212;
/// let tangent_a = 0.1;
/// let tangent_b = 0.082787;
/// let result = hermite_scalar(a, b, tangent_a, tangent_b, t);
/// ```
#[inline(always)]
pub fn hermite_scalar(a: f32, b: f32, tangent_a: f32, tangent_b: f32, t: f32) -> f32 {
    let t2 = t * t;
    let t3 = t2 * t;

    // Hermite basis functions
    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = 3.0 * t2 - 2.0 * t3;
    let h11 = t3 - t2;

    h00 * a + h10 * tangent_a + h01 * b + h11 * tangent_b
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DefaultSimdVector;

    #[test]
    fn test_lerp_midpoint() {
        let a = DefaultSimdVector::splat(0.0);
        let b = DefaultSimdVector::splat(10.0);
        let t = DefaultSimdVector::splat(0.5);
        let result = lerp(a, b, t);
        let value = result.horizontal_sum() / DefaultSimdVector::LANES as f32;
        assert!((value - 5.0).abs() < 1e-4, "lerp midpoint should be 5.0");
    }

    #[test]
    fn test_lerp_boundaries() {
        let a = DefaultSimdVector::splat(0.0);
        let b = DefaultSimdVector::splat(10.0);

        // t = 0 -> a
        let t0 = DefaultSimdVector::splat(0.0);
        let result0 = lerp(a, b, t0);
        let value0 = result0.horizontal_sum() / DefaultSimdVector::LANES as f32;
        assert!((value0 - 0.0).abs() < 1e-4, "lerp(0) should be a");

        // t = 1 -> b
        let t1 = DefaultSimdVector::splat(1.0);
        let result1 = lerp(a, b, t1);
        let value1 = result1.horizontal_sum() / DefaultSimdVector::LANES as f32;
        assert!((value1 - 10.0).abs() < 1e-4, "lerp(1) should be b");
    }

    #[test]
    fn test_hermite_scalar_boundaries() {
        // At t=0, should return a
        let result = hermite_scalar(5.0, 10.0, 1.0, 2.0, 0.0);
        assert!(
            (result - 5.0).abs() < 1e-6,
            "hermite_scalar(t=0) should return a, got {}",
            result
        );

        // At t=1, should return b
        let result = hermite_scalar(5.0, 10.0, 1.0, 2.0, 1.0);
        assert!(
            (result - 10.0).abs() < 1e-6,
            "hermite_scalar(t=1) should return b, got {}",
            result
        );
    }

    #[test]
    fn test_hermite_scalar_midpoint_zero_tangents() {
        // With zero tangents, hermite at t=0.5 should be exactly (a+b)/2
        let a = 0.0;
        let b = 10.0;
        let result = hermite_scalar(a, b, 0.0, 0.0, 0.5);
        let expected = 5.0;
        assert!(
            (result - expected).abs() < 1e-6,
            "hermite_scalar midpoint with zero tangents should be 5.0, got {}",
            result
        );
    }

    #[test]
    fn test_hermite_scalar_tangent_influence() {
        // Positive tangent at a should pull the curve up at t=0.25
        let result_pos = hermite_scalar(0.0, 0.0, 10.0, 0.0, 0.25);
        assert!(
            result_pos > 0.0,
            "positive tangent_a should pull curve up, got {}",
            result_pos
        );

        // Negative tangent at a should pull the curve down at t=0.25
        let result_neg = hermite_scalar(0.0, 0.0, -10.0, 0.0, 0.25);
        assert!(
            result_neg < 0.0,
            "negative tangent_a should pull curve down, got {}",
            result_neg
        );
    }

    #[test]
    fn test_hermite_scalar_symmetry() {
        // With symmetric endpoints and tangents, curve should be symmetric
        let t1 = 0.25;
        let t2 = 0.75;
        let result1 = hermite_scalar(0.0, 10.0, 5.0, 5.0, t1);
        let result2 = hermite_scalar(0.0, 10.0, 5.0, 5.0, t2);

        // result1 and result2 should be symmetric around the midpoint (5.0)
        let dist1 = (result1 - 5.0).abs();
        let dist2 = (result2 - 5.0).abs();
        assert!(
            (dist1 - dist2).abs() < 1e-4,
            "curve should be symmetric: dist1={}, dist2={}",
            dist1,
            dist2
        );
    }

    #[test]
    fn test_hermite_scalar_matches_simd() {
        // hermite_scalar should produce the same results as cubic_hermite SIMD version
        let a = 2.5;
        let b = 7.5;
        let ta = 1.0;
        let tb = -0.5;
        let t = 0.3;

        let scalar_result = hermite_scalar(a, b, ta, tb, t);

        let simd_result = cubic_hermite(
            DefaultSimdVector::splat(a),
            DefaultSimdVector::splat(b),
            DefaultSimdVector::splat(ta),
            DefaultSimdVector::splat(tb),
            DefaultSimdVector::splat(t),
        );
        let simd_value = simd_result.horizontal_sum() / DefaultSimdVector::LANES as f32;

        assert!(
            (scalar_result - simd_value).abs() < 1e-5,
            "scalar and SIMD should match: scalar={}, simd={}",
            scalar_result,
            simd_value
        );
    }

    #[test]
    fn test_hermite_scalar_continuity() {
        // Test that small changes in t produce small changes in output (continuity)
        let a = 1.0;
        let b = 5.0;
        let ta = 2.0;
        let tb = 1.0;

        let epsilon = 1e-4;
        for i in 0..10 {
            let t = i as f32 * 0.1;
            let result1 = hermite_scalar(a, b, ta, tb, t);
            let result2 = hermite_scalar(a, b, ta, tb, t + epsilon);

            let diff = (result2 - result1).abs();
            assert!(
                diff < 0.1,
                "discontinuity detected at t={}: diff={}",
                t,
                diff
            );
        }
    }
}
