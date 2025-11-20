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
}
