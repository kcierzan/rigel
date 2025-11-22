//! Basic arithmetic operations (T040)
//!
//! Provides functional-style arithmetic operations for SIMD vectors.

use crate::traits::SimdVector;

/// Add two SIMD vectors element-wise
///
/// # Example
///
/// ```rust
/// use rigel_math::{DefaultSimdVector, SimdVector};
/// use rigel_math::ops::add;
///
/// let a = DefaultSimdVector::splat(2.0);
/// let b = DefaultSimdVector::splat(3.0);
/// let result = add(a, b);
/// assert_eq!(result.horizontal_sum(), 5.0 * DefaultSimdVector::LANES as f32);
/// ```
#[inline(always)]
pub fn add<V: SimdVector>(a: V, b: V) -> V {
    a.add(b)
}

/// Subtract two SIMD vectors element-wise
///
/// # Example
///
/// ```rust
/// use rigel_math::{DefaultSimdVector, SimdVector};
/// use rigel_math::ops::sub;
///
/// let a = DefaultSimdVector::splat(5.0);
/// let b = DefaultSimdVector::splat(3.0);
/// let result = sub(a, b);
/// assert_eq!(result.horizontal_sum(), 2.0 * DefaultSimdVector::LANES as f32);
/// ```
#[inline(always)]
pub fn sub<V: SimdVector>(a: V, b: V) -> V {
    a.sub(b)
}

/// Multiply two SIMD vectors element-wise
///
/// # Example
///
/// ```rust
/// use rigel_math::{DefaultSimdVector, SimdVector};
/// use rigel_math::ops::mul;
///
/// let a = DefaultSimdVector::splat(2.0);
/// let b = DefaultSimdVector::splat(3.0);
/// let result = mul(a, b);
/// assert_eq!(result.horizontal_sum(), 6.0 * DefaultSimdVector::LANES as f32);
/// ```
#[inline(always)]
pub fn mul<V: SimdVector>(a: V, b: V) -> V {
    a.mul(b)
}

/// Divide two SIMD vectors element-wise
///
/// # Example
///
/// ```rust
/// use rigel_math::{DefaultSimdVector, SimdVector};
/// use rigel_math::ops::div;
///
/// let a = DefaultSimdVector::splat(6.0);
/// let b = DefaultSimdVector::splat(3.0);
/// let result = div(a, b);
/// assert_eq!(result.horizontal_sum(), 2.0 * DefaultSimdVector::LANES as f32);
/// ```
#[inline(always)]
pub fn div<V: SimdVector>(a: V, b: V) -> V {
    a.div(b)
}

/// Negate a SIMD vector element-wise
///
/// # Example
///
/// ```rust
/// use rigel_math::{DefaultSimdVector, SimdVector};
/// use rigel_math::ops::neg;
///
/// let a = DefaultSimdVector::splat(2.0);
/// let result = neg(a);
/// assert_eq!(result.horizontal_sum(), -2.0 * DefaultSimdVector::LANES as f32);
/// ```
#[inline(always)]
pub fn neg<V: SimdVector>(a: V) -> V {
    a.neg()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::scalar::ScalarVector;

    #[test]
    fn test_add() {
        let a = ScalarVector(2.0);
        let b = ScalarVector(3.0);
        let result = add(a, b);
        assert_eq!(result.0, 5.0);
    }

    #[test]
    fn test_sub() {
        let a = ScalarVector(5.0);
        let b = ScalarVector(3.0);
        let result = sub(a, b);
        assert_eq!(result.0, 2.0);
    }

    #[test]
    fn test_mul() {
        let a = ScalarVector(2.0);
        let b = ScalarVector(3.0);
        let result = mul(a, b);
        assert_eq!(result.0, 6.0);
    }

    #[test]
    fn test_div() {
        let a = ScalarVector(6.0);
        let b = ScalarVector(3.0);
        let result = div(a, b);
        assert_eq!(result.0, 2.0);
    }

    #[test]
    fn test_neg() {
        let a = ScalarVector(2.0);
        let result = neg(a);
        assert_eq!(result.0, -2.0);
    }
}
