//! Fused multiply-add operations (T041)
//!
//! Provides fused multiply-add (FMA) operations which compute a * b + c
//! in a single operation, potentially with higher precision and performance.

use crate::traits::SimdVector;

/// Fused multiply-add: computes `a * b + c` in a single operation
///
/// This operation may be implemented using a single CPU instruction (like vfmadd)
/// which can provide better performance and potentially higher precision than
/// separate multiply and add operations.
///
/// # Example
///
/// ```rust
/// use rigel_math::{DefaultSimdVector, SimdVector};
/// use rigel_math::ops::fma;
///
/// let a = DefaultSimdVector::splat(2.0);
/// let b = DefaultSimdVector::splat(3.0);
/// let c = DefaultSimdVector::splat(1.0);
/// let result = fma(a, b, c);
/// // result = 2.0 * 3.0 + 1.0 = 7.0
/// assert_eq!(result.horizontal_sum(), 7.0 * DefaultSimdVector::LANES as f32);
/// ```
#[inline(always)]
pub fn fma<V: SimdVector>(a: V, b: V, c: V) -> V {
    a.fma(b, c)
}

/// Fused multiply-subtract: computes `a * b - c`
///
/// Equivalent to `fma(a, b, neg(c))` but potentially more efficient.
///
/// # Example
///
/// ```rust
/// use rigel_math::{DefaultSimdVector, SimdVector};
/// use rigel_math::ops::fms;
///
/// let a = DefaultSimdVector::splat(2.0);
/// let b = DefaultSimdVector::splat(3.0);
/// let c = DefaultSimdVector::splat(1.0);
/// let result = fms(a, b, c);
/// // result = 2.0 * 3.0 - 1.0 = 5.0
/// assert_eq!(result.horizontal_sum(), 5.0 * DefaultSimdVector::LANES as f32);
/// ```
#[inline(always)]
pub fn fms<V: SimdVector>(a: V, b: V, c: V) -> V {
    a.fma(b, c.neg())
}

/// Fused negated multiply-add: computes `-(a * b) + c` = `c - a * b`
///
/// Useful for computing differences after multiplication.
///
/// # Example
///
/// ```rust
/// use rigel_math::{DefaultSimdVector, SimdVector};
/// use rigel_math::ops::fnma;
///
/// let a = DefaultSimdVector::splat(2.0);
/// let b = DefaultSimdVector::splat(3.0);
/// let c = DefaultSimdVector::splat(10.0);
/// let result = fnma(a, b, c);
/// // result = -(2.0 * 3.0) + 10.0 = 4.0
/// assert_eq!(result.horizontal_sum(), 4.0 * DefaultSimdVector::LANES as f32);
/// ```
#[inline(always)]
pub fn fnma<V: SimdVector>(a: V, b: V, c: V) -> V {
    a.neg().fma(b, c)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::scalar::ScalarVector;

    #[test]
    fn test_fma() {
        let a = ScalarVector(2.0);
        let b = ScalarVector(3.0);
        let c = ScalarVector(1.0);
        let result = fma(a, b, c);
        assert_eq!(result.0, 7.0); // 2*3 + 1 = 7
    }

    #[test]
    fn test_fms() {
        let a = ScalarVector(2.0);
        let b = ScalarVector(3.0);
        let c = ScalarVector(1.0);
        let result = fms(a, b, c);
        assert_eq!(result.0, 5.0); // 2*3 - 1 = 5
    }

    #[test]
    fn test_fnma() {
        let a = ScalarVector(2.0);
        let b = ScalarVector(3.0);
        let c = ScalarVector(10.0);
        let result = fnma(a, b, c);
        assert_eq!(result.0, 4.0); // -(2*3) + 10 = 4
    }
}
