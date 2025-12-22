//! Comparison operations (T043)
//!
//! Provides element-wise comparison operations that return boolean masks.

use crate::traits::{SimdMask, SimdVector};

/// Element-wise equality comparison
///
/// Returns a mask where each lane is true if the corresponding elements are equal.
///
/// # Example
///
/// ```rust
/// use rigel_simd::{DefaultSimdVector, SimdVector, SimdMask};
/// use rigel_simd::ops::eq;
///
/// let a = DefaultSimdVector::splat(2.0);
/// let b = DefaultSimdVector::splat(2.0);
/// let result = eq(a, b);
/// assert!(result.all());
/// ```
#[inline(always)]
pub fn eq<V: SimdVector>(a: V, b: V) -> V::Mask {
    a.eq(b)
}

/// Element-wise inequality comparison
///
/// Returns a mask where each lane is true if the corresponding elements are not equal.
///
/// # Example
///
/// ```rust
/// use rigel_simd::{DefaultSimdVector, SimdVector, SimdMask};
/// use rigel_simd::ops::ne;
///
/// let a = DefaultSimdVector::splat(2.0);
/// let b = DefaultSimdVector::splat(3.0);
/// let result = ne(a, b);
/// assert!(result.all());
/// ```
#[inline(always)]
pub fn ne<V: SimdVector>(a: V, b: V) -> V::Mask {
    a.eq(b).not()
}

/// Element-wise less-than comparison
///
/// Returns a mask where each lane is true if the corresponding element in `a` is less than `b`.
///
/// # Example
///
/// ```rust
/// use rigel_simd::{DefaultSimdVector, SimdVector, SimdMask};
/// use rigel_simd::ops::lt;
///
/// let a = DefaultSimdVector::splat(2.0);
/// let b = DefaultSimdVector::splat(3.0);
/// let result = lt(a, b);
/// assert!(result.all());
/// ```
#[inline(always)]
pub fn lt<V: SimdVector>(a: V, b: V) -> V::Mask {
    a.lt(b)
}

/// Element-wise less-than-or-equal comparison
///
/// Returns a mask where each lane is true if the corresponding element in `a` is less than or equal to `b`.
///
/// # Example
///
/// ```rust
/// use rigel_simd::{DefaultSimdVector, SimdVector, SimdMask};
/// use rigel_simd::ops::le;
///
/// let a = DefaultSimdVector::splat(2.0);
/// let b = DefaultSimdVector::splat(2.0);
/// let result = le(a, b);
/// assert!(result.all());
/// ```
#[inline(always)]
pub fn le<V: SimdVector>(a: V, b: V) -> V::Mask {
    a.gt(b).not()
}

/// Element-wise greater-than comparison
///
/// Returns a mask where each lane is true if the corresponding element in `a` is greater than `b`.
///
/// # Example
///
/// ```rust
/// use rigel_simd::{DefaultSimdVector, SimdVector, SimdMask};
/// use rigel_simd::ops::gt;
///
/// let a = DefaultSimdVector::splat(3.0);
/// let b = DefaultSimdVector::splat(2.0);
/// let result = gt(a, b);
/// assert!(result.all());
/// ```
#[inline(always)]
pub fn gt<V: SimdVector>(a: V, b: V) -> V::Mask {
    a.gt(b)
}

/// Element-wise greater-than-or-equal comparison
///
/// Returns a mask where each lane is true if the corresponding element in `a` is greater than or equal to `b`.
///
/// # Example
///
/// ```rust
/// use rigel_simd::{DefaultSimdVector, SimdVector, SimdMask};
/// use rigel_simd::ops::ge;
///
/// let a = DefaultSimdVector::splat(3.0);
/// let b = DefaultSimdVector::splat(3.0);
/// let result = ge(a, b);
/// assert!(result.all());
/// ```
#[inline(always)]
pub fn ge<V: SimdVector>(a: V, b: V) -> V::Mask {
    a.lt(b).not()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::scalar::{ScalarMask, ScalarVector};

    #[test]
    fn test_eq() {
        let a = ScalarVector(2.0);
        let b = ScalarVector(2.0);
        let result: ScalarMask = eq(a, b);
        assert!(result.all());

        let c = ScalarVector(3.0);
        let result: ScalarMask = eq(a, c);
        assert!(!result.all());
    }

    #[test]
    fn test_ne() {
        let a = ScalarVector(2.0);
        let b = ScalarVector(3.0);
        let result: ScalarMask = ne(a, b);
        assert!(result.all());

        let result: ScalarMask = ne(a, a);
        assert!(!result.all());
    }

    #[test]
    fn test_lt() {
        let a = ScalarVector(2.0);
        let b = ScalarVector(3.0);
        let result: ScalarMask = lt(a, b);
        assert!(result.all());

        let result: ScalarMask = lt(b, a);
        assert!(!result.all());
    }

    #[test]
    fn test_le() {
        let a = ScalarVector(2.0);
        let b = ScalarVector(2.0);
        let result: ScalarMask = le(a, b);
        assert!(result.all());

        let c = ScalarVector(3.0);
        let result: ScalarMask = le(a, c);
        assert!(result.all());

        let result: ScalarMask = le(c, a);
        assert!(!result.all());
    }

    #[test]
    fn test_gt() {
        let a = ScalarVector(3.0);
        let b = ScalarVector(2.0);
        let result: ScalarMask = gt(a, b);
        assert!(result.all());

        let result: ScalarMask = gt(b, a);
        assert!(!result.all());
    }

    #[test]
    fn test_ge() {
        let a = ScalarVector(3.0);
        let b = ScalarVector(3.0);
        let result: ScalarMask = ge(a, b);
        assert!(result.all());

        let c = ScalarVector(2.0);
        let result: ScalarMask = ge(a, c);
        assert!(result.all());

        let result: ScalarMask = ge(c, a);
        assert!(!result.all());
    }
}
