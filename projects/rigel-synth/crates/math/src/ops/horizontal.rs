//! Horizontal reduction operations (T044)
//!
//! Provides operations that reduce a SIMD vector to a single scalar value
//! by combining all lanes.

use crate::traits::SimdVector;

/// Horizontal sum: add all lanes together
///
/// Returns the sum of all elements in the SIMD vector.
///
/// # Example
///
/// ```rust
/// use rigel_math::{DefaultSimdVector, SimdVector};
/// use rigel_math::ops::horizontal_sum;
///
/// let a = DefaultSimdVector::splat(2.0);
/// let result = horizontal_sum(a);
/// assert_eq!(result, 2.0 * DefaultSimdVector::LANES as f32);
/// ```
#[inline(always)]
pub fn horizontal_sum<V: SimdVector>(a: V) -> V::Scalar {
    a.horizontal_sum()
}

/// Horizontal maximum: find the maximum element
///
/// Returns the maximum value among all elements in the SIMD vector.
///
/// # Example
///
/// ```rust
/// use rigel_math::{DefaultSimdVector, SimdVector};
/// use rigel_math::ops::horizontal_max;
///
/// let a = DefaultSimdVector::splat(2.0);
/// let result = horizontal_max(a);
/// assert_eq!(result, 2.0);
/// ```
#[inline(always)]
pub fn horizontal_max<V: SimdVector>(a: V) -> V::Scalar {
    a.horizontal_max()
}

/// Horizontal minimum: find the minimum element
///
/// Returns the minimum value among all elements in the SIMD vector.
///
/// # Example
///
/// ```rust
/// use rigel_math::{DefaultSimdVector, SimdVector};
/// use rigel_math::ops::horizontal_min;
///
/// let a = DefaultSimdVector::splat(2.0);
/// let result = horizontal_min(a);
/// assert_eq!(result, 2.0);
/// ```
#[inline(always)]
pub fn horizontal_min<V: SimdVector>(a: V) -> V::Scalar {
    a.horizontal_min()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::scalar::ScalarVector;

    #[test]
    fn test_horizontal_sum() {
        let a = ScalarVector(2.0);
        let result = horizontal_sum(a);
        assert_eq!(result, 2.0);
    }

    #[test]
    fn test_horizontal_max() {
        let a = ScalarVector(2.0);
        let result = horizontal_max(a);
        assert_eq!(result, 2.0);
    }

    #[test]
    fn test_horizontal_min() {
        let a = ScalarVector(2.0);
        let result = horizontal_min(a);
        assert_eq!(result, 2.0);
    }
}
