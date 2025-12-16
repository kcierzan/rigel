//! Min/max and absolute value operations (T042)
//!
//! Provides element-wise minimum, maximum, and absolute value operations.

use crate::traits::SimdVector;

/// Element-wise minimum of two SIMD vectors
///
/// # Example
///
/// ```rust
/// use rigel_simd::{DefaultSimdVector, SimdVector};
/// use rigel_simd::ops::min;
///
/// let a = DefaultSimdVector::splat(2.0);
/// let b = DefaultSimdVector::splat(3.0);
/// let result = min(a, b);
/// assert_eq!(result.horizontal_sum(), 2.0 * DefaultSimdVector::LANES as f32);
/// ```
#[inline(always)]
pub fn min<V: SimdVector>(a: V, b: V) -> V {
    a.min(b)
}

/// Element-wise maximum of two SIMD vectors
///
/// # Example
///
/// ```rust
/// use rigel_simd::{DefaultSimdVector, SimdVector};
/// use rigel_simd::ops::max;
///
/// let a = DefaultSimdVector::splat(2.0);
/// let b = DefaultSimdVector::splat(3.0);
/// let result = max(a, b);
/// assert_eq!(result.horizontal_sum(), 3.0 * DefaultSimdVector::LANES as f32);
/// ```
#[inline(always)]
pub fn max<V: SimdVector>(a: V, b: V) -> V {
    a.max(b)
}

/// Element-wise absolute value
///
/// # Example
///
/// ```rust
/// use rigel_simd::{DefaultSimdVector, SimdVector};
/// use rigel_simd::ops::abs;
///
/// let a = DefaultSimdVector::splat(-2.0);
/// let result = abs(a);
/// assert_eq!(result.horizontal_sum(), 2.0 * DefaultSimdVector::LANES as f32);
/// ```
#[inline(always)]
pub fn abs<V: SimdVector>(a: V) -> V {
    a.abs()
}

/// Clamp a SIMD vector to a range [min_val, max_val]
///
/// Each element is clamped independently to the range.
///
/// # Example
///
/// ```rust
/// use rigel_simd::{DefaultSimdVector, SimdVector};
/// use rigel_simd::ops::clamp;
///
/// let a = DefaultSimdVector::splat(5.0);
/// let min_val = DefaultSimdVector::splat(0.0);
/// let max_val = DefaultSimdVector::splat(3.0);
/// let result = clamp(a, min_val, max_val);
/// assert_eq!(result.horizontal_sum(), 3.0 * DefaultSimdVector::LANES as f32);
/// ```
#[inline(always)]
pub fn clamp<V: SimdVector>(a: V, min_val: V, max_val: V) -> V {
    a.max(min_val).min(max_val)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::scalar::ScalarVector;

    #[test]
    fn test_min() {
        let a = ScalarVector(2.0);
        let b = ScalarVector(3.0);
        let result = min(a, b);
        assert_eq!(result.0, 2.0);
    }

    #[test]
    fn test_max() {
        let a = ScalarVector(2.0);
        let b = ScalarVector(3.0);
        let result = max(a, b);
        assert_eq!(result.0, 3.0);
    }

    #[test]
    fn test_abs() {
        let a = ScalarVector(-2.0);
        let result = abs(a);
        assert_eq!(result.0, 2.0);
    }

    #[test]
    fn test_clamp() {
        let a = ScalarVector(5.0);
        let min_val = ScalarVector(0.0);
        let max_val = ScalarVector(3.0);
        let result = clamp(a, min_val, max_val);
        assert_eq!(result.0, 3.0);

        let b = ScalarVector(-5.0);
        let result = clamp(b, min_val, max_val);
        assert_eq!(result.0, 0.0);

        let c = ScalarVector(1.5);
        let result = clamp(c, min_val, max_val);
        assert_eq!(result.0, 1.5);
    }
}
