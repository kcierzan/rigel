//! Scalar backend implementation
//!
//! This backend provides a scalar (non-SIMD) fallback that always works on any platform.
//! It serves as the reference implementation and is useful for testing backend consistency.

use crate::traits::{SimdMask, SimdVector};

/// Scalar vector wrapper (single-lane SIMD)
///
/// This wraps a single scalar value to implement the SimdVector trait,
/// providing a fallback when SIMD is not available or desired.
#[derive(Debug, Copy, Clone, PartialEq)]
#[repr(transparent)]
pub struct ScalarVector<T>(pub T);

/// Scalar mask wrapper (single boolean)
#[derive(Debug, Copy, Clone, PartialEq)]
#[repr(transparent)]
pub struct ScalarMask(pub bool);

// Implement SimdVector for ScalarVector<f32>
impl SimdVector for ScalarVector<f32> {
    type Scalar = f32;
    type Mask = ScalarMask;

    const LANES: usize = 1;

    #[inline(always)]
    fn splat(value: Self::Scalar) -> Self {
        ScalarVector(value)
    }

    #[inline(always)]
    fn from_slice(slice: &[Self::Scalar]) -> Self {
        assert!(slice.len() >= Self::LANES, "Slice too short for scalar load");
        ScalarVector(slice[0])
    }

    #[inline(always)]
    fn to_slice(self, slice: &mut [Self::Scalar]) {
        assert!(
            slice.len() >= Self::LANES,
            "Slice too short for scalar store"
        );
        slice[0] = self.0;
    }

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        ScalarVector(self.0 + rhs.0)
    }

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        ScalarVector(self.0 - rhs.0)
    }

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        ScalarVector(self.0 * rhs.0)
    }

    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        ScalarVector(self.0 / rhs.0)
    }

    #[inline(always)]
    fn neg(self) -> Self {
        ScalarVector(-self.0)
    }

    #[inline(always)]
    fn abs(self) -> Self {
        ScalarVector(libm::fabsf(self.0))
    }

    #[inline(always)]
    fn fma(self, b: Self, c: Self) -> Self {
        // Note: libm::fmaf would use hardware FMA if available, but for scalar fallback we use multiply-add
        ScalarVector(libm::fmaf(self.0, b.0, c.0))
    }

    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        ScalarVector(libm::fminf(self.0, rhs.0))
    }

    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        ScalarVector(libm::fmaxf(self.0, rhs.0))
    }

    #[inline(always)]
    fn lt(self, rhs: Self) -> Self::Mask {
        ScalarMask(self.0 < rhs.0)
    }

    #[inline(always)]
    fn gt(self, rhs: Self) -> Self::Mask {
        ScalarMask(self.0 > rhs.0)
    }

    #[inline(always)]
    fn eq(self, rhs: Self) -> Self::Mask {
        ScalarMask(self.0 == rhs.0)
    }

    #[inline(always)]
    fn select(mask: Self::Mask, true_val: Self, false_val: Self) -> Self {
        if mask.0 {
            true_val
        } else {
            false_val
        }
    }

    #[inline(always)]
    fn horizontal_sum(self) -> Self::Scalar {
        self.0
    }

    #[inline(always)]
    fn horizontal_max(self) -> Self::Scalar {
        self.0
    }

    #[inline(always)]
    fn horizontal_min(self) -> Self::Scalar {
        self.0
    }
}

// Implement SimdVector for ScalarVector<f64>
impl SimdVector for ScalarVector<f64> {
    type Scalar = f64;
    type Mask = ScalarMask;

    const LANES: usize = 1;

    #[inline(always)]
    fn splat(value: Self::Scalar) -> Self {
        ScalarVector(value)
    }

    #[inline(always)]
    fn from_slice(slice: &[Self::Scalar]) -> Self {
        assert!(slice.len() >= Self::LANES, "Slice too short for scalar load");
        ScalarVector(slice[0])
    }

    #[inline(always)]
    fn to_slice(self, slice: &mut [Self::Scalar]) {
        assert!(
            slice.len() >= Self::LANES,
            "Slice too short for scalar store"
        );
        slice[0] = self.0;
    }

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        ScalarVector(self.0 + rhs.0)
    }

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        ScalarVector(self.0 - rhs.0)
    }

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        ScalarVector(self.0 * rhs.0)
    }

    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        ScalarVector(self.0 / rhs.0)
    }

    #[inline(always)]
    fn neg(self) -> Self {
        ScalarVector(-self.0)
    }

    #[inline(always)]
    fn abs(self) -> Self {
        ScalarVector(libm::fabs(self.0))
    }

    #[inline(always)]
    fn fma(self, b: Self, c: Self) -> Self {
        ScalarVector(libm::fma(self.0, b.0, c.0))
    }

    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        ScalarVector(libm::fmin(self.0, rhs.0))
    }

    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        ScalarVector(libm::fmax(self.0, rhs.0))
    }

    #[inline(always)]
    fn lt(self, rhs: Self) -> Self::Mask {
        ScalarMask(self.0 < rhs.0)
    }

    #[inline(always)]
    fn gt(self, rhs: Self) -> Self::Mask {
        ScalarMask(self.0 > rhs.0)
    }

    #[inline(always)]
    fn eq(self, rhs: Self) -> Self::Mask {
        ScalarMask(self.0 == rhs.0)
    }

    #[inline(always)]
    fn select(mask: Self::Mask, true_val: Self, false_val: Self) -> Self {
        if mask.0 {
            true_val
        } else {
            false_val
        }
    }

    #[inline(always)]
    fn horizontal_sum(self) -> Self::Scalar {
        self.0
    }

    #[inline(always)]
    fn horizontal_max(self) -> Self::Scalar {
        self.0
    }

    #[inline(always)]
    fn horizontal_min(self) -> Self::Scalar {
        self.0
    }
}

// Implement SimdMask for ScalarMask
impl SimdMask for ScalarMask {
    #[inline(always)]
    fn all(self) -> bool {
        self.0
    }

    #[inline(always)]
    fn any(self) -> bool {
        self.0
    }

    #[inline(always)]
    fn none(self) -> bool {
        !self.0
    }

    #[inline(always)]
    fn and(self, rhs: Self) -> Self {
        ScalarMask(self.0 && rhs.0)
    }

    #[inline(always)]
    fn or(self, rhs: Self) -> Self {
        ScalarMask(self.0 || rhs.0)
    }

    #[inline(always)]
    fn not(self) -> Self {
        ScalarMask(!self.0)
    }

    #[inline(always)]
    fn xor(self, rhs: Self) -> Self {
        ScalarMask(self.0 ^ rhs.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_arithmetic() {
        let a = ScalarVector(2.0f32);
        let b = ScalarVector(3.0f32);

        assert_eq!(a.add(b), ScalarVector(5.0));
        assert_eq!(a.sub(b), ScalarVector(-1.0));
        assert_eq!(a.mul(b), ScalarVector(6.0));
        assert_eq!(a.div(b).0, 2.0 / 3.0);
    }

    #[test]
    fn test_scalar_fma() {
        let a = ScalarVector(2.0f32);
        let b = ScalarVector(3.0f32);
        let c = ScalarVector(1.0f32);

        let result = a.fma(b, c);
        assert_eq!(result, ScalarVector(7.0)); // 2 * 3 + 1
    }

    #[test]
    fn test_scalar_minmax() {
        let a = ScalarVector(2.0f32);
        let b = ScalarVector(3.0f32);

        assert_eq!(a.min(b), ScalarVector(2.0));
        assert_eq!(a.max(b), ScalarVector(3.0));
    }

    #[test]
    fn test_scalar_comparison() {
        let a = ScalarVector(2.0f32);
        let b = ScalarVector(3.0f32);

        assert!(a.lt(b).0);
        assert!(!a.gt(b).0);
        assert!(!a.eq(b).0);
    }

    #[test]
    fn test_scalar_select() {
        let a = ScalarVector(1.0f32);
        let b = ScalarVector(2.0f32);
        let mask_true = ScalarMask(true);
        let mask_false = ScalarMask(false);

        assert_eq!(ScalarVector::select(mask_true, a, b), a);
        assert_eq!(ScalarVector::select(mask_false, a, b), b);
    }

    #[test]
    fn test_scalar_horizontal() {
        let a = ScalarVector(5.0f32);

        assert_eq!(a.horizontal_sum(), 5.0);
        assert_eq!(a.horizontal_max(), 5.0);
        assert_eq!(a.horizontal_min(), 5.0);
    }

    #[test]
    fn test_scalar_mask() {
        let mask_true = ScalarMask(true);
        let mask_false = ScalarMask(false);

        assert!(mask_true.all());
        assert!(mask_true.any());
        assert!(!mask_true.none());

        assert!(!mask_false.all());
        assert!(!mask_false.any());
        assert!(mask_false.none());

        assert!(mask_true.and(mask_true).0);
        assert!(!mask_true.and(mask_false).0);
        assert!(mask_true.or(mask_false).0);
        assert!(!mask_true.xor(mask_true).0);
        assert!(mask_true.xor(mask_false).0);
    }
}
