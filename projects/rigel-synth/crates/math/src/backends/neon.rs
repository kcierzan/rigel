//! NEON backend implementation (ARM64)
//!
//! This backend provides 4-lane (128-bit) SIMD operations using ARM NEON instructions.
//! Available on all ARM64 CPUs (Apple Silicon, AWS Graviton, Raspberry Pi 4+, etc.).
//!
//! **Note**: This implementation assumes NEON is available on aarch64 targets.
//! NEON is mandatory for ARM64, so no runtime detection is needed.

// This backend only compiles on aarch64 targets
#![cfg(target_arch = "aarch64")]

use crate::traits::{SimdMask, SimdVector};
use core::arch::aarch64::*;

/// NEON vector wrapper (4 lanes of f32)
///
/// Wraps float32x4_t intrinsic type to provide SimdVector trait implementation.
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct NeonVector(float32x4_t);

/// NEON mask wrapper (4-lane mask)
///
/// Uses uint32x4_t to represent per-lane boolean values.
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct NeonMask(uint32x4_t);

// Implement SimdVector for NeonVector
impl SimdVector for NeonVector {
    type Scalar = f32;
    type Mask = NeonMask;

    const LANES: usize = 4;

    #[inline(always)]
    fn splat(value: Self::Scalar) -> Self {
        unsafe { NeonVector(vdupq_n_f32(value)) }
    }

    #[inline(always)]
    fn from_slice(slice: &[Self::Scalar]) -> Self {
        assert!(slice.len() >= Self::LANES, "Slice too short for NEON load");
        unsafe { NeonVector(vld1q_f32(slice.as_ptr())) }
    }

    #[inline(always)]
    fn to_slice(self, slice: &mut [Self::Scalar]) {
        assert!(slice.len() >= Self::LANES, "Slice too short for NEON store");
        unsafe { vst1q_f32(slice.as_mut_ptr(), self.0) }
    }

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        unsafe { NeonVector(vaddq_f32(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        unsafe { NeonVector(vsubq_f32(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        unsafe { NeonVector(vmulq_f32(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        unsafe { NeonVector(vdivq_f32(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn neg(self) -> Self {
        unsafe { NeonVector(vnegq_f32(self.0)) }
    }

    #[inline(always)]
    fn abs(self) -> Self {
        unsafe { NeonVector(vabsq_f32(self.0)) }
    }

    #[inline(always)]
    fn fma(self, b: Self, c: Self) -> Self {
        // NEON has vfmaq_f32: a + (b * c), but we need self * b + c
        // So we use vfmaq_f32(c, self, b) = c + (self * b) = self * b + c
        unsafe { NeonVector(vfmaq_f32(c.0, self.0, b.0)) }
    }

    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        unsafe { NeonVector(vminq_f32(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        unsafe { NeonVector(vmaxq_f32(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn lt(self, rhs: Self) -> Self::Mask {
        unsafe { NeonMask(vcltq_f32(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn gt(self, rhs: Self) -> Self::Mask {
        unsafe { NeonMask(vcgtq_f32(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn eq(self, rhs: Self) -> Self::Mask {
        unsafe { NeonMask(vceqq_f32(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn select(mask: Self::Mask, true_val: Self, false_val: Self) -> Self {
        // NEON: vbslq_f32(mask, true_val, false_val)
        unsafe { NeonVector(vbslq_f32(mask.0, true_val.0, false_val.0)) }
    }

    #[inline(always)]
    fn horizontal_sum(self) -> Self::Scalar {
        unsafe {
            // Pairwise addition
            let sum_pair = vpaddq_f32(self.0, self.0); // [a+b, c+d, a+b, c+d]
            let sum = vpaddq_f32(sum_pair, sum_pair); // [a+b+c+d, ...]
            vgetq_lane_f32(sum, 0)
        }
    }

    #[inline(always)]
    fn horizontal_max(self) -> Self::Scalar {
        unsafe {
            // Pairwise maximum
            let max_pair = vpmaxq_f32(self.0, self.0);
            let max = vpmaxq_f32(max_pair, max_pair);
            vgetq_lane_f32(max, 0)
        }
    }

    #[inline(always)]
    fn horizontal_min(self) -> Self::Scalar {
        unsafe {
            // Pairwise minimum
            let min_pair = vpminq_f32(self.0, self.0);
            let min = vpminq_f32(min_pair, min_pair);
            vgetq_lane_f32(min, 0)
        }
    }
}

// Implement SimdMask for NeonMask
impl SimdMask for NeonMask {
    #[inline(always)]
    fn all(self) -> bool {
        unsafe {
            // Check if all lanes are 0xFFFFFFFF (all bits set)
            let min = vminvq_u32(self.0);
            min == 0xFFFFFFFF
        }
    }

    #[inline(always)]
    fn any(self) -> bool {
        unsafe {
            // Check if any lane is non-zero
            let max = vmaxvq_u32(self.0);
            max != 0
        }
    }

    #[inline(always)]
    fn none(self) -> bool {
        unsafe {
            // Check if all lanes are 0
            let max = vmaxvq_u32(self.0);
            max == 0
        }
    }

    #[inline(always)]
    fn and(self, rhs: Self) -> Self {
        unsafe { NeonMask(vandq_u32(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn or(self, rhs: Self) -> Self {
        unsafe { NeonMask(vorrq_u32(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn not(self) -> Self {
        unsafe { NeonMask(vmvnq_u32(self.0)) }
    }

    #[inline(always)]
    fn xor(self, rhs: Self) -> Self {
        unsafe { NeonMask(veorq_u32(self.0, rhs.0)) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neon_arithmetic() {
        let a = NeonVector::splat(2.0);
        let b = NeonVector::splat(3.0);

        let sum = a.add(b);
        assert_eq!(sum.horizontal_sum(), 20.0); // 5.0 * 4 lanes

        let diff = a.sub(b);
        assert_eq!(diff.horizontal_sum(), -4.0); // -1.0 * 4 lanes

        let prod = a.mul(b);
        assert_eq!(prod.horizontal_sum(), 24.0); // 6.0 * 4 lanes
    }

    #[test]
    fn test_neon_fma() {
        let a = NeonVector::splat(2.0);
        let b = NeonVector::splat(3.0);
        let c = NeonVector::splat(1.0);

        let result = a.fma(b, c);
        assert_eq!(result.horizontal_sum(), 28.0); // 7.0 * 4 lanes
    }

    #[test]
    fn test_neon_minmax() {
        let a = NeonVector::splat(2.0);
        let b = NeonVector::splat(3.0);

        assert_eq!(a.min(b).horizontal_sum(), 8.0); // 2.0 * 4
        assert_eq!(a.max(b).horizontal_sum(), 12.0); // 3.0 * 4
    }

    #[test]
    fn test_neon_comparison() {
        let a = NeonVector::splat(2.0);
        let b = NeonVector::splat(3.0);

        let mask_lt = a.lt(b);
        assert!(mask_lt.all());

        let mask_gt = a.gt(b);
        assert!(mask_gt.none());

        let mask_eq = a.eq(a);
        assert!(mask_eq.all());
    }

    #[test]
    fn test_neon_select() {
        let a = NeonVector::splat(1.0);
        let b = NeonVector::splat(2.0);
        let mask_true = a.lt(b); // All lanes true
        let mask_false = a.gt(b); // All lanes false

        let result_true = NeonVector::select(mask_true, a, b);
        assert_eq!(result_true.horizontal_sum(), 4.0); // a selected

        let result_false = NeonVector::select(mask_false, a, b);
        assert_eq!(result_false.horizontal_sum(), 8.0); // b selected
    }

    #[test]
    fn test_neon_horizontal() {
        let a = NeonVector::splat(5.0);

        assert_eq!(a.horizontal_sum(), 20.0); // 5.0 * 4
        assert_eq!(a.horizontal_max(), 5.0);
        assert_eq!(a.horizontal_min(), 5.0);
    }

    #[test]
    fn test_neon_mask() {
        let a = NeonVector::splat(2.0);
        let b = NeonVector::splat(3.0);

        let mask_true = a.lt(b);
        let mask_false = a.gt(b);

        assert!(mask_true.all());
        assert!(mask_true.any());
        assert!(!mask_true.none());

        assert!(!mask_false.all());
        assert!(!mask_false.any());
        assert!(mask_false.none());

        assert!(mask_true.and(mask_true).all());
        assert!(!mask_true.and(mask_false).any());
        assert!(mask_true.or(mask_false).all());
        assert!(!mask_true.xor(mask_true).any());
        assert!(mask_true.xor(mask_false).all());
    }
}
