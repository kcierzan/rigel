//! NEON backend implementation (ARM64)
//!
//! This backend provides 4-lane (128-bit) SIMD operations using ARM NEON instructions.
//! Available on all ARM64 CPUs (Apple Silicon, AWS Graviton, Raspberry Pi 4+, etc.).
//!
//! **Note**: This implementation assumes NEON is available on aarch64 targets.
//! NEON is mandatory for ARM64, so no runtime detection is needed.

// This backend only compiles on aarch64 targets
#![cfg(target_arch = "aarch64")]

use crate::traits::{SimdInt, SimdMask, SimdVector};
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

/// NEON integer vector wrapper (4 lanes of u32)
///
/// Used for bit manipulation operations in IEEE 754 logarithm extraction.
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct NeonInt(uint32x4_t);

// Implement SimdInt for NeonInt
impl SimdInt for NeonInt {
    const LANES: usize = 4;
    type FloatVec = NeonVector;

    #[inline(always)]
    fn splat(value: u32) -> Self {
        unsafe { NeonInt(vdupq_n_u32(value)) }
    }

    #[inline(always)]
    fn shr(self, count: u32) -> Self {
        unsafe {
            // For variable shift, use vshlq_u32 with negative count (right shift)
            let shift_vec = vdupq_n_s32(-(count as i32));
            NeonInt(vshlq_u32(self.0, shift_vec))
        }
    }

    #[inline(always)]
    fn shl(self, count: u32) -> Self {
        unsafe {
            // For variable shift, use vshlq_u32 with positive count (left shift)
            let shift_vec = vdupq_n_s32(count as i32);
            NeonInt(vshlq_u32(self.0, shift_vec))
        }
    }

    #[inline(always)]
    fn bitwise_and(self, rhs: u32) -> Self {
        unsafe {
            let rhs_vec = vdupq_n_u32(rhs);
            NeonInt(vandq_u32(self.0, rhs_vec))
        }
    }

    #[inline(always)]
    fn bitwise_or(self, rhs: u32) -> Self {
        unsafe {
            let rhs_vec = vdupq_n_u32(rhs);
            NeonInt(vorrq_u32(self.0, rhs_vec))
        }
    }

    #[inline(always)]
    fn sub_scalar(self, rhs: u32) -> Self {
        unsafe {
            let rhs_vec = vdupq_n_u32(rhs);
            NeonInt(vsubq_u32(self.0, rhs_vec))
        }
    }

    #[inline(always)]
    fn add_scalar(self, rhs: u32) -> Self {
        unsafe {
            let rhs_vec = vdupq_n_u32(rhs);
            NeonInt(vaddq_u32(self.0, rhs_vec))
        }
    }

    #[inline(always)]
    fn from_f32_to_i32(float_vec: Self::FloatVec) -> Self {
        unsafe {
            let s32 = vcvtq_s32_f32(float_vec.0);
            NeonInt(vreinterpretq_u32_s32(s32))
        }
    }

    #[inline(always)]
    fn to_f32(self) -> Self::FloatVec {
        unsafe { NeonVector(vcvtq_f32_u32(self.0)) }
    }
}

// Implement SimdVector for NeonVector
impl SimdVector for NeonVector {
    type Scalar = f32;
    type Mask = NeonMask;
    type IntBits = NeonInt;

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

    #[inline(always)]
    fn floor(self) -> Self {
        unsafe { NeonVector(vrndmq_f32(self.0)) }
    }

    #[inline(always)]
    fn to_int_bits_i32(self) -> Self::IntBits {
        unsafe {
            let s32 = vcvtq_s32_f32(self.0);
            NeonInt(vreinterpretq_u32_s32(s32))
        }
    }

    #[inline(always)]
    fn to_bits(self) -> Self::IntBits {
        unsafe { NeonInt(vreinterpretq_u32_f32(self.0)) }
    }

    #[inline(always)]
    fn from_bits(bits: Self::IntBits) -> Self {
        unsafe { NeonVector(vreinterpretq_f32_u32(bits.0)) }
    }

    #[inline(always)]
    fn from_int_cast(int_vec: Self::IntBits) -> Self {
        unsafe { NeonVector(vcvtq_f32_u32(int_vec.0)) }
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

    #[test]
    fn test_neon_bit_manipulation() {
        use crate::traits::SimdInt;

        // Test to_bits / from_bits round trip
        let vec = NeonVector::splat(1.0);
        let bits = vec.to_bits();
        let restored = NeonVector::from_bits(bits);
        assert_eq!(restored.horizontal_sum(), 4.0); // 1.0 * 4 lanes

        // Test from_int_cast (numerical conversion)
        let int_vec = NeonInt::splat(5);
        let float_vec = NeonVector::from_int_cast(int_vec);
        assert_eq!(float_vec.horizontal_sum(), 20.0); // 5.0 * 4 lanes

        // Test NeonInt operations
        let int_a = NeonInt::splat(0xF0);
        let int_b = int_a.bitwise_and(0x0F);
        let result = NeonVector::from_int_cast(int_b);
        assert_eq!(result.horizontal_sum(), 0.0); // (0xF0 & 0x0F) = 0

        // Test shift operations
        let int_val = NeonInt::splat(8);
        let shifted_right = int_val.shr(2); // 8 >> 2 = 2
        let result_shr = NeonVector::from_int_cast(shifted_right);
        assert_eq!(result_shr.horizontal_sum(), 8.0); // 2.0 * 4 lanes

        let shifted_left = int_val.shl(2); // 8 << 2 = 32
        let result_shl = NeonVector::from_int_cast(shifted_left);
        assert_eq!(result_shl.horizontal_sum(), 128.0); // 32.0 * 4 lanes
    }

    #[test]
    fn test_neon_infinity_handling() {
        // Test that min with f32::MAX works correctly on infinity
        let inf_vec = NeonVector::splat(f32::INFINITY);
        let max_vec = NeonVector::splat(f32::MAX);
        let result = inf_vec.min(max_vec);

        let mut buf = [0.0f32; 4];
        result.to_slice(&mut buf);

        // Should clamp to f32::MAX
        assert!(
            buf[0].is_finite(),
            "min(inf, MAX) should be finite, got {}",
            buf[0]
        );
        assert_eq!(buf[0], f32::MAX, "min(inf, MAX) should equal MAX");
    }

    #[test]
    fn test_neon_div_edge_cases() {
        // Test division edge cases
        let a = NeonVector::splat(1.0);
        let b = NeonVector::splat(0.0);
        let result = a.div(b);

        let mut buf = [0.0f32; 4];
        result.to_slice(&mut buf);

        // 1.0 / 0.0 should be infinity
        assert!(
            buf[0].is_infinite(),
            "1.0/0.0 should be inf, got {}",
            buf[0]
        );

        // Test large number division
        let large = NeonVector::splat(1e38);
        let small = NeonVector::splat(1e-10);
        let result2 = large.div(small);
        result2.to_slice(&mut buf);

        // Should produce infinity or very large number
        assert!(
            buf[0] > 1e30 || buf[0].is_infinite(),
            "1e38/1e-10 should be very large or inf, got {}",
            buf[0]
        );
    }
}
