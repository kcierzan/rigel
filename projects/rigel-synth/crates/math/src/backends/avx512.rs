//! AVX512 backend implementation (x86-64)
//!
//! This backend provides 16-lane (512-bit) SIMD operations using AVX-512 instructions.
//! Requires x86-64 CPU with AVX-512 Foundation support (Intel Skylake-X 2017+, AMD Zen 4 2022+).
//!
//! **Note**: This implementation assumes AVX-512 is available when the `avx512` feature is enabled.
//! Runtime CPU detection is not performed - use feature flags at compile time only.

// This backend only compiles on x86/x86_64 targets
#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]

use crate::traits::{SimdMask, SimdVector};

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "x86")]
use core::arch::x86::*;

/// AVX512 vector wrapper (16 lanes of f32)
///
/// Wraps __m512 intrinsic type to provide SimdVector trait implementation.
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct Avx512Vector(__m512);

/// AVX512 mask wrapper (16-lane mask)
///
/// Uses __mmask16 to represent per-lane boolean values using AVX-512's native mask registers.
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct Avx512Mask(__mmask16);

// Implement SimdVector for Avx512Vector
impl SimdVector for Avx512Vector {
    type Scalar = f32;
    type Mask = Avx512Mask;

    const LANES: usize = 16;

    #[inline(always)]
    fn splat(value: Self::Scalar) -> Self {
        unsafe { Avx512Vector(_mm512_set1_ps(value)) }
    }

    #[inline(always)]
    fn from_slice(slice: &[Self::Scalar]) -> Self {
        assert!(
            slice.len() >= Self::LANES,
            "Slice too short for AVX512 load"
        );
        unsafe { Avx512Vector(_mm512_loadu_ps(slice.as_ptr())) }
    }

    #[inline(always)]
    fn to_slice(self, slice: &mut [Self::Scalar]) {
        assert!(
            slice.len() >= Self::LANES,
            "Slice too short for AVX512 store"
        );
        unsafe { _mm512_storeu_ps(slice.as_mut_ptr(), self.0) }
    }

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        unsafe { Avx512Vector(_mm512_add_ps(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        unsafe { Avx512Vector(_mm512_sub_ps(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        unsafe { Avx512Vector(_mm512_mul_ps(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        unsafe { Avx512Vector(_mm512_div_ps(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn neg(self) -> Self {
        unsafe {
            let zero = _mm512_setzero_ps();
            Avx512Vector(_mm512_sub_ps(zero, self.0))
        }
    }

    #[inline(always)]
    fn abs(self) -> Self {
        unsafe { Avx512Vector(_mm512_abs_ps(self.0)) }
    }

    #[inline(always)]
    fn fma(self, b: Self, c: Self) -> Self {
        unsafe { Avx512Vector(_mm512_fmadd_ps(self.0, b.0, c.0)) }
    }

    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        unsafe { Avx512Vector(_mm512_min_ps(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        unsafe { Avx512Vector(_mm512_max_ps(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn lt(self, rhs: Self) -> Self::Mask {
        unsafe { Avx512Mask(_mm512_cmp_ps_mask::<_CMP_LT_OQ>(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn gt(self, rhs: Self) -> Self::Mask {
        unsafe { Avx512Mask(_mm512_cmp_ps_mask::<_CMP_GT_OQ>(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn eq(self, rhs: Self) -> Self::Mask {
        unsafe { Avx512Mask(_mm512_cmp_ps_mask::<_CMP_EQ_OQ>(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn select(mask: Self::Mask, true_val: Self, false_val: Self) -> Self {
        unsafe { Avx512Vector(_mm512_mask_blend_ps(mask.0, false_val.0, true_val.0)) }
    }

    #[inline(always)]
    fn horizontal_sum(self) -> Self::Scalar {
        unsafe { _mm512_reduce_add_ps(self.0) }
    }

    #[inline(always)]
    fn horizontal_max(self) -> Self::Scalar {
        unsafe { _mm512_reduce_max_ps(self.0) }
    }

    #[inline(always)]
    fn horizontal_min(self) -> Self::Scalar {
        unsafe { _mm512_reduce_min_ps(self.0) }
    }
}

// Implement SimdMask for Avx512Mask
impl SimdMask for Avx512Mask {
    #[inline(always)]
    fn all(self) -> bool {
        self.0 == 0xffff
    }

    #[inline(always)]
    fn any(self) -> bool {
        self.0 != 0
    }

    #[inline(always)]
    fn none(self) -> bool {
        self.0 == 0
    }

    #[inline(always)]
    fn and(self, rhs: Self) -> Self {
        Avx512Mask(self.0 & rhs.0)
    }

    #[inline(always)]
    fn or(self, rhs: Self) -> Self {
        Avx512Mask(self.0 | rhs.0)
    }

    #[inline(always)]
    fn not(self) -> Self {
        Avx512Mask(!self.0 & 0xffff)
    }

    #[inline(always)]
    fn xor(self, rhs: Self) -> Self {
        Avx512Mask(self.0 ^ rhs.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_avx512_arithmetic() {
        if is_x86_feature_detected!("avx512f") {
            let a = Avx512Vector::splat(2.0);
            let b = Avx512Vector::splat(3.0);

            let sum = a.add(b);
            assert_eq!(sum.horizontal_sum(), 80.0); // 5.0 * 16 lanes

            let diff = a.sub(b);
            assert_eq!(diff.horizontal_sum(), -16.0); // -1.0 * 16 lanes

            let prod = a.mul(b);
            assert_eq!(prod.horizontal_sum(), 96.0); // 6.0 * 16 lanes
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_avx512_fma() {
        if is_x86_feature_detected!("avx512f") {
            let a = Avx512Vector::splat(2.0);
            let b = Avx512Vector::splat(3.0);
            let c = Avx512Vector::splat(1.0);

            let result = a.fma(b, c);
            assert_eq!(result.horizontal_sum(), 112.0); // 7.0 * 16 lanes
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_avx512_comparison() {
        if is_x86_feature_detected!("avx512f") {
            let a = Avx512Vector::splat(2.0);
            let b = Avx512Vector::splat(3.0);

            let mask_lt = a.lt(b);
            assert!(mask_lt.all());

            let mask_gt = a.gt(b);
            assert!(mask_gt.none());
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_avx512_mask_operations() {
        if is_x86_feature_detected!("avx512f") {
            let a = Avx512Vector::splat(1.0);
            let b = Avx512Vector::splat(2.0);

            let mask_lt = a.lt(b);
            let mask_gt = a.gt(b);

            // AND: all true AND all false = all false
            let and_mask = mask_lt.and(mask_gt);
            assert!(and_mask.none());

            // OR: all true OR all false = all true
            let or_mask = mask_lt.or(mask_gt);
            assert!(or_mask.all());

            // NOT: NOT all true = all false
            let not_mask = mask_lt.not();
            assert!(not_mask.none());

            // XOR: all true XOR all false = all true
            let xor_mask = mask_lt.xor(mask_gt);
            assert!(xor_mask.all());
        }
    }
}
