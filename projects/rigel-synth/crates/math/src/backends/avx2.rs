//! AVX2 backend implementation (x86-64)
//!
//! This backend provides 8-lane (256-bit) SIMD operations using AVX2 instructions.
//! Requires x86-64 CPU with AVX2 support (Intel Haswell 2013+, AMD Excavator 2015+).
//!
//! **Note**: This implementation assumes AVX2 is available when the `avx2` feature is enabled.
//! Runtime CPU detection is not performed - use feature flags at compile time only.

// This backend only compiles on x86/x86_64 targets
#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]

use crate::traits::{SimdMask, SimdVector};

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "x86")]
use core::arch::x86::*;

/// AVX2 vector wrapper (8 lanes of f32)
///
/// Wraps __m256 intrinsic type to provide SimdVector trait implementation.
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct Avx2Vector(__m256);

/// AVX2 mask wrapper (8-lane mask)
///
/// Uses __m256 to represent per-lane boolean values.
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct Avx2Mask(__m256);

// Implement SimdVector for Avx2Vector
impl SimdVector for Avx2Vector {
    type Scalar = f32;
    type Mask = Avx2Mask;

    const LANES: usize = 8;

    #[inline(always)]
    fn splat(value: Self::Scalar) -> Self {
        unsafe { Avx2Vector(_mm256_set1_ps(value)) }
    }

    #[inline(always)]
    fn from_slice(slice: &[Self::Scalar]) -> Self {
        assert!(slice.len() >= Self::LANES, "Slice too short for AVX2 load");
        unsafe { Avx2Vector(_mm256_loadu_ps(slice.as_ptr())) }
    }

    #[inline(always)]
    fn to_slice(self, slice: &mut [Self::Scalar]) {
        assert!(slice.len() >= Self::LANES, "Slice too short for AVX2 store");
        unsafe { _mm256_storeu_ps(slice.as_mut_ptr(), self.0) }
    }

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        unsafe { Avx2Vector(_mm256_add_ps(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        unsafe { Avx2Vector(_mm256_sub_ps(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        unsafe { Avx2Vector(_mm256_mul_ps(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        unsafe { Avx2Vector(_mm256_div_ps(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn neg(self) -> Self {
        unsafe {
            let zero = _mm256_setzero_ps();
            Avx2Vector(_mm256_sub_ps(zero, self.0))
        }
    }

    #[inline(always)]
    fn abs(self) -> Self {
        unsafe {
            let mask = _mm256_set1_ps(f32::from_bits(0x7fffffff));
            Avx2Vector(_mm256_and_ps(self.0, mask))
        }
    }

    #[inline(always)]
    fn fma(self, b: Self, c: Self) -> Self {
        unsafe { Avx2Vector(_mm256_fmadd_ps(self.0, b.0, c.0)) }
    }

    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        unsafe { Avx2Vector(_mm256_min_ps(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        unsafe { Avx2Vector(_mm256_max_ps(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn lt(self, rhs: Self) -> Self::Mask {
        unsafe { Avx2Mask(_mm256_cmp_ps::<_CMP_LT_OQ>(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn gt(self, rhs: Self) -> Self::Mask {
        unsafe { Avx2Mask(_mm256_cmp_ps::<_CMP_GT_OQ>(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn eq(self, rhs: Self) -> Self::Mask {
        unsafe { Avx2Mask(_mm256_cmp_ps::<_CMP_EQ_OQ>(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn select(mask: Self::Mask, true_val: Self, false_val: Self) -> Self {
        unsafe { Avx2Vector(_mm256_blendv_ps(false_val.0, true_val.0, mask.0)) }
    }

    #[inline(always)]
    fn horizontal_sum(self) -> Self::Scalar {
        unsafe {
            // Extract high and low 128-bit halves
            let high = _mm256_extractf128_ps::<1>(self.0);
            let low = _mm256_castps256_ps128(self.0);
            let sum128 = _mm_add_ps(high, low);

            // Horizontal add within 128-bit
            let shuf = _mm_movehdup_ps(sum128);
            let sums = _mm_add_ps(sum128, shuf);
            let shuf = _mm_movehl_ps(shuf, sums);
            let result = _mm_add_ss(sums, shuf);

            _mm_cvtss_f32(result)
        }
    }

    #[inline(always)]
    fn horizontal_max(self) -> Self::Scalar {
        unsafe {
            let high = _mm256_extractf128_ps::<1>(self.0);
            let low = _mm256_castps256_ps128(self.0);
            let max128 = _mm_max_ps(high, low);

            let shuf = _mm_movehdup_ps(max128);
            let maxs = _mm_max_ps(max128, shuf);
            let shuf = _mm_movehl_ps(shuf, maxs);
            let result = _mm_max_ss(maxs, shuf);

            _mm_cvtss_f32(result)
        }
    }

    #[inline(always)]
    fn horizontal_min(self) -> Self::Scalar {
        unsafe {
            let high = _mm256_extractf128_ps::<1>(self.0);
            let low = _mm256_castps256_ps128(self.0);
            let min128 = _mm_min_ps(high, low);

            let shuf = _mm_movehdup_ps(min128);
            let mins = _mm_min_ps(min128, shuf);
            let shuf = _mm_movehl_ps(shuf, mins);
            let result = _mm_min_ss(mins, shuf);

            _mm_cvtss_f32(result)
        }
    }
}

// Implement SimdMask for Avx2Mask
impl SimdMask for Avx2Mask {
    #[inline(always)]
    fn all(self) -> bool {
        unsafe { _mm256_movemask_ps(self.0) == 0xff }
    }

    #[inline(always)]
    fn any(self) -> bool {
        unsafe { _mm256_movemask_ps(self.0) != 0 }
    }

    #[inline(always)]
    fn none(self) -> bool {
        unsafe { _mm256_movemask_ps(self.0) == 0 }
    }

    #[inline(always)]
    fn and(self, rhs: Self) -> Self {
        unsafe { Avx2Mask(_mm256_and_ps(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn or(self, rhs: Self) -> Self {
        unsafe { Avx2Mask(_mm256_or_ps(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn not(self) -> Self {
        unsafe {
            // Create all-ones mask
            let ones = _mm256_cmp_ps::<_CMP_EQ_OQ>(_mm256_setzero_ps(), _mm256_setzero_ps());
            // XOR with all-ones to invert
            Avx2Mask(_mm256_andnot_ps(self.0, ones))
        }
    }

    #[inline(always)]
    fn xor(self, rhs: Self) -> Self {
        unsafe {
            // XOR via (A AND NOT B) OR (NOT A AND B)
            let a_and_not_b = _mm256_andnot_ps(rhs.0, self.0);
            let not_a_and_b = _mm256_andnot_ps(self.0, rhs.0);
            Avx2Mask(_mm256_or_ps(a_and_not_b, not_a_and_b))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_avx2_arithmetic() {
        if is_x86_feature_detected!("avx2") {
            let a = Avx2Vector::splat(2.0);
            let b = Avx2Vector::splat(3.0);

            let sum = a.add(b);
            assert_eq!(sum.horizontal_sum(), 40.0); // 5.0 * 8 lanes

            let diff = a.sub(b);
            assert_eq!(diff.horizontal_sum(), -8.0); // -1.0 * 8 lanes

            let prod = a.mul(b);
            assert_eq!(prod.horizontal_sum(), 48.0); // 6.0 * 8 lanes
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_avx2_fma() {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            let a = Avx2Vector::splat(2.0);
            let b = Avx2Vector::splat(3.0);
            let c = Avx2Vector::splat(1.0);

            let result = a.fma(b, c);
            assert_eq!(result.horizontal_sum(), 56.0); // 7.0 * 8 lanes
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_avx2_comparison() {
        if is_x86_feature_detected!("avx2") {
            let a = Avx2Vector::splat(2.0);
            let b = Avx2Vector::splat(3.0);

            let mask_lt = a.lt(b);
            assert!(mask_lt.all());

            let mask_gt = a.gt(b);
            assert!(mask_gt.none());
        }
    }
}
