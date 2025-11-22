//! AVX512 backend implementation (x86-64)
//!
//! This backend provides 16-lane (512-bit) SIMD operations using AVX-512 instructions.
//! Requires x86-64 CPU with AVX-512 Foundation support (Intel Skylake-X 2017+, AMD Zen 4 2022+).
//!
//! **Note**: This implementation assumes AVX-512 is available when the `avx512` feature is enabled.
//! Runtime CPU detection is not performed - use feature flags at compile time only.

// This backend only compiles on x86/x86_64 targets
#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]

use crate::traits::{SimdInt, SimdMask, SimdVector};

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

/// AVX512 integer vector wrapper (16 lanes of u32)
///
/// Used for bit manipulation operations in IEEE 754 logarithm extraction.
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct Avx512Int(__m512i);

// Implement SimdInt for Avx512Int
impl SimdInt for Avx512Int {
    const LANES: usize = 16;
    type FloatVec = Avx512Vector;

    #[inline(always)]
    fn splat(value: u32) -> Self {
        unsafe { Avx512Int(_mm512_set1_epi32(value as i32)) }
    }

    #[inline(always)]
    fn shr(self, count: u32) -> Self {
        unsafe {
            // AVX-512 shift requires a 128-bit count vector (same as AVX2)
            let shift_count = _mm_cvtsi32_si128(count as i32);
            Avx512Int(_mm512_srl_epi32(self.0, shift_count))
        }
    }

    #[inline(always)]
    fn shl(self, count: u32) -> Self {
        unsafe {
            // AVX-512 shift requires a 128-bit count vector (same as AVX2)
            let shift_count = _mm_cvtsi32_si128(count as i32);
            Avx512Int(_mm512_sll_epi32(self.0, shift_count))
        }
    }

    #[inline(always)]
    fn bitwise_and(self, rhs: u32) -> Self {
        unsafe {
            let rhs_vec = _mm512_set1_epi32(rhs as i32);
            Avx512Int(_mm512_and_si512(self.0, rhs_vec))
        }
    }

    #[inline(always)]
    fn bitwise_or(self, rhs: u32) -> Self {
        unsafe {
            let rhs_vec = _mm512_set1_epi32(rhs as i32);
            Avx512Int(_mm512_or_si512(self.0, rhs_vec))
        }
    }

    #[inline(always)]
    fn sub_scalar(self, rhs: u32) -> Self {
        unsafe {
            let rhs_vec = _mm512_set1_epi32(rhs as i32);
            Avx512Int(_mm512_sub_epi32(self.0, rhs_vec))
        }
    }

    #[inline(always)]
    fn add_scalar(self, rhs: u32) -> Self {
        unsafe {
            let rhs_vec = _mm512_set1_epi32(rhs as i32);
            Avx512Int(_mm512_add_epi32(self.0, rhs_vec))
        }
    }

    #[inline(always)]
    fn from_f32_to_i32(float_vec: Self::FloatVec) -> Self {
        unsafe { Avx512Int(_mm512_cvtps_epi32(float_vec.0)) }
    }

    #[inline(always)]
    fn to_f32(self) -> Self::FloatVec {
        unsafe { Avx512Vector(_mm512_cvtepi32_ps(self.0)) }
    }
}

// Implement SimdVector for Avx512Vector
impl SimdVector for Avx512Vector {
    type Scalar = f32;
    type Mask = Avx512Mask;
    type IntBits = Avx512Int;

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
        unsafe {
            // IEEE 754-2008 minNum semantics: if one value is NaN, return the other
            // AVX-512's _mm512_min_ps may still propagate NaN in some cases
            let min_result = _mm512_min_ps(self.0, rhs.0);

            // Check for NaN using mask: value != value
            let self_is_nan = _mm512_cmp_ps_mask::<_CMP_UNORD_Q>(self.0, self.0);
            let rhs_is_nan = _mm512_cmp_ps_mask::<_CMP_UNORD_Q>(rhs.0, rhs.0);

            // If self is NaN, use rhs; if rhs is NaN, use self; else use min_result
            let result = _mm512_mask_blend_ps(
                self_is_nan,
                _mm512_mask_blend_ps(rhs_is_nan, min_result, self.0),
                rhs.0,
            );
            Avx512Vector(result)
        }
    }

    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        unsafe {
            // IEEE 754-2008 maxNum semantics: if one value is NaN, return the other
            // AVX-512's _mm512_max_ps may still propagate NaN in some cases
            let max_result = _mm512_max_ps(self.0, rhs.0);

            // Check for NaN using mask: value != value
            let self_is_nan = _mm512_cmp_ps_mask::<_CMP_UNORD_Q>(self.0, self.0);
            let rhs_is_nan = _mm512_cmp_ps_mask::<_CMP_UNORD_Q>(rhs.0, rhs.0);

            // If self is NaN, use rhs; if rhs is NaN, use self; else use max_result
            let result = _mm512_mask_blend_ps(
                self_is_nan,
                _mm512_mask_blend_ps(rhs_is_nan, max_result, self.0),
                rhs.0,
            );
            Avx512Vector(result)
        }
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

    #[inline(always)]
    fn floor(self) -> Self {
        // _mm512_floor_ps doesn't exist in Rust std::arch
        // Use _mm512_roundscale_ps with floor rounding mode instead
        // 0x09 = _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC
        unsafe { Avx512Vector(_mm512_roundscale_ps::<0x09>(self.0)) }
    }

    #[inline(always)]
    fn to_int_bits_i32(self) -> Self::IntBits {
        unsafe { Avx512Int(_mm512_cvtps_epi32(self.0)) }
    }

    #[inline(always)]
    fn to_bits(self) -> Self::IntBits {
        unsafe { Avx512Int(_mm512_castps_si512(self.0)) }
    }

    #[inline(always)]
    fn from_bits(bits: Self::IntBits) -> Self {
        unsafe { Avx512Vector(_mm512_castsi512_ps(bits.0)) }
    }

    #[inline(always)]
    fn from_int_cast(int_vec: Self::IntBits) -> Self {
        unsafe { Avx512Vector(_mm512_cvtepi32_ps(int_vec.0)) }
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

    // Tests assume AVX-512 is available when compiled with target-feature=+avx512f
    // No runtime detection needed - this is a compile-time backend selection library

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_avx512_arithmetic() {
        #[target_feature(enable = "avx512f")]
        unsafe fn inner() {
            let a = Avx512Vector::splat(2.0);
            let b = Avx512Vector::splat(3.0);

            let sum = a.add(b);
            assert_eq!(sum.horizontal_sum(), 80.0); // 5.0 * 16 lanes

            let diff = a.sub(b);
            assert_eq!(diff.horizontal_sum(), -16.0); // -1.0 * 16 lanes

            let prod = a.mul(b);
            assert_eq!(prod.horizontal_sum(), 96.0); // 6.0 * 16 lanes
        }
        unsafe { inner() }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_avx512_fma() {
        #[target_feature(enable = "avx512f")]
        unsafe fn inner() {
            let a = Avx512Vector::splat(2.0);
            let b = Avx512Vector::splat(3.0);
            let c = Avx512Vector::splat(1.0);

            let result = a.fma(b, c);
            assert_eq!(result.horizontal_sum(), 112.0); // 7.0 * 16 lanes
        }
        unsafe { inner() }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_avx512_comparison() {
        #[target_feature(enable = "avx512f")]
        unsafe fn inner() {
            let a = Avx512Vector::splat(2.0);
            let b = Avx512Vector::splat(3.0);

            let mask_lt = a.lt(b);
            assert!(mask_lt.all());

            let mask_gt = a.gt(b);
            assert!(mask_gt.none());
        }
        unsafe { inner() }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_avx512_mask_operations() {
        #[target_feature(enable = "avx512f")]
        unsafe fn inner() {
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
        unsafe { inner() }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_avx512_bit_manipulation() {
        #[target_feature(enable = "avx512f")]
        unsafe fn inner() {
            use crate::traits::SimdInt;

            // Test to_bits / from_bits round trip
            let vec = Avx512Vector::splat(1.0);
            let bits = vec.to_bits();
            let restored = Avx512Vector::from_bits(bits);
            assert_eq!(restored.horizontal_sum(), 16.0); // 1.0 * 16 lanes

            // Test from_int_cast (numerical conversion)
            let int_vec = Avx512Int::splat(5);
            let float_vec = Avx512Vector::from_int_cast(int_vec);
            assert_eq!(float_vec.horizontal_sum(), 80.0); // 5.0 * 16 lanes

            // Test Avx512Int operations
            let int_a = Avx512Int::splat(0xF0);
            let int_b = int_a.bitwise_and(0x0F);
            let result = Avx512Vector::from_int_cast(int_b);
            assert_eq!(result.horizontal_sum(), 0.0); // (0xF0 & 0x0F) = 0

            // Test shift operations
            let int_val = Avx512Int::splat(8);
            let shifted_right = int_val.shr(2); // 8 >> 2 = 2
            let result_shr = Avx512Vector::from_int_cast(shifted_right);
            assert_eq!(result_shr.horizontal_sum(), 32.0); // 2.0 * 16 lanes

            let shifted_left = int_val.shl(2); // 8 << 2 = 32
            let result_shl = Avx512Vector::from_int_cast(shifted_left);
            assert_eq!(result_shl.horizontal_sum(), 512.0); // 32.0 * 16 lanes
        }
        unsafe { inner() }
    }
}
