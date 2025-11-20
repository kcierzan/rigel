//! Scalar backend implementation
//!
//! This backend provides a scalar (non-SIMD) fallback that always works on any platform.
//! It serves as the reference implementation and is useful for testing backend consistency.

use crate::traits::{SimdInt, SimdMask, SimdVector};

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

/// Scalar integer vector wrapper (single u32)
///
/// Used for bit manipulation operations in IEEE 754 logarithm extraction.
#[derive(Debug, Copy, Clone, PartialEq)]
#[repr(transparent)]
pub struct ScalarInt(pub u32);

/// Scalar integer vector wrapper for f64 (single u64)
///
/// Used for bit manipulation operations in IEEE 754 logarithm extraction for f64.
#[derive(Debug, Copy, Clone, PartialEq)]
#[repr(transparent)]
pub struct ScalarInt64(pub u64);

// Implement SimdInt for ScalarInt
impl SimdInt for ScalarInt {
    const LANES: usize = 1;
    type FloatVec = ScalarVector<f32>;

    #[inline(always)]
    fn splat(value: u32) -> Self {
        ScalarInt(value)
    }

    #[inline(always)]
    fn shr(self, count: u32) -> Self {
        ScalarInt(self.0 >> count)
    }

    #[inline(always)]
    fn shl(self, count: u32) -> Self {
        ScalarInt(self.0 << count)
    }

    #[inline(always)]
    fn bitwise_and(self, rhs: u32) -> Self {
        ScalarInt(self.0 & rhs)
    }

    #[inline(always)]
    fn bitwise_or(self, rhs: u32) -> Self {
        ScalarInt(self.0 | rhs)
    }

    #[inline(always)]
    fn sub_scalar(self, rhs: u32) -> Self {
        ScalarInt(self.0.wrapping_sub(rhs))
    }

    #[inline(always)]
    fn to_f32(self) -> Self::FloatVec {
        ScalarVector(self.0 as f32)
    }
}

// Implement SimdVector for ScalarVector<f32>
impl SimdVector for ScalarVector<f32> {
    type Scalar = f32;
    type Mask = ScalarMask;
    type IntBits = ScalarInt;

    const LANES: usize = 1;

    #[inline(always)]
    fn splat(value: Self::Scalar) -> Self {
        ScalarVector(value)
    }

    #[inline(always)]
    fn from_slice(slice: &[Self::Scalar]) -> Self {
        assert!(
            slice.len() >= Self::LANES,
            "Slice too short for scalar load"
        );
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

    #[inline(always)]
    fn to_bits(self) -> Self::IntBits {
        ScalarInt(self.0.to_bits())
    }

    #[inline(always)]
    fn from_bits(bits: Self::IntBits) -> Self {
        ScalarVector(f32::from_bits(bits.0))
    }

    #[inline(always)]
    fn from_int_cast(int_vec: Self::IntBits) -> Self {
        ScalarVector(int_vec.0 as f32)
    }
}

// Libm-optimized math functions for scalar backend
//
// These provide best-in-class scalar performance by using libm directly,
// while the generic polynomial implementations in src/math/*.rs are used for
// actual SIMD backends (AVX2, AVX512, NEON).
//
// Usage: For scalar code, call these methods directly (e.g., x.sin_libm()).
// For generic SIMD code, use the generic functions (e.g., sin(x)).
impl ScalarVector<f32> {
    /// Natural exponential function using libm
    #[inline(always)]
    pub fn exp_libm(self) -> Self {
        ScalarVector(libm::expf(self.0))
    }

    /// Natural logarithm using libm
    #[inline(always)]
    pub fn log_libm(self) -> Self {
        ScalarVector(libm::logf(self.0))
    }

    /// Base-2 logarithm using libm
    #[inline(always)]
    pub fn log2_libm(self) -> Self {
        ScalarVector(libm::log2f(self.0))
    }

    /// Base-10 logarithm using libm
    #[inline(always)]
    pub fn log10_libm(self) -> Self {
        ScalarVector(libm::log10f(self.0))
    }

    /// Natural logarithm of 1+x using libm
    #[inline(always)]
    pub fn log1p_libm(self) -> Self {
        ScalarVector(libm::log1pf(self.0))
    }

    /// Sine function using libm
    #[inline(always)]
    pub fn sin_libm(self) -> Self {
        ScalarVector(libm::sinf(self.0))
    }

    /// Cosine function using libm
    #[inline(always)]
    pub fn cos_libm(self) -> Self {
        ScalarVector(libm::cosf(self.0))
    }

    /// Tangent function using libm
    #[inline(always)]
    pub fn tan_libm(self) -> Self {
        ScalarVector(libm::tanf(self.0))
    }

    /// Arctangent function using libm
    #[inline(always)]
    pub fn atan_libm(self) -> Self {
        ScalarVector(libm::atanf(self.0))
    }

    /// Two-argument arctangent using libm
    #[inline(always)]
    pub fn atan2_libm(self, x: Self) -> Self {
        ScalarVector(libm::atan2f(self.0, x.0))
    }

    /// Hyperbolic tangent using libm
    #[inline(always)]
    pub fn tanh_libm(self) -> Self {
        ScalarVector(libm::tanhf(self.0))
    }

    /// Power function using libm
    #[inline(always)]
    pub fn pow_libm(self, exp: Self) -> Self {
        ScalarVector(libm::powf(self.0, exp.0))
    }

    /// Square root using libm
    #[inline(always)]
    pub fn sqrt_libm(self) -> Self {
        ScalarVector(libm::sqrtf(self.0))
    }

    /// Compute sine and cosine simultaneously using libm
    #[inline(always)]
    pub fn sincos_libm(self) -> (Self, Self) {
        let sin = libm::sinf(self.0);
        let cos = libm::cosf(self.0);
        (ScalarVector(sin), ScalarVector(cos))
    }
}

// Implement SimdInt for ScalarInt64 (f64 support - placeholder)
impl SimdInt for ScalarInt64 {
    const LANES: usize = 1;
    type FloatVec = ScalarVector<f64>;

    #[inline(always)]
    fn splat(value: u32) -> Self {
        ScalarInt64(value as u64)
    }

    #[inline(always)]
    fn shr(self, count: u32) -> Self {
        ScalarInt64(self.0 >> count)
    }

    #[inline(always)]
    fn shl(self, count: u32) -> Self {
        ScalarInt64(self.0 << count)
    }

    #[inline(always)]
    fn bitwise_and(self, rhs: u32) -> Self {
        ScalarInt64(self.0 & (rhs as u64))
    }

    #[inline(always)]
    fn bitwise_or(self, rhs: u32) -> Self {
        ScalarInt64(self.0 | (rhs as u64))
    }

    #[inline(always)]
    fn sub_scalar(self, rhs: u32) -> Self {
        ScalarInt64(self.0.wrapping_sub(rhs as u64))
    }

    #[inline(always)]
    fn to_f32(self) -> Self::FloatVec {
        ScalarVector(self.0 as f64)
    }
}

// Implement SimdVector for ScalarVector<f64>
impl SimdVector for ScalarVector<f64> {
    type Scalar = f64;
    type Mask = ScalarMask;
    type IntBits = ScalarInt64;

    const LANES: usize = 1;

    #[inline(always)]
    fn splat(value: Self::Scalar) -> Self {
        ScalarVector(value)
    }

    #[inline(always)]
    fn from_slice(slice: &[Self::Scalar]) -> Self {
        assert!(
            slice.len() >= Self::LANES,
            "Slice too short for scalar load"
        );
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

    #[inline(always)]
    fn to_bits(self) -> Self::IntBits {
        ScalarInt64(self.0.to_bits())
    }

    #[inline(always)]
    fn from_bits(bits: Self::IntBits) -> Self {
        ScalarVector(f64::from_bits(bits.0))
    }

    #[inline(always)]
    fn from_int_cast(int_vec: Self::IntBits) -> Self {
        ScalarVector(int_vec.0 as f64)
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

    // Tests for libm-optimized math functions
    #[test]
    fn test_libm_exp() {
        let x = ScalarVector(1.0f32);
        let result = x.exp_libm();
        let expected = core::f32::consts::E;
        assert!((result.0 - expected).abs() < 1e-6);
    }

    #[test]
    fn test_libm_log() {
        let x = ScalarVector(core::f32::consts::E);
        let result = x.log_libm();
        assert!((result.0 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_libm_log2() {
        let x = ScalarVector(8.0f32);
        let result = x.log2_libm();
        assert!((result.0 - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_libm_log10() {
        let x = ScalarVector(100.0f32);
        let result = x.log10_libm();
        assert!((result.0 - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_libm_trig() {
        let x = ScalarVector(core::f32::consts::FRAC_PI_4);
        let sin_result = x.sin_libm();
        let cos_result = x.cos_libm();

        // sin(π/4) ≈ cos(π/4) ≈ √2/2
        let expected = 0.7071067811865476f32;
        assert!((sin_result.0 - expected).abs() < 1e-6);
        assert!((cos_result.0 - expected).abs() < 1e-6);
    }

    #[test]
    fn test_libm_sincos() {
        let x = ScalarVector(core::f32::consts::FRAC_PI_4);
        let (sin_result, cos_result) = x.sincos_libm();

        let expected = 0.7071067811865476f32;
        assert!((sin_result.0 - expected).abs() < 1e-6);
        assert!((cos_result.0 - expected).abs() < 1e-6);
    }

    #[test]
    fn test_libm_atan() {
        let x = ScalarVector(1.0f32);
        let result = x.atan_libm();
        let expected = core::f32::consts::FRAC_PI_4;
        assert!((result.0 - expected).abs() < 1e-6);
    }

    #[test]
    fn test_libm_atan2() {
        let y = ScalarVector(1.0f32);
        let x = ScalarVector(1.0f32);
        let result = y.atan2_libm(x);
        let expected = core::f32::consts::FRAC_PI_4;
        assert!((result.0 - expected).abs() < 1e-6);
    }

    #[test]
    fn test_libm_tanh() {
        let x = ScalarVector(0.0f32);
        let result = x.tanh_libm();
        assert!((result.0 - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_libm_pow() {
        let base = ScalarVector(2.0f32);
        let exp = ScalarVector(3.0f32);
        let result = base.pow_libm(exp);
        assert!((result.0 - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_libm_sqrt() {
        let x = ScalarVector(16.0f32);
        let result = x.sqrt_libm();
        assert!((result.0 - 4.0).abs() < 1e-6);
    }
}
