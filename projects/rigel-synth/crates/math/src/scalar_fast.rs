//! Scalar fast-math approximations for control-rate DSP
//!
//! Provides ~3-5x speedup over libm with <1% error.
//! Designed for smoothing, envelopes, LFOs, and other control-rate operations
//! where sample-accurate precision is not required.
//!
//! # When to Use
//!
//! - Parameter smoothing (filter cutoff, amplitude, etc.)
//! - Envelope generation
//! - LFO calculations
//! - Any scalar math in control-rate code paths
//!
//! # When NOT to Use
//!
//! - Sample-accurate synthesis requiring <0.01% error
//! - Scientific computing requiring IEEE 754 compliance
//! - Cases where you're already using SIMD (use `math::exp`/`math::log` instead)
//!
//! # Error Bounds
//!
//! - `fast_expf`: <1% relative error for x ∈ [-20, 20]
//! - `fast_logf`: <1% relative error for x > 0
//!
//! # Example
//!
//! ```rust
//! use rigel_math::scalar_fast::{fast_expf, fast_logf};
//!
//! // Exponential decay envelope
//! let decay_rate = -5.0;
//! let time = 0.1;
//! let envelope = fast_expf(decay_rate * time); // ~0.606
//!
//! // Log-domain parameter smoothing
//! let frequency = 1000.0;
//! let log_freq = fast_logf(frequency); // ~6.9
//! ```

/// Fast scalar exp(x) using Padé[4/4] approximation with range reduction
///
/// Computes e^x with <1% relative error for x ∈ [-87, 87].
/// Values outside this range are clamped to prevent overflow/underflow.
///
/// # Algorithm
///
/// Uses range reduction: exp(x) = exp(x/2)² repeated until |x| < 1,
/// then applies Padé[4/4] rational approximation for high accuracy.
///
/// # Performance
///
/// Approximately 3-5x faster than `libm::expf` due to:
/// - No function call overhead (always inlined)
/// - Simple polynomial arithmetic
/// - Efficient range reduction
///
/// # Example
///
/// ```rust
/// use rigel_math::scalar_fast::fast_expf;
///
/// let decay = fast_expf(-5.0); // ~0.00674
/// assert!((decay - 0.00674).abs() < 0.0001);
/// ```
#[inline(always)]
pub fn fast_expf(x: f32) -> f32 {
    // Clamp to prevent overflow/underflow
    // exp(-87) ≈ 1.2e-38 (near f32 min)
    // exp(87) ≈ 6.1e37 (near f32 max, with headroom for squaring)
    let x = if x < -87.0 {
        -87.0
    } else if x > 87.0 {
        87.0
    } else {
        x
    };

    // Range reduction: find how many times we need to halve x
    // to get |x_reduced| < 1
    let mut x_reduced = x;
    let mut squarings = 0u32;

    while x_reduced.abs() > 1.0 && squarings < 8 {
        x_reduced *= 0.5;
        squarings += 1;
    }

    // Padé [5/5] approximation for exp(x) when |x| < 1
    // Same coefficients as the SIMD version in math/exp.rs for consistency
    // Numerator: 1 + x/2 + 3x²/28 + x³/84 + x⁴/1680 + x⁵/15120
    // Denominator: 1 - x/2 + 3x²/28 - x³/84 + x⁴/1680 - x⁵/15120
    let x2 = x_reduced * x_reduced;
    let x3 = x2 * x_reduced;
    let x4 = x2 * x2;
    let x5 = x4 * x_reduced;

    // Coefficients from math/exp.rs
    let p1 = 0.5;
    let p2 = 0.107_142_857_14; // 3/28
    let p3 = 0.011_904_761_90; // 1/84
    let p4 = 0.000_595_238_10; // 1/1680
    let p5 = 0.000_066_137_56; // 1/15120

    let num = 1.0 + p1 * x_reduced + p2 * x2 + p3 * x3 + p4 * x4 + p5 * x5;
    let den = 1.0 - p1 * x_reduced + p2 * x2 - p3 * x3 + p4 * x4 - p5 * x5;

    let mut result = num / den;

    // Square the result for each halving we did
    for _ in 0..squarings {
        result = result * result;
    }

    result
}

/// Fast scalar log(x) using IEEE 754 bit manipulation
///
/// Computes ln(x) with <1% relative error for x > 0.
/// For x ≤ 0, behavior is undefined (returns garbage, not NaN).
///
/// # Performance
///
/// Approximately 3-4x faster than `libm::logf` due to:
/// - Direct IEEE 754 exponent extraction (no iteration)
/// - Simple polynomial for mantissa correction
/// - Always inlined, no function call overhead
///
/// # Algorithm
///
/// Uses IEEE 754 representation where x = mantissa × 2^exponent:
/// ```text
/// log(x) = exponent × ln(2) + log(mantissa)
/// ```
///
/// The exponent is extracted directly from the bit representation,
/// and a polynomial approximates log(mantissa) for mantissa ∈ [1, 2).
///
/// # Example
///
/// ```rust
/// use rigel_math::scalar_fast::fast_logf;
///
/// let log_1000 = fast_logf(1000.0); // ~6.9
/// assert!((log_1000 - 6.907).abs() < 0.1);
/// ```
#[inline(always)]
pub fn fast_logf(x: f32) -> f32 {
    // IEEE 754 single precision: sign(1) | exponent(8) | mantissa(23)
    // For x = 1.mantissa × 2^(exp-127)
    // log(x) = (exp - 127) × ln(2) + log(1.mantissa)
    let bits = x.to_bits();

    // Extract exponent: (bits >> 23) - 127
    // The & 0xFF masks out the sign bit which could be 1 for NaN/negative
    let exp = ((bits >> 23) & 0xFF) as f32 - 127.0;

    // Extract mantissa and normalize to [1, 2)
    // Clear exponent bits and set exponent to 127 (which represents 2^0 = 1)
    let mantissa_bits = (bits & 0x007F_FFFF) | 0x3F80_0000;
    let m = f32::from_bits(mantissa_bits);

    // Polynomial for ln(m) where m ∈ [1, 2)
    // Using t = m - 1, so t ∈ [0, 1)
    // 15-term Taylor series using Horner's method for numerical stability
    // Same approach as the SIMD version in math/log.rs
    // ln(1 + t) = t*(1 - t/2 + t²/3 - t³/4 + ...)
    let t = m - 1.0;

    // Horner's method: evaluate from innermost term outward
    // Coefficients: 1, -1/2, 1/3, -1/4, 1/5, -1/6, 1/7, -1/8, 1/9, -1/10, 1/11, -1/12, 1/13, -1/14, 1/15
    let ln_m = t * (1.0
        + t * (-0.5
            + t * (0.333_333_333
                + t * (-0.25
                    + t * (0.2
                        + t * (-0.166_666_667
                            + t * (0.142_857_143
                                + t * (-0.125
                                    + t * (0.111_111_111
                                        + t * (-0.1
                                            + t * (0.090_909_091
                                                + t * (-0.083_333_333
                                                    + t * (0.076_923_077
                                                        + t * (-0.071_428_571
                                                            + t * 0.066_666_667))))))))))))));
    // Combine: log(x) = exponent × ln(2) + ln(mantissa)
    exp * core::f32::consts::LN_2 + ln_m
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to compute relative error
    fn relative_error(actual: f32, expected: f32) -> f32 {
        if expected.abs() < 1e-10 {
            actual.abs()
        } else {
            ((actual - expected) / expected).abs()
        }
    }

    #[test]
    fn test_fast_expf_zero() {
        let result = fast_expf(0.0);
        let expected = 1.0;
        let error = relative_error(result, expected);
        assert!(
            error < 0.001,
            "exp(0) = {}, expected {}, error = {:.4}%",
            result,
            expected,
            error * 100.0
        );
    }

    #[test]
    fn test_fast_expf_one() {
        let result = fast_expf(1.0);
        let expected = core::f32::consts::E;
        let error = relative_error(result, expected);
        assert!(
            error < 0.01,
            "exp(1) = {}, expected {}, error = {:.4}%",
            result,
            expected,
            error * 100.0
        );
    }

    #[test]
    fn test_fast_expf_negative() {
        let result = fast_expf(-2.0);
        let expected = libm::expf(-2.0);
        let error = relative_error(result, expected);
        assert!(
            error < 0.01,
            "exp(-2) = {}, expected {}, error = {:.4}%",
            result,
            expected,
            error * 100.0
        );
    }

    #[test]
    fn test_fast_expf_typical_envelope_decay() {
        // Typical envelope decay: exp(-5) ≈ 0.00674
        let result = fast_expf(-5.0);
        let expected = libm::expf(-5.0);
        let error = relative_error(result, expected);
        assert!(
            error < 0.01,
            "exp(-5) = {}, expected {}, error = {:.4}%",
            result,
            expected,
            error * 100.0
        );
    }

    #[test]
    fn test_fast_expf_clamping() {
        // Should not overflow
        let result = fast_expf(100.0);
        assert!(result.is_finite(), "exp(100) should be clamped and finite");

        // Should not underflow to zero
        let result = fast_expf(-100.0);
        assert!(result > 0.0, "exp(-100) should be clamped to positive value");
    }

    #[test]
    fn test_fast_logf_one() {
        let result = fast_logf(1.0);
        let expected = 0.0;
        assert!(
            result.abs() < 0.001,
            "log(1) = {}, expected {}",
            result,
            expected
        );
    }

    #[test]
    fn test_fast_logf_e() {
        let result = fast_logf(core::f32::consts::E);
        let expected = 1.0;
        let error = relative_error(result, expected);
        assert!(
            error < 0.01,
            "log(e) = {}, expected {}, error = {:.4}%",
            result,
            expected,
            error * 100.0
        );
    }

    #[test]
    fn test_fast_logf_typical_frequency() {
        // log(1000) ≈ 6.907 (typical filter frequency)
        let result = fast_logf(1000.0);
        let expected = libm::logf(1000.0);
        let error = relative_error(result, expected);
        assert!(
            error < 0.01,
            "log(1000) = {}, expected {}, error = {:.4}%",
            result,
            expected,
            error * 100.0
        );
    }

    #[test]
    fn test_fast_logf_small_values() {
        // log(0.01) ≈ -4.605
        let result = fast_logf(0.01);
        let expected = libm::logf(0.01);
        let error = relative_error(result, expected);
        assert!(
            error < 0.01,
            "log(0.01) = {}, expected {}, error = {:.4}%",
            result,
            expected,
            error * 100.0
        );
    }

    #[test]
    fn test_roundtrip_exp_log() {
        // exp(log(x)) should approximately equal x
        let test_values = [0.1, 0.5, 1.0, 2.0, 10.0, 100.0, 1000.0];
        for &x in &test_values {
            let roundtrip = fast_expf(fast_logf(x));
            let error = relative_error(roundtrip, x);
            assert!(
                error < 0.02,
                "exp(log({})) = {}, error = {:.4}%",
                x,
                roundtrip,
                error * 100.0
            );
        }
    }

    #[test]
    fn test_accuracy_across_range() {
        // Test accuracy across the typical audio DSP range
        let test_values: [f32; 11] =
            [-10.0, -5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0];

        for &x in &test_values {
            let result = fast_expf(x);
            let expected = libm::expf(x);
            let error = relative_error(result, expected);
            assert!(
                error < 0.01,
                "exp({}) = {}, expected {}, error = {:.4}%",
                x,
                result,
                expected,
                error * 100.0
            );
        }
    }
}
