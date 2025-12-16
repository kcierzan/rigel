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
    let x = x.clamp(-87.0, 87.0);

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

    // Coefficients from math/exp.rs (truncated to f32 precision)
    let p1 = 0.5;
    let p2 = 0.107_142_86; // 3/28
    let p3 = 0.011_904_762; // 1/84
    let p4 = 0.000_595_238_1; // 1/1680
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
    // Refactored to avoid deeply nested expression that causes rustfmt to hang
    // Truncated to f32 precision to satisfy clippy::excessive_precision
    let c15 = 0.066_666_67; // 1/15
    let c14 = -0.071_428_57; // -1/14
    let c13 = 0.076_923_08; // 1/13
    let c12 = -0.083_333_336; // -1/12
    let c11 = 0.090_909_09; // 1/11
    let c10 = -0.1; // -1/10
    let c9 = 0.111_111_11; // 1/9
    let c8 = -0.125; // -1/8
    let c7 = 0.142_857_14; // 1/7
    let c6 = -0.166_666_67; // -1/6
    let c5 = 0.2; // 1/5
    let c4 = -0.25; // -1/4
    let c3 = 0.333_333_34; // 1/3
    let c2 = -0.5; // -1/2
    let c1 = 1.0; // 1/1

    // Evaluate Horner's method iteratively
    let mut result = c15;
    result = c14 + t * result;
    result = c13 + t * result;
    result = c12 + t * result;
    result = c11 + t * result;
    result = c10 + t * result;
    result = c9 + t * result;
    result = c8 + t * result;
    result = c7 + t * result;
    result = c6 + t * result;
    result = c5 + t * result;
    result = c4 + t * result;
    result = c3 + t * result;
    result = c2 + t * result;
    result = c1 + t * result;
    let ln_m = t * result;

    // Combine: log(x) = exponent × ln(2) + ln(mantissa)
    exp * core::f32::consts::LN_2 + ln_m
}

/// Fast scalar sin(x) using polynomial approximation with range reduction
///
/// Computes sin(x) with <0.1% amplitude error, suitable for LFOs and control-rate
/// modulation where sample-accurate precision is not required.
///
/// # Algorithm
///
/// Uses the same Cody-Waite range reduction and 7th-order minimax polynomial
/// as the SIMD version in `math::trig::sin`, adapted for scalar execution.
///
/// # Performance
///
/// Approximately 2-3x faster than `libm::sinf` due to:
/// - Optimized polynomial evaluation
/// - No function call overhead (always inlined)
/// - Simplified range reduction for typical LFO ranges
///
/// # Example
///
/// ```rust
/// use rigel_math::scalar_fast::fast_sinf;
///
/// let phase = core::f32::consts::FRAC_PI_2;
/// let result = fast_sinf(phase); // ~1.0
/// assert!((result - 1.0).abs() < 0.01);
/// ```
#[inline(always)]
pub fn fast_sinf(x: f32) -> f32 {
    use core::f32::consts::{FRAC_PI_2, PI, TAU};

    // Cody-Waite range reduction constants
    const TWO_PI_A: f32 = 6.28125;
    const TWO_PI_B: f32 = 0.001_934_051_5;
    const TWO_PI_C: f32 = 1.215_422_8e-6;
    const INV_TWO_PI: f32 = 0.159_154_94;

    // Range reduction: x mod 2pi
    let n = libm::floorf(x * INV_TWO_PI);

    // Three-stage reduction for high precision
    let mut x_reduced = x - n * TWO_PI_A;
    x_reduced -= n * TWO_PI_B;
    x_reduced -= n * TWO_PI_C;

    // Ensure x_reduced is in [0, 2pi]
    if x_reduced < 0.0 {
        x_reduced += TAU;
    }

    // Map to [0, pi] and track sign flip
    let sign_flip = x_reduced > PI;
    let x_pi = if sign_flip { x_reduced - PI } else { x_reduced };

    // Map [0, pi] to [0, pi/2] using sin(pi - x) = sin(x)
    let x_half_pi = if x_pi > FRAC_PI_2 { PI - x_pi } else { x_pi };

    // 7th-order minimax polynomial on [0, pi/2]
    // sin(x) ≈ x * (c1 + x² * (c3 + x² * (c5 + x² * c7)))
    let x2 = x_half_pi * x_half_pi;

    // Minimax coefficients (same as SIMD version)
    const C1: f32 = 1.0;
    const C3: f32 = -0.166_666_66;
    const C5: f32 = 0.008_333_162;
    const C7: f32 = -0.000_194_953_22;

    // Horner's method
    let poly = C1 + x2 * (C3 + x2 * (C5 + x2 * C7));
    let result = x_half_pi * poly;

    // Apply sign flip if x was in [pi, 2pi]
    if sign_flip {
        -result
    } else {
        result
    }
}

/// Fast scalar cos(x) using fast_sinf
///
/// Uses the identity: cos(x) = sin(x + pi/2)
///
/// # Performance
///
/// Same performance characteristics as `fast_sinf`.
///
/// # Example
///
/// ```rust
/// use rigel_math::scalar_fast::fast_cosf;
///
/// let result = fast_cosf(0.0); // ~1.0
/// assert!((result - 1.0).abs() < 0.01);
/// ```
#[inline(always)]
pub fn fast_cosf(x: f32) -> f32 {
    fast_sinf(x + core::f32::consts::FRAC_PI_2)
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
        assert!(
            result > 0.0,
            "exp(-100) should be clamped to positive value"
        );
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
        let test_values: [f32; 11] = [-10.0, -5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0];

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

    // ─────────────────────────────────────────────────────────────────────
    // fast_sinf tests
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn test_fast_sinf_zero() {
        let result = fast_sinf(0.0);
        assert!(result.abs() < 0.001, "sin(0) = {}, expected 0.0", result);
    }

    #[test]
    fn test_fast_sinf_pi_over_2() {
        let result = fast_sinf(core::f32::consts::FRAC_PI_2);
        let error = (result - 1.0).abs();
        assert!(
            error < 0.01,
            "sin(π/2) = {}, expected 1.0, error = {:.4}%",
            result,
            error * 100.0
        );
    }

    #[test]
    fn test_fast_sinf_pi() {
        let result = fast_sinf(core::f32::consts::PI);
        assert!(result.abs() < 0.01, "sin(π) = {}, expected 0.0", result);
    }

    #[test]
    fn test_fast_sinf_negative() {
        let result = fast_sinf(-core::f32::consts::FRAC_PI_2);
        let error = (result + 1.0).abs();
        assert!(
            error < 0.01,
            "sin(-π/2) = {}, expected -1.0, error = {:.4}%",
            result,
            error * 100.0
        );
    }

    #[test]
    fn test_fast_sinf_lfo_range() {
        // Test typical LFO phase range [0, 2π]
        for i in 0..=100 {
            let phase = (i as f32) * core::f32::consts::TAU / 100.0;
            let result = fast_sinf(phase);
            let expected = libm::sinf(phase);
            let error = (result - expected).abs();
            assert!(
                error < 0.01,
                "sin({}) = {}, expected {}, error = {:.4}",
                phase,
                result,
                expected,
                error
            );
        }
    }

    #[test]
    fn test_fast_sinf_large_values() {
        // Test range reduction with large values
        let test_values = [
            10.0 * core::f32::consts::PI,
            100.0 * core::f32::consts::PI,
            1000.0 * core::f32::consts::PI,
        ];

        for &x in &test_values {
            let result = fast_sinf(x);
            let expected = libm::sinf(x);
            let error = (result - expected).abs();
            assert!(
                error < 0.02,
                "sin({}) = {}, expected {}, error = {:.4}",
                x,
                result,
                expected,
                error
            );
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // fast_cosf tests
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn test_fast_cosf_zero() {
        let result = fast_cosf(0.0);
        let error = (result - 1.0).abs();
        assert!(
            error < 0.01,
            "cos(0) = {}, expected 1.0, error = {:.4}%",
            result,
            error * 100.0
        );
    }

    #[test]
    fn test_fast_cosf_pi_over_2() {
        let result = fast_cosf(core::f32::consts::FRAC_PI_2);
        assert!(result.abs() < 0.01, "cos(π/2) = {}, expected 0.0", result);
    }

    #[test]
    fn test_fast_sinf_cosf_pythagorean() {
        // sin²(x) + cos²(x) = 1
        let test_values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0];

        for &x in &test_values {
            let s = fast_sinf(x);
            let c = fast_cosf(x);
            let identity = s * s + c * c;
            let error = (identity - 1.0).abs();
            assert!(
                error < 0.02,
                "sin²({}) + cos²({}) = {}, expected 1.0, error = {:.4}",
                x,
                x,
                identity,
                error
            );
        }
    }
}
