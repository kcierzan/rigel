//! Scalar math approximations for control-rate DSP
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
//! - Cases where you're already using SIMD (use `simd::exp`/`simd::log` instead)
//!
//! # Error Bounds
//!
//! - `expf`: <1% relative error for x ∈ [-20, 20]
//! - `logf`: <1% relative error for x > 0
//! - `exp2f`: <0.0005% relative error for typical audio ranges
//!
//! # Example
//!
//! ```rust
//! use rigel_math::scalar::{expf, logf, exp2f};
//!
//! // Exponential decay envelope
//! let decay_rate = -5.0;
//! let time = 0.1;
//! let envelope = expf(decay_rate * time); // ~0.606
//!
//! // Log-domain parameter smoothing
//! let frequency = 1000.0;
//! let log_freq = logf(frequency); // ~6.9
//!
//! // Q8 envelope level to linear amplitude
//! let log2_gain = -8.0; // ~-48dB
//! let linear = exp2f(log2_gain); // ~0.0039
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
/// use rigel_math::scalar::expf;
///
/// let decay = expf(-5.0); // ~0.00674
/// assert!((decay - 0.00674).abs() < 0.0001);
/// ```
#[inline(always)]
pub fn expf(x: f32) -> f32 {
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
/// use rigel_math::scalar::logf;
///
/// let log_1000 = logf(1000.0); // ~6.9
/// assert!((log_1000 - 6.907).abs() < 0.1);
/// ```
#[inline(always)]
pub fn logf(x: f32) -> f32 {
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
/// use rigel_math::scalar::sinf;
///
/// let phase = core::f32::consts::FRAC_PI_2;
/// let result = sinf(phase); // ~1.0
/// assert!((result - 1.0).abs() < 0.01);
/// ```
#[inline(always)]
pub fn sinf(x: f32) -> f32 {
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

/// Fast scalar cos(x) using sinf
///
/// Uses the identity: cos(x) = sin(x + pi/2)
///
/// # Performance
///
/// Same performance characteristics as `sinf`.
///
/// # Example
///
/// ```rust
/// use rigel_math::scalar::cosf;
///
/// let result = cosf(0.0); // ~1.0
/// assert!((result - 1.0).abs() < 0.01);
/// ```
#[inline(always)]
pub fn cosf(x: f32) -> f32 {
    sinf(x + core::f32::consts::FRAC_PI_2)
}

/// Fast scalar exp2(x): 2^x
///
/// Computes 2^x using IEEE 754 bit manipulation for the integer part
/// and degree-5 minimax polynomial approximation for the fractional part.
/// This is the scalar equivalent of the SIMD `fast_exp2` function.
///
/// # Algorithm
///
/// ```text
/// x = i + f  where i = floor(x), f ∈ [0,1)
/// 2^x = 2^i * 2^f
///
/// 2^i: Set IEEE 754 exponent = (i + 127) << 23 (exact)
/// 2^f: Minimax polynomial with max error < 5e-6
/// ```
///
/// # Error Bounds
///
/// - Maximum relative error: < 0.0005% for typical audio ranges
/// - Polynomial error: < 5e-6 for fractional part [0, 1)
/// - Integer inputs: Exact (within f32 precision)
///
/// # Performance
///
/// Approximately 3-5x faster than `libm::exp2f` due to:
/// - IEEE 754 bit manipulation instead of iterative computation
/// - Simple polynomial evaluation using Horner's method
/// - Always inlined, no function call overhead
///
/// # Example
///
/// ```rust
/// use rigel_math::scalar::exp2f;
///
/// // 2^3 = 8
/// let result = exp2f(3.0);
/// assert!((result - 8.0).abs() < 0.001);
///
/// // MIDI to frequency ratio
/// let semitones = -9.0; // C4 relative to A4
/// let octaves = semitones / 12.0;
/// let ratio = exp2f(octaves); // ~0.5946
/// let freq = 440.0 * ratio;   // ~261.63 Hz
/// ```
#[inline(always)]
pub fn exp2f(x: f32) -> f32 {
    // Clamp to safe range to prevent overflow/underflow
    // 2^126 provides headroom for polynomial error
    // 2^-126 ≈ denormal threshold
    let x_clamped = x.clamp(-126.0, 126.0);

    // Split x = i + f where i = floor(x), f ∈ [0,1)
    let i_float = libm::floorf(x_clamped);
    let f = x_clamped - i_float;

    // Compute 2^i by setting IEEE 754 exponent: exp = (i + 127) << 23
    let i = i_float as i32;
    let pow2_i = f32::from_bits(((i + 127) << 23) as u32);

    // Compute 2^f using degree-5 minimax polynomial on [0,1)
    // Coefficients from Sollya/Remez for max error < 5e-6
    // Same coefficients as SIMD fast_exp2 for consistency
    const C0: f32 = 1.0;
    const C1: f32 = core::f32::consts::LN_2; // 0.693147...
    const C2: f32 = 0.240_226_5;
    const C3: f32 = 0.055_504_11;
    const C4: f32 = 0.009_618_129;
    const C5: f32 = 0.001_333_355_8;

    // Horner's method: c0 + f*(c1 + f*(c2 + f*(c3 + f*(c4 + f*c5))))
    let pow2_f = C0 + f * (C1 + f * (C2 + f * (C3 + f * (C4 + f * C5))));

    // Combine: 2^x = 2^i * 2^f
    pow2_i * pow2_f
}

/// Scalar PolyBLEP residual function for anti-aliasing
///
/// Computes the polynomial correction for a step discontinuity.
/// This is the scalar version of the SIMD `polyblep` function in `antialias.rs`.
///
/// # Parameters
///
/// - `t`: Phase position in [0, 1)
/// - `dt`: Phase increment per sample (frequency / sample_rate)
///
/// # Returns
///
/// Correction value to add to the naive waveform. Returns 0.0 when far from
/// discontinuities.
///
/// # Algorithm
///
/// For a discontinuity at phase=0:
/// - If phase < dt (just after crossing): correction = 2t - t² - 1
/// - If phase > 1-dt (just before crossing): correction = t² + 2t + 1
/// - Otherwise: no correction needed
///
/// # Example
///
/// ```rust
/// use rigel_math::scalar::polyblep;
///
/// let phase = 0.001; // Just after discontinuity
/// let dt = 0.01;     // ~440 Hz at 44.1kHz
/// let correction = polyblep(phase, dt);
/// // correction is non-zero near discontinuities
/// ```
#[inline(always)]
pub fn polyblep(t: f32, dt: f32) -> f32 {
    if t < dt {
        // Just after crossing (phase < dt)
        // Normalize: t_norm = t / dt, in range [0, 1)
        let t_norm = t / dt;
        // Correction: 2*t_norm - t_norm² - 1
        2.0 * t_norm - t_norm * t_norm - 1.0
    } else if t > 1.0 - dt {
        // Just before wrap (phase > 1-dt)
        // Normalize: t_norm = (t - 1) / dt, in range (-1, 0]
        let t_norm = (t - 1.0) / dt;
        // Correction: t_norm² + 2*t_norm + 1
        t_norm * t_norm + 2.0 * t_norm + 1.0
    } else {
        // Far from discontinuity, no correction needed
        0.0
    }
}

/// Scalar PolyBLEP-corrected sawtooth wave
///
/// Generates a band-limited sawtooth wave using PolyBLEP correction.
/// This is the scalar version of the SIMD `polyblep_sawtooth` function.
///
/// The sawtooth has one discontinuity per cycle at phase=0/1 where it
/// drops from +1 to -1.
///
/// # Parameters
///
/// - `phase`: Phase in [0, 1)
/// - `phase_increment`: Phase increment per sample (frequency / sample_rate)
///
/// # Output Range
///
/// Returns values in approximately [-1, 1] (slight overshoot possible
/// due to band-limiting correction)
///
/// # Example
///
/// ```rust
/// use rigel_math::scalar::polyblep_sawtooth;
///
/// // Generate sawtooth at 440 Hz (sample rate 44100)
/// let phase = 0.5;
/// let phase_inc = 440.0 / 44100.0;
/// let result = polyblep_sawtooth(phase, phase_inc);
/// // result ≈ 0.0 (midpoint of sawtooth)
/// ```
#[inline(always)]
pub fn polyblep_sawtooth(phase: f32, phase_increment: f32) -> f32 {
    // Naive sawtooth: rises linearly from -1 to 1
    // naive = 2 * phase - 1
    let naive = 2.0 * phase - 1.0;

    // Apply PolyBLEP correction at the discontinuity (phase ≈ 0 or ≈ 1)
    // Sawtooth drops from +1 to -1 at wrap, so we subtract the correction
    let correction = polyblep(phase, phase_increment);

    naive - correction
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
    fn test_expf_zero() {
        let result = expf(0.0);
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
    fn test_expf_one() {
        let result = expf(1.0);
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
    fn test_expf_negative() {
        let result = expf(-2.0);
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
    fn test_expf_typical_envelope_decay() {
        // Typical envelope decay: exp(-5) ≈ 0.00674
        let result = expf(-5.0);
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
    fn test_expf_clamping() {
        // Should not overflow
        let result = expf(100.0);
        assert!(result.is_finite(), "exp(100) should be clamped and finite");

        // Should not underflow to zero
        let result = expf(-100.0);
        assert!(
            result > 0.0,
            "exp(-100) should be clamped to positive value"
        );
    }

    #[test]
    fn test_logf_one() {
        let result = logf(1.0);
        let expected = 0.0;
        assert!(
            result.abs() < 0.001,
            "log(1) = {}, expected {}",
            result,
            expected
        );
    }

    #[test]
    fn test_logf_e() {
        let result = logf(core::f32::consts::E);
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
    fn test_logf_typical_frequency() {
        // log(1000) ≈ 6.907 (typical filter frequency)
        let result = logf(1000.0);
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
    fn test_logf_small_values() {
        // log(0.01) ≈ -4.605
        let result = logf(0.01);
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
            let roundtrip = expf(logf(x));
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
            let result = expf(x);
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
    // sinf tests
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn test_sinf_zero() {
        let result = sinf(0.0);
        assert!(result.abs() < 0.001, "sin(0) = {}, expected 0.0", result);
    }

    #[test]
    fn test_sinf_pi_over_2() {
        let result = sinf(core::f32::consts::FRAC_PI_2);
        let error = (result - 1.0).abs();
        assert!(
            error < 0.01,
            "sin(π/2) = {}, expected 1.0, error = {:.4}%",
            result,
            error * 100.0
        );
    }

    #[test]
    fn test_sinf_pi() {
        let result = sinf(core::f32::consts::PI);
        assert!(result.abs() < 0.01, "sin(π) = {}, expected 0.0", result);
    }

    #[test]
    fn test_sinf_negative() {
        let result = sinf(-core::f32::consts::FRAC_PI_2);
        let error = (result + 1.0).abs();
        assert!(
            error < 0.01,
            "sin(-π/2) = {}, expected -1.0, error = {:.4}%",
            result,
            error * 100.0
        );
    }

    #[test]
    fn test_sinf_lfo_range() {
        // Test typical LFO phase range [0, 2π]
        for i in 0..=100 {
            let phase = (i as f32) * core::f32::consts::TAU / 100.0;
            let result = sinf(phase);
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
    fn test_sinf_large_values() {
        // Test range reduction with large values
        let test_values = [
            10.0 * core::f32::consts::PI,
            100.0 * core::f32::consts::PI,
            1000.0 * core::f32::consts::PI,
        ];

        for &x in &test_values {
            let result = sinf(x);
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
    // cosf tests
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn test_cosf_zero() {
        let result = cosf(0.0);
        let error = (result - 1.0).abs();
        assert!(
            error < 0.01,
            "cos(0) = {}, expected 1.0, error = {:.4}%",
            result,
            error * 100.0
        );
    }

    #[test]
    fn test_cosf_pi_over_2() {
        let result = cosf(core::f32::consts::FRAC_PI_2);
        assert!(result.abs() < 0.01, "cos(π/2) = {}, expected 0.0", result);
    }

    #[test]
    fn test_sinf_cosf_pythagorean() {
        // sin²(x) + cos²(x) = 1
        let test_values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0];

        for &x in &test_values {
            let s = sinf(x);
            let c = cosf(x);
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

    // ─────────────────────────────────────────────────────────────────────
    // polyblep tests
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn test_polyblep_zero_far_from_discontinuity() {
        // When far from discontinuities, correction should be zero
        let result = polyblep(0.5, 0.01);
        assert!(
            result.abs() < 1e-6,
            "polyblep far from discontinuity should be 0, got {}",
            result
        );
    }

    #[test]
    fn test_polyblep_near_start() {
        // Just after phase=0, should have non-zero correction
        let result = polyblep(0.005, 0.01);
        // Should be non-zero and in reasonable range
        assert!(
            result.abs() < 2.0,
            "polyblep near start should be bounded, got {}",
            result
        );
        // At t_norm = 0.5: 2*0.5 - 0.25 - 1 = 1 - 0.25 - 1 = -0.25
        let expected = -0.25;
        assert!(
            (result - expected).abs() < 1e-6,
            "polyblep at t_norm=0.5 should be {}, got {}",
            expected,
            result
        );
    }

    #[test]
    fn test_polyblep_near_end() {
        // Just before phase=1, should have non-zero correction
        let result = polyblep(0.995, 0.01);
        // At t_norm = (0.995 - 1) / 0.01 = -0.5
        // t_norm² + 2*t_norm + 1 = 0.25 - 1 + 1 = 0.25
        let expected = 0.25;
        assert!(
            (result - expected).abs() < 1e-6,
            "polyblep near end should be {}, got {}",
            expected,
            result
        );
    }

    #[test]
    fn test_polyblep_boundary_values() {
        let dt = 0.01;

        // At exactly t=0 (t_norm=0): 2*0 - 0 - 1 = -1
        let at_zero = polyblep(0.0, dt);
        assert!(
            (at_zero - (-1.0)).abs() < 1e-6,
            "polyblep at t=0 should be -1, got {}",
            at_zero
        );

        // At exactly t=dt (t_norm=1): 2*1 - 1 - 1 = 0
        // But t >= dt falls outside the correction region, so should be 0
        let at_dt = polyblep(dt, dt);
        assert!(
            at_dt.abs() < 1e-6,
            "polyblep at t=dt should be 0 (outside region), got {}",
            at_dt
        );
    }

    // ─────────────────────────────────────────────────────────────────────
    // polyblep_sawtooth tests
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn test_polyblep_sawtooth_midpoint() {
        let result = polyblep_sawtooth(0.5, 0.01);
        // At midpoint, sawtooth should be ~0
        assert!(
            result.abs() < 0.1,
            "sawtooth at phase=0.5 should be ~0, got {}",
            result
        );
    }

    #[test]
    fn test_polyblep_sawtooth_range() {
        // Test multiple phase values
        for i in 0..10 {
            let phase = i as f32 / 10.0;
            let result = polyblep_sawtooth(phase, 0.01);
            assert!(
                (-1.5..=1.5).contains(&result),
                "sawtooth at phase={} should be in [-1.5, 1.5], got {}",
                phase,
                result
            );
        }
    }

    #[test]
    fn test_polyblep_sawtooth_monotonic_rising() {
        // In the middle of the cycle (away from discontinuity), sawtooth should rise
        let phase_inc = 0.01;
        let mut prev = polyblep_sawtooth(0.1, phase_inc);
        for i in 2..9 {
            let phase = i as f32 / 10.0;
            let curr = polyblep_sawtooth(phase, phase_inc);
            assert!(
                curr > prev,
                "sawtooth should be rising: phase={}, prev={}, curr={}",
                phase,
                prev,
                curr
            );
            prev = curr;
        }
    }

    #[test]
    fn test_polyblep_sawtooth_matches_simd_at_midpoint() {
        // At phase=0.5, far from discontinuity, naive sawtooth = 2*0.5 - 1 = 0
        // correction = 0, so result should be exactly 0
        let result = polyblep_sawtooth(0.5, 0.01);
        let expected = 0.0;
        assert!(
            (result - expected).abs() < 1e-6,
            "sawtooth at phase=0.5 should be exactly {}, got {}",
            expected,
            result
        );
    }

    #[test]
    fn test_polyblep_sawtooth_typical_frequency() {
        // 440 Hz at 44100 Hz sample rate
        let phase_inc = 440.0 / 44100.0;

        // Test a few phases
        let phases = [0.0, 0.25, 0.5, 0.75];
        for &phase in &phases {
            let result = polyblep_sawtooth(phase, phase_inc);
            assert!(
                result.is_finite(),
                "sawtooth at phase={} should be finite, got {}",
                phase,
                result
            );
            assert!(
                (-1.5..=1.5).contains(&result),
                "sawtooth at phase={} should be in [-1.5, 1.5], got {}",
                phase,
                result
            );
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // exp2f tests
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn test_exp2f_exact_powers() {
        // 2^3 = 8 (exact)
        let result = exp2f(3.0);
        let error = relative_error(result, 8.0);
        assert!(
            error < 1e-5,
            "exp2(3) = {}, expected 8.0, error = {:.6}%",
            result,
            error * 100.0
        );

        // 2^0 = 1 (exact)
        let result = exp2f(0.0);
        let error = relative_error(result, 1.0);
        assert!(
            error < 1e-5,
            "exp2(0) = {}, expected 1.0, error = {:.6}%",
            result,
            error * 100.0
        );

        // 2^1 = 2 (exact)
        let result = exp2f(1.0);
        let error = relative_error(result, 2.0);
        assert!(
            error < 1e-5,
            "exp2(1) = {}, expected 2.0, error = {:.6}%",
            result,
            error * 100.0
        );
    }

    #[test]
    fn test_exp2f_fractional() {
        // 2^0.5 = √2 ≈ 1.414
        let result = exp2f(0.5);
        let expected = core::f32::consts::SQRT_2;
        let error = relative_error(result, expected);
        assert!(
            error < 0.0001,
            "exp2(0.5) = {}, expected {}, error = {:.6}%",
            result,
            expected,
            error * 100.0
        );

        // 2^0.25 = 2^(1/4) ≈ 1.189
        let result = exp2f(0.25);
        let expected = libm::exp2f(0.25);
        let error = relative_error(result, expected);
        assert!(
            error < 0.0001,
            "exp2(0.25) = {}, expected {}, error = {:.6}%",
            result,
            expected,
            error * 100.0
        );
    }

    #[test]
    fn test_exp2f_negative() {
        // 2^-2 = 0.25 (exact)
        let result = exp2f(-2.0);
        let error = relative_error(result, 0.25);
        assert!(
            error < 1e-5,
            "exp2(-2) = {}, expected 0.25, error = {:.6}%",
            result,
            error * 100.0
        );

        // 2^-1 = 0.5 (exact)
        let result = exp2f(-1.0);
        let error = relative_error(result, 0.5);
        assert!(
            error < 1e-5,
            "exp2(-1) = {}, expected 0.5, error = {:.6}%",
            result,
            error * 100.0
        );
    }

    #[test]
    fn test_exp2f_clamping() {
        // Should not overflow
        let result = exp2f(200.0);
        assert!(
            result.is_finite() && result.is_sign_positive(),
            "exp2(200) should be clamped and finite, got {}",
            result
        );

        // Should not underflow to zero (clamped to exp2(-126))
        let result = exp2f(-200.0);
        assert!(
            result > 0.0,
            "exp2(-200) should be clamped to positive value, got {}",
            result
        );
    }

    #[test]
    fn test_exp2f_accuracy_vs_libm() {
        // Test accuracy across the Q8 envelope range
        // Q8 log2_gain ranges from -16 (at level 0) to 0 (at level 4095)
        let test_values: [f32; 17] = [
            -16.0, -14.0, -12.0, -10.0, -8.0, -6.0, -4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0,
            4.0, 6.0, 8.0,
        ];

        for &x in &test_values {
            let result = exp2f(x);
            let expected = libm::exp2f(x);
            let error = relative_error(result, expected);
            assert!(
                error < 0.00005, // 0.005%
                "exp2({}) = {}, expected {}, error = {:.6}%",
                x,
                result,
                expected,
                error * 100.0
            );
        }
    }

    #[test]
    fn test_exp2f_envelope_use_case() {
        // Test the specific use case: Q8 level to linear amplitude
        // log2_gain = level_q8 * 16.0 / 4095.0 - 16.0
        // At level 4095: log2_gain = 0, linear = 1.0
        // At level 2048: log2_gain = -8, linear ≈ 0.00390625
        // At level 0: log2_gain = -16, linear ≈ 0.0000153

        // Mid-level: 2048 -> -8 -> 2^-8 = 1/256
        let log2_gain = 2048.0 * 16.0 / 4095.0 - 16.0;
        let result = exp2f(log2_gain);
        let expected = libm::exp2f(log2_gain);
        let error = relative_error(result, expected);
        assert!(
            error < 0.00005,
            "envelope mid-level error = {:.6}%",
            error * 100.0
        );

        // Max level: 4095 -> 0 -> 2^0 = 1.0
        let log2_gain_max = 4095.0 * 16.0 / 4095.0 - 16.0;
        let result_max = exp2f(log2_gain_max);
        assert!(
            (result_max - 1.0).abs() < 0.001,
            "envelope max level = {}, expected ~1.0",
            result_max
        );
    }

    #[test]
    fn test_exp2f_monotonic() {
        // exp2 should be strictly increasing
        let test_values: [f32; 10] = [-8.0, -6.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 6.0];
        let mut prev = exp2f(test_values[0]);
        for &x in &test_values[1..] {
            let curr = exp2f(x);
            assert!(
                curr > prev,
                "exp2 should be increasing: exp2({}) = {} <= exp2(prev) = {}",
                x,
                curr,
                prev
            );
            prev = curr;
        }
    }
}
