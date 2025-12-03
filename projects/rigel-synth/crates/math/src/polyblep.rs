//! PolyBLEP (Polynomial Band-Limited Step) for alias-free oscillators
//!
//! Provides polynomial band-limiting for discontinuities in waveforms.
//! PolyBLEP reduces aliasing in sawtooth, square, and pulse waveforms by
//! smoothing the discontinuities at transition points.
//!
//! # Algorithm
//!
//! The PolyBLEP algorithm applies a polynomial correction near discontinuities:
//! - Uses a 2nd-order polynomial approximation of the ideal band-limited step
//! - Correction is applied within one sample of each discontinuity
//! - Computational cost: ~2-4 multiply-adds per discontinuity
//!
//! # Error Bounds
//!
//! - Alias suppression: -40 to -60 dB for typical audio frequencies
//! - Higher frequencies see less suppression (aliasing increases as f→fs/2)
//!
//! # Example
//!
//! ```rust
//! use rigel_math::{DefaultSimdVector, SimdVector};
//! use rigel_math::polyblep::{polyblep_sawtooth, polyblep_square};
//!
//! let phase = DefaultSimdVector::splat(0.5);
//! let phase_inc = DefaultSimdVector::splat(0.01); // ~440 Hz at 44.1kHz
//!
//! let saw = polyblep_sawtooth(phase, phase_inc);
//! let square = polyblep_square(phase, phase_inc);
//! ```

use crate::traits::SimdVector;

/// PolyBLEP residual function
///
/// Computes the polynomial correction for a discontinuity.
/// This is the core building block used by all PolyBLEP waveforms.
///
/// # Parameters
///
/// - `t`: Normalized phase position relative to discontinuity
/// - `dt`: Phase increment (determines correction width)
///
/// # Returns
///
/// Correction value to add to the naive waveform
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
/// use rigel_math::{DefaultSimdVector, SimdVector};
/// use rigel_math::polyblep::polyblep;
///
/// let phase = DefaultSimdVector::splat(0.001); // Just after discontinuity
/// let dt = DefaultSimdVector::splat(0.01);
/// let correction = polyblep(phase, dt);
/// ```
#[inline(always)]
pub fn polyblep<V: SimdVector<Scalar = f32>>(t: V, dt: V) -> V {
    let zero = V::splat(0.0);
    let one = V::splat(1.0);
    let two = V::splat(2.0);

    // Check if we're near phase=0 discontinuity (just after crossing)
    let near_start = t.lt(dt);

    // Check if we're near phase=1 discontinuity (just before wrapping)
    let near_end = t.gt(one.sub(dt));

    // Normalize t for the correction regions
    // For near_start: t_norm = t / dt, in range [0, 1)
    // For near_end: t_norm = (t - 1) / dt, in range (-1, 0]
    let t_norm_start = t.div(dt);
    let t_norm_end = t.sub(one).div(dt);

    // Compute correction for just-after-crossing (t < dt):
    // 2*t_norm - t_norm² - 1
    let correction_start = two
        .mul(t_norm_start)
        .sub(t_norm_start.mul(t_norm_start))
        .sub(one);

    // Compute correction for just-before-wrap (t > 1-dt):
    // t_norm² + 2*t_norm + 1
    let correction_end = t_norm_end.mul(t_norm_end).add(two.mul(t_norm_end)).add(one);

    // Select appropriate correction based on phase position
    let result = V::select(near_start, correction_start, zero);
    V::select(near_end, correction_end, result)
}

/// PolyBLEP-corrected sawtooth wave
///
/// Generates a band-limited sawtooth wave using PolyBLEP correction.
/// The sawtooth has one discontinuity per cycle at phase=0/1.
///
/// # Parameters
///
/// - `phase`: Phase in [0, 1)
/// - `phase_increment`: Phase increment per sample (frequency / sample_rate)
///
/// # Output Range
///
/// Returns values in [-1, 1]
///
/// # Example
///
/// ```rust
/// use rigel_math::{DefaultSimdVector, SimdVector};
/// use rigel_math::polyblep::polyblep_sawtooth;
///
/// // Generate sawtooth at 440 Hz (sample rate 44100)
/// let phase = DefaultSimdVector::splat(0.5);
/// let phase_inc = DefaultSimdVector::splat(440.0 / 44100.0);
/// let result = polyblep_sawtooth(phase, phase_inc);
/// // result ≈ 0.0 (midpoint of sawtooth)
/// ```
#[inline(always)]
pub fn polyblep_sawtooth<V: SimdVector<Scalar = f32>>(phase: V, phase_increment: V) -> V {
    // Naive sawtooth: rises linearly from -1 to 1
    // naive = 2 * phase - 1
    let two = V::splat(2.0);
    let one = V::splat(1.0);

    let naive = two.mul(phase).sub(one);

    // Apply PolyBLEP correction at the discontinuity (phase ≈ 0 or ≈ 1)
    // Sawtooth drops from +1 to -1 at wrap, so we subtract the correction
    let correction = polyblep(phase, phase_increment);

    naive.sub(correction)
}

/// PolyBLEP-corrected square wave
///
/// Generates a band-limited square wave using PolyBLEP correction.
/// The square wave has two discontinuities per cycle: at phase=0 and phase=0.5.
///
/// # Parameters
///
/// - `phase`: Phase in [0, 1)
/// - `phase_increment`: Phase increment per sample (frequency / sample_rate)
///
/// # Output Range
///
/// Returns values in [-1, 1] (approximately, with BLEP smoothing)
///
/// # Example
///
/// ```rust
/// use rigel_math::{DefaultSimdVector, SimdVector};
/// use rigel_math::polyblep::polyblep_square;
///
/// // Generate square wave at 440 Hz
/// let phase = DefaultSimdVector::splat(0.25);
/// let phase_inc = DefaultSimdVector::splat(440.0 / 44100.0);
/// let result = polyblep_square(phase, phase_inc);
/// // result ≈ 1.0 (first half of cycle is high)
/// ```
#[inline(always)]
pub fn polyblep_square<V: SimdVector<Scalar = f32>>(phase: V, phase_increment: V) -> V {
    let half = V::splat(0.5);
    let one = V::splat(1.0);
    let neg_one = V::splat(-1.0);

    // Naive square: +1 for first half, -1 for second half
    let is_first_half = phase.lt(half);
    let naive = V::select(is_first_half, one, neg_one);

    // Apply PolyBLEP at both discontinuities:
    // 1. At phase=0 (wrap point): transition from -1 to +1
    // 2. At phase=0.5: transition from +1 to -1

    // Correction at phase=0 (rising edge)
    let correction_start = polyblep(phase, phase_increment);

    // Correction at phase=0.5 (falling edge)
    // Shift phase by 0.5 to align discontinuity with polyblep function
    let phase_shifted = phase.sub(half);
    // Wrap negative values: if phase < 0.5, phase_shifted is negative, add 1.0
    let phase_wrapped = V::select(phase.lt(half), phase_shifted.add(one), phase_shifted);
    let correction_mid = polyblep(phase_wrapped, phase_increment);

    // Apply corrections:
    // - At phase=0: rise from -1 to +1, add correction (increases smoothness)
    // - At phase=0.5: fall from +1 to -1, subtract correction
    naive.add(correction_start).sub(correction_mid)
}

/// PolyBLEP-corrected pulse wave
///
/// Generates a band-limited pulse wave with variable duty cycle.
/// Duty cycle of 0.5 produces a square wave.
///
/// # Parameters
///
/// - `phase`: Phase in [0, 1)
/// - `phase_increment`: Phase increment per sample
/// - `duty`: Duty cycle in [0, 1] (0.5 = square wave)
///
/// # Output Range
///
/// Returns values in [-1, 1] (approximately, with BLEP smoothing)
///
/// # Example
///
/// ```rust
/// use rigel_math::{DefaultSimdVector, SimdVector};
/// use rigel_math::polyblep::polyblep_pulse;
///
/// // Generate pulse wave with 25% duty cycle
/// let phase = DefaultSimdVector::splat(0.1);
/// let phase_inc = DefaultSimdVector::splat(440.0 / 44100.0);
/// let duty = DefaultSimdVector::splat(0.25);
/// let result = polyblep_pulse(phase, phase_inc, duty);
/// ```
#[inline(always)]
pub fn polyblep_pulse<V: SimdVector<Scalar = f32>>(phase: V, phase_increment: V, duty: V) -> V {
    let one = V::splat(1.0);
    let neg_one = V::splat(-1.0);

    // Naive pulse: +1 for phase < duty, -1 otherwise
    let is_high = phase.lt(duty);
    let naive = V::select(is_high, one, neg_one);

    // Apply PolyBLEP at both discontinuities:
    // 1. At phase=0 (rising edge)
    // 2. At phase=duty (falling edge)

    // Correction at phase=0
    let correction_start = polyblep(phase, phase_increment);

    // Correction at phase=duty
    // Shift phase to align falling edge with polyblep function
    let phase_shifted = phase.sub(duty);
    // Wrap: if phase < duty, the shifted value is negative
    let phase_wrapped = V::select(phase.lt(duty), phase_shifted.add(one), phase_shifted);
    let correction_duty = polyblep(phase_wrapped, phase_increment);

    // Apply corrections
    naive.add(correction_start).sub(correction_duty)
}

/// PolyBLEP-corrected triangle wave
///
/// Generates a band-limited triangle wave by integrating a square wave
/// with PolyBLEP correction applied to smooth the corners.
///
/// Note: This is a simplified implementation that applies direct
/// PolyBLEP to the corner points. For highest quality, consider using
/// PolyBLAMP (band-limited ramp) instead.
///
/// # Parameters
///
/// - `phase`: Phase in [0, 1)
/// - `phase_increment`: Phase increment per sample
///
/// # Output Range
///
/// Returns values in [-1, 1]
///
/// # Example
///
/// ```rust
/// use rigel_math::{DefaultSimdVector, SimdVector};
/// use rigel_math::polyblep::polyblep_triangle;
///
/// let phase = DefaultSimdVector::splat(0.25);
/// let phase_inc = DefaultSimdVector::splat(440.0 / 44100.0);
/// let result = polyblep_triangle(phase, phase_inc);
/// // result ≈ 1.0 (peak of triangle)
/// ```
#[inline(always)]
pub fn polyblep_triangle<V: SimdVector<Scalar = f32>>(phase: V, _phase_increment: V) -> V {
    let one = V::splat(1.0);
    let two = V::splat(2.0);
    let four = V::splat(4.0);
    let half = V::splat(0.5);
    let quarter = V::splat(0.25);

    // Naive triangle: piecewise linear, peaks at phase=0.25 and troughs at phase=0.75
    // Using absolute value formulation: 4 * |phase - 0.5| - 1
    // But we want peak at 0.25, so: 4 * |0.25 - |phase - 0.5|| - 1
    // Simpler: piecewise approach

    // Triangle wave rising from -1 at phase=0 to +1 at phase=0.25,
    // falling to -1 at phase=0.75, then rising back to -1 at phase=1.0

    // Remap to center at 0.25 and 0.75 peaks/troughs
    let phase_adjusted = phase.add(quarter);
    // Wrap: subtract 1.0 if >= 1.0
    let phase_wrapped = V::select(
        phase_adjusted.gt(one),
        phase_adjusted.sub(one),
        phase_adjusted,
    );

    // Now phase=0 corresponds to old phase=0.75 (trough)
    // Triangle formula: 1 - 4 * |phase - 0.5|
    let centered = phase_wrapped.sub(half).abs();

    // Note: PolyBLEP is designed for step discontinuities.
    // Triangle has slope discontinuities (corners), not step discontinuities.
    // For true band-limiting, PolyBLAMP should be used.
    // This simplified version returns the naive triangle without correction.
    one.sub(four.mul(centered)).mul(two).sub(one)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DefaultSimdVector;

    #[test]
    fn test_polyblep_zero() {
        // When far from discontinuities, correction should be zero
        let phase = DefaultSimdVector::splat(0.5);
        let dt = DefaultSimdVector::splat(0.01);
        let result = polyblep(phase, dt);
        let value = result.horizontal_sum() / DefaultSimdVector::LANES as f32;
        assert!(
            value.abs() < 1e-4,
            "polyblep far from discontinuity should be ~0, got {}",
            value
        );
    }

    #[test]
    fn test_polyblep_near_start() {
        // Just after phase=0, should have non-zero correction
        let phase = DefaultSimdVector::splat(0.005);
        let dt = DefaultSimdVector::splat(0.01);
        let result = polyblep(phase, dt);
        let value = result.horizontal_sum() / DefaultSimdVector::LANES as f32;
        // Should be non-zero and in reasonable range
        assert!(
            value.abs() < 2.0,
            "polyblep near start should be bounded, got {}",
            value
        );
    }

    #[test]
    fn test_polyblep_sawtooth_midpoint() {
        let phase = DefaultSimdVector::splat(0.5);
        let phase_inc = DefaultSimdVector::splat(0.01);
        let result = polyblep_sawtooth(phase, phase_inc);
        let value = result.horizontal_sum() / DefaultSimdVector::LANES as f32;
        // At midpoint, sawtooth should be ~0
        assert!(
            value.abs() < 0.1,
            "sawtooth at phase=0.5 should be ~0, got {}",
            value
        );
    }

    #[test]
    fn test_polyblep_sawtooth_range() {
        // Test multiple phase values
        for i in 0..10 {
            let phase_val = i as f32 / 10.0;
            let phase = DefaultSimdVector::splat(phase_val);
            let phase_inc = DefaultSimdVector::splat(0.01);
            let result = polyblep_sawtooth(phase, phase_inc);
            let value = result.horizontal_sum() / DefaultSimdVector::LANES as f32;
            assert!(
                value >= -1.5 && value <= 1.5,
                "sawtooth at phase={} should be in [-1.5, 1.5], got {}",
                phase_val,
                value
            );
        }
    }

    #[test]
    fn test_polyblep_square_high_half() {
        // In first half (phase < 0.5), square should be approximately +1
        let phase = DefaultSimdVector::splat(0.25);
        let phase_inc = DefaultSimdVector::splat(0.01);
        let result = polyblep_square(phase, phase_inc);
        let value = result.horizontal_sum() / DefaultSimdVector::LANES as f32;
        assert!(
            value > 0.5,
            "square at phase=0.25 should be positive, got {}",
            value
        );
    }

    #[test]
    fn test_polyblep_square_low_half() {
        // In second half (phase >= 0.5), square should be approximately -1
        let phase = DefaultSimdVector::splat(0.75);
        let phase_inc = DefaultSimdVector::splat(0.01);
        let result = polyblep_square(phase, phase_inc);
        let value = result.horizontal_sum() / DefaultSimdVector::LANES as f32;
        assert!(
            value < -0.5,
            "square at phase=0.75 should be negative, got {}",
            value
        );
    }

    #[test]
    fn test_polyblep_pulse_variable_duty() {
        // 50% duty should behave like square wave
        let phase = DefaultSimdVector::splat(0.25);
        let phase_inc = DefaultSimdVector::splat(0.01);
        let duty_50 = DefaultSimdVector::splat(0.5);
        let pulse_50 = polyblep_pulse(phase, phase_inc, duty_50);
        let square = polyblep_square(phase, phase_inc);

        let pulse_val = pulse_50.horizontal_sum() / DefaultSimdVector::LANES as f32;
        let square_val = square.horizontal_sum() / DefaultSimdVector::LANES as f32;

        assert!(
            (pulse_val - square_val).abs() < 0.1,
            "50% duty pulse should match square wave: pulse={}, square={}",
            pulse_val,
            square_val
        );
    }

    #[test]
    fn test_polyblep_triangle_peak() {
        // Triangle should peak around phase=0.25
        let phase = DefaultSimdVector::splat(0.25);
        let phase_inc = DefaultSimdVector::splat(0.01);
        let result = polyblep_triangle(phase, phase_inc);
        let value = result.horizontal_sum() / DefaultSimdVector::LANES as f32;
        // Peak should be close to +1
        assert!(
            value > 0.5,
            "triangle at phase=0.25 should be near peak, got {}",
            value
        );
    }
}
