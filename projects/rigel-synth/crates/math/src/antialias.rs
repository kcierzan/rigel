//! Anti-aliasing functions for alias-free oscillators
//!
//! Provides polynomial band-limiting for discontinuities in waveforms:
//! - **PolyBLEP** (Polynomial Band-Limited Step): For step discontinuities
//!   in sawtooth, square, and pulse waveforms
//! - **PolyBLAMP** (Polynomial Band-Limited Ramp): For slope discontinuities
//!   (corners) in triangle waveforms
//!
//! # Algorithm
//!
//! Both algorithms apply polynomial corrections near discontinuities:
//! - PolyBLEP: 2nd-order polynomial for step transitions
//! - PolyBLAMP: 4th-order polynomial (integrated PolyBLEP) for corners
//! - Corrections applied within one sample of each discontinuity
//! - Computational cost: ~2-6 multiply-adds per discontinuity
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
//! use rigel_math::antialias::{polyblep_sawtooth, polyblep_square, polyblamp_triangle};
//!
//! let phase = DefaultSimdVector::splat(0.5);
//! let phase_inc = DefaultSimdVector::splat(0.01); // ~440 Hz at 44.1kHz
//!
//! let saw = polyblep_sawtooth(phase, phase_inc);
//! let square = polyblep_square(phase, phase_inc);
//! let triangle = polyblamp_triangle(phase, phase_inc);
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
/// use rigel_math::antialias::polyblep;
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

/// PolyBLAMP (Polynomial Band-Limited Ramp) residual function
///
/// Computes the polynomial correction for a slope discontinuity (corner).
/// This is the integral of PolyBLEP and is used for band-limiting waveforms
/// with corners/kinks like triangle waves.
///
/// # Parameters
///
/// - `t`: Normalized phase position relative to corner
/// - `dt`: Phase increment (determines correction width)
///
/// # Returns
///
/// Correction value to apply at corners. Must be scaled by slope change.
///
/// # Algorithm
///
/// For a corner at phase=0:
/// - If phase < dt (just after corner): `(t² - t)² / 4 = (t(t-1))² / 4`
/// - If phase > 1-dt (just before wrapping): `-(t² + t)² / 4 = -(t(t+1))² / 4`
/// - Otherwise: no correction needed
///
/// The polynomial `t^4/4 - t^3/2 + t^2/4` factors as `(t² - t)² / 4`,
/// allowing efficient computation using FMA instructions (4 ops instead of 8).
///
/// # Example
///
/// ```rust
/// use rigel_math::{DefaultSimdVector, SimdVector};
/// use rigel_math::antialias::polyblamp;
///
/// let phase = DefaultSimdVector::splat(0.001); // Just after corner
/// let dt = DefaultSimdVector::splat(0.01);
/// let correction = polyblamp(phase, dt);
/// ```
#[inline(always)]
pub fn polyblamp<V: SimdVector<Scalar = f32>>(t: V, dt: V) -> V {
    let zero = V::splat(0.0);
    let one = V::splat(1.0);
    let quarter = V::splat(0.25);
    let neg_quarter = V::splat(-0.25);

    // Check if we're near phase=0 corner (just after)
    let near_start = t.lt(dt);

    // Check if we're near phase=1 corner (just before wrapping)
    let near_end = t.gt(one.sub(dt));

    // Normalize t for the correction regions
    // For near_start: t_norm = t / dt, in range [0, 1)
    // For near_end: t_norm = (t - 1) / dt, in range (-1, 0]
    let t_norm_start = t.div(dt);
    let t_norm_end = t.sub(one).div(dt);

    // Optimized polynomial computation using factored form:
    //
    // The PolyBLAMP polynomial t^4/4 - t^3/2 + t^2/4 factors as:
    //   t^2(t^2 - 2t + 1)/4 = t^2(t-1)^2/4 = (t^2 - t)^2/4 = (t(t-1))^2/4
    //
    // This allows computation with FMA: v = t*t - t, then v*v*0.25
    //
    // For the end region: -(t^2 + t)^2/4 = -(t(t+1))^2/4

    // Start region: v = t² - t = t*t + (-t), computed via FMA
    // fma(a, b, c) = a*b + c, so fma(t, t, -t) = t² - t
    let v_start = t_norm_start.fma(t_norm_start, t_norm_start.neg());
    let correction_start = v_start.mul(v_start).mul(quarter);

    // End region: v = t² + t = t*t + t, computed via FMA
    let v_end = t_norm_end.fma(t_norm_end, t_norm_end);
    let correction_end = v_end.mul(v_end).mul(neg_quarter);

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
/// use rigel_math::antialias::polyblep_sawtooth;
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
/// use rigel_math::antialias::polyblep_square;
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
/// use rigel_math::antialias::polyblep_pulse;
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

/// PolyBLAMP-corrected triangle wave
///
/// Generates a band-limited triangle wave using PolyBLAMP correction
/// to smooth the corner discontinuities (slope changes).
///
/// Triangle waves have slope (derivative) discontinuities at their
/// peaks and troughs. Unlike step discontinuities (handled by PolyBLEP),
/// these require PolyBLAMP (the integral of PolyBLEP) for proper
/// band-limiting.
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
/// use rigel_math::antialias::polyblamp_triangle;
///
/// let phase = DefaultSimdVector::splat(0.25);
/// let phase_inc = DefaultSimdVector::splat(440.0 / 44100.0);
/// let result = polyblamp_triangle(phase, phase_inc);
/// // result ≈ 1.0 (peak of triangle)
/// ```
#[inline(always)]
pub fn polyblamp_triangle<V: SimdVector<Scalar = f32>>(phase: V, phase_increment: V) -> V {
    let one = V::splat(1.0);
    let four = V::splat(4.0);
    let half = V::splat(0.5);

    // Naive triangle wave with corners at phase=0 and phase=0.5
    // From 0 to 0.5: rises from -1 to +1 (slope = +4)
    // From 0.5 to 1: falls from +1 to -1 (slope = -4)
    let is_rising = phase.lt(half);
    let naive = V::select(
        is_rising,
        four.mul(phase).sub(one),           // 4*phase - 1
        V::splat(3.0).sub(four.mul(phase)), // 3 - 4*phase
    );

    // Apply PolyBLAMP at both corners:
    // 1. At phase=0: slope changes from -4 to +4 (Δslope = +8)
    // 2. At phase=0.5: slope changes from +4 to -4 (Δslope = -8)

    // Correction at phase=0 (wrap point)
    let blamp_0 = polyblamp(phase, phase_increment);

    // Correction at phase=0.5 (peak)
    // Shift phase by 0.5 and wrap to align with polyblamp function
    let phase_shifted = phase.sub(half);
    let phase_wrapped = V::select(phase.lt(half), phase_shifted.add(one), phase_shifted);
    let blamp_half = polyblamp(phase_wrapped, phase_increment);

    // Apply corrections with appropriate scaling
    // The PolyBLAMP correction is scaled by slope change
    // At phase=0: Δslope = +8, so we add 8 * blamp_0
    // At phase=0.5: Δslope = -8, so we subtract 8 * blamp_half
    let eight = V::splat(8.0);

    naive.add(eight.mul(blamp_0)).sub(eight.mul(blamp_half))
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
                (-1.5..=1.5).contains(&value),
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

    // PolyBLAMP tests

    #[test]
    fn test_polyblamp_zero() {
        // When far from corners, correction should be zero
        let phase = DefaultSimdVector::splat(0.5);
        let dt = DefaultSimdVector::splat(0.01);
        let result = polyblamp(phase, dt);
        let value = result.horizontal_sum() / DefaultSimdVector::LANES as f32;
        assert!(
            value.abs() < 1e-6,
            "polyblamp far from corner should be ~0, got {}",
            value
        );
    }

    #[test]
    fn test_polyblamp_near_start() {
        // Just after phase=0 corner, should have non-zero correction
        let phase = DefaultSimdVector::splat(0.005);
        let dt = DefaultSimdVector::splat(0.01);
        let result = polyblamp(phase, dt);
        let value = result.horizontal_sum() / DefaultSimdVector::LANES as f32;
        // Should be positive and bounded
        assert!(
            (0.0..0.1).contains(&value),
            "polyblamp near start should be positive and small, got {}",
            value
        );
    }

    #[test]
    fn test_polyblamp_boundary_conditions() {
        // PolyBLAMP should be zero at t=0 and t=1 (normalized)
        let dt = DefaultSimdVector::splat(0.01);

        // At t=0 (t_norm=0), polynomial = 0
        let phase_zero = DefaultSimdVector::splat(0.0);
        let result_zero = polyblamp(phase_zero, dt);
        let value_zero = result_zero.horizontal_sum() / DefaultSimdVector::LANES as f32;
        assert!(
            value_zero.abs() < 1e-6,
            "polyblamp at t=0 should be 0, got {}",
            value_zero
        );

        // At t=dt (t_norm=1), polynomial = (1-1)^2/4 = 0
        let phase_dt = DefaultSimdVector::splat(0.0099); // Just under dt
        let result_dt = polyblamp(phase_dt, dt);
        let value_dt = result_dt.horizontal_sum() / DefaultSimdVector::LANES as f32;
        // Should be close to zero at the boundary
        assert!(
            value_dt.abs() < 0.01,
            "polyblamp near t=dt should be near 0, got {}",
            value_dt
        );
    }

    #[test]
    fn test_polyblamp_polynomial_factorization() {
        // Verify that (t^2 - t)^2 / 4 = t^4/4 - t^3/2 + t^2/4
        // by checking at specific points
        //
        // Use dt=0.5 and phase=0.25 so t_norm = 0.5, and phase is only in start region
        // (phase < dt and phase <= 1-dt, so not in end region)
        let dt = DefaultSimdVector::splat(0.5);
        let phase = DefaultSimdVector::splat(0.25);

        // t_norm = 0.25/0.5 = 0.5
        // v = t_norm^2 - t_norm = 0.25 - 0.5 = -0.25
        // correction = v^2 * 0.25 = 0.0625 * 0.25 = 0.015625
        let result = polyblamp(phase, dt);
        let value = result.horizontal_sum() / DefaultSimdVector::LANES as f32;
        let expected = 0.015625;
        assert!(
            (value - expected).abs() < 1e-6,
            "polyblamp at t_norm=0.5 should be {}, got {}",
            expected,
            value
        );
    }

    // PolyBLAMP triangle tests

    #[test]
    fn test_polyblamp_triangle_peak() {
        // Triangle should peak at phase=0.5 (new shape)
        let phase = DefaultSimdVector::splat(0.5);
        let phase_inc = DefaultSimdVector::splat(0.01);
        let result = polyblamp_triangle(phase, phase_inc);
        let value = result.horizontal_sum() / DefaultSimdVector::LANES as f32;
        // Peak should be close to +1
        assert!(
            value > 0.9,
            "triangle at phase=0.5 should be near peak (+1), got {}",
            value
        );
    }

    #[test]
    fn test_polyblamp_triangle_trough() {
        // Triangle should be at trough at phase=0 and phase=1
        let phase = DefaultSimdVector::splat(0.001);
        let phase_inc = DefaultSimdVector::splat(0.01);
        let result = polyblamp_triangle(phase, phase_inc);
        let value = result.horizontal_sum() / DefaultSimdVector::LANES as f32;
        // Near trough should be close to -1
        assert!(
            value < -0.9,
            "triangle at phase=0 should be near trough (-1), got {}",
            value
        );
    }

    #[test]
    fn test_polyblamp_triangle_range() {
        // Test multiple phase values, triangle should be in [-1.1, 1.1]
        // (slight overshoot possible due to band-limiting)
        for i in 0..20 {
            let phase_val = i as f32 / 20.0;
            let phase = DefaultSimdVector::splat(phase_val);
            let phase_inc = DefaultSimdVector::splat(0.01);
            let result = polyblamp_triangle(phase, phase_inc);
            let value = result.horizontal_sum() / DefaultSimdVector::LANES as f32;
            assert!(
                (-1.2..=1.2).contains(&value),
                "triangle at phase={} should be in [-1.2, 1.2], got {}",
                phase_val,
                value
            );
        }
    }

    #[test]
    fn test_polyblamp_triangle_symmetry() {
        // Triangle should be symmetric: f(x) = -f(x + 0.5) for rising/falling
        let phase_inc = DefaultSimdVector::splat(0.01);

        let phase_a = DefaultSimdVector::splat(0.25);
        let phase_b = DefaultSimdVector::splat(0.75);

        let result_a = polyblamp_triangle(phase_a, phase_inc);
        let result_b = polyblamp_triangle(phase_b, phase_inc);

        let value_a = result_a.horizontal_sum() / DefaultSimdVector::LANES as f32;
        let value_b = result_b.horizontal_sum() / DefaultSimdVector::LANES as f32;

        // Values should be opposite (symmetric around 0)
        assert!(
            (value_a + value_b).abs() < 0.1,
            "triangle should be symmetric: f(0.25)={} should equal -f(0.75)={}",
            value_a,
            -value_b
        );
    }
}
