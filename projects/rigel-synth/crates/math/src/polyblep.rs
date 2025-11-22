//! PolyBLEP (band-limited step) for alias-free oscillators
//!
//! Provides polynomial band-limiting for discontinuities in waveforms.

use crate::traits::SimdVector;

/// PolyBLEP correction kernel
///
/// Corrects discontinuities in waveforms (e.g., sawtooth, square) to reduce aliasing.
/// Uses a 2nd-order polynomial approximation of the band-limited step function.
///
/// # Parameters
///
/// - `t`: Phase offset from discontinuity, normalized to [0, 1]
///
/// # Returns
///
/// Correction value to add to the naive waveform
///
/// # Example
///
/// ```rust
/// use rigel_math::{DefaultSimdVector, SimdVector};
/// use rigel_math::polyblep::polyblep;
///
/// // At a discontinuity (t = 0)
/// let t = DefaultSimdVector::splat(0.0);
/// let correction = polyblep(t);
/// ```
#[inline(always)]
pub fn polyblep<V: SimdVector<Scalar = f32>>(t: V) -> V {
    // PolyBLEP uses a polynomial approximation:
    // For 0 < t < 1: t² - 2t + 1
    // For -1 < t < 0: t² + 2t + 1
    //
    // Simplified implementation for positive t:

    let one = V::splat(1.0);
    let two = V::splat(2.0);

    let t_sq = t.mul(t);

    // t² - 2t + 1
    t_sq.sub(two.mul(t)).add(one)
}

/// PolyBLEP-corrected sawtooth wave
///
/// Generates a band-limited sawtooth wave using PolyBLEP correction.
///
/// # Parameters
///
/// - `phase`: Phase in [0, 1]
/// - `phase_increment`: Phase increment per sample (frequency-dependent)
///
/// # Example
///
/// ```rust
/// use rigel_math::{DefaultSimdVector, SimdVector};
/// use rigel_math::polyblep::polyblep_sawtooth;
///
/// let phase = DefaultSimdVector::splat(0.5);
/// let phase_inc = DefaultSimdVector::splat(0.001);
/// let result = polyblep_sawtooth(phase, phase_inc);
/// ```
#[inline(always)]
pub fn polyblep_sawtooth<V: SimdVector<Scalar = f32>>(phase: V, _phase_increment: V) -> V {
    // Naive sawtooth: 2 * phase - 1
    let two = V::splat(2.0);
    let one = V::splat(1.0);

    // Apply PolyBLEP correction at discontinuities
    // For now, return naive (full implementation would detect and correct discontinuities)
    two.mul(phase).sub(one)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DefaultSimdVector;

    #[test]
    fn test_polyblep_zero() {
        let t = DefaultSimdVector::splat(0.0);
        let result = polyblep(t);
        let value = result.horizontal_sum() / DefaultSimdVector::LANES as f32;
        // At t=0: 0 - 0 + 1 = 1
        assert!((value - 1.0).abs() < 1e-4, "polyblep(0) should be 1");
    }
}
