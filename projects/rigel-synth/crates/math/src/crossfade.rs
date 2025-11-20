//! Crossfade and parameter ramping utilities for smooth audio transitions
//!
//! This module provides utilities for crossfading between audio signals and
//! smoothly ramping parameters to avoid clicks and zipper noise.

use crate::SimdVector;

/// Perform linear crossfade between two signals
///
/// Simple linear interpolation: `a * (1 - t) + b * t`
///
/// # Arguments
///
/// * `a` - First signal
/// * `b` - Second signal
/// * `t` - Crossfade position [0, 1] where 0=all A, 1=all B
///
/// # Example
///
/// ```rust
/// use rigel_math::{DefaultSimdVector, SimdVector};
/// use rigel_math::crossfade::crossfade_linear;
///
/// let signal_a = DefaultSimdVector::splat(1.0);
/// let signal_b = DefaultSimdVector::splat(0.0);
/// let mix = DefaultSimdVector::splat(0.5);
/// let result = crossfade_linear(signal_a, signal_b, mix);
/// ```
pub fn crossfade_linear<V: SimdVector<Scalar = f32>>(a: V, b: V, t: V) -> V {
    let one = V::splat(1.0);
    let one_minus_t = one.sub(t);

    // a * (1 - t) + b * t
    a.mul(one_minus_t).add(b.mul(t))
}

/// Perform equal-power crossfade between two signals
///
/// Uses sine/cosine panning law to maintain constant energy during crossfade,
/// avoiding the perceived volume dip of linear crossfading.
///
/// # Arguments
///
/// * `a` - First signal
/// * `b` - Second signal
/// * `t` - Crossfade position [0, 1] where 0=all A, 1=all B
///
/// # Example
///
/// ```rust
/// use rigel_math::{DefaultSimdVector, SimdVector};
/// use rigel_math::crossfade::crossfade_equal_power;
///
/// let signal_a = DefaultSimdVector::splat(1.0);
/// let signal_b = DefaultSimdVector::splat(1.0);
/// let mix = DefaultSimdVector::splat(0.5);
/// let result = crossfade_equal_power(signal_a, signal_b, mix);
/// ```
pub fn crossfade_equal_power<V: SimdVector<Scalar = f32>>(a: V, b: V, t: V) -> V {
    use crate::math::{cos, sin};

    // Equal power crossfade using sin/cos pan law
    // angle = t * pi/2
    let pi_over_2 = V::splat(core::f32::consts::PI / 2.0);
    let angle = t.mul(pi_over_2);

    let gain_a = cos(angle);
    let gain_b = sin(angle);

    a.mul(gain_a).add(b.mul(gain_b))
}

/// Perform S-curve crossfade between two signals
///
/// Uses smoothstep for a more gradual transition at the edges.
///
/// # Arguments
///
/// * `a` - First signal
/// * `b` - Second signal
/// * `t` - Crossfade position [0, 1] where 0=all A, 1=all B
///
/// # Example
///
/// ```rust
/// use rigel_math::{DefaultSimdVector, SimdVector};
/// use rigel_math::crossfade::crossfade_scurve;
///
/// let signal_a = DefaultSimdVector::splat(1.0);
/// let signal_b = DefaultSimdVector::splat(0.0);
/// let mix = DefaultSimdVector::splat(0.5);
/// let result = crossfade_scurve(signal_a, signal_b, mix);
/// ```
pub fn crossfade_scurve<V: SimdVector<Scalar = f32>>(a: V, b: V, t: V) -> V {
    use crate::sigmoid::smoothstep;

    // Apply smoothstep to crossfade position
    let smooth_t = smoothstep(t);
    crossfade_linear(a, b, smooth_t)
}

/// Parameter ramp for smooth parameter changes
///
/// Generates a linear ramp from start to end value over a specified number of samples,
/// useful for avoiding zipper noise when changing parameters.
#[derive(Debug, Clone, Copy)]
pub struct ParameterRamp {
    /// Current value
    current: f32,
    /// Target value
    target: f32,
    /// Increment per sample
    increment: f32,
    /// Remaining samples
    remaining_samples: usize,
}

impl ParameterRamp {
    /// Create a new parameter ramp
    ///
    /// # Arguments
    ///
    /// * `start` - Starting value
    /// * `end` - Target value
    /// * `num_samples` - Number of samples to ramp over
    ///
    /// # Example
    ///
    /// ```rust
    /// use rigel_math::crossfade::ParameterRamp;
    ///
    /// // Ramp from 0.0 to 1.0 over 64 samples
    /// let mut ramp = ParameterRamp::new(0.0, 1.0, 64);
    /// ```
    pub fn new(start: f32, end: f32, num_samples: usize) -> Self {
        let increment = if num_samples > 0 {
            (end - start) / num_samples as f32
        } else {
            0.0
        };

        Self {
            current: start,
            target: end,
            increment,
            remaining_samples: num_samples,
        }
    }

    /// Get the next sample value
    ///
    /// Returns the current value and advances the ramp.
    pub fn next(&mut self) -> f32 {
        let value = self.current;

        if self.remaining_samples > 0 {
            self.current += self.increment;
            self.remaining_samples -= 1;
        } else {
            self.current = self.target;
        }

        value
    }

    /// Check if the ramp has completed
    pub fn is_complete(&self) -> bool {
        self.remaining_samples == 0
    }

    /// Get the current value without advancing
    pub fn current(&self) -> f32 {
        self.current
    }

    /// Fill a SIMD vector with ramped values
    ///
    /// Efficiently fills a SIMD vector with sequential ramp values.
    ///
    /// # Example
    ///
    /// ```rust
    /// use rigel_math::{DefaultSimdVector, SimdVector};
    /// use rigel_math::crossfade::ParameterRamp;
    ///
    /// let mut ramp = ParameterRamp::new(0.0, 1.0, 64);
    /// let values = ramp.fill_simd::<DefaultSimdVector>();
    /// ```
    pub fn fill_simd<V: SimdVector<Scalar = f32>>(&mut self) -> V {
        let mut values = [0.0f32; 16]; // Max SIMD width

        for i in 0..V::LANES {
            values[i] = self.next();
        }

        V::from_slice(&values)
    }

    /// Fill a block of samples with ramped values
    ///
    /// # Arguments
    ///
    /// * `output` - Output buffer to fill
    ///
    /// # Example
    ///
    /// ```rust
    /// use rigel_math::crossfade::ParameterRamp;
    /// use rigel_math::Block64;
    ///
    /// let mut ramp = ParameterRamp::new(0.0, 1.0, 64);
    /// let mut block = Block64::new();
    /// ramp.fill_block(block.as_slice_mut());
    /// ```
    pub fn fill_block(&mut self, output: &mut [f32]) {
        for sample in output.iter_mut() {
            *sample = self.next();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DefaultSimdVector, SimdVector};

    #[test]
    fn test_crossfade_linear() {
        let a = DefaultSimdVector::splat(1.0);
        let b = DefaultSimdVector::splat(0.0);
        let t = DefaultSimdVector::splat(0.5);

        let result = crossfade_linear(a, b, t);
        let sum = result.horizontal_sum();
        let avg = sum / DefaultSimdVector::LANES as f32;

        assert!(
            (avg - 0.5).abs() < 0.001,
            "Linear crossfade at 0.5 should be 0.5"
        );
    }

    #[test]
    fn test_parameter_ramp() {
        let mut ramp = ParameterRamp::new(0.0, 1.0, 10);

        assert_eq!(ramp.current(), 0.0);
        assert!(!ramp.is_complete());

        // Sample first value
        let first = ramp.next();
        assert!((first - 0.0).abs() < 0.001);

        // Sample all values
        for _ in 1..10 {
            ramp.next();
        }

        assert!(ramp.is_complete());
        assert!((ramp.current() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_ramp_fill_block() {
        let mut ramp = ParameterRamp::new(0.0, 1.0, 64);
        let mut block = [0.0f32; 64];

        ramp.fill_block(&mut block);

        // Check first and last values
        assert!((block[0] - 0.0).abs() < 0.001);
        assert!((block[63] - 63.0 / 64.0).abs() < 0.01);
    }
}
