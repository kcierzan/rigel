//! LFO waveshape definitions.

use core::f32::consts::TAU;
use rigel_math::{fast_cosf, fast_sinf};

/// Available LFO waveshapes.
///
/// Each waveshape maps phase [0.0, 1.0) to an output value.
/// Output is bipolar [-1.0, 1.0] before polarity scaling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LfoWaveshape {
    /// Smooth sinusoidal oscillation.
    /// Output: sin(phase * 2pi)
    #[default]
    Sine,

    /// Linear ramp up and down.
    /// Output: rises to +1 at phase=0.25, falls to -1 at phase=0.75
    Triangle,

    /// Linear ramp up with instant reset.
    /// Output: rises from -1 to +1, then resets
    Saw,

    /// 50% duty cycle square wave.
    /// Output: +1 for phase < 0.5, -1 otherwise
    Square,

    /// Variable duty cycle pulse wave.
    /// Output: +1 for phase < pulse_width, -1 otherwise
    Pulse,

    /// Random value held for one cycle.
    /// Output: new random value sampled at each phase wrap
    SampleAndHold,

    /// Continuously varying random values.
    /// Output: new random value on each update
    Noise,
}

impl LfoWaveshape {
    /// Generate output value for the given phase.
    ///
    /// # Arguments
    /// * `phase` - Current phase in [0.0, 1.0)
    /// * `pulse_width` - Duty cycle for Pulse waveshape [0.01, 0.99]
    /// * `held_value` - Stored value for SampleAndHold waveshape
    /// * `noise_value` - Random value for Noise waveshape
    ///
    /// # Returns
    /// Output value in [-1.0, 1.0] (bipolar)
    #[inline]
    pub fn generate(self, phase: f32, pulse_width: f32, held_value: f32, noise_value: f32) -> f32 {
        match self {
            LfoWaveshape::Sine => fast_sinf(phase * TAU),
            LfoWaveshape::Triangle => {
                // Triangle wave: rises to +1 at 0.25, falls to -1 at 0.75
                if phase < 0.25 {
                    phase * 4.0
                } else if phase < 0.75 {
                    2.0 - phase * 4.0
                } else {
                    phase * 4.0 - 4.0
                }
            }
            LfoWaveshape::Saw => {
                // Saw wave: -1 at phase=0, +1 at phase=1
                phase * 2.0 - 1.0
            }
            LfoWaveshape::Square => {
                // Square wave: +1 for first half, -1 for second half
                if phase < 0.5 {
                    1.0
                } else {
                    -1.0
                }
            }
            LfoWaveshape::Pulse => {
                // Pulse wave: +1 for phase < pulse_width, -1 otherwise
                if phase < pulse_width {
                    1.0
                } else {
                    -1.0
                }
            }
            LfoWaveshape::SampleAndHold => held_value,
            LfoWaveshape::Noise => noise_value,
        }
    }

    /// Compute the analytical derivative at the given phase.
    ///
    /// Used for cubic Hermite interpolation to compute tangent values.
    /// The derivative represents the rate of change of the waveform.
    ///
    /// # Arguments
    /// * `phase` - Current phase in [0.0, 1.0)
    /// * `_pulse_width` - Duty cycle for Pulse waveshape (unused for derivative)
    ///
    /// # Returns
    /// Derivative value. For step functions (Square, Pulse, S&H), returns 0.
    #[inline]
    pub fn derivative(self, phase: f32, _pulse_width: f32) -> f32 {
        match self {
            LfoWaveshape::Sine => {
                // d/d(phase) sin(2*pi*phase) = 2*pi * cos(2*pi*phase)
                TAU * fast_cosf(phase * TAU)
            }
            LfoWaveshape::Triangle => {
                // Piecewise linear with slopes:
                // - Rising from 0 to 0.25: slope = 4
                // - Falling from 0.25 to 0.75: slope = -4
                // - Rising from 0.75 to 1.0: slope = 4
                if !(0.25..0.75).contains(&phase) {
                    4.0
                } else {
                    -4.0
                }
            }
            LfoWaveshape::Saw => {
                // d/d(phase) (2*phase - 1) = 2
                2.0
            }
            // Step functions have zero derivative (technically undefined at edges)
            LfoWaveshape::Square | LfoWaveshape::Pulse | LfoWaveshape::SampleAndHold => 0.0,
            // Noise is not interpolated, so derivative is irrelevant
            LfoWaveshape::Noise => 0.0,
        }
    }
}
