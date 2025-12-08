//! Trait interface for modulation sources.

use crate::Timebase;

/// Interface for modulation sources (LFOs, envelopes, etc.)
///
/// Implementors produce time-varying values that can drive synthesizer parameters.
/// This trait provides a common interface for all modulation sources, enabling
/// consistent timing and control rate behavior.
///
/// # Implementation Requirements
///
/// - `reset()` must return the source to its initial state
/// - `update()` is called at control rate and should update internal state
/// - `value()` can be called at any time and should return the current value
/// - `value()` must return a sensible default before the first `update()` call
///
/// # Output Range
///
/// Modulation values are typically in the range:
/// - [-1.0, 1.0] for bipolar modulation (e.g., LFO)
/// - [0.0, 1.0] for unipolar modulation (e.g., envelope)
pub trait ModulationSource {
    /// Reset the modulation source to initial state.
    fn reset(&mut self);

    /// Update the modulation source state.
    ///
    /// Called at control rate (typically every 32-64 samples).
    ///
    /// # Arguments
    /// * `timebase` - Current timing context
    fn update(&mut self, timebase: &Timebase);

    /// Get the current output value.
    ///
    /// Can be called at any time; returns the last computed value.
    ///
    /// # Returns
    /// Current modulation value (typically in range [-1.0, 1.0] or [0.0, 1.0])
    fn value(&self) -> f32;
}
