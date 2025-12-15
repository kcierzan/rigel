//! Trait interface for modulation sources.
//!
//! This module defines the `ModulationSource` trait - the common interface
//! for all modulation sources (LFOs, envelopes, sequencers, etc.).

use rigel_timing::Timebase;

/// Interface for modulation sources (LFOs, envelopes, sequencers, etc.)
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
/// - All methods must complete in constant time (O(1))
/// - No heap allocations permitted
///
/// # Output Range
///
/// Modulation values are typically in the range:
/// - [-1.0, 1.0] for bipolar modulation (e.g., LFO)
/// - [0.0, 1.0] for unipolar modulation (e.g., envelope)
///
/// The exact range depends on the implementor's polarity configuration.
pub trait ModulationSource {
    /// Reset the modulation source to initial state.
    ///
    /// After calling reset:
    /// - Phase should return to start position
    /// - Any accumulated state should be cleared
    /// - Random generators may be re-seeded (implementation-defined)
    fn reset(&mut self, timebase: &Timebase);

    /// Update the modulation source state.
    ///
    /// Called at control rate (typically every 32-64 samples).
    /// This method advances the internal state based on the timebase.
    ///
    /// # Arguments
    /// * `timebase` - Current timing context providing sample rate and position
    ///
    /// # Real-Time Safety
    /// This method must complete in constant time with no allocations.
    fn update(&mut self, timebase: &Timebase);

    /// Get the current output value.
    ///
    /// Can be called at any time; returns the last computed value.
    /// This method does not advance state - it only reads.
    ///
    /// # Returns
    /// Current modulation value within the configured range.
    fn value(&self) -> f32;
}
