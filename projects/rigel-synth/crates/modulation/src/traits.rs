//! Trait interfaces for modulation sources.
//!
//! This module defines:
//! - [`ModulationSource`] - Common interface for all modulation sources (LFOs, envelopes, etc.)
//! - [`SimdAwareComponent`] - Trait for DSP components that use SIMD-accelerated processing

use rigel_math::SimdVector;
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

/// Trait for DSP components that use SIMD-accelerated processing.
///
/// Components expose their SIMD vector type as an associated type.
/// All backend properties (lanes, name, etc.) can be derived from it.
///
/// This trait enables:
/// - Reporting which SIMD backend is in use
/// - Querying the number of SIMD lanes available
/// - Generic code that operates on any SIMD-aware component
///
/// # Example
///
/// ```ignore
/// use rigel_modulation::SimdAwareComponent;
///
/// fn describe_component<T: SimdAwareComponent>(_c: &T) {
///     println!("Using {} lanes", T::Vector::LANES);
/// }
/// ```
pub trait SimdAwareComponent {
    /// The SIMD vector type used by this component.
    ///
    /// Typically `DefaultSimdVector` which resolves to the best
    /// available backend at compile time:
    /// - AVX-512: 16 lanes (x86_64)
    /// - AVX2: 8 lanes (x86_64)
    /// - NEON: 4 lanes (aarch64)
    /// - Scalar: 1 lane (fallback)
    type Vector: SimdVector<Scalar = f32>;

    /// Get the number of SIMD lanes available.
    ///
    /// This is a convenience method that returns `Self::Vector::LANES`.
    fn lanes() -> usize {
        Self::Vector::LANES
    }
}
