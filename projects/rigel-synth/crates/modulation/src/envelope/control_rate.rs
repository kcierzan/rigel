//! Control-rate envelope wrapper for CPU-efficient envelope processing.
//!
//! This module provides a wrapper around the standard envelope that updates
//! at control rate (~1.5ms intervals) rather than per-sample, with linear
//! interpolation for smooth audio output.
//!
//! # CPU Savings
//!
//! Control-rate processing reduces CPU usage by 2-5x compared to per-sample:
//! - State machine updates happen every 64 samples instead of every sample
//! - Linear interpolation (~5ns) replaces full envelope tick (~20-50ns)
//!
//! # Audio Quality
//!
//! At 64-sample intervals (~1.45ms at 44.1kHz), linear interpolation provides
//! perceptually smooth envelopes, matching classic DX7/SY99 behavior.
//!
//! # Example
//!
//! ```ignore
//! use rigel_modulation::envelope::{ControlRateFmEnvelope, FmEnvelopeConfig};
//! use rigel_timing::Timebase;
//!
//! let config = FmEnvelopeConfig::adsr(0.1, 0.2, 0.7, 0.5, 44100.0);
//! let mut env = ControlRateFmEnvelope::new(config, 64);
//!
//! let mut timebase = Timebase::new(44100.0);
//! env.note_on(60);
//!
//! // Process a 64-sample block
//! timebase.advance_block(64);
//! env.update(&timebase);
//!
//! // Get interpolated values within the block
//! for i in 0..64 {
//!     let value = env.sample();
//!     // Use value for audio processing
//! }
//! ```

use super::{Envelope, EnvelopeConfig, EnvelopePhase};
use crate::ModulationSource;
use rigel_timing::Timebase;

/// Control-rate envelope wrapper.
///
/// Wraps a standard envelope to process state updates at control rate
/// with linear interpolation between updates for smooth output.
///
/// # Type Parameters
///
/// * `KEY_ON_SEGS` - Number of key-on segments (attack/decay)
/// * `RELEASE_SEGS` - Number of release segments
#[derive(Debug, Clone, Copy)]
pub struct ControlRateEnvelope<const KEY_ON_SEGS: usize, const RELEASE_SEGS: usize> {
    /// Inner envelope for state management
    inner: Envelope<KEY_ON_SEGS, RELEASE_SEGS>,

    /// Value at start of current interval (for interpolation)
    previous_value: f32,

    /// Target value at end of current interval
    target_value: f32,

    /// Samples processed since last control-rate update
    samples_since_update: u32,

    /// Update interval in samples (power of 2)
    update_interval: u32,
}

/// Type alias for control-rate FM envelope (6 key-on + 2 release).
pub type ControlRateFmEnvelope = ControlRateEnvelope<6, 2>;

/// Type alias for control-rate AWM envelope (5 key-on + 5 release).
pub type ControlRateAwmEnvelope = ControlRateEnvelope<5, 5>;

/// Type alias for control-rate 7-segment envelope (5 key-on + 2 release).
pub type ControlRateSevenSegEnvelope = ControlRateEnvelope<5, 2>;

impl<const K: usize, const R: usize> ControlRateEnvelope<K, R> {
    // =========================================================================
    // Construction
    // =========================================================================

    /// Create a new control-rate envelope.
    ///
    /// # Arguments
    ///
    /// * `config` - Envelope configuration
    /// * `update_interval` - Update interval in samples (must be power of 2, 1-128)
    ///
    /// # Panics
    ///
    /// Panics if `update_interval` is not a valid power of 2 in range [1, 128]
    pub fn new(config: EnvelopeConfig<K, R>, update_interval: u32) -> Self {
        assert!(
            update_interval > 0 && update_interval <= 128,
            "Interval must be 1-128"
        );
        assert!(
            update_interval.is_power_of_two(),
            "Interval must be a power of 2"
        );

        Self {
            inner: Envelope::with_config(config),
            previous_value: 0.0,
            target_value: 0.0,
            samples_since_update: 0,
            update_interval,
        }
    }

    /// Create a new control-rate envelope with default sample rate.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate in Hz
    /// * `update_interval` - Update interval in samples
    pub fn with_sample_rate(sample_rate: f32, update_interval: u32) -> Self {
        Self::new(
            EnvelopeConfig::default_with_sample_rate(sample_rate),
            update_interval,
        )
    }

    // =========================================================================
    // Note Events
    // =========================================================================

    /// Trigger note-on event.
    ///
    /// Starts the envelope attack sequence.
    ///
    /// # Arguments
    ///
    /// * `midi_note` - MIDI note number (0-127) for rate scaling
    pub fn note_on(&mut self, midi_note: u8) {
        self.inner.note_on(midi_note);

        // Initialize interpolation state
        self.previous_value = self.inner.value();
        self.target_value = self.previous_value;
        self.samples_since_update = 0;
    }

    /// Trigger note-off event.
    ///
    /// Transitions to release phase.
    pub fn note_off(&mut self) {
        self.inner.note_off();

        // Update interpolation targets for release transition
        self.previous_value = self.target_value;
        self.samples_since_update = 0;
    }

    // =========================================================================
    // Processing
    // =========================================================================

    /// Perform control-rate update.
    ///
    /// Advances the inner envelope by `update_interval` samples and
    /// updates interpolation targets. Call this once per control-rate
    /// period (typically every 64 samples).
    ///
    /// Uses optimized `advance_by()` to skip per-sample iteration,
    /// providing significant CPU savings over per-sample processing.
    pub fn tick(&mut self) {
        // Store current value as previous for interpolation
        self.previous_value = self.target_value;

        // Advance inner envelope by update_interval samples in one step
        self.inner.advance_by(self.update_interval);

        // Store new target for interpolation
        self.target_value = self.inner.value();

        // Reset sample counter
        self.samples_since_update = 0;
    }

    /// Get a single interpolated sample and advance position.
    ///
    /// Returns a linearly interpolated value between the previous
    /// and target control-rate values.
    ///
    /// # Returns
    ///
    /// Linear amplitude in range [0.0, 1.0]
    #[inline]
    pub fn sample(&mut self) -> f32 {
        let t = self.samples_since_update as f32 / self.update_interval as f32;
        let value = self.previous_value + (self.target_value - self.previous_value) * t;

        self.samples_since_update = (self.samples_since_update + 1) % self.update_interval;
        value
    }

    /// Get current interpolated value without advancing.
    ///
    /// # Returns
    ///
    /// Current interpolated linear amplitude in range [0.0, 1.0]
    #[inline]
    pub fn current_value(&self) -> f32 {
        let t = self.samples_since_update as f32 / self.update_interval as f32;
        self.previous_value + (self.target_value - self.previous_value) * t
    }

    /// Fill a buffer with interpolated samples.
    ///
    /// Efficiently generates a block of interpolated values.
    /// Calls `tick()` automatically when crossing control-rate boundaries.
    ///
    /// # Arguments
    ///
    /// * `output` - Buffer to fill with interpolated values
    pub fn generate_block(&mut self, output: &mut [f32]) {
        let interval = self.update_interval;
        let inv_interval = 1.0 / interval as f32;

        for sample in output.iter_mut() {
            let t = self.samples_since_update as f32 * inv_interval;
            *sample = self.previous_value + (self.target_value - self.previous_value) * t;

            self.samples_since_update += 1;

            // Wrap and update at interval boundary
            if self.samples_since_update >= interval {
                self.tick();
            }
        }
    }

    /// Process block with explicit control-rate updates.
    ///
    /// This method is more efficient when the caller manages control-rate
    /// timing externally (e.g., via `ControlRateClock`).
    ///
    /// # Arguments
    ///
    /// * `output` - Buffer to fill with interpolated values
    /// * `should_tick` - Whether to perform a control-rate update first
    pub fn process_block_with_tick(&mut self, output: &mut [f32], should_tick: bool) {
        if should_tick {
            self.tick();
        }

        let inv_interval = 1.0 / self.update_interval as f32;

        for sample in output.iter_mut() {
            let t = self.samples_since_update as f32 * inv_interval;
            *sample = self.previous_value + (self.target_value - self.previous_value) * t;
            self.samples_since_update += 1;
        }
    }

    // =========================================================================
    // State Queries
    // =========================================================================

    /// Get current envelope phase.
    #[inline]
    pub fn phase(&self) -> EnvelopePhase {
        self.inner.phase()
    }

    /// Check if envelope is active.
    #[inline]
    pub fn is_active(&self) -> bool {
        self.inner.is_active()
    }

    /// Check if envelope is in release phase.
    #[inline]
    pub fn is_releasing(&self) -> bool {
        self.inner.is_releasing()
    }

    /// Check if envelope has completed.
    #[inline]
    pub fn is_complete(&self) -> bool {
        self.inner.is_complete()
    }

    /// Get current segment index.
    #[inline]
    pub fn current_segment(&self) -> usize {
        self.inner.current_segment()
    }

    /// Get update interval in samples.
    #[inline]
    pub fn update_interval(&self) -> u32 {
        self.update_interval
    }

    /// Get reference to inner envelope.
    #[inline]
    pub fn inner(&self) -> &Envelope<K, R> {
        &self.inner
    }

    // =========================================================================
    // Configuration
    // =========================================================================

    /// Update configuration.
    ///
    /// New configuration takes effect on next note_on().
    pub fn set_config(&mut self, config: EnvelopeConfig<K, R>) {
        self.inner.set_config(config);
    }

    /// Set update interval.
    ///
    /// # Panics
    ///
    /// Panics if interval is not a power of 2 in range [1, 128]
    pub fn set_update_interval(&mut self, interval: u32) {
        assert!(interval > 0 && interval <= 128, "Interval must be 1-128");
        assert!(interval.is_power_of_two(), "Interval must be a power of 2");
        self.update_interval = interval;
    }

    /// Reset envelope to idle state.
    pub fn reset(&mut self) {
        self.inner.reset();
        self.previous_value = 0.0;
        self.target_value = 0.0;
        self.samples_since_update = 0;
    }
}

// =========================================================================
// ModulationSource Implementation
// =========================================================================

impl<const K: usize, const R: usize> ModulationSource for ControlRateEnvelope<K, R> {
    fn reset(&mut self, _timebase: &Timebase) {
        ControlRateEnvelope::reset(self);
    }

    fn update(&mut self, _timebase: &Timebase) {
        // Perform control-rate update
        self.tick();
    }

    fn value(&self) -> f32 {
        self.current_value()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::envelope::FmEnvelopeConfig;

    const SAMPLE_RATE: f32 = 44100.0;

    #[test]
    fn test_control_rate_envelope_creation() {
        let config = FmEnvelopeConfig::default_with_sample_rate(SAMPLE_RATE);
        let env = ControlRateFmEnvelope::new(config, 64);

        assert_eq!(env.phase(), EnvelopePhase::Idle);
        assert!(!env.is_active());
        assert_eq!(env.update_interval(), 64);
    }

    #[test]
    fn test_note_on_triggers_key_on() {
        let config = FmEnvelopeConfig::default_with_sample_rate(SAMPLE_RATE);
        let mut env = ControlRateFmEnvelope::new(config, 64);

        env.note_on(60);
        assert_eq!(env.phase(), EnvelopePhase::KeyOn);
        assert!(env.is_active());
    }

    #[test]
    fn test_note_off_triggers_release() {
        let config = FmEnvelopeConfig::default_with_sample_rate(SAMPLE_RATE);
        let mut env = ControlRateFmEnvelope::new(config, 64);

        env.note_on(60);
        env.tick(); // Advance once
        env.note_off();

        assert_eq!(env.phase(), EnvelopePhase::Release);
    }

    #[test]
    fn test_output_range() {
        let config = FmEnvelopeConfig::adsr(0.1, 0.2, 0.7, 0.5, SAMPLE_RATE);
        let mut env = ControlRateFmEnvelope::new(config, 64);

        env.note_on(60);

        // Process many samples
        for _ in 0..10 {
            env.tick();
            for _ in 0..64 {
                let value = env.sample();
                assert!(
                    (0.0..=1.0).contains(&value),
                    "Output {} not in range [0.0, 1.0]",
                    value
                );
            }
        }
    }

    #[test]
    fn test_interpolation() {
        let config = FmEnvelopeConfig::adsr(0.5, 0.2, 0.7, 0.5, SAMPLE_RATE);
        let mut env = ControlRateFmEnvelope::new(config, 64);

        env.note_on(60);
        env.tick();

        // Get samples across the interval and check for discontinuities
        let mut prev_sample = env.sample();
        for _ in 1..64 {
            let curr_sample = env.sample();

            // Values should not jump discontinuously
            let diff = (curr_sample - prev_sample).abs();
            assert!(
                diff < 0.1,
                "Large discontinuity: {} -> {}",
                prev_sample,
                curr_sample
            );

            prev_sample = curr_sample;
        }
    }

    #[test]
    fn test_generate_block() {
        let config = FmEnvelopeConfig::adsr(0.1, 0.2, 0.7, 0.5, SAMPLE_RATE);
        let mut env = ControlRateFmEnvelope::new(config, 64);

        env.note_on(60);

        let mut output = [0.0f32; 64];
        env.generate_block(&mut output);

        // All values should be in valid range
        for &v in &output {
            assert!((0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn test_reset() {
        let config = FmEnvelopeConfig::default_with_sample_rate(SAMPLE_RATE);
        let mut env = ControlRateFmEnvelope::new(config, 64);

        env.note_on(60);
        env.tick();
        env.reset();

        assert_eq!(env.phase(), EnvelopePhase::Idle);
        assert!(!env.is_active());
    }

    #[test]
    #[should_panic(expected = "Interval must be a power of 2")]
    fn test_invalid_interval() {
        let config = FmEnvelopeConfig::default_with_sample_rate(SAMPLE_RATE);
        let _env = ControlRateFmEnvelope::new(config, 50); // Not power of 2
    }

    #[test]
    fn test_modulation_source_trait() {
        let config = FmEnvelopeConfig::adsr(0.1, 0.2, 0.7, 0.5, SAMPLE_RATE);
        let mut env = ControlRateFmEnvelope::new(config, 64);
        let timebase = Timebase::new(SAMPLE_RATE);

        env.note_on(60);

        // Test update via trait
        <ControlRateFmEnvelope as ModulationSource>::update(&mut env, &timebase);

        // Value should be accessible via trait
        let value = <ControlRateFmEnvelope as ModulationSource>::value(&env);
        assert!((0.0..=1.0).contains(&value));
    }
}
