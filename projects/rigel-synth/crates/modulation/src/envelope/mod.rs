//! SY-Style Multi-Segment Envelope Generator
//!
//! This module provides a Yamaha SY99-style envelope generator with
//! MSFA-compatible rate calculations, outputting linear amplitude values.
//!
//! # Features
//!
//! - MSFA-compatible rate calculations and timing (STATICS table)
//! - Multiple segment configurations via const generics (FM 8-seg, AWM 10-seg, 7-seg)
//! - Rate scaling by MIDI note position
//! - Delayed envelope start for evolving sounds
//! - Segment looping for rhythmic textures
//! - Instantaneous attack jump for FM "punch"
//! - SIMD batch processing for polyphonic efficiency
//!
//! # Internal Format: f32 Linear Amplitude
//!
//! The envelope uses f32 floating-point format internally for arbitrary
//! precision timing. Level range is 0.0 to 1.0 (linear amplitude).
//! Timing uses the STATICS table to support durations from instant to 40+ seconds.
//!
//! # Example
//!
//! ```ignore
//! use rigel_modulation::envelope::{FmEnvelope, Segment, EnvelopeConfig, LoopConfig};
//!
//! // Create envelope with default settings
//! let mut env = FmEnvelope::new(44100.0);
//!
//! // Trigger note-on (middle C)
//! env.note_on(60);
//!
//! // Process audio samples
//! for _ in 0..44100 {
//!     let amplitude = env.process();
//!     // amplitude is in range [0.0, 1.0]
//! }
//!
//! // Release the note
//! env.note_off();
//!
//! // Continue processing release phase
//! while env.is_active() {
//!     let _amplitude = env.process();
//! }
//! ```

mod batch;
mod config;
mod rates;
mod segment;
mod state;

// Re-export public types
pub use batch::{EnvelopeBatch, FmEnvelopeBatch16, FmEnvelopeBatch4, FmEnvelopeBatch8};
pub use config::{
    AwmEnvelopeConfig, EnvelopeConfig, FmEnvelopeConfig, LoopConfig, SevenSegEnvelopeConfig,
};
pub use rates::{
    calculate_increment_f32, calculate_increment_f32_scaled, calculate_increment_f32_with_max_rate,
    get_static_count, linear_to_param_level, max_rate_for_sample_rate, param_to_level_f32,
    scale_rate, seconds_to_rate, JUMP_TARGET, MIN_SEGMENT_TIME_SECONDS, STATICS,
};
pub use segment::Segment;
pub use state::{EnvelopeLevel, EnvelopePhase, EnvelopeState, LEVEL_MAX, LEVEL_MIN};

use crate::ModulationSource;
use rigel_timing::Timebase;

/// SY-style multi-segment envelope generator.
///
/// Uses f32 linear amplitude (0.0-1.0) internally for precise timing.
/// Implements MSFA-compatible rate calculations and attack behavior.
///
/// # Type Parameters
///
/// * `KEY_ON_SEGS` - Number of key-on segments (attack/decay)
/// * `RELEASE_SEGS` - Number of release segments
///
/// # Real-Time Safety
///
/// All methods are allocation-free and complete in constant time.
/// The envelope is `Copy` and `Clone` for efficient voice management.
#[derive(Debug, Clone, Copy)]
pub struct Envelope<const KEY_ON_SEGS: usize, const RELEASE_SEGS: usize> {
    /// Immutable configuration
    config: EnvelopeConfig<KEY_ON_SEGS, RELEASE_SEGS>,

    /// Mutable runtime state
    state: EnvelopeState,
}

/// Type alias for 8-segment FM envelope (6 key-on + 2 release).
pub type FmEnvelope = Envelope<6, 2>;

/// Type alias for 10-segment AWM envelope (5 key-on + 5 release).
pub type AwmEnvelope = Envelope<5, 5>;

/// Type alias for 7-segment envelope (5 key-on + 2 release).
pub type SevenSegEnvelope = Envelope<5, 2>;

impl<const K: usize, const R: usize> Envelope<K, R> {
    // =========================================================================
    // Construction
    // =========================================================================

    /// Create new envelope with default configuration.
    ///
    /// Default configuration:
    /// - All key-on segments: rate=99, level=99 (instant full)
    /// - All release segments: rate=50, level=0 (medium fade to silence)
    /// - No rate scaling, no delay, no looping
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate in Hz (e.g., 44100.0)
    pub fn new(sample_rate: f32) -> Self {
        Self {
            config: EnvelopeConfig::default_with_sample_rate(sample_rate),
            state: EnvelopeState::default(),
        }
    }

    /// Create envelope with specific configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Envelope configuration
    pub fn with_config(config: EnvelopeConfig<K, R>) -> Self {
        Self {
            config,
            state: EnvelopeState::default(),
        }
    }

    // =========================================================================
    // Note Events
    // =========================================================================

    /// Trigger note-on event.
    ///
    /// Starts the envelope attack sequence. If a delay is configured,
    /// the envelope waits before beginning attack. If the envelope is
    /// already active, it restarts from the current level (no click).
    ///
    /// # Arguments
    ///
    /// * `midi_note` - MIDI note number (0-127) for rate scaling
    pub fn note_on(&mut self, midi_note: u8) {
        // Cache MIDI note for rate scaling
        self.state.midi_note = midi_note;

        // Check if we should start with delay
        if self.config.delay_samples > 0 {
            self.state.phase = EnvelopePhase::Delay;
            self.state.delay_remaining = self.config.delay_samples;
            self.state.segment_index = 0;
            // Don't reset level - allows retrigger from current position
        } else {
            // Start attack immediately
            self.start_key_on_phase();
        }
    }

    /// Trigger note-off event.
    ///
    /// Transitions to release phase. If looping, exits the loop.
    /// If in delay phase, skips to release immediately.
    pub fn note_off(&mut self) {
        match self.state.phase {
            EnvelopePhase::Idle | EnvelopePhase::Complete => {
                // Already off, nothing to do
            }
            EnvelopePhase::Delay => {
                // Skip to release immediately
                self.start_release_phase();
            }
            EnvelopePhase::KeyOn | EnvelopePhase::Sustain => {
                // Transition to release
                self.start_release_phase();
            }
            EnvelopePhase::Release => {
                // Already releasing, nothing to do
            }
        }
    }

    // =========================================================================
    // Processing
    // =========================================================================

    /// Process one sample and return linear amplitude.
    ///
    /// Advances the envelope state by one sample and returns the
    /// current amplitude value.
    ///
    /// # Returns
    ///
    /// Linear amplitude in range [0.0, 1.0]
    ///
    /// # Performance
    ///
    /// O(1) time, no allocations, ~20-50 nanoseconds typical
    #[inline]
    pub fn process(&mut self) -> f32 {
        match self.state.phase {
            EnvelopePhase::Idle => 0.0,
            EnvelopePhase::Delay => self.process_delay(),
            EnvelopePhase::KeyOn => self.process_key_on(),
            EnvelopePhase::Sustain => self.process_sustain(),
            EnvelopePhase::Release => self.process_release(),
            EnvelopePhase::Complete => {
                // Transition to idle
                self.state.phase = EnvelopePhase::Idle;
                0.0
            }
        }
    }

    /// Process block of samples.
    ///
    /// Optimized batch processing for audio blocks.
    ///
    /// # Arguments
    ///
    /// * `output` - Slice to fill with linear amplitude values
    pub fn process_block(&mut self, output: &mut [f32]) {
        for sample in output.iter_mut() {
            *sample = self.process();
        }
    }

    /// Get current linear amplitude without advancing state.
    ///
    /// Useful for UI display or modulation routing without
    /// consuming a sample.
    ///
    /// # Returns
    ///
    /// Current linear amplitude in range [0.0, 1.0]
    #[inline]
    pub fn value(&self) -> f32 {
        self.state.level
    }

    // =========================================================================
    // State Queries
    // =========================================================================

    /// Get current envelope phase.
    #[inline]
    pub fn phase(&self) -> EnvelopePhase {
        self.state.phase
    }

    /// Check if envelope is active (not idle or complete).
    #[inline]
    pub fn is_active(&self) -> bool {
        self.state.is_active()
    }

    /// Check if envelope is in release phase.
    #[inline]
    pub fn is_releasing(&self) -> bool {
        self.state.is_releasing()
    }

    /// Check if envelope has completed.
    #[inline]
    pub fn is_complete(&self) -> bool {
        self.state.is_complete()
    }

    /// Get current segment index.
    #[inline]
    pub fn current_segment(&self) -> usize {
        self.state.segment_index as usize
    }

    // =========================================================================
    // Configuration
    // =========================================================================

    /// Update configuration.
    ///
    /// New configuration takes effect on next note_on().
    /// Does not affect currently playing envelope.
    pub fn set_config(&mut self, config: EnvelopeConfig<K, R>) {
        self.config = config;
    }

    /// Get current configuration (read-only).
    #[inline]
    pub fn config(&self) -> &EnvelopeConfig<K, R> {
        &self.config
    }

    /// Get current state (read-only).
    #[inline]
    pub fn state(&self) -> &EnvelopeState {
        &self.state
    }

    /// Reset to idle state.
    ///
    /// Immediately stops envelope and resets to idle.
    /// Level is set to minimum (silent).
    pub fn reset(&mut self) {
        self.state = EnvelopeState::default();
    }

    // =========================================================================
    // Internal: Phase Processing
    // =========================================================================

    /// Process delay phase - count down, return 0.
    #[inline]
    fn process_delay(&mut self) -> f32 {
        if self.state.delay_remaining > 0 {
            self.state.delay_remaining -= 1;
            0.0
        } else {
            // Delay complete, start attack
            self.start_key_on_phase();
            self.process_key_on()
        }
    }

    /// Process key-on segments (attack/decay).
    #[inline]
    fn process_key_on(&mut self) -> f32 {
        // Attack jump is applied in start_key_on_phase() before setup_segment()
        // to ensure correct increment calculation based on post-jump distance.

        // Apply appropriate increment based on direction
        if self.state.rising {
            self.apply_attack_increment();
        } else {
            self.apply_decay_increment();
        }

        // Check if we've reached target
        if self.reached_target() {
            self.advance_segment();
        }

        self.state.level
    }

    /// Process sustain phase - hold at current level.
    #[inline]
    fn process_sustain(&mut self) -> f32 {
        // Just return current level, no change
        self.state.level
    }

    /// Process release segments.
    #[inline]
    fn process_release(&mut self) -> f32 {
        // Release always decays (even if target is higher, which is unusual)
        self.apply_decay_increment();

        // Check if we've reached target
        if self.reached_target() {
            self.advance_release_segment();
        }

        self.state.level
    }

    // =========================================================================
    // Internal: Phase Transitions
    // =========================================================================

    /// Start key-on phase from first segment.
    fn start_key_on_phase(&mut self) {
        self.state.phase = EnvelopePhase::KeyOn;
        self.state.segment_index = 0;

        // Apply attack jump BEFORE setup_segment calculates increment.
        // This jump to ~0.16% amplitude (-56dB) gets the envelope out of the
        // sub-perceptual range quickly. The DX7 uses 1716 in Q8 format (~40dB
        // above minimum). This preserves FM "punch" while allowing proper
        // slow attacks that build from near-silence.
        if self.state.level < JUMP_TARGET {
            self.state.level = JUMP_TARGET;
        }

        self.setup_segment(0, true);
    }

    /// Start release phase.
    fn start_release_phase(&mut self) {
        if R == 0 {
            // No release segments, go directly to complete
            self.state.phase = EnvelopePhase::Complete;
            self.state.level = LEVEL_MIN;
        } else {
            self.state.phase = EnvelopePhase::Release;
            self.state.segment_index = 0;
            self.setup_segment(0, false);
        }
    }

    /// Setup a segment for processing.
    fn setup_segment(&mut self, segment_index: u8, is_key_on: bool) {
        let segment = if is_key_on {
            &self.config.key_on_segments[segment_index as usize]
        } else {
            &self.config.release_segments[segment_index as usize]
        };

        // Calculate target level from segment (linear 0.0-1.0)
        self.state.target_level = param_to_level_f32(segment.level);

        // Calculate the distance to travel
        let distance = (self.state.target_level - self.state.level).abs();

        // If already at target, no movement needed
        if distance < f32::EPSILON {
            self.state.increment = 0.0;
            self.state.decay_factor = 1.0;
            self.state.rising = false;
            return;
        }

        // Calculate per-sample increment with rate scaling applied.
        // The raw increment is for full 0.0-1.0 range; scale by actual distance
        // so that the configured time matches the actual transition time.
        // Uses pre-computed max_rate to avoid O(100) search per segment change.
        let full_range_increment = calculate_increment_f32_with_max_rate(
            segment.rate,
            self.state.midi_note,
            self.config.rate_scaling,
            self.config.sample_rate,
            self.config.max_rate(),
        );

        // Scale increment by distance to maintain correct timing
        let scaled_increment = full_range_increment * distance;

        // Determine direction
        self.state.rising = self.state.target_level > self.state.level;

        if self.state.rising {
            // For attack (rising) segments, use MSFA-faithful exponential approach.
            // The average factor is ~9 (integral of 1+16(1-x) from 0 to 1 = 9),
            // so we divide the increment by 9 to maintain correct timing.
            self.state.attack_base_level = self.state.level;
            self.state.increment = scaled_increment / 9.0;
            self.state.decay_factor = 1.0; // Not used for attack
        } else {
            // For decay (falling) segments, use exponential decay (linear-in-dB).
            // This matches authentic DX7/SY99 behavior where decay is constant dB/second.
            //
            // Calculate decay factor: level *= decay_factor each sample
            // To go from current_level to target_level in N samples:
            //   target = current * decay_factor^N
            //   decay_factor = (target / current)^(1/N)
            //
            // N samples = distance / scaled_increment (since increment covers full distance)
            let samples = if scaled_increment > f32::EPSILON {
                distance / scaled_increment
            } else {
                1.0
            };

            // Handle target = 0 case: use a small minimum to avoid log(0)
            // -120dB is well below audible threshold
            const MIN_TARGET: f32 = 1e-6;
            let effective_target = self.state.target_level.max(MIN_TARGET);
            let current = self.state.level.max(MIN_TARGET);

            // decay_factor = (target / current)^(1/samples)
            let ratio = effective_target / current;
            self.state.decay_factor = libm::powf(ratio, 1.0 / samples);

            // Store increment for timing reference (used by reached_target check)
            self.state.increment = -scaled_increment;
        }
    }

    // =========================================================================
    // Internal: Segment Advancement
    // =========================================================================

    /// Advance to next key-on segment.
    fn advance_segment(&mut self) {
        // Snap to target
        self.state.level = self.state.target_level;

        let next_segment = self.state.segment_index + 1;

        // Check for looping
        if self.config.loop_config.is_enabled()
            && self.config.loop_config.is_valid(K)
            && self.state.segment_index == self.config.loop_config.end_segment
        {
            // Loop back to start
            self.state.segment_index = self.config.loop_config.start_segment;
            self.setup_segment(self.state.segment_index, true);
            return;
        }

        if (next_segment as usize) < K {
            // More key-on segments
            self.state.segment_index = next_segment;
            self.setup_segment(next_segment, true);
        } else {
            // Enter sustain at final key-on level
            self.state.phase = EnvelopePhase::Sustain;
        }
    }

    /// Advance to next release segment.
    fn advance_release_segment(&mut self) {
        // Snap to target
        self.state.level = self.state.target_level;

        let next_segment = self.state.segment_index + 1;

        if (next_segment as usize) < R {
            // More release segments
            self.state.segment_index = next_segment;
            self.setup_segment(next_segment, false);
        } else {
            // Release complete
            self.state.phase = EnvelopePhase::Complete;
            self.state.level = LEVEL_MIN;
        }
    }

    // =========================================================================
    // Internal: Increment Application
    // =========================================================================

    /// MSFA-faithful exponential approach for attack (rising) phases.
    ///
    /// The factor varies from ~17 at start to ~1 near target, creating
    /// the characteristic FM "snap" - fast through quiet region, slowing at top.
    /// This matches the DX7/MSFA formula: `level += ((max - level) >> 24) * inc`
    #[inline]
    fn apply_attack_increment(&mut self) {
        // Calculate progress through attack (0.0 at start, 1.0 at end)
        let remaining = self.state.target_level - self.state.level;
        let total = self.state.target_level - self.state.attack_base_level;

        let progress = if total > f32::EPSILON {
            1.0 - (remaining / total)
        } else {
            1.0
        };

        // Factor varies from 17.0 (at start) to 1.0 (at end)
        // This matches MSFA's ~17x variation for authentic FM attack character
        let factor = 1.0 + 16.0 * (1.0 - progress);

        self.state.level += self.state.increment * factor;
        self.state.level = self.state.level.min(self.state.target_level);
    }

    /// Exponential decay for falling phases (linear-in-dB).
    ///
    /// The DX7/SY99/MSFA operates internally in dB domain where decay is
    /// linear subtraction in dB. In linear amplitude domain, this translates
    /// to multiplicative decay: level *= decay_factor each sample.
    ///
    /// This produces the characteristic where "slope increases as level
    /// approaches zero" - faster decay through quiet regions, matching
    /// authentic FM synth behavior.
    #[inline]
    fn apply_decay_increment(&mut self) {
        // Exponential decay: multiply by factor each sample
        self.state.level *= self.state.decay_factor;

        // Clamp to valid range and snap to minimum when very small
        // to avoid denormal numbers and infinite asymptotic approach
        if self.state.level < LEVEL_MIN + 1e-7 {
            self.state.level = LEVEL_MIN;
        }
    }

    /// Check if we've reached the target level.
    ///
    /// For decay (exponential), we check if we're within a small tolerance
    /// of the target to handle floating-point precision and ensure we don't
    /// get stuck in asymptotic approach.
    #[inline]
    fn reached_target(&self) -> bool {
        if self.state.rising {
            self.state.level >= self.state.target_level
        } else {
            // For decay, check if we've reached or slightly passed target,
            // or if we're within 0.1% of target (handles exponential asymptotic approach)
            self.state.level <= self.state.target_level
                || (self.state.target_level > 0.0
                    && self.state.level <= self.state.target_level * 1.001)
        }
    }
}

// =========================================================================
// ModulationSource Implementation
// =========================================================================

impl<const K: usize, const R: usize> ModulationSource for Envelope<K, R> {
    fn reset(&mut self, _timebase: &Timebase) {
        Envelope::reset(self);
    }

    fn update(&mut self, _timebase: &Timebase) {
        // Envelope doesn't use control-rate updates; it processes per-sample
        // This is intentionally a no-op
    }

    fn value(&self) -> f32 {
        Envelope::value(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_envelope_creation() {
        let env = FmEnvelope::new(44100.0);
        assert_eq!(env.phase(), EnvelopePhase::Idle);
        assert!(!env.is_active());
    }

    #[test]
    fn test_note_on_triggers_key_on() {
        let mut env = FmEnvelope::new(44100.0);
        env.note_on(60);
        assert_eq!(env.phase(), EnvelopePhase::KeyOn);
        assert!(env.is_active());
    }

    #[test]
    fn test_note_off_triggers_release() {
        let mut env = FmEnvelope::new(44100.0);
        env.note_on(60);
        env.note_off();
        assert_eq!(env.phase(), EnvelopePhase::Release);
        assert!(env.is_releasing());
    }

    #[test]
    fn test_output_range() {
        let mut env = FmEnvelope::new(44100.0);
        env.note_on(60);

        // Process many samples
        for _ in 0..44100 {
            let value = env.process();
            assert!(
                (0.0..=1.0).contains(&value),
                "Output {} not in range [0.0, 1.0]",
                value
            );
        }
    }

    #[test]
    fn test_segment_transitions() {
        let mut env = FmEnvelope::new(44100.0);
        env.note_on(60);

        // Process until we've gone through segments
        let mut last_segment = 0;
        let mut segment_changes = 0;

        for _ in 0..44100 {
            env.process();
            let current = env.current_segment();
            if current != last_segment {
                segment_changes += 1;
                last_segment = current;
            }
        }

        // Should have transitioned through at least some segments
        assert!(segment_changes > 0, "No segment transitions occurred");
    }

    #[test]
    fn test_value_without_advance() {
        let mut env = FmEnvelope::new(44100.0);
        env.note_on(60);
        env.process(); // Advance once

        let v1 = env.value();
        let v2 = env.value();
        assert_eq!(v1, v2, "value() should not change state");
    }

    #[test]
    fn test_reset() {
        let mut env = FmEnvelope::new(44100.0);
        env.note_on(60);
        env.process();
        env.reset();

        assert_eq!(env.phase(), EnvelopePhase::Idle);
        assert!(!env.is_active());
    }

    #[test]
    fn test_delayed_start() {
        let mut config = FmEnvelopeConfig::default_with_sample_rate(44100.0);
        config.delay_samples = 1000;

        let mut env = FmEnvelope::with_config(config);
        env.note_on(60);

        assert_eq!(env.phase(), EnvelopePhase::Delay);

        // Process through delay
        for _ in 0..1000 {
            let value = env.process();
            assert_eq!(value, 0.0, "Should be silent during delay");
        }

        // After delay, should be in KeyOn
        env.process();
        assert_eq!(env.phase(), EnvelopePhase::KeyOn);
    }

    #[test]
    fn test_note_off_during_delay() {
        let mut config = FmEnvelopeConfig::default_with_sample_rate(44100.0);
        config.delay_samples = 1000;

        let mut env = FmEnvelope::with_config(config);
        env.note_on(60);
        env.note_off();

        // Should skip to release
        assert_eq!(env.phase(), EnvelopePhase::Release);
    }

    #[test]
    fn test_copy() {
        let env1 = FmEnvelope::new(44100.0);
        let env2 = env1; // Copy

        assert_eq!(env1.phase(), env2.phase());
    }

    #[test]
    fn test_envelope_completes() {
        let mut env = FmEnvelope::new(44100.0);
        env.note_on(60);
        env.note_off();

        // Process until complete
        let mut iterations = 0;
        while env.is_active() && iterations < 100000 {
            env.process();
            iterations += 1;
        }

        assert!(
            !env.is_active(),
            "Envelope should complete within reasonable time"
        );
    }
}
