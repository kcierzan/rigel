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
//! # Internal Format: Hybrid Q8/f32
//!
//! The envelope uses a hybrid approach for hardware-authentic amplitude
//! with precise timing:
//!
//! - **Output levels** are Q8 fixed-point (i16, 0-4095) providing authentic
//!   DX7-style 96dB dynamic range with 4096 discrete steps
//! - **Internal accumulation** uses f32 for sub-sample timing precision,
//!   allowing slow rates to achieve proper multi-second durations
//!
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
mod control_rate;
mod rates;
mod segment;
mod state;

// Re-export public types
pub use batch::{EnvelopeBatch, FmEnvelopeBatch16, FmEnvelopeBatch4, FmEnvelopeBatch8};
pub use config::{
    AwmEnvelopeConfig, EnvelopeConfig, FmEnvelopeConfig, LoopConfig, SevenSegEnvelopeConfig,
};
pub use control_rate::{
    ControlRateAwmEnvelope, ControlRateEnvelope, ControlRateFmEnvelope, ControlRateSevenSegEnvelope,
};
pub use rates::{
    calculate_increment_f32, calculate_increment_f32_scaled, calculate_increment_f32_with_max_rate,
    get_static_count, level_to_linear, linear_to_param_level, max_rate_for_sample_rate,
    param_to_level_q8, scale_rate, seconds_to_rate, JUMP_TARGET_Q8, MIN_SEGMENT_TIME_SECONDS,
    STATICS,
};
pub use segment::Segment;
pub use state::{EnvelopeLevel, EnvelopePhase, EnvelopeState, LEVEL_MAX, LEVEL_MIN};

use crate::ModulationSource;
use rigel_timing::Timebase;

/// SY-style multi-segment envelope generator.
///
/// Uses a hybrid Q8/f32 approach: output levels are Q8 fixed-point (i16)
/// for authentic 96dB dynamic range, while internal accumulation uses f32
/// for sub-sample timing precision. Implements MSFA-compatible rate
/// calculations and attack behavior.
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
    /// current amplitude value. Internal processing uses Q8 fixed-point;
    /// conversion to linear amplitude happens only at output.
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
    /// consuming a sample. Converts from internal Q8 format.
    ///
    /// # Returns
    ///
    /// Current linear amplitude in range [0.0, 1.0]
    #[inline]
    pub fn value(&self) -> f32 {
        level_to_linear(self.state.level)
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

        level_to_linear(self.state.level)
    }

    /// Process sustain phase - hold at current level.
    #[inline]
    fn process_sustain(&mut self) -> f32 {
        // Just return current level, no change
        level_to_linear(self.state.level)
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

        level_to_linear(self.state.level)
    }

    // =========================================================================
    // Internal: Phase Transitions
    // =========================================================================

    /// Start key-on phase from first segment.
    fn start_key_on_phase(&mut self) {
        self.state.phase = EnvelopePhase::KeyOn;
        self.state.segment_index = 0;

        // Apply attack jump BEFORE setup_segment calculates increment.
        // This jump to 1716 Q8 (~-56dB) gets the envelope out of the
        // sub-perceptual range quickly. This preserves FM "punch" while
        // allowing proper slow attacks that build from near-silence.
        // Reference: MSFA/Dexed `const int jumptarget = 1716`
        if self.state.level_precise < JUMP_TARGET_Q8 as f32 {
            self.state.level_precise = JUMP_TARGET_Q8 as f32;
            self.state.level = JUMP_TARGET_Q8;
        }

        self.setup_segment(0, true);
    }

    /// Start release phase.
    fn start_release_phase(&mut self) {
        if R == 0 {
            // No release segments, go directly to complete (Q8 minimum = silence)
            self.state.phase = EnvelopePhase::Complete;
            self.state.level = LEVEL_MIN;
            self.state.level_precise = LEVEL_MIN as f32;
        } else {
            self.state.phase = EnvelopePhase::Release;
            self.state.segment_index = 0;
            self.setup_segment(0, false);
        }
    }

    /// Setup a segment for processing.
    ///
    /// Uses f32 increments for precise timing with sub-sample accuracy.
    /// Both attack and decay use additive increments in the Q8 domain,
    /// which matches MSFA/DX7 behavior (additive in log domain).
    ///
    /// The fractional accumulator approach allows slow rates to achieve
    /// proper multi-second timing (matching STATICS table) while maintaining
    /// authentic Q8 amplitude quantization.
    fn setup_segment(&mut self, segment_index: u8, is_key_on: bool) {
        let segment = if is_key_on {
            &self.config.key_on_segments[segment_index as usize]
        } else {
            &self.config.release_segments[segment_index as usize]
        };

        // Calculate target level from segment in Q8 format
        self.state.target_level = param_to_level_q8(segment.level);

        // Calculate the distance to travel in Q8 units
        let distance = (self.state.target_level as f32 - self.state.level_precise).abs();

        // If already at target, no movement needed
        if distance < 0.5 {
            self.state.increment = 0.0;
            self.state.rising = false;
            return;
        }

        // Calculate per-sample increment with rate scaling applied.
        // The f32 increment is for full 0.0-1.0 range; convert to Q8 units.
        // Uses pre-computed max_rate to avoid O(100) search per segment change.
        let full_range_increment_f32 = calculate_increment_f32_with_max_rate(
            segment.rate,
            self.state.midi_note,
            self.config.rate_scaling,
            self.config.sample_rate,
            self.config.max_rate(),
        );

        // Convert f32 increment (0.0-1.0 range) to Q8 units (0.0-4095.0 range)
        // No minimum clamping - f32 allows fractional increments for slow rates
        let increment_q8 = full_range_increment_f32 * LEVEL_MAX as f32;

        // Determine direction
        self.state.rising = self.state.target_level > self.state.level;

        if self.state.rising {
            // For attack (rising) segments, use MSFA-style exponential approach.
            // The increment is scaled by remaining distance to max, creating
            // the characteristic FM "snap" - fast through quiet region, slowing at top.
            // Divisor 6 compensates for average factor ~5.7 over the 1716→4095 range
            // (attacks start from JUMP_TARGET_Q8, not from 0).
            self.state.increment = increment_q8 / 6.0;
        } else {
            // For decay (falling) segments, use additive decrement in Q8 domain.
            // In log domain, additive decrement = exponential decay in linear.
            // This matches authentic DX7/SY99 behavior.
            self.state.increment = -increment_q8;
        }
    }

    // =========================================================================
    // Internal: Segment Advancement
    // =========================================================================

    /// Advance to next key-on segment.
    fn advance_segment(&mut self) {
        // Snap to target (both Q8 and precise)
        self.state.level = self.state.target_level;
        self.state.level_precise = self.state.target_level as f32;

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
        // Snap to target (both Q8 and precise)
        self.state.level = self.state.target_level;
        self.state.level_precise = self.state.target_level as f32;

        let next_segment = self.state.segment_index + 1;

        if (next_segment as usize) < R {
            // More release segments
            self.state.segment_index = next_segment;
            self.setup_segment(next_segment, false);
        } else {
            // Release complete - set to Q8 minimum (silence)
            self.state.phase = EnvelopePhase::Complete;
            self.state.level = LEVEL_MIN;
            self.state.level_precise = LEVEL_MIN as f32;
        }
    }

    // =========================================================================
    // Internal: Increment Application
    // =========================================================================

    /// MSFA-faithful exponential approach for attack (rising) phases.
    ///
    /// Uses remaining distance to max to create the characteristic FM "snap" -
    /// fast through quiet region, slowing at top. This matches the DX7/MSFA
    /// formula: `level += ((max - level) >> shift) * inc`
    ///
    /// Uses level_precise (f32) for sub-sample timing accuracy, then
    /// quantizes to Q8 (i16) for authentic amplitude representation.
    #[inline]
    fn apply_attack_increment(&mut self) {
        // Factor based on remaining distance to max (not to target)
        // This creates the characteristic FM "snap" - fast at low levels, slow at top
        // Using f32 version of the shift: remaining / 256.0 gives 0.0-16.0 range, +1 gives 1.0-17.0
        let remaining_to_max = LEVEL_MAX as f32 - self.state.level_precise;
        let factor = (remaining_to_max / 256.0) + 1.0;

        // Apply increment scaled by factor
        // Increment was already scaled by 1/9 in setup_segment
        // to compensate for the average factor being ~9
        let delta = self.state.increment * factor;
        self.state.level_precise =
            (self.state.level_precise + delta).min(self.state.target_level as f32);

        // Quantize to Q8 for output
        self.state.level = self.state.level_precise as i16;
    }

    /// Additive decrement for falling phases (exponential in linear domain).
    ///
    /// In Q8 (log) domain, additive decrement produces exponential decay
    /// in linear amplitude domain. This matches authentic DX7/SY99 behavior.
    ///
    /// Uses level_precise (f32) for sub-sample timing accuracy, then
    /// quantizes to Q8 (i16) for authentic amplitude representation.
    #[inline]
    fn apply_decay_increment(&mut self) {
        // Additive decrement in Q8 domain (increment is negative for decay)
        // f32 allows fractional increments for slow rates
        self.state.level_precise =
            (self.state.level_precise + self.state.increment).max(LEVEL_MIN as f32);

        // Quantize to Q8 for output
        self.state.level = self.state.level_precise as i16;
    }

    /// Check if we've reached the target level.
    ///
    /// Uses level_precise for accurate detection with fractional increments.
    #[inline]
    fn reached_target(&self) -> bool {
        if self.state.rising {
            self.state.level_precise >= self.state.target_level as f32
        } else {
            self.state.level_precise <= self.state.target_level as f32
        }
    }

    // =========================================================================
    // Control-Rate Support
    // =========================================================================

    /// Advance envelope by multiple samples in a single step.
    ///
    /// This is an optimized method for control-rate processing that advances
    /// the envelope state by `samples` worth of change without iterating
    /// through each sample individually.
    ///
    /// For decay phases, this uses direct multiplication since the increment
    /// is constant. For attack phases, it uses an average factor approximation
    /// to account for the distance-dependent scaling.
    ///
    /// # Arguments
    ///
    /// * `samples` - Number of samples to advance (typically 32-128)
    ///
    /// # Note
    ///
    /// Segment transitions are checked after advancement, so timing may be
    /// up to `samples-1` samples late. This is acceptable for control-rate
    /// modulation (~1.45ms at 64 samples, 44.1kHz).
    pub fn advance_by(&mut self, samples: u32) {
        match self.state.phase {
            EnvelopePhase::Idle | EnvelopePhase::Complete => {
                // Nothing to do
            }
            EnvelopePhase::Delay => {
                self.advance_delay(samples);
            }
            EnvelopePhase::KeyOn => {
                self.advance_key_on(samples);
            }
            EnvelopePhase::Sustain => {
                // Sustain holds level, nothing to advance
            }
            EnvelopePhase::Release => {
                self.advance_release(samples);
            }
        }
    }

    /// Advance through delay phase by multiple samples.
    #[inline]
    fn advance_delay(&mut self, samples: u32) {
        if self.state.delay_remaining > samples {
            self.state.delay_remaining -= samples;
        } else {
            // Delay complete, start attack with remaining samples
            let remaining = samples - self.state.delay_remaining;
            self.start_key_on_phase();
            if remaining > 0 {
                self.advance_key_on(remaining);
            }
        }
    }

    /// Advance key-on phase by multiple samples using average factor approximation.
    #[inline]
    fn advance_key_on(&mut self, samples: u32) {
        if self.state.rising {
            // Attack: use average factor approximation
            self.advance_attack(samples);
        } else {
            // Decay: direct multiplication (constant increment)
            self.advance_decay(samples);
        }

        // Check for segment transition
        if self.reached_target() {
            self.advance_segment();
        }
    }

    /// Advance release phase by multiple samples.
    #[inline]
    fn advance_release(&mut self, samples: u32) {
        // Release always decays (constant increment)
        self.advance_decay(samples);

        // Check for segment transition
        if self.reached_target() {
            self.advance_release_segment();
        }
    }

    /// Advance attack (rising) phase using average factor approximation.
    ///
    /// The attack increment is scaled by `(remaining_to_max / 256) + 1`,
    /// which varies as the level rises. We approximate by using the average
    /// of the start and estimated end factors.
    #[inline]
    fn advance_attack(&mut self, samples: u32) {
        let samples_f32 = samples as f32;

        // Calculate factor at current position
        let remaining_start = LEVEL_MAX as f32 - self.state.level_precise;
        let factor_start = (remaining_start / 256.0) + 1.0;

        // Estimate where we'll end up using start factor
        let estimated_delta = self.state.increment * factor_start * samples_f32;
        let estimated_end = (self.state.level_precise + estimated_delta).min(LEVEL_MAX as f32);

        // Calculate factor at estimated end position
        let remaining_end = LEVEL_MAX as f32 - estimated_end;
        let factor_end = (remaining_end / 256.0) + 1.0;

        // Use average of start and end factors
        let avg_factor = (factor_start + factor_end) * 0.5;

        // Apply averaged increment
        let delta = self.state.increment * avg_factor * samples_f32;
        self.state.level_precise =
            (self.state.level_precise + delta).min(self.state.target_level as f32);

        // Quantize to Q8
        self.state.level = self.state.level_precise as i16;
    }

    /// Advance decay (falling) phase by direct multiplication.
    ///
    /// Decay uses constant increment, so we can simply multiply.
    #[inline]
    fn advance_decay(&mut self, samples: u32) {
        // Increment is negative for decay
        let delta = self.state.increment * samples as f32;
        self.state.level_precise =
            (self.state.level_precise + delta).max(self.state.target_level as f32);

        // Quantize to Q8
        self.state.level = self.state.level_precise as i16;
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
        let mut first_change_at = 0u32;
        let mut final_phase = env.phase();

        for i in 0..44100 {
            env.process();
            let current = env.current_segment();
            if current != last_segment {
                if segment_changes == 0 {
                    first_change_at = i;
                }
                segment_changes += 1;
                last_segment = current;
            }
            final_phase = env.phase();
        }

        // Should have transitioned through at least some segments
        assert!(
            segment_changes > 0,
            "No segment transitions occurred. Final phase: {:?}, final segment: {}, level: {}",
            final_phase,
            env.current_segment(),
            env.state().level_q8()
        );

        // Verify we reached sustain or later
        assert!(
            segment_changes >= 5,
            "Expected at least 5 segment changes (0→1→2→3→4→5), got {}. First change at sample {}",
            segment_changes,
            first_change_at
        );
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
