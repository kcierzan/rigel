//! Envelope state and phase definitions.
//!
//! This module defines:
//! - [`EnvelopePhase`] - Operational phases of the envelope
//! - [`EnvelopeState`] - Runtime mutable state
//! - [`EnvelopeLevel`] - Q8 fixed-point level type (i16)
//!
//! ## Q8 Fixed-Point Format with Fractional Accumulator
//!
//! The envelope uses a hybrid approach for hardware-authentic amplitude
//! representation with precise timing:
//!
//! - **Output levels** are Q8 fixed-point (i16, 0-4095) for authentic amplitude
//! - **Internal accumulation** uses f32 for sub-sample precision timing
//!
//! This allows slow envelope rates to achieve proper multi-second timing
//! while maintaining the 4096-step quantization of real DX7 hardware.
//!
//! Q8 level mapping:
//! - 0 = ~-96dB (silence)
//! - 4095 = 0dB (full amplitude)
//! - Each step ≈ 0.0235 dB

use super::rates::level_to_linear;

/// Envelope level type in Q8 fixed-point format.
///
/// Range: 0 to 4095 (~96dB dynamic range, ~0.0235 dB per step)
/// This provides hardware-authentic amplitude representation.
pub type EnvelopeLevel = i16;

/// Maximum envelope level (full amplitude, 0dB)
pub const LEVEL_MAX: i16 = 4095;

/// Minimum envelope level (silence, ~-96dB)
pub const LEVEL_MIN: i16 = 0;

/// Envelope operational phases.
///
/// Represents the current state in the envelope's lifecycle.
/// State transitions follow a defined state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EnvelopePhase {
    /// Envelope is idle (note off, fully released).
    /// This is the initial state and the state after completion.
    #[default]
    Idle,
    /// Waiting for delay period to complete before attack begins.
    Delay,
    /// Processing key-on segments (attack/decay phases).
    KeyOn,
    /// Holding at sustain level (final key-on segment or loop point).
    Sustain,
    /// Processing release segments after note-off.
    Release,
    /// Envelope completed (reached minimum level after release).
    /// Automatically transitions to Idle.
    Complete,
}

/// Runtime state for envelope processing.
///
/// Contains all mutable state that changes during envelope operation.
/// Uses a hybrid approach: Q8 (i16) for output levels, f32 for timing precision.
///
/// # Fractional Accumulator Design
///
/// The `level_precise` field holds the exact accumulated level as f32 (0.0-4095.0),
/// while `level` is the quantized Q8 output derived from it. This allows:
/// - Sub-sample timing precision for slow rates (multi-second envelopes)
/// - Hardware-authentic 4096-step amplitude quantization for output
#[derive(Debug, Clone, Copy)]
pub struct EnvelopeState {
    /// Current level in Q8 fixed-point (0-4095).
    /// This is the quantized output level, derived from `level_precise`.
    pub(crate) level: i16,

    /// Precise accumulated level (0.0-4095.0).
    /// Allows sub-sample increment precision for accurate timing.
    /// The Q8 `level` is derived as `level_precise as i16`.
    pub(crate) level_precise: f32,

    /// Target level for current segment in Q8 (0-4095).
    pub(crate) target_level: i16,

    /// Level change per sample (f32 for sub-sample precision).
    /// Positive for rising (attack), negative for falling (decay).
    /// Can be fractional (e.g., 0.1) for slow rates.
    pub(crate) increment: f32,

    /// Current segment index.
    /// 0..(K-1) for key-on, 0..(R-1) for release.
    pub(crate) segment_index: u8,

    /// Current envelope phase.
    pub(crate) phase: EnvelopePhase,

    /// True if level is rising toward target.
    pub(crate) rising: bool,

    /// Remaining delay samples (counts down to 0).
    pub(crate) delay_remaining: u32,

    /// MIDI note for rate scaling (cached from note_on).
    pub(crate) midi_note: u8,
}

impl Default for EnvelopeState {
    fn default() -> Self {
        Self {
            level: LEVEL_MIN,
            level_precise: LEVEL_MIN as f32,
            target_level: LEVEL_MIN,
            increment: 0.0,
            segment_index: 0,
            phase: EnvelopePhase::Idle,
            rising: false,
            delay_remaining: 0,
            midi_note: 60, // Middle C default
        }
    }
}

impl EnvelopeState {
    /// Get current level in Q8 fixed-point (0-4095).
    #[inline]
    pub fn level_q8(&self) -> i16 {
        self.level
    }

    /// Get current level converted to linear amplitude (0.0-1.0).
    ///
    /// Converts from Q8 fixed-point to linear amplitude for output.
    #[inline]
    pub fn level(&self) -> f32 {
        level_to_linear(self.level)
    }

    /// Get current phase.
    #[inline]
    pub fn phase(&self) -> EnvelopePhase {
        self.phase
    }

    /// Check if envelope is active (not idle or complete).
    #[inline]
    pub fn is_active(&self) -> bool {
        !matches!(self.phase, EnvelopePhase::Idle | EnvelopePhase::Complete)
    }

    /// Check if envelope is in release phase.
    #[inline]
    pub fn is_releasing(&self) -> bool {
        matches!(self.phase, EnvelopePhase::Release)
    }

    /// Check if envelope has completed.
    #[inline]
    pub fn is_complete(&self) -> bool {
        matches!(self.phase, EnvelopePhase::Complete)
    }

    /// Get current segment index.
    #[inline]
    pub fn segment_index(&self) -> u8 {
        self.segment_index
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_state() {
        let state = EnvelopeState::default();
        assert_eq!(state.level, LEVEL_MIN);
        assert_eq!(state.phase, EnvelopePhase::Idle);
        assert!(!state.is_active());
        assert!(!state.is_releasing());
    }

    #[test]
    fn test_phase_transitions() {
        // Idle -> KeyOn
        let mut state = EnvelopeState {
            phase: EnvelopePhase::KeyOn,
            ..Default::default()
        };
        assert!(state.is_active());
        assert!(!state.is_releasing());

        // KeyOn -> Sustain
        state.phase = EnvelopePhase::Sustain;
        assert!(state.is_active());

        // Sustain -> Release
        state.phase = EnvelopePhase::Release;
        assert!(state.is_active());
        assert!(state.is_releasing());

        // Release -> Complete
        state.phase = EnvelopePhase::Complete;
        assert!(!state.is_active());
        assert!(state.is_complete());
    }

    #[test]
    fn test_level_constants() {
        // Q8 format: 0 = silence, 4095 = full
        assert_eq!(LEVEL_MIN, 0);
        assert_eq!(LEVEL_MAX, 4095);
    }

    #[test]
    fn test_level_to_linear_conversion() {
        // Q8 0 should give exactly 0 (silence)
        let linear_min = level_to_linear(LEVEL_MIN);
        assert_eq!(
            linear_min, 0.0,
            "LEVEL_MIN should be 0.0, got {}",
            linear_min
        );

        // Q8 4095 should give exactly 1.0 (full amplitude)
        let linear_max = level_to_linear(LEVEL_MAX);
        assert_eq!(
            linear_max, 1.0,
            "LEVEL_MAX should be 1.0, got {}",
            linear_max
        );

        // Monotonic: higher Q8 = higher linear
        let linear_low = level_to_linear(1000);
        let linear_mid = level_to_linear(2000);
        let linear_high = level_to_linear(3000);
        assert!(linear_low < linear_mid);
        assert!(linear_mid < linear_high);

        // Midpoint should be roughly -48dB
        let linear_mid_check = level_to_linear(2048);
        assert!(
            linear_mid_check > 0.003 && linear_mid_check < 0.005,
            "Mid-level ~2048 should be around -48dB, got {}",
            linear_mid_check
        );
    }
}
