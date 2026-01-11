//! Envelope state and phase definitions.
//!
//! This module defines:
//! - [`EnvelopePhase`] - Operational phases of the envelope
//! - [`EnvelopeState`] - Runtime mutable state
//! - [`EnvelopeLevel`] - Q8 fixed-point level type

/// Envelope level type in Q8 fixed-point format (matches DX7 hardware).
///
/// Range: 0 to 4095 (12 bits used, ~96dB dynamic range)
/// Conversion to linear: `2^(level / 256.0)`
///
/// Q8 format means 256 steps = 6dB (one amplitude doubling).
/// This provides ~0.0234 dB per step resolution.
pub type EnvelopeLevel = i16;

/// Maximum envelope level (0dB, full amplitude)
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
/// Uses i16/Q8 fixed-point format for hardware-authentic behavior.
///
/// # Memory Layout
///
/// Designed to be compact (16 bytes) for cache efficiency when
/// processing 1536+ concurrent envelopes.
#[derive(Debug, Clone, Copy)]
pub struct EnvelopeState {
    /// Current level in Q8 fixed-point format (0-4095).
    /// Represents log2 amplitude (256 units = 6dB).
    pub(crate) level: i16,

    /// Target level for current segment (Q8).
    pub(crate) target_level: i16,

    /// Level change per sample (Q8, signed).
    /// Positive for rising, negative for falling.
    pub(crate) increment: i16,

    /// Current segment index.
    /// 0..(K-1) for key-on, K..(K+R-1) for release.
    pub(crate) segment_index: u8,

    /// Current envelope phase.
    pub(crate) phase: EnvelopePhase,

    /// True if level is rising toward target.
    pub(crate) rising: bool,

    /// Remaining delay samples (counts down to 0).
    pub(crate) delay_remaining: u32,

    /// Scaled qRate for current segment (with rate scaling applied).
    pub(crate) current_qrate: u8,

    /// MIDI note for rate scaling (cached from note_on).
    pub(crate) midi_note: u8,
}

impl Default for EnvelopeState {
    fn default() -> Self {
        Self {
            level: LEVEL_MIN,
            target_level: LEVEL_MIN,
            increment: 0,
            segment_index: 0,
            phase: EnvelopePhase::Idle,
            rising: false,
            delay_remaining: 0,
            current_qrate: 0,
            midi_note: 60, // Middle C default
        }
    }
}

impl EnvelopeState {
    /// Get current level in Q8 format (0-4095).
    #[inline]
    pub fn level_q8(&self) -> i16 {
        self.level
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
        assert_eq!(LEVEL_MIN, 0);
        assert_eq!(LEVEL_MAX, 4095);
        // 4095 / 256 = ~16 octaves of range (~96dB)
    }
}
