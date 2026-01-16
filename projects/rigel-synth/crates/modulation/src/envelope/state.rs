//! Envelope state and phase definitions.
//!
//! This module defines:
//! - [`EnvelopePhase`] - Operational phases of the envelope
//! - [`EnvelopeState`] - Runtime mutable state
//! - [`EnvelopeLevel`] - Linear amplitude level type (f32)

/// Envelope level type in linear amplitude format.
///
/// Range: 0.0 to 1.0 (full dynamic range)
/// This provides arbitrary precision for envelope timing.
pub type EnvelopeLevel = f32;

/// Maximum envelope level (full amplitude)
pub const LEVEL_MAX: f32 = 1.0;

/// Minimum envelope level (silence)
pub const LEVEL_MIN: f32 = 0.0;

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
/// Uses f32 floating-point format for precise timing at any duration.
///
/// # Memory Layout
///
/// Slightly larger than Q8 version but provides much better timing accuracy.
#[derive(Debug, Clone, Copy)]
pub struct EnvelopeState {
    /// Current level in linear amplitude (0.0-1.0).
    pub(crate) level: f32,

    /// Target level for current segment (0.0-1.0).
    pub(crate) target_level: f32,

    /// Level change per sample (signed).
    /// Positive for rising, negative for falling.
    /// For attack phases only; decay uses decay_factor.
    pub(crate) increment: f32,

    /// Multiplicative decay factor for exponential decay.
    /// Each sample: level *= decay_factor (where 0 < decay_factor < 1).
    /// This produces linear-in-dB decay (exponential in linear amplitude),
    /// matching authentic DX7/SY99 behavior.
    /// Only meaningful when `rising` is false.
    pub(crate) decay_factor: f32,

    /// Starting level for current attack segment.
    /// Used to calculate exponential approach factor.
    /// Only meaningful when `rising` is true.
    pub(crate) attack_base_level: f32,

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
            target_level: LEVEL_MIN,
            increment: 0.0,
            decay_factor: 1.0, // No decay by default
            attack_base_level: LEVEL_MIN,
            segment_index: 0,
            phase: EnvelopePhase::Idle,
            rising: false,
            delay_remaining: 0,
            midi_note: 60, // Middle C default
        }
    }
}

impl EnvelopeState {
    /// Get current level in linear amplitude (0.0-1.0).
    #[inline]
    pub fn level(&self) -> f32 {
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
        assert!((state.level - LEVEL_MIN).abs() < f32::EPSILON);
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
        assert!((LEVEL_MIN - 0.0).abs() < f32::EPSILON);
        assert!((LEVEL_MAX - 1.0).abs() < f32::EPSILON);
    }
}
