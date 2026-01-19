//! Type definitions for rigel-modulation crate.
//!
//! All types are Copy + Clone + Send + Sync for real-time safety.

// ─────────────────────────────────────────────────────────────────────────────
// Waveshape
// ─────────────────────────────────────────────────────────────────────────────

/// Available LFO waveshapes.
///
/// Each waveshape maps phase [0.0, 1.0) to output value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LfoWaveshape {
    /// Smooth sinusoidal oscillation.
    /// Output: sin(phase * 2π)
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

// ─────────────────────────────────────────────────────────────────────────────
// Phase Mode
// ─────────────────────────────────────────────────────────────────────────────

/// LFO phase behavior on note events.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LfoPhaseMode {
    /// Phase continues uninterrupted on note events.
    /// LFO runs independently of note triggers.
    #[default]
    FreeRunning,

    /// Phase resets to start_phase on note-on events.
    /// Ensures consistent modulation from note start.
    Retrigger,
}

// ─────────────────────────────────────────────────────────────────────────────
// Polarity
// ─────────────────────────────────────────────────────────────────────────────

/// LFO output value range.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LfoPolarity {
    /// Bipolar output range [-1.0, 1.0].
    /// Modulation oscillates around zero.
    #[default]
    Bipolar,

    /// Unipolar output range [0.0, 1.0].
    /// Modulation is always positive.
    Unipolar,
}

// ─────────────────────────────────────────────────────────────────────────────
// Rate Mode
// ─────────────────────────────────────────────────────────────────────────────

/// How LFO rate is determined.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LfoRateMode {
    /// Fixed rate in Hz.
    ///
    /// Valid range: [0.01, 100.0] Hz
    Hz(f32),

    /// Tempo-synchronized rate.
    ///
    /// Rate is calculated from BPM and note division.
    TempoSync {
        /// Musical note division
        division: NoteDivision,
        /// Current tempo in BPM (updated externally)
        bpm: f32,
    },
}

impl Default for LfoRateMode {
    fn default() -> Self {
        Self::Hz(1.0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Note Division
// ─────────────────────────────────────────────────────────────────────────────

/// Musical note division for tempo sync.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct NoteDivision {
    /// Base note value (whole, half, quarter, etc.)
    pub base: NoteBase,
    /// Modifier (normal, dotted, triplet)
    pub modifier: NoteModifier,
}

impl NoteDivision {
    /// Create a new note division.
    pub const fn new(base: NoteBase, modifier: NoteModifier) -> Self {
        Self { base, modifier }
    }

    /// Create a normal (unmodified) note division.
    pub const fn normal(base: NoteBase) -> Self {
        Self {
            base,
            modifier: NoteModifier::Normal,
        }
    }

    /// Create a dotted note division (1.5× duration).
    pub const fn dotted(base: NoteBase) -> Self {
        Self {
            base,
            modifier: NoteModifier::Dotted,
        }
    }

    /// Create a triplet note division (2/3× duration).
    pub const fn triplet(base: NoteBase) -> Self {
        Self {
            base,
            modifier: NoteModifier::Triplet,
        }
    }

    /// Get the rate multiplier for this division.
    ///
    /// At 60 BPM (1 beat/second), returns cycles per second.
    pub fn multiplier(&self) -> f32 {
        let base_mult = match self.base {
            NoteBase::Whole => 0.25,
            NoteBase::Half => 0.5,
            NoteBase::Quarter => 1.0,
            NoteBase::Eighth => 2.0,
            NoteBase::Sixteenth => 4.0,
            NoteBase::ThirtySecond => 8.0,
        };

        match self.modifier {
            NoteModifier::Normal => base_mult,
            NoteModifier::Dotted => base_mult * (2.0 / 3.0),
            NoteModifier::Triplet => base_mult * 1.5,
        }
    }

    /// Convert to Hz given a tempo.
    pub fn to_hz(&self, bpm: f32) -> f32 {
        (bpm / 60.0) * self.multiplier()
    }
}

/// Base note value for tempo sync.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum NoteBase {
    /// Whole note (1/1)
    Whole,
    /// Half note (1/2)
    Half,
    /// Quarter note (1/4)
    #[default]
    Quarter,
    /// Eighth note (1/8)
    Eighth,
    /// Sixteenth note (1/16)
    Sixteenth,
    /// Thirty-second note (1/32)
    ThirtySecond,
}

/// Note duration modifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum NoteModifier {
    /// Standard duration
    #[default]
    Normal,
    /// 1.5× duration (slower rate = 2/3× multiplier)
    Dotted,
    /// 2/3× duration (faster rate = 1.5× multiplier)
    Triplet,
}

// ─────────────────────────────────────────────────────────────────────────────
// Random Number Generator
// ─────────────────────────────────────────────────────────────────────────────

/// PCG32 pseudo-random number generator.
///
/// Used internally for sample-and-hold and noise waveshapes.
/// Provides deterministic, reproducible random sequences.
#[derive(Clone, Copy, Debug)]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Create a new RNG with the given seed.
    pub const fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x2C9277B5_27D4EB2D_u64),
        }
    }

    /// Generate the next random u32.
    #[inline(always)]
    pub fn next_u32(&mut self) -> u32 {
        let old_state = self.state;
        self.state = old_state
            .wrapping_mul(6364136223846793005_u64)
            .wrapping_add(1442695040888963407_u64);

        let xor_shifted = (((old_state >> 18) ^ old_state) >> 27) as u32;
        let rot = (old_state >> 59) as u32;
        xor_shifted.rotate_right(rot)
    }

    /// Generate a random f32 in [-1.0, 1.0].
    #[inline(always)]
    pub fn next_f32(&mut self) -> f32 {
        let value = self.next_u32();
        // Use high 24 bits for better distribution
        let normalized = ((value >> 8) as f32) / 16777215.0;
        normalized * 2.0 - 1.0
    }

    /// Generate a random f32 in [0.0, 1.0].
    #[inline(always)]
    pub fn next_f32_unipolar(&mut self) -> f32 {
        let value = self.next_u32();
        ((value >> 8) as f32) / 16777215.0
    }
}

impl Default for Rng {
    fn default() -> Self {
        Self::new(0x12345678)
    }
}
