//! LFO rate mode definitions.

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

impl LfoRateMode {
    /// Get the effective rate in Hz.
    ///
    /// For Hz mode, returns the configured rate.
    /// For TempoSync mode, calculates rate from BPM and division.
    pub fn effective_hz(&self) -> f32 {
        match *self {
            LfoRateMode::Hz(hz) => hz,
            LfoRateMode::TempoSync { division, bpm } => division.to_hz(bpm),
        }
    }
}

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

    /// Create a dotted note division (1.5x duration).
    pub const fn dotted(base: NoteBase) -> Self {
        Self {
            base,
            modifier: NoteModifier::Dotted,
        }
    }

    /// Create a triplet note division (2/3x duration).
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
            // Dotted = 1.5x duration = 2/3 rate
            NoteModifier::Dotted => base_mult * (2.0 / 3.0),
            // Triplet = 2/3 duration = 1.5x rate
            NoteModifier::Triplet => base_mult * 1.5,
        }
    }

    /// Convert to Hz given a tempo.
    ///
    /// # Arguments
    /// * `bpm` - Tempo in beats per minute
    ///
    /// # Returns
    /// Rate in Hz
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
    /// 1.5x duration (slower rate = 2/3x multiplier)
    Dotted,
    /// 2/3x duration (faster rate = 1.5x multiplier)
    Triplet,
}
