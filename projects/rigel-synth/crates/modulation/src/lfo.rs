//! LFO implementation.

use crate::rate::LfoRateMode;
use crate::rng::Rng;
use crate::traits::ModulationSource;
use crate::waveshape::LfoWaveshape;
use rigel_timing::Timebase;

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

/// Low Frequency Oscillator for parameter modulation.
///
/// The LFO generates periodic waveforms that can modulate synthesizer parameters.
/// It supports multiple waveshapes, tempo synchronization, and phase control.
///
/// # Real-Time Safety
///
/// All LFO operations are real-time safe:
/// - No heap allocations
/// - Constant-time operations
/// - Copy/Clone semantics
///
/// # Example
///
/// ```ignore
/// use rigel_modulation::{Lfo, LfoWaveshape, LfoRateMode, ModulationSource};
/// use rigel_timing::Timebase;
///
/// let mut lfo = Lfo::new();
/// lfo.set_waveshape(LfoWaveshape::Sine);
/// lfo.set_rate(LfoRateMode::Hz(2.0));
///
/// let mut timebase = Timebase::new(44100.0);
/// timebase.advance_block(64);
///
/// lfo.update(&timebase);
/// let modulation = lfo.value(); // Returns value in [-1.0, 1.0]
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Lfo {
    // Configuration (set once or via setters)
    waveshape: LfoWaveshape,
    rate_mode: LfoRateMode,
    phase_mode: LfoPhaseMode,
    polarity: LfoPolarity,
    start_phase: f32,
    pulse_width: f32,

    // Runtime state (changes during processing)
    phase: f32,
    current_value: f32,
    rng: Rng,
    held_value: f32,
}

impl Lfo {
    // ─────────────────────────────────────────────────────────────────────
    // Construction
    // ─────────────────────────────────────────────────────────────────────

    /// Create a new LFO with default settings.
    ///
    /// Defaults:
    /// - Waveshape: Sine
    /// - Rate: 1.0 Hz
    /// - Phase mode: FreeRunning
    /// - Polarity: Bipolar
    /// - Start phase: 0.0
    /// - Pulse width: 0.5
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an LFO with specific configuration.
    ///
    /// # Arguments
    /// * `waveshape` - The waveform shape to generate
    /// * `rate_mode` - How the LFO rate is determined (Hz or tempo-sync)
    /// * `phase_mode` - Whether phase resets on note triggers
    /// * `polarity` - Output range (bipolar or unipolar)
    pub fn with_config(
        waveshape: LfoWaveshape,
        rate_mode: LfoRateMode,
        phase_mode: LfoPhaseMode,
        polarity: LfoPolarity,
    ) -> Self {
        Self {
            waveshape,
            rate_mode,
            phase_mode,
            polarity,
            start_phase: 0.0,
            pulse_width: 0.5,
            phase: 0.0,
            current_value: 0.0,
            rng: Rng::default(),
            held_value: 0.0,
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // Configuration Setters
    // ─────────────────────────────────────────────────────────────────────

    /// Set the LFO waveshape.
    pub fn set_waveshape(&mut self, waveshape: LfoWaveshape) {
        self.waveshape = waveshape;
    }

    /// Set the LFO rate mode.
    ///
    /// # Arguments
    /// * `rate` - Either a fixed Hz value or tempo-synchronized division
    ///
    /// # Panics
    /// Panics in debug builds if Hz rate is outside [0.01, 100.0] range.
    pub fn set_rate(&mut self, rate: LfoRateMode) {
        if let LfoRateMode::Hz(hz) = rate {
            debug_assert!(
                (0.01..=100.0).contains(&hz),
                "Hz rate must be in [0.01, 100.0], got {}",
                hz
            );
        }
        self.rate_mode = rate;
    }

    /// Set the phase mode (free-running or retrigger).
    pub fn set_phase_mode(&mut self, mode: LfoPhaseMode) {
        self.phase_mode = mode;
    }

    /// Set the output polarity.
    pub fn set_polarity(&mut self, polarity: LfoPolarity) {
        self.polarity = polarity;
    }

    /// Set the starting phase for reset/retrigger.
    ///
    /// # Arguments
    /// * `phase` - Normalized phase [0.0, 1.0] where 0.0 = 0 deg, 0.5 = 180 deg, etc.
    ///
    /// # Panics
    /// Panics in debug builds if phase is outside [0.0, 1.0] range.
    pub fn set_start_phase(&mut self, phase: f32) {
        debug_assert!(
            (0.0..=1.0).contains(&phase),
            "Start phase must be in [0.0, 1.0], got {}",
            phase
        );
        self.start_phase = phase;
    }

    /// Set the pulse width for Pulse waveshape.
    ///
    /// # Arguments
    /// * `width` - Duty cycle [0.01, 0.99] where 0.5 = square wave
    ///
    /// # Panics
    /// Panics in debug builds if width is outside [0.01, 0.99] range.
    pub fn set_pulse_width(&mut self, width: f32) {
        debug_assert!(
            (0.01..=0.99).contains(&width),
            "Pulse width must be in [0.01, 0.99], got {}",
            width
        );
        self.pulse_width = width;
    }

    /// Set the tempo (BPM) for tempo-sync mode.
    ///
    /// # Arguments
    /// * `bpm` - Beats per minute [1.0, 999.0]
    ///
    /// Note: This only affects the rate when in TempoSync mode.
    pub fn set_tempo(&mut self, bpm: f32) {
        if let LfoRateMode::TempoSync { division, .. } = self.rate_mode {
            self.rate_mode = LfoRateMode::TempoSync { division, bpm };
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // Configuration Getters
    // ─────────────────────────────────────────────────────────────────────

    /// Get the current waveshape.
    pub fn waveshape(&self) -> LfoWaveshape {
        self.waveshape
    }

    /// Get the current rate mode.
    pub fn rate_mode(&self) -> LfoRateMode {
        self.rate_mode
    }

    /// Get the current phase mode.
    pub fn phase_mode(&self) -> LfoPhaseMode {
        self.phase_mode
    }

    /// Get the current polarity.
    pub fn polarity(&self) -> LfoPolarity {
        self.polarity
    }

    /// Get the starting phase.
    pub fn start_phase(&self) -> f32 {
        self.start_phase
    }

    /// Get the pulse width.
    pub fn pulse_width(&self) -> f32 {
        self.pulse_width
    }

    // ─────────────────────────────────────────────────────────────────────
    // Runtime State
    // ─────────────────────────────────────────────────────────────────────

    /// Get the current phase position.
    ///
    /// # Returns
    /// Phase in [0.0, 1.0) representing position in the cycle.
    pub fn phase(&self) -> f32 {
        self.phase
    }

    /// Get the effective rate in Hz.
    ///
    /// For Hz mode, returns the configured rate.
    /// For TempoSync mode, calculates rate from BPM and division.
    pub fn effective_rate_hz(&self) -> f32 {
        self.rate_mode.effective_hz()
    }

    // ─────────────────────────────────────────────────────────────────────
    // Triggering
    // ─────────────────────────────────────────────────────────────────────

    /// Trigger phase reset.
    ///
    /// Call this on note-on events. The effect depends on phase_mode:
    /// - FreeRunning: No effect
    /// - Retrigger: Phase resets to start_phase
    ///
    /// For sample-and-hold waveshape, also samples a new random value.
    pub fn trigger(&mut self) {
        match self.phase_mode {
            LfoPhaseMode::FreeRunning => {
                // No effect - phase continues
            }
            LfoPhaseMode::Retrigger => {
                self.phase = self.start_phase;
                // Sample new S&H value on trigger
                if self.waveshape == LfoWaveshape::SampleAndHold {
                    self.held_value = self.rng.next_f32();
                }
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // Internal Helpers
    // ─────────────────────────────────────────────────────────────────────

    /// Apply polarity scaling to a bipolar value.
    #[inline]
    fn apply_polarity(&self, bipolar_value: f32) -> f32 {
        match self.polarity {
            LfoPolarity::Bipolar => bipolar_value,
            LfoPolarity::Unipolar => (bipolar_value + 1.0) * 0.5,
        }
    }
}

impl Default for Lfo {
    fn default() -> Self {
        Self {
            waveshape: LfoWaveshape::Sine,
            rate_mode: LfoRateMode::Hz(1.0),
            phase_mode: LfoPhaseMode::FreeRunning,
            polarity: LfoPolarity::Bipolar,
            start_phase: 0.0,
            pulse_width: 0.5,
            phase: 0.0,
            current_value: 0.0,
            rng: Rng::new(0x12345678),
            held_value: 0.0,
        }
    }
}

impl ModulationSource for Lfo {
    fn reset(&mut self, _timebase: &Timebase) {
        self.phase = self.start_phase;
        // Sample initial S&H value
        if self.waveshape == LfoWaveshape::SampleAndHold {
            self.held_value = self.rng.next_f32();
        }
        // Generate initial value
        let noise_value = if self.waveshape == LfoWaveshape::Noise {
            self.rng.next_f32()
        } else {
            0.0
        };
        let bipolar =
            self.waveshape
                .generate(self.phase, self.pulse_width, self.held_value, noise_value);
        self.current_value = self.apply_polarity(bipolar);
    }

    fn update(&mut self, timebase: &Timebase) {
        let elapsed_samples = timebase.block_size() as f32;
        let rate_hz = self.effective_rate_hz();
        let sample_rate = timebase.sample_rate();

        // Calculate phase increment based on elapsed time
        let phase_increment = rate_hz * elapsed_samples / sample_rate;

        // Store old phase to detect cycle wrap
        let old_phase = self.phase;

        // Advance phase and wrap to [0.0, 1.0)
        let new_phase = self.phase + phase_increment;
        self.phase = new_phase - libm::floorf(new_phase);

        // Handle negative phase (shouldn't happen, but defensive)
        if self.phase < 0.0 {
            self.phase += 1.0;
        }

        // Detect cycle wrap for S&H
        let wrapped = self.phase < old_phase;
        if wrapped && self.waveshape == LfoWaveshape::SampleAndHold {
            self.held_value = self.rng.next_f32();
        }

        // Generate noise value if needed
        let noise_value = if self.waveshape == LfoWaveshape::Noise {
            self.rng.next_f32()
        } else {
            0.0
        };

        // Generate new value based on waveshape
        let bipolar =
            self.waveshape
                .generate(self.phase, self.pulse_width, self.held_value, noise_value);
        self.current_value = self.apply_polarity(bipolar);
    }

    fn value(&self) -> f32 {
        self.current_value
    }
}
