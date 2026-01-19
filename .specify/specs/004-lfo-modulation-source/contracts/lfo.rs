//! LFO API contracts for rigel-modulation crate.
//!
//! This file defines the public API for the LFO implementation.

use crate::{LfoPhaseMode, LfoPolarity, LfoRateMode, LfoWaveshape, ModulationSource};
use rigel_timing::Timebase;

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
///
/// // In audio callback:
/// timebase.advance_block(64);
/// lfo.update(&timebase);
/// let modulation = lfo.value(); // Returns value in [-1.0, 1.0]
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Lfo {
    // Implementation details hidden
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
    pub fn new() -> Self;

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
    ) -> Self;

    // ─────────────────────────────────────────────────────────────────────
    // Configuration Setters
    // ─────────────────────────────────────────────────────────────────────

    /// Set the LFO waveshape.
    pub fn set_waveshape(&mut self, waveshape: LfoWaveshape);

    /// Set the LFO rate mode.
    ///
    /// # Arguments
    /// * `rate` - Either a fixed Hz value or tempo-synchronized division
    ///
    /// # Panics
    /// Panics if Hz rate is outside [0.01, 100.0] range.
    pub fn set_rate(&mut self, rate: LfoRateMode);

    /// Set the phase mode (free-running or retrigger).
    pub fn set_phase_mode(&mut self, mode: LfoPhaseMode);

    /// Set the output polarity.
    pub fn set_polarity(&mut self, polarity: LfoPolarity);

    /// Set the starting phase for reset/retrigger.
    ///
    /// # Arguments
    /// * `phase` - Normalized phase [0.0, 1.0] where 0.0 = 0°, 0.5 = 180°, etc.
    ///
    /// # Panics
    /// Panics if phase is outside [0.0, 1.0] range.
    pub fn set_start_phase(&mut self, phase: f32);

    /// Set the pulse width for Pulse waveshape.
    ///
    /// # Arguments
    /// * `width` - Duty cycle [0.01, 0.99] where 0.5 = square wave
    ///
    /// # Panics
    /// Panics if width is outside [0.01, 0.99] range.
    pub fn set_pulse_width(&mut self, width: f32);

    /// Set the tempo (BPM) for tempo-sync mode.
    ///
    /// # Arguments
    /// * `bpm` - Beats per minute [1.0, 999.0]
    ///
    /// Note: This only affects the rate when in TempoSync mode.
    pub fn set_tempo(&mut self, bpm: f32);

    // ─────────────────────────────────────────────────────────────────────
    // Configuration Getters
    // ─────────────────────────────────────────────────────────────────────

    /// Get the current waveshape.
    pub fn waveshape(&self) -> LfoWaveshape;

    /// Get the current rate mode.
    pub fn rate_mode(&self) -> LfoRateMode;

    /// Get the current phase mode.
    pub fn phase_mode(&self) -> LfoPhaseMode;

    /// Get the current polarity.
    pub fn polarity(&self) -> LfoPolarity;

    /// Get the starting phase.
    pub fn start_phase(&self) -> f32;

    /// Get the pulse width.
    pub fn pulse_width(&self) -> f32;

    // ─────────────────────────────────────────────────────────────────────
    // Runtime State
    // ─────────────────────────────────────────────────────────────────────

    /// Get the current phase position.
    ///
    /// # Returns
    /// Phase in [0.0, 1.0) representing position in the cycle.
    pub fn phase(&self) -> f32;

    /// Get the effective rate in Hz.
    ///
    /// For Hz mode, returns the configured rate.
    /// For TempoSync mode, calculates rate from BPM and division.
    ///
    /// # Arguments
    /// * `bpm` - Current tempo (only used if in TempoSync mode with stored BPM)
    pub fn effective_rate_hz(&self) -> f32;

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
    pub fn trigger(&mut self);
}

impl Default for Lfo {
    fn default() -> Self {
        Self::new()
    }
}

impl ModulationSource for Lfo {
    fn reset(&mut self);
    fn update(&mut self, timebase: &Timebase);
    fn value(&self) -> f32;
}
