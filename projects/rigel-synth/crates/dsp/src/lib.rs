#![no_std]

//! # Rigel DSP Core
//!
//! No-std monophonic DSP core for the Rigel wavetable synthesizer.
//! Provides fast, deterministic audio processing components.

use core::f32::consts::TAU;
use rigel_math::expf;
use rigel_math::scalar::polyblep_sawtooth;
use rigel_modulation::envelope::{FmEnvelope, FmEnvelopeConfig};
use rigel_simd::DenormalGuard;
use rigel_simd_dispatch::SimdContext;

// Re-export timing infrastructure from rigel-timing for backward compatibility
pub use rigel_timing::{
    ControlRateClock, ControlRateUpdates, Smoother, SmoothingMode, Timebase, DEFAULT_SAMPLE_RATE,
    DEFAULT_SMOOTHING_TIME_MS,
};

// Re-export envelope types for users who want direct access
pub use rigel_modulation::envelope::{
    EnvelopePhase, FmEnvelopeConfig as EnvelopeConfig, Segment, Segment as EnvelopeSegment,
};

// Backward compatibility alias - SegmentParams is now Segment from rigel-modulation
#[deprecated(
    since = "0.2.0",
    note = "Use Segment from rigel_modulation::envelope instead"
)]
pub type SegmentParams = Segment;

/// Sample rate type
pub type SampleRate = f32;

/// Audio sample type
pub type Sample = f32;

/// MIDI note number (0-127)
pub type NoteNumber = u8;

/// MIDI velocity (0.0 to 1.0)
pub type Velocity = f32;

/// Full FM envelope parameters (6 key-on + 2 release segments)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FmEnvelopeParams {
    /// Key-on segments (6 segments: attack through sustain)
    pub key_on: [Segment; 6],
    /// Release segments (2 segments)
    pub release: [Segment; 2],
    /// Rate scaling sensitivity (0-7, higher = more keyboard tracking)
    pub rate_scaling: u8,
    /// Whether envelope looping is enabled
    pub loop_enabled: bool,
    /// Start segment index for loop (0-5)
    pub loop_start: u8,
    /// End segment index for loop (0-5)
    pub loop_end: u8,
}

impl Default for FmEnvelopeParams {
    fn default() -> Self {
        // Default to a typical ADSR-style envelope
        Self {
            key_on: [
                Segment::new(85, 99), // Attack: fast to full
                Segment::new(50, 69), // Decay: medium to sustain (~70%)
                Segment::new(99, 69), // Hold at sustain
                Segment::new(99, 69),
                Segment::new(99, 69),
                Segment::new(99, 69),
            ],
            release: [
                Segment::new(45, 0), // Release: medium to silence
                Segment::new(99, 0), // Immediate if needed
            ],
            rate_scaling: 0,
            loop_enabled: false,
            loop_start: 0,
            loop_end: 1,
        }
    }
}

impl FmEnvelopeParams {
    /// Create from ADSR time-based parameters
    pub fn from_adsr(
        attack_secs: f32,
        decay_secs: f32,
        sustain_linear: f32,
        release_secs: f32,
        sample_rate: f32,
    ) -> Self {
        use rigel_modulation::envelope::{linear_to_param_level, seconds_to_rate};

        let attack_rate = seconds_to_rate(attack_secs, sample_rate);
        let decay_rate = seconds_to_rate(decay_secs, sample_rate);
        let sustain_level = linear_to_param_level(sustain_linear);
        let release_rate = seconds_to_rate(release_secs, sample_rate);

        Self {
            key_on: [
                Segment::new(attack_rate, 99),
                Segment::new(decay_rate, sustain_level),
                Segment::new(99, sustain_level),
                Segment::new(99, sustain_level),
                Segment::new(99, sustain_level),
                Segment::new(99, sustain_level),
            ],
            release: [Segment::new(release_rate, 0), Segment::new(99, 0)],
            rate_scaling: 0,
            loop_enabled: false,
            loop_start: 0,
            loop_end: 1,
        }
    }
}

/// Synthesis parameters with full FM envelope control
#[derive(Debug, Clone, Copy)]
pub struct SynthParams {
    /// Master volume (0.0 to 1.0)
    pub volume: f32,
    /// Pitch offset in semitones (-24.0 to 24.0)
    pub pitch_offset: f32,
    /// FM envelope parameters
    pub envelope: FmEnvelopeParams,
}

impl Default for SynthParams {
    fn default() -> Self {
        Self {
            volume: 0.7,
            pitch_offset: 0.0,
            envelope: FmEnvelopeParams::default(),
        }
    }
}

impl SynthParams {
    /// Create SynthParams from simple ADSR values (for backward compatibility)
    ///
    /// Converts time-based ADSR to FM envelope format.
    pub fn from_adsr(
        volume: f32,
        pitch_offset: f32,
        attack_secs: f32,
        decay_secs: f32,
        sustain_linear: f32,
        release_secs: f32,
        sample_rate: f32,
    ) -> Self {
        Self {
            volume,
            pitch_offset,
            envelope: FmEnvelopeParams::from_adsr(
                attack_secs,
                decay_secs,
                sustain_linear,
                release_secs,
                sample_rate,
            ),
        }
    }
}

/// Convert MIDI note number to frequency in Hz
#[inline]
pub fn midi_to_freq(note: NoteNumber) -> f32 {
    440.0 * libm::powf(2.0, (note as f32 - 69.0) / 12.0)
}

/// Linear interpolation between two values
#[inline]
pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + t * (b - a)
}

/// Clamp value between min and max
#[inline]
pub fn clamp(value: f32, min: f32, max: f32) -> f32 {
    value.clamp(min, max)
}

/// Simple soft clipping
#[inline]
pub fn soft_clip(sample: f32) -> f32 {
    if sample > 1.0 {
        1.0 - expf(-(sample - 1.0))
    } else if sample < -1.0 {
        -1.0 + expf(sample + 1.0)
    } else {
        sample
    }
}

/// Simple sine wave oscillator
#[derive(Debug, Clone, Copy)]
pub struct SimpleOscillator {
    phase: f32,
    phase_increment: f32,
}

impl SimpleOscillator {
    /// Create new oscillator
    pub fn new() -> Self {
        Self {
            phase: 0.0,
            phase_increment: 0.0,
        }
    }

    /// Set frequency and sample rate
    pub fn set_frequency(&mut self, frequency: f32, sample_rate: f32) {
        self.phase_increment = TAU * frequency / sample_rate;
    }

    /// Process one sample (sine wave)
    pub fn process_sample(&mut self) -> f32 {
        let output = libm::sinf(self.phase);

        self.phase += self.phase_increment;
        if self.phase >= TAU {
            self.phase -= TAU;
        }

        output
    }

    /// Reset phase
    pub fn reset(&mut self) {
        self.phase = 0.0;
    }
}

impl Default for SimpleOscillator {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BandlimitedSawOscillator {
    phase: f32,
    phase_increment: f32,
}

impl BandlimitedSawOscillator {
    /// Create new bandlimited sawtooth oscillator
    pub fn new() -> Self {
        Self {
            phase: 0.0,
            phase_increment: 0.0,
        }
    }

    /// Set frequency and sample rate
    pub fn set_frequency(&mut self, frequency: f32, sample_rate: f32) {
        self.phase_increment = frequency / sample_rate;
    }

    /// Process one sample (bandlimited sawtooth wave)
    pub fn process_sample(&mut self) -> f32 {
        // Naive bandlimited sawtooth using PolyBLEP
        let output = polyblep_sawtooth(self.phase, self.phase_increment);

        // Advance phase
        self.phase += self.phase_increment;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }

        output
    }

    /// Reset phase
    pub fn reset(&mut self) {
        self.phase = 0.0;
    }
}

impl Default for BandlimitedSawOscillator {
    fn default() -> Self {
        Self::new()
    }
}

// Note: The old simple Envelope has been replaced by FmEnvelope from rigel-modulation.
// Use EnvelopePhase (re-exported above) instead of the old EnvelopeStage.

/// Monophonic synthesis engine
#[derive(Debug, Clone)]
pub struct SynthEngine {
    oscillator: BandlimitedSawOscillator,
    envelope: FmEnvelope,
    sample_rate: f32,
    current_note: Option<NoteNumber>,
    current_velocity: f32,
    master_volume: f32,
    #[allow(dead_code)] // Will be used for future SIMD block processing
    simd_ctx: SimdContext,
    /// Cached base frequency (from MIDI note) to avoid repeated powf calls
    cached_base_freq: f32,
    /// Last pitch offset used for frequency calculation
    last_pitch_offset: f32,
    /// Cached final frequency (base_freq * pitch_factor)
    cached_frequency: f32,
    /// Sample-accurate timing infrastructure
    timebase: Timebase,
    /// Last envelope params for detecting changes
    last_env_params: FmEnvelopeParams,
}

impl SynthEngine {
    /// Create new synthesis engine
    pub fn new(sample_rate: f32) -> Self {
        // Initialize SIMD context (selects optimal backend for this CPU)
        let simd_ctx = SimdContext::new();

        // Debug logging for backend selection (only in debug builds)
        #[cfg(debug_assertions)]
        {
            // Note: In no_std, we can't use println! or eprintln!
            // This will be logged when the engine is used in std environments
            // For now, we just initialize silently
            let _backend_name = simd_ctx.backend_name();
        }

        // Default envelope params
        let default_env_params = FmEnvelopeParams::default();

        // Create envelope with default config
        let envelope_config = Self::fm_params_to_config(&default_env_params, sample_rate);
        let envelope = FmEnvelope::with_config(envelope_config);

        Self {
            oscillator: BandlimitedSawOscillator::new(),
            envelope,
            sample_rate,
            current_note: None,
            current_velocity: 0.0,
            master_volume: 0.7,
            simd_ctx,
            cached_base_freq: 440.0,
            last_pitch_offset: 0.0,
            cached_frequency: 440.0,
            timebase: Timebase::new(sample_rate),
            last_env_params: default_env_params,
        }
    }

    /// Convert FmEnvelopeParams to FmEnvelopeConfig
    fn fm_params_to_config(params: &FmEnvelopeParams, sample_rate: f32) -> FmEnvelopeConfig {
        use rigel_modulation::envelope::{LoopConfig, Segment};

        // Build loop config from params
        let loop_config = if params.loop_enabled && params.loop_start < params.loop_end {
            LoopConfig::new(params.loop_start, params.loop_end).unwrap_or_else(LoopConfig::disabled)
        } else {
            LoopConfig::disabled()
        };

        FmEnvelopeConfig::new(
            [
                Segment::new(params.key_on[0].rate, params.key_on[0].level),
                Segment::new(params.key_on[1].rate, params.key_on[1].level),
                Segment::new(params.key_on[2].rate, params.key_on[2].level),
                Segment::new(params.key_on[3].rate, params.key_on[3].level),
                Segment::new(params.key_on[4].rate, params.key_on[4].level),
                Segment::new(params.key_on[5].rate, params.key_on[5].level),
            ],
            [
                Segment::new(params.release[0].rate, params.release[0].level),
                Segment::new(params.release[1].rate, params.release[1].level),
            ],
            params.rate_scaling,
            0, // No delay
            loop_config,
            sample_rate,
        )
    }

    /// Get the SIMD backend name (for debugging/logging)
    pub fn simd_backend(&self) -> &'static str {
        self.simd_ctx.backend_name()
    }

    /// Start playing a note
    pub fn note_on(&mut self, note: NoteNumber, velocity: Velocity) {
        self.current_note = Some(note);
        self.current_velocity = velocity;

        // Cache base frequency (only calculated on note-on, not every sample)
        self.cached_base_freq = midi_to_freq(note);
        self.cached_frequency = self.cached_base_freq;
        self.last_pitch_offset = 0.0;

        self.oscillator
            .set_frequency(self.cached_frequency, self.sample_rate);

        // opinionated oscillator phase reset for now
        self.oscillator.reset();

        // Trigger envelope with MIDI note for rate scaling
        self.envelope.note_on(note);
    }

    /// Start playing a note with specific envelope parameters
    ///
    /// This variant allows configuring the envelope at note-on time,
    /// which is more efficient than updating params every sample.
    pub fn note_on_with_params(
        &mut self,
        note: NoteNumber,
        velocity: Velocity,
        params: &SynthParams,
    ) {
        self.current_note = Some(note);
        self.current_velocity = velocity;

        // Cache base frequency
        self.cached_base_freq = midi_to_freq(note);
        self.cached_frequency = self.cached_base_freq;
        self.last_pitch_offset = 0.0;

        self.oscillator
            .set_frequency(self.cached_frequency, self.sample_rate);

        // Update envelope config if params changed
        if params.envelope != self.last_env_params {
            let config = Self::fm_params_to_config(&params.envelope, self.sample_rate);
            self.envelope.set_config(config);
            self.last_env_params = params.envelope;
        }

        // Trigger envelope with MIDI note for rate scaling
        self.envelope.note_on(note);
    }

    /// Stop playing the current note
    pub fn note_off(&mut self, note: NoteNumber) {
        // Only release if it's the current note
        if self.current_note == Some(note) {
            self.envelope.note_off();
        }
    }

    /// Process one sample
    pub fn process_sample(&mut self, params: &SynthParams) -> f32 {
        // Only recalculate frequency when pitch_offset changes (avoids powf every sample)
        if self.current_note.is_some() && params.pitch_offset != self.last_pitch_offset {
            self.last_pitch_offset = params.pitch_offset;
            let pitch_factor = libm::powf(2.0, params.pitch_offset / 12.0);
            self.cached_frequency = self.cached_base_freq * pitch_factor;
            self.oscillator
                .set_frequency(self.cached_frequency, self.sample_rate);
        }

        // Check if envelope params changed and update config
        // Note: This is less efficient than note_on_with_params, but provides
        // real-time parameter updates during playback
        if params.envelope != self.last_env_params {
            let config = Self::fm_params_to_config(&params.envelope, self.sample_rate);
            self.envelope.set_config(config);
            self.last_env_params = params.envelope;
        }

        // Generate audio
        let osc_output = self.oscillator.process_sample();
        let env_value = self.envelope.process();

        // If envelope finished, clear current note
        if !self.envelope.is_active() {
            self.current_note = None;
            self.current_velocity = 0.0;
        }

        // Apply envelope and velocity
        let output = osc_output * env_value * self.current_velocity;

        // Apply master volume and params volume
        let result = output * params.volume * self.master_volume;

        // Soft clipping
        soft_clip(result)
    }

    /// Reset engine
    pub fn reset(&mut self) {
        self.oscillator.reset();
        self.envelope.reset();
        self.current_note = None;
        self.current_velocity = 0.0;
        self.cached_base_freq = 440.0;
        self.last_pitch_offset = 0.0;
        self.cached_frequency = 440.0;
        self.timebase.reset();
        // Reset to default envelope params
        self.last_env_params = FmEnvelopeParams::default();
    }

    /// Check if any note is currently playing
    pub fn is_active(&self) -> bool {
        self.envelope.is_active()
    }

    /// Get the currently playing note (if any)
    pub fn current_note(&self) -> Option<NoteNumber> {
        self.current_note
    }

    /// Set master volume
    pub fn set_master_volume(&mut self, volume: f32) {
        self.master_volume = clamp(volume, 0.0, 1.0);
    }

    /// Get master volume
    pub fn master_volume(&self) -> f32 {
        self.master_volume
    }

    /// Get sample rate
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    /// Process a block of samples with denormal protection
    ///
    /// This is the recommended method for audio processing as it:
    /// 1. Enables FTZ/DAZ flags to prevent denormal slowdowns
    /// 2. Processes samples efficiently in a batch
    /// 3. Advances the timebase for sample-accurate timing
    ///
    /// # Arguments
    ///
    /// * `output` - Buffer to fill with audio samples
    /// * `params` - Synthesis parameters for this block
    ///
    /// # Performance
    ///
    /// Denormal protection prevents 10-100x slowdowns during silence/release.
    #[inline]
    pub fn process_block(&mut self, output: &mut [f32], params: &SynthParams) {
        let _guard = DenormalGuard::new();

        // Advance timebase for this block (enables sample-accurate timing)
        self.timebase.advance_block(output.len() as u32);

        for sample in output.iter_mut() {
            *sample = self.process_sample(params);
        }
    }

    /// Get a reference to the timebase
    ///
    /// The timebase provides sample-accurate timing information for DSP modules.
    pub fn timebase(&self) -> &Timebase {
        &self.timebase
    }

    /// Get a mutable reference to the timebase
    ///
    /// Allows direct modification of timebase (e.g., for reset or sample rate changes).
    pub fn timebase_mut(&mut self) -> &mut Timebase {
        &mut self.timebase
    }

    /// Get the current envelope phase
    ///
    /// Useful for UI display or debugging envelope behavior.
    pub fn envelope_phase(&self) -> EnvelopePhase {
        self.envelope.phase()
    }
}

impl Default for SynthEngine {
    fn default() -> Self {
        Self::new(44100.0)
    }
}
