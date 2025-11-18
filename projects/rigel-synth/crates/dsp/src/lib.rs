#![no_std]

//! # Rigel DSP Core
//!
//! No-std monophonic DSP core for the Rigel wavetable synthesizer.
//! Provides fast, deterministic audio processing components.

use core::f32::consts::TAU;

/// Sample rate type
pub type SampleRate = f32;

/// Audio sample type
pub type Sample = f32;

/// MIDI note number (0-127)
pub type NoteNumber = u8;

/// MIDI velocity (0.0 to 1.0)
pub type Velocity = f32;

/// Simple synthesis parameters
#[derive(Debug, Clone, Copy)]
pub struct SynthParams {
    /// Master volume (0.0 to 1.0)
    pub volume: f32,
    /// Pitch offset in semitones (-24.0 to 24.0)
    pub pitch_offset: f32,
    /// Envelope attack time in seconds
    pub env_attack: f32,
    /// Envelope decay time in seconds
    pub env_decay: f32,
    /// Envelope sustain level (0.0 to 1.0)
    pub env_sustain: f32,
    /// Envelope release time in seconds
    pub env_release: f32,
}

impl Default for SynthParams {
    fn default() -> Self {
        Self {
            volume: 0.7,
            pitch_offset: 0.0,
            env_attack: 0.01,
            env_decay: 0.3,
            env_sustain: 0.7,
            env_release: 0.5,
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
        1.0 - libm::expf(-(sample - 1.0))
    } else if sample < -1.0 {
        -1.0 + libm::expf(sample + 1.0)
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

/// Envelope stages
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EnvelopeStage {
    Idle,
    Attack,
    Decay,
    Sustain,
    Release,
}

/// Simple ADSR envelope
#[derive(Debug, Clone, Copy)]
pub struct Envelope {
    stage: EnvelopeStage,
    current_value: f32,
    samples_in_stage: u32,
    sample_rate: f32,
}

impl Envelope {
    /// Create new envelope
    pub fn new(sample_rate: f32) -> Self {
        Self {
            stage: EnvelopeStage::Idle,
            current_value: 0.0,
            samples_in_stage: 0,
            sample_rate,
        }
    }

    /// Trigger note on
    pub fn note_on(&mut self) {
        self.stage = EnvelopeStage::Attack;
        self.samples_in_stage = 0;
    }

    /// Trigger note off
    pub fn note_off(&mut self) {
        if self.stage != EnvelopeStage::Idle {
            self.stage = EnvelopeStage::Release;
            self.samples_in_stage = 0;
        }
    }

    /// Process one sample
    pub fn process_sample(&mut self, params: &SynthParams) -> f32 {
        match self.stage {
            EnvelopeStage::Idle => {
                self.current_value = 0.0;
            }
            EnvelopeStage::Attack => {
                let attack_samples = (params.env_attack * self.sample_rate) as u32;
                if attack_samples > 0 {
                    let progress = self.samples_in_stage as f32 / attack_samples as f32;
                    self.current_value = progress.min(1.0);

                    if self.samples_in_stage >= attack_samples {
                        self.stage = EnvelopeStage::Decay;
                        self.samples_in_stage = 0;
                    } else {
                        self.samples_in_stage += 1;
                    }
                } else {
                    self.current_value = 1.0;
                    self.stage = EnvelopeStage::Decay;
                    self.samples_in_stage = 0;
                }
            }
            EnvelopeStage::Decay => {
                let decay_samples = (params.env_decay * self.sample_rate) as u32;
                if decay_samples > 0 {
                    let progress = self.samples_in_stage as f32 / decay_samples as f32;
                    self.current_value = lerp(1.0, params.env_sustain, progress.min(1.0));

                    if self.samples_in_stage >= decay_samples {
                        self.stage = EnvelopeStage::Sustain;
                        self.samples_in_stage = 0;
                    } else {
                        self.samples_in_stage += 1;
                    }
                } else {
                    self.current_value = params.env_sustain;
                    self.stage = EnvelopeStage::Sustain;
                    self.samples_in_stage = 0;
                }
            }
            EnvelopeStage::Sustain => {
                self.current_value = params.env_sustain;
            }
            EnvelopeStage::Release => {
                let release_samples = (params.env_release * self.sample_rate) as u32;
                if release_samples > 0 {
                    let progress = self.samples_in_stage as f32 / release_samples as f32;
                    let sustain_at_release_start = if self.samples_in_stage == 0 {
                        self.current_value // Capture the value when release started
                    } else {
                        params.env_sustain // Fallback if we missed it somehow
                    };

                    self.current_value = lerp(sustain_at_release_start, 0.0, progress.min(1.0));

                    if self.samples_in_stage >= release_samples || self.current_value <= 0.001 {
                        self.stage = EnvelopeStage::Idle;
                        self.current_value = 0.0;
                        self.samples_in_stage = 0;
                    } else {
                        self.samples_in_stage += 1;
                    }
                } else {
                    self.stage = EnvelopeStage::Idle;
                    self.current_value = 0.0;
                }
            }
        }

        self.current_value
    }

    /// Reset envelope
    pub fn reset(&mut self) {
        self.stage = EnvelopeStage::Idle;
        self.current_value = 0.0;
        self.samples_in_stage = 0;
    }

    /// Get current stage
    pub fn current_stage(&self) -> EnvelopeStage {
        self.stage
    }

    /// Check if envelope is active
    pub fn is_active(&self) -> bool {
        self.stage != EnvelopeStage::Idle
    }
}

/// Monophonic synthesis engine
#[derive(Debug, Clone, Copy)]
pub struct SynthEngine {
    oscillator: SimpleOscillator,
    envelope: Envelope,
    sample_rate: f32,
    current_note: Option<NoteNumber>,
    current_velocity: f32,
    master_volume: f32,
}

impl SynthEngine {
    /// Create new synthesis engine
    pub fn new(sample_rate: f32) -> Self {
        Self {
            oscillator: SimpleOscillator::new(),
            envelope: Envelope::new(sample_rate),
            sample_rate,
            current_note: None,
            current_velocity: 0.0,
            master_volume: 0.7,
        }
    }

    /// Start playing a note
    pub fn note_on(&mut self, note: NoteNumber, velocity: Velocity) {
        self.current_note = Some(note);
        self.current_velocity = velocity;

        let frequency = midi_to_freq(note);
        self.oscillator.set_frequency(frequency, self.sample_rate);
        self.envelope.note_on();
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
        // Update pitch if we have a current note
        if let Some(note) = self.current_note {
            let base_freq = midi_to_freq(note);
            let pitch_factor = libm::powf(2.0, params.pitch_offset / 12.0);
            self.oscillator
                .set_frequency(base_freq * pitch_factor, self.sample_rate);
        }

        // Generate audio
        let osc_output = self.oscillator.process_sample();
        let env_value = self.envelope.process_sample(params);

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
}

impl Default for SynthEngine {
    fn default() -> Self {
        Self::new(44100.0)
    }
}
