#![no_std]

//! # Rigel Modulation
//!
//! No-std modulation sources for audio DSP.
//!
//! This crate provides modulation sources for the Rigel synthesizer:
//!
//! - [`Lfo`] - Low Frequency Oscillator with multiple waveshapes
//! - [`ModulationSource`] - Trait interface for all modulation sources
//!
//! All types are `Copy`/`Clone`, zero-allocation, and suitable for real-time use.
//!
//! # Features
//!
//! - 7 waveshapes: Sine, Triangle, Saw, Square, Pulse, Sample-and-Hold, Noise
//! - Hz rate or tempo-synchronized rate (note divisions)
//! - Phase reset modes: Free-running or Retrigger on note events
//! - Polarity modes: Bipolar [-1.0, 1.0] or Unipolar [0.0, 1.0]
//! - Control-rate integration with `rigel-timing::Timebase`
//!
//! # Example
//!
//! ```ignore
//! use rigel_modulation::{Lfo, LfoWaveshape, LfoRateMode, ModulationSource};
//! use rigel_timing::Timebase;
//!
//! let mut lfo = Lfo::new();
//! lfo.set_waveshape(LfoWaveshape::Sine);
//! lfo.set_rate(LfoRateMode::Hz(2.0));
//!
//! let mut timebase = Timebase::new(44100.0);
//! timebase.advance_block(64);
//!
//! lfo.update(&timebase);
//! let modulation = lfo.value(); // Returns value in [-1.0, 1.0]
//! ```

mod lfo;
mod rate;
mod rng;
mod traits;
mod waveshape;

// Re-export all public types
pub use lfo::Lfo;
pub use rate::{LfoRateMode, NoteBase, NoteDivision, NoteModifier};
pub use rng::Rng;
pub use traits::ModulationSource;
pub use waveshape::LfoWaveshape;

// Re-export phase and polarity types (defined in lfo.rs for now)
pub use lfo::{LfoPhaseMode, LfoPolarity};
