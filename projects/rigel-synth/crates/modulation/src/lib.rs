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
//! - Configurable interpolation: Linear or Cubic Hermite
//! - Control-rate integration with `rigel-timing::Timebase`
//! - Runtime SIMD dispatch via `rigel-math::simd::SimdContext`
//!
//! # Example
//!
//! ```ignore
//! use rigel_modulation::{Lfo, LfoWaveshape, LfoRateMode, InterpolationStrategy, ModulationSource};
//! use rigel_math::simd::SimdContext;
//! use rigel_timing::Timebase;
//!
//! let simd_ctx = SimdContext::new();
//! let mut lfo = Lfo::new();
//! lfo.set_waveshape(LfoWaveshape::Sine);
//! lfo.set_rate(LfoRateMode::Hz(2.0));
//! lfo.set_interpolation(InterpolationStrategy::CubicHermite);
//!
//! let mut timebase = Timebase::new(44100.0);
//! timebase.advance_block(64);
//!
//! lfo.update(&timebase);
//!
//! // Block-based processing (most efficient, uses runtime SIMD dispatch)
//! let mut output = [0.0f32; 64];
//! lfo.generate_block(&mut output, &simd_ctx);
//!
//! // Or single-sample access (uses scalar interpolation)
//! let value = lfo.sample();
//! ```

pub mod envelope;
mod lfo;
mod rate;
mod simd_rng;
mod traits;
mod waveshape;

// Re-export all public types
pub use lfo::{InterpolationStrategy, Lfo, LfoPhaseMode, LfoPolarity};
pub use rate::{LfoRateMode, NoteBase, NoteDivision, NoteModifier};
pub use simd_rng::SimdXorshift128;
pub use traits::ModulationSource;
pub use waveshape::LfoWaveshape;
