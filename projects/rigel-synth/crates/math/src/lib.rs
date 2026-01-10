#![no_std]
#![doc = include_str!("../README.md")]
#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(unexpected_cfgs)]

//! rigel-math: Fast math kernels and DSP utilities for real-time audio
//!
//! This library provides vectorized math functions and DSP building blocks
//! built on top of rigel-simd's SIMD abstractions.
//!
//! # Features
//!
//! - **Fast math kernels**: Vectorized tanh, exp, log, sin/cos, inverse, sqrt, pow
//! - **Scalar math**: Control-rate approximations (expf, logf, sinf, cosf)
//! - **Lookup tables**: Wavetable synthesis with linear/cubic interpolation
//! - **Interpolation**: Linear, cubic hermite, quintic polynomials
//! - **Waveshaping**: Saturation curves, sigmoid functions
//! - **Anti-aliasing**: PolyBLEP/PolyBLAMP for band-limited synthesis
//! - **No allocations**: All operations stack-based, real-time safe
//!
//! # Quick Start
//!
//! ```rust
//! use rigel_math::{Block64, DefaultSimdVector, DenormalGuard, SimdVector};
//! use rigel_math::ops::mul;
//!
//! fn apply_gain(input: &Block64, output: &mut Block64, gain: f32) {
//!     let _guard = DenormalGuard::new();
//!     let gain_vec = DefaultSimdVector::splat(gain);
//!
//!     for mut out_chunk in output.as_chunks_mut::<DefaultSimdVector>().iter_mut() {
//!         let value = out_chunk.load();
//!         out_chunk.store(mul(value, gain_vec));
//!     }
//! }
//! ```

// Re-export libm for use in math kernels (test-only reference implementations)
extern crate libm;

// ============================================================================
// Re-export everything from rigel-simd for backward compatibility
// ============================================================================

// Re-export all public items from rigel-simd
pub use rigel_simd::*;

// Re-export modules from rigel-simd for explicit module access (e.g., rigel_math::ops::mul)
pub use rigel_simd::backends;
pub use rigel_simd::block;
pub use rigel_simd::denormal;
pub use rigel_simd::ops;
pub use rigel_simd::traits;

// ============================================================================
// Math-specific modules (not from rigel-simd)
// ============================================================================

// SIMD vectorized math kernels
pub mod simd;

// Saturation and waveshaping
pub mod saturate;
pub mod sigmoid;

// Interpolation
pub mod interpolate;

// Anti-aliasing (PolyBLEP for steps, PolyBLAMP for corners)
pub mod antialias;

// Noise generation
pub mod noise;

// Lookup tables
pub mod table;

// Crossfade and parameter ramping
pub mod crossfade;

// Scalar math for control-rate operations
pub mod scalar;

// Fast exp2 lookup table for level-to-linear conversion
pub mod exp2_lut;

// ============================================================================
// Additional re-exports for convenience
// ============================================================================

// Re-export scalar math functions at top level
pub use scalar::{cosf, expf, logf, sinf};

// Re-export hermite_scalar for convenience (commonly used)
pub use interpolate::hermite_scalar;

// Re-export exp2 LUT functions for level-to-linear conversion
pub use exp2_lut::{db_to_linear, exp2_lut, exp2_lut_slice};
