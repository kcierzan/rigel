//! Fast math kernels for audio DSP
//!
//! This module provides vectorized implementations of common mathematical functions
//! optimized for audio DSP applications. All functions work across all SIMD backends
//! (scalar, AVX2, AVX512, NEON) through the `SimdVector` trait abstraction.
//!
//! # Modules
//!
//! - `tanh`: Hyperbolic tangent for waveshaping and soft clipping
//! - `exp`: Exponential functions for envelopes and decay curves
//! - `log`: Logarithm functions for frequency calculations
//! - `trig`: Sine, cosine, and sincos for oscillators and modulation
//! - `inverse`: Fast reciprocal (1/x) for division-free algorithms
//! - `sqrt`: Square root and reciprocal square root
//! - `pow`: Power functions
//! - `atan`: Arctangent for phase calculations
//! - `exp2_log2`: Base-2 exponential and logarithm with IEEE 754 optimization
//!
//! # Example
//!
//! ```rust
//! use rigel_math::{DefaultSimdVector, SimdVector};
//! use rigel_math::math::{tanh, exp, sin};
//!
//! // Waveshaping with tanh
//! let input = DefaultSimdVector::splat(2.0);
//! let shaped = tanh(input);
//!
//! // Envelope generation with exp
//! let decay = DefaultSimdVector::splat(-5.0);
//! let envelope = exp(decay);
//!
//! // Oscillator with sin
//! let phase = DefaultSimdVector::splat(0.5);
//! let output = sin(phase);
//! ```

// Math kernel modules
pub mod atan;
pub mod exp;
pub mod exp2_log2;
pub mod inverse;
pub mod log;
pub mod pow;
pub mod sqrt;
pub mod tanh;
pub mod trig;

// Re-export commonly used functions
pub use self::atan::{atan, atan2};
pub use self::exp::{exp, exp_envelope};
pub use self::exp2_log2::{fast_exp2, fast_log2};
pub use self::inverse::{recip, recip_rough};
pub use self::log::{log, log10, log1p, log2};
pub use self::pow::pow;
pub use self::sqrt::{rsqrt, sqrt};
pub use self::tanh::{tanh, tanh_fast};
pub use self::trig::{cos, sin, sincos};
