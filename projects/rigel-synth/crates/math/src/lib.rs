#![no_std]
#![doc = include_str!("../README.md")]
#![warn(missing_docs)]
#![warn(clippy::all)]

//! rigel-math: Trait-based SIMD abstraction library for real-time audio DSP
//!
//! This library provides zero-cost SIMD abstractions that enable writing DSP algorithms
//! once and compiling to optimal SIMD instructions for each platform without #[cfg] directives.
//!
//! # Features
//!
//! - **Trait-based SIMD abstraction**: Write backend-agnostic code using `SimdVector` trait
//! - **Compile-time backend selection**: Choose scalar, AVX2, AVX512, or NEON via cargo features
//! - **Block processing**: Fixed-size aligned buffers (64/128 samples) for cache efficiency
//! - **Fast math kernels**: Vectorized tanh, exp, log, sin/cos, inverse, sqrt, pow
//! - **Lookup tables**: Wavetable synthesis with linear/cubic interpolation
//! - **Denormal protection**: RAII-based FTZ/DAZ flags for real-time stability
//! - **No allocations**: All operations stack-based, real-time safe
//!
//! # Quick Start
//!
//! ```rust
//! use rigel_math::{Block64, DefaultSimdVector, DenormalGuard};
//! use rigel_math::ops::mul;
//!
//! fn apply_gain(input: &Block64, output: &mut Block64, gain: f32) {
//!     let _guard = DenormalGuard::new();
//!     let gain_vec = DefaultSimdVector::splat(gain);
//!
//!     for (in_chunk, out_chunk) in input.as_chunks::<DefaultSimdVector>()
//!         .iter()
//!         .zip(output.as_chunks_mut::<DefaultSimdVector>().iter_mut())
//!     {
//!         *out_chunk = mul(*in_chunk, gain_vec);
//!     }
//! }
//! ```

// Re-export libm for use in math kernels (test-only reference implementations)
extern crate libm;

// Core trait definitions
pub mod traits;

// Backend implementations
pub mod backends;

// Public re-exports for convenience
pub use traits::{SimdMask, SimdVector};

// Re-export backend types
pub use backends::scalar::{ScalarMask, ScalarVector};

#[cfg(feature = "avx2")]
pub use backends::avx2::{Avx2Mask, Avx2Vector};

#[cfg(feature = "avx512")]
pub use backends::avx512::{Avx512Mask, Avx512Vector};

#[cfg(feature = "neon")]
pub use backends::neon::{NeonMask, NeonVector};

/// Default SIMD vector type based on enabled feature
///
/// This type alias resolves to the appropriate SIMD backend selected at compile time:
/// - `scalar` feature (default): `ScalarVector<f32>` (1 lane)
/// - `avx2` feature: `Avx2Vector` (8 lanes, x86-64)
/// - `avx512` feature: `Avx512Vector` (16 lanes, x86-64)
/// - `neon` feature: `NeonVector` (4 lanes, ARM64)
#[cfg(all(
    not(feature = "avx2"),
    not(feature = "avx512"),
    not(feature = "neon")
))]
pub type DefaultSimdVector = ScalarVector<f32>;

#[cfg(feature = "avx2")]
#[cfg(target_arch = "x86_64")]
pub type DefaultSimdVector = Avx2Vector;

#[cfg(feature = "avx512")]
#[cfg(target_arch = "x86_64")]
pub type DefaultSimdVector = Avx512Vector;

#[cfg(feature = "neon")]
pub type DefaultSimdVector = NeonVector;

// Compile-time checks to prevent multiple backends
#[cfg(all(feature = "avx2", feature = "avx512"))]
compile_error!("Cannot enable both avx2 and avx512 features simultaneously");

#[cfg(all(feature = "avx2", feature = "neon"))]
compile_error!("Cannot enable both avx2 and neon features simultaneously");

#[cfg(all(feature = "avx512", feature = "neon"))]
compile_error!("Cannot enable both avx512 and neon features simultaneously");
