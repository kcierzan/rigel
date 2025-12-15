#![no_std]
#![doc = include_str!("../README.md")]
#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(unexpected_cfgs)]

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
//! use rigel_math::{Block64, DefaultSimdVector, DenormalGuard, SimdVector};
//! use rigel_math::ops::mul;
//!
//! fn apply_gain(input: &Block64, output: &mut Block64, gain: f32) {
//!     let _guard = DenormalGuard::new();
//!     let gain_vec = DefaultSimdVector::splat(gain);
//!
//!     for mut out_chunk in output.as_chunks_mut::<DefaultSimdVector>().iter_mut() {
//!         // In a real implementation, you'd zip with input chunks
//!         let value = out_chunk.load();
//!         out_chunk.store(mul(value, gain_vec));
//!     }
//! }
//!
//! // Example usage
//! let mut input = Block64::new();
//! let mut output = Block64::new();
//! apply_gain(&input, &mut output, 0.5);
//! ```

// Re-export libm for use in math kernels (test-only reference implementations)
extern crate libm;

// Core trait definitions
pub mod traits;

// Backend implementations
pub mod backends;

// Block processing
pub mod block;

// Functional-style vector operations
pub mod ops;

// Denormal protection
pub mod denormal;

// Fast math kernels
pub mod math;

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

// Scalar fast-math for control-rate operations
pub mod scalar_fast;

// Runtime SIMD dispatch
pub mod simd;

// Public re-exports for convenience
pub use traits::{SimdInt, SimdMask, SimdVector};

// Re-export block types
pub use block::{AudioBlock, Block128, Block64};

// Re-export denormal protection
pub use denormal::DenormalGuard;

// Re-export scalar fast-math functions
pub use scalar_fast::{fast_cosf, fast_expf, fast_logf, fast_sinf};

// Re-export backend types
pub use backends::scalar::{ScalarInt, ScalarInt64, ScalarMask, ScalarVector};

// Only re-export AVX2 types when both feature is enabled AND we're targeting x86/x86_64
#[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
pub use backends::avx2::{Avx2Mask, Avx2Vector};

// Only re-export AVX512 types when both feature is enabled AND we're targeting x86/x86_64
#[cfg(all(feature = "avx512", any(target_arch = "x86", target_arch = "x86_64")))]
pub use backends::avx512::{Avx512Mask, Avx512Vector};

// Only re-export NEON types when both feature is enabled AND we're targeting aarch64
#[cfg(all(feature = "neon", target_arch = "aarch64"))]
pub use backends::neon::{NeonMask, NeonVector};

/// Default SIMD vector type based on enabled feature
///
/// This type alias resolves to the appropriate SIMD backend selected at compile time:
/// - `scalar` feature (default): `ScalarVector<f32>` (1 lane)
/// - `avx2` feature: `Avx2Vector` (8 lanes, x86-64)
/// - `avx512` feature: `Avx512Vector` (16 lanes, x86-64)
/// - `neon` feature: `NeonVector` (4 lanes, ARM64)
///
/// Note: When `runtime-dispatch` is enabled, multiple backends can coexist and
/// DefaultSimdVector is set to ScalarVector as a safe fallback.
#[cfg(all(
    not(feature = "runtime-dispatch"),
    not(feature = "avx2"),
    not(feature = "avx512"),
    not(feature = "neon")
))]
pub type DefaultSimdVector = ScalarVector<f32>;

/// Default SIMD vector type (AVX2 backend for x86-64)
#[cfg(all(
    not(feature = "runtime-dispatch"),
    feature = "avx2",
    target_arch = "x86_64"
))]
pub type DefaultSimdVector = Avx2Vector;

/// Default SIMD vector type (AVX512 backend for x86-64)
#[cfg(all(
    not(feature = "runtime-dispatch"),
    feature = "avx512",
    target_arch = "x86_64"
))]
pub type DefaultSimdVector = Avx512Vector;

/// Default SIMD vector type (NEON backend for ARM64)
#[cfg(all(
    not(feature = "runtime-dispatch"),
    feature = "neon",
    target_arch = "aarch64"
))]
pub type DefaultSimdVector = NeonVector;

/// When runtime-dispatch is enabled, DefaultSimdVector defaults to scalar
/// (users should use the dispatcher instead of DefaultSimdVector)
#[cfg(feature = "runtime-dispatch")]
pub type DefaultSimdVector = ScalarVector<f32>;

// Compile-time checks to prevent conflicting backends for the target architecture
// Note: These checks are disabled when runtime-dispatch is enabled, as multiple
// backends need to coexist for runtime selection

// Prevent both AVX2 and AVX512 on x86/x86_64 (they conflict) - UNLESS runtime-dispatch is enabled
#[cfg(all(
    not(feature = "runtime-dispatch"),
    feature = "avx2",
    feature = "avx512",
    any(target_arch = "x86", target_arch = "x86_64")
))]
compile_error!(
    "Cannot enable both avx2 and avx512 features simultaneously on x86/x86_64. Choose one backend, or use runtime-dispatch feature."
);
