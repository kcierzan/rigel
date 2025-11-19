//! SIMD backend implementations
//!
//! This module contains platform-specific SIMD implementations selected at compile time
//! via cargo features. Only one backend is active per build.

// Scalar backend (always available as fallback)
pub mod scalar;

// Platform-specific backends (feature-gated)
#[cfg(feature = "avx2")]
pub mod avx2;

#[cfg(feature = "avx512")]
pub mod avx512;

#[cfg(feature = "neon")]
pub mod neon;

// Compile-time checks to prevent multiple backends
#[cfg(all(feature = "avx2", feature = "avx512"))]
compile_error!("Cannot enable both avx2 and avx512 features simultaneously. Choose one backend per build.");

#[cfg(all(feature = "avx2", feature = "neon"))]
compile_error!("Cannot enable both avx2 and neon features simultaneously. Choose one backend per build.");

#[cfg(all(feature = "avx512", feature = "neon"))]
compile_error!("Cannot enable both avx512 and neon features simultaneously. Choose one backend per build.");

#[cfg(all(feature = "avx2", feature = "avx512", feature = "neon"))]
compile_error!("Cannot enable multiple SIMD backends simultaneously. Choose one backend per build.");
