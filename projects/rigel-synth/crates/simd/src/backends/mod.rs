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

// Compile-time checks to prevent conflicting backends for the target architecture
// Note: AVX2/AVX512 are x86/x86_64 only, NEON is aarch64 only, so they don't conflict across architectures
// These checks are disabled when runtime-dispatch is enabled

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
