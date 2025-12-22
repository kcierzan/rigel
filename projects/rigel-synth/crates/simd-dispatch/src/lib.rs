#![no_std]
#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(unexpected_cfgs)]

//! rigel-simd-dispatch: Runtime SIMD dispatch and SimdContext API
//!
//! This crate provides runtime SIMD backend selection and dispatch for optimal
//! performance across different CPU architectures. It enables a single binary to
//! automatically use the best available SIMD instruction set (scalar → AVX2 → AVX-512 on x86_64,
//! NEON on aarch64).
//!
//! # Primary Public API
//!
//! **Use `SimdContext` for all DSP code** - it provides a unified interface that works
//! identically across all platforms while automatically selecting the optimal backend.
//!
//! # Architecture
//!
//! - `context`: SimdContext unified API (PRIMARY PUBLIC INTERFACE)
//! - `backend`: SimdBackend trait definition and ProcessParams struct
//! - `scalar`: Scalar (non-SIMD) fallback implementation
//! - `dispatcher`: CPU feature detection and BackendType selection
//! - Platform-specific backends: `avx2`, `avx512`, `neon`
//!
//! # Feature Flags
//!
//! - `runtime-dispatch`: Enable runtime CPU detection and backend selection
//! - `avx2`: Compile AVX2 backend (x86_64)
//! - `avx512`: Compile AVX-512 backend (x86_64, experimental)
//! - `neon`: Compile NEON backend (aarch64)
//! - `force-scalar`: Force scalar backend for deterministic testing
//! - `force-avx2`: Force AVX2 backend for deterministic testing
//! - `force-avx512`: Force AVX-512 backend for experimental testing
//! - `force-neon`: Force NEON backend for deterministic testing
//!
//! # Example Usage
//!
//! ```ignore
//! use rigel_simd_dispatch::{SimdContext, ProcessParams};
//!
//! // Initialize once during engine startup
//! let ctx = SimdContext::new();
//! println!("Using SIMD backend: {}", ctx.backend_name());
//!
//! // Use for DSP operations
//! let input = [1.0f32; 64];
//! let mut output = [0.0f32; 64];
//! let params = ProcessParams {
//!     gain: 0.5,
//!     frequency: 440.0,
//!     sample_rate: 44100.0,
//! };
//!
//! ctx.process_block(&input, &mut output, &params);
//! ```

// Re-export everything from rigel-math for convenience
pub use rigel_math::*;

// Internal modules
pub mod backend;
pub mod context;
pub mod dispatcher;
pub mod helpers;
pub mod scalar;

// Platform-specific backend modules
#[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
pub mod avx2;

#[cfg(all(feature = "avx512", any(target_arch = "x86", target_arch = "x86_64")))]
pub mod avx512;

#[cfg(all(feature = "neon", target_arch = "aarch64"))]
pub mod neon;

// Re-export primary public API
pub use context::SimdContext;

// Re-export supporting types
pub use backend::{ProcessParams, SimdBackend};
pub use dispatcher::{BackendDispatcher, BackendType, CpuFeatures};
pub use scalar::ScalarBackend;

// Conditionally re-export SIMD backends
#[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
pub use avx2::Avx2Backend;

#[cfg(all(feature = "avx512", any(target_arch = "x86", target_arch = "x86_64")))]
pub use avx512::Avx512Backend;

#[cfg(all(feature = "neon", target_arch = "aarch64"))]
pub use neon::NeonBackend;
