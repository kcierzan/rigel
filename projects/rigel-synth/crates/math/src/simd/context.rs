//! SIMD Context - Unified Public API
//!
//! This module provides the `SimdContext` type, which is the ONLY public SIMD interface
//! used by DSP code. It abstracts away platform differences:
//!
//! - **x86_64 with runtime-dispatch**: Wraps BackendDispatcher for runtime CPU detection
//! - **aarch64 or forced backends**: Zero-sized type with compile-time backend selection
//!
//! # Example Usage
//!
//! ```ignore
//! use rigel_math::simd::{SimdContext, ProcessParams};
//!
//! // Initialize once during engine startup
//! let ctx = SimdContext::new();
//!
//! // Query selected backend (for logging/debugging)
//! println!("Using SIMD backend: {}", ctx.backend_name());
//!
//! // Use the context for DSP operations
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

#![allow(unused)]

use super::backend::ProcessParams;
use super::scalar::ScalarBackend;

#[cfg(feature = "runtime-dispatch")]
use super::dispatcher::BackendDispatcher;

/// SIMD Context - Unified Public API
///
/// This is the primary interface for all SIMD operations in rigel-math.
/// It provides a consistent API across all platforms while optimizing for each:
///
/// - **x86_64 with runtime-dispatch**: Contains BackendDispatcher for runtime selection
/// - **aarch64 or forced backends**: Zero-sized type, compiles to direct backend calls
///
/// # Platform Behavior
///
/// **x86_64 with runtime-dispatch enabled:**
/// - First `new()` call detects CPU features (AVX2, AVX-512)
/// - Selects optimal backend: AVX-512 → AVX2 → Scalar
/// - All operations dispatch through function pointers (<1% overhead)
///
/// **aarch64 or forced backend features:**
/// - `new()` is zero-cost (optimized away)
/// - All operations compile to direct backend calls
/// - No runtime overhead whatsoever
///
/// # Performance
///
/// - Initialization: ~100-200 CPU cycles (call once at startup)
/// - Dispatch overhead: <1% for block operations (typically 64-512 samples)
/// - Individual sample operations: Use batch methods for best performance
///
/// # Safety
///
/// All methods are safe to call. Internal unsafe SIMD intrinsics are properly
/// encapsulated and verified through property-based testing.
#[derive(Clone, Debug)]
pub struct SimdContext {
    #[cfg(feature = "runtime-dispatch")]
    dispatcher: BackendDispatcher,
}

impl SimdContext {
    /// Initialize SIMD context with optimal backend
    ///
    /// This function:
    /// - **x86_64 with runtime-dispatch**: Detects CPU features and selects best backend
    /// - **aarch64 or forced backend**: Zero-cost, optimized away by compiler
    ///
    /// # Performance
    ///
    /// - First call: ~100-200 CPU cycles (CPUID on x86_64)
    /// - Should be called once at plugin initialization
    /// - Not real-time safe (contains CPUID on first call)
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Initialize once during plugin startup
    /// let ctx = SimdContext::new();
    ///
    /// // Store in your audio engine and reuse for all processing
    /// let mut engine = AudioEngine {
    ///     simd_ctx: ctx,
    ///     // ...
    /// };
    /// ```
    pub fn new() -> Self {
        #[cfg(feature = "runtime-dispatch")]
        {
            Self {
                dispatcher: BackendDispatcher::init(),
            }
        }

        #[cfg(not(feature = "runtime-dispatch"))]
        {
            Self {}
        }
    }

    /// Get backend name for logging/debugging
    ///
    /// # Returns
    ///
    /// Static string: "scalar", "avx2", "avx512", or "neon"
    ///
    /// # Example
    ///
    /// ```ignore
    /// let ctx = SimdContext::new();
    /// println!("Using SIMD backend: {}", ctx.backend_name());
    /// ```
    pub fn backend_name(&self) -> &'static str {
        #[cfg(feature = "runtime-dispatch")]
        {
            self.dispatcher.backend_name()
        }

        #[cfg(not(feature = "runtime-dispatch"))]
        {
            Self::compile_time_backend_name()
        }
    }

    /// Get compile-time backend name (when runtime-dispatch disabled)
    #[cfg(not(feature = "runtime-dispatch"))]
    fn compile_time_backend_name() -> &'static str {
        #[cfg(feature = "force-scalar")]
        {
            "scalar"
        }

        #[cfg(feature = "force-avx2")]
        {
            "avx2"
        }

        #[cfg(feature = "force-avx512")]
        {
            "avx512"
        }

        #[cfg(feature = "force-neon")]
        {
            "neon"
        }

        #[cfg(not(any(
            feature = "force-scalar",
            feature = "force-avx2",
            feature = "force-avx512",
            feature = "force-neon"
        )))]
        {
            // Default to compile-time backend based on architecture
            #[cfg(all(feature = "neon", target_arch = "aarch64"))]
            {
                "neon"
            }

            #[cfg(not(all(feature = "neon", target_arch = "aarch64")))]
            {
                "scalar"
            }
        }
    }

    /// Process a block of audio samples
    ///
    /// Applies gain to input samples and writes to output buffer.
    /// Dispatches to the selected SIMD backend for optimal performance.
    ///
    /// # Parameters
    ///
    /// - `input`: Input samples (any length)
    /// - `output`: Output buffer (must be same length as input)
    /// - `params`: Processing parameters (gain, frequency, sample_rate)
    ///
    /// # Performance
    ///
    /// - Dispatch overhead: <1% for typical block sizes (64-512 samples)
    /// - SIMD speedup: 2-8x depending on backend (AVX2: ~2-4x, AVX-512: ~4-8x)
    ///
    /// # Panics
    ///
    /// May panic if input and output lengths differ (backend-dependent)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let ctx = SimdContext::new();
    /// let input = [1.0f32; 128];
    /// let mut output = [0.0f32; 128];
    /// let params = ProcessParams {
    ///     gain: 0.5,
    ///     frequency: 440.0,
    ///     sample_rate: 44100.0,
    /// };
    ///
    /// ctx.process_block(&input, &mut output, &params);
    /// ```
    #[inline]
    pub fn process_block(&self, input: &[f32], output: &mut [f32], params: &ProcessParams) {
        #[cfg(feature = "runtime-dispatch")]
        {
            self.dispatcher.process_block(input, output, params)
        }

        #[cfg(not(feature = "runtime-dispatch"))]
        {
            Self::compile_time_process_block(input, output, params)
        }
    }

    #[cfg(not(feature = "runtime-dispatch"))]
    fn compile_time_process_block(input: &[f32], output: &mut [f32], params: &ProcessParams) {
        use super::backend::SimdBackend;

        #[cfg(feature = "force-scalar")]
        {
            ScalarBackend::process_block(input, output, params);
        }

        #[cfg(all(
            feature = "force-avx2",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            use super::avx2::Avx2Backend;
            Avx2Backend::process_block(input, output, params);
        }

        #[cfg(all(
            feature = "force-avx512",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            use super::avx512::Avx512Backend;
            Avx512Backend::process_block(input, output, params);
        }

        #[cfg(all(feature = "force-neon", target_arch = "aarch64"))]
        {
            use super::neon::NeonBackend;
            NeonBackend::process_block(input, output, params);
        }

        #[cfg(not(any(
            feature = "force-scalar",
            feature = "force-avx2",
            feature = "force-avx512",
            feature = "force-neon"
        )))]
        {
            #[cfg(all(feature = "neon", target_arch = "aarch64"))]
            {
                use super::neon::NeonBackend;
                NeonBackend::process_block(input, output, params);
            }

            #[cfg(not(all(feature = "neon", target_arch = "aarch64")))]
            {
                ScalarBackend::process_block(input, output, params);
            }
        }
    }

    /// Advance oscillator phases with SIMD vectorization
    ///
    /// Adds phase increments to phases and wraps to [0, TAU) range.
    /// Uses SIMD vectorization for optimal performance on large phase arrays.
    ///
    /// # Parameters
    ///
    /// - `phases`: Mutable phase array (modified in-place)
    /// - `phase_increments`: Increment values to add to each phase
    /// - `count`: Number of phases to advance (must be <= phases.len() and phase_increments.len())
    ///
    /// # Performance
    ///
    /// - Processes 4-16 phases per iteration depending on SIMD backend
    /// - Automatically handles phase wrapping in vectorized form
    ///
    /// # Example
    ///
    /// ```ignore
    /// let ctx = SimdContext::new();
    /// let mut phases = [0.0f32; 64];
    /// let increments = [0.01f32; 64];
    ///
    /// ctx.advance_phase(&mut phases, &increments, 64);
    /// ```
    #[inline]
    pub fn advance_phase(&self, phases: &mut [f32], phase_increments: &[f32], count: usize) {
        #[cfg(feature = "runtime-dispatch")]
        {
            self.dispatcher
                .advance_phase(phases, phase_increments, count)
        }

        #[cfg(not(feature = "runtime-dispatch"))]
        {
            Self::compile_time_advance_phase(phases, phase_increments, count)
        }
    }

    #[cfg(not(feature = "runtime-dispatch"))]
    fn compile_time_advance_phase(phases: &mut [f32], phase_increments: &[f32], count: usize) {
        use super::backend::SimdBackend;

        #[cfg(feature = "force-scalar")]
        {
            ScalarBackend::advance_phase_vectorized(phases, phase_increments, count);
        }

        #[cfg(all(
            feature = "force-avx2",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            use super::avx2::Avx2Backend;
            Avx2Backend::advance_phase_vectorized(phases, phase_increments, count);
        }

        #[cfg(all(
            feature = "force-avx512",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            use super::avx512::Avx512Backend;
            Avx512Backend::advance_phase_vectorized(phases, phase_increments, count);
        }

        #[cfg(all(feature = "force-neon", target_arch = "aarch64"))]
        {
            use super::neon::NeonBackend;
            NeonBackend::advance_phase_vectorized(phases, phase_increments, count);
        }

        #[cfg(not(any(
            feature = "force-scalar",
            feature = "force-avx2",
            feature = "force-avx512",
            feature = "force-neon"
        )))]
        {
            #[cfg(all(feature = "neon", target_arch = "aarch64"))]
            {
                use super::neon::NeonBackend;
                NeonBackend::advance_phase_vectorized(phases, phase_increments, count);
            }

            #[cfg(not(all(feature = "neon", target_arch = "aarch64")))]
            {
                ScalarBackend::advance_phase_vectorized(phases, phase_increments, count);
            }
        }
    }

    /// Wavetable interpolation with SIMD
    ///
    /// Performs linear interpolation of wavetable samples at fractional positions.
    /// Positions are automatically wrapped to [0.0, 1.0) range.
    ///
    /// # Parameters
    ///
    /// - `wavetable`: Source wavetable samples (any length, typically power of 2)
    /// - `positions`: Normalized positions in [0.0, 1.0) range (auto-wrapped if outside)
    /// - `output`: Output buffer (must be same length as positions)
    ///
    /// # Performance
    ///
    /// - Processes 4-16 samples per iteration depending on SIMD backend
    /// - Linear interpolation for balance of quality and performance
    ///
    /// # Example
    ///
    /// ```ignore
    /// let ctx = SimdContext::new();
    /// let wavetable = [0.0f32, 1.0, 0.0, -1.0]; // Simple sine approximation
    /// let positions = [0.0, 0.25, 0.5, 0.75];
    /// let mut output = [0.0f32; 4];
    ///
    /// ctx.interpolate_linear(&wavetable, &positions, &mut output);
    /// ```
    #[inline]
    pub fn interpolate_linear(&self, wavetable: &[f32], positions: &[f32], output: &mut [f32]) {
        #[cfg(feature = "runtime-dispatch")]
        {
            self.dispatcher.interpolate(wavetable, positions, output)
        }

        #[cfg(not(feature = "runtime-dispatch"))]
        {
            Self::compile_time_interpolate(wavetable, positions, output)
        }
    }

    #[cfg(not(feature = "runtime-dispatch"))]
    fn compile_time_interpolate(wavetable: &[f32], positions: &[f32], output: &mut [f32]) {
        use super::backend::SimdBackend;

        #[cfg(feature = "force-scalar")]
        {
            ScalarBackend::interpolate_wavetable(wavetable, positions, output);
        }

        #[cfg(all(
            feature = "force-avx2",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            use super::avx2::Avx2Backend;
            Avx2Backend::interpolate_wavetable(wavetable, positions, output);
        }

        #[cfg(all(
            feature = "force-avx512",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            use super::avx512::Avx512Backend;
            Avx512Backend::interpolate_wavetable(wavetable, positions, output);
        }

        #[cfg(all(feature = "force-neon", target_arch = "aarch64"))]
        {
            use super::neon::NeonBackend;
            NeonBackend::interpolate_wavetable(wavetable, positions, output);
        }

        #[cfg(not(any(
            feature = "force-scalar",
            feature = "force-avx2",
            feature = "force-avx512",
            feature = "force-neon"
        )))]
        {
            #[cfg(all(feature = "neon", target_arch = "aarch64"))]
            {
                use super::neon::NeonBackend;
                NeonBackend::interpolate_wavetable(wavetable, positions, output);
            }

            #[cfg(not(all(feature = "neon", target_arch = "aarch64")))]
            {
                ScalarBackend::interpolate_wavetable(wavetable, positions, output);
            }
        }
    }

    /// Apply gain to audio buffer
    ///
    /// Convenience method for applying a constant gain value to all samples.
    /// Equivalent to process_block with gain parameter.
    ///
    /// # Parameters
    ///
    /// - `input`: Input samples
    /// - `output`: Output buffer (must be same length as input)
    /// - `gain`: Gain multiplier
    ///
    /// # Example
    ///
    /// ```ignore
    /// let ctx = SimdContext::new();
    /// let input = [1.0f32, 2.0, 3.0, 4.0];
    /// let mut output = [0.0f32; 4];
    ///
    /// ctx.apply_gain(&input, &mut output, 0.5);
    /// // output is now [0.5, 1.0, 1.5, 2.0]
    /// ```
    #[inline]
    pub fn apply_gain(&self, input: &[f32], output: &mut [f32], gain: f32) {
        let params = ProcessParams {
            gain,
            frequency: 0.0,
            sample_rate: 0.0,
        };
        self.process_block(input, output, &params);
    }

    // ========================================================================
    // Arithmetic Operations (Binary)
    // ========================================================================

    /// Element-wise addition: output[i] = a[i] + b[i]
    ///
    /// # Parameters
    /// - `a`: First input array
    /// - `b`: Second input array
    /// - `output`: Output array (must be same length as inputs)
    ///
    /// # Example
    /// ```ignore
    /// let ctx = SimdContext::new();
    /// let a = [1.0, 2.0, 3.0, 4.0];
    /// let b = [0.5, 0.5, 0.5, 0.5];
    /// let mut output = [0.0; 4];
    /// ctx.add(&a, &b, &mut output);
    /// // output is now [1.5, 2.5, 3.5, 4.5]
    /// ```
    #[inline]
    pub fn add(&self, a: &[f32], b: &[f32], output: &mut [f32]) {
        #[cfg(feature = "runtime-dispatch")]
        {
            self.dispatcher.add(a, b, output)
        }

        #[cfg(not(feature = "runtime-dispatch"))]
        {
            Self::compile_time_add(a, b, output)
        }
    }

    #[cfg(not(feature = "runtime-dispatch"))]
    fn compile_time_add(a: &[f32], b: &[f32], output: &mut [f32]) {
        use super::backend::SimdBackend;

        #[cfg(feature = "force-scalar")]
        {
            ScalarBackend::add(a, b, output);
        }

        #[cfg(all(
            feature = "force-avx2",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            use super::avx2::Avx2Backend;
            Avx2Backend::add(a, b, output);
        }

        #[cfg(all(
            feature = "force-avx512",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            use super::avx512::Avx512Backend;
            Avx512Backend::add(a, b, output);
        }

        #[cfg(all(feature = "force-neon", target_arch = "aarch64"))]
        {
            use super::neon::NeonBackend;
            NeonBackend::add(a, b, output);
        }

        #[cfg(not(any(
            feature = "force-scalar",
            feature = "force-avx2",
            feature = "force-avx512",
            feature = "force-neon"
        )))]
        {
            #[cfg(all(feature = "neon", target_arch = "aarch64"))]
            {
                use super::neon::NeonBackend;
                NeonBackend::add(a, b, output);
            }

            #[cfg(not(all(feature = "neon", target_arch = "aarch64")))]
            {
                ScalarBackend::add(a, b, output);
            }
        }
    }

    /// Element-wise subtraction: output[i] = a[i] - b[i]
    ///
    /// # Parameters
    /// - `a`: First input array
    /// - `b`: Second input array
    /// - `output`: Output array (must be same length as inputs)
    ///
    /// # Example
    /// ```ignore
    /// let ctx = SimdContext::new();
    /// let a = [1.0, 2.0, 3.0, 4.0];
    /// let b = [0.5, 0.5, 0.5, 0.5];
    /// let mut output = [0.0; 4];
    /// ctx.sub(&a, &b, &mut output);
    /// // output is now [0.5, 1.5, 2.5, 3.5]
    /// ```
    #[inline]
    pub fn sub(&self, a: &[f32], b: &[f32], output: &mut [f32]) {
        #[cfg(feature = "runtime-dispatch")]
        {
            self.dispatcher.sub(a, b, output)
        }

        #[cfg(not(feature = "runtime-dispatch"))]
        {
            Self::compile_time_sub(a, b, output)
        }
    }

    #[cfg(not(feature = "runtime-dispatch"))]
    fn compile_time_sub(a: &[f32], b: &[f32], output: &mut [f32]) {
        use super::backend::SimdBackend;

        #[cfg(feature = "force-scalar")]
        {
            ScalarBackend::sub(a, b, output);
        }

        #[cfg(all(
            feature = "force-avx2",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            use super::avx2::Avx2Backend;
            Avx2Backend::sub(a, b, output);
        }

        #[cfg(all(
            feature = "force-avx512",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            use super::avx512::Avx512Backend;
            Avx512Backend::sub(a, b, output);
        }

        #[cfg(all(feature = "force-neon", target_arch = "aarch64"))]
        {
            use super::neon::NeonBackend;
            NeonBackend::sub(a, b, output);
        }

        #[cfg(not(any(
            feature = "force-scalar",
            feature = "force-avx2",
            feature = "force-avx512",
            feature = "force-neon"
        )))]
        {
            #[cfg(all(feature = "neon", target_arch = "aarch64"))]
            {
                use super::neon::NeonBackend;
                NeonBackend::sub(a, b, output);
            }

            #[cfg(not(all(feature = "neon", target_arch = "aarch64")))]
            {
                ScalarBackend::sub(a, b, output);
            }
        }
    }

    /// Element-wise multiplication: output[i] = a[i] * b[i]
    ///
    /// # Parameters
    /// - `a`: First input array
    /// - `b`: Second input array
    /// - `output`: Output array (must be same length as inputs)
    ///
    /// # Example
    /// ```ignore
    /// let ctx = SimdContext::new();
    /// let a = [1.0, 2.0, 3.0, 4.0];
    /// let b = [2.0, 2.0, 2.0, 2.0];
    /// let mut output = [0.0; 4];
    /// ctx.mul(&a, &b, &mut output);
    /// // output is now [2.0, 4.0, 6.0, 8.0]
    /// ```
    #[inline]
    pub fn mul(&self, a: &[f32], b: &[f32], output: &mut [f32]) {
        #[cfg(feature = "runtime-dispatch")]
        {
            self.dispatcher.mul(a, b, output)
        }

        #[cfg(not(feature = "runtime-dispatch"))]
        {
            Self::compile_time_mul(a, b, output)
        }
    }

    #[cfg(not(feature = "runtime-dispatch"))]
    fn compile_time_mul(a: &[f32], b: &[f32], output: &mut [f32]) {
        use super::backend::SimdBackend;

        #[cfg(feature = "force-scalar")]
        {
            ScalarBackend::mul(a, b, output);
        }

        #[cfg(all(
            feature = "force-avx2",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            use super::avx2::Avx2Backend;
            Avx2Backend::mul(a, b, output);
        }

        #[cfg(all(
            feature = "force-avx512",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            use super::avx512::Avx512Backend;
            Avx512Backend::mul(a, b, output);
        }

        #[cfg(all(feature = "force-neon", target_arch = "aarch64"))]
        {
            use super::neon::NeonBackend;
            NeonBackend::mul(a, b, output);
        }

        #[cfg(not(any(
            feature = "force-scalar",
            feature = "force-avx2",
            feature = "force-avx512",
            feature = "force-neon"
        )))]
        {
            #[cfg(all(feature = "neon", target_arch = "aarch64"))]
            {
                use super::neon::NeonBackend;
                NeonBackend::mul(a, b, output);
            }

            #[cfg(not(all(feature = "neon", target_arch = "aarch64")))]
            {
                ScalarBackend::mul(a, b, output);
            }
        }
    }

    /// Element-wise division: output[i] = a[i] / b[i]
    ///
    /// # Parameters
    /// - `a`: First input array (numerator)
    /// - `b`: Second input array (denominator)
    /// - `output`: Output array (must be same length as inputs)
    ///
    /// # Example
    /// ```ignore
    /// let ctx = SimdContext::new();
    /// let a = [1.0, 2.0, 3.0, 4.0];
    /// let b = [2.0, 2.0, 2.0, 2.0];
    /// let mut output = [0.0; 4];
    /// ctx.div(&a, &b, &mut output);
    /// // output is now [0.5, 1.0, 1.5, 2.0]
    /// ```
    #[inline]
    pub fn div(&self, a: &[f32], b: &[f32], output: &mut [f32]) {
        #[cfg(feature = "runtime-dispatch")]
        {
            self.dispatcher.div(a, b, output)
        }

        #[cfg(not(feature = "runtime-dispatch"))]
        {
            Self::compile_time_div(a, b, output)
        }
    }

    #[cfg(not(feature = "runtime-dispatch"))]
    fn compile_time_div(a: &[f32], b: &[f32], output: &mut [f32]) {
        use super::backend::SimdBackend;

        #[cfg(feature = "force-scalar")]
        {
            ScalarBackend::div(a, b, output);
        }

        #[cfg(all(
            feature = "force-avx2",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            use super::avx2::Avx2Backend;
            Avx2Backend::div(a, b, output);
        }

        #[cfg(all(
            feature = "force-avx512",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            use super::avx512::Avx512Backend;
            Avx512Backend::div(a, b, output);
        }

        #[cfg(all(feature = "force-neon", target_arch = "aarch64"))]
        {
            use super::neon::NeonBackend;
            NeonBackend::div(a, b, output);
        }

        #[cfg(not(any(
            feature = "force-scalar",
            feature = "force-avx2",
            feature = "force-avx512",
            feature = "force-neon"
        )))]
        {
            #[cfg(all(feature = "neon", target_arch = "aarch64"))]
            {
                use super::neon::NeonBackend;
                NeonBackend::div(a, b, output);
            }

            #[cfg(not(all(feature = "neon", target_arch = "aarch64")))]
            {
                ScalarBackend::div(a, b, output);
            }
        }
    }

    /// Element-wise fused multiply-add: output[i] = a[i] * b[i] + c[i]
    ///
    /// # Parameters
    /// - `a`: First input array (multiplier)
    /// - `b`: Second input array (multiplier)
    /// - `c`: Third input array (addend)
    /// - `output`: Output array (must be same length as inputs)
    ///
    /// # Example
    /// ```ignore
    /// let ctx = SimdContext::new();
    /// let a = [1.0, 2.0, 3.0, 4.0];
    /// let b = [2.0, 2.0, 2.0, 2.0];
    /// let c = [0.5, 0.5, 0.5, 0.5];
    /// let mut output = [0.0; 4];
    /// ctx.fma(&a, &b, &c, &mut output);
    /// // output is now [2.5, 4.5, 6.5, 8.5]
    /// ```
    #[inline]
    pub fn fma(&self, a: &[f32], b: &[f32], c: &[f32], output: &mut [f32]) {
        #[cfg(feature = "runtime-dispatch")]
        {
            self.dispatcher.fma(a, b, c, output)
        }

        #[cfg(not(feature = "runtime-dispatch"))]
        {
            Self::compile_time_fma(a, b, c, output)
        }
    }

    #[cfg(not(feature = "runtime-dispatch"))]
    fn compile_time_fma(a: &[f32], b: &[f32], c: &[f32], output: &mut [f32]) {
        use super::backend::SimdBackend;

        #[cfg(feature = "force-scalar")]
        {
            ScalarBackend::fma(a, b, c, output);
        }

        #[cfg(all(
            feature = "force-avx2",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            use super::avx2::Avx2Backend;
            Avx2Backend::fma(a, b, c, output);
        }

        #[cfg(all(
            feature = "force-avx512",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            use super::avx512::Avx512Backend;
            Avx512Backend::fma(a, b, c, output);
        }

        #[cfg(all(feature = "force-neon", target_arch = "aarch64"))]
        {
            use super::neon::NeonBackend;
            NeonBackend::fma(a, b, c, output);
        }

        #[cfg(not(any(
            feature = "force-scalar",
            feature = "force-avx2",
            feature = "force-avx512",
            feature = "force-neon"
        )))]
        {
            #[cfg(all(feature = "neon", target_arch = "aarch64"))]
            {
                use super::neon::NeonBackend;
                NeonBackend::fma(a, b, c, output);
            }

            #[cfg(not(all(feature = "neon", target_arch = "aarch64")))]
            {
                ScalarBackend::fma(a, b, c, output);
            }
        }
    }

    // ========================================================================
    // Arithmetic Operations (Unary)
    // ========================================================================

    /// Element-wise negation: output[i] = -input[i]
    ///
    /// # Parameters
    /// - `input`: Input array
    /// - `output`: Output array (must be same length as input)
    ///
    /// # Example
    /// ```ignore
    /// let ctx = SimdContext::new();
    /// let input = [1.0, -2.0, 3.0, -4.0];
    /// let mut output = [0.0; 4];
    /// ctx.neg(&input, &mut output);
    /// // output is now [-1.0, 2.0, -3.0, 4.0]
    /// ```
    #[inline]
    pub fn neg(&self, input: &[f32], output: &mut [f32]) {
        #[cfg(feature = "runtime-dispatch")]
        {
            self.dispatcher.neg(input, output)
        }

        #[cfg(not(feature = "runtime-dispatch"))]
        {
            Self::compile_time_neg(input, output)
        }
    }

    #[cfg(not(feature = "runtime-dispatch"))]
    fn compile_time_neg(input: &[f32], output: &mut [f32]) {
        use super::backend::SimdBackend;

        #[cfg(feature = "force-scalar")]
        {
            ScalarBackend::neg(input, output);
        }

        #[cfg(all(
            feature = "force-avx2",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            use super::avx2::Avx2Backend;
            Avx2Backend::neg(input, output);
        }

        #[cfg(all(
            feature = "force-avx512",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            use super::avx512::Avx512Backend;
            Avx512Backend::neg(input, output);
        }

        #[cfg(all(feature = "force-neon", target_arch = "aarch64"))]
        {
            use super::neon::NeonBackend;
            NeonBackend::neg(input, output);
        }

        #[cfg(not(any(
            feature = "force-scalar",
            feature = "force-avx2",
            feature = "force-avx512",
            feature = "force-neon"
        )))]
        {
            #[cfg(all(feature = "neon", target_arch = "aarch64"))]
            {
                use super::neon::NeonBackend;
                NeonBackend::neg(input, output);
            }

            #[cfg(not(all(feature = "neon", target_arch = "aarch64")))]
            {
                ScalarBackend::neg(input, output);
            }
        }
    }

    /// Element-wise absolute value: output[i] = |input[i]|
    ///
    /// # Parameters
    /// - `input`: Input array
    /// - `output`: Output array (must be same length as input)
    ///
    /// # Example
    /// ```ignore
    /// let ctx = SimdContext::new();
    /// let input = [1.0, -2.0, 3.0, -4.0];
    /// let mut output = [0.0; 4];
    /// ctx.abs(&input, &mut output);
    /// // output is now [1.0, 2.0, 3.0, 4.0]
    /// ```
    #[inline]
    pub fn abs(&self, input: &[f32], output: &mut [f32]) {
        #[cfg(feature = "runtime-dispatch")]
        {
            self.dispatcher.abs(input, output)
        }

        #[cfg(not(feature = "runtime-dispatch"))]
        {
            Self::compile_time_abs(input, output)
        }
    }

    #[cfg(not(feature = "runtime-dispatch"))]
    fn compile_time_abs(input: &[f32], output: &mut [f32]) {
        use super::backend::SimdBackend;

        #[cfg(feature = "force-scalar")]
        {
            ScalarBackend::abs(input, output);
        }

        #[cfg(all(
            feature = "force-avx2",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            use super::avx2::Avx2Backend;
            Avx2Backend::abs(input, output);
        }

        #[cfg(all(
            feature = "force-avx512",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            use super::avx512::Avx512Backend;
            Avx512Backend::abs(input, output);
        }

        #[cfg(all(feature = "force-neon", target_arch = "aarch64"))]
        {
            use super::neon::NeonBackend;
            NeonBackend::abs(input, output);
        }

        #[cfg(not(any(
            feature = "force-scalar",
            feature = "force-avx2",
            feature = "force-avx512",
            feature = "force-neon"
        )))]
        {
            #[cfg(all(feature = "neon", target_arch = "aarch64"))]
            {
                use super::neon::NeonBackend;
                NeonBackend::abs(input, output);
            }

            #[cfg(not(all(feature = "neon", target_arch = "aarch64")))]
            {
                ScalarBackend::abs(input, output);
            }
        }
    }

    // ========================================================================
    // Comparison Operations
    // ========================================================================

    /// Element-wise minimum: output[i] = min(a[i], b[i])
    ///
    /// # Parameters
    /// - `a`: First input array
    /// - `b`: Second input array
    /// - `output`: Output array (must be same length as inputs)
    ///
    /// # Example
    /// ```ignore
    /// let ctx = SimdContext::new();
    /// let a = [1.0, 2.0, 3.0, 4.0];
    /// let b = [2.0, 1.0, 4.0, 3.0];
    /// let mut output = [0.0; 4];
    /// ctx.min(&a, &b, &mut output);
    /// // output is now [1.0, 1.0, 3.0, 3.0]
    /// ```
    #[inline]
    pub fn min(&self, a: &[f32], b: &[f32], output: &mut [f32]) {
        #[cfg(feature = "runtime-dispatch")]
        {
            self.dispatcher.min(a, b, output)
        }

        #[cfg(not(feature = "runtime-dispatch"))]
        {
            Self::compile_time_min(a, b, output)
        }
    }

    #[cfg(not(feature = "runtime-dispatch"))]
    fn compile_time_min(a: &[f32], b: &[f32], output: &mut [f32]) {
        use super::backend::SimdBackend;

        #[cfg(feature = "force-scalar")]
        {
            ScalarBackend::min(a, b, output);
        }

        #[cfg(all(
            feature = "force-avx2",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            use super::avx2::Avx2Backend;
            Avx2Backend::min(a, b, output);
        }

        #[cfg(all(
            feature = "force-avx512",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            use super::avx512::Avx512Backend;
            Avx512Backend::min(a, b, output);
        }

        #[cfg(all(feature = "force-neon", target_arch = "aarch64"))]
        {
            use super::neon::NeonBackend;
            NeonBackend::min(a, b, output);
        }

        #[cfg(not(any(
            feature = "force-scalar",
            feature = "force-avx2",
            feature = "force-avx512",
            feature = "force-neon"
        )))]
        {
            #[cfg(all(feature = "neon", target_arch = "aarch64"))]
            {
                use super::neon::NeonBackend;
                NeonBackend::min(a, b, output);
            }

            #[cfg(not(all(feature = "neon", target_arch = "aarch64")))]
            {
                ScalarBackend::min(a, b, output);
            }
        }
    }

    /// Element-wise maximum: output[i] = max(a[i], b[i])
    ///
    /// # Parameters
    /// - `a`: First input array
    /// - `b`: Second input array
    /// - `output`: Output array (must be same length as inputs)
    ///
    /// # Example
    /// ```ignore
    /// let ctx = SimdContext::new();
    /// let a = [1.0, 2.0, 3.0, 4.0];
    /// let b = [2.0, 1.0, 4.0, 3.0];
    /// let mut output = [0.0; 4];
    /// ctx.max(&a, &b, &mut output);
    /// // output is now [2.0, 2.0, 4.0, 4.0]
    /// ```
    #[inline]
    pub fn max(&self, a: &[f32], b: &[f32], output: &mut [f32]) {
        #[cfg(feature = "runtime-dispatch")]
        {
            self.dispatcher.max(a, b, output)
        }

        #[cfg(not(feature = "runtime-dispatch"))]
        {
            Self::compile_time_max(a, b, output)
        }
    }

    #[cfg(not(feature = "runtime-dispatch"))]
    fn compile_time_max(a: &[f32], b: &[f32], output: &mut [f32]) {
        use super::backend::SimdBackend;

        #[cfg(feature = "force-scalar")]
        {
            ScalarBackend::max(a, b, output);
        }

        #[cfg(all(
            feature = "force-avx2",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            use super::avx2::Avx2Backend;
            Avx2Backend::max(a, b, output);
        }

        #[cfg(all(
            feature = "force-avx512",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            use super::avx512::Avx512Backend;
            Avx512Backend::max(a, b, output);
        }

        #[cfg(all(feature = "force-neon", target_arch = "aarch64"))]
        {
            use super::neon::NeonBackend;
            NeonBackend::max(a, b, output);
        }

        #[cfg(not(any(
            feature = "force-scalar",
            feature = "force-avx2",
            feature = "force-avx512",
            feature = "force-neon"
        )))]
        {
            #[cfg(all(feature = "neon", target_arch = "aarch64"))]
            {
                use super::neon::NeonBackend;
                NeonBackend::max(a, b, output);
            }

            #[cfg(not(all(feature = "neon", target_arch = "aarch64")))]
            {
                ScalarBackend::max(a, b, output);
            }
        }
    }

    // ========================================================================
    // Basic Math Functions
    // ========================================================================

    /// Element-wise square root: output[i] = sqrt(input[i])
    ///
    /// # Parameters
    /// - `input`: Input array
    /// - `output`: Output array (must be same length as input)
    ///
    /// # Example
    /// ```ignore
    /// let ctx = SimdContext::new();
    /// let input = [1.0, 4.0, 9.0, 16.0];
    /// let mut output = [0.0; 4];
    /// ctx.sqrt(&input, &mut output);
    /// // output is now [1.0, 2.0, 3.0, 4.0]
    /// ```
    #[inline]
    pub fn sqrt(&self, input: &[f32], output: &mut [f32]) {
        #[cfg(feature = "runtime-dispatch")]
        {
            self.dispatcher.sqrt(input, output)
        }

        #[cfg(not(feature = "runtime-dispatch"))]
        {
            Self::compile_time_sqrt(input, output)
        }
    }

    #[cfg(not(feature = "runtime-dispatch"))]
    fn compile_time_sqrt(input: &[f32], output: &mut [f32]) {
        use super::backend::SimdBackend;

        #[cfg(feature = "force-scalar")]
        {
            ScalarBackend::sqrt(input, output);
        }

        #[cfg(all(
            feature = "force-avx2",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            use super::avx2::Avx2Backend;
            Avx2Backend::sqrt(input, output);
        }

        #[cfg(all(
            feature = "force-avx512",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            use super::avx512::Avx512Backend;
            Avx512Backend::sqrt(input, output);
        }

        #[cfg(all(feature = "force-neon", target_arch = "aarch64"))]
        {
            use super::neon::NeonBackend;
            NeonBackend::sqrt(input, output);
        }

        #[cfg(not(any(
            feature = "force-scalar",
            feature = "force-avx2",
            feature = "force-avx512",
            feature = "force-neon"
        )))]
        {
            #[cfg(all(feature = "neon", target_arch = "aarch64"))]
            {
                use super::neon::NeonBackend;
                NeonBackend::sqrt(input, output);
            }

            #[cfg(not(all(feature = "neon", target_arch = "aarch64")))]
            {
                ScalarBackend::sqrt(input, output);
            }
        }
    }

    /// Element-wise exponential: output[i] = e^input[i]
    ///
    /// # Parameters
    /// - `input`: Input array
    /// - `output`: Output array (must be same length as input)
    ///
    /// # Example
    /// ```ignore
    /// let ctx = SimdContext::new();
    /// let input = [0.0, 1.0, 2.0];
    /// let mut output = [0.0; 3];
    /// ctx.exp(&input, &mut output);
    /// // output is approximately [1.0, 2.718, 7.389]
    /// ```
    #[inline]
    pub fn exp(&self, input: &[f32], output: &mut [f32]) {
        #[cfg(feature = "runtime-dispatch")]
        {
            self.dispatcher.exp(input, output)
        }

        #[cfg(not(feature = "runtime-dispatch"))]
        {
            Self::compile_time_exp(input, output)
        }
    }

    #[cfg(not(feature = "runtime-dispatch"))]
    fn compile_time_exp(input: &[f32], output: &mut [f32]) {
        use super::backend::SimdBackend;

        #[cfg(feature = "force-scalar")]
        {
            ScalarBackend::exp(input, output);
        }

        #[cfg(all(
            feature = "force-avx2",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            use super::avx2::Avx2Backend;
            Avx2Backend::exp(input, output);
        }

        #[cfg(all(
            feature = "force-avx512",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            use super::avx512::Avx512Backend;
            Avx512Backend::exp(input, output);
        }

        #[cfg(all(feature = "force-neon", target_arch = "aarch64"))]
        {
            use super::neon::NeonBackend;
            NeonBackend::exp(input, output);
        }

        #[cfg(not(any(
            feature = "force-scalar",
            feature = "force-avx2",
            feature = "force-avx512",
            feature = "force-neon"
        )))]
        {
            #[cfg(all(feature = "neon", target_arch = "aarch64"))]
            {
                use super::neon::NeonBackend;
                NeonBackend::exp(input, output);
            }

            #[cfg(not(all(feature = "neon", target_arch = "aarch64")))]
            {
                ScalarBackend::exp(input, output);
            }
        }
    }

    /// Element-wise natural logarithm: output[i] = ln(input[i])
    ///
    /// # Parameters
    /// - `input`: Input array
    /// - `output`: Output array (must be same length as input)
    ///
    /// # Example
    /// ```ignore
    /// let ctx = SimdContext::new();
    /// let input = [1.0, 2.718, 7.389];
    /// let mut output = [0.0; 3];
    /// ctx.log(&input, &mut output);
    /// // output is approximately [0.0, 1.0, 2.0]
    /// ```
    #[inline]
    pub fn log(&self, input: &[f32], output: &mut [f32]) {
        #[cfg(feature = "runtime-dispatch")]
        {
            self.dispatcher.log(input, output)
        }

        #[cfg(not(feature = "runtime-dispatch"))]
        {
            Self::compile_time_log(input, output)
        }
    }

    #[cfg(not(feature = "runtime-dispatch"))]
    fn compile_time_log(input: &[f32], output: &mut [f32]) {
        use super::backend::SimdBackend;

        #[cfg(feature = "force-scalar")]
        {
            ScalarBackend::log(input, output);
        }

        #[cfg(all(
            feature = "force-avx2",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            use super::avx2::Avx2Backend;
            Avx2Backend::log(input, output);
        }

        #[cfg(all(
            feature = "force-avx512",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            use super::avx512::Avx512Backend;
            Avx512Backend::log(input, output);
        }

        #[cfg(all(feature = "force-neon", target_arch = "aarch64"))]
        {
            use super::neon::NeonBackend;
            NeonBackend::log(input, output);
        }

        #[cfg(not(any(
            feature = "force-scalar",
            feature = "force-avx2",
            feature = "force-avx512",
            feature = "force-neon"
        )))]
        {
            #[cfg(all(feature = "neon", target_arch = "aarch64"))]
            {
                use super::neon::NeonBackend;
                NeonBackend::log(input, output);
            }

            #[cfg(not(all(feature = "neon", target_arch = "aarch64")))]
            {
                ScalarBackend::log(input, output);
            }
        }
    }

    /// Element-wise base-2 logarithm: output[i] = log2(input[i])
    ///
    /// # Parameters
    /// - `input`: Input array
    /// - `output`: Output array (must be same length as input)
    ///
    /// # Example
    /// ```ignore
    /// let ctx = SimdContext::new();
    /// let input = [1.0, 2.0, 4.0, 8.0];
    /// let mut output = [0.0; 4];
    /// ctx.log2(&input, &mut output);
    /// // output is now [0.0, 1.0, 2.0, 3.0]
    /// ```
    #[inline]
    pub fn log2(&self, input: &[f32], output: &mut [f32]) {
        #[cfg(feature = "runtime-dispatch")]
        {
            self.dispatcher.log2(input, output)
        }

        #[cfg(not(feature = "runtime-dispatch"))]
        {
            Self::compile_time_log2(input, output)
        }
    }

    #[cfg(not(feature = "runtime-dispatch"))]
    fn compile_time_log2(input: &[f32], output: &mut [f32]) {
        use super::backend::SimdBackend;

        #[cfg(feature = "force-scalar")]
        {
            ScalarBackend::log2(input, output);
        }

        #[cfg(all(
            feature = "force-avx2",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            use super::avx2::Avx2Backend;
            Avx2Backend::log2(input, output);
        }

        #[cfg(all(
            feature = "force-avx512",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            use super::avx512::Avx512Backend;
            Avx512Backend::log2(input, output);
        }

        #[cfg(all(feature = "force-neon", target_arch = "aarch64"))]
        {
            use super::neon::NeonBackend;
            NeonBackend::log2(input, output);
        }

        #[cfg(not(any(
            feature = "force-scalar",
            feature = "force-avx2",
            feature = "force-avx512",
            feature = "force-neon"
        )))]
        {
            #[cfg(all(feature = "neon", target_arch = "aarch64"))]
            {
                use super::neon::NeonBackend;
                NeonBackend::log2(input, output);
            }

            #[cfg(not(all(feature = "neon", target_arch = "aarch64")))]
            {
                ScalarBackend::log2(input, output);
            }
        }
    }

    /// Element-wise base-10 logarithm: output[i] = log10(input[i])
    ///
    /// # Parameters
    /// - `input`: Input array
    /// - `output`: Output array (must be same length as input)
    ///
    /// # Example
    /// ```ignore
    /// let ctx = SimdContext::new();
    /// let input = [1.0, 10.0, 100.0, 1000.0];
    /// let mut output = [0.0; 4];
    /// ctx.log10(&input, &mut output);
    /// // output is now [0.0, 1.0, 2.0, 3.0]
    /// ```
    #[inline]
    pub fn log10(&self, input: &[f32], output: &mut [f32]) {
        #[cfg(feature = "runtime-dispatch")]
        {
            self.dispatcher.log10(input, output)
        }

        #[cfg(not(feature = "runtime-dispatch"))]
        {
            Self::compile_time_log10(input, output)
        }
    }

    #[cfg(not(feature = "runtime-dispatch"))]
    fn compile_time_log10(input: &[f32], output: &mut [f32]) {
        use super::backend::SimdBackend;

        #[cfg(feature = "force-scalar")]
        {
            ScalarBackend::log10(input, output);
        }

        #[cfg(all(
            feature = "force-avx2",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            use super::avx2::Avx2Backend;
            Avx2Backend::log10(input, output);
        }

        #[cfg(all(
            feature = "force-avx512",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            use super::avx512::Avx512Backend;
            Avx512Backend::log10(input, output);
        }

        #[cfg(all(feature = "force-neon", target_arch = "aarch64"))]
        {
            use super::neon::NeonBackend;
            NeonBackend::log10(input, output);
        }

        #[cfg(not(any(
            feature = "force-scalar",
            feature = "force-avx2",
            feature = "force-avx512",
            feature = "force-neon"
        )))]
        {
            #[cfg(all(feature = "neon", target_arch = "aarch64"))]
            {
                use super::neon::NeonBackend;
                NeonBackend::log10(input, output);
            }

            #[cfg(not(all(feature = "neon", target_arch = "aarch64")))]
            {
                ScalarBackend::log10(input, output);
            }
        }
    }

    /// Element-wise power: output[i] = base[i]^exponent[i]
    ///
    /// # Parameters
    /// - `base`: Base values
    /// - `exponent`: Exponent values
    /// - `output`: Output array (must be same length as inputs)
    ///
    /// # Example
    /// ```ignore
    /// let ctx = SimdContext::new();
    /// let base = [2.0, 3.0, 4.0, 5.0];
    /// let exponent = [2.0, 2.0, 2.0, 2.0];
    /// let mut output = [0.0; 4];
    /// ctx.pow(&base, &exponent, &mut output);
    /// // output is now [4.0, 9.0, 16.0, 25.0]
    /// ```
    #[inline]
    pub fn pow(&self, base: &[f32], exponent: &[f32], output: &mut [f32]) {
        #[cfg(feature = "runtime-dispatch")]
        {
            self.dispatcher.pow(base, exponent, output)
        }

        #[cfg(not(feature = "runtime-dispatch"))]
        {
            Self::compile_time_pow(base, exponent, output)
        }
    }

    #[cfg(not(feature = "runtime-dispatch"))]
    fn compile_time_pow(base: &[f32], exponent: &[f32], output: &mut [f32]) {
        use super::backend::SimdBackend;

        #[cfg(feature = "force-scalar")]
        {
            ScalarBackend::pow(base, exponent, output);
        }

        #[cfg(all(
            feature = "force-avx2",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            use super::avx2::Avx2Backend;
            Avx2Backend::pow(base, exponent, output);
        }

        #[cfg(all(
            feature = "force-avx512",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            use super::avx512::Avx512Backend;
            Avx512Backend::pow(base, exponent, output);
        }

        #[cfg(all(feature = "force-neon", target_arch = "aarch64"))]
        {
            use super::neon::NeonBackend;
            NeonBackend::pow(base, exponent, output);
        }

        #[cfg(not(any(
            feature = "force-scalar",
            feature = "force-avx2",
            feature = "force-avx512",
            feature = "force-neon"
        )))]
        {
            #[cfg(all(feature = "neon", target_arch = "aarch64"))]
            {
                use super::neon::NeonBackend;
                NeonBackend::pow(base, exponent, output);
            }

            #[cfg(not(all(feature = "neon", target_arch = "aarch64")))]
            {
                ScalarBackend::pow(base, exponent, output);
            }
        }
    }

    // ========================================================================
    // Trigonometric Functions
    // ========================================================================

    /// Element-wise sine: output[i] = sin(input[i])
    ///
    /// # Parameters
    /// - `input`: Input array (radians)
    /// - `output`: Output array (must be same length as input)
    ///
    /// # Example
    /// ```ignore
    /// let ctx = SimdContext::new();
    /// let input = [0.0, std::f32::consts::PI / 2.0];
    /// let mut output = [0.0; 2];
    /// ctx.sin(&input, &mut output);
    /// // output is approximately [0.0, 1.0]
    /// ```
    #[inline]
    pub fn sin(&self, input: &[f32], output: &mut [f32]) {
        #[cfg(feature = "runtime-dispatch")]
        {
            self.dispatcher.sin(input, output)
        }

        #[cfg(not(feature = "runtime-dispatch"))]
        {
            Self::compile_time_sin(input, output)
        }
    }

    #[cfg(not(feature = "runtime-dispatch"))]
    fn compile_time_sin(input: &[f32], output: &mut [f32]) {
        use super::backend::SimdBackend;

        #[cfg(feature = "force-scalar")]
        {
            ScalarBackend::sin(input, output);
        }

        #[cfg(all(
            feature = "force-avx2",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            use super::avx2::Avx2Backend;
            Avx2Backend::sin(input, output);
        }

        #[cfg(all(
            feature = "force-avx512",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            use super::avx512::Avx512Backend;
            Avx512Backend::sin(input, output);
        }

        #[cfg(all(feature = "force-neon", target_arch = "aarch64"))]
        {
            use super::neon::NeonBackend;
            NeonBackend::sin(input, output);
        }

        #[cfg(not(any(
            feature = "force-scalar",
            feature = "force-avx2",
            feature = "force-avx512",
            feature = "force-neon"
        )))]
        {
            #[cfg(all(feature = "neon", target_arch = "aarch64"))]
            {
                use super::neon::NeonBackend;
                NeonBackend::sin(input, output);
            }

            #[cfg(not(all(feature = "neon", target_arch = "aarch64")))]
            {
                ScalarBackend::sin(input, output);
            }
        }
    }

    /// Element-wise cosine: output[i] = cos(input[i])
    ///
    /// # Parameters
    /// - `input`: Input array (radians)
    /// - `output`: Output array (must be same length as input)
    ///
    /// # Example
    /// ```ignore
    /// let ctx = SimdContext::new();
    /// let input = [0.0, std::f32::consts::PI];
    /// let mut output = [0.0; 2];
    /// ctx.cos(&input, &mut output);
    /// // output is approximately [1.0, -1.0]
    /// ```
    #[inline]
    pub fn cos(&self, input: &[f32], output: &mut [f32]) {
        #[cfg(feature = "runtime-dispatch")]
        {
            self.dispatcher.cos(input, output)
        }

        #[cfg(not(feature = "runtime-dispatch"))]
        {
            Self::compile_time_cos(input, output)
        }
    }

    #[cfg(not(feature = "runtime-dispatch"))]
    fn compile_time_cos(input: &[f32], output: &mut [f32]) {
        use super::backend::SimdBackend;

        #[cfg(feature = "force-scalar")]
        {
            ScalarBackend::cos(input, output);
        }

        #[cfg(all(
            feature = "force-avx2",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            use super::avx2::Avx2Backend;
            Avx2Backend::cos(input, output);
        }

        #[cfg(all(
            feature = "force-avx512",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            use super::avx512::Avx512Backend;
            Avx512Backend::cos(input, output);
        }

        #[cfg(all(feature = "force-neon", target_arch = "aarch64"))]
        {
            use super::neon::NeonBackend;
            NeonBackend::cos(input, output);
        }

        #[cfg(not(any(
            feature = "force-scalar",
            feature = "force-avx2",
            feature = "force-avx512",
            feature = "force-neon"
        )))]
        {
            #[cfg(all(feature = "neon", target_arch = "aarch64"))]
            {
                use super::neon::NeonBackend;
                NeonBackend::cos(input, output);
            }

            #[cfg(not(all(feature = "neon", target_arch = "aarch64")))]
            {
                ScalarBackend::cos(input, output);
            }
        }
    }

    /// Element-wise tangent: output[i] = tan(input[i])
    ///
    /// # Parameters
    /// - `input`: Input array (radians)
    /// - `output`: Output array (must be same length as input)
    ///
    /// # Example
    /// ```ignore
    /// let ctx = SimdContext::new();
    /// let input = [0.0, std::f32::consts::PI / 4.0];
    /// let mut output = [0.0; 2];
    /// ctx.tan(&input, &mut output);
    /// // output is approximately [0.0, 1.0]
    /// ```
    #[inline]
    pub fn tan(&self, input: &[f32], output: &mut [f32]) {
        #[cfg(feature = "runtime-dispatch")]
        {
            self.dispatcher.tan(input, output)
        }

        #[cfg(not(feature = "runtime-dispatch"))]
        {
            Self::compile_time_tan(input, output)
        }
    }

    #[cfg(not(feature = "runtime-dispatch"))]
    fn compile_time_tan(input: &[f32], output: &mut [f32]) {
        use super::backend::SimdBackend;

        #[cfg(feature = "force-scalar")]
        {
            ScalarBackend::tan(input, output);
        }

        #[cfg(all(
            feature = "force-avx2",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            use super::avx2::Avx2Backend;
            Avx2Backend::tan(input, output);
        }

        #[cfg(all(
            feature = "force-avx512",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            use super::avx512::Avx512Backend;
            Avx512Backend::tan(input, output);
        }

        #[cfg(all(feature = "force-neon", target_arch = "aarch64"))]
        {
            use super::neon::NeonBackend;
            NeonBackend::tan(input, output);
        }

        #[cfg(not(any(
            feature = "force-scalar",
            feature = "force-avx2",
            feature = "force-avx512",
            feature = "force-neon"
        )))]
        {
            #[cfg(all(feature = "neon", target_arch = "aarch64"))]
            {
                use super::neon::NeonBackend;
                NeonBackend::tan(input, output);
            }

            #[cfg(not(all(feature = "neon", target_arch = "aarch64")))]
            {
                ScalarBackend::tan(input, output);
            }
        }
    }

    /// Element-wise arcsine: output[i] = asin(input[i])
    ///
    /// # Parameters
    /// - `input`: Input array (range [-1, 1])
    /// - `output`: Output array (must be same length as input)
    ///
    /// # Example
    /// ```ignore
    /// let ctx = SimdContext::new();
    /// let input = [0.0, 0.5, 1.0];
    /// let mut output = [0.0; 3];
    /// ctx.asin(&input, &mut output);
    /// // output is approximately [0.0, 0.524, 1.571]
    /// ```
    #[inline]
    pub fn asin(&self, input: &[f32], output: &mut [f32]) {
        #[cfg(feature = "runtime-dispatch")]
        {
            self.dispatcher.asin(input, output)
        }

        #[cfg(not(feature = "runtime-dispatch"))]
        {
            Self::compile_time_asin(input, output)
        }
    }

    #[cfg(not(feature = "runtime-dispatch"))]
    fn compile_time_asin(input: &[f32], output: &mut [f32]) {
        use super::backend::SimdBackend;

        #[cfg(feature = "force-scalar")]
        {
            ScalarBackend::asin(input, output);
        }

        #[cfg(all(
            feature = "force-avx2",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            use super::avx2::Avx2Backend;
            Avx2Backend::asin(input, output);
        }

        #[cfg(all(
            feature = "force-avx512",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            use super::avx512::Avx512Backend;
            Avx512Backend::asin(input, output);
        }

        #[cfg(all(feature = "force-neon", target_arch = "aarch64"))]
        {
            use super::neon::NeonBackend;
            NeonBackend::asin(input, output);
        }

        #[cfg(not(any(
            feature = "force-scalar",
            feature = "force-avx2",
            feature = "force-avx512",
            feature = "force-neon"
        )))]
        {
            #[cfg(all(feature = "neon", target_arch = "aarch64"))]
            {
                use super::neon::NeonBackend;
                NeonBackend::asin(input, output);
            }

            #[cfg(not(all(feature = "neon", target_arch = "aarch64")))]
            {
                ScalarBackend::asin(input, output);
            }
        }
    }

    /// Element-wise arccosine: output[i] = acos(input[i])
    ///
    /// # Parameters
    /// - `input`: Input array (range [-1, 1])
    /// - `output`: Output array (must be same length as input)
    ///
    /// # Example
    /// ```ignore
    /// let ctx = SimdContext::new();
    /// let input = [0.0, 0.5, 1.0];
    /// let mut output = [0.0; 3];
    /// ctx.acos(&input, &mut output);
    /// // output is approximately [1.571, 1.047, 0.0]
    /// ```
    #[inline]
    pub fn acos(&self, input: &[f32], output: &mut [f32]) {
        #[cfg(feature = "runtime-dispatch")]
        {
            self.dispatcher.acos(input, output)
        }

        #[cfg(not(feature = "runtime-dispatch"))]
        {
            Self::compile_time_acos(input, output)
        }
    }

    #[cfg(not(feature = "runtime-dispatch"))]
    fn compile_time_acos(input: &[f32], output: &mut [f32]) {
        use super::backend::SimdBackend;

        #[cfg(feature = "force-scalar")]
        {
            ScalarBackend::acos(input, output);
        }

        #[cfg(all(
            feature = "force-avx2",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            use super::avx2::Avx2Backend;
            Avx2Backend::acos(input, output);
        }

        #[cfg(all(
            feature = "force-avx512",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            use super::avx512::Avx512Backend;
            Avx512Backend::acos(input, output);
        }

        #[cfg(all(feature = "force-neon", target_arch = "aarch64"))]
        {
            use super::neon::NeonBackend;
            NeonBackend::acos(input, output);
        }

        #[cfg(not(any(
            feature = "force-scalar",
            feature = "force-avx2",
            feature = "force-avx512",
            feature = "force-neon"
        )))]
        {
            #[cfg(all(feature = "neon", target_arch = "aarch64"))]
            {
                use super::neon::NeonBackend;
                NeonBackend::acos(input, output);
            }

            #[cfg(not(all(feature = "neon", target_arch = "aarch64")))]
            {
                ScalarBackend::acos(input, output);
            }
        }
    }

    /// Element-wise arctangent: output[i] = atan(input[i])
    ///
    /// # Parameters
    /// - `input`: Input array
    /// - `output`: Output array (must be same length as input)
    ///
    /// # Example
    /// ```ignore
    /// let ctx = SimdContext::new();
    /// let input = [0.0, 1.0, -1.0];
    /// let mut output = [0.0; 3];
    /// ctx.atan(&input, &mut output);
    /// // output is approximately [0.0, 0.785, -0.785]
    /// ```
    #[inline]
    pub fn atan(&self, input: &[f32], output: &mut [f32]) {
        #[cfg(feature = "runtime-dispatch")]
        {
            self.dispatcher.atan(input, output)
        }

        #[cfg(not(feature = "runtime-dispatch"))]
        {
            Self::compile_time_atan(input, output)
        }
    }

    #[cfg(not(feature = "runtime-dispatch"))]
    fn compile_time_atan(input: &[f32], output: &mut [f32]) {
        use super::backend::SimdBackend;

        #[cfg(feature = "force-scalar")]
        {
            ScalarBackend::atan(input, output);
        }

        #[cfg(all(
            feature = "force-avx2",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            use super::avx2::Avx2Backend;
            Avx2Backend::atan(input, output);
        }

        #[cfg(all(
            feature = "force-avx512",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            use super::avx512::Avx512Backend;
            Avx512Backend::atan(input, output);
        }

        #[cfg(all(feature = "force-neon", target_arch = "aarch64"))]
        {
            use super::neon::NeonBackend;
            NeonBackend::atan(input, output);
        }

        #[cfg(not(any(
            feature = "force-scalar",
            feature = "force-avx2",
            feature = "force-avx512",
            feature = "force-neon"
        )))]
        {
            #[cfg(all(feature = "neon", target_arch = "aarch64"))]
            {
                use super::neon::NeonBackend;
                NeonBackend::atan(input, output);
            }

            #[cfg(not(all(feature = "neon", target_arch = "aarch64")))]
            {
                ScalarBackend::atan(input, output);
            }
        }
    }

    /// Element-wise two-argument arctangent: output[i] = atan2(y[i], x[i])
    ///
    /// # Parameters
    /// - `y`: Y-coordinates
    /// - `x`: X-coordinates
    /// - `output`: Output array (must be same length as inputs)
    ///
    /// # Example
    /// ```ignore
    /// let ctx = SimdContext::new();
    /// let y = [1.0, 1.0, -1.0, -1.0];
    /// let x = [1.0, -1.0, 1.0, -1.0];
    /// let mut output = [0.0; 4];
    /// ctx.atan2(&y, &x, &mut output);
    /// // output contains angles in radians for each (y, x) pair
    /// ```
    #[inline]
    pub fn atan2(&self, y: &[f32], x: &[f32], output: &mut [f32]) {
        #[cfg(feature = "runtime-dispatch")]
        {
            self.dispatcher.atan2(y, x, output)
        }

        #[cfg(not(feature = "runtime-dispatch"))]
        {
            Self::compile_time_atan2(y, x, output)
        }
    }

    #[cfg(not(feature = "runtime-dispatch"))]
    fn compile_time_atan2(y: &[f32], x: &[f32], output: &mut [f32]) {
        use super::backend::SimdBackend;

        #[cfg(feature = "force-scalar")]
        {
            ScalarBackend::atan2(y, x, output);
        }

        #[cfg(all(
            feature = "force-avx2",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            use super::avx2::Avx2Backend;
            Avx2Backend::atan2(y, x, output);
        }

        #[cfg(all(
            feature = "force-avx512",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            use super::avx512::Avx512Backend;
            Avx512Backend::atan2(y, x, output);
        }

        #[cfg(all(feature = "force-neon", target_arch = "aarch64"))]
        {
            use super::neon::NeonBackend;
            NeonBackend::atan2(y, x, output);
        }

        #[cfg(not(any(
            feature = "force-scalar",
            feature = "force-avx2",
            feature = "force-avx512",
            feature = "force-neon"
        )))]
        {
            #[cfg(all(feature = "neon", target_arch = "aarch64"))]
            {
                use super::neon::NeonBackend;
                NeonBackend::atan2(y, x, output);
            }

            #[cfg(not(all(feature = "neon", target_arch = "aarch64")))]
            {
                ScalarBackend::atan2(y, x, output);
            }
        }
    }

    // ========================================================================
    // Hyperbolic Functions
    // ========================================================================

    /// Element-wise hyperbolic sine: output[i] = sinh(input[i])
    ///
    /// # Parameters
    /// - `input`: Input array
    /// - `output`: Output array (must be same length as input)
    ///
    /// # Example
    /// ```ignore
    /// let ctx = SimdContext::new();
    /// let input = [0.0, 1.0, -1.0];
    /// let mut output = [0.0; 3];
    /// ctx.sinh(&input, &mut output);
    /// // output is approximately [0.0, 1.175, -1.175]
    /// ```
    #[inline]
    pub fn sinh(&self, input: &[f32], output: &mut [f32]) {
        #[cfg(feature = "runtime-dispatch")]
        {
            self.dispatcher.sinh(input, output)
        }

        #[cfg(not(feature = "runtime-dispatch"))]
        {
            Self::compile_time_sinh(input, output)
        }
    }

    #[cfg(not(feature = "runtime-dispatch"))]
    fn compile_time_sinh(input: &[f32], output: &mut [f32]) {
        use super::backend::SimdBackend;

        #[cfg(feature = "force-scalar")]
        {
            ScalarBackend::sinh(input, output);
        }

        #[cfg(all(
            feature = "force-avx2",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            use super::avx2::Avx2Backend;
            Avx2Backend::sinh(input, output);
        }

        #[cfg(all(
            feature = "force-avx512",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            use super::avx512::Avx512Backend;
            Avx512Backend::sinh(input, output);
        }

        #[cfg(all(feature = "force-neon", target_arch = "aarch64"))]
        {
            use super::neon::NeonBackend;
            NeonBackend::sinh(input, output);
        }

        #[cfg(not(any(
            feature = "force-scalar",
            feature = "force-avx2",
            feature = "force-avx512",
            feature = "force-neon"
        )))]
        {
            #[cfg(all(feature = "neon", target_arch = "aarch64"))]
            {
                use super::neon::NeonBackend;
                NeonBackend::sinh(input, output);
            }

            #[cfg(not(all(feature = "neon", target_arch = "aarch64")))]
            {
                ScalarBackend::sinh(input, output);
            }
        }
    }

    /// Element-wise hyperbolic cosine: output[i] = cosh(input[i])
    ///
    /// # Parameters
    /// - `input`: Input array
    /// - `output`: Output array (must be same length as input)
    ///
    /// # Example
    /// ```ignore
    /// let ctx = SimdContext::new();
    /// let input = [0.0, 1.0, -1.0];
    /// let mut output = [0.0; 3];
    /// ctx.cosh(&input, &mut output);
    /// // output is approximately [1.0, 1.543, 1.543]
    /// ```
    #[inline]
    pub fn cosh(&self, input: &[f32], output: &mut [f32]) {
        #[cfg(feature = "runtime-dispatch")]
        {
            self.dispatcher.cosh(input, output)
        }

        #[cfg(not(feature = "runtime-dispatch"))]
        {
            Self::compile_time_cosh(input, output)
        }
    }

    #[cfg(not(feature = "runtime-dispatch"))]
    fn compile_time_cosh(input: &[f32], output: &mut [f32]) {
        use super::backend::SimdBackend;

        #[cfg(feature = "force-scalar")]
        {
            ScalarBackend::cosh(input, output);
        }

        #[cfg(all(
            feature = "force-avx2",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            use super::avx2::Avx2Backend;
            Avx2Backend::cosh(input, output);
        }

        #[cfg(all(
            feature = "force-avx512",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            use super::avx512::Avx512Backend;
            Avx512Backend::cosh(input, output);
        }

        #[cfg(all(feature = "force-neon", target_arch = "aarch64"))]
        {
            use super::neon::NeonBackend;
            NeonBackend::cosh(input, output);
        }

        #[cfg(not(any(
            feature = "force-scalar",
            feature = "force-avx2",
            feature = "force-avx512",
            feature = "force-neon"
        )))]
        {
            #[cfg(all(feature = "neon", target_arch = "aarch64"))]
            {
                use super::neon::NeonBackend;
                NeonBackend::cosh(input, output);
            }

            #[cfg(not(all(feature = "neon", target_arch = "aarch64")))]
            {
                ScalarBackend::cosh(input, output);
            }
        }
    }

    /// Element-wise hyperbolic tangent: output[i] = tanh(input[i])
    ///
    /// # Parameters
    /// - `input`: Input array
    /// - `output`: Output array (must be same length as input)
    ///
    /// # Example
    /// ```ignore
    /// let ctx = SimdContext::new();
    /// let input = [0.0, 1.0, -1.0];
    /// let mut output = [0.0; 3];
    /// ctx.tanh(&input, &mut output);
    /// // output is approximately [0.0, 0.762, -0.762]
    /// ```
    #[inline]
    pub fn tanh(&self, input: &[f32], output: &mut [f32]) {
        #[cfg(feature = "runtime-dispatch")]
        {
            self.dispatcher.tanh(input, output)
        }

        #[cfg(not(feature = "runtime-dispatch"))]
        {
            Self::compile_time_tanh(input, output)
        }
    }

    #[cfg(not(feature = "runtime-dispatch"))]
    fn compile_time_tanh(input: &[f32], output: &mut [f32]) {
        use super::backend::SimdBackend;

        #[cfg(feature = "force-scalar")]
        {
            ScalarBackend::tanh(input, output);
        }

        #[cfg(all(
            feature = "force-avx2",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            use super::avx2::Avx2Backend;
            Avx2Backend::tanh(input, output);
        }

        #[cfg(all(
            feature = "force-avx512",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            use super::avx512::Avx512Backend;
            Avx512Backend::tanh(input, output);
        }

        #[cfg(all(feature = "force-neon", target_arch = "aarch64"))]
        {
            use super::neon::NeonBackend;
            NeonBackend::tanh(input, output);
        }

        #[cfg(not(any(
            feature = "force-scalar",
            feature = "force-avx2",
            feature = "force-avx512",
            feature = "force-neon"
        )))]
        {
            #[cfg(all(feature = "neon", target_arch = "aarch64"))]
            {
                use super::neon::NeonBackend;
                NeonBackend::tanh(input, output);
            }

            #[cfg(not(all(feature = "neon", target_arch = "aarch64")))]
            {
                ScalarBackend::tanh(input, output);
            }
        }
    }

    // ========================================================================
    // Rounding Functions
    // ========================================================================

    /// Element-wise floor: output[i] = floor(input[i])
    ///
    /// # Parameters
    /// - `input`: Input array
    /// - `output`: Output array (must be same length as input)
    ///
    /// # Example
    /// ```ignore
    /// let ctx = SimdContext::new();
    /// let input = [1.5, 2.7, -1.3, -2.8];
    /// let mut output = [0.0; 4];
    /// ctx.floor(&input, &mut output);
    /// // output is now [1.0, 2.0, -2.0, -3.0]
    /// ```
    #[inline]
    pub fn floor(&self, input: &[f32], output: &mut [f32]) {
        #[cfg(feature = "runtime-dispatch")]
        {
            self.dispatcher.floor(input, output)
        }

        #[cfg(not(feature = "runtime-dispatch"))]
        {
            Self::compile_time_floor(input, output)
        }
    }

    #[cfg(not(feature = "runtime-dispatch"))]
    fn compile_time_floor(input: &[f32], output: &mut [f32]) {
        use super::backend::SimdBackend;

        #[cfg(feature = "force-scalar")]
        {
            ScalarBackend::floor(input, output);
        }

        #[cfg(all(
            feature = "force-avx2",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            use super::avx2::Avx2Backend;
            Avx2Backend::floor(input, output);
        }

        #[cfg(all(
            feature = "force-avx512",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            use super::avx512::Avx512Backend;
            Avx512Backend::floor(input, output);
        }

        #[cfg(all(feature = "force-neon", target_arch = "aarch64"))]
        {
            use super::neon::NeonBackend;
            NeonBackend::floor(input, output);
        }

        #[cfg(not(any(
            feature = "force-scalar",
            feature = "force-avx2",
            feature = "force-avx512",
            feature = "force-neon"
        )))]
        {
            #[cfg(all(feature = "neon", target_arch = "aarch64"))]
            {
                use super::neon::NeonBackend;
                NeonBackend::floor(input, output);
            }

            #[cfg(not(all(feature = "neon", target_arch = "aarch64")))]
            {
                ScalarBackend::floor(input, output);
            }
        }
    }

    /// Element-wise ceiling: output[i] = ceil(input[i])
    ///
    /// # Parameters
    /// - `input`: Input array
    /// - `output`: Output array (must be same length as input)
    ///
    /// # Example
    /// ```ignore
    /// let ctx = SimdContext::new();
    /// let input = [1.5, 2.7, -1.3, -2.8];
    /// let mut output = [0.0; 4];
    /// ctx.ceil(&input, &mut output);
    /// // output is now [2.0, 3.0, -1.0, -2.0]
    /// ```
    #[inline]
    pub fn ceil(&self, input: &[f32], output: &mut [f32]) {
        #[cfg(feature = "runtime-dispatch")]
        {
            self.dispatcher.ceil(input, output)
        }

        #[cfg(not(feature = "runtime-dispatch"))]
        {
            Self::compile_time_ceil(input, output)
        }
    }

    #[cfg(not(feature = "runtime-dispatch"))]
    fn compile_time_ceil(input: &[f32], output: &mut [f32]) {
        use super::backend::SimdBackend;

        #[cfg(feature = "force-scalar")]
        {
            ScalarBackend::ceil(input, output);
        }

        #[cfg(all(
            feature = "force-avx2",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            use super::avx2::Avx2Backend;
            Avx2Backend::ceil(input, output);
        }

        #[cfg(all(
            feature = "force-avx512",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            use super::avx512::Avx512Backend;
            Avx512Backend::ceil(input, output);
        }

        #[cfg(all(feature = "force-neon", target_arch = "aarch64"))]
        {
            use super::neon::NeonBackend;
            NeonBackend::ceil(input, output);
        }

        #[cfg(not(any(
            feature = "force-scalar",
            feature = "force-avx2",
            feature = "force-avx512",
            feature = "force-neon"
        )))]
        {
            #[cfg(all(feature = "neon", target_arch = "aarch64"))]
            {
                use super::neon::NeonBackend;
                NeonBackend::ceil(input, output);
            }

            #[cfg(not(all(feature = "neon", target_arch = "aarch64")))]
            {
                ScalarBackend::ceil(input, output);
            }
        }
    }

    /// Element-wise rounding: output[i] = round(input[i])
    ///
    /// # Parameters
    /// - `input`: Input array
    /// - `output`: Output array (must be same length as input)
    ///
    /// # Example
    /// ```ignore
    /// let ctx = SimdContext::new();
    /// let input = [1.5, 2.7, -1.3, -2.8];
    /// let mut output = [0.0; 4];
    /// ctx.round(&input, &mut output);
    /// // output is now [2.0, 3.0, -1.0, -3.0]
    /// ```
    #[inline]
    pub fn round(&self, input: &[f32], output: &mut [f32]) {
        #[cfg(feature = "runtime-dispatch")]
        {
            self.dispatcher.round(input, output)
        }

        #[cfg(not(feature = "runtime-dispatch"))]
        {
            Self::compile_time_round(input, output)
        }
    }

    #[cfg(not(feature = "runtime-dispatch"))]
    fn compile_time_round(input: &[f32], output: &mut [f32]) {
        use super::backend::SimdBackend;

        #[cfg(feature = "force-scalar")]
        {
            ScalarBackend::round(input, output);
        }

        #[cfg(all(
            feature = "force-avx2",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            use super::avx2::Avx2Backend;
            Avx2Backend::round(input, output);
        }

        #[cfg(all(
            feature = "force-avx512",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            use super::avx512::Avx512Backend;
            Avx512Backend::round(input, output);
        }

        #[cfg(all(feature = "force-neon", target_arch = "aarch64"))]
        {
            use super::neon::NeonBackend;
            NeonBackend::round(input, output);
        }

        #[cfg(not(any(
            feature = "force-scalar",
            feature = "force-avx2",
            feature = "force-avx512",
            feature = "force-neon"
        )))]
        {
            #[cfg(all(feature = "neon", target_arch = "aarch64"))]
            {
                use super::neon::NeonBackend;
                NeonBackend::round(input, output);
            }

            #[cfg(not(all(feature = "neon", target_arch = "aarch64")))]
            {
                ScalarBackend::round(input, output);
            }
        }
    }

    /// Element-wise truncation: output[i] = trunc(input[i])
    ///
    /// # Parameters
    /// - `input`: Input array
    /// - `output`: Output array (must be same length as input)
    ///
    /// # Example
    /// ```ignore
    /// let ctx = SimdContext::new();
    /// let input = [1.5, 2.7, -1.3, -2.8];
    /// let mut output = [0.0; 4];
    /// ctx.trunc(&input, &mut output);
    /// // output is now [1.0, 2.0, -1.0, -2.0]
    /// ```
    #[inline]
    pub fn trunc(&self, input: &[f32], output: &mut [f32]) {
        #[cfg(feature = "runtime-dispatch")]
        {
            self.dispatcher.trunc(input, output)
        }

        #[cfg(not(feature = "runtime-dispatch"))]
        {
            Self::compile_time_trunc(input, output)
        }
    }

    #[cfg(not(feature = "runtime-dispatch"))]
    fn compile_time_trunc(input: &[f32], output: &mut [f32]) {
        use super::backend::SimdBackend;

        #[cfg(feature = "force-scalar")]
        {
            ScalarBackend::trunc(input, output);
        }

        #[cfg(all(
            feature = "force-avx2",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            use super::avx2::Avx2Backend;
            Avx2Backend::trunc(input, output);
        }

        #[cfg(all(
            feature = "force-avx512",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            use super::avx512::Avx512Backend;
            Avx512Backend::trunc(input, output);
        }

        #[cfg(all(feature = "force-neon", target_arch = "aarch64"))]
        {
            use super::neon::NeonBackend;
            NeonBackend::trunc(input, output);
        }

        #[cfg(not(any(
            feature = "force-scalar",
            feature = "force-avx2",
            feature = "force-avx512",
            feature = "force-neon"
        )))]
        {
            #[cfg(all(feature = "neon", target_arch = "aarch64"))]
            {
                use super::neon::NeonBackend;
                NeonBackend::trunc(input, output);
            }

            #[cfg(not(all(feature = "neon", target_arch = "aarch64")))]
            {
                ScalarBackend::trunc(input, output);
            }
        }
    }
}

impl Default for SimdContext {
    /// Default implementation calls `SimdContext::new()`
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_context_new() {
        let ctx = SimdContext::new();
        let backend = ctx.backend_name();

        // Verify we got a valid backend name
        assert!(
            backend == "scalar" || backend == "avx2" || backend == "avx512" || backend == "neon"
        );
    }

    #[test]
    fn test_simd_context_backend_name() {
        let ctx = SimdContext::new();
        let name = ctx.backend_name();

        // Should be a non-empty string
        assert!(!name.is_empty());

        // Should be one of the known backends
        let valid_backends = ["scalar", "avx2", "avx512", "neon"];
        assert!(valid_backends.contains(&name));
    }

    #[test]
    fn test_simd_context_process_block() {
        let ctx = SimdContext::new();
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut output = [0.0; 4];
        let params = ProcessParams {
            gain: 0.5,
            frequency: 440.0,
            sample_rate: 44100.0,
        };

        ctx.process_block(&input, &mut output, &params);

        assert_eq!(output, [0.5, 1.0, 1.5, 2.0]);
    }

    #[test]
    fn test_simd_context_apply_gain() {
        let ctx = SimdContext::new();
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut output = [0.0; 4];

        ctx.apply_gain(&input, &mut output, 0.5);

        assert_eq!(output, [0.5, 1.0, 1.5, 2.0]);
    }

    #[test]
    fn test_simd_context_advance_phase() {
        let ctx = SimdContext::new();
        let mut phases = [0.0, 1.0, 2.0, 3.0];
        let increments = [0.1, 0.2, 0.3, 0.4];

        ctx.advance_phase(&mut phases, &increments, 4);

        // Verify phases advanced correctly
        assert!((phases[0] - 0.1).abs() < 1e-6);
        assert!((phases[1] - 1.2).abs() < 1e-6);
        assert!((phases[2] - 2.3).abs() < 1e-6);
        assert!((phases[3] - 3.4).abs() < 1e-6);
    }

    #[test]
    fn test_simd_context_interpolate_linear() {
        let ctx = SimdContext::new();
        let wavetable = [0.0, 1.0, 0.0, -1.0];
        let positions = [0.0, 0.25, 0.5, 0.75];
        let mut output = [0.0; 4];

        ctx.interpolate_linear(&wavetable, &positions, &mut output);

        // Verify interpolation worked (exact values depend on implementation)
        // Just check output was modified
        assert!(output.iter().any(|&x| x != 0.0));
    }

    #[test]
    fn test_simd_context_default() {
        let ctx = SimdContext::default();
        let name = ctx.backend_name();

        assert!(!name.is_empty());
    }
}
