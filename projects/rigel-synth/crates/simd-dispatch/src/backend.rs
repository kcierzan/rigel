//! SIMD Backend Trait
//!
//! This module defines the contract that all SIMD backend implementations must satisfy.
//! All backends (Scalar, AVX2, AVX-512, NEON) implement the `SimdBackend` trait with
//! functionally identical behavior - only performance characteristics differ.

#![allow(unused)]

use core::f32::consts::TAU;

/// Parameters for audio processing operations
#[derive(Debug, Clone, Copy)]
pub struct ProcessParams {
    /// Gain multiplier (0.0 to 1.0+)
    pub gain: f32,
    /// Frequency in Hz
    pub frequency: f32,
    /// Sample rate in Hz
    pub sample_rate: f32,
}

/// SIMD Backend Trait
///
/// All SIMD backend implementations (Scalar, AVX2, AVX-512, NEON) must implement this trait.
///
/// # Contract Requirements
///
/// 1. **Functional Equivalence**: All backends MUST produce identical output within floating-point precision
/// 2. **no_std Compatible**: No heap allocations, no std library dependencies
/// 3. **Copy Semantic**: All backends are zero-sized types (ZSTs) and Copy
/// 4. **Inline**: All methods should be marked #[inline] for optimization
/// 5. **Safety**: No undefined behavior, handle all edge cases (NaN, infinity, etc.)
///
/// # Example Usage
///
/// ```ignore
/// // Initialize backend (compile-time or runtime selection)
/// let backend = ScalarBackend;
///
/// // Process audio block
/// let input = [1.0f32; 1024];
/// let mut output = [0.0f32; 1024];
/// let params = ProcessParams { gain: 0.5, frequency: 440.0, sample_rate: 44100.0 };
///
/// backend.process_block(&input, &mut output, &params);
/// ```
pub trait SimdBackend: Copy {
    /// Process a block of audio samples
    ///
    /// # Parameters
    /// - `input`: Input audio buffer (immutable)
    /// - `output`: Output audio buffer (same length as input, mutable)
    /// - `params`: Processing parameters (gain, frequency, etc.)
    ///
    /// # Invariants
    /// - `input.len() == output.len()` (caller ensures)
    /// - No allocations allowed
    /// - Must handle NaN/infinity gracefully (propagate or clamp)
    /// - Must produce identical results across all backends (within 1e-6 tolerance)
    ///
    /// # Performance
    /// - Scalar: Baseline performance (1.0x)
    /// - AVX2: ~2-4x faster (processes 8 f32s per iteration)
    /// - AVX-512: ~4-8x faster (processes 16 f32s per iteration)
    /// - NEON: ~2-4x faster (processes 4 f32s per iteration)
    fn process_block(input: &[f32], output: &mut [f32], params: &ProcessParams);

    /// Advance oscillator phases with SIMD vectorization
    ///
    /// # Parameters
    /// - `phases`: Current phase values in radians (0.0 to TAU), mutable
    /// - `phase_increments`: Phase delta per sample
    /// - `count`: Number of phases to advance
    ///
    /// # Invariants
    /// - `phases.len() >= count`
    /// - `phase_increments.len() >= count`
    /// - Output phases wrap to [0.0, TAU) range
    /// - No allocations allowed
    ///
    /// # Example
    /// ```ignore
    /// let mut phases = [0.0f32; 64];
    /// let increments = [0.1f32; 64]; // Phase increment per sample
    /// backend.advance_phase_vectorized(&mut phases, &increments, 64);
    /// // phases[i] = (phases[i] + increments[i]) % TAU
    /// ```
    fn advance_phase_vectorized(phases: &mut [f32], phase_increments: &[f32], count: usize);

    /// Wavetable interpolation with SIMD
    ///
    /// Reads from a wavetable at specified positions using linear interpolation.
    ///
    /// # Parameters
    /// - `wavetable`: Source wavetable data (periodic waveform, typically 2048 samples)
    /// - `positions`: Normalized read positions (0.0 to 1.0)
    /// - `output`: Interpolated output samples (same length as positions)
    ///
    /// # Invariants
    /// - `positions.len() == output.len()` (caller ensures)
    /// - `positions[i]` in range [0.0, 1.0) (wraps if out of range)
    /// - Uses linear interpolation between samples
    /// - No allocations allowed
    ///
    /// # Example
    /// ```ignore
    /// let wavetable = vec![/* 2048 samples */];
    /// let positions = [0.0, 0.25, 0.5, 0.75]; // Read at 0%, 25%, 50%, 75%
    /// let mut output = [0.0f32; 4];
    /// backend.interpolate_wavetable(&wavetable, &positions, &mut output);
    /// ```
    fn interpolate_wavetable(wavetable: &[f32], positions: &[f32], output: &mut [f32]);

    /// Backend identifier for debugging and logging
    ///
    /// # Returns
    /// Static string identifying the backend: "scalar", "avx2", "avx512", or "neon"
    fn name() -> &'static str;

    // ========================================================================
    // Arithmetic Operations (Binary)
    // ========================================================================

    /// Element-wise addition: output[i] = a[i] + b[i]
    fn add(a: &[f32], b: &[f32], output: &mut [f32]);

    /// Element-wise subtraction: output[i] = a[i] - b[i]
    fn sub(a: &[f32], b: &[f32], output: &mut [f32]);

    /// Element-wise multiplication: output[i] = a[i] * b[i]
    fn mul(a: &[f32], b: &[f32], output: &mut [f32]);

    /// Element-wise division: output[i] = a[i] / b[i]
    fn div(a: &[f32], b: &[f32], output: &mut [f32]);

    /// Element-wise fused multiply-add: output[i] = a[i] * b[i] + c[i]
    fn fma(a: &[f32], b: &[f32], c: &[f32], output: &mut [f32]);

    // ========================================================================
    // Arithmetic Operations (Unary)
    // ========================================================================

    /// Element-wise negation: output[i] = -input[i]
    fn neg(input: &[f32], output: &mut [f32]);

    /// Element-wise absolute value: output[i] = |input[i]|
    fn abs(input: &[f32], output: &mut [f32]);

    // ========================================================================
    // Comparison Operations
    // ========================================================================

    /// Element-wise minimum: output[i] = min(a[i], b[i])
    fn min(a: &[f32], b: &[f32], output: &mut [f32]);

    /// Element-wise maximum: output[i] = max(a[i], b[i])
    fn max(a: &[f32], b: &[f32], output: &mut [f32]);

    // ========================================================================
    // Basic Math Functions
    // ========================================================================

    /// Element-wise square root: output[i] = sqrt(input[i])
    fn sqrt(input: &[f32], output: &mut [f32]);

    /// Element-wise exponential: output[i] = e^input[i]
    fn exp(input: &[f32], output: &mut [f32]);

    /// Element-wise natural logarithm: output[i] = ln(input[i])
    fn log(input: &[f32], output: &mut [f32]);

    /// Element-wise base-2 logarithm: output[i] = log2(input[i])
    fn log2(input: &[f32], output: &mut [f32]);

    /// Element-wise base-10 logarithm: output[i] = log10(input[i])
    fn log10(input: &[f32], output: &mut [f32]);

    /// Element-wise power: output[i] = base[i]^exponent[i]
    fn pow(base: &[f32], exponent: &[f32], output: &mut [f32]);

    // ========================================================================
    // Trigonometric Functions
    // ========================================================================

    /// Element-wise sine: output[i] = sin(input[i])
    fn sin(input: &[f32], output: &mut [f32]);

    /// Element-wise cosine: output[i] = cos(input[i])
    fn cos(input: &[f32], output: &mut [f32]);

    /// Element-wise tangent: output[i] = tan(input[i])
    fn tan(input: &[f32], output: &mut [f32]);

    /// Element-wise arcsine: output[i] = asin(input[i])
    fn asin(input: &[f32], output: &mut [f32]);

    /// Element-wise arccosine: output[i] = acos(input[i])
    fn acos(input: &[f32], output: &mut [f32]);

    /// Element-wise arctangent: output[i] = atan(input[i])
    fn atan(input: &[f32], output: &mut [f32]);

    /// Element-wise two-argument arctangent: output[i] = atan2(y[i], x[i])
    fn atan2(y: &[f32], x: &[f32], output: &mut [f32]);

    // ========================================================================
    // Hyperbolic Functions
    // ========================================================================

    /// Element-wise hyperbolic sine: output[i] = sinh(input[i])
    fn sinh(input: &[f32], output: &mut [f32]);

    /// Element-wise hyperbolic cosine: output[i] = cosh(input[i])
    fn cosh(input: &[f32], output: &mut [f32]);

    /// Element-wise hyperbolic tangent: output[i] = tanh(input[i])
    fn tanh(input: &[f32], output: &mut [f32]);

    // ========================================================================
    // Rounding Functions
    // ========================================================================

    /// Element-wise floor: output[i] = floor(input[i])
    fn floor(input: &[f32], output: &mut [f32]);

    /// Element-wise ceiling: output[i] = ceil(input[i])
    fn ceil(input: &[f32], output: &mut [f32]);

    /// Element-wise rounding: output[i] = round(input[i])
    fn round(input: &[f32], output: &mut [f32]);

    /// Element-wise truncation: output[i] = trunc(input[i])
    fn trunc(input: &[f32], output: &mut [f32]);
}
