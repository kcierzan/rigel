/// SIMD Backend Contract
///
/// This file defines the contract that all SIMD backend implementations must satisfy.
/// It is a specification document, not executable code.
///
/// Location: projects/rigel-synth/crates/math/src/simd/backend.rs

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
/// ```rust
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
    /// ```rust
    /// let mut phases = [0.0f32; 64];
    /// let increments = [0.1f32; 64]; // Phase increment per sample
    /// backend.advance_phase_vectorized(&mut phases, &increments, 64);
    /// // phases[i] = (phases[i] + increments[i]) % TAU
    /// ```
    fn advance_phase_vectorized(
        phases: &mut [f32],
        phase_increments: &[f32],
        count: usize,
    );

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
    /// ```rust
    /// let wavetable = [/* 2048 samples */];
    /// let positions = [0.0, 0.25, 0.5, 0.75]; // Read at 0%, 25%, 50%, 75%
    /// let mut output = [0.0f32; 4];
    /// backend.interpolate_wavetable(&wavetable, &positions, &mut output);
    /// ```
    fn interpolate_wavetable(
        wavetable: &[f32],
        positions: &[f32],
        output: &mut [f32],
    );

    /// Backend identifier for debugging and logging
    ///
    /// # Returns
    /// Static string identifying the backend: "scalar", "avx2", "avx512", or "neon"
    fn name() -> &'static str;
}

/// Backend Selection Priority
///
/// When runtime dispatch is enabled, backends are selected in this priority order:
///
/// 1. **AVX-512** (if available and compiled with `avx512` feature)
///    - Requires: x86_64 CPU with AVX-512F, AVX-512BW, AVX-512DQ, AVX-512VL
///    - Performance: Highest (16 f32s per SIMD operation)
///    - Status: Experimental (local testing only)
///
/// 2. **AVX2** (if available and compiled with `avx2` feature)
///    - Requires: x86_64 CPU with AVX2 support
///    - Performance: High (8 f32s per SIMD operation)
///    - Status: Production-ready (CI tested)
///
/// 3. **NEON** (if aarch64 platform)
///    - Requires: aarch64 CPU (always available on Apple Silicon)
///    - Performance: High (4 f32s per SIMD operation)
///    - Status: Production-ready (compile-time selection)
///
/// 4. **Scalar** (always available)
///    - Requires: Any platform
///    - Performance: Baseline
///    - Status: Fallback for CPUs without SIMD

/// Property-Based Testing Requirements
///
/// All backends must pass these property-based tests to ensure correctness:
///
/// 1. **Cross-Backend Equivalence**:
///    ```
///    For all inputs (random f32 arrays):
///        scalar_output = ScalarBackend::process_block(input, params)
///        avx2_output = Avx2Backend::process_block(input, params)
///        assert_approx_eq!(scalar_output, avx2_output, epsilon=1e-6)
///    ```
///
/// 2. **Edge Case Handling**:
///    - NaN inputs: Must produce consistent results (propagate or replace with 0.0)
///    - Infinity inputs: Must handle gracefully
///    - Zero/very small values: No denormal performance penalties
///    - Out-of-range positions (wavetable): Must wrap correctly
///
/// 3. **Phase Wrapping**:
///    ```
///    For all phase values and increments:
///        advance_phase_vectorized(phases, increments)
///        assert!(all phases in [0.0, TAU))
///    ```
///
/// 4. **Interpolation Accuracy**:
///    ```
///    For known wavetable + positions:
///        output = interpolate_wavetable(wavetable, positions)
///        expected = linear_interp_reference(wavetable, positions)
///        assert_approx_eq!(output, expected, epsilon=1e-6)
///    ```

/// Performance Validation Requirements
///
/// Benchmarks must validate these performance characteristics:
///
/// 1. **Dispatch Overhead**: <1% compared to direct backend call
///    ```
///    overhead = (dispatch_time - direct_time) / direct_time
///    assert!(overhead < 0.01) // <1%
///    ```
///
/// 2. **SIMD Speedup**:
///    - AVX2: 2-4x faster than scalar (measured via Criterion)
///    - AVX-512: 4-8x faster than scalar
///    - NEON: 2-4x faster than scalar
///
/// 3. **CPU Usage** (real-time constraint):
///    - Single voice: ~0.1% at 44.1kHz
///    - Full polyphonic: <1% CPU usage
