//! AVX2 Backend Implementation
//!
//! This module provides the AVX2 SIMD backend for x86_64 processors with AVX2 support.
//! AVX2 uses 256-bit registers to process 8 f32 values simultaneously, providing
//! approximately 2-4x speedup over scalar code.
//!
//! # Requirements
//! - x86_64 architecture
//! - AVX2 CPU support (Intel Haswell 2013+, AMD Excavator 2015+)
//! - Compiled with `avx2` feature flag
//!
//! # Implementation Strategy
//!
//! This backend uses the generic helpers in `super::helpers` to bridge slice-based
//! operations to the optimized SIMD kernels in the `math` module. The kernels are
//! implemented using `Avx2Vector` which provides 8-lane SIMD processing with
//! optimized algorithms (Padé approximations, polynomial methods, etc.).
//!
//! # Safety
//! All AVX2 intrinsics are marked unsafe, but we ensure safety by:
//! - Only calling intrinsics when AVX2 feature is enabled
//! - Properly handling alignment (unaligned loads/stores)
//! - Processing remainder samples with scalar fallback

#![cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
#![allow(unused)]

use super::backend::{ProcessParams, SimdBackend};
use super::helpers::{process_binary, process_ternary, process_unary};
use crate::math;
use crate::Avx2Vector;
use core::arch::x86_64::*;
use core::f32::consts::TAU;

/// AVX2 Backend (256-bit SIMD)
///
/// Zero-sized type that implements SIMD operations using AVX2 intrinsics.
/// Processes 8 f32 values per iteration with scalar fallback for remainder.
///
/// # Performance
/// - Throughput: ~2-4x faster than scalar
/// - Processes 8 samples per iteration (256-bit / 32-bit)
/// - Requires AVX2 CPU support
#[derive(Copy, Clone, Debug)]
pub struct Avx2Backend;

impl SimdBackend for Avx2Backend {
    #[inline]
    fn process_block(input: &[f32], output: &mut [f32], params: &ProcessParams) {
        let len = input.len();
        let mut i = 0;

        // Process 8 f32s at a time using AVX2
        let gain_vec = unsafe { _mm256_set1_ps(params.gain) };

        while i + 8 <= len {
            unsafe {
                // Load 8 f32s from input (unaligned)
                let input_vec = _mm256_loadu_ps(input.as_ptr().add(i));

                // Multiply by gain
                let result = _mm256_mul_ps(input_vec, gain_vec);

                // Store 8 f32s to output (unaligned)
                _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
            }
            i += 8;
        }

        // Scalar fallback for remainder samples
        while i < len {
            output[i] = input[i] * params.gain;
            i += 1;
        }
    }

    #[inline]
    fn advance_phase_vectorized(phases: &mut [f32], phase_increments: &[f32], count: usize) {
        let mut i = 0;
        let tau_vec = unsafe { _mm256_set1_ps(TAU) };

        // Process 8 phases at a time
        while i + 8 <= count {
            unsafe {
                // Load 8 phases and 8 increments
                let phase_vec = _mm256_loadu_ps(phases.as_ptr().add(i));
                let increment_vec = _mm256_loadu_ps(phase_increments.as_ptr().add(i));

                // Add increment to phase
                let mut new_phase = _mm256_add_ps(phase_vec, increment_vec);

                // Wrap phase to [0, TAU) range
                // while (phase >= TAU) phase -= TAU
                let mask = _mm256_cmp_ps(new_phase, tau_vec, _CMP_GE_OQ);
                let wrapped = _mm256_sub_ps(new_phase, tau_vec);
                new_phase = _mm256_blendv_ps(new_phase, wrapped, mask);

                // Handle negative phases (unlikely in normal use)
                let zero_vec = _mm256_setzero_ps();
                let neg_mask = _mm256_cmp_ps(new_phase, zero_vec, _CMP_LT_OQ);
                let neg_wrapped = _mm256_add_ps(new_phase, tau_vec);
                new_phase = _mm256_blendv_ps(new_phase, neg_wrapped, neg_mask);

                // Store result
                _mm256_storeu_ps(phases.as_mut_ptr().add(i), new_phase);
            }
            i += 8;
        }

        // Scalar fallback for remainder
        while i < count {
            phases[i] += phase_increments[i];
            while phases[i] >= TAU {
                phases[i] -= TAU;
            }
            while phases[i] < 0.0 {
                phases[i] += TAU;
            }
            i += 1;
        }
    }

    #[inline]
    fn interpolate_wavetable(wavetable: &[f32], positions: &[f32], output: &mut [f32]) {
        let table_len = wavetable.len();
        let table_len_f32 = table_len as f32;
        let mut i = 0;

        let table_len_vec = unsafe { _mm256_set1_ps(table_len_f32) };
        let one_vec = unsafe { _mm256_set1_ps(1.0) };

        // Process 8 positions at a time
        while i + 8 <= positions.len() {
            unsafe {
                // Load 8 positions
                let pos_vec = _mm256_loadu_ps(positions.as_ptr().add(i));

                // Wrap position to [0.0, 1.0) range
                // normalized_pos = pos - floor(pos)
                let floor_vec = _mm256_floor_ps(pos_vec);
                let normalized_pos = _mm256_sub_ps(pos_vec, floor_vec);

                // Convert to wavetable index: index_f = normalized_pos * table_len
                let index_f_vec = _mm256_mul_ps(normalized_pos, table_len_vec);

                // Get integer index and fractional part
                let index0_vec = _mm256_floor_ps(index_f_vec);
                let frac_vec = _mm256_sub_ps(index_f_vec, index0_vec);

                // Convert to integer indices (we need to extract to do table lookups)
                let mut indices: [f32; 8] = [0.0; 8];
                let mut fracs: [f32; 8] = [0.0; 8];
                _mm256_storeu_ps(indices.as_mut_ptr(), index0_vec);
                _mm256_storeu_ps(fracs.as_mut_ptr(), frac_vec);

                // Perform table lookups and interpolation (scalar, unfortunately)
                // AVX2 doesn't have efficient gather for this pattern
                let mut results: [f32; 8] = [0.0; 8];
                for j in 0..8 {
                    let index0 = indices[j] as usize % table_len;
                    let index1 = (index0 + 1) % table_len;
                    let frac = fracs[j];

                    let sample0 = wavetable[index0];
                    let sample1 = wavetable[index1];

                    results[j] = sample0 + frac * (sample1 - sample0);
                }

                // Store results
                let result_vec = _mm256_loadu_ps(results.as_ptr());
                _mm256_storeu_ps(output.as_mut_ptr().add(i), result_vec);
            }
            i += 8;
        }

        // Scalar fallback for remainder
        while i < positions.len() {
            let mut normalized_pos = positions[i];
            normalized_pos = normalized_pos - libm::floorf(normalized_pos);

            let index_f = normalized_pos * table_len_f32;
            let index0 = index_f as usize % table_len;
            let index1 = (index0 + 1) % table_len;
            let frac = index_f - index0 as f32;

            let sample0 = wavetable[index0];
            let sample1 = wavetable[index1];

            output[i] = sample0 + frac * (sample1 - sample0);
            i += 1;
        }
    }

    #[inline]
    fn name() -> &'static str {
        "avx2"
    }

    // ========================================================================
    // Arithmetic Operations (Binary)
    // ========================================================================

    #[inline]
    fn add(a: &[f32], b: &[f32], output: &mut [f32]) {
        let len = output.len();
        let mut i = 0;

        // Process 8 f32s at a time using AVX2
        while i + 8 <= len {
            unsafe {
                let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
                let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
                let result = _mm256_add_ps(a_vec, b_vec);
                _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
            }
            i += 8;
        }

        // Scalar fallback for remainder
        while i < len {
            output[i] = a[i] + b[i];
            i += 1;
        }
    }

    #[inline]
    fn sub(a: &[f32], b: &[f32], output: &mut [f32]) {
        let len = output.len();
        let mut i = 0;

        // Process 8 f32s at a time using AVX2
        while i + 8 <= len {
            unsafe {
                let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
                let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
                let result = _mm256_sub_ps(a_vec, b_vec);
                _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
            }
            i += 8;
        }

        // Scalar fallback for remainder
        while i < len {
            output[i] = a[i] - b[i];
            i += 1;
        }
    }

    #[inline]
    fn mul(a: &[f32], b: &[f32], output: &mut [f32]) {
        let len = output.len();
        let mut i = 0;

        // Process 8 f32s at a time using AVX2
        while i + 8 <= len {
            unsafe {
                let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
                let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
                let result = _mm256_mul_ps(a_vec, b_vec);
                _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
            }
            i += 8;
        }

        // Scalar fallback for remainder
        while i < len {
            output[i] = a[i] * b[i];
            i += 1;
        }
    }

    #[inline]
    fn div(a: &[f32], b: &[f32], output: &mut [f32]) {
        let len = output.len();
        let mut i = 0;

        // Process 8 f32s at a time using AVX2
        while i + 8 <= len {
            unsafe {
                let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
                let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
                let result = _mm256_div_ps(a_vec, b_vec);
                _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
            }
            i += 8;
        }

        // Scalar fallback for remainder
        while i < len {
            output[i] = a[i] / b[i];
            i += 1;
        }
    }

    #[inline]
    fn fma(a: &[f32], b: &[f32], c: &[f32], output: &mut [f32]) {
        let len = output.len();
        let mut i = 0;

        // Process 8 f32s at a time using AVX2 FMA
        while i + 8 <= len {
            unsafe {
                let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
                let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
                let c_vec = _mm256_loadu_ps(c.as_ptr().add(i));
                let result = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
            }
            i += 8;
        }

        // Scalar fallback for remainder
        while i < len {
            output[i] = a[i] * b[i] + c[i];
            i += 1;
        }
    }

    // ========================================================================
    // Arithmetic Operations (Unary)
    // ========================================================================

    #[inline]
    fn neg(input: &[f32], output: &mut [f32]) {
        let len = output.len();
        let mut i = 0;

        // Process 8 f32s at a time using AVX2
        // Negation via XOR with sign bit (0x80000000)
        let sign_bit = unsafe { _mm256_set1_ps(-0.0) };

        while i + 8 <= len {
            unsafe {
                let input_vec = _mm256_loadu_ps(input.as_ptr().add(i));
                let result = _mm256_xor_ps(input_vec, sign_bit);
                _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
            }
            i += 8;
        }

        // Scalar fallback for remainder
        while i < len {
            output[i] = -input[i];
            i += 1;
        }
    }

    #[inline]
    fn abs(input: &[f32], output: &mut [f32]) {
        let len = output.len();
        let mut i = 0;

        // Process 8 f32s at a time using AVX2
        // Clear sign bit via AND with 0x7FFFFFFF
        let abs_mask = unsafe { _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)) };

        while i + 8 <= len {
            unsafe {
                let input_vec = _mm256_loadu_ps(input.as_ptr().add(i));
                let result = _mm256_and_ps(input_vec, abs_mask);
                _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
            }
            i += 8;
        }

        // Scalar fallback for remainder
        while i < len {
            output[i] = input[i].abs();
            i += 1;
        }
    }

    // ========================================================================
    // Comparison Operations
    // ========================================================================

    #[inline]
    fn min(a: &[f32], b: &[f32], output: &mut [f32]) {
        let len = output.len();
        let mut i = 0;

        // Process 8 f32s at a time using AVX2
        while i + 8 <= len {
            unsafe {
                let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
                let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
                let result = _mm256_min_ps(a_vec, b_vec);
                _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
            }
            i += 8;
        }

        // Scalar fallback for remainder
        while i < len {
            output[i] = if a[i] < b[i] { a[i] } else { b[i] };
            i += 1;
        }
    }

    #[inline]
    fn max(a: &[f32], b: &[f32], output: &mut [f32]) {
        let len = output.len();
        let mut i = 0;

        // Process 8 f32s at a time using AVX2
        while i + 8 <= len {
            unsafe {
                let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
                let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
                let result = _mm256_max_ps(a_vec, b_vec);
                _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
            }
            i += 8;
        }

        // Scalar fallback for remainder
        while i < len {
            output[i] = if a[i] > b[i] { a[i] } else { b[i] };
            i += 1;
        }
    }

    // ========================================================================
    // Basic Math Functions
    // ========================================================================

    #[inline]
    fn sqrt(input: &[f32], output: &mut [f32]) {
        let len = output.len();
        let mut i = 0;

        // Process 8 f32s at a time using AVX2
        while i + 8 <= len {
            unsafe {
                let input_vec = _mm256_loadu_ps(input.as_ptr().add(i));
                let result = _mm256_sqrt_ps(input_vec);
                _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
            }
            i += 8;
        }

        // Scalar fallback for remainder
        while i < len {
            output[i] = libm::sqrtf(input[i]);
            i += 1;
        }
    }

    #[inline]
    fn exp(input: &[f32], output: &mut [f32]) {
        process_unary::<Avx2Vector, _>(input, output, math::exp::exp, libm::expf);
    }

    #[inline]
    fn log(input: &[f32], output: &mut [f32]) {
        process_unary::<Avx2Vector, _>(input, output, math::log::log, libm::logf);
    }

    #[inline]
    fn log2(input: &[f32], output: &mut [f32]) {
        process_unary::<Avx2Vector, _>(input, output, math::log::log2, libm::log2f);
    }

    #[inline]
    fn log10(input: &[f32], output: &mut [f32]) {
        process_unary::<Avx2Vector, _>(input, output, math::log::log10, libm::log10f);
    }

    #[inline]
    fn pow(base: &[f32], exponent: &[f32], output: &mut [f32]) {
        // math::pow takes scalar exponent, not vector exponent
        // So we can't use the helper here - just use libm::powf
        for i in 0..output.len() {
            output[i] = libm::powf(base[i], exponent[i]);
        }
    }

    // ========================================================================
    // Trigonometric Functions
    // ========================================================================

    #[inline]
    fn sin(input: &[f32], output: &mut [f32]) {
        process_unary::<Avx2Vector, _>(input, output, math::trig::sin, libm::sinf);
    }

    #[inline]
    fn cos(input: &[f32], output: &mut [f32]) {
        process_unary::<Avx2Vector, _>(input, output, math::trig::cos, libm::cosf);
    }

    #[inline]
    fn tan(input: &[f32], output: &mut [f32]) {
        // No dedicated tan kernel in math module, use libm
        for i in 0..output.len() {
            output[i] = libm::tanf(input[i]);
        }
    }

    #[inline]
    fn asin(input: &[f32], output: &mut [f32]) {
        // No vectorized asin in math module, use libm
        for i in 0..output.len() {
            output[i] = libm::asinf(input[i]);
        }
    }

    #[inline]
    fn acos(input: &[f32], output: &mut [f32]) {
        // No vectorized acos in math module, use libm
        for i in 0..output.len() {
            output[i] = libm::acosf(input[i]);
        }
    }

    #[inline]
    fn atan(input: &[f32], output: &mut [f32]) {
        process_unary::<Avx2Vector, _>(input, output, math::atan::atan, libm::atanf);
    }

    #[inline]
    fn atan2(y: &[f32], x: &[f32], output: &mut [f32]) {
        process_binary::<Avx2Vector, _>(y, x, output, math::atan::atan2, libm::atan2f);
    }

    // ========================================================================
    // Hyperbolic Functions
    // ========================================================================

    #[inline]
    fn sinh(input: &[f32], output: &mut [f32]) {
        // No vectorized sinh in math module, use libm
        for i in 0..output.len() {
            output[i] = libm::sinhf(input[i]);
        }
    }

    #[inline]
    fn cosh(input: &[f32], output: &mut [f32]) {
        // No vectorized cosh in math module, use libm
        for i in 0..output.len() {
            output[i] = libm::coshf(input[i]);
        }
    }

    #[inline]
    fn tanh(input: &[f32], output: &mut [f32]) {
        process_unary::<Avx2Vector, _>(input, output, math::tanh::tanh, libm::tanhf);
    }

    // ========================================================================
    // Rounding Functions
    // ========================================================================

    #[inline]
    fn floor(input: &[f32], output: &mut [f32]) {
        let len = output.len();
        let mut i = 0;

        // Process 8 f32s at a time using AVX2
        while i + 8 <= len {
            unsafe {
                let input_vec = _mm256_loadu_ps(input.as_ptr().add(i));
                let result = _mm256_floor_ps(input_vec);
                _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
            }
            i += 8;
        }

        // Scalar fallback for remainder
        while i < len {
            output[i] = libm::floorf(input[i]);
            i += 1;
        }
    }

    #[inline]
    fn ceil(input: &[f32], output: &mut [f32]) {
        let len = output.len();
        let mut i = 0;

        // Process 8 f32s at a time using AVX2
        while i + 8 <= len {
            unsafe {
                let input_vec = _mm256_loadu_ps(input.as_ptr().add(i));
                let result = _mm256_ceil_ps(input_vec);
                _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
            }
            i += 8;
        }

        // Scalar fallback for remainder
        while i < len {
            output[i] = libm::ceilf(input[i]);
            i += 1;
        }
    }

    #[inline]
    fn round(input: &[f32], output: &mut [f32]) {
        let len = output.len();
        let mut i = 0;

        // Process 8 f32s at a time using AVX2
        // _MM_FROUND_TO_NEAREST_INT = 0x00 (round to nearest, ties to even)
        while i + 8 <= len {
            unsafe {
                let input_vec = _mm256_loadu_ps(input.as_ptr().add(i));
                let result =
                    _mm256_round_ps(input_vec, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
            }
            i += 8;
        }

        // Scalar fallback for remainder
        while i < len {
            output[i] = libm::roundf(input[i]);
            i += 1;
        }
    }

    #[inline]
    fn trunc(input: &[f32], output: &mut [f32]) {
        let len = output.len();
        let mut i = 0;

        // Process 8 f32s at a time using AVX2
        // _MM_FROUND_TO_ZERO = 0x03 (truncate towards zero)
        while i + 8 <= len {
            unsafe {
                let input_vec = _mm256_loadu_ps(input.as_ptr().add(i));
                let result = _mm256_round_ps(input_vec, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
                _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
            }
            i += 8;
        }

        // Scalar fallback for remainder
        while i < len {
            output[i] = libm::truncf(input[i]);
            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_avx2_process_block() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut output = [0.0; 8];
        let params = ProcessParams {
            gain: 0.5,
            frequency: 440.0,
            sample_rate: 44100.0,
        };

        Avx2Backend::process_block(&input, &mut output, &params);

        assert_eq!(output, [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]);
    }

    #[test]
    fn test_avx2_process_block_non_multiple_of_8() {
        // Test with length that's not a multiple of 8
        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut output = [0.0; 5];
        let params = ProcessParams {
            gain: 0.5,
            frequency: 440.0,
            sample_rate: 44100.0,
        };

        Avx2Backend::process_block(&input, &mut output, &params);

        assert_eq!(output, [0.5, 1.0, 1.5, 2.0, 2.5]);
    }

    #[test]
    fn test_avx2_advance_phase() {
        let mut phases = [0.0, 1.0, 2.0, 3.0, TAU - 0.1, TAU - 0.05, 0.5, 1.5];
        let increments = [0.1, 0.2, 0.3, 0.4, 0.2, 0.1, 0.1, 0.2];

        Avx2Backend::advance_phase_vectorized(&mut phases, &increments, 8);

        // Verify phases advanced and wrapped correctly
        assert!((phases[0] - 0.1).abs() < 1e-5);
        assert!((phases[1] - 1.2).abs() < 1e-5);
        // Phase 4 should wrap: (TAU - 0.1) + 0.2 = TAU + 0.1 -> 0.1
        assert!(phases[4] >= 0.0 && phases[4] < TAU);
        assert!((phases[4] - 0.1).abs() < 1e-5);
    }

    #[test]
    fn test_avx2_backend_name() {
        assert_eq!(Avx2Backend::name(), "avx2");
    }
}
