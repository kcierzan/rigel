//! AVX-512 Backend Implementation (Experimental)
//!
//! This module provides the AVX-512 SIMD backend for x86_64 processors with AVX-512 support.
//! AVX-512 uses 512-bit registers to process 16 f32 values simultaneously, providing
//! approximately 4-8x speedup over scalar code.
//!
//! # Requirements
//! - x86_64 architecture
//! - AVX-512F, AVX-512BW, AVX-512DQ, AVX-512VL support (Intel Skylake-X 2017+)
//! - Compiled with `avx512` feature flag
//!
//! # Status
//! **Experimental** - AVX-512 support in Rust is still maturing, and not all server
//! CPUs have full AVX-512 support. Use for local testing only, not in CI.
//!
//! # Implementation Strategy
//!
//! This backend uses the generic helpers in `super::helpers` to bridge slice-based
//! operations to the optimized SIMD kernels in the `math` module. The kernels are
//! implemented using `Avx512Vector` which provides 16-lane SIMD processing with
//! optimized algorithms (Padé approximations, polynomial methods, etc.).
//!
//! # Safety
//! All AVX-512 intrinsics are marked unsafe, but we ensure safety by:
//! - Only calling intrinsics when AVX-512 feature is enabled
//! - Properly handling alignment (unaligned loads/stores)
//! - Processing remainder samples with scalar fallback

#![cfg(all(feature = "avx512", any(target_arch = "x86", target_arch = "x86_64")))]
#![allow(unused)]

use super::backend::{ProcessParams, SimdBackend};
use super::helpers::{process_binary, process_ternary, process_unary};
use crate::simd;
use crate::Avx512Vector;
use core::arch::x86_64::*;
use core::f32::consts::TAU;

/// AVX-512 Backend (512-bit SIMD, Experimental)
///
/// Zero-sized type that implements SIMD operations using AVX-512 intrinsics.
/// Processes 16 f32 values per iteration with scalar fallback for remainder.
///
/// # Performance
/// - Throughput: ~4-8x faster than scalar
/// - Processes 16 samples per iteration (512-bit / 32-bit)
/// - Requires AVX-512F+BW+DQ+VL CPU support
/// - Experimental status: Use for local testing only
#[derive(Copy, Clone, Debug)]
pub struct Avx512Backend;

impl SimdBackend for Avx512Backend {
    #[inline]
    fn process_block(input: &[f32], output: &mut [f32], params: &ProcessParams) {
        let len = input.len();
        let mut i = 0;

        // Process 16 f32s at a time using AVX-512
        let gain_vec = unsafe { _mm512_set1_ps(params.gain) };

        while i + 16 <= len {
            unsafe {
                // Load 16 f32s from input (unaligned)
                let input_vec = _mm512_loadu_ps(input.as_ptr().add(i));

                // Multiply by gain
                let result = _mm512_mul_ps(input_vec, gain_vec);

                // Store 16 f32s to output (unaligned)
                _mm512_storeu_ps(output.as_mut_ptr().add(i), result);
            }
            i += 16;
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
        let tau_vec = unsafe { _mm512_set1_ps(TAU) };

        // Process 16 phases at a time
        while i + 16 <= count {
            unsafe {
                // Load 16 phases and 16 increments
                let phase_vec = _mm512_loadu_ps(phases.as_ptr().add(i));
                let increment_vec = _mm512_loadu_ps(phase_increments.as_ptr().add(i));

                // Add increment to phase
                let mut new_phase = _mm512_add_ps(phase_vec, increment_vec);

                // Wrap phase to [0, TAU) range
                // while (phase >= TAU) phase -= TAU
                let mask = _mm512_cmp_ps_mask(new_phase, tau_vec, _CMP_GE_OQ);
                let wrapped = _mm512_sub_ps(new_phase, tau_vec);
                new_phase = _mm512_mask_blend_ps(mask, new_phase, wrapped);

                // Handle negative phases (unlikely in normal use)
                let zero_vec = _mm512_setzero_ps();
                let neg_mask = _mm512_cmp_ps_mask(new_phase, zero_vec, _CMP_LT_OQ);
                let neg_wrapped = _mm512_add_ps(new_phase, tau_vec);
                new_phase = _mm512_mask_blend_ps(neg_mask, new_phase, neg_wrapped);

                // Store result
                _mm512_storeu_ps(phases.as_mut_ptr().add(i), new_phase);
            }
            i += 16;
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

        let table_len_vec = unsafe { _mm512_set1_ps(table_len_f32) };

        // Process 16 positions at a time
        while i + 16 <= positions.len() {
            unsafe {
                // Load 16 positions
                let pos_vec = _mm512_loadu_ps(positions.as_ptr().add(i));

                // Wrap position to [0.0, 1.0) range
                // normalized_pos = pos - floor(pos)
                // Use roundscale with _MM_FROUND_TO_NEG_INF mode (0x01 | 0x08 = floor)
                let floor_vec = _mm512_roundscale_ps(pos_vec, 0x01 | 0x08);
                let normalized_pos = _mm512_sub_ps(pos_vec, floor_vec);

                // Convert to wavetable index: index_f = normalized_pos * table_len
                let index_f_vec = _mm512_mul_ps(normalized_pos, table_len_vec);

                // Get integer index and fractional part
                let index0_vec = _mm512_roundscale_ps(index_f_vec, 0x01 | 0x08);
                let frac_vec = _mm512_sub_ps(index_f_vec, index0_vec);

                // Convert to integer indices (we need to extract to do table lookups)
                let mut indices: [f32; 16] = [0.0; 16];
                let mut fracs: [f32; 16] = [0.0; 16];
                _mm512_storeu_ps(indices.as_mut_ptr(), index0_vec);
                _mm512_storeu_ps(fracs.as_mut_ptr(), frac_vec);

                // Perform table lookups and interpolation
                // AVX-512 has gather instructions, but they're complex for this pattern
                let mut results: [f32; 16] = [0.0; 16];
                for j in 0..16 {
                    let index0 = indices[j] as usize % table_len;
                    let index1 = (index0 + 1) % table_len;
                    let frac = fracs[j];

                    let sample0 = wavetable[index0];
                    let sample1 = wavetable[index1];

                    results[j] = sample0 + frac * (sample1 - sample0);
                }

                // Store results
                let result_vec = _mm512_loadu_ps(results.as_ptr());
                _mm512_storeu_ps(output.as_mut_ptr().add(i), result_vec);
            }
            i += 16;
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
        "avx512"
    }

    // ========================================================================
    // Arithmetic Operations (Binary)
    // ========================================================================

    #[inline]
    fn add(a: &[f32], b: &[f32], output: &mut [f32]) {
        let len = output.len();
        let mut i = 0;

        // Process 16 f32s at a time using AVX-512
        while i + 16 <= len {
            unsafe {
                let a_vec = _mm512_loadu_ps(a.as_ptr().add(i));
                let b_vec = _mm512_loadu_ps(b.as_ptr().add(i));
                let result = _mm512_add_ps(a_vec, b_vec);
                _mm512_storeu_ps(output.as_mut_ptr().add(i), result);
            }
            i += 16;
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

        // Process 16 f32s at a time using AVX-512
        while i + 16 <= len {
            unsafe {
                let a_vec = _mm512_loadu_ps(a.as_ptr().add(i));
                let b_vec = _mm512_loadu_ps(b.as_ptr().add(i));
                let result = _mm512_sub_ps(a_vec, b_vec);
                _mm512_storeu_ps(output.as_mut_ptr().add(i), result);
            }
            i += 16;
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

        // Process 16 f32s at a time using AVX-512
        while i + 16 <= len {
            unsafe {
                let a_vec = _mm512_loadu_ps(a.as_ptr().add(i));
                let b_vec = _mm512_loadu_ps(b.as_ptr().add(i));
                let result = _mm512_mul_ps(a_vec, b_vec);
                _mm512_storeu_ps(output.as_mut_ptr().add(i), result);
            }
            i += 16;
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

        // Process 16 f32s at a time using AVX-512
        while i + 16 <= len {
            unsafe {
                let a_vec = _mm512_loadu_ps(a.as_ptr().add(i));
                let b_vec = _mm512_loadu_ps(b.as_ptr().add(i));
                let result = _mm512_div_ps(a_vec, b_vec);
                _mm512_storeu_ps(output.as_mut_ptr().add(i), result);
            }
            i += 16;
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

        // Process 16 f32s at a time using AVX-512
        while i + 16 <= len {
            unsafe {
                let a_vec = _mm512_loadu_ps(a.as_ptr().add(i));
                let b_vec = _mm512_loadu_ps(b.as_ptr().add(i));
                let c_vec = _mm512_loadu_ps(c.as_ptr().add(i));
                let result = _mm512_fmadd_ps(a_vec, b_vec, c_vec);
                _mm512_storeu_ps(output.as_mut_ptr().add(i), result);
            }
            i += 16;
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

        // Process 16 f32s at a time using AVX-512
        let zero_vec = unsafe { _mm512_setzero_ps() };

        while i + 16 <= len {
            unsafe {
                let input_vec = _mm512_loadu_ps(input.as_ptr().add(i));
                let result = _mm512_sub_ps(zero_vec, input_vec);
                _mm512_storeu_ps(output.as_mut_ptr().add(i), result);
            }
            i += 16;
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

        // Process 16 f32s at a time using AVX-512
        // Create a mask with sign bit cleared (0x7FFFFFFF for each f32)
        let abs_mask = unsafe { _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFF_FFFF)) };

        while i + 16 <= len {
            unsafe {
                let input_vec = _mm512_loadu_ps(input.as_ptr().add(i));
                let result = _mm512_and_ps(input_vec, abs_mask);
                _mm512_storeu_ps(output.as_mut_ptr().add(i), result);
            }
            i += 16;
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

        // Process 16 f32s at a time using AVX-512
        while i + 16 <= len {
            unsafe {
                let a_vec = _mm512_loadu_ps(a.as_ptr().add(i));
                let b_vec = _mm512_loadu_ps(b.as_ptr().add(i));
                let result = _mm512_min_ps(a_vec, b_vec);
                _mm512_storeu_ps(output.as_mut_ptr().add(i), result);
            }
            i += 16;
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

        // Process 16 f32s at a time using AVX-512
        while i + 16 <= len {
            unsafe {
                let a_vec = _mm512_loadu_ps(a.as_ptr().add(i));
                let b_vec = _mm512_loadu_ps(b.as_ptr().add(i));
                let result = _mm512_max_ps(a_vec, b_vec);
                _mm512_storeu_ps(output.as_mut_ptr().add(i), result);
            }
            i += 16;
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

        // Process 16 f32s at a time using AVX-512
        while i + 16 <= len {
            unsafe {
                let input_vec = _mm512_loadu_ps(input.as_ptr().add(i));
                let result = _mm512_sqrt_ps(input_vec);
                _mm512_storeu_ps(output.as_mut_ptr().add(i), result);
            }
            i += 16;
        }

        // Scalar fallback for remainder
        while i < len {
            output[i] = libm::sqrtf(input[i]);
            i += 1;
        }
    }

    #[inline]
    fn exp(input: &[f32], output: &mut [f32]) {
        process_unary::<Avx512Vector, _>(input, output, simd::exp::exp, libm::expf);
    }

    #[inline]
    fn log(input: &[f32], output: &mut [f32]) {
        process_unary::<Avx512Vector, _>(input, output, simd::log::log, libm::logf);
    }

    #[inline]
    fn log2(input: &[f32], output: &mut [f32]) {
        process_unary::<Avx512Vector, _>(input, output, simd::log::log2, libm::log2f);
    }

    #[inline]
    fn log10(input: &[f32], output: &mut [f32]) {
        process_unary::<Avx512Vector, _>(input, output, simd::log::log10, libm::log10f);
    }

    #[inline]
    fn pow(base: &[f32], exponent: &[f32], output: &mut [f32]) {
        // simd::pow takes scalar exponent, not vector exponent
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
        process_unary::<Avx512Vector, _>(input, output, simd::trig::sin, libm::sinf);
    }

    #[inline]
    fn cos(input: &[f32], output: &mut [f32]) {
        process_unary::<Avx512Vector, _>(input, output, simd::trig::cos, libm::cosf);
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
        process_unary::<Avx512Vector, _>(input, output, simd::atan::atan, libm::atanf);
    }

    #[inline]
    fn atan2(y: &[f32], x: &[f32], output: &mut [f32]) {
        process_binary::<Avx512Vector, _>(y, x, output, simd::atan::atan2, libm::atan2f);
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
        process_unary::<Avx512Vector, _>(input, output, simd::tanh::tanh, libm::tanhf);
    }

    // ========================================================================
    // Rounding Functions
    // ========================================================================

    #[inline]
    fn floor(input: &[f32], output: &mut [f32]) {
        let len = output.len();
        let mut i = 0;

        // Process 16 f32s at a time using AVX-512
        // _MM_FROUND_TO_NEG_INF = 0x01, _MM_FROUND_NO_EXC = 0x08
        const ROUNDING_MODE: i32 = 0x01 | 0x08;

        while i + 16 <= len {
            unsafe {
                let input_vec = _mm512_loadu_ps(input.as_ptr().add(i));
                let result = _mm512_roundscale_ps(input_vec, ROUNDING_MODE);
                _mm512_storeu_ps(output.as_mut_ptr().add(i), result);
            }
            i += 16;
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

        // Process 16 f32s at a time using AVX-512
        // _MM_FROUND_TO_POS_INF = 0x02, _MM_FROUND_NO_EXC = 0x08
        const ROUNDING_MODE: i32 = 0x02 | 0x08;

        while i + 16 <= len {
            unsafe {
                let input_vec = _mm512_loadu_ps(input.as_ptr().add(i));
                let result = _mm512_roundscale_ps(input_vec, ROUNDING_MODE);
                _mm512_storeu_ps(output.as_mut_ptr().add(i), result);
            }
            i += 16;
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

        // Process 16 f32s at a time using AVX-512
        // _MM_FROUND_TO_NEAREST_INT = 0x00, _MM_FROUND_NO_EXC = 0x08
        const ROUNDING_MODE: i32 = 0x08;

        while i + 16 <= len {
            unsafe {
                let input_vec = _mm512_loadu_ps(input.as_ptr().add(i));
                let result = _mm512_roundscale_ps(input_vec, ROUNDING_MODE);
                _mm512_storeu_ps(output.as_mut_ptr().add(i), result);
            }
            i += 16;
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

        // Process 16 f32s at a time using AVX-512
        // _MM_FROUND_TO_ZERO = 0x03, _MM_FROUND_NO_EXC = 0x08
        const ROUNDING_MODE: i32 = 0x03 | 0x08;

        while i + 16 <= len {
            unsafe {
                let input_vec = _mm512_loadu_ps(input.as_ptr().add(i));
                let result = _mm512_roundscale_ps(input_vec, ROUNDING_MODE);
                _mm512_storeu_ps(output.as_mut_ptr().add(i), result);
            }
            i += 16;
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
    fn test_avx512_process_block() {
        let input = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let mut output = [0.0; 16];
        let params = ProcessParams {
            gain: 0.5,
            frequency: 440.0,
            sample_rate: 44100.0,
        };

        Avx512Backend::process_block(&input, &mut output, &params);

        assert_eq!(
            output,
            [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
        );
    }

    #[test]
    fn test_avx512_process_block_non_multiple_of_16() {
        // Test with length that's not a multiple of 16
        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut output = [0.0; 5];
        let params = ProcessParams {
            gain: 0.5,
            frequency: 440.0,
            sample_rate: 44100.0,
        };

        Avx512Backend::process_block(&input, &mut output, &params);

        assert_eq!(output, [0.5, 1.0, 1.5, 2.0, 2.5]);
    }

    #[test]
    fn test_avx512_backend_name() {
        assert_eq!(Avx512Backend::name(), "avx512");
    }
}
