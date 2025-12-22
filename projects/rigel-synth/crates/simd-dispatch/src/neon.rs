//! NEON Backend Implementation
//!
//! This module provides the NEON SIMD backend for aarch64 processors (ARM64).
//! NEON uses 128-bit registers to process 4 f32 values simultaneously, providing
//! approximately 2-4x speedup over scalar code.
//!
//! # Requirements
//! - aarch64 architecture (Apple Silicon, ARM64 servers)
//! - NEON support (always available on modern ARM64 CPUs)
//! - Compiled with `neon` feature flag
//!
//! # Implementation Strategy
//!
//! This backend uses the generic helpers in `super::helpers` to bridge slice-based
//! operations to the optimized SIMD kernels in the `math` module. The kernels are
//! implemented using `NeonVector` which provides 4-lane SIMD processing with
//! optimized algorithms (Padé approximations, polynomial methods, etc.).
//!
//! # Note
//! On Apple Silicon (M1, M2, M3+), NEON is always available and this backend
//! should be used via compile-time selection (no runtime detection needed).

#![cfg(all(feature = "neon", target_arch = "aarch64"))]
#![allow(unused)]

use super::backend::{ProcessParams, SimdBackend};
use super::helpers::{process_binary, process_ternary, process_unary};
use crate::simd;
use crate::NeonVector;
use core::arch::aarch64::*;
use core::f32::consts::TAU;

/// NEON Backend (128-bit SIMD)
///
/// Zero-sized type that implements SIMD operations using NEON intrinsics.
/// Processes 4 f32 values per iteration with scalar fallback for remainder.
///
/// # Performance
/// - Throughput: ~2-4x faster than scalar
/// - Processes 4 samples per iteration (128-bit / 32-bit)
/// - NEON always available on Apple Silicon and modern ARM64
#[derive(Copy, Clone, Debug)]
pub struct NeonBackend;

impl SimdBackend for NeonBackend {
    #[inline]
    fn process_block(input: &[f32], output: &mut [f32], params: &ProcessParams) {
        let len = input.len();
        let mut i = 0;

        // Process 4 f32s at a time using NEON
        let gain_vec = unsafe { vdupq_n_f32(params.gain) };

        while i + 4 <= len {
            unsafe {
                // Load 4 f32s from input
                let input_vec = vld1q_f32(input.as_ptr().add(i));

                // Multiply by gain
                let result = vmulq_f32(input_vec, gain_vec);

                // Store 4 f32s to output
                vst1q_f32(output.as_mut_ptr().add(i), result);
            }
            i += 4;
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
        let tau_vec = unsafe { vdupq_n_f32(TAU) };
        let zero_vec = unsafe { vdupq_n_f32(0.0) };

        // Process 4 phases at a time
        while i + 4 <= count {
            unsafe {
                // Load 4 phases and 4 increments
                let phase_vec = vld1q_f32(phases.as_ptr().add(i));
                let increment_vec = vld1q_f32(phase_increments.as_ptr().add(i));

                // Add increment to phase
                let mut new_phase = vaddq_f32(phase_vec, increment_vec);

                // Wrap phase to [0, TAU) range
                // while (phase >= TAU) phase -= TAU
                let ge_mask = vcgeq_f32(new_phase, tau_vec);
                let wrapped = vsubq_f32(new_phase, tau_vec);
                new_phase = vbslq_f32(ge_mask, wrapped, new_phase);

                // Handle negative phases (unlikely in normal use)
                let lt_mask = vcltq_f32(new_phase, zero_vec);
                let neg_wrapped = vaddq_f32(new_phase, tau_vec);
                new_phase = vbslq_f32(lt_mask, neg_wrapped, new_phase);

                // Store result
                vst1q_f32(phases.as_mut_ptr().add(i), new_phase);
            }
            i += 4;
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

        let table_len_vec = unsafe { vdupq_n_f32(table_len_f32) };

        // Process 4 positions at a time
        while i + 4 <= positions.len() {
            unsafe {
                // Load 4 positions
                let pos_vec = vld1q_f32(positions.as_ptr().add(i));

                // Wrap position to [0.0, 1.0) range
                // normalized_pos = pos - floor(pos)
                // Note: NEON doesn't have vrndmq_f32 (floor) in all versions,
                // so we use a manual approach
                let mut positions_arr: [f32; 4] = [0.0; 4];
                vst1q_f32(positions_arr.as_mut_ptr(), pos_vec);

                // Normalize positions
                for j in 0..4 {
                    positions_arr[j] = positions_arr[j] - libm::floorf(positions_arr[j]);
                }

                let normalized_pos = vld1q_f32(positions_arr.as_ptr());

                // Convert to wavetable index: index_f = normalized_pos * table_len
                let index_f_vec = vmulq_f32(normalized_pos, table_len_vec);

                // Extract to array for table lookups
                let mut indices_f: [f32; 4] = [0.0; 4];
                vst1q_f32(indices_f.as_mut_ptr(), index_f_vec);

                // Perform table lookups and interpolation (scalar)
                let mut results: [f32; 4] = [0.0; 4];
                for j in 0..4 {
                    let index_f = indices_f[j];
                    let index0 = index_f as usize % table_len;
                    let index1 = (index0 + 1) % table_len;
                    let frac = index_f - index0 as f32;

                    let sample0 = wavetable[index0];
                    let sample1 = wavetable[index1];

                    results[j] = sample0 + frac * (sample1 - sample0);
                }

                // Store results
                let result_vec = vld1q_f32(results.as_ptr());
                vst1q_f32(output.as_mut_ptr().add(i), result_vec);
            }
            i += 4;
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
        "neon"
    }

    // ========================================================================
    // Arithmetic Operations (Binary)
    // ========================================================================

    #[inline]
    fn add(a: &[f32], b: &[f32], output: &mut [f32]) {
        let len = output.len();
        let mut i = 0;

        // Process 4 f32s at a time using NEON
        while i + 4 <= len {
            unsafe {
                let a_vec = vld1q_f32(a.as_ptr().add(i));
                let b_vec = vld1q_f32(b.as_ptr().add(i));
                let result = vaddq_f32(a_vec, b_vec);
                vst1q_f32(output.as_mut_ptr().add(i), result);
            }
            i += 4;
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

        // Process 4 f32s at a time using NEON
        while i + 4 <= len {
            unsafe {
                let a_vec = vld1q_f32(a.as_ptr().add(i));
                let b_vec = vld1q_f32(b.as_ptr().add(i));
                let result = vsubq_f32(a_vec, b_vec);
                vst1q_f32(output.as_mut_ptr().add(i), result);
            }
            i += 4;
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

        // Process 4 f32s at a time using NEON
        while i + 4 <= len {
            unsafe {
                let a_vec = vld1q_f32(a.as_ptr().add(i));
                let b_vec = vld1q_f32(b.as_ptr().add(i));
                let result = vmulq_f32(a_vec, b_vec);
                vst1q_f32(output.as_mut_ptr().add(i), result);
            }
            i += 4;
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

        // Process 4 f32s at a time using NEON
        while i + 4 <= len {
            unsafe {
                let a_vec = vld1q_f32(a.as_ptr().add(i));
                let b_vec = vld1q_f32(b.as_ptr().add(i));
                let result = vdivq_f32(a_vec, b_vec);
                vst1q_f32(output.as_mut_ptr().add(i), result);
            }
            i += 4;
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

        // Process 4 f32s at a time using NEON
        while i + 4 <= len {
            unsafe {
                let a_vec = vld1q_f32(a.as_ptr().add(i));
                let b_vec = vld1q_f32(b.as_ptr().add(i));
                let c_vec = vld1q_f32(c.as_ptr().add(i));
                // vfmaq_f32(c, a, b) computes a * b + c
                let result = vfmaq_f32(c_vec, a_vec, b_vec);
                vst1q_f32(output.as_mut_ptr().add(i), result);
            }
            i += 4;
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

        // Process 4 f32s at a time using NEON
        while i + 4 <= len {
            unsafe {
                let input_vec = vld1q_f32(input.as_ptr().add(i));
                let result = vnegq_f32(input_vec);
                vst1q_f32(output.as_mut_ptr().add(i), result);
            }
            i += 4;
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

        // Process 4 f32s at a time using NEON
        while i + 4 <= len {
            unsafe {
                let input_vec = vld1q_f32(input.as_ptr().add(i));
                let result = vabsq_f32(input_vec);
                vst1q_f32(output.as_mut_ptr().add(i), result);
            }
            i += 4;
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

        // Process 4 f32s at a time using NEON
        while i + 4 <= len {
            unsafe {
                let a_vec = vld1q_f32(a.as_ptr().add(i));
                let b_vec = vld1q_f32(b.as_ptr().add(i));
                let result = vminq_f32(a_vec, b_vec);
                vst1q_f32(output.as_mut_ptr().add(i), result);
            }
            i += 4;
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

        // Process 4 f32s at a time using NEON
        while i + 4 <= len {
            unsafe {
                let a_vec = vld1q_f32(a.as_ptr().add(i));
                let b_vec = vld1q_f32(b.as_ptr().add(i));
                let result = vmaxq_f32(a_vec, b_vec);
                vst1q_f32(output.as_mut_ptr().add(i), result);
            }
            i += 4;
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

        // Process 4 f32s at a time using NEON
        while i + 4 <= len {
            unsafe {
                let input_vec = vld1q_f32(input.as_ptr().add(i));
                let result = vsqrtq_f32(input_vec);
                vst1q_f32(output.as_mut_ptr().add(i), result);
            }
            i += 4;
        }

        // Scalar fallback for remainder
        while i < len {
            output[i] = libm::sqrtf(input[i]);
            i += 1;
        }
    }

    #[inline]
    fn exp(input: &[f32], output: &mut [f32]) {
        process_unary::<NeonVector, _>(input, output, simd::exp::exp, libm::expf);
    }

    #[inline]
    fn log(input: &[f32], output: &mut [f32]) {
        process_unary::<NeonVector, _>(input, output, simd::log::log, libm::logf);
    }

    #[inline]
    fn log2(input: &[f32], output: &mut [f32]) {
        process_unary::<NeonVector, _>(input, output, simd::log::log2, libm::log2f);
    }

    #[inline]
    fn log10(input: &[f32], output: &mut [f32]) {
        process_unary::<NeonVector, _>(input, output, simd::log::log10, libm::log10f);
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
        process_unary::<NeonVector, _>(input, output, simd::trig::sin, libm::sinf);
    }

    #[inline]
    fn cos(input: &[f32], output: &mut [f32]) {
        process_unary::<NeonVector, _>(input, output, simd::trig::cos, libm::cosf);
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
        process_unary::<NeonVector, _>(input, output, simd::atan::atan, libm::atanf);
    }

    #[inline]
    fn atan2(y: &[f32], x: &[f32], output: &mut [f32]) {
        process_binary::<NeonVector, _>(y, x, output, simd::atan::atan2, libm::atan2f);
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
        process_unary::<NeonVector, _>(input, output, simd::tanh::tanh, libm::tanhf);
    }

    // ========================================================================
    // Rounding Functions
    // ========================================================================

    #[inline]
    fn floor(input: &[f32], output: &mut [f32]) {
        let len = output.len();
        let mut i = 0;

        // Process 4 f32s at a time using NEON
        while i + 4 <= len {
            unsafe {
                let input_vec = vld1q_f32(input.as_ptr().add(i));
                let result = vrndmq_f32(input_vec);
                vst1q_f32(output.as_mut_ptr().add(i), result);
            }
            i += 4;
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

        // Process 4 f32s at a time using NEON
        while i + 4 <= len {
            unsafe {
                let input_vec = vld1q_f32(input.as_ptr().add(i));
                let result = vrndpq_f32(input_vec);
                vst1q_f32(output.as_mut_ptr().add(i), result);
            }
            i += 4;
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

        // Process 4 f32s at a time using NEON
        while i + 4 <= len {
            unsafe {
                let input_vec = vld1q_f32(input.as_ptr().add(i));
                let result = vrndnq_f32(input_vec);
                vst1q_f32(output.as_mut_ptr().add(i), result);
            }
            i += 4;
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

        // Process 4 f32s at a time using NEON
        // vrndq_f32 rounds towards zero (truncation)
        while i + 4 <= len {
            unsafe {
                let input_vec = vld1q_f32(input.as_ptr().add(i));
                let result = vrndq_f32(input_vec);
                vst1q_f32(output.as_mut_ptr().add(i), result);
            }
            i += 4;
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
    fn test_neon_process_block() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut output = [0.0; 4];
        let params = ProcessParams {
            gain: 0.5,
            frequency: 440.0,
            sample_rate: 44100.0,
        };

        NeonBackend::process_block(&input, &mut output, &params);

        assert_eq!(output, [0.5, 1.0, 1.5, 2.0]);
    }

    #[test]
    fn test_neon_process_block_non_multiple_of_4() {
        // Test with length that's not a multiple of 4
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut output = [0.0; 6];
        let params = ProcessParams {
            gain: 0.5,
            frequency: 440.0,
            sample_rate: 44100.0,
        };

        NeonBackend::process_block(&input, &mut output, &params);

        assert_eq!(output, [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]);
    }

    #[test]
    fn test_neon_advance_phase() {
        let mut phases = [0.0, 1.0, 2.0, 3.0];
        let increments = [0.1, 0.2, 0.3, 0.4];

        NeonBackend::advance_phase_vectorized(&mut phases, &increments, 4);

        // Verify phases advanced correctly
        assert!((phases[0] - 0.1).abs() < 1e-6);
        assert!((phases[1] - 1.2).abs() < 1e-6);
        assert!((phases[2] - 2.3).abs() < 1e-6);
        assert!((phases[3] - 3.4).abs() < 1e-6);
    }

    #[test]
    fn test_neon_backend_name() {
        assert_eq!(NeonBackend::name(), "neon");
    }
}
