//! Scalar Backend Implementation
//!
//! This module provides the scalar (non-SIMD) fallback backend.
//! It processes one sample at a time using standard Rust operations.
//! This backend is always available and serves as the reference implementation
//! for correctness testing of SIMD backends.
//!
//! # Implementation Strategy
//!
//! This backend uses the generic helpers in `super::helpers` to bridge slice-based
//! operations to the optimized SIMD kernels in the `math` module. Even though this
//! is the "scalar" backend, it still benefits from the optimized algorithms (e.g.,
//! Padé approximations for exp, rational polynomials for tanh) implemented in the
//! math module.

#![allow(unused)]

use super::backend::{ProcessParams, SimdBackend};
use super::helpers::{process_binary, process_ternary, process_unary};
use crate::math;
use crate::ops;
use crate::ScalarVector;
use core::f32::consts::TAU;

/// Scalar Backend (No SIMD)
///
/// Zero-sized type that implements SIMD operations using scalar code.
/// Always available as fallback when no SIMD instructions are supported.
///
/// # Performance
/// - Baseline: 1.0x (reference performance)
/// - Processes 1 sample per iteration
/// - No alignment requirements
#[derive(Copy, Clone, Debug)]
pub struct ScalarBackend;

impl SimdBackend for ScalarBackend {
    #[inline]
    fn process_block(input: &[f32], output: &mut [f32], params: &ProcessParams) {
        // Apply gain to each sample
        for (inp, out) in input.iter().zip(output.iter_mut()) {
            *out = inp * params.gain;
        }
    }

    #[inline]
    fn advance_phase_vectorized(phases: &mut [f32], phase_increments: &[f32], count: usize) {
        for i in 0..count {
            phases[i] += phase_increments[i];

            // Wrap phase to [0, TAU) range
            while phases[i] >= TAU {
                phases[i] -= TAU;
            }
            while phases[i] < 0.0 {
                phases[i] += TAU;
            }
        }
    }

    #[inline]
    fn interpolate_wavetable(wavetable: &[f32], positions: &[f32], output: &mut [f32]) {
        let table_len = wavetable.len() as f32;

        for (pos, out) in positions.iter().zip(output.iter_mut()) {
            // Normalize position to wavetable index
            let mut normalized_pos = *pos;

            // Wrap position to [0.0, 1.0) range (manual rem_euclid for no_std)
            normalized_pos = normalized_pos - libm::floorf(normalized_pos);

            // Convert to wavetable index
            let index_f = normalized_pos * table_len;
            let index0 = index_f as usize;
            let index1 = (index0 + 1) % wavetable.len();

            // Linear interpolation
            let frac = index_f - index0 as f32;
            let sample0 = wavetable[index0];
            let sample1 = wavetable[index1];

            *out = sample0 + frac * (sample1 - sample0);
        }
    }

    #[inline]
    fn name() -> &'static str {
        "scalar"
    }

    // ========================================================================
    // Arithmetic Operations (Binary)
    // ========================================================================

    #[inline]
    fn add(a: &[f32], b: &[f32], output: &mut [f32]) {
        for i in 0..output.len() {
            output[i] = a[i] + b[i];
        }
    }

    #[inline]
    fn sub(a: &[f32], b: &[f32], output: &mut [f32]) {
        for i in 0..output.len() {
            output[i] = a[i] - b[i];
        }
    }

    #[inline]
    fn mul(a: &[f32], b: &[f32], output: &mut [f32]) {
        for i in 0..output.len() {
            output[i] = a[i] * b[i];
        }
    }

    #[inline]
    fn div(a: &[f32], b: &[f32], output: &mut [f32]) {
        for i in 0..output.len() {
            output[i] = a[i] / b[i];
        }
    }

    #[inline]
    fn fma(a: &[f32], b: &[f32], c: &[f32], output: &mut [f32]) {
        for i in 0..output.len() {
            output[i] = a[i] * b[i] + c[i];
        }
    }

    // ========================================================================
    // Arithmetic Operations (Unary)
    // ========================================================================

    #[inline]
    fn neg(input: &[f32], output: &mut [f32]) {
        for i in 0..output.len() {
            output[i] = -input[i];
        }
    }

    #[inline]
    fn abs(input: &[f32], output: &mut [f32]) {
        for i in 0..output.len() {
            output[i] = input[i].abs();
        }
    }

    // ========================================================================
    // Comparison Operations
    // ========================================================================

    #[inline]
    fn min(a: &[f32], b: &[f32], output: &mut [f32]) {
        for i in 0..output.len() {
            output[i] = if a[i] < b[i] { a[i] } else { b[i] };
        }
    }

    #[inline]
    fn max(a: &[f32], b: &[f32], output: &mut [f32]) {
        for i in 0..output.len() {
            output[i] = if a[i] > b[i] { a[i] } else { b[i] };
        }
    }

    // ========================================================================
    // Basic Math Functions
    // ========================================================================

    #[inline]
    fn sqrt(input: &[f32], output: &mut [f32]) {
        process_unary::<ScalarVector<f32>, _>(input, output, math::sqrt::sqrt, libm::sqrtf);
    }

    #[inline]
    fn exp(input: &[f32], output: &mut [f32]) {
        process_unary::<ScalarVector<f32>, _>(input, output, math::exp::exp, libm::expf);
    }

    #[inline]
    fn log(input: &[f32], output: &mut [f32]) {
        process_unary::<ScalarVector<f32>, _>(input, output, math::log::log, libm::logf);
    }

    #[inline]
    fn log2(input: &[f32], output: &mut [f32]) {
        process_unary::<ScalarVector<f32>, _>(input, output, math::log::log2, libm::log2f);
    }

    #[inline]
    fn log10(input: &[f32], output: &mut [f32]) {
        process_unary::<ScalarVector<f32>, _>(input, output, math::log::log10, libm::log10f);
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
        process_unary::<ScalarVector<f32>, _>(input, output, math::trig::sin, libm::sinf);
    }

    #[inline]
    fn cos(input: &[f32], output: &mut [f32]) {
        process_unary::<ScalarVector<f32>, _>(input, output, math::trig::cos, libm::cosf);
    }

    #[inline]
    fn tan(input: &[f32], output: &mut [f32]) {
        // Use sin/cos to compute tan (no dedicated tan kernel in math module)
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
        process_unary::<ScalarVector<f32>, _>(input, output, math::atan::atan, libm::atanf);
    }

    #[inline]
    fn atan2(y: &[f32], x: &[f32], output: &mut [f32]) {
        process_binary::<ScalarVector<f32>, _>(y, x, output, math::atan::atan2, libm::atan2f);
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
        process_unary::<ScalarVector<f32>, _>(input, output, math::tanh::tanh, libm::tanhf);
    }

    // ========================================================================
    // Rounding Functions
    // ========================================================================

    #[inline]
    fn floor(input: &[f32], output: &mut [f32]) {
        for i in 0..output.len() {
            output[i] = libm::floorf(input[i]);
        }
    }

    #[inline]
    fn ceil(input: &[f32], output: &mut [f32]) {
        for i in 0..output.len() {
            output[i] = libm::ceilf(input[i]);
        }
    }

    #[inline]
    fn round(input: &[f32], output: &mut [f32]) {
        for i in 0..output.len() {
            output[i] = libm::roundf(input[i]);
        }
    }

    #[inline]
    fn trunc(input: &[f32], output: &mut [f32]) {
        for i in 0..output.len() {
            output[i] = libm::truncf(input[i]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_process_block_basic() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut output = [0.0; 4];
        let params = ProcessParams {
            gain: 0.5,
            frequency: 440.0,
            sample_rate: 44100.0,
        };

        ScalarBackend::process_block(&input, &mut output, &params);

        assert_eq!(output, [0.5, 1.0, 1.5, 2.0]);
    }

    #[test]
    fn test_advance_phase_wrapping() {
        let mut phases = [TAU - 0.1, 0.0, TAU / 2.0];
        let increments = [0.2, 0.1, 0.1];

        ScalarBackend::advance_phase_vectorized(&mut phases, &increments, 3);

        // First phase should wrap
        assert!(phases[0] >= 0.0 && phases[0] < TAU);
        assert!((phases[0] - 0.1).abs() < 1e-6);

        // Others should just increment
        assert!((phases[1] - 0.1).abs() < 1e-6);
        assert!((phases[2] - (TAU / 2.0 + 0.1)).abs() < 1e-6);
    }

    #[test]
    fn test_interpolate_wavetable_basic() {
        // Simple wavetable: 0, 1, 2, 3
        let wavetable = [0.0, 1.0, 2.0, 3.0];
        let positions = [0.0, 0.25, 0.5, 0.75];
        let mut output = [0.0; 4];

        ScalarBackend::interpolate_wavetable(&wavetable, &positions, &mut output);

        // Position 0.0 -> index 0, value 0.0
        assert!((output[0] - 0.0).abs() < 1e-6);

        // Position 0.25 -> index 1, value 1.0
        assert!((output[1] - 1.0).abs() < 1e-6);

        // Position 0.5 -> index 2, value 2.0
        assert!((output[2] - 2.0).abs() < 1e-6);

        // Position 0.75 -> index 3, value 3.0
        assert!((output[3] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_interpolate_wavetable_interpolation() {
        // Simple wavetable: 0, 2, 4, 6
        let wavetable = [0.0, 2.0, 4.0, 6.0];
        // Position 0.125 = halfway between index 0 and 1
        let positions = [0.125];
        let mut output = [0.0; 1];

        ScalarBackend::interpolate_wavetable(&wavetable, &positions, &mut output);

        // Should interpolate halfway between 0.0 and 2.0 = 1.0
        assert!((output[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_backend_name() {
        assert_eq!(ScalarBackend::name(), "scalar");
    }
}
