//! Property-Based Backend Equivalence Tests
//!
//! This test suite uses proptest to verify that all SIMD backend implementations
//! produce functionally identical results. We generate thousands of random inputs
//! and ensure that Scalar, AVX2, AVX-512, and NEON backends all produce the same
//! output within floating-point precision (1e-6 tolerance).
//!
//! These tests are critical for ensuring correctness - if a backend produces different
//! results, it's either buggy or has a precision issue that needs investigation.

use rigel_math::simd::{ProcessParams, ScalarBackend, SimdBackend};

#[cfg(test)]
mod backend_equivalence_tests {
    use super::*;
    use proptest::prelude::*;

    /// Tolerance for floating-point comparisons (1e-6 as per contract)
    #[allow(dead_code)] // Used in conditional compilation blocks
    const EPSILON: f32 = 1e-6;

    /// Helper to compare two f32 slices with tolerance
    #[allow(dead_code)] // Used in conditional compilation blocks
    fn assert_slices_approx_eq(a: &[f32], b: &[f32], epsilon: f32, context: &str) {
        assert_eq!(
            a.len(),
            b.len(),
            "{}: slice lengths differ: {} vs {}",
            context,
            a.len(),
            b.len()
        );

        for (i, (val_a, val_b)) in a.iter().zip(b.iter()).enumerate() {
            let diff = (val_a - val_b).abs();
            assert!(
                diff < epsilon || (val_a.is_nan() && val_b.is_nan()),
                "{}: index {}: values differ by {}: {} vs {} (epsilon: {})",
                context,
                i,
                diff,
                val_a,
                val_b,
                epsilon
            );
        }
    }

    proptest! {
        /// Test process_block equivalence across all backends
        ///
        /// Generates random input buffers and ProcessParams, then verifies that
        /// all backends produce identical output.
        #[test]
        fn test_process_block_equivalence(
            input in prop::collection::vec(-10.0f32..10.0f32, 1..1024),
            gain in -2.0f32..2.0f32,
            frequency in 20.0f32..20000.0f32,
            sample_rate in prop::sample::select(vec![44100.0f32, 48000.0f32, 96000.0f32]),
        ) {
            let params = ProcessParams { gain, frequency, sample_rate };
            let mut scalar_output = vec![0.0f32; input.len()];

            // Scalar backend (reference implementation)
            ScalarBackend::process_block(&input, &mut scalar_output, &params);

            // AVX2 backend (if compiled)
            #[cfg(all(feature = "avx2", target_arch = "x86_64"))]
            {
                use rigel_math::simd::Avx2Backend;
                let mut avx2_output = vec![0.0f32; input.len()];
                Avx2Backend::process_block(&input, &mut avx2_output, &params);
                assert_slices_approx_eq(
                    &scalar_output,
                    &avx2_output,
                    EPSILON,
                    "AVX2 vs Scalar process_block"
                );
            }

            // AVX-512 backend (if compiled)
            #[cfg(all(feature = "avx512", target_arch = "x86_64"))]
            {
                use rigel_math::simd::Avx512Backend;
                let mut avx512_output = vec![0.0f32; input.len()];
                Avx512Backend::process_block(&input, &mut avx512_output, &params);
                assert_slices_approx_eq(
                    &scalar_output,
                    &avx512_output,
                    EPSILON,
                    "AVX512 vs Scalar process_block"
                );
            }

            // NEON backend (if compiled)
            #[cfg(all(feature = "neon", target_arch = "aarch64"))]
            {
                use rigel_math::simd::NeonBackend;
                let mut neon_output = vec![0.0f32; input.len()];
                NeonBackend::process_block(&input, &mut neon_output, &params);
                assert_slices_approx_eq(
                    &scalar_output,
                    &neon_output,
                    EPSILON,
                    "NEON vs Scalar process_block"
                );
            }
        }

        /// Test advance_phase_vectorized equivalence across all backends
        ///
        /// Verifies that phase advancement (with wrapping) produces identical results.
        #[test]
        fn test_advance_phase_equivalence(
            initial_phases in prop::collection::vec(0.0f32..std::f32::consts::TAU, 1..256),
            increments in prop::collection::vec(0.0f32..1.0f32, 1..256),
        ) {
            let count = initial_phases.len().min(increments.len());
            let mut scalar_phases = initial_phases.clone();
            #[allow(unused_variables)] // Used in conditional compilation blocks
            let phases_copy = initial_phases.clone();

            // Scalar backend (reference)
            ScalarBackend::advance_phase_vectorized(&mut scalar_phases, &increments, count);

            // AVX2 backend
            #[cfg(all(feature = "avx2", target_arch = "x86_64"))]
            {
                use rigel_math::simd::Avx2Backend;
                let mut avx2_phases = phases_copy.clone();
                Avx2Backend::advance_phase_vectorized(&mut avx2_phases, &increments, count);
                assert_slices_approx_eq(
                    &scalar_phases[..count],
                    &avx2_phases[..count],
                    EPSILON,
                    "AVX2 vs Scalar advance_phase"
                );
            }

            // AVX-512 backend
            #[cfg(all(feature = "avx512", target_arch = "x86_64"))]
            {
                use rigel_math::simd::Avx512Backend;
                let mut avx512_phases = phases_copy.clone();
                Avx512Backend::advance_phase_vectorized(&mut avx512_phases, &increments, count);
                assert_slices_approx_eq(
                    &scalar_phases[..count],
                    &avx512_phases[..count],
                    EPSILON,
                    "AVX512 vs Scalar advance_phase"
                );
            }

            // NEON backend
            #[cfg(all(feature = "neon", target_arch = "aarch64"))]
            {
                use rigel_math::simd::NeonBackend;
                let mut neon_phases = phases_copy;
                NeonBackend::advance_phase_vectorized(&mut neon_phases, &increments, count);
                assert_slices_approx_eq(
                    &scalar_phases[..count],
                    &neon_phases[..count],
                    EPSILON,
                    "NEON vs Scalar advance_phase"
                );
            }
        }

        /// Test interpolate_wavetable equivalence across all backends
        ///
        /// Verifies that wavetable interpolation produces identical results.
        #[test]
        fn test_interpolate_wavetable_equivalence(
            wavetable in prop::collection::vec(-1.0f32..1.0f32, 64..2048),
            positions in prop::collection::vec(0.0f32..1.0f32, 1..512),
        ) {
            let mut scalar_output = vec![0.0f32; positions.len()];

            // Scalar backend (reference)
            ScalarBackend::interpolate_wavetable(&wavetable, &positions, &mut scalar_output);

            // AVX2 backend
            #[cfg(all(feature = "avx2", target_arch = "x86_64"))]
            {
                use rigel_math::simd::Avx2Backend;
                let mut avx2_output = vec![0.0f32; positions.len()];
                Avx2Backend::interpolate_wavetable(&wavetable, &positions, &mut avx2_output);
                assert_slices_approx_eq(
                    &scalar_output,
                    &avx2_output,
                    EPSILON,
                    "AVX2 vs Scalar interpolate_wavetable"
                );
            }

            // AVX-512 backend
            #[cfg(all(feature = "avx512", target_arch = "x86_64"))]
            {
                use rigel_math::simd::Avx512Backend;
                let mut avx512_output = vec![0.0f32; positions.len()];
                Avx512Backend::interpolate_wavetable(&wavetable, &positions, &mut avx512_output);
                assert_slices_approx_eq(
                    &scalar_output,
                    &avx512_output,
                    EPSILON,
                    "AVX512 vs Scalar interpolate_wavetable"
                );
            }

            // NEON backend
            #[cfg(all(feature = "neon", target_arch = "aarch64"))]
            {
                use rigel_math::simd::NeonBackend;
                let mut neon_output = vec![0.0f32; positions.len()];
                NeonBackend::interpolate_wavetable(&wavetable, &positions, &mut neon_output);
                assert_slices_approx_eq(
                    &scalar_output,
                    &neon_output,
                    EPSILON,
                    "NEON vs Scalar interpolate_wavetable"
                );
            }
        }
    }
}
