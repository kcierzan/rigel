//! SimdContext API Integration Tests
//!
//! These tests verify that the SimdContext unified API works identically across
//! all platforms and backend selection mechanisms (runtime dispatch on x86_64,
//! compile-time selection on aarch64).
//!
//! The SimdContext API is the primary public interface that DSP code should use,
//! abstracting away platform differences.

#[cfg(test)]
mod simd_context_api_tests {
    use rigel_math::simd::{ProcessParams, SimdContext};

    #[test]
    fn test_simd_context_initialization() {
        // SimdContext should initialize without panicking on any platform
        let ctx = SimdContext::new();

        // Backend name should be non-empty
        assert!(!ctx.backend_name().is_empty());

        println!(
            "Initialized SimdContext with backend: {}",
            ctx.backend_name()
        );
    }

    #[test]
    fn test_simd_context_process_block() {
        let ctx = SimdContext::new();

        let input = [1.0f32, 2.0, 3.0, 4.0];
        let mut output = [0.0f32; 4];
        let params = ProcessParams {
            gain: 0.5,
            frequency: 440.0,
            sample_rate: 44100.0,
        };

        ctx.process_block(&input, &mut output, &params);

        // Verify output is correct (simple gain multiplication)
        assert_eq!(output, [0.5, 1.0, 1.5, 2.0]);
    }

    #[test]
    fn test_simd_context_advance_phase() {
        let ctx = SimdContext::new();

        let mut phases = [0.0f32, 1.0, 2.0, 3.0];
        let increments = [0.1f32, 0.2, 0.3, 0.4];

        ctx.advance_phase(&mut phases, &increments, 4);

        // Verify phases were advanced correctly
        assert!((phases[0] - 0.1).abs() < 1e-6);
        assert!((phases[1] - 1.2).abs() < 1e-6);
        assert!((phases[2] - 2.3).abs() < 1e-6);
        assert!((phases[3] - 3.4).abs() < 1e-6);
    }

    #[test]
    fn test_simd_context_interpolate() {
        let ctx = SimdContext::new();

        let wavetable = [0.0f32, 1.0, 2.0, 3.0];
        let positions = [0.0, 0.25, 0.5, 0.75];
        let mut output = [0.0f32; 4];

        ctx.interpolate_linear(&wavetable, &positions, &mut output);

        // Verify interpolation worked (output should be modified)
        // The exact values depend on the interpolation implementation,
        // but we can verify the output is non-zero and reasonable
        assert!(output.iter().any(|&x| x != 0.0));

        // Verify output values are within the wavetable range
        for &val in &output {
            assert!(val >= wavetable[0] && val <= wavetable[wavetable.len() - 1]);
        }
    }

    #[test]
    fn test_simd_context_works_across_platforms() {
        // This test verifies that the same code works on x86_64 (runtime dispatch)
        // and aarch64 (compile-time NEON)

        let ctx = SimdContext::new();

        println!("Platform: {}", std::env::consts::ARCH);
        println!("Backend: {}", ctx.backend_name());

        // Process some data
        let input = vec![1.0f32; 1024];
        let mut output = vec![0.0f32; 1024];
        let params = ProcessParams {
            gain: 0.5,
            frequency: 440.0,
            sample_rate: 44100.0,
        };

        ctx.process_block(&input, &mut output, &params);

        // Verify all outputs are correct
        for val in &output {
            assert!((val - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn test_simd_context_zero_overhead_on_aarch64() {
        // On aarch64 with compile-time NEON selection, SimdContext should be zero-sized
        // and have no runtime overhead

        #[cfg(all(target_arch = "aarch64", not(feature = "runtime-dispatch")))]
        {
            use rigel_math::simd::SimdContext;
            // Verify SimdContext is zero-sized (compile-time optimization)
            assert_eq!(std::mem::size_of::<SimdContext>(), 0);
        }
    }

    #[test]
    fn test_simd_context_backend_selection_on_x86_64() {
        // Test backend selection on x86_64 (both compile-time and runtime-dispatch modes)

        // Compile-time AVX2 selection (CI runs: cargo test --features avx2)
        #[cfg(all(
            target_arch = "x86_64",
            feature = "avx2",
            not(feature = "avx512"),
            not(feature = "runtime-dispatch")
        ))]
        {
            use rigel_math::simd::SimdContext;
            let ctx = SimdContext::new();
            let backend_name = ctx.backend_name();

            assert_eq!(
                backend_name, "avx2",
                "Expected avx2 backend with --features avx2, got: {}",
                backend_name
            );
        }

        // Compile-time AVX-512 selection
        #[cfg(all(
            target_arch = "x86_64",
            feature = "avx512",
            not(feature = "runtime-dispatch")
        ))]
        {
            use rigel_math::simd::SimdContext;
            let ctx = SimdContext::new();
            let backend_name = ctx.backend_name();

            assert_eq!(
                backend_name, "avx512",
                "Expected avx512 backend with --features avx512, got: {}",
                backend_name
            );
        }

        // Runtime dispatch mode (CPU feature detection)
        #[cfg(all(target_arch = "x86_64", feature = "runtime-dispatch"))]
        {
            use rigel_math::simd::SimdContext;
            let ctx = SimdContext::new();

            // Backend should be selected based on CPU features
            let backend_name = ctx.backend_name();
            assert!(
                backend_name == "avx512" || backend_name == "avx2" || backend_name == "scalar",
                "Unexpected backend: {}",
                backend_name
            );
        }

        // Scalar fallback (no SIMD features on x86_64)
        #[cfg(all(
            target_arch = "x86_64",
            not(feature = "avx2"),
            not(feature = "avx512"),
            not(feature = "runtime-dispatch")
        ))]
        {
            use rigel_math::simd::SimdContext;
            let ctx = SimdContext::new();
            let backend_name = ctx.backend_name();

            assert_eq!(
                backend_name, "scalar",
                "Expected scalar backend without SIMD features, got: {}",
                backend_name
            );
        }
    }

    #[test]
    fn test_simd_context_backend_selection_on_aarch64() {
        // Test backend selection on aarch64

        // Compile-time NEON selection (CI runs: cargo test --features neon)
        #[cfg(all(
            target_arch = "aarch64",
            feature = "neon",
            not(feature = "force-scalar")
        ))]
        {
            use rigel_math::simd::SimdContext;
            let ctx = SimdContext::new();
            let backend_name = ctx.backend_name();

            assert_eq!(
                backend_name, "neon",
                "Expected neon backend on aarch64, got: {}",
                backend_name
            );
        }

        // Scalar fallback on aarch64 (no neon feature)
        #[cfg(all(target_arch = "aarch64", not(feature = "neon")))]
        {
            use rigel_math::simd::SimdContext;
            let ctx = SimdContext::new();
            let backend_name = ctx.backend_name();

            assert_eq!(
                backend_name, "scalar",
                "Expected scalar backend without neon feature, got: {}",
                backend_name
            );
        }
    }
}
