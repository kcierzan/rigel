//! Backend Selection Integration Tests
//!
//! These tests verify that the runtime dispatcher correctly selects backends based on:
//! 1. CPU features detected at runtime
//! 2. Forced backend feature flags (for deterministic testing)
//! 3. Edge cases (NaN, infinity, zero inputs)

use rigel_math::simd::{BackendType, CpuFeatures, ProcessParams, ScalarBackend, SimdBackend};

#[cfg(test)]
mod backend_selection_tests {
    use super::*;

    #[test]
    fn test_cpu_feature_detection() {
        // Test that CPU feature detection doesn't panic and returns valid data
        let features = CpuFeatures::detect();

        #[cfg(target_arch = "x86_64")]
        {
            // On x86_64, at least one of the flags should be determinable
            // We can't assert specific values since it depends on the CPU
            println!("Detected CPU features:");
            println!("  AVX2: {}", features.has_avx2);
            println!("  AVX-512F: {}", features.has_avx512_f);
            println!("  AVX-512BW: {}", features.has_avx512_bw);
            println!("  AVX-512DQ: {}", features.has_avx512_dq);
            println!("  AVX-512VL: {}", features.has_avx512_vl);
            println!("  Full AVX-512: {}", features.has_avx512_full());
        }

        #[cfg(target_arch = "aarch64")]
        {
            // On aarch64, all AVX flags should be false (NEON is assumed always present)
            assert!(!features.has_avx2);
            assert!(!features.has_avx512_f);
            assert!(!features.has_avx512_full());
        }
    }

    #[test]
    fn test_backend_type_selection() {
        // Test that backend selection logic works
        let features = CpuFeatures::detect();
        let backend_type = BackendType::select(features);

        println!("Selected backend: {:?}", backend_type);
        assert!(!backend_type.name().is_empty());

        // Verify forced backend flags work (compile-time check)
        #[cfg(feature = "force-scalar")]
        {
            assert_eq!(backend_type, BackendType::Scalar);
            assert_eq!(backend_type.name(), "scalar");
        }

        #[cfg(feature = "force-avx2")]
        {
            assert_eq!(backend_type, BackendType::Avx2);
            assert_eq!(backend_type.name(), "avx2");
        }

        #[cfg(feature = "force-avx512")]
        {
            assert_eq!(backend_type, BackendType::Avx512);
            assert_eq!(backend_type.name(), "avx512");
        }

        #[cfg(feature = "force-neon")]
        {
            assert_eq!(backend_type, BackendType::Neon);
            assert_eq!(backend_type.name(), "neon");
        }

        // Without forced flags, verify selection matches detected features
        #[cfg(not(any(
            feature = "force-scalar",
            feature = "force-avx2",
            feature = "force-avx512",
            feature = "force-neon"
        )))]
        {
            #[cfg(target_arch = "x86_64")]
            {
                // Priority order: AVX-512 → AVX2 → Scalar
                #[cfg(feature = "avx512")]
                if features.has_avx512_full() {
                    assert_eq!(backend_type, BackendType::Avx512);
                } else if features.has_avx2 && cfg!(feature = "avx2") {
                    assert_eq!(backend_type, BackendType::Avx2);
                } else {
                    assert_eq!(backend_type, BackendType::Scalar);
                }

                #[cfg(not(feature = "avx512"))]
                if features.has_avx2 && cfg!(feature = "avx2") {
                    assert_eq!(backend_type, BackendType::Avx2);
                } else {
                    assert_eq!(backend_type, BackendType::Scalar);
                }
            }

            #[cfg(target_arch = "aarch64")]
            {
                #[cfg(feature = "neon")]
                {
                    assert_eq!(backend_type, BackendType::Neon);
                }

                #[cfg(not(feature = "neon"))]
                {
                    assert_eq!(backend_type, BackendType::Scalar);
                }
            }
        }
    }

    #[test]
    fn test_backend_name_consistency() {
        // Verify BackendType name() method returns correct strings
        assert_eq!(BackendType::Scalar.name(), "scalar");
        assert_eq!(BackendType::Avx2.name(), "avx2");
        assert_eq!(BackendType::Avx512.name(), "avx512");
        assert_eq!(BackendType::Neon.name(), "neon");
    }

    #[test]
    fn test_dispatcher_init_validation() {
        // FR-010: dispatcher.init() correctly validates CPU features and never selects
        // an unsupported backend
        use rigel_math::simd::BackendDispatcher;

        let dispatcher = BackendDispatcher::init();
        let backend_type = dispatcher.backend_type();
        let backend_name = dispatcher.backend_name();

        println!(
            "Dispatcher initialized with backend: {} ({:?})",
            backend_name, backend_type
        );

        // Verify backend name and type are consistent
        assert_eq!(backend_name, backend_type.name());

        // Verify the selected backend matches CPU capabilities
        let features = CpuFeatures::detect();

        #[cfg(not(any(
            feature = "force-scalar",
            feature = "force-avx2",
            feature = "force-avx512",
            feature = "force-neon"
        )))]
        {
            #[cfg(target_arch = "x86_64")]
            {
                // Dispatcher should never select a backend that the CPU doesn't support
                match backend_type {
                    BackendType::Avx512 => {
                        assert!(
                            features.has_avx512_full(),
                            "AVX-512 backend selected but CPU doesn't support it"
                        );
                    }
                    BackendType::Avx2 => {
                        // If AVX2 is selected, either:
                        // 1. CPU has AVX2, or
                        // 2. CPU doesn't have AVX-512 (so AVX2 is next best)
                        if !features.has_avx512_full() {
                            assert!(
                                features.has_avx2 || !cfg!(feature = "avx2"),
                                "AVX2 backend selected but CPU doesn't support it"
                            );
                        }
                    }
                    BackendType::Scalar => {
                        // Scalar is always valid as fallback
                    }
                    BackendType::Neon => {
                        panic!("NEON backend should not be selected on x86_64");
                    }
                }
            }

            #[cfg(target_arch = "aarch64")]
            {
                // On aarch64, should select NEON or scalar
                match backend_type {
                    BackendType::Neon => {
                        // NEON is always available on aarch64
                        assert!(true);
                    }
                    BackendType::Scalar => {
                        // Scalar is always valid as fallback
                    }
                    BackendType::Avx2 | BackendType::Avx512 => {
                        panic!("x86 backend should not be selected on aarch64");
                    }
                }
            }
        }

        // Verify forced backend flags are respected
        #[cfg(feature = "force-scalar")]
        {
            assert_eq!(
                backend_type,
                BackendType::Scalar,
                "force-scalar flag not respected"
            );
        }

        #[cfg(feature = "force-avx2")]
        {
            assert_eq!(
                backend_type,
                BackendType::Avx2,
                "force-avx2 flag not respected"
            );
        }

        #[cfg(feature = "force-avx512")]
        {
            assert_eq!(
                backend_type,
                BackendType::Avx512,
                "force-avx512 flag not respected"
            );
        }

        #[cfg(feature = "force-neon")]
        {
            assert_eq!(
                backend_type,
                BackendType::Neon,
                "force-neon flag not respected"
            );
        }
    }
}

#[cfg(test)]
mod edge_case_tests {
    use super::*;

    /// Helper to test edge cases across all compiled backends
    fn test_edge_case_across_backends<F>(test_fn: F, test_name: &str)
    where
        F: Fn(&dyn Fn(&[f32], &mut [f32], &ProcessParams)),
    {
        println!("Testing edge case: {}", test_name);

        // Scalar backend (always available)
        test_fn(&|input, output, params| {
            ScalarBackend::process_block(input, output, params);
        });

        // AVX2 backend
        #[cfg(all(feature = "avx2", target_arch = "x86_64"))]
        {
            use rigel_math::simd::Avx2Backend;
            test_fn(&|input, output, params| {
                Avx2Backend::process_block(input, output, params);
            });
        }

        // AVX-512 backend
        #[cfg(all(feature = "avx512", target_arch = "x86_64"))]
        {
            use rigel_math::simd::Avx512Backend;
            test_fn(&|input, output, params| {
                Avx512Backend::process_block(input, output, params);
            });
        }

        // NEON backend
        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        {
            use rigel_math::simd::NeonBackend;
            test_fn(&|input, output, params| {
                NeonBackend::process_block(input, output, params);
            });
        }
    }

    #[test]
    fn test_nan_inputs() {
        let params = ProcessParams {
            gain: 0.5,
            frequency: 440.0,
            sample_rate: 44100.0,
        };

        test_edge_case_across_backends(
            |process_fn| {
                let input = [f32::NAN, 1.0, f32::NAN, 2.0];
                let mut output = [0.0f32; 4];

                process_fn(&input, &mut output, &params);

                // NaN should propagate (NaN * anything = NaN)
                assert!(output[0].is_nan(), "NaN input should produce NaN output");
                assert!(
                    !output[1].is_nan(),
                    "Normal input should produce normal output"
                );
                assert!(output[2].is_nan(), "NaN input should produce NaN output");
            },
            "NaN inputs",
        );
    }

    #[test]
    fn test_infinity_inputs() {
        let params = ProcessParams {
            gain: 0.5,
            frequency: 440.0,
            sample_rate: 44100.0,
        };

        test_edge_case_across_backends(
            |process_fn| {
                let input = [f32::INFINITY, f32::NEG_INFINITY, 1.0, -1.0];
                let mut output = [0.0f32; 4];

                process_fn(&input, &mut output, &params);

                // Infinity * 0.5 should still be infinity
                assert!(output[0].is_infinite() && output[0].is_sign_positive());
                assert!(output[1].is_infinite() && output[1].is_sign_negative());
                assert_eq!(output[2], 0.5);
                assert_eq!(output[3], -0.5);
            },
            "Infinity inputs",
        );
    }

    #[test]
    fn test_zero_inputs() {
        let params = ProcessParams {
            gain: 0.5,
            frequency: 440.0,
            sample_rate: 44100.0,
        };

        test_edge_case_across_backends(
            |process_fn| {
                let input = [0.0, -0.0, 1.0, 0.0];
                let mut output = [0.0f32; 4];

                process_fn(&input, &mut output, &params);

                // Zero should remain zero
                assert_eq!(output[0], 0.0);
                assert_eq!(output[1], 0.0);
                assert_eq!(output[2], 0.5);
                assert_eq!(output[3], 0.0);
            },
            "Zero inputs",
        );
    }

    #[test]
    fn test_very_small_values() {
        let params = ProcessParams {
            gain: 0.5,
            frequency: 440.0,
            sample_rate: 44100.0,
        };

        test_edge_case_across_backends(
            |process_fn| {
                // Denormal numbers
                let input = [1e-40f32, -1e-40f32, 1e-38f32, -1e-38f32];
                let mut output = [0.0f32; 4];

                process_fn(&input, &mut output, &params);

                // Very small values should be handled correctly (may flush to zero with FTZ)
                // We just verify no crashes or NaNs
                for val in &output {
                    assert!(!val.is_nan(), "Small values should not produce NaN");
                }
            },
            "Very small values (denormals)",
        );
    }

    #[test]
    fn test_mixed_edge_cases() {
        let params = ProcessParams {
            gain: 0.5,
            frequency: 440.0,
            sample_rate: 44100.0,
        };

        test_edge_case_across_backends(
            |process_fn| {
                let input = [
                    f32::NAN,
                    f32::INFINITY,
                    f32::NEG_INFINITY,
                    0.0,
                    1.0,
                    -1.0,
                    1e-40f32,
                    1e38f32,
                ];
                let mut output = [0.0f32; 8];

                process_fn(&input, &mut output, &params);

                // Verify no crashes and basic sanity
                assert!(output[0].is_nan());
                assert!(output[1].is_infinite());
                assert!(output[2].is_infinite());
                assert_eq!(output[3], 0.0);
                assert_eq!(output[4], 0.5);
                assert_eq!(output[5], -0.5);
                // output[6] and output[7] may vary due to denormals/rounding
            },
            "Mixed edge cases",
        );
    }
}
