//! Backend consistency tests (T024, T025)
//!
//! Validates that scalar and SIMD backends produce identical (or within tolerance) results.
//! Tests simple DSP algorithms to ensure backend abstraction works correctly.

use rigel_math::{DefaultSimdVector, SimdMask, SimdVector};

/// Test T024: Backend consistency - scalar vs SIMD backends produce identical results
///
/// This test verifies that operations produce the same results regardless of backend.
#[test]
fn test_backend_arithmetic_consistency() {
    let test_values = [
        (2.0, 3.0),
        (1.5, 2.5),
        (-1.0, 4.0),
        (0.0, 5.0),
        (100.0, 0.01),
    ];

    for (a, b) in test_values.iter() {
        let vec_a = DefaultSimdVector::splat(*a);
        let vec_b = DefaultSimdVector::splat(*b);

        // Expected values (computed with scalar arithmetic)
        let expected_sum = a + b;
        let expected_diff = a - b;
        let expected_prod = a * b;
        let expected_div = if *b != 0.0 { a / b } else { f32::INFINITY };

        // SIMD results (per-lane should match expected)
        let sum = vec_a.add(vec_b);
        let diff = vec_a.sub(vec_b);
        let prod = vec_a.mul(vec_b);
        let quot = vec_a.div(vec_b);

        // Verify each lane matches expected value
        let mut sum_result = vec![0.0; DefaultSimdVector::LANES];
        let mut diff_result = vec![0.0; DefaultSimdVector::LANES];
        let mut prod_result = vec![0.0; DefaultSimdVector::LANES];
        let mut quot_result = vec![0.0; DefaultSimdVector::LANES];

        sum.to_slice(&mut sum_result);
        diff.to_slice(&mut diff_result);
        prod.to_slice(&mut prod_result);
        quot.to_slice(&mut quot_result);

        for i in 0..DefaultSimdVector::LANES {
            assert_eq!(
                sum_result[i], expected_sum,
                "Addition mismatch at lane {}",
                i
            );
            assert_eq!(
                diff_result[i], expected_diff,
                "Subtraction mismatch at lane {}",
                i
            );
            assert_eq!(
                prod_result[i], expected_prod,
                "Multiplication mismatch at lane {}",
                i
            );
            assert_eq!(
                quot_result[i], expected_div,
                "Division mismatch at lane {}",
                i
            );
        }
    }
}

/// Test FMA consistency
#[test]
fn test_backend_fma_consistency() {
    let test_values = [(2.0, 3.0, 1.0), (1.5, 2.0, 0.5), (-1.0, 4.0, 2.0)];

    for (a, b, c) in test_values.iter() {
        let vec_a = DefaultSimdVector::splat(*a);
        let vec_b = DefaultSimdVector::splat(*b);
        let vec_c = DefaultSimdVector::splat(*c);

        let expected_fma = a * b + c;
        let fma_result = vec_a.fma(vec_b, vec_c);

        let mut result = vec![0.0; DefaultSimdVector::LANES];
        fma_result.to_slice(&mut result);

        for i in 0..DefaultSimdVector::LANES {
            // FMA might have slightly different precision
            let diff = (result[i] - expected_fma).abs();
            assert!(
                diff < 1e-6 || diff / expected_fma.abs() < 1e-6,
                "FMA mismatch at lane {}: got {}, expected {}",
                i,
                result[i],
                expected_fma
            );
        }
    }
}

/// Test min/max consistency
#[test]
fn test_backend_minmax_consistency() {
    let test_values = [(2.0, 3.0), (5.0, 1.0), (-1.0, 4.0), (0.0, 0.0)];

    for (a, b) in test_values.iter() {
        let vec_a = DefaultSimdVector::splat(*a);
        let vec_b = DefaultSimdVector::splat(*b);

        let expected_min = a.min(*b);
        let expected_max = a.max(*b);

        let min_result = vec_a.min(vec_b);
        let max_result = vec_a.max(vec_b);

        let mut min_vals = vec![0.0; DefaultSimdVector::LANES];
        let mut max_vals = vec![0.0; DefaultSimdVector::LANES];

        min_result.to_slice(&mut min_vals);
        max_result.to_slice(&mut max_vals);

        for i in 0..DefaultSimdVector::LANES {
            assert_eq!(min_vals[i], expected_min, "Min mismatch at lane {}", i);
            assert_eq!(max_vals[i], expected_max, "Max mismatch at lane {}", i);
        }
    }
}

/// Test comparison operations consistency
#[test]
fn test_backend_comparison_consistency() {
    let test_values = [
        (2.0, 3.0, true, false, false),
        (3.0, 2.0, false, true, false),
        (2.0, 2.0, false, false, true),
    ];

    for (a, b, expected_lt, expected_gt, expected_eq) in test_values.iter() {
        let vec_a = DefaultSimdVector::splat(*a);
        let vec_b = DefaultSimdVector::splat(*b);

        let mask_lt = vec_a.lt(vec_b);
        let mask_gt = vec_a.gt(vec_b);
        let mask_eq = vec_a.eq(vec_b);

        assert_eq!(
            mask_lt.all(),
            *expected_lt,
            "Less-than comparison mismatch for {} < {}",
            a,
            b
        );
        assert_eq!(
            mask_gt.all(),
            *expected_gt,
            "Greater-than comparison mismatch for {} > {}",
            a,
            b
        );
        assert_eq!(
            mask_eq.all(),
            *expected_eq,
            "Equality comparison mismatch for {} == {}",
            a,
            b
        );
    }
}

/// Test T025: Integration test - Simple DSP algorithm compiles and runs with all backends
///
/// Implements a simple gain adjustment as a representative DSP operation.
#[test]
fn test_simple_dsp_gain_adjustment() {
    const BLOCK_SIZE: usize = 64;
    let mut input = [0.0f32; BLOCK_SIZE];
    let mut output = [0.0f32; BLOCK_SIZE];

    // Create a simple test signal (sine-like pattern)
    for i in 0..BLOCK_SIZE {
        input[i] = (i as f32 / BLOCK_SIZE as f32) * 2.0 - 1.0; // -1.0 to 1.0
    }

    let gain = 0.5;
    let gain_vec = DefaultSimdVector::splat(gain);

    // Process in SIMD chunks
    let lanes = DefaultSimdVector::LANES;
    for chunk_start in (0..BLOCK_SIZE).step_by(lanes) {
        if chunk_start + lanes <= BLOCK_SIZE {
            let input_vec = DefaultSimdVector::from_slice(&input[chunk_start..]);
            let output_vec = input_vec.mul(gain_vec);
            output_vec.to_slice(&mut output[chunk_start..]);
        }
    }

    // Verify results
    for i in 0..(BLOCK_SIZE / lanes) * lanes {
        let expected = input[i] * gain;
        assert!(
            (output[i] - expected).abs() < 1e-6,
            "Gain adjustment failed at sample {}: got {}, expected {}",
            i,
            output[i],
            expected
        );
    }
}

/// Test simple mixing algorithm (two signals combined)
#[test]
fn test_simple_dsp_mixing() {
    const BLOCK_SIZE: usize = 64;
    let mut signal_a = [0.0f32; BLOCK_SIZE];
    let mut signal_b = [0.0f32; BLOCK_SIZE];
    let mut output = [0.0f32; BLOCK_SIZE];

    // Create test signals
    for i in 0..BLOCK_SIZE {
        signal_a[i] = (i as f32 / BLOCK_SIZE as f32);
        signal_b[i] = 1.0 - (i as f32 / BLOCK_SIZE as f32);
    }

    let mix = 0.7; // 70% signal A, 30% signal B
    let mix_a = DefaultSimdVector::splat(mix);
    let mix_b = DefaultSimdVector::splat(1.0 - mix);

    // Process in SIMD chunks
    let lanes = DefaultSimdVector::LANES;
    for chunk_start in (0..BLOCK_SIZE).step_by(lanes) {
        if chunk_start + lanes <= BLOCK_SIZE {
            let vec_a = DefaultSimdVector::from_slice(&signal_a[chunk_start..]);
            let vec_b = DefaultSimdVector::from_slice(&signal_b[chunk_start..]);

            // output = signal_a * mix + signal_b * (1 - mix)
            let weighted_a = vec_a.mul(mix_a);
            let weighted_b = vec_b.mul(mix_b);
            let mixed = weighted_a.add(weighted_b);

            mixed.to_slice(&mut output[chunk_start..]);
        }
    }

    // Verify results
    for i in 0..(BLOCK_SIZE / lanes) * lanes {
        let expected = signal_a[i] * mix + signal_b[i] * (1.0 - mix);
        assert!(
            (output[i] - expected).abs() < 1e-5,
            "Mixing failed at sample {}: got {}, expected {}",
            i,
            output[i],
            expected
        );
    }
}

/// Test clipping/saturation algorithm
#[test]
fn test_simple_dsp_clipping() {
    const BLOCK_SIZE: usize = 64;
    let mut input = [0.0f32; BLOCK_SIZE];
    let mut output = [0.0f32; BLOCK_SIZE];

    // Create test signal with values exceeding [-1, 1]
    for i in 0..BLOCK_SIZE {
        input[i] = (i as f32 / BLOCK_SIZE as f32) * 4.0 - 2.0; // -2.0 to 2.0
    }

    let min_val = DefaultSimdVector::splat(-1.0);
    let max_val = DefaultSimdVector::splat(1.0);

    // Process in SIMD chunks
    let lanes = DefaultSimdVector::LANES;
    for chunk_start in (0..BLOCK_SIZE).step_by(lanes) {
        if chunk_start + lanes <= BLOCK_SIZE {
            let input_vec = DefaultSimdVector::from_slice(&input[chunk_start..]);
            let clamped = input_vec.max(min_val).min(max_val);
            clamped.to_slice(&mut output[chunk_start..]);
        }
    }

    // Verify results
    for i in 0..(BLOCK_SIZE / lanes) * lanes {
        let expected = input[i].clamp(-1.0, 1.0);
        assert_eq!(
            output[i], expected,
            "Clipping failed at sample {}: got {}, expected {}",
            i, output[i], expected
        );
    }
}
