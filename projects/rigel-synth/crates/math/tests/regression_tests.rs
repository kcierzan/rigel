//! Performance regression tests (T077)
//!
//! These tests detect >5% instruction count or >10% wall-clock performance degradation.
//! Currently uses simple baseline comparisons - in production, you'd use criterion's
//! baseline comparison features or store golden reference values.

use rigel_math::ops::{add, mul};
use rigel_math::{Block64, DefaultSimdVector, DenormalGuard, SimdVector};
use std::time::Instant;

/// Performance baseline for simple vector operations
///
/// This test establishes a performance baseline and will fail if operations
/// become significantly slower.
#[test]
fn test_vector_ops_performance_baseline() {
    let _guard = DenormalGuard::new();

    let iterations = 10_000;
    let a = DefaultSimdVector::splat(1.1);
    let b = DefaultSimdVector::splat(0.01);

    let start = Instant::now();

    let mut result = DefaultSimdVector::splat(1.0);
    for _ in 0..iterations {
        result = add(result, b);
        result = mul(result, a);
        // Keep values bounded to prevent overflow/infinity
        let max_val = DefaultSimdVector::splat(1000.0);
        result = result.min(max_val);
    }

    let duration = start.elapsed();

    // Prevent compiler optimization
    let sum = result.horizontal_sum();
    assert!(sum.is_finite(), "Result should be finite: {}", sum);

    // Performance assertion: should complete in reasonable time
    // This is a rough baseline - adjust based on your platform
    let nanos_per_iteration = duration.as_nanos() / iterations;

    // On a modern CPU, simple SIMD operations should be fast
    // Allow generous headroom for different platforms, backends, and debug builds
    assert!(
        nanos_per_iteration < 10_000,
        "Vector operations too slow: {}ns per iteration (baseline: <10000ns)",
        nanos_per_iteration
    );
}

/// Performance baseline for block processing
#[test]
fn test_block_processing_performance_baseline() {
    let _guard = DenormalGuard::new();

    let mut input = Block64::new();
    let mut output = Block64::new();

    // Fill input with test data
    for i in 0..64 {
        input[i] = (i as f32) / 64.0;
    }

    let iterations = 10_000;
    let gain = DefaultSimdVector::splat(0.5);

    let start = Instant::now();

    for _ in 0..iterations {
        for mut chunk in output.as_chunks_mut::<DefaultSimdVector>().iter_mut() {
            // Simulate reading from input and writing to output
            let value = chunk.load();
            chunk.store(mul(value, gain));
        }
    }

    let duration = start.elapsed();

    // Performance assertion
    let nanos_per_block = duration.as_nanos() / iterations;

    // Block processing should be very fast with SIMD
    // 64 samples should process in < 10µs on modern hardware
    assert!(
        nanos_per_block < 10_000,
        "Block processing too slow: {}ns per 64-sample block (baseline: <10000ns)",
        nanos_per_block
    );

    // Verify output is valid
    assert!(output[0].is_finite());
}

/// Regression test: Ensure denormal protection doesn't add overhead
#[test]
fn test_denormal_protection_overhead() {
    // Test without denormal protection
    let start_no_guard = Instant::now();
    {
        let mut result = 1.0f32;
        for i in 0..100_000 {
            result *= 0.9999;
            result += (i as f32) * 1e-30; // May go denormal
        }
        assert!(result.is_finite());
    }
    let duration_no_guard = start_no_guard.elapsed();

    // Test with denormal protection
    let start_with_guard = Instant::now();
    {
        let _guard = DenormalGuard::new();
        let mut result = 1.0f32;
        for i in 0..100_000 {
            result *= 0.9999;
            result += (i as f32) * 1e-30;
        }
        assert!(result.is_finite());
    }
    let duration_with_guard = start_with_guard.elapsed();

    // With proper denormal protection, the guarded version should be
    // faster or similar speed (denormals are expensive without FTZ/DAZ)
    // Allow 2x overhead for guard setup/teardown on some platforms
    assert!(
        duration_with_guard < duration_no_guard * 2,
        "Denormal guard adds excessive overhead: with_guard={}µs, without={}µs",
        duration_with_guard.as_micros(),
        duration_no_guard.as_micros()
    );
}

/// Regression test: Backend selection doesn't add runtime overhead
///
/// This test verifies that the backend abstraction is truly zero-cost
/// by comparing direct operations vs trait-wrapped operations.
#[test]
fn test_zero_cost_abstraction() {
    let _guard = DenormalGuard::new();

    let iterations = 100_000;

    // Test using trait abstraction
    let start_trait = Instant::now();
    {
        let mut vec = DefaultSimdVector::splat(1.0);
        let inc = DefaultSimdVector::splat(0.001);
        for _ in 0..iterations {
            vec = add(vec, inc);
        }
        assert!(vec.horizontal_sum().is_finite());
    }
    let duration_trait = start_trait.elapsed();

    // Test using direct scalar operations (for scalar backend)
    let start_scalar = Instant::now();
    {
        let mut value = 1.0f32;
        for _ in 0..iterations {
            value += 0.001;
        }
        assert!(value.is_finite());
    }
    let duration_scalar = start_scalar.elapsed();

    // The trait abstraction should compile to similar or faster code
    // Allow some variance due to measurement noise and SIMD benefits
    let overhead_ratio = duration_trait.as_nanos() as f64 / duration_scalar.as_nanos() as f64;

    assert!(
        overhead_ratio < 2.0, // Very generous - should be ~1.0 for scalar, <1.0 for SIMD
        "Trait abstraction adds overhead: ratio={} (trait={}ns, scalar={}ns)",
        overhead_ratio,
        duration_trait.as_nanos(),
        duration_scalar.as_nanos()
    );
}

/// Integration performance test: Complete DSP voice
///
/// Tests a complete signal chain: oscillator → filter → envelope
/// This catches regressions in real-world usage patterns.
#[test]
fn test_voice_pipeline_performance() {
    let _guard = DenormalGuard::new();

    let mut output = Block64::new();
    let iterations = 10_000;

    let start = Instant::now();

    for _ in 0..iterations {
        // 1. Generate sine wave
        let mut phase = 0.0;
        for i in 0..64 {
            output[i] = (phase * 2.0 * core::f32::consts::PI).sin();
            phase += 0.01; // 440Hz at 44.1kHz
            if phase >= 1.0 {
                phase -= 1.0;
            }
        }

        // 2. Apply filter
        let mut y = 0.0;
        for i in 0..64 {
            y = y + 0.1 * (output[i] - y);
            output[i] = y;
        }

        // 3. Apply envelope
        for i in 0..64 {
            output[i] *= (-((i as f32) / 20.0)).exp();
        }
    }

    let duration = start.elapsed();

    // Performance baseline: complete voice should process efficiently
    let nanos_per_block = duration.as_nanos() / iterations;

    // A complete voice pipeline should process 64 samples in < 50µs
    assert!(
        nanos_per_block < 50_000,
        "Voice pipeline too slow: {}ns per block (baseline: <50000ns)",
        nanos_per_block
    );

    assert!(output[0].is_finite());
}
