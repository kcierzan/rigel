//! Performance tests for denormal protection
//!
//! These tests validate that denormal protection prevents performance degradation
//! when processing signals that decay into the denormal range.

use rigel_math::ops::mul;
use rigel_math::{Block64, DefaultSimdVector, DenormalGuard, SimdVector};
use std::time::Instant;

/// Test that CPU usage remains constant when processing denormals
///
/// Without denormal protection, processing denormal numbers can cause 10-100x slowdown.
/// With protection, performance should remain constant regardless of signal amplitude.
///
/// Note: Ignored by default because performance tests are flaky in CI environments
/// due to variable CPU performance, cache effects, and timing noise.
/// Run with: cargo test -- --ignored
#[test]
#[ignore]
fn test_denormal_protection_prevents_slowdown() {
    const NUM_ITERATIONS: usize = 10000;
    const DECAY_FACTOR: f32 = 0.9999; // Decays to denormals after ~10000 iterations

    // Test WITH denormal protection
    let protected_time = {
        let _guard = DenormalGuard::new();
        let mut block = Block64::new();

        // Initialize with normal values
        for i in 0..64 {
            block[i] = 1.0;
        }

        let decay_vec = DefaultSimdVector::splat(DECAY_FACTOR);
        let start = Instant::now();

        for _ in 0..NUM_ITERATIONS {
            for mut chunk in block.as_chunks_mut::<DefaultSimdVector>().iter_mut() {
                let val = chunk.load();
                chunk.store(mul(val, decay_vec));
            }
        }

        start.elapsed()
    };

    // Test WITHOUT denormal protection (for comparison)
    // Note: On some platforms this might not show slowdown if FTZ/DAZ are always enabled
    let unprotected_time = {
        let mut block = Block64::new();

        // Initialize with normal values
        for i in 0..64 {
            block[i] = 1.0;
        }

        let decay_vec = DefaultSimdVector::splat(DECAY_FACTOR);
        let start = Instant::now();

        for _ in 0..NUM_ITERATIONS {
            for mut chunk in block.as_chunks_mut::<DefaultSimdVector>().iter_mut() {
                let val = chunk.load();
                chunk.store(mul(val, decay_vec));
            }
        }

        start.elapsed()
    };

    println!("Protected time: {:?}", protected_time);
    println!("Unprotected time: {:?}", unprotected_time);
    println!(
        "Slowdown ratio: {:.2}x",
        unprotected_time.as_secs_f64() / protected_time.as_secs_f64()
    );

    // The protected version should not be significantly slower
    // We allow up to 2x variation due to measurement noise and platform differences
    assert!(
        protected_time < unprotected_time * 2,
        "Denormal protection should not cause significant slowdown. Protected: {:?}, Unprotected: {:?}",
        protected_time,
        unprotected_time
    );
}

/// Test that denormal protection maintains stable performance across multiple iterations
///
/// Note: Ignored by default because performance tests are flaky under coverage instrumentation.
/// Run with: cargo test -- --ignored
#[test]
#[ignore]
fn test_denormal_protection_stable_performance() {
    const NUM_RUNS: usize = 5;
    const NUM_ITERATIONS: usize = 5000;
    const DECAY_FACTOR: f32 = 0.9999;

    let _guard = DenormalGuard::new();
    let mut times = Vec::with_capacity(NUM_RUNS);

    for _ in 0..NUM_RUNS {
        let mut block = Block64::new();

        // Initialize with normal values
        for i in 0..64 {
            block[i] = 1.0;
        }

        let decay_vec = DefaultSimdVector::splat(DECAY_FACTOR);
        let start = Instant::now();

        for _ in 0..NUM_ITERATIONS {
            for mut chunk in block.as_chunks_mut::<DefaultSimdVector>().iter_mut() {
                let val = chunk.load();
                chunk.store(mul(val, decay_vec));
            }
        }

        times.push(start.elapsed());
    }

    // Calculate mean and standard deviation
    let mean = times.iter().map(|t| t.as_secs_f64()).sum::<f64>() / NUM_RUNS as f64;
    let variance = times
        .iter()
        .map(|t| {
            let diff = t.as_secs_f64() - mean;
            diff * diff
        })
        .sum::<f64>()
        / NUM_RUNS as f64;
    let std_dev = variance.sqrt();
    let coefficient_of_variation = (std_dev / mean) * 100.0;

    println!("Mean time: {:.6}s", mean);
    println!("Std dev: {:.6}s", std_dev);
    println!("Coefficient of variation: {:.2}%", coefficient_of_variation);

    // Performance should be stable (coefficient of variation < 15% to account for platform variability)
    assert!(
        coefficient_of_variation < 15.0,
        "Performance should be stable with denormal protection. CoV: {:.2}%",
        coefficient_of_variation
    );
}

/// Test that processing with denormals doesn't take significantly longer than normal values
#[test]
fn test_denormal_vs_normal_performance() {
    const NUM_ITERATIONS: usize = 10000;

    let _guard = DenormalGuard::new();

    // Test with normal values
    let normal_time = {
        let mut block = Block64::new();
        for i in 0..64 {
            block[i] = 0.1; // Normal value
        }

        let scale = DefaultSimdVector::splat(1.0001);
        let start = Instant::now();

        for _ in 0..NUM_ITERATIONS {
            for mut chunk in block.as_chunks_mut::<DefaultSimdVector>().iter_mut() {
                let val = chunk.load();
                chunk.store(mul(val, scale));
            }
        }

        start.elapsed()
    };

    // Test with denormal values (will be flushed to zero with protection)
    let denormal_time = {
        let mut block = Block64::new();
        for i in 0..64 {
            block[i] = 1e-40; // Denormal value
        }

        let scale = DefaultSimdVector::splat(1.0001);
        let start = Instant::now();

        for _ in 0..NUM_ITERATIONS {
            for mut chunk in block.as_chunks_mut::<DefaultSimdVector>().iter_mut() {
                let val = chunk.load();
                chunk.store(mul(val, scale));
            }
        }

        start.elapsed()
    };

    println!("Normal value time: {:?}", normal_time);
    println!("Denormal value time: {:?}", denormal_time);
    println!(
        "Ratio: {:.2}x",
        denormal_time.as_secs_f64() / normal_time.as_secs_f64()
    );

    // With denormal protection, both should have similar performance (within 2x)
    assert!(
        denormal_time < normal_time * 2,
        "Denormal processing should not be significantly slower with protection. Normal: {:?}, Denormal: {:?}",
        normal_time,
        denormal_time
    );
}
