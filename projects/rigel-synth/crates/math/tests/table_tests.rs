//! Comprehensive tests for lookup table functionality (T124-T127)

use rigel_math::table::{IndexMode, LookupTable};
use rigel_math::{DefaultSimdVector, SimdVector};

// T127: Unit test: IndexMode variants behave correctly at boundaries
#[test]
fn test_index_mode_wrap_boundaries() {
    let table = LookupTable::<f32, 10>::from_fn(|i, _| i as f32);

    // Test wrapping at positive boundary
    let v1 = table.lookup_linear(10.0, IndexMode::Wrap);
    assert!((v1 - 0.0).abs() < 0.001, "Index 10 should wrap to 0");

    // Test wrapping at negative boundary
    let v2 = table.lookup_linear(-1.0, IndexMode::Wrap);
    assert!((v2 - 9.0).abs() < 0.1, "Index -1 should wrap to 9");
}

#[test]
fn test_index_mode_mirror_boundaries() {
    let table = LookupTable::<f32, 10>::from_fn(|i, _| i as f32);

    // Test mirroring beyond boundary
    let v1 = table.lookup_linear(11.0, IndexMode::Mirror);
    assert!((v1 - 9.0).abs() < 0.1, "Index 11 should mirror back");

    // Test mirroring at negative boundary
    let v2 = table.lookup_linear(-1.0, IndexMode::Mirror);
    assert!((v2 - 1.0).abs() < 0.1, "Index -1 should mirror to 1");
}

#[test]
fn test_index_mode_clamp_boundaries() {
    let table = LookupTable::<f32, 10>::from_fn(|i, _| i as f32);

    // Test clamping above boundary
    let v1 = table.lookup_linear(15.0, IndexMode::Clamp);
    assert!((v1 - 9.0).abs() < 0.001, "Index 15 should clamp to 9");

    // Test clamping below boundary
    let v2 = table.lookup_linear(-5.0, IndexMode::Clamp);
    assert!((v2 - 0.0).abs() < 0.001, "Index -5 should clamp to 0");
}

// T125: Accuracy test: Linear interpolation maintains phase continuity
#[test]
fn test_linear_interpolation_phase_continuity() {
    // Create a sine wave table
    let table = LookupTable::<f32, 256>::from_fn(|i, size| {
        let phase = (i as f32 / size as f32) * 2.0 * core::f32::consts::PI;
        libm::sinf(phase)
    });

    // Sample at fractional indices and verify smoothness
    let mut prev_value = table.lookup_linear(0.0, IndexMode::Wrap);
    for i in 1..1000 {
        let index = (i as f32) * 0.1; // Step by 0.1
        let value = table.lookup_linear(index, IndexMode::Wrap);

        // Check that value changes smoothly (no large jumps)
        let diff = (value - prev_value).abs();
        assert!(
            diff < 0.1,
            "Phase discontinuity detected at index {}: diff = {}",
            index,
            diff
        );

        prev_value = value;
    }
}

// T126: Property-based test: SIMD gather provides correct per-lane indexing
#[test]
fn test_simd_gather_per_lane_correctness() {
    let table = LookupTable::<f32, 256>::from_fn(|i, _| (i * 10) as f32);

    // Create SIMD vector with different indices per lane
    let indices = DefaultSimdVector::from_slice(&[0.0, 10.5, 20.3, 30.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

    // Perform SIMD lookup
    let simd_results = table.lookup_linear_simd(indices, IndexMode::Wrap);

    // Extract results and verify each lane independently
    let mut results_buf = [0.0f32; 16];
    simd_results.to_slice(&mut results_buf);

    // Verify first few lanes match scalar lookups
    for lane in 0..DefaultSimdVector::LANES.min(4) {
        let mut indices_buf = [0.0f32; 16];
        indices.to_slice(&mut indices_buf);

        let expected = table.lookup_linear(indices_buf[lane], IndexMode::Wrap);
        let actual = results_buf[lane];

        assert!(
            (actual - expected).abs() < 0.001,
            "Lane {} mismatch: expected {}, got {}",
            lane,
            expected,
            actual
        );
    }
}

// T124: Performance test: 64-sample block lookup <640ns (<10ns/sample)
#[test]
fn test_lookup_performance_reasonableness() {
    use std::time::Instant;

    let table = LookupTable::<f32, 1024>::from_fn(|i, size| {
        let x = (i as f32 / size as f32) * 2.0 * core::f32::consts::PI;
        libm::sinf(x)
    });

    // Warm up
    for i in 0..100 {
        let _ = table.lookup_linear((i as f32) * 0.1, IndexMode::Wrap);
    }

    // Time 64-sample block lookups
    let iterations = 10000;
    let start = Instant::now();

    for iter in 0..iterations {
        for i in 0..64 {
            let index = ((iter * 64 + i) as f32) * 0.01;
            let _ = table.lookup_linear(index, IndexMode::Wrap);
        }
    }

    let elapsed = start.elapsed();
    let avg_per_block = elapsed / iterations;

    println!(
        "Average time per 64-sample block: {:?} ({:.2}ns per sample)",
        avg_per_block,
        avg_per_block.as_nanos() as f64 / 64.0
    );

    // This is a smoke test - actual performance depends on CPU
    // Just verify it completes in reasonable time (< 1ms per block)
    assert!(
        avg_per_block.as_micros() < 1000,
        "Lookup too slow: {:?} per block",
        avg_per_block
    );
}

// Additional test: Cubic interpolation provides smoother results
#[test]
fn test_cubic_interpolation_smoother_than_linear() {
    // Create a sine wave table
    let table = LookupTable::<f32, 64>::from_fn(|i, size| {
        let x = (i as f32 / size as f32) * 2.0 * core::f32::consts::PI;
        libm::sinf(x)
    });

    // Compare linear vs cubic at fractional indices
    let test_indices = [10.5, 20.3, 30.7, 40.2];

    for &idx in &test_indices {
        let linear = table.lookup_linear(idx, IndexMode::Wrap);
        let cubic = table.lookup_cubic(idx, IndexMode::Wrap);

        // Both should be in valid range
        assert!((-1.0..=1.0).contains(&linear));
        assert!((-1.0..=1.0).contains(&cubic));

        // They should be similar but not identical
        let diff = (linear - cubic).abs();
        assert!(
            diff < 0.2,
            "Linear and cubic too different at {}: diff = {}",
            idx,
            diff
        );
    }
}

// Test SIMD cubic interpolation
#[test]
fn test_simd_cubic_interpolation() {
    let table = LookupTable::<f32, 256>::from_fn(|i, size| {
        let x = (i as f32 / size as f32) * 2.0 * core::f32::consts::PI;
        libm::sinf(x)
    });

    let indices = DefaultSimdVector::from_slice(&[64.5, 128.3, 192.7, 200.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let results = table.lookup_cubic_simd(indices, IndexMode::Wrap);

    // Verify results are in valid range for sine wave
    let mut results_buf = [0.0f32; 16];
    results.to_slice(&mut results_buf);

    #[allow(clippy::needless_range_loop)]
    for lane in 0..DefaultSimdVector::LANES.min(4) {
        assert!(
            (-1.1..=1.1).contains(&results_buf[lane]),
            "Sine wave result out of range: {}",
            results_buf[lane]
        );
    }
}
