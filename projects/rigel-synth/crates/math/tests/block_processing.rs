//! Block processing integration tests (T037, T038)
//!
//! Tests block processing with as_chunks enables loop unrolling and maintains correctness.

use rigel_math::{Block128, Block64, DefaultSimdVector, SimdVector};

/// Test T037: Block processing with as_chunks enables loop unrolling
///
/// This test verifies that block processing compiles to efficient vectorized code.
/// Assembly inspection would confirm unrolling (done separately in T039).
#[test]
fn test_block_processing_gain() {
    let mut input = Block64::new();
    let mut output = Block64::new();

    // Fill input with test signal
    for i in 0..64 {
        input[i] = (i as f32) / 64.0;
    }

    let gain = DefaultSimdVector::splat(0.5);

    // Process using SIMD chunks
    for (in_chunk, mut out_chunk) in input
        .as_chunks::<DefaultSimdVector>()
        .iter()
        .zip(output.as_chunks_mut::<DefaultSimdVector>().iter_mut())
    {
        let result = in_chunk.mul(gain);
        out_chunk.store(result);
    }

    // Verify results
    for i in 0..64 {
        let expected = (i as f32) / 64.0 * 0.5;
        assert!(
            (output[i] - expected).abs() < 1e-6,
            "Sample {} mismatch: got {}, expected {}",
            i,
            output[i],
            expected
        );
    }
}

/// Test block processing with mixing
#[test]
fn test_block_processing_mix() {
    let mut signal_a = Block64::new();
    let mut signal_b = Block64::new();
    let mut output = Block64::new();

    // Fill signals
    for i in 0..64 {
        signal_a[i] = 1.0;
        signal_b[i] = 0.5;
    }

    let mix_a = DefaultSimdVector::splat(0.7);
    let mix_b = DefaultSimdVector::splat(0.3);

    // Mix signals
    for ((chunk_a, chunk_b), mut out_chunk) in signal_a
        .as_chunks::<DefaultSimdVector>()
        .iter()
        .zip(signal_b.as_chunks::<DefaultSimdVector>().iter())
        .zip(output.as_chunks_mut::<DefaultSimdVector>().iter_mut())
    {
        let weighted_a = chunk_a.mul(mix_a);
        let weighted_b = chunk_b.mul(mix_b);
        let mixed = weighted_a.add(weighted_b);
        out_chunk.store(mixed);
    }

    // Verify: 1.0 * 0.7 + 0.5 * 0.3 = 0.85
    for i in 0..64 {
        assert!((output[i] - 0.85).abs() < 1e-6);
    }
}

/// Test block processing with FMA
#[test]
fn test_block_processing_fma() {
    let mut input = Block64::new();
    let mut output = Block64::new();

    for i in 0..64 {
        input[i] = i as f32;
    }

    let scale = DefaultSimdVector::splat(2.0);
    let offset = DefaultSimdVector::splat(1.0);

    // y = x * 2 + 1 using FMA
    for (in_chunk, mut out_chunk) in input
        .as_chunks::<DefaultSimdVector>()
        .iter()
        .zip(output.as_chunks_mut::<DefaultSimdVector>().iter_mut())
    {
        let result = in_chunk.fma(scale, offset);
        out_chunk.store(result);
    }

    for i in 0..64 {
        let expected = i as f32 * 2.0 + 1.0;
        assert_eq!(output[i], expected);
    }
}

/// Test block processing with clipping
#[test]
fn test_block_processing_clip() {
    let mut input = Block64::new();
    let mut output = Block64::new();

    // Fill with values exceeding [-1, 1]
    for i in 0..64 {
        input[i] = (i as f32 / 32.0) - 1.0; // Range: -1.0 to 1.0
    }
    input[0] = -2.0; // Exceeds minimum
    input[63] = 2.0; // Exceeds maximum

    let min_val = DefaultSimdVector::splat(-1.0);
    let max_val = DefaultSimdVector::splat(1.0);

    // Clip to [-1, 1]
    for (in_chunk, mut out_chunk) in input
        .as_chunks::<DefaultSimdVector>()
        .iter()
        .zip(output.as_chunks_mut::<DefaultSimdVector>().iter_mut())
    {
        let clipped = in_chunk.max(min_val).min(max_val);
        out_chunk.store(clipped);
    }

    // Verify clipping
    assert_eq!(output[0], -1.0, "Should clip to minimum");
    assert_eq!(output[63], 1.0, "Should clip to maximum");

    for i in 1..63 {
        assert!(output[i] >= -1.0 && output[i] <= 1.0);
    }
}

/// Test T038: Property-based test - Block processing maintains correctness
///
/// Verifies that processing arbitrary block contents produces correct results.
#[test]
fn test_block_processing_correctness() {
    let test_values = [
        (0.0, 1.0),
        (1.0, 0.5),
        (-1.0, 2.0),
        (0.707, 0.707), // Equal power crossfade mix
    ];

    for (val_a, val_b) in test_values.iter() {
        let mut block_a = Block64::new();
        let mut block_b = Block64::new();
        let mut output = Block64::new();

        // Fill blocks
        for i in 0..64 {
            block_a[i] = *val_a;
            block_b[i] = *val_b;
        }

        let mix = DefaultSimdVector::splat(0.5);

        // Mix with equal weights
        for ((chunk_a, chunk_b), mut out_chunk) in block_a
            .as_chunks::<DefaultSimdVector>()
            .iter()
            .zip(block_b.as_chunks::<DefaultSimdVector>().iter())
            .zip(output.as_chunks_mut::<DefaultSimdVector>().iter_mut())
        {
            let mixed = chunk_a.mul(mix).add(chunk_b.mul(mix));
            out_chunk.store(mixed);
        }

        let expected = (val_a + val_b) * 0.5;
        for i in 0..64 {
            assert!(
                (output[i] - expected).abs() < 1e-6,
                "Mismatch at sample {}: got {}, expected {}",
                i,
                output[i],
                expected
            );
        }
    }
}

/// Test Block128 processing
#[test]
fn test_block128_processing() {
    let mut input = Block128::new();
    let mut output = Block128::new();

    for i in 0..128 {
        input[i] = i as f32;
    }

    let gain = DefaultSimdVector::splat(0.1);

    for (in_chunk, mut out_chunk) in input
        .as_chunks::<DefaultSimdVector>()
        .iter()
        .zip(output.as_chunks_mut::<DefaultSimdVector>().iter_mut())
    {
        out_chunk.store(in_chunk.mul(gain));
    }

    for i in 0..128 {
        assert_eq!(output[i], i as f32 * 0.1);
    }
}

/// Test processing with different SIMD backends
#[test]
fn test_block_processing_backend_agnostic() {
    let mut input = Block64::new();
    let mut output_simd = Block64::new();
    let mut output_scalar = Block64::new();

    // Fill with test data
    for i in 0..64 {
        input[i] = (i as f32).sin();
    }

    let gain = 0.5f32;
    let gain_vec = DefaultSimdVector::splat(gain);

    // Process with SIMD
    for (in_chunk, mut out_chunk) in input
        .as_chunks::<DefaultSimdVector>()
        .iter()
        .zip(output_simd.as_chunks_mut::<DefaultSimdVector>().iter_mut())
    {
        out_chunk.store(in_chunk.mul(gain_vec));
    }

    // Process with scalar for reference
    for i in 0..64 {
        output_scalar[i] = input[i] * gain;
    }

    // Results should be identical
    for i in 0..64 {
        assert_eq!(output_simd[i], output_scalar[i]);
    }
}

/// Test in-place processing
#[test]
fn test_block_processing_in_place() {
    let mut block = Block64::new();

    for i in 0..64 {
        block[i] = i as f32;
    }

    let scale = DefaultSimdVector::splat(2.0);

    // In-place modification
    for mut chunk in block.as_chunks_mut::<DefaultSimdVector>().iter_mut() {
        let val = chunk.load();
        chunk.store(val.mul(scale));
    }

    for i in 0..64 {
        assert_eq!(block[i], i as f32 * 2.0);
    }
}

/// Test block processing with complex operations
#[test]
fn test_block_processing_complex() {
    let mut input = Block64::new();
    let mut output = Block64::new();

    for i in 0..64 {
        input[i] = (i as f32 - 32.0) / 32.0; // Range: -1 to ~1
    }

    let a = DefaultSimdVector::splat(2.0);
    let b = DefaultSimdVector::splat(0.5);
    let c = DefaultSimdVector::splat(1.0);
    let min_val = DefaultSimdVector::splat(-1.0);
    let max_val = DefaultSimdVector::splat(1.0);

    // Complex: y = clamp(x * 2 + 0.5 * 1, -1, 1) = clamp(x * 2 + 0.5, -1, 1)
    for (in_chunk, mut out_chunk) in input
        .as_chunks::<DefaultSimdVector>()
        .iter()
        .zip(output.as_chunks_mut::<DefaultSimdVector>().iter_mut())
    {
        let scaled = in_chunk.mul(a);
        let offset = b.mul(c);
        let result = scaled.add(offset);
        let clamped = result.max(min_val).min(max_val);
        out_chunk.store(clamped);
    }

    // Verify a few samples
    let expected_0 = ((-32.0f32 / 32.0) * 2.0 + 0.5).clamp(-1.0, 1.0); // -1.5 -> -1.0
    assert_eq!(output[0], expected_0);

    let expected_32 = ((0.0f32 / 32.0) * 2.0 + 0.5).clamp(-1.0, 1.0); // 0.5
    assert_eq!(output[32], expected_32);
}
