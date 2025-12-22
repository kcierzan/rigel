//! Comprehensive integration test for SIMD backend selection and dispatch
//!
//! This test verifies that:
//! 1. Backend selection works correctly (compile-time and runtime)
//! 2. SimdContext provides a unified API regardless of backend
//! 3. ops module functions work through the abstraction
//! 4. simd module functions work through the abstraction
//! 5. table module functions work through the abstraction

use rigel_simd_dispatch::table::{IndexMode, LookupTable};
use rigel_simd_dispatch::{ops, simd, Block64, DefaultSimdVector, SimdVector};
use rigel_simd_dispatch::{ProcessParams, SimdContext};

#[test]
fn test_runtime_backend_selection() {
    // Initialize SimdContext - should select best backend automatically
    let ctx = SimdContext::new();
    let backend_name = ctx.backend_name();

    // Verify backend name is not empty
    assert!(!backend_name.is_empty());
    println!("Backend selected: {}", backend_name);
    println!("DefaultSimdVector::LANES: {}", DefaultSimdVector::LANES);

    // Compile-time AVX2 selection (CI runs: cargo test --features avx2)
    #[cfg(all(
        target_arch = "x86_64",
        feature = "avx2",
        not(feature = "avx512"),
        not(feature = "runtime-dispatch")
    ))]
    {
        assert_eq!(
            backend_name, "avx2",
            "Expected avx2 backend with --features avx2, got: {}",
            backend_name
        );
        assert_eq!(
            DefaultSimdVector::LANES,
            8,
            "AVX2 should have 8 lanes, got: {}",
            DefaultSimdVector::LANES
        );
    }

    // Compile-time AVX-512 selection
    #[cfg(all(
        target_arch = "x86_64",
        feature = "avx512",
        not(feature = "runtime-dispatch")
    ))]
    {
        assert_eq!(
            backend_name, "avx512",
            "Expected avx512 backend with --features avx512, got: {}",
            backend_name
        );
        assert_eq!(
            DefaultSimdVector::LANES,
            16,
            "AVX-512 should have 16 lanes, got: {}",
            DefaultSimdVector::LANES
        );
    }

    // Runtime dispatch on x86_64 (when runtime-dispatch feature is enabled)
    #[cfg(all(target_arch = "x86_64", feature = "runtime-dispatch"))]
    {
        println!("Runtime dispatch selected: {}", backend_name);
        assert!(
            backend_name == "scalar" || backend_name == "avx2" || backend_name == "avx512",
            "Expected scalar, avx2, or avx512, got: {}",
            backend_name
        );
    }

    // On aarch64 with NEON feature, should be NEON
    #[cfg(all(
        target_arch = "aarch64",
        feature = "neon",
        not(feature = "force-scalar")
    ))]
    {
        assert_eq!(
            backend_name, "neon",
            "Expected neon backend on aarch64, got: {}",
            backend_name
        );
        assert_eq!(
            DefaultSimdVector::LANES,
            4,
            "NEON should have 4 lanes, got: {}",
            DefaultSimdVector::LANES
        );
    }

    // Scalar fallback (no SIMD features enabled)
    #[cfg(all(
        not(feature = "avx2"),
        not(feature = "avx512"),
        not(feature = "neon"),
        not(feature = "runtime-dispatch")
    ))]
    {
        assert_eq!(
            backend_name, "scalar",
            "Expected scalar backend without SIMD features, got: {}",
            backend_name
        );
        assert_eq!(
            DefaultSimdVector::LANES,
            1,
            "Scalar should have 1 lane, got: {}",
            DefaultSimdVector::LANES
        );
    }
}

#[test]
fn test_simd_context_apply_gain() {
    let ctx = SimdContext::new();
    let mut input = Block64::new();
    let mut output = Block64::new();

    // Fill input with test data
    for i in 0..64 {
        input[i] = i as f32;
    }

    // Apply gain using SimdContext
    ctx.apply_gain(input.as_slice(), output.as_slice_mut(), 0.5);

    // Verify output
    for i in 0..64 {
        assert_eq!(output[i], (i as f32) * 0.5, "Failed at index {}", i);
    }

    println!(
        "✓ SimdContext::apply_gain works with {} backend",
        ctx.backend_name()
    );
}

#[test]
fn test_simd_context_process_block() {
    let ctx = SimdContext::new();
    let mut input = Block64::new();
    let mut output = Block64::new();

    // Fill input
    for i in 0..64 {
        input[i] = i as f32;
    }

    // Apply gain of 2.0 using process_block
    let params = ProcessParams {
        gain: 2.0,
        frequency: 440.0,
        sample_rate: 44100.0,
    };
    ctx.process_block(input.as_slice(), output.as_slice_mut(), &params);

    // Verify
    for i in 0..64 {
        assert_eq!(output[i], (i as f32) * 2.0, "Failed at index {}", i);
    }

    println!(
        "✓ SimdContext::process_block works with {} backend",
        ctx.backend_name()
    );
}

#[test]
fn test_ops_module_functions() {
    let ctx = SimdContext::new();

    // Test add, sub, mul, div
    let a = DefaultSimdVector::splat(10.0);
    let b = DefaultSimdVector::splat(2.0);

    let sum = ops::add(a, b);
    let diff = ops::sub(a, b);
    let product = ops::mul(a, b);
    let quotient = ops::div(a, b);

    // Extract values for verification (size 16 for AVX-512 compatibility)
    let mut sum_buf = [0.0f32; 16];
    let mut diff_buf = [0.0f32; 16];
    let mut prod_buf = [0.0f32; 16];
    let mut quot_buf = [0.0f32; 16];

    sum.to_slice(&mut sum_buf[..DefaultSimdVector::LANES]);
    diff.to_slice(&mut diff_buf[..DefaultSimdVector::LANES]);
    product.to_slice(&mut prod_buf[..DefaultSimdVector::LANES]);
    quotient.to_slice(&mut quot_buf[..DefaultSimdVector::LANES]);

    // Verify results
    for (i, (((&sum_val, &diff_val), &prod_val), &quot_val)) in sum_buf[..DefaultSimdVector::LANES]
        .iter()
        .zip(&diff_buf[..DefaultSimdVector::LANES])
        .zip(&prod_buf[..DefaultSimdVector::LANES])
        .zip(&quot_buf[..DefaultSimdVector::LANES])
        .enumerate()
    {
        assert_eq!(sum_val, 12.0, "add failed at lane {}", i);
        assert_eq!(diff_val, 8.0, "sub failed at lane {}", i);
        assert_eq!(prod_val, 20.0, "mul failed at lane {}", i);
        assert_eq!(quot_val, 5.0, "div failed at lane {}", i);
    }

    // Test fma (fused multiply-add)
    let c = DefaultSimdVector::splat(1.0);
    let fma_result = ops::fma(a, b, c); // a * b + c = 10 * 2 + 1 = 21
    let mut fma_buf = [0.0f32; 16];
    fma_result.to_slice(&mut fma_buf[..DefaultSimdVector::LANES]);

    for (i, &val) in fma_buf[..DefaultSimdVector::LANES].iter().enumerate() {
        assert_eq!(val, 21.0, "fma failed at lane {}", i);
    }

    // Test min, max, abs
    let neg = DefaultSimdVector::splat(-5.0);
    let pos = DefaultSimdVector::splat(3.0);

    let min_result = ops::min(neg, pos);
    let max_result = ops::max(neg, pos);
    let abs_result = ops::abs(neg);

    let mut min_buf = [0.0f32; 16];
    let mut max_buf = [0.0f32; 16];
    let mut abs_buf = [0.0f32; 16];

    min_result.to_slice(&mut min_buf[..DefaultSimdVector::LANES]);
    max_result.to_slice(&mut max_buf[..DefaultSimdVector::LANES]);
    abs_result.to_slice(&mut abs_buf[..DefaultSimdVector::LANES]);

    for (i, ((&min_val, &max_val), &abs_val)) in min_buf[..DefaultSimdVector::LANES]
        .iter()
        .zip(&max_buf[..DefaultSimdVector::LANES])
        .zip(&abs_buf[..DefaultSimdVector::LANES])
        .enumerate()
    {
        assert_eq!(min_val, -5.0, "min failed at lane {}", i);
        assert_eq!(max_val, 3.0, "max failed at lane {}", i);
        assert_eq!(abs_val, 5.0, "abs failed at lane {}", i);
    }

    println!(
        "✓ ops module (add, sub, mul, div, fma, min, max, abs) works with {} backend",
        ctx.backend_name()
    );
}

#[test]
fn test_math_module_functions() {
    let ctx = SimdContext::new();

    // Test sqrt (allow tolerance for fast approximation)
    let x = DefaultSimdVector::splat(16.0);
    let sqrt_result = simd::sqrt(x);
    let mut sqrt_buf = [0.0f32; 16];
    sqrt_result.to_slice(&mut sqrt_buf[..DefaultSimdVector::LANES]);

    for (i, &val) in sqrt_buf[..DefaultSimdVector::LANES].iter().enumerate() {
        assert!(
            (val - 4.0).abs() < 0.2,
            "sqrt(16) should be ~4.0, got {} at lane {}",
            val,
            i
        );
    }

    // Test tanh (soft clipping) - should be in range [-1, 1]
    let values = DefaultSimdVector::splat(0.5);
    let tanh_result = simd::tanh(values);
    let mut tanh_buf = [0.0f32; 16];
    tanh_result.to_slice(&mut tanh_buf[..DefaultSimdVector::LANES]);

    for (i, &val) in tanh_buf[..DefaultSimdVector::LANES].iter().enumerate() {
        assert!(
            val > 0.4 && val < 0.6,
            "tanh(0.5) should be ~0.46, got {} at lane {}",
            val,
            i
        );
    }

    // Test exp
    let zero = DefaultSimdVector::splat(0.0);
    let exp_result = simd::exp(zero);
    let mut exp_buf = [0.0f32; 16];
    exp_result.to_slice(&mut exp_buf[..DefaultSimdVector::LANES]);

    for (i, &val) in exp_buf[..DefaultSimdVector::LANES].iter().enumerate() {
        assert!(
            (val - 1.0).abs() < 0.01,
            "exp(0) should be ~1.0, got {} at lane {}",
            val,
            i
        );
    }

    println!(
        "✓ simd module (sqrt, tanh, exp) works with {} backend",
        ctx.backend_name()
    );
}

#[test]
fn test_table_module_interpolation() {
    let ctx = SimdContext::new();

    // Create a simple wavetable: linear ramp from 0.0 to 1.0
    const TABLE_SIZE: usize = 64;
    let table = LookupTable::<f32, TABLE_SIZE>::from_fn(|i, size| i as f32 / (size - 1) as f32);

    // Test scalar linear interpolation at various positions
    let positions = [0.0, 15.75, 31.5, 47.25, 63.0];
    let expected = [0.0, 0.25, 0.5, 0.75, 1.0];

    for (&pos, &exp) in positions.iter().zip(expected.iter()) {
        let result = table.lookup_linear(pos, IndexMode::Clamp);
        assert!(
            (result - exp).abs() < 0.02,
            "linear lookup at pos {} should be ~{}, got {}",
            pos,
            exp,
            result
        );
    }

    // Test SIMD cubic interpolation
    let indices = DefaultSimdVector::splat(31.5);
    let result = table.lookup_cubic_simd(indices, IndexMode::Clamp);

    let mut result_buf = [0.0f32; 16];
    result.to_slice(&mut result_buf[..DefaultSimdVector::LANES]);

    for (i, &val) in result_buf[..DefaultSimdVector::LANES].iter().enumerate() {
        assert!(
            (val - 0.5).abs() < 0.02,
            "SIMD cubic lookup should be ~0.5, got {} at lane {}",
            val,
            i
        );
    }

    println!(
        "✓ table module (LookupTable, linear & cubic SIMD) works with {} backend",
        ctx.backend_name()
    );
}

#[test]
fn test_complete_dsp_pipeline() {
    // This test demonstrates a complete DSP processing pipeline using all modules
    let ctx = SimdContext::new();

    println!("\n=== Complete DSP Pipeline Test ===");
    println!("Backend: {}", ctx.backend_name());
    println!("SIMD Lanes: {}", DefaultSimdVector::LANES);

    let mut input = Block64::new();
    let mut output = Block64::new();

    // Initialize input with a simple waveform
    for i in 0..64 {
        input[i] = (i as f32 / 64.0) * 2.0 - 1.0; // Range: [-1, 1]
    }

    // Process: apply gain, soft clipping, and final normalization
    for (in_vec, mut out_chunk) in input
        .as_chunks::<DefaultSimdVector>()
        .iter()
        .zip(output.as_chunks_mut::<DefaultSimdVector>().iter_mut())
    {
        // Step 1: Apply gain (ops module)
        let gain = DefaultSimdVector::splat(2.0);
        let gained = ops::mul(in_vec, gain);

        // Step 2: Soft clipping (simd module)
        let clipped = simd::tanh(gained);

        // Step 3: Final scaling (ops module)
        let scale = DefaultSimdVector::splat(0.8);
        let result = ops::mul(clipped, scale);

        out_chunk.store(result);
    }

    // Verify output is in reasonable range
    for i in 0..64 {
        assert!(
            output[i] >= -1.0 && output[i] <= 1.0,
            "Output at {} out of range: {}",
            i,
            output[i]
        );
    }

    println!("✓ Complete DSP pipeline (ops + simd) works correctly");
    println!(
        "✓ All SIMD modules verified working with {} backend",
        ctx.backend_name()
    );
    println!("=====================================\n");
}
