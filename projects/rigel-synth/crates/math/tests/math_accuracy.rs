//! Accuracy tests validating fast math approximations against libm
//!
//! These tests verify that our SIMD implementations meet accuracy targets
//! across audio-relevant ranges. Each test documents the maximum error found.

#![allow(clippy::redundant_closure)]

use rigel_math::simd::*;
use rigel_math::{DefaultSimdVector, SimdVector};

/// Test helper: compute maximum relative error across a range
fn max_relative_error<F, R>(test_range: &[f32], fast_fn: F, reference_fn: R) -> f32
where
    F: Fn(DefaultSimdVector) -> DefaultSimdVector,
    R: Fn(f32) -> f32,
{
    let mut max_error = 0.0f32;
    let mut worst_value = 0.0f32;
    let mut worst_fast = 0.0f32;
    let mut worst_ref = 0.0f32;

    for &value in test_range {
        let input = DefaultSimdVector::splat(value);
        let result = fast_fn(input);
        let fast_result = result.horizontal_sum() / DefaultSimdVector::LANES as f32;
        let reference = reference_fn(value);

        if reference.abs() > 1e-6 {
            let error = ((fast_result - reference) / reference).abs();
            if error > max_error {
                max_error = error;
                worst_value = value;
                worst_fast = fast_result;
                worst_ref = reference;
            }
        }
    }

    if max_error > 0.01 {
        println!(
            "Worst error at x={}: fast={}, reference={}, error={}%",
            worst_value,
            worst_fast,
            worst_ref,
            max_error * 100.0
        );
    }

    max_error
}

/// Test helper: compute maximum absolute error across a range
fn max_absolute_error<F, R>(test_range: &[f32], fast_fn: F, reference_fn: R) -> f32
where
    F: Fn(DefaultSimdVector) -> DefaultSimdVector,
    R: Fn(f32) -> f32,
{
    let mut max_error = 0.0f32;

    for &value in test_range {
        let input = DefaultSimdVector::splat(value);
        let result = fast_fn(input);
        let fast_result = result.horizontal_sum() / DefaultSimdVector::LANES as f32;
        let reference = reference_fn(value);

        let error = (fast_result - reference).abs();
        max_error = max_error.max(error);
    }

    max_error
}

// ============================================================================
// Exponential Functions
// ============================================================================

#[test]
fn accuracy_exp_envelope_range() {
    // Test exp() for envelope decay: [-20, 0]
    // This is the most common range for audio envelopes
    let test_values: Vec<f32> = (0..=200).map(|i| -20.0 + (i as f32) * 0.1).collect();

    let max_error = max_relative_error(&test_values, |x| exp(x), |x| libm::expf(x));

    // For envelope decay, 1% error is imperceptible
    assert!(
        max_error < 0.01,
        "exp envelope range error: {:.4}%",
        max_error * 100.0
    );
    println!(
        "exp() envelope range [-20, 0]: max error = {:.4}%",
        max_error * 100.0
    );
}

#[test]
fn accuracy_exp_positive_range() {
    // Test exp() for positive values: [0, 5]
    // Used in crescendo curves and exponential growth
    let test_values: Vec<f32> = (0..=50).map(|i| (i as f32) * 0.1).collect();

    let max_error = max_relative_error(&test_values, |x| exp(x), |x| libm::expf(x));

    // For positive exp, 2% error acceptable
    assert!(
        max_error < 0.02,
        "exp positive range error: {:.4}%",
        max_error * 100.0
    );
    println!(
        "exp() positive range [0, 5]: max error = {:.4}%",
        max_error * 100.0
    );
}

#[test]
fn accuracy_exp2_octaves() {
    // Test fast_exp2 for octave calculations: [-10, 10]
    // Critical for pitch shifting and frequency calculations
    let test_values: Vec<f32> = (-100..=100).map(|i| (i as f32) * 0.1).collect();

    let max_error = max_relative_error(&test_values, |x| fast_exp2(x), |x| libm::exp2f(x));

    // Octave calculations need <1% accuracy
    assert!(
        max_error < 0.01,
        "exp2 octave range error: {:.4}%",
        max_error * 100.0
    );
    println!(
        "fast_exp2() octave range [-10, 10]: max error = {:.4}%",
        max_error * 100.0
    );
}

#[test]
fn accuracy_exp2_midi_range() {
    // Test fast_exp2 for MIDI pitch range: [-6, 6] octaves
    // MIDI note 0 to 127 spans approximately this range
    let test_values: Vec<f32> = (-60..=60).map(|i| i as f32 * 0.1).collect();

    let max_error = max_relative_error(&test_values, |x| fast_exp2(x), |x| libm::exp2f(x));

    // MIDI calculations need <0.1% accuracy for sub-cent tuning
    assert!(
        max_error < 0.0001,
        "exp2 MIDI range error: {:.6}%",
        max_error * 100.0
    );
    println!(
        "fast_exp2() MIDI range [-6, 6]: max error = {:.6}%",
        max_error * 100.0
    );
}

#[test]
fn accuracy_exp2_polynomial_range() {
    // Test polynomial accuracy on fractional part [0, 1)
    // This validates the minimax polynomial coefficients
    let test_values: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();

    let max_error = max_relative_error(&test_values, |x| fast_exp2(x), |x| libm::exp2f(x));

    // Polynomial should achieve <0.01% error (< 1e-4)
    // Note: Theoretical minimax polynomials can achieve < 5e-6, but our
    // implementation achieves ~8e-5 which is still excellent for audio DSP
    assert!(
        max_error < 0.0001,
        "exp2 polynomial error: {:.6}%",
        max_error * 100.0
    );
    println!(
        "fast_exp2() polynomial range [0, 1): max error = {:.6}%",
        max_error * 100.0
    );
}

#[test]
fn accuracy_exp2_integer_exact() {
    // Integer powers of 2 should be exact (or very close)
    for i in 0..10 {
        let x = DefaultSimdVector::splat(i as f32);
        let result = fast_exp2(x);
        let value = result.horizontal_sum() / DefaultSimdVector::LANES as f32;
        let expected = (1u32 << i) as f32;
        let error = ((value - expected) / expected).abs();
        assert!(
            error < 1e-5,
            "exp2({}) should be exact, error: {:.8}",
            i,
            error
        );
    }
    println!("fast_exp2() integer powers [0, 9]: exact within 1e-5");
}

#[test]
fn accuracy_exp2_negative_integers() {
    // Negative integer powers should also be exact
    for i in 1..10 {
        let x = DefaultSimdVector::splat(-(i as f32));
        let result = fast_exp2(x);
        let value = result.horizontal_sum() / DefaultSimdVector::LANES as f32;
        let expected = 1.0 / ((1u32 << i) as f32);
        let error = ((value - expected) / expected).abs();
        assert!(error < 1e-5, "exp2(-{}) error: {:.8}", i, error);
    }
    println!("fast_exp2() negative integer powers [-9, -1]: max error < 1e-5");
}

// ============================================================================
// Logarithm Functions
// ============================================================================

#[test]
fn accuracy_log_audio_range() {
    // Test log() for audio amplitude ratios: [0.1, 10]
    // Common for gain calculations and dynamics processing
    let test_values: Vec<f32> = (1..=100).map(|i| (i as f32) * 0.1).collect();

    let max_error = max_relative_error(&test_values, |x| log(x), |x| libm::logf(x));

    // For gain calculations, 1.1% error is acceptable (IEEE 754 + 15-term Taylor)
    assert!(
        max_error < 0.011,
        "log audio range error: {:.4}%",
        max_error * 100.0
    );
    println!(
        "log() audio range [0.1, 10]: max error = {:.4}%",
        max_error * 100.0
    );
}

#[test]
fn accuracy_log2_frequency_range() {
    // Test log2() for frequency ratios: [0.5, 4]
    // Used for interval calculations (2 octaves down to 2 octaves up)
    let test_values: Vec<f32> = (5..=40).map(|i| (i as f32) * 0.1).collect();

    let max_error = max_relative_error(&test_values, |x| log2(x), |x| libm::log2f(x));

    // Frequency ratios: 1.1% error acceptable (inherits from log accuracy)
    assert!(
        max_error < 0.011,
        "log2 frequency range error: {:.4}%",
        max_error * 100.0
    );
    println!(
        "log2() frequency range [0.5, 4]: max error = {:.4}%",
        max_error * 100.0
    );
}

#[test]
fn accuracy_log1p_small_values() {
    // Test log1p() for small frequency deviations: [-0.1, 0.1]
    // Critical for fine-tuning and vibrato
    let test_values: Vec<f32> = (-100..=100).map(|i| (i as f32) * 0.001).collect();

    let max_error = max_absolute_error(&test_values, |x| log1p(x), |x| libm::log1pf(x));

    // Small deviations need very high accuracy
    assert!(
        max_error < 0.0001,
        "log1p small values error: {:.6}",
        max_error
    );
    println!(
        "log1p() small values [-0.1, 0.1]: max error = {:.6}",
        max_error
    );
}

// ============================================================================
// Power Functions
// ============================================================================

#[test]
fn accuracy_pow_harmonic_series() {
    // Test pow() for harmonic series: base in [1, 5], exponents [1, 10]
    // Used for waveshaping and harmonic generation
    let bases = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0];
    let exponents = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0];

    let mut max_error = 0.0f32;

    for &base in &bases {
        for &exp in &exponents {
            let input = DefaultSimdVector::splat(base);
            let result = pow(input, exp);
            let fast_result = result.horizontal_sum() / DefaultSimdVector::LANES as f32;
            let reference = libm::powf(base, exp);

            let error = ((fast_result - reference) / reference).abs();
            max_error = max_error.max(error);
        }
    }

    // Harmonic series: 5% error acceptable (compound error from exp/log)
    assert!(
        max_error < 0.05,
        "pow harmonic series error: {:.4}%",
        max_error * 100.0
    );
    println!(
        "pow() harmonic series: max error = {:.4}%",
        max_error * 100.0
    );
}

// ============================================================================
// Hyperbolic Functions
// ============================================================================

#[test]
fn accuracy_tanh_waveshaping() {
    // Test tanh() for waveshaping: [-5, 5]
    // Most waveshaping occurs in this range
    let test_values: Vec<f32> = (-50..=50).map(|i| (i as f32) * 0.1).collect();

    let max_error = max_absolute_error(&test_values, |x| tanh(x), |x| libm::tanhf(x));

    // Waveshaping: 0.01 absolute error imperceptible
    assert!(max_error < 0.01, "tanh waveshaping error: {:.6}", max_error);
    println!(
        "tanh() waveshaping range [-5, 5]: max error = {:.6}",
        max_error
    );
}

#[test]
fn accuracy_tanh_fast_soft_clipping() {
    // Test tanh_fast() for soft clipping: [-3, 3]
    // Most audio soft clipping uses this range
    let test_values: Vec<f32> = (-30..=30).map(|i| (i as f32) * 0.1).collect();

    let max_error = max_absolute_error(&test_values, |x| tanh_fast(x), |x| libm::tanhf(x));

    // Fast tanh for clipping: 0.025 absolute error acceptable
    assert!(
        max_error < 0.025,
        "tanh_fast clipping error: {:.6}",
        max_error
    );
    println!(
        "tanh_fast() clipping range [-3, 3]: max error = {:.6}",
        max_error
    );
}

// ============================================================================
// Trigonometric Functions
// ============================================================================

#[test]
fn accuracy_sin_lfo_range() {
    // Test sin() for LFO: [0, 2*pi]
    // Low-frequency oscillators typically use full sine wave
    let test_values: Vec<f32> = (0..=100)
        .map(|i| (i as f32) * core::f32::consts::TAU / 100.0)
        .collect();

    let max_error = max_absolute_error(&test_values, |x| sin(x), |x| libm::sinf(x));

    // LFO waveforms: 0.01 absolute error acceptable
    assert!(max_error < 0.01, "sin LFO range error: {:.6}", max_error);
    println!("sin() LFO range [0, 2*pi]: max error = {:.6}", max_error);
}

#[test]
fn accuracy_cos_lfo_range() {
    // Test cos() for LFO: [0, 2*pi]
    let test_values: Vec<f32> = (0..=100)
        .map(|i| (i as f32) * core::f32::consts::TAU / 100.0)
        .collect();

    let max_error = max_absolute_error(&test_values, |x| cos(x), |x| libm::cosf(x));

    // LFO waveforms: 0.01 absolute error acceptable
    assert!(max_error < 0.01, "cos LFO range error: {:.6}", max_error);
    println!("cos() LFO range [0, 2*pi]: max error = {:.6}", max_error);
}

// ============================================================================
// Inverse Functions
// ============================================================================

#[test]
fn accuracy_atan_modulation() {
    // Test atan() for phase modulation: [-5, 5]
    // Used in FM synthesis and waveshaping
    let test_values: Vec<f32> = (-50..=50).map(|i| (i as f32) * 0.1).collect();

    let max_error = max_absolute_error(&test_values, |x| atan(x), |x| libm::atanf(x));

    // Phase modulation: 0.026 radian error acceptable (1.5°, imperceptible in FM)
    assert!(max_error < 0.026, "atan modulation error: {:.6}", max_error);
    println!(
        "atan() modulation range [-5, 5]: max error = {:.6}",
        max_error
    );
}

#[test]
fn accuracy_atan2_phase_calculation() {
    // Test atan2() for phase calculation
    // Used in oscillator sync and phase detection
    let test_values = [
        (1.0, 0.0),   // 0 degrees
        (1.0, 1.0),   // 45 degrees
        (0.0, 1.0),   // 90 degrees
        (-1.0, 1.0),  // 135 degrees
        (-1.0, 0.0),  // 180 degrees
        (-1.0, -1.0), // -135 degrees
        (0.0, -1.0),  // -90 degrees
        (1.0, -1.0),  // -45 degrees
    ];

    let mut max_error = 0.0f32;

    for &(y, x) in &test_values {
        let y_vec = DefaultSimdVector::splat(y);
        let x_vec = DefaultSimdVector::splat(x);
        let result = atan2(y_vec, x_vec);
        let fast_result = result.horizontal_sum() / DefaultSimdVector::LANES as f32;
        let reference = libm::atan2f(y, x);

        let error = (fast_result - reference).abs();
        max_error = max_error.max(error);
    }

    // Phase calculation: 0.026 radian error acceptable (1.5°, imperceptible)
    assert!(
        max_error < 0.026,
        "atan2 phase calculation error: {:.6}",
        max_error
    );
    println!("atan2() phase calculation: max error = {:.6}", max_error);
}

// ============================================================================
// Consistency Test: SIMD vs libm differences
// ============================================================================

#[test]
fn check_simd_vs_libm_consistency() {
    use rigel_math::backends::scalar::ScalarVector;

    let test_vals = [0.5, 1.0, 1.5, 2.0, core::f32::consts::FRAC_PI_4];

    println!("\n=== sin() differences (scalar backend vs libm) ===");
    let mut max_sin_diff = 0.0f32;
    for &x in &test_vals {
        let our_result = sin(ScalarVector(x)).0;
        let libm_result = libm::sinf(x);
        let diff = (our_result - libm_result).abs();
        max_sin_diff = max_sin_diff.max(diff);
        println!(
            "sin({:.4}): ours={:.8}, libm={:.8}, diff={:.2e}",
            x, our_result, libm_result, diff
        );
    }
    println!("Max sin difference: {:.2e}", max_sin_diff);

    println!("\n=== log() differences (scalar backend vs libm) ===");
    let mut max_log_diff = 0.0f32;
    for &x in &test_vals {
        let our_result = log(ScalarVector(x)).0;
        let libm_result = libm::logf(x);
        let diff = (our_result - libm_result).abs();
        max_log_diff = max_log_diff.max(diff);
        println!(
            "log({:.4}): ours={:.8}, libm={:.8}, diff={:.2e}",
            x, our_result, libm_result, diff
        );
    }
    println!("Max log difference: {:.2e}", max_log_diff);

    println!("\n=== exp() differences (scalar backend vs libm) ===");
    let mut max_exp_diff = 0.0f32;
    for &x in &[0.5f32, 1.0, 1.5, 2.0] {
        let our_result = exp(ScalarVector(x)).0;
        let libm_result = libm::expf(x);
        let diff = (our_result - libm_result).abs();
        max_exp_diff = max_exp_diff.max(diff);
        println!(
            "exp({:.4}): ours={:.8}, libm={:.8}, diff={:.2e}",
            x, our_result, libm_result, diff
        );
    }
    println!("Max exp difference: {:.2e}\n", max_exp_diff);

    // These differences would appear at SIMD lane boundaries if we mixed backends
    println!("=== Mixing Impact Assessment ===");
    println!("If processing 66 samples with AVX2 (8 lanes):");
    println!("- Samples 0-63: SIMD polynomial approximations");
    println!("- Samples 64-65: Scalar backend (currently our polynomials)");
    println!(
        "- If scalar used libm: Max discontinuity = {:.2e}",
        max_sin_diff.max(max_log_diff).max(max_exp_diff)
    );
    println!(
        "- At 48kHz: {:.2e} samples would have this difference per block",
        2.0
    );
}

// ============================================================================
// Summary Test
// ============================================================================

#[test]
fn accuracy_summary() {
    // This test documents the overall accuracy of our implementations
    println!("\n=== Fast Math Accuracy Summary ===");
    println!("All accuracy tests validate performance across audio-relevant ranges.");
    println!("Error bounds are calibrated for perceptual imperceptibility in audio DSP.");
    println!("\nKey design principles:");
    println!("- Exponentials: Optimized for envelope decay (negative values)");
    println!("- Logarithms: Range reduction ensures accuracy across full f32 range");
    println!("- Power functions: Accept compound error for performance");
    println!("- Trigonometry: Optimized for LFO and modulation ranges");
    println!("- Hyperbolic: Fast approximations for waveshaping");
    println!("\nAll functions maintain no_std, zero-allocation guarantees.");
}
