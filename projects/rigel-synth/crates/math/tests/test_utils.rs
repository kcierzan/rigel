//! Test utilities for rigel-math (T069, T070, T071)
//!
//! Provides reference implementations, proptest strategies, and assertion helpers
//! for validating SIMD operations across all backends.

use proptest::prelude::*;
use rigel_math::{DefaultSimdVector, SimdVector};

/// Relative error tolerance for floating-point comparisons
pub const RELATIVE_ERROR_TOLERANCE: f32 = 1e-5;

/// Absolute error tolerance for floating-point comparisons
pub const ABSOLUTE_ERROR_TOLERANCE: f32 = 1e-6;

// ============================================================================
// Reference Implementations using libm (T069)
// ============================================================================

/// Reference implementation of addition
#[inline]
pub fn ref_add(a: f32, b: f32) -> f32 {
    a + b
}

/// Reference implementation of subtraction
#[inline]
pub fn ref_sub(a: f32, b: f32) -> f32 {
    a - b
}

/// Reference implementation of multiplication
#[inline]
pub fn ref_mul(a: f32, b: f32) -> f32 {
    a * b
}

/// Reference implementation of division
#[inline]
pub fn ref_div(a: f32, b: f32) -> f32 {
    a / b
}

/// Reference implementation of FMA
#[inline]
pub fn ref_fma(a: f32, b: f32, c: f32) -> f32 {
    libm::fmaf(a, b, c)
}

/// Reference implementation of min
#[inline]
pub fn ref_min(a: f32, b: f32) -> f32 {
    libm::fminf(a, b)
}

/// Reference implementation of max
#[inline]
pub fn ref_max(a: f32, b: f32) -> f32 {
    libm::fmaxf(a, b)
}

/// Reference implementation of clamp
#[inline]
pub fn ref_clamp(value: f32, min: f32, max: f32) -> f32 {
    ref_min(ref_max(value, min), max)
}

/// Reference implementation of abs
#[inline]
pub fn ref_abs(x: f32) -> f32 {
    libm::fabsf(x)
}

/// Reference implementation of sqrt
#[inline]
pub fn ref_sqrt(x: f32) -> f32 {
    libm::sqrtf(x)
}

// ============================================================================
// Proptest Strategies (T070)
// ============================================================================

/// Strategy for generating normal floating-point values
///
/// Generates values in the range [-1000.0, 1000.0] excluding denormals,
/// infinities, and NaN.
pub fn normal_f32() -> impl Strategy<Value = f32> {
    (-1000.0f32..=1000.0f32).prop_filter("not denormal or special", |&x| x.is_normal() || x == 0.0)
}

/// Strategy for generating small normal floating-point values
///
/// Generates values in the range [-1.0, 1.0] useful for testing precision.
pub fn small_normal_f32() -> impl Strategy<Value = f32> {
    (-1.0f32..=1.0f32).prop_filter("not denormal or special", |&x| x.is_normal() || x == 0.0)
}

/// Strategy for generating positive normal floating-point values
///
/// Useful for operations that require positive inputs (sqrt, log, etc.)
pub fn positive_f32() -> impl Strategy<Value = f32> {
    (f32::MIN_POSITIVE..=1000.0f32).prop_filter("positive normal", |&x| x.is_normal())
}

/// Strategy for generating denormal (subnormal) floating-point values
///
/// Generates values in the denormal range to test denormal handling.
pub fn denormal_f32() -> impl Strategy<Value = f32> {
    prop_oneof![
        Just(1e-40f32),
        Just(-1e-40f32),
        Just(f32::MIN_POSITIVE / 2.0),
        Just(-f32::MIN_POSITIVE / 2.0),
    ]
}

/// Strategy for generating edge case floating-point values
///
/// Includes zero, denormals, min/max normal values, and special values.
pub fn edge_case_f32() -> impl Strategy<Value = f32> {
    prop_oneof![
        Just(0.0f32),
        Just(-0.0f32),
        Just(f32::MIN_POSITIVE),
        Just(-f32::MIN_POSITIVE),
        Just(f32::MAX),
        Just(-f32::MAX),
        denormal_f32(),
    ]
}

/// Strategy for generating any floating-point value including special values
///
/// Includes NaN and infinity for robust testing.
pub fn any_f32() -> impl Strategy<Value = f32> {
    prop_oneof![
        normal_f32(),
        edge_case_f32(),
        Just(f32::NAN),
        Just(f32::INFINITY),
        Just(f32::NEG_INFINITY),
    ]
}

/// Strategy for generating a pair of normal floats
pub fn normal_f32_pair() -> impl Strategy<Value = (f32, f32)> {
    (normal_f32(), normal_f32())
}

/// Strategy for generating a triple of normal floats
pub fn normal_f32_triple() -> impl Strategy<Value = (f32, f32, f32)> {
    (normal_f32(), normal_f32(), normal_f32())
}

// ============================================================================
// Assertion Helpers (T071)
// ============================================================================

/// Assert that two floating-point values are approximately equal
///
/// Uses both relative and absolute error tolerance to handle different
/// magnitude ranges.
pub fn assert_approx_eq(actual: f32, expected: f32, context: &str) {
    if expected.is_nan() {
        assert!(actual.is_nan(), "{}: expected NaN, got {}", context, actual);
        return;
    }

    if expected.is_infinite() {
        assert_eq!(
            actual.is_infinite(),
            true,
            "{}: expected infinite, got {}",
            context,
            actual
        );
        assert_eq!(
            actual.signum(),
            expected.signum(),
            "{}: expected {} infinity, got {} infinity",
            context,
            expected.signum(),
            actual.signum()
        );
        return;
    }

    let abs_diff = (actual - expected).abs();
    let abs_expected = expected.abs();
    let relative_error = if abs_expected > 0.0 {
        abs_diff / abs_expected
    } else {
        abs_diff
    };

    assert!(
        abs_diff <= ABSOLUTE_ERROR_TOLERANCE || relative_error <= RELATIVE_ERROR_TOLERANCE,
        "{}: values not approximately equal. Expected: {}, Actual: {}, Abs diff: {:.2e}, Rel error: {:.2e}",
        context,
        expected,
        actual,
        abs_diff,
        relative_error
    );
}

/// Assert that a SIMD vector matches expected scalar values
///
/// Extracts each lane and compares against expected array using approximate equality.
pub fn assert_vector_approx_eq(vector: DefaultSimdVector, expected: &[f32], context: &str) {
    assert_eq!(
        expected.len(),
        DefaultSimdVector::LANES,
        "{}: expected array length must match vector lanes",
        context
    );

    let mut actual = vec![0.0f32; DefaultSimdVector::LANES];
    vector.to_slice(&mut actual);

    for (i, (&actual_val, &expected_val)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_approx_eq(
            actual_val,
            expected_val,
            &format!("{} (lane {})", context, i),
        );
    }
}

/// Assert that a SIMD operation produces results consistent with scalar reference
///
/// This is the primary helper for backend consistency testing (T071).
/// It applies a SIMD operation and scalar reference operation to the same inputs
/// and verifies they produce approximately equal results.
pub fn assert_backend_consistency<F, R>(input: &[f32], simd_op: F, scalar_op: R, context: &str)
where
    F: Fn(DefaultSimdVector) -> DefaultSimdVector,
    R: Fn(f32) -> f32,
{
    assert_eq!(
        input.len(),
        DefaultSimdVector::LANES,
        "{}: input length must match vector lanes",
        context
    );

    // Apply SIMD operation
    let input_vec = DefaultSimdVector::from_slice(input);
    let result_vec = simd_op(input_vec);

    // Apply scalar operation to each element
    let expected: Vec<f32> = input.iter().map(|&x| scalar_op(x)).collect();

    // Compare results
    assert_vector_approx_eq(result_vec, &expected, context);
}

/// Assert that a binary SIMD operation produces results consistent with scalar reference
pub fn assert_binary_backend_consistency<F, R>(
    input_a: &[f32],
    input_b: &[f32],
    simd_op: F,
    scalar_op: R,
    context: &str,
) where
    F: Fn(DefaultSimdVector, DefaultSimdVector) -> DefaultSimdVector,
    R: Fn(f32, f32) -> f32,
{
    assert_eq!(
        input_a.len(),
        DefaultSimdVector::LANES,
        "{}: input_a length must match vector lanes",
        context
    );
    assert_eq!(
        input_b.len(),
        DefaultSimdVector::LANES,
        "{}: input_b length must match vector lanes",
        context
    );

    // Apply SIMD operation
    let vec_a = DefaultSimdVector::from_slice(input_a);
    let vec_b = DefaultSimdVector::from_slice(input_b);
    let result_vec = simd_op(vec_a, vec_b);

    // Apply scalar operation to each element pair
    let expected: Vec<f32> = input_a
        .iter()
        .zip(input_b.iter())
        .map(|(&a, &b)| scalar_op(a, b))
        .collect();

    // Compare results
    assert_vector_approx_eq(result_vec, &expected, context);
}

/// Assert that a ternary SIMD operation produces results consistent with scalar reference
pub fn assert_ternary_backend_consistency<F, R>(
    input_a: &[f32],
    input_b: &[f32],
    input_c: &[f32],
    simd_op: F,
    scalar_op: R,
    context: &str,
) where
    F: Fn(DefaultSimdVector, DefaultSimdVector, DefaultSimdVector) -> DefaultSimdVector,
    R: Fn(f32, f32, f32) -> f32,
{
    assert_eq!(
        input_a.len(),
        DefaultSimdVector::LANES,
        "{}: input_a length must match vector lanes",
        context
    );
    assert_eq!(
        input_b.len(),
        DefaultSimdVector::LANES,
        "{}: input_b length must match vector lanes",
        context
    );
    assert_eq!(
        input_c.len(),
        DefaultSimdVector::LANES,
        "{}: input_c length must match vector lanes",
        context
    );

    // Apply SIMD operation
    let vec_a = DefaultSimdVector::from_slice(input_a);
    let vec_b = DefaultSimdVector::from_slice(input_b);
    let vec_c = DefaultSimdVector::from_slice(input_c);
    let result_vec = simd_op(vec_a, vec_b, vec_c);

    // Apply scalar operation to each element triple
    let expected: Vec<f32> = input_a
        .iter()
        .zip(input_b.iter())
        .zip(input_c.iter())
        .map(|((&a, &b), &c)| scalar_op(a, b, c))
        .collect();

    // Compare results
    assert_vector_approx_eq(result_vec, &expected, context);
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Generate an array of normal float values for testing
///
/// Useful for creating test inputs without proptest overhead.
pub fn generate_normal_array<const N: usize>(seed: u64) -> [f32; N] {
    let mut values = [0.0f32; N];
    for (i, val) in values.iter_mut().enumerate() {
        // Simple LCG for deterministic pseudo-random values
        let x = (seed.wrapping_mul(1103515245).wrapping_add(i as u64)) as u32;
        let normalized = (x as f32) / (u32::MAX as f32);
        *val = (normalized * 2000.0) - 1000.0; // Range [-1000, 1000]
    }
    values
}
