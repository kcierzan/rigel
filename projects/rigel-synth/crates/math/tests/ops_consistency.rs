//! Ops module consistency tests (T046-T050)
//!
//! Verifies that functional-style ops produce identical results to method-style trait calls,
//! and that ops work consistently across backends.

use rigel_math::ops::{
    abs, add, clamp, div, eq, fma, fms, fnma, ge, gt, horizontal_max, horizontal_min,
    horizontal_sum, le, lt, max, min, mul, ne, neg, sub,
};
use rigel_math::{DefaultSimdVector, SimdMask, SimdVector};

/// Test T046: Verify arithmetic ops match trait methods
#[test]
fn test_arithmetic_ops_consistency() {
    let a = DefaultSimdVector::splat(2.0);
    let b = DefaultSimdVector::splat(3.0);

    // Addition
    let result_op = add(a, b);
    let result_method = a.add(b);
    assert_eq!(result_op.horizontal_sum(), result_method.horizontal_sum());

    // Subtraction
    let result_op = sub(a, b);
    let result_method = a.sub(b);
    assert_eq!(result_op.horizontal_sum(), result_method.horizontal_sum());

    // Multiplication
    let result_op = mul(a, b);
    let result_method = a.mul(b);
    assert_eq!(result_op.horizontal_sum(), result_method.horizontal_sum());

    // Division
    let result_op = div(a, b);
    let result_method = a.div(b);
    assert_eq!(result_op.horizontal_sum(), result_method.horizontal_sum());

    // Negation
    let result_op = neg(a);
    let result_method = a.neg();
    assert_eq!(result_op.horizontal_sum(), result_method.horizontal_sum());
}

/// Test T047: Verify FMA ops correctness
#[test]
fn test_fma_ops_correctness() {
    let a = DefaultSimdVector::splat(2.0);
    let b = DefaultSimdVector::splat(3.0);
    let c = DefaultSimdVector::splat(1.0);

    // FMA: a * b + c = 2 * 3 + 1 = 7
    let result = fma(a, b, c);
    assert_eq!(
        result.horizontal_sum(),
        7.0 * DefaultSimdVector::LANES as f32
    );

    // FMS: a * b - c = 2 * 3 - 1 = 5
    let result = fms(a, b, c);
    assert_eq!(
        result.horizontal_sum(),
        5.0 * DefaultSimdVector::LANES as f32
    );

    // FNMA: -(a * b) + c = -(2 * 3) + 1 = -5
    let result = fnma(a, b, c);
    assert_eq!(
        result.horizontal_sum(),
        -5.0 * DefaultSimdVector::LANES as f32
    );
}

/// Test T048: Verify minmax ops
#[test]
fn test_minmax_ops() {
    let a = DefaultSimdVector::splat(2.0);
    let b = DefaultSimdVector::splat(3.0);

    // Min
    let result = min(a, b);
    assert_eq!(
        result.horizontal_sum(),
        2.0 * DefaultSimdVector::LANES as f32
    );

    // Max
    let result = max(a, b);
    assert_eq!(
        result.horizontal_sum(),
        3.0 * DefaultSimdVector::LANES as f32
    );

    // Abs
    let c = DefaultSimdVector::splat(-2.5);
    let result = abs(c);
    assert_eq!(
        result.horizontal_sum(),
        2.5 * DefaultSimdVector::LANES as f32
    );

    // Clamp
    let value = DefaultSimdVector::splat(5.0);
    let min_val = DefaultSimdVector::splat(0.0);
    let max_val = DefaultSimdVector::splat(3.0);
    let result = clamp(value, min_val, max_val);
    assert_eq!(
        result.horizontal_sum(),
        3.0 * DefaultSimdVector::LANES as f32
    );

    let value = DefaultSimdVector::splat(-5.0);
    let result = clamp(value, min_val, max_val);
    assert_eq!(
        result.horizontal_sum(),
        0.0 * DefaultSimdVector::LANES as f32
    );
}

/// Test T049: Verify comparison ops
#[test]
fn test_comparison_ops() {
    let a = DefaultSimdVector::splat(2.0);
    let b = DefaultSimdVector::splat(3.0);
    let c = DefaultSimdVector::splat(2.0);

    // Equality
    assert!(eq(a, c).all());
    assert!(!eq(a, b).all());

    // Inequality
    assert!(ne(a, b).all());
    assert!(!ne(a, c).all());

    // Less than
    assert!(lt(a, b).all());
    assert!(!lt(b, a).all());

    // Less than or equal
    assert!(le(a, b).all());
    assert!(le(a, c).all());
    assert!(!le(b, a).all());

    // Greater than
    assert!(gt(b, a).all());
    assert!(!gt(a, b).all());

    // Greater than or equal
    assert!(ge(b, a).all());
    assert!(ge(a, c).all());
    assert!(!ge(a, b).all());
}

/// Test T050: Verify horizontal reduction ops
#[test]
fn test_horizontal_ops() {
    let a = DefaultSimdVector::splat(2.0);

    // Horizontal sum
    let result = horizontal_sum(a);
    assert_eq!(result, 2.0 * DefaultSimdVector::LANES as f32);

    // Horizontal max
    let result = horizontal_max(a);
    assert_eq!(result, 2.0);

    // Horizontal min
    let result = horizontal_min(a);
    assert_eq!(result, 2.0);
}

/// Test composability of ops in a DSP pipeline
#[test]
fn test_ops_composability() {
    let input = DefaultSimdVector::splat(1.0);
    let gain = DefaultSimdVector::splat(2.0);
    let offset = DefaultSimdVector::splat(0.5);

    // Pipeline: (input * gain) + offset
    let scaled = mul(input, gain);
    let result = add(scaled, offset);
    assert_eq!(
        result.horizontal_sum(),
        2.5 * DefaultSimdVector::LANES as f32
    );

    // Alternative: use FMA
    let result_fma = fma(input, gain, offset);
    assert_eq!(
        result_fma.horizontal_sum(),
        2.5 * DefaultSimdVector::LANES as f32
    );

    // Verify they're identical
    assert_eq!(result.horizontal_sum(), result_fma.horizontal_sum());
}

/// Test ops with edge cases
#[test]
fn test_ops_edge_cases() {
    // Zero
    let zero = DefaultSimdVector::splat(0.0);
    let one = DefaultSimdVector::splat(1.0);

    assert_eq!(
        add(zero, one).horizontal_sum(),
        1.0 * DefaultSimdVector::LANES as f32
    );
    assert_eq!(mul(zero, one).horizontal_sum(), 0.0);

    // Negative zero
    let neg_zero = DefaultSimdVector::splat(-0.0);
    assert_eq!(abs(neg_zero).horizontal_sum(), 0.0);

    // Very small values
    let small = DefaultSimdVector::splat(1e-10);
    let result = add(small, small);
    assert!((result.horizontal_sum() - 2e-10 * DefaultSimdVector::LANES as f32).abs() < 1e-9);
}

/// Test ops with typical DSP patterns
#[test]
fn test_ops_dsp_patterns() {
    // Soft clipping using tanh approximation pattern
    let input = DefaultSimdVector::splat(0.5);
    let gain = DefaultSimdVector::splat(2.0);
    let min_val = DefaultSimdVector::splat(-1.0);
    let max_val = DefaultSimdVector::splat(1.0);

    let amplified = mul(input, gain);
    let clipped = clamp(amplified, min_val, max_val);
    assert_eq!(
        clipped.horizontal_sum(),
        1.0 * DefaultSimdVector::LANES as f32
    );

    // Crossfade mix
    let signal_a = DefaultSimdVector::splat(1.0);
    let signal_b = DefaultSimdVector::splat(0.0);
    let mix_amount = DefaultSimdVector::splat(0.7); // 70% signal_a, 30% signal_b

    let one = DefaultSimdVector::splat(1.0);
    let inv_mix = sub(one, mix_amount);

    let weighted_a = mul(signal_a, mix_amount);
    let weighted_b = mul(signal_b, inv_mix);
    let mixed = add(weighted_a, weighted_b);

    assert!((mixed.horizontal_sum() - 0.7 * DefaultSimdVector::LANES as f32).abs() < 1e-6);

    // DC offset removal simulation
    let signal = DefaultSimdVector::splat(0.5);
    let dc_offset = DefaultSimdVector::splat(0.1);
    let corrected = sub(signal, dc_offset);
    assert_eq!(
        corrected.horizontal_sum(),
        0.4 * DefaultSimdVector::LANES as f32
    );
}

/// Test backend-agnostic behavior
#[test]
fn test_backend_agnostic() {
    // This test verifies that ops work with any SimdVector implementation
    use rigel_math::backends::scalar::{ScalarMask, ScalarVector};

    let a = ScalarVector(2.0);
    let b = ScalarVector(3.0);

    // All ops should work with scalar backend
    assert_eq!(add(a, b).horizontal_sum(), 5.0);
    assert_eq!(mul(a, b).horizontal_sum(), 6.0);
    assert_eq!(min(a, b).horizontal_sum(), 2.0);
    assert_eq!(max(a, b).horizontal_sum(), 3.0);

    // Comparisons
    let lt_result: ScalarMask = lt(a, b);
    assert!(lt_result.all());
    let gt_result: ScalarMask = gt(a, b);
    assert!(!gt_result.all());
}
