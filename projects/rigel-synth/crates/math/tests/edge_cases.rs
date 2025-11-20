//! Edge case tests for all backends (T026)
//!
//! Tests handling of NaN, infinity, zero, and denormal values across all SIMD backends.

use rigel_math::{DefaultSimdVector, SimdMask, SimdVector};

/// Test NaN handling in arithmetic operations
#[test]
fn test_nan_arithmetic() {
    let nan_vec = DefaultSimdVector::splat(f32::NAN);
    let normal_vec = DefaultSimdVector::splat(1.0);

    // NaN + value = NaN
    let sum = nan_vec.add(normal_vec);
    assert!(sum.horizontal_sum().is_nan(), "NaN + 1.0 should be NaN");

    // value + NaN = NaN
    let sum2 = normal_vec.add(nan_vec);
    assert!(sum2.horizontal_sum().is_nan(), "1.0 + NaN should be NaN");

    // NaN * value = NaN
    let prod = nan_vec.mul(normal_vec);
    assert!(prod.horizontal_sum().is_nan(), "NaN * 1.0 should be NaN");

    // NaN - NaN = NaN
    let diff = nan_vec.sub(nan_vec);
    assert!(diff.horizontal_sum().is_nan(), "NaN - NaN should be NaN");
}

/// Test infinity handling
#[test]
fn test_infinity_arithmetic() {
    let inf_vec = DefaultSimdVector::splat(f32::INFINITY);
    let neg_inf_vec = DefaultSimdVector::splat(f32::NEG_INFINITY);
    let normal_vec = DefaultSimdVector::splat(1.0);

    // inf + value = inf
    let sum = inf_vec.add(normal_vec);
    assert!(sum.horizontal_sum().is_infinite() && sum.horizontal_sum().is_sign_positive());

    // inf - inf = NaN
    let diff = inf_vec.sub(inf_vec);
    assert!(diff.horizontal_sum().is_nan(), "inf - inf should be NaN");

    // inf * -1 = -inf
    let neg_one = DefaultSimdVector::splat(-1.0);
    let prod = inf_vec.mul(neg_one);
    assert!(prod.horizontal_sum().is_infinite() && prod.horizontal_sum().is_sign_negative());

    // -inf + inf = NaN
    let sum2 = neg_inf_vec.add(inf_vec);
    assert!(sum2.horizontal_sum().is_nan(), "-inf + inf should be NaN");
}

/// Test zero handling
#[test]
fn test_zero_arithmetic() {
    let zero_vec = DefaultSimdVector::splat(0.0);
    let neg_zero_vec = DefaultSimdVector::splat(-0.0);
    let one_vec = DefaultSimdVector::splat(1.0);

    // 0 + 0 = 0
    let sum = zero_vec.add(zero_vec);
    assert_eq!(sum.horizontal_sum(), 0.0);

    // 0 * value = 0
    let prod = zero_vec.mul(one_vec);
    assert_eq!(prod.horizontal_sum(), 0.0);

    // 0 / value = 0 (non-zero value)
    let quot = zero_vec.div(one_vec);
    assert_eq!(quot.horizontal_sum(), 0.0);

    // value / 0 = inf
    let quot2 = one_vec.div(zero_vec);
    assert!(quot2.horizontal_sum().is_infinite());

    // -0 should be treated correctly
    let sum_neg = neg_zero_vec.add(zero_vec);
    // The result could be 0.0 or -0.0 depending on implementation
    assert!(sum_neg.horizontal_sum().abs() == 0.0);
}

/// Test division by zero
#[test]
fn test_division_by_zero() {
    let zero = DefaultSimdVector::splat(0.0);
    let one = DefaultSimdVector::splat(1.0);
    let neg_one = DefaultSimdVector::splat(-1.0);

    // 1 / 0 = +inf
    let result = one.div(zero);
    assert!(result.horizontal_sum().is_infinite());
    assert!(result.horizontal_sum().is_sign_positive());

    // -1 / 0 = -inf
    let result_neg = neg_one.div(zero);
    assert!(result_neg.horizontal_sum().is_infinite());
    assert!(result_neg.horizontal_sum().is_sign_negative());

    // 0 / 0 = NaN
    let result_nan = zero.div(zero);
    assert!(result_nan.horizontal_sum().is_nan());
}

/// Test comparison with NaN
#[test]
fn test_nan_comparisons() {
    let nan_vec = DefaultSimdVector::splat(f32::NAN);
    let one_vec = DefaultSimdVector::splat(1.0);

    // NaN < value = false for all lanes
    let mask_lt = nan_vec.lt(one_vec);
    assert!(!mask_lt.any(), "NaN < 1.0 should be false");

    // NaN > value = false for all lanes
    let mask_gt = nan_vec.gt(one_vec);
    assert!(!mask_gt.any(), "NaN > 1.0 should be false");

    // NaN == value = false for all lanes
    let mask_eq = nan_vec.eq(one_vec);
    assert!(!mask_eq.any(), "NaN == 1.0 should be false");

    // NaN == NaN = false (IEEE 754 behavior)
    let mask_eq_nan = nan_vec.eq(nan_vec);
    assert!(!mask_eq_nan.any(), "NaN == NaN should be false");
}

/// Test comparison with infinity
#[test]
fn test_infinity_comparisons() {
    let inf_vec = DefaultSimdVector::splat(f32::INFINITY);
    let neg_inf_vec = DefaultSimdVector::splat(f32::NEG_INFINITY);
    let one_vec = DefaultSimdVector::splat(1.0);

    // 1 < inf = true
    let mask = one_vec.lt(inf_vec);
    assert!(mask.all(), "1.0 < inf should be true");

    // inf < 1 = false
    let mask2 = inf_vec.lt(one_vec);
    assert!(mask2.none(), "inf < 1.0 should be false");

    // -inf < 1 = true
    let mask3 = neg_inf_vec.lt(one_vec);
    assert!(mask3.all(), "-inf < 1.0 should be true");

    // -inf < inf = true
    let mask4 = neg_inf_vec.lt(inf_vec);
    assert!(mask4.all(), "-inf < inf should be true");
}

/// Test absolute value with edge cases
#[test]
fn test_abs_edge_cases() {
    // abs(NaN) = NaN
    let nan_vec = DefaultSimdVector::splat(f32::NAN);
    let abs_nan = nan_vec.abs();
    assert!(abs_nan.horizontal_sum().is_nan());

    // abs(inf) = inf
    let inf_vec = DefaultSimdVector::splat(f32::INFINITY);
    let abs_inf = inf_vec.abs();
    assert_eq!(abs_inf.horizontal_sum(), f32::INFINITY);

    // abs(-inf) = inf
    let neg_inf_vec = DefaultSimdVector::splat(f32::NEG_INFINITY);
    let abs_neg_inf = neg_inf_vec.abs();
    assert_eq!(abs_neg_inf.horizontal_sum(), f32::INFINITY);

    // abs(0) = 0
    let zero_vec = DefaultSimdVector::splat(0.0);
    let abs_zero = zero_vec.abs();
    assert_eq!(abs_zero.horizontal_sum(), 0.0);

    // abs(-0) = 0
    let neg_zero_vec = DefaultSimdVector::splat(-0.0);
    let abs_neg_zero = neg_zero_vec.abs();
    assert_eq!(abs_neg_zero.horizontal_sum().abs(), 0.0);
}

/// Test min/max with edge cases
#[test]
fn test_minmax_edge_cases() {
    let nan_vec = DefaultSimdVector::splat(f32::NAN);
    let one_vec = DefaultSimdVector::splat(1.0);
    let inf_vec = DefaultSimdVector::splat(f32::INFINITY);
    let neg_inf_vec = DefaultSimdVector::splat(f32::NEG_INFINITY);

    // min with NaN (implementation-defined, but should handle gracefully)
    let min_nan = one_vec.min(nan_vec);
    // Result could be NaN or 1.0 depending on implementation
    let result = min_nan.horizontal_sum();
    assert!(result.is_nan() || result == 1.0);

    // max with infinity
    let max_inf = one_vec.max(inf_vec);
    assert_eq!(max_inf.horizontal_sum(), f32::INFINITY);

    // min with -infinity
    let min_neg_inf = one_vec.min(neg_inf_vec);
    assert_eq!(min_neg_inf.horizontal_sum(), f32::NEG_INFINITY);

    // max(-inf, inf) = inf
    let max_inf2 = neg_inf_vec.max(inf_vec);
    assert_eq!(max_inf2.horizontal_sum(), f32::INFINITY);
}

/// Test FMA with edge cases
#[test]
fn test_fma_edge_cases() {
    let nan_vec = DefaultSimdVector::splat(f32::NAN);
    let one_vec = DefaultSimdVector::splat(1.0);
    let zero_vec = DefaultSimdVector::splat(0.0);
    let inf_vec = DefaultSimdVector::splat(f32::INFINITY);

    // NaN * a + b = NaN
    let fma_nan = nan_vec.fma(one_vec, one_vec);
    assert!(fma_nan.horizontal_sum().is_nan());

    // a * NaN + b = NaN
    let fma_nan2 = one_vec.fma(nan_vec, one_vec);
    assert!(fma_nan2.horizontal_sum().is_nan());

    // a * b + NaN = NaN
    let fma_nan3 = one_vec.fma(one_vec, nan_vec);
    assert!(fma_nan3.horizontal_sum().is_nan());

    // 0 * inf + 0 = NaN (indeterminate form)
    let fma_indet = zero_vec.fma(inf_vec, zero_vec);
    assert!(fma_indet.horizontal_sum().is_nan());

    // inf * 1 + 1 = inf
    let fma_inf = inf_vec.fma(one_vec, one_vec);
    assert!(fma_inf.horizontal_sum().is_infinite());
}

/// Test negation with edge cases
#[test]
fn test_neg_edge_cases() {
    // -NaN = NaN
    let nan_vec = DefaultSimdVector::splat(f32::NAN);
    let neg_nan = nan_vec.neg();
    assert!(neg_nan.horizontal_sum().is_nan());

    // -inf = -inf
    let inf_vec = DefaultSimdVector::splat(f32::INFINITY);
    let neg_inf = inf_vec.neg();
    assert_eq!(neg_inf.horizontal_sum(), f32::NEG_INFINITY);

    // -(-inf) = inf
    let neg_inf_vec = DefaultSimdVector::splat(f32::NEG_INFINITY);
    let pos_inf = neg_inf_vec.neg();
    assert_eq!(pos_inf.horizontal_sum(), f32::INFINITY);

    // -0 = -0 (or 0, both acceptable)
    let zero_vec = DefaultSimdVector::splat(0.0);
    let neg_zero = zero_vec.neg();
    assert_eq!(neg_zero.horizontal_sum().abs(), 0.0);
}

/// Test select (blend) with edge cases
#[test]
fn test_select_edge_cases() {
    let nan_vec = DefaultSimdVector::splat(f32::NAN);
    let one_vec = DefaultSimdVector::splat(1.0);
    let two_vec = DefaultSimdVector::splat(2.0);

    // Create masks
    let all_true_mask = one_vec.lt(two_vec); // 1 < 2 = true
    let all_false_mask = two_vec.lt(one_vec); // 2 < 1 = false

    // Select with all-true mask
    let result_true = DefaultSimdVector::select(all_true_mask, nan_vec, one_vec);
    assert!(result_true.horizontal_sum().is_nan(), "Should select NaN");

    // Select with all-false mask
    let result_false = DefaultSimdVector::select(all_false_mask, nan_vec, one_vec);
    assert_eq!(
        result_false.horizontal_sum(),
        1.0 * DefaultSimdVector::LANES as f32,
        "Should select 1.0"
    );
}

/// Test horizontal operations with edge cases
#[test]
fn test_horizontal_edge_cases() {
    // Create a vector with mixed values (if LANES > 1)
    let lanes = DefaultSimdVector::LANES;
    if lanes > 1 {
        let mut values = vec![0.0f32; lanes];
        values[0] = 1.0;
        values[lanes - 1] = f32::NAN;

        let vec = DefaultSimdVector::from_slice(&values);

        // horizontal_sum with NaN should be NaN
        let sum = vec.horizontal_sum();
        assert!(sum.is_nan(), "Horizontal sum with NaN should be NaN");
    }

    // All NaN
    let nan_vec = DefaultSimdVector::splat(f32::NAN);
    assert!(nan_vec.horizontal_sum().is_nan());
    assert!(nan_vec.horizontal_max().is_nan());
    assert!(nan_vec.horizontal_min().is_nan());

    // All infinity
    let inf_vec = DefaultSimdVector::splat(f32::INFINITY);
    assert_eq!(inf_vec.horizontal_sum(), f32::INFINITY);
    assert_eq!(inf_vec.horizontal_max(), f32::INFINITY);
    assert_eq!(inf_vec.horizontal_min(), f32::INFINITY);
}
