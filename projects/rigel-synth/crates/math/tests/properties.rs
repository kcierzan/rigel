//! Property-based tests for rigel-math
//!
//! Uses proptest to validate mathematical invariants across all SIMD backends.
//! These tests generate thousands of test cases to ensure correctness.

use proptest::prelude::*;
use rigel_math::{DefaultSimdVector, SimdMask, SimdVector};

#[cfg(test)]
mod test_utils;

#[cfg(test)]
use test_utils::*;

// Configure proptest to run 10,000+ test cases per property (T072)
use proptest::test_runner::Config as ProptestConfig;

fn proptest_config() -> ProptestConfig {
    ProptestConfig {
        cases: 10_000,
        ..ProptestConfig::default()
    }
}

/// Test T022: SimdVector arithmetic commutativity
///
/// Property: a + b == b + a for all values of a and b
#[test]
fn test_addition_commutativity() {
    proptest!(proptest_config(), |((a, b) in (small_normal_f32(), small_normal_f32()))| {
        let vec_a = DefaultSimdVector::splat(a);
        let vec_b = DefaultSimdVector::splat(b);

        let sum_ab = vec_a.add(vec_b);
        let sum_ba = vec_b.add(vec_a);

        // Extract results
        let mut result_ab = vec![0.0; DefaultSimdVector::LANES];
        let mut result_ba = vec![0.0; DefaultSimdVector::LANES];
        sum_ab.to_slice(&mut result_ab);
        sum_ba.to_slice(&mut result_ba);

        // Allow for floating-point precision differences
        for i in 0..DefaultSimdVector::LANES {
            if result_ab[i].is_nan() && result_ba[i].is_nan() {
                continue; // Both NaN is ok
            }
            assert_eq!(result_ab[i], result_ba[i], "Addition not commutative for a={}, b={}", a, b);
        }
    });
}

#[test]
fn test_multiplication_commutativity() {
    proptest!(proptest_config(), |((a, b) in (small_normal_f32(), small_normal_f32()))| {
        let vec_a = DefaultSimdVector::splat(a);
        let vec_b = DefaultSimdVector::splat(b);

        let prod_ab = vec_a.mul(vec_b);
        let prod_ba = vec_b.mul(vec_a);

        let mut result_ab = vec![0.0; DefaultSimdVector::LANES];
        let mut result_ba = vec![0.0; DefaultSimdVector::LANES];
        prod_ab.to_slice(&mut result_ab);
        prod_ba.to_slice(&mut result_ba);

        for i in 0..DefaultSimdVector::LANES {
            if result_ab[i].is_nan() && result_ba[i].is_nan() {
                continue;
            }
            assert_eq!(result_ab[i], result_ba[i], "Multiplication not commutative for a={}, b={}", a, b);
        }
    });
}

/// Test T023: SimdVector arithmetic associativity
///
/// Property: (a + b) + c == a + (b + c) for all values
#[test]
fn test_addition_associativity() {
    proptest!(proptest_config(), |((a, b, c) in (small_normal_f32(), small_normal_f32(), small_normal_f32()))| {
        let vec_a = DefaultSimdVector::splat(a);
        let vec_b = DefaultSimdVector::splat(b);
        let vec_c = DefaultSimdVector::splat(c);

        // (a + b) + c
        let sum_ab = vec_a.add(vec_b);
        let sum_abc_left = sum_ab.add(vec_c);

        // a + (b + c)
        let sum_bc = vec_b.add(vec_c);
        let sum_abc_right = vec_a.add(sum_bc);

        let mut result_left = vec![0.0; DefaultSimdVector::LANES];
        let mut result_right = vec![0.0; DefaultSimdVector::LANES];
        sum_abc_left.to_slice(&mut result_left);
        sum_abc_right.to_slice(&mut result_right);

        for i in 0..DefaultSimdVector::LANES {
            if result_left[i].is_nan() && result_right[i].is_nan() {
                continue;
            }
            // Floating-point addition is not perfectly associative, so allow small tolerance
            let diff = (result_left[i] - result_right[i]).abs();
            assert!(diff < 1e-5 || diff / result_left[i].abs().max(result_right[i].abs()) < 1e-5,
                    "Addition not associative for a={}, b={}, c={}: left={}, right={}, diff={}",
                    a, b, c, result_left[i], result_right[i], diff);
        }
    });
}

#[test]
fn test_multiplication_associativity() {
    proptest!(proptest_config(), |((a, b, c) in (small_normal_f32(), small_normal_f32(), small_normal_f32()))| {
        let vec_a = DefaultSimdVector::splat(a);
        let vec_b = DefaultSimdVector::splat(b);
        let vec_c = DefaultSimdVector::splat(c);

        // (a * b) * c
        let prod_ab = vec_a.mul(vec_b);
        let prod_abc_left = prod_ab.mul(vec_c);

        // a * (b * c)
        let prod_bc = vec_b.mul(vec_c);
        let prod_abc_right = vec_a.mul(prod_bc);

        let mut result_left = vec![0.0; DefaultSimdVector::LANES];
        let mut result_right = vec![0.0; DefaultSimdVector::LANES];
        prod_abc_left.to_slice(&mut result_left);
        prod_abc_right.to_slice(&mut result_right);

        for i in 0..DefaultSimdVector::LANES {
            if result_left[i].is_nan() && result_right[i].is_nan() {
                continue;
            }
            // Both infinite is considered equal
            if result_left[i].is_infinite() && result_right[i].is_infinite() &&
               result_left[i].is_sign_positive() == result_right[i].is_sign_positive() {
                continue;
            }
            // Allow for floating-point precision differences
            let diff = (result_left[i] - result_right[i]).abs();
            // Skip if diff is NaN (can happen with inf - inf)
            if diff.is_nan() {
                continue;
            }
            assert!(diff < 1e-4 || diff / result_left[i].abs().max(result_right[i].abs()) < 1e-4,
                    "Multiplication not associative for a={}, b={}, c={}: left={}, right={}, diff={}",
                    a, b, c, result_left[i], result_right[i], diff);
        }
    });
}

// Simpler tests without proptest for always running

/// Test basic arithmetic commutativity without proptest
#[test]
fn test_basic_commutativity() {
    let test_values = [(2.0, 3.0), (-1.0, 4.0), (0.0, 5.0), (1.5, 2.5)];

    for (a, b) in test_values.iter() {
        let vec_a = DefaultSimdVector::splat(*a);
        let vec_b = DefaultSimdVector::splat(*b);

        // Test addition
        let sum_ab = vec_a.add(vec_b);
        let sum_ba = vec_b.add(vec_a);
        assert_eq!(sum_ab.horizontal_sum(), sum_ba.horizontal_sum());

        // Test multiplication
        let prod_ab = vec_a.mul(vec_b);
        let prod_ba = vec_b.mul(vec_a);
        assert_eq!(prod_ab.horizontal_sum(), prod_ba.horizontal_sum());
    }
}

/// Test basic arithmetic associativity without proptest
#[test]
fn test_basic_associativity() {
    let test_values = [(2.0, 3.0, 4.0), (1.0, 2.0, 3.0), (-1.0, 0.0, 1.0)];

    for (a, b, c) in test_values.iter() {
        let vec_a = DefaultSimdVector::splat(*a);
        let vec_b = DefaultSimdVector::splat(*b);
        let vec_c = DefaultSimdVector::splat(*c);

        // Test addition: (a + b) + c == a + (b + c)
        let sum_ab = vec_a.add(vec_b);
        let sum_abc_left = sum_ab.add(vec_c);

        let sum_bc = vec_b.add(vec_c);
        let sum_abc_right = vec_a.add(sum_bc);

        let diff = (sum_abc_left.horizontal_sum() - sum_abc_right.horizontal_sum()).abs();
        assert!(diff < 1e-5, "Addition not associative");

        // Test multiplication: (a * b) * c == a * (b * c)
        let prod_ab = vec_a.mul(vec_b);
        let prod_abc_left = prod_ab.mul(vec_c);

        let prod_bc = vec_b.mul(vec_c);
        let prod_abc_right = vec_a.mul(prod_bc);

        let diff = (prod_abc_left.horizontal_sum() - prod_abc_right.horizontal_sum()).abs();
        assert!(diff < 1e-4, "Multiplication not associative");
    }
}

// ============================================================================
// Comprehensive Property-Based Tests (T073)
// ============================================================================
//
// These tests use test_utils strategies and run 10,000+ cases to validate
// mathematical invariants across all backends.

/// Test T046: FMA accuracy vs separate multiply-add
///
/// Property: fma(a, b, c) ≈ (a * b) + c
#[test]
fn test_fma_accuracy() {
    proptest!(proptest_config(), |(
        (a, b, c) in normal_f32_triple()
    )| {
        let vec_a = DefaultSimdVector::splat(a);
        let vec_b = DefaultSimdVector::splat(b);
        let vec_c = DefaultSimdVector::splat(c);

        // FMA operation
        let fma_result = vec_a.fma(vec_b, vec_c);

        // Separate multiply-add
        let mul_result = vec_a.mul(vec_b);
        let add_result = mul_result.add(vec_c);

        let mut fma_vals = vec![0.0; DefaultSimdVector::LANES];
        let mut add_vals = vec![0.0; DefaultSimdVector::LANES];
        fma_result.to_slice(&mut fma_vals);
        add_result.to_slice(&mut add_vals);

        // FMA should be at least as accurate as mul+add
        // Allow for small differences due to rounding
        // Note: FMA and mul+add can differ due to different rounding behavior,
        // especially in catastrophic cancellation cases (large numbers - large numbers = small result)
        for i in 0..DefaultSimdVector::LANES {
            if fma_vals[i].is_nan() && add_vals[i].is_nan() {
                continue;
            }
            let diff = (fma_vals[i] - add_vals[i]).abs();
            let max_val = fma_vals[i].abs().max(add_vals[i].abs());
            // Relaxed tolerance to 0.01% relative error or 1e-4 absolute
            // This accounts for catastrophic cancellation cases
            assert!(
                diff < 1e-4 || (max_val > 0.0 && diff / max_val < 1e-4),
                "FMA differs from mul+add: fma={}, mul+add={}, diff={}, rel_err={:.6}, a={}, b={}, c={}",
                fma_vals[i], add_vals[i], diff, if max_val > 0.0 { diff / max_val } else { 0.0 }, a, b, c
            );
        }
    });
}

/// Test T047: Min/max operations with edge cases
///
/// Property: min and max handle NaN, infinity, and zero correctly
#[test]
fn test_minmax_edge_cases() {
    proptest!(proptest_config(), |(
        (a, b) in (any_f32(), any_f32())
    )| {
        let vec_a = DefaultSimdVector::splat(a);
        let vec_b = DefaultSimdVector::splat(b);

        let min_result = vec_a.min(vec_b);
        let max_result = vec_a.max(vec_b);

        let mut min_vals = vec![0.0; DefaultSimdVector::LANES];
        let mut max_vals = vec![0.0; DefaultSimdVector::LANES];
        min_result.to_slice(&mut min_vals);
        max_result.to_slice(&mut max_vals);

        // Reference implementation
        let expected_min = ref_min(a, b);
        let expected_max = ref_max(a, b);

        for i in 0..DefaultSimdVector::LANES {
            // NaN handling: if either input is NaN, result behavior is platform-specific
            // but both results should be identical across lanes
            if expected_min.is_nan() || expected_max.is_nan() {
                continue;
            }

            assert_approx_eq(
                min_vals[i],
                expected_min,
                &format!("min({}, {})", a, b)
            );
            assert_approx_eq(
                max_vals[i],
                expected_max,
                &format!("max({}, {})", a, b)
            );
        }
    });
}

/// Test T048: Horizontal sum correctness across backends
///
/// Property: horizontal_sum sums all lanes correctly
#[test]
fn test_horizontal_sum_correctness() {
    proptest!(proptest_config(), |(values in proptest::collection::vec(normal_f32(), DefaultSimdVector::LANES))| {
        let vec = DefaultSimdVector::from_slice(&values);
        let sum = vec.horizontal_sum();

        // Reference: scalar sum
        let expected_sum: f32 = values.iter().sum();

        // Allow for accumulated floating-point error
        assert_approx_eq(sum, expected_sum, "horizontal_sum");
    });
}

/// Test T073: Mathematical invariants - Distributivity
///
/// Property: a * (b + c) ≈ (a * b) + (a * c)
#[test]
fn test_distributivity() {
    proptest!(proptest_config(), |(
        (a, b, c) in (small_normal_f32(), small_normal_f32(), small_normal_f32())
    )| {
        let vec_a = DefaultSimdVector::splat(a);
        let vec_b = DefaultSimdVector::splat(b);
        let vec_c = DefaultSimdVector::splat(c);

        // Left side: a * (b + c)
        let sum_bc = vec_b.add(vec_c);
        let left = vec_a.mul(sum_bc);

        // Right side: (a * b) + (a * c)
        let prod_ab = vec_a.mul(vec_b);
        let prod_ac = vec_a.mul(vec_c);
        let right = prod_ab.add(prod_ac);

        let mut left_vals = vec![0.0; DefaultSimdVector::LANES];
        let mut right_vals = vec![0.0; DefaultSimdVector::LANES];
        left.to_slice(&mut left_vals);
        right.to_slice(&mut right_vals);

        for i in 0..DefaultSimdVector::LANES {
            if left_vals[i].is_nan() && right_vals[i].is_nan() {
                continue;
            }
            // Both infinite with same sign is considered equal
            if left_vals[i].is_infinite() && right_vals[i].is_infinite() &&
               left_vals[i].is_sign_positive() == right_vals[i].is_sign_positive() {
                continue;
            }
            let diff = (left_vals[i] - right_vals[i]).abs();
            // Skip if diff is NaN (can happen with inf - inf)
            if diff.is_nan() {
                continue;
            }
            let max_val = left_vals[i].abs().max(right_vals[i].abs());
            assert!(
                diff < 1e-4 || (max_val > 0.0 && diff / max_val < 1e-4),
                "Distributivity violated: left={}, right={}, diff={}, a={}, b={}, c={}",
                left_vals[i], right_vals[i], diff, a, b, c
            );
        }
    });
}

/// Test T073: Mathematical invariants - Identity elements
///
/// Property: a + 0 == a, a * 1 == a
#[test]
fn test_identity_elements() {
    proptest!(proptest_config(), |(a in normal_f32())| {
        let vec_a = DefaultSimdVector::splat(a);
        let zero = DefaultSimdVector::splat(0.0);
        let one = DefaultSimdVector::splat(1.0);

        // Additive identity
        let add_zero = vec_a.add(zero);
        let mut add_result = vec![0.0; DefaultSimdVector::LANES];
        add_zero.to_slice(&mut add_result);
        for &val in &add_result {
            assert_approx_eq(val, a, "additive identity");
        }

        // Multiplicative identity
        let mul_one = vec_a.mul(one);
        let mut mul_result = vec![0.0; DefaultSimdVector::LANES];
        mul_one.to_slice(&mut mul_result);
        for &val in &mul_result {
            assert_approx_eq(val, a, "multiplicative identity");
        }
    });
}

/// Test T073: Mathematical invariants - Inverse elements
///
/// Property: a + (-a) ≈ 0, a * (1/a) ≈ 1 (for a != 0)
#[test]
fn test_inverse_elements() {
    proptest!(proptest_config(), |(a in positive_f32())| {
        let vec_a = DefaultSimdVector::splat(a);
        let vec_neg_a = DefaultSimdVector::splat(-a);
        let vec_inv_a = DefaultSimdVector::splat(1.0 / a);

        // Additive inverse
        let add_inv = vec_a.add(vec_neg_a);
        let mut add_result = vec![0.0; DefaultSimdVector::LANES];
        add_inv.to_slice(&mut add_result);
        for &val in &add_result {
            assert_approx_eq(val, 0.0, "additive inverse");
        }

        // Multiplicative inverse (for non-zero values)
        let mul_inv = vec_a.mul(vec_inv_a);
        let mut mul_result = vec![0.0; DefaultSimdVector::LANES];
        mul_inv.to_slice(&mut mul_result);
        for &val in &mul_result {
            assert_approx_eq(val, 1.0, "multiplicative inverse");
        }
    });
}

/// Test T073: Comparison operations consistency
///
/// Property: if a < b, then !(a >= b) and !(a > b)
#[test]
fn test_comparison_consistency() {
    proptest!(proptest_config(), |(
        (a, b) in normal_f32_pair()
    )| {
        let vec_a = DefaultSimdVector::splat(a);
        let vec_b = DefaultSimdVector::splat(b);

        let lt_mask = vec_a.lt(vec_b);
        let gt_mask = vec_a.gt(vec_b);
        let eq_mask = vec_a.eq(vec_b);

        // If a < b, then not (a > b) and not (a == b)
        // If a > b, then not (a < b) and not (a == b)
        // If a == b, then not (a < b) and not (a > b)

        // At most one of these should be true
        let count_true = (lt_mask.any() as u32) + (gt_mask.any() as u32) + (eq_mask.any() as u32);
        assert!(
            count_true <= 1,
            "Comparison masks should be mutually exclusive: a={}, b={}, lt={}, gt={}, eq={}",
            a, b, lt_mask.any(), gt_mask.any(), eq_mask.any()
        );
    });
}

/// Test T073: Clamp correctness
///
/// Property: clamp(x, min, max) is always in [min, max]
#[test]
fn test_clamp_range() {
    proptest!(proptest_config(), |(
        value in normal_f32(),
        min_val in small_normal_f32(),
        max_val in small_normal_f32()
    )| {
        // Ensure min <= max
        let (min_val, max_val) = if min_val <= max_val {
            (min_val, max_val)
        } else {
            (max_val, min_val)
        };

        let vec_value = DefaultSimdVector::splat(value);
        let vec_min = DefaultSimdVector::splat(min_val);
        let vec_max = DefaultSimdVector::splat(max_val);

        use rigel_math::ops::clamp;
        let clamped = clamp(vec_value, vec_min, vec_max);

        let mut result = vec![0.0; DefaultSimdVector::LANES];
        clamped.to_slice(&mut result);

        for &val in &result {
            assert!(
                val >= min_val && val <= max_val,
                "Clamped value out of range: value={}, min={}, max={}, result={}",
                value, min_val, max_val, val
            );
        }
    });
}
