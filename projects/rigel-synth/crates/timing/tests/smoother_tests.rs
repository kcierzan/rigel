//! Smoother unit tests
//!
//! Tests for parameter smoothing with multiple curve types.

use rigel_timing::{Smoother, SmoothingMode};

const SAMPLE_RATE: f32 = 44100.0;

#[test]
fn test_smoother_new_initializes_correctly() {
    let smoother = Smoother::new(1.0, SmoothingMode::Linear, 10.0, SAMPLE_RATE);
    assert_eq!(smoother.current(), 1.0);
    assert_eq!(smoother.target(), 1.0);
    assert!(!smoother.is_active());
}

#[test]
fn test_smoother_default() {
    let smoother = Smoother::default();
    assert_eq!(smoother.current(), 0.0);
    assert_eq!(smoother.target(), 0.0);
    assert!(!smoother.is_active());
}

#[test]
fn test_smoother_with_defaults() {
    let smoother = Smoother::with_defaults(0.5);
    assert_eq!(smoother.current(), 0.5);
    assert_eq!(smoother.target(), 0.5);
    assert!(!smoother.is_active());
}

#[test]
#[should_panic(expected = "Smoothing time must be non-negative")]
fn test_smoother_new_panics_on_negative_smoothing_time() {
    let _smoother = Smoother::new(1.0, SmoothingMode::Linear, -1.0, SAMPLE_RATE);
}

#[test]
#[should_panic(expected = "Sample rate must be positive")]
fn test_smoother_new_panics_on_zero_sample_rate() {
    let _smoother = Smoother::new(1.0, SmoothingMode::Linear, 10.0, 0.0);
}

// ============================================================================
// Instant Mode Tests
// ============================================================================

#[test]
fn test_instant_mode_changes_immediately() {
    let mut smoother = Smoother::new(0.0, SmoothingMode::Instant, 10.0, SAMPLE_RATE);

    smoother.set_target(1.0);

    // Value should already be at target
    assert_eq!(smoother.current(), 1.0);
    assert_eq!(smoother.target(), 1.0);
    assert!(!smoother.is_active());
}

#[test]
fn test_instant_mode_no_samples_needed() {
    let mut smoother = Smoother::new(0.0, SmoothingMode::Instant, 10.0, SAMPLE_RATE);

    smoother.set_target(1.0);

    // Should already be at target before any samples processed
    assert_eq!(smoother.current(), 1.0);
    assert!(!smoother.is_active());

    // Processing more samples should just return the target
    for _ in 0..100 {
        assert_eq!(smoother.process_sample(), 1.0);
    }
}

// ============================================================================
// Linear Mode Tests
// ============================================================================

#[test]
fn test_linear_smoothing_reaches_target_in_exact_time() {
    // 10ms smoothing at 44100 Hz = 441 samples
    let mut smoother = Smoother::new(0.0, SmoothingMode::Linear, 10.0, SAMPLE_RATE);

    smoother.set_target(1.0);
    assert!(smoother.is_active());

    // Process exactly 441 samples
    let expected_samples = 441u32;
    for i in 0..expected_samples {
        let value = smoother.process_sample();

        // Value should increase linearly
        if i < expected_samples - 1 {
            assert!(smoother.is_active(), "Should still be active at sample {}", i);
        }

        // Value should be between 0 and 1
        assert!(
            (0.0..=1.0).contains(&value),
            "Value {} out of range at sample {}",
            value,
            i
        );
    }

    // After 441 samples, should be exactly at target
    assert_eq!(smoother.current(), 1.0);
    assert!(!smoother.is_active());
}

#[test]
fn test_linear_smoothing_constant_rate() {
    let mut smoother = Smoother::new(0.0, SmoothingMode::Linear, 10.0, SAMPLE_RATE);

    smoother.set_target(1.0);

    let val1 = smoother.process_sample();
    let val2 = smoother.process_sample();
    let val3 = smoother.process_sample();

    // Increment should be constant
    let inc1 = val2 - val1;
    let inc2 = val3 - val2;

    assert!(
        (inc1 - inc2).abs() < 1e-6,
        "Linear increment should be constant"
    );
}

#[test]
fn test_linear_smoothing_downward() {
    let mut smoother = Smoother::new(1.0, SmoothingMode::Linear, 10.0, SAMPLE_RATE);

    smoother.set_target(0.0);

    // Process all samples
    for _ in 0..441 {
        smoother.process_sample();
    }

    // Should reach target
    assert_eq!(smoother.current(), 0.0);
    assert!(!smoother.is_active());
}

// ============================================================================
// Exponential Mode Tests
// ============================================================================

#[test]
fn test_exponential_smoothing_reaches_threshold() {
    let mut smoother = Smoother::new(0.0, SmoothingMode::Exponential, 10.0, SAMPLE_RATE);

    smoother.set_target(1.0);
    assert!(smoother.is_active());

    // Process until we reach the target (should happen within reasonable time)
    let mut samples_processed = 0;
    let max_samples = 10000; // Safety limit

    while smoother.is_active() && samples_processed < max_samples {
        smoother.process_sample();
        samples_processed += 1;
    }

    // Should have reached target within threshold
    assert!(!smoother.is_active(), "Should have reached target");

    // Current should be exactly at target (snapped when threshold reached)
    assert_eq!(smoother.current(), 1.0);

    // Should complete in reasonable time (not too many samples)
    // For 10ms at 44100Hz, typically ~5-6x time constant = 2200-2600 samples
    assert!(
        samples_processed < 5000,
        "Took too long: {} samples",
        samples_processed
    );
}

#[test]
fn test_exponential_smoothing_asymptotic_approach() {
    let mut smoother = Smoother::new(0.0, SmoothingMode::Exponential, 10.0, SAMPLE_RATE);

    smoother.set_target(1.0);

    // Process some samples and verify asymptotic behavior
    let val1 = smoother.process_sample();
    let val2 = smoother.process_sample();
    let val3 = smoother.process_sample();

    // Increments should get smaller (exponential decay)
    let inc1 = val2 - val1;
    let inc2 = val3 - val2;

    assert!(inc2 < inc1, "Exponential increments should decrease");
}

#[test]
fn test_exponential_smoothing_0_1_percent_threshold() {
    let mut smoother = Smoother::new(0.0, SmoothingMode::Exponential, 10.0, SAMPLE_RATE);

    smoother.set_target(1.0);

    // Process until complete
    while smoother.is_active() {
        smoother.process_sample();
    }

    // Should be at target (snapped)
    assert_eq!(smoother.current(), 1.0);

    // Verify it didn't just snap immediately (actually smoothed)
    let mut test_smoother = Smoother::new(0.0, SmoothingMode::Exponential, 10.0, SAMPLE_RATE);
    test_smoother.set_target(1.0);
    test_smoother.process_sample();
    assert!(
        test_smoother.current() < 1.0,
        "Should not reach target in 1 sample"
    );
}

// ============================================================================
// Logarithmic Mode Tests
// ============================================================================

#[test]
fn test_logarithmic_smoothing_for_frequency() {
    // Test smoothing from 1000 Hz to 5000 Hz
    let mut smoother = Smoother::new(1000.0, SmoothingMode::Logarithmic, 10.0, SAMPLE_RATE);

    smoother.set_target(5000.0);
    assert!(smoother.is_active());

    // Process until complete
    let mut samples_processed = 0;
    let max_samples = 10000;

    while smoother.is_active() && samples_processed < max_samples {
        smoother.process_sample();
        samples_processed += 1;
    }

    // Should have reached target
    assert!(!smoother.is_active(), "Should have reached target");

    // Should be at target (or very close)
    let diff = (smoother.current() - 5000.0).abs();
    assert!(diff < 1.0, "Should be at target, diff = {}", diff);
}

#[test]
fn test_logarithmic_smoothing_handles_small_values() {
    // Test smoothing near zero
    let mut smoother = Smoother::new(0.001, SmoothingMode::Logarithmic, 10.0, SAMPLE_RATE);

    smoother.set_target(1.0);

    // Should not panic or produce NaN
    for _ in 0..1000 {
        let value = smoother.process_sample();
        assert!(!value.is_nan(), "Should not produce NaN");
        assert!(!value.is_infinite(), "Should not produce infinity");
    }
}

#[test]
fn test_logarithmic_smoothing_perceptually_linear() {
    // In log domain, the midpoint should be the geometric mean
    // Geometric mean of 100 and 10000 = sqrt(100 * 10000) = 1000
    let mut smoother = Smoother::new(100.0, SmoothingMode::Logarithmic, 100.0, SAMPLE_RATE);

    smoother.set_target(10000.0);

    // Find the approximate midpoint value
    let total_samples = (100.0 / 1000.0 * SAMPLE_RATE) as u32;
    let mid_samples = total_samples / 2;

    for _ in 0..mid_samples {
        smoother.process_sample();
    }

    let mid_value = smoother.current();

    // Should be close to geometric mean (1000), not arithmetic mean (5050)
    // Allow some tolerance due to exponential nature of approach
    let geometric_mean = 1000.0;
    let relative_diff = (mid_value - geometric_mean).abs() / geometric_mean;

    // Should be closer to geometric mean than arithmetic mean
    assert!(
        relative_diff < 0.5,
        "Midpoint {} should be close to geometric mean 1000",
        mid_value
    );
}

// ============================================================================
// Configuration Methods Tests
// ============================================================================

#[test]
fn test_set_immediate() {
    let mut smoother = Smoother::new(0.0, SmoothingMode::Linear, 10.0, SAMPLE_RATE);

    smoother.set_target(0.5); // Start smoothing
    assert!(smoother.is_active());

    // Force immediate value
    smoother.set_immediate(1.0);

    assert_eq!(smoother.current(), 1.0);
    assert_eq!(smoother.target(), 1.0);
    assert!(!smoother.is_active());
}

#[test]
fn test_set_smoothing_time() {
    let mut smoother = Smoother::new(0.0, SmoothingMode::Linear, 10.0, SAMPLE_RATE);

    smoother.set_smoothing_time(20.0);
    smoother.set_target(1.0);

    // With 20ms at 44100 Hz = 882 samples
    for _ in 0..882 {
        smoother.process_sample();
    }

    assert!(!smoother.is_active());
    assert_eq!(smoother.current(), 1.0);
}

#[test]
fn test_set_mode() {
    let mut smoother = Smoother::new(0.0, SmoothingMode::Linear, 10.0, SAMPLE_RATE);

    smoother.set_mode(SmoothingMode::Instant);
    smoother.set_target(1.0);

    // Should now behave as Instant
    assert_eq!(smoother.current(), 1.0);
    assert!(!smoother.is_active());
}

#[test]
fn test_set_sample_rate_recalculates_coefficients() {
    let mut smoother = Smoother::new(0.0, SmoothingMode::Linear, 10.0, SAMPLE_RATE);

    // Change to 96kHz
    smoother.set_sample_rate(96000.0);
    smoother.set_target(1.0);

    // 10ms at 96kHz = 960 samples
    for _ in 0..960 {
        smoother.process_sample();
    }

    assert!(!smoother.is_active());
    assert_eq!(smoother.current(), 1.0);
}

#[test]
fn test_reset() {
    let mut smoother = Smoother::new(0.0, SmoothingMode::Linear, 10.0, SAMPLE_RATE);

    smoother.set_target(1.0);
    for _ in 0..200 {
        smoother.process_sample();
    }

    // Reset to a new value
    smoother.reset(0.5);

    assert_eq!(smoother.current(), 0.5);
    assert_eq!(smoother.target(), 0.5);
    assert!(!smoother.is_active());
}

// ============================================================================
// process_block Tests
// ============================================================================

#[test]
fn test_process_block() {
    let mut smoother = Smoother::new(0.0, SmoothingMode::Linear, 10.0, SAMPLE_RATE);
    smoother.set_target(1.0);

    let mut buffer = [0.0f32; 64];
    smoother.process_block(&mut buffer);

    // Buffer should contain increasing values
    for i in 1..buffer.len() {
        assert!(buffer[i] >= buffer[i - 1], "Values should be non-decreasing");
    }

    // First value should be non-zero (we started from 0, targeting 1)
    assert!(buffer[0] > 0.0);
}

#[test]
fn test_process_block_when_inactive() {
    let smoother_value = 0.5;
    let mut smoother = Smoother::new(smoother_value, SmoothingMode::Linear, 10.0, SAMPLE_RATE);

    // Not setting a target, so smoother is inactive
    let mut buffer = [0.0f32; 64];
    smoother.process_block(&mut buffer);

    // All values should be the current value
    for value in buffer.iter() {
        assert_eq!(*value, smoother_value);
    }
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_zero_smoothing_time_acts_as_instant() {
    let mut smoother = Smoother::new(0.0, SmoothingMode::Linear, 0.0, SAMPLE_RATE);

    smoother.set_target(1.0);

    // Should be instant
    assert_eq!(smoother.current(), 1.0);
    assert!(!smoother.is_active());
}

#[test]
fn test_setting_same_target_when_not_active() {
    let mut smoother = Smoother::new(1.0, SmoothingMode::Linear, 10.0, SAMPLE_RATE);

    // Set target to same value when not active
    smoother.set_target(1.0);

    // Should remain inactive
    assert!(!smoother.is_active());
}

#[test]
fn test_smoother_is_copy_and_clone() {
    let smoother = Smoother::new(1.0, SmoothingMode::Linear, 10.0, SAMPLE_RATE);

    // Test Copy
    let copy = smoother;
    assert_eq!(copy.current(), 1.0);

    // Original should still work
    assert_eq!(smoother.current(), 1.0);

    // Test Clone (explicit clone() to verify Clone trait even though Copy is available)
    #[allow(clippy::clone_on_copy)]
    let clone = smoother.clone();
    assert_eq!(clone.current(), 1.0);
}
