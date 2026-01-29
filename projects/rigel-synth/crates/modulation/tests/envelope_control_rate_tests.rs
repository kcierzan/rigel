//! Integration tests for control-rate envelope processing.
//!
//! These tests verify that control-rate envelopes:
//! 1. Reach the same levels at checkpoints as per-sample envelopes
//! 2. Handle segment transitions correctly mid-interval
//! 3. Process note events at any point in the interval
//! 4. Keep all output values in [0.0, 1.0]
//! 5. Behave consistently across different sample rates

use rigel_modulation::envelope::{
    ControlRateFmEnvelope, EnvelopePhase, FmEnvelope, FmEnvelopeConfig,
};
use rigel_modulation::ModulationSource;
use rigel_timing::Timebase;

const SAMPLE_RATE_44K: f32 = 44100.0;
const SAMPLE_RATE_48K: f32 = 48000.0;
const SAMPLE_RATE_96K: f32 = 96000.0;

/// Tolerance for comparing control-rate vs per-sample values.
/// Linear interpolation introduces errors due to the envelope's
/// exponential nature being approximated linearly. A 10% tolerance
/// is acceptable since control-rate is for CPU savings, not precision.
const VALUE_TOLERANCE: f32 = 0.10;

// =============================================================================
// Timing Accuracy Tests
// =============================================================================

/// Test that control-rate envelope reaches approximately the same levels
/// at key checkpoints as a per-sample envelope.
#[test]
fn test_timing_accuracy_at_checkpoints() {
    let config = FmEnvelopeConfig::adsr(0.1, 0.2, 0.7, 0.5, SAMPLE_RATE_44K);
    let update_interval = 64;

    let mut per_sample = FmEnvelope::with_config(config);
    let mut control_rate = ControlRateFmEnvelope::new(config, update_interval);

    per_sample.note_on(60);
    control_rate.note_on(60);

    // Compare values at regular checkpoints (every 64 samples = ~1.45ms)
    let num_checkpoints = 100;
    let samples_per_checkpoint = update_interval;

    for checkpoint in 0..num_checkpoints {
        // Advance per-sample envelope
        for _ in 0..samples_per_checkpoint {
            per_sample.process();
        }

        // Advance control-rate envelope
        // tick() advances inner by update_interval samples and sets target_value
        control_rate.tick();

        // Compare values immediately after tick() - this is when both envelopes
        // are synchronized (both have processed N*64 samples from their inner state)
        // The control-rate envelope's inner().value() should match per_sample.value()
        let per_sample_val = per_sample.value();
        let control_rate_val = control_rate.inner().value();

        let diff = (per_sample_val - control_rate_val).abs();
        assert!(
            diff < VALUE_TOLERANCE,
            "Checkpoint {}: per_sample={:.4}, control_rate={:.4}, diff={:.4}",
            checkpoint,
            per_sample_val,
            control_rate_val,
            diff
        );

        // Consume samples to keep the control-rate envelope in sync
        for _ in 0..samples_per_checkpoint {
            control_rate.sample();
        }
    }
}

// =============================================================================
// Segment Transition Tests
// =============================================================================

/// Test that segment transitions work correctly when they occur mid-interval.
#[test]
fn test_segment_transitions_mid_interval() {
    // Use very short attack to force transition
    let config = FmEnvelopeConfig::adsr(0.001, 0.1, 0.7, 0.3, SAMPLE_RATE_44K);
    let mut env = ControlRateFmEnvelope::new(config, 64);

    env.note_on(60);

    // Process enough blocks to go through attack and into decay
    for _ in 0..20 {
        env.tick();

        // Sample through the interval
        for _ in 0..64 {
            let value = env.sample();
            assert!(
                (0.0..=1.0).contains(&value),
                "Value {} out of range during segment transition",
                value
            );
        }
    }
}

/// Test that fast segment transitions don't cause discontinuities.
#[test]
fn test_no_discontinuities_on_fast_transitions() {
    // Very fast envelope for stress testing
    let config = FmEnvelopeConfig::adsr(0.005, 0.01, 0.5, 0.01, SAMPLE_RATE_44K);
    let mut env = ControlRateFmEnvelope::new(config, 32);

    env.note_on(60);

    let mut prev_value = env.current_value();

    // Process through full lifecycle
    for block in 0..100 {
        env.tick();

        for sample in 0..32 {
            let value = env.sample();

            // Check for large discontinuities (> 0.3 is suspicious)
            let diff = (value - prev_value).abs();
            assert!(
                diff < 0.3,
                "Discontinuity at block {} sample {}: {} -> {} (diff={})",
                block,
                sample,
                prev_value,
                value,
                diff
            );

            prev_value = value;
        }
    }
}

// =============================================================================
// Note Event Tests
// =============================================================================

/// Test that note_on works at any point in the interval.
#[test]
fn test_note_on_mid_interval() {
    let config = FmEnvelopeConfig::adsr(0.1, 0.2, 0.7, 0.5, SAMPLE_RATE_44K);
    let mut env = ControlRateFmEnvelope::new(config, 64);

    // Start envelope, process partially
    env.note_on(60);
    env.tick();
    for _ in 0..30 {
        env.sample();
    }

    // Retrigger mid-interval
    env.note_on(72);

    // Should still produce valid output
    for _ in 0..34 {
        let value = env.sample();
        assert!((0.0..=1.0).contains(&value));
    }
}

/// Test that note_off works at any point in the interval.
#[test]
fn test_note_off_mid_interval() {
    let config = FmEnvelopeConfig::adsr(0.1, 0.2, 0.7, 0.5, SAMPLE_RATE_44K);
    let mut env = ControlRateFmEnvelope::new(config, 64);

    env.note_on(60);

    // Process a few blocks
    for _ in 0..10 {
        env.tick();
        for _ in 0..64 {
            env.sample();
        }
    }

    // Release mid-interval
    env.tick();
    for _ in 0..30 {
        env.sample();
    }
    env.note_off();

    // Should transition to release
    assert_eq!(env.phase(), EnvelopePhase::Release);

    // Continue processing - should produce valid output
    for _ in 0..34 {
        let value = env.sample();
        assert!((0.0..=1.0).contains(&value));
    }
}

/// Test note_off during attack phase.
#[test]
fn test_note_off_during_attack() {
    let config = FmEnvelopeConfig::adsr(0.5, 0.2, 0.7, 0.3, SAMPLE_RATE_44K);
    let mut env = ControlRateFmEnvelope::new(config, 64);

    env.note_on(60);
    env.tick();

    // Release immediately during attack
    env.note_off();
    assert_eq!(env.phase(), EnvelopePhase::Release);

    // Should complete release without errors
    let mut iterations = 0;
    while env.is_active() && iterations < 1000 {
        env.tick();
        for _ in 0..64 {
            let value = env.sample();
            assert!((0.0..=1.0).contains(&value));
        }
        iterations += 1;
    }

    assert!(!env.is_active(), "Envelope should complete");
}

// =============================================================================
// Output Range Tests
// =============================================================================

/// Test that all output values are in [0.0, 1.0] throughout full lifecycle.
#[test]
fn test_output_range_full_lifecycle() {
    let config = FmEnvelopeConfig::adsr(0.1, 0.2, 0.7, 0.5, SAMPLE_RATE_44K);
    let mut env = ControlRateFmEnvelope::new(config, 64);

    env.note_on(60);

    // Process 500ms key-on
    for _ in 0..(22050 / 64) {
        env.tick();
        for _ in 0..64 {
            let value = env.sample();
            assert!(
                (0.0..=1.0).contains(&value),
                "Value {} out of range during key-on",
                value
            );
        }
    }

    env.note_off();

    // Process 500ms release
    for _ in 0..(22050 / 64) {
        env.tick();
        for _ in 0..64 {
            let value = env.sample();
            assert!(
                (0.0..=1.0).contains(&value),
                "Value {} out of range during release",
                value
            );
        }
    }
}

/// Test output range with extreme envelope settings.
#[test]
fn test_output_range_extreme_settings() {
    // Instant attack, instant release
    let config = FmEnvelopeConfig::adsr(0.001, 0.001, 1.0, 0.001, SAMPLE_RATE_44K);
    let mut env = ControlRateFmEnvelope::new(config, 64);

    env.note_on(60);

    for _ in 0..50 {
        env.tick();
        for _ in 0..64 {
            let value = env.sample();
            assert!(
                (0.0..=1.0).contains(&value),
                "Value {} out of range with extreme settings",
                value
            );
        }
    }
}

// =============================================================================
// Sample Rate Independence Tests
// =============================================================================

/// Test that envelope behavior is consistent across sample rates.
/// The same config should produce similar envelope shapes at different rates.
#[test]
fn test_sample_rate_independence() {
    let rates = [SAMPLE_RATE_44K, SAMPLE_RATE_48K, SAMPLE_RATE_96K];

    for rate in rates {
        let config = FmEnvelopeConfig::adsr(0.1, 0.2, 0.7, 0.5, rate);

        // Interval should scale with sample rate
        let interval = if rate >= 96000.0 { 128 } else { 64 };
        let mut env = ControlRateFmEnvelope::new(config, interval);

        env.note_on(60);

        // Process ~100ms worth of samples
        let samples_100ms = (rate * 0.1) as usize;
        let blocks = samples_100ms / interval as usize;

        for _ in 0..blocks {
            env.tick();
            for _ in 0..interval {
                let value = env.sample();
                assert!(
                    (0.0..=1.0).contains(&value),
                    "Value {} out of range at {}Hz",
                    value,
                    rate
                );
            }
        }

        // Should be well into the envelope by now (past the jump target)
        let value = env.current_value();
        assert!(
            value > 0.4,
            "Expected value > 0.4 after 100ms at {}Hz, got {}",
            rate,
            value
        );
    }
}

// =============================================================================
// ModulationSource Trait Tests
// =============================================================================

/// Test ModulationSource trait implementation.
#[test]
fn test_modulation_source_trait() {
    let config = FmEnvelopeConfig::adsr(0.1, 0.2, 0.7, 0.5, SAMPLE_RATE_44K);
    let mut env = ControlRateFmEnvelope::new(config, 64);
    let timebase = Timebase::new(SAMPLE_RATE_44K);

    env.note_on(60);

    // Use trait methods
    <ControlRateFmEnvelope as ModulationSource>::update(&mut env, &timebase);

    let value = <ControlRateFmEnvelope as ModulationSource>::value(&env);
    assert!((0.0..=1.0).contains(&value));

    // Reset via trait
    <ControlRateFmEnvelope as ModulationSource>::reset(&mut env, &timebase);
    assert_eq!(env.phase(), EnvelopePhase::Idle);
}

// =============================================================================
// Block Processing Tests
// =============================================================================

/// Test generate_block produces valid output.
#[test]
fn test_generate_block() {
    let config = FmEnvelopeConfig::adsr(0.1, 0.2, 0.7, 0.5, SAMPLE_RATE_44K);
    let mut env = ControlRateFmEnvelope::new(config, 64);

    env.note_on(60);

    let mut output = [0.0f32; 64];

    for _ in 0..100 {
        env.generate_block(&mut output);

        for (i, &value) in output.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&value),
                "generate_block produced {} at index {}",
                value,
                i
            );
        }
    }
}

/// Test generate_block handles partial blocks.
#[test]
fn test_generate_block_partial() {
    let config = FmEnvelopeConfig::adsr(0.1, 0.2, 0.7, 0.5, SAMPLE_RATE_44K);
    let mut env = ControlRateFmEnvelope::new(config, 64);

    env.note_on(60);

    // Process smaller block
    let mut output = [0.0f32; 32];
    env.generate_block(&mut output);

    for &value in &output {
        assert!((0.0..=1.0).contains(&value));
    }

    // Process another partial block
    env.generate_block(&mut output);

    for &value in &output {
        assert!((0.0..=1.0).contains(&value));
    }
}

// =============================================================================
// Reset Tests
// =============================================================================

/// Test that reset properly clears state.
#[test]
fn test_reset_clears_state() {
    let config = FmEnvelopeConfig::adsr(0.1, 0.2, 0.7, 0.5, SAMPLE_RATE_44K);
    let mut env = ControlRateFmEnvelope::new(config, 64);

    env.note_on(60);

    // Process for a while
    for _ in 0..50 {
        env.tick();
        for _ in 0..64 {
            env.sample();
        }
    }

    assert!(env.is_active());

    // Reset
    env.reset();

    assert!(!env.is_active());
    assert_eq!(env.phase(), EnvelopePhase::Idle);
    assert_eq!(env.current_value(), 0.0);
}

/// Test that envelope can be reused after reset.
#[test]
fn test_reuse_after_reset() {
    let config = FmEnvelopeConfig::adsr(0.1, 0.2, 0.7, 0.5, SAMPLE_RATE_44K);
    let mut env = ControlRateFmEnvelope::new(config, 64);

    // First use
    env.note_on(60);
    for _ in 0..20 {
        env.tick();
        for _ in 0..64 {
            env.sample();
        }
    }
    env.reset();

    // Second use - should work identically
    env.note_on(60);
    assert_eq!(env.phase(), EnvelopePhase::KeyOn);

    for _ in 0..20 {
        env.tick();
        for _ in 0..64 {
            let value = env.sample();
            assert!((0.0..=1.0).contains(&value));
        }
    }
}

// =============================================================================
// Edge Case Tests
// =============================================================================

/// Test with minimum interval (1 sample = per-sample equivalent).
#[test]
fn test_minimum_interval() {
    let config = FmEnvelopeConfig::adsr(0.1, 0.2, 0.7, 0.5, SAMPLE_RATE_44K);
    let mut env = ControlRateFmEnvelope::new(config, 1);

    env.note_on(60);

    // Should behave essentially like per-sample
    for _ in 0..1000 {
        env.tick();
        let value = env.sample();
        assert!((0.0..=1.0).contains(&value));
    }
}

/// Test with maximum interval (128 samples).
#[test]
fn test_maximum_interval() {
    let config = FmEnvelopeConfig::adsr(0.1, 0.2, 0.7, 0.5, SAMPLE_RATE_44K);
    let mut env = ControlRateFmEnvelope::new(config, 128);

    env.note_on(60);

    for _ in 0..100 {
        env.tick();
        for _ in 0..128 {
            let value = env.sample();
            assert!((0.0..=1.0).contains(&value));
        }
    }
}
