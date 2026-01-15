//! Integration tests verifying envelope timing matches UI labels.
//!
//! These tests ensure that the time parameters displayed in the UI
//! correspond to the actual envelope behavior in audio output.

use rigel_modulation::envelope::{
    seconds_to_rate, EnvelopePhase, FmEnvelope, FmEnvelopeConfig, Segment,
};

/// Test that attack time matches configured value.
#[test]
fn test_attack_time_100ms() {
    let sample_rate = 44100.0;
    let attack_time = 0.1; // 100ms

    let config = FmEnvelopeConfig::adsr(attack_time, 0.1, 0.7, 0.1, sample_rate);
    let mut env = FmEnvelope::with_config(config);
    env.note_on(60);

    // Count samples to reach 90% of target
    let mut samples = 0;
    while env.value() < 0.9 && samples < 50000 {
        env.process();
        samples += 1;
    }

    let measured_time = samples as f32 / sample_rate;
    let tolerance = 0.3; // 30% tolerance

    assert!(
        (measured_time - attack_time).abs() < attack_time * tolerance,
        "Attack time mismatch: expected {}ms, got {}ms",
        attack_time * 1000.0,
        measured_time * 1000.0
    );
}

/// Test 500ms attack time.
#[test]
fn test_attack_time_500ms() {
    let sample_rate = 44100.0;
    let attack_time = 0.5; // 500ms

    let config = FmEnvelopeConfig::adsr(attack_time, 0.1, 0.7, 0.1, sample_rate);
    let mut env = FmEnvelope::with_config(config);
    env.note_on(60);

    // Count samples to reach 90% of target
    let mut samples = 0;
    while env.value() < 0.9 && samples < 100000 {
        env.process();
        samples += 1;
    }

    let measured_time = samples as f32 / sample_rate;
    let tolerance = 0.3; // 30% tolerance

    assert!(
        (measured_time - attack_time).abs() < attack_time * tolerance,
        "Attack time mismatch: expected {}ms, got {}ms",
        attack_time * 1000.0,
        measured_time * 1000.0
    );
}

/// Test release time matches configured value.
#[test]
fn test_release_time_300ms() {
    let sample_rate = 44100.0;
    let release_time = 0.3; // 300ms

    let config = FmEnvelopeConfig::adsr(0.01, 0.01, 1.0, release_time, sample_rate);
    let mut env = FmEnvelope::with_config(config);
    env.note_on(60);

    // Process through attack/decay to sustain
    for _ in 0..5000 {
        env.process();
    }

    env.note_off();
    assert_eq!(
        env.phase(),
        EnvelopePhase::Release,
        "Should be in release phase"
    );

    // Get starting level
    let start_level = env.value();
    assert!(start_level > 0.5, "Should start release at high level");

    // Count samples to reach 10% of starting level
    let mut samples = 0;
    while env.value() > start_level * 0.1 && samples < 100000 {
        env.process();
        samples += 1;
    }

    let measured_time = samples as f32 / sample_rate;
    let tolerance = 0.3; // 30% tolerance

    assert!(
        (measured_time - release_time).abs() < release_time * tolerance,
        "Release time mismatch: expected {}ms, got {}ms",
        release_time * 1000.0,
        measured_time * 1000.0
    );
}

/// Test 1 second release time.
#[test]
fn test_release_time_1s() {
    let sample_rate = 44100.0;
    let release_time = 1.0; // 1 second

    let config = FmEnvelopeConfig::adsr(0.01, 0.01, 1.0, release_time, sample_rate);
    let mut env = FmEnvelope::with_config(config);
    env.note_on(60);

    // Process through attack/decay to sustain
    for _ in 0..5000 {
        env.process();
    }

    env.note_off();

    // Get starting level
    let start_level = env.value();

    // Count samples to reach 10% of starting level
    let mut samples = 0;
    while env.value() > start_level * 0.1 && samples < 200000 {
        env.process();
        samples += 1;
    }

    let measured_time = samples as f32 / sample_rate;
    let tolerance = 0.3; // 30% tolerance

    assert!(
        (measured_time - release_time).abs() < release_time * tolerance,
        "Release time mismatch: expected {}ms, got {}ms",
        release_time * 1000.0,
        measured_time * 1000.0
    );
}

/// Test multiple time values for attack parametrically.
#[test]
fn test_attack_timing_parametric() {
    let sample_rate = 44100.0;
    let test_times = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0];

    for expected_time in test_times {
        let config = FmEnvelopeConfig::adsr(expected_time, 0.01, 0.7, 0.01, sample_rate);
        let mut env = FmEnvelope::with_config(config);
        env.note_on(60);

        let mut samples = 0;
        while env.value() < 0.9 && samples < 500000 {
            env.process();
            samples += 1;
        }

        let measured = samples as f32 / sample_rate;

        // Use larger tolerance for very short times
        let tolerance = if expected_time < 0.1 { 0.5 } else { 0.3 };

        assert!(
            (measured - expected_time).abs() < expected_time * tolerance,
            "Attack time {}s: expected {}ms, got {}ms",
            expected_time,
            expected_time * 1000.0,
            measured * 1000.0
        );
    }
}

/// Test multiple time values for release parametrically.
#[test]
fn test_release_timing_parametric() {
    let sample_rate = 44100.0;
    let test_times = [0.1, 0.3, 0.5, 1.0, 2.0];

    for expected_time in test_times {
        let config = FmEnvelopeConfig::adsr(0.01, 0.01, 1.0, expected_time, sample_rate);
        let mut env = FmEnvelope::with_config(config);
        env.note_on(60);

        // Process to sustain
        for _ in 0..5000 {
            env.process();
        }

        env.note_off();
        let start_level = env.value();

        let mut samples = 0;
        while env.value() > start_level * 0.1 && samples < 500000 {
            env.process();
            samples += 1;
        }

        let measured = samples as f32 / sample_rate;
        let tolerance = if expected_time < 0.2 { 0.5 } else { 0.3 };

        assert!(
            (measured - expected_time).abs() < expected_time * tolerance,
            "Release time {}s: expected {}ms, got {}ms",
            expected_time,
            expected_time * 1000.0,
            measured * 1000.0
        );
    }
}

/// Test release actually works (not broken).
#[test]
fn test_release_completes() {
    let mut env = FmEnvelope::new(44100.0);
    env.note_on(60);

    // Process some samples
    for _ in 0..1000 {
        env.process();
    }

    let level_before = env.value();
    assert!(level_before > 0.5, "Should be at high level before release");

    env.note_off();
    assert_eq!(
        env.phase(),
        EnvelopePhase::Release,
        "Should be in release phase"
    );

    // Process release - give it plenty of time
    for _ in 0..100000 {
        env.process();
    }

    assert!(
        env.value() < 0.01,
        "Should be near zero after release, got {}",
        env.value()
    );
}

/// Test envelope transitions through all phases correctly.
#[test]
fn test_full_envelope_lifecycle() {
    let sample_rate = 44100.0;
    let config = FmEnvelopeConfig::adsr(0.1, 0.1, 0.7, 0.3, sample_rate);
    let mut env = FmEnvelope::with_config(config);

    // Initial state
    assert_eq!(env.phase(), EnvelopePhase::Idle);
    assert!(env.value() < f32::EPSILON);

    // Note on -> KeyOn
    env.note_on(60);
    assert_eq!(env.phase(), EnvelopePhase::KeyOn);

    // Process through attack/decay to sustain
    let mut reached_sustain = false;
    for _ in 0..50000 {
        env.process();
        if env.phase() == EnvelopePhase::Sustain {
            reached_sustain = true;
            break;
        }
    }
    assert!(reached_sustain, "Should reach sustain phase");

    // Verify sustain level is approximately correct
    let sustain_value = env.value();
    assert!(
        (sustain_value - 0.7).abs() < 0.1,
        "Sustain should be ~0.7, got {}",
        sustain_value
    );

    // Note off -> Release
    env.note_off();
    assert_eq!(env.phase(), EnvelopePhase::Release);

    // Process through release to completion
    let mut completed = false;
    for _ in 0..100000 {
        env.process();
        if !env.is_active() {
            completed = true;
            break;
        }
    }
    assert!(completed, "Envelope should complete");
    assert!(
        env.value() < 0.01,
        "Final value should be near zero, got {}",
        env.value()
    );
}

/// Test that seconds_to_rate produces reasonable rates.
#[test]
fn test_seconds_to_rate_coverage() {
    let sample_rate = 44100.0;

    // Very fast: 1ms -> high rate
    let rate_1ms = seconds_to_rate(0.001, sample_rate);
    assert!(
        rate_1ms >= 80,
        "1ms should give rate >= 80, got {}",
        rate_1ms
    );

    // Fast: 10ms -> high rate
    let rate_10ms = seconds_to_rate(0.01, sample_rate);
    assert!(
        rate_10ms >= 60,
        "10ms should give rate >= 60, got {}",
        rate_10ms
    );

    // Medium: 300ms -> medium rate
    let rate_300ms = seconds_to_rate(0.3, sample_rate);
    assert!(
        (30..=60).contains(&rate_300ms),
        "300ms should give rate 30-60, got {}",
        rate_300ms
    );

    // Slow: 2s -> low rate
    let rate_2s = seconds_to_rate(2.0, sample_rate);
    assert!(rate_2s <= 30, "2s should give rate <= 30, got {}", rate_2s);

    // Very slow: 10s -> very low rate
    let rate_10s = seconds_to_rate(10.0, sample_rate);
    assert!(
        rate_10s <= 15,
        "10s should give rate <= 15, got {}",
        rate_10s
    );
}

/// Test that long envelope times work (previously capped at ~93ms).
#[test]
fn test_long_envelope_times() {
    let sample_rate = 44100.0;

    // This was the bug: Q8 capped envelopes at ~93ms
    // With f32, we should be able to have 5+ second envelopes

    let attack_time = 5.0; // 5 seconds
    let config = FmEnvelopeConfig::adsr(attack_time, 0.1, 0.7, 0.1, sample_rate);
    let mut env = FmEnvelope::with_config(config);
    env.note_on(60);

    // After 1 second, we should still be in attack and not at full level
    for _ in 0..44100 {
        env.process();
    }

    assert_eq!(
        env.phase(),
        EnvelopePhase::KeyOn,
        "Should still be in attack after 1s of 5s attack"
    );
    let level_at_1s = env.value();

    // After 3 seconds total, still in attack but higher
    for _ in 0..(2 * 44100) {
        env.process();
    }
    let level_at_3s = env.value();

    assert!(
        level_at_3s > level_at_1s,
        "Level should increase: 1s={}, 3s={}",
        level_at_1s,
        level_at_3s
    );
    assert!(
        level_at_3s < 0.95,
        "Should not be at max yet after 3s of 5s attack, got {}",
        level_at_3s
    );
}

/// Test that 48kHz sample rate produces correct timing.
#[test]
fn test_48khz_timing() {
    let sample_rate = 48000.0;
    let attack_time = 0.5; // 500ms

    let config = FmEnvelopeConfig::adsr(attack_time, 0.1, 0.7, 0.1, sample_rate);
    let mut env = FmEnvelope::with_config(config);
    env.note_on(60);

    let mut samples = 0;
    while env.value() < 0.9 && samples < 100000 {
        env.process();
        samples += 1;
    }

    let measured_time = samples as f32 / sample_rate;
    let tolerance = 0.3;

    assert!(
        (measured_time - attack_time).abs() < attack_time * tolerance,
        "48kHz attack time mismatch: expected {}ms, got {}ms",
        attack_time * 1000.0,
        measured_time * 1000.0
    );
}

/// Test custom segment configuration with specific rate values.
#[test]
fn test_custom_segment_rates() {
    let sample_rate = 44100.0;

    // Configure with specific rate value
    let mut config = FmEnvelopeConfig::default_with_sample_rate(sample_rate);
    config.key_on_segments[0] = Segment::new(50, 99); // Medium rate attack to max

    let mut env = FmEnvelope::with_config(config);
    env.note_on(60);

    // Count samples to reach 90%
    let mut samples = 0;
    while env.value() < 0.9 && samples < 50000 {
        env.process();
        samples += 1;
    }

    // Rate 50 should give medium speed - not instant but not too slow
    let time = samples as f32 / sample_rate;
    assert!(
        time > 0.1 && time < 2.0,
        "Rate 50 attack should be 0.1-2s, got {}s",
        time
    );
}

/// Test that rate scaling affects envelope timing.
#[test]
fn test_rate_scaling_effect() {
    let sample_rate = 44100.0;

    let mut config = FmEnvelopeConfig::default_with_sample_rate(sample_rate);
    config.rate_scaling = 7; // Maximum rate scaling
    config.key_on_segments[0] = Segment::new(40, 99); // Slower attack to see the effect

    // Low note
    let mut env_low = FmEnvelope::with_config(config);
    env_low.note_on(36); // C2

    // High note
    let mut env_high = FmEnvelope::with_config(config);
    env_high.note_on(96); // C7

    // Count samples to reach 90%
    let mut samples_low = 0;
    while env_low.value() < 0.9 && samples_low < 200000 {
        env_low.process();
        samples_low += 1;
    }

    let mut samples_high = 0;
    while env_high.value() < 0.9 && samples_high < 200000 {
        env_high.process();
        samples_high += 1;
    }

    // High note should be faster due to rate scaling
    assert!(
        samples_high < samples_low,
        "High note should be faster with rate scaling: high={}samples, low={}samples",
        samples_high,
        samples_low
    );
}

// =============================================================================
// MSFA-Faithful Attack Behavior Tests
// =============================================================================

/// Test that JUMP_TARGET is at the correct dB level (~-56dB).
#[test]
fn test_jump_target_db_level() {
    use rigel_modulation::envelope::JUMP_TARGET;

    // JUMP_TARGET should be ~-56dB from full scale (40dB above minimum)
    let db = 20.0 * libm::log10f(JUMP_TARGET);
    assert!(
        (db - (-55.8)).abs() < 2.0,
        "JUMP_TARGET should be ~-56dB, got {}dB",
        db
    );

    // Verify it's much smaller than 0.5 (the old incorrect value)
    // This is a compile-time check to ensure the constant wasn't accidentally changed
    const _: () = assert!(JUMP_TARGET < 0.01);
}

/// Test exponential approach curve shape (fast start, slow finish).
#[test]
fn test_attack_exponential_curve() {
    let sample_rate = 44100.0;
    // Use a moderate attack time for observable curve shape
    let config = FmEnvelopeConfig::adsr(0.1, 0.01, 0.99, 0.1, sample_rate);
    let mut env = FmEnvelope::with_config(config);
    env.note_on(60);

    // Sample at 25%, 50%, 75% of expected attack time (~100ms = 4410 samples)
    let checkpoint_samples = 4410 / 4;

    let mut prev_level = env.value();
    let mut rises = Vec::new();

    for _i in 1..=4 {
        // Process to next checkpoint
        for _ in 0..checkpoint_samples {
            env.process();
        }
        let level = env.value();
        rises.push(level - prev_level);
        prev_level = level;
    }

    // With exponential approach, each quarter should rise LESS than the previous
    // (fast at start, slow at end)
    assert!(
        rises[0] > rises[1],
        "First quarter should rise faster than second: {} vs {}",
        rises[0],
        rises[1]
    );
    assert!(
        rises[1] > rises[2],
        "Second quarter should rise faster than third: {} vs {}",
        rises[1],
        rises[2]
    );
}

/// Test slow attack builds from near-silence (not 50%!).
#[test]
fn test_slow_attack_from_silence() {
    let sample_rate = 44100.0;
    let attack_time = 5.0; // 5 seconds

    let config = FmEnvelopeConfig::adsr(attack_time, 0.1, 0.7, 0.1, sample_rate);
    let mut env = FmEnvelope::with_config(config);
    env.note_on(60);

    // After 1 second (20% of attack), should be building but NOT at 50%
    for _ in 0..44100 {
        env.process();
    }

    let level_1s = env.value();

    // With the old broken JUMP_TARGET=0.5, this would be ~50%+
    // With correct JUMP_TARGET=0.00163, should be much lower
    assert!(
        level_1s < 0.5,
        "1s into 5s attack should be <50%, got {:.1}%",
        level_1s * 100.0
    );

    // But should have made some progress (not stuck at jump target)
    assert!(
        level_1s > 0.01,
        "1s into 5s attack should have made progress, got {:.3}%",
        level_1s * 100.0
    );
}

/// Test that very slow attacks complete correctly.
#[test]
fn test_very_slow_attack_completes() {
    let sample_rate = 44100.0;
    let attack_time = 10.0; // 10 seconds

    let config = FmEnvelopeConfig::adsr(attack_time, 0.01, 0.99, 0.1, sample_rate);
    let mut env = FmEnvelope::with_config(config);
    env.note_on(60);

    // Process 12 seconds (more than attack time + some decay)
    for _ in 0..(12 * 44100) {
        env.process();
    }

    // Should have completed attack and be at/near sustain level
    let level = env.value();
    assert!(
        level > 0.9,
        "Should reach sustain level (~0.99) after 12s, got {}",
        level
    );
}

/// Test that attack timing is still reasonably accurate with exponential approach.
#[test]
fn test_attack_timing_with_exponential() {
    let sample_rate = 44100.0;

    // Test a few different attack times
    let test_cases = [(0.1, 0.5), (0.5, 0.4), (1.0, 0.35)]; // (time, tolerance)

    for (attack_time, tolerance) in test_cases {
        let config = FmEnvelopeConfig::adsr(attack_time, 0.01, 0.99, 0.1, sample_rate);
        let mut env = FmEnvelope::with_config(config);
        env.note_on(60);

        // Count samples to reach 90% of target (0.99 * 0.9 ≈ 0.89)
        let mut samples = 0;
        while env.value() < 0.89 && samples < 500000 {
            env.process();
            samples += 1;
        }

        let measured_time = samples as f32 / sample_rate;

        // Allow wider tolerance due to exponential curve shape
        assert!(
            (measured_time - attack_time).abs() < attack_time * tolerance,
            "Attack time {}s: expected ~{}ms, got {}ms",
            attack_time,
            attack_time * 1000.0,
            measured_time * 1000.0
        );
    }
}

/// Test that minimum segment time is enforced across all sample rates.
#[test]
fn test_min_segment_time_enforced() {
    use rigel_modulation::envelope::{
        get_static_count, max_rate_for_sample_rate, MIN_SEGMENT_TIME_SECONDS,
    };

    for &sr in &[22050.0, 44100.0, 48000.0, 96000.0] {
        let max_rate = max_rate_for_sample_rate(sr);
        let min_samples = get_static_count(max_rate, sr);
        let min_time = min_samples as f32 / sr;

        assert!(
            min_time >= MIN_SEGMENT_TIME_SECONDS * 0.8,
            "At {}Hz: min time {}ms should be >= {}ms",
            sr,
            min_time * 1000.0,
            MIN_SEGMENT_TIME_SECONDS * 800.0
        );
    }
}

/// Test that various rate/note/scaling combinations don't produce clicks.
#[test]
fn test_rate_scaling_combinations_no_clicks() {
    use rigel_modulation::envelope::{calculate_increment_f32_scaled, MIN_SEGMENT_TIME_SECONDS};

    // Test problematic combinations at various sample rates
    let test_cases = [
        (85, 127, 7),
        (90, 96, 7),
        (95, 84, 5),
        (99, 60, 4),
        (99, 127, 7),
    ];

    for &sr in &[22050.0, 44100.0, 48000.0, 96000.0] {
        for (rate, note, scaling) in test_cases {
            let inc = calculate_increment_f32_scaled(rate, note, scaling, sr);
            let samples = (1.0 / inc) as u32;
            let time = samples as f32 / sr;

            assert!(
                time >= MIN_SEGMENT_TIME_SECONDS * 0.8,
                "At {}Hz: rate {} + note {} + scaling {} gives {}ms, want >= {}ms",
                sr,
                rate,
                note,
                scaling,
                time * 1000.0,
                MIN_SEGMENT_TIME_SECONDS * 800.0
            );
        }
    }
}
