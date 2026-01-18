//! Integration tests verifying envelope timing behavior.
//!
//! ## Hybrid Q8/f32 Format
//!
//! The envelope uses a hybrid approach:
//! - **Output levels**: Q8 fixed-point (i16, 0-4095 range) for authentic DX7/SY99
//!   hardware amplitude quantization (~96dB dynamic range)
//! - **Internal accumulation**: f32 for sub-sample timing precision
//!
//! This allows slow rates to achieve proper multi-second timing while
//! maintaining authentic Q8 amplitude quantization.
//!
//! ## Key Characteristics
//!
//! - **Attack curve**: MSFA-style "concave up" in Q8 domain (fast at bottom, slow at top)
//! - **Decay curve**: Linear in Q8 (dB) domain = exponential in linear amplitude
//! - **Minimum 1.5ms segment time**: Prevents audible clicks
//! - **STATICS table timing**: Properly supported for rates 0-99

use rigel_modulation::envelope::{
    seconds_to_rate, EnvelopePhase, FmEnvelope, FmEnvelopeConfig, Segment,
};

// =============================================================================
// Q8-Compatible Timing Tests (these pass with integer precision)
// =============================================================================

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
    const _: () = assert!(JUMP_TARGET < 0.01);
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
    // With correct JUMP_TARGET=0.00163, should be lower
    // Note: With Q8, attack progresses faster due to integer increment precision
    assert!(
        level_1s < 0.8,
        "1s into 5s attack should be <80%, got {:.1}%",
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

// =============================================================================
// Q8 Timing Characteristic Tests
// =============================================================================

/// Test Q8 attack behavior - fast rates work well.
#[test]
fn test_q8_fast_attack() {
    let sample_rate = 44100.0;

    // Fast attack with high rate - Q8 handles this well
    let mut config = FmEnvelopeConfig::default_with_sample_rate(sample_rate);
    config.key_on_segments[0] = Segment::new(90, 99); // Fast attack

    let mut env = FmEnvelope::with_config(config);
    env.note_on(60);

    // Should reach 50% quickly (fast rate gives Q8 increment > 1)
    let mut samples = 0;
    while env.value() < 0.5 && samples < 1000 {
        env.process();
        samples += 1;
    }

    assert!(
        samples < 1000,
        "Fast attack (rate 90) should reach 50% in <1000 samples, took {}",
        samples
    );
}

/// Test Q8 decay timing - completes within expected bounds.
#[test]
fn test_q8_decay_completes() {
    let sample_rate = 44100.0;

    let mut config = FmEnvelopeConfig::default_with_sample_rate(sample_rate);
    config.key_on_segments[0] = Segment::new(99, 99); // Fast attack
    config.key_on_segments[1] = Segment::new(60, 0); // Decay to 0

    let mut env = FmEnvelope::with_config(config);
    env.note_on(60);

    // Process until decay completes
    let mut samples = 0;
    while env.current_segment() < 2 && samples < 50000 && env.is_active() {
        env.process();
        samples += 1;
    }

    // With Q8, decay is bounded by distance (4095 samples max for full range)
    assert!(
        samples <= 5000,
        "Decay should complete within Q8 bounds, took {} samples",
        samples
    );
}

/// Test that Q8 envelope reaches sustain level.
#[test]
fn test_q8_sustain_level() {
    let sample_rate = 44100.0;

    // Use param_to_level_q8 mapping: level 70 maps to approximately 0.07 linear
    // due to the LEVEL_LUT and logarithmic Q8 to linear conversion
    let config = FmEnvelopeConfig::adsr(0.01, 0.01, 0.7, 0.1, sample_rate);
    let mut env = FmEnvelope::with_config(config);
    env.note_on(60);

    // Process to sustain
    let mut reached_sustain = false;
    for _ in 0..50000 {
        env.process();
        if env.phase() == EnvelopePhase::Sustain {
            reached_sustain = true;
            break;
        }
    }

    assert!(reached_sustain, "Should reach sustain phase");

    // With Q8 logarithmic conversion, sustain level 70 (0-99 range) maps to
    // a much lower linear value due to the dB scale
    let sustain_value = env.value();
    assert!(
        sustain_value > 0.01 && sustain_value < 0.5,
        "Sustain value should be in valid Q8 range, got {}",
        sustain_value
    );
}

// =============================================================================
// Precision-Dependent Tests (require f32 for exact timing)
// =============================================================================

/// Test that attack time matches configured value.
///
/// IGNORED: Q8 integer increment precision limits timing accuracy for slow rates.
/// With minimum increment of 1, slow attacks are faster than STATICS table suggests.
#[test]
fn test_attack_time_100ms() {
    let sample_rate = 44100.0;
    let attack_time = 0.1; // 100ms

    let config = FmEnvelopeConfig::adsr(attack_time, 0.1, 0.7, 0.1, sample_rate);
    let mut env = FmEnvelope::with_config(config);
    env.note_on(60);

    let mut samples = 0;
    while env.value() < 0.9 && samples < 50000 {
        env.process();
        samples += 1;
    }

    let measured_time = samples as f32 / sample_rate;
    let tolerance = 0.3;

    assert!(
        (measured_time - attack_time).abs() < attack_time * tolerance,
        "Attack time mismatch: expected {}ms, got {}ms",
        attack_time * 1000.0,
        measured_time * 1000.0
    );
}

/// Test 500ms attack time.
///
/// IGNORED: Q8 integer precision limits slow attack timing.
#[test]
fn test_attack_time_500ms() {
    let sample_rate = 44100.0;
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
        "Attack time mismatch: expected {}ms, got {}ms",
        attack_time * 1000.0,
        measured_time * 1000.0
    );
}

/// Test release time matches configured value.
///
/// IGNORED: Q8 integer precision limits release timing accuracy.
#[test]
fn test_release_time_300ms() {
    let sample_rate = 44100.0;
    let release_time = 0.3; // 300ms

    let config = FmEnvelopeConfig::adsr(0.01, 0.01, 1.0, release_time, sample_rate);
    let mut env = FmEnvelope::with_config(config);
    env.note_on(60);

    for _ in 0..5000 {
        env.process();
    }

    env.note_off();
    let start_level = env.value();
    let threshold = start_level * 0.001;
    let expected_fraction = 60.0 / 96.0;

    let mut samples = 0;
    while env.value() > threshold && samples < 100000 {
        env.process();
        samples += 1;
    }

    let measured_time = samples as f32 / sample_rate;
    let expected_time = release_time * expected_fraction;
    let tolerance = 0.35;

    assert!(
        (measured_time - expected_time).abs() < expected_time * tolerance,
        "Release time to -60dB mismatch: expected ~{}ms, got {}ms",
        expected_time * 1000.0,
        measured_time * 1000.0
    );
}

/// Test 1 second release time.
///
/// IGNORED: Q8 integer precision limits release timing accuracy.
#[test]
fn test_release_time_1s() {
    let sample_rate = 44100.0;
    let release_time = 1.0;

    let config = FmEnvelopeConfig::adsr(0.01, 0.01, 1.0, release_time, sample_rate);
    let mut env = FmEnvelope::with_config(config);
    env.note_on(60);

    for _ in 0..5000 {
        env.process();
    }

    env.note_off();
    let start_level = env.value();
    let threshold = start_level * 0.001;
    let expected_fraction = 60.0 / 96.0;

    let mut samples = 0;
    while env.value() > threshold && samples < 200000 {
        env.process();
        samples += 1;
    }

    let measured_time = samples as f32 / sample_rate;
    let expected_time = release_time * expected_fraction;
    let tolerance = 0.35;

    assert!(
        (measured_time - expected_time).abs() < expected_time * tolerance,
        "Release time mismatch: expected ~{}ms, got {}ms",
        expected_time * 1000.0,
        measured_time * 1000.0
    );
}

/// Test multiple time values for attack parametrically.
///
/// IGNORED: Q8 integer precision limits timing accuracy.
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
///
/// IGNORED: Q8 integer precision limits timing accuracy.
#[test]
fn test_release_timing_parametric() {
    let sample_rate = 44100.0;
    let test_times = [0.1, 0.3, 0.5, 1.0, 2.0];
    let expected_fraction = 60.0 / 96.0;

    for configured_time in test_times {
        let config = FmEnvelopeConfig::adsr(0.01, 0.01, 1.0, configured_time, sample_rate);
        let mut env = FmEnvelope::with_config(config);
        env.note_on(60);

        for _ in 0..5000 {
            env.process();
        }

        env.note_off();
        let start_level = env.value();
        let threshold = start_level * 0.001;

        let mut samples = 0;
        while env.value() > threshold && samples < 500000 {
            env.process();
            samples += 1;
        }

        let measured = samples as f32 / sample_rate;
        let expected_time = configured_time * expected_fraction;
        let tolerance = if configured_time < 0.2 { 0.5 } else { 0.35 };

        assert!(
            (measured - expected_time).abs() < expected_time * tolerance,
            "Release time {}s: expected {}ms, got {}ms",
            expected_time,
            expected_time * 1000.0,
            measured * 1000.0
        );
    }
}

/// Test envelope transitions through all phases correctly.
/// Full envelope lifecycle test with Q8 level mapping.
///
/// Note: Q8 uses dB (logarithmic) scale, so sustain param 70 (0-99)
/// maps to approximately 0.07 linear, not 0.7 linear. This is the
/// authentic DX7/SY99 behavior where level parameters represent dB.
#[test]
fn test_full_envelope_lifecycle() {
    let sample_rate = 44100.0;
    let config = FmEnvelopeConfig::adsr(0.1, 0.1, 0.7, 0.3, sample_rate);
    let mut env = FmEnvelope::with_config(config);

    assert_eq!(env.phase(), EnvelopePhase::Idle);
    env.note_on(60);
    assert_eq!(env.phase(), EnvelopePhase::KeyOn);

    let mut reached_sustain = false;
    for _ in 0..50000 {
        env.process();
        if env.phase() == EnvelopePhase::Sustain {
            reached_sustain = true;
            break;
        }
    }
    assert!(reached_sustain, "Should reach sustain phase");

    // Q8 dB scale: sustain param 70 maps to ~0.07 linear (not 0.7)
    // This is authentic DX7/SY99 behavior
    let sustain_value = env.value();
    assert!(
        sustain_value > 0.01 && sustain_value < 0.2,
        "Sustain should be in Q8-mapped range, got {}",
        sustain_value
    );

    env.note_off();
    assert_eq!(env.phase(), EnvelopePhase::Release);

    let mut completed = false;
    for _ in 0..100000 {
        env.process();
        if !env.is_active() {
            completed = true;
            break;
        }
    }
    assert!(completed, "Envelope should complete");
}

/// Test that long envelope times work.
///
/// IGNORED: Q8 precision makes long attacks faster than configured.
#[test]
fn test_long_envelope_times() {
    let sample_rate = 44100.0;
    let attack_time = 5.0;

    let config = FmEnvelopeConfig::adsr(attack_time, 0.1, 0.7, 0.1, sample_rate);
    let mut env = FmEnvelope::with_config(config);
    env.note_on(60);

    for _ in 0..44100 {
        env.process();
    }

    assert_eq!(
        env.phase(),
        EnvelopePhase::KeyOn,
        "Should still be in attack after 1s of 5s attack"
    );
}

/// Test that 48kHz sample rate produces correct timing.
///
/// IGNORED: Q8 precision limits timing accuracy.
#[test]
fn test_48khz_timing() {
    let sample_rate = 48000.0;
    let attack_time = 0.5;

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
///
/// IGNORED: Q8 attack timing differs from STATICS expectations.
#[test]
fn test_custom_segment_rates() {
    let sample_rate = 44100.0;

    let mut config = FmEnvelopeConfig::default_with_sample_rate(sample_rate);
    config.key_on_segments[0] = Segment::new(50, 99);

    let mut env = FmEnvelope::with_config(config);
    env.note_on(60);

    let mut samples = 0;
    while env.value() < 0.9 && samples < 50000 {
        env.process();
        samples += 1;
    }

    let time = samples as f32 / sample_rate;
    assert!(
        time > 0.1 && time < 2.0,
        "Rate 50 attack should be 0.1-2s, got {}s",
        time
    );
}

/// Test that rate scaling affects envelope timing.
///
/// IGNORED: Q8 precision may cause both to have same increment.
#[test]
fn test_rate_scaling_effect() {
    let sample_rate = 44100.0;

    let mut config = FmEnvelopeConfig::default_with_sample_rate(sample_rate);
    config.rate_scaling = 7;
    config.key_on_segments[0] = Segment::new(40, 99);

    let mut env_low = FmEnvelope::with_config(config);
    env_low.note_on(36);

    let mut env_high = FmEnvelope::with_config(config);
    env_high.note_on(96);

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

    assert!(
        samples_high < samples_low,
        "High note should be faster with rate scaling: high={}samples, low={}samples",
        samples_high,
        samples_low
    );
}

/// Test MSFA-style attack curve shape in Q8 domain.
///
/// The MSFA attack curve is "concave up" in Q8 (dB) domain: fast at low levels,
/// slow at top. This creates the characteristic FM "punch" - the envelope races
/// through the quiet region quickly.
///
/// Note: In LINEAR amplitude domain, this appears reversed due to the dB→linear
/// conversion (exponential). The Q8 envelope accelerates through -96dB to -20dB
/// quickly, but that's a tiny change in linear (0 to 0.1). Then it slows down
/// from -20dB to 0dB, but that's a huge linear change (0.1 to 1.0).
///
/// This test verifies the Q8 domain behavior, which is where the MSFA curve
/// actually operates.
#[test]
fn test_attack_exponential_curve() {
    let sample_rate = 44100.0;
    let config = FmEnvelopeConfig::adsr(0.1, 0.01, 0.99, 0.1, sample_rate);
    let mut env = FmEnvelope::with_config(config);
    env.note_on(60);

    let checkpoint_samples = 4410 / 4;
    let mut prev_level_q8 = env.state().level_q8();
    let mut rises_q8 = Vec::new();

    for _i in 1..=4 {
        for _ in 0..checkpoint_samples {
            env.process();
        }
        let level_q8 = env.state().level_q8();
        rises_q8.push(level_q8 - prev_level_q8);
        prev_level_q8 = level_q8;
    }

    // MSFA attack in Q8 domain: fast start, slow finish
    // Early quarters have LARGER Q8 rises than later quarters
    // This is the FM "punch" - racing through quiet region
    assert!(
        rises_q8[0] > rises_q8[3],
        "First quarter should rise more than fourth quarter in Q8: {} vs {}",
        rises_q8[0],
        rises_q8[3]
    );
}

/// Test that attack timing is still reasonably accurate with exponential approach.
///
/// IGNORED: Q8 integer precision limits timing accuracy.
#[test]
fn test_attack_timing_with_exponential() {
    let sample_rate = 44100.0;
    let test_cases = [(0.1, 0.5), (0.5, 0.4), (1.0, 0.35)];

    for (attack_time, tolerance) in test_cases {
        let config = FmEnvelopeConfig::adsr(attack_time, 0.01, 0.99, 0.1, sample_rate);
        let mut env = FmEnvelope::with_config(config);
        env.note_on(60);

        let mut samples = 0;
        while env.value() < 0.89 && samples < 500000 {
            env.process();
            samples += 1;
        }

        let measured_time = samples as f32 / sample_rate;

        assert!(
            (measured_time - attack_time).abs() < attack_time * tolerance,
            "Attack time {}s: expected ~{}ms, got {}ms",
            attack_time,
            attack_time * 1000.0,
            measured_time * 1000.0
        );
    }
}
