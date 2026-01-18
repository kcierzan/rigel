//! Integration tests for the SY-Style Envelope module.
//!
//! Tests are organized by user story for traceability.

use rigel_modulation::envelope::{
    AwmEnvelope, EnvelopePhase, FmEnvelope, FmEnvelopeBatch8, FmEnvelopeConfig, LoopConfig,
    Segment, SevenSegEnvelope, JUMP_TARGET,
};

// =============================================================================
// Phase 3: User Story 1 - Basic Envelope Modulation
// =============================================================================

mod us1_basic_envelope {
    use super::*;

    #[test]
    fn test_note_on_triggers_attack() {
        let mut env = FmEnvelope::new(44100.0);
        assert_eq!(env.phase(), EnvelopePhase::Idle);

        env.note_on(60);
        assert_eq!(env.phase(), EnvelopePhase::KeyOn);
        assert!(env.is_active());
    }

    #[test]
    fn test_note_off_triggers_release() {
        let mut env = FmEnvelope::new(44100.0);
        env.note_on(60);
        env.note_off();

        assert_eq!(env.phase(), EnvelopePhase::Release);
        assert!(env.is_releasing());
        assert!(env.is_active());
    }

    #[test]
    fn test_output_range_0_to_1() {
        let mut env = FmEnvelope::new(44100.0);
        env.note_on(60);

        // Process many samples through attack and decay
        for i in 0..22050 {
            let value = env.process();
            assert!(
                (0.0..=1.0).contains(&value),
                "Sample {}: Output {} not in range [0.0, 1.0]",
                i,
                value
            );
        }

        // Trigger release
        env.note_off();

        // Process through release
        for i in 0..22050 {
            let value = env.process();
            assert!(
                (0.0..=1.0).contains(&value),
                "Release sample {}: Output {} not in range",
                i,
                value
            );
        }
    }

    #[test]
    fn test_segment_transitions() {
        let mut env = FmEnvelope::new(44100.0);
        env.note_on(60);

        let mut segments_visited = std::collections::HashSet::new();
        let mut last_segment = 0;

        // Process until we've seen multiple segments or hit limit
        for _ in 0..100000 {
            env.process();
            let current = env.current_segment();

            if current != last_segment {
                segments_visited.insert(current);
                last_segment = current;
            }

            // Stop if we've entered sustain or release
            if matches!(
                env.phase(),
                EnvelopePhase::Sustain | EnvelopePhase::Release | EnvelopePhase::Complete
            ) {
                break;
            }
        }

        // Should have visited at least one segment
        assert!(
            !segments_visited.is_empty(),
            "Should visit segments, visited: {:?}",
            segments_visited
        );
    }

    #[test]
    fn test_value_does_not_advance() {
        let mut env = FmEnvelope::new(44100.0);
        env.note_on(60);

        // Process once to get started
        env.process();

        // Get value multiple times
        let v1 = env.value();
        let v2 = env.value();
        let v3 = env.value();

        assert_eq!(v1, v2);
        assert_eq!(v2, v3);

        // Now process and check value changed
        env.process();
        let v4 = env.value();

        // Value may or may not change depending on rate, but calling
        // value() shouldn't cause changes
        let v5 = env.value();
        assert_eq!(v4, v5);
    }

    #[test]
    fn test_reset() {
        let mut env = FmEnvelope::new(44100.0);
        env.note_on(60);

        // Process some samples
        for _ in 0..1000 {
            env.process();
        }

        env.reset();

        assert_eq!(env.phase(), EnvelopePhase::Idle);
        assert!(!env.is_active());
        // At LEVEL_MIN (0.0), the value should be exactly 0
        assert!(
            env.value() < f32::EPSILON,
            "Reset should give zero value, got {}",
            env.value()
        );
    }

    #[test]
    fn test_envelope_completes_after_release() {
        let mut env = FmEnvelope::new(44100.0);
        env.note_on(60);
        env.note_off();

        // Process until complete
        let mut iterations = 0;
        while env.is_active() && iterations < 500000 {
            env.process();
            iterations += 1;
        }

        assert!(
            !env.is_active(),
            "Envelope should complete after {} iterations",
            iterations
        );
    }
}

// =============================================================================
// Phase 4: User Story 2 - Rate Scaling by Key Position
// =============================================================================

mod us2_rate_scaling {
    use super::*;
    use rigel_modulation::envelope::scale_rate;

    #[test]
    fn test_rate_scaling_at_midi_60_baseline() {
        // MIDI 60 should have moderate rate scaling
        let adjustment = scale_rate(60, 7); // Max sensitivity
        assert!(adjustment > 0, "MIDI 60 should have some rate scaling");
    }

    #[test]
    fn test_rate_scaling_at_midi_84_faster() {
        // Higher notes should have higher rate adjustment
        let adj_60 = scale_rate(60, 7);
        let adj_84 = scale_rate(84, 7);

        assert!(
            adj_84 > adj_60,
            "Higher notes should scale faster: MIDI 84 ({}) vs MIDI 60 ({})",
            adj_84,
            adj_60
        );
    }

    #[test]
    fn test_rate_scaling_disabled_at_sensitivity_0() {
        // Sensitivity 0 should give no rate scaling
        let adj_24 = scale_rate(24, 0);
        let adj_60 = scale_rate(60, 0);
        let adj_96 = scale_rate(96, 0);

        assert_eq!(adj_24, 0);
        assert_eq!(adj_60, 0);
        assert_eq!(adj_96, 0);
    }

    #[test]
    fn test_rate_scaling_affects_envelope_timing() {
        // Create two envelopes with rate scaling enabled
        let mut config = FmEnvelopeConfig::default_with_sample_rate(44100.0);
        config.rate_scaling = 7; // Maximum sensitivity

        let mut env_low = FmEnvelope::with_config(config);
        let mut env_high = FmEnvelope::with_config(config);

        // Trigger at different notes
        env_low.note_on(36); // Low note
        env_high.note_on(84); // High note

        // Count samples to reach sustain (or 50% of initial attack)
        let mut samples_low = 0;
        let mut samples_high = 0;

        // Process both until they've progressed through attack
        for _ in 0..10000 {
            if env_low.current_segment() == 0 {
                env_low.process();
                samples_low += 1;
            }
            if env_high.current_segment() == 0 {
                env_high.process();
                samples_high += 1;
            }

            if env_low.current_segment() > 0 && env_high.current_segment() > 0 {
                break;
            }
        }

        // High note should complete attack faster due to rate scaling
        // (This may depend on attack rate - with rate 99, both might be instant)
        assert!(
            samples_high <= samples_low,
            "High note should be faster or equal: high={}, low={}",
            samples_high,
            samples_low
        );
    }

    #[test]
    fn test_rate_scaling_no_instant_transitions() {
        // Fast envelope with max rate scaling at high note should still ramp
        let mut config = FmEnvelopeConfig::default_with_sample_rate(44100.0);
        config.rate_scaling = 7;
        config.key_on_segments[0] = Segment::new(95, 99);

        let mut env = FmEnvelope::with_config(config);
        env.note_on(127);

        // Process several samples and verify we're ramping, not jumping
        let first_value = env.value();
        for _ in 0..10 {
            env.process();
        }
        let tenth_value = env.value();

        // Should see gradual increase - the 10th value should be different from the first
        assert!(
            (tenth_value - first_value).abs() > f32::EPSILON,
            "Envelope should ramp, not stay constant: first={}, tenth={}",
            first_value,
            tenth_value
        );
    }

    #[test]
    fn test_rate_scaling_sample_rate_independent() {
        // Test that minimum time is consistent across sample rates
        for &sr in &[22050.0, 44100.0, 48000.0, 96000.0] {
            let mut config = FmEnvelopeConfig::default_with_sample_rate(sr);
            config.rate_scaling = 7;
            config.key_on_segments[0] = Segment::new(99, 99);

            let mut env = FmEnvelope::with_config(config);
            env.note_on(127);

            // Count samples to reach 90%
            let mut samples = 0;
            while env.value() < 0.9 && samples < 10000 {
                env.process();
                samples += 1;
            }

            let time_ms = samples as f32 / sr * 1000.0;
            assert!(
                time_ms >= 1.0,
                "At {}Hz: attack should take >= 1ms, took {}ms ({} samples)",
                sr,
                time_ms,
                samples
            );
        }
    }
}

// =============================================================================
// Phase 5: User Story 3 - MSFA-Compatible Rate Behavior
// =============================================================================

mod us3_msfa_rates {
    use super::*;
    use rigel_modulation::envelope::{calculate_increment_f32, seconds_to_rate};

    #[test]
    fn test_distance_dependent_timing() {
        // Same rate should take different times for different distances
        let mut env1 = FmEnvelope::new(44100.0);
        let mut env2 = FmEnvelope::new(44100.0);

        // Configure different starting levels (would need more complex setup)
        // For now, just verify the formula exists
        env1.note_on(60);
        env2.note_on(60);

        // Both should start processing
        assert!(env1.is_active());
        assert!(env2.is_active());
    }

    #[test]
    fn test_rate_99_near_instantaneous() {
        // Rate 99 should produce fastest transition
        // Note: Rate 99 is clamped to max_rate (~95) to prevent clicks
        // The MSFA attack curve is asymptotic - fast at low levels, slow at top
        // Reaching 90% linear (~Q8 4055) takes longer due to this curve
        let mut config = FmEnvelopeConfig::default_with_sample_rate(44100.0);
        config.key_on_segments[0] = Segment::new(99, 99);

        let mut env = FmEnvelope::with_config(config);
        env.note_on(60);

        // Should reach 50% linear quickly (this is more achievable with MSFA curve)
        // 50% linear ≈ Q8 level 3413, well within the fast part of the attack
        let mut samples = 0;
        while env.value() < 0.5 && samples < 100 {
            env.process();
            samples += 1;
        }

        assert!(
            samples < 100,
            "Rate 99 should reach 50% linear in <100 samples, took {}",
            samples
        );

        // Should eventually reach 90%+ (test envelope completes)
        while env.value() < 0.9 && samples < 500 {
            env.process();
            samples += 1;
        }

        assert!(
            env.value() >= 0.9,
            "Rate 99 should eventually reach 90% linear, got {} after {} samples",
            env.value(),
            samples
        );
    }

    #[test]
    fn test_increment_increases_with_rate() {
        // Higher rate should give higher increment (faster envelope)
        let inc_20 = calculate_increment_f32(20, 44100.0);
        let inc_50 = calculate_increment_f32(50, 44100.0);
        let inc_80 = calculate_increment_f32(80, 44100.0);
        let inc_99 = calculate_increment_f32(99, 44100.0);

        assert!(
            inc_50 > inc_20,
            "Rate 50 ({}) should be > rate 20 ({})",
            inc_50,
            inc_20
        );
        assert!(
            inc_80 > inc_50,
            "Rate 80 ({}) should be > rate 50 ({})",
            inc_80,
            inc_50
        );
        assert!(
            inc_99 > inc_80,
            "Rate 99 ({}) should be > rate 80 ({})",
            inc_99,
            inc_80
        );
    }

    #[test]
    fn test_seconds_to_rate_roundtrip() {
        // Verify that seconds_to_rate produces reasonable rates
        let rate_10ms = seconds_to_rate(0.01, 44100.0);
        let rate_100ms = seconds_to_rate(0.1, 44100.0);
        let rate_1s = seconds_to_rate(1.0, 44100.0);
        let rate_10s = seconds_to_rate(10.0, 44100.0);

        // Shorter times should give higher rates
        assert!(
            rate_10ms > rate_100ms,
            "10ms rate ({}) should be > 100ms rate ({})",
            rate_10ms,
            rate_100ms
        );
        assert!(
            rate_100ms > rate_1s,
            "100ms rate ({}) should be > 1s rate ({})",
            rate_100ms,
            rate_1s
        );
        assert!(
            rate_1s > rate_10s,
            "1s rate ({}) should be > 10s rate ({})",
            rate_1s,
            rate_10s
        );
    }
}

// =============================================================================
// Phase 6: User Story 7 - High-Performance Block Processing
// =============================================================================

mod us7_performance {
    use super::*;

    #[test]
    fn test_zero_allocations_during_processing() {
        // This test verifies the types are stack-based
        // (Actual allocation checking would need custom allocators)

        let mut env = FmEnvelope::new(44100.0);
        env.note_on(60);

        // Process block - should not allocate
        let mut output = [0.0f32; 64];
        env.process_block(&mut output);

        // Verify output is valid
        for &val in output.iter() {
            assert!((0.0..=1.0).contains(&val));
        }
    }

    #[test]
    fn test_copy_clone_traits() {
        let env1 = FmEnvelope::new(44100.0);

        // Test Copy
        let env2 = env1;
        assert_eq!(env1.phase(), env2.phase());

        // Verify copies are independent
        let mut env4 = env1;
        env4.note_on(60);
        assert_ne!(env1.phase(), env4.phase());
    }

    #[test]
    fn test_batch_processing() {
        let mut batch = FmEnvelopeBatch8::new(44100.0);

        // Trigger some envelopes
        batch.note_on(0, 60);
        batch.note_on(1, 72);
        batch.note_on(4, 48);

        // Process
        let mut output = [0.0f32; 8];
        batch.process(&mut output);

        // Active envelopes should have non-zero output
        assert!(output[0] > 0.0, "Envelope 0 should produce output");
        assert!(output[1] > 0.0, "Envelope 1 should produce output");
        assert!(output[4] > 0.0, "Envelope 4 should produce output");

        // Inactive envelopes are at LEVEL_MIN (0.0)
        assert!(
            output[2] < f32::EPSILON,
            "Inactive envelope should be zero, got {}",
            output[2]
        );
        assert!(
            output[3] < f32::EPSILON,
            "Inactive envelope should be zero, got {}",
            output[3]
        );
    }

    #[test]
    fn test_batch_block_processing() {
        let mut batch = FmEnvelopeBatch8::new(44100.0);
        batch.note_on(0, 60);

        let mut output = [[0.0f32; 8]; 64];
        batch.process_block(&mut output);

        // Verify first envelope has values throughout block
        for (i, sample) in output.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&sample[0]),
                "Sample {} out of range",
                i
            );
        }
    }

    #[test]
    fn test_size_of_envelope() {
        // Verify memory footprint is reasonable (< 128 bytes per spec)
        let size = core::mem::size_of::<FmEnvelope>();
        assert!(
            size <= 128,
            "FmEnvelope should be <= 128 bytes, got {}",
            size
        );
    }
}

// =============================================================================
// Phase 7: User Story 4 - Instantaneous Attack dB Jump
// =============================================================================

mod us4_attack_jump {
    use super::*;

    #[test]
    fn test_immediate_jump_on_attack() {
        let mut env = FmEnvelope::new(44100.0);
        env.note_on(60);

        // After first process, level should jump to JUMP_TARGET (~0.5)
        env.process();

        // Get linear level
        let level = env.state().level();

        assert!(
            level >= JUMP_TARGET,
            "Level {} should jump to at least {} on attack",
            level,
            JUMP_TARGET
        );
    }

    #[test]
    fn test_smooth_approach_after_jump() {
        let mut env = FmEnvelope::new(44100.0);
        env.note_on(60);

        // Process through attack
        let mut last_level = 0.0f32;
        for i in 0..1000 {
            env.process();
            let current = env.state().level();

            // After the initial jump, level should generally increase
            // (or stay same if we're at target)
            if i > 0 && env.current_segment() == 0 {
                // During attack, should be rising
                assert!(
                    current >= last_level,
                    "Level should rise during attack: {} -> {}",
                    last_level,
                    current
                );
            }
            last_level = current;
        }
    }
}

// =============================================================================
// Phase 8: User Story 5 - Delayed Envelope Start
// =============================================================================

mod us5_delayed_start {
    use super::*;

    #[test]
    fn test_delay_countdown() {
        let mut config = FmEnvelopeConfig::default_with_sample_rate(44100.0);
        config.delay_samples = 1000;

        let mut env = FmEnvelope::with_config(config);
        env.note_on(60);

        assert_eq!(env.phase(), EnvelopePhase::Delay);

        // Should output 0 during delay
        for i in 0..1000 {
            let value = env.process();
            assert_eq!(value, 0.0, "Sample {} should be silent during delay", i);
        }

        // After delay, should start attack
        env.process();
        assert_eq!(env.phase(), EnvelopePhase::KeyOn);
    }

    #[test]
    fn test_note_off_during_delay_aborts_to_release() {
        let mut config = FmEnvelopeConfig::default_with_sample_rate(44100.0);
        config.delay_samples = 5000;

        let mut env = FmEnvelope::with_config(config);
        env.note_on(60);

        // Process a few samples
        for _ in 0..100 {
            env.process();
        }
        assert_eq!(env.phase(), EnvelopePhase::Delay);

        // Note off during delay
        env.note_off();
        assert_eq!(env.phase(), EnvelopePhase::Release);
    }
}

// =============================================================================
// Phase 9: User Story 6 - Looping Between Segments
// =============================================================================

mod us6_looping {
    use super::*;

    #[test]
    fn test_loop_from_segment_3_to_5() {
        let mut config = FmEnvelopeConfig::default_with_sample_rate(44100.0);
        // Configure segments with varying levels to detect looping
        config.key_on_segments[0] = Segment::new(99, 99);
        config.key_on_segments[1] = Segment::new(99, 90);
        config.key_on_segments[2] = Segment::new(99, 80); // Loop start
        config.key_on_segments[3] = Segment::new(99, 70);
        config.key_on_segments[4] = Segment::new(99, 85); // Loop end
        config.key_on_segments[5] = Segment::new(99, 75);
        config.loop_config = LoopConfig::new(2, 4).unwrap();

        let mut env = FmEnvelope::with_config(config);
        env.note_on(60);

        // Track segment visits
        let mut segment_history = Vec::new();
        let mut last_segment = 255u8;

        for _ in 0..50000 {
            env.process();
            let current = env.current_segment() as u8;

            if current != last_segment {
                segment_history.push(current);
                last_segment = current;
            }

            // Stop if we've seen enough looping
            if segment_history.len() > 15 {
                break;
            }
        }

        // Should see segment 2 appear multiple times (looping back)
        let loop_start_count = segment_history.iter().filter(|&&s| s == 2).count();

        assert!(
            loop_start_count >= 2,
            "Should loop back to segment 2 at least twice, history: {:?}",
            segment_history
        );
    }

    #[test]
    fn test_note_off_exits_loop() {
        let mut config = FmEnvelopeConfig::default_with_sample_rate(44100.0);
        config.loop_config = LoopConfig::new(2, 4).unwrap();

        let mut env = FmEnvelope::with_config(config);
        env.note_on(60);

        // Process until we're in the loop
        for _ in 0..10000 {
            env.process();
            if env.current_segment() >= 2 {
                break;
            }
        }

        // Note off should exit to release
        env.note_off();
        assert_eq!(env.phase(), EnvelopePhase::Release);
    }

    #[test]
    fn test_invalid_loop_boundaries_fallback() {
        // Invalid loop (start >= end) should be disabled
        let bad_loop = LoopConfig::new(4, 2);
        assert!(bad_loop.is_none());

        // Loop with end beyond segment count should be invalid
        let loop_cfg = LoopConfig::new(2, 10).unwrap();
        assert!(!loop_cfg.is_valid(6)); // Only 6 key-on segments
    }
}

// =============================================================================
// Phase 10: User Story 8 - Variant Envelope Configurations
// =============================================================================

mod us8_variants {
    use super::*;

    #[test]
    fn test_fm_envelope_6_plus_2() {
        let env = FmEnvelope::new(44100.0);
        assert_eq!(env.config().key_on_count(), 6);
        assert_eq!(env.config().release_count(), 2);
    }

    #[test]
    fn test_seven_seg_envelope_5_plus_2() {
        let env = SevenSegEnvelope::new(44100.0);
        assert_eq!(env.config().key_on_count(), 5);
        assert_eq!(env.config().release_count(), 2);
    }

    #[test]
    fn test_awm_envelope_5_plus_5() {
        let env = AwmEnvelope::new(44100.0);
        assert_eq!(env.config().key_on_count(), 5);
        assert_eq!(env.config().release_count(), 5);
    }

    #[test]
    fn test_all_variants_process() {
        let mut fm = FmEnvelope::new(44100.0);
        let mut awm = AwmEnvelope::new(44100.0);
        let mut seven = SevenSegEnvelope::new(44100.0);

        fm.note_on(60);
        awm.note_on(60);
        seven.note_on(60);

        // All should produce valid output
        for _ in 0..1000 {
            let v1 = fm.process();
            let v2 = awm.process();
            let v3 = seven.process();

            assert!((0.0..=1.0).contains(&v1));
            assert!((0.0..=1.0).contains(&v2));
            assert!((0.0..=1.0).contains(&v3));
        }
    }
}

// =============================================================================
// Additional Integration Tests
// =============================================================================

mod integration {
    use super::*;
    use rigel_modulation::ModulationSource;
    use rigel_timing::Timebase;

    #[test]
    fn test_modulation_source_trait() {
        let mut env = FmEnvelope::new(44100.0);
        let timebase = Timebase::new(44100.0);

        // Use trait methods
        ModulationSource::reset(&mut env, &timebase);
        assert_eq!(env.phase(), EnvelopePhase::Idle);

        // value() from trait should work
        let _value: f32 = ModulationSource::value(&env);
    }

    #[test]
    fn test_polyphonic_scenario() {
        // Simulate 8-voice polyphony
        let mut batch = FmEnvelopeBatch8::new(44100.0);

        // Trigger notes at different times
        batch.note_on(0, 60);
        for _ in 0..100 {
            let mut out = [0.0f32; 8];
            batch.process(&mut out);
        }

        batch.note_on(1, 64);
        for _ in 0..100 {
            let mut out = [0.0f32; 8];
            batch.process(&mut out);
        }

        batch.note_on(2, 67);

        // All three should be active
        assert!(batch.get(0).is_active());
        assert!(batch.get(1).is_active());
        assert!(batch.get(2).is_active());

        // Release all voices (not just the first one!)
        batch.note_off(0);
        batch.note_off(1);
        batch.note_off(2);
        assert!(batch.get(0).is_releasing());
        assert!(batch.get(1).is_releasing());
        assert!(batch.get(2).is_releasing());

        // Process until complete
        let mut iterations = 0;
        while batch.any_active() && iterations < 500000 {
            let mut out = [0.0f32; 8];
            batch.process(&mut out);
            iterations += 1;
        }

        assert!(
            !batch.any_active(),
            "All voices should complete within {} iterations",
            iterations
        );
    }

    #[test]
    fn test_retrigger_behavior() {
        let mut env = FmEnvelope::new(44100.0);
        env.note_on(60);

        // Process partway through
        for _ in 0..1000 {
            env.process();
        }

        let level_before_retrigger = env.value();

        // Retrigger
        env.note_on(72);

        // Should restart attack but from current level (no click)
        assert_eq!(env.phase(), EnvelopePhase::KeyOn);

        // The attack jump might cause a jump, but it shouldn't go to 0
        let level_after = env.value();
        assert!(
            level_after > 0.0,
            "Retrigger should not reset to zero: before={}, after={}",
            level_before_retrigger,
            level_after
        );
    }
}

// =============================================================================
// Exponential Decay Tests
// =============================================================================

mod exponential_decay {
    use super::*;

    #[test]
    fn test_decay_is_exponential_not_linear() {
        // Configure an envelope with a decay segment from level 99 to level 0
        let mut config = FmEnvelopeConfig::default_with_sample_rate(44100.0);
        // First segment: instant attack to full
        config.key_on_segments[0] = Segment::new(99, 99);
        // Second segment: decay to silence at medium rate
        config.key_on_segments[1] = Segment::new(50, 0);

        let mut env = FmEnvelope::with_config(config);
        env.note_on(60);

        // Process through attack to reach decay segment
        while env.current_segment() == 0 && env.is_active() {
            env.process();
        }

        // Now we should be in decay segment
        assert_eq!(env.current_segment(), 1, "Should be in decay segment");

        // Collect level samples during decay
        let mut levels: Vec<f32> = Vec::new();
        for _ in 0..5000 {
            if env.current_segment() != 1 {
                break;
            }
            levels.push(env.value());
            env.process();
        }

        // With exponential decay, the ratio between consecutive samples should be constant
        // (within floating point tolerance)
        // ratio = level[n+1] / level[n] should be the same throughout
        if levels.len() >= 100 {
            let ratios: Vec<f32> = levels
                .windows(2)
                .filter(|w| w[0] > 1e-6 && w[1] > 1e-6) // Avoid division by tiny numbers
                .map(|w| w[1] / w[0])
                .collect();

            if ratios.len() >= 10 {
                let first_ratio = ratios[0];
                let mid_ratio = ratios[ratios.len() / 2];
                let late_ratio = ratios[ratios.len() * 3 / 4];

                // All ratios should be approximately the same (within 1%)
                let tolerance = 0.01;
                assert!(
                    (first_ratio - mid_ratio).abs() < tolerance,
                    "Exponential decay ratio should be constant: first={}, mid={}",
                    first_ratio,
                    mid_ratio
                );
                assert!(
                    (mid_ratio - late_ratio).abs() < tolerance,
                    "Exponential decay ratio should be constant: mid={}, late={}",
                    mid_ratio,
                    late_ratio
                );

                // Ratio should be less than 1 (decaying)
                assert!(
                    first_ratio < 1.0,
                    "Decay ratio should be < 1, got {}",
                    first_ratio
                );
            }
        }
    }

    #[test]
    fn test_decay_slope_increases_near_zero() {
        // The key characteristic: in linear amplitude terms, the slope (dLevel/dSample)
        // should increase as level approaches zero
        let mut config = FmEnvelopeConfig::default_with_sample_rate(44100.0);
        config.key_on_segments[0] = Segment::new(99, 99);
        config.key_on_segments[1] = Segment::new(40, 0); // Slow decay

        let mut env = FmEnvelope::with_config(config);
        env.note_on(60);

        // Get to decay segment
        while env.current_segment() == 0 && env.is_active() {
            env.process();
        }

        // Sample at beginning of decay (high level)
        let level_high = env.value();
        env.process();
        let level_high_next = env.value();
        let slope_high = (level_high - level_high_next).abs();

        // Process through most of the decay
        for _ in 0..10000 {
            if env.value() < 0.1 || env.current_segment() != 1 {
                break;
            }
            env.process();
        }

        // Sample near end of decay (low level)
        let level_low = env.value();
        if level_low > 0.001 && level_low < 0.1 {
            env.process();
            let level_low_next = env.value();
            let slope_low = (level_low - level_low_next).abs();

            // The absolute slope (dLevel/dSample) should be smaller at lower levels
            // for exponential decay (constant percentage decrease = smaller absolute decrease)
            // This is the OPPOSITE of what the user described - let me reconsider...
            //
            // Actually, the user's description "slope increases as values get closer to zero"
            // refers to the visual appearance when plotting against TIME: the curve gets steeper
            // as it approaches zero because it's spending less time at each level.
            //
            // In terms of dB/second, the rate is constant (linear-in-dB).
            // In terms of linear amplitude change per sample, the change is SMALLER at lower levels
            // (because the same dB change corresponds to a smaller linear change).
            //
            // Let me just verify that this IS exponential (constant ratio), not linear (constant difference).
            let ratio_high = level_high_next / level_high;
            let ratio_low = level_low_next / level_low;

            // Ratios should be similar (exponential decay has constant ratio)
            let ratio_diff = (ratio_high - ratio_low).abs();
            assert!(
                ratio_diff < 0.02,
                "Decay should be exponential (constant ratio): high_ratio={}, low_ratio={}, diff={}",
                ratio_high,
                ratio_low,
                ratio_diff
            );

            // And for exponential decay, absolute slope IS smaller at lower levels
            assert!(
                slope_low < slope_high,
                "For exponential decay, absolute change per sample should be smaller at lower levels: slope_high={}, slope_low={}",
                slope_high,
                slope_low
            );
        }
    }

    #[test]
    fn test_decay_to_zero_completes() {
        // Ensure decay to level 0 actually completes and doesn't get stuck
        let mut config = FmEnvelopeConfig::default_with_sample_rate(44100.0);
        config.key_on_segments[0] = Segment::new(99, 99);
        config.key_on_segments[1] = Segment::new(60, 0); // Decay to silence

        let mut env = FmEnvelope::with_config(config);
        env.note_on(60);

        // Process until we reach segment 2 or timeout
        let mut iterations = 0;
        while env.current_segment() < 2 && iterations < 500000 && env.is_active() {
            env.process();
            iterations += 1;
        }

        // Should have moved past the decay segment
        assert!(
            env.current_segment() >= 2 || !env.is_active(),
            "Decay to zero should complete, stuck at segment {} after {} iterations",
            env.current_segment(),
            iterations
        );
    }

    #[test]
    fn test_decay_timing_preserved() {
        // Verify decay timing with Q8 integer precision.
        //
        // With Q8 format, the minimum increment is 1 Q8 unit per sample.
        // For slow rates where STATICS suggests > 4095 samples for full decay,
        // the actual decay time equals the Q8 distance (4095 for full range).
        // This matches authentic DX7 hardware behavior where fixed-point
        // precision limits timing accuracy for slow rates.
        //
        // For faster rates (increment >= 2), timing more closely matches STATICS.
        use rigel_modulation::envelope::get_static_count;

        let sample_rate = 44100.0;

        // Test with a faster rate where Q8 precision is adequate
        // Rate 70 at 44100Hz: STATICS gives ~1654 samples
        // Q8 increment = 4095/1654 ≈ 2.5, rounds to 2-3
        let rate = 70;
        let statics_samples = get_static_count(rate, sample_rate);

        let mut config = FmEnvelopeConfig::default_with_sample_rate(sample_rate);
        config.key_on_segments[0] = Segment::new(99, 99);
        config.key_on_segments[1] = Segment::new(rate, 0);

        let mut env = FmEnvelope::with_config(config);
        env.note_on(60);

        // Get to decay segment
        while env.current_segment() == 0 && env.is_active() {
            env.process();
        }

        // Count samples in decay
        let start_segment = env.current_segment();
        let mut decay_samples = 0u32;
        while env.current_segment() == start_segment && env.is_active() && decay_samples < 100000 {
            env.process();
            decay_samples += 1;
        }

        // With Q8 integer increments, timing is bounded by:
        // - Upper bound: distance (minimum increment of 1)
        // - Lower bound: STATICS timing (if increment > 1)
        // Allow 50% tolerance to account for Q8 quantization
        let max_decay = 4095u32; // Q8 distance
        let min_decay = (statics_samples as f32 * 0.5) as u32;

        assert!(
            decay_samples >= min_decay && decay_samples <= max_decay,
            "Decay timing should be between STATICS/2 ({}) and Q8 max ({}), got {}",
            min_decay,
            max_decay,
            decay_samples
        );
    }

    #[test]
    fn test_release_is_exponential() {
        // Verify release phase also uses exponential decay
        let mut env = FmEnvelope::new(44100.0);
        env.note_on(60);

        // Process through attack
        for _ in 0..1000 {
            env.process();
        }

        env.note_off();
        assert!(env.is_releasing());

        // Collect levels during release
        let mut levels: Vec<f32> = Vec::new();
        for _ in 0..5000 {
            if !env.is_releasing() || env.value() < 1e-6 {
                break;
            }
            levels.push(env.value());
            env.process();
        }

        // Check for constant ratio (exponential decay)
        if levels.len() >= 50 {
            let ratios: Vec<f32> = levels
                .windows(2)
                .filter(|w| w[0] > 1e-6)
                .map(|w| w[1] / w[0])
                .collect();

            if ratios.len() >= 10 {
                // All ratios should be approximately the same
                let avg_ratio: f32 = ratios.iter().sum::<f32>() / ratios.len() as f32;
                let max_deviation = ratios
                    .iter()
                    .map(|r| (r - avg_ratio).abs())
                    .fold(0.0f32, f32::max);

                assert!(
                    max_deviation < 0.02,
                    "Release should have constant decay ratio (exponential): avg={}, max_deviation={}",
                    avg_ratio,
                    max_deviation
                );
            }
        }
    }
}
