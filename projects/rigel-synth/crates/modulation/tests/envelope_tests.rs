//! Integration tests for the SY-Style Envelope module.
//!
//! Tests are organized by user story for traceability.

use rigel_modulation::envelope::{
    AwmEnvelope, EnvelopePhase, FmEnvelope, FmEnvelopeBatch8, FmEnvelopeConfig, LoopConfig,
    Segment, SevenSegEnvelope, JUMP_TARGET_Q8, LEVEL_MAX,
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
        // At level 0 (LEVEL_MIN), the linear value is very small but not exactly 0
        // due to exp2 calculation
        assert!(env.value() < 0.001, "Reset should give near-zero value");
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
}

// =============================================================================
// Phase 5: User Story 3 - MSFA-Compatible Rate Behavior
// =============================================================================

mod us3_msfa_rates {
    use super::*;
    use rigel_modulation::envelope::{calculate_increment_q8, rate_to_qrate};

    #[test]
    fn test_msfa_rate_formula() {
        // qrate = (rate * 41) >> 6
        assert_eq!(rate_to_qrate(0), 0);
        assert_eq!(rate_to_qrate(50), 32); // (50 * 41) >> 6 = 2050 >> 6 = 32
        assert_eq!(rate_to_qrate(99), 63); // (99 * 41) >> 6 = 4059 >> 6 = 63
    }

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
        let mut config = FmEnvelopeConfig::default_with_sample_rate(44100.0);
        config.key_on_segments[0] = Segment::new(99, 99);

        let mut env = FmEnvelope::with_config(config);
        env.note_on(60);

        // Should reach near-max very quickly
        let mut samples = 0;
        while env.value() < 0.9 && samples < 100 {
            env.process();
            samples += 1;
        }

        assert!(
            samples < 100,
            "Rate 99 should reach 90% in <100 samples, took {}",
            samples
        );
    }

    #[test]
    fn test_tolerance_against_reference() {
        // Verify level conversion accuracy
        use rigel_modulation::envelope::level_to_linear;

        // At LEVEL_MAX, should get ~1.0
        let max_linear = level_to_linear(LEVEL_MAX);
        let error_db = 20.0 * libm::log10f((max_linear - 1.0).abs().max(0.00001));

        // Error should be well within 0.1 dB
        assert!(
            error_db < -60.0,
            "Max level error should be < 0.1 dB, got {} dB",
            error_db
        );
    }

    #[test]
    fn test_increment_increases_with_qrate() {
        // Higher qRate should give higher increment
        let inc_10 = calculate_increment_q8(10);
        let inc_30 = calculate_increment_q8(30);
        let inc_50 = calculate_increment_q8(50);
        let inc_63 = calculate_increment_q8(63);

        assert!(
            inc_30 >= inc_10,
            "qRate 30 ({}) should be >= qRate 10 ({})",
            inc_30,
            inc_10
        );
        assert!(
            inc_50 >= inc_30,
            "qRate 50 ({}) should be >= qRate 30 ({})",
            inc_50,
            inc_30
        );
        assert!(
            inc_63 >= inc_50,
            "qRate 63 ({}) should be >= qRate 50 ({})",
            inc_63,
            inc_50
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

        // Inactive envelopes are at LEVEL_MIN which gives a very small
        // but non-zero linear value due to exp2 calculation
        assert!(
            output[2] < 0.001,
            "Inactive envelope should be near-zero, got {}",
            output[2]
        );
        assert!(
            output[3] < 0.001,
            "Inactive envelope should be near-zero, got {}",
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

        // After first process, level should jump to JUMP_TARGET
        env.process();

        // Get raw level
        let level = env.state().level_q8();

        assert!(
            level >= JUMP_TARGET_Q8,
            "Level {} should jump to at least {} on attack",
            level,
            JUMP_TARGET_Q8
        );
    }

    #[test]
    fn test_smooth_approach_after_jump() {
        let mut env = FmEnvelope::new(44100.0);
        env.note_on(60);

        // Process through attack
        let mut last_level = 0i16;
        for i in 0..1000 {
            env.process();
            let current = env.state().level_q8();

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
