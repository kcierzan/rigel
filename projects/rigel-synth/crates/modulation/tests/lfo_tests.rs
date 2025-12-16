//! LFO integration tests.

use rigel_modulation::{
    Lfo, LfoPhaseMode, LfoPolarity, LfoRateMode, LfoWaveshape, ModulationSource, NoteBase,
    NoteDivision,
};
use rigel_timing::Timebase;

// ─────────────────────────────────────────────────────────────────────────────
// User Story 1: Basic LFO Modulation (P1)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_lfo_default_creation() {
    let lfo = Lfo::new();
    assert_eq!(lfo.waveshape(), LfoWaveshape::Sine);
    assert_eq!(lfo.rate_mode(), LfoRateMode::Hz(1.0));
    assert_eq!(lfo.phase_mode(), LfoPhaseMode::FreeRunning);
    assert_eq!(lfo.polarity(), LfoPolarity::Bipolar);
    assert_eq!(lfo.start_phase(), 0.0);
    assert_eq!(lfo.pulse_width(), 0.5);
    assert_eq!(lfo.phase(), 0.0);
}

#[test]
fn test_lfo_with_config() {
    let lfo = Lfo::with_config(
        LfoWaveshape::Triangle,
        LfoRateMode::Hz(2.0),
        LfoPhaseMode::Retrigger,
        LfoPolarity::Unipolar,
    );
    assert_eq!(lfo.waveshape(), LfoWaveshape::Triangle);
    assert_eq!(lfo.rate_mode(), LfoRateMode::Hz(2.0));
    assert_eq!(lfo.phase_mode(), LfoPhaseMode::Retrigger);
    assert_eq!(lfo.polarity(), LfoPolarity::Unipolar);
}

#[test]
fn test_lfo_output_range_bipolar() {
    let mut lfo = Lfo::new();
    lfo.set_waveshape(LfoWaveshape::Sine);
    lfo.set_rate(LfoRateMode::Hz(10.0)); // Fast rate for quick testing
    lfo.set_polarity(LfoPolarity::Bipolar);

    let mut timebase = Timebase::new(44100.0);

    let mut min_value = f32::MAX;
    let mut max_value = f32::MIN;

    // Run through multiple cycles
    for _ in 0..1000 {
        timebase.advance_block(64);
        lfo.update(&timebase);
        let value = lfo.value();
        min_value = min_value.min(value);
        max_value = max_value.max(value);
    }

    // Bipolar range should be approximately [-1.0, 1.0]
    assert!(
        (-1.01..=-0.9).contains(&min_value),
        "Bipolar min should be near -1.0, got {}",
        min_value
    );
    assert!(
        (0.9..=1.01).contains(&max_value),
        "Bipolar max should be near 1.0, got {}",
        max_value
    );
}

#[test]
fn test_lfo_output_range_unipolar() {
    let mut lfo = Lfo::new();
    lfo.set_waveshape(LfoWaveshape::Sine);
    lfo.set_rate(LfoRateMode::Hz(10.0));
    lfo.set_polarity(LfoPolarity::Unipolar);

    let mut timebase = Timebase::new(44100.0);

    let mut min_value = f32::MAX;
    let mut max_value = f32::MIN;

    for _ in 0..1000 {
        timebase.advance_block(64);
        lfo.update(&timebase);
        let value = lfo.value();
        min_value = min_value.min(value);
        max_value = max_value.max(value);
    }

    // Unipolar range should be approximately [0.0, 1.0]
    assert!(
        (-0.01..=0.1).contains(&min_value),
        "Unipolar min should be near 0.0, got {}",
        min_value
    );
    assert!(
        (0.9..=1.01).contains(&max_value),
        "Unipolar max should be near 1.0, got {}",
        max_value
    );
}

#[test]
fn test_lfo_cycle_frequency() {
    // Test that LFO completes one cycle in approximately the expected time
    let mut lfo = Lfo::new();
    lfo.set_waveshape(LfoWaveshape::Saw); // Saw wave is easy to detect crossings
    lfo.set_rate(LfoRateMode::Hz(1.0)); // 1 Hz = 1 cycle per second

    let sample_rate = 44100.0;
    let mut timebase = Timebase::new(sample_rate);

    // Reset and get initial value
    lfo.reset(&timebase);
    let mut last_value = lfo.value();

    let mut cycle_count = 0;
    let block_size = 64;
    let total_samples = (sample_rate * 2.0) as usize; // 2 seconds

    for _ in 0..(total_samples / block_size) {
        timebase.advance_block(block_size as u32);
        lfo.update(&timebase);
        let value = lfo.value();

        // Detect negative-to-positive crossing (start of saw cycle)
        if last_value < -0.5 && value > -0.5 {
            cycle_count += 1;
        }
        last_value = value;
    }

    // At 1 Hz over 2 seconds, we should see approximately 2 cycles
    // Allow some tolerance for block boundary effects
    assert!(
        (1..=3).contains(&cycle_count),
        "Expected ~2 cycles at 1 Hz over 2 seconds, got {}",
        cycle_count
    );
}

#[test]
fn test_waveshape_sine() {
    let mut lfo = Lfo::new();
    lfo.set_waveshape(LfoWaveshape::Sine);
    lfo.set_rate(LfoRateMode::Hz(1.0));

    let timebase = Timebase::new(44100.0);

    // At phase 0, sine should be 0
    lfo.reset(&timebase);
    let value_at_zero = lfo.value();
    assert!(
        value_at_zero.abs() < 0.1,
        "Sine at phase 0 should be near 0, got {}",
        value_at_zero
    );
}

#[test]
fn test_waveshape_triangle() {
    let mut lfo = Lfo::new();
    lfo.set_waveshape(LfoWaveshape::Triangle);
    lfo.set_rate(LfoRateMode::Hz(100.0)); // Fast for testing

    let mut timebase = Timebase::new(44100.0);
    lfo.reset(&timebase);

    let mut values = Vec::new();
    for _ in 0..100 {
        timebase.advance_block(8);
        lfo.update(&timebase);
        values.push(lfo.value());
    }

    // Triangle wave should have linear slopes
    // All values should be in [-1, 1]
    for value in &values {
        assert!(
            *value >= -1.01 && *value <= 1.01,
            "Triangle value out of range: {}",
            value
        );
    }
}

#[test]
fn test_waveshape_saw() {
    let mut lfo = Lfo::new();
    lfo.set_waveshape(LfoWaveshape::Saw);
    lfo.set_rate(LfoRateMode::Hz(100.0));

    let timebase = Timebase::new(44100.0);
    lfo.reset(&timebase);

    // Saw wave starts at -1 at phase 0 and rises to +1
    let initial_value = lfo.value();
    assert!(
        (-1.01..=-0.9).contains(&initial_value),
        "Saw at phase 0 should be near -1, got {}",
        initial_value
    );
}

#[test]
fn test_waveshape_square() {
    let mut lfo = Lfo::new();
    lfo.set_waveshape(LfoWaveshape::Square);
    lfo.set_rate(LfoRateMode::Hz(100.0));

    let mut timebase = Timebase::new(44100.0);
    lfo.reset(&timebase);

    let mut values = Vec::new();
    for _ in 0..100 {
        timebase.advance_block(8);
        lfo.update(&timebase);
        values.push(lfo.value());
    }

    // Square wave should only produce values near -1 or +1
    for value in &values {
        assert!(
            (*value - 1.0).abs() < 0.01 || (*value + 1.0).abs() < 0.01,
            "Square value should be -1 or +1, got {}",
            value
        );
    }
}

#[test]
fn test_configuration_setters() {
    let mut lfo = Lfo::new();

    lfo.set_waveshape(LfoWaveshape::Triangle);
    assert_eq!(lfo.waveshape(), LfoWaveshape::Triangle);

    lfo.set_rate(LfoRateMode::Hz(5.0));
    assert_eq!(lfo.rate_mode(), LfoRateMode::Hz(5.0));

    lfo.set_phase_mode(LfoPhaseMode::Retrigger);
    assert_eq!(lfo.phase_mode(), LfoPhaseMode::Retrigger);

    lfo.set_polarity(LfoPolarity::Unipolar);
    assert_eq!(lfo.polarity(), LfoPolarity::Unipolar);

    lfo.set_start_phase(0.25);
    assert_eq!(lfo.start_phase(), 0.25);

    lfo.set_pulse_width(0.75);
    assert_eq!(lfo.pulse_width(), 0.75);
}

#[test]
fn test_value_is_cached() {
    let mut lfo = Lfo::new();
    lfo.set_waveshape(LfoWaveshape::Sine);
    lfo.set_rate(LfoRateMode::Hz(1.0));

    let mut timebase = Timebase::new(44100.0);
    timebase.advance_block(64);
    lfo.update(&timebase);

    // Multiple calls to value() should return the same result
    let value1 = lfo.value();
    let value2 = lfo.value();
    let value3 = lfo.value();

    assert_eq!(value1, value2);
    assert_eq!(value2, value3);
}

// ─────────────────────────────────────────────────────────────────────────────
// User Story 2: Tempo-Synchronized Modulation (P1)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_note_division_multipliers() {
    // Quarter note at 60 BPM = 1 Hz
    assert!(
        (NoteDivision::normal(NoteBase::Quarter).to_hz(60.0) - 1.0).abs() < 0.001,
        "Quarter at 60 BPM should be 1 Hz"
    );

    // Half note at 60 BPM = 0.5 Hz
    assert!(
        (NoteDivision::normal(NoteBase::Half).to_hz(60.0) - 0.5).abs() < 0.001,
        "Half at 60 BPM should be 0.5 Hz"
    );

    // Eighth note at 60 BPM = 2 Hz
    assert!(
        (NoteDivision::normal(NoteBase::Eighth).to_hz(60.0) - 2.0).abs() < 0.001,
        "Eighth at 60 BPM should be 2 Hz"
    );

    // Whole note at 60 BPM = 0.25 Hz
    assert!(
        (NoteDivision::normal(NoteBase::Whole).to_hz(60.0) - 0.25).abs() < 0.001,
        "Whole at 60 BPM should be 0.25 Hz"
    );

    // Sixteenth at 60 BPM = 4 Hz
    assert!(
        (NoteDivision::normal(NoteBase::Sixteenth).to_hz(60.0) - 4.0).abs() < 0.001,
        "Sixteenth at 60 BPM should be 4 Hz"
    );

    // 32nd at 60 BPM = 8 Hz
    assert!(
        (NoteDivision::normal(NoteBase::ThirtySecond).to_hz(60.0) - 8.0).abs() < 0.001,
        "ThirtySecond at 60 BPM should be 8 Hz"
    );
}

#[test]
fn test_note_division_modifiers() {
    // Dotted quarter = 2/3 rate of quarter
    let quarter_hz = NoteDivision::normal(NoteBase::Quarter).to_hz(120.0);
    let dotted_quarter_hz = NoteDivision::dotted(NoteBase::Quarter).to_hz(120.0);
    assert!(
        ((dotted_quarter_hz / quarter_hz) - (2.0 / 3.0)).abs() < 0.001,
        "Dotted should be 2/3 rate"
    );

    // Triplet quarter = 1.5x rate of quarter
    let triplet_quarter_hz = NoteDivision::triplet(NoteBase::Quarter).to_hz(120.0);
    assert!(
        ((triplet_quarter_hz / quarter_hz) - 1.5).abs() < 0.001,
        "Triplet should be 1.5x rate"
    );
}

#[test]
fn test_tempo_sync_rate_calculation() {
    let mut lfo = Lfo::new();

    // Set to quarter note at 120 BPM
    lfo.set_rate(LfoRateMode::TempoSync {
        division: NoteDivision::normal(NoteBase::Quarter),
        bpm: 120.0,
    });

    // At 120 BPM, quarter note = 2 Hz (120/60 * 1.0)
    let effective_hz = lfo.effective_rate_hz();
    assert!(
        (effective_hz - 2.0).abs() < 0.001,
        "Quarter at 120 BPM should be 2 Hz, got {}",
        effective_hz
    );
}

#[test]
fn test_tempo_sync_bpm_changes() {
    let mut lfo = Lfo::new();

    lfo.set_rate(LfoRateMode::TempoSync {
        division: NoteDivision::normal(NoteBase::Quarter),
        bpm: 120.0,
    });

    assert!(
        (lfo.effective_rate_hz() - 2.0).abs() < 0.001,
        "Initial rate should be 2 Hz"
    );

    // Change tempo to 60 BPM
    lfo.set_tempo(60.0);
    assert!(
        (lfo.effective_rate_hz() - 1.0).abs() < 0.001,
        "After tempo change to 60 BPM, should be 1 Hz"
    );

    // Change tempo to 180 BPM
    lfo.set_tempo(180.0);
    assert!(
        (lfo.effective_rate_hz() - 3.0).abs() < 0.001,
        "After tempo change to 180 BPM, should be 3 Hz"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// User Story 3: Phase Reset on Note Trigger (P2)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_trigger_free_running_mode() {
    let mut lfo = Lfo::new();
    lfo.set_phase_mode(LfoPhaseMode::FreeRunning);
    lfo.set_rate(LfoRateMode::Hz(10.0));

    let mut timebase = Timebase::new(44100.0);

    // Advance to get non-zero phase
    for _ in 0..10 {
        timebase.advance_block(64);
        lfo.update(&timebase);
    }

    let phase_before = lfo.phase();

    // Trigger should have no effect in FreeRunning mode
    lfo.trigger();

    assert_eq!(
        lfo.phase(),
        phase_before,
        "Phase should not change in FreeRunning mode"
    );
}

#[test]
fn test_trigger_retrigger_mode() {
    let mut lfo = Lfo::new();
    lfo.set_phase_mode(LfoPhaseMode::Retrigger);
    lfo.set_start_phase(0.25);
    lfo.set_rate(LfoRateMode::Hz(10.0));

    let mut timebase = Timebase::new(44100.0);

    // Advance to get non-zero phase different from start_phase
    for _ in 0..10 {
        timebase.advance_block(64);
        lfo.update(&timebase);
    }

    assert!(
        (lfo.phase() - 0.25).abs() > 0.01,
        "Phase should have advanced from start_phase"
    );

    // Trigger should reset phase to start_phase
    lfo.trigger();

    assert!(
        (lfo.phase() - 0.25).abs() < 0.01,
        "Phase should reset to start_phase (0.25), got {}",
        lfo.phase()
    );
}

#[test]
fn test_start_phase_variations() {
    let mut lfo = Lfo::new();
    lfo.set_phase_mode(LfoPhaseMode::Retrigger);

    let mut timebase = Timebase::new(44100.0);

    // Test different start phases
    for start_phase in [0.0, 0.25, 0.5, 0.75, 1.0] {
        let clamped_phase = if start_phase >= 1.0 { 0.0 } else { start_phase };
        lfo.set_start_phase(clamped_phase);

        // Advance phase
        timebase.advance_block(64);
        lfo.update(&timebase);

        // Trigger reset
        lfo.trigger();

        // Phase should be at start_phase (or 0 if start_phase was 1.0)
        assert!(
            (lfo.phase() - clamped_phase).abs() < 0.01,
            "After trigger, phase should be {}, got {}",
            clamped_phase,
            lfo.phase()
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// User Story 4: PWM Control for Pulse Wave (P2)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_pulse_width_effect() {
    let mut lfo = Lfo::new();
    lfo.set_waveshape(LfoWaveshape::Pulse);
    lfo.set_rate(LfoRateMode::Hz(100.0));

    let mut timebase = Timebase::new(44100.0);

    // Test 25% duty cycle
    lfo.set_pulse_width(0.25);
    lfo.reset(&timebase);

    let mut high_count = 0;
    let mut low_count = 0;

    for _ in 0..100 {
        timebase.advance_block(8);
        lfo.update(&timebase);
        if lfo.value() > 0.0 {
            high_count += 1;
        } else {
            low_count += 1;
        }
    }

    // With 25% duty cycle, should have roughly 25% high, 75% low
    let high_ratio = high_count as f32 / (high_count + low_count) as f32;
    assert!(
        (high_ratio - 0.25).abs() < 0.15,
        "With 25% pulse width, high ratio should be ~25%, got {}%",
        high_ratio * 100.0
    );
}

#[test]
fn test_pulse_width_50_equals_square() {
    let mut lfo_pulse = Lfo::new();
    lfo_pulse.set_waveshape(LfoWaveshape::Pulse);
    lfo_pulse.set_pulse_width(0.5);
    lfo_pulse.set_rate(LfoRateMode::Hz(1.0));

    let mut lfo_square = Lfo::new();
    lfo_square.set_waveshape(LfoWaveshape::Square);
    lfo_square.set_rate(LfoRateMode::Hz(1.0));

    let mut timebase = Timebase::new(44100.0);

    // Both should produce the same output
    for _ in 0..10 {
        timebase.advance_block(64);
        lfo_pulse.update(&timebase);
        lfo_square.update(&timebase);

        assert!(
            (lfo_pulse.value() - lfo_square.value()).abs() < 0.01,
            "Pulse(0.5) and Square should be equivalent"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// User Story 5: Sample and Hold Modulation (P2)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_sample_and_hold_stability_within_cycle() {
    let mut lfo = Lfo::new();
    lfo.set_waveshape(LfoWaveshape::SampleAndHold);
    lfo.set_rate(LfoRateMode::Hz(0.5)); // Slow rate - one cycle every 2 seconds

    let mut timebase = Timebase::new(44100.0);
    lfo.reset(&timebase);

    // Get initial S&H value
    let initial_value = lfo.value();

    // Advance partway through the cycle (should stay the same)
    for _ in 0..100 {
        timebase.advance_block(64);
        lfo.update(&timebase);
        assert_eq!(
            lfo.value(),
            initial_value,
            "S&H value should remain constant within cycle"
        );
    }
}

#[test]
fn test_sample_and_hold_changes_at_cycle_boundary() {
    let mut lfo = Lfo::new();
    lfo.set_waveshape(LfoWaveshape::SampleAndHold);
    lfo.set_rate(LfoRateMode::Hz(10.0)); // Fast rate for testing

    let mut timebase = Timebase::new(44100.0);
    lfo.reset(&timebase);

    let mut values_seen = Vec::new();
    let mut last_value = lfo.value();
    values_seen.push(last_value);

    // Run through multiple cycles
    for _ in 0..500 {
        timebase.advance_block(64);
        lfo.update(&timebase);
        let current = lfo.value();
        if (current - last_value).abs() > 0.01 {
            values_seen.push(current);
            last_value = current;
        }
    }

    // Should have seen multiple distinct values (one per cycle)
    assert!(
        values_seen.len() > 1,
        "S&H should change values at cycle boundaries, saw {} distinct values",
        values_seen.len()
    );
}

#[test]
fn test_noise_continuous_variation() {
    let mut lfo = Lfo::new();
    lfo.set_waveshape(LfoWaveshape::Noise);
    lfo.set_rate(LfoRateMode::Hz(1.0));

    let mut timebase = Timebase::new(44100.0);
    lfo.reset(&timebase);

    let mut values = Vec::new();
    for _ in 0..100 {
        timebase.advance_block(64);
        lfo.update(&timebase);
        values.push(lfo.value());
    }

    // Count unique values (should be many for noise)
    let mut unique_count = 1;
    for i in 1..values.len() {
        if (values[i] - values[i - 1]).abs() > 0.001 {
            unique_count += 1;
        }
    }

    assert!(
        unique_count > 50,
        "Noise should produce many different values, got {} unique",
        unique_count
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// User Story 6: Polarity Mode Selection (P3)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_polarity_bipolar_range() {
    let mut lfo = Lfo::new();
    lfo.set_waveshape(LfoWaveshape::Sine);
    lfo.set_rate(LfoRateMode::Hz(10.0));
    lfo.set_polarity(LfoPolarity::Bipolar);

    let mut timebase = Timebase::new(44100.0);

    for _ in 0..1000 {
        timebase.advance_block(64);
        lfo.update(&timebase);
        let value = lfo.value();
        assert!(
            (-1.01..=1.01).contains(&value),
            "Bipolar value {} out of range [-1, 1]",
            value
        );
    }
}

#[test]
fn test_polarity_unipolar_range() {
    let mut lfo = Lfo::new();
    lfo.set_waveshape(LfoWaveshape::Sine);
    lfo.set_rate(LfoRateMode::Hz(10.0));
    lfo.set_polarity(LfoPolarity::Unipolar);

    let mut timebase = Timebase::new(44100.0);

    for _ in 0..1000 {
        timebase.advance_block(64);
        lfo.update(&timebase);
        let value = lfo.value();
        assert!(
            (-0.01..=1.01).contains(&value),
            "Unipolar value {} out of range [0, 1]",
            value
        );
    }
}

#[test]
fn test_polarity_mode_switching() {
    // Test that polarity affects the relationship between values correctly.
    // We use two separate LFOs at the same phase to verify the math.
    let mut lfo_bipolar = Lfo::new();
    lfo_bipolar.set_waveshape(LfoWaveshape::Saw);
    lfo_bipolar.set_rate(LfoRateMode::Hz(10.0));
    lfo_bipolar.set_polarity(LfoPolarity::Bipolar);

    let mut lfo_unipolar = Lfo::new();
    lfo_unipolar.set_waveshape(LfoWaveshape::Saw);
    lfo_unipolar.set_rate(LfoRateMode::Hz(10.0));
    lfo_unipolar.set_polarity(LfoPolarity::Unipolar);

    let mut timebase = Timebase::new(44100.0);

    // Both LFOs at same phase
    timebase.advance_block(64);
    lfo_bipolar.update(&timebase);
    lfo_unipolar.update(&timebase);

    let bipolar_value = lfo_bipolar.value();
    let unipolar_value = lfo_unipolar.value();

    // Unipolar should be (bipolar + 1) / 2
    let expected_unipolar = (bipolar_value + 1.0) * 0.5;
    assert!(
        (unipolar_value - expected_unipolar).abs() < 0.01,
        "Unipolar {} should be (bipolar {} + 1) / 2 = {}",
        unipolar_value,
        bipolar_value,
        expected_unipolar
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// User Story 7: Control Rate Processing (P1)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_control_rate_different_intervals() {
    let sample_rate = 44100.0;

    // Test with different control rate intervals
    for block_size in [1, 32, 64, 128] {
        let mut lfo = Lfo::new();
        lfo.set_waveshape(LfoWaveshape::Saw);
        lfo.set_rate(LfoRateMode::Hz(1.0));

        let mut timebase = Timebase::new(sample_rate);

        let total_samples = sample_rate as usize; // 1 second
        let num_blocks = total_samples / block_size;

        for _ in 0..num_blocks {
            timebase.advance_block(block_size as u32);
            lfo.update(&timebase);
        }

        // After 1 second at 1 Hz, phase should have wrapped back to near 0
        // (with some tolerance for block boundary effects)
        assert!(
            lfo.phase() < 0.1 || lfo.phase() > 0.9,
            "After 1 second at 1 Hz with block size {}, phase {} should be near 0 or 1",
            block_size,
            lfo.phase()
        );
    }
}

#[test]
fn test_value_returns_cached_no_computation() {
    let mut lfo = Lfo::new();
    lfo.set_waveshape(LfoWaveshape::Sine);
    lfo.set_rate(LfoRateMode::Hz(1.0));

    let mut timebase = Timebase::new(44100.0);
    timebase.advance_block(64);
    lfo.update(&timebase);

    // Calling value() many times should return identical results
    // (proving it's just reading a cached value)
    let values: Vec<f32> = (0..1000).map(|_| lfo.value()).collect();

    let first = values[0];
    for (i, &v) in values.iter().enumerate() {
        assert_eq!(v, first, "value() call {} returned different result", i);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ModulationSource Trait Tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_modulation_source_trait_reset() {
    let mut lfo = Lfo::new();
    lfo.set_start_phase(0.5);
    lfo.set_rate(LfoRateMode::Hz(10.0));

    let mut timebase = Timebase::new(44100.0);

    // Advance phase
    for _ in 0..10 {
        timebase.advance_block(64);
        lfo.update(&timebase);
    }

    // Reset should restore to start_phase
    lfo.reset(&timebase);

    assert!(
        (lfo.phase() - 0.5).abs() < 0.01,
        "After reset, phase should be at start_phase"
    );
}

#[test]
fn test_modulation_source_trait_value_before_update() {
    let lfo = Lfo::new();

    // value() should work even before any update() calls
    let value = lfo.value();
    assert!(
        value.is_finite(),
        "value() should return finite number before update"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// New API Tests: Interpolation Strategy
// ─────────────────────────────────────────────────────────────────────────────

use rigel_modulation::{InterpolationStrategy, SimdAwareComponent};

#[test]
fn test_interpolation_strategy_default() {
    let lfo = Lfo::new();
    assert_eq!(lfo.interpolation(), InterpolationStrategy::Linear);
}

#[test]
fn test_interpolation_strategy_setter() {
    let mut lfo = Lfo::new();

    lfo.set_interpolation(InterpolationStrategy::CubicHermite);
    assert_eq!(lfo.interpolation(), InterpolationStrategy::CubicHermite);

    lfo.set_interpolation(InterpolationStrategy::Linear);
    assert_eq!(lfo.interpolation(), InterpolationStrategy::Linear);
}

// ─────────────────────────────────────────────────────────────────────────────
// New API Tests: Block Generation
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_generate_block_fills_buffer() {
    let mut lfo = Lfo::new();
    lfo.set_waveshape(LfoWaveshape::Sine);
    lfo.set_rate(LfoRateMode::Hz(10.0));

    let mut timebase = Timebase::new(44100.0);
    lfo.reset(&timebase);
    timebase.advance_block(64);
    lfo.update(&timebase);

    let mut output = [0.0f32; 64];
    lfo.generate_block(&mut output);

    // All values should be valid and in range
    for (i, &value) in output.iter().enumerate() {
        assert!(
            value.is_finite(),
            "generate_block value at {} should be finite",
            i
        );
        assert!(
            (-1.01..=1.01).contains(&value),
            "generate_block value {} at {} out of bipolar range",
            value,
            i
        );
    }
}

#[test]
fn test_generate_block_linear_interpolation() {
    let mut lfo = Lfo::new();
    lfo.set_waveshape(LfoWaveshape::Saw);
    lfo.set_rate(LfoRateMode::Hz(100.0)); // Faster LFO for visible interpolation
    lfo.set_interpolation(InterpolationStrategy::Linear);

    let mut timebase = Timebase::new(44100.0);
    lfo.reset(&timebase);

    // First block
    timebase.advance_block(64);
    lfo.update(&timebase);

    let mut output = [0.0f32; 64];
    lfo.generate_block(&mut output);

    // Linear interpolation should produce smoothly changing values
    // Values should be smoothly changing (not stepping)
    let mut num_changes = 0;
    for i in 1..64 {
        if (output[i] - output[i - 1]).abs() > 0.0001 {
            num_changes += 1;
        }
    }

    // At 100Hz with 64 samples, phase advances by ~0.145, giving significant interpolation
    // Should have gradual changes across the block
    assert!(
        num_changes > 30,
        "Linear interpolation should produce smooth transitions, got {} changes",
        num_changes
    );

    // All values should be in valid range
    for (i, &value) in output.iter().enumerate() {
        assert!(
            (-1.01..=1.01).contains(&value),
            "Value {} at {} out of range",
            value,
            i
        );
    }
}

#[test]
fn test_generate_block_hermite_interpolation() {
    let mut lfo = Lfo::new();
    lfo.set_waveshape(LfoWaveshape::Sine);
    lfo.set_rate(LfoRateMode::Hz(1.0));
    lfo.set_interpolation(InterpolationStrategy::CubicHermite);

    let mut timebase = Timebase::new(44100.0);
    lfo.reset(&timebase);

    timebase.advance_block(64);
    lfo.update(&timebase);

    let mut output = [0.0f32; 64];
    lfo.generate_block(&mut output);

    // Hermite should also produce smooth transitions
    for (i, &value) in output.iter().enumerate() {
        assert!(value.is_finite(), "Hermite value at {} should be finite", i);
        assert!(
            (-1.1..=1.1).contains(&value),
            "Hermite value {} at {} out of range",
            value,
            i
        );
    }
}

#[test]
fn test_generate_block_unipolar() {
    let mut lfo = Lfo::new();
    lfo.set_waveshape(LfoWaveshape::Sine);
    lfo.set_rate(LfoRateMode::Hz(10.0));
    lfo.set_polarity(LfoPolarity::Unipolar);

    let mut timebase = Timebase::new(44100.0);
    lfo.reset(&timebase);

    timebase.advance_block(64);
    lfo.update(&timebase);

    let mut output = [0.0f32; 64];
    lfo.generate_block(&mut output);

    // All values should be in unipolar range [0, 1]
    for (i, &value) in output.iter().enumerate() {
        assert!(
            (-0.01..=1.01).contains(&value),
            "Unipolar generate_block value {} at {} out of range",
            value,
            i
        );
    }
}

#[test]
fn test_generate_block_noise() {
    let mut lfo = Lfo::new();
    lfo.set_waveshape(LfoWaveshape::Noise);
    lfo.set_rate(LfoRateMode::Hz(1.0));

    let mut timebase = Timebase::new(44100.0);
    lfo.reset(&timebase);

    timebase.advance_block(64);
    lfo.update(&timebase);

    let mut output = [0.0f32; 64];
    lfo.generate_block_mut(&mut output);

    // Noise should produce varied values within range
    let mut unique_count = 1;
    for i in 1..64 {
        if (output[i] - output[i - 1]).abs() > 0.001 {
            unique_count += 1;
        }
    }

    assert!(
        unique_count > 50,
        "Noise generate_block should produce varied values, got {} unique",
        unique_count
    );

    for (i, &value) in output.iter().enumerate() {
        assert!(
            (-1.01..=1.01).contains(&value),
            "Noise value {} at {} out of range",
            value,
            i
        );
    }
}

#[test]
fn test_generate_block_sample_and_hold() {
    let mut lfo = Lfo::new();
    lfo.set_waveshape(LfoWaveshape::SampleAndHold);
    lfo.set_rate(LfoRateMode::Hz(0.5)); // Slow rate - constant within block

    let mut timebase = Timebase::new(44100.0);
    lfo.reset(&timebase);

    timebase.advance_block(64);
    lfo.update(&timebase);

    let mut output = [0.0f32; 64];
    lfo.generate_block(&mut output);

    // S&H should produce constant values within the block
    let first = output[0];
    for (i, &value) in output.iter().enumerate() {
        assert!(
            (value - first).abs() < 0.001,
            "S&H block should be constant, but value {} at {} differs from first {}",
            value,
            i,
            first
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// New API Tests: Single Sample Access
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_sample_returns_values() {
    let mut lfo = Lfo::new();
    lfo.set_waveshape(LfoWaveshape::Sine);
    lfo.set_rate(LfoRateMode::Hz(10.0));

    let mut timebase = Timebase::new(44100.0);
    lfo.reset(&timebase);

    timebase.advance_block(64);
    lfo.update(&timebase);

    // Get samples
    for i in 0..64 {
        let value = lfo.sample();
        assert!(
            value.is_finite(),
            "sample() {} should return finite value",
            i
        );
        assert!(
            (-1.01..=1.01).contains(&value),
            "sample() {} value {} out of range",
            i,
            value
        );
    }
}

#[test]
fn test_sample_matches_generate_block() {
    let mut lfo1 = Lfo::new();
    lfo1.set_waveshape(LfoWaveshape::Sine);
    lfo1.set_rate(LfoRateMode::Hz(5.0));
    lfo1.set_interpolation(InterpolationStrategy::Linear);

    let mut lfo2 = Lfo::new();
    lfo2.set_waveshape(LfoWaveshape::Sine);
    lfo2.set_rate(LfoRateMode::Hz(5.0));
    lfo2.set_interpolation(InterpolationStrategy::Linear);

    let mut timebase1 = Timebase::new(44100.0);
    let mut timebase2 = Timebase::new(44100.0);

    lfo1.reset(&timebase1);
    lfo2.reset(&timebase2);

    timebase1.advance_block(64);
    timebase2.advance_block(64);
    lfo1.update(&timebase1);
    lfo2.update(&timebase2);

    // Generate block
    let mut block_output = [0.0f32; 64];
    lfo1.generate_block(&mut block_output);

    // Get samples
    let mut sample_output = [0.0f32; 64];
    for item in &mut sample_output {
        *item = lfo2.sample();
    }

    // Both should match
    for i in 0..64 {
        assert!(
            (block_output[i] - sample_output[i]).abs() < 0.001,
            "sample() and generate_block() should match at {}: {} vs {}",
            i,
            sample_output[i],
            block_output[i]
        );
    }
}

#[test]
fn test_sample_cache_refresh() {
    let mut lfo = Lfo::new();
    lfo.set_waveshape(LfoWaveshape::Sine);
    lfo.set_rate(LfoRateMode::Hz(1.0));

    let mut timebase = Timebase::new(44100.0);
    lfo.reset(&timebase);

    timebase.advance_block(64);
    lfo.update(&timebase);

    // Read more than cache size (64 samples)
    let mut values = Vec::new();
    for _ in 0..128 {
        values.push(lfo.sample());
    }

    // All values should be valid
    for (i, &value) in values.iter().enumerate() {
        assert!(
            value.is_finite() && (-1.01..=1.01).contains(&value),
            "sample() {} value {} invalid",
            i,
            value
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// New API Tests: SimdAwareComponent Trait
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_simd_aware_component_lanes() {
    // LFO implements SimdAwareComponent
    let lanes = Lfo::lanes();

    // Lanes should be at least 1 (scalar) and at most 16 (AVX-512)
    assert!(
        (1..=16).contains(&lanes),
        "SIMD lanes {} should be in [1, 16]",
        lanes
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Accuracy Validation Tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_sine_accuracy_linear() {
    // Test that interpolated sine is reasonably close to ideal
    let mut lfo = Lfo::new();
    lfo.set_waveshape(LfoWaveshape::Sine);
    lfo.set_rate(LfoRateMode::Hz(1.0));
    lfo.set_interpolation(InterpolationStrategy::Linear);

    let sample_rate = 44100.0;
    let mut timebase = Timebase::new(sample_rate);
    lfo.reset(&timebase);

    let mut max_error = 0.0f32;
    let num_blocks = 689; // ~1 second at 64 samples per block

    for block_idx in 0..num_blocks {
        timebase.advance_block(64);
        lfo.update(&timebase);

        let mut output = [0.0f32; 64];
        lfo.generate_block(&mut output);

        // Compare to ideal sine at each sample position
        for (i, &sample) in output.iter().enumerate() {
            let global_sample = block_idx * 64 + i;
            let t = global_sample as f32 / sample_rate;
            let ideal = (core::f32::consts::TAU * t).sin();
            let error = (sample - ideal).abs();
            max_error = max_error.max(error);
        }
    }

    // Linear interpolation should be within 10% of ideal for 64-sample blocks
    assert!(
        max_error < 0.10,
        "Linear sine max error {} exceeds 10%",
        max_error
    );
}

#[test]
fn test_sine_accuracy_hermite() {
    // Test that Hermite interpolated sine is closer to ideal than linear
    let mut lfo = Lfo::new();
    lfo.set_waveshape(LfoWaveshape::Sine);
    lfo.set_rate(LfoRateMode::Hz(1.0));
    lfo.set_interpolation(InterpolationStrategy::CubicHermite);

    let sample_rate = 44100.0;
    let mut timebase = Timebase::new(sample_rate);
    lfo.reset(&timebase);

    let mut max_error = 0.0f32;
    let num_blocks = 689; // ~1 second at 64 samples per block

    for block_idx in 0..num_blocks {
        timebase.advance_block(64);
        lfo.update(&timebase);

        let mut output = [0.0f32; 64];
        lfo.generate_block(&mut output);

        // Compare to ideal sine at each sample position
        for (i, &sample) in output.iter().enumerate() {
            let global_sample = block_idx * 64 + i;
            let t = global_sample as f32 / sample_rate;
            let ideal = (core::f32::consts::TAU * t).sin();
            let error = (sample - ideal).abs();
            max_error = max_error.max(error);
        }
    }

    // Hermite interpolation should be within 5% of ideal
    assert!(
        max_error < 0.05,
        "Hermite sine max error {} exceeds 5%",
        max_error
    );
}

#[test]
fn test_waveshape_derivative_sine() {
    // Test that sine derivative is correct
    let phase = 0.25; // At peak, derivative should be 0
    let derivative = LfoWaveshape::Sine.derivative(phase, 0.5);

    // d/dφ sin(2πφ) at φ=0.25 should be 2π·cos(π/2) = 0
    assert!(
        derivative.abs() < 0.1,
        "Sine derivative at phase 0.25 should be near 0, got {}",
        derivative
    );

    // At phase 0, derivative should be 2π
    let derivative_at_0 = LfoWaveshape::Sine.derivative(0.0, 0.5);
    let expected = core::f32::consts::TAU;
    assert!(
        (derivative_at_0 - expected).abs() < 0.1,
        "Sine derivative at phase 0 should be ~{}, got {}",
        expected,
        derivative_at_0
    );
}

#[test]
fn test_waveshape_derivative_triangle() {
    // Triangle should have constant slopes
    let d1 = LfoWaveshape::Triangle.derivative(0.1, 0.5); // Rising phase
    let d2 = LfoWaveshape::Triangle.derivative(0.5, 0.5); // Falling phase
    let d3 = LfoWaveshape::Triangle.derivative(0.9, 0.5); // Rising phase again

    assert!(
        (d1 - 4.0).abs() < 0.01,
        "Triangle rising slope should be 4, got {}",
        d1
    );
    assert!(
        (d2 + 4.0).abs() < 0.01,
        "Triangle falling slope should be -4, got {}",
        d2
    );
    assert!(
        (d3 - 4.0).abs() < 0.01,
        "Triangle rising slope at end should be 4, got {}",
        d3
    );
}

#[test]
fn test_waveshape_derivative_saw() {
    // Saw should have constant slope of 2
    let d1 = LfoWaveshape::Saw.derivative(0.0, 0.5);
    let d2 = LfoWaveshape::Saw.derivative(0.5, 0.5);
    let d3 = LfoWaveshape::Saw.derivative(0.99, 0.5);

    assert!((d1 - 2.0).abs() < 0.01, "Saw slope should be 2, got {}", d1);
    assert!((d2 - 2.0).abs() < 0.01, "Saw slope should be 2, got {}", d2);
    assert!((d3 - 2.0).abs() < 0.01, "Saw slope should be 2, got {}", d3);
}

#[test]
fn test_waveshape_derivative_step_functions() {
    // Square, Pulse, S&H should have zero derivative
    assert_eq!(LfoWaveshape::Square.derivative(0.25, 0.5), 0.0);
    assert_eq!(LfoWaveshape::Pulse.derivative(0.25, 0.5), 0.0);
    assert_eq!(LfoWaveshape::SampleAndHold.derivative(0.25, 0.5), 0.0);
    assert_eq!(LfoWaveshape::Noise.derivative(0.25, 0.5), 0.0);
}
