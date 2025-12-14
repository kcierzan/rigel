//! Timebase unit tests
//!
//! Tests for sample-accurate timing infrastructure.

use rigel_timing::Timebase;

#[test]
fn test_timebase_new_initializes_to_zero() {
    let timebase = Timebase::new(44100.0);
    assert_eq!(timebase.sample_position(), 0);
    assert_eq!(timebase.block_start(), 0);
    assert_eq!(timebase.block_size(), 0);
    assert_eq!(timebase.sample_rate(), 44100.0);
}

#[test]
fn test_timebase_default() {
    let timebase = Timebase::default();
    assert_eq!(timebase.sample_position(), 0);
    assert_eq!(timebase.sample_rate(), 44100.0);
}

#[test]
#[should_panic(expected = "Sample rate must be positive")]
fn test_timebase_new_panics_on_zero_sample_rate() {
    let _timebase = Timebase::new(0.0);
}

#[test]
#[should_panic(expected = "Sample rate must be positive")]
fn test_timebase_new_panics_on_negative_sample_rate() {
    let _timebase = Timebase::new(-44100.0);
}

#[test]
fn test_timebase_advance_block_accuracy() {
    let mut timebase = Timebase::new(44100.0);

    // First block
    timebase.advance_block(64);
    assert_eq!(timebase.sample_position(), 64);
    assert_eq!(timebase.block_start(), 0);
    assert_eq!(timebase.block_size(), 64);

    // Second block
    timebase.advance_block(64);
    assert_eq!(timebase.sample_position(), 128);
    assert_eq!(timebase.block_start(), 64);
    assert_eq!(timebase.block_size(), 64);

    // Third block with different size
    timebase.advance_block(128);
    assert_eq!(timebase.sample_position(), 256);
    assert_eq!(timebase.block_start(), 128);
    assert_eq!(timebase.block_size(), 128);
}

#[test]
fn test_timebase_advance_exactly_block_size_samples() {
    let mut timebase = Timebase::new(44100.0);

    // Process 100 blocks of 64 samples
    for i in 0..100 {
        let expected_pos = i as u64 * 64;
        assert_eq!(timebase.sample_position(), expected_pos);
        timebase.advance_block(64);
    }

    // Final position should be exactly 100 * 64 = 6400
    assert_eq!(timebase.sample_position(), 6400);
}

#[test]
fn test_timebase_24_hour_overflow_safety_at_192khz() {
    // 24 hours at 192kHz = 24 * 60 * 60 * 192000 = 16,588,800,000 samples
    // u64 max is ~18.4 quintillion, so this should easily fit
    let samples_24_hours: u64 = 24 * 60 * 60 * 192_000;

    let mut timebase = Timebase::new(192000.0);

    // Simulate advancing to 24 hours
    // We can't actually process this many blocks, so we verify the math
    // by checking that adding this value doesn't overflow

    // Process a single large "block" representing 24 hours
    // Note: This is a simulation - real blocks would be smaller
    let block_size = u32::MAX; // Max block size we can pass

    // Calculate how many max-size blocks needed for 24 hours
    let total_samples = samples_24_hours;
    let blocks_needed = total_samples / block_size as u64;

    // Process enough blocks to exceed 24 hours
    for _ in 0..(blocks_needed + 1) {
        timebase.advance_block(block_size);
    }

    // Verify we didn't overflow and position is large enough
    assert!(timebase.sample_position() > samples_24_hours);

    // Verify time conversion still works correctly
    let time_seconds = timebase.samples_to_seconds(samples_24_hours);
    let expected_seconds = 24.0 * 60.0 * 60.0; // 86400 seconds

    // Allow small floating point tolerance
    assert!((time_seconds - expected_seconds).abs() < 0.001);
}

#[test]
fn test_timebase_sample_rate_change_preserves_position() {
    let mut timebase = Timebase::new(44100.0);

    // Advance several blocks
    timebase.advance_block(64);
    timebase.advance_block(64);
    timebase.advance_block(64);

    let position_before = timebase.sample_position();
    assert_eq!(position_before, 192);

    // Change sample rate
    timebase.set_sample_rate(96000.0);

    // Position should be unchanged
    assert_eq!(timebase.sample_position(), position_before);
    assert_eq!(timebase.sample_rate(), 96000.0);

    // Advancing should still work
    timebase.advance_block(64);
    assert_eq!(timebase.sample_position(), 256);
}

#[test]
#[should_panic(expected = "Sample rate must be positive")]
fn test_timebase_set_sample_rate_panics_on_zero() {
    let mut timebase = Timebase::new(44100.0);
    timebase.set_sample_rate(0.0);
}

#[test]
fn test_timebase_samples_to_seconds_conversion() {
    let timebase = Timebase::new(44100.0);

    // 44100 samples at 44100 Hz = 1 second
    assert!((timebase.samples_to_seconds(44100) - 1.0).abs() < 1e-10);

    // 22050 samples at 44100 Hz = 0.5 seconds
    assert!((timebase.samples_to_seconds(22050) - 0.5).abs() < 1e-10);

    // 0 samples = 0 seconds
    assert!((timebase.samples_to_seconds(0) - 0.0).abs() < 1e-10);
}

#[test]
fn test_timebase_seconds_to_samples_conversion() {
    let timebase = Timebase::new(44100.0);

    // 1 second at 44100 Hz = 44100 samples
    assert_eq!(timebase.seconds_to_samples(1.0), 44100);

    // 0.5 seconds at 44100 Hz = 22050 samples
    assert_eq!(timebase.seconds_to_samples(0.5), 22050);

    // 0 seconds = 0 samples
    assert_eq!(timebase.seconds_to_samples(0.0), 0);
}

#[test]
fn test_timebase_ms_to_samples_conversion() {
    let timebase = Timebase::new(44100.0);

    // 1000ms at 44100 Hz = 44100 samples
    assert_eq!(timebase.ms_to_samples(1000.0), 44100);

    // 10ms at 44100 Hz = 441 samples
    assert_eq!(timebase.ms_to_samples(10.0), 441);

    // 5ms at 44100 Hz = 220.5 -> rounds to 220 or 221
    let samples_5ms = timebase.ms_to_samples(5.0);
    assert!(samples_5ms == 220 || samples_5ms == 221);
}

#[test]
fn test_timebase_reset() {
    let mut timebase = Timebase::new(44100.0);

    // Advance several blocks
    timebase.advance_block(64);
    timebase.advance_block(64);
    timebase.advance_block(64);

    assert_eq!(timebase.sample_position(), 192);

    // Reset
    timebase.reset();

    // Position should be back to 0
    assert_eq!(timebase.sample_position(), 0);
    assert_eq!(timebase.block_start(), 0);
    assert_eq!(timebase.block_size(), 0);

    // Sample rate should be preserved
    assert_eq!(timebase.sample_rate(), 44100.0);
}

#[test]
fn test_timebase_multiple_modules_same_block() {
    // Simulate multiple DSP modules querying timebase within the same block
    let mut timebase = Timebase::new(44100.0);
    timebase.advance_block(64);

    // All queries within the same block should return identical values
    let module_a_pos = timebase.sample_position();
    let module_a_start = timebase.block_start();
    let module_a_size = timebase.block_size();

    let module_b_pos = timebase.sample_position();
    let module_b_start = timebase.block_start();
    let module_b_size = timebase.block_size();

    let module_c_pos = timebase.sample_position();
    let module_c_start = timebase.block_start();
    let module_c_size = timebase.block_size();

    assert_eq!(module_a_pos, module_b_pos);
    assert_eq!(module_b_pos, module_c_pos);

    assert_eq!(module_a_start, module_b_start);
    assert_eq!(module_b_start, module_c_start);

    assert_eq!(module_a_size, module_b_size);
    assert_eq!(module_b_size, module_c_size);
}

#[test]
fn test_timebase_one_second_at_44_1khz() {
    // Verify that exactly 44100 samples = 1 second
    let mut timebase = Timebase::new(44100.0);

    // Process 689 blocks of 64 samples = 44096 samples
    // Plus 1 block of 4 samples = 44100 samples total
    for _ in 0..689 {
        timebase.advance_block(64);
    }
    timebase.advance_block(4);

    assert_eq!(timebase.sample_position(), 44100);

    let elapsed_seconds = timebase.samples_to_seconds(timebase.sample_position());
    assert!((elapsed_seconds - 1.0).abs() < 1e-10);
}

#[test]
fn test_timebase_is_copy_and_clone() {
    let timebase = Timebase::new(44100.0);

    // Test Copy
    let copy = timebase;
    assert_eq!(copy.sample_rate(), 44100.0);

    // Original should still work (Copy semantics)
    assert_eq!(timebase.sample_rate(), 44100.0);

    // Test Clone (explicit clone() to verify Clone trait even though Copy is available)
    #[allow(clippy::clone_on_copy)]
    let clone = timebase.clone();
    assert_eq!(clone.sample_rate(), 44100.0);
}
