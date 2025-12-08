//! Control rate and modulation source tests
//!
//! Tests for ControlRateClock timing and ModulationSource trait.

use rigel_timing::{ControlRateClock, ModulationSource, Timebase};

// ============================================================================
// ControlRateClock Tests
// ============================================================================

#[test]
fn test_control_rate_clock_new() {
    let clock = ControlRateClock::new(64);
    assert_eq!(clock.interval(), 64);
}

#[test]
fn test_control_rate_clock_valid_intervals() {
    // All valid power-of-2 intervals
    let valid_intervals = [1, 8, 16, 32, 64, 128];

    for interval in valid_intervals {
        let clock = ControlRateClock::new(interval);
        assert_eq!(clock.interval(), interval);
    }
}

#[test]
#[should_panic(expected = "Interval must be 1-128")]
fn test_control_rate_clock_panics_on_zero() {
    let _clock = ControlRateClock::new(0);
}

#[test]
#[should_panic(expected = "Interval must be 1-128")]
fn test_control_rate_clock_panics_on_too_large() {
    let _clock = ControlRateClock::new(256);
}

#[test]
#[should_panic(expected = "Interval must be a power of 2")]
fn test_control_rate_clock_panics_on_non_power_of_2() {
    let _clock = ControlRateClock::new(48);
}

#[test]
fn test_control_rate_clock_update_count_accuracy() {
    let mut clock = ControlRateClock::new(64);

    // 128-sample block should have exactly 2 updates (at 0 and 64)
    let updates: Vec<u32> = clock.advance(128).collect();
    assert_eq!(updates.len(), 2);
    assert_eq!(updates, vec![0, 64]);
}

#[test]
fn test_control_rate_clock_exact_updates_at_interval() {
    let mut clock = ControlRateClock::new(32);

    // 128-sample block should have 4 updates
    let updates: Vec<u32> = clock.advance(128).collect();
    assert_eq!(updates, vec![0, 32, 64, 96]);
}

#[test]
fn test_control_rate_clock_remainder_carryover() {
    let mut clock = ControlRateClock::new(64);

    // First block: 48 samples (no complete interval)
    let updates1: Vec<u32> = clock.advance(48).collect();
    assert_eq!(updates1, vec![0]); // Update at start

    // Second block: 48 samples (should complete an interval)
    // Remainder was 48, so next update at 64 - 48 = 16 into block
    let updates2: Vec<u32> = clock.advance(48).collect();
    assert_eq!(updates2, vec![16]); // Update at offset 16

    // Third block: 48 samples
    // Remainder was (48 + 48) % 64 = 32
    // Next update at 64 - 32 = 32 into block
    let updates3: Vec<u32> = clock.advance(48).collect();
    assert_eq!(updates3, vec![32]); // Update at offset 32
}

#[test]
fn test_control_rate_clock_block_smaller_than_interval() {
    let mut clock = ControlRateClock::new(128);

    // First 64-sample block - should have 1 update at 0
    let updates1: Vec<u32> = clock.advance(64).collect();
    assert_eq!(updates1, vec![0]);

    // Second 64-sample block - no update yet
    // Remainder is now 64, next update at 128 - 64 = 64 into block
    // But block is only 64 samples, so 64 >= block_size (64), no update
    let updates2: Vec<u32> = clock.advance(64).collect();
    assert_eq!(updates2, vec![]); // No updates - 64 is out of bounds for 64-sample block

    // Third 64-sample block - update at 0
    // Remainder is now (64 + 64) % 128 = 0, so update at 0
    let updates3: Vec<u32> = clock.advance(64).collect();
    assert_eq!(updates3, vec![0]);
}

#[test]
fn test_control_rate_clock_multiple_control_rates() {
    // Two clocks with different rates
    let mut clock32 = ControlRateClock::new(32);
    let mut clock64 = ControlRateClock::new(64);

    // Process 128 samples
    let updates32: Vec<u32> = clock32.advance(128).collect();
    let updates64: Vec<u32> = clock64.advance(128).collect();

    // 32-sample interval should have 4 updates
    assert_eq!(updates32, vec![0, 32, 64, 96]);

    // 64-sample interval should have 2 updates
    assert_eq!(updates64, vec![0, 64]);
}

#[test]
fn test_control_rate_clock_reset() {
    let mut clock = ControlRateClock::new(64);

    // Advance and accumulate remainder
    clock.advance(48);

    // Reset
    clock.reset();

    // Next block should start with remainder = 0
    let updates: Vec<u32> = clock.advance(64).collect();
    assert_eq!(updates, vec![0]); // Update at start
}

#[test]
fn test_control_rate_clock_exact_block_multiple() {
    let mut clock = ControlRateClock::new(64);

    // 64-sample block = exactly 1 interval
    let updates: Vec<u32> = clock.advance(64).collect();
    assert_eq!(updates, vec![0]);

    // Next 64-sample block should also have update at 0
    let updates2: Vec<u32> = clock.advance(64).collect();
    assert_eq!(updates2, vec![0]);
}

#[test]
fn test_control_rate_updates_iterator_size_hint() {
    let mut clock = ControlRateClock::new(32);

    let updates = clock.advance(128);

    // Should be exactly 4 updates
    assert_eq!(updates.len(), 4);
}

#[test]
fn test_control_rate_clock_is_copy_and_clone() {
    let clock = ControlRateClock::new(64);

    // Test Copy
    let copy = clock;
    assert_eq!(copy.interval(), 64);

    // Original should still work
    assert_eq!(clock.interval(), 64);

    // Test Clone (explicit clone() to verify Clone trait even though Copy is available)
    #[allow(clippy::clone_on_copy)]
    let clone = clock.clone();
    assert_eq!(clone.interval(), 64);
}

// ============================================================================
// ModulationSource Tests with ConstantModulator
// ============================================================================

/// A simple modulation source that outputs a constant value.
/// Used for testing the ModulationSource trait.
struct ConstantModulator {
    value: f32,
    update_count: u32,
}

impl ConstantModulator {
    fn new(value: f32) -> Self {
        Self {
            value,
            update_count: 0,
        }
    }

    fn update_count(&self) -> u32 {
        self.update_count
    }
}

impl ModulationSource for ConstantModulator {
    fn reset(&mut self) {
        self.update_count = 0;
    }

    fn update(&mut self, _timebase: &Timebase) {
        self.update_count += 1;
    }

    fn value(&self) -> f32 {
        self.value
    }
}

#[test]
fn test_modulation_source_constant_value() {
    let modulator = ConstantModulator::new(0.5);
    assert_eq!(modulator.value(), 0.5);
}

#[test]
fn test_modulation_source_value_before_update() {
    // value() should return sensible default before first update
    let modulator = ConstantModulator::new(1.0);
    assert_eq!(modulator.value(), 1.0);
}

#[test]
fn test_modulation_source_update_receives_timebase() {
    let mut modulator = ConstantModulator::new(0.5);
    let timebase = Timebase::new(44100.0);

    modulator.update(&timebase);

    assert_eq!(modulator.update_count(), 1);
}

#[test]
fn test_modulation_source_reset() {
    let mut modulator = ConstantModulator::new(0.5);
    let timebase = Timebase::new(44100.0);

    modulator.update(&timebase);
    modulator.update(&timebase);
    assert_eq!(modulator.update_count(), 2);

    modulator.reset();
    assert_eq!(modulator.update_count(), 0);
}

#[test]
fn test_modulation_source_with_control_rate_clock() {
    let mut clock = ControlRateClock::new(64);
    let mut modulator = ConstantModulator::new(0.5);
    let mut timebase = Timebase::new(44100.0);

    // Process 128 samples
    timebase.advance_block(128);

    for _offset in clock.advance(128) {
        modulator.update(&timebase);
    }

    // Should have 2 updates (at 0 and 64)
    assert_eq!(modulator.update_count(), 2);
}

#[test]
fn test_modulation_source_multiple_blocks() {
    let mut clock = ControlRateClock::new(64);
    let mut modulator = ConstantModulator::new(0.5);
    let mut timebase = Timebase::new(44100.0);

    // Process 4 blocks of 64 samples each
    for _ in 0..4 {
        timebase.advance_block(64);

        for _offset in clock.advance(64) {
            modulator.update(&timebase);
        }
    }

    // Should have 4 updates (1 per block)
    assert_eq!(modulator.update_count(), 4);
}

/// A linear ramp modulator for testing time-based modulation
struct LinearRampModulator {
    current_value: f32,
    increment: f32,
}

impl LinearRampModulator {
    fn new(start: f32, increment: f32) -> Self {
        Self {
            current_value: start,
            increment,
        }
    }
}

impl ModulationSource for LinearRampModulator {
    fn reset(&mut self) {
        self.current_value = 0.0;
    }

    fn update(&mut self, _timebase: &Timebase) {
        self.current_value += self.increment;
    }

    fn value(&self) -> f32 {
        self.current_value
    }
}

#[test]
fn test_linear_ramp_modulator() {
    let mut ramp = LinearRampModulator::new(0.0, 0.1);
    let timebase = Timebase::new(44100.0);

    assert_eq!(ramp.value(), 0.0);

    ramp.update(&timebase);
    assert!((ramp.value() - 0.1).abs() < 1e-6);

    ramp.update(&timebase);
    assert!((ramp.value() - 0.2).abs() < 1e-6);
}

#[test]
fn test_modulation_source_timing_accuracy() {
    // Verify that updates happen at exactly the right sample boundaries
    let mut clock = ControlRateClock::new(64);
    let mut update_offsets: Vec<u32> = Vec::new();

    // Process 256 samples
    for offset in clock.advance(256) {
        update_offsets.push(offset);
    }

    // Should have updates at 0, 64, 128, 192
    assert_eq!(update_offsets, vec![0, 64, 128, 192]);
}

#[test]
fn test_control_rate_with_irregular_block_sizes() {
    let mut clock = ControlRateClock::new(64);
    let mut total_updates = 0;

    // Process blocks of varying sizes
    let block_sizes = [100, 50, 128, 34, 200];

    for &size in &block_sizes {
        total_updates += clock.advance(size).count();
    }

    // Total samples: 100 + 50 + 128 + 34 + 200 = 512
    // With 64-sample interval: 512 / 64 = 8 updates
    assert_eq!(total_updates, 8);
}
