//! Control-rate scheduling with carry-over remainder tracking.

/// Maximum number of control rate updates that can occur within a single block.
/// This is sufficient for block sizes up to 1024 samples with interval of 1.
/// Reserved for future use in stack-allocated iterator optimization.
#[allow(dead_code)]
const MAX_UPDATES_PER_BLOCK: usize = 16;

/// Default target interval in milliseconds for timebased control rate.
/// ~1.5ms matches classic DX7/SY99 behavior for smooth envelopes.
pub const DEFAULT_CONTROL_RATE_MS: f32 = 1.5;

/// Manages timing for control-rate updates.
///
/// Ensures modulation sources update at consistent sample boundaries
/// regardless of block size. Uses carry-over remainder tracking to
/// maintain exact timing across block boundaries.
///
/// # Example
///
/// ```ignore
/// use rigel_timing::{ControlRateClock, Timebase, ModulationSource};
///
/// let mut clock = ControlRateClock::new(64);
/// let mut timebase = Timebase::new(44100.0);
///
/// // Process a 128-sample block
/// timebase.advance_block(128);
/// for offset in clock.advance(128) {
///     // offset will be 0, 64 for a 128-sample block
///     // lfo.update(&timebase);
/// }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ControlRateClock {
    /// Update interval in samples (power of 2: 1, 8, 16, 32, 64, 128)
    interval: u32,
    /// Samples accumulated since last update (0..interval)
    remainder: u32,
}

impl ControlRateClock {
    /// Create a new control rate clock.
    ///
    /// # Arguments
    /// * `interval` - Update interval in samples (must be power of 2: 1, 8, 16, 32, 64, 128)
    ///
    /// # Panics
    /// Panics if interval is not a valid power of 2 in the allowed range (1-128)
    pub fn new(interval: u32) -> Self {
        assert!(interval > 0 && interval <= 128, "Interval must be 1-128");
        assert!(interval.is_power_of_two(), "Interval must be a power of 2");

        Self {
            interval,
            remainder: 0,
        }
    }

    /// Get the update interval in samples.
    #[inline]
    pub fn interval(&self) -> u32 {
        self.interval
    }

    /// Advance the clock by one block and get update points.
    ///
    /// Returns an iterator of sample offsets within the block where
    /// updates should occur.
    ///
    /// # Arguments
    /// * `block_size` - Size of the block in samples
    ///
    /// # Returns
    /// Iterator yielding sample offsets where updates occur
    pub fn advance(&mut self, block_size: u32) -> ControlRateUpdates {
        // Calculate the first update point within this block
        // If remainder is 0, first update is at 0
        // If remainder > 0, first update is at (interval - remainder)
        let first_offset = if self.remainder == 0 {
            0
        } else {
            self.interval - self.remainder
        };

        let updates = ControlRateUpdates {
            interval: self.interval,
            block_size,
            next_offset: first_offset,
        };

        // Update remainder for next block
        self.remainder = (self.remainder + block_size) % self.interval;

        updates
    }

    /// Reset the clock phase.
    ///
    /// Next block will start with remainder = 0 (update at sample 0).
    pub fn reset(&mut self) {
        self.remainder = 0;
    }
}

/// Iterator over control rate update points within a block.
///
/// Yields sample offsets (0..block_size) where modulation sources
/// should be updated.
#[derive(Debug, Clone)]
pub struct ControlRateUpdates {
    /// Update interval in samples
    interval: u32,
    /// Total block size
    block_size: u32,
    /// Next offset to yield (or >= block_size if done)
    next_offset: u32,
}

impl Iterator for ControlRateUpdates {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_offset < self.block_size {
            let current = self.next_offset;
            self.next_offset += self.interval;
            Some(current)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.next_offset >= self.block_size {
            (0, Some(0))
        } else {
            let remaining = (self.block_size - self.next_offset).div_ceil(self.interval);
            (remaining as usize, Some(remaining as usize))
        }
    }
}

impl ExactSizeIterator for ControlRateUpdates {}

/// Sample-rate independent control rate clock.
///
/// Computes update intervals from a target time in milliseconds,
/// automatically adjusting when sample rate changes. Rounds to
/// power-of-2 intervals for efficient remainder tracking.
///
/// # Example
///
/// ```ignore
/// use rigel_timing::TimebasedControlRateClock;
///
/// // Create clock targeting ~1.5ms update interval
/// let mut clock = TimebasedControlRateClock::new(1.5, 44100.0);
/// assert_eq!(clock.interval(), 64); // Rounds to nearest power of 2
///
/// // At 96kHz, interval adjusts automatically
/// clock.set_sample_rate(96000.0);
/// assert_eq!(clock.interval(), 128); // ~1.33ms at 96kHz
/// ```
#[derive(Debug, Clone, Copy)]
pub struct TimebasedControlRateClock {
    /// Target update interval in milliseconds
    target_interval_ms: f32,
    /// Computed sample interval (power of 2)
    sample_interval: u32,
    /// Samples accumulated since last update (0..sample_interval)
    remainder: u32,
    /// Current sample rate in Hz
    sample_rate: f32,
}

impl TimebasedControlRateClock {
    /// Create a new timebased control rate clock.
    ///
    /// # Arguments
    /// * `target_interval_ms` - Target update interval in milliseconds (e.g., 1.5)
    /// * `sample_rate` - Sample rate in Hz (e.g., 44100.0)
    ///
    /// # Panics
    /// Panics if `target_interval_ms <= 0.0` or `sample_rate <= 0.0`
    pub fn new(target_interval_ms: f32, sample_rate: f32) -> Self {
        assert!(target_interval_ms > 0.0, "Target interval must be positive");
        assert!(sample_rate > 0.0, "Sample rate must be positive");

        let sample_interval = Self::calculate_interval(target_interval_ms, sample_rate);

        Self {
            target_interval_ms,
            sample_interval,
            remainder: 0,
            sample_rate,
        }
    }

    /// Create with default ~1.5ms target interval.
    ///
    /// # Arguments
    /// * `sample_rate` - Sample rate in Hz
    pub fn with_default_interval(sample_rate: f32) -> Self {
        Self::new(DEFAULT_CONTROL_RATE_MS, sample_rate)
    }

    /// Calculate power-of-2 interval from target milliseconds.
    fn calculate_interval(target_ms: f32, sample_rate: f32) -> u32 {
        // Convert ms to samples: samples = (ms / 1000) * sample_rate
        let target_samples = (target_ms / 1000.0) * sample_rate;

        // Round to nearest power of 2, clamped to valid range [1, 128]
        let rounded = Self::nearest_power_of_two(target_samples as u32);
        rounded.clamp(1, 128)
    }

    /// Find nearest power of 2 to the given value.
    fn nearest_power_of_two(n: u32) -> u32 {
        if n == 0 {
            return 1;
        }

        // Find the highest set bit position
        let high_bit = 31 - n.leading_zeros();
        let lower = 1u32 << high_bit;
        let upper = lower << 1;

        // Return whichever is closer
        if upper.saturating_sub(n) < n.saturating_sub(lower) {
            upper.min(128) // Clamp to max
        } else {
            lower.max(1) // Clamp to min
        }
    }

    /// Get the computed update interval in samples.
    #[inline]
    pub fn interval(&self) -> u32 {
        self.sample_interval
    }

    /// Get the target interval in milliseconds.
    #[inline]
    pub fn target_interval_ms(&self) -> f32 {
        self.target_interval_ms
    }

    /// Get the actual interval in milliseconds (may differ from target due to power-of-2 rounding).
    #[inline]
    pub fn actual_interval_ms(&self) -> f32 {
        (self.sample_interval as f32 / self.sample_rate) * 1000.0
    }

    /// Get the current sample rate.
    #[inline]
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    /// Update the sample rate.
    ///
    /// Recalculates the sample interval to maintain the target
    /// time interval at the new rate. Resets the remainder.
    ///
    /// # Arguments
    /// * `new_rate` - New sample rate in Hz
    ///
    /// # Panics
    /// Panics if `new_rate <= 0.0`
    pub fn set_sample_rate(&mut self, new_rate: f32) {
        assert!(new_rate > 0.0, "Sample rate must be positive");
        self.sample_rate = new_rate;
        self.sample_interval = Self::calculate_interval(self.target_interval_ms, new_rate);
        self.remainder = 0;
    }

    /// Advance the clock by one block and get update points.
    ///
    /// Returns an iterator of sample offsets within the block where
    /// updates should occur.
    ///
    /// # Arguments
    /// * `block_size` - Size of the block in samples
    ///
    /// # Returns
    /// Iterator yielding sample offsets where updates occur
    pub fn advance(&mut self, block_size: u32) -> ControlRateUpdates {
        // Calculate the first update point within this block
        let first_offset = if self.remainder == 0 {
            0
        } else {
            self.sample_interval - self.remainder
        };

        let updates = ControlRateUpdates {
            interval: self.sample_interval,
            block_size,
            next_offset: first_offset,
        };

        // Update remainder for next block
        self.remainder = (self.remainder + block_size) % self.sample_interval;

        updates
    }

    /// Reset the clock phase.
    ///
    /// Next block will start with remainder = 0 (update at sample 0).
    pub fn reset(&mut self) {
        self.remainder = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timebased_clock_creation() {
        let clock = TimebasedControlRateClock::new(1.5, 44100.0);

        // 1.5ms at 44100Hz = 66.15 samples, rounds to 64 (power of 2)
        assert_eq!(clock.interval(), 64);
        assert_eq!(clock.target_interval_ms(), 1.5);
    }

    #[test]
    fn test_timebased_clock_sample_rates() {
        // Test at different sample rates
        let clock_44k = TimebasedControlRateClock::new(1.5, 44100.0);
        let clock_48k = TimebasedControlRateClock::new(1.5, 48000.0);
        let clock_96k = TimebasedControlRateClock::new(1.5, 96000.0);

        // 1.5ms at 44.1kHz ≈ 66 samples → 64
        assert_eq!(clock_44k.interval(), 64);

        // 1.5ms at 48kHz = 72 samples → 64
        assert_eq!(clock_48k.interval(), 64);

        // 1.5ms at 96kHz = 144 samples → 128
        assert_eq!(clock_96k.interval(), 128);
    }

    #[test]
    fn test_timebased_clock_set_sample_rate() {
        let mut clock = TimebasedControlRateClock::new(1.5, 44100.0);
        assert_eq!(clock.interval(), 64);

        clock.set_sample_rate(96000.0);
        assert_eq!(clock.interval(), 128);
        assert_eq!(clock.sample_rate(), 96000.0);
    }

    #[test]
    fn test_timebased_clock_advance() {
        let mut clock = TimebasedControlRateClock::new(1.5, 44100.0);

        // Process 128-sample block with 64-sample interval
        let mut updates = [0u32; 4];
        let mut count = 0;
        for offset in clock.advance(128) {
            updates[count] = offset;
            count += 1;
        }
        assert_eq!(count, 2);
        assert_eq!(updates[0], 0);
        assert_eq!(updates[1], 64);
    }

    #[test]
    fn test_timebased_clock_remainder() {
        let mut clock = TimebasedControlRateClock::new(1.5, 44100.0);

        // 100-sample block with 64-sample interval
        let mut updates1 = [0u32; 4];
        let mut count1 = 0;
        for offset in clock.advance(100) {
            updates1[count1] = offset;
            count1 += 1;
        }
        assert_eq!(count1, 2);
        assert_eq!(updates1[0], 0);
        assert_eq!(updates1[1], 64);

        // Next block should account for remainder (36 samples)
        // First update at 64 - 36 = 28
        let mut updates2 = [0u32; 4];
        let mut count2 = 0;
        for offset in clock.advance(100) {
            updates2[count2] = offset;
            count2 += 1;
        }
        assert_eq!(count2, 2);
        assert_eq!(updates2[0], 28);
        assert_eq!(updates2[1], 92);
    }

    #[test]
    fn test_timebased_clock_reset() {
        let mut clock = TimebasedControlRateClock::new(1.5, 44100.0);

        clock.advance(50); // Partial block
        clock.reset();

        let mut first_offset = None;
        for offset in clock.advance(128) {
            if first_offset.is_none() {
                first_offset = Some(offset);
            }
        }
        assert_eq!(first_offset, Some(0)); // Should start at 0 after reset
    }

    #[test]
    fn test_nearest_power_of_two() {
        assert_eq!(TimebasedControlRateClock::nearest_power_of_two(0), 1);
        assert_eq!(TimebasedControlRateClock::nearest_power_of_two(1), 1);
        assert_eq!(TimebasedControlRateClock::nearest_power_of_two(2), 2);
        assert_eq!(TimebasedControlRateClock::nearest_power_of_two(3), 2); // Equidistant, rounds down
        assert_eq!(TimebasedControlRateClock::nearest_power_of_two(5), 4);
        assert_eq!(TimebasedControlRateClock::nearest_power_of_two(6), 4); // Equidistant, rounds down
        assert_eq!(TimebasedControlRateClock::nearest_power_of_two(7), 8);
        assert_eq!(TimebasedControlRateClock::nearest_power_of_two(63), 64);
        assert_eq!(TimebasedControlRateClock::nearest_power_of_two(65), 64);
        assert_eq!(TimebasedControlRateClock::nearest_power_of_two(96), 64); // Equidistant, rounds down
        assert_eq!(TimebasedControlRateClock::nearest_power_of_two(97), 128); // Closer to 128
        assert_eq!(TimebasedControlRateClock::nearest_power_of_two(127), 128);
        assert_eq!(TimebasedControlRateClock::nearest_power_of_two(129), 128); // Clamped to max
    }

    #[test]
    fn test_default_interval() {
        let clock = TimebasedControlRateClock::with_default_interval(44100.0);
        assert_eq!(clock.target_interval_ms(), DEFAULT_CONTROL_RATE_MS);
    }

    #[test]
    fn test_actual_interval_ms() {
        let clock = TimebasedControlRateClock::new(1.5, 44100.0);

        // 64 samples at 44100Hz = 1.451ms
        let actual = clock.actual_interval_ms();
        assert!((actual - 1.451).abs() < 0.01);
    }
}
