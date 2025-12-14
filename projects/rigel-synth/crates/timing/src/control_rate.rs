//! Control-rate scheduling with carry-over remainder tracking.

/// Maximum number of control rate updates that can occur within a single block.
/// This is sufficient for block sizes up to 1024 samples with interval of 1.
/// Reserved for future use in stack-allocated iterator optimization.
#[allow(dead_code)]
const MAX_UPDATES_PER_BLOCK: usize = 16;

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
