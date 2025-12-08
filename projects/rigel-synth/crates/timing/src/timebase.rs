//! Sample-accurate timing context for DSP modules.

/// Global timing context for the synthesizer.
///
/// Provides sample-accurate timing information that is consistent
/// across all DSP modules during a single audio block.
///
/// # Example
///
/// ```ignore
/// let mut timebase = Timebase::new(44100.0);
///
/// // Process blocks
/// loop {
///     timebase.advance_block(64);
///
///     // All queries within block return consistent values
///     let pos = timebase.sample_position();
///     let time_sec = timebase.samples_to_seconds(pos);
///
///     // Process audio...
/// }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Timebase {
    /// Global sample counter (starts at 0, advances monotonically)
    sample_position: u64,
    /// Current sample rate in Hz
    sample_rate: f32,
    /// Sample position at start of current block
    block_start: u64,
    /// Number of samples in current block
    block_size: u32,
}

impl Default for Timebase {
    fn default() -> Self {
        Self {
            sample_position: 0,
            sample_rate: 44100.0,
            block_start: 0,
            block_size: 0,
        }
    }
}

impl Timebase {
    /// Create a new timebase with the specified sample rate.
    ///
    /// Sample position starts at 0.
    ///
    /// # Arguments
    /// * `sample_rate` - Sample rate in Hz (e.g., 44100.0)
    ///
    /// # Panics
    /// Panics if `sample_rate <= 0.0`
    pub fn new(sample_rate: f32) -> Self {
        assert!(sample_rate > 0.0, "Sample rate must be positive");
        Self {
            sample_position: 0,
            sample_rate,
            block_start: 0,
            block_size: 0,
        }
    }

    /// Get the current global sample position.
    ///
    /// This value increases monotonically and never resets during a session.
    #[inline]
    pub fn sample_position(&self) -> u64 {
        self.sample_position
    }

    /// Get the current sample rate in Hz.
    #[inline]
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    /// Get the sample position at the start of the current block.
    #[inline]
    pub fn block_start(&self) -> u64 {
        self.block_start
    }

    /// Get the size of the current block in samples.
    #[inline]
    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    /// Convert a sample count to time in seconds.
    ///
    /// # Arguments
    /// * `samples` - Number of samples
    ///
    /// # Returns
    /// Time in seconds
    #[inline]
    pub fn samples_to_seconds(&self, samples: u64) -> f64 {
        samples as f64 / self.sample_rate as f64
    }

    /// Convert time in seconds to sample count.
    ///
    /// # Arguments
    /// * `seconds` - Time in seconds
    ///
    /// # Returns
    /// Number of samples (rounded)
    #[inline]
    pub fn seconds_to_samples(&self, seconds: f64) -> u64 {
        libm::round(seconds * self.sample_rate as f64) as u64
    }

    /// Convert milliseconds to sample count.
    ///
    /// # Arguments
    /// * `ms` - Time in milliseconds
    ///
    /// # Returns
    /// Number of samples
    #[inline]
    pub fn ms_to_samples(&self, ms: f32) -> u32 {
        libm::roundf((ms / 1000.0) * self.sample_rate) as u32
    }

    /// Advance to the next block.
    ///
    /// Called once at the start of each audio processing block.
    /// Updates `block_start` to the previous `sample_position` and
    /// increments `sample_position` by the block size.
    ///
    /// # Arguments
    /// * `block_size` - Size of the new block in samples
    #[inline]
    pub fn advance_block(&mut self, block_size: u32) {
        self.block_start = self.sample_position;
        self.block_size = block_size;
        self.sample_position += block_size as u64;
    }

    /// Update the sample rate.
    ///
    /// Preserves the sample position (does not reset).
    ///
    /// # Arguments
    /// * `new_rate` - New sample rate in Hz
    ///
    /// # Panics
    /// Panics if `new_rate <= 0.0`
    pub fn set_sample_rate(&mut self, new_rate: f32) {
        assert!(new_rate > 0.0, "Sample rate must be positive");
        self.sample_rate = new_rate;
    }

    /// Reset to initial state.
    ///
    /// Sets sample position back to 0.
    pub fn reset(&mut self) {
        self.sample_position = 0;
        self.block_start = 0;
        self.block_size = 0;
    }
}
