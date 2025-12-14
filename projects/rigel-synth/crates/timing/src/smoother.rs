//! Parameter smoothing with configurable curve types.

use crate::{DEFAULT_SAMPLE_RATE, DEFAULT_SMOOTHING_TIME_MS};
use rigel_math::{fast_expf, fast_logf};

/// Threshold for exponential smoothing completion (0.1%)
const EXPONENTIAL_THRESHOLD: f32 = 0.001;

/// Absolute threshold for near-zero targets
const ABSOLUTE_THRESHOLD: f32 = 1e-6;

/// Minimum value for logarithmic smoothing (to avoid log(0))
const LOG_MIN_VALUE: f32 = 1e-6;

/// Type of smoothing curve to apply.
///
/// Controls how parameter values transition from current to target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SmoothingMode {
    /// No smoothing; value changes immediately.
    #[default]
    Instant,

    /// Linear interpolation; constant rate of change.
    /// Reaches target in exactly the configured smoothing time.
    Linear,

    /// Exponential (one-pole filter) smoothing.
    /// Asymptotically approaches target, terminates at 0.1% threshold.
    Exponential,

    /// Logarithmic smoothing for frequency/amplitude parameters.
    /// Perceptually linear; operates in log domain.
    Logarithmic,
}

/// Parameter smoother with configurable curve type.
///
/// Provides smooth transitions between parameter values to avoid
/// audible clicks and zipper noise.
///
/// # Example
///
/// ```ignore
/// use rigel_timing::{Smoother, SmoothingMode};
///
/// // Create smoother for filter cutoff
/// let mut cutoff = Smoother::new(
///     1000.0,                    // Initial value: 1000 Hz
///     SmoothingMode::Logarithmic, // Log for frequency
///     10.0,                       // 10ms smoothing time
///     44100.0,                    // Sample rate
/// );
///
/// // User changes cutoff
/// cutoff.set_target(5000.0);
///
/// // Process audio block
/// let mut buffer = [0.0f32; 64];
/// cutoff.process_block(&mut buffer);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Smoother {
    /// Current smoothed value
    current: f32,
    /// Target value to approach
    target: f32,
    /// Type of smoothing curve
    mode: SmoothingMode,
    /// Smoothing duration in milliseconds
    smoothing_time_ms: f32,
    /// Pre-calculated smoothing coefficient (for exponential)
    coefficient: f32,
    /// Cached sample rate for coefficient calculation
    sample_rate: f32,
    /// True if smoothing is currently active
    is_active: bool,
    /// Increment per sample (for linear mode)
    linear_increment: f32,
    /// Samples remaining until target reached (for linear mode)
    linear_samples_remaining: u32,
    /// Log of current value (for logarithmic mode)
    log_current: f32,
    /// Log of target value (for logarithmic mode)
    log_target: f32,
}

impl Default for Smoother {
    /// Default smoother: value 0.0, Linear mode, 5ms, 44100 Hz
    fn default() -> Self {
        Self::new(
            0.0,
            SmoothingMode::Linear,
            DEFAULT_SMOOTHING_TIME_MS,
            DEFAULT_SAMPLE_RATE,
        )
    }
}

impl Smoother {
    /// Create a new smoother with initial configuration.
    ///
    /// # Arguments
    /// * `initial_value` - Starting value
    /// * `mode` - Type of smoothing curve
    /// * `smoothing_time_ms` - Transition time in milliseconds
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Panics
    /// Panics if `smoothing_time_ms < 0.0` or `sample_rate <= 0.0`
    pub fn new(
        initial_value: f32,
        mode: SmoothingMode,
        smoothing_time_ms: f32,
        sample_rate: f32,
    ) -> Self {
        assert!(
            smoothing_time_ms >= 0.0,
            "Smoothing time must be non-negative"
        );
        assert!(sample_rate > 0.0, "Sample rate must be positive");

        let coefficient = Self::calculate_coefficient(smoothing_time_ms, sample_rate);
        let log_value = Self::safe_log(initial_value);

        Self {
            current: initial_value,
            target: initial_value,
            mode,
            smoothing_time_ms,
            coefficient,
            sample_rate,
            is_active: false,
            linear_increment: 0.0,
            linear_samples_remaining: 0,
            log_current: log_value,
            log_target: log_value,
        }
    }

    /// Create a new smoother with default settings.
    ///
    /// Uses Linear mode with 5ms smoothing time at 44100 Hz.
    pub fn with_defaults(initial_value: f32) -> Self {
        Self::new(
            initial_value,
            SmoothingMode::Linear,
            DEFAULT_SMOOTHING_TIME_MS,
            DEFAULT_SAMPLE_RATE,
        )
    }

    /// Calculate the exponential smoothing coefficient.
    ///
    /// Uses one-pole filter formulation: coeff = 1 - exp(-1 / (tau * sample_rate))
    /// where tau = smoothing_time_ms / 1000
    fn calculate_coefficient(smoothing_time_ms: f32, sample_rate: f32) -> f32 {
        if smoothing_time_ms <= 0.0 {
            return 1.0; // Instant transition
        }
        let tau = smoothing_time_ms / 1000.0;
        1.0 - fast_expf(-1.0 / (tau * sample_rate))
    }

    /// Safe logarithm that handles zero and negative values
    #[inline]
    fn safe_log(value: f32) -> f32 {
        let clamped = if value > LOG_MIN_VALUE {
            value
        } else {
            LOG_MIN_VALUE
        };
        fast_logf(clamped)
    }

    /// Set a new target value.
    ///
    /// The smoother will begin transitioning toward this value.
    ///
    /// # Arguments
    /// * `target` - New target value
    pub fn set_target(&mut self, target: f32) {
        if (target - self.target).abs() < ABSOLUTE_THRESHOLD && !self.is_active {
            return; // Already at target
        }

        self.target = target;
        self.is_active = true;

        match self.mode {
            SmoothingMode::Instant => {
                self.current = target;
                self.is_active = false;
            }
            SmoothingMode::Linear => {
                // Calculate samples needed and increment per sample
                let samples = ((self.smoothing_time_ms / 1000.0) * self.sample_rate) as u32;
                if samples > 0 {
                    self.linear_samples_remaining = samples;
                    self.linear_increment = (target - self.current) / samples as f32;
                } else {
                    self.current = target;
                    self.is_active = false;
                }
            }
            SmoothingMode::Exponential => {
                // Coefficient already calculated, just start smoothing
            }
            SmoothingMode::Logarithmic => {
                // Calculate log-domain target
                self.log_target = Self::safe_log(target);
            }
        }
    }

    /// Set value immediately without smoothing.
    ///
    /// Both current and target are set to the new value.
    ///
    /// # Arguments
    /// * `value` - Value to set immediately
    pub fn set_immediate(&mut self, value: f32) {
        self.current = value;
        self.target = value;
        self.is_active = false;
        self.linear_increment = 0.0;
        self.linear_samples_remaining = 0;
        self.log_current = Self::safe_log(value);
        self.log_target = self.log_current;
    }

    /// Get the current smoothed value.
    #[inline]
    pub fn current(&self) -> f32 {
        self.current
    }

    /// Get the target value.
    #[inline]
    pub fn target(&self) -> f32 {
        self.target
    }

    /// Check if smoothing is currently active.
    ///
    /// Returns `true` if current value differs from target (within tolerance).
    #[inline]
    pub fn is_active(&self) -> bool {
        self.is_active
    }

    /// Check if smoothing has reached the target (within threshold).
    #[inline]
    fn has_reached_target(&self) -> bool {
        let diff = (self.current - self.target).abs();

        // Use relative threshold for non-zero targets, absolute for near-zero
        if self.target.abs() > ABSOLUTE_THRESHOLD {
            diff / self.target.abs() < EXPONENTIAL_THRESHOLD
        } else {
            diff < ABSOLUTE_THRESHOLD
        }
    }

    /// Process one sample and advance the smoother.
    ///
    /// # Returns
    /// The new current value after this sample
    #[inline]
    pub fn process_sample(&mut self) -> f32 {
        if !self.is_active {
            return self.current;
        }

        match self.mode {
            SmoothingMode::Instant => {
                self.current = self.target;
                self.is_active = false;
            }
            SmoothingMode::Linear => {
                if self.linear_samples_remaining > 0 {
                    self.current += self.linear_increment;
                    self.linear_samples_remaining -= 1;

                    if self.linear_samples_remaining == 0 {
                        self.current = self.target; // Ensure exact target
                        self.is_active = false;
                    }
                } else {
                    self.current = self.target;
                    self.is_active = false;
                }
            }
            SmoothingMode::Exponential => {
                // One-pole filter: current = current + coeff * (target - current)
                self.current += self.coefficient * (self.target - self.current);

                // Check 0.1% threshold
                if self.has_reached_target() {
                    self.current = self.target;
                    self.is_active = false;
                }
            }
            SmoothingMode::Logarithmic => {
                // Smooth in log domain
                self.log_current += self.coefficient * (self.log_target - self.log_current);

                // Convert back to linear domain
                self.current = fast_expf(self.log_current);

                // Check threshold
                if self.has_reached_target() {
                    self.current = self.target;
                    self.log_current = self.log_target;
                    self.is_active = false;
                }
            }
        }

        self.current
    }

    /// Process a block of samples, filling the output buffer.
    ///
    /// # Arguments
    /// * `output` - Mutable slice to fill with smoothed values
    pub fn process_block(&mut self, output: &mut [f32]) {
        for sample in output.iter_mut() {
            *sample = self.process_sample();
        }
    }

    /// Update the smoothing time.
    ///
    /// Takes effect on the next target change.
    ///
    /// # Arguments
    /// * `ms` - New smoothing time in milliseconds
    pub fn set_smoothing_time(&mut self, ms: f32) {
        assert!(ms >= 0.0, "Smoothing time must be non-negative");
        self.smoothing_time_ms = ms;
        self.coefficient = Self::calculate_coefficient(ms, self.sample_rate);
    }

    /// Update the smoothing mode.
    ///
    /// Takes effect on the next target change.
    ///
    /// # Arguments
    /// * `mode` - New smoothing mode
    pub fn set_mode(&mut self, mode: SmoothingMode) {
        self.mode = mode;
    }

    /// Update the sample rate.
    ///
    /// Recalculates internal coefficients.
    ///
    /// # Arguments
    /// * `sample_rate` - New sample rate in Hz
    pub fn set_sample_rate(&mut self, sample_rate: f32) {
        assert!(sample_rate > 0.0, "Sample rate must be positive");
        self.sample_rate = sample_rate;
        self.coefficient = Self::calculate_coefficient(self.smoothing_time_ms, sample_rate);

        // Recalculate linear increment if active
        if self.is_active && self.mode == SmoothingMode::Linear && self.linear_samples_remaining > 0
        {
            let remaining_distance = self.target - self.current;
            let new_samples = ((self.smoothing_time_ms / 1000.0) * sample_rate) as u32;
            if new_samples > 0 {
                self.linear_samples_remaining = new_samples;
                self.linear_increment = remaining_distance / new_samples as f32;
            }
        }
    }

    /// Reset the smoother to a specific value.
    ///
    /// Both current and target are set, smoothing is deactivated.
    ///
    /// # Arguments
    /// * `value` - Value to reset to
    pub fn reset(&mut self, value: f32) {
        self.set_immediate(value);
    }
}
