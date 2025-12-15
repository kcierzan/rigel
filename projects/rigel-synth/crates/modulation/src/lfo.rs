//! LFO implementation with SIMD-accelerated interpolation.

use crate::rate::LfoRateMode;
use crate::simd_rng::SimdXorshift128;
use crate::traits::{ModulationSource, SimdAwareComponent};
use crate::waveshape::LfoWaveshape;
use rigel_math::interpolate::{cubic_hermite, lerp};
use rigel_math::{DefaultSimdVector, SimdVector};
use rigel_timing::Timebase;

/// Default block size for sample cache.
const CACHE_SIZE: usize = 64;

/// LFO phase behavior on note events.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LfoPhaseMode {
    /// Phase continues uninterrupted on note events.
    /// LFO runs independently of note triggers.
    #[default]
    FreeRunning,

    /// Phase resets to start_phase on note-on events.
    /// Ensures consistent modulation from note start.
    Retrigger,
}

/// LFO output value range.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LfoPolarity {
    /// Bipolar output range [-1.0, 1.0].
    /// Modulation oscillates around zero.
    #[default]
    Bipolar,

    /// Unipolar output range [0.0, 1.0].
    /// Modulation is always positive.
    Unipolar,
}

/// Interpolation strategy for smooth LFO output.
///
/// Controls how values are interpolated between control-rate updates.
/// Higher quality interpolation reduces "zipper" artifacts for fast LFOs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum InterpolationStrategy {
    /// Linear interpolation (fastest, good for most use cases).
    /// Produces straight-line segments between control points.
    #[default]
    Linear,

    /// Cubic Hermite interpolation with analytical tangents.
    /// Produces smooth curves with continuous first derivatives.
    /// Best for slow LFOs where smoothness is audible.
    CubicHermite,
}

/// Low Frequency Oscillator for parameter modulation.
///
/// The LFO generates periodic waveforms that can modulate synthesizer parameters.
/// It supports multiple waveshapes, tempo synchronization, and phase control.
///
/// # Control-Rate Processing with Interpolation
///
/// The LFO operates at control rate (typically once per 64-sample block) but
/// provides smooth per-sample values through interpolation:
///
/// - `update()` advances phase and computes target values (call once per block)
/// - `generate_block()` fills a buffer with interpolated samples (SIMD-optimized)
/// - `sample()` returns individual samples from an internal cache
///
/// # Real-Time Safety
///
/// All LFO operations are real-time safe:
/// - No heap allocations
/// - Constant-time operations
/// - Copy/Clone semantics
///
/// # Example
///
/// ```ignore
/// use rigel_modulation::{Lfo, LfoWaveshape, LfoRateMode, InterpolationStrategy};
/// use rigel_timing::Timebase;
///
/// let mut lfo = Lfo::new();
/// lfo.set_waveshape(LfoWaveshape::Sine);
/// lfo.set_rate(LfoRateMode::Hz(2.0));
/// lfo.set_interpolation(InterpolationStrategy::CubicHermite);
///
/// let mut timebase = Timebase::new(44100.0);
/// timebase.advance_block(64);
///
/// lfo.update(&timebase);
///
/// // Block-based processing (most efficient)
/// let mut output = [0.0f32; 64];
/// lfo.generate_block(&mut output);
///
/// // Or single-sample access (uses internal SIMD cache)
/// for _ in 0..64 {
///     let value = lfo.sample();
/// }
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Lfo {
    // ─────────────────────────────────────────────────────────────────────
    // Configuration (set once or via setters)
    // ─────────────────────────────────────────────────────────────────────
    waveshape: LfoWaveshape,
    rate_mode: LfoRateMode,
    phase_mode: LfoPhaseMode,
    polarity: LfoPolarity,
    start_phase: f32,
    pulse_width: f32,
    interpolation: InterpolationStrategy,

    // ─────────────────────────────────────────────────────────────────────
    // Runtime state (changes during processing)
    // ─────────────────────────────────────────────────────────────────────
    phase: f32,
    phase_increment: f32, // How much phase changes per block (for tangent scaling)
    previous_value: f32,
    target_value: f32,
    previous_tangent: f32,
    target_tangent: f32,
    rng: SimdXorshift128,
    held_value: f32,

    // ─────────────────────────────────────────────────────────────────────
    // Sample cache for efficient single-sample access
    // ─────────────────────────────────────────────────────────────────────
    sample_cache: [f32; CACHE_SIZE],
    cache_index: usize,
    cache_valid: bool,
}

impl Lfo {
    // ─────────────────────────────────────────────────────────────────────
    // Construction
    // ─────────────────────────────────────────────────────────────────────

    /// Create a new LFO with default settings.
    ///
    /// Defaults:
    /// - Waveshape: Sine
    /// - Rate: 1.0 Hz
    /// - Phase mode: FreeRunning
    /// - Polarity: Bipolar
    /// - Start phase: 0.0
    /// - Pulse width: 0.5
    /// - Interpolation: Linear
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an LFO with specific configuration.
    ///
    /// # Arguments
    /// * `waveshape` - The waveform shape to generate
    /// * `rate_mode` - How the LFO rate is determined (Hz or tempo-sync)
    /// * `phase_mode` - Whether phase resets on note triggers
    /// * `polarity` - Output range (bipolar or unipolar)
    pub fn with_config(
        waveshape: LfoWaveshape,
        rate_mode: LfoRateMode,
        phase_mode: LfoPhaseMode,
        polarity: LfoPolarity,
    ) -> Self {
        Self {
            waveshape,
            rate_mode,
            phase_mode,
            polarity,
            start_phase: 0.0,
            pulse_width: 0.5,
            interpolation: InterpolationStrategy::Linear,
            phase: 0.0,
            phase_increment: 0.0,
            previous_value: 0.0,
            target_value: 0.0,
            previous_tangent: 0.0,
            target_tangent: 0.0,
            rng: SimdXorshift128::default(),
            held_value: 0.0,
            sample_cache: [0.0; CACHE_SIZE],
            cache_index: CACHE_SIZE, // Force refresh on first sample()
            cache_valid: false,
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // Configuration Setters
    // ─────────────────────────────────────────────────────────────────────

    /// Set the LFO waveshape.
    pub fn set_waveshape(&mut self, waveshape: LfoWaveshape) {
        self.waveshape = waveshape;
        self.invalidate_cache();
    }

    /// Set the LFO rate mode.
    ///
    /// # Arguments
    /// * `rate` - Either a fixed Hz value or tempo-synchronized division
    ///
    /// # Panics
    /// Panics in debug builds if Hz rate is outside [0.01, 100.0] range.
    pub fn set_rate(&mut self, rate: LfoRateMode) {
        if let LfoRateMode::Hz(hz) = rate {
            debug_assert!(
                (0.01..=100.0).contains(&hz),
                "Hz rate must be in [0.01, 100.0], got {}",
                hz
            );
        }
        self.rate_mode = rate;
        self.invalidate_cache();
    }

    /// Set the phase mode (free-running or retrigger).
    pub fn set_phase_mode(&mut self, mode: LfoPhaseMode) {
        self.phase_mode = mode;
    }

    /// Set the output polarity.
    pub fn set_polarity(&mut self, polarity: LfoPolarity) {
        self.polarity = polarity;
        self.invalidate_cache();
    }

    /// Set the starting phase for reset/retrigger.
    ///
    /// # Arguments
    /// * `phase` - Normalized phase [0.0, 1.0] where 0.0 = 0 deg, 0.5 = 180 deg, etc.
    ///
    /// # Panics
    /// Panics in debug builds if phase is outside [0.0, 1.0] range.
    pub fn set_start_phase(&mut self, phase: f32) {
        debug_assert!(
            (0.0..=1.0).contains(&phase),
            "Start phase must be in [0.0, 1.0], got {}",
            phase
        );
        self.start_phase = phase;
    }

    /// Set the pulse width for Pulse waveshape.
    ///
    /// # Arguments
    /// * `width` - Duty cycle [0.01, 0.99] where 0.5 = square wave
    ///
    /// # Panics
    /// Panics in debug builds if width is outside [0.01, 0.99] range.
    pub fn set_pulse_width(&mut self, width: f32) {
        debug_assert!(
            (0.01..=0.99).contains(&width),
            "Pulse width must be in [0.01, 0.99], got {}",
            width
        );
        self.pulse_width = width;
        self.invalidate_cache();
    }

    /// Set the interpolation strategy.
    ///
    /// # Arguments
    /// * `strategy` - Linear for speed, CubicHermite for smoothness
    pub fn set_interpolation(&mut self, strategy: InterpolationStrategy) {
        self.interpolation = strategy;
        self.invalidate_cache();
    }

    /// Set the tempo (BPM) for tempo-sync mode.
    ///
    /// # Arguments
    /// * `bpm` - Beats per minute [1.0, 999.0]
    ///
    /// Note: This only affects the rate when in TempoSync mode.
    pub fn set_tempo(&mut self, bpm: f32) {
        if let LfoRateMode::TempoSync { division, .. } = self.rate_mode {
            self.rate_mode = LfoRateMode::TempoSync { division, bpm };
            self.invalidate_cache();
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // Configuration Getters
    // ─────────────────────────────────────────────────────────────────────

    /// Get the current waveshape.
    pub fn waveshape(&self) -> LfoWaveshape {
        self.waveshape
    }

    /// Get the current rate mode.
    pub fn rate_mode(&self) -> LfoRateMode {
        self.rate_mode
    }

    /// Get the current phase mode.
    pub fn phase_mode(&self) -> LfoPhaseMode {
        self.phase_mode
    }

    /// Get the current polarity.
    pub fn polarity(&self) -> LfoPolarity {
        self.polarity
    }

    /// Get the starting phase.
    pub fn start_phase(&self) -> f32 {
        self.start_phase
    }

    /// Get the pulse width.
    pub fn pulse_width(&self) -> f32 {
        self.pulse_width
    }

    /// Get the current interpolation strategy.
    pub fn interpolation(&self) -> InterpolationStrategy {
        self.interpolation
    }

    // ─────────────────────────────────────────────────────────────────────
    // Runtime State
    // ─────────────────────────────────────────────────────────────────────

    /// Get the current phase position.
    ///
    /// # Returns
    /// Phase in [0.0, 1.0) representing position in the cycle.
    pub fn phase(&self) -> f32 {
        self.phase
    }

    /// Get the effective rate in Hz.
    ///
    /// For Hz mode, returns the configured rate.
    /// For TempoSync mode, calculates rate from BPM and division.
    pub fn effective_rate_hz(&self) -> f32 {
        self.rate_mode.effective_hz()
    }

    // ─────────────────────────────────────────────────────────────────────
    // Block Processing
    // ─────────────────────────────────────────────────────────────────────

    /// Generate a block of interpolated samples.
    ///
    /// Fills the output buffer with smoothly interpolated values between
    /// the previous and target control-rate values. Uses SIMD acceleration
    /// when available.
    ///
    /// # Arguments
    /// * `output` - Buffer to fill with LFO values
    ///
    /// # Note
    /// Call `update()` before each block to advance the LFO state.
    pub fn generate_block(&self, output: &mut [f32]) {
        match self.waveshape {
            LfoWaveshape::Noise => self.generate_noise_block(output),
            LfoWaveshape::SampleAndHold => self.generate_sh_block(output),
            _ => self.generate_interpolated_block(output),
        }
    }

    /// Generate a block using a mutable reference (needed for noise generation).
    ///
    /// This is the primary block generation method that handles all waveshapes.
    pub fn generate_block_mut(&mut self, output: &mut [f32]) {
        match self.waveshape {
            LfoWaveshape::Noise => self.generate_noise_block_mut(output),
            LfoWaveshape::SampleAndHold => self.generate_sh_block(output),
            _ => self.generate_interpolated_block(output),
        }
    }

    /// Generate interpolated block (for smooth waveshapes).
    fn generate_interpolated_block(&self, output: &mut [f32]) {
        let block_size = output.len();
        if block_size == 0 {
            return;
        }

        let block_size_f = block_size as f32;
        let lanes = DefaultSimdVector::LANES;

        // Process in SIMD chunks
        let mut idx = 0;
        while idx + lanes <= block_size {
            // Build t vector for this chunk
            let mut t_values = [0.0f32; 16]; // Max lanes (AVX-512)
            for (i, t_val) in t_values[..lanes].iter_mut().enumerate() {
                *t_val = (idx + i) as f32 / block_size_f;
            }

            let t = DefaultSimdVector::from_slice(&t_values[..lanes]);
            let a = DefaultSimdVector::splat(self.previous_value);
            let b = DefaultSimdVector::splat(self.target_value);

            let result = match self.interpolation {
                InterpolationStrategy::Linear => lerp(a, b, t),
                InterpolationStrategy::CubicHermite => {
                    // Scale tangents by phase increment for proper interpolation
                    // Derivative is d(value)/d(phase), tangent is d(value)/dt = derivative * d(phase)/dt
                    let ta = DefaultSimdVector::splat(self.previous_tangent * self.phase_increment);
                    let tb = DefaultSimdVector::splat(self.target_tangent * self.phase_increment);
                    cubic_hermite(a, b, ta, tb, t)
                }
            };

            result.to_slice(&mut output[idx..idx + lanes]);
            idx += lanes;
        }

        // Handle remainder with scalar fallback
        while idx < block_size {
            let t = idx as f32 / block_size_f;
            output[idx] = match self.interpolation {
                InterpolationStrategy::Linear => {
                    self.previous_value + t * (self.target_value - self.previous_value)
                }
                InterpolationStrategy::CubicHermite => hermite_scalar(
                    self.previous_value,
                    self.target_value,
                    self.previous_tangent * self.phase_increment,
                    self.target_tangent * self.phase_increment,
                    t,
                ),
            };
            idx += 1;
        }

        // Apply polarity transformation
        self.apply_polarity_block(output);
    }

    /// Generate noise block (per-sample random values).
    fn generate_noise_block(&self, output: &mut [f32]) {
        // Noise needs mutable RNG, but we have immutable self here.
        // Fill with last known values (caller should use generate_block_mut for noise).
        for sample in output.iter_mut() {
            *sample = self.apply_polarity(self.target_value);
        }
    }

    /// Generate noise block with mutable RNG access.
    fn generate_noise_block_mut(&mut self, output: &mut [f32]) {
        self.rng.fill_buffer(output);
        self.apply_polarity_block(output);
    }

    /// Generate sample-and-hold block (constant value).
    fn generate_sh_block(&self, output: &mut [f32]) {
        let value = self.apply_polarity(self.held_value);
        let value_vec = DefaultSimdVector::splat(value);
        let lanes = DefaultSimdVector::LANES;

        let mut idx = 0;
        while idx + lanes <= output.len() {
            value_vec.to_slice(&mut output[idx..idx + lanes]);
            idx += lanes;
        }
        while idx < output.len() {
            output[idx] = value;
            idx += 1;
        }
    }

    /// Apply polarity transformation to a block.
    fn apply_polarity_block(&self, output: &mut [f32]) {
        if self.polarity == LfoPolarity::Unipolar {
            let half = DefaultSimdVector::splat(0.5);
            let one = DefaultSimdVector::splat(1.0);
            let lanes = DefaultSimdVector::LANES;

            let mut idx = 0;
            while idx + lanes <= output.len() {
                let v = DefaultSimdVector::from_slice(&output[idx..idx + lanes]);
                let result = v.add(one).mul(half);
                result.to_slice(&mut output[idx..idx + lanes]);
                idx += lanes;
            }
            while idx < output.len() {
                output[idx] = (output[idx] + 1.0) * 0.5;
                idx += 1;
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // Single-Sample Access
    // ─────────────────────────────────────────────────────────────────────

    /// Get the next interpolated sample value.
    ///
    /// Uses an internal SIMD-generated cache for efficiency. When the cache
    /// is exhausted, automatically regenerates a new block.
    ///
    /// This method advances the internal read position - each call returns
    /// the next sample in sequence.
    ///
    /// # Example
    /// ```ignore
    /// for i in 0..1024 {
    ///     let mod_value = lfo.sample();  // Efficient single-sample access
    ///     apply_modulation(mod_value);
    /// }
    /// ```
    #[inline]
    pub fn sample(&mut self) -> f32 {
        if !self.cache_valid || self.cache_index >= CACHE_SIZE {
            self.refresh_cache();
        }

        let value = self.sample_cache[self.cache_index];
        self.cache_index += 1;
        value
    }

    /// Refresh internal sample cache using SIMD block generation.
    fn refresh_cache(&mut self) {
        // Inline block generation to avoid double-mutable-borrow
        match self.waveshape {
            LfoWaveshape::Noise => {
                self.rng.fill_buffer(&mut self.sample_cache);
                self.apply_polarity_cache();
            }
            LfoWaveshape::SampleAndHold => {
                let value = self.apply_polarity(self.held_value);
                for sample in self.sample_cache.iter_mut() {
                    *sample = value;
                }
            }
            _ => {
                self.generate_interpolated_cache();
            }
        }
        self.cache_index = 0;
        self.cache_valid = true;
    }

    /// Generate interpolated samples into the cache.
    fn generate_interpolated_cache(&mut self) {
        let block_size_f = CACHE_SIZE as f32;
        let lanes = DefaultSimdVector::LANES;

        // Process in SIMD chunks
        let mut idx = 0;
        while idx + lanes <= CACHE_SIZE {
            // Build t vector for this chunk
            let mut t_values = [0.0f32; 16]; // Max lanes (AVX-512)
            for (i, t_val) in t_values[..lanes].iter_mut().enumerate() {
                *t_val = (idx + i) as f32 / block_size_f;
            }

            let t = DefaultSimdVector::from_slice(&t_values[..lanes]);
            let a = DefaultSimdVector::splat(self.previous_value);
            let b = DefaultSimdVector::splat(self.target_value);

            let result = match self.interpolation {
                InterpolationStrategy::Linear => lerp(a, b, t),
                InterpolationStrategy::CubicHermite => {
                    // Scale tangents by phase increment for proper interpolation
                    // Derivative is d(value)/d(phase), tangent is d(value)/dt = derivative * d(phase)/dt
                    let ta = DefaultSimdVector::splat(self.previous_tangent * self.phase_increment);
                    let tb = DefaultSimdVector::splat(self.target_tangent * self.phase_increment);
                    cubic_hermite(a, b, ta, tb, t)
                }
            };

            result.to_slice(&mut self.sample_cache[idx..idx + lanes]);
            idx += lanes;
        }

        // Handle remainder with scalar fallback
        while idx < CACHE_SIZE {
            let t = idx as f32 / block_size_f;
            self.sample_cache[idx] = match self.interpolation {
                InterpolationStrategy::Linear => {
                    self.previous_value + t * (self.target_value - self.previous_value)
                }
                InterpolationStrategy::CubicHermite => hermite_scalar(
                    self.previous_value,
                    self.target_value,
                    self.previous_tangent * self.phase_increment,
                    self.target_tangent * self.phase_increment,
                    t,
                ),
            };
            idx += 1;
        }

        // Apply polarity transformation
        self.apply_polarity_cache();
    }

    /// Apply polarity transformation to the cache.
    fn apply_polarity_cache(&mut self) {
        if self.polarity == LfoPolarity::Unipolar {
            for sample in self.sample_cache.iter_mut() {
                *sample = (*sample + 1.0) * 0.5;
            }
        }
    }

    /// Invalidate cache (called after parameter changes).
    fn invalidate_cache(&mut self) {
        self.cache_valid = false;
        self.cache_index = CACHE_SIZE; // Force refresh on next sample()
    }

    // ─────────────────────────────────────────────────────────────────────
    // Triggering
    // ─────────────────────────────────────────────────────────────────────

    /// Trigger phase reset.
    ///
    /// Call this on note-on events. The effect depends on phase_mode:
    /// - FreeRunning: No effect
    /// - Retrigger: Phase resets to start_phase
    ///
    /// For sample-and-hold waveshape, also samples a new random value.
    pub fn trigger(&mut self) {
        match self.phase_mode {
            LfoPhaseMode::FreeRunning => {
                // No effect - phase continues
            }
            LfoPhaseMode::Retrigger => {
                self.phase = self.start_phase;
                // Sample new S&H value on trigger
                if self.waveshape == LfoWaveshape::SampleAndHold {
                    self.held_value = self.rng.next_f32();
                }
                self.invalidate_cache();
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // Internal Helpers
    // ─────────────────────────────────────────────────────────────────────

    /// Apply polarity scaling to a bipolar value.
    #[inline]
    fn apply_polarity(&self, bipolar_value: f32) -> f32 {
        match self.polarity {
            LfoPolarity::Bipolar => bipolar_value,
            LfoPolarity::Unipolar => (bipolar_value + 1.0) * 0.5,
        }
    }

    /// Compute the raw waveshape value at the current phase.
    #[inline]
    fn compute_value_at_phase(&self, phase: f32) -> f32 {
        let noise_value = 0.0; // Not used for interpolated shapes
        self.waveshape
            .generate(phase, self.pulse_width, self.held_value, noise_value)
    }
}

impl Default for Lfo {
    fn default() -> Self {
        Self {
            waveshape: LfoWaveshape::Sine,
            rate_mode: LfoRateMode::Hz(1.0),
            phase_mode: LfoPhaseMode::FreeRunning,
            polarity: LfoPolarity::Bipolar,
            start_phase: 0.0,
            pulse_width: 0.5,
            interpolation: InterpolationStrategy::Linear,
            phase: 0.0,
            phase_increment: 0.0,
            previous_value: 0.0,
            target_value: 0.0,
            previous_tangent: 0.0,
            target_tangent: 0.0,
            rng: SimdXorshift128::new(0x1234_5678),
            held_value: 0.0,
            sample_cache: [0.0; CACHE_SIZE],
            cache_index: CACHE_SIZE, // Force refresh on first sample()
            cache_valid: false,
        }
    }
}

impl ModulationSource for Lfo {
    fn reset(&mut self, _timebase: &Timebase) {
        self.phase = self.start_phase;

        // Sample initial S&H value
        if self.waveshape == LfoWaveshape::SampleAndHold {
            self.held_value = self.rng.next_f32();
        }

        // Compute initial value and tangent
        // For noise, generate a random value
        if self.waveshape == LfoWaveshape::Noise {
            self.previous_value = self.rng.next_f32();
        } else {
            self.previous_value = self.compute_value_at_phase(self.phase);
        }
        self.target_value = self.previous_value;
        self.previous_tangent = self.waveshape.derivative(self.phase, self.pulse_width);
        self.target_tangent = self.previous_tangent;

        self.invalidate_cache();
    }

    fn update(&mut self, timebase: &Timebase) {
        // Store previous state for interpolation
        self.previous_value = self.target_value;
        self.previous_tangent = self.target_tangent;

        let elapsed_samples = timebase.block_size() as f32;
        let rate_hz = self.effective_rate_hz();
        let sample_rate = timebase.sample_rate();

        // Calculate and store phase increment (for tangent scaling in generate_block)
        self.phase_increment = rate_hz * elapsed_samples / sample_rate;

        // Store old phase to detect cycle wrap
        let old_phase = self.phase;

        // Advance phase and wrap to [0.0, 1.0)
        let new_phase = self.phase + self.phase_increment;
        self.phase = new_phase - libm::floorf(new_phase);

        // Handle negative phase (shouldn't happen, but defensive)
        if self.phase < 0.0 {
            self.phase += 1.0;
        }

        // Detect cycle wrap for S&H
        let wrapped = self.phase < old_phase;
        if wrapped && self.waveshape == LfoWaveshape::SampleAndHold {
            self.held_value = self.rng.next_f32();
        }

        // Compute target value and tangent for interpolation
        // For noise, generate a new random value each update
        if self.waveshape == LfoWaveshape::Noise {
            self.target_value = self.rng.next_f32();
        } else {
            self.target_value = self.compute_value_at_phase(self.phase);
        }
        self.target_tangent = self.waveshape.derivative(self.phase, self.pulse_width);

        // Invalidate cache since state changed
        self.invalidate_cache();
    }

    fn value(&self) -> f32 {
        // Return the target value (most recent computed value)
        self.apply_polarity(self.target_value)
    }
}

impl SimdAwareComponent for Lfo {
    type Vector = DefaultSimdVector;
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper Functions
// ─────────────────────────────────────────────────────────────────────────────

/// Scalar cubic Hermite interpolation for remainder samples.
#[inline]
fn hermite_scalar(a: f32, b: f32, tangent_a: f32, tangent_b: f32, t: f32) -> f32 {
    let t2 = t * t;
    let t3 = t2 * t;

    // Hermite basis functions
    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = 3.0 * t2 - 2.0 * t3;
    let h11 = t3 - t2;

    h00 * a + h10 * tangent_a + h01 * b + h11 * tangent_b
}
