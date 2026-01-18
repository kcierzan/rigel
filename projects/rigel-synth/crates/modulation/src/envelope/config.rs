//! Envelope configuration types.
//!
//! This module defines:
//! - [`LoopConfig`] - Loop configuration for key-on segments
//! - [`EnvelopeConfig`] - Immutable envelope configuration

use super::rates::{linear_to_param_level, max_rate_for_sample_rate, seconds_to_rate};
use super::segment::Segment;

/// Loop configuration for envelope segments.
///
/// Allows the envelope to loop between two key-on segments
/// while the note is held, useful for rhythmic or evolving textures.
///
/// # Example
///
/// ```ignore
/// use rigel_modulation::envelope::LoopConfig;
///
/// // Loop between segments 2 and 4
/// let loop_cfg = LoopConfig::new(2, 4).unwrap();
///
/// // Or disable looping
/// let no_loop = LoopConfig::disabled();
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LoopConfig {
    /// Whether looping is enabled.
    pub enabled: bool,

    /// Index of first segment in loop (0-based).
    /// Must be < end_segment and within key-on range.
    pub start_segment: u8,

    /// Index of last segment in loop (0-based).
    /// After completing this segment, loop back to start_segment.
    pub end_segment: u8,
}

impl LoopConfig {
    /// Create disabled loop config.
    #[inline]
    pub const fn disabled() -> Self {
        Self {
            enabled: false,
            start_segment: 0,
            end_segment: 0,
        }
    }

    /// Create enabled loop config.
    ///
    /// Returns `None` if boundaries are invalid (start >= end).
    ///
    /// # Arguments
    ///
    /// * `start` - First segment index in the loop
    /// * `end` - Last segment index in the loop
    #[inline]
    pub const fn new(start: u8, end: u8) -> Option<Self> {
        if start < end {
            Some(Self {
                enabled: true,
                start_segment: start,
                end_segment: end,
            })
        } else {
            None
        }
    }

    /// Validate loop boundaries against key-on segment count.
    ///
    /// # Arguments
    ///
    /// * `key_on_segments` - Number of key-on segments in the envelope
    #[inline]
    pub const fn is_valid(&self, key_on_segments: usize) -> bool {
        !self.enabled
            || (self.start_segment < self.end_segment
                && (self.end_segment as usize) < key_on_segments)
    }

    /// Check if looping is enabled.
    #[inline]
    pub const fn is_enabled(&self) -> bool {
        self.enabled
    }
}

/// Envelope configuration (immutable after creation).
///
/// Uses const generics for segment count optimization. Different
/// synthesizer architectures use different segment counts:
///
/// - FM (DX7/SY77/SY99): 6 key-on + 2 release = 8 segments
/// - AWM (SY77/SY99): 5 key-on + 5 release = 10 segments
/// - 7-segment: 5 key-on + 2 release = 7 segments
///
/// # Type Parameters
///
/// * `KEY_ON_SEGS` - Number of key-on segments (attack/decay)
/// * `RELEASE_SEGS` - Number of release segments
#[derive(Debug, Clone, Copy)]
pub struct EnvelopeConfig<const KEY_ON_SEGS: usize, const RELEASE_SEGS: usize> {
    /// Key-on segments (attack, decay, etc.).
    pub key_on_segments: [Segment; KEY_ON_SEGS],

    /// Release segments (key-off behavior).
    pub release_segments: [Segment; RELEASE_SEGS],

    /// Rate scaling sensitivity (0-7).
    /// Higher = more rate variation across keyboard.
    pub rate_scaling: u8,

    /// Delay before attack begins (in samples).
    pub delay_samples: u32,

    /// Loop configuration for key-on segments.
    pub loop_config: LoopConfig,

    /// Sample rate for timing calculations.
    pub sample_rate: f32,

    /// Pre-computed maximum rate that ensures minimum segment time.
    ///
    /// This is cached at config creation to avoid O(100) search at every
    /// segment transition when rate scaling is applied. The value depends
    /// only on sample rate and is computed via [`max_rate_for_sample_rate`].
    cached_max_rate: u8,
}

impl<const K: usize, const R: usize> EnvelopeConfig<K, R> {
    /// Total number of segments.
    pub const TOTAL_SEGMENTS: usize = K + R;

    /// Create default configuration (all segments at max rate/level).
    ///
    /// Default configuration:
    /// - All key-on segments: rate=99, level=99 (instant full)
    /// - All release segments: rate=50, level=0 (medium fade)
    /// - No rate scaling, no delay, no looping
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate in Hz
    pub fn default_with_sample_rate(sample_rate: f32) -> Self {
        Self {
            key_on_segments: [Segment::new(99, 99); K],
            release_segments: [Segment::new(50, 0); R],
            rate_scaling: 0,
            delay_samples: 0,
            loop_config: LoopConfig::disabled(),
            sample_rate,
            cached_max_rate: max_rate_for_sample_rate(sample_rate),
        }
    }

    /// Create configuration from raw parameters.
    ///
    /// # Arguments
    ///
    /// * `key_on_segments` - Key-on segment configuration
    /// * `release_segments` - Release segment configuration
    /// * `rate_scaling` - Rate scaling sensitivity (0-7)
    /// * `delay_samples` - Delay before attack (in samples)
    /// * `loop_config` - Loop configuration
    /// * `sample_rate` - Sample rate in Hz
    pub fn new(
        key_on_segments: [Segment; K],
        release_segments: [Segment; R],
        rate_scaling: u8,
        delay_samples: u32,
        loop_config: LoopConfig,
        sample_rate: f32,
    ) -> Self {
        Self {
            key_on_segments,
            release_segments,
            rate_scaling: if rate_scaling > 7 { 7 } else { rate_scaling },
            delay_samples,
            loop_config,
            sample_rate,
            cached_max_rate: max_rate_for_sample_rate(sample_rate),
        }
    }

    /// Get the number of key-on segments.
    #[inline]
    pub const fn key_on_count(&self) -> usize {
        K
    }

    /// Get the number of release segments.
    #[inline]
    pub const fn release_count(&self) -> usize {
        R
    }

    /// Check if loop configuration is valid for this envelope.
    #[inline]
    pub const fn is_loop_valid(&self) -> bool {
        self.loop_config.is_valid(K)
    }

    /// Calculate delay in milliseconds.
    #[inline]
    pub fn delay_ms(&self) -> f32 {
        (self.delay_samples as f32 / self.sample_rate) * 1000.0
    }

    /// Set delay from milliseconds.
    #[inline]
    pub fn set_delay_ms(&mut self, ms: f32) {
        self.delay_samples = ((ms / 1000.0) * self.sample_rate) as u32;
    }

    /// Get cached maximum rate for this sample rate.
    ///
    /// This is the highest rate that still ensures minimum segment time
    /// (1.5ms) to prevent audible clicks from rate scaling.
    #[inline]
    pub fn max_rate(&self) -> u8 {
        self.cached_max_rate
    }
}

/// Type alias for 8-segment FM envelope configuration (6 key-on + 2 release).
pub type FmEnvelopeConfig = EnvelopeConfig<6, 2>;

/// Type alias for 10-segment AWM envelope configuration (5 key-on + 5 release).
pub type AwmEnvelopeConfig = EnvelopeConfig<5, 5>;

/// Type alias for 7-segment envelope configuration (5 key-on + 2 release).
pub type SevenSegEnvelopeConfig = EnvelopeConfig<5, 2>;

impl FmEnvelopeConfig {
    /// Create ADSR-style envelope from user-friendly time-based parameters.
    ///
    /// This builder converts familiar ADSR parameters (attack/decay/sustain/release
    /// in seconds and linear levels) into the multi-segment FM envelope format.
    ///
    /// The resulting envelope has:
    /// - Attack: Fast ramp to full level (rate based on attack_secs)
    /// - Decay: Transition to sustain level (rate based on decay_secs)
    /// - Sustain: Hold at sustain level (remaining key-on segments)
    /// - Release: Fade to silence (rate based on release_secs)
    ///
    /// # Arguments
    ///
    /// * `attack_secs` - Attack time in seconds (0.001 to 40.0)
    /// * `decay_secs` - Decay time in seconds (0.001 to 40.0)
    /// * `sustain_linear` - Sustain level (0.0 to 1.0, linear amplitude)
    /// * `release_secs` - Release time in seconds (0.001 to 40.0)
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Example
    ///
    /// ```ignore
    /// use rigel_modulation::envelope::FmEnvelopeConfig;
    ///
    /// // Create typical synth pad envelope
    /// let config = FmEnvelopeConfig::adsr(
    ///     0.5,    // 500ms attack
    ///     1.0,    // 1s decay
    ///     0.7,    // 70% sustain
    ///     2.0,    // 2s release
    ///     44100.0,
    /// );
    /// ```
    pub fn adsr(
        attack_secs: f32,
        decay_secs: f32,
        sustain_linear: f32,
        release_secs: f32,
        sample_rate: f32,
    ) -> Self {
        let attack_rate = seconds_to_rate(attack_secs, sample_rate);
        let decay_rate = seconds_to_rate(decay_secs, sample_rate);
        let sustain_level = linear_to_param_level(sustain_linear);
        let release_rate = seconds_to_rate(release_secs, sample_rate);

        Self {
            key_on_segments: [
                Segment::new(attack_rate, 99),           // Attack: ramp to full
                Segment::new(decay_rate, sustain_level), // Decay: to sustain level
                Segment::new(99, sustain_level),         // Hold at sustain (remaining segments)
                Segment::new(99, sustain_level),
                Segment::new(99, sustain_level),
                Segment::new(99, sustain_level),
            ],
            release_segments: [
                Segment::new(release_rate, 0), // Release: fade to silence
                Segment::new(99, 0),           // Immediate if needed
            ],
            rate_scaling: 0,
            delay_samples: 0,
            loop_config: LoopConfig::disabled(),
            sample_rate,
            cached_max_rate: max_rate_for_sample_rate(sample_rate),
        }
    }

    /// Create a typical FM piano envelope.
    pub fn piano(sample_rate: f32) -> Self {
        Self {
            key_on_segments: [
                Segment::new(99, 99), // Attack: instant to full
                Segment::new(70, 80), // Decay 1: fast to 80%
                Segment::new(50, 65), // Decay 2: medium to 65%
                Segment::new(35, 50), // Decay 3: slow to 50%
                Segment::new(25, 40), // Decay 4: slower to 40%
                Segment::new(20, 35), // Sustain: very slow decay
            ],
            release_segments: [
                Segment::new(50, 15), // Release 1: medium to 15%
                Segment::new(35, 0),  // Release 2: slow to silence
            ],
            rate_scaling: 3,
            delay_samples: 0,
            loop_config: LoopConfig::disabled(),
            sample_rate,
            cached_max_rate: max_rate_for_sample_rate(sample_rate),
        }
    }

    /// Create an organ-style sustaining envelope.
    pub fn organ(sample_rate: f32) -> Self {
        Self {
            key_on_segments: [
                Segment::new(99, 99), // Attack: instant
                Segment::new(99, 99), // Hold at max
                Segment::new(99, 99),
                Segment::new(99, 99),
                Segment::new(99, 99),
                Segment::new(99, 99), // Sustain: hold at max
            ],
            release_segments: [
                Segment::new(80, 0), // Fast release
                Segment::new(99, 0),
            ],
            rate_scaling: 0,
            delay_samples: 0,
            loop_config: LoopConfig::disabled(),
            sample_rate,
            cached_max_rate: max_rate_for_sample_rate(sample_rate),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loop_config_disabled() {
        let cfg = LoopConfig::disabled();
        assert!(!cfg.enabled);
        assert!(cfg.is_valid(6));
    }

    #[test]
    fn test_loop_config_valid() {
        let cfg = LoopConfig::new(2, 4).unwrap();
        assert!(cfg.enabled);
        assert_eq!(cfg.start_segment, 2);
        assert_eq!(cfg.end_segment, 4);
        assert!(cfg.is_valid(6)); // 6 key-on segments
    }

    #[test]
    fn test_loop_config_invalid_boundaries() {
        assert!(LoopConfig::new(4, 2).is_none()); // start >= end
        assert!(LoopConfig::new(2, 2).is_none()); // start == end
    }

    #[test]
    fn test_loop_config_out_of_range() {
        let cfg = LoopConfig::new(2, 10).unwrap();
        assert!(!cfg.is_valid(6)); // end_segment >= key_on_count
    }

    #[test]
    fn test_envelope_config_default() {
        let cfg = FmEnvelopeConfig::default_with_sample_rate(44100.0);
        assert_eq!(cfg.key_on_count(), 6);
        assert_eq!(cfg.release_count(), 2);
        assert_eq!(cfg.rate_scaling, 0);
        assert_eq!(cfg.delay_samples, 0);
        assert!(!cfg.loop_config.enabled);
    }

    #[test]
    fn test_envelope_config_rate_scaling_clamping() {
        let cfg = EnvelopeConfig::<6, 2>::new(
            [Segment::default(); 6],
            [Segment::default(); 2],
            15, // Should be clamped to 7
            0,
            LoopConfig::disabled(),
            44100.0,
        );
        assert_eq!(cfg.rate_scaling, 7);
    }

    #[test]
    fn test_delay_conversion() {
        let mut cfg = FmEnvelopeConfig::default_with_sample_rate(44100.0);
        cfg.set_delay_ms(500.0);
        assert_eq!(cfg.delay_samples, 22050);
        assert!((cfg.delay_ms() - 500.0).abs() < 0.1);
    }

    #[test]
    fn test_piano_preset() {
        let cfg = FmEnvelopeConfig::piano(44100.0);
        assert_eq!(cfg.key_on_segments[0].rate, 99);
        assert_eq!(cfg.rate_scaling, 3);
    }

    #[test]
    fn test_adsr_builder() {
        let cfg = FmEnvelopeConfig::adsr(0.01, 0.3, 0.7, 0.5, 44100.0);

        // Attack should be fast (high rate) for 10ms
        assert!(
            cfg.key_on_segments[0].rate > 60,
            "Attack rate should be high for 10ms"
        );
        assert_eq!(cfg.key_on_segments[0].level, 99); // Attack to full

        // Decay should target sustain level (0.7 -> ~69)
        let sustain_level = cfg.key_on_segments[1].level;
        assert!(
            (65..=73).contains(&sustain_level),
            "Sustain level should be ~69 for 0.7, got {}",
            sustain_level
        );

        // Remaining key-on segments should hold at sustain
        for seg in &cfg.key_on_segments[2..] {
            assert_eq!(seg.level, sustain_level);
            assert_eq!(seg.rate, 99); // Instant transition to same level
        }

        // Release should fade to 0
        assert_eq!(cfg.release_segments[0].level, 0);
        assert_eq!(cfg.release_segments[1].level, 0);

        // No advanced features
        assert_eq!(cfg.rate_scaling, 0);
        assert_eq!(cfg.delay_samples, 0);
        assert!(!cfg.loop_config.enabled);
    }

    #[test]
    fn test_adsr_fast_attack() {
        let cfg = FmEnvelopeConfig::adsr(0.001, 0.3, 0.7, 0.5, 44100.0);

        // 1ms attack should give very high rate (77-99)
        assert!(
            cfg.key_on_segments[0].rate >= 77,
            "1ms attack should give rate >= 77, got {}",
            cfg.key_on_segments[0].rate
        );
    }

    #[test]
    fn test_adsr_slow_release() {
        let cfg = FmEnvelopeConfig::adsr(0.01, 0.3, 0.7, 5.0, 44100.0);

        // 5s release should give low rate
        assert!(
            cfg.release_segments[0].rate < 30,
            "5s release should give rate < 30, got {}",
            cfg.release_segments[0].rate
        );
    }

    #[test]
    fn test_cached_max_rate_matches_computed() {
        // Verify cached max_rate matches what max_rate_for_sample_rate computes
        for &sr in &[22050.0, 44100.0, 48000.0, 96000.0] {
            let cfg = FmEnvelopeConfig::default_with_sample_rate(sr);
            let computed = max_rate_for_sample_rate(sr);
            assert_eq!(
                cfg.max_rate(),
                computed,
                "Cached max_rate should match computed for sample rate {}",
                sr
            );
        }
    }

    #[test]
    fn test_cached_max_rate_in_presets() {
        // Verify all preset constructors correctly cache max_rate
        let sr = 44100.0;
        let expected = max_rate_for_sample_rate(sr);

        let default_cfg = FmEnvelopeConfig::default_with_sample_rate(sr);
        assert_eq!(default_cfg.max_rate(), expected);

        let adsr_cfg = FmEnvelopeConfig::adsr(0.1, 0.2, 0.7, 0.5, sr);
        assert_eq!(adsr_cfg.max_rate(), expected);

        let piano_cfg = FmEnvelopeConfig::piano(sr);
        assert_eq!(piano_cfg.max_rate(), expected);

        let organ_cfg = FmEnvelopeConfig::organ(sr);
        assert_eq!(organ_cfg.max_rate(), expected);
    }
}
