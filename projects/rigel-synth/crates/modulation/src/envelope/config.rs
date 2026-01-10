//! Envelope configuration types.
//!
//! This module defines:
//! - [`LoopConfig`] - Loop configuration for key-on segments
//! - [`EnvelopeConfig`] - Immutable envelope configuration

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

    /// Output level scaling (pre-computed from operator level).
    /// In internal units (~0.75dB per step).
    pub output_level: u8,

    /// Delay before attack begins (in samples).
    pub delay_samples: u32,

    /// Loop configuration for key-on segments.
    pub loop_config: LoopConfig,

    /// Sample rate for timing calculations.
    pub sample_rate: f32,
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
            output_level: 127, // Full output
            delay_samples: 0,
            loop_config: LoopConfig::disabled(),
            sample_rate,
        }
    }

    /// Create configuration from raw parameters.
    ///
    /// # Arguments
    ///
    /// * `key_on_segments` - Key-on segment configuration
    /// * `release_segments` - Release segment configuration
    /// * `rate_scaling` - Rate scaling sensitivity (0-7)
    /// * `output_level` - Output level (0-127)
    /// * `delay_samples` - Delay before attack (in samples)
    /// * `loop_config` - Loop configuration
    /// * `sample_rate` - Sample rate in Hz
    pub const fn new(
        key_on_segments: [Segment; K],
        release_segments: [Segment; R],
        rate_scaling: u8,
        output_level: u8,
        delay_samples: u32,
        loop_config: LoopConfig,
        sample_rate: f32,
    ) -> Self {
        Self {
            key_on_segments,
            release_segments,
            rate_scaling: if rate_scaling > 7 { 7 } else { rate_scaling },
            output_level,
            delay_samples,
            loop_config,
            sample_rate,
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
}

/// Type alias for 8-segment FM envelope configuration (6 key-on + 2 release).
pub type FmEnvelopeConfig = EnvelopeConfig<6, 2>;

/// Type alias for 10-segment AWM envelope configuration (5 key-on + 5 release).
pub type AwmEnvelopeConfig = EnvelopeConfig<5, 5>;

/// Type alias for 7-segment envelope configuration (5 key-on + 2 release).
pub type SevenSegEnvelopeConfig = EnvelopeConfig<5, 2>;

impl FmEnvelopeConfig {
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
            output_level: 127,
            delay_samples: 0,
            loop_config: LoopConfig::disabled(),
            sample_rate,
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
            output_level: 127,
            delay_samples: 0,
            loop_config: LoopConfig::disabled(),
            sample_rate,
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
            127,
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
}
