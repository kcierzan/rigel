//! Envelope segment definition.
//!
//! A segment represents one leg of the envelope's journey,
//! with a target level and a rate to reach it.

/// A single envelope segment.
///
/// Each segment defines a target level and the rate at which
/// to approach it. This matches the DX7/SY99 convention.
///
/// # Rate
///
/// - 0 = slowest (~10 minutes for full range)
/// - 50 = medium (~750ms for 96dB)
/// - 99 = fastest (~6ms for full range)
///
/// # Level
///
/// - 0 = silence (~-96dB)
/// - 50 = ~37dB below full scale
/// - 99 = full scale (0dB)
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Segment {
    /// Rate parameter (0-99 DX7 convention).
    /// Higher values = faster transition.
    pub rate: u8,

    /// Target level parameter (0-99 DX7 convention).
    /// 99 = 0dB (full), 0 = ~-96dB (silent).
    pub level: u8,
}

impl Segment {
    /// Maximum rate (near-instantaneous).
    pub const MAX_RATE: u8 = 99;

    /// Maximum level (0dB).
    pub const MAX_LEVEL: u8 = 99;

    /// Minimum rate (very slow).
    pub const MIN_RATE: u8 = 0;

    /// Minimum level (silent).
    pub const MIN_LEVEL: u8 = 0;

    /// Create a new segment with specified rate and level.
    ///
    /// Values are clamped to valid range (0-99).
    ///
    /// # Arguments
    ///
    /// * `rate` - Rate parameter (0-99)
    /// * `level` - Target level parameter (0-99)
    ///
    /// # Example
    ///
    /// ```ignore
    /// use rigel_modulation::envelope::Segment;
    ///
    /// // Fast attack to full level
    /// let attack = Segment::new(99, 99);
    ///
    /// // Medium decay to half level
    /// let decay = Segment::new(50, 50);
    ///
    /// // Slow release to silence
    /// let release = Segment::new(30, 0);
    /// ```
    #[inline]
    pub const fn new(rate: u8, level: u8) -> Self {
        Self {
            rate: if rate > 99 { 99 } else { rate },
            level: if level > 99 { 99 } else { level },
        }
    }

    /// Create a segment for instant attack (rate=99, level=99).
    #[inline]
    pub const fn instant_attack() -> Self {
        Self::new(99, 99)
    }

    /// Create a segment for instant silence (rate=99, level=0).
    #[inline]
    pub const fn instant_silence() -> Self {
        Self::new(99, 0)
    }

    /// Create a segment that holds at current level.
    ///
    /// Uses maximum rate to minimize transition time.
    #[inline]
    pub const fn hold(level: u8) -> Self {
        Self::new(99, level)
    }

    /// Check if this segment has maximum rate.
    #[inline]
    pub const fn is_instant(&self) -> bool {
        self.rate >= 95 // Near-instant threshold
    }

    /// Check if this segment targets silence.
    #[inline]
    pub const fn is_silent(&self) -> bool {
        self.level == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_segment_creation() {
        let seg = Segment::new(50, 75);
        assert_eq!(seg.rate, 50);
        assert_eq!(seg.level, 75);
    }

    #[test]
    fn test_segment_clamping() {
        let seg = Segment::new(150, 200);
        assert_eq!(seg.rate, 99);
        assert_eq!(seg.level, 99);
    }

    #[test]
    fn test_instant_attack() {
        let seg = Segment::instant_attack();
        assert_eq!(seg.rate, 99);
        assert_eq!(seg.level, 99);
        assert!(seg.is_instant());
    }

    #[test]
    fn test_instant_silence() {
        let seg = Segment::instant_silence();
        assert_eq!(seg.rate, 99);
        assert_eq!(seg.level, 0);
        assert!(seg.is_instant());
        assert!(seg.is_silent());
    }

    #[test]
    fn test_default() {
        let seg = Segment::default();
        assert_eq!(seg.rate, 0);
        assert_eq!(seg.level, 0);
    }
}
