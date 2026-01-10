//! MSFA-compatible rate calculations and lookup tables.
//!
//! This module implements the DX7/MSFA envelope rate system:
//! - Rate (0-99) to qRate (0-63) conversion
//! - Rate scaling by MIDI note position
//! - Increment calculation for per-sample processing
//! - Level conversion (Q8 to linear amplitude)
//! - Output level scaling with LEVEL_LUT

// Use rigel_math's exp2 LUT for level-to-linear conversion
use rigel_math::exp2_lut::{exp2_lut, exp2_lut_slice};

/// Attack jump threshold (Q8 format, ~40dB above minimum).
///
/// The envelope immediately jumps to this level at attack start
/// to provide the characteristic FM "punch".
pub const JUMP_TARGET_Q8: i16 = 1716;

/// Level lookup table for output levels 0-19.
///
/// Creates a non-linear curve with faster falloff at low levels,
/// matching original DX7 behavior. Without this, level 0 would
/// only be -74.5dB instead of -95.5dB.
pub const LEVEL_LUT: [u8; 20] = [
    0, 5, 9, 13, 17, 20, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 42, 43, 45, 46,
];

/// Static timing table for same-level transitions (44100Hz base).
///
/// When target equals current level, the standard increment calculation
/// doesn't apply. MSFA uses this lookup table for these cases.
/// Values are sample counts at 44100Hz.
pub const STATICS: [u32; 77] = [
    1764000, 1764000, 1411200, 1411200, 1190700, 1014300, 992250, 882000, 705600, 705600, 584325,
    507150, 502740, 441000, 418950, 352800, 308700, 286650, 253575, 220500, 220500, 176400, 145530,
    145530, 125685, 110250, 110250, 88200, 88200, 74970, 61740, 61740, 55125, 48510, 44100, 37485,
    31311, 30870, 27562, 27562, 22050, 18522, 17640, 15435, 14112, 13230, 11025, 9261, 9261, 7717,
    6615, 6615, 5512, 5512, 4410, 3969, 3969, 3439, 2866, 2690, 2249, 1984, 1896, 1808, 1411, 1367,
    1234, 1146, 926, 837, 837, 705, 573, 573, 529, 441, 441,
];

/// Convert DX7 rate (0-99) to internal qRate (0-63).
///
/// Uses MSFA formula: `qrate = (rate * 41) >> 6`
///
/// # Arguments
///
/// * `rate` - DX7 rate parameter (0-99)
///
/// # Returns
///
/// Internal quantized rate (0-63)
#[inline]
pub fn rate_to_qrate(rate: u8) -> u8 {
    ((rate as u32 * 41) >> 6) as u8
}

/// Scale output level using MSFA lookup table.
///
/// Levels 0-19 use the non-linear LEVEL_LUT, levels 20-99 use
/// linear formula (28 + level).
///
/// # Arguments
///
/// * `outlevel` - Output level (0-99)
///
/// # Returns
///
/// Scaled level in internal units (~0.75dB per step)
#[inline]
pub fn scale_output_level(outlevel: u8) -> u8 {
    if outlevel >= 20 {
        28 + outlevel
    } else {
        LEVEL_LUT[outlevel as usize]
    }
}

/// Calculate rate scaling adjustment for MIDI note.
///
/// Produces musically appropriate rate scaling where higher notes
/// have shorter envelopes, mimicking acoustic instrument behavior.
///
/// # Arguments
///
/// * `midi_note` - MIDI note number (0-127)
/// * `sensitivity` - Rate scaling sensitivity (0-7)
///
/// # Returns
///
/// qRate delta to add to base qRate (0-31 typical range)
///
/// # Example
///
/// ```ignore
/// // At sensitivity 7:
/// // C1 (MIDI 24) -> +1 qRate
/// // C3 (MIDI 48) -> +9 qRate
/// // C5 (MIDI 72) -> +17 qRate
/// // C7 (MIDI 96) -> +25 qRate
/// ```
#[inline]
pub fn scale_rate(midi_note: u8, sensitivity: u8) -> u8 {
    // Divide keyboard into groups of 3 notes, offset by 7 groups
    // Centers scaling around MIDI note 21 (A0)
    let x = ((midi_note as i32 / 3) - 7).clamp(0, 31) as u8;

    // Apply sensitivity scaling
    ((sensitivity as u32 * x as u32) >> 3) as u8
}

/// Calculate increment from qRate in Q8 format.
///
/// Scaled for per-sample processing with i16 level representation.
/// The increment determines how fast the envelope moves per sample.
///
/// # Arguments
///
/// * `qrate` - Internal quantized rate (0-63)
///
/// # Returns
///
/// Level increment per sample (Q8 format)
#[inline]
pub fn calculate_increment_q8(qrate: u8) -> i16 {
    // The qrate maps to how fast the envelope moves.
    // Higher qrate = larger increment per sample.
    //
    // We use a simple exponential mapping:
    // Each increment in qrate doubles the speed every 4 units.
    //
    // At 44.1kHz with Q8 levels (4096 range):
    // - qrate 0: very slow (~1 per 1000 samples)
    // - qrate 32: moderate
    // - qrate 63: very fast (~full range in <100 samples)

    if qrate == 0 {
        return 0;
    }

    // Use bit shifting to create exponential curve
    // integer_part = qrate >> 2 (0-15)
    // fractional_part = qrate & 3 (0-3)
    // increment = (4 + fractional) << (integer - 2)
    //
    // For small qrates, ensure minimum increment of 1
    let int_part = (qrate >> 2) as i32;
    let frac_part = (qrate & 3) as i32;

    let base = 4 + frac_part;

    // Shift amount: int_part - 2, clamped to valid range
    let shift = (int_part - 2).max(0);

    if shift > 0 {
        (base << shift).min(4096) as i16
    } else {
        // For low qrates, use fractional approach
        (base >> (2 - int_part).max(0)).max(1) as i16
    }
}

/// Convert Q8 level (0-4095) to linear amplitude (0.0 to 1.0).
///
/// Uses pre-computed lookup table from rigel_math for maximum performance.
/// Q8 format: 256 steps = 6dB (one octave)
///
/// Formula: `linear = 2^((level - LEVEL_MAX) / 256)`
///
/// # Arguments
///
/// * `level_q8` - Level in Q8 format (0-4095)
///
/// # Returns
///
/// Linear amplitude in range [0.0, 1.0]
#[inline]
pub fn level_to_linear(level_q8: i16) -> f32 {
    // Delegate to rigel_math's exp2 LUT
    exp2_lut(level_q8)
}

/// Convert multiple Q8 levels to linear amplitudes using LUT.
///
/// Uses pre-computed lookup table from rigel_math for O(1) conversion per level.
/// This is the fastest method for batch conversion.
///
/// # Arguments
///
/// * `levels` - Slice of Q8 levels (0-4095)
/// * `output` - Output slice for linear amplitudes (must be same length)
///
/// # Panics
///
/// Panics if `levels` and `output` have different lengths.
#[inline]
pub fn levels_to_linear_simd(levels: &[i16], output: &mut [f32]) {
    // Delegate to rigel_math's batch exp2 LUT
    exp2_lut_slice(levels, output);
}

/// Convert linear amplitude (0.0 to 1.0) to Q8 level.
///
/// Inverse of `level_to_linear`.
///
/// # Arguments
///
/// * `linear` - Linear amplitude (0.0 to 1.0)
///
/// # Returns
///
/// Level in Q8 format (0-4095)
#[inline]
pub fn linear_to_level(linear: f32) -> i16 {
    // level = log2(linear) * 256 + LEVEL_MAX
    // Handle zero/negative to prevent log of zero
    if linear <= 0.0 {
        return super::state::LEVEL_MIN;
    }

    let log2_val = libm::log2f(linear);
    let level = log2_val * 256.0 + super::state::LEVEL_MAX as f32;

    // Clamp to valid range
    level.clamp(
        super::state::LEVEL_MIN as f32,
        super::state::LEVEL_MAX as f32,
    ) as i16
}

/// Get static timing count for same-level transitions.
///
/// For rates >= 77: samples = 20 * (99 - rate)
///
/// # Arguments
///
/// * `rate` - DX7 rate parameter (0-99)
/// * `sample_rate` - Target sample rate in Hz
///
/// # Returns
///
/// Sample count for the transition
#[inline]
pub fn get_static_count(rate: u8, sample_rate: f32) -> u32 {
    let base_count = if rate < 77 {
        STATICS[rate as usize]
    } else {
        20 * (99 - rate as u32)
    };

    // Scale for sample rate (table is for 44100Hz)
    (base_count as f32 * sample_rate / 44100.0) as u32
}

/// Convert time in seconds to DX7 rate (0-99).
///
/// Inverts the STATICS table to find the rate that produces
/// approximately the target duration. Shorter time = higher rate.
///
/// # Arguments
///
/// * `time_seconds` - Target envelope segment time in seconds
/// * `sample_rate` - Sample rate in Hz
///
/// # Returns
///
/// DX7 rate parameter (0-99) that approximates the target time
///
/// # Example
///
/// ```ignore
/// // 10ms attack at 44.1kHz
/// let rate = seconds_to_rate(0.01, 44100.0);
/// assert!(rate > 70); // Fast rate for short time
///
/// // 2 second decay at 44.1kHz
/// let rate = seconds_to_rate(2.0, 44100.0);
/// assert!(rate < 30); // Slow rate for long time
/// ```
#[inline]
pub fn seconds_to_rate(time_seconds: f32, sample_rate: f32) -> u8 {
    // Handle edge cases
    if time_seconds <= 0.0 {
        return 99; // Instant
    }

    let target_samples = (time_seconds * sample_rate) as u32;

    // For very short times (rates 77-99), use formula: samples = 20 * (99 - rate)
    // Solving for rate: rate = 99 - samples / 20
    if target_samples < 440 {
        // 440 = 20 * (99 - 77), threshold where formula begins
        let rate = 99u32.saturating_sub(target_samples / 20);
        return rate.clamp(77, 99) as u8;
    }

    // Scale target to 44100Hz base (STATICS table is for 44100Hz)
    let target_at_44100 = (target_samples as f32 * 44100.0 / sample_rate) as u32;

    // Binary search the STATICS table (sorted in descending order)
    // Lower rate = longer time, higher rate = shorter time
    let mut low = 0usize;
    let mut high = 76usize; // STATICS has 77 entries (0-76)

    while low < high {
        let mid = (low + high) / 2;
        if STATICS[mid] > target_at_44100 {
            low = mid + 1;
        } else {
            high = mid;
        }
    }

    // Clamp to valid range
    low.min(76) as u8
}

/// Convert linear amplitude (0.0-1.0) to DX7 level parameter (0-99).
///
/// This is useful for converting user-friendly sustain levels
/// to the 0-99 range expected by envelope segments.
///
/// # Arguments
///
/// * `linear` - Linear amplitude in range [0.0, 1.0]
///
/// # Returns
///
/// DX7 level parameter (0-99)
#[inline]
pub fn linear_to_param_level(linear: f32) -> u8 {
    // Simple linear mapping with clamping
    // 0.0 -> 0, 1.0 -> 99
    libm::roundf(linear.clamp(0.0, 1.0) * 99.0) as u8
}

/// Convert DX7 level parameter (0-99) to Q8 internal level.
///
/// # Arguments
///
/// * `param_level` - DX7 level parameter (0-99)
///
/// # Returns
///
/// Internal Q8 level (0-4095)
#[inline]
pub fn param_to_q8_level(param_level: u8) -> i16 {
    // Scale 0-99 to roughly 0-4095
    // Using similar curve to output level scaling
    let scaled = scale_output_level(param_level) as i32;

    // The scaled value is in ~0.75dB units, convert to Q8
    // Q8 has 256 steps per 6dB, so ~42.67 steps per dB
    // 0.75dB * 42.67 ≈ 32 Q8 steps per scaled unit
    (scaled * 32).clamp(0, super::state::LEVEL_MAX as i32) as i16
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::envelope::state::{LEVEL_MAX, LEVEL_MIN};

    #[test]
    fn test_rate_to_qrate() {
        assert_eq!(rate_to_qrate(0), 0);
        assert_eq!(rate_to_qrate(50), 32);
        assert_eq!(rate_to_qrate(99), 63);
    }

    #[test]
    fn test_scale_output_level() {
        // Low levels use lookup table
        assert_eq!(scale_output_level(0), 0);
        assert_eq!(scale_output_level(10), 31);
        assert_eq!(scale_output_level(19), 46);

        // High levels use linear formula
        assert_eq!(scale_output_level(20), 48);
        assert_eq!(scale_output_level(50), 78);
        assert_eq!(scale_output_level(99), 127);
    }

    #[test]
    fn test_scale_rate() {
        // At sensitivity 0, no scaling
        assert_eq!(scale_rate(60, 0), 0);
        assert_eq!(scale_rate(96, 0), 0);

        // At sensitivity 7, maximum scaling
        let rate_c1 = scale_rate(24, 7);
        let rate_c4 = scale_rate(60, 7);
        let rate_c7 = scale_rate(96, 7);

        // Higher notes should have higher rate adjustment
        assert!(rate_c4 > rate_c1);
        assert!(rate_c7 > rate_c4);
    }

    #[test]
    fn test_level_to_linear() {
        // Maximum level should give ~1.0
        let max_linear = level_to_linear(LEVEL_MAX);
        assert!(
            (max_linear - 1.0).abs() < 0.01,
            "LEVEL_MAX should give ~1.0, got {}",
            max_linear
        );

        // Minimum level should give very small value
        let min_linear = level_to_linear(LEVEL_MIN);
        assert!(
            min_linear < 0.001,
            "LEVEL_MIN should give very small value, got {}",
            min_linear
        );

        // Middle value should give reasonable result
        let mid_linear = level_to_linear(LEVEL_MAX / 2);
        assert!(mid_linear > 0.0 && mid_linear < 1.0);
    }

    #[test]
    fn test_linear_to_level() {
        // 1.0 should give LEVEL_MAX
        let level_1 = linear_to_level(1.0);
        assert!(
            (level_1 - LEVEL_MAX).abs() < 10,
            "1.0 should give ~LEVEL_MAX, got {}",
            level_1
        );

        // Very small value should give near LEVEL_MIN
        let level_small = linear_to_level(0.00001);
        assert!(level_small < LEVEL_MAX / 4);

        // 0.0 should give LEVEL_MIN
        let level_0 = linear_to_level(0.0);
        assert_eq!(level_0, LEVEL_MIN);
    }

    #[test]
    fn test_roundtrip_level_conversion() {
        // Test roundtrip conversion for various levels
        for level in [0, 1000, 2000, 3000, LEVEL_MAX].iter() {
            let linear = level_to_linear(*level);
            let back = linear_to_level(linear);

            // Allow some tolerance due to precision
            let diff = (back - *level).abs();
            assert!(
                diff < 50,
                "Roundtrip failed for level {}: got back {}, diff {}",
                level,
                back,
                diff
            );
        }
    }

    #[test]
    fn test_calculate_increment_q8() {
        // Higher qrate should give higher increment
        let inc_low = calculate_increment_q8(10);
        let inc_high = calculate_increment_q8(50);
        assert!(
            inc_high > inc_low,
            "Higher qrate should give higher increment"
        );

        // Maximum rate should give substantial increment
        let inc_max = calculate_increment_q8(63);
        assert!(inc_max > 0);
    }

    #[test]
    fn test_get_static_count() {
        // Rate 0 should give longest time
        let count_0 = get_static_count(0, 44100.0);
        assert_eq!(count_0, STATICS[0]);

        // Rate 99 should give shortest time
        let count_99 = get_static_count(99, 44100.0);
        assert_eq!(count_99, 0); // 20 * (99 - 99) = 0

        // Sample rate scaling
        let count_48k = get_static_count(50, 48000.0);
        let count_44k = get_static_count(50, 44100.0);
        assert!(count_48k > count_44k);
    }

    #[test]
    fn test_param_to_q8_level() {
        // Level 0 should map to low Q8
        let q8_0 = param_to_q8_level(0);
        assert_eq!(q8_0, 0);

        // Level 99 should map to high Q8
        let q8_99 = param_to_q8_level(99);
        assert!(q8_99 > LEVEL_MAX / 2);
    }

    #[test]
    fn test_seconds_to_rate_instant() {
        // Zero or negative time should give instant (rate 99)
        assert_eq!(seconds_to_rate(0.0, 44100.0), 99);
        assert_eq!(seconds_to_rate(-1.0, 44100.0), 99);
    }

    #[test]
    fn test_seconds_to_rate_very_short() {
        // Very short times should give high rates (77-99)
        let rate = seconds_to_rate(0.001, 44100.0); // 1ms
        assert!(rate >= 77, "1ms should give rate >= 77, got {}", rate);
    }

    #[test]
    fn test_seconds_to_rate_short() {
        // 10ms attack should give high rate
        let rate = seconds_to_rate(0.01, 44100.0);
        assert!(rate > 60, "10ms should give rate > 60, got {}", rate);
    }

    #[test]
    fn test_seconds_to_rate_medium() {
        // 500ms should give medium rate
        let rate = seconds_to_rate(0.5, 44100.0);
        assert!(
            rate > 20 && rate < 60,
            "500ms should give rate 20-60, got {}",
            rate
        );
    }

    #[test]
    fn test_seconds_to_rate_long() {
        // 5 seconds should give low rate
        let rate = seconds_to_rate(5.0, 44100.0);
        assert!(rate < 30, "5s should give rate < 30, got {}", rate);
    }

    #[test]
    fn test_seconds_to_rate_very_long() {
        // 40 seconds (max) should give rate 0
        let rate = seconds_to_rate(40.0, 44100.0);
        assert_eq!(rate, 0, "40s should give rate 0");
    }

    #[test]
    fn test_seconds_to_rate_monotonic() {
        // Longer times should give lower rates (slower envelopes)
        let rate_short = seconds_to_rate(0.1, 44100.0);
        let rate_medium = seconds_to_rate(1.0, 44100.0);
        let rate_long = seconds_to_rate(5.0, 44100.0);

        assert!(
            rate_short > rate_medium,
            "Shorter time should give higher rate: {} vs {}",
            rate_short,
            rate_medium
        );
        assert!(
            rate_medium > rate_long,
            "Medium time should give higher rate than long: {} vs {}",
            rate_medium,
            rate_long
        );
    }

    #[test]
    fn test_seconds_to_rate_sample_rate_scaling() {
        // Same time at different sample rates should give similar results
        // (the rate is relative to envelope behavior, not sample count)
        let rate_44100 = seconds_to_rate(0.5, 44100.0);
        let rate_48000 = seconds_to_rate(0.5, 48000.0);

        // Should be within a few units of each other
        let diff = (rate_44100 as i32 - rate_48000 as i32).abs();
        assert!(
            diff <= 5,
            "Rates at different sample rates should be similar: {} vs {}, diff {}",
            rate_44100,
            rate_48000,
            diff
        );
    }

    #[test]
    fn test_linear_to_param_level() {
        // 0.0 -> 0
        assert_eq!(linear_to_param_level(0.0), 0);

        // 1.0 -> 99
        assert_eq!(linear_to_param_level(1.0), 99);

        // 0.5 -> ~50
        let mid = linear_to_param_level(0.5);
        assert!((49..=50).contains(&mid), "0.5 should give ~50, got {}", mid);

        // Clamping
        assert_eq!(linear_to_param_level(-0.5), 0);
        assert_eq!(linear_to_param_level(1.5), 99);
    }
}
