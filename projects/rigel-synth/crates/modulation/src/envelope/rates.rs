//! MSFA-compatible rate calculations and lookup tables.
//!
//! This module implements the DX7/MSFA envelope rate system:
//! - Rate (0-99) to sample count conversion via STATICS table
//! - Rate scaling by MIDI note position
//! - Increment calculation for per-sample processing (f32)
//! - Level conversion functions
//!
//! ## Attribution
//!
//! Rate tables, timing constants, and conversion formulas in this module are
//! derived from the Music Synthesizer for Android (MSFA) project:
//!
//! Copyright 2012 Google Inc.
//! Licensed under the Apache License, Version 2.0
//! <https://github.com/google/music-synthesizer-for-android>
//!
//! See THIRD_PARTY_LICENSES.md in the repository root for the full license text.

/// Attack jump threshold in linear amplitude (0.0-1.0).
///
/// The DX7 envelope immediately jumps to 1716 in Q8 format at attack start,
/// which is ~40dB above minimum (~-56dB from full scale). This gets the
/// envelope out of the sub-perceptual range quickly while preserving the
/// exponential approach character for the rest of the attack.
///
/// Calculation: 1716 Q8 steps = 40.2dB above minimum = -55.8dB from 0dB
/// Linear amplitude: 10^(-55.8/20) ≈ 0.00163 (0.16%)
///
/// Reference: MSFA/Dexed `const int jumptarget = 1716`
// TODO: this seems pretty high for long envelopes. Tweak this by ear later.
pub const JUMP_TARGET: f32 = 0.00163;

/// Minimum envelope segment transition time in seconds.
///
/// This prevents rate scaling from producing instant (clicking) transitions
/// when high notes are combined with fast base rates and high sensitivity.
/// 1.5ms is within the safe range to avoid audible clicks (1-2ms recommended).
///
/// This is enforced as a time value rather than a fixed rate to ensure
/// consistent behavior across all sample rates.
pub const MIN_SEGMENT_TIME_SECONDS: f32 = 0.0015; // 1.5ms

/// Static timing table for DX7 rates (44100Hz base).
///
/// Maps DX7 rate parameter (0-76) to sample counts at 44100Hz.
/// Lower rate = longer time (more samples).
/// Rates 77-99 use formula: samples = 20 * (99 - rate).
pub const STATICS: [u32; 77] = [
    1764000, 1764000, 1411200, 1411200, 1190700, 1014300, 992250, 882000, 705600, 705600, 584325,
    507150, 502740, 441000, 418950, 352800, 308700, 286650, 253575, 220500, 220500, 176400, 145530,
    145530, 125685, 110250, 110250, 88200, 88200, 74970, 61740, 61740, 55125, 48510, 44100, 37485,
    31311, 30870, 27562, 27562, 22050, 18522, 17640, 15435, 14112, 13230, 11025, 9261, 9261, 7717,
    6615, 6615, 5512, 5512, 4410, 3969, 3969, 3439, 2866, 2690, 2249, 1984, 1896, 1808, 1411, 1367,
    1234, 1146, 926, 837, 837, 705, 573, 573, 529, 441, 441,
];

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
/// Rate delta to add to base rate (0-31 typical range)
///
/// # Example
///
/// ```ignore
/// // At sensitivity 7:
/// // C1 (MIDI 24) -> small adjustment
/// // C4 (MIDI 60) -> moderate adjustment
/// // C7 (MIDI 96) -> large adjustment (faster envelope)
/// ```
#[inline]
pub fn scale_rate(midi_note: u8, sensitivity: u8) -> u8 {
    // Divide keyboard into groups of 3 notes, offset by 7 groups
    // Centers scaling around MIDI note 21 (A0)
    let x = ((midi_note as i32 / 3) - 7).clamp(0, 31) as u8;

    // Apply sensitivity scaling
    ((sensitivity as u32 * x as u32) >> 3) as u8
}

/// Calculate per-sample increment from rate using STATICS table.
///
/// Uses the STATICS timing table to determine the correct increment
/// for accurate envelope timing.
///
/// # Arguments
///
/// * `rate` - DX7 rate parameter (0-99)
/// * `sample_rate` - Sample rate in Hz
///
/// # Returns
///
/// Per-sample increment (f32) to traverse full 0.0-1.0 range
///
/// # Example
///
/// ```ignore
/// // 300ms release at 44.1kHz
/// let rate = seconds_to_rate(0.3, 44100.0); // ~rate 40
/// let inc = calculate_increment_f32(rate, 44100.0);
/// // inc ≈ 1.0 / 13230 ≈ 0.0000756
/// // Full transition takes 13230 samples = 300ms
/// ```
#[inline]
pub fn calculate_increment_f32(rate: u8, sample_rate: f32) -> f32 {
    let samples = get_static_count(rate, sample_rate);
    if samples == 0 {
        return 1.0; // Instant transition
    }
    1.0 / samples as f32
}

/// Calculate per-sample increment with rate scaling applied.
///
/// Applies MIDI note-based rate scaling before calculating increment.
/// Higher notes produce faster envelopes when rate_scaling > 0.
///
/// The scaled rate is clamped to ensure a minimum transition time
/// (see [`MIN_SEGMENT_TIME_SECONDS`]) to prevent audible clicks when
/// rate scaling pushes envelope rates toward instant transitions.
///
/// # Arguments
///
/// * `rate` - Base DX7 rate parameter (0-99)
/// * `midi_note` - MIDI note number (0-127)
/// * `rate_scaling` - Rate scaling sensitivity (0-7)
/// * `sample_rate` - Sample rate in Hz
///
/// # Returns
///
/// Per-sample increment (f32) with rate scaling applied
#[inline]
pub fn calculate_increment_f32_scaled(
    rate: u8,
    midi_note: u8,
    rate_scaling: u8,
    sample_rate: f32,
) -> f32 {
    // Apply rate scaling: higher notes get faster rates
    let rate_adjustment = scale_rate(midi_note, rate_scaling);

    // Clamp to max rate that ensures minimum segment time (sample-rate independent)
    let max_rate = max_rate_for_sample_rate(sample_rate);
    let scaled_rate = (rate as u16 + rate_adjustment as u16).min(max_rate as u16) as u8;

    calculate_increment_f32(scaled_rate, sample_rate)
}

/// Calculate per-sample increment with rate scaling and pre-computed max rate.
///
/// Same as [`calculate_increment_f32_scaled`] but takes a pre-computed `max_rate`
/// parameter to avoid the O(100) search in [`max_rate_for_sample_rate`].
/// Use this in hot paths where the max rate has been cached at configuration time.
///
/// # Arguments
///
/// * `rate` - Base DX7 rate parameter (0-99)
/// * `midi_note` - MIDI note number (0-127)
/// * `rate_scaling` - Rate scaling sensitivity (0-7)
/// * `sample_rate` - Sample rate in Hz
/// * `max_rate` - Pre-computed maximum rate from [`max_rate_for_sample_rate`]
///
/// # Returns
///
/// Per-sample increment (f32) with rate scaling applied
#[inline]
pub fn calculate_increment_f32_with_max_rate(
    rate: u8,
    midi_note: u8,
    rate_scaling: u8,
    sample_rate: f32,
    max_rate: u8,
) -> f32 {
    // Apply rate scaling: higher notes get faster rates
    let rate_adjustment = scale_rate(midi_note, rate_scaling);

    // Clamp to provided max rate (pre-computed at config time)
    let scaled_rate = (rate as u16 + rate_adjustment as u16).min(max_rate as u16) as u8;

    calculate_increment_f32(scaled_rate, sample_rate)
}

/// Get the maximum rate that ensures minimum segment time.
///
/// Returns the highest rate that still provides at least [`MIN_SEGMENT_TIME_SECONDS`]
/// transition time at the given sample rate. This is used to prevent rate scaling
/// from producing instant (clicking) transitions.
///
/// Note: This searches from high to low rates to find the first rate that meets
/// the minimum time requirement, ensuring we never return a rate that's too fast.
///
/// # Arguments
///
/// * `sample_rate` - Sample rate in Hz
///
/// # Returns
///
/// Maximum rate parameter (0-99) that provides minimum transition time
#[inline]
pub fn max_rate_for_sample_rate(sample_rate: f32) -> u8 {
    let min_samples = (MIN_SEGMENT_TIME_SECONDS * sample_rate) as u32;

    // Search from fastest to slowest, return first rate that meets minimum time
    for rate in (0..=99).rev() {
        if get_static_count(rate, sample_rate) >= min_samples {
            return rate;
        }
    }
    0 // Fallback to slowest rate (should never happen with reasonable sample rates)
}

/// Get static timing count for a rate.
///
/// For rates 0-76: uses STATICS table lookup.
/// For rates 77-99: samples = 20 * (99 - rate).
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

/// Convert DX7 level parameter (0-99) to linear amplitude (0.0-1.0).
///
/// Simple linear mapping for envelope levels.
///
/// # Arguments
///
/// * `param_level` - DX7 level parameter (0-99)
///
/// # Returns
///
/// Linear amplitude in range [0.0, 1.0]
#[inline]
pub fn param_to_level_f32(param_level: u8) -> f32 {
    param_level as f32 / 99.0
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_calculate_increment_f32() {
        // Rate 99 should give instant transition
        let inc_99 = calculate_increment_f32(99, 44100.0);
        assert_eq!(inc_99, 1.0, "Rate 99 should be instant");

        // Rate 0 should give very slow transition
        let inc_0 = calculate_increment_f32(0, 44100.0);
        let expected_samples = STATICS[0] as f32;
        let expected_inc = 1.0 / expected_samples;
        assert!(
            (inc_0 - expected_inc).abs() < 1e-10,
            "Rate 0 increment mismatch: {} vs {}",
            inc_0,
            expected_inc
        );

        // Higher rate = higher increment (faster envelope)
        let inc_low = calculate_increment_f32(20, 44100.0);
        let inc_high = calculate_increment_f32(60, 44100.0);
        assert!(
            inc_high > inc_low,
            "Higher rate should give higher increment: {} vs {}",
            inc_high,
            inc_low
        );
    }

    #[test]
    fn test_calculate_increment_f32_timing() {
        // Test that increment produces correct timing
        let sample_rate = 44100.0;

        // 300ms release should take ~13230 samples
        let rate = seconds_to_rate(0.3, sample_rate);
        let inc = calculate_increment_f32(rate, sample_rate);

        // Simulate envelope from 1.0 to 0.0
        let mut level = 1.0f32;
        let mut samples = 0u32;
        while level > 0.0 && samples < 100000 {
            level -= inc;
            samples += 1;
        }

        let actual_time = samples as f32 / sample_rate;
        let tolerance = 0.3; // 30% tolerance

        assert!(
            (actual_time - 0.3).abs() < 0.3 * tolerance,
            "300ms timing mismatch: expected ~300ms, got {}ms",
            actual_time * 1000.0
        );
    }

    #[test]
    fn test_param_to_level_f32() {
        // Level 0 -> 0.0
        assert!((param_to_level_f32(0) - 0.0).abs() < f32::EPSILON);

        // Level 99 -> 1.0
        assert!((param_to_level_f32(99) - 1.0).abs() < f32::EPSILON);

        // Level 50 -> ~0.505
        let mid = param_to_level_f32(50);
        assert!(
            (mid - 0.505).abs() < 0.01,
            "Level 50 should give ~0.505, got {}",
            mid
        );
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

    #[test]
    fn test_jump_target_constant() {
        // JUMP_TARGET should be approximately -56dB from full scale
        // (40dB above minimum in DX7 terms, where minimum is ~-96dB)
        // Linear amplitude: 10^(-55.8/20) ≈ 0.00163
        assert!(
            (JUMP_TARGET - 0.00163).abs() < 0.0001,
            "JUMP_TARGET should be ~0.00163, got {}",
            JUMP_TARGET
        );

        // Verify dB level
        let db = 20.0 * libm::log10f(JUMP_TARGET);
        assert!(
            (db - (-55.8)).abs() < 1.0,
            "JUMP_TARGET should be ~-56dB, got {}dB",
            db
        );
    }

    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn test_min_segment_time_constant() {
        // 1.5ms is within the safe range (1-2ms)
        // These assertions validate the constant is in a reasonable range
        assert!(
            MIN_SEGMENT_TIME_SECONDS >= 0.001,
            "Min time should be >= 1ms"
        );
        assert!(
            MIN_SEGMENT_TIME_SECONDS <= 0.003,
            "Min time should be <= 3ms"
        );
    }

    #[test]
    fn test_max_rate_sample_rate_independent() {
        // The actual minimum time should be consistent across sample rates
        let sample_rates = [22050.0, 44100.0, 48000.0, 96000.0];

        for &sr in &sample_rates {
            let max_rate = max_rate_for_sample_rate(sr);
            let samples = get_static_count(max_rate, sr);
            let actual_time = samples as f32 / sr;

            // Should be within 20% of target (due to rate quantization)
            assert!(
                actual_time >= MIN_SEGMENT_TIME_SECONDS * 0.8,
                "At {}Hz: rate {} gives {}ms, should be >= {}ms",
                sr,
                max_rate,
                actual_time * 1000.0,
                MIN_SEGMENT_TIME_SECONDS * 800.0
            );
        }
    }

    #[test]
    fn test_rate_scaling_respects_min_time() {
        // Even with maximum scaling, should not produce instant transition
        let sample_rate = 44100.0;
        let inc = calculate_increment_f32_scaled(99, 127, 7, sample_rate);

        // Calculate expected minimum samples
        let max_rate = max_rate_for_sample_rate(sample_rate);
        let min_samples = get_static_count(max_rate, sample_rate);
        let max_inc = 1.0 / min_samples as f32;

        assert!(
            inc <= max_inc + f32::EPSILON,
            "Increment {} should be <= {} (from min samples {})",
            inc,
            max_inc,
            min_samples
        );
    }

    #[test]
    fn test_rate_scaling_extreme_no_instant() {
        // Various sample rates should never produce instant (inc=1.0)
        for &sr in &[22050.0, 44100.0, 48000.0, 96000.0] {
            let inc = calculate_increment_f32_scaled(99, 127, 7, sr);
            assert!(
                inc < 1.0,
                "At {}Hz: rate 99 + max scaling should not be instant, got inc={}",
                sr,
                inc
            );
        }
    }

    #[test]
    fn test_rate_scaling_below_max_unchanged() {
        // When scaled rate is below the max, behavior is unchanged
        let sample_rate = 44100.0;
        let max_rate = max_rate_for_sample_rate(sample_rate);

        // Low note + low rate + low scaling = well under max
        let inc = calculate_increment_f32_scaled(50, 36, 3, sample_rate);
        let adj = scale_rate(36, 3);
        let expected_rate = 50 + adj;

        assert!(
            expected_rate < max_rate,
            "Test setup: should be under max rate"
        );

        let expected_inc = calculate_increment_f32(expected_rate, sample_rate);
        assert!(
            (inc - expected_inc).abs() < f32::EPSILON,
            "Below-max scaling should behave normally"
        );
    }
}
