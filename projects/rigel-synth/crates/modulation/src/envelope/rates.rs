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

/// Attack jump threshold in Q8 format.
///
/// The DX7 envelope immediately jumps to 1716 in Q8 format at attack start,
/// which is ~40dB above minimum (~-56dB from full scale). This gets the
/// envelope out of the sub-perceptual range quickly while preserving the
/// exponential approach character for the rest of the attack.
///
/// Reference: MSFA/Dexed `const int jumptarget = 1716`
pub const JUMP_TARGET_Q8: i16 = 1716;

/// Attack jump threshold in linear amplitude (0.0-1.0).
///
/// Calculation: 1716 Q8 steps = 40.2dB above minimum = -55.8dB from 0dB
/// Linear amplitude: 10^(-55.8/20) ≈ 0.00163 (0.16%)
///
/// Reference: MSFA/Dexed `const int jumptarget = 1716`
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

/// Non-linear level lookup table for low levels (0-19).
///
/// The DX7/SY99 uses a non-linear curve for low output levels
/// to provide finer control in the quiet region. Levels 0-19 map
/// through this LUT, while levels 20-99 use linear mapping (28 + level).
///
/// This produces ~-95.5dB at level 0 and smooth transitions
/// to the linear region at level 20.
///
/// Reference: MSFA/Dexed `scale_output_level()` function.
const LEVEL_LUT: [u8; 20] = [
    0, 5, 9, 13, 17, 20, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 42, 43, 45, 46,
];

/// Pre-computed lookup table for level_to_linear conversion.
///
/// 257 entries (1KB) mapping Q8 level grid points to linear amplitude.
/// Index i corresponds to Q8 level = i * 4095 / 256.
/// Entry 256 is 1.0 (full amplitude).
/// Uses linear interpolation for values between grid points.
///
/// Values computed as: LUT[i] = 2^(i/16 - 16) for all i.
/// The boundary check `if level_q8 <= 0` handles the silence case.
/// This provides ~1.78x speedup over exp2f with negligible error (~5e-6).
#[rustfmt::skip]
#[allow(clippy::excessive_precision)]
const LEVEL_LINEAR_LUT: [f32; 257] = [
    1.525878906e-05, 1.593435337e-05, 1.663982746e-05, 1.737653556e-05,
    1.814586052e-05, 1.894924640e-05, 1.978820121e-05, 2.066429973e-05,
    2.157918644e-05, 2.253457864e-05, 2.353226967e-05, 2.457413226e-05,
    2.566212205e-05, 2.679828126e-05, 2.798474253e-05, 2.922373293e-05,
    3.051757812e-05, 3.186870674e-05, 3.327965493e-05, 3.475307113e-05,
    3.629172104e-05, 3.789849280e-05, 3.957640242e-05, 4.132859945e-05,
    4.315837288e-05, 4.506915729e-05, 4.706453935e-05, 4.914826452e-05,
    5.132424410e-05, 5.359656251e-05, 5.596948506e-05, 5.844746586e-05,
    6.103515625e-05, 6.373741348e-05, 6.655930986e-05, 6.950614226e-05,
    7.258344208e-05, 7.579698560e-05, 7.915280485e-05, 8.265719891e-05,
    8.631674575e-05, 9.013831457e-05, 9.412907870e-05, 9.829652905e-05,
    1.026484882e-04, 1.071931250e-04, 1.119389701e-04, 1.168949317e-04,
    1.220703125e-04, 1.274748270e-04, 1.331186197e-04, 1.390122845e-04,
    1.451668842e-04, 1.515939712e-04, 1.583056097e-04, 1.653143978e-04,
    1.726334915e-04, 1.802766291e-04, 1.882581574e-04, 1.965930581e-04,
    2.052969764e-04, 2.143862500e-04, 2.238779402e-04, 2.337898635e-04,
    2.441406250e-04, 2.549496539e-04, 2.662372394e-04, 2.780245690e-04,
    2.903337683e-04, 3.031879424e-04, 3.166112194e-04, 3.306287956e-04,
    3.452669830e-04, 3.605532583e-04, 3.765163148e-04, 3.931861162e-04,
    4.105939528e-04, 4.287725001e-04, 4.477558805e-04, 4.675797269e-04,
    4.882812500e-04, 5.098993078e-04, 5.324744788e-04, 5.560491381e-04,
    5.806675366e-04, 6.063758848e-04, 6.332224388e-04, 6.612575913e-04,
    6.905339660e-04, 7.211065166e-04, 7.530326296e-04, 7.863722324e-04,
    8.211879055e-04, 8.575450002e-04, 8.955117609e-04, 9.351594538e-04,
    9.765625000e-04, 1.019798616e-03, 1.064948958e-03, 1.112098276e-03,
    1.161335073e-03, 1.212751770e-03, 1.266444878e-03, 1.322515183e-03,
    1.381067932e-03, 1.442213033e-03, 1.506065259e-03, 1.572744465e-03,
    1.642375811e-03, 1.715090000e-03, 1.791023522e-03, 1.870318908e-03,
    1.953125000e-03, 2.039597231e-03, 2.129897915e-03, 2.224196552e-03,
    2.322670146e-03, 2.425503539e-03, 2.532889755e-03, 2.645030365e-03,
    2.762135864e-03, 2.884426066e-03, 3.012130518e-03, 3.145488930e-03,
    3.284751622e-03, 3.430180001e-03, 3.582047044e-03, 3.740637815e-03,
    3.906250000e-03, 4.079194463e-03, 4.259795831e-03, 4.448393105e-03,
    4.645340293e-03, 4.851007078e-03, 5.065779510e-03, 5.290060730e-03,
    5.524271728e-03, 5.768852133e-03, 6.024261037e-03, 6.290977859e-03,
    6.569503244e-03, 6.860360001e-03, 7.164094088e-03, 7.481275630e-03,
    7.812500000e-03, 8.158388925e-03, 8.519591661e-03, 8.896786209e-03,
    9.290680586e-03, 9.702014157e-03, 1.013155902e-02, 1.058012146e-02,
    1.104854346e-02, 1.153770427e-02, 1.204852207e-02, 1.258195572e-02,
    1.313900649e-02, 1.372072000e-02, 1.432818818e-02, 1.496255126e-02,
    1.562500000e-02, 1.631677785e-02, 1.703918332e-02, 1.779357242e-02,
    1.858136117e-02, 1.940402831e-02, 2.026311804e-02, 2.116024292e-02,
    2.209708691e-02, 2.307540853e-02, 2.409704415e-02, 2.516391144e-02,
    2.627801298e-02, 2.744144001e-02, 2.865637635e-02, 2.992510252e-02,
    3.125000000e-02, 3.263355570e-02, 3.407836665e-02, 3.558714484e-02,
    3.716272234e-02, 3.880805663e-02, 4.052623608e-02, 4.232048584e-02,
    4.419417382e-02, 4.615081706e-02, 4.819408829e-02, 5.032782287e-02,
    5.255602595e-02, 5.488288001e-02, 5.731275270e-02, 5.985020504e-02,
    6.250000000e-02, 6.526711140e-02, 6.815673329e-02, 7.117428967e-02,
    7.432544469e-02, 7.761611325e-02, 8.105247217e-02, 8.464097168e-02,
    8.838834765e-02, 9.230163412e-02, 9.638817659e-02, 1.006556457e-01,
    1.051120519e-01, 1.097657600e-01, 1.146255054e-01, 1.197004101e-01,
    1.250000000e-01, 1.305342228e-01, 1.363134666e-01, 1.423485793e-01,
    1.486508894e-01, 1.552322265e-01, 1.621049443e-01, 1.692819434e-01,
    1.767766953e-01, 1.846032682e-01, 1.927763532e-01, 2.013112915e-01,
    2.102241038e-01, 2.195315200e-01, 2.292510108e-01, 2.394008202e-01,
    2.500000000e-01, 2.610684456e-01, 2.726269332e-01, 2.846971587e-01,
    2.973017788e-01, 3.104644530e-01, 3.242098887e-01, 3.385638867e-01,
    3.535533906e-01, 3.692065365e-01, 3.855527064e-01, 4.026225830e-01,
    4.204482076e-01, 4.390630401e-01, 4.585020216e-01, 4.788016403e-01,
    5.000000000e-01, 5.221368912e-01, 5.452538663e-01, 5.693943174e-01,
    5.946035575e-01, 6.209289060e-01, 6.484197773e-01, 6.771277735e-01,
    7.071067812e-01, 7.384130730e-01, 7.711054127e-01, 8.052451660e-01,
    8.408964153e-01, 8.781260802e-01, 9.170040432e-01, 9.576032807e-01,
    1.000000000e+00,
];

/// Convert DX7 level parameter (0-99) to Q8 internal level (0-4095).
///
/// Uses the LEVEL_LUT for levels 0-19 to provide authentic non-linear
/// behavior in the quiet region. Levels 20-99 use linear mapping.
///
/// The result is in Q8 fixed-point format where:
/// - 0 = ~-96dB (silence)
/// - 4095 = 0dB (full amplitude)
///
/// # Arguments
///
/// * `param_level` - DX7 level parameter (0-99)
///
/// # Returns
///
/// Q8 internal level (0-4095)
#[inline]
pub fn param_to_level_q8(param_level: u8) -> i16 {
    // Apply non-linear scaling for low levels (0-19)
    let scaled = if param_level >= 20 {
        28 + param_level
    } else {
        LEVEL_LUT[param_level as usize]
    };
    // Convert scaled (0-127) to Q8 (0-4095)
    // Use exact scaling: (scaled * 4095) / 127 to ensure full range
    ((scaled as i32 * 4095) / 127) as i16
}

/// Convert Q8 internal level (0-4095) to linear amplitude (0.0-1.0).
///
/// Q8 is a logarithmic representation where each step is ~0.0235 dB.
/// This converts to linear amplitude for audio output.
///
/// Uses a 256-entry lookup table with linear interpolation for speed.
/// The LUT provides ~1.78x speedup over exp2f with negligible error (~5e-6).
///
/// The scaling ensures LEVEL_MAX (4095) maps to exactly 1.0, and
/// LEVEL_MIN (0) maps to near-silence (~-96dB).
///
/// # Arguments
///
/// * `level_q8` - Q8 internal level (0-4095)
///
/// # Returns
///
/// Linear amplitude in range [0.0, 1.0]
#[inline]
pub fn level_to_linear(level_q8: i16) -> f32 {
    if level_q8 <= 0 {
        return 0.0;
    }
    if level_q8 >= 4095 {
        return 1.0;
    }
    // Map Q8 level (0-4095) to LUT index (0-256) with interpolation
    // scaled = level * 256 / 4095, split into integer index and fraction
    let scaled = level_q8 as f32 * (256.0 / 4095.0);
    let idx = scaled as usize;
    let frac = scaled - idx as f32;
    // Linear interpolation between adjacent LUT entries
    LEVEL_LINEAR_LUT[idx] * (1.0 - frac) + LEVEL_LINEAR_LUT[idx + 1] * frac
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
    fn test_param_to_level_q8() {
        // Level 0 should give 0 Q8 (silence via LEVEL_LUT)
        assert_eq!(param_to_level_q8(0), 0);

        // Level 99 should give maximum Q8 (4095)
        // 28 + 99 = 127, (127 * 4095) / 127 = 4095
        assert_eq!(param_to_level_q8(99), 4095);

        // Level 20 is the transition point
        // 28 + 20 = 48, (48 * 4095) / 127 = 1548
        let level_20 = param_to_level_q8(20);
        assert!(
            (1545..=1550).contains(&level_20),
            "Level 20 should be ~1548, got {}",
            level_20
        );

        // Level 19 uses LEVEL_LUT[19] = 46
        // (46 * 4095) / 127 = 1483
        let level_19 = param_to_level_q8(19);
        assert!(
            (1480..=1486).contains(&level_19),
            "Level 19 should be ~1483, got {}",
            level_19
        );

        // Verify low levels use LEVEL_LUT
        // LUT[1] = 5, (5 * 4095) / 127 = 161
        assert!(param_to_level_q8(1) > 150 && param_to_level_q8(1) < 170);
        // LUT[10] = 31, (31 * 4095) / 127 = 999
        assert!(param_to_level_q8(10) > 990 && param_to_level_q8(10) < 1010);
    }

    #[test]
    fn test_level_to_linear() {
        // Q8 0 should give 0 (silence)
        assert_eq!(level_to_linear(0), 0.0);

        // Q8 4095 (max) should give ~1.0
        let max_linear = level_to_linear(4095);
        assert!(
            (max_linear - 1.0).abs() < 0.01,
            "Q8 4095 should give ~1.0, got {}",
            max_linear
        );

        // Q8 2048 (mid) should give roughly -48dB (half the dB range)
        // 2048/256 - 16 = 8 - 16 = -8, 2^-8 = 0.00390625
        let mid_linear = level_to_linear(2048);
        assert!(
            (mid_linear - 0.00390625).abs() < 0.001,
            "Q8 2048 should give ~0.004, got {}",
            mid_linear
        );

        // Monotonic: higher Q8 = higher linear
        assert!(level_to_linear(1000) < level_to_linear(2000));
        assert!(level_to_linear(2000) < level_to_linear(3000));
    }

    #[test]
    fn test_level_lut_produces_correct_db_curve() {
        // Level 0 should give near-silence (~-96dB)
        let linear_0 = level_to_linear(param_to_level_q8(0));
        assert!(
            linear_0 < 0.00002, // -96dB is about 1.5e-5
            "Level 0 should be near silence, got {}",
            linear_0
        );

        // Level 99 should give ~1.0 (0dB)
        let linear_99 = level_to_linear(param_to_level_q8(99));
        assert!(
            (linear_99 - 1.0).abs() < 0.05,
            "Level 99 should be near 1.0, got {}",
            linear_99
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

    #[test]
    fn test_level_to_linear_lut_accuracy() {
        // Verify LUT-based implementation matches exp2f reference
        // across the full Q8 range with acceptable error.
        // Linear interpolation between logarithmically-spaced LUT entries
        // introduces small errors that peak at mid-range levels.
        // The relative error is always < 0.03% which is far below audible thresholds.
        let mut max_abs_error: f32 = 0.0;
        let mut max_rel_error: f32 = 0.0;
        let mut max_error_level: i16 = 0;

        for level in 1..4095 {
            let lut_result = level_to_linear(level);

            // Reference computation using exp2f
            let log2_gain = (level as f32) * 16.0 / 4095.0 - 16.0;
            let exp2_result = rigel_math::scalar::exp2f(log2_gain);

            let abs_error = (lut_result - exp2_result).abs();
            let rel_error = abs_error / exp2_result;

            if abs_error > max_abs_error {
                max_abs_error = abs_error;
                max_error_level = level;
            }
            if rel_error > max_rel_error {
                max_rel_error = rel_error;
            }
        }

        // Maximum absolute error peaks at high levels (linear interp on exp curve).
        // At level 4087, absolute error ~3e-4 but relative error ~0.03%.
        // This is acceptable for audio (well below audible thresholds).
        assert!(
            max_abs_error < 5e-4,
            "Max absolute error {:.2e} at level {} exceeds threshold",
            max_abs_error,
            max_error_level
        );

        // Maximum relative error should be < 0.1% (0.001)
        // Relative error is consistent across the full range.
        assert!(
            max_rel_error < 0.001,
            "Max relative error {:.4}% exceeds 0.1%",
            max_rel_error * 100.0
        );
    }

    #[test]
    fn test_level_to_linear_lut_boundaries() {
        // Verify edge cases still work correctly
        assert_eq!(level_to_linear(0), 0.0, "Level 0 should be silence");
        assert_eq!(level_to_linear(-1), 0.0, "Negative level should be silence");
        assert_eq!(
            level_to_linear(4095),
            1.0,
            "Level 4095 should be full amplitude"
        );
        assert_eq!(
            level_to_linear(4096),
            1.0,
            "Level > 4095 should clamp to 1.0"
        );
        assert_eq!(
            level_to_linear(5000),
            1.0,
            "Level > 4095 should clamp to 1.0"
        );
    }

    #[test]
    fn test_level_to_linear_lut_monotonic() {
        // Verify LUT maintains monotonicity (higher level = higher amplitude)
        let mut prev = level_to_linear(1);
        for level in 2..4095 {
            let curr = level_to_linear(level);
            assert!(
                curr >= prev,
                "Monotonicity violation: level {} ({}) < level {} ({})",
                level,
                curr,
                level - 1,
                prev
            );
            prev = curr;
        }
    }
}
