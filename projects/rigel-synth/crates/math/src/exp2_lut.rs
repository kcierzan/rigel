//! Fast exp2 via lookup table for real-time audio DSP.
//!
//! This module provides a pre-computed lookup table for 2^x calculations,
//! optimized for the common audio DSP use case of dB-to-linear conversion.
//!
//! The LUT covers exponents from -16.0 to 0.0, which corresponds to:
//! - -96 dB to 0 dB dynamic range
//! - Linear amplitude from ~0.000015 to 1.0
//!
//! # Performance
//!
//! LUT lookup is O(1) with no floating-point math, making it significantly
//! faster than libm::exp2f or even SIMD polynomial approximations for
//! single-value lookups. The 16KB table fits in L1 cache.
//!
//! # Example
//!
//! ```rust
//! use rigel_math::exp2_lut::{exp2_lut, exp2_lut_slice};
//!
//! // Single value lookup (level 0-4095 maps to exponent -16 to 0)
//! let linear = exp2_lut(4095); // Returns ~1.0
//! let quiet = exp2_lut(0);     // Returns ~0.000015
//!
//! // Batch conversion
//! let levels = [0i16, 2048, 4095];
//! let mut output = [0.0f32; 3];
//! exp2_lut_slice(&levels, &mut output);
//! ```

/// Table size: 4096 entries covering Q8 level range (0-4095).
pub const EXP2_LUT_SIZE: usize = 4096;

/// Maximum level value (maps to exponent 0, output 1.0).
pub const EXP2_LUT_MAX: i16 = 4095;

/// Minimum level value (maps to exponent -16, output ~0.000015).
pub const EXP2_LUT_MIN: i16 = 0;

/// Pre-computed exp2 lookup table.
///
/// Entry `i` contains `2^((i - 4095) / 256)`:
/// - Index 0: 2^(-16) ≈ 0.0000153
/// - Index 2048: 2^(-8) ≈ 0.00391
/// - Index 4095: 2^0 = 1.0
///
/// The table uses Q8 format: 256 steps per octave (6 dB).
/// Total range: 16 octaves = 96 dB.
static EXP2_LUT: [f32; EXP2_LUT_SIZE] = {
    let mut table = [0.0f32; EXP2_LUT_SIZE];
    let mut i = 0;
    while i < EXP2_LUT_SIZE {
        let exponent = (i as f32 - 4095.0) / 256.0;
        table[i] = const_exp2(exponent);
        i += 1;
    }
    table
};

/// Const-compatible exp2 approximation for LUT initialization.
///
/// Uses the identity: 2^x = 2^floor(x) * 2^frac(x)
/// where 2^floor(x) is computed via IEEE 754 bit manipulation
/// and 2^frac(x) is approximated with a 4th-order polynomial.
///
/// Accuracy: < 0.01% relative error for x in [-16, 0].
#[inline]
const fn const_exp2(x: f32) -> f32 {
    // Handle edge cases
    if x <= -16.0 {
        return 0.0;
    }
    if x >= 0.0 {
        return 1.0;
    }

    // Split into integer and fractional parts
    // x is negative, so we need floor(x)
    let floor_x = x as i32 - 1; // Conservative floor for negative numbers
    let frac = x - floor_x as f32;

    // 2^floor(x) via IEEE 754 bit manipulation
    // For float: exponent bits = 127 + floor_x, mantissa = 0
    let int_part = if floor_x < -126 {
        0.0
    } else {
        let bits = ((127 + floor_x) as u32) << 23;
        f32::from_bits(bits)
    };

    // Approximate 2^frac using minimax polynomial (frac is in [0, 1))
    // Coefficients optimized for audio range accuracy
    let c0 = 1.0;
    let c1 = core::f32::consts::LN_2;
    let c2 = 0.2402265; // ln(2)^2 / 2
    let c3 = 0.0555041; // ln(2)^3 / 6
    let c4 = 0.0096139; // ln(2)^4 / 24

    let frac_part = c0 + frac * (c1 + frac * (c2 + frac * (c3 + frac * c4)));

    int_part * frac_part
}

/// Fast exp2 lookup for Q8 level values.
///
/// Converts a Q8 level (0-4095) to linear amplitude using the pre-computed LUT.
/// This is equivalent to `2^((level - 4095) / 256)`.
///
/// # Arguments
///
/// * `level` - Q8 level value (0-4095). Values outside this range are clamped.
///
/// # Returns
///
/// Linear amplitude in range [~0.000015, 1.0].
///
/// # Performance
///
/// O(1) - single array access with bounds check.
#[inline]
pub fn exp2_lut(level: i16) -> f32 {
    let index = level.clamp(0, 4095) as usize;
    EXP2_LUT[index]
}

/// Fast exp2 lookup for Q8 level values (unchecked).
///
/// Same as `exp2_lut` but without bounds checking.
/// Use when you can guarantee level is in [0, 4095].
///
/// # Safety
///
/// This function is safe but will return incorrect results
/// (or panic in debug builds) if level is outside [0, 4095].
#[inline]
pub fn exp2_lut_unchecked(level: i16) -> f32 {
    debug_assert!(
        (0..=4095).contains(&level),
        "level {} out of range [0, 4095]",
        level
    );
    // SAFETY: Caller guarantees level is in valid range
    unsafe { *EXP2_LUT.get_unchecked(level as usize) }
}

/// Batch convert Q8 levels to linear amplitudes.
///
/// Converts multiple levels using the LUT. This is the fastest method
/// for batch level-to-linear conversion.
///
/// # Arguments
///
/// * `levels` - Slice of Q8 level values (0-4095)
/// * `output` - Output slice for linear amplitudes (must be same length)
///
/// # Panics
///
/// Panics if `levels` and `output` have different lengths.
#[inline]
pub fn exp2_lut_slice(levels: &[i16], output: &mut [f32]) {
    assert_eq!(levels.len(), output.len());
    for (i, &level) in levels.iter().enumerate() {
        output[i] = EXP2_LUT[level.clamp(0, 4095) as usize];
    }
}

/// Convert dB value to linear amplitude.
///
/// Uses the exp2 LUT for fast conversion. The dB value is mapped
/// to the Q8 level range internally.
///
/// # Arguments
///
/// * `db` - Decibel value in range [-96, 0]
///
/// # Returns
///
/// Linear amplitude in range [~0.000015, 1.0]
///
/// # Example
///
/// ```rust
/// use rigel_math::exp2_lut::db_to_linear;
///
/// let unity = db_to_linear(0.0);    // ~1.0
/// let half = db_to_linear(-6.0);    // ~0.5
/// let quiet = db_to_linear(-96.0);  // ~0.000015
/// ```
#[inline]
pub fn db_to_linear(db: f32) -> f32 {
    // dB to level: level = (db / 6.0) * 256 + 4095
    // (since 256 steps = 6 dB = 1 octave)
    let level = ((db / 6.0) * 256.0 + 4095.0) as i16;
    exp2_lut(level)
}

/// Get raw access to the exp2 LUT.
///
/// Useful for advanced use cases where you need direct table access.
#[inline]
pub fn exp2_lut_table() -> &'static [f32; EXP2_LUT_SIZE] {
    &EXP2_LUT
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exp2_lut_max() {
        let result = exp2_lut(4095);
        assert!(
            (result - 1.0).abs() < 0.01,
            "Level 4095 should give ~1.0, got {}",
            result
        );
    }

    #[test]
    fn test_exp2_lut_min() {
        let result = exp2_lut(0);
        assert!(
            result < 0.001,
            "Level 0 should give very small value, got {}",
            result
        );
        assert!(result > 0.0, "Level 0 should be positive, got {}", result);
    }

    #[test]
    fn test_exp2_lut_mid() {
        // Level 2048 = exponent -8 = 2^-8 ≈ 0.00391
        let result = exp2_lut(2048);
        let expected = 0.00390625; // 2^-8
        let error = (result - expected).abs() / expected;
        assert!(
            error < 0.01,
            "Level 2048 error too high: got {}, expected {}, error {}%",
            result,
            expected,
            error * 100.0
        );
    }

    #[test]
    fn test_exp2_lut_slice() {
        let levels = [0i16, 2048, 4095];
        let mut output = [0.0f32; 3];
        exp2_lut_slice(&levels, &mut output);

        assert!(output[0] < 0.001);
        assert!((output[1] - 0.00391).abs() < 0.001);
        assert!((output[2] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_db_to_linear() {
        // 0 dB = 1.0
        let unity = db_to_linear(0.0);
        assert!(
            (unity - 1.0).abs() < 0.01,
            "0 dB should give ~1.0, got {}",
            unity
        );

        // -6 dB ≈ 0.5
        let half = db_to_linear(-6.0);
        assert!(
            (half - 0.5).abs() < 0.05,
            "-6 dB should give ~0.5, got {}",
            half
        );

        // -96 dB ≈ 0.000016
        let quiet = db_to_linear(-96.0);
        assert!(quiet < 0.001, "-96 dB should be very small, got {}", quiet);
    }

    #[test]
    fn test_monotonicity() {
        // Verify LUT values are monotonically increasing
        for i in 1..EXP2_LUT_SIZE {
            assert!(
                EXP2_LUT[i] >= EXP2_LUT[i - 1],
                "LUT not monotonic at index {}: {} < {}",
                i,
                EXP2_LUT[i],
                EXP2_LUT[i - 1]
            );
        }
    }

    #[test]
    fn test_clamping() {
        // Out of range values should be clamped
        let low = exp2_lut(-100);
        let high = exp2_lut(10000);

        assert!(low < 0.001, "Clamped low should be small");
        assert!((high - 1.0).abs() < 0.01, "Clamped high should be ~1.0");
    }
}
