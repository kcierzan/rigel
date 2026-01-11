//! Custom parameter formatters for envelope time display.
//!
//! Provides conversion between time values (seconds) used in the plugin UI
//! and DX7-compatible rate values (0-99) used internally by the DSP.
//!
//! # Thread Safety
//!
//! - `time_to_string` and `string_to_time`: Used by nih-plug for UI display only.
//!   These allocate strings and are wrapped in Arc, but are NEVER called from
//!   the audio thread. They're only invoked by the host/GUI for parameter display.
//!
//! - `time_to_rate`: Called from the audio thread in `process()`. This is a simple
//!   no-allocation function that delegates to `seconds_to_rate()` from rigel_modulation.

use rigel_modulation::envelope::seconds_to_rate;

/// Format time in seconds to human-readable string.
///
/// # Examples
/// - 0.0005 -> "0.5ms"
/// - 0.150 -> "150ms"
/// - 1.5 -> "1.50s"
pub fn time_to_string(seconds: f32) -> String {
    if seconds >= 1.0 {
        format!("{:.2}s", seconds)
    } else if seconds >= 0.1 {
        format!("{:.0}ms", seconds * 1000.0)
    } else {
        format!("{:.1}ms", seconds * 1000.0)
    }
}

/// Parse time string to seconds.
///
/// Accepts formats:
/// - "500ms" -> 0.5
/// - "0.5s" -> 0.5
/// - "1.2s" -> 1.2
/// - "0.5" -> 0.5 (interpreted as seconds)
pub fn string_to_time(input: &str) -> Option<f32> {
    let trimmed = input.trim().to_lowercase();

    if let Some(s) = trimmed.strip_suffix("ms") {
        s.trim().parse::<f32>().ok().map(|ms| ms / 1000.0)
    } else if let Some(s) = trimmed.strip_suffix('s') {
        s.trim().parse::<f32>().ok()
    } else {
        // Assume seconds if no unit
        trimmed.parse::<f32>().ok()
    }
}

/// Convert time (seconds) to DX7 rate (0-99) for DSP.
///
/// This function delegates to the rigel_modulation rate conversion,
/// which uses the MSFA-compatible STATICS table and formulas.
#[inline]
pub fn time_to_rate(seconds: f32, sample_rate: f32) -> u8 {
    seconds_to_rate(seconds, sample_rate)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_to_string_seconds() {
        assert_eq!(time_to_string(1.0), "1.00s");
        assert_eq!(time_to_string(1.5), "1.50s");
        assert_eq!(time_to_string(10.0), "10.00s");
        assert_eq!(time_to_string(40.0), "40.00s");
    }

    #[test]
    fn test_time_to_string_milliseconds() {
        assert_eq!(time_to_string(0.5), "500ms");
        assert_eq!(time_to_string(0.150), "150ms");
        assert_eq!(time_to_string(0.1), "100ms");
    }

    #[test]
    fn test_time_to_string_sub_100ms() {
        assert_eq!(time_to_string(0.05), "50.0ms");
        assert_eq!(time_to_string(0.01), "10.0ms");
        assert_eq!(time_to_string(0.001), "1.0ms");
        assert_eq!(time_to_string(0.0005), "0.5ms");
    }

    #[test]
    fn test_string_to_time_milliseconds() {
        assert_eq!(string_to_time("500ms"), Some(0.5));
        assert_eq!(string_to_time("150ms"), Some(0.150));
        assert_eq!(string_to_time("1ms"), Some(0.001));
        assert_eq!(string_to_time("0.5ms"), Some(0.0005));
    }

    #[test]
    fn test_string_to_time_seconds() {
        assert_eq!(string_to_time("1s"), Some(1.0));
        assert_eq!(string_to_time("1.5s"), Some(1.5));
        assert_eq!(string_to_time("0.5s"), Some(0.5));
    }

    #[test]
    fn test_string_to_time_raw_float() {
        assert_eq!(string_to_time("0.5"), Some(0.5));
        assert_eq!(string_to_time("1.0"), Some(1.0));
    }

    #[test]
    fn test_string_to_time_whitespace() {
        assert_eq!(string_to_time("  500ms  "), Some(0.5));
        assert_eq!(string_to_time("  1.5s  "), Some(1.5));
    }

    #[test]
    fn test_string_to_time_case_insensitive() {
        assert_eq!(string_to_time("500MS"), Some(0.5));
        assert_eq!(string_to_time("1.5S"), Some(1.5));
    }

    #[test]
    fn test_string_to_time_invalid() {
        assert_eq!(string_to_time("abc"), None);
        assert_eq!(string_to_time(""), None);
    }

    #[test]
    fn test_time_to_rate() {
        // Very short time -> high rate (fast envelope)
        let fast_rate = time_to_rate(0.001, 44100.0);
        assert!(
            fast_rate > 70,
            "1ms should give high rate, got {}",
            fast_rate
        );

        // Long time -> low rate (slow envelope)
        let slow_rate = time_to_rate(5.0, 44100.0);
        assert!(slow_rate < 30, "5s should give low rate, got {}", slow_rate);

        // Medium time -> medium rate
        let mid_rate = time_to_rate(0.5, 44100.0);
        assert!(
            mid_rate > 20 && mid_rate < 60,
            "500ms should give medium rate, got {}",
            mid_rate
        );
    }

    #[test]
    fn test_roundtrip_formatting() {
        // Test that formatting and parsing are consistent
        let original = 0.150;
        let formatted = time_to_string(original);
        let parsed = string_to_time(&formatted).unwrap();

        // Allow for some floating point imprecision
        assert!(
            (original - parsed).abs() < 0.001,
            "Roundtrip failed: {} -> {} -> {}",
            original,
            formatted,
            parsed
        );
    }
}
