//! Validation functions for wavetable metadata and audio data.
//!
//! This module provides validation functions to ensure wavetable files
//! conform to the interchange format specification.
//!
//! # Error Codes
//!
//! All validation errors include an error code for programmatic handling:
//!
//! | Code | Description |
//! |------|-------------|
//! | E001 | Invalid schema version |
//! | E002 | Invalid frame length |
//! | E003 | Invalid number of frames |
//! | E004 | Invalid number of mip levels |
//! | E005 | Mip frame lengths count mismatch |
//! | E006 | Mip frame lengths[0] != frame_length |
//! | E007 | Mip frame lengths not decreasing |
//! | E008 | Mip level count mismatch |
//! | E009 | Frame count mismatch in mip level |
//! | E010 | Frame length mismatch |
//! | E011 | Non-finite sample value |
//! | E012 | Audio data length mismatch |

use crate::proto;
use crate::types::WavetableFile;
use std::fmt;

/// Error codes for validation failures.
///
/// These codes can be used for programmatic error handling and localization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationErrorCode {
    /// E001: Schema version is less than 1
    InvalidSchemaVersion,
    /// E002: Frame length is 0
    InvalidFrameLength,
    /// E003: Number of frames is 0
    InvalidNumFrames,
    /// E004: Number of mip levels is 0
    InvalidNumMipLevels,
    /// E005: mip_frame_lengths array has wrong number of entries
    MipFrameLengthsCountMismatch,
    /// E006: mip_frame_lengths[0] does not equal frame_length
    MipFrameLengthsFirstMismatch,
    /// E007: mip_frame_lengths values are not strictly decreasing
    MipFrameLengthsNotDecreasing,
    /// E008: Audio data has wrong number of mip levels
    MipLevelCountMismatch,
    /// E009: A mip level has the wrong number of frames
    FrameCountMismatch,
    /// E010: A frame has the wrong number of samples
    FrameLengthMismatch,
    /// E011: A sample value is NaN or Infinity
    NonFiniteSample,
    /// E012: Total audio data length doesn't match metadata
    AudioDataLengthMismatch,
}

impl ValidationErrorCode {
    /// Get the error code string (e.g., "E001").
    pub fn code(&self) -> &'static str {
        match self {
            Self::InvalidSchemaVersion => "E001",
            Self::InvalidFrameLength => "E002",
            Self::InvalidNumFrames => "E003",
            Self::InvalidNumMipLevels => "E004",
            Self::MipFrameLengthsCountMismatch => "E005",
            Self::MipFrameLengthsFirstMismatch => "E006",
            Self::MipFrameLengthsNotDecreasing => "E007",
            Self::MipLevelCountMismatch => "E008",
            Self::FrameCountMismatch => "E009",
            Self::FrameLengthMismatch => "E010",
            Self::NonFiniteSample => "E011",
            Self::AudioDataLengthMismatch => "E012",
        }
    }

    /// Get guidance on how to fix this error.
    pub fn guidance(&self) -> &'static str {
        match self {
            Self::InvalidSchemaVersion => {
                "Set schema_version to 1 (or higher) in the WavetableMetadata protobuf message."
            }
            Self::InvalidFrameLength => {
                "Set frame_length to a positive value (typically a power of 2 like 256, 512, 1024, or 2048)."
            }
            Self::InvalidNumFrames => {
                "Set num_frames to the number of waveform keyframes (typically 64 for classic wavetables)."
            }
            Self::InvalidNumMipLevels => {
                "Set num_mip_levels to the number of mip levels in the audio data (at least 1)."
            }
            Self::MipFrameLengthsCountMismatch => {
                "Ensure mip_frame_lengths has exactly num_mip_levels entries, one for each mip level."
            }
            Self::MipFrameLengthsFirstMismatch => {
                "mip_frame_lengths[0] must equal frame_length since mip level 0 is the highest resolution."
            }
            Self::MipFrameLengthsNotDecreasing => {
                "mip_frame_lengths must be strictly decreasing (each mip level is half the previous)."
            }
            Self::MipLevelCountMismatch => {
                "Audio data must contain exactly num_mip_levels mip levels."
            }
            Self::FrameCountMismatch => {
                "Each mip level must contain exactly num_frames frames."
            }
            Self::FrameLengthMismatch => {
                "Each frame in mip level N must have exactly mip_frame_lengths[N] samples."
            }
            Self::NonFiniteSample => {
                "All audio samples must be finite floating-point values (no NaN or Infinity)."
            }
            Self::AudioDataLengthMismatch => {
                "Total audio samples must equal sum(mip_frame_lengths[i] * num_frames for each mip level)."
            }
        }
    }
}

impl fmt::Display for ValidationErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.code())
    }
}

/// Error type for validation failures.
///
/// Provides detailed information for debugging, including:
/// - Error code for programmatic handling
/// - Human-readable message
/// - Optional field path
/// - Guidance on how to fix the error
#[derive(Debug, Clone)]
pub struct ValidationError {
    /// The error code for programmatic handling.
    pub code: ValidationErrorCode,
    /// The error message.
    pub message: String,
    /// The field that failed validation, if applicable.
    pub field: Option<String>,
}

impl ValidationError {
    /// Create a new validation error with a code.
    pub fn new(code: ValidationErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            field: None,
        }
    }

    /// Create a validation error for a specific field.
    pub fn for_field(
        code: ValidationErrorCode,
        field: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            code,
            message: message.into(),
            field: Some(field.into()),
        }
    }

    /// Get guidance on how to fix this error.
    pub fn guidance(&self) -> &'static str {
        self.code.guidance()
    }

    /// Format the error with full details for debugging.
    pub fn detailed_message(&self) -> String {
        let field_info = self
            .field
            .as_ref()
            .map(|f| format!(" (field: {})", f))
            .unwrap_or_default();
        format!(
            "[{}]{}: {}\n  Guidance: {}",
            self.code.code(),
            field_info,
            self.message,
            self.guidance()
        )
    }
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(field) = &self.field {
            write!(f, "[{}] {}: {}", self.code.code(), field, self.message)
        } else {
            write!(f, "[{}] {}", self.code.code(), self.message)
        }
    }
}

impl std::error::Error for ValidationError {}

/// Result of validation with optional warnings.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether validation passed.
    pub valid: bool,
    /// List of error messages (empty if valid).
    pub errors: Vec<String>,
    /// List of detailed errors with codes (empty if valid).
    pub detailed_errors: Vec<ValidationError>,
    /// List of warnings (may be present even if valid).
    pub warnings: Vec<String>,
}

impl ValidationResult {
    /// Create a successful validation result.
    pub fn success() -> Self {
        Self {
            valid: true,
            errors: Vec::new(),
            detailed_errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    /// Create a successful validation result with warnings.
    pub fn success_with_warnings(warnings: Vec<String>) -> Self {
        Self {
            valid: true,
            errors: Vec::new(),
            detailed_errors: Vec::new(),
            warnings,
        }
    }

    /// Create a failed validation result.
    pub fn failure(detailed_errors: Vec<ValidationError>) -> Self {
        Self {
            valid: false,
            errors: detailed_errors.iter().map(|e| e.message.clone()).collect(),
            detailed_errors,
            warnings: Vec::new(),
        }
    }

    /// Create a failed validation result with warnings.
    pub fn failure_with_warnings(
        detailed_errors: Vec<ValidationError>,
        warnings: Vec<String>,
    ) -> Self {
        Self {
            valid: false,
            errors: detailed_errors.iter().map(|e| e.message.clone()).collect(),
            detailed_errors,
            warnings,
        }
    }
}

/// Validate a complete wavetable file.
///
/// This validates both the metadata and audio data for consistency.
///
/// # Arguments
///
/// * `wavetable` - The wavetable file to validate.
///
/// # Returns
///
/// `Ok(())` if validation passes, or an error describing the first failure.
///
/// # Example
///
/// ```ignore
/// use wavetable_io::{read_wavetable, validate_wavetable};
///
/// let wavetable = read_wavetable("my_wavetable.wav")?;
/// match validate_wavetable(&wavetable) {
///     Ok(()) => println!("Valid!"),
///     Err(e) => {
///         eprintln!("Validation failed: {}", e);
///         eprintln!("Details: {}", e.detailed_message());
///     }
/// }
/// ```
pub fn validate_wavetable(wavetable: &WavetableFile) -> Result<(), ValidationError> {
    // Validate metadata
    let metadata_result = validate_metadata(&wavetable.metadata);
    if !metadata_result.valid {
        // Return the first error with its code
        if let Some(first_error) = metadata_result.detailed_errors.first() {
            return Err(first_error.clone());
        }
        // Fallback (shouldn't happen)
        return Err(ValidationError::new(
            ValidationErrorCode::InvalidSchemaVersion,
            metadata_result.errors.join("; "),
        ));
    }

    // Validate audio data matches metadata
    validate_audio_data(wavetable)?;

    Ok(())
}

/// Validate wavetable metadata for structural correctness.
///
/// This performs MUST-pass validation per the spec:
/// - schema_version >= 1
/// - num_frames > 0
/// - num_mip_levels > 0
/// - mip_frame_lengths has correct count
/// - frame_length > 0
///
/// Returns a ValidationResult with detailed error codes for each failure.
pub fn validate_metadata(metadata: &proto::WavetableMetadata) -> ValidationResult {
    let mut errors = Vec::new();
    let mut warnings = Vec::new();

    // Schema version check
    if metadata.schema_version < 1 {
        errors.push(ValidationError::for_field(
            ValidationErrorCode::InvalidSchemaVersion,
            "schema_version",
            format!(
                "schema_version must be >= 1, got {}. The current schema version is 1.",
                metadata.schema_version
            ),
        ));
    }

    // Frame length check
    if metadata.frame_length == 0 {
        errors.push(ValidationError::for_field(
            ValidationErrorCode::InvalidFrameLength,
            "frame_length",
            "frame_length must be > 0. Typical values are 256 (classic), 512, 1024, or 2048 (high-res).".to_string(),
        ));
    } else if !is_power_of_two(metadata.frame_length) {
        warnings.push(format!(
            "frame_length should be a power of 2, got {}. Non-power-of-2 values may affect playback quality.",
            metadata.frame_length
        ));
    }

    // Num frames check
    if metadata.num_frames == 0 {
        errors.push(ValidationError::for_field(
            ValidationErrorCode::InvalidNumFrames,
            "num_frames",
            "num_frames must be > 0. This is the number of waveform keyframes (typically 64)."
                .to_string(),
        ));
    }

    // Num mip levels check
    if metadata.num_mip_levels == 0 {
        errors.push(ValidationError::for_field(
            ValidationErrorCode::InvalidNumMipLevels,
            "num_mip_levels",
            "num_mip_levels must be > 0. At minimum, provide 1 mip level for the base wavetable."
                .to_string(),
        ));
    }

    // Mip frame lengths check
    let expected_count = metadata.num_mip_levels as usize;
    let actual_count = metadata.mip_frame_lengths.len();
    if actual_count != expected_count {
        errors.push(ValidationError::for_field(
            ValidationErrorCode::MipFrameLengthsCountMismatch,
            "mip_frame_lengths",
            format!(
                "mip_frame_lengths has {} entries, expected {} (must match num_mip_levels). \
                 Add or remove entries so mip_frame_lengths.len() == num_mip_levels.",
                actual_count, expected_count
            ),
        ));
    } else if !metadata.mip_frame_lengths.is_empty() {
        // Check mip_frame_lengths[0] == frame_length
        if metadata.mip_frame_lengths[0] != metadata.frame_length {
            errors.push(ValidationError::for_field(
                ValidationErrorCode::MipFrameLengthsFirstMismatch,
                "mip_frame_lengths[0]",
                format!(
                    "mip_frame_lengths[0] ({}) must equal frame_length ({}). \
                     Mip level 0 is the highest resolution and must match the declared frame_length.",
                    metadata.mip_frame_lengths[0], metadata.frame_length
                ),
            ));
        }

        // Check mip_frame_lengths are decreasing
        for i in 1..metadata.mip_frame_lengths.len() {
            if metadata.mip_frame_lengths[i] > metadata.mip_frame_lengths[i - 1] {
                errors.push(ValidationError::for_field(
                    ValidationErrorCode::MipFrameLengthsNotDecreasing,
                    format!("mip_frame_lengths[{}]", i),
                    format!(
                        "mip_frame_lengths must be strictly decreasing (lower mip levels have fewer samples), \
                         but mip_frame_lengths[{}]={} > mip_frame_lengths[{}]={}. \
                         Example valid sequence: [2048, 1024, 512, 256, 128].",
                        i, metadata.mip_frame_lengths[i], i - 1, metadata.mip_frame_lengths[i - 1]
                    ),
                ));
                break;
            }
        }

        // Check mip_frame_lengths are powers of 2 (warning only)
        for (i, &length) in metadata.mip_frame_lengths.iter().enumerate() {
            if !is_power_of_two(length) {
                warnings.push(format!(
                    "mip_frame_lengths[{}] should be a power of 2, got {}. \
                     Non-power-of-2 values may affect playback quality and SIMD optimization.",
                    i, length
                ));
            }
        }
    }

    // Wavetable type check (warning only)
    if metadata.wavetable_type() == proto::WavetableType::Unspecified {
        warnings.push(
            "wavetable_type is UNSPECIFIED (0), will be treated as CUSTOM. \
             Consider setting an explicit type for better handling hints."
                .to_string(),
        );
    }

    if errors.is_empty() {
        ValidationResult::success_with_warnings(warnings)
    } else {
        ValidationResult::failure_with_warnings(errors, warnings)
    }
}

/// Validate audio data matches metadata declarations.
fn validate_audio_data(wavetable: &WavetableFile) -> Result<(), ValidationError> {
    let metadata = &wavetable.metadata;

    // Check number of mip levels
    if wavetable.mip_levels.len() != metadata.num_mip_levels as usize {
        return Err(ValidationError::for_field(
            ValidationErrorCode::MipLevelCountMismatch,
            "mip_levels",
            format!(
                "Audio data contains {} mip levels, but metadata declares num_mip_levels={}. \
                 The audio data must be organized as [mip0_frames][mip1_frames]...[mipN_frames] \
                 with exactly num_mip_levels sections.",
                wavetable.mip_levels.len(),
                metadata.num_mip_levels
            ),
        ));
    }

    // Check each mip level
    for (i, mip) in wavetable.mip_levels.iter().enumerate() {
        let expected_frame_length = metadata.mip_frame_lengths.get(i).copied().unwrap_or(0);
        let expected_num_frames = metadata.num_frames as usize;

        // Check number of frames
        if mip.frames.len() != expected_num_frames {
            return Err(ValidationError::for_field(
                ValidationErrorCode::FrameCountMismatch,
                format!("mip_levels[{}]", i),
                format!(
                    "Mip level {} has {} frames, expected {} (num_frames). \
                     Each mip level must contain exactly num_frames waveform frames.",
                    i,
                    mip.frames.len(),
                    expected_num_frames
                ),
            ));
        }

        // Check frame length for each frame
        for (j, frame) in mip.frames.iter().enumerate() {
            if frame.len() != expected_frame_length as usize {
                return Err(ValidationError::for_field(
                    ValidationErrorCode::FrameLengthMismatch,
                    format!("mip_levels[{}].frames[{}]", i, j),
                    format!(
                        "Frame {} in mip level {} has {} samples, expected {} (mip_frame_lengths[{}]). \
                         Each frame at mip level N must have exactly mip_frame_lengths[N] samples.",
                        j,
                        i,
                        frame.len(),
                        expected_frame_length,
                        i
                    ),
                ));
            }

            // Check for NaN/Inf
            for (k, &sample) in frame.iter().enumerate() {
                if !sample.is_finite() {
                    return Err(ValidationError::for_field(
                        ValidationErrorCode::NonFiniteSample,
                        format!("mip_levels[{}].frames[{}][{}]", i, j, k),
                        format!(
                            "Sample at mip {}, frame {}, index {} is non-finite: {}. \
                             All samples must be valid IEEE 754 floating-point values in range [-1.0, 1.0]. \
                             Check for division by zero or other arithmetic errors in generation.",
                            i, j, k, sample
                        ),
                    ));
                }
            }
        }
    }

    Ok(())
}

/// Calculate the expected total number of samples for a wavetable.
pub fn calculate_expected_samples(metadata: &proto::WavetableMetadata) -> usize {
    metadata
        .mip_frame_lengths
        .iter()
        .map(|&frame_length| frame_length as usize * metadata.num_frames as usize)
        .sum()
}

/// Check if a number is a power of two.
fn is_power_of_two(n: u32) -> bool {
    n > 0 && (n & (n - 1)) == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_metadata_valid() {
        let metadata = proto::WavetableMetadata {
            schema_version: 1,
            wavetable_type: proto::WavetableType::HighResolution.into(),
            frame_length: 2048,
            num_frames: 64,
            num_mip_levels: 3,
            mip_frame_lengths: vec![2048, 1024, 512],
            ..Default::default()
        };

        let result = validate_metadata(&metadata);
        assert!(
            result.valid,
            "Expected valid, got errors: {:?}",
            result.errors
        );
    }

    #[test]
    fn test_validate_metadata_invalid_schema_version() {
        let metadata = proto::WavetableMetadata {
            schema_version: 0,
            frame_length: 256,
            num_frames: 64,
            num_mip_levels: 1,
            mip_frame_lengths: vec![256],
            ..Default::default()
        };

        let result = validate_metadata(&metadata);
        assert!(!result.valid);
        assert!(result.errors.iter().any(|e| e.contains("schema_version")));

        // Check error code
        assert!(result
            .detailed_errors
            .iter()
            .any(|e| e.code == ValidationErrorCode::InvalidSchemaVersion));
    }

    #[test]
    fn test_validate_metadata_mismatched_mip_lengths() {
        let metadata = proto::WavetableMetadata {
            schema_version: 1,
            frame_length: 256,
            num_frames: 64,
            num_mip_levels: 3,
            mip_frame_lengths: vec![256, 128], // Only 2 entries, expected 3
            ..Default::default()
        };

        let result = validate_metadata(&metadata);
        assert!(!result.valid);
        assert!(result
            .errors
            .iter()
            .any(|e| e.contains("mip_frame_lengths")));

        // Check error code
        assert!(result
            .detailed_errors
            .iter()
            .any(|e| e.code == ValidationErrorCode::MipFrameLengthsCountMismatch));
    }

    #[test]
    fn test_validate_metadata_mip_lengths_not_decreasing() {
        let metadata = proto::WavetableMetadata {
            schema_version: 1,
            frame_length: 256,
            num_frames: 64,
            num_mip_levels: 3,
            mip_frame_lengths: vec![256, 128, 256], // Not decreasing!
            ..Default::default()
        };

        let result = validate_metadata(&metadata);
        assert!(!result.valid);
        assert!(result
            .detailed_errors
            .iter()
            .any(|e| e.code == ValidationErrorCode::MipFrameLengthsNotDecreasing));
    }

    #[test]
    fn test_validate_metadata_mip_lengths_first_mismatch() {
        let metadata = proto::WavetableMetadata {
            schema_version: 1,
            frame_length: 256,
            num_frames: 64,
            num_mip_levels: 2,
            mip_frame_lengths: vec![512, 256], // mip_frame_lengths[0] != frame_length
            ..Default::default()
        };

        let result = validate_metadata(&metadata);
        assert!(!result.valid);
        assert!(result
            .detailed_errors
            .iter()
            .any(|e| e.code == ValidationErrorCode::MipFrameLengthsFirstMismatch));
    }

    #[test]
    fn test_is_power_of_two() {
        assert!(is_power_of_two(1));
        assert!(is_power_of_two(2));
        assert!(is_power_of_two(4));
        assert!(is_power_of_two(256));
        assert!(is_power_of_two(2048));
        assert!(!is_power_of_two(0));
        assert!(!is_power_of_two(3));
        assert!(!is_power_of_two(100));
    }

    #[test]
    fn test_error_code_strings() {
        assert_eq!(ValidationErrorCode::InvalidSchemaVersion.code(), "E001");
        assert_eq!(ValidationErrorCode::InvalidFrameLength.code(), "E002");
        assert_eq!(ValidationErrorCode::InvalidNumFrames.code(), "E003");
        assert_eq!(ValidationErrorCode::InvalidNumMipLevels.code(), "E004");
        assert_eq!(
            ValidationErrorCode::MipFrameLengthsCountMismatch.code(),
            "E005"
        );
        assert_eq!(
            ValidationErrorCode::MipFrameLengthsFirstMismatch.code(),
            "E006"
        );
        assert_eq!(
            ValidationErrorCode::MipFrameLengthsNotDecreasing.code(),
            "E007"
        );
        assert_eq!(ValidationErrorCode::MipLevelCountMismatch.code(), "E008");
        assert_eq!(ValidationErrorCode::FrameCountMismatch.code(), "E009");
        assert_eq!(ValidationErrorCode::FrameLengthMismatch.code(), "E010");
        assert_eq!(ValidationErrorCode::NonFiniteSample.code(), "E011");
        assert_eq!(ValidationErrorCode::AudioDataLengthMismatch.code(), "E012");
    }

    #[test]
    fn test_error_guidance() {
        // All error codes should have non-empty guidance
        let codes = [
            ValidationErrorCode::InvalidSchemaVersion,
            ValidationErrorCode::InvalidFrameLength,
            ValidationErrorCode::InvalidNumFrames,
            ValidationErrorCode::InvalidNumMipLevels,
            ValidationErrorCode::MipFrameLengthsCountMismatch,
            ValidationErrorCode::MipFrameLengthsFirstMismatch,
            ValidationErrorCode::MipFrameLengthsNotDecreasing,
            ValidationErrorCode::MipLevelCountMismatch,
            ValidationErrorCode::FrameCountMismatch,
            ValidationErrorCode::FrameLengthMismatch,
            ValidationErrorCode::NonFiniteSample,
            ValidationErrorCode::AudioDataLengthMismatch,
        ];

        for code in codes {
            assert!(
                !code.guidance().is_empty(),
                "Error code {:?} should have guidance",
                code
            );
        }
    }

    #[test]
    fn test_detailed_message_format() {
        let error = ValidationError::for_field(
            ValidationErrorCode::InvalidSchemaVersion,
            "schema_version",
            "Test message",
        );

        let detailed = error.detailed_message();
        assert!(detailed.contains("[E001]"));
        assert!(detailed.contains("schema_version"));
        assert!(detailed.contains("Test message"));
        assert!(detailed.contains("Guidance:"));
    }
}
