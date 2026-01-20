//! Wavetable file writer.
//!
//! This module provides functionality to write wavetables as WAV files
//! with embedded protobuf metadata in the WTBL chunk.
//!
//! Note: This is primarily used for round-trip testing. The main
//! generation workflow uses Python (wtgen).

use crate::proto;
use crate::riff::build_wav_with_wtbl;
use crate::types::WavetableFile;
use crate::validation::{validate_metadata, ValidationError, ValidationErrorCode};
use anyhow::{Context, Result};
use prost::Message;
use std::fs;
use std::path::Path;

/// Write a wavetable to a WAV file with embedded metadata.
///
/// # Arguments
///
/// * `path` - Output file path.
/// * `wavetable` - The wavetable to write.
/// * `validate` - Whether to validate the data before writing.
///
/// # Errors
///
/// Returns an error if:
/// - Validation fails (and `validate` is true)
/// - The file cannot be written
pub fn write_wavetable(path: &Path, wavetable: &WavetableFile, validate: bool) -> Result<()> {
    // Validate if requested
    if validate {
        let result = validate_metadata(&wavetable.metadata);
        if !result.valid {
            // Use the first detailed error if available
            if let Some(first_error) = result.detailed_errors.first() {
                return Err(first_error.clone().into());
            }
            return Err(ValidationError::new(
                ValidationErrorCode::InvalidSchemaVersion,
                result.errors.join("; "),
            )
            .into());
        }
    }

    // Serialize metadata
    let wtbl_data = wavetable.metadata.encode_to_vec();

    // Flatten samples in mip-major, frame-secondary order
    let samples = flatten_mip_levels(wavetable);

    // Build WAV data
    let wav_data = build_wav_with_wtbl(&samples, wavetable.sample_rate, &wtbl_data);

    // Write to file
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).context("Failed to create parent directories")?;
    }
    fs::write(path, wav_data).context("Failed to write wavetable file")?;

    Ok(())
}

/// Write a wavetable to bytes (in-memory).
///
/// This is useful for testing or when you need the WAV data without writing to disk.
pub fn write_wavetable_to_bytes(wavetable: &WavetableFile) -> Vec<u8> {
    let wtbl_data = wavetable.metadata.encode_to_vec();
    let samples = flatten_mip_levels(wavetable);
    build_wav_with_wtbl(&samples, wavetable.sample_rate, &wtbl_data)
}

/// Flatten mip levels into a single sample array.
///
/// Data is organized as mip-major, frame-secondary:
/// [mip0_frame0][mip0_frame1]...[mip1_frame0][mip1_frame1]...
fn flatten_mip_levels(wavetable: &WavetableFile) -> Vec<f32> {
    let total_samples = wavetable.total_samples();
    let mut samples = Vec::with_capacity(total_samples);

    for mip in &wavetable.mip_levels {
        for frame in &mip.frames {
            samples.extend(frame);
        }
    }

    samples
}

/// Builder for creating wavetable files.
///
/// This provides a convenient API for constructing wavetables programmatically.
#[derive(Debug, Clone)]
pub struct WavetableBuilder {
    wavetable_type: proto::WavetableType,
    frame_length: u32,
    num_frames: u32,
    sample_rate: u32,
    name: Option<String>,
    author: Option<String>,
    description: Option<String>,
    normalization_method: proto::NormalizationMethod,
    mip_data: Vec<Vec<Vec<f32>>>, // mip_level -> frame -> samples
}

impl WavetableBuilder {
    /// Create a new builder with the given basic parameters.
    pub fn new(wavetable_type: proto::WavetableType, frame_length: u32, num_frames: u32) -> Self {
        Self {
            wavetable_type,
            frame_length,
            num_frames,
            sample_rate: 44100,
            name: None,
            author: None,
            description: None,
            normalization_method: proto::NormalizationMethod::NormalizationUnspecified,
            mip_data: Vec::new(),
        }
    }

    /// Set the sample rate.
    pub fn sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = rate;
        self
    }

    /// Set the name.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the author.
    pub fn author(mut self, author: impl Into<String>) -> Self {
        self.author = Some(author.into());
        self
    }

    /// Set the description.
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Set the normalization method.
    pub fn normalization_method(mut self, method: proto::NormalizationMethod) -> Self {
        self.normalization_method = method;
        self
    }

    /// Add a mip level with the given frames.
    ///
    /// Each frame should have the appropriate length for that mip level.
    /// Mip levels must be added in order (0, 1, 2, ...).
    pub fn add_mip_level(mut self, frames: Vec<Vec<f32>>) -> Self {
        self.mip_data.push(frames);
        self
    }

    /// Build the wavetable file.
    pub fn build(self) -> Result<WavetableFile> {
        use crate::types::MipLevel;

        if self.mip_data.is_empty() {
            anyhow::bail!("At least one mip level is required");
        }

        // Build metadata
        let mut metadata = proto::WavetableMetadata {
            schema_version: 1,
            frame_length: self.frame_length,
            num_frames: self.num_frames,
            num_mip_levels: self.mip_data.len() as u32,
            normalization_method: self.normalization_method.into(),
            name: self.name,
            author: self.author,
            description: self.description,
            sample_rate: Some(self.sample_rate),
            ..Default::default()
        };
        metadata.set_wavetable_type(self.wavetable_type);

        // Calculate mip frame lengths and build mip levels
        let mut mip_levels = Vec::with_capacity(self.mip_data.len());

        for (level, frames) in self.mip_data.into_iter().enumerate() {
            if frames.is_empty() {
                anyhow::bail!("Mip level {} has no frames", level);
            }

            let frame_length = frames[0].len() as u32;

            // Verify all frames have the same length
            for (i, frame) in frames.iter().enumerate() {
                if frame.len() != frame_length as usize {
                    anyhow::bail!(
                        "Mip level {} frame {} has {} samples, expected {}",
                        level,
                        i,
                        frame.len(),
                        frame_length
                    );
                }
            }

            // Verify frame count matches
            if frames.len() != self.num_frames as usize {
                anyhow::bail!(
                    "Mip level {} has {} frames, expected {}",
                    level,
                    frames.len(),
                    self.num_frames
                );
            }

            metadata.mip_frame_lengths.push(frame_length);
            mip_levels.push(MipLevel::new(level, frame_length, frames));
        }

        Ok(WavetableFile::new(metadata, mip_levels, self.sample_rate))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::read_wavetable_from_bytes;

    #[test]
    fn test_roundtrip_write_read() {
        // Build a test wavetable
        let frame0 = vec![0.0f32, 0.5, 1.0, 0.5];
        let frame1 = vec![0.0f32, -0.5, -1.0, -0.5];

        let wavetable = WavetableBuilder::new(proto::WavetableType::Custom, 4, 2)
            .name("Test Wavetable")
            .author("Test Author")
            .sample_rate(48000)
            .add_mip_level(vec![frame0.clone(), frame1.clone()])
            .build()
            .unwrap();

        // Write to bytes
        let wav_data = write_wavetable_to_bytes(&wavetable);

        // Read back
        let loaded = read_wavetable_from_bytes(&wav_data).unwrap();

        // Verify
        assert_eq!(loaded.metadata.name, Some("Test Wavetable".to_string()));
        assert_eq!(loaded.metadata.author, Some("Test Author".to_string()));
        assert_eq!(loaded.sample_rate, 48000);
        assert_eq!(loaded.mip_levels.len(), 1);
        assert_eq!(loaded.mip_levels[0].frames[0], frame0);
        assert_eq!(loaded.mip_levels[0].frames[1], frame1);
    }

    #[test]
    fn test_builder_validation() {
        // Empty mip data should fail
        let result = WavetableBuilder::new(proto::WavetableType::Custom, 4, 2).build();
        assert!(result.is_err());
    }
}
