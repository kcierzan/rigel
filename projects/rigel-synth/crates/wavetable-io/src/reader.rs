//! Wavetable file reader.
//!
//! This module provides functionality to load wavetables from WAV files
//! with embedded protobuf metadata in the WTBL chunk.

use crate::proto;
use crate::riff::{extract_wtbl_chunk, read_data_chunk, read_fmt_chunk};
use crate::types::{MipLevel, WavetableFile};
use anyhow::{bail, Context, Result};
use prost::Message;
use std::fs;
use std::path::Path;

/// Maximum file size in bytes (100 MB per FR-030b).
pub const MAX_FILE_SIZE_BYTES: u64 = 100 * 1024 * 1024;

/// Read a wavetable from a WAV file with embedded metadata.
///
/// # Arguments
///
/// * `path` - Path to the WAV file.
///
/// # Returns
///
/// A `WavetableFile` containing the metadata and audio data.
///
/// # Errors
///
/// Returns an error if:
/// - The file cannot be read
/// - The file exceeds 100 MB (FR-030b)
/// - The file is not a valid WAV file
/// - The WTBL chunk is missing
/// - The protobuf metadata cannot be decoded
/// - The audio data doesn't match the metadata
/// - The audio samples contain NaN or Infinity values (FR-028)
pub fn read_wavetable(path: &Path) -> Result<WavetableFile> {
    // FR-030b: Check file size limit (100 MB)
    let file_size = fs::metadata(path)
        .context("Failed to read file metadata")?
        .len();
    if file_size > MAX_FILE_SIZE_BYTES {
        bail!(
            "File size ({:.1} MB) exceeds maximum allowed size of 100 MB per FR-030b",
            file_size as f64 / (1024.0 * 1024.0)
        );
    }

    // Read the entire file
    let data = fs::read(path).context("Failed to read wavetable file")?;

    read_wavetable_from_bytes(&data)
}

/// Read a wavetable from raw bytes.
///
/// This is useful for testing or when the file is already in memory.
pub fn read_wavetable_from_bytes(data: &[u8]) -> Result<WavetableFile> {
    // Extract and parse metadata
    let wtbl_data = extract_wtbl_chunk(data)?;
    let metadata = proto::WavetableMetadata::decode(wtbl_data.as_slice())
        .context("Failed to decode protobuf metadata")?;

    // Read audio format
    let (num_channels, sample_rate, bits_per_sample) = read_fmt_chunk(data)?;

    if num_channels != 1 {
        bail!("Expected mono audio, got {} channels", num_channels);
    }

    // Read audio data
    let audio_data = read_data_chunk(data)?;

    // Convert bytes to samples
    let samples = match bits_per_sample {
        32 => {
            // IEEE float (most common for our format)
            audio_data
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect::<Vec<_>>()
        }
        16 => {
            // 16-bit PCM
            audio_data
                .chunks_exact(2)
                .map(|chunk| {
                    let value = i16::from_le_bytes([chunk[0], chunk[1]]);
                    value as f32 / 32768.0
                })
                .collect::<Vec<_>>()
        }
        24 => {
            // 24-bit PCM
            decode_24bit_pcm(&audio_data)
        }
        _ => bail!("Unsupported bits per sample: {}", bits_per_sample),
    };

    // FR-028: Check for NaN/Infinity values immediately after parsing
    // This catches issues early before processing
    validate_samples_finite(&samples)?;

    // Split samples into mip levels
    let mip_levels = split_into_mip_levels(&samples, &metadata)?;

    Ok(WavetableFile::new(metadata, mip_levels, sample_rate))
}

/// Validate that all samples are finite (no NaN or Infinity) per FR-028.
fn validate_samples_finite(samples: &[f32]) -> Result<()> {
    let mut nan_count = 0;
    let mut inf_count = 0;
    let mut first_bad_idx = None;
    let mut first_bad_val = 0.0f32;

    for (i, &sample) in samples.iter().enumerate() {
        if !sample.is_finite() {
            if first_bad_idx.is_none() {
                first_bad_idx = Some(i);
                first_bad_val = sample;
            }
            if sample.is_nan() {
                nan_count += 1;
            } else {
                inf_count += 1;
            }
        }
    }

    if nan_count > 0 || inf_count > 0 {
        bail!(
            "Audio samples contain non-finite values per FR-028: {} NaN, {} Infinity. \
             First occurrence at sample index {} (value: {})",
            nan_count,
            inf_count,
            first_bad_idx.unwrap_or(0),
            first_bad_val
        );
    }

    Ok(())
}

/// Split flat sample array into mip levels.
///
/// Data is expected in mip-major, frame-secondary order:
/// [mip0_frame0][mip0_frame1]...[mip1_frame0][mip1_frame1]...
fn split_into_mip_levels(
    samples: &[f32],
    metadata: &proto::WavetableMetadata,
) -> Result<Vec<MipLevel>> {
    let mut mip_levels = Vec::with_capacity(metadata.num_mip_levels as usize);
    let mut offset = 0;

    for level in 0..metadata.num_mip_levels as usize {
        let frame_length = metadata.mip_frame_lengths.get(level).copied().unwrap_or(0) as usize;

        if frame_length == 0 {
            bail!("mip_frame_lengths[{}] is 0", level);
        }

        let total_samples = frame_length * metadata.num_frames as usize;

        if offset + total_samples > samples.len() {
            bail!(
                "Not enough samples for mip level {}: expected {} samples at offset {}, but only {} available",
                level, total_samples, offset, samples.len() - offset
            );
        }

        // Split into frames
        let mut frames = Vec::with_capacity(metadata.num_frames as usize);
        for frame_idx in 0..metadata.num_frames as usize {
            let frame_start = offset + frame_idx * frame_length;
            let frame_end = frame_start + frame_length;
            frames.push(samples[frame_start..frame_end].to_vec());
        }

        mip_levels.push(MipLevel::new(level, frame_length as u32, frames));
        offset += total_samples;
    }

    Ok(mip_levels)
}

/// Decode 24-bit PCM samples to f32.
fn decode_24bit_pcm(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(3)
        .map(|chunk| {
            // Read 3 bytes as little-endian 24-bit signed integer
            let value = chunk[0] as i32 | ((chunk[1] as i32) << 8) | ((chunk[2] as i32) << 16);

            // Sign-extend from 24-bit to 32-bit
            let value = if value & 0x800000 != 0 {
                value | 0xFF000000_u32 as i32
            } else {
                value
            };

            value as f32 / 8388608.0 // 2^23
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::riff::build_wav_with_wtbl;

    #[test]
    fn test_read_wavetable_basic() {
        // Create a minimal test wavetable
        let metadata = proto::WavetableMetadata {
            schema_version: 1,
            wavetable_type: proto::WavetableType::Custom.into(),
            frame_length: 4,
            num_frames: 2,
            num_mip_levels: 1,
            mip_frame_lengths: vec![4],
            ..Default::default()
        };

        let wtbl_data = metadata.encode_to_vec();

        // Create samples: 2 frames of 4 samples each
        let samples = vec![0.0f32, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0, -0.5];

        let wav_data = build_wav_with_wtbl(&samples, 44100, &wtbl_data);

        // Read it back
        let wavetable = read_wavetable_from_bytes(&wav_data).unwrap();

        assert_eq!(wavetable.metadata.schema_version, 1);
        assert_eq!(wavetable.metadata.frame_length, 4);
        assert_eq!(wavetable.metadata.num_frames, 2);
        assert_eq!(wavetable.mip_levels.len(), 1);
        assert_eq!(wavetable.mip_levels[0].frames.len(), 2);
        assert_eq!(wavetable.mip_levels[0].frames[0].len(), 4);
    }

    #[test]
    fn test_read_wavetable_multi_mip() {
        // Create test wavetable with multiple mip levels
        let metadata = proto::WavetableMetadata {
            schema_version: 1,
            wavetable_type: proto::WavetableType::HighResolution.into(),
            frame_length: 8,
            num_frames: 2,
            num_mip_levels: 2,
            mip_frame_lengths: vec![8, 4],
            ..Default::default()
        };

        let wtbl_data = metadata.encode_to_vec();

        // Create samples: mip0 (2 frames * 8 samples) + mip1 (2 frames * 4 samples)
        let mut samples = vec![0.5f32; 2 * 8];
        samples.extend(vec![0.25f32; 2 * 4]);

        let wav_data = build_wav_with_wtbl(&samples, 48000, &wtbl_data);

        let wavetable = read_wavetable_from_bytes(&wav_data).unwrap();

        assert_eq!(wavetable.mip_levels.len(), 2);
        assert_eq!(wavetable.mip_levels[0].frame_length, 8);
        assert_eq!(wavetable.mip_levels[1].frame_length, 4);
        assert_eq!(wavetable.sample_rate, 48000);
    }
}
