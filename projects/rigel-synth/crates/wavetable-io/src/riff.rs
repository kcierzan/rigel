//! RIFF/WAV chunk utilities for wavetable files.
//!
//! This module provides utilities for reading and writing custom RIFF chunks,
//! specifically the WTBL chunk that contains protobuf-encoded wavetable metadata.
//!
//! Rather than depending on the riff crate's complex API, we implement simple
//! RIFF parsing directly since we only need basic chunk reading/writing.

use anyhow::{bail, Context, Result};
use std::io::{Seek, SeekFrom, Write};

/// The FourCC identifier for the wavetable metadata chunk.
pub const WTBL_CHUNK_ID: [u8; 4] = *b"WTBL";

/// The FourCC identifier for the WAVE format.
pub const WAVE_FORMAT_ID: [u8; 4] = *b"WAVE";

/// The FourCC identifier for the fmt chunk.
pub const FMT_CHUNK_ID: [u8; 4] = *b"fmt ";

/// The FourCC identifier for the data chunk.
pub const DATA_CHUNK_ID: [u8; 4] = *b"data";

/// The FourCC identifier for RIFF.
pub const RIFF_ID: [u8; 4] = *b"RIFF";

/// A simple RIFF chunk representation.
#[derive(Debug, Clone)]
pub struct RiffChunk {
    /// The chunk's FourCC identifier.
    pub id: [u8; 4],
    /// The chunk's data content.
    pub data: Vec<u8>,
}

/// Parse a RIFF file and extract all chunks.
///
/// # Arguments
///
/// * `data` - The raw bytes of the RIFF file.
///
/// # Returns
///
/// A vector of chunks found in the file.
pub fn parse_riff_chunks(data: &[u8]) -> Result<Vec<RiffChunk>> {
    // Verify RIFF header
    if data.len() < 12 {
        bail!("File too small to be a valid RIFF file");
    }

    if &data[0..4] != b"RIFF" {
        bail!("Not a RIFF file (missing RIFF header)");
    }

    let file_size = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
    if &data[8..12] != b"WAVE" {
        bail!("Not a WAVE file (missing WAVE format)");
    }

    // Parse chunks starting at offset 12
    let mut chunks = Vec::new();
    let mut offset = 12;
    // Use saturating_add to prevent overflow on malicious input (H-1 fix)
    let end = file_size.saturating_add(8).min(data.len());

    while offset + 8 <= end {
        let chunk_id: [u8; 4] = data[offset..offset + 4].try_into().unwrap();
        let chunk_size = u32::from_le_bytes([
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]) as usize;

        let data_start = offset + 8;
        // Use checked arithmetic to prevent overflow on malicious chunk sizes (H-2 fix)
        let data_end = data_start
            .checked_add(chunk_size)
            .map(|v| v.min(data.len()))
            .unwrap_or(data.len());

        if data_end > data.len() {
            break;
        }

        chunks.push(RiffChunk {
            id: chunk_id,
            data: data[data_start..data_end].to_vec(),
        });

        // Move to next chunk (chunks are word-aligned)
        offset = data_end + (chunk_size % 2);
    }

    Ok(chunks)
}

/// Extract the WTBL chunk data from a RIFF/WAV file.
///
/// # Arguments
///
/// * `data` - The raw bytes of the RIFF file.
///
/// # Returns
///
/// The raw bytes of the WTBL chunk content, or an error if the chunk is not found.
pub fn extract_wtbl_chunk(data: &[u8]) -> Result<Vec<u8>> {
    let chunks = parse_riff_chunks(data).context("Failed to parse RIFF structure")?;

    for chunk in chunks {
        if chunk.id == WTBL_CHUNK_ID {
            return Ok(chunk.data);
        }
    }

    bail!("WTBL chunk not found in WAV file")
}

/// Read the fmt chunk to extract audio format information.
///
/// # Returns
///
/// A tuple of (num_channels, sample_rate, bits_per_sample)
pub fn read_fmt_chunk(data: &[u8]) -> Result<(u16, u32, u16)> {
    let chunks = parse_riff_chunks(data).context("Failed to parse RIFF structure")?;

    for chunk in chunks {
        if chunk.id == FMT_CHUNK_ID {
            let content = &chunk.data;
            if content.len() < 16 {
                bail!("fmt chunk too small");
            }

            // Parse PCM format
            let audio_format = u16::from_le_bytes([content[0], content[1]]);
            let num_channels = u16::from_le_bytes([content[2], content[3]]);
            let sample_rate = u32::from_le_bytes([content[4], content[5], content[6], content[7]]);
            let bits_per_sample = u16::from_le_bytes([content[14], content[15]]);

            // Verify it's PCM or IEEE float
            if audio_format != 1 && audio_format != 3 {
                bail!(
                    "Unsupported audio format: {} (expected PCM=1 or IEEE_FLOAT=3)",
                    audio_format
                );
            }

            return Ok((num_channels, sample_rate, bits_per_sample));
        }
    }

    bail!("fmt chunk not found")
}

/// Read the data chunk to extract audio samples.
///
/// # Returns
///
/// The raw bytes of the audio data.
pub fn read_data_chunk(data: &[u8]) -> Result<Vec<u8>> {
    let chunks = parse_riff_chunks(data).context("Failed to parse RIFF structure")?;

    for chunk in chunks {
        if chunk.id == DATA_CHUNK_ID {
            return Ok(chunk.data);
        }
    }

    bail!("data chunk not found")
}

/// Append a WTBL chunk to an existing WAV file.
///
/// This modifies the file in place, appending the chunk and updating the RIFF size.
///
/// # Arguments
///
/// * `writer` - A writable and seekable stream positioned at the end of the WAV file.
/// * `wtbl_data` - The protobuf-encoded metadata to store in the WTBL chunk.
pub fn append_wtbl_chunk<W: Write + Seek>(writer: &mut W, wtbl_data: &[u8]) -> Result<()> {
    // Seek to end to append the chunk
    writer.seek(SeekFrom::End(0))?;

    // Write WTBL chunk header (FourCC + size)
    writer.write_all(&WTBL_CHUNK_ID)?;
    let chunk_size = wtbl_data.len() as u32;
    writer.write_all(&chunk_size.to_le_bytes())?;

    // Write chunk data
    writer.write_all(wtbl_data)?;

    // Pad to word boundary if needed
    if wtbl_data.len() % 2 != 0 {
        writer.write_all(&[0u8])?;
    }

    // Calculate new RIFF size (file size - 8 for RIFF header)
    let new_file_size = writer.seek(SeekFrom::End(0))?;
    let new_riff_size = (new_file_size - 8) as u32;

    // Update RIFF size at offset 4
    writer.seek(SeekFrom::Start(4))?;
    writer.write_all(&new_riff_size.to_le_bytes())?;

    // Seek back to end
    writer.seek(SeekFrom::End(0))?;

    Ok(())
}

/// Build a complete RIFF/WAV file with audio data and WTBL chunk.
///
/// # Arguments
///
/// * `samples` - Audio samples as f32 values (mono, 32-bit float).
/// * `sample_rate` - The sample rate in Hz.
/// * `wtbl_data` - The protobuf-encoded metadata for the WTBL chunk.
///
/// # Returns
///
/// The complete WAV file as bytes.
pub fn build_wav_with_wtbl(samples: &[f32], sample_rate: u32, wtbl_data: &[u8]) -> Vec<u8> {
    let num_samples = samples.len();
    let data_size = (num_samples * 4) as u32; // 4 bytes per f32

    // Build fmt chunk (IEEE float format)
    let fmt_content = build_fmt_chunk_ieee_float(1, sample_rate);

    // Build data chunk
    let mut data_content = Vec::with_capacity(num_samples * 4);
    for sample in samples {
        data_content.extend_from_slice(&sample.to_le_bytes());
    }

    // Calculate total file size
    // RIFF header (12) + fmt chunk (8 + 16) + data chunk (8 + data_size) + WTBL chunk (8 + wtbl_size + padding)
    let wtbl_padded_size = wtbl_data.len() + (wtbl_data.len() % 2);
    let total_size = 12 + 24 + 8 + data_size as usize + 8 + wtbl_padded_size;

    let mut wav = Vec::with_capacity(total_size);

    // RIFF header
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&((total_size - 8) as u32).to_le_bytes());
    wav.extend_from_slice(b"WAVE");

    // fmt chunk
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes()); // chunk size
    wav.extend_from_slice(&fmt_content);

    // data chunk
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_size.to_le_bytes());
    wav.extend_from_slice(&data_content);

    // WTBL chunk
    wav.extend_from_slice(b"WTBL");
    wav.extend_from_slice(&(wtbl_data.len() as u32).to_le_bytes());
    wav.extend_from_slice(wtbl_data);
    if wtbl_data.len() % 2 != 0 {
        wav.push(0); // padding
    }

    wav
}

/// Build the fmt chunk content for IEEE float format.
fn build_fmt_chunk_ieee_float(num_channels: u16, sample_rate: u32) -> Vec<u8> {
    let mut fmt = Vec::with_capacity(16);

    let audio_format: u16 = 3; // IEEE float
    let bits_per_sample: u16 = 32;
    let byte_rate = sample_rate * num_channels as u32 * 4; // 4 bytes per sample
    let block_align = num_channels * 4;

    fmt.extend_from_slice(&audio_format.to_le_bytes());
    fmt.extend_from_slice(&num_channels.to_le_bytes());
    fmt.extend_from_slice(&sample_rate.to_le_bytes());
    fmt.extend_from_slice(&byte_rate.to_le_bytes());
    fmt.extend_from_slice(&block_align.to_le_bytes());
    fmt.extend_from_slice(&bits_per_sample.to_le_bytes());

    fmt
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_wav_with_wtbl_roundtrip() {
        let samples = vec![0.0f32, 0.5, 1.0, -0.5, -1.0, 0.0];
        let sample_rate = 44100;
        let wtbl_data = b"test metadata";

        let wav = build_wav_with_wtbl(&samples, sample_rate, wtbl_data);

        // Verify we can extract the WTBL chunk
        let extracted = extract_wtbl_chunk(&wav).unwrap();
        assert_eq!(extracted, wtbl_data);

        // Verify fmt chunk
        let (channels, rate, bits) = read_fmt_chunk(&wav).unwrap();
        assert_eq!(channels, 1);
        assert_eq!(rate, sample_rate);
        assert_eq!(bits, 32);

        // Verify data chunk
        let data = read_data_chunk(&wav).unwrap();
        assert_eq!(data.len(), samples.len() * 4);
    }

    #[test]
    fn test_parse_riff_chunks() {
        let samples = vec![0.0f32, 0.5, 1.0];
        let wtbl_data = b"metadata";

        let wav = build_wav_with_wtbl(&samples, 44100, wtbl_data);
        let chunks = parse_riff_chunks(&wav).unwrap();

        // Should have fmt, data, and WTBL chunks
        assert_eq!(chunks.len(), 3);
        assert_eq!(&chunks[0].id, b"fmt ");
        assert_eq!(&chunks[1].id, b"data");
        assert_eq!(&chunks[2].id, b"WTBL");
    }

    #[test]
    fn test_invalid_riff() {
        let invalid = b"not a riff file";
        let result = parse_riff_chunks(invalid);
        assert!(result.is_err());
    }
}
