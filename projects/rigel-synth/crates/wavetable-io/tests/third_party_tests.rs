//! Tests for third-party wavetable file creation.
//!
//! These tests verify that wavetable files can be created manually following
//! the format specification, simulating what a third-party implementation would do.
//! This validates the format documentation completeness.

use prost::Message;
use wavetable_io::proto;
use wavetable_io::reader::read_wavetable_from_bytes;
use wavetable_io::validation::{validate_metadata, validate_wavetable};

/// Build a minimal WAV file with WTBL chunk manually (simulating third-party creation).
///
/// This function creates a WAV file byte-by-byte without using wavetable-io writer,
/// demonstrating what a third-party implementation would do.
fn build_manual_wavetable(
    frame_length: u32,
    num_frames: u32,
    mip_frame_lengths: &[u32],
    wavetable_type: proto::WavetableType,
    sample_rate: u32,
) -> Vec<u8> {
    // Create metadata
    let metadata = proto::WavetableMetadata {
        schema_version: 1,
        wavetable_type: wavetable_type.into(),
        frame_length,
        num_frames,
        num_mip_levels: mip_frame_lengths.len() as u32,
        mip_frame_lengths: mip_frame_lengths.to_vec(),
        name: Some("Third-Party Created".to_string()),
        author: Some("Manual Test".to_string()),
        ..Default::default()
    };

    let wtbl_data = metadata.encode_to_vec();

    // Calculate total samples
    let total_samples: u32 = mip_frame_lengths.iter().map(|&len| len * num_frames).sum();

    // Create sample data (simple sine wave for each frame)
    let mut samples: Vec<f32> = Vec::with_capacity(total_samples as usize);
    for mip_len in mip_frame_lengths {
        for frame_idx in 0..num_frames {
            for sample_idx in 0..*mip_len {
                let phase = (sample_idx as f32 / *mip_len as f32) * std::f32::consts::TAU;
                let value = (phase * (frame_idx + 1) as f32).sin() * 0.5;
                samples.push(value);
            }
        }
    }

    // Build WAV manually
    let mut wav_data = Vec::new();

    // RIFF header (12 bytes)
    wav_data.extend_from_slice(b"RIFF");
    let riff_size_pos = wav_data.len();
    wav_data.extend_from_slice(&0u32.to_le_bytes()); // Placeholder
    wav_data.extend_from_slice(b"WAVE");

    // fmt chunk (24 bytes)
    wav_data.extend_from_slice(b"fmt ");
    wav_data.extend_from_slice(&16u32.to_le_bytes()); // chunk size
    wav_data.extend_from_slice(&3u16.to_le_bytes()); // format = IEEE float
    wav_data.extend_from_slice(&1u16.to_le_bytes()); // channels = 1
    wav_data.extend_from_slice(&sample_rate.to_le_bytes()); // sample rate
    let byte_rate = sample_rate * 4; // 4 bytes per sample (float32)
    wav_data.extend_from_slice(&byte_rate.to_le_bytes()); // byte rate
    wav_data.extend_from_slice(&4u16.to_le_bytes()); // block align
    wav_data.extend_from_slice(&32u16.to_le_bytes()); // bits per sample

    // data chunk header + samples
    wav_data.extend_from_slice(b"data");
    let data_size = total_samples * 4;
    wav_data.extend_from_slice(&data_size.to_le_bytes());
    for sample in &samples {
        wav_data.extend_from_slice(&sample.to_le_bytes());
    }

    // WTBL chunk
    wav_data.extend_from_slice(b"WTBL");
    wav_data.extend_from_slice(&(wtbl_data.len() as u32).to_le_bytes());
    wav_data.extend_from_slice(&wtbl_data);

    // Pad for word alignment if needed
    if wtbl_data.len() % 2 == 1 {
        wav_data.push(0);
    }

    // Update RIFF size
    let riff_size = (wav_data.len() - 8) as u32;
    wav_data[riff_size_pos..riff_size_pos + 4].copy_from_slice(&riff_size.to_le_bytes());

    wav_data
}

/// Test that a manually-crafted simple wavetable passes validation.
#[test]
fn test_third_party_simple_wavetable() {
    let wav_data = build_manual_wavetable(
        256,                                  // frame_length
        64,                                   // num_frames
        &[256, 128, 64, 32, 16, 8, 4],        // mip_frame_lengths
        proto::WavetableType::ClassicDigital, // type
        44100,                                // sample_rate
    );

    // Verify it can be read
    let wavetable = read_wavetable_from_bytes(&wav_data).expect("Should read successfully");

    // Verify metadata
    assert_eq!(wavetable.metadata.schema_version, 1);
    assert_eq!(
        wavetable.metadata.wavetable_type(),
        proto::WavetableType::ClassicDigital
    );
    assert_eq!(wavetable.metadata.frame_length, 256);
    assert_eq!(wavetable.metadata.num_frames, 64);
    assert_eq!(wavetable.metadata.num_mip_levels, 7);
    assert_eq!(
        wavetable.metadata.name,
        Some("Third-Party Created".to_string())
    );

    // Verify validation passes
    let result = validate_wavetable(&wavetable);
    assert!(result.is_ok(), "Validation should pass: {:?}", result);
}

/// Test that a high-resolution third-party wavetable passes validation.
#[test]
fn test_third_party_high_resolution_wavetable() {
    let wav_data = build_manual_wavetable(
        2048,
        64,
        &[2048, 1024, 512, 256, 128, 64, 32, 16],
        proto::WavetableType::HighResolution,
        48000,
    );

    let wavetable = read_wavetable_from_bytes(&wav_data).expect("Should read successfully");

    assert_eq!(
        wavetable.metadata.wavetable_type(),
        proto::WavetableType::HighResolution
    );
    assert_eq!(wavetable.mip_levels.len(), 8);

    let result = validate_wavetable(&wavetable);
    assert!(result.is_ok(), "Validation should pass: {:?}", result);
}

/// Test that a minimal single-mip wavetable passes validation.
#[test]
fn test_third_party_minimal_wavetable() {
    let wav_data = build_manual_wavetable(
        64,
        4,
        &[64], // Single mip level
        proto::WavetableType::Custom,
        44100,
    );

    let wavetable = read_wavetable_from_bytes(&wav_data).expect("Should read successfully");

    assert_eq!(wavetable.metadata.num_mip_levels, 1);
    assert_eq!(wavetable.mip_levels.len(), 1);
    assert_eq!(wavetable.mip_levels[0].frames.len(), 4);

    let result = validate_wavetable(&wavetable);
    assert!(result.is_ok(), "Validation should pass: {:?}", result);
}

/// Test that metadata validation catches common third-party errors.
#[test]
#[allow(clippy::field_reassign_with_default)]
fn test_third_party_validation_errors() {
    // Test 1: Missing schema version (0 is invalid)
    let mut metadata = proto::WavetableMetadata::default();
    metadata.schema_version = 0; // Invalid!
    metadata.frame_length = 256;
    metadata.num_frames = 64;
    metadata.num_mip_levels = 1;
    metadata.mip_frame_lengths = vec![256];

    let result = validate_metadata(&metadata);
    assert!(!result.valid, "Should fail with invalid schema version");
    assert!(result
        .detailed_errors
        .iter()
        .any(|e| e.code == wavetable_io::validation::ValidationErrorCode::InvalidSchemaVersion));

    // Test 2: Mismatched mip_frame_lengths count
    let mut metadata = proto::WavetableMetadata::default();
    metadata.schema_version = 1;
    metadata.frame_length = 256;
    metadata.num_frames = 64;
    metadata.num_mip_levels = 3;
    metadata.mip_frame_lengths = vec![256, 128]; // Only 2 entries!

    let result = validate_metadata(&metadata);
    assert!(!result.valid, "Should fail with mismatched mip count");
    assert!(result
        .detailed_errors
        .iter()
        .any(|e| e.code
            == wavetable_io::validation::ValidationErrorCode::MipFrameLengthsCountMismatch));

    // Test 3: mip_frame_lengths[0] != frame_length
    let mut metadata = proto::WavetableMetadata::default();
    metadata.schema_version = 1;
    metadata.frame_length = 256;
    metadata.num_frames = 64;
    metadata.num_mip_levels = 2;
    metadata.mip_frame_lengths = vec![512, 256]; // First entry wrong!

    let result = validate_metadata(&metadata);
    assert!(
        !result.valid,
        "Should fail with mip_frame_lengths[0] mismatch"
    );
    assert!(result
        .detailed_errors
        .iter()
        .any(|e| e.code
            == wavetable_io::validation::ValidationErrorCode::MipFrameLengthsFirstMismatch));

    // Test 4: mip_frame_lengths not decreasing
    let mut metadata = proto::WavetableMetadata::default();
    metadata.schema_version = 1;
    metadata.frame_length = 256;
    metadata.num_frames = 64;
    metadata.num_mip_levels = 3;
    metadata.mip_frame_lengths = vec![256, 128, 256]; // Not decreasing!

    let result = validate_metadata(&metadata);
    assert!(!result.valid, "Should fail with non-decreasing mip lengths");
    assert!(result
        .detailed_errors
        .iter()
        .any(|e| e.code
            == wavetable_io::validation::ValidationErrorCode::MipFrameLengthsNotDecreasing));
}

/// Test that all wavetable types can be manually created and validated.
#[test]
fn test_third_party_all_wavetable_types() {
    let types = vec![
        (proto::WavetableType::Unspecified, "UNSPECIFIED"),
        (proto::WavetableType::ClassicDigital, "CLASSIC_DIGITAL"),
        (proto::WavetableType::HighResolution, "HIGH_RESOLUTION"),
        (proto::WavetableType::VintageEmulation, "VINTAGE_EMULATION"),
        (proto::WavetableType::PcmSample, "PCM_SAMPLE"),
        (proto::WavetableType::Custom, "CUSTOM"),
    ];

    for (wt_type, type_name) in types {
        let wav_data = build_manual_wavetable(128, 16, &[128, 64, 32], wt_type, 44100);

        let wavetable = read_wavetable_from_bytes(&wav_data)
            .unwrap_or_else(|_| panic!("Should read {} type", type_name));

        // UNSPECIFIED gets treated as Custom internally in some contexts,
        // but the raw value should be preserved
        assert_eq!(
            wavetable.metadata.wavetable_type(),
            wt_type,
            "Type mismatch for {}",
            type_name
        );

        // All types should pass validation
        let result = validate_wavetable(&wavetable);
        assert!(
            result.is_ok(),
            "Validation should pass for {}: {:?}",
            type_name,
            result
        );
    }
}

/// Test that the format is backward compatible (unknown fields are preserved).
#[test]
fn test_third_party_forward_compatibility() {
    // Create a wavetable with standard fields
    let metadata = proto::WavetableMetadata {
        schema_version: 1,
        wavetable_type: proto::WavetableType::Custom.into(),
        frame_length: 64,
        num_frames: 4,
        num_mip_levels: 1,
        mip_frame_lengths: vec![64],
        name: Some("Forward Compat Test".to_string()),
        ..Default::default()
    };

    let original_encoded = metadata.encode_to_vec();

    // Append unknown field data (simulating a newer schema version)
    // Field 99, wire type 2 (length-delimited)
    let mut modified_encoded = original_encoded;
    modified_encoded.push(0x9A); // Low byte of tag (99 << 3) | 2
    modified_encoded.push(0x06); // High byte
    modified_encoded.push(0x0B); // Length = 11
    modified_encoded.extend_from_slice(b"future_data");

    // Create samples
    let total_samples: u32 = 64 * 4;
    let samples: Vec<f32> = (0..total_samples).map(|_| 0.5f32).collect();

    // Build WAV manually with modified protobuf
    let mut wav_data = Vec::new();

    // RIFF header
    wav_data.extend_from_slice(b"RIFF");
    let riff_size_pos = wav_data.len();
    wav_data.extend_from_slice(&0u32.to_le_bytes());
    wav_data.extend_from_slice(b"WAVE");

    // fmt chunk
    wav_data.extend_from_slice(b"fmt ");
    wav_data.extend_from_slice(&16u32.to_le_bytes());
    wav_data.extend_from_slice(&3u16.to_le_bytes());
    wav_data.extend_from_slice(&1u16.to_le_bytes());
    wav_data.extend_from_slice(&44100u32.to_le_bytes());
    wav_data.extend_from_slice(&(44100u32 * 4).to_le_bytes());
    wav_data.extend_from_slice(&4u16.to_le_bytes());
    wav_data.extend_from_slice(&32u16.to_le_bytes());

    // data chunk
    wav_data.extend_from_slice(b"data");
    wav_data.extend_from_slice(&(total_samples * 4).to_le_bytes());
    for s in &samples {
        wav_data.extend_from_slice(&s.to_le_bytes());
    }

    // WTBL chunk with modified metadata
    wav_data.extend_from_slice(b"WTBL");
    wav_data.extend_from_slice(&(modified_encoded.len() as u32).to_le_bytes());
    wav_data.extend_from_slice(&modified_encoded);
    if modified_encoded.len() % 2 == 1 {
        wav_data.push(0);
    }

    // Update RIFF size
    let riff_size = (wav_data.len() - 8) as u32;
    wav_data[riff_size_pos..riff_size_pos + 4].copy_from_slice(&riff_size.to_le_bytes());

    // Verify it can be read and validated
    let wavetable =
        read_wavetable_from_bytes(&wav_data).expect("Should read wavetable with unknown fields");

    // Known fields should be correct
    assert_eq!(
        wavetable.metadata.name,
        Some("Forward Compat Test".to_string())
    );

    // Should still validate
    let result = validate_wavetable(&wavetable);
    assert!(
        result.is_ok(),
        "Validation should pass with unknown fields: {:?}",
        result
    );
}

/// Test that error messages include helpful guidance.
#[test]
fn test_third_party_error_guidance() {
    use wavetable_io::validation::ValidationErrorCode;

    // Verify all error codes have guidance
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
        let guidance = code.guidance();
        assert!(
            !guidance.is_empty(),
            "Error code {:?} should have guidance",
            code
        );
        assert!(
            guidance.len() > 20,
            "Error code {:?} should have meaningful guidance (got: {})",
            code,
            guidance
        );
    }
}
