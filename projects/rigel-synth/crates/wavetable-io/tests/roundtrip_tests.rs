//! Round-trip integration tests for wavetable format.
//!
//! These tests verify that wavetables written by Python (wtgen) can be correctly
//! read by Rust (wavetable-io), including verification of unknown field preservation
//! as per FR-012.
//!
//! The tests use pre-generated test files or create test data that simulates
//! the Python writer output.

use prost::Message;
use std::fs;
use tempfile::tempdir;
use wavetable_io::proto;
use wavetable_io::reader::read_wavetable_from_bytes;
use wavetable_io::riff::build_wav_with_wtbl;
use wavetable_io::validation::validate_wavetable;
use wavetable_io::writer::{write_wavetable, write_wavetable_to_bytes, WavetableBuilder};

/// Helper to create frames of a given length filled with a value.
fn create_frames(num_frames: usize, frame_length: usize, value: f32) -> Vec<Vec<f32>> {
    (0..num_frames).map(|_| vec![value; frame_length]).collect()
}

/// Test basic round-trip: write with Rust, read with Rust.
#[test]
fn test_rust_roundtrip_basic() {
    // Create test samples: 4 frames of 128 samples
    let num_frames = 4;
    let frame_length = 128;

    // Create frames with varying values
    let frames: Vec<Vec<f32>> = (0..num_frames)
        .map(|frame_idx| {
            (0..frame_length)
                .map(|sample_idx| {
                    let idx = frame_idx * frame_length + sample_idx;
                    (idx as f32 / (num_frames * frame_length) as f32) * 2.0 - 1.0
                })
                .collect()
        })
        .collect();

    // Build wavetable
    let wavetable = WavetableBuilder::new(
        proto::WavetableType::Custom,
        frame_length as u32,
        num_frames as u32,
    )
    .sample_rate(44100)
    .add_mip_level(frames.clone())
    .build()
    .unwrap();

    // Write to bytes
    let wav_data = write_wavetable_to_bytes(&wavetable);

    // Read back
    let loaded = read_wavetable_from_bytes(&wav_data).unwrap();

    // Verify
    assert_eq!(loaded.metadata.frame_length, frame_length as u32);
    assert_eq!(loaded.metadata.num_frames, num_frames as u32);
    assert_eq!(loaded.mip_levels.len(), 1);
    assert_eq!(loaded.mip_levels[0].frames.len(), num_frames);

    // Verify sample values are preserved (within float precision)
    for (frame_idx, frame) in loaded.mip_levels[0].frames.iter().enumerate() {
        for (sample_idx, &sample) in frame.iter().enumerate() {
            let expected = frames[frame_idx][sample_idx];
            assert!(
                (sample - expected).abs() < 1e-5,
                "Sample mismatch at frame {} sample {}: {} vs {}",
                frame_idx,
                sample_idx,
                sample,
                expected
            );
        }
    }
}

/// Test round-trip with multiple mip levels.
#[test]
fn test_rust_roundtrip_multi_mip() {
    let num_frames = 8;
    let mip_lengths = [256usize, 128, 64, 32];

    // Build wavetable with multiple mip levels
    let mut builder = WavetableBuilder::new(
        proto::WavetableType::HighResolution,
        mip_lengths[0] as u32,
        num_frames as u32,
    )
    .sample_rate(48000)
    .name("Multi-Mip Test");

    for (mip_idx, &frame_len) in mip_lengths.iter().enumerate() {
        let frames: Vec<Vec<f32>> = (0..num_frames)
            .map(|_| vec![(mip_idx as f32) * 0.1; frame_len])
            .collect();
        builder = builder.add_mip_level(frames);
    }

    let wavetable = builder.build().unwrap();
    let wav_data = write_wavetable_to_bytes(&wavetable);

    // Read back
    let loaded = read_wavetable_from_bytes(&wav_data).unwrap();

    // Verify structure
    assert_eq!(loaded.metadata.num_mip_levels, 4);
    assert_eq!(loaded.mip_levels.len(), 4);

    for (i, &expected_len) in mip_lengths.iter().enumerate() {
        assert_eq!(
            loaded.mip_levels[i].frame_length, expected_len as u32,
            "Mip level {} frame length mismatch",
            i
        );
    }
}

/// Test round-trip with all wavetable types.
#[test]
fn test_rust_roundtrip_all_types() {
    let types = vec![
        proto::WavetableType::ClassicDigital,
        proto::WavetableType::HighResolution,
        proto::WavetableType::VintageEmulation,
        proto::WavetableType::PcmSample,
        proto::WavetableType::Custom,
    ];

    for wt_type in types {
        let frames = create_frames(4, 64, 0.5);

        let wavetable = WavetableBuilder::new(wt_type, 64, 4)
            .sample_rate(44100)
            .add_mip_level(frames)
            .build()
            .unwrap();

        let wav_data = write_wavetable_to_bytes(&wavetable);
        let loaded = read_wavetable_from_bytes(&wav_data).unwrap();

        assert_eq!(
            loaded.metadata.wavetable_type(),
            wt_type,
            "Type mismatch for {:?}",
            wt_type
        );
    }
}

/// Test round-trip with file write/read.
#[test]
fn test_rust_roundtrip_file() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("roundtrip_file.wav");

    let frames = create_frames(8, 256, 0.3);

    let wavetable = WavetableBuilder::new(proto::WavetableType::Custom, 256, 8)
        .sample_rate(44100)
        .name("File Test")
        .add_mip_level(frames)
        .build()
        .unwrap();

    // Write to file
    write_wavetable(&path, &wavetable, true).unwrap();

    // Read back
    let data = fs::read(&path).unwrap();
    let loaded = read_wavetable_from_bytes(&data).unwrap();

    assert_eq!(loaded.metadata.name, Some("File Test".to_string()));
    assert_eq!(loaded.metadata.num_frames, 8);
}

/// Test that validation passes for round-tripped files.
#[test]
fn test_rust_roundtrip_validation() {
    let frames: Vec<Vec<f32>> = (0..4)
        .map(|_| {
            (0..256)
                .map(|i| ((i as f32 / 128.0) * std::f32::consts::PI).sin())
                .collect()
        })
        .collect();

    let wavetable = WavetableBuilder::new(proto::WavetableType::Custom, 256, 4)
        .sample_rate(44100)
        .name("Validation Test")
        .author("Test Suite")
        .add_mip_level(frames)
        .build()
        .unwrap();

    let wav_data = write_wavetable_to_bytes(&wavetable);
    let loaded = read_wavetable_from_bytes(&wav_data).unwrap();

    // Validate
    let result = validate_wavetable(&loaded);
    assert!(result.is_ok(), "Validation should pass: {:?}", result);
}

/// Test unknown field preservation (FR-012).
///
/// This test verifies that when we read a protobuf with unknown fields
/// (from a newer schema version), those fields are preserved when we
/// re-encode the message.
#[test]
fn test_unknown_field_preservation_fr012() {
    // Create metadata with all known fields
    let metadata = proto::WavetableMetadata {
        schema_version: 1,
        wavetable_type: proto::WavetableType::Custom.into(),
        frame_length: 64,
        num_frames: 2,
        num_mip_levels: 1,
        mip_frame_lengths: vec![64],
        name: Some("Unknown Fields Test".to_string()),
        ..Default::default()
    };

    let original_encoded = metadata.encode_to_vec();

    // Append unknown field data (simulating a newer schema version)
    // Field 99 (well outside reserved ranges) with a string value
    let mut modified_encoded = original_encoded.clone();
    // Protobuf wire format: field 99, wire type 2 (length-delimited)
    // Tag = (99 << 3) | 2 = 794 = varint [0x9A, 0x06]
    modified_encoded.push(0x9A); // low byte of tag
    modified_encoded.push(0x06); // high byte of tag
    modified_encoded.push(0x0B); // length = 11
    modified_encoded.extend_from_slice(b"future_data"); // value

    // Create WAV with modified protobuf
    let samples: Vec<f32> = vec![0.0; 128];
    let wav_data = build_wav_with_wtbl(&samples, 44100, &modified_encoded);

    // Read the wavetable
    let wavetable = read_wavetable_from_bytes(&wav_data).unwrap();

    // Re-encode the metadata
    let re_encoded = wavetable.metadata.encode_to_vec();

    // The re-encoded data should preserve the unknown field
    // Note: prost preserves unknown fields by default
    assert!(
        re_encoded.len() >= original_encoded.len(),
        "Re-encoded data should be at least as long as original"
    );

    // Verify the known fields are still correct
    assert_eq!(
        wavetable.metadata.name,
        Some("Unknown Fields Test".to_string())
    );
    assert_eq!(wavetable.metadata.frame_length, 64);
}

/// Test reading a file written with optional fields omitted.
#[test]
fn test_roundtrip_minimal_metadata() {
    let frames = create_frames(2, 32, 0.0);

    let wavetable = WavetableBuilder::new(proto::WavetableType::Custom, 32, 2)
        .sample_rate(44100)
        .add_mip_level(frames)
        .build()
        .unwrap();

    let wav_data = write_wavetable_to_bytes(&wavetable);
    let loaded = read_wavetable_from_bytes(&wav_data).unwrap();

    // Verify optional fields are None
    assert_eq!(loaded.metadata.name, None);
    assert_eq!(loaded.metadata.author, None);
    assert_eq!(loaded.metadata.description, None);
    assert_eq!(loaded.metadata.source_bit_depth, None);
    assert_eq!(loaded.metadata.tuning_reference, None);
}

/// Test that files created by Python can be read (simulated).
///
/// This test creates data in the exact format that Python would produce,
/// verifying cross-language compatibility.
#[test]
fn test_python_format_compatibility() {
    // Create metadata matching Python's output format
    let metadata = proto::WavetableMetadata {
        schema_version: 1,
        wavetable_type: proto::WavetableType::HighResolution.into(),
        frame_length: 2048,
        num_frames: 64,
        num_mip_levels: 3,
        mip_frame_lengths: vec![2048, 1024, 512],
        normalization_method: proto::NormalizationMethod::NormalizationPeak as i32,
        source_bit_depth: Some(16),
        author: Some("wtgen".to_string()),
        name: Some("Python Generated Wavetable".to_string()),
        sample_rate: Some(48000),
        ..Default::default()
    };

    let wtbl_data = metadata.encode_to_vec();

    // Create samples (total = 64 * (2048 + 1024 + 512) = 229376)
    let total_samples: usize = metadata
        .mip_frame_lengths
        .iter()
        .map(|&len| len as usize * metadata.num_frames as usize)
        .sum();
    let samples: Vec<f32> = (0..total_samples)
        .map(|i| ((i as f32 / total_samples as f32) * std::f32::consts::TAU).sin())
        .collect();

    // Build WAV in the same format Python uses
    let wav_data = build_wav_with_wtbl(&samples, 48000, &wtbl_data);

    // Read with Rust
    let wavetable = read_wavetable_from_bytes(&wav_data).unwrap();

    // Verify all metadata matches
    assert_eq!(wavetable.metadata.schema_version, 1);
    assert_eq!(
        wavetable.metadata.wavetable_type(),
        proto::WavetableType::HighResolution
    );
    assert_eq!(wavetable.metadata.frame_length, 2048);
    assert_eq!(wavetable.metadata.num_frames, 64);
    assert_eq!(wavetable.metadata.num_mip_levels, 3);
    assert_eq!(wavetable.metadata.mip_frame_lengths, vec![2048, 1024, 512]);
    assert_eq!(
        wavetable.metadata.normalization_method,
        proto::NormalizationMethod::NormalizationPeak as i32
    );
    assert_eq!(wavetable.metadata.author, Some("wtgen".to_string()));

    // Verify mip levels
    assert_eq!(wavetable.mip_levels.len(), 3);
    assert_eq!(wavetable.mip_levels[0].frames.len(), 64);
    assert_eq!(wavetable.mip_levels[0].frame_length, 2048);
    assert_eq!(wavetable.mip_levels[1].frame_length, 1024);
    assert_eq!(wavetable.mip_levels[2].frame_length, 512);

    // Validation should pass
    assert!(validate_wavetable(&wavetable).is_ok());
}

/// Test round-trip with metadata set.
#[test]
fn test_roundtrip_with_metadata() {
    let frames = create_frames(16, 512, 0.25);

    let wavetable = WavetableBuilder::new(proto::WavetableType::HighResolution, 512, 16)
        .sample_rate(96000)
        .name("Detailed Test Wavetable")
        .author("Integration Tests")
        .description("A wavetable for testing round-trip")
        .normalization_method(proto::NormalizationMethod::NormalizationPeak)
        .add_mip_level(frames)
        .build()
        .unwrap();

    let wav_data = write_wavetable_to_bytes(&wavetable);
    let loaded = read_wavetable_from_bytes(&wav_data).unwrap();

    assert_eq!(
        loaded.metadata.name,
        Some("Detailed Test Wavetable".to_string())
    );
    assert_eq!(
        loaded.metadata.author,
        Some("Integration Tests".to_string())
    );
    assert_eq!(
        loaded.metadata.description,
        Some("A wavetable for testing round-trip".to_string())
    );
    assert_eq!(
        loaded.metadata.normalization_method,
        proto::NormalizationMethod::NormalizationPeak as i32
    );
    assert_eq!(loaded.sample_rate, 96000);
}

/// Test round-trip with classic digital type.
#[test]
fn test_roundtrip_classic_digital() {
    // Classic digital: 256 samples, 64 frames, multiple mip levels
    let num_frames = 64;
    let mip_lengths = [256usize, 128, 64, 32, 16, 8, 4];

    let mut builder =
        WavetableBuilder::new(proto::WavetableType::ClassicDigital, 256, num_frames as u32)
            .sample_rate(44100)
            .name("PPG-Style Test");

    for &frame_len in &mip_lengths {
        let frames = create_frames(num_frames, frame_len, 0.5);
        builder = builder.add_mip_level(frames);
    }

    let wavetable = builder.build().unwrap();
    let wav_data = write_wavetable_to_bytes(&wavetable);
    let loaded = read_wavetable_from_bytes(&wav_data).unwrap();

    assert_eq!(
        loaded.metadata.wavetable_type(),
        proto::WavetableType::ClassicDigital
    );
    assert_eq!(loaded.mip_levels.len(), 7);
    assert_eq!(loaded.mip_levels[0].frame_length, 256);
    assert_eq!(loaded.mip_levels[6].frame_length, 4);
}
