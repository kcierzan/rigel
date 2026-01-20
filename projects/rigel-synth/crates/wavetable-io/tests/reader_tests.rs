//! Unit tests for wavetable reader module.
//!
//! These tests verify the reader can correctly load wavetable files
//! with various configurations and edge cases.

use prost::Message;
use wavetable_io::proto;
use wavetable_io::reader::read_wavetable_from_bytes;
use wavetable_io::riff::build_wav_with_wtbl;
use wavetable_io::validation::validate_wavetable;

/// Helper to create a basic metadata with common defaults.
fn create_metadata(
    wavetable_type: proto::WavetableType,
    frame_length: u32,
    num_frames: u32,
    mip_frame_lengths: Vec<u32>,
) -> proto::WavetableMetadata {
    proto::WavetableMetadata {
        schema_version: 1,
        wavetable_type: wavetable_type.into(),
        frame_length,
        num_frames,
        num_mip_levels: mip_frame_lengths.len() as u32,
        mip_frame_lengths,
        ..Default::default()
    }
}

/// Helper to create samples for given metadata.
fn create_samples(metadata: &proto::WavetableMetadata, value: f32) -> Vec<f32> {
    let total: usize = metadata
        .mip_frame_lengths
        .iter()
        .map(|&len| len as usize * metadata.num_frames as usize)
        .sum();
    vec![value; total]
}

#[test]
fn test_read_classic_digital_wavetable() {
    let metadata = create_metadata(
        proto::WavetableType::ClassicDigital,
        256,
        64,
        vec![256, 128, 64, 32],
    );
    let wtbl_data = metadata.encode_to_vec();
    let samples = create_samples(&metadata, 0.5);
    let wav_data = build_wav_with_wtbl(&samples, 44100, &wtbl_data);

    let wavetable = read_wavetable_from_bytes(&wav_data).unwrap();

    assert_eq!(
        wavetable.metadata.wavetable_type(),
        proto::WavetableType::ClassicDigital
    );
    assert_eq!(wavetable.metadata.frame_length, 256);
    assert_eq!(wavetable.metadata.num_frames, 64);
    assert_eq!(wavetable.mip_levels.len(), 4);
}

#[test]
fn test_read_high_resolution_wavetable() {
    let metadata = create_metadata(
        proto::WavetableType::HighResolution,
        2048,
        128,
        vec![2048, 1024, 512],
    );
    let wtbl_data = metadata.encode_to_vec();
    let samples = create_samples(&metadata, 0.3);
    let wav_data = build_wav_with_wtbl(&samples, 48000, &wtbl_data);

    let wavetable = read_wavetable_from_bytes(&wav_data).unwrap();

    assert_eq!(
        wavetable.metadata.wavetable_type(),
        proto::WavetableType::HighResolution
    );
    assert_eq!(wavetable.metadata.frame_length, 2048);
    assert_eq!(wavetable.sample_rate, 48000);
}

#[test]
fn test_read_vintage_emulation_wavetable() {
    let metadata = create_metadata(
        proto::WavetableType::VintageEmulation,
        512,
        8,
        vec![512, 256],
    );
    let wtbl_data = metadata.encode_to_vec();
    let samples = create_samples(&metadata, 0.7);
    let wav_data = build_wav_with_wtbl(&samples, 44100, &wtbl_data);

    let wavetable = read_wavetable_from_bytes(&wav_data).unwrap();

    assert_eq!(
        wavetable.metadata.wavetable_type(),
        proto::WavetableType::VintageEmulation
    );
    assert_eq!(wavetable.metadata.num_frames, 8);
}

#[test]
fn test_read_pcm_sample_wavetable() {
    let metadata = create_metadata(proto::WavetableType::PcmSample, 1024, 1, vec![1024]);
    let wtbl_data = metadata.encode_to_vec();
    let samples = create_samples(&metadata, 0.8);
    let wav_data = build_wav_with_wtbl(&samples, 22050, &wtbl_data);

    let wavetable = read_wavetable_from_bytes(&wav_data).unwrap();

    assert_eq!(
        wavetable.metadata.wavetable_type(),
        proto::WavetableType::PcmSample
    );
    assert_eq!(wavetable.metadata.num_mip_levels, 1);
}

#[test]
fn test_read_custom_wavetable() {
    let metadata = create_metadata(proto::WavetableType::Custom, 64, 16, vec![64, 32, 16]);
    let wtbl_data = metadata.encode_to_vec();
    let samples = create_samples(&metadata, -0.5);
    let wav_data = build_wav_with_wtbl(&samples, 96000, &wtbl_data);

    let wavetable = read_wavetable_from_bytes(&wav_data).unwrap();

    assert_eq!(
        wavetable.metadata.wavetable_type(),
        proto::WavetableType::Custom
    );
}

#[test]
fn test_read_with_optional_metadata() {
    let mut metadata = create_metadata(proto::WavetableType::Custom, 128, 4, vec![128]);
    metadata.name = Some("Test Wavetable".to_string());
    metadata.author = Some("Test Author".to_string());
    metadata.description = Some("A wavetable for testing".to_string());
    metadata.source_bit_depth = Some(16);
    metadata.tuning_reference = Some(440.0);
    metadata.sample_rate = Some(44100);

    let wtbl_data = metadata.encode_to_vec();
    let samples = create_samples(&metadata, 0.1);
    let wav_data = build_wav_with_wtbl(&samples, 44100, &wtbl_data);

    let wavetable = read_wavetable_from_bytes(&wav_data).unwrap();

    assert_eq!(wavetable.metadata.name, Some("Test Wavetable".to_string()));
    assert_eq!(wavetable.metadata.author, Some("Test Author".to_string()));
    assert_eq!(
        wavetable.metadata.description,
        Some("A wavetable for testing".to_string())
    );
    assert_eq!(wavetable.metadata.source_bit_depth, Some(16));
    assert_eq!(wavetable.metadata.tuning_reference, Some(440.0));
}

#[test]
fn test_read_with_type_specific_metadata_classic() {
    let mut metadata = create_metadata(proto::WavetableType::ClassicDigital, 256, 64, vec![256]);

    let classic_meta = proto::ClassicDigitalMetadata {
        original_bit_depth: Some(8),
        original_sample_rate: Some(31250),
        source_hardware: Some("PPG Wave 2.2".to_string()),
        harmonic_caps: vec![128, 64, 32],
    };
    metadata.type_metadata = Some(proto::wavetable_metadata::TypeMetadata::ClassicDigital(
        classic_meta,
    ));

    let wtbl_data = metadata.encode_to_vec();
    let samples = create_samples(&metadata, 0.25);
    let wav_data = build_wav_with_wtbl(&samples, 44100, &wtbl_data);

    let wavetable = read_wavetable_from_bytes(&wav_data).unwrap();

    match &wavetable.metadata.type_metadata {
        Some(proto::wavetable_metadata::TypeMetadata::ClassicDigital(m)) => {
            assert_eq!(m.original_bit_depth, Some(8));
            assert_eq!(m.source_hardware, Some("PPG Wave 2.2".to_string()));
        }
        _ => panic!("Expected ClassicDigitalMetadata"),
    }
}

#[test]
fn test_read_with_type_specific_metadata_high_resolution() {
    let mut metadata = create_metadata(proto::WavetableType::HighResolution, 2048, 64, vec![2048]);

    let hires_meta = proto::HighResolutionMetadata {
        max_harmonics: Some(1024),
        interpolation_hint: proto::InterpolationHint::InterpolationCubic as i32,
        source_synth: Some("AN1x".to_string()),
    };
    metadata.type_metadata = Some(proto::wavetable_metadata::TypeMetadata::HighResolution(
        hires_meta,
    ));

    let wtbl_data = metadata.encode_to_vec();
    let samples = create_samples(&metadata, 0.6);
    let wav_data = build_wav_with_wtbl(&samples, 48000, &wtbl_data);

    let wavetable = read_wavetable_from_bytes(&wav_data).unwrap();

    match &wavetable.metadata.type_metadata {
        Some(proto::wavetable_metadata::TypeMetadata::HighResolution(m)) => {
            assert_eq!(m.max_harmonics, Some(1024));
            assert_eq!(m.source_synth, Some("AN1x".to_string()));
        }
        _ => panic!("Expected HighResolutionMetadata"),
    }
}

#[test]
fn test_validation_on_read() {
    // Create a valid wavetable and verify it passes validation
    let metadata = create_metadata(proto::WavetableType::Custom, 256, 8, vec![256, 128]);
    let wtbl_data = metadata.encode_to_vec();
    let samples = create_samples(&metadata, 0.4);
    let wav_data = build_wav_with_wtbl(&samples, 44100, &wtbl_data);

    let wavetable = read_wavetable_from_bytes(&wav_data).unwrap();
    let result = validate_wavetable(&wavetable);

    assert!(result.is_ok(), "Validation should pass: {:?}", result);
}

#[test]
fn test_sample_values_preserved() {
    let metadata = create_metadata(proto::WavetableType::Custom, 4, 2, vec![4]);
    let wtbl_data = metadata.encode_to_vec();

    // Use specific sample values to verify precision
    let samples = vec![0.0, 0.25, 0.5, 0.75, -0.25, -0.5, -0.75, -1.0];
    let wav_data = build_wav_with_wtbl(&samples, 44100, &wtbl_data);

    let wavetable = read_wavetable_from_bytes(&wav_data).unwrap();

    // Verify frame 0
    assert_eq!(wavetable.mip_levels[0].frames[0].len(), 4);
    assert!((wavetable.mip_levels[0].frames[0][0] - 0.0).abs() < 1e-6);
    assert!((wavetable.mip_levels[0].frames[0][1] - 0.25).abs() < 1e-6);
    assert!((wavetable.mip_levels[0].frames[0][2] - 0.5).abs() < 1e-6);
    assert!((wavetable.mip_levels[0].frames[0][3] - 0.75).abs() < 1e-6);

    // Verify frame 1
    assert!((wavetable.mip_levels[0].frames[1][0] - (-0.25)).abs() < 1e-6);
    assert!((wavetable.mip_levels[0].frames[1][1] - (-0.5)).abs() < 1e-6);
    assert!((wavetable.mip_levels[0].frames[1][2] - (-0.75)).abs() < 1e-6);
    assert!((wavetable.mip_levels[0].frames[1][3] - (-1.0)).abs() < 1e-6);
}

#[test]
fn test_read_error_missing_wtbl() {
    // Create a minimal valid WAV file without WTBL chunk
    // RIFF header (12 bytes) + fmt chunk (24 bytes) + data chunk (minimal)
    let mut wav_data = Vec::new();

    // RIFF header
    wav_data.extend_from_slice(b"RIFF");
    wav_data.extend_from_slice(&36u32.to_le_bytes()); // file size - 8
    wav_data.extend_from_slice(b"WAVE");

    // fmt chunk (PCM format)
    wav_data.extend_from_slice(b"fmt ");
    wav_data.extend_from_slice(&16u32.to_le_bytes()); // chunk size
    wav_data.extend_from_slice(&3u16.to_le_bytes()); // format (IEEE float)
    wav_data.extend_from_slice(&1u16.to_le_bytes()); // channels
    wav_data.extend_from_slice(&44100u32.to_le_bytes()); // sample rate
    wav_data.extend_from_slice(&176400u32.to_le_bytes()); // byte rate
    wav_data.extend_from_slice(&4u16.to_le_bytes()); // block align
    wav_data.extend_from_slice(&32u16.to_le_bytes()); // bits per sample

    // Empty data chunk
    wav_data.extend_from_slice(b"data");
    wav_data.extend_from_slice(&0u32.to_le_bytes());

    let result = read_wavetable_from_bytes(&wav_data);
    assert!(result.is_err());
}

#[test]
fn test_read_error_invalid_file() {
    let invalid_data = b"not a valid wav file";
    let result = read_wavetable_from_bytes(invalid_data);
    assert!(result.is_err());
}

// ============================================================================
// Edge Case Tests
// ============================================================================

#[test]
fn test_edge_case_single_sample_frame() {
    // Minimum valid frame size: 1 sample per frame
    let metadata = create_metadata(proto::WavetableType::Custom, 1, 4, vec![1]);
    let wtbl_data = metadata.encode_to_vec();

    // 4 frames of 1 sample each
    let samples = vec![0.0f32, 0.5, -0.5, 1.0];
    let wav_data = build_wav_with_wtbl(&samples, 44100, &wtbl_data);

    let wavetable = read_wavetable_from_bytes(&wav_data).unwrap();

    assert_eq!(wavetable.metadata.frame_length, 1);
    assert_eq!(wavetable.mip_levels[0].frames.len(), 4);
    assert_eq!(wavetable.mip_levels[0].frames[0].len(), 1);
    assert!((wavetable.mip_levels[0].frames[0][0] - 0.0).abs() < 1e-6);
    assert!((wavetable.mip_levels[0].frames[1][0] - 0.5).abs() < 1e-6);
}

#[test]
fn test_edge_case_single_frame_wavetable() {
    // Minimum valid: single frame with multiple samples
    let metadata = create_metadata(proto::WavetableType::Custom, 16, 1, vec![16]);
    let wtbl_data = metadata.encode_to_vec();
    let samples = create_samples(&metadata, 0.25);
    let wav_data = build_wav_with_wtbl(&samples, 44100, &wtbl_data);

    let wavetable = read_wavetable_from_bytes(&wav_data).unwrap();

    assert_eq!(wavetable.metadata.num_frames, 1);
    assert_eq!(wavetable.mip_levels[0].frames.len(), 1);
}

#[test]
fn test_edge_case_minimal_wavetable() {
    // Absolute minimum: 1 frame of 1 sample
    let metadata = create_metadata(proto::WavetableType::Custom, 1, 1, vec![1]);
    let wtbl_data = metadata.encode_to_vec();
    let samples = vec![0.42f32];
    let wav_data = build_wav_with_wtbl(&samples, 44100, &wtbl_data);

    let wavetable = read_wavetable_from_bytes(&wav_data).unwrap();

    assert_eq!(wavetable.metadata.frame_length, 1);
    assert_eq!(wavetable.metadata.num_frames, 1);
    assert_eq!(wavetable.mip_levels.len(), 1);
    assert!((wavetable.mip_levels[0].frames[0][0] - 0.42).abs() < 1e-6);
}

#[test]
fn test_edge_case_many_mip_levels() {
    // Test with many mip levels (8 levels)
    let metadata = create_metadata(
        proto::WavetableType::HighResolution,
        2048,
        8,
        vec![2048, 1024, 512, 256, 128, 64, 32, 16],
    );
    let wtbl_data = metadata.encode_to_vec();
    let samples = create_samples(&metadata, 0.1);
    let wav_data = build_wav_with_wtbl(&samples, 48000, &wtbl_data);

    let wavetable = read_wavetable_from_bytes(&wav_data).unwrap();

    assert_eq!(wavetable.mip_levels.len(), 8);
    assert_eq!(wavetable.mip_levels[0].frame_length, 2048);
    assert_eq!(wavetable.mip_levels[7].frame_length, 16);
}

#[test]
fn test_edge_case_two_sample_frame() {
    // Two samples per frame - smallest power of 2 greater than 1
    let metadata = create_metadata(proto::WavetableType::Custom, 2, 2, vec![2]);
    let wtbl_data = metadata.encode_to_vec();
    let samples = vec![0.5f32, -0.5, 0.25, -0.25];
    let wav_data = build_wav_with_wtbl(&samples, 44100, &wtbl_data);

    let wavetable = read_wavetable_from_bytes(&wav_data).unwrap();

    assert_eq!(wavetable.mip_levels[0].frames[0], vec![0.5, -0.5]);
    assert_eq!(wavetable.mip_levels[0].frames[1], vec![0.25, -0.25]);
}

#[test]
fn test_edge_case_boundary_sample_values() {
    // Test with boundary float values (not NaN/Inf but at edges)
    let metadata = create_metadata(proto::WavetableType::Custom, 4, 1, vec![4]);
    let wtbl_data = metadata.encode_to_vec();

    // Use extreme but valid values
    let samples = vec![1.0f32, -1.0, 0.0, f32::MIN_POSITIVE];
    let wav_data = build_wav_with_wtbl(&samples, 44100, &wtbl_data);

    let wavetable = read_wavetable_from_bytes(&wav_data).unwrap();

    assert!((wavetable.mip_levels[0].frames[0][0] - 1.0).abs() < 1e-6);
    assert!((wavetable.mip_levels[0].frames[0][1] - (-1.0)).abs() < 1e-6);
    assert!((wavetable.mip_levels[0].frames[0][2] - 0.0).abs() < 1e-6);
}

#[test]
fn test_edge_case_large_frame_count() {
    // Test with many frames (256 frames)
    let metadata = create_metadata(proto::WavetableType::Custom, 4, 256, vec![4]);
    let wtbl_data = metadata.encode_to_vec();
    let samples = create_samples(&metadata, 0.5);
    let wav_data = build_wav_with_wtbl(&samples, 44100, &wtbl_data);

    let wavetable = read_wavetable_from_bytes(&wav_data).unwrap();

    assert_eq!(wavetable.mip_levels[0].frames.len(), 256);
}

// ============================================================================
// RIFF Parser Edge Case Tests (for fuzz coverage)
// ============================================================================

#[test]
fn test_riff_parser_truncated_header() {
    // Less than 12 bytes - should fail gracefully
    let data = b"RIFF\x04\x00";
    let result = read_wavetable_from_bytes(data);
    assert!(result.is_err());
}

#[test]
fn test_riff_parser_invalid_magic() {
    // Valid size but wrong magic
    let mut data = vec![0u8; 44];
    data[0..4].copy_from_slice(b"XXXX"); // Not RIFF
    let result = read_wavetable_from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_riff_parser_not_wave_format() {
    // RIFF but not WAVE format
    let mut data = vec![0u8; 44];
    data[0..4].copy_from_slice(b"RIFF");
    data[4..8].copy_from_slice(&36u32.to_le_bytes());
    data[8..12].copy_from_slice(b"AVI "); // Not WAVE
    let result = read_wavetable_from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_riff_parser_huge_declared_size() {
    // RIFF header with size claiming file is huge (overflow test)
    let mut data = vec![0u8; 44];
    data[0..4].copy_from_slice(b"RIFF");
    data[4..8].copy_from_slice(&u32::MAX.to_le_bytes()); // Huge size
    data[8..12].copy_from_slice(b"WAVE");
    // Should not panic due to our saturating_add fix
    let result = read_wavetable_from_bytes(&data);
    assert!(result.is_err()); // Will error but shouldn't panic
}

#[test]
fn test_riff_parser_chunk_size_overflow() {
    // Chunk with size that would overflow when added to offset
    let mut data = vec![0u8; 100];
    data[0..4].copy_from_slice(b"RIFF");
    data[4..8].copy_from_slice(&92u32.to_le_bytes());
    data[8..12].copy_from_slice(b"WAVE");
    // fmt chunk
    data[12..16].copy_from_slice(b"fmt ");
    data[16..20].copy_from_slice(&u32::MAX.to_le_bytes()); // Overflow-inducing chunk size
    // Should not panic due to our checked_add fix
    let result = read_wavetable_from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_riff_parser_empty_after_header() {
    // Just the RIFF/WAVE header with no chunks
    let mut data = vec![0u8; 12];
    data[0..4].copy_from_slice(b"RIFF");
    data[4..8].copy_from_slice(&4u32.to_le_bytes()); // Just WAVE
    data[8..12].copy_from_slice(b"WAVE");
    let result = read_wavetable_from_bytes(&data);
    assert!(result.is_err()); // No WTBL chunk
}
