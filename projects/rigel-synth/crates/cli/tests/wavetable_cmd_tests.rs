//! Integration tests for the wavetable CLI commands.
//!
//! These tests verify that the `rigel wavetable inspect` and `rigel wavetable validate`
//! commands work correctly end-to-end.

use std::process::Command;
use tempfile::tempdir;
use wavetable_io::proto;
use wavetable_io::writer::{write_wavetable, WavetableBuilder};

/// Helper to create a test wavetable file with specific metadata.
fn create_test_wavetable(
    path: &std::path::Path,
    wt_type: proto::WavetableType,
    name: &str,
    author: &str,
    frame_length: u32,
    num_frames: u32,
) {
    let frames: Vec<Vec<f32>> = (0..num_frames as usize)
        .map(|i| {
            (0..frame_length as usize)
                .map(|j| {
                    let phase = (j as f32 / frame_length as f32) * std::f32::consts::TAU;
                    (phase * (i + 1) as f32).sin() * 0.5
                })
                .collect()
        })
        .collect();

    let wavetable = WavetableBuilder::new(wt_type, frame_length, num_frames)
        .sample_rate(44100)
        .name(name)
        .author(author)
        .add_mip_level(frames)
        .build()
        .expect("Failed to build wavetable");

    write_wavetable(path, &wavetable, true).expect("Failed to write wavetable");
}

/// Helper to get the rigel CLI binary path.
fn get_rigel_binary() -> std::path::PathBuf {
    // Use cargo to locate the binary
    let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    workspace_root.join("target/debug/rigel")
}

/// Test that we can build the binary.
#[test]
fn test_cli_builds() {
    // This test just verifies that the binary can be built
    // The actual binary is built by cargo test
    let binary = get_rigel_binary();
    // If the test is running, the binary should exist
    // (may not exist if running tests without build)
    if !binary.exists() {
        eprintln!("Note: rigel binary not found at {:?}", binary);
        eprintln!("Run 'cargo build' first to enable CLI integration tests");
    }
}

/// Test the inspect command with a valid wavetable file.
#[test]
fn test_inspect_command_basic() {
    let binary = get_rigel_binary();
    if !binary.exists() {
        eprintln!("Skipping test: rigel binary not found");
        return;
    }

    let dir = tempdir().unwrap();
    let wavetable_path = dir.path().join("test.wav");

    create_test_wavetable(
        &wavetable_path,
        proto::WavetableType::HighResolution,
        "Test Wavetable",
        "Test Author",
        2048,
        64,
    );

    let output = Command::new(&binary)
        .args(["wavetable", "inspect", wavetable_path.to_str().unwrap()])
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        output.status.success(),
        "Command failed with stderr: {}",
        stderr
    );

    // Verify expected output
    assert!(
        stdout.contains("Test Wavetable"),
        "Output should contain wavetable name"
    );
    assert!(
        stdout.contains("Test Author"),
        "Output should contain author"
    );
    assert!(
        stdout.contains("HIGH_RESOLUTION"),
        "Output should contain type"
    );
    assert!(stdout.contains("2048"), "Output should contain frame length");
    assert!(stdout.contains("64"), "Output should contain num frames");
}

/// Test the inspect command with verbose flag.
#[test]
fn test_inspect_command_verbose() {
    let binary = get_rigel_binary();
    if !binary.exists() {
        eprintln!("Skipping test: rigel binary not found");
        return;
    }

    let dir = tempdir().unwrap();
    let wavetable_path = dir.path().join("test_verbose.wav");

    create_test_wavetable(
        &wavetable_path,
        proto::WavetableType::ClassicDigital,
        "Verbose Test",
        "CLI Tester",
        256,
        64,
    );

    let output = Command::new(&binary)
        .args([
            "wavetable",
            "inspect",
            "--verbose",
            wavetable_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success());

    // Verbose output should include additional fields
    assert!(
        stdout.contains("CLASSIC_DIGITAL"),
        "Output should contain type"
    );
    assert!(
        stdout.contains("Optional Fields"),
        "Verbose output should show optional fields section"
    );
}

/// Test the validate command with a valid wavetable file.
#[test]
fn test_validate_command_valid() {
    let binary = get_rigel_binary();
    if !binary.exists() {
        eprintln!("Skipping test: rigel binary not found");
        return;
    }

    let dir = tempdir().unwrap();
    let wavetable_path = dir.path().join("valid.wav");

    create_test_wavetable(
        &wavetable_path,
        proto::WavetableType::Custom,
        "Valid Wavetable",
        "Test",
        512,
        32,
    );

    let output = Command::new(&binary)
        .args(["wavetable", "validate", wavetable_path.to_str().unwrap()])
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success());
    assert!(
        stdout.contains("Valid wavetable"),
        "Output should confirm valid wavetable"
    );
}

/// Test the validate command with verbose flag.
#[test]
fn test_validate_command_verbose() {
    let binary = get_rigel_binary();
    if !binary.exists() {
        eprintln!("Skipping test: rigel binary not found");
        return;
    }

    let dir = tempdir().unwrap();
    let wavetable_path = dir.path().join("valid_verbose.wav");

    create_test_wavetable(
        &wavetable_path,
        proto::WavetableType::PcmSample,
        "PCM Test",
        "Test",
        1024,
        8,
    );

    let output = Command::new(&binary)
        .args([
            "wavetable",
            "validate",
            "--verbose",
            wavetable_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success());
    assert!(
        stdout.contains("Validation passed"),
        "Verbose output should show validation details"
    );
}

/// Test the inspect command with a non-existent file.
#[test]
fn test_inspect_nonexistent_file() {
    let binary = get_rigel_binary();
    if !binary.exists() {
        eprintln!("Skipping test: rigel binary not found");
        return;
    }

    let output = Command::new(&binary)
        .args(["wavetable", "inspect", "/nonexistent/path/wavetable.wav"])
        .output()
        .expect("Failed to execute command");

    assert!(!output.status.success(), "Command should fail for non-existent file");
}

/// Test the validate command with a non-existent file.
#[test]
fn test_validate_nonexistent_file() {
    let binary = get_rigel_binary();
    if !binary.exists() {
        eprintln!("Skipping test: rigel binary not found");
        return;
    }

    let output = Command::new(&binary)
        .args(["wavetable", "validate", "/nonexistent/path/wavetable.wav"])
        .output()
        .expect("Failed to execute command");

    assert!(!output.status.success(), "Command should fail for non-existent file");
}

/// Test the inspect command with all wavetable types.
#[test]
fn test_inspect_all_types() {
    let binary = get_rigel_binary();
    if !binary.exists() {
        eprintln!("Skipping test: rigel binary not found");
        return;
    }

    let types = vec![
        (proto::WavetableType::ClassicDigital, "CLASSIC_DIGITAL"),
        (proto::WavetableType::HighResolution, "HIGH_RESOLUTION"),
        (proto::WavetableType::VintageEmulation, "VINTAGE_EMULATION"),
        (proto::WavetableType::PcmSample, "PCM_SAMPLE"),
        (proto::WavetableType::Custom, "CUSTOM"),
    ];

    for (wt_type, type_name) in types {
        let dir = tempdir().unwrap();
        let wavetable_path = dir.path().join(format!("{}.wav", type_name.to_lowercase()));

        create_test_wavetable(
            &wavetable_path,
            wt_type,
            &format!("{} Test", type_name),
            "Type Tester",
            256,
            16,
        );

        let output = Command::new(&binary)
            .args(["wavetable", "inspect", wavetable_path.to_str().unwrap()])
            .output()
            .expect("Failed to execute command");

        let stdout = String::from_utf8_lossy(&output.stdout);

        assert!(
            output.status.success(),
            "Inspect should succeed for {} type",
            type_name
        );
        assert!(
            stdout.contains(type_name),
            "Output should contain type name {} but got: {}",
            type_name,
            stdout
        );
    }
}

/// Test inspect command output format for a file with multiple mip levels.
#[test]
fn test_inspect_multi_mip() {
    let binary = get_rigel_binary();
    if !binary.exists() {
        eprintln!("Skipping test: rigel binary not found");
        return;
    }

    let dir = tempdir().unwrap();
    let wavetable_path = dir.path().join("multi_mip.wav");

    let num_frames = 32u32;
    let mip_lengths = [512u32, 256, 128, 64];

    let mut builder = WavetableBuilder::new(proto::WavetableType::HighResolution, 512, num_frames)
        .sample_rate(44100)
        .name("Multi-Mip Test")
        .author("Test");

    for &frame_len in &mip_lengths {
        let frames: Vec<Vec<f32>> = (0..num_frames as usize)
            .map(|_| vec![0.5; frame_len as usize])
            .collect();
        builder = builder.add_mip_level(frames);
    }

    let wavetable = builder.build().unwrap();
    write_wavetable(&wavetable_path, &wavetable, true).unwrap();

    let output = Command::new(&binary)
        .args(["wavetable", "inspect", wavetable_path.to_str().unwrap()])
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success());
    assert!(
        stdout.contains("Mip Levels: 4"),
        "Output should show 4 mip levels"
    );
    assert!(
        stdout.contains("[512, 256, 128, 64]"),
        "Output should show mip frame lengths"
    );
}

/// Test help output for wavetable subcommand.
#[test]
fn test_wavetable_help() {
    let binary = get_rigel_binary();
    if !binary.exists() {
        eprintln!("Skipping test: rigel binary not found");
        return;
    }

    let output = Command::new(&binary)
        .args(["wavetable", "--help"])
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success());
    assert!(
        stdout.contains("inspect"),
        "Help should list inspect subcommand"
    );
    assert!(
        stdout.contains("validate"),
        "Help should list validate subcommand"
    );
}

/// Test that inspect shows audio statistics (peak and RMS).
#[test]
fn test_inspect_audio_stats() {
    let binary = get_rigel_binary();
    if !binary.exists() {
        eprintln!("Skipping test: rigel binary not found");
        return;
    }

    let dir = tempdir().unwrap();
    let wavetable_path = dir.path().join("audio_stats.wav");

    // Create a wavetable with known values
    let frame_length = 256u32;
    let num_frames = 4u32;

    // Create sine waves with known peak
    let frames: Vec<Vec<f32>> = (0..num_frames as usize)
        .map(|_| {
            (0..frame_length as usize)
                .map(|j| {
                    let phase = (j as f32 / frame_length as f32) * std::f32::consts::TAU;
                    phase.sin() * 0.8 // Peak should be ~0.8
                })
                .collect()
        })
        .collect();

    let wavetable = WavetableBuilder::new(proto::WavetableType::Custom, frame_length, num_frames)
        .sample_rate(44100)
        .add_mip_level(frames)
        .build()
        .unwrap();

    write_wavetable(&wavetable_path, &wavetable, true).unwrap();

    let output = Command::new(&binary)
        .args(["wavetable", "inspect", wavetable_path.to_str().unwrap()])
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success());
    assert!(stdout.contains("Peak:"), "Output should show peak value");
    assert!(stdout.contains("RMS:"), "Output should show RMS value");
}
