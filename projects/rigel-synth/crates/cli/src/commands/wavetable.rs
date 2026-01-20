//! Wavetable file inspection and validation commands.
//!
//! This module provides CLI commands for inspecting wavetable files with
//! embedded protobuf metadata in the WTBL chunk.

use anyhow::{Context, Result};
use clap::Subcommand;
use std::path::{Path, PathBuf};
use wavetable_io::proto;
use wavetable_io::{read_wavetable, validate_wavetable, WavetableFile};

/// Wavetable file operations.
#[derive(Subcommand)]
pub enum WavetableCommands {
    /// Inspect a wavetable file and display its metadata
    Inspect {
        /// Path to the wavetable file
        file: PathBuf,

        /// Show verbose output including type-specific metadata
        #[arg(short, long)]
        verbose: bool,
    },

    /// Validate a wavetable file
    Validate {
        /// Path to the wavetable file
        file: PathBuf,

        /// Show detailed validation results
        #[arg(short, long)]
        verbose: bool,
    },
}

/// Execute a wavetable command.
pub fn execute(cmd: WavetableCommands) -> Result<()> {
    match cmd {
        WavetableCommands::Inspect { file, verbose } => inspect_wavetable(&file, verbose),
        WavetableCommands::Validate { file, verbose } => validate_wavetable_file(&file, verbose),
    }
}

/// Inspect a wavetable file and print its metadata.
fn inspect_wavetable(path: &Path, verbose: bool) -> Result<()> {
    let wavetable = read_wavetable(path)
        .with_context(|| format!("Failed to read wavetable: {}", path.display()))?;

    print_wavetable_info(&wavetable, path, verbose);

    Ok(())
}

/// Print wavetable information in a formatted display.
fn print_wavetable_info(wavetable: &WavetableFile, path: &Path, verbose: bool) {
    let filename = path.file_name().unwrap_or_default().to_string_lossy();
    let separator = "─".repeat(filename.len() + 12);

    println!("Wavetable: {}", filename);
    println!("{}", separator);
    println!();

    // Schema and type info
    println!("Schema Version: {}", wavetable.metadata.schema_version);
    println!(
        "Type: {}",
        wavetable_type_name(wavetable.metadata.wavetable_type())
    );

    // Optional metadata
    if let Some(ref name) = wavetable.metadata.name {
        println!("Name: {}", name);
    }
    if let Some(ref author) = wavetable.metadata.author {
        println!("Author: {}", author);
    }
    if let Some(ref description) = wavetable.metadata.description {
        println!("Description: {}", description);
    }

    println!();
    println!("Structure:");
    println!(
        "  Frame Length: {} samples",
        wavetable.metadata.frame_length
    );
    println!("  Frames: {} keyframes", wavetable.metadata.num_frames);
    println!("  Mip Levels: {}", wavetable.metadata.num_mip_levels);
    println!(
        "  Mip Frame Lengths: {:?}",
        wavetable.metadata.mip_frame_lengths
    );

    println!();
    println!("Audio Data:");
    let total_samples = wavetable.total_samples();
    println!("  Total Samples: {}", format_number(total_samples));
    let file_size_bytes = total_samples * 4; // 32-bit float
    println!("  Audio Size: {}", format_bytes(file_size_bytes));
    println!("  Sample Rate: {} Hz", wavetable.sample_rate);

    // Calculate peak and RMS from audio data
    let (peak, rms) = calculate_audio_stats(&wavetable.mip_levels);
    println!("  Peak: {:.4}", peak);
    println!("  RMS: {:.4}", rms);

    // Normalization method
    if wavetable.metadata.normalization_method != 0 {
        println!(
            "  Normalization: {}",
            normalization_method_name(wavetable.metadata.normalization_method)
        );
    }

    // Optional fields
    if verbose {
        println!();
        println!("Optional Fields:");
        if let Some(bit_depth) = wavetable.metadata.source_bit_depth {
            println!("  Source Bit Depth: {}", bit_depth);
        }
        if let Some(tuning) = wavetable.metadata.tuning_reference {
            println!("  Tuning Reference: {} Hz", tuning);
        }
        if let Some(ref gen_params) = wavetable.metadata.generation_parameters {
            println!("  Generation Parameters: {}", gen_params);
        }
        if let Some(sample_rate) = wavetable.metadata.sample_rate {
            println!("  Stored Sample Rate: {} Hz", sample_rate);
        }

        // Type-specific metadata
        if let Some(ref type_meta) = wavetable.metadata.type_metadata {
            println!();
            println!("Type-Specific Metadata:");
            print_type_metadata(type_meta);
        }

        // Unknown fields indicator (forward compatibility)
        // Note: prost doesn't expose unknown fields directly, but we can check
        // if the re-encoded size differs from decoded to detect their presence
    }
}

/// Calculate peak and RMS from mip level audio data.
fn calculate_audio_stats(mip_levels: &[wavetable_io::MipLevel]) -> (f32, f32) {
    let mut peak: f32 = 0.0;
    let mut sum_squares: f64 = 0.0;
    let mut total_samples: usize = 0;

    for mip in mip_levels {
        for frame in &mip.frames {
            for &sample in frame {
                let abs_sample = sample.abs();
                if abs_sample > peak {
                    peak = abs_sample;
                }
                sum_squares += (sample as f64) * (sample as f64);
                total_samples += 1;
            }
        }
    }

    let rms = if total_samples > 0 {
        (sum_squares / total_samples as f64).sqrt() as f32
    } else {
        0.0
    };

    (peak, rms)
}

/// Print type-specific metadata.
fn print_type_metadata(type_meta: &proto::wavetable_metadata::TypeMetadata) {
    match type_meta {
        proto::wavetable_metadata::TypeMetadata::ClassicDigital(m) => {
            if let Some(bit_depth) = m.original_bit_depth {
                println!("  Original Bit Depth: {}", bit_depth);
            }
            if let Some(sample_rate) = m.original_sample_rate {
                println!("  Original Sample Rate: {} Hz", sample_rate);
            }
            if let Some(ref hardware) = m.source_hardware {
                println!("  Source Hardware: {}", hardware);
            }
            if !m.harmonic_caps.is_empty() {
                println!("  Harmonic Caps: {:?}", m.harmonic_caps);
            }
        }
        proto::wavetable_metadata::TypeMetadata::HighResolution(m) => {
            if let Some(max_harmonics) = m.max_harmonics {
                println!("  Max Harmonics: {}", max_harmonics);
            }
            if m.interpolation_hint != 0 {
                println!(
                    "  Interpolation Hint: {}",
                    interpolation_hint_name(m.interpolation_hint)
                );
            }
            if let Some(ref synth) = m.source_synth {
                println!("  Source Synth: {}", synth);
            }
        }
        proto::wavetable_metadata::TypeMetadata::VintageEmulation(m) => {
            if let Some(ref hardware) = m.emulated_hardware {
                println!("  Emulated Hardware: {}", hardware);
            }
            if let Some(ref osc_type) = m.oscillator_type {
                println!("  Oscillator Type: {}", osc_type);
            }
            if let Some(preserves_aliasing) = m.preserves_aliasing {
                println!("  Preserves Aliasing: {}", preserves_aliasing);
            }
        }
        proto::wavetable_metadata::TypeMetadata::PcmSample(m) => {
            if let Some(sample_rate) = m.original_sample_rate {
                println!("  Original Sample Rate: {} Hz", sample_rate);
            }
            if let Some(root_note) = m.root_note {
                println!("  Root Note: {} (MIDI)", root_note);
            }
            if let Some(loop_start) = m.loop_start {
                println!("  Loop Start: {}", loop_start);
            }
            if let Some(loop_end) = m.loop_end {
                println!("  Loop End: {}", loop_end);
            }
        }
    }
}

/// Validate a wavetable file and report results.
fn validate_wavetable_file(path: &Path, verbose: bool) -> Result<()> {
    let wavetable = read_wavetable(path)
        .with_context(|| format!("Failed to read wavetable: {}", path.display()))?;

    let result = validate_wavetable(&wavetable);

    match result {
        Ok(()) => {
            println!("✓ Valid wavetable: {}", path.display());
            if verbose {
                println!();
                println!("Validation passed for:");
                println!("  - RIFF/WAV structure");
                println!("  - WTBL chunk present");
                println!("  - Protobuf metadata decodable");
                println!("  - Schema version valid");
                println!("  - Frame structure consistent");
                println!("  - Audio data length matches metadata");
            }
            Ok(())
        }
        Err(e) => {
            println!("✗ Invalid wavetable: {}", path.display());
            println!();
            println!("Validation errors:");
            println!("  {}", e);
            Err(e.into())
        }
    }
}

/// Get human-readable name for wavetable type.
fn wavetable_type_name(wt_type: proto::WavetableType) -> &'static str {
    match wt_type {
        proto::WavetableType::Unspecified => "UNSPECIFIED",
        proto::WavetableType::ClassicDigital => "CLASSIC_DIGITAL",
        proto::WavetableType::HighResolution => "HIGH_RESOLUTION",
        proto::WavetableType::VintageEmulation => "VINTAGE_EMULATION",
        proto::WavetableType::PcmSample => "PCM_SAMPLE",
        proto::WavetableType::Custom => "CUSTOM",
    }
}

/// Get human-readable name for normalization method.
fn normalization_method_name(method: i32) -> &'static str {
    match method {
        1 => "PEAK",
        2 => "RMS",
        3 => "NONE",
        _ => "UNSPECIFIED",
    }
}

/// Get human-readable name for interpolation hint.
fn interpolation_hint_name(hint: i32) -> &'static str {
    match hint {
        1 => "LINEAR",
        2 => "CUBIC",
        3 => "SINC",
        _ => "UNSPECIFIED",
    }
}

/// Format a number with comma separators.
fn format_number(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.insert(0, ',');
        }
        result.insert(0, c);
    }
    result
}

/// Format bytes into a human-readable size string.
fn format_bytes(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{} B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    }
}
