//! # Rigel CLI
//!
//! Command-line interface for testing the Rigel wavetable synthesizer.
//! Generates audio files for testing and development purposes.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use hound::{SampleFormat, WavSpec, WavWriter};
use rigel_dsp::{SynthEngine, SynthParams};
use std::path::PathBuf;

/// CLI tool for generating test audio with Rigel synthesizer
#[derive(Parser)]
#[command(name = "rigel")]
#[command(about = "A CLI for the Rigel wavetable synthesizer")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate a single note
    Note {
        /// Output file path
        #[arg(short, long, default_value = "output.wav")]
        output: PathBuf,

        /// MIDI note number (0-127)
        #[arg(short, long, default_value = "60")]
        note: u8,

        /// Duration in seconds
        #[arg(short, long, default_value = "2.0")]
        duration: f32,

        /// Sample rate in Hz
        #[arg(short, long, default_value = "44100")]
        sample_rate: u32,

        /// Velocity (0.0 to 1.0)
        #[arg(short, long, default_value = "0.8")]
        velocity: f32,
    },

    /// Generate a chord
    Chord {
        /// Output file path
        #[arg(short, long, default_value = "chord.wav")]
        output: PathBuf,

        /// Root note MIDI number
        #[arg(short, long, default_value = "60")]
        root: u8,

        /// Chord type
        #[arg(short, long, default_value = "major")]
        chord_type: String,

        /// Duration in seconds
        #[arg(short, long, default_value = "3.0")]
        duration: f32,

        /// Sample rate in Hz
        #[arg(short, long, default_value = "44100")]
        sample_rate: u32,

        /// Velocity (0.0 to 1.0)
        #[arg(short, long, default_value = "0.7")]
        velocity: f32,
    },

    /// Generate a scale
    Scale {
        /// Output file path
        #[arg(short, long, default_value = "scale.wav")]
        output: PathBuf,

        /// Starting MIDI note number
        #[arg(short, long, default_value = "60")]
        start_note: u8,

        /// Number of octaves
        #[arg(short, long, default_value = "1")]
        octaves: u8,

        /// Note duration in seconds
        #[arg(short, long, default_value = "0.5")]
        note_duration: f32,

        /// Sample rate in Hz
        #[arg(short, long, default_value = "44100")]
        sample_rate: u32,

        /// Scale type (major, minor, chromatic)
        #[arg(short, long, default_value = "major")]
        scale_type: String,
    },

    /// Test wavetable morphing
    Morph {
        /// Output file path
        #[arg(short, long, default_value = "morph.wav")]
        output: PathBuf,

        /// MIDI note number
        #[arg(short, long, default_value = "60")]
        note: u8,

        /// Duration in seconds
        #[arg(short, long, default_value = "4.0")]
        duration: f32,

        /// Sample rate in Hz
        #[arg(short, long, default_value = "44100")]
        sample_rate: u32,

        /// Morph speed (cycles per second)
        #[arg(short, long, default_value = "0.5")]
        morph_speed: f32,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Note {
            output,
            note,
            duration,
            sample_rate,
            velocity,
        } => generate_note(&output, note, duration, sample_rate as f32, velocity),

        Commands::Chord {
            output,
            root,
            chord_type,
            duration,
            sample_rate,
            velocity,
        } => generate_chord(
            &output,
            root,
            &chord_type,
            duration,
            sample_rate as f32,
            velocity,
        ),

        Commands::Scale {
            output,
            start_note,
            octaves,
            note_duration,
            sample_rate,
            scale_type,
        } => generate_scale(
            &output,
            start_note,
            octaves,
            note_duration,
            sample_rate as f32,
            &scale_type,
        ),

        Commands::Morph {
            output,
            note,
            duration,
            sample_rate,
            morph_speed,
        } => generate_morph(&output, note, duration, sample_rate as f32, morph_speed),
    }
}

fn generate_note(
    output_path: &PathBuf,
    note: u8,
    duration: f32,
    sample_rate: f32,
    velocity: f32,
) -> Result<()> {
    println!(
        "Generating note {} for {:.2}s at {}Hz sample rate...",
        note, duration, sample_rate
    );

    let mut synth = SynthEngine::new(sample_rate);

    let spec = WavSpec {
        channels: 1,
        sample_rate: sample_rate as u32,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };

    let mut writer = WavWriter::create(output_path, spec).context("Failed to create WAV file")?;

    let total_samples = (duration * sample_rate) as usize;
    let attack_samples = (0.1 * sample_rate) as usize;
    let release_samples = (0.1 * sample_rate) as usize;

    // Create synthesis parameters
    let mut params = SynthParams {
        volume: 0.7,
        ..Default::default()
    };

    // Start the note
    synth.note_on(note, velocity);

    for i in 0..total_samples {
        // Simple envelope for note start/end
        let envelope = if i < attack_samples {
            i as f32 / attack_samples as f32
        } else if i > total_samples - release_samples {
            (total_samples - i) as f32 / release_samples as f32
        } else {
            1.0
        };

        params.volume = 0.7 * envelope;

        let sample = synth.process_sample(&params);
        writer
            .write_sample(sample)
            .context("Failed to write sample")?;
    }

    synth.note_off(note);

    writer.finalize().context("Failed to finalize WAV file")?;
    println!("Generated: {}", output_path.display());

    Ok(())
}

fn generate_chord(
    output_path: &PathBuf,
    root: u8,
    chord_type: &str,
    duration: f32,
    sample_rate: f32,
    velocity: f32,
) -> Result<()> {
    println!(
        "Generating {} chord from note {} for {:.2}s...",
        chord_type, root, duration
    );

    let chord_intervals = match chord_type {
        "major" => vec![0, 4, 7],
        "minor" => vec![0, 3, 7],
        "sus2" => vec![0, 2, 7],
        "sus4" => vec![0, 5, 7],
        "dim" => vec![0, 3, 6],
        "aug" => vec![0, 4, 8],
        "maj7" => vec![0, 4, 7, 11],
        "min7" => vec![0, 3, 7, 10],
        _ => {
            println!("Unknown chord type '{}', using major", chord_type);
            vec![0, 4, 7]
        }
    };

    let mut synth = SynthEngine::new(sample_rate);

    let spec = WavSpec {
        channels: 1,
        sample_rate: sample_rate as u32,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };

    let mut writer = WavWriter::create(output_path, spec).context("Failed to create WAV file")?;

    let total_samples = (duration * sample_rate) as usize;
    let attack_samples = (0.1 * sample_rate) as usize;
    let release_samples = (0.2 * sample_rate) as usize;

    let params = SynthParams::default();

    // Start all chord notes
    for &interval in &chord_intervals {
        if root + interval <= 127 {
            synth.note_on(root + interval, velocity);
        }
    }

    for i in 0..total_samples {
        // Simple envelope
        let envelope = if i < attack_samples {
            i as f32 / attack_samples as f32
        } else if i > total_samples - release_samples {
            (total_samples - i) as f32 / release_samples as f32
        } else {
            1.0
        };

        let mut params = params;
        params.volume = 0.5 * envelope; // Lower volume for multiple notes

        let sample = synth.process_sample(&params);
        writer
            .write_sample(sample)
            .context("Failed to write sample")?;
    }

    // Stop all chord notes
    for &interval in &chord_intervals {
        if root + interval <= 127 {
            synth.note_off(root + interval);
        }
    }

    writer.finalize().context("Failed to finalize WAV file")?;
    println!("Generated: {}", output_path.display());

    Ok(())
}

fn generate_scale(
    output_path: &PathBuf,
    start_note: u8,
    octaves: u8,
    note_duration: f32,
    sample_rate: f32,
    scale_type: &str,
) -> Result<()> {
    println!(
        "Generating {} scale starting from note {} for {} octaves...",
        scale_type, start_note, octaves
    );

    let scale_intervals = match scale_type {
        "major" => vec![0, 2, 4, 5, 7, 9, 11],
        "minor" => vec![0, 2, 3, 5, 7, 8, 10],
        "chromatic" => vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "pentatonic" => vec![0, 2, 4, 7, 9],
        _ => {
            println!("Unknown scale type '{}', using major", scale_type);
            vec![0, 2, 4, 5, 7, 9, 11]
        }
    };

    let mut synth = SynthEngine::new(sample_rate);

    let spec = WavSpec {
        channels: 1,
        sample_rate: sample_rate as u32,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };

    let mut writer = WavWriter::create(output_path, spec).context("Failed to create WAV file")?;

    let note_samples = (note_duration * sample_rate) as usize;
    let attack_samples = (0.05 * sample_rate) as usize;
    let release_samples = (0.1 * sample_rate) as usize;

    let params = SynthParams::default();

    for octave in 0..octaves {
        for &interval in &scale_intervals {
            let note = start_note + (octave * 12) + interval;
            if note > 127 {
                break;
            }

            synth.note_on(note, 0.8);

            for i in 0..note_samples {
                let envelope = if i < attack_samples {
                    i as f32 / attack_samples as f32
                } else if i > note_samples - release_samples {
                    (note_samples - i) as f32 / release_samples as f32
                } else {
                    1.0
                };

                let mut params = params;
                params.volume = 0.7 * envelope;

                let sample = synth.process_sample(&params);
                writer
                    .write_sample(sample)
                    .context("Failed to write sample")?;
            }

            synth.note_off(note);
        }
    }

    writer.finalize().context("Failed to finalize WAV file")?;
    println!("Generated: {}", output_path.display());

    Ok(())
}

fn generate_morph(
    output_path: &PathBuf,
    note: u8,
    duration: f32,
    sample_rate: f32,
    morph_speed: f32,
) -> Result<()> {
    println!(
        "Generating wavetable morph on note {} for {:.2}s at {:.2} Hz morph rate...",
        note, duration, morph_speed
    );

    let mut synth = SynthEngine::new(sample_rate);

    let spec = WavSpec {
        channels: 1,
        sample_rate: sample_rate as u32,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };

    let mut writer = WavWriter::create(output_path, spec).context("Failed to create WAV file")?;

    let total_samples = (duration * sample_rate) as usize;
    let attack_samples = (0.1 * sample_rate) as usize;
    let release_samples = (0.1 * sample_rate) as usize;

    let mut params = SynthParams::default();

    synth.note_on(note, 0.8);

    for i in 0..total_samples {
        let time = i as f32 / sample_rate;

        // For now, just modulate pitch slightly for morphing effect
        let morph_phase = 2.0 * std::f32::consts::PI * morph_speed * time;
        params.pitch_offset = morph_phase.sin() * 2.0; // ±2 semitones

        // Simple envelope
        let envelope = if i < attack_samples {
            i as f32 / attack_samples as f32
        } else if i > total_samples - release_samples {
            (total_samples - i) as f32 / release_samples as f32
        } else {
            1.0
        };

        params.volume = 0.7 * envelope;

        let sample = synth.process_sample(&params);
        writer
            .write_sample(sample)
            .context("Failed to write sample")?;
    }

    synth.note_off(note);

    writer.finalize().context("Failed to finalize WAV file")?;
    println!("Generated: {}", output_path.display());

    Ok(())
}
