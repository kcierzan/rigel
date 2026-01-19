//! # Rigel CLI
//!
//! Command-line interface for testing the Rigel wavetable synthesizer.
//! Generates audio files for testing and development purposes.

mod commands;

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
    /// Wavetable file inspection and validation
    #[command(subcommand)]
    Wavetable(commands::wavetable::WavetableCommands),

    /// Generate a single note
    Note {
        /// Output file path
        #[arg(short, long, default_value = "output.wav")]
        output: PathBuf,

        /// MIDI note number (0-127)
        #[arg(short, long, default_value = "60")]
        note: u8,

        /// Duration in seconds (note-on duration, release added automatically)
        #[arg(short, long, default_value = "1.0")]
        duration: f32,

        /// Sample rate in Hz
        #[arg(short, long, default_value = "44100")]
        sample_rate: u32,

        /// Velocity (0.0 to 1.0)
        #[arg(short, long, default_value = "0.8")]
        velocity: f32,

        /// Envelope attack time in seconds
        #[arg(long, default_value = "0.01")]
        attack: f32,

        /// Envelope decay time in seconds
        #[arg(long, default_value = "0.3")]
        decay: f32,

        /// Envelope sustain level (0.0 to 1.0)
        #[arg(long, default_value = "0.7")]
        sustain: f32,

        /// Envelope release time in seconds
        #[arg(long, default_value = "0.5")]
        release: f32,
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

        /// Duration in seconds (note-on duration, release added automatically)
        #[arg(short, long, default_value = "2.0")]
        duration: f32,

        /// Sample rate in Hz
        #[arg(short, long, default_value = "44100")]
        sample_rate: u32,

        /// Velocity (0.0 to 1.0)
        #[arg(short, long, default_value = "0.7")]
        velocity: f32,

        /// Envelope attack time in seconds
        #[arg(long, default_value = "0.01")]
        attack: f32,

        /// Envelope decay time in seconds
        #[arg(long, default_value = "0.3")]
        decay: f32,

        /// Envelope sustain level (0.0 to 1.0)
        #[arg(long, default_value = "0.7")]
        sustain: f32,

        /// Envelope release time in seconds
        #[arg(long, default_value = "0.5")]
        release: f32,
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

        /// Note duration in seconds (note-on duration per note)
        #[arg(short, long, default_value = "0.3")]
        note_duration: f32,

        /// Sample rate in Hz
        #[arg(short, long, default_value = "44100")]
        sample_rate: u32,

        /// Scale type (major, minor, chromatic)
        #[arg(short, long, default_value = "major")]
        scale_type: String,

        /// Envelope attack time in seconds
        #[arg(long, default_value = "0.01")]
        attack: f32,

        /// Envelope decay time in seconds
        #[arg(long, default_value = "0.1")]
        decay: f32,

        /// Envelope sustain level (0.0 to 1.0)
        #[arg(long, default_value = "0.8")]
        sustain: f32,

        /// Envelope release time in seconds
        #[arg(long, default_value = "0.15")]
        release: f32,
    },

    /// Test wavetable morphing
    Morph {
        /// Output file path
        #[arg(short, long, default_value = "morph.wav")]
        output: PathBuf,

        /// MIDI note number
        #[arg(short, long, default_value = "60")]
        note: u8,

        /// Duration in seconds (note-on duration, release added automatically)
        #[arg(short, long, default_value = "3.0")]
        duration: f32,

        /// Sample rate in Hz
        #[arg(short, long, default_value = "44100")]
        sample_rate: u32,

        /// Morph speed (cycles per second)
        #[arg(short, long, default_value = "0.5")]
        morph_speed: f32,

        /// Envelope attack time in seconds
        #[arg(long, default_value = "0.5")]
        attack: f32,

        /// Envelope decay time in seconds
        #[arg(long, default_value = "0.5")]
        decay: f32,

        /// Envelope sustain level (0.0 to 1.0)
        #[arg(long, default_value = "0.8")]
        sustain: f32,

        /// Envelope release time in seconds
        #[arg(long, default_value = "1.0")]
        release: f32,
    },
}

/// Envelope parameters for CLI commands
struct EnvelopeParams {
    attack: f32,
    decay: f32,
    sustain: f32,
    release: f32,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Wavetable(cmd) => commands::wavetable::execute(cmd),

        Commands::Note {
            output,
            note,
            duration,
            sample_rate,
            velocity,
            attack,
            decay,
            sustain,
            release,
        } => generate_note(
            &output,
            note,
            duration,
            sample_rate as f32,
            velocity,
            EnvelopeParams {
                attack,
                decay,
                sustain,
                release,
            },
        ),

        Commands::Chord {
            output,
            root,
            chord_type,
            duration,
            sample_rate,
            velocity,
            attack,
            decay,
            sustain,
            release,
        } => generate_chord(
            &output,
            root,
            &chord_type,
            duration,
            sample_rate as f32,
            velocity,
            EnvelopeParams {
                attack,
                decay,
                sustain,
                release,
            },
        ),

        Commands::Scale {
            output,
            start_note,
            octaves,
            note_duration,
            sample_rate,
            scale_type,
            attack,
            decay,
            sustain,
            release,
        } => generate_scale(
            &output,
            start_note,
            octaves,
            note_duration,
            sample_rate as f32,
            &scale_type,
            EnvelopeParams {
                attack,
                decay,
                sustain,
                release,
            },
        ),

        Commands::Morph {
            output,
            note,
            duration,
            sample_rate,
            morph_speed,
            attack,
            decay,
            sustain,
            release,
        } => generate_morph(
            &output,
            note,
            duration,
            sample_rate as f32,
            morph_speed,
            EnvelopeParams {
                attack,
                decay,
                sustain,
                release,
            },
        ),
    }
}

fn generate_note(
    output_path: &PathBuf,
    note: u8,
    duration: f32,
    sample_rate: f32,
    velocity: f32,
    env: EnvelopeParams,
) -> Result<()> {
    println!(
        "Generating note {} for {:.2}s (+ {:.2}s release) at {}Hz...",
        note, duration, env.release, sample_rate
    );

    let mut synth = SynthEngine::new(sample_rate);

    let spec = WavSpec {
        channels: 1,
        sample_rate: sample_rate as u32,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };

    let mut writer = WavWriter::create(output_path, spec).context("Failed to create WAV file")?;

    // Note-on duration + release tail
    let note_on_samples = (duration * sample_rate) as usize;
    let release_samples = (env.release * sample_rate * 1.5) as usize; // Extra buffer for release
    let total_samples = note_on_samples + release_samples;

    // Create synthesis parameters with envelope settings
    let params = SynthParams::from_adsr(
        0.7,
        0.0,
        env.attack,
        env.decay,
        env.sustain,
        env.release,
        sample_rate,
    );

    // Start the note with configured envelope
    synth.note_on_with_params(note, velocity, &params);

    for i in 0..total_samples {
        // Trigger note-off at the right time
        if i == note_on_samples {
            synth.note_off(note);
        }

        let sample = synth.process_sample(&params);
        writer
            .write_sample(sample)
            .context("Failed to write sample")?;

        // Stop early if envelope completed
        if i > note_on_samples && !synth.is_active() {
            break;
        }
    }

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
    env: EnvelopeParams,
) -> Result<()> {
    println!(
        "Generating {} chord from note {} for {:.2}s (+ {:.2}s release)...",
        chord_type, root, duration, env.release
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

    // Note-on duration + release tail
    let note_on_samples = (duration * sample_rate) as usize;
    let release_samples = (env.release * sample_rate * 1.5) as usize;
    let total_samples = note_on_samples + release_samples;

    // Create synthesis parameters with envelope settings
    let params = SynthParams::from_adsr(
        0.5, // Lower volume for chords
        0.0,
        env.attack,
        env.decay,
        env.sustain,
        env.release,
        sample_rate,
    );

    // Start all chord notes (only the first note uses note_on_with_params for config)
    let mut first = true;
    for &interval in &chord_intervals {
        if root + interval <= 127 {
            if first {
                synth.note_on_with_params(root + interval, velocity, &params);
                first = false;
            } else {
                synth.note_on(root + interval, velocity);
            }
        }
    }

    for i in 0..total_samples {
        // Trigger note-off at the right time
        if i == note_on_samples {
            for &interval in &chord_intervals {
                if root + interval <= 127 {
                    synth.note_off(root + interval);
                }
            }
        }

        let sample = synth.process_sample(&params);
        writer
            .write_sample(sample)
            .context("Failed to write sample")?;

        // Stop early if envelope completed
        if i > note_on_samples && !synth.is_active() {
            break;
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
    env: EnvelopeParams,
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

    // Note-on duration + release tail per note
    let note_on_samples = (note_duration * sample_rate) as usize;
    let release_samples = (env.release * sample_rate * 1.5) as usize;
    let samples_per_note = note_on_samples + release_samples;

    // Create synthesis parameters with envelope settings
    let params = SynthParams::from_adsr(
        0.7,
        0.0,
        env.attack,
        env.decay,
        env.sustain,
        env.release,
        sample_rate,
    );

    for octave in 0..octaves {
        for &interval in &scale_intervals {
            let note = start_note + (octave * 12) + interval;
            if note > 127 {
                break;
            }

            // Start note with configured envelope
            synth.note_on_with_params(note, 0.8, &params);

            for i in 0..samples_per_note {
                // Trigger note-off at the right time
                if i == note_on_samples {
                    synth.note_off(note);
                }

                let sample = synth.process_sample(&params);
                writer
                    .write_sample(sample)
                    .context("Failed to write sample")?;

                // Stop early if envelope completed
                if i > note_on_samples && !synth.is_active() {
                    break;
                }
            }
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
    env: EnvelopeParams,
) -> Result<()> {
    println!(
        "Generating wavetable morph on note {} for {:.2}s (+ {:.2}s release) at {:.2} Hz morph rate...",
        note, duration, env.release, morph_speed
    );

    let mut synth = SynthEngine::new(sample_rate);

    let spec = WavSpec {
        channels: 1,
        sample_rate: sample_rate as u32,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };

    let mut writer = WavWriter::create(output_path, spec).context("Failed to create WAV file")?;

    // Note-on duration + release tail
    let note_on_samples = (duration * sample_rate) as usize;
    let release_samples = (env.release * sample_rate * 1.5) as usize;
    let total_samples = note_on_samples + release_samples;

    // Create synthesis parameters with envelope settings
    let mut params = SynthParams::from_adsr(
        0.7,
        0.0,
        env.attack,
        env.decay,
        env.sustain,
        env.release,
        sample_rate,
    );

    // Start note with configured envelope
    synth.note_on_with_params(note, 0.8, &params);

    for i in 0..total_samples {
        // Trigger note-off at the right time
        if i == note_on_samples {
            synth.note_off(note);
        }

        let time = i as f32 / sample_rate;

        // Modulate pitch slightly for morphing effect
        let morph_phase = 2.0 * std::f32::consts::PI * morph_speed * time;
        params.pitch_offset = morph_phase.sin() * 2.0; // ±2 semitones

        let sample = synth.process_sample(&params);
        writer
            .write_sample(sample)
            .context("Failed to write sample")?;

        // Stop early if envelope completed
        if i > note_on_samples && !synth.is_active() {
            break;
        }
    }

    writer.finalize().context("Failed to finalize WAV file")?;
    println!("Generated: {}", output_path.display());

    Ok(())
}
