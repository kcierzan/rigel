//! # wavetable-io
//!
//! Wavetable file I/O library for reading and writing wavetable files in the
//! RIFF/WAV format with embedded protobuf metadata (WTBL chunk).
//!
//! ## Format Overview
//!
//! The wavetable interchange format uses standard WAV files with a custom RIFF chunk:
//!
//! ```text
//! ┌────────────────────────────────────────┐
//! │ RIFF Header ("WAVE")                   │
//! ├────────────────────────────────────────┤
//! │ fmt  chunk (audio format)              │
//! ├────────────────────────────────────────┤
//! │ data chunk (waveform samples)          │
//! │   - 32-bit float                       │
//! │   - Mip-major ordering                 │
//! ├────────────────────────────────────────┤
//! │ WTBL chunk (protobuf metadata)         │
//! │   - Schema version                     │
//! │   - Wavetable type                     │
//! │   - Frame structure                    │
//! │   - Optional metadata                  │
//! └────────────────────────────────────────┘
//! ```
//!
//! ## Example
//!
//! ```rust,ignore
//! use wavetable_io::{read_wavetable, WavetableFile};
//! use std::path::Path;
//!
//! let wavetable = read_wavetable(Path::new("my_wavetable.wav"))?;
//! println!("Name: {:?}", wavetable.metadata.name);
//! println!("Type: {:?}", wavetable.metadata.wavetable_type);
//! ```

pub mod reader;
pub mod riff;
pub mod types;
pub mod validation;
pub mod writer;

// Re-export main types for convenience
pub use reader::read_wavetable;
pub use types::{MipLevel, WavetableFile};
pub use validation::{validate_wavetable, ValidationError};
pub use writer::write_wavetable;

/// The protobuf-generated types from the wavetable.proto schema.
///
/// This module contains the Protocol Buffers message types for wavetable metadata.
/// These are generated at build time by prost-build.
pub mod proto {
    include!(concat!(env!("OUT_DIR"), "/rigel.wavetable.rs"));
}
