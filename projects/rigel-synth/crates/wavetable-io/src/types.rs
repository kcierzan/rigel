//! Rust types for wavetable data structures.
//!
//! These types represent the in-memory structure of a wavetable file,
//! providing a more ergonomic API than the raw protobuf types.

use crate::proto;

/// A complete wavetable file with audio data and metadata.
#[derive(Debug, Clone)]
pub struct WavetableFile {
    /// The protobuf metadata from the WTBL chunk.
    pub metadata: proto::WavetableMetadata,

    /// The mip levels containing audio data.
    ///
    /// Each mip level contains `num_frames` frames, with frame length
    /// determined by `mip_frame_lengths[mip_level]`.
    pub mip_levels: Vec<MipLevel>,

    /// Sample rate of the audio data.
    pub sample_rate: u32,
}

/// A single mip level containing multiple frames.
#[derive(Debug, Clone)]
pub struct MipLevel {
    /// The mip level index (0 = highest resolution).
    pub level: usize,

    /// Frame length at this mip level (samples per frame).
    pub frame_length: u32,

    /// The frames at this mip level.
    /// Each frame is a vector of f32 samples.
    pub frames: Vec<Vec<f32>>,
}

impl WavetableFile {
    /// Create a new wavetable file with the given metadata and mip levels.
    pub fn new(
        metadata: proto::WavetableMetadata,
        mip_levels: Vec<MipLevel>,
        sample_rate: u32,
    ) -> Self {
        Self {
            metadata,
            mip_levels,
            sample_rate,
        }
    }

    /// Get the total number of samples across all mip levels.
    pub fn total_samples(&self) -> usize {
        self.mip_levels
            .iter()
            .map(|mip| mip.frames.iter().map(|f| f.len()).sum::<usize>())
            .sum()
    }

    /// Get the wavetable type as an enum.
    pub fn wavetable_type(&self) -> WavetableType {
        WavetableType::from_proto(self.metadata.wavetable_type())
    }

    /// Get the name if present.
    pub fn name(&self) -> Option<&str> {
        self.metadata.name.as_deref()
    }

    /// Get the author if present.
    pub fn author(&self) -> Option<&str> {
        self.metadata.author.as_deref()
    }

    /// Get the description if present.
    pub fn description(&self) -> Option<&str> {
        self.metadata.description.as_deref()
    }
}

impl MipLevel {
    /// Create a new mip level.
    pub fn new(level: usize, frame_length: u32, frames: Vec<Vec<f32>>) -> Self {
        Self {
            level,
            frame_length,
            frames,
        }
    }

    /// Get the number of frames at this mip level.
    pub fn num_frames(&self) -> usize {
        self.frames.len()
    }
}

/// High-level wavetable type classification.
///
/// This provides a more ergonomic API than the protobuf enum, with
/// automatic handling of unknown values as Custom.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WavetableType {
    /// Unspecified or unknown type (treated as Custom).
    Unspecified,
    /// PPG Wave-style wavetables (256 samples, 64 frames, 8-bit source).
    ClassicDigital,
    /// Modern high-resolution wavetables (2048+ samples, many frames).
    HighResolution,
    /// Vintage oscillator emulations (OSCar, Wasp, etc.).
    VintageEmulation,
    /// Single-cycle PCM samples (AWM/SY99-style).
    PcmSample,
    /// User-defined wavetables.
    Custom,
}

impl WavetableType {
    /// Convert from the protobuf enum.
    pub fn from_proto(proto_type: proto::WavetableType) -> Self {
        match proto_type {
            proto::WavetableType::Unspecified => Self::Unspecified,
            proto::WavetableType::ClassicDigital => Self::ClassicDigital,
            proto::WavetableType::HighResolution => Self::HighResolution,
            proto::WavetableType::VintageEmulation => Self::VintageEmulation,
            proto::WavetableType::PcmSample => Self::PcmSample,
            proto::WavetableType::Custom => Self::Custom,
        }
    }

    /// Convert to the protobuf enum.
    pub fn to_proto(self) -> proto::WavetableType {
        match self {
            Self::Unspecified => proto::WavetableType::Unspecified,
            Self::ClassicDigital => proto::WavetableType::ClassicDigital,
            Self::HighResolution => proto::WavetableType::HighResolution,
            Self::VintageEmulation => proto::WavetableType::VintageEmulation,
            Self::PcmSample => proto::WavetableType::PcmSample,
            Self::Custom => proto::WavetableType::Custom,
        }
    }

    /// Get a human-readable name for this type.
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Unspecified => "Unspecified",
            Self::ClassicDigital => "Classic Digital",
            Self::HighResolution => "High Resolution",
            Self::VintageEmulation => "Vintage Emulation",
            Self::PcmSample => "PCM Sample",
            Self::Custom => "Custom",
        }
    }
}

impl std::fmt::Display for WavetableType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

/// High-level normalization method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalizationMethod {
    /// Unspecified normalization.
    Unspecified,
    /// Normalized to peak amplitude.
    Peak,
    /// Normalized to target RMS level.
    Rms,
    /// No normalization applied.
    None,
}

impl NormalizationMethod {
    /// Convert from the protobuf enum.
    pub fn from_proto(proto_method: proto::NormalizationMethod) -> Self {
        match proto_method {
            proto::NormalizationMethod::NormalizationUnspecified => Self::Unspecified,
            proto::NormalizationMethod::NormalizationPeak => Self::Peak,
            proto::NormalizationMethod::NormalizationRms => Self::Rms,
            proto::NormalizationMethod::NormalizationNone => Self::None,
        }
    }

    /// Convert to the protobuf enum.
    pub fn to_proto(self) -> proto::NormalizationMethod {
        match self {
            Self::Unspecified => proto::NormalizationMethod::NormalizationUnspecified,
            Self::Peak => proto::NormalizationMethod::NormalizationPeak,
            Self::Rms => proto::NormalizationMethod::NormalizationRms,
            Self::None => proto::NormalizationMethod::NormalizationNone,
        }
    }
}

/// High-level interpolation hint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationHint {
    /// Use reader default.
    Unspecified,
    /// Linear interpolation.
    Linear,
    /// Cubic (Catmull-Rom) interpolation.
    Cubic,
    /// Windowed sinc interpolation.
    Sinc,
}

impl InterpolationHint {
    /// Convert from the protobuf enum.
    pub fn from_proto(proto_hint: proto::InterpolationHint) -> Self {
        match proto_hint {
            proto::InterpolationHint::InterpolationUnspecified => Self::Unspecified,
            proto::InterpolationHint::InterpolationLinear => Self::Linear,
            proto::InterpolationHint::InterpolationCubic => Self::Cubic,
            proto::InterpolationHint::InterpolationSinc => Self::Sinc,
        }
    }

    /// Convert to the protobuf enum.
    pub fn to_proto(self) -> proto::InterpolationHint {
        match self {
            Self::Unspecified => proto::InterpolationHint::InterpolationUnspecified,
            Self::Linear => proto::InterpolationHint::InterpolationLinear,
            Self::Cubic => proto::InterpolationHint::InterpolationCubic,
            Self::Sinc => proto::InterpolationHint::InterpolationSinc,
        }
    }
}
