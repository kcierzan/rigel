"""Wavetable file reader.

This module provides functionality to load wavetables from WAV files
with embedded protobuf metadata in the WTBL chunk.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from wtgen.format.proto import wavetable_pb2 as pb
from wtgen.format.riff import (
    WAVE_FORMAT_IEEE_FLOAT,
    WAVE_FORMAT_PCM,
    RiffError,
    extract_wtbl_chunk,
    read_data_chunk,
    read_fmt_chunk,
)
from wtgen.format.types import (
    ClassicDigitalMetadata,
    HighResolutionMetadata,
    NormalizationMethod,
    PcmSampleMetadata,
    TypeMetadata,
    VintageEmulationMetadata,
    WavetableType,
)
from wtgen.format.validation import ValidationError, validate_audio_data, validate_metadata

# Maximum file size in bytes (100 MB per FR-030b)
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024


@dataclass
class WavetableFile:
    """A complete wavetable file with audio data and metadata."""

    metadata: pb.WavetableMetadata
    """The protobuf metadata from the WTBL chunk."""

    mipmaps: list[NDArray[np.float32]]
    """List of mipmap arrays, each with shape (num_frames, frame_length)."""

    sample_rate: int
    """Sample rate from the WAV file."""

    @property
    def wavetable_type(self) -> WavetableType:
        """Get the wavetable type as an enum."""
        return WavetableType.from_proto(self.metadata.wavetable_type)

    @property
    def name(self) -> str | None:
        """Get the name if present."""
        return self.metadata.name if self.metadata.HasField("name") else None

    @property
    def author(self) -> str | None:
        """Get the author if present."""
        return self.metadata.author if self.metadata.HasField("author") else None

    @property
    def description(self) -> str | None:
        """Get the description if present."""
        return self.metadata.description if self.metadata.HasField("description") else None

    @property
    def normalization_method(self) -> NormalizationMethod:
        """Get the normalization method."""
        return NormalizationMethod.from_proto(self.metadata.normalization_method)

    @property
    def num_frames(self) -> int:
        """Get the number of frames per mip level."""
        return self.metadata.num_frames

    @property
    def frame_length(self) -> int:
        """Get the frame length at mip level 0."""
        return self.metadata.frame_length

    @property
    def num_mip_levels(self) -> int:
        """Get the number of mip levels."""
        return self.metadata.num_mip_levels

    @property
    def mip_frame_lengths(self) -> list[int]:
        """Get the frame lengths for each mip level."""
        return list(self.metadata.mip_frame_lengths)

    def total_samples(self) -> int:
        """Get the total number of samples across all mip levels."""
        return sum(
            self.metadata.mip_frame_lengths[i] * self.metadata.num_frames
            for i in range(self.metadata.num_mip_levels)
        )

    def get_type_metadata(self) -> TypeMetadata | None:
        """Get the type-specific metadata if present."""
        which = self.metadata.WhichOneof("type_metadata")
        if which == "classic_digital":
            return ClassicDigitalMetadata.from_proto(self.metadata.classic_digital)
        elif which == "high_resolution":
            return HighResolutionMetadata.from_proto(self.metadata.high_resolution)
        elif which == "vintage_emulation":
            return VintageEmulationMetadata.from_proto(self.metadata.vintage_emulation)
        elif which == "pcm_sample":
            return PcmSampleMetadata.from_proto(self.metadata.pcm_sample)
        return None


def load_wavetable_wav(
    path: Path | str,
    *,
    validate: bool = True,
) -> WavetableFile:
    """Load a wavetable from a WAV file with embedded metadata.

    Args:
        path: Path to the WAV file.
        validate: Whether to validate the loaded data.

    Returns:
        WavetableFile containing metadata and audio data.

    Raises:
        RiffError: If the file is not a valid WAV or WTBL chunk is missing.
        ValidationError: If validation fails and validate=True, or if file
            exceeds 100MB (FR-030b), or if samples contain NaN/Infinity (FR-028).
    """
    path = Path(path)

    # FR-030b: Check file size limit (100 MB)
    try:
        file_size = path.stat().st_size
    except FileNotFoundError as e:
        raise RiffError(f"File not found: {path}") from e
    if file_size > MAX_FILE_SIZE_BYTES:
        raise ValidationError(
            f"File size ({file_size / (1024 * 1024):.1f} MB) exceeds maximum "
            f"allowed size of 100 MB per FR-030b"
        )

    # Read WTBL chunk and parse metadata
    wtbl_data = extract_wtbl_chunk(path)
    metadata = pb.WavetableMetadata()
    metadata.ParseFromString(wtbl_data)

    # Validate metadata if requested
    if validate:
        result = validate_metadata(metadata)
        if not result.valid:
            raise ValidationError(f"Metadata validation failed: {result.errors}")

    # Read audio format info
    audio_format, num_channels, sample_rate, bits_per_sample = read_fmt_chunk(path)

    if num_channels != 1:
        raise ValidationError(f"Expected mono audio, got {num_channels} channels")

    # Read audio data
    data_bytes = read_data_chunk(path)

    # Convert bytes to samples based on format
    if audio_format == WAVE_FORMAT_IEEE_FLOAT and bits_per_sample == 32:
        samples = np.frombuffer(data_bytes, dtype=np.float32)
    elif audio_format == WAVE_FORMAT_PCM and bits_per_sample == 16:
        samples = np.frombuffer(data_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    elif audio_format == WAVE_FORMAT_PCM and bits_per_sample == 24:
        # 24-bit PCM requires special handling
        samples = _decode_24bit_pcm(data_bytes)
    elif audio_format == WAVE_FORMAT_PCM and bits_per_sample == 32:
        samples = np.frombuffer(data_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValidationError(
            f"Unsupported audio format: format={audio_format}, bits={bits_per_sample}"
        )

    # FR-028: Check for NaN/Infinity values immediately after parsing
    # This catches issues early before processing
    if not np.all(np.isfinite(samples)):
        nan_count = int(np.sum(np.isnan(samples)))
        inf_count = int(np.sum(np.isinf(samples)))
        # Find first non-finite sample index for debugging
        non_finite_indices = np.where(~np.isfinite(samples))[0]
        first_bad_idx = int(non_finite_indices[0]) if len(non_finite_indices) > 0 else -1
        first_bad_val = float(samples[first_bad_idx]) if first_bad_idx >= 0 else 0.0
        raise ValidationError(
            f"Audio samples contain non-finite values per FR-028: "
            f"{nan_count} NaN, {inf_count} Infinity. "
            f"First occurrence at sample index {first_bad_idx} (value: {first_bad_val})"
        )

    # Split samples into mipmaps
    mipmaps = _split_into_mipmaps(samples, metadata)

    # Validate audio data if requested
    if validate:
        result = validate_audio_data(mipmaps, metadata)
        if not result.valid:
            raise ValidationError(f"Audio data validation failed: {result.errors}")

    return WavetableFile(
        metadata=metadata,
        mipmaps=mipmaps,
        sample_rate=sample_rate,
    )


def _split_into_mipmaps(
    samples: NDArray[np.float32],
    metadata: pb.WavetableMetadata,
) -> list[NDArray[np.float32]]:
    """Split flat sample array into mipmaps.

    Data is expected in mip-major, frame-secondary order:
    [mip0_frame0][mip0_frame1]...[mip1_frame0][mip1_frame1]...

    Args:
        samples: Flat array of all samples.
        metadata: Metadata describing the expected structure.

    Returns:
        List of 2D arrays with shape (num_frames, frame_length).
    """
    mipmaps = []
    offset = 0

    for i in range(metadata.num_mip_levels):
        frame_length = metadata.mip_frame_lengths[i]
        total_samples = frame_length * metadata.num_frames

        mip_samples = samples[offset : offset + total_samples]
        mip_2d = mip_samples.reshape((metadata.num_frames, frame_length))
        mipmaps.append(mip_2d)

        offset += total_samples

    return mipmaps


def _decode_24bit_pcm(data: bytes) -> NDArray[np.float32]:
    """Decode 24-bit PCM samples to float32.

    Uses vectorized NumPy operations for performance (H-4 fix).
    The naive Python loop was extremely slow for large files.

    Args:
        data: Raw 24-bit PCM data (3 bytes per sample, little-endian).

    Returns:
        Float32 samples normalized to [-1, 1].
    """
    num_samples = len(data) // 3
    if num_samples == 0:
        return np.array([], dtype=np.float32)

    # Truncate to exact multiple of 3 bytes and reshape to (N, 3)
    raw = np.frombuffer(data[: num_samples * 3], dtype=np.uint8).reshape(-1, 3)

    # Combine bytes as little-endian 24-bit unsigned integers
    # raw[:, 0] is LSB, raw[:, 2] is MSB
    values = (
        raw[:, 0].astype(np.int32)
        | (raw[:, 1].astype(np.int32) << 8)
        | (raw[:, 2].astype(np.int32) << 16)
    )

    # Sign extend from 24-bit: if bit 23 is set, the value is negative
    # We need to set bits 24-31 to 1 for negative values
    sign_mask = (values & 0x800000) != 0
    values = np.where(sign_mask, values | np.int32(0xFF000000), values)

    # Convert to float32 normalized to [-1, 1]
    # 2^23 = 8388608 is the maximum positive value for 24-bit signed
    return (values / 8388608.0).astype(np.float32)
