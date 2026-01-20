"""Wavetable file writer.

This module provides functionality to export wavetables as WAV files
with embedded protobuf metadata in the WTBL chunk.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from wtgen.format.proto import wavetable_pb2 as pb
from wtgen.format.riff import build_wav_with_wtbl
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


def save_wavetable_wav(
    path: Path | str,
    mipmaps: Sequence[NDArray[np.float32]],
    wavetable_type: WavetableType,
    *,
    name: str | None = None,
    author: str | None = None,
    description: str | None = None,
    normalization_method: NormalizationMethod = NormalizationMethod.UNSPECIFIED,
    source_bit_depth: int | None = None,
    tuning_reference: float | None = None,
    generation_parameters: str | None = None,
    sample_rate: int = 44100,
    type_metadata: TypeMetadata | None = None,
    validate: bool = True,
) -> None:
    """Save a wavetable to a WAV file with embedded metadata.

    Args:
        path: Output file path.
        mipmaps: List of numpy arrays, one per mip level.
                 Each array should be shape (num_frames, frame_length) or flattened.
                 Mip level 0 is highest resolution.
        wavetable_type: Classification of the wavetable.
        name: Human-readable name for the wavetable.
        author: Creator attribution.
        description: Extended description.
        normalization_method: How the audio was normalized.
        source_bit_depth: Original source bit depth (8, 16, 24, 32).
        tuning_reference: Reference frequency in Hz (default: 440.0).
        generation_parameters: JSON-encoded generation parameters.
        sample_rate: Sample rate for the WAV file.
        type_metadata: Type-specific metadata (ClassicDigitalMetadata, etc.).
        validate: Whether to validate the data before writing.

    Raises:
        ValidationError: If validation fails and validate=True.
        ValueError: If arguments are inconsistent.
    """
    path = Path(path)

    # Validate mipmaps shape
    if not mipmaps:
        raise ValueError("mipmaps cannot be empty")

    # Determine structure from first mipmap
    mip0 = np.asarray(mipmaps[0], dtype=np.float32)
    if mip0.ndim == 2:
        num_frames, frame_length = mip0.shape
    elif mip0.ndim == 1:
        # Assume it's flattened, need to infer structure
        # Try to determine from subsequent mipmaps or raise error
        raise ValueError(
            "mipmaps should be 2D arrays with shape (num_frames, frame_length). "
            "Got 1D array; please reshape to 2D."
        )
    else:
        raise ValueError(f"mipmaps[0] has unexpected shape {mip0.shape}")

    # Extract mip frame lengths from mipmaps
    mip_frame_lengths = []
    for i, mip in enumerate(mipmaps):
        mip_arr = np.asarray(mip, dtype=np.float32)
        if mip_arr.ndim != 2:
            raise ValueError(f"mipmaps[{i}] should be 2D, got shape {mip_arr.shape}")
        if mip_arr.shape[0] != num_frames:
            raise ValueError(f"mipmaps[{i}] has {mip_arr.shape[0]} frames, expected {num_frames}")
        mip_frame_lengths.append(mip_arr.shape[1])

    # Build metadata
    metadata = pb.WavetableMetadata()
    metadata.schema_version = 1
    metadata.wavetable_type = wavetable_type.to_proto()
    metadata.frame_length = frame_length
    metadata.num_frames = num_frames
    metadata.num_mip_levels = len(mipmaps)
    metadata.mip_frame_lengths.extend(mip_frame_lengths)

    # Optional fields
    metadata.normalization_method = normalization_method.to_proto()
    if source_bit_depth is not None:
        metadata.source_bit_depth = source_bit_depth
    if author is not None:
        metadata.author = author
    if name is not None:
        metadata.name = name
    if description is not None:
        metadata.description = description
    if tuning_reference is not None:
        metadata.tuning_reference = tuning_reference
    if generation_parameters is not None:
        metadata.generation_parameters = generation_parameters
    if sample_rate is not None:
        metadata.sample_rate = sample_rate

    # Type-specific metadata
    if type_metadata is not None:
        if isinstance(type_metadata, ClassicDigitalMetadata):
            metadata.classic_digital.CopyFrom(type_metadata.to_proto())
        elif isinstance(type_metadata, HighResolutionMetadata):
            metadata.high_resolution.CopyFrom(type_metadata.to_proto())
        elif isinstance(type_metadata, VintageEmulationMetadata):
            metadata.vintage_emulation.CopyFrom(type_metadata.to_proto())
        elif isinstance(type_metadata, PcmSampleMetadata):
            metadata.pcm_sample.CopyFrom(type_metadata.to_proto())

    # Validate if requested
    if validate:
        result = validate_metadata(metadata)
        if not result.valid:
            raise ValidationError(f"Metadata validation failed: {result.errors}")

        result = validate_audio_data(mipmaps, metadata)
        if not result.valid:
            raise ValidationError(f"Audio data validation failed: {result.errors}")

    # Concatenate all mipmap data in mip-major, frame-secondary order
    samples = _concatenate_mipmaps(mipmaps)

    # Serialize metadata to protobuf
    wtbl_data = metadata.SerializeToString()

    # Build WAV file
    wav_bytes = build_wav_with_wtbl(
        samples=samples.tobytes(),
        sample_rate=sample_rate,
        wtbl_data=wtbl_data,
        num_channels=1,
        bits_per_sample=32,
    )

    # Write to file
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(wav_bytes)


def _concatenate_mipmaps(mipmaps: Sequence[NDArray[np.float32]]) -> NDArray[np.float32]:
    """Concatenate mipmaps into a single flat array.

    Data is organized as mip-major, frame-secondary:
    [mip0_frame0][mip0_frame1]...[mip1_frame0][mip1_frame1]...

    Args:
        mipmaps: List of 2D arrays with shape (num_frames, frame_length).

    Returns:
        Flattened array of all samples.
    """
    # Flatten each mipmap in row-major (C) order: frame0, frame1, ...
    arrays = [np.asarray(mip, dtype=np.float32).flatten() for mip in mipmaps]
    return np.concatenate(arrays)
