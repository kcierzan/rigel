"""Validation functions for wavetable metadata and audio data.

This module provides validation functions to ensure wavetable files
conform to the interchange format specification.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from wtgen.format.proto import wavetable_pb2 as pb


class ValidationError(Exception):
    """Error during wavetable validation."""

    def __init__(self, message: str, field: str | None = None) -> None:
        self.field = field
        super().__init__(message)


@dataclass
class ValidationResult:
    """Result of validation with optional warnings."""

    valid: bool
    errors: list[str]
    warnings: list[str]

    @classmethod
    def success(cls, warnings: list[str] | None = None) -> "ValidationResult":
        """Create a successful validation result."""
        return cls(valid=True, errors=[], warnings=warnings or [])

    @classmethod
    def failure(cls, errors: list[str], warnings: list[str] | None = None) -> "ValidationResult":
        """Create a failed validation result."""
        return cls(valid=False, errors=errors, warnings=warnings or [])


def validate_metadata(metadata: pb.WavetableMetadata) -> ValidationResult:
    """Validate wavetable metadata for structural correctness.

    This performs MUST-pass validation per the spec:
    - schema_version >= 1
    - num_frames > 0
    - num_mip_levels > 0
    - mip_frame_lengths has correct count
    - frame_length > 0

    Args:
        metadata: The protobuf metadata message to validate.

    Returns:
        ValidationResult with errors and warnings.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # Schema version check
    if metadata.schema_version < 1:
        errors.append(f"schema_version must be >= 1, got {metadata.schema_version}")

    # Frame length check
    if metadata.frame_length == 0:
        errors.append("frame_length must be > 0")
    elif not _is_power_of_two(metadata.frame_length):
        warnings.append(f"frame_length should be a power of 2, got {metadata.frame_length}")

    # Num frames check
    if metadata.num_frames == 0:
        errors.append("num_frames must be > 0")

    # Num mip levels check
    if metadata.num_mip_levels == 0:
        errors.append("num_mip_levels must be > 0")

    # Mip frame lengths check
    if len(metadata.mip_frame_lengths) != metadata.num_mip_levels:
        errors.append(
            f"mip_frame_lengths has {len(metadata.mip_frame_lengths)} entries, "
            f"expected {metadata.num_mip_levels}"
        )
    else:
        # Check mip_frame_lengths[0] == frame_length
        if metadata.mip_frame_lengths and metadata.mip_frame_lengths[0] != metadata.frame_length:
            errors.append(
                f"mip_frame_lengths[0] ({metadata.mip_frame_lengths[0]}) "
                f"must equal frame_length ({metadata.frame_length})"
            )

        # Check mip_frame_lengths are decreasing
        for i in range(1, len(metadata.mip_frame_lengths)):
            if metadata.mip_frame_lengths[i] > metadata.mip_frame_lengths[i - 1]:
                errors.append(
                    f"mip_frame_lengths must be decreasing, but "
                    f"mip_frame_lengths[{i}]={metadata.mip_frame_lengths[i]} > "
                    f"mip_frame_lengths[{i - 1}]={metadata.mip_frame_lengths[i - 1]}"
                )
                break

        # Check mip_frame_lengths are powers of 2
        for i, length in enumerate(metadata.mip_frame_lengths):
            if not _is_power_of_two(length):
                warnings.append(f"mip_frame_lengths[{i}] should be a power of 2, got {length}")

    # Wavetable type check
    if metadata.wavetable_type == pb.WAVETABLE_TYPE_UNSPECIFIED:
        warnings.append("wavetable_type is UNSPECIFIED, will be treated as CUSTOM")

    if errors:
        return ValidationResult.failure(errors, warnings)
    return ValidationResult.success(warnings)


def validate_audio_data(
    mipmaps: Sequence[NDArray[np.float32]],
    metadata: pb.WavetableMetadata,
) -> ValidationResult:
    """Validate audio data matches metadata declarations.

    This validates:
    - Number of mip levels matches metadata
    - Each mip level has correct total samples (frame_length * num_frames)
    - Samples are finite (no NaN/Inf)

    Args:
        mipmaps: List of numpy arrays, one per mip level.
                 Each array should be shape (total_samples,) or (num_frames, frame_length).
        metadata: The protobuf metadata that describes the expected structure.

    Returns:
        ValidationResult with errors and warnings.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # Check number of mip levels
    if len(mipmaps) != metadata.num_mip_levels:
        errors.append(f"Expected {metadata.num_mip_levels} mip levels, got {len(mipmaps)}")
        return ValidationResult.failure(errors, warnings)

    # Check each mip level
    for i, mipmap in enumerate(mipmaps):
        if i < len(metadata.mip_frame_lengths):
            expected_frame_length = metadata.mip_frame_lengths[i]
        else:
            expected_frame_length = 0
        expected_total = expected_frame_length * metadata.num_frames

        # Flatten to 1D for total sample count
        flat = mipmap.flatten()
        actual_total = len(flat)

        if actual_total != expected_total:
            errors.append(
                f"Mip level {i}: expected {expected_total} samples "
                f"({expected_frame_length} * {metadata.num_frames}), got {actual_total}"
            )

        # Check for NaN/Inf
        if not np.all(np.isfinite(flat)):
            nan_count = np.sum(np.isnan(flat))
            inf_count = np.sum(np.isinf(flat))
            errors.append(
                f"Mip level {i}: contains non-finite values ({nan_count} NaN, {inf_count} Inf)"
            )

        # Check sample range (warning only)
        if np.max(np.abs(flat)) > 1.0:
            max_abs = float(np.max(np.abs(flat)))
            warnings.append(
                f"Mip level {i}: samples exceed [-1, 1] range, max |sample| = {max_abs:.4f}"
            )

    if errors:
        return ValidationResult.failure(errors, warnings)
    return ValidationResult.success(warnings)


def calculate_expected_samples(metadata: pb.WavetableMetadata) -> int:
    """Calculate the expected total number of samples for a wavetable.

    Args:
        metadata: The wavetable metadata.

    Returns:
        Total expected sample count across all mip levels.
    """
    total = 0
    for frame_length in metadata.mip_frame_lengths:
        total += frame_length * metadata.num_frames
    return total


def _is_power_of_two(n: int) -> bool:
    """Check if a number is a power of two."""
    return n > 0 and (n & (n - 1)) == 0
