"""Raw PCM wavetable import utilities.

This module handles importing raw (headerless) PCM audio data as wavetables.
Common sources include:
- PPG Wave ROM dumps
- Vintage synth wavetable extractions
- Custom-exported binary wavetables

Supported bit depths: 8, 16, 24, 32 (integer or float)
"""

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from wtgen.format.analysis.inference import infer_wavetable_type
from wtgen.format.types import WavetableType


def import_raw_pcm(
    path: Path | str,
    frame_length: int,
    num_frames: int,
    bit_depth: int = 8,
    signed: bool = True,
    byte_order: str = "little",
    normalize: bool = True,
) -> tuple[list[NDArray[np.float32]], dict[str, Any]]:
    """Import raw PCM audio data as a wavetable.

    Args:
        path: Path to the raw PCM file.
        frame_length: Number of samples per frame.
        num_frames: Number of frames in the wavetable.
        bit_depth: Sample bit depth (8, 16, 24, or 32).
        signed: Whether samples are signed integers.
        byte_order: Byte order ("little" or "big").
        normalize: Whether to normalize audio to [-1, 1].

    Returns:
        Tuple of (mipmaps, metadata) where:
        - mipmaps: List with single mip level array (num_frames, frame_length)
        - metadata: Dictionary with inferred metadata

    Raises:
        ValueError: If file size doesn't match expected dimensions.
        FileNotFoundError: If file doesn't exist.

    Example:
        >>> mipmaps, meta = import_raw_pcm(
        ...     "ppg_wave.raw",
        ...     frame_length=256,
        ...     num_frames=64,
        ...     bit_depth=8
        ... )
        >>> # mipmaps[0].shape == (64, 256)
    """
    path = Path(path)
    data = path.read_bytes()

    expected_samples = frame_length * num_frames
    bytes_per_sample = (bit_depth + 7) // 8  # Round up for 24-bit
    expected_bytes = expected_samples * bytes_per_sample

    if len(data) < expected_bytes:
        raise ValueError(
            f"File has {len(data)} bytes, expected at least {expected_bytes} "
            f"for {num_frames} frames * {frame_length} samples * {bytes_per_sample} bytes"
        )

    # Convert bytes to numpy array
    samples = _decode_pcm_samples(data[:expected_bytes], bit_depth, signed, byte_order)

    # Reshape to frames
    wavetable = samples.reshape(num_frames, frame_length)

    # Normalize if requested
    if normalize:
        max_val = np.max(np.abs(wavetable))
        if max_val > 0:
            wavetable = wavetable / max_val

    wavetable = wavetable.astype(np.float32)

    # Infer wavetable type using unified inference
    wavetable_type, _ = infer_wavetable_type(
        wavetable,
        source_bit_depth=bit_depth,
    )

    # Build metadata
    metadata: dict[str, Any] = {
        "wavetable_type": wavetable_type,
        "source_bit_depth": bit_depth,
        "normalization_method": "peak" if normalize else "none",
    }

    # Add type-specific metadata
    if wavetable_type == WavetableType.CLASSIC_DIGITAL:
        metadata["type_metadata"] = {
            "original_bit_depth": bit_depth,
        }

    return [wavetable], metadata


def _decode_pcm_samples(
    data: bytes,
    bit_depth: int,
    signed: bool,
    byte_order: str,
) -> NDArray[np.float32]:
    """Decode raw PCM bytes to float samples."""
    endian = "<" if byte_order == "little" else ">"

    if bit_depth == 8:
        samples = _decode_8bit(data, signed)
    elif bit_depth == 16:
        samples = _decode_16bit(data, signed, endian)
    elif bit_depth == 24:
        samples = _decode_24bit(data, signed, byte_order)
    elif bit_depth == 32:
        samples = _decode_32bit(data, signed, endian)
    else:
        raise ValueError(f"Unsupported bit depth: {bit_depth}")

    return samples


def _decode_8bit(data: bytes, signed: bool) -> NDArray[np.float32]:
    """Decode 8-bit PCM samples."""
    if signed:
        samples = np.frombuffer(data, dtype=np.int8)
        return samples.astype(np.float32) / 128.0
    else:
        samples = np.frombuffer(data, dtype=np.uint8)
        return (samples.astype(np.float32) - 128.0) / 128.0


def _decode_16bit(data: bytes, signed: bool, endian: str) -> NDArray[np.float32]:
    """Decode 16-bit PCM samples."""
    dtype = np.dtype(f"{endian}i2") if signed else np.dtype(f"{endian}u2")
    samples = np.frombuffer(data, dtype=dtype)
    if signed:
        return samples.astype(np.float32) / 32768.0
    else:
        return (samples.astype(np.float32) - 32768.0) / 32768.0


def _decode_24bit(data: bytes, signed: bool, byte_order: str) -> NDArray[np.float32]:
    """Decode 24-bit PCM samples (requires manual unpacking)."""
    num_samples = len(data) // 3
    samples = np.zeros(num_samples, dtype=np.float32)

    for i in range(num_samples):
        idx = i * 3
        if byte_order == "little":
            value = data[idx] | (data[idx + 1] << 8) | (data[idx + 2] << 16)
        else:
            value = (data[idx] << 16) | (data[idx + 1] << 8) | data[idx + 2]

        if signed and value >= 0x800000:  # Sign extend
            value -= 0x1000000

        samples[i] = value / 8388608.0  # 2^23

    return samples


def _decode_32bit(data: bytes, signed: bool, endian: str) -> NDArray[np.float32]:
    """Decode 32-bit PCM samples (int32 or float32)."""
    dtype = np.dtype(f"{endian}i4") if signed else np.dtype(f"{endian}f4")
    samples = np.frombuffer(data, dtype=dtype)

    if signed:
        return samples.astype(np.float32) / 2147483648.0  # 2^31
    else:
        return samples.astype(np.float32)


def detect_raw_format(
    path: Path | str,
    possible_frame_lengths: list[int] | None = None,
    possible_num_frames: list[int] | None = None,
) -> dict[str, Any]:
    """Attempt to detect the format of a raw PCM file.

    This uses file size and common wavetable dimensions to guess
    the most likely format parameters.

    Args:
        path: Path to the raw file.
        possible_frame_lengths: Frame lengths to try (default: common values).
        possible_num_frames: Frame counts to try (default: common values).

    Returns:
        Dictionary with detected format parameters, or empty if no match.

    Example:
        >>> info = detect_raw_format("mystery.raw")
        >>> if info:
        ...     print(f"Detected: {info['frame_length']}x{info['num_frames']}")  # doctest: +SKIP
    """
    path = Path(path)
    file_size = path.stat().st_size

    if possible_frame_lengths is None:
        possible_frame_lengths = [64, 128, 256, 512, 1024, 2048]

    if possible_num_frames is None:
        possible_num_frames = [1, 4, 8, 16, 32, 64, 128, 256]

    # Try common bit depths
    for bit_depth in [8, 16, 32]:
        bytes_per_sample = bit_depth // 8

        for frame_length in possible_frame_lengths:
            for num_frames in possible_num_frames:
                expected_size = frame_length * num_frames * bytes_per_sample

                if file_size == expected_size:
                    return {
                        "frame_length": frame_length,
                        "num_frames": num_frames,
                        "bit_depth": bit_depth,
                        "total_samples": frame_length * num_frames,
                        "confidence": "exact_match",
                    }

    # No exact match found
    return {}
