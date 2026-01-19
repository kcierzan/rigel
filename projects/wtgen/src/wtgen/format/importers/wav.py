"""High-resolution WAV wavetable import utilities.

This module handles importing standard WAV files that contain wavetable data.
Common sources include:
- Serum/Vital wavetable exports
- AN1x/Nord wavetable captures
- Other synth wavetable exports

The audio data is automatically split into frames based on specified parameters.
"""

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

from wtgen.format.analysis.inference import infer_wavetable_type
from wtgen.format.types import WavetableType


def import_hires_wav(
    path: Path | str,
    num_frames: int,
    frame_length: int | None = None,
    normalize: bool = True,
) -> tuple[list[NDArray[np.float32]], dict[str, Any]]:
    """Import a WAV file as a high-resolution wavetable.

    The WAV file should contain concatenated waveform frames.
    Either num_frames or frame_length can be specified to define
    how to split the audio data.

    Args:
        path: Path to the WAV file.
        num_frames: Number of frames to extract.
        frame_length: Samples per frame (auto-calculated if None).
        normalize: Whether to normalize audio to [-1, 1].

    Returns:
        Tuple of (mipmaps, metadata) where:
        - mipmaps: List with single mip level array (num_frames, frame_length)
        - metadata: Dictionary with inferred metadata

    Raises:
        ValueError: If audio cannot be evenly divided into frames.
        FileNotFoundError: If file doesn't exist.

    Example:
        >>> mipmaps, meta = import_hires_wav("serum_table.wav", num_frames=256)
        >>> # Auto-calculates frame_length from total samples / num_frames
    """
    path = Path(path)
    data, sample_rate = sf.read(path, dtype="float32")

    # Handle stereo by taking first channel
    if data.ndim > 1:
        data = data[:, 0]

    total_samples = len(data)

    # Calculate or validate frame length
    if frame_length is None:
        if total_samples % num_frames != 0:
            raise ValueError(
                f"Total samples ({total_samples}) is not evenly divisible "
                f"by num_frames ({num_frames}). Specify frame_length explicitly."
            )
        frame_length = total_samples // num_frames
    else:
        expected_samples = frame_length * num_frames
        if total_samples < expected_samples:
            raise ValueError(
                f"WAV has {total_samples} samples, but {expected_samples} "
                f"are required for {num_frames} frames * {frame_length} samples"
            )
        # Truncate to expected size
        data = data[:expected_samples]

    # Reshape to frames
    wavetable = data.reshape(num_frames, frame_length)

    # Normalize if requested
    if normalize:
        max_val = np.max(np.abs(wavetable))
        if max_val > 0:
            wavetable = wavetable / max_val

    wavetable = wavetable.astype(np.float32)

    # Determine source bit depth from file
    info = sf.info(path)
    bit_depth = _get_bit_depth_from_subtype(info.subtype)

    # Infer wavetable type using unified inference
    wavetable_type, _ = infer_wavetable_type(
        wavetable,
        source_bit_depth=bit_depth,
    )

    # Build metadata
    metadata: dict[str, Any] = {
        "wavetable_type": wavetable_type,
        "sample_rate": sample_rate,
        "source_bit_depth": bit_depth,
        "normalization_method": "peak" if normalize else "none",
    }

    # Add type-specific metadata
    if wavetable_type == WavetableType.HIGH_RESOLUTION:
        # Estimate max harmonics from frame length
        max_harmonics = frame_length // 2  # Nyquist limit
        metadata["type_metadata"] = {
            "max_harmonics": max_harmonics,
        }

    return [wavetable], metadata


def _get_bit_depth_from_subtype(subtype: str) -> int:
    """Get bit depth from soundfile subtype string."""
    subtype = subtype.upper()
    if "8" in subtype:
        return 8
    elif "16" in subtype:
        return 16
    elif "24" in subtype:
        return 24
    elif "32" in subtype or "FLOAT" in subtype:
        return 32
    return 16  # Default assumption


def detect_wav_wavetable(
    path: Path | str,
    possible_num_frames: list[int] | None = None,
) -> dict[str, Any]:
    """Analyze a WAV file and suggest wavetable parameters.

    This function examines a WAV file and suggests possible interpretations
    as a wavetable, based on total sample count and common wavetable dimensions.

    Args:
        path: Path to the WAV file.
        possible_num_frames: Frame counts to check (default: common values).

    Returns:
        Dictionary with analysis results and suggested parameters.

    Example:
        >>> info = detect_wav_wavetable("unknown.wav")
        >>> for suggestion in info["suggestions"]:
        ...     print(f"{suggestion['num_frames']} frames x {suggestion['frame_length']} samples")
    """
    path = Path(path)
    info = sf.info(path)
    data, sample_rate = sf.read(path, dtype="float32")

    # Handle stereo
    if data.ndim > 1:
        data = data[:, 0]

    total_samples = len(data)

    if possible_num_frames is None:
        possible_num_frames = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    # Find valid frame counts (exact divisions)
    suggestions = []
    for num_frames in possible_num_frames:
        if total_samples % num_frames == 0:
            frame_length = total_samples // num_frames
            # Check if frame_length is reasonable
            if 32 <= frame_length <= 8192:
                is_power_of_two = frame_length > 0 and (frame_length & (frame_length - 1)) == 0

                # Create temporary wavetable for type inference
                temp_wavetable = data[:num_frames * frame_length].reshape(num_frames, frame_length)
                temp_wavetable = temp_wavetable.astype(np.float32)
                wt_type, _ = infer_wavetable_type(temp_wavetable)

                suggestions.append({
                    "num_frames": num_frames,
                    "frame_length": frame_length,
                    "is_power_of_two": is_power_of_two,
                    "likely_type": wt_type.name,
                })

    # Sort by likelihood (prefer power-of-two frame lengths)
    suggestions.sort(key=lambda s: (not s["is_power_of_two"], s["num_frames"]))

    return {
        "file_info": {
            "path": str(path),
            "total_samples": total_samples,
            "sample_rate": sample_rate,
            "channels": info.channels,
            "duration_seconds": info.duration,
            "bit_depth": _get_bit_depth_from_subtype(info.subtype),
        },
        "suggestions": suggestions,
        "recommended": suggestions[0] if suggestions else None,
    }


def import_wav_with_mips(
    path: Path | str,
    num_frames: int,
    num_mip_levels: int = 7,
    normalize: bool = True,
) -> tuple[list[NDArray[np.float32]], dict[str, Any]]:
    """Import a WAV file and generate mip levels automatically.

    This is a convenience function that imports a WAV file and generates
    bandwidth-limited mip levels using simple decimation.

    For production use, consider using wtgen's proper mip generation
    which includes anti-aliasing filtering.

    Args:
        path: Path to the WAV file.
        num_frames: Number of frames to extract.
        num_mip_levels: Number of mip levels to generate.
        normalize: Whether to normalize audio to [-1, 1].

    Returns:
        Tuple of (mipmaps, metadata) where mipmaps has multiple levels.
    """
    # Import base level
    mipmaps, metadata = import_hires_wav(path, num_frames, normalize=normalize)
    base = mipmaps[0]

    # Generate additional mip levels
    current = base
    for _ in range(1, num_mip_levels):
        # Simple decimation (production code should low-pass filter first)
        if current.shape[1] < 8:  # Minimum frame length
            break
        decimated = current[:, ::2]  # Take every other sample
        mipmaps.append(decimated.astype(np.float32))
        current = decimated

    return mipmaps, metadata
