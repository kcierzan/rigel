"""Wavetable type inference utilities.

This module provides tools for analyzing wavetable data and inferring
the appropriate wavetable type based on characteristics like frame size,
harmonic content, and source metadata.

## Classification Thresholds

The magic numbers used in this module are derived from historical
synthesizer specifications and industry conventions:

### Classic Digital Wavetables
- **256 samples, 64 frames**: PPG Wave 2.2/2.3 standard (1982-1983)
  - 8-bit resolution, 31.25kHz effective sample rate
  - Reference: "PPG Wave 2.2/2.3 Service Manual", Wolfgang Palm, 1983
- **128-512 samples, 32-128 frames**: Waldorf Microwave/Wave family (1989-1997)
  - Extended PPG format with variable table sizes
- **≤12-bit depth**: Characteristic of 1980s-1990s digital wavetable synths

### High Resolution Wavetables
- **≥1024 samples/frame**: Modern "supersaw" and complex waveform tables
  - Serum (2014+): 2048 samples default
  - Vital (2020+): 2048 samples
  - Reference: Common industry practice for alias-free synthesis
- **≥512 samples with ≥128 frames**: Extended morphing tables
  - Allows smooth morphing with high harmonic content

### Vintage Emulation
- **≤16 frames**: Characteristic of simple analog-style oscillators
  - EDP Wasp (1978): 8 waveforms
  - Casio CZ series (1984): 8 phase distortion shapes
- **Aliasing artifacts present**: Intentional non-band-limited waveforms

### PCM Sample
- **≤4 frames**: Single-cycle or minimal morphing waveforms
  - Common for sampled instrument tones
  - Roland D-50 (1987): Single-cycle PCM attack transients
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from wtgen.format.analysis.harmonics import analyze_harmonic_content
from wtgen.format.types import WavetableType

# =============================================================================
# Classification Threshold Constants
# =============================================================================

# Classic Digital: PPG Wave 2.2/2.3 standard (Wolfgang Palm, 1982)
# The PPG Wave used 256 samples per frame, 64 frames per wavetable, 8-bit resolution
PPG_STANDARD_FRAME_LENGTH = 256
PPG_STANDARD_NUM_FRAMES = 64

# Classic Digital: Waldorf Microwave/Wave family extensions
# Extended PPG format allowed 128-512 samples and 32-128 frames
CLASSIC_DIGITAL_MIN_FRAME_LENGTH = 128
CLASSIC_DIGITAL_MAX_FRAME_LENGTH = 512
CLASSIC_DIGITAL_MIN_FRAMES = 32
CLASSIC_DIGITAL_MAX_FRAMES = 128
CLASSIC_DIGITAL_MAX_BIT_DEPTH = 12  # 1980s-90s hardware limitation

# High Resolution: Modern synths (Serum, Vital, Zebra)
# Industry standard for alias-free, high-quality wavetable synthesis
HIGH_RES_MIN_FRAME_LENGTH = 1024  # Serum/Vital default: 2048
HIGH_RES_EXTENDED_FRAME_LENGTH = 512  # For large morph tables
HIGH_RES_EXTENDED_MIN_FRAMES = 128

# Vintage Emulation: Simple analog-style oscillators
# EDP Wasp: 8 waves, Casio CZ: 8 shapes, early digital: limited memory
VINTAGE_MAX_FRAMES = 16

# PCM Sample: Single-cycle or minimal morphing
# Common for sampled transients and simple waveforms
PCM_MAX_FRAMES = 4


def infer_wavetable_type(
    wavetable: NDArray[np.float32],
    source_bit_depth: int | None = None,
    source_hardware: str | None = None,
) -> tuple[WavetableType, dict[str, Any]]:
    """Infer the wavetable type from data characteristics.

    This function analyzes the wavetable data and returns the most appropriate
    WavetableType along with supporting evidence.

    Args:
        wavetable: 2D array of shape (num_frames, frame_length).
        source_bit_depth: Original bit depth if known.
        source_hardware: Source hardware name if known.

    Returns:
        Tuple of (WavetableType, analysis_dict) where analysis_dict contains
        the reasoning behind the classification.

    Example:
        >>> wt_type, analysis = infer_wavetable_type(data)
        >>> print(f"Type: {wt_type.name}, Confidence: {analysis['confidence']}")
    """
    num_frames, frame_length = wavetable.shape

    analysis: dict[str, Any] = {
        "num_frames": num_frames,
        "frame_length": frame_length,
        "source_bit_depth": source_bit_depth,
        "source_hardware": source_hardware,
        "evidence": [],
    }

    # Check for known hardware signatures
    if source_hardware:
        hw_type = _check_hardware_signature(source_hardware)
        if hw_type is not None:
            analysis["evidence"].append(f"Hardware signature match: {source_hardware}")
            analysis["confidence"] = "high"
            return hw_type, analysis

    # Classic Digital: PPG-style
    if _is_classic_digital(frame_length, num_frames, source_bit_depth):
        analysis["evidence"].append(
            f"Matches classic digital profile: {frame_length} samples, "
            f"{num_frames} frames, {source_bit_depth or 'unknown'}-bit"
        )
        analysis["confidence"] = "high" if source_bit_depth == 8 else "medium"
        return WavetableType.CLASSIC_DIGITAL, analysis

    # High Resolution: Modern synths
    if _is_high_resolution(frame_length, num_frames):
        analysis["evidence"].append(f"High resolution profile: {frame_length} samples/frame")
        harmonic_info = analyze_harmonic_content(wavetable)
        if harmonic_info["has_high_harmonics"]:
            max_harm = harmonic_info["estimated_max_harmonic"]
            analysis["evidence"].append(
                f"Rich harmonic content detected: up to {max_harm} harmonics"
            )
            analysis["confidence"] = "high"
        else:
            analysis["confidence"] = "medium"
        return WavetableType.HIGH_RESOLUTION, analysis

    # Vintage Emulation: Simple waveforms, few frames
    if _is_vintage_emulation(frame_length, num_frames, wavetable):
        analysis["evidence"].append(
            f"Vintage profile: {num_frames} frames, simple waveform structure"
        )
        analysis["confidence"] = "medium"
        return WavetableType.VINTAGE_EMULATION, analysis

    # PCM Sample: Single frame or very few frames
    if _is_pcm_sample(frame_length, num_frames):
        analysis["evidence"].append(f"PCM sample profile: {num_frames} frame(s)")
        analysis["confidence"] = "medium"
        return WavetableType.PCM_SAMPLE, analysis

    # Default to Custom
    analysis["evidence"].append("No specific type signature detected")
    analysis["confidence"] = "low"
    return WavetableType.CUSTOM, analysis


def _check_hardware_signature(hardware: str) -> WavetableType | None:
    """Check if hardware name matches known synth signatures."""
    hardware_lower = hardware.lower()

    # Classic digital synthesizers
    classic_digital_keywords = [
        "ppg",
        "wave",
        "waldorf",
        "microwave",
        "blofeld",
    ]
    if any(kw in hardware_lower for kw in classic_digital_keywords):
        return WavetableType.CLASSIC_DIGITAL

    # High-resolution digital
    hires_keywords = [
        "an1x",
        "nord",
        "serum",
        "vital",
        "zebra",
        "massive",
    ]
    if any(kw in hardware_lower for kw in hires_keywords):
        return WavetableType.HIGH_RESOLUTION

    # Vintage emulation targets
    vintage_keywords = [
        "oscar",
        "wasp",
        "edp",
        "casio",
        "cz",
    ]
    if any(kw in hardware_lower for kw in vintage_keywords):
        return WavetableType.VINTAGE_EMULATION

    # PCM sample sources
    pcm_keywords = [
        "sy99",
        "awm",
        "roland",
        "korg",
        "m1",
        "sample",
    ]
    if any(kw in hardware_lower for kw in pcm_keywords):
        return WavetableType.PCM_SAMPLE

    return None


def _is_classic_digital(
    frame_length: int,
    num_frames: int,
    bit_depth: int | None,
) -> bool:
    """Check if characteristics match classic digital wavetables.

    Detection is based on PPG Wave and Waldorf specifications:
    - PPG standard: 256 samples/frame, 64 frames (strict match)
    - Waldorf extensions: 128-512 samples, 32-128 frames with ≤12-bit depth

    See module docstring for detailed historical references.
    """
    # PPG Wave 2.2/2.3 exact specification
    if frame_length == PPG_STANDARD_FRAME_LENGTH and num_frames == PPG_STANDARD_NUM_FRAMES:
        return True

    # Waldorf Microwave/Wave family variations
    in_classic_frame_range = (
        CLASSIC_DIGITAL_MIN_FRAME_LENGTH <= frame_length <= CLASSIC_DIGITAL_MAX_FRAME_LENGTH
    )
    in_classic_count_range = CLASSIC_DIGITAL_MIN_FRAMES <= num_frames <= CLASSIC_DIGITAL_MAX_FRAMES

    if in_classic_frame_range and in_classic_count_range:
        # Require low bit depth to confirm classic digital origin
        if bit_depth is not None and bit_depth <= CLASSIC_DIGITAL_MAX_BIT_DEPTH:
            return True

    return False


def _is_high_resolution(frame_length: int, num_frames: int) -> bool:
    """Check if characteristics match high-resolution wavetables.

    Detection is based on modern synth standards:
    - Primary: ≥1024 samples/frame (Serum, Vital, Zebra standard)
    - Extended: ≥512 samples with ≥128 frames (large morph tables)

    See module docstring for detailed references.
    """
    # Modern high-resolution standard (Serum/Vital/etc.)
    if frame_length >= HIGH_RES_MIN_FRAME_LENGTH:
        return True

    # Extended morph tables with moderate resolution
    has_extended_frames = frame_length >= HIGH_RES_EXTENDED_FRAME_LENGTH
    has_many_frames = num_frames >= HIGH_RES_EXTENDED_MIN_FRAMES
    if has_extended_frames and has_many_frames:
        return True

    return False


def _is_vintage_emulation(
    frame_length: int,
    num_frames: int,
    wavetable: NDArray[np.float32],
) -> bool:
    """Check if characteristics match vintage emulation wavetables.

    Detection is based on early digital/analog hybrid synths:
    - Few frames (≤16): EDP Wasp, Casio CZ, early ROMs
    - Aliasing artifacts present: intentional non-band-limited character

    See module docstring for detailed references.
    """
    # Vintage synths had limited frame counts due to memory constraints
    if num_frames > VINTAGE_MAX_FRAMES:
        return False

    # Vintage emulations often preserve intentional aliasing
    harmonic_info = analyze_harmonic_content(wavetable)
    return harmonic_info["has_aliasing_artifacts"]


def _is_pcm_sample(frame_length: int, num_frames: int) -> bool:
    """Check if characteristics match PCM sample wavetables.

    Detection is based on single-cycle sample conventions:
    - Very few frames (≤4): single-cycle or minimal morphing
    - Common in ROM-based synths for attack transients

    See module docstring for detailed references.
    """
    return num_frames <= PCM_MAX_FRAMES


def suggest_type_metadata(
    wavetable_type: WavetableType,
    wavetable: NDArray[np.float32],
    source_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Suggest type-specific metadata based on analysis.

    Args:
        wavetable_type: The determined wavetable type.
        wavetable: The wavetable data.
        source_info: Optional source information dictionary.

    Returns:
        Dictionary with suggested type-specific metadata.
    """
    source_info = source_info or {}
    num_frames, frame_length = wavetable.shape
    harmonic_info = analyze_harmonic_content(wavetable)

    if wavetable_type == WavetableType.CLASSIC_DIGITAL:
        return {
            "original_bit_depth": source_info.get("bit_depth", 8),
            "source_hardware": source_info.get("hardware"),
            "harmonic_caps": _estimate_harmonic_caps(frame_length),
        }

    elif wavetable_type == WavetableType.HIGH_RESOLUTION:
        return {
            "max_harmonics": harmonic_info["estimated_max_harmonic"],
            "interpolation_hint": "cubic" if frame_length >= 2048 else "linear",
            "source_synth": source_info.get("hardware"),
        }

    elif wavetable_type == WavetableType.VINTAGE_EMULATION:
        return {
            "emulated_hardware": source_info.get("hardware"),
            "preserves_aliasing": harmonic_info["has_aliasing_artifacts"],
        }

    elif wavetable_type == WavetableType.PCM_SAMPLE:
        return {
            "original_sample_rate": source_info.get("sample_rate", 44100),
            "root_note": source_info.get("root_note", 60),  # Middle C
        }

    return {}


def _estimate_harmonic_caps(frame_length: int) -> list[int]:
    """Estimate harmonic caps for mip levels based on frame length."""
    caps = []
    current_length = frame_length

    while current_length >= 4:
        # Harmonics limited by Nyquist
        max_harmonics = current_length // 2
        caps.append(min(max_harmonics, 128))  # Cap at 128 for practical use
        current_length //= 2

    return caps
