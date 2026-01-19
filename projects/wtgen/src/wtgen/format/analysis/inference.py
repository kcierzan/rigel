"""Wavetable type inference utilities.

This module provides tools for analyzing wavetable data and inferring
the appropriate wavetable type based on characteristics like frame size,
harmonic content, and source metadata.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from wtgen.format.analysis.harmonics import analyze_harmonic_content
from wtgen.format.types import WavetableType


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
        analysis["evidence"].append(
            f"High resolution profile: {frame_length} samples/frame"
        )
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
        analysis["evidence"].append(
            f"PCM sample profile: {num_frames} frame(s)"
        )
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
        "ppg", "wave", "waldorf", "microwave", "blofeld",
    ]
    if any(kw in hardware_lower for kw in classic_digital_keywords):
        return WavetableType.CLASSIC_DIGITAL

    # High-resolution digital
    hires_keywords = [
        "an1x", "nord", "serum", "vital", "zebra", "massive",
    ]
    if any(kw in hardware_lower for kw in hires_keywords):
        return WavetableType.HIGH_RESOLUTION

    # Vintage emulation targets
    vintage_keywords = [
        "oscar", "wasp", "edp", "casio", "cz",
    ]
    if any(kw in hardware_lower for kw in vintage_keywords):
        return WavetableType.VINTAGE_EMULATION

    # PCM sample sources
    pcm_keywords = [
        "sy99", "awm", "roland", "korg", "m1", "sample",
    ]
    if any(kw in hardware_lower for kw in pcm_keywords):
        return WavetableType.PCM_SAMPLE

    return None


def _is_classic_digital(
    frame_length: int,
    num_frames: int,
    bit_depth: int | None,
) -> bool:
    """Check if characteristics match classic digital wavetables."""
    # PPG-style: 256 samples, 64 frames, 8-bit
    if frame_length == 256 and num_frames == 64:
        return True
    # Variations: 128-512 samples, 32-128 frames
    if 128 <= frame_length <= 512 and 32 <= num_frames <= 128:
        if bit_depth is not None and bit_depth <= 12:
            return True
    return False


def _is_high_resolution(frame_length: int, num_frames: int) -> bool:
    """Check if characteristics match high-resolution wavetables."""
    # Modern synths: 1024+ samples, 64+ frames
    return frame_length >= 1024 or (frame_length >= 512 and num_frames >= 128)


def _is_vintage_emulation(
    frame_length: int,
    num_frames: int,
    wavetable: NDArray[np.float32],
) -> bool:
    """Check if characteristics match vintage emulation wavetables."""
    # Vintage style: few frames, simpler waveforms
    if num_frames > 16:
        return False

    # Check for simple waveform characteristics
    # (steep transitions, quantization artifacts)
    harmonic_info = analyze_harmonic_content(wavetable)
    return harmonic_info["has_aliasing_artifacts"]


def _is_pcm_sample(frame_length: int, num_frames: int) -> bool:
    """Check if characteristics match PCM sample wavetables."""
    # PCM samples: typically single cycle
    return num_frames <= 4


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
