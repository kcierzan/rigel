"""Harmonic content analysis utilities.

This module provides FFT-based analysis tools for examining the harmonic
content of wavetable frames, useful for type classification and quality analysis.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray


def analyze_harmonic_content(
    wavetable: NDArray[np.float32],
    frame_index: int = 0,
) -> dict[str, Any]:
    """Analyze the harmonic content of a wavetable frame.

    Performs FFT analysis to determine harmonic characteristics
    that can help classify the wavetable type.

    Args:
        wavetable: 2D array of shape (num_frames, frame_length).
        frame_index: Which frame to analyze (default: first).

    Returns:
        Dictionary with harmonic analysis results including:
        - estimated_max_harmonic: Highest harmonic with significant energy
        - num_significant_harmonics: Count of harmonics above noise floor
        - spectral_density: Ratio of significant harmonics to Nyquist
        - has_high_harmonics: Whether harmonics exceed half Nyquist
        - has_aliasing_artifacts: Whether high-frequency energy suggests aliasing
        - high_freq_energy_ratio: Ratio of high-frequency energy to total
        - fundamental_frequency_bin: Detected fundamental frequency bin

    Example:
        >>> info = analyze_harmonic_content(wavetable)
        >>> print(f"Max harmonic: {info['estimated_max_harmonic']}")
    """
    if frame_index >= wavetable.shape[0]:
        frame_index = 0

    frame = wavetable[frame_index]
    frame_length = len(frame)

    # Compute FFT
    spectrum = np.abs(np.fft.rfft(frame))
    spectrum_normalized = spectrum / (np.max(spectrum) + 1e-10)

    # Find significant harmonics (above noise floor)
    noise_floor = 0.001
    significant_harmonics = np.where(spectrum_normalized > noise_floor)[0]

    max_harmonic = int(significant_harmonics[-1]) if len(significant_harmonics) > 0 else 0

    # Check for aliasing artifacts (high-frequency energy relative to frame size)
    nyquist = frame_length // 2
    high_freq_energy = np.sum(spectrum_normalized[nyquist * 3 // 4 :])
    total_energy = np.sum(spectrum_normalized)
    high_freq_ratio = high_freq_energy / (total_energy + 1e-10)

    # Detect if there are aliasing artifacts
    has_aliasing = high_freq_ratio > 0.1

    # Check spectral complexity
    num_significant = len(significant_harmonics)
    spectral_density = num_significant / nyquist

    return {
        "estimated_max_harmonic": max_harmonic,
        "num_significant_harmonics": num_significant,
        "spectral_density": spectral_density,
        "has_high_harmonics": max_harmonic > nyquist // 2,
        "has_aliasing_artifacts": has_aliasing,
        "high_freq_energy_ratio": high_freq_ratio,
        "fundamental_frequency_bin": _find_fundamental(spectrum_normalized),
    }


def _find_fundamental(spectrum: NDArray[np.float32]) -> int:
    """Find the fundamental frequency bin in a spectrum.

    Args:
        spectrum: Normalized spectrum array from FFT.

    Returns:
        Bin index of the fundamental frequency.
    """
    if len(spectrum) < 2:
        return 0

    # Skip DC component and find first significant peak
    for i in range(1, len(spectrum)):
        if spectrum[i] > 0.1:  # 10% of max
            return i

    return 1
