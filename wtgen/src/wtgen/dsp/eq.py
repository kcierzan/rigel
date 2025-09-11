"""
EQ-style filtering for wavetable mipmaps using FFT-based band filtering.

This module provides parametric EQ functionality that can be applied to mipmap
chains while preserving phase relationships and RMS levels.
"""

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from wtgen.dsp.process import align_to_zero_crossing
from wtgen.utils import EPSILON

# Type aliases
EQBand: TypeAlias = dict[str, float]
EQSettings: TypeAlias = list[EQBand]


def create_eq_band(
    frequency_hz: float,
    gain_db: float,
    q_factor: float = 1.0,
    sample_rate: float = 44100.0,
) -> EQBand:
    """
    Create an EQ band specification.

    Args:
        frequency_hz: Center frequency in Hz
        gain_db: Gain in dB (positive = boost, negative = cut)
        q_factor: Q factor (bandwidth control, higher = narrower)
        sample_rate: Sample rate in Hz (default: 44100)

    Returns:
        Dictionary containing EQ band parameters

    Examples:
        # Boost at 1000 Hz with 3dB gain and Q=2
        band = create_eq_band(1000.0, 3.0, 2.0)

        # Cut at 5000 Hz with -6dB gain and default Q
        band = create_eq_band(5000.0, -6.0)
    """
    nyquist_freq = sample_rate / 2.0

    if not 0.0 < frequency_hz < nyquist_freq:
        raise ValueError(f"Frequency must be between 0.0 and {nyquist_freq} Hz, got {frequency_hz}")

    # Convert Hz to normalized frequency (ratio of Nyquist)
    frequency_normalized = frequency_hz / nyquist_freq

    return {
        "frequency": frequency_normalized,
        "frequency_hz": frequency_hz,
        "gain_db": gain_db,
        "q_factor": q_factor,
        "sample_rate": sample_rate,
    }


def apply_parametric_eq_fft(
    wavetable: NDArray[np.floating],
    eq_bands: EQSettings,
    preserve_rms: bool = True,
    preserve_phase: bool = True,
    sample_rate: float = 44100.0,
) -> NDArray[np.floating]:
    """
    Apply parametric EQ to a wavetable using FFT-based filtering.

    This function applies multiple EQ bands to a wavetable while preserving
    key characteristics like RMS level and zero-crossing alignment.

    Args:
        wavetable: Input wavetable to process
        eq_bands: List of EQ band specifications
        preserve_rms: Whether to maintain original RMS level after EQ
        preserve_phase: Whether to preserve zero-crossing alignment
        sample_rate: Sample rate in Hz (default: 44100)

    Returns:
        EQ-processed wavetable

    Notes:
        - Frequency values in eq_bands should be in Hz
        - Q factor determines bandwidth: higher Q = narrower band
        - EQ is applied in frequency domain for phase-linear filtering
        - Original RMS and phase alignment are preserved by default
    """
    if len(wavetable) == 0:
        return wavetable.copy()

    if not eq_bands:
        return wavetable.copy()

    # Store original characteristics
    original_rms = np.sqrt(np.mean(wavetable**2))

    # Transform to frequency domain
    spectrum = np.fft.fft(wavetable.astype(np.complex128))
    N = len(spectrum)

    # Create frequency axis normalized to Nyquist
    freq_axis = np.fft.fftfreq(N)

    # Apply each EQ band
    for band in eq_bands:
        # Handle both old (normalized) and new (Hz) formats for backward compatibility
        if "frequency_hz" in band:
            nyquist_freq = sample_rate / 2.0
            freq_norm = band["frequency_hz"] / nyquist_freq
        else:
            freq_norm = band["frequency"]  # Backward compatibility

        gain_db = band["gain_db"]
        q_factor = band["q_factor"]

        # Skip if no gain change
        if abs(gain_db) < 1e-6:
            continue

        # Convert gain to linear scale
        gain_linear = 10.0 ** (gain_db / 20.0)

        # Calculate bandwidth from Q factor
        # Q = center_freq / bandwidth
        bandwidth = freq_norm / q_factor if q_factor > 0 else 0.1

        # Create bell-shaped filter response
        for i in range(N // 2 + 1):  # Only process positive frequencies + DC/Nyquist
            freq = abs(freq_axis[i])

            if freq == 0:  # Skip DC component
                continue

            # Calculate distance from center frequency
            freq_distance = abs(freq - freq_norm)

            # Bell curve response using Gaussian-like shape
            # More sophisticated than simple rectangular bands
            if bandwidth > 0:
                response = np.exp(-0.5 * (freq_distance / (bandwidth / 2)) ** 2)
            else:
                response = 1.0 if freq_distance < 0.01 else 0.0

            # Apply gain with smooth transition
            filter_gain = 1.0 + response * (gain_linear - 1.0)

            # Apply to spectrum (maintain conjugate symmetry)
            spectrum[i] *= filter_gain
            if i > 0 and i < N // 2:  # Don't double-apply to DC or Nyquist
                spectrum[N - i] *= filter_gain

    # Transform back to time domain
    eq_wavetable = np.fft.ifft(spectrum).real.astype(wavetable.dtype)

    # Preserve RMS level if requested
    if preserve_rms and original_rms > EPSILON:
        current_rms = np.sqrt(np.mean(eq_wavetable**2))
        if current_rms > EPSILON:
            rms_scale = original_rms / current_rms
            eq_wavetable *= rms_scale

    # Preserve phase alignment if requested
    if preserve_phase:
        eq_wavetable = align_to_zero_crossing(eq_wavetable)

    return eq_wavetable


def parse_eq_string(eq_string: str, sample_rate: float = 44100.0) -> EQSettings:
    """
    Parse EQ settings from a string format.

    Format: "freq1:gain1:q1,freq2:gain2:q2,..."
    Where:
    - freq: Frequency in Hz
    - gain: Gain in dB (can be negative)
    - q: Q factor (optional, defaults to 1.0)

    Args:
        eq_string: String representation of EQ bands
        sample_rate: Sample rate in Hz (default: 44100)

    Returns:
        List of EQ band specifications

    Examples:
        "1000:3.0:2.0,5000:-6.0:1.5"  # Two bands with custom Q
        "2000:2.5,8000:-3.0"          # Two bands with default Q

    Raises:
        ValueError: If string format is invalid
    """
    if not eq_string.strip():
        return []

    eq_bands = []

    try:
        for band_str in eq_string.split(","):
            parts = band_str.strip().split(":")

            if len(parts) < 2 or len(parts) > 3:
                raise ValueError(f"Invalid EQ band format: {band_str}")

            frequency_hz = float(parts[0])
            gain_db = float(parts[1])
            q_factor = float(parts[2]) if len(parts) == 3 else 1.0

            eq_bands.append(create_eq_band(frequency_hz, gain_db, q_factor, sample_rate))

    except (ValueError, IndexError) as e:
        raise ValueError(f"Error parsing EQ string '{eq_string}': {e}") from e

    return eq_bands


def create_high_tilt_eq(
    start_freq_ratio: float,
    tilt_db: float,
    sample_rate: float = 44100.0,
) -> NDArray[np.floating]:
    """
    Create a high-frequency tilt EQ filter.

    This creates a filter that applies a linear tilt to frequencies above
    the specified start frequency, reaching the full tilt amount at Nyquist.

    Args:
        start_freq_ratio: Start frequency as ratio of Nyquist (0.0 to 1.0)
        tilt_db: Amount of tilt in dB (positive = boost highs, negative = cut highs)
        sample_rate: Sample rate in Hz (default: 44100)

    Returns:
        Frequency domain filter response for FFT application

    Examples:
        # Boost highs starting at half Nyquist
        filter_response = create_high_tilt_eq(0.5, 6.0)

        # Cut highs starting at quarter Nyquist
        filter_response = create_high_tilt_eq(0.25, -3.0)
    """
    if not 0.0 <= start_freq_ratio <= 1.0:
        raise ValueError(f"start_freq_ratio must be between 0.0 and 1.0, got {start_freq_ratio}")

    # Create frequency axis normalized to Nyquist for half spectrum
    num_points = 1024  # Standard wavetable size
    freq_axis = np.fft.fftfreq(num_points)[: num_points // 2 + 1]

    # Initialize response to unity gain
    response = np.ones(len(freq_axis))

    # Apply tilt above start frequency
    for i, freq in enumerate(freq_axis):
        freq_norm = abs(freq)

        # Scale start_freq_ratio to match the normalized frequency range [0, 0.5]
        scaled_start_freq = start_freq_ratio * 0.5

        if freq_norm >= scaled_start_freq:
            # Linear tilt from start frequency to Nyquist (0.5 in normalized freq)
            if scaled_start_freq < 0.5:
                tilt_progress = (freq_norm - scaled_start_freq) / (0.5 - scaled_start_freq)
            else:
                tilt_progress = 0.0

            # Convert tilt dB to linear gain
            tilt_linear = 10.0 ** (tilt_db * tilt_progress / 20.0)
            response[i] = tilt_linear

    return response


def create_low_tilt_eq(
    start_freq_ratio: float,
    tilt_db: float,
    sample_rate: float = 44100.0,
) -> NDArray[np.floating]:
    """
    Create a low-frequency tilt EQ filter.

    This creates a filter that applies a linear tilt to frequencies below
    the specified start frequency, reaching the full tilt amount at DC.

    Args:
        start_freq_ratio: Start frequency as ratio of Nyquist (0.0 to 1.0)
        tilt_db: Amount of tilt in dB (positive = boost lows, negative = cut lows)
        sample_rate: Sample rate in Hz (default: 44100)

    Returns:
        Frequency domain filter response for FFT application

    Examples:
        # Boost lows below half Nyquist
        filter_response = create_low_tilt_eq(0.5, 4.0)

        # Cut lows below quarter Nyquist
        filter_response = create_low_tilt_eq(0.25, -2.0)
    """
    if not 0.0 <= start_freq_ratio <= 1.0:
        raise ValueError(f"start_freq_ratio must be between 0.0 and 1.0, got {start_freq_ratio}")

    # Create frequency axis normalized to Nyquist for half spectrum
    num_points = 1024  # Standard wavetable size
    freq_axis = np.fft.fftfreq(num_points)[: num_points // 2 + 1]

    # Initialize response to unity gain
    response = np.ones(len(freq_axis))

    # Apply tilt below start frequency
    for i, freq in enumerate(freq_axis):
        freq_norm = abs(freq)

        # Need to scale the start_freq_ratio since freq_axis goes to 0.5 (Nyquist)
        scaled_start_freq = start_freq_ratio * 0.5

        if freq_norm <= scaled_start_freq and freq_norm > 0:  # Skip DC
            # Linear tilt from DC to start frequency
            if scaled_start_freq > 0.0:
                tilt_progress = (scaled_start_freq - freq_norm) / scaled_start_freq
            else:
                tilt_progress = 0.0

            # Convert tilt dB to linear gain
            tilt_linear = 10.0 ** (tilt_db * tilt_progress / 20.0)
            response[i] = tilt_linear

    return response


def apply_tilt_eq_fft(
    wavetable: NDArray[np.floating],
    start_freq_ratio: float,
    tilt_db: float,
    tilt_type: str = "high",
    preserve_rms: bool = True,
    preserve_phase: bool = True,
    sample_rate: float = 44100.0,
) -> NDArray[np.floating]:
    """
    Apply tilt EQ to a wavetable using FFT-based filtering.

    Args:
        wavetable: Input wavetable to process
        start_freq_ratio: Start frequency as ratio of Nyquist (0.0 to 1.0)
        tilt_db: Amount of tilt in dB (positive = boost, negative = cut)
        tilt_type: Type of tilt - "high" or "low"
        preserve_rms: Whether to maintain original RMS level after EQ
        preserve_phase: Whether to preserve zero-crossing alignment
        sample_rate: Sample rate in Hz (default: 44100)

    Returns:
        Tilt-EQ processed wavetable

    Examples:
        # Boost highs starting at 0.3 * Nyquist
        tilted = apply_tilt_eq_fft(wave, 0.3, 6.0, "high")

        # Cut lows below 0.4 * Nyquist
        tilted = apply_tilt_eq_fft(wave, 0.4, -3.0, "low")
    """
    if len(wavetable) == 0:
        return wavetable.copy()

    if abs(tilt_db) < 1e-6:
        return wavetable.copy()

    if tilt_type not in ["high", "low"]:
        raise ValueError(f"tilt_type must be 'high' or 'low', got '{tilt_type}'")

    # Store original characteristics
    original_rms = np.sqrt(np.mean(wavetable**2))

    # Transform to frequency domain
    spectrum = np.fft.fft(wavetable.astype(np.complex128))
    N = len(spectrum)

    # Create tilt filter response
    if tilt_type == "high":
        tilt_response = create_high_tilt_eq(start_freq_ratio, tilt_db, sample_rate)
    else:
        tilt_response = create_low_tilt_eq(start_freq_ratio, tilt_db, sample_rate)

    # Resize filter response to match spectrum length
    if len(tilt_response) != N // 2 + 1:
        # Resample the filter response to match spectrum size
        original_freqs = np.linspace(0, 1, len(tilt_response))
        target_freqs = np.linspace(0, 1, N // 2 + 1)
        tilt_response = np.interp(target_freqs, original_freqs, tilt_response)

    # Apply tilt to positive frequencies
    for i in range(N // 2 + 1):
        spectrum[i] *= tilt_response[i]

        # Apply to negative frequencies (maintain conjugate symmetry)
        if i > 0 and i < N // 2:
            spectrum[N - i] *= tilt_response[i]

    # Transform back to time domain
    tilted_wavetable = np.fft.ifft(spectrum).real.astype(wavetable.dtype)

    # Preserve RMS level if requested
    if preserve_rms and original_rms > EPSILON:
        current_rms = np.sqrt(np.mean(tilted_wavetable**2))
        if current_rms > 1e-12:
            rms_scale = original_rms / current_rms
            tilted_wavetable *= rms_scale

    # Preserve phase alignment if requested
    if preserve_phase:
        tilted_wavetable = align_to_zero_crossing(tilted_wavetable)

    return tilted_wavetable


def analyze_eq_response(
    eq_bands: EQSettings,
    num_points: int = 512,
    sample_rate: float = 44100.0,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Calculate the frequency response of EQ settings.

    Args:
        eq_bands: EQ band specifications
        num_points: Number of frequency points to calculate
        sample_rate: Sample rate in Hz (default: 44100)

    Returns:
        Tuple of (frequencies_hz, magnitudes_db)

    Notes:
        Frequencies are returned in Hz
        Magnitudes are in dB
    """
    nyquist_freq = sample_rate / 2.0
    frequencies_hz = np.linspace(0, nyquist_freq, num_points)
    frequencies_norm = frequencies_hz / nyquist_freq  # Normalized for calculations
    magnitudes_db = np.zeros(num_points)

    for band in eq_bands:
        # Handle both old (normalized) and new (Hz) formats for backward compatibility
        if "frequency_hz" in band:
            freq_center_norm = band["frequency_hz"] / nyquist_freq
        else:
            freq_center_norm = band["frequency"]  # Backward compatibility

        gain_db = band["gain_db"]
        q_factor = band["q_factor"]

        bandwidth = freq_center_norm / q_factor if q_factor > 0 else 0.1

        for i, freq_norm in enumerate(frequencies_norm):
            if freq_norm == 0:  # Skip DC
                continue

            freq_distance = abs(freq_norm - freq_center_norm)

            # Bell curve response
            if bandwidth > 0:
                response = np.exp(-0.5 * (freq_distance / (bandwidth / 2)) ** 2)
            else:
                response = 1.0 if freq_distance < 0.01 else 0.0

            # Add this band's contribution
            magnitudes_db[i] += response * gain_db

    return frequencies_hz, magnitudes_db


def analyze_tilt_eq_response(
    start_freq_ratio: float,
    tilt_db: float,
    tilt_type: str = "high",
    num_points: int = 512,
    sample_rate: float = 44100.0,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Calculate the frequency response of tilt EQ settings.

    Args:
        start_freq_ratio: Start frequency as ratio of Nyquist (0.0 to 1.0)
        tilt_db: Amount of tilt in dB
        tilt_type: Type of tilt - "high" or "low"
        num_points: Number of frequency points to calculate
        sample_rate: Sample rate in Hz (default: 44100)

    Returns:
        Tuple of (frequencies_hz, magnitudes_db)
    """
    nyquist_freq = sample_rate / 2.0
    frequencies_hz = np.linspace(0, nyquist_freq, num_points)
    frequencies_norm = frequencies_hz / nyquist_freq
    magnitudes_db = np.zeros(num_points)

    # Create the appropriate tilt filter
    if tilt_type == "high":
        tilt_response = create_high_tilt_eq(start_freq_ratio, tilt_db, sample_rate)
    else:
        tilt_response = create_low_tilt_eq(start_freq_ratio, tilt_db, sample_rate)

    # Resample to match requested resolution
    if len(tilt_response) != num_points:
        original_freqs = np.linspace(0, 1, len(tilt_response))
        target_freqs = frequencies_norm
        response_resampled = np.interp(target_freqs, original_freqs, tilt_response)
    else:
        response_resampled = tilt_response

    # Convert linear response to dB
    magnitudes_db = 20.0 * np.log10(np.maximum(response_resampled, 1e-12))

    return frequencies_hz, magnitudes_db
