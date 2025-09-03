from typing import Any, TypeAlias

import numpy as np
from numpy import cos, pi
from numpy.typing import NDArray

from wtgen.dsp.process import align_to_zero_crossing

# Type alias for mipmap levels
MipmapLevel: TypeAlias = NDArray[np.float32]
MipmapChain: TypeAlias = list[MipmapLevel]


def build_mipmap(
    base_wavetable: NDArray[np.floating],
    num_octaves: int = 10,
    sample_rate: float = 44100.0,
    rolloff_method: str = "raised_cosine",
    decimate: bool = False,
) -> MipmapChain:
    """
    Build a mipmap chain from a base wavetable with smooth bandlimiting to avoid Gibbs phenomenon.

    Args:
        base_wavetable: Base wavetable for one cycle (full bandwidth)
        num_octaves: Number of octave levels to generate (default 10 for full MIDI range)
        sample_rate: Sample rate in Hz (default 44100.0)
        rolloff_method: Method for smooth rolloff ("raised_cosine", "tukey", "hann", "blackman",
            "brick_wall")
        decimate: Whether to decimate each level by half (2048→1024→512...)

    Returns:
        List of wavetables with progressively reduced bandwidth,
        using smooth rolloff to eliminate Gibbs phenomenon artifacts

    Notes:
        Uses smooth spectral domain rolloff for all levels to prevent both aliasing
        and Gibbs phenomenon artifacts across the complete MIDI range (0-127).

        Rolloff methods:
        - "raised_cosine": Smooth S-curve transition (recommended)
        - "tukey": Flat passband with cosine rolloff edges
        - "hann": Hann window rolloff
        - "blackman": Blackman window rolloff (smoothest, widest transition)
        - "brick_wall": Original hard cutoff method (for comparison)

        Level selection for playback should be:
        level = max(
            0, min(num_octaves, int(np.log2(fundamental_freq / (sample_rate / table_size))))
        )
    """
    table_size = float(len(base_wavetable))
    nyquist = sample_rate / 2

    # Target RMS level for consistent synthesizer behavior
    TARGET_RMS = 0.35  # This provides good headroom while maximizing signal level

    mipmap_levels: MipmapChain = []

    # Apply zero-crossing alignment to base wavetable first
    # This ensures the lowest mip starts at 0,0 ideally
    aligned_base = align_to_zero_crossing(base_wavetable)

    for octave_level in range(0, num_octaves + 1):
        # Improved frequency calculation for smoother transitions
        # Each level should handle frequencies up to a specific cutoff
        base_freq = sample_rate / table_size
        max_freq_in_level = base_freq * (2**octave_level)

        # Conservative safety margins to prevent aliasing at 44.1kHz
        # Higher sample rates provide more headroom but we prioritize safety
        if sample_rate <= 48000:
            # Conservative margins for standard sample rates, tested for critical MIDI notes
            if octave_level == 0:
                safety_margin = 0.78  # Tested safe for MIDI 21
            elif octave_level == 1:
                safety_margin = 0.65  # Tested safe for MIDI 36
            elif octave_level == 2:
                safety_margin = 0.62  # Tested safe for MIDI 48
            elif octave_level == 3:
                safety_margin = 0.65  # Conservative for level 3
            elif octave_level == 4:
                safety_margin = 0.60  # More conservative for level 4
            elif octave_level <= 6:
                safety_margin = 0.65  # Conservative for mid levels
            else:
                safety_margin = 0.70  # Conservative for higher levels
        else:
            # Higher sample rates: use conservative base margins but allow some scaling
            # The key insight is that MIDI frequencies don't scale with sample rate
            scale_factor = min(
                1.1, 0.9 + (sample_rate / 96000.0) * 0.2
            )  # Much more conservative scaling
            if octave_level == 0:
                safety_margin = min(0.80, 0.72 * scale_factor)  # Very conservative for level 0
            elif octave_level == 1:
                safety_margin = min(0.80, 0.65 * scale_factor)
            elif octave_level == 2:
                safety_margin = min(0.78, 0.62 * scale_factor)
            elif octave_level == 3:
                safety_margin = min(0.80, 0.65 * scale_factor)
            elif octave_level == 4:
                safety_margin = min(0.75, 0.60 * scale_factor)
            elif octave_level <= 6:
                safety_margin = min(0.80, 0.65 * scale_factor)
            else:
                safety_margin = min(0.85, 0.70 * scale_factor)

        # Calculate maximum safe harmonics for this level
        cutoff_freq = safety_margin * nyquist / max_freq_in_level
        max_safe_harmonics = int(cutoff_freq)
        max_safe_harmonics = max(1, max_safe_harmonics)  # Always keep at least fundamental

        # Apply smooth spectral domain bandlimiting
        spectrum = np.fft.fft(aligned_base)
        N = len(spectrum)

        # Apply smooth rolloff instead of brick wall filtering
        smooth_spectrum = _apply_smooth_rolloff(spectrum, max_safe_harmonics, N, rolloff_method)

        # Convert back to time domain
        bandlimited_level = np.fft.ifft(smooth_spectrum).real

        # Remove any DC offset that may have been introduced during filtering
        bandlimited_level = bandlimited_level - np.mean(bandlimited_level)

        # Apply zero-crossing alignment to maintain consistent phase across all levels
        # This ensures each mipmap level starts at a zero crossing
        bandlimited_level = align_to_zero_crossing(bandlimited_level)

        # Improved RMS normalization with clipping prevention
        current_rms = np.sqrt(np.mean(bandlimited_level**2))
        if current_rms > 1e-12:
            # First, find the maximum possible scale factor without clipping
            current_peak = np.max(np.abs(bandlimited_level))
            max_scale_for_no_clipping = 1.0 / current_peak if current_peak > 0 else 1.0

            # Calculate the scale factor needed for target RMS
            rms_scale_factor = TARGET_RMS / current_rms

            # Use the smaller of the two scale factors to prevent clipping
            # while getting as close to target RMS as possible
            final_scale_factor = min(rms_scale_factor, max_scale_for_no_clipping)
            normalized_level = bandlimited_level * final_scale_factor

            # If we had to reduce the scale factor to prevent clipping,
            # the final RMS will be lower than target, but this is necessary
            # to maintain waveform integrity
        else:
            # Handle silent waveforms
            normalized_level = bandlimited_level

        # Final safety clamp (should not be needed with improved logic above)
        normalized_level = np.clip(normalized_level, -1.0, 1.0)

        # Apply decimation if requested
        if decimate and octave_level > 0:
            # Calculate target size for this level (halve size each level)
            target_size = len(base_wavetable) // (2**octave_level)
            target_size = max(target_size, 64)  # Don't go below 64 samples

            if target_size < len(normalized_level):
                # Decimate by taking every Nth sample to preserve phase alignment
                # This maintains better phase relationships than interpolation
                decimation_factor = len(normalized_level) // target_size
                decimated_level = normalized_level[::decimation_factor][:target_size]

                # Apply zero-crossing alignment after decimation to ensure consistent phase
                decimated_level = align_to_zero_crossing(decimated_level)

                mipmap_levels.append(decimated_level.astype(np.float32))
            else:
                mipmap_levels.append(normalized_level.astype(np.float32))
        else:
            mipmap_levels.append(normalized_level.astype(np.float32))

    return mipmap_levels


def _apply_raised_cosine_rolloff(
    spectrum: NDArray[np.complexfloating], cutoff_harmonic: int, transition_bandwidth: int, N: int
) -> None:
    """
    Apply raised cosine rolloff to spectrum.

    Raised cosine rolloff provides a smooth S-curve transition that effectively
    eliminates Gibbs phenomenon artifacts. Mathematical form: 0.5 * (1 + cos(π * t))

    Args:
        spectrum: Complex FFT spectrum (modified in-place)
        cutoff_harmonic: Harmonic number where rolloff begins
        transition_bandwidth: Number of harmonics for the rolloff transition
        N: Length of spectrum
    """
    for k in range(1, N // 2):  # Process positive frequencies only
        if k <= cutoff_harmonic:
            # Passband - no attenuation
            continue
        elif k <= cutoff_harmonic + transition_bandwidth:
            # Transition band - raised cosine rolloff
            transition_pos = (k - cutoff_harmonic) / transition_bandwidth
            attenuation = 0.5 * (1.0 + cos(pi * transition_pos))
            spectrum[k] *= attenuation
            spectrum[N - k] *= attenuation  # Mirror for negative frequencies
        else:
            # Stopband - full attenuation
            spectrum[k] = 0
            spectrum[N - k] = 0


def _apply_tukey_rolloff(
    spectrum: NDArray[np.complexfloating], cutoff_harmonic: int, transition_bandwidth: int, N: int
) -> None:
    """
    Apply Tukey window rolloff to spectrum.

    Tukey window provides a flat passband with cosine rolloff, offering the
    sharpest transition while still being smooth. Uses cosine taper: cos²(π/2 * t)

    Args:
        spectrum: Complex FFT spectrum (modified in-place)
        cutoff_harmonic: Harmonic number where rolloff begins
        transition_bandwidth: Number of harmonics for the rolloff transition
        N: Length of spectrum
    """
    for k in range(1, N // 2):
        if k <= cutoff_harmonic:
            continue
        elif k <= cutoff_harmonic + transition_bandwidth:
            transition_pos = (k - cutoff_harmonic) / transition_bandwidth
            # Cosine taper: cos²(π/2 * t) for t from 0 to 1
            attenuation = cos(pi * transition_pos / 2.0) ** 2
            spectrum[k] *= attenuation
            spectrum[N - k] *= attenuation
        else:
            spectrum[k] = 0
            spectrum[N - k] = 0


def _apply_hann_rolloff(
    spectrum: NDArray[np.complexfloating], cutoff_harmonic: int, transition_bandwidth: int, N: int
) -> None:
    """
    Apply Hann window rolloff to spectrum.

    Hann window rolloff provides a good balance of smoothness and transition width.
    Uses Hann window: 0.5 * (1 + cos(π * t)) applied as rolloff.

    Args:
        spectrum: Complex FFT spectrum (modified in-place)
        cutoff_harmonic: Harmonic number where rolloff begins
        transition_bandwidth: Number of harmonics for the rolloff transition
        N: Length of spectrum
    """
    for k in range(1, N // 2):
        if k <= cutoff_harmonic:
            continue
        elif k <= cutoff_harmonic + transition_bandwidth:
            transition_pos = (k - cutoff_harmonic) / transition_bandwidth
            # Hann window: 0.5 * (1 + cos(π * t)) but applied as rolloff
            attenuation = 0.5 * (1.0 + cos(pi * transition_pos))
            spectrum[k] *= attenuation
            spectrum[N - k] *= attenuation
        else:
            spectrum[k] = 0
            spectrum[N - k] = 0


def _apply_blackman_rolloff(
    spectrum: NDArray[np.complexfloating], cutoff_harmonic: int, transition_bandwidth: int, N: int
) -> None:
    """
    Apply Blackman window rolloff to spectrum.

    Blackman window rolloff provides very smooth, minimal ripple with the widest
    transition band. Best for eliminating Gibbs phenomenon but with wider transition.

    Args:
        spectrum: Complex FFT spectrum (modified in-place)
        cutoff_harmonic: Harmonic number where rolloff begins
        transition_bandwidth: Number of harmonics for the rolloff transition
        N: Length of spectrum
    """
    for k in range(1, N // 2):
        if k <= cutoff_harmonic:
            continue
        elif k <= cutoff_harmonic + transition_bandwidth:
            transition_pos = (k - cutoff_harmonic) / transition_bandwidth
            # Blackman window coefficients for extremely smooth rolloff
            attenuation = (
                0.42 + 0.5 * cos(pi * transition_pos) + 0.08 * cos(2 * pi * transition_pos)
            )
            spectrum[k] *= attenuation
            spectrum[N - k] *= attenuation
        else:
            spectrum[k] = 0
            spectrum[N - k] = 0


def _apply_brick_wall_rolloff(
    spectrum: NDArray[np.complexfloating], cutoff_harmonic: int, transition_bandwidth: int, N: int
) -> None:
    """
    Apply brick wall (hard cutoff) rolloff to spectrum.

    Original brick wall method for comparison - causes Gibbs phenomenon.
    Provided mainly for legacy compatibility and comparison purposes.

    Args:
        spectrum: Complex FFT spectrum (modified in-place)
        cutoff_harmonic: Harmonic number where rolloff begins
        transition_bandwidth: Number of harmonics for rolloff transition (unused)
        N: Length of spectrum
    """
    for k in range(1, N // 2):
        if k > cutoff_harmonic:
            spectrum[k] = 0
            spectrum[N - k] = 0


def _apply_smooth_rolloff(
    spectrum: NDArray[np.complexfloating], cutoff_harmonic: int, N: int, method: str
) -> NDArray[np.complexfloating]:
    """
    Apply smooth rolloff to spectrum to avoid Gibbs phenomenon.

    Instead of abruptly cutting off harmonics (which causes ringing), this function
    applies smooth transitions that eliminate the sharp edges in frequency domain
    that cause Gibbs phenomenon artifacts in the time domain.

    Args:
        spectrum: Complex FFT spectrum
        cutoff_harmonic: Harmonic number where rolloff begins
        N: Length of spectrum
        method: Rolloff method to use

    Returns:
        Spectrum with smooth rolloff applied, free from Gibbs artifacts
    """
    smooth_spectrum = spectrum.copy()

    # Define transition bandwidth (how many harmonics for the rolloff)
    # Wider transitions = smoother rolloff = less Gibbs phenomenon
    transition_bandwidth = max(3, cutoff_harmonic // 3)  # At least 3, up to 33% of cutoff

    # Dispatch to appropriate rolloff method
    rolloff_methods = {
        "raised_cosine": _apply_raised_cosine_rolloff,
        "tukey": _apply_tukey_rolloff,
        "hann": _apply_hann_rolloff,
        "blackman": _apply_blackman_rolloff,
        "brick_wall": _apply_brick_wall_rolloff,
    }

    rolloff_function = rolloff_methods.get(method, _apply_brick_wall_rolloff)
    rolloff_function(smooth_spectrum, cutoff_harmonic, transition_bandwidth, N)

    return smooth_spectrum


def analyze_mipmap_quality(mipmap_chain: MipmapChain, method_name: str = "") -> dict[str, Any]:
    """
    Analyze the quality of a mipmap chain, detecting Gibbs phenomenon artifacts.

    Args:
        mipmap_chain: List of mipmap levels to analyze
        method_name: Name of the method used (for reporting)

    Returns:
        Dictionary with quality metrics including Gibbs artifact detection
    """
    results: dict[str, Any] = {
        "method": method_name,
        "levels": len(mipmap_chain),
        "gibbs_artifacts": [],
        "rms_levels": [],
        "peak_levels": [],
    }

    for level_idx, level in enumerate(mipmap_chain):
        # Calculate RMS and peak levels
        rms = np.sqrt(np.mean(level**2))
        peak = np.max(np.abs(level))

        results["rms_levels"].append(rms)
        results["peak_levels"].append(peak)

        # Detect Gibbs phenomenon by looking for high-frequency oscillations
        # Take the derivative to emphasize rapid changes
        derivative = np.diff(level)

        # Look for excessive high-frequency content (potential Gibbs artifacts)
        # High standard deviation in the derivative indicates oscillations/ripple
        derivative_std = np.std(derivative)
        derivative_mean = np.mean(np.abs(derivative))

        # Ratio of std to mean indicates oscillatory behavior
        # High ratio suggests Gibbs phenomenon artifacts
        oscillation_ratio = derivative_std / (derivative_mean + 1e-12)

        results["gibbs_artifacts"].append(
            {
                "level": level_idx,
                "oscillation_ratio": oscillation_ratio,
                "has_artifacts": oscillation_ratio > 3.0,  # Threshold for artifact detection
            }
        )

    return results
