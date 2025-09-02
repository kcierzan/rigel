import numpy as np
from numpy.typing import NDArray


def dc_remove(base_wavetable: NDArray[np.floating]) -> NDArray[np.floating]:
    """Remove DC component from a wavetable by subtracting the mean value.

    Args:
        base_wavetable: Input wavetable array containing audio samples

    Returns:
        Wavetable with DC component removed (zero mean)
    """
    return base_wavetable - np.mean(base_wavetable)


def normalize(base_wavetable: NDArray[np.floating], peak: float = 0.999) -> NDArray[np.floating]:
    """Normalize wavetable amplitude to a specified peak level.

    Args:
        base_wavetable: Input wavetable array to normalize
        peak: Target peak amplitude (default: 0.999 to avoid clipping)

    Returns:
        Normalized wavetable with specified peak amplitude
    """
    max_amplitude = np.max(np.abs(base_wavetable))
    if max_amplitude <= 1e-12:  # Handle near-zero signals (inclusive of boundary)
        return np.zeros_like(base_wavetable)
    return (base_wavetable / max_amplitude) * peak


def normalize_to_range(
    wavetable: NDArray[np.floating], target_min: float = -0.999, target_max: float = 0.999
) -> NDArray[np.floating]:
    """Normalize wavetable to use the full specified range.

    This ensures the waveform uses the complete dynamic range available,
    with the minimum value scaled to target_min and maximum to target_max.

    Args:
        wavetable: Input wavetable to normalize
        target_min: Target minimum value (default: -0.999)
        target_max: Target maximum value (default: 0.999)

    Returns:
        Range-normalized wavetable
    """
    if len(wavetable) == 0:
        return wavetable

    current_min = np.min(wavetable)
    current_max = np.max(wavetable)
    current_range = current_max - current_min

    if current_range < 1e-12:
        # Constant signal - return zeros
        return np.zeros_like(wavetable)

    # Scale and shift to target range
    target_range = target_max - target_min
    normalized = (wavetable - current_min) / current_range * target_range + target_min

    # Remove any DC offset introduced by the range normalization
    normalized = normalized - np.mean(normalized)

    # Re-scale to maintain the target range after DC removal
    new_min = np.min(normalized)
    new_max = np.max(normalized)
    new_range = new_max - new_min

    if new_range > 1e-12:
        # Scale to use the target range while maintaining zero mean
        normalized = (normalized / new_range) * target_range

    return normalized


def align_to_zero_crossing(wavetable: NDArray[np.floating]) -> NDArray[np.floating]:
    """Align wavetable to start at a zero crossing with consistent phase.

    This ensures all wavetables start close to zero but maintains the
    natural phase relationships of the waveform.

    Args:
        wavetable: Input wavetable to align

    Returns:
        Wavetable aligned to start close to zero without phase inversion
    """
    if len(wavetable) < 2:
        return wavetable

    # Find the sample closest to zero, but limit search to prevent large phase shifts
    # Search the first half of the waveform for the best zero crossing
    search_range = len(wavetable) // 2
    min_value = float("inf")
    best_idx = 0

    for i in range(search_range):
        if abs(wavetable[i]) < min_value:
            min_value = abs(wavetable[i])
            best_idx = i

    # Only apply rotation if we found a significantly better starting point
    # This prevents unnecessary rotations when the current start is already good
    current_start_value = abs(wavetable[0])
    improvement_threshold = current_start_value * 0.8  # Must be at least 20% better

    if min_value < improvement_threshold:
        # Rotate the wavetable to start at the better zero crossing
        aligned_wavetable = np.roll(wavetable, -best_idx)
    else:
        # Keep original if no significant improvement is possible
        aligned_wavetable = wavetable.copy()

    return aligned_wavetable


def ensure_consistent_phase_alignment(
    mipmap_chain: list[NDArray[np.floating]],
) -> list[NDArray[np.floating]]:
    """Ensure all mipmap levels have consistent phase alignment.

    Aligns all levels to match the phase of the first level while keeping
    start values close to zero. The first level is used as-is to preserve
    the original processed wavetable phase.

    Args:
        mipmap_chain: List of mipmap levels to align

    Returns:
        Phase-aligned mipmap chain with consistent phase across all levels
    """
    if not mipmap_chain:
        return mipmap_chain

    aligned_chain = []

    # Use level 0 as-is to preserve the original processed wavetable phase
    # Don't apply additional alignment to level 0
    reference_level = mipmap_chain[0].copy()
    aligned_chain.append(reference_level)

    # For all other levels, find the rotation that maximizes correlation
    # with the reference level (level 0) to maintain phase consistency
    for level in mipmap_chain[1:]:
        best_correlation = -float("inf")
        best_shift = 0

        # Test different rotations to find the best phase match
        # Limit search to prevent large phase shifts
        max_shift = len(level) // 8  # Test up to 1/8 rotation

        for shift in range(-max_shift, max_shift + 1):
            shifted_level = np.roll(level, shift)

            # Calculate correlation with reference level (level 0)
            # Use the shorter length to handle different sized arrays
            subset_size = min(len(reference_level), len(shifted_level), 512)

            if subset_size > 1:  # Need at least 2 points for correlation
                try:
                    correlation = np.corrcoef(
                        reference_level[:subset_size], shifted_level[:subset_size]
                    )[0, 1]

                    # Handle NaN case (constant signals or other edge cases)
                    if np.isnan(correlation):
                        correlation = 0.0
                except (ValueError, IndexError):
                    # Handle any other edge cases
                    correlation = 0.0
            else:
                correlation = 0.0

            # Prioritize phase consistency - find the best correlation
            if correlation > best_correlation:
                best_correlation = correlation
                best_shift = shift

        # Apply the best shift found
        aligned_level = np.roll(level, best_shift)
        aligned_chain.append(aligned_level)

    return aligned_chain


def min_phase_realign(wavetable: NDArray[np.floating]) -> NDArray[np.floating]:
    """Convert wavetable to minimum phase representation using the Hilbert transform method.

    Minimum phase signals have all their energy concentrated at the beginning of the
    time domain representation, which can be beneficial for wavetable synthesis.

    Args:
        wavetable: Input wavetable array to convert to minimum phase

    Returns:
        Minimum phase version of the input wavetable, normalized
    """
    # Transform to frequency domain
    frequency_spectrum = np.fft.rfft(wavetable)

    # Compute log magnitude spectrum (add epsilon to avoid log(0))
    log_magnitude = np.log(np.abs(frequency_spectrum) + 1e-12)

    # Compute cepstrum (inverse FFT of log magnitude)
    cepstrum = np.fft.irfft(log_magnitude, n=len(wavetable))

    # Create minimum phase cepstrum by doubling positive time samples
    # and zeroing negative time samples (causal sequence)
    cepstrum[1:] *= 2  # Double positive time samples
    cepstrum[len(cepstrum) // 2 + 1 :] = 0  # Zero negative time samples

    # Reconstruct minimum phase spectrum
    min_phase_magnitude = np.exp(np.fft.rfft(cepstrum))
    min_phase_spectrum = min_phase_magnitude * np.exp(1j * np.angle(frequency_spectrum))

    # Transform back to time domain and normalize
    min_phase_wavetable = np.fft.irfft(min_phase_spectrum, n=len(wavetable))
    return normalize(min_phase_wavetable)
