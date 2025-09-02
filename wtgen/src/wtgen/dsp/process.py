import numpy as np
from numpy.typing import NDArray
from scipy import signal


def dc_remove(base_wavetable: NDArray[np.floating]) -> NDArray[np.floating]:
    """Remove DC component from a wavetable by subtracting the mean value.

    Args:
        base_wavetable: Input wavetable array containing audio samples

    Returns:
        Wavetable with DC component removed (zero mean)
    """
    return base_wavetable - np.mean(base_wavetable)


def _estimate_inter_sample_peak(wavetable: NDArray[np.floating]) -> float:
    """Estimate the true peak value accounting for inter-sample peaks.

    Uses upsampling and filtering to detect peaks that occur between samples
    which could cause clipping during playback with interpolation.

    Args:
        wavetable: Input wavetable to analyze

    Returns:
        Estimated true peak value (always >= sample peak)
    """
    if len(wavetable) < 2:
        return np.max(np.abs(wavetable)) if len(wavetable) > 0 else 0.0

    # Upsample by factor of 4 to catch inter-sample peaks
    oversample_factor = 4
    upsampled = signal.resample(wavetable, len(wavetable) * oversample_factor)

    # The true peak is the maximum of the upsampled version
    true_peak = np.max(np.abs(upsampled))
    sample_peak = np.max(np.abs(wavetable))

    # Return the larger of the two (true peak should always be >= sample peak)
    return max(true_peak, sample_peak)


def normalize(base_wavetable: NDArray[np.floating], peak: float = 0.999) -> NDArray[np.floating]:
    """Normalize wavetable amplitude to a specified peak level, preventing inter-sample clipping.

    This function accounts for inter-sample peaks that can occur during playback
    when using interpolation, ensuring the true peak (including interpolated values)
    does not exceed the target peak level.

    Args:
        base_wavetable: Input wavetable array to normalize
        peak: Target peak amplitude (default: 0.999 to avoid clipping)

    Returns:
        Normalized wavetable with specified peak amplitude, safe from inter-sample clipping
    """
    if len(base_wavetable) == 0:
        return base_wavetable

    true_peak = _estimate_inter_sample_peak(base_wavetable)
    if true_peak <= 1e-12:  # Handle near-zero signals (inclusive of boundary)
        return np.zeros_like(base_wavetable)
    return (base_wavetable / true_peak) * peak


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
