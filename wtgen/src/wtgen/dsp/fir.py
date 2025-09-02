from typing import Literal

import numpy as np
from numba import njit
from numpy.typing import NDArray
from scipy.signal import firwin

# Type aliases for clarity
WindowSpec = str | tuple[str, float]
ConvolveMode = Literal["full", "valid", "same"]


def lowpass_fir(
    cutoff_frequency: float,
    num_taps: int = 129,
    window_spec: WindowSpec = "kaiser",
    # ~60 dB stopband attenuation
    kaiser_beta: float = 8.6,
    sample_rate: float = 1.0,
) -> NDArray[np.floating]:
    """
    Create a lowpass FIR filter using scipy.signal.firwin.

    Args:
        cutoff_frequency: Cutoff frequency (0-1 for normalized frequency,
                          or actual freq if sample_rate specified)
        num_taps: Number of filter taps (filter length)
        window_spec: Window specification - string name or (name, parameter) tuple
        kaiser_beta: Kaiser window beta parameter for stopband attenuation
        sample_rate: Sampling frequency (1.0 for normalized)

    Returns:
        FIR filter coefficients as floating point array
    """
    if isinstance(window_spec, str) and window_spec.lower() == "kaiser":
        window_spec = ("kaiser", kaiser_beta)
    return firwin(
        numtaps=num_taps,
        cutoff=cutoff_frequency,
        window=window_spec,
        pass_zero="lowpass",
        fs=sample_rate,
    )


@njit
def _convolve_same_core(
    signal: NDArray[np.floating], kernel: NDArray[np.floating]
) -> NDArray[np.floating]:
    """Numba-optimized convolution with 'same' mode."""
    signal_len = len(signal)
    kernel_len = len(kernel)
    output_len = signal_len
    result = np.zeros(output_len, dtype=np.float64)

    # Calculate padding
    pad_left = kernel_len // 2

    for i in range(output_len):
        conv_sum = 0.0
        for j in range(kernel_len):
            signal_idx = i - pad_left + j
            if 0 <= signal_idx < signal_len:
                conv_sum += signal[signal_idx] * kernel[j]
        result[i] = conv_sum

    return result


@njit
def _normalize_signal_core(signal: NDArray[np.floating]) -> NDArray[np.floating]:
    """Numba-optimized signal normalization."""
    max_abs = 0.0
    for i in range(len(signal)):
        abs_val = abs(signal[i])
        max_abs = max(max_abs, abs_val)

    norm_factor = max_abs + 1e-12
    result = np.zeros_like(signal)
    for i in range(len(signal)):
        result[i] = signal[i] / norm_factor

    return result


def bandlimit_table(
    wavetable: NDArray[np.floating], cutoff_frequency: float
) -> NDArray[np.floating]:
    """
    Apply bandlimiting to a wavetable using FIR lowpass filtering with numba optimization.

    Args:
        wavetable: Input wavetable to be bandlimited
        cutoff_frequency: Normalized cutoff frequency (0-1)

    Returns:
        Bandlimited wavetable, renormalized to preserve peak amplitude
    """
    filter_coeffs: NDArray[np.floating] = lowpass_fir(cutoff_frequency=cutoff_frequency)

    # Use numba-optimized convolution for better performance
    NUMBA_THRESHOLD = 8192  # Size threshold for numba optimization
    if len(wavetable) < NUMBA_THRESHOLD:  # Use numba for smaller tables
        filtered_signal = _convolve_same_core(
            wavetable.astype(np.float64), filter_coeffs.astype(np.float64)
        )
        normalized_result = _normalize_signal_core(filtered_signal)
    else:  # Use numpy for very large tables
        filtered_signal_large: NDArray[np.floating] = np.convolve(
            wavetable, filter_coeffs, mode="same"
        )
        normalized_result = filtered_signal_large / (np.max(np.abs(filtered_signal_large)) + 1e-12)

    assert isinstance(normalized_result, np.ndarray), "Result must be NDArray"
    assert np.issubdtype(normalized_result.dtype, np.floating), "Result must contain floats"

    return normalized_result
