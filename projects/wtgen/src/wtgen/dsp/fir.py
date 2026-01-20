from typing import Literal

import numpy as np
from numpy import cos, pi
from numpy.typing import NDArray

from wtgen.utils import assert_exhaustiveness

type RolloffMethod = Literal["raised_cosine", "tukey", "hann", "blackman", "brick_wall"]


class FIRFilter:
    def __init__(
        self,
        signal: NDArray[np.floating],
        cutoff_harmonic: int,
    ):
        """
        Initialize an FIR filter

        Args:
            signal: Input signal for filtering
            cutoff_harmonic: Harmonic number where the filtering will begin

        Returns:
            FIR filter object that exposes various filtering methods
        """
        # Frequency domain representation of the input signal
        self._spectrum = np.fft.fft(signal.copy())
        # N
        self._N = len(self._spectrum)
        # Harmonic where filtering will start
        self._cutoff_harmonic = cutoff_harmonic
        # Define transition bandwidth (how many harmonics for the rolloff)
        # Wider transitions = smoother rolloff = less Gibbs phenomenon
        # Here we keep at least 3 harmonics, up to 33% of cutoff
        self._transition_bandwidth = max(3, cutoff_harmonic // 3)

    def lowpass(
        self,
        method: RolloffMethod,
    ) -> NDArray[np.floating]:
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
        # Dispatch to appropriate rolloff method
        match method:
            case "raised_cosine":
                self._apply_raised_cosine_rolloff()
            case "tukey":
                self._apply_tukey_rolloff()
            case "hann":
                self._apply_hann_rolloff()
            case "blackman":
                self._apply_blackman_rolloff()
            case "brick_wall":
                self._apply_brick_wall_rolloff()
            case _:
                assert_exhaustiveness(method)

        # Convert back to time domain
        return np.fft.ifft(self._spectrum).real

    def _apply_raised_cosine_rolloff(self) -> None:
        """
        Apply raised cosine rolloff to spectrum.

        Raised cosine rolloff provides a smooth S-curve transition that effectively
        eliminates Gibbs phenomenon artifacts. Mathematical form: 0.5 * (1 + cos(π * t))
        """
        for k in range(1, self._N // 2):  # Process positive frequencies only
            if k <= self._cutoff_harmonic:
                # Passband - no attenuation
                continue
            elif k <= self._cutoff_harmonic + self._transition_bandwidth:
                # Transition band - raised cosine rolloff
                transition_pos = (k - self._cutoff_harmonic) / self._transition_bandwidth
                attenuation = 0.5 * (1.0 + cos(pi * transition_pos))
                self._spectrum[k] *= attenuation
                self._spectrum[self._N - k] *= attenuation  # Mirror for negative frequencies
            else:
                # Stopband - full attenuation
                self._spectrum[k] = 0
                self._spectrum[self._N - k] = 0

    def _apply_tukey_rolloff(self) -> None:
        """
        Apply Tukey window rolloff to spectrum.

        Tukey window provides a flat passband with cosine rolloff, offering the
        sharpest transition while still being smooth. Uses cosine taper: cos²(π/2 * t)

        Args:
            cutoff_harmonic: Harmonic number where rolloff begins
            transition_bandwidth: Number of harmonics for the rolloff transition
        """
        for k in range(1, self._N // 2):
            if k <= self._cutoff_harmonic:
                continue
            elif k <= self._cutoff_harmonic + self._transition_bandwidth:
                transition_pos = (k - self._cutoff_harmonic) / self._transition_bandwidth
                # Cosine taper: cos²(π/2 * t) for t from 0 to 1
                attenuation = cos(pi * transition_pos / 2.0) ** 2
                self._spectrum[k] *= attenuation
                self._spectrum[self._N - k] *= attenuation
            else:
                self._spectrum[k] = 0
                self._spectrum[self._N - k] = 0

    def _apply_hann_rolloff(self) -> None:
        """
        Apply Hann window rolloff to spectrum.

        Hann window rolloff provides a good balance of smoothness and transition width.
        Uses Hann window: 0.5 * (1 + cos(π * t)) applied as rolloff.

        Args:
            cutoff_harmonic: Harmonic number where rolloff begins
            transition_bandwidth: Number of harmonics for the rolloff transition
        """
        for k in range(1, self._N // 2):
            if k <= self._cutoff_harmonic:
                continue
            elif k <= self._cutoff_harmonic + self._transition_bandwidth:
                transition_pos = (k - self._cutoff_harmonic) / self._transition_bandwidth
                # Hann window: 0.5 * (1 + cos(π * t)) but applied as rolloff
                attenuation = 0.5 * (1.0 + cos(pi * transition_pos))
                self._spectrum[k] *= attenuation
                self._spectrum[self._N - k] *= attenuation
            else:
                self._spectrum[k] = 0
                self._spectrum[self._N - k] = 0

    def _apply_blackman_rolloff(self) -> None:
        """
        Apply Blackman window rolloff to spectrum.

        Blackman window rolloff provides very smooth, minimal ripple with the widest
        transition band. Best for eliminating Gibbs phenomenon but with wider transition.

        Args:
            cutoff_harmonic: Harmonic number where rolloff begins
            transition_bandwidth: Number of harmonics for the rolloff transition
        """
        for k in range(1, self._N // 2):
            if k <= self._cutoff_harmonic:
                continue
            elif k <= self._cutoff_harmonic + self._transition_bandwidth:
                transition_pos = (k - self._cutoff_harmonic) / self._transition_bandwidth
                # Blackman window coefficients for extremely smooth rolloff
                attenuation = (
                    0.42 + 0.5 * cos(pi * transition_pos) + 0.08 * cos(2 * pi * transition_pos)
                )
                self._spectrum[k] *= attenuation
                self._spectrum[self._N - k] *= attenuation
            else:
                self._spectrum[k] = 0
                self._spectrum[self._N - k] = 0

    def _apply_brick_wall_rolloff(self) -> None:
        """
        Apply brick wall (hard cutoff) rolloff to spectrum.

        Original brick wall method for comparison - causes Gibbs phenomenon.
        Provided mainly for legacy compatibility and comparison purposes.

        Args:
            cutoff_harmonic: Harmonic number where rolloff begins
            transition_bandwidth: Number of harmonics for rolloff transition (unused)
        """
        for k in range(1, self._N // 2):
            if k > self._cutoff_harmonic:
                self._spectrum[k] = 0
                self._spectrum[self._N - k] = 0
