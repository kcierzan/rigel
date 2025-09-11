from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from wtgen.dsp import process
from wtgen.dsp.fir import FIRFilter, RolloffMethod
from wtgen.utils import EPSILON

# Type alias for mipmap levels
MipmapLevel: TypeAlias = NDArray[np.float32]
MipmapChain: TypeAlias = list[MipmapLevel]

TARGET_RMS = 0.35  # This provides good headroom while maximizing signal level


class Mipmap:
    """
    Initialize a mipmap chain builder from a base wavetable with smooth bandlimiting.

    Args:
        base_wavetable: Base wavetable for one cycle (full bandwidth)
        num_octaves: Number of octave levels to generate (default 10 for full MIDI range)
        sample_rate: Sample rate in Hz (default 44100.0)
        rolloff_method: Method for smooth FIR window rolloff of LP filter
        decimate: Whether to decimate level by half (2018->1024->512...)
    """

    def __init__(
        self,
        base_wavetable: NDArray[np.floating],
        num_octaves: int = 10,
        sample_rate: float = 44100.0,
        rolloff_method: RolloffMethod = "raised_cosine",
        decimate: bool = False,
    ):
        self._base_wavetable = base_wavetable
        self._num_octaves = num_octaves
        self._sample_rate = sample_rate
        self._rolloff_method: RolloffMethod = rolloff_method
        self._decimate = decimate
        self._mipmap_levels: MipmapChain = []

    def generate(self) -> MipmapChain:
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
        # Apply zero-crossing alignment to base wavetable first
        # This ensures the lowest mip starts at 0,0 ideally
        aligned_base = process.align_to_zero_crossing(self._base_wavetable)

        for octave_level in range(0, self._num_octaves + 1):
            max_safe_harmonics = self._safe_cutoff_harmonics(octave_level=octave_level)
            fir = FIRFilter(signal=aligned_base, cutoff_harmonic=max_safe_harmonics)
            bandlimited_level = fir.lowpass(method=self._rolloff_method)

            # Remove any DC offset that may have been introduced during filtering
            bandlimited_level = bandlimited_level - np.mean(bandlimited_level)

            # Apply zero-crossing alignment to maintain consistent phase across all levels
            # This ensures each mipmap level starts at a zero crossing
            bandlimited_level = process.align_to_zero_crossing(bandlimited_level)
            normalized_level = self._smart_normalize(input_signal=bandlimited_level)

            self._maybe_decimate(input_signal=normalized_level, octave_level=octave_level)

        return self._mipmap_levels

    def _safe_cutoff_harmonics(self, octave_level: int) -> int:
        # Improved frequency calculation for smoother transitions
        # Each level should handle frequencies up to a specific cutoff
        nyquist = self._sample_rate / 2
        table_size = float(len(self._base_wavetable))
        base_freq = self._sample_rate / table_size
        max_freq_in_level = base_freq * (2**octave_level)

        # Conservative safety margins to prevent aliasing at 44.1kHz
        # Higher sample rates provide more headroom but we prioritize safety
        if self._sample_rate <= 48000:
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
                1.1, 0.9 + (self._sample_rate / 96000.0) * 0.2
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
        return max(1, max_safe_harmonics)  # Always keep at least fundamental

    def _smart_normalize(self, input_signal: NDArray[np.floating]) -> NDArray[np.floating]:
        # Target RMS level for consistent synthesizer behavior
        # Improved RMS normalization with clipping prevention
        current_rms = np.sqrt(np.mean(input_signal**2))
        if current_rms > EPSILON:
            # First, find the maximum possible scale factor without clipping
            current_peak = np.max(np.abs(input_signal))
            max_scale_for_no_clipping = (
                1.0 / process.estimate_inter_sample_peak(input_signal) if current_peak > 0 else 1.0
            )

            # Calculate the scale factor needed for target RMS
            rms_scale_factor = TARGET_RMS / current_rms

            # Use the smaller of the two scale factors to prevent clipping
            # while getting as close to target RMS as possible
            final_scale_factor = min(rms_scale_factor, max_scale_for_no_clipping)
            normalized_level = input_signal * final_scale_factor

            # If we had to reduce the scale factor to prevent clipping,
            # the final RMS will be lower than target, but this is necessary
            # to maintain waveform integrity
        else:
            # Handle silent waveforms
            normalized_level = input_signal
        # Final safety clamp (should not be needed with improved logic above)
        return np.clip(normalized_level, -1.0, 1.0)

    def _maybe_decimate(
        self,
        input_signal: NDArray[np.floating],
        octave_level: int,
        minimum_level_size: int = 64,  # Don't go below 64 samples
    ) -> None:
        # Apply decimation if requested
        if self._decimate and octave_level > 0:
            # Calculate target size for this level (halve size each level)
            target_size = len(self._base_wavetable) // (2**octave_level)
            target_size = max(target_size, minimum_level_size)

            if target_size < len(input_signal):
                # Decimate by taking every Nth sample to preserve phase alignment
                # This maintains better phase relationships than interpolation
                decimation_factor = len(input_signal) // target_size
                decimated_level = input_signal[::decimation_factor][:target_size]

                # Apply zero-crossing alignment after decimation to ensure consistent phase
                decimated_level = process.align_to_zero_crossing(decimated_level)

                self._mipmap_levels.append(decimated_level.astype(np.float32))
            else:
                self._mipmap_levels.append(input_signal.astype(np.float32))
        else:
            self._mipmap_levels.append(input_signal.astype(np.float32))

    @staticmethod
    def analyze_quality(mipmap_chain: MipmapChain, method_name: str = "") -> dict[str, Any]:
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
            oscillation_ratio = derivative_std / (derivative_mean + EPSILON)

            results["gibbs_artifacts"].append(
                {
                    "level": level_idx,
                    "oscillation_ratio": oscillation_ratio,
                    "has_artifacts": oscillation_ratio > 3.0,  # Threshold for artifact detection
                }
            )

        return results
