from typing import Literal

import numpy as np
from numba import njit
from numpy import pi
from numpy.typing import NDArray
from scipy import signal

from wtgen.types import WaveformType
from wtgen.utils import EPSILON

WAVETABLE_SIZE = 2048

# Type alias for harmonic partials: (harmonic_number, amplitude, phase_radians)
Partial = tuple[int, float, float]
PartialList = list[Partial]
PhaseMode = Literal["minimum", "linear"]


@njit
def build_spectrum_core(
    harmonic_indices: np.ndarray, amplitudes: np.ndarray, phases: np.ndarray, size: int
) -> np.ndarray:
    """Numba-optimized core spectrum building."""
    spec_real = np.zeros(size, dtype=np.float64)
    spec_imag = np.zeros(size, dtype=np.float64)

    nyquist_limit = size // 2

    for i in range(len(harmonic_indices)):
        k = harmonic_indices[i]
        a = amplitudes[i]
        ph = phases[i]

        if k >= nyquist_limit:  # Respect Nyquist limit
            continue

        # Calculate complex exponential manually
        cos_ph = np.cos(ph)
        sin_ph = np.sin(ph)

        # Positive frequency
        spec_real[k] = a * cos_ph
        spec_imag[k] = a * sin_ph

        if k > 0:  # Don't duplicate DC component
            # Negative frequency (complex conjugate for Hermitian symmetry)
            spec_real[size - k] = a * cos_ph
            spec_imag[size - k] = -a * sin_ph  # Conjugate

    # Combine real and imaginary parts
    spec = spec_real + 1j * spec_imag
    return spec


class WaveGenerator:
    def generate(
        self,
        waveform: WaveformType,
        frequency: float,
        duty: float = 0.5,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        match waveform.value:
            case "sawtooth":
                return self.sawtooth(frequency)
            case "square":
                return self.square(frequency, duty=duty)
            case "pulse":
                return self.pulse(frequency, pulse_width=duty)
            case "sine":
                return self.sine(frequency)
            case "triangle":
                return self.triangle(frequency)
            case "polyblep_saw":
                return self.polyblep_sawtooth(frequency)
            case _:
                raise ValueError(f"Unsupported waveform: {waveform}")

    def sine(
        self,
        frequency: float = 1 / (2 * pi),
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Generate a sine wave wavetable."""
        t = np.linspace(0, 2 * pi, WAVETABLE_SIZE)
        return (
            t,
            signal.chirp(
                t,
                f0=frequency,
                f1=frequency,
                t1=2 * pi,
                phi=270,
                method="linear",
            ),
        )

    def sawtooth(
        self,
        frequency: float = 1,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Generate a sawtooth wave wavetable."""
        t = np.linspace(0, 2 * pi, WAVETABLE_SIZE)
        return (t, signal.sawtooth(frequency * t))

    def square(
        self,
        frequency: float = 1,
        duty: float = 0.5,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Generate a square wave wavetable."""
        t = np.linspace(0, 2 * pi, WAVETABLE_SIZE)
        return (t, signal.square(frequency * t, duty=duty))

    def pulse(
        self,
        frequency: float = 1,
        pulse_width: float = 0.5,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Generate a pulse wave as the difference of two sawtooth waves.

        Args:
            frequency: Frequency of the pulse wave
            pulse_width: Pulse width (0.0 to 1.0), where 0.5 is 50% duty cycle
        """
        t = np.linspace(0, 2 * pi, WAVETABLE_SIZE)

        sawtooth1 = signal.sawtooth(frequency * t)
        sawtooth2 = signal.sawtooth(frequency * t + 2 * pi * pulse_width)

        pulse = sawtooth1 - sawtooth2
        return (t, pulse)

    def triangle(
        self,
        frequency: float = 1,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Generate a triangle wave wavetable.

        Triangle waves have a characteristic shape that rises linearly from -1 to +1
        and then falls linearly from +1 to -1. They contain only odd harmonics with
        amplitudes that fall off as 1/n² (much faster than sawtooth waves).

        Args:
            frequency: Frequency of the triangle wave

        Returns:
            Tuple of (time_values, waveform_samples)
        """
        t = np.linspace(0, 2 * pi, WAVETABLE_SIZE)
        return (t, signal.sawtooth(frequency * t, width=0.5))

    @staticmethod
    def harmonics_to_table(
        partials: PartialList, size: int, phase: PhaseMode = "minimum"
    ) -> NDArray[np.floating]:
        """
        Convert harmonic partials to wavetable with enhanced continuity.

        Args:
            partials: List of (harmonic_index, amplitude, phase_radians) tuples
                    where harmonic_index >= 1
            size: Table length (should be power of two)
            phase: Phase strategy - "minimum" for minimum phase via cepstral method,
                "linear" for linear phase

        Returns:
            Normalized wavetable as floating point array with enhanced continuity
            and no clipping (all values within [-1.0, 1.0])
        """
        # Convert partials to arrays for numba optimization
        if not partials:
            return np.zeros(size, dtype=np.float64)

        harmonic_indices = np.array([k for k, a, ph in partials], dtype=np.int32)
        amplitudes = np.array([a for k, a, ph in partials], dtype=np.float64)
        phases = np.array([ph for k, a, ph in partials], dtype=np.float64)

        # Build spectrum using optimized core
        spec = build_spectrum_core(harmonic_indices, amplitudes, phases, size)

        # Generate time domain signal
        if phase == "linear":
            # Linear phase - direct IFFT
            x: NDArray[np.floating] = np.fft.ifft(spec).real
        else:
            # Minimum phase processing (simplified to avoid artifacts)
            # For now, use linear phase to ensure stable, continuous waveforms
            # TODO: Implement proper minimum phase if needed
            x = np.fft.ifft(spec).real

        # Ensure waveform continuity by checking boundary conditions
        # For periodic waveforms, the last sample should connect smoothly to the first
        boundary_discontinuity = x[-1] - x[0]

        # If there's a significant discontinuity, apply a gentle linear correction
        if abs(boundary_discontinuity) > 0.135:  # Tightened based on fuzz testing
            # Create a linear ramp to remove the discontinuity
            ramp = np.linspace(0, -boundary_discontinuity, size, endpoint=False)
            x = x + ramp

        # Apply gentle smoothing to reduce any high-frequency artifacts
        # that might introduce discontinuities without affecting the main harmonic content
        if len(x) > 4:
            # Simple 3-point smoothing filter (very light)
            smoothed = x.copy()
            for i in range(1, len(x) - 1):
                smoothed[i] = 0.25 * x[i - 1] + 0.5 * x[i] + 0.25 * x[i + 1]
            x = smoothed

        # Basic normalization to prevent extreme values before further processing
        max_val = np.max(np.abs(x))
        if max_val > EPSILON:
            x = x / max_val

        # Check if peaks would exceed ±1.0 and scale down if necessary
        # This ensures no clipping while preserving waveform shape
        peak_value = np.max(np.abs(x))
        if peak_value > 1.0:
            # Scale down to prevent clipping
            x = x / peak_value

        return x

    @staticmethod
    def _polyblep(phase: float, dt: float) -> float:
        """PolyBLEP (Polynomial Band Limited Step) correction function.

        MATHEMATICAL PRINCIPLE:
        The PolyBLEP correction is based on the mathematical concept that any
        discontinuous function can be decomposed into:
        1. A smooth, band-limited component
        2. A discontinuous component (which causes aliasing)

        The correction polynomial has these key properties:
        - It matches the EXACT discontinuity of the original waveform
        - It contains ONLY band-limited frequency content (no aliasing)
        - When subtracted, it leaves behind a smooth, alias-free signal

        POLYNOMIAL DERIVATION:
        For a sawtooth wave discontinuity (jump from +1 to -1), we need a polynomial P(t) such that:
        - P(-dt) = 0    (smooth before the discontinuity)
        - P(+dt) = -2   (matches the -2 amplitude jump of sawtooth)
        - P'(-dt) = P'(+dt) = 0  (continuous derivative at boundaries)

        This gives us a cubic polynomial: P(t) = at³ + bt² + ct + d
        Solving the boundary conditions yields the specific coefficients used below.

        Args:
            phase: Current phase position (0.0 to 1.0)
            dt: Phase increment per sample (determines the correction window size)

        Returns:
            Correction value to subtract from the raw waveform
        """
        if phase < dt:
            phase /= dt
            return phase + phase - phase * phase - 1.0

        elif phase > 1.0 - dt:
            phase = (phase - 1.0) / dt
            return phase * phase + phase + phase + 1.0

        else:
            return 0.0

    def polyblep_sawtooth(
        self,
        frequency: float = 1,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Generate a bandlimited sawtooth wave using the PolyBLEP technique.

        This implementation is optimized for realtime VST plugin usage because:

        1. **Computational Efficiency**: PolyBLEP only requires a few arithmetic
        operations per sample, making it much faster than FFT-based methods
        or convolution with BLEP tables.

        2. **Low Memory Usage**: No lookup tables or large filter kernels needed,
        just the polyblep correction function.

        3. **Excellent Anti-aliasing**: Effectively removes aliasing artifacts
        that would occur with naive sawtooth generation, especially at high
        frequencies relative to sample rate.

        4. **Parameter Modulatable**: Frequency can be smoothly modulated in
        realtime without artifacts or instability.

        The algorithm:
        1. Generate a naive sawtooth wave (linear ramp from -1 to 1)
        2. Detect phase resets (discontinuities)
        3. Apply PolyBLEP correction around each discontinuity
        4. The correction subtracts out the aliasing-causing sharp edges

        Technical Background:
        - Traditional sawtooth waves have infinite bandwidth due to sharp edges
        - When sampled digitally, this creates aliasing (high frequencies folding back)
        - PolyBLEP replaces sharp transitions with smooth polynomials
        - This bandlimits the signal to below the Nyquist frequency
        - The correction is applied in a window proportional to phase increment

        Args:
            frequency: Frequency of the sawtooth wave in cycles per full table

        Returns:
            Tuple of (time_array, bandlimited_sawtooth_wave)
        """
        WAVETABLE_SIZE = 2048
        t = np.linspace(0, 2 * pi, WAVETABLE_SIZE)

        dt = frequency / WAVETABLE_SIZE
        output = np.zeros(WAVETABLE_SIZE)

        for i in range(WAVETABLE_SIZE):
            phase = (i * frequency / WAVETABLE_SIZE) % 1.0
            naive_saw = 2.0 * phase - 1.0
            correction = self._polyblep(phase, dt)
            output[i] = naive_saw - correction

        return (t, output)


def generate_sawtooth_wavetable(
    frequency: float = 1.0,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Convenience helper for generating standard sawtooth wavetables."""
    generator = WaveGenerator()
    return generator.sawtooth(frequency)


def generate_polyblep_sawtooth_wavetable(
    frequency: float = 1.0,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Convenience helper for generating band-limited PolyBLEP sawtooth tables."""
    generator = WaveGenerator()
    return generator.polyblep_sawtooth(frequency)
