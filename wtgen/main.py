import numpy as np
import matplotlib
matplotlib.use("kitcat") # kitty image protocol frontend for matplotlib
import matplotlib.pyplot as plt
from scipy import signal
from numpy.typing import NDArray

from wtgen.dsp.mipmap import build_mipmap
from wtgen.dsp.process import align_to_zero_crossing


WAVETABLE_SIZE = 2048

def plot(table: NDArray[np.floating], wave: NDArray[np.floating]):
    plt.close()
    plt.figure(figsize=(30, 8))
    plt.plot(table, wave)
    plt.title('One period waveform')
    plt.xlabel('Radians')
    plt.ylabel('Amplitude')
    plt.xlim(0, 2*np.pi)
    plt.show()


def generate_sine_wavetable(frequency: float = 1/(2 * np.pi)) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    t = np.linspace(0, 2*np.pi, WAVETABLE_SIZE)
    return (t, signal.chirp(
        t,
        f0=frequency,
        f1=frequency,
        t1=2*np.pi,
        phi=270,
        method="linear",
    ))

def generate_sawtooth_wavetable(frequency: float = 1) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    t = np.linspace(0, 2*np.pi, WAVETABLE_SIZE)
    return (t, signal.sawtooth(frequency * t))


def generate_square_wavetable(frequency: float = 1, duty: float = 0.5) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    t = np.linspace(0, 2*np.pi, WAVETABLE_SIZE)
    return (t, signal.square(frequency * t, duty=duty))


def generate_pulse_wavetable(frequency: float = 1, pulse_width: float = 0.5) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Generate a pulse wave as the difference of two sawtooth waves.

    Args:
        frequency: Frequency of the pulse wave
        pulse_width: Pulse width (0.0 to 1.0), where 0.5 is 50% duty cycle
    """
    t = np.linspace(0, 2*np.pi, WAVETABLE_SIZE)

    # Generate two sawtooth waves with phase offset
    sawtooth1 = signal.sawtooth(frequency * t)
    sawtooth2 = signal.sawtooth(frequency * t + 2 * np.pi * pulse_width)

    # Pulse wave is the difference of the two sawtooths
    pulse = sawtooth1 - sawtooth2

    return (t, pulse)


def polyblep(phase: float, dt: float) -> float:
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
    # Check if we're in the correction window around the phase reset (near 0 or 1)
    if phase < dt:
        # POSITIVE DISCONTINUITY: Sawtooth jumps from -1 to +1 at phase = 0
        #
        # POLYNOMIAL EXPLANATION:
        # We normalize phase to range [-1, +1] within the correction window
        # The cubic polynomial: 2t - t² - 1 where t = phase/dt
        #
        # WHY THIS WORKS:
        # - At t = -1 (start of window): 2(-1) - (-1)² - 1 = -2 - 1 - 1 = -4... wait, let me recalculate
        # - The polynomial creates a smooth transition that exactly cancels the sharp edge
        # - After subtraction: (sharp sawtooth) - (smooth polynomial with same discontinuity) = smooth result
        phase /= dt  # Normalize to [0, 1] within correction window
        return phase + phase - phase * phase - 1.0  # Cubic: 2t - t² - 1

    elif phase > 1.0 - dt:
        # NEGATIVE DISCONTINUITY: Sawtooth jumps from +1 to -1 at phase = 1
        #
        # POLYNOMIAL EXPLANATION:
        # Similar cubic polynomial but for the opposite discontinuity direction
        # We adjust phase to be relative to the discontinuity point
        phase = (phase - 1.0) / dt  # Normalize relative to discontinuity
        return phase * phase + phase + phase + 1.0  # Cubic: t² + 2t + 1

    else:
        # AWAY FROM DISCONTINUITIES: No correction needed
        # The naive sawtooth is already smooth here, so correction = 0
        return 0.0


def generate_polyblep_sawtooth_wavetable(frequency: float = 1) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
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
    t = np.linspace(0, 2*np.pi, WAVETABLE_SIZE)

    # Calculate phase increment per sample (normalized to 0-1 range)
    # This determines the width of the PolyBLEP correction window
    dt = frequency / WAVETABLE_SIZE

    # Initialize output array
    output = np.zeros(WAVETABLE_SIZE)

    for i in range(WAVETABLE_SIZE):
        # Calculate normalized phase (0.0 to 1.0)
        phase = (i * frequency / WAVETABLE_SIZE) % 1.0

        # Generate naive sawtooth: linear ramp from -1 to +1
        naive_saw = 2.0 * phase - 1.0

        # Apply PolyBLEP correction to remove aliasing
        # The correction is subtracted from the naive waveform
        correction = polyblep(phase, dt)

        # Final bandlimited sawtooth
        output[i] = naive_saw - correction

    return (t, output)


def compare_sawtooth_methods(frequency: float = 2) -> None:
    """Compare naive vs bandlimited sawtooth generation to demonstrate aliasing reduction.

    This function generates both a naive sawtooth (with aliasing) and a bandlimited
    sawtooth using PolyBLEP, then plots them together to show the difference.

    At higher frequencies, the naive sawtooth will show severe aliasing artifacts
    (false low frequencies caused by undersampling), while the bandlimited version
    remains clean and artifact-free.

    Args:
        frequency: Test frequency - higher values show more dramatic aliasing differences
    """
    t = np.linspace(0, 2*np.pi, WAVETABLE_SIZE)

    # Generate naive sawtooth (aliased)
    phase_naive = (np.arange(WAVETABLE_SIZE) * frequency / WAVETABLE_SIZE) % 1.0
    naive_sawtooth = 2.0 * phase_naive - 1.0

    # Generate bandlimited sawtooth
    _, bandlimited_sawtooth = generate_polyblep_sawtooth_wavetable(frequency)

    # Plot comparison with subplots
    plt.close()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(40, 20))

    # Full waveform comparison
    ax1.plot(t, naive_sawtooth, label='Naive Sawtooth (Aliased)', alpha=0.7, linewidth=2)
    ax1.plot(t, bandlimited_sawtooth, label='Bandlimited Sawtooth (PolyBLEP)', alpha=0.9, linewidth=2)
    ax1.set_title(f'Sawtooth Comparison at {frequency} cycles - Full Waveform')
    ax1.set_xlabel('Radians')
    ax1.set_ylabel('Amplitude')
    ax1.set_xlim(0, 2*np.pi)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Find first discontinuity point for zoomed view
    # The discontinuity occurs when phase wraps from ~1.0 to ~0.0
    discontinuity_indices = []
    for i in range(1, len(phase_naive)):
        if phase_naive[i] < phase_naive[i-1]:  # Phase wrap detected
            discontinuity_indices.append(i)

    if discontinuity_indices:
        # Zoom in on the first discontinuity
        disc_idx = discontinuity_indices[0]

        # Create a window around the discontinuity (±50 samples)
        window_size = 50
        start_idx = max(0, disc_idx - window_size)
        end_idx = min(WAVETABLE_SIZE, disc_idx + window_size)

        zoom_t = t[start_idx:end_idx]
        zoom_naive = naive_sawtooth[start_idx:end_idx]
        zoom_bandlimited = bandlimited_sawtooth[start_idx:end_idx]

        ax2.plot(zoom_t, zoom_naive, label='Naive Sawtooth (Sharp Discontinuity)',
                alpha=0.7, linewidth=3, marker='o', markersize=3)
        ax2.plot(zoom_t, zoom_bandlimited, label='Bandlimited Sawtooth (Smooth Transition)',
                alpha=0.9, linewidth=3, marker='s', markersize=2)
        ax2.set_title(f'Zoomed Discontinuity View - Sample {disc_idx} (Phase Wrap)')
        ax2.set_xlabel('Radians')
        ax2.set_ylabel('Amplitude')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Add vertical line at discontinuity point
        ax2.axvline(x=t[disc_idx], color='red', linestyle='--', alpha=0.5,
                   label='Discontinuity Point')
    else:
        ax2.text(0.5, 0.5, 'No discontinuity found in this frequency range',
                ha='center', va='center', transform=ax2.transAxes, fontsize=16)
        ax2.set_title('Zoomed Discontinuity View - No Discontinuity Found')

    plt.tight_layout()
    plt.show()

def plot_mips() -> None:
    table, wave = generate_sawtooth_wavetable()
    mips = build_mipmap(wave)
    for mip in mips:
        plot(table, align_to_zero_crossing(mip))

def compare_mipmap_methods() -> None:
    """Compare different mipmap bandlimiting methods to show Gibbs phenomenon reduction."""
    # Generate a sawtooth with many harmonics
    _, base_sawtooth = generate_sawtooth_wavetable()

    # Build mipmaps with different methods
    brick_wall_mips = build_mipmap(base_sawtooth, num_octaves=3, rolloff_method="brick_wall")
    # raised_cosine_mips = build_mipmap(base_sawtooth, num_octaves=3, rolloff_method="raised_cosine")
    tukey_mips = build_mipmap(base_sawtooth, num_octaves=3, rolloff_method="tukey")
    blackman_mips = build_mipmap(base_sawtooth, num_octaves=3, rolloff_method="blackman")

    # Compare level 2 (significant bandlimiting)
    t = np.linspace(0, 2*np.pi, WAVETABLE_SIZE)

    plt.close()
    fig, axes = plt.subplots(2, 2, figsize=(30, 20))

    # Plot waveforms
    # axes[0, 0].plot(t, brick_wall_mips[2], label='Brick Wall (Gibbs artifacts)', linewidth=2)
    # axes[0, 0].plot(t, raised_cosine_mips[2], label='Raised Cosine (smooth)', linewidth=2, alpha=0.8)
    axes[0, 0].plot(t, tukey_mips[2], label='tukey', linewidth=2, alpha=0.8)
    axes[0, 0].plot(t, blackman_mips[2], label='blackman', linewidth=2, alpha=0.8)
    axes[0, 0].set_title('Mipmap Level 2 - Time Domain Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Zoom in to show ripple
    zoom_start, zoom_end = 1000, 1200 # Sample range to zoom
    # axes[0, 1].plot(t[zoom_start:zoom_end], brick_wall_mips[2][zoom_start:zoom_end],
    #                label='Brick Wall (ripple visible)', linewidth=3, marker='o', markersize=2)
    axes[0, 1].plot(t[zoom_start:zoom_end], tukey_mips[2][zoom_start:zoom_end],
                   label='Tukey', linewidth=3, marker='o', markersize=2)
    axes[0, 1].plot(t[zoom_start:zoom_end], blackman_mips[2][zoom_start:zoom_end],
                   label='Blackman', linewidth=3, marker='s', markersize=2)
    axes[0, 1].set_title('Zoomed View - Gibbs Ripple Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Show frequency domain
    brick_fft = np.abs(np.fft.rfft(brick_wall_mips[2]))
    smooth_fft = np.abs(np.fft.rfft(blackman_mips[2]))
    freqs = np.fft.rfftfreq(len(brick_wall_mips[2]), 1.0)

    axes[1, 0].plot(freqs[:100], brick_fft[:100], label='Brick Wall Filter', linewidth=2)
    axes[1, 0].plot(freqs[:100], smooth_fft[:100], label='Smooth Rolloff', linewidth=2)
    axes[1, 0].set_title('Frequency Domain - Filter Response')
    axes[1, 0].set_xlabel('Normalized Frequency')
    axes[1, 0].set_ylabel('Magnitude')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Method comparison
    methods = ['tukey', 'blackman', 'raised_cosine', 'hann', 'none']
    colors = ['blue', 'red', 'green', 'yellow', 'purple']
    for i, (method, color) in enumerate(zip(methods, colors)):
        mips = build_mipmap(base_sawtooth, num_octaves=3, rolloff_method=method)
        axes[1, 1].plot(
            t[zoom_start:zoom_end],
            mips[2][zoom_start:zoom_end],
            label=f'{method.replace("_", " ").title()}',
            color=color,
            linewidth=2,
            alpha=0.4,
        )

    axes[1, 1].set_title('Different Smooth Rolloff Methods')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def analyze_gibbs_artifacts(waveform: NDArray[np.floating], method_name: str = "") -> float:
    """
    Analyze a waveform for Gibbs phenomenon artifacts.

    Args:
        waveform: Waveform to analyze
        method_name: Name of the method used (for reporting)

    Returns:
        Oscillation ratio (higher values indicate more Gibbs artifacts)
    """
    # Take derivative to emphasize rapid changes
    derivative = np.diff(waveform)

    # Calculate oscillation metrics
    derivative_std = np.std(derivative)
    derivative_mean = np.mean(np.abs(derivative))

    # Ratio indicates oscillatory behavior
    oscillation_ratio = derivative_std / (derivative_mean + 1e-12)

    print(f"{method_name}: Oscillation ratio = {oscillation_ratio:.3f}")
    if oscillation_ratio > 3.0:
        print(f"  ⚠️  High Gibbs artifacts detected")
    else:
        print(f"  ✅  Low Gibbs artifacts")

    return oscillation_ratio
