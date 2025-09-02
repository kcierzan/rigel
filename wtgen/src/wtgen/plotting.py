import matplotlib
import numpy as np

matplotlib.use("kitcat")
import matplotlib.pyplot as plt
from numpy.typing import NDArray

from wtgen.dsp.mipmap import build_mipmap
from wtgen.dsp.process import align_to_zero_crossing
from wtgen.dsp.waves import WAVETABLE_SIZE, generate_sawtooth_wavetable

def plot_waveform(
    table: NDArray[np.floating], wave: NDArray[np.floating], title: str = "Waveform"
) -> None:
    """Plot a single waveform period optimized for iPython REPL use."""
    plt.close()
    plt.figure(figsize=(30, 8))
    plt.plot(table, wave)
    plt.title(title)
    plt.xlabel("Radians")
    plt.ylabel("Amplitude")
    plt.xlim(0, 2 * np.pi)
    plt.show()


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
    from wtgen.dsp.waves import generate_polyblep_sawtooth_wavetable

    t = np.linspace(0, 2 * np.pi, WAVETABLE_SIZE)

    phase_naive = (np.arange(WAVETABLE_SIZE) * frequency / WAVETABLE_SIZE) % 1.0
    naive_sawtooth = 2.0 * phase_naive - 1.0

    _, bandlimited_sawtooth = generate_polyblep_sawtooth_wavetable(frequency)

    plt.close()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(40, 20))

    ax1.plot(t, naive_sawtooth, label="Naive Sawtooth (Aliased)", alpha=0.7, linewidth=2)
    ax1.plot(
        t, bandlimited_sawtooth, label="Bandlimited Sawtooth (PolyBLEP)", alpha=0.9, linewidth=2
    )
    ax1.set_title(f"Sawtooth Comparison at {frequency} cycles - Full Waveform")
    ax1.set_xlabel("Radians")
    ax1.set_ylabel("Amplitude")
    ax1.set_xlim(0, 2 * np.pi)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    discontinuity_indices = []
    for i in range(1, len(phase_naive)):
        if phase_naive[i] < phase_naive[i - 1]:
            discontinuity_indices.append(i)

    if discontinuity_indices:
        disc_idx = discontinuity_indices[0]

        window_size = 50
        start_idx = max(0, disc_idx - window_size)
        end_idx = min(WAVETABLE_SIZE, disc_idx + window_size)

        zoom_t = t[start_idx:end_idx]
        zoom_naive = naive_sawtooth[start_idx:end_idx]
        zoom_bandlimited = bandlimited_sawtooth[start_idx:end_idx]

        ax2.plot(
            zoom_t,
            zoom_naive,
            label="Naive Sawtooth (Sharp Discontinuity)",
            alpha=0.7,
            linewidth=3,
            marker="o",
            markersize=3,
        )
        ax2.plot(
            zoom_t,
            zoom_bandlimited,
            label="Bandlimited Sawtooth (Smooth Transition)",
            alpha=0.9,
            linewidth=3,
            marker="s",
            markersize=2,
        )
        ax2.set_title(f"Zoomed Discontinuity View - Sample {disc_idx} (Phase Wrap)")
        ax2.set_xlabel("Radians")
        ax2.set_ylabel("Amplitude")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        ax2.axvline(
            x=t[disc_idx], color="red", linestyle="--", alpha=0.5, label="Discontinuity Point"
        )
    else:
        ax2.text(
            0.5,
            0.5,
            "No discontinuity found in this frequency range",
            ha="center",
            va="center",
            transform=ax2.transAxes,
            fontsize=16,
        )
        ax2.set_title("Zoomed Discontinuity View - No Discontinuity Found")

    plt.tight_layout()
    plt.show()


def plot_mipmaps(wave: NDArray[np.floating]) -> None:
    """Plot all mipmap levels for a given waveform."""
    table = np.linspace(0, 2 * np.pi, WAVETABLE_SIZE)
    mips = build_mipmap(wave)
    for i, mip in enumerate(mips):
        plot_waveform(table, align_to_zero_crossing(mip), f"Mipmap Level {i}")


def compare_mipmap_methods() -> None:
    """Compare different mipmap bandlimiting methods to show Gibbs phenomenon reduction."""
    _, base_sawtooth = generate_sawtooth_wavetable()

    brick_wall_mips = build_mipmap(base_sawtooth, num_octaves=3, rolloff_method="brick_wall")
    tukey_mips = build_mipmap(base_sawtooth, num_octaves=3, rolloff_method="tukey")
    blackman_mips = build_mipmap(base_sawtooth, num_octaves=3, rolloff_method="blackman")

    t = np.linspace(0, 2 * np.pi, WAVETABLE_SIZE)

    plt.close()
    fig, axes = plt.subplots(2, 2, figsize=(30, 20))

    axes[0, 0].plot(t, tukey_mips[2], label="tukey", linewidth=2, alpha=0.8)
    axes[0, 0].plot(t, blackman_mips[2], label="blackman", linewidth=2, alpha=0.8)
    axes[0, 0].set_title("Mipmap Level 2 - Time Domain Comparison")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    zoom_start, zoom_end = 1000, 1200
    axes[0, 1].plot(
        t[zoom_start:zoom_end],
        tukey_mips[2][zoom_start:zoom_end],
        label="Tukey",
        linewidth=3,
        marker="o",
        markersize=2,
    )
    axes[0, 1].plot(
        t[zoom_start:zoom_end],
        blackman_mips[2][zoom_start:zoom_end],
        label="Blackman",
        linewidth=3,
        marker="s",
        markersize=2,
    )
    axes[0, 1].set_title("Zoomed View - Gibbs Ripple Comparison")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    brick_fft = np.abs(np.fft.rfft(brick_wall_mips[2]))
    smooth_fft = np.abs(np.fft.rfft(blackman_mips[2]))
    freqs = np.fft.rfftfreq(len(brick_wall_mips[2]), 1.0)

    axes[1, 0].plot(freqs[:100], brick_fft[:100], label="Brick Wall Filter", linewidth=2)
    axes[1, 0].plot(freqs[:100], smooth_fft[:100], label="Smooth Rolloff", linewidth=2)
    axes[1, 0].set_title("Frequency Domain - Filter Response")
    axes[1, 0].set_xlabel("Normalized Frequency")
    axes[1, 0].set_ylabel("Magnitude")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    methods = ["tukey", "blackman", "raised_cosine", "hann", "none"]
    colors = ["blue", "red", "green", "yellow", "purple"]
    for i, (method, color) in enumerate(zip(methods, colors, strict=False)):
        mips = build_mipmap(base_sawtooth, num_octaves=3, rolloff_method=method)
        axes[1, 1].plot(
            t[zoom_start:zoom_end],
            mips[2][zoom_start:zoom_end],
            label=f'{method.replace("_", " ").title()}',
            color=color,
            linewidth=2,
            alpha=0.4,
        )

    axes[1, 1].set_title("Different Smooth Rolloff Methods")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
