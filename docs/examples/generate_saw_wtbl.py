#!/usr/bin/env python3
"""Generate a bandlimited mipmapped sawtooth wavetable in WTBL format.

This script demonstrates the complete workflow for creating wavetables
with wtgen and exporting them to the WTBL interchange format.
"""

import json
from pathlib import Path

import numpy as np

from wtgen.dsp.mipmap import Mipmap
from wtgen.dsp.process import align_to_zero_crossing, dc_remove, normalize
from wtgen.dsp.waves import WaveGenerator
from wtgen.types import WaveformType
from wtgen.format.types import (
    HighResolutionMetadata,
    InterpolationHint,
    NormalizationMethod,
    WavetableType,
)
from wtgen.format.writer import save_wavetable_wav


def generate_bandlimited_saw_wtbl(
    output_path: Path,
    frame_length: int = 2048,
    num_octaves: int = 8,
    rolloff_method: str = "raised_cosine",
) -> None:
    """Generate a bandlimited sawtooth wavetable and save as WTBL.

    Args:
        output_path: Output file path for the WTBL WAV file.
        frame_length: Samples per frame (power of 2).
        num_octaves: Number of mipmap levels.
        rolloff_method: FIR rolloff method for bandlimiting.
    """
    print(f"Generating bandlimited sawtooth wavetable...")
    print(f"  Frame length: {frame_length} samples")
    print(f"  Mipmap levels: {num_octaves}")
    print(f"  Rolloff method: {rolloff_method}")

    # 1. Generate base sawtooth waveform
    wave_gen = WaveGenerator()
    _, base_wave = wave_gen.generate(WaveformType.sawtooth, frequency=1.0)

    # Resize to target frame length if needed
    if len(base_wave) != frame_length:
        indices = np.linspace(0, len(base_wave) - 1, frame_length)
        base_wave = np.interp(indices, np.arange(len(base_wave)), base_wave)

    print(f"  Base waveform RMS: {np.sqrt(np.mean(base_wave**2)):.4f}")

    # 2. Create bandlimited mipmaps
    mipmap_gen = Mipmap(
        base_wavetable=base_wave,
        num_octaves=num_octaves,
        rolloff_method=rolloff_method,
        decimate=False,  # Keep same length for all levels
    )
    raw_mipmaps = mipmap_gen.generate()

    # 3. Process each mipmap level
    processed_mipmaps = []
    for i, mip in enumerate(raw_mipmaps):
        # Standard processing: remove DC, normalize, align to zero crossing
        processed = align_to_zero_crossing(dc_remove(normalize(mip)))
        # Reshape to 2D: (1 frame, frame_length samples)
        processed_2d = processed.reshape(1, -1).astype(np.float32)
        processed_mipmaps.append(processed_2d)
        print(f"  Mip level {i}: {len(mip)} samples, RMS={np.sqrt(np.mean(processed**2)):.4f}")

    # 4. Prepare generation parameters as JSON
    generation_params = json.dumps({
        "waveform": "sawtooth",
        "frequency": 1.0,
        "rolloff_method": rolloff_method,
        "decimate": False,
    })

    # 5. Save to WTBL format
    print(f"\nSaving to {output_path}...")
    save_wavetable_wav(
        path=output_path,
        mipmaps=processed_mipmaps,
        wavetable_type=WavetableType.HIGH_RESOLUTION,
        name="Ideal Sawtooth",
        author="wtgen",
        description="Bandlimited sawtooth wavetable with FIR-filtered mipmaps",
        normalization_method=NormalizationMethod.PEAK,
        tuning_reference=440.0,
        generation_parameters=generation_params,
        sample_rate=44100,
        type_metadata=HighResolutionMetadata(
            max_harmonics=frame_length // 2,
            interpolation_hint=InterpolationHint.LINEAR,
            source_synth="wtgen",
        ),
    )

    print(f"Successfully saved {output_path}")
    print(f"  File size: {output_path.stat().st_size} bytes")


if __name__ == "__main__":
    output = Path(__file__).parent / "ideal_saw.wav"
    generate_bandlimited_saw_wtbl(output)
