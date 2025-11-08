import sys
from pathlib import Path
from typing import Annotated

import numpy as np
from cyclopts import App, Parameter
from rich.console import Console

from wtgen.cli.validators import (
    validate_eq_string,
    validate_power_of_two_integer,
    validate_tilt_string,
)
from wtgen.dsp.eq import Equalizer
from wtgen.dsp.fir import RolloffMethod
from wtgen.dsp.mipmap import Mipmap
from wtgen.dsp.process import align_to_zero_crossing, dc_remove, normalize
from wtgen.dsp.waves import WaveGenerator
from wtgen.export import (
    create_wavetable_metadata,
    handle_wavetable_export,
    load_wavetable_npz,
)
from wtgen.types import (
    BitDepth,
    ExportParams,
    HarmonicPartial,
    WaveformType,
)

app = App(name="wtgen", help="A utility for generating and analyzing wavetables")
console = Console()


def print_error(message: str) -> None:
    console.print(message, style="bold red")


def parse_partials_string(partials: str) -> list[HarmonicPartial]:
    """Parse partials string into HarmonicPartial objects."""
    partial_list = []
    for partial_str in partials.split(","):
        parts = partial_str.split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid partial format: {partial_str}")
        harmonic = int(parts[0])
        amplitude = float(parts[1])
        phase = float(parts[2])
        partial_list.append(HarmonicPartial(harmonic, amplitude, phase))
    return partial_list


@app.command
def generate(
    waveform: WaveformType = WaveformType.sawtooth,
    output: Path = Path("wavetable.npz"),
    octaves: int = 8,
    rolloff: RolloffMethod = "raised_cosine",
    frequency: float = 1.0,
    duty: float = 0.5,
    size: Annotated[int, Parameter(validator=validate_power_of_two_integer)] = 2048,
    decimate: bool = False,
    export_wav: bool = False,
    wav_dir: Path | None = None,
    wav_sample_rate: int = 44100,
    wav_bit_depth: BitDepth = 16,
    eq: Annotated[str | None, Parameter(validator=validate_eq_string)] = None,
    high_tilt: Annotated[str | None, Parameter(validator=validate_tilt_string)] = None,
    low_tilt: Annotated[str | None, Parameter(validator=validate_tilt_string)] = None,
) -> int:
    """
    Generate wavetable mipmaps and export to .npz or .wav format.

    Parameters
    ----------
    waveform: WaveformType
        The base wave shape for the wavetable
    output: Path
        The output destination for the .npz file
    octaves: int
        The number of mipmap octaves
    rolloff_method: RolloffMethod
        The rolloff method for FIR bandlimiting
    frequency: float
        The base frequency of the wavetable
    duty: float
        The duty cycle for square/pulse waveforms
    size: int
        The length of the wavetable in samples. Must be a power of 2.
    decimate: bool
        Whether to decimate the wavetable for each octave above base
    export_wav: bool
        Whether to export the wavetable also in .wav format
    wav_dir: Path | None
        The output directory of the .wav wavetables
    wav_sample_rate: int
        The sample rate for .wav wavetables in Hz
    wav_bit_depth: BitDepth
        The bit depths for .wav wavetables
    eq: str | None
        EQ settings in the format 'freq:gain:q,freq:gain:q'
        (freq in Hz, gain in dB, q=factor)
    high_tilt: str | None
        High-frequency tilt spectral shaper in the format 'start_ratio:gain'
        (ratio of Nyquist 0.0-1.0, gain in dB)
    low_tilt: str | None
        Low-frequency spectral tilt shaper in the format 'start_ratio:gain'
        (ratio of Nyquist 0.0-1.0, gain in dB)
    """
    export_params = ExportParams(
        export_wav=export_wav,
        wav_dir=wav_dir,
        wav_sample_rate=wav_sample_rate,
        wav_bit_depth=wav_bit_depth,
    )

    console.print(f"Generating {waveform} wavetable with {octaves} octaves...")

    wave_generator = WaveGenerator()

    _, wave = wave_generator.generate(
        waveform=waveform,
        frequency=frequency,
        duty=duty,
    )

    # Resize if different from default
    if len(wave) != size:
        # Simple resampling - could be improved with proper interpolation
        indices = np.linspace(0, len(wave) - 1, size)
        wave = np.interp(indices, np.arange(len(wave)), wave)

    # Apply processing
    equalizer = Equalizer(
        eq_settings=eq,
        high_tilt_settings=high_tilt,
        low_tilt_settings=low_tilt,
        sample_rate=44100.0,
    )
    wave = equalizer.apply(wave)

    # Build mipmaps
    console.print(f"Building mipmaps with [cyan bold]{rolloff}[/] rolloff...")
    mipmaps = Mipmap(
        base_wavetable=wave,
        num_octaves=octaves,
        rolloff_method=rolloff,
        decimate=decimate,
    ).generate()

    # Process each mipmap level
    processed_mipmaps = []
    for i, mip in enumerate(mipmaps):
        # Apply standard processing pipeline
        processed = align_to_zero_crossing(dc_remove(normalize(mip)))
        processed_mipmaps.append(processed)
        console.print(f"Processed mipmap level {i}: RMS={np.sqrt(np.mean(processed**2)):.3f}")

    # Export
    console.print(f"Exporting to {output}...")

    tables = {"base": processed_mipmaps}
    meta = create_wavetable_metadata(
        name=f"{waveform}_wavetable",
        waveform=waveform,
        octaves=octaves,
        rolloff=rolloff,
        size=size,
        decimate=decimate,
        sample_rate=export_params.wav_sample_rate,
        frequency=frequency,
        duty=duty,
    )

    _, message = handle_wavetable_export(output, tables, meta, export_params)
    console.print(message)
    return 0


@app.command
def harmonic(
    output: Path = Path("harmonic_wavetable.npz"),
    partials: str | None = "1:1.0:0.0",
    octaves: int = 8,
    rolloff: RolloffMethod = "raised_cosine",
    size: Annotated[int, Parameter(validator=validate_power_of_two_integer)] = 2048,
    decimate: bool = False,
    export_wav: bool = False,
    wav_dir: Path | None = None,
    wav_sample_rate: int = 44100,
    wav_bit_depth: int = 16,
    eq: str | None = None,
    high_tilt: str | None = None,
    low_tilt: str | None = None,
) -> int:
    """
    Generate wavetable from harmonic partials.

    Parameters
    ----------
    output: Path
        Output path for the .npz wavetable (default: harmonic_wavetable.npz)
    partials: str | None
        Harmonic partials as 'h1:a1:p1,h2:a2:p2...'
        where h is harmonic index, a is amplitude and p is phase
    rolloff: RolloffMethod
        Rolloff method for FIR bandlimiting
    size: int
        The length of the wavetable in samples. Must be a power of 2 (default: 2048)
    decimate: bool
        Whether to decimate the wavetable for each octave above base (default: False)
    export_wav: bool
        Whether to export the wavetable also in .wav format
    wav_dir: Path | None
        The output directory of the .wav wavetables (default: <output>.wav)
    wav_sample_rate: int
        The sample rate for .wav wavetables in Hz (default: 44100)
    wav_bit_depth: BitDepth
        The bit depths for .wav wavetables (default: 16)
    eq: str | None
        EQ settings in the format 'freq:gain:q,freq:gain:q'
        (freq in Hz, gain in dB, q=factor)
    high_tilt: str | None
        High-frequency tilt spectral shaper in the format 'start_ratio:gain'
        (ratio of Nyquist 0.0-1.0, gain in dB)
    low_tilt: str | None
        Low-frequency spectral tilt shaper in the format 'start_ratio:gain'
        (ratio of Nyquist 0.0-1.0, gain in dB)
    """

    export_params = ExportParams(
        export_wav=export_wav,
        wav_dir=wav_dir,
        wav_sample_rate=wav_sample_rate,
        wav_bit_depth=wav_bit_depth,
    )

    # Parse partials
    if partials:
        partial_list = parse_partials_string(partials)
        harmonic_partials = [(p.harmonic, p.amplitude, p.phase) for p in partial_list]
    else:
        # Default: sawtooth-like harmonics
        harmonic_partials = [(i, 1.0 / i, 0.0) for i in range(1, 17)]

    console.print(f"Generating wavetable from {len(harmonic_partials)} harmonics...")

    # Generate base waveform from harmonics
    wave_generator = WaveGenerator()
    wave = wave_generator.harmonics_to_table(harmonic_partials, size)

    # Apply processing
    equalizer = Equalizer(
        eq_settings=eq,
        high_tilt_settings=high_tilt,
        low_tilt_settings=low_tilt,
        sample_rate=44100.0,
    )
    wave = equalizer.apply(wave)

    # Build mipmaps
    console.print(f"Building mipmaps with {rolloff} rolloff...")
    mipmaps = Mipmap(
        base_wavetable=wave,
        num_octaves=octaves,
        rolloff_method=rolloff,
        decimate=decimate,
    ).generate()

    # Process each mipmap level
    processed_mipmaps = []
    for i, mip in enumerate(mipmaps):
        processed = align_to_zero_crossing(dc_remove(normalize(mip)))
        processed_mipmaps.append(processed)
        console.print(f"Processed mipmap level {i}: RMS={np.sqrt(np.mean(processed**2)):.3f}")

    # Export
    console.print(f"Exporting to {output}...")

    tables = {"base": processed_mipmaps}
    meta = create_wavetable_metadata(
        name="harmonic_wavetable",
        waveform="harmonic",
        octaves=octaves,
        rolloff=rolloff,
        size=size,
        decimate=decimate,
        sample_rate=export_params.wav_sample_rate,
        partials=harmonic_partials,
    )

    _, message = handle_wavetable_export(output, tables, meta, export_params)
    console.print(message)
    return 0


@app.command
def info(file: Path) -> int:
    """
    Display information about a wavetable file.

    Parameters
    ----------
    file: Path
        The path to the .npz wavetable
    """

    if not file.exists():
        print_error(f"Error: File {file} does not exist")
        raise ValueError("File does not exist")

    # Try new manifest format first
    wt_data = load_wavetable_npz(file)
    manifest = wt_data["manifest"]
    tables = wt_data["tables"]

    console.print(f"Wavetable: {file}")
    console.print(f"  Name: {manifest.get('name', 'unknown')}")
    console.print(f"  Version: {manifest.get('version', 'unknown')}")
    console.print(f"  Author: {manifest.get('author', 'unknown')}")
    console.print(f"  Sample rate: {manifest.get('sample_rate_hz', 'unknown')} Hz")
    console.print(f"  Base cycle length: {manifest.get('cycle_length_samples', 'unknown')}")

    if "generation" in manifest:
        gen = manifest["generation"]
        console.print(f"  Waveform: {gen.get('waveform', 'unknown')}")
        console.print(f"  Rolloff: {gen.get('rolloff', 'unknown')}")
        if gen.get("decimate"):
            console.print("  Decimated: Yes")

    # Show table information
    for table_info in manifest.get("tables", []):
        table_id = table_info["id"]
        mipmaps = tables.get(table_id, [])
        console.print(f"  Table '{table_id}': {len(mipmaps)} mipmap levels")

        # Show mipmap sizes and RMS levels
        console.print("    Mipmap levels:")
        for i, mip in enumerate(mipmaps):
            rms = np.sqrt(np.mean(mip**2))
            console.print(f"      Level {i}: size={len(mip)}, RMS={rms:.3f}")

    # Show statistics
    if "stats" in manifest:
        stats = manifest["stats"]
        console.print(f"  DC offset: {stats.get('dc_offset_mean', 'unknown'):.6f}")
        console.print(f"  Peak: {stats.get('peak', 'unknown'):.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(app())
