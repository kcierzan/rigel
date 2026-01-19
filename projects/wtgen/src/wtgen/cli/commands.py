import json
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
from wtgen.export import save_mipmaps_as_wav
from wtgen.format import (
    HighResolutionMetadata,
    NormalizationMethod,
    WavetableType,
    load_wavetable_wav,
    save_wavetable_wav,
)
from wtgen.types import (
    BitDepth,
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
    output: Path = Path("wavetable.wav"),
    octaves: int = 8,
    rolloff: RolloffMethod = "raised_cosine",
    frequency: float = 1.0,
    duty: float = 0.5,
    size: Annotated[int, Parameter(validator=validate_power_of_two_integer)] = 2048,
    decimate: bool = False,
    export_individual_wavs: bool = False,
    wav_dir: Path | None = None,
    wav_sample_rate: int = 44100,
    wav_bit_depth: BitDepth = 16,
    eq: Annotated[str | None, Parameter(validator=validate_eq_string)] = None,
    high_tilt: Annotated[str | None, Parameter(validator=validate_tilt_string)] = None,
    low_tilt: Annotated[str | None, Parameter(validator=validate_tilt_string)] = None,
) -> int:
    """
    Generate wavetable mipmaps and export to WTBL format (WAV with embedded metadata).

    Parameters
    ----------
    waveform: WaveformType
        The base wave shape for the wavetable
    output: Path
        The output destination for the .wav WTBL file
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
    export_individual_wavs: bool
        Whether to also export individual .wav files per mipmap level
    wav_dir: Path | None
        The output directory of the individual .wav wavetables
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

    # Export to WTBL format
    console.print(f"Exporting to {output}...")

    # Reshape mipmaps to 2D arrays (num_frames=1, frame_length)
    mipmaps_2d = [mip.reshape(1, -1).astype(np.float32) for mip in processed_mipmaps]

    # Create generation parameters JSON
    generation_params = {
        "waveform": str(waveform.value),
        "octaves": octaves,
        "rolloff": str(rolloff),
        "size": size,
        "decimate": decimate,
        "frequency": frequency,
        "duty": duty,
    }

    # Save as WTBL format
    save_wavetable_wav(
        output,
        mipmaps_2d,
        wavetable_type=WavetableType.HIGH_RESOLUTION,
        name=f"{waveform.value}_wavetable",
        author="wtgen",
        normalization_method=NormalizationMethod.RMS,
        generation_parameters=json.dumps(generation_params),
        sample_rate=wav_sample_rate,
        type_metadata=HighResolutionMetadata(
            max_harmonics=size // 2,
        ),
    )

    message = f"Exported {len(mipmaps_2d)} mipmap levels to {output}"

    # Optionally export individual WAV files
    if export_individual_wavs:
        if wav_dir is None:
            wav_dir = output.parent / f"{output.stem}_wavs"
        tables = {"base": processed_mipmaps}
        save_mipmaps_as_wav(wav_dir, tables, wav_sample_rate, wav_bit_depth)
        message += f" and individual WAV files to {wav_dir}"

    console.print(message)
    return 0


@app.command
def harmonic(
    output: Path = Path("harmonic_wavetable.wav"),
    partials: str | None = "1:1.0:0.0",
    octaves: int = 8,
    rolloff: RolloffMethod = "raised_cosine",
    size: Annotated[int, Parameter(validator=validate_power_of_two_integer)] = 2048,
    decimate: bool = False,
    export_individual_wavs: bool = False,
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
        Output path for the .wav WTBL file (default: harmonic_wavetable.wav)
    partials: str | None
        Harmonic partials as 'h1:a1:p1,h2:a2:p2...'
        where h is harmonic index, a is amplitude and p is phase
    rolloff: RolloffMethod
        Rolloff method for FIR bandlimiting
    size: int
        The length of the wavetable in samples. Must be a power of 2 (default: 2048)
    decimate: bool
        Whether to decimate the wavetable for each octave above base (default: False)
    export_individual_wavs: bool
        Whether to also export individual .wav files per mipmap level
    wav_dir: Path | None
        The output directory of the individual .wav wavetables
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

    # Export to WTBL format
    console.print(f"Exporting to {output}...")

    # Reshape mipmaps to 2D arrays (num_frames=1, frame_length)
    mipmaps_2d = [mip.reshape(1, -1).astype(np.float32) for mip in processed_mipmaps]

    # Create generation parameters JSON
    generation_params = {
        "waveform": "harmonic",
        "octaves": octaves,
        "rolloff": str(rolloff),
        "size": size,
        "decimate": decimate,
        "num_partials": len(harmonic_partials),
        "partials": harmonic_partials,
    }

    # Save as WTBL format
    save_wavetable_wav(
        output,
        mipmaps_2d,
        wavetable_type=WavetableType.HIGH_RESOLUTION,
        name="harmonic_wavetable",
        author="wtgen",
        normalization_method=NormalizationMethod.RMS,
        generation_parameters=json.dumps(generation_params),
        sample_rate=wav_sample_rate,
        type_metadata=HighResolutionMetadata(
            max_harmonics=size // 2,
        ),
    )

    message = f"Exported {len(mipmaps_2d)} mipmap levels to {output}"

    # Optionally export individual WAV files
    if export_individual_wavs:
        if wav_dir is None:
            wav_dir = output.parent / f"{output.stem}_wavs"
        tables = {"base": processed_mipmaps}
        save_mipmaps_as_wav(wav_dir, tables, wav_sample_rate, wav_bit_depth)
        message += f" and individual WAV files to {wav_dir}"

    console.print(message)
    return 0


@app.command
def info(file: Path) -> int:
    """
    Display information about a wavetable file.

    Parameters
    ----------
    file: Path
        The path to the .wav WTBL file
    """

    if not file.exists():
        print_error(f"Error: File {file} does not exist")
        raise ValueError("File does not exist")

    # Load WTBL format
    wt_file = load_wavetable_wav(file)

    console.print(f"Wavetable: {file}")
    console.print(f"  Schema version: {wt_file.metadata.schema_version}")
    console.print(f"  Wavetable type: {wt_file.wavetable_type.display_name}")
    if wt_file.name:
        console.print(f"  Name: {wt_file.name}")
    if wt_file.author:
        console.print(f"  Author: {wt_file.author}")
    if wt_file.description:
        console.print(f"  Description: {wt_file.description}")
    console.print(f"  Sample rate: {wt_file.sample_rate} Hz")
    console.print(f"  Normalization: {wt_file.normalization_method.name}")

    # Frame structure
    console.print(f"  Frames: {wt_file.num_frames}")
    console.print(f"  Frame length (mip 0): {wt_file.frame_length}")
    console.print(f"  Mip levels: {wt_file.num_mip_levels}")

    # Show mipmap frame lengths
    console.print("  Mipmap levels:")
    for i, (mip, frame_len) in enumerate(
        zip(wt_file.mipmaps, wt_file.mip_frame_lengths, strict=True)
    ):
        rms = np.sqrt(np.mean(mip**2))
        console.print(f"    Level {i}: frame_length={frame_len}, RMS={rms:.3f}")

    # Parse and display generation parameters if present
    if wt_file.metadata.generation_parameters:
        try:
            gen_params = json.loads(wt_file.metadata.generation_parameters)
            console.print("  Generation parameters:")
            if "waveform" in gen_params:
                console.print(f"    Waveform: {gen_params['waveform']}")
            if "rolloff" in gen_params:
                console.print(f"    Rolloff: {gen_params['rolloff']}")
            if "octaves" in gen_params:
                console.print(f"    Octaves: {gen_params['octaves']}")
            if "size" in gen_params:
                console.print(f"    Size: {gen_params['size']}")
            if gen_params.get("decimate"):
                console.print("    Decimated: Yes")
            if "num_partials" in gen_params:
                console.print(f"    Num partials: {gen_params['num_partials']}")
        except json.JSONDecodeError:
            console.print("  Generation parameters: (invalid JSON)")

    # Show type-specific metadata
    type_meta = wt_file.get_type_metadata()
    if type_meta:
        console.print(f"  Type metadata: {type(type_meta).__name__}")
        if isinstance(type_meta, HighResolutionMetadata):
            if type_meta.max_harmonics:
                console.print(f"    Max harmonics: {type_meta.max_harmonics}")
            if type_meta.source_synth:
                console.print(f"    Source synth: {type_meta.source_synth}")

    return 0


if __name__ == "__main__":
    sys.exit(app())
