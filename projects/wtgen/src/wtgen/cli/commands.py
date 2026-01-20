import json
import sys
from pathlib import Path
from typing import Annotated

import numpy as np
from cyclopts import App, Parameter
from rich.console import Console
from rich.table import Table

from wtgen.cli.validators import (
    validate_eq_string,
    validate_positive_integer,
    validate_power_of_two_integer,
    validate_tilt_string,
)
from wtgen.dsp.eq import Equalizer
from wtgen.dsp.fir import RolloffMethod
from wtgen.dsp.mipmap import Mipmap, build_multiframe_mipmap
from wtgen.dsp.process import align_to_zero_crossing, dc_remove, normalize
from wtgen.dsp.waves import WaveGenerator
from wtgen.export import save_mipmaps_as_wav
from wtgen.format import (
    HighResolutionMetadata,
    NormalizationMethod,
    ValidationError,
    WavetableType,
    load_wavetable_wav,
    save_wavetable_wav,
    validate_audio_data,
    validate_metadata,
)
from wtgen.format.importers import detect_wav_wavetable, import_hires_wav
from wtgen.format.riff import RiffError, extract_wtbl_chunk
from wtgen.types import (
    BitDepth,
    HarmonicPartial,
    WaveformType,
)

app = App(name="wtgen", help="A utility for generating and analyzing wavetables")
console = Console()


def print_error(message: str) -> None:
    """Print an error message in red."""
    console.print(message, style="bold red")


def print_success(message: str) -> None:
    """Print a success message in green."""
    console.print(message, style="bold green")


def print_warning(message: str) -> None:
    """Print a warning message in yellow."""
    console.print(message, style="bold yellow")


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


@app.command
def validate(
    file: Path,
    strict: bool = False,
    output_json: Annotated[bool, Parameter(name=["--json"])] = False,
) -> int:
    """
    Validate a WTBL wavetable file.

    Checks RIFF structure, WTBL chunk presence, metadata consistency,
    and audio data integrity.

    Parameters
    ----------
    file: Path
        The path to the .wav WTBL file to validate
    strict: bool
        Treat warnings as errors (default: False)
    output_json: bool
        Output results as JSON (default: False)
    """
    results: dict[str, object] = {
        "file": str(file),
        "valid": True,
        "errors": [],
        "warnings": [],
    }
    errors: list[str] = []
    warnings: list[str] = []

    # Check file exists
    if not file.exists():
        errors.append(f"File not found: {file}")
        results["valid"] = False
        results["errors"] = errors
        if output_json:
            console.print(json.dumps(results, indent=2))
        else:
            print_error(f"[FAIL] File not found: {file}")
        return 1

    # Check WTBL chunk exists
    try:
        extract_wtbl_chunk(file)
    except RiffError as e:
        errors.append(f"RIFF/WTBL error: {e}")
        results["valid"] = False
        results["errors"] = errors
        if output_json:
            console.print(json.dumps(results, indent=2))
        else:
            print_error(f"[FAIL] RIFF/WTBL error: {e}")
            console.print(
                "  Suggestion: This may not be a valid WTBL file. "
                "Use 'wtgen analyze' to check if it's a raw WAV wavetable."
            )
        return 1

    # Load and validate the file
    try:
        wt_file = load_wavetable_wav(file, validate=True)
    except ValidationError as e:
        errors.append(f"Validation error: {e}")
        results["valid"] = False
        results["errors"] = errors
        if output_json:
            console.print(json.dumps(results, indent=2))
        else:
            print_error(f"[FAIL] Validation error: {e}")
        return 1

    # Run metadata validation to get warnings
    meta_result = validate_metadata(wt_file.metadata)
    warnings.extend(meta_result.warnings)

    # Run audio data validation to get warnings
    audio_result = validate_audio_data(wt_file.mipmaps, wt_file.metadata)
    warnings.extend(audio_result.warnings)

    results["warnings"] = warnings

    # In strict mode, warnings become errors
    if strict and warnings:
        results["valid"] = False
        results["errors"] = [f"Strict mode: {w}" for w in warnings]

    if output_json:
        console.print(json.dumps(results, indent=2))
        return 0 if results["valid"] else 1

    # Rich formatted output
    if results["valid"]:
        print_success(f"[PASS] {file}")
        console.print(f"  Type: {wt_file.wavetable_type.display_name}")
        console.print(f"  Frames: {wt_file.num_frames}")
        console.print(f"  Frame length: {wt_file.frame_length}")
        console.print(f"  Mip levels: {wt_file.num_mip_levels}")

        if warnings:
            console.print("")
            for warning in warnings:
                print_warning(f"  [WARN] {warning}")
    else:
        print_error(f"[FAIL] {file}")
        err_list = results.get("errors", [])
        if isinstance(err_list, list):
            for error in err_list:
                console.print(f"  {error}")

    return 0 if results["valid"] else 1


@app.command
def analyze(
    file: Path,
    output_json: Annotated[bool, Parameter(name=["--json"])] = False,
) -> int:
    """
    Analyze a WAV file and suggest wavetable parameters.

    This command examines a WAV file and suggests possible interpretations
    as a wavetable, showing frame count and frame size combinations.

    Parameters
    ----------
    file: Path
        The path to the .wav file to analyze
    output_json: bool
        Output results as JSON (default: False)
    """
    if not file.exists():
        print_error(f"Error: File not found: {file}")
        return 1

    try:
        analysis = detect_wav_wavetable(file)
    except Exception as e:
        print_error(f"Error analyzing file: {e}")
        return 1

    if output_json:
        console.print(json.dumps(analysis, indent=2))
        return 0

    # Rich formatted output
    file_info = analysis["file_info"]
    console.print(f"[bold]File Analysis: {file}[/bold]")
    console.print("")
    console.print("[bold]File Info:[/bold]")
    console.print(f"  Total samples: {file_info['total_samples']:,}")
    console.print(f"  Sample rate: {file_info['sample_rate']} Hz")
    console.print(f"  Duration: {file_info['duration_seconds']:.3f}s")
    console.print(f"  Bit depth: {file_info['bit_depth']}-bit")
    console.print(f"  Channels: {file_info['channels']}")

    suggestions = analysis["suggestions"]
    if not suggestions:
        print_warning("\nNo valid wavetable interpretations found.")
        console.print("The total sample count doesn't divide evenly into common frame counts.")
        return 0

    console.print("")
    console.print("[bold]Possible Interpretations:[/bold]")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Frames", justify="right")
    table.add_column("Frame Size", justify="right")
    table.add_column("Power of 2", justify="center")
    table.add_column("Type", justify="left")
    table.add_column("Import Command", justify="left")

    for suggestion in suggestions:
        is_pow2 = suggestion["is_power_of_two"]
        power_of_two = "[green]Yes[/green]" if is_pow2 else "[yellow]No[/yellow]"
        import_cmd = f"wtgen import {file.name} --frames {suggestion['num_frames']}"
        table.add_row(
            str(suggestion["num_frames"]),
            str(suggestion["frame_length"]),
            power_of_two,
            suggestion["likely_type"],
            import_cmd,
        )

    console.print(table)

    # Show recommended import command
    recommended = analysis.get("recommended")
    if recommended:
        console.print("")
        console.print("[bold]Recommended:[/bold]")
        console.print(f"  wtgen import {file} --frames {recommended['num_frames']}")

    return 0


@app.command(name="import")
def import_(
    source: Path,
    frames: Annotated[int, Parameter(validator=validate_positive_integer)],
    output: Path = Path("imported.wav"),
    frame_size: int | None = None,
    wavetable_type: WavetableType = WavetableType.HIGH_RESOLUTION,
    mip_levels: Annotated[int, Parameter(validator=validate_positive_integer)] = 7,
    no_mipmaps: bool = False,
    rolloff: RolloffMethod = "raised_cosine",
    do_normalize: bool = True,
    dry_run: bool = False,
    name: str | None = None,
    author: str | None = None,
) -> int:
    """
    Import a WAV file as a WTBL wavetable with explicit parameters.

    Converts a raw WAV file containing wavetable data into the standardized
    WTBL format with optional mipmap generation using proper FIR anti-aliasing.

    Parameters
    ----------
    source: Path
        The source WAV file to import
    frames: int
        Number of frames to extract (required)
    output: Path
        Output path for the WTBL file (default: imported.wav)
    frame_size: int | None
        Samples per frame (auto-calculated from total_samples/frames if omitted)
    wavetable_type: WavetableType
        Wavetable type classification (default: high-resolution)
    mip_levels: int
        Number of mipmap levels to generate (default: 7)
    no_mipmaps: bool
        Skip mipmap generation, output single level only (default: False)
    rolloff: RolloffMethod
        FIR filter rolloff method for mipmap anti-aliasing. Options:
        raised_cosine (default), tukey, hann, blackman, brick_wall
    do_normalize: bool
        Normalize audio to [-1, 1] range (default: True)
    dry_run: bool
        Preview import without writing output (default: False)
    name: str | None
        Wavetable name metadata
    author: str | None
        Wavetable author metadata
    """
    if not source.exists():
        print_error(f"Error: Source file not found: {source}")
        console.print("  Suggestion: Check the file path and try again.")
        return 1

    # Import base wavetable
    try:
        base_mipmaps, import_metadata = import_hires_wav(
            source,
            num_frames=frames,
            frame_length=frame_size,
            normalize=do_normalize,
        )
    except ValueError as e:
        print_error(f"Error: {e}")
        console.print("")
        console.print("  Suggestion: Use 'wtgen analyze' to find valid frame counts:")
        console.print(f"    wtgen analyze {source}")
        return 1
    except Exception as e:
        print_error(f"Error importing file: {e}")
        return 1

    base_wavetable = base_mipmaps[0]
    actual_frame_size = base_wavetable.shape[1]
    actual_frames = base_wavetable.shape[0]

    # Generate mipmaps if requested
    if no_mipmaps:
        final_mipmaps = [base_wavetable]
        num_mip_levels = 1
    else:
        # Generate mipmaps with proper FIR anti-aliasing filtering
        console.print(
            f"Generating {mip_levels} mipmap levels with [cyan bold]{rolloff}[/] rolloff..."
        )
        final_mipmaps = build_multiframe_mipmap(
            wavetable=base_wavetable,
            num_octaves=mip_levels - 1,  # num_octaves + 1 = total levels
            rolloff_method=rolloff,
            decimate=True,
        )
        num_mip_levels = len(final_mipmaps)

    # Dry run output
    if dry_run:
        console.print("")
        console.print("[bold]Dry Run - No files written[/bold]")
        console.print("")
        console.print("[bold]Import Summary:[/bold]")
        console.print(f"  Source: {source}")
        console.print(f"  Frames: {actual_frames}")
        console.print(f"  Frame size: {actual_frame_size}")
        console.print(f"  Type: {wavetable_type.display_name}")
        console.print(f"  Mip levels: {num_mip_levels}")
        if name:
            console.print(f"  Name: {name}")
        if author:
            console.print(f"  Author: {author}")
        console.print(f"  Output: {output}")
        console.print("")

        # Show mip level sizes
        console.print("[bold]Mipmap Levels:[/bold]")
        for i, mip in enumerate(final_mipmaps):
            rms = np.sqrt(np.mean(mip**2))
            frames, samples = mip.shape[0], mip.shape[1]
            console.print(f"  Level {i}: {frames} frames x {samples} samples, RMS={rms:.3f}")

        return 0

    # Write output file
    try:
        norm_method = NormalizationMethod.PEAK if do_normalize else NormalizationMethod.NONE
        type_meta = None
        if wavetable_type == WavetableType.HIGH_RESOLUTION:
            type_meta = HighResolutionMetadata(max_harmonics=actual_frame_size // 2)

        save_wavetable_wav(
            output,
            final_mipmaps,
            wavetable_type=wavetable_type,
            name=name or source.stem,
            author=author,
            normalization_method=norm_method,
            sample_rate=int(import_metadata.get("sample_rate", 44100)),
            type_metadata=type_meta,
        )
    except Exception as e:
        print_error(f"Error writing output: {e}")
        return 1

    print_success(f"Imported {source} -> {output}")
    console.print(f"  Frames: {actual_frames}")
    console.print(f"  Frame size: {actual_frame_size}")
    console.print(f"  Mip levels: {num_mip_levels}")

    return 0


if __name__ == "__main__":
    sys.exit(app())
