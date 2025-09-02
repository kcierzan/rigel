from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from wtgen.dsp.mipmap import build_mipmap
from wtgen.dsp.process import align_to_zero_crossing, dc_remove, normalize
from wtgen.dsp.waves import (
    RolloffMethod,
    WaveformType,
    generate_polyblep_sawtooth_wavetable,
    harmonics_to_table,
)
from wtgen.export import load_wavetable_npz, save_wavetable_npz, save_wavetable_with_wav_export
from wtgen.plotting import (
    generate_pulse_wavetable,
    generate_sawtooth_wavetable,
    generate_square_wavetable,
    generate_triangle_wavetable,
)

app = typer.Typer(help="Wavetable generation and processing CLI")


@app.command()
def generate(
    waveform: Annotated[
        WaveformType, typer.Argument(help="Waveform type to generate")
    ] = WaveformType.sawtooth,
    output: Annotated[Path, typer.Option("--output", "-o", help="Output .npz file path")] = Path(
        "wavetable.npz"
    ),
    octaves: Annotated[int, typer.Option("--octaves", "-n", help="Number of mipmap octaves")] = 8,
    rolloff: Annotated[
        RolloffMethod, typer.Option("--rolloff", "-r", help="Rolloff method for bandlimiting")
    ] = RolloffMethod.tukey,
    frequency: Annotated[
        float, typer.Option("--frequency", "-f", help="Base frequency for waveform")
    ] = 1.0,
    duty: Annotated[
        float, typer.Option("--duty", "-d", help="Duty cycle for square/pulse waves (0.0-1.0)")
    ] = 0.5,
    size: Annotated[int, typer.Option("--size", "-s", help="Wavetable size (power of 2)")] = 2048,
    decimate: Annotated[
        bool, typer.Option("--decimate", help="Decimate each mipmap level (2048→1024→512...)")
    ] = False,
    export_wav: Annotated[
        bool,
        typer.Option("--export-wav", help="Also export mipmap levels as individual .wav files"),
    ] = False,
    wav_dir: Annotated[
        Path | None,
        typer.Option("--wav-dir", help="Directory for .wav files (default: <output_name>_wav)"),
    ] = None,
    wav_sample_rate: Annotated[
        int, typer.Option("--wav-sample-rate", help="Sample rate for .wav files")
    ] = 44100,
    wav_bit_depth: Annotated[
        int, typer.Option("--wav-bit-depth", help="Bit depth for .wav files (16, 24, or 32)")
    ] = 16,
) -> None:
    """Generate wavetable mipmaps and export to .npz format."""

    if size & (size - 1) != 0:
        typer.echo("Error: Size must be a power of 2", err=True)
        raise typer.Exit(1)

    if wav_bit_depth not in [16, 24, 32]:
        typer.echo("Error: WAV bit depth must be 16, 24, or 32", err=True)
        raise typer.Exit(1)

    typer.echo(f"Generating {waveform} wavetable with {octaves} octaves...")

    # Generate base waveform
    if waveform == "sawtooth":
        _, wave = generate_sawtooth_wavetable(frequency)
    elif waveform == "square":
        _, wave = generate_square_wavetable(frequency, duty)
    elif waveform == "pulse":
        _, wave = generate_pulse_wavetable(frequency, duty)
    elif waveform == "triangle":
        _, wave = generate_triangle_wavetable(frequency)
    elif waveform == "polyblep_saw":
        _, wave = generate_polyblep_sawtooth_wavetable(frequency)
    else:
        typer.echo(f"Error: Unknown waveform type '{waveform}'", err=True)
        raise typer.Exit(1)

    # Resize if different from default
    if len(wave) != size:
        # Simple resampling - could be improved with proper interpolation
        indices = np.linspace(0, len(wave) - 1, size)
        wave = np.interp(indices, np.arange(len(wave)), wave)

    # Build mipmaps
    typer.echo(f"Building mipmaps with {rolloff} rolloff...")
    mipmaps = build_mipmap(wave, num_octaves=octaves, rolloff_method=rolloff, decimate=decimate)

    # Process each mipmap level
    processed_mipmaps = []
    for i, mip in enumerate(mipmaps):
        # Apply standard processing pipeline
        processed = align_to_zero_crossing(dc_remove(normalize(mip)))
        processed_mipmaps.append(processed)
        typer.echo(f"Processed mipmap level {i}: RMS={np.sqrt(np.mean(processed**2)):.3f}")

    # Export to .npz
    typer.echo(f"Exporting to {output}...")

    # Prepare tables dict
    tables = {"base": processed_mipmaps}

    # Prepare metadata in manifest format
    meta = {
        "version": 1,
        "name": f"{waveform}_wavetable",
        "author": "wtgen",
        "sample_rate_hz": wav_sample_rate,
        "cycle_length_samples": size,
        "phase_convention": "zero_at_idx0",
        "normalization": "rms_0.35",
        "tuning": {"root_midi_note": 69, "cents_offset": 0},
        "generation": {
            "waveform": waveform,
            "octaves": octaves,
            "rolloff": rolloff.value,
            "frequency": frequency,
            "duty": duty,
            "size": size,
            "decimate": decimate,
        },
    }

    if export_wav:
        # Determine WAV output directory
        if wav_dir is None:
            wav_dir = output.parent / f"{output.stem}_wav"

        typer.echo(f"Exporting WAV files to {wav_dir}...")
        save_wavetable_with_wav_export(
            output,
            wav_dir,
            tables,
            meta,
            compress=True,
            sample_rate=wav_sample_rate,
            bit_depth=wav_bit_depth,
        )
        typer.echo(
            f" Exported {len(processed_mipmaps)} mipmap levels to {output} "
            f"and WAV files to {wav_dir}"
        )
    else:
        save_wavetable_npz(output, tables, meta, compress=True)

        typer.echo(f" Exported {len(processed_mipmaps)} mipmap levels to {output}")


@app.command()
def harmonic(
    output: Annotated[Path, typer.Option("--output", "-o", help="Output .npz file path")] = Path(
        "harmonic_wavetable.npz"
    ),
    partials: Annotated[
        str | None,
        typer.Option("--partials", "-p", help="Harmonic partials as 'h1:a1:p1,h2:a2:p2,...'"),
    ] = None,
    octaves: Annotated[int, typer.Option("--octaves", "-n", help="Number of mipmap octaves")] = 8,
    rolloff: Annotated[
        RolloffMethod, typer.Option("--rolloff", "-r", help="Rolloff method for bandlimiting")
    ] = RolloffMethod.raised_cosine,
    size: Annotated[int, typer.Option("--size", "-s", help="Wavetable size (power of 2)")] = 2048,
    decimate: Annotated[
        bool, typer.Option("--decimate", help="Decimate each mipmap level (2048→1024→512...)")
    ] = False,
    export_wav: Annotated[
        bool,
        typer.Option("--export-wav", help="Also export mipmap levels as individual .wav files"),
    ] = False,
    wav_dir: Annotated[
        Path | None,
        typer.Option("--wav-dir", help="Directory for .wav files (default: <output_name>_wav)"),
    ] = None,
    wav_sample_rate: Annotated[
        int, typer.Option("--wav-sample-rate", help="Sample rate for .wav files")
    ] = 44100,
    wav_bit_depth: Annotated[
        int, typer.Option("--wav-bit-depth", help="Bit depth for .wav files (16, 24, or 32)")
    ] = 16,
) -> None:
    """Generate wavetable from harmonic partials."""

    if size & (size - 1) != 0:
        typer.echo("Error: Size must be a power of 2", err=True)
        raise typer.Exit(1)

    # Parse partials
    partial_list = []
    if partials:
        try:
            for partial_str in partials.split(","):
                parts = partial_str.split(":")
                if len(parts) != 3:
                    raise ValueError(f"Invalid partial format: {partial_str}")
                harmonic = int(parts[0])
                amplitude = float(parts[1])
                phase = float(parts[2])
                partial_list.append((harmonic, amplitude, phase))
        except ValueError as e:
            typer.echo(f"Error parsing partials: {e}", err=True)
            raise typer.Exit(1)
    else:
        # Default: sawtooth-like harmonics
        partial_list = [(i, 1.0 / i, 0.0) for i in range(1, 17)]

    typer.echo(f"Generating wavetable from {len(partial_list)} harmonics...")

    # Generate base waveform from harmonics
    wave = harmonics_to_table(partial_list, size)

    # Build mipmaps
    typer.echo(f"Building mipmaps with {rolloff} rolloff...")
    mipmaps = build_mipmap(wave, num_octaves=octaves, rolloff_method=rolloff, decimate=decimate)

    # Process each mipmap level
    processed_mipmaps = []
    for i, mip in enumerate(mipmaps):
        processed = align_to_zero_crossing(dc_remove(normalize(mip)))
        processed_mipmaps.append(processed)
        typer.echo(f"Processed mipmap level {i}: RMS={np.sqrt(np.mean(processed**2)):.3f}")

    # Export to .npz
    typer.echo(f"Exporting to {output}...")

    # Prepare tables dict
    tables = {"base": processed_mipmaps}

    # Prepare metadata in manifest format
    meta = {
        "version": 1,
        "name": "harmonic_wavetable",
        "author": "wtgen",
        "sample_rate_hz": wav_sample_rate,
        "cycle_length_samples": size,
        "phase_convention": "zero_at_idx0",
        "normalization": "rms_0.35",
        "tuning": {"root_midi_note": 69, "cents_offset": 0},
        "generation": {
            "waveform": "harmonic",
            "octaves": octaves,
            "rolloff": rolloff.value,
            "size": size,
            "num_partials": len(partial_list),
            "partials": partial_list,
            "decimate": decimate,
        },
    }

    if export_wav:
        # Determine WAV output directory
        if wav_dir is None:
            wav_dir = output.parent / f"{output.stem}_wav"

        typer.echo(f"Exporting WAV files to {wav_dir}...")
        save_wavetable_with_wav_export(
            output,
            wav_dir,
            tables,
            meta,
            compress=True,
            sample_rate=wav_sample_rate,
            bit_depth=wav_bit_depth,
        )
        typer.echo(
            f" Exported {len(processed_mipmaps)} mipmap levels to {output} "
            f"and WAV files to {wav_dir}"
        )
    else:
        save_wavetable_npz(output, tables, meta, compress=True)
        typer.echo(f" Exported {len(processed_mipmaps)} mipmap levels to {output}")


@app.command()
def info(file: Annotated[Path, typer.Argument(help="Input .npz wavetable file")]) -> None:
    """Display information about a wavetable file."""

    if not file.exists():
        typer.echo(f"Error: File {file} does not exist", err=True)
        raise typer.Exit(1)

    try:
        # Try new manifest format first
        wt_data = load_wavetable_npz(file)
        manifest = wt_data["manifest"]
        tables = wt_data["tables"]

        typer.echo(f"Wavetable: {file}")
        typer.echo(f"  Name: {manifest.get('name', 'unknown')}")
        typer.echo(f"  Version: {manifest.get('version', 'unknown')}")
        typer.echo(f"  Author: {manifest.get('author', 'unknown')}")
        typer.echo(f"  Sample rate: {manifest.get('sample_rate_hz', 'unknown')} Hz")
        typer.echo(f"  Base cycle length: {manifest.get('cycle_length_samples', 'unknown')}")

        if "generation" in manifest:
            gen = manifest["generation"]
            typer.echo(f"  Waveform: {gen.get('waveform', 'unknown')}")
            typer.echo(f"  Rolloff: {gen.get('rolloff', 'unknown')}")
            if gen.get("decimate"):
                typer.echo("  Decimated: Yes")

        # Show table information
        for table_info in manifest.get("tables", []):
            table_id = table_info["id"]
            mipmaps = tables.get(table_id, [])
            typer.echo(f"  Table '{table_id}': {len(mipmaps)} mipmap levels")

            # Show mipmap sizes and RMS levels
            typer.echo("    Mipmap levels:")
            for i, mip in enumerate(mipmaps):
                rms = np.sqrt(np.mean(mip**2))
                typer.echo(f"      Level {i}: size={len(mip)}, RMS={rms:.3f}")

        # Show statistics
        if "stats" in manifest:
            stats = manifest["stats"]
            typer.echo(f"  DC offset: {stats.get('dc_offset_mean', 'unknown'):.6f}")
            typer.echo(f"  Peak: {stats.get('peak', 'unknown'):.6f}")

    except Exception as e:
        typer.echo(f"Error reading file: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
