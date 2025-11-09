import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile

from wtgen.types import ExportParams, WavetableMetadata, WavetableTables


def save_wavetable_npz(
    out_path: Path | str,
    tables: WavetableTables,
    meta: WavetableMetadata,
    compress: bool = True,
) -> None:
    """
    Save wavetables to NPZ format with manifest.json schema.

    Args:
        out_path: Output file path
        tables: Dict mapping table IDs to lists of mipmap arrays
        meta: Metadata dict (excluding "tables" field which is auto-generated)
        compress: Whether to compress the ZIP file
    """
    compression = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED

    with zipfile.ZipFile(out_path, "w", compression=compression) as z:
        manifest = dict(meta)
        manifest["tables"] = []

        for table_id, mip_list in tables.items():
            entry: dict[str, Any] = {
                "id": table_id,
                "description": f"{table_id} wavetable",
                "mipmaps": [],
            }

            for lvl, arr in enumerate(mip_list):
                arr_data = np.asarray(arr, dtype="<f4", order="C")
                name = f"{table_id}/mip_{lvl:02d}_len{arr_data.shape[0]}.npy"

                # Serialize NPY into memory so we can write into the zip
                buf = io.BytesIO()
                np.save(buf, arr_data, allow_pickle=False)
                z.writestr(name, buf.getvalue())

                entry["mipmaps"].append({"npz_path": name, "length": int(arr_data.shape[0])})

            manifest["tables"].append(entry)

        # Add statistics
        if tables:
            first_table = next(iter(tables.values()))
            if first_table:
                all_samples = np.concatenate([mip for mip in first_table])
                manifest["stats"] = {
                    "dc_offset_mean": float(np.mean(all_samples)),
                    "peak": float(np.max(np.abs(all_samples))),
                }

        z.writestr("manifest.json", json.dumps(manifest, indent=2))


def load_wavetable_npz(file_path: Path | str) -> dict[str, Any]:
    """
    Load wavetable from custom NPZ format.

    Returns:
        Dict containing "manifest" and "tables" keys
    """
    result: dict[str, Any] = {"manifest": {}, "tables": {}}

    with zipfile.ZipFile(file_path, "r") as z:
        # Load manifest
        manifest_data = z.read("manifest.json")
        result["manifest"] = json.loads(manifest_data)

        # Load tables
        for table_info in result["manifest"]["tables"]:
            table_id = table_info["id"]
            result["tables"][table_id] = []

            for mipmap_info in table_info["mipmaps"]:
                npy_path = mipmap_info["npz_path"]
                npy_data = z.read(npy_path)

                # Load NPY from bytes
                buf = io.BytesIO(npy_data)
                arr = np.load(buf, allow_pickle=False)
                result["tables"][table_id].append(arr)

    return result


def save_mipmaps_as_wav(
    output_dir: Path | str,
    tables: WavetableTables,
    sample_rate: int = 44100,
    bit_depth: int = 16,
) -> None:
    """
    Save each mipmap level as individual single-cycle .wav files.

    Args:
        output_dir: Directory to save .wav files
        tables: Dict mapping table IDs to lists of mipmap arrays
        sample_rate: Sample rate for .wav files (default 44100 Hz)
        bit_depth: Bit depth for .wav files (16 or 24 or 32)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Map bit depths to numpy dtypes and scaling
    if bit_depth == 16:
        dtype: type[np.signedinteger[Any]] = np.int16
        scale = 32767.0
    elif bit_depth == 24:
        # scipy.wavfile doesn't support 24-bit directly, use 32-bit instead
        dtype = np.int32
        scale = 8388607.0  # 2^23 - 1 for 24-bit range
    elif bit_depth == 32:
        dtype = np.int32
        scale = 2147483647.0  # 2^31 - 1
    else:
        raise ValueError(f"Unsupported bit depth: {bit_depth}. Use 16, 24, or 32.")

    for table_id, mip_list in tables.items():
        table_dir = output_dir / table_id
        table_dir.mkdir(exist_ok=True)

        for level, mip_array in enumerate(mip_list):
            # Normalize to [-1, 1] range and convert to integer format
            # Handle NaN/inf values by replacing with 0
            mip_clean = np.where(np.isfinite(mip_array), mip_array, 0.0)
            mip_normalized = np.clip(mip_clean, -1.0, 1.0)
            mip_scaled = (mip_normalized * scale).astype(dtype)

            # Create filename
            filename = f"mip_{level:02d}_len{len(mip_array)}.wav"
            filepath = table_dir / filename

            # Save as .wav file
            # Ensure array is 1D for wavfile.write
            mip_1d = np.asarray(mip_scaled).flatten()
            wavfile.write(str(filepath), sample_rate, mip_1d)


def save_wavetable_with_wav_export(
    npz_path: Path | str,
    wav_dir: Path | str,
    tables: WavetableTables,
    meta: WavetableMetadata,
    compress: bool = True,
    sample_rate: int = 44100,
    bit_depth: int = 16,
) -> None:
    """
    Save wavetable to both .npz format and individual .wav files.

    Args:
        npz_path: Output path for .npz file
        wav_dir: Output directory for .wav files
        tables: Dict mapping table IDs to lists of mipmap arrays
        meta: Metadata dict
        compress: Whether to compress the .npz file
        sample_rate: Sample rate for .wav files
        bit_depth: Bit depth for .wav files (16, 24, or 32)
    """
    # Save NPZ format
    save_wavetable_npz(npz_path, tables, meta, compress)

    # Save WAV files
    save_mipmaps_as_wav(wav_dir, tables, sample_rate, bit_depth)


def create_wavetable_metadata(
    name: str,
    waveform: str,
    octaves: int,
    rolloff: str,
    size: int,
    decimate: bool,
    sample_rate: int,
    frequency: float | None = None,
    duty: float | None = None,
    partials: list[tuple[int, float, float]] | None = None,
) -> WavetableMetadata:
    """Create standardized wavetable metadata."""
    meta: WavetableMetadata = {
        "version": 1,
        "name": name,
        "author": "wtgen",
        "sample_rate_hz": sample_rate,
        "cycle_length_samples": size,
        "phase_convention": "zero_at_idx0",
        "normalization": "rms_0.35",
        "tuning": {"root_midi_note": 69, "cents_offset": 0},
        "generation": {
            "waveform": waveform,
            "octaves": octaves,
            "rolloff": rolloff,
            "size": size,
            "decimate": decimate,
        },
    }

    if frequency is not None:
        meta["generation"]["frequency"] = frequency

    if duty is not None:
        meta["generation"]["duty"] = duty

    if partials is not None:
        meta["generation"]["num_partials"] = len(partials)
        meta["generation"]["partials"] = partials

    return meta


def handle_wavetable_export(
    output_path: Path,
    tables: WavetableTables,
    meta: WavetableMetadata,
    params: ExportParams,
) -> tuple[Path | None, str]:
    """Handle wavetable export based on parameters.

    Returns:
        Tuple of (wav_directory, success_message)
    """
    wav_dir = None
    if params.export_wav:
        if params.wav_dir is None:
            wav_dir = output_path.parent / f"{output_path.stem}_wav"
        else:
            wav_dir = params.wav_dir

        save_wavetable_with_wav_export(
            output_path,
            wav_dir,
            tables,
            meta,
            compress=True,
            sample_rate=params.wav_sample_rate,
            bit_depth=params.wav_bit_depth,
        )

        num_mipmaps = len(next(iter(tables.values())))
        message = (
            f"Exported {num_mipmaps} mipmap levels to {output_path} and WAV files to {wav_dir}"
        )
    else:
        save_wavetable_npz(output_path, tables, meta, compress=True)
        num_mipmaps = len(next(iter(tables.values())))
        message = f"Exported {num_mipmaps} mipmap levels to {output_path}"

    return wav_dir, message
