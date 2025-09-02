import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.io import wavfile


def save_wavetable_npz(
    out_path: Path | str,
    tables: dict[str, list[NDArray[np.float32]]],
    meta: dict[str, Any],
    compress: bool = True,
) -> None:
    """
    Save wavetables to NPZ format with manifest.json schema.

    Args:
        out_path: Output file path
        tables: Dict mapping table IDs to lists of mipmap arrays
            e.g. {"base": [np.array(2048), np.array(1024), ...]}
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
                arr = np.asarray(arr, dtype="<f4", order="C")
                name = f"{table_id}/mip_{lvl:02d}_len{arr.shape[0]}.npy"

                # Serialize NPY into memory so we can write into the zip
                buf = io.BytesIO()
                np.save(buf, arr, allow_pickle=False)
                z.writestr(name, buf.getvalue())

                entry["mipmaps"].append({"npz_path": name, "length": int(arr.shape[0])})

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
    tables: dict[str, list[NDArray[np.float32]]],
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
            wavfile.write(str(filepath), sample_rate, mip_scaled)


def save_wavetable_with_wav_export(
    npz_path: Path | str,
    wav_dir: Path | str,
    tables: dict[str, list[NDArray[np.float32]]],
    meta: dict[str, Any],
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
