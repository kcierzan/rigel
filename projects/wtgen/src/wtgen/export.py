from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile

from wtgen.types import WavetableTables


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
