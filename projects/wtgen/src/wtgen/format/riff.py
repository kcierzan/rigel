"""RIFF/WAV chunk utilities for wavetable files.

This module provides utilities for reading and writing custom RIFF chunks,
specifically the WTBL chunk that contains protobuf-encoded wavetable metadata.
"""

import struct
from pathlib import Path
from typing import BinaryIO

# FourCC identifiers
RIFF_ID = b"RIFF"
WAVE_ID = b"WAVE"
FMT_ID = b"fmt "
DATA_ID = b"data"
WTBL_ID = b"WTBL"

# Audio format codes
WAVE_FORMAT_PCM = 1
WAVE_FORMAT_IEEE_FLOAT = 3


class RiffError(Exception):
    """Error reading or writing RIFF files."""


def read_chunk_header(f: BinaryIO) -> tuple[bytes, int]:
    """Read a RIFF chunk header (FourCC + size).

    Args:
        f: File handle positioned at the start of a chunk.

    Returns:
        Tuple of (chunk_id, chunk_size).

    Raises:
        RiffError: If the header cannot be read.
    """
    header = f.read(8)
    if len(header) < 8:
        raise RiffError("Unexpected end of file reading chunk header")

    chunk_id = header[:4]
    chunk_size = struct.unpack("<I", header[4:8])[0]
    return chunk_id, chunk_size


def find_chunk(f: BinaryIO, target_id: bytes, file_size: int) -> bytes | None:
    """Find a chunk by its FourCC identifier.

    Args:
        f: File handle positioned after the RIFF header (at first chunk).
        target_id: The FourCC identifier to search for.
        file_size: Total file size for bounds checking.

    Returns:
        The chunk data if found, None otherwise.
    """
    while f.tell() < file_size:
        try:
            chunk_id, chunk_size = read_chunk_header(f)
        except RiffError:
            return None

        if chunk_id == target_id:
            return f.read(chunk_size)

        # Skip to next chunk (with word alignment padding)
        skip_size = chunk_size + (chunk_size % 2)
        f.seek(skip_size, 1)  # Seek relative to current position

    return None


def extract_wtbl_chunk(file_path: Path | str) -> bytes:
    """Extract the WTBL chunk data from a WAV file.

    Args:
        file_path: Path to the WAV file.

    Returns:
        The raw bytes of the WTBL chunk content.

    Raises:
        RiffError: If the file is not a valid WAV or WTBL chunk is missing.
    """
    file_path = Path(file_path)

    try:
        f = open(file_path, "rb")
    except FileNotFoundError as e:
        raise RiffError(f"File not found: {file_path}") from e
    except OSError as e:
        raise RiffError(f"Cannot open file: {file_path}") from e

    with f:
        # Read and validate RIFF header
        riff_header = f.read(12)
        if len(riff_header) < 12:
            raise RiffError("File too small to be a valid WAV file")

        if riff_header[:4] != RIFF_ID:
            raise RiffError("Not a RIFF file")

        if riff_header[8:12] != WAVE_ID:
            raise RiffError("Not a WAVE file")

        file_size = struct.unpack("<I", riff_header[4:8])[0] + 8

        # Search for WTBL chunk
        wtbl_data = find_chunk(f, WTBL_ID, file_size)
        if wtbl_data is None:
            raise RiffError("WTBL chunk not found in WAV file")

        return wtbl_data


def read_fmt_chunk(file_path: Path | str) -> tuple[int, int, int, int]:
    """Read the fmt chunk to extract audio format information.

    Args:
        file_path: Path to the WAV file.

    Returns:
        Tuple of (audio_format, num_channels, sample_rate, bits_per_sample).

    Raises:
        RiffError: If the file is not a valid WAV or fmt chunk is missing.
    """
    file_path = Path(file_path)

    with open(file_path, "rb") as f:
        # Read and validate RIFF header
        riff_header = f.read(12)
        if len(riff_header) < 12:
            raise RiffError("File too small to be a valid WAV file")

        if riff_header[:4] != RIFF_ID or riff_header[8:12] != WAVE_ID:
            raise RiffError("Not a valid WAVE file")

        file_size = struct.unpack("<I", riff_header[4:8])[0] + 8

        # Search for fmt chunk
        fmt_data = find_chunk(f, FMT_ID, file_size)
        if fmt_data is None:
            raise RiffError("fmt chunk not found in WAV file")

        if len(fmt_data) < 16:
            raise RiffError("fmt chunk too small")

        audio_format = struct.unpack("<H", fmt_data[0:2])[0]
        num_channels = struct.unpack("<H", fmt_data[2:4])[0]
        sample_rate = struct.unpack("<I", fmt_data[4:8])[0]
        bits_per_sample = struct.unpack("<H", fmt_data[14:16])[0]

        return audio_format, num_channels, sample_rate, bits_per_sample


def read_data_chunk(file_path: Path | str) -> bytes:
    """Read the data chunk to extract raw audio data.

    Args:
        file_path: Path to the WAV file.

    Returns:
        The raw bytes of the audio data.

    Raises:
        RiffError: If the file is not a valid WAV or data chunk is missing.
    """
    file_path = Path(file_path)

    with open(file_path, "rb") as f:
        # Read and validate RIFF header
        riff_header = f.read(12)
        if len(riff_header) < 12:
            raise RiffError("File too small to be a valid WAV file")

        if riff_header[:4] != RIFF_ID or riff_header[8:12] != WAVE_ID:
            raise RiffError("Not a valid WAVE file")

        file_size = struct.unpack("<I", riff_header[4:8])[0] + 8

        # Search for data chunk
        data = find_chunk(f, DATA_ID, file_size)
        if data is None:
            raise RiffError("data chunk not found in WAV file")

        return data


def append_wtbl_chunk(file_path: Path | str, wtbl_data: bytes) -> None:
    """Append a WTBL chunk to an existing WAV file.

    This modifies the file in place, appending the chunk and updating the RIFF size.

    Args:
        file_path: Path to the WAV file.
        wtbl_data: The protobuf-encoded metadata to store in the WTBL chunk.

    Raises:
        RiffError: If the file is not a valid WAV file.
    """
    file_path = Path(file_path)

    with open(file_path, "r+b") as f:
        # Verify it's a RIFF/WAVE file
        riff_header = f.read(12)
        if len(riff_header) < 12:
            raise RiffError("File too small to be a valid WAV file")

        if riff_header[:4] != RIFF_ID or riff_header[8:12] != WAVE_ID:
            raise RiffError("Not a valid WAVE file")

        # Read current RIFF size
        current_riff_size = struct.unpack("<I", riff_header[4:8])[0]

        # Seek to end
        f.seek(0, 2)

        # Write WTBL chunk header + data
        f.write(WTBL_ID)
        f.write(struct.pack("<I", len(wtbl_data)))
        f.write(wtbl_data)

        # Pad for word alignment
        if len(wtbl_data) % 2:
            f.write(b"\x00")
            chunk_total_size = 8 + len(wtbl_data) + 1
        else:
            chunk_total_size = 8 + len(wtbl_data)

        # Update RIFF size
        new_riff_size = current_riff_size + chunk_total_size
        f.seek(4)
        f.write(struct.pack("<I", new_riff_size))


def build_wav_with_wtbl(
    samples: bytes,
    sample_rate: int,
    wtbl_data: bytes,
    num_channels: int = 1,
    bits_per_sample: int = 32,
) -> bytes:
    """Build a complete WAV file with audio data and WTBL chunk.

    Args:
        samples: Raw audio sample data (already in the correct byte format).
        sample_rate: The sample rate in Hz.
        wtbl_data: The protobuf-encoded metadata for the WTBL chunk.
        num_channels: Number of audio channels (default: 1 for mono).
        bits_per_sample: Bits per sample (default: 32 for float).

    Returns:
        The complete WAV file as bytes.
    """
    # Build fmt chunk (IEEE float format for 32-bit)
    audio_format = WAVE_FORMAT_IEEE_FLOAT if bits_per_sample == 32 else WAVE_FORMAT_PCM
    bytes_per_sample = bits_per_sample // 8
    byte_rate = sample_rate * num_channels * bytes_per_sample
    block_align = num_channels * bytes_per_sample

    fmt_chunk = struct.pack(
        "<HHIIHH",
        audio_format,
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
    )

    # Calculate sizes
    data_size = len(samples)
    wtbl_size = len(wtbl_data)
    wtbl_padded_size = wtbl_size + (wtbl_size % 2)

    # Total RIFF size = file size - 8 (RIFF header)
    # 4 (WAVE) + 8+16 (fmt chunk) + 8+data_size (data chunk) + 8+wtbl_padded_size
    riff_size = 4 + 8 + 16 + 8 + data_size + 8 + wtbl_padded_size

    # Build the WAV file
    wav = bytearray()

    # RIFF header
    wav.extend(RIFF_ID)
    wav.extend(struct.pack("<I", riff_size))
    wav.extend(WAVE_ID)

    # fmt chunk
    wav.extend(FMT_ID)
    wav.extend(struct.pack("<I", 16))  # fmt chunk size
    wav.extend(fmt_chunk)

    # data chunk
    wav.extend(DATA_ID)
    wav.extend(struct.pack("<I", data_size))
    wav.extend(samples)

    # WTBL chunk
    wav.extend(WTBL_ID)
    wav.extend(struct.pack("<I", wtbl_size))
    wav.extend(wtbl_data)
    if wtbl_size % 2:
        wav.extend(b"\x00")

    return bytes(wav)
