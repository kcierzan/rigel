#!/usr/bin/env python3
"""
Third-Party Wavetable Creation Example

This example demonstrates how to create wavetable files compatible with Rigel
using only the format specification and protobuf schema, without using wtgen.

Requirements:
  pip install protobuf numpy soundfile

Usage:
  python third_party_python.py

This will create 'output.wav' containing a simple wavetable with sine waves.
"""

import struct
import math
from pathlib import Path
from typing import Optional

import numpy as np

# Import the generated protobuf bindings
# You need to generate these from proto/wavetable.proto using:
#   protoc --python_out=. proto/wavetable.proto
# Then adjust the import path as needed.
# For this example, we'll create a minimal inline version.

# ============================================================================
# Option 1: Use generated protobuf bindings (recommended for production)
# ============================================================================
#
# from proto import wavetable_pb2
#
# def create_metadata(frame_length, num_frames, mip_frame_lengths):
#     metadata = wavetable_pb2.WavetableMetadata()
#     metadata.schema_version = 1
#     metadata.wavetable_type = wavetable_pb2.WAVETABLE_TYPE_CUSTOM
#     metadata.frame_length = frame_length
#     metadata.num_frames = num_frames
#     metadata.num_mip_levels = len(mip_frame_lengths)
#     metadata.mip_frame_lengths.extend(mip_frame_lengths)
#     return metadata.SerializeToString()

# ============================================================================
# Option 2: Inline protobuf serialization (for demonstration/testing)
# ============================================================================
#
# This shows the raw protobuf wire format for educational purposes.
# In production, always use the official protobuf library.


def encode_varint(value: int) -> bytes:
    """Encode an unsigned integer as a protobuf varint."""
    parts = []
    while value > 127:
        parts.append((value & 0x7F) | 0x80)
        value >>= 7
    parts.append(value)
    return bytes(parts)


def encode_field(field_number: int, wire_type: int, value: bytes) -> bytes:
    """Encode a protobuf field with tag."""
    tag = (field_number << 3) | wire_type
    return encode_varint(tag) + value


def encode_uint32(field_number: int, value: int) -> bytes:
    """Encode a uint32 field."""
    return encode_field(field_number, 0, encode_varint(value))  # wire type 0 = varint


def encode_string(field_number: int, value: str) -> bytes:
    """Encode a string field."""
    encoded = value.encode("utf-8")
    return encode_field(field_number, 2, encode_varint(len(encoded)) + encoded)


def encode_repeated_uint32(field_number: int, values: list[int]) -> bytes:
    """Encode a repeated uint32 field (packed)."""
    packed = b"".join(encode_varint(v) for v in values)
    return encode_field(field_number, 2, encode_varint(len(packed)) + packed)


def create_wavetable_metadata(
    frame_length: int,
    num_frames: int,
    mip_frame_lengths: list[int],
    wavetable_type: int = 5,  # WAVETABLE_TYPE_CUSTOM
    name: Optional[str] = None,
    author: Optional[str] = None,
) -> bytes:
    """
    Create WavetableMetadata protobuf message.

    Field numbers per proto/wavetable.proto:
      1: schema_version (uint32)
      2: wavetable_type (enum)
      3: frame_length (uint32)
      4: num_frames (uint32)
      5: num_mip_levels (uint32)
      6: mip_frame_lengths (repeated uint32)
      18: author (optional string)
      19: name (optional string)
    """
    parts = []

    # Core fields (required)
    parts.append(encode_uint32(1, 1))  # schema_version = 1
    parts.append(encode_uint32(2, wavetable_type))  # wavetable_type
    parts.append(encode_uint32(3, frame_length))  # frame_length
    parts.append(encode_uint32(4, num_frames))  # num_frames
    parts.append(encode_uint32(5, len(mip_frame_lengths)))  # num_mip_levels
    parts.append(encode_repeated_uint32(6, mip_frame_lengths))  # mip_frame_lengths

    # Optional fields
    if author:
        parts.append(encode_string(18, author))
    if name:
        parts.append(encode_string(19, name))

    return b"".join(parts)


# ============================================================================
# RIFF/WAV utilities
# ============================================================================


def write_wav_header(
    num_samples: int,
    sample_rate: int,
    bits_per_sample: int = 32,
    num_channels: int = 1,
) -> bytes:
    """Write a WAV file header for floating-point audio."""
    # Calculate sizes
    bytes_per_sample = bits_per_sample // 8
    byte_rate = sample_rate * num_channels * bytes_per_sample
    block_align = num_channels * bytes_per_sample
    data_size = num_samples * bytes_per_sample

    # Audio format: 3 = IEEE float, 1 = PCM
    audio_format = 3 if bits_per_sample == 32 else 1

    # fmt chunk (18 bytes for extended format)
    fmt_chunk = struct.pack(
        "<4sI2H2I2H",
        b"fmt ",
        16,  # chunk size (16 for PCM, 18 for IEEE float with extension)
        audio_format,  # audio format
        num_channels,  # num channels
        sample_rate,  # sample rate
        byte_rate,  # byte rate
        block_align,  # block align
        bits_per_sample,  # bits per sample
    )

    # data chunk header (size will include all samples)
    data_header = struct.pack("<4sI", b"data", data_size)

    # RIFF header (size will be updated after WTBL chunk is added)
    # Placeholder size; will be updated when finalizing
    riff_size = 4 + len(fmt_chunk) + len(data_header) + data_size
    riff_header = struct.pack("<4sI4s", b"RIFF", riff_size, b"WAVE")

    return riff_header + fmt_chunk + data_header


def write_wtbl_chunk(metadata_bytes: bytes) -> bytes:
    """Create a WTBL chunk from protobuf metadata bytes."""
    chunk_size = len(metadata_bytes)
    header = struct.pack("<4sI", b"WTBL", chunk_size)
    data = header + metadata_bytes

    # Add padding byte for word alignment if needed
    if chunk_size % 2 == 1:
        data += b"\x00"

    return data


def update_riff_size(wav_data: bytearray, additional_size: int) -> None:
    """Update the RIFF size field in WAV data."""
    current_size = struct.unpack_from("<I", wav_data, 4)[0]
    new_size = current_size + additional_size
    struct.pack_into("<I", wav_data, 4, new_size)


# ============================================================================
# Wavetable generation
# ============================================================================


def generate_sine_wavetable(
    frame_length: int = 256,
    num_frames: int = 64,
) -> np.ndarray:
    """
    Generate a simple wavetable with sine waves of increasing harmonic content.

    Args:
        frame_length: Samples per frame
        num_frames: Number of waveform frames

    Returns:
        2D numpy array of shape (num_frames, frame_length)
    """
    wavetable = np.zeros((num_frames, frame_length), dtype=np.float32)

    for frame_idx in range(num_frames):
        # Vary harmonic content across frames
        num_harmonics = 1 + int(frame_idx * 32 / num_frames)
        phase = np.linspace(0, 2 * np.pi, frame_length, endpoint=False)

        waveform = np.zeros(frame_length, dtype=np.float32)
        for h in range(1, num_harmonics + 1):
            # Add harmonics with decreasing amplitude
            amplitude = 1.0 / h
            waveform += amplitude * np.sin(h * phase)

        # Normalize to [-1, 1]
        max_val = np.max(np.abs(waveform))
        if max_val > 0:
            waveform /= max_val

        wavetable[frame_idx] = waveform

    return wavetable


def create_mip_levels(
    base_wavetable: np.ndarray,
    num_mips: int = 7,
) -> list[np.ndarray]:
    """
    Create mip levels by decimating the base wavetable.

    For proper anti-aliasing, you should low-pass filter before decimation.
    This simple example just uses every Nth sample.

    Args:
        base_wavetable: Shape (num_frames, frame_length)
        num_mips: Number of mip levels to create

    Returns:
        List of mip level arrays, each shape (num_frames, mip_frame_length)
    """
    mip_levels = [base_wavetable]
    current = base_wavetable

    for _ in range(1, num_mips):
        # Simple decimation (production code should low-pass filter first)
        decimated = current[:, ::2]  # Take every other sample
        if decimated.shape[1] < 4:  # Minimum reasonable frame length
            break
        mip_levels.append(decimated)
        current = decimated

    return mip_levels


def flatten_mip_levels(mip_levels: list[np.ndarray]) -> np.ndarray:
    """
    Flatten mip levels into a single sample array.

    Order: [mip0_frame0][mip0_frame1]...[mip0_frameN][mip1_frame0]...[mipM_frameN]
    """
    parts = []
    for mip in mip_levels:
        for frame in mip:
            parts.extend(frame)
    return np.array(parts, dtype=np.float32)


# ============================================================================
# Main function
# ============================================================================


def create_wavetable_file(
    output_path: Path,
    frame_length: int = 256,
    num_frames: int = 64,
    sample_rate: int = 44100,
    name: str = "Third-Party Wavetable",
    author: str = "Third-Party Developer",
) -> None:
    """
    Create a complete wavetable file.

    Args:
        output_path: Output WAV file path
        frame_length: Samples per frame at mip level 0
        num_frames: Number of waveform keyframes
        sample_rate: Sample rate for the WAV file
        name: Wavetable name (stored in metadata)
        author: Author attribution (stored in metadata)
    """
    print(f"Creating wavetable: {output_path}")
    print(f"  Frame length: {frame_length}")
    print(f"  Num frames: {num_frames}")

    # Generate wavetable data
    base_wavetable = generate_sine_wavetable(frame_length, num_frames)
    mip_levels = create_mip_levels(base_wavetable)

    mip_frame_lengths = [mip.shape[1] for mip in mip_levels]
    print(f"  Mip levels: {len(mip_levels)}")
    print(f"  Mip frame lengths: {mip_frame_lengths}")

    # Flatten to single array
    samples = flatten_mip_levels(mip_levels)
    total_samples = len(samples)
    print(f"  Total samples: {total_samples}")

    # Create protobuf metadata
    metadata_bytes = create_wavetable_metadata(
        frame_length=frame_length,
        num_frames=num_frames,
        mip_frame_lengths=mip_frame_lengths,
        wavetable_type=5,  # WAVETABLE_TYPE_CUSTOM
        name=name,
        author=author,
    )
    print(f"  Metadata size: {len(metadata_bytes)} bytes")

    # Create WAV file
    wav_header = write_wav_header(total_samples, sample_rate)
    sample_bytes = samples.astype("<f4").tobytes()  # Little-endian float32
    wtbl_chunk = write_wtbl_chunk(metadata_bytes)

    # Combine into final file
    wav_data = bytearray(wav_header + sample_bytes)

    # Update RIFF size to include WTBL chunk
    wtbl_size = len(wtbl_chunk)
    update_riff_size(wav_data, wtbl_size)

    # Append WTBL chunk
    wav_data.extend(wtbl_chunk)

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(bytes(wav_data))

    file_size = len(wav_data)
    print(f"  File size: {file_size} bytes ({file_size / 1024:.1f} KB)")
    print(f"  Written to: {output_path}")


def main():
    """Create example wavetable files."""
    output_dir = Path(__file__).parent

    # Example 1: Simple custom wavetable
    create_wavetable_file(
        output_dir / "simple_wavetable.wav",
        frame_length=256,
        num_frames=64,
        name="Simple Sine Wavetable",
        author="Third-Party Example",
    )
    print()

    # Example 2: High-resolution wavetable
    create_wavetable_file(
        output_dir / "hires_wavetable.wav",
        frame_length=2048,
        num_frames=64,
        name="Hi-Res Wavetable",
        author="Third-Party Example",
    )
    print()

    # Verify the files can be read
    print("=" * 60)
    print("Verification: Use 'rigel wavetable inspect <file>' to verify")
    print("  rigel wavetable inspect simple_wavetable.wav")
    print("  rigel wavetable validate simple_wavetable.wav")
    print()
    print("Or validate with ffprobe:")
    print("  ffprobe simple_wavetable.wav")


if __name__ == "__main__":
    main()
