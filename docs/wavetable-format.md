# Wavetable Interchange Format Specification

**Version**: 1.0
**Date**: 2026-01-19
**Status**: Stable

This document specifies the Rigel wavetable interchange format, a standardized file format for wavetable data storage and exchange between tools.

## Overview

The wavetable interchange format uses a standard WAV file container with an embedded custom RIFF chunk (`WTBL`) containing Protocol Buffers encoded metadata. This design ensures:

1. **Compatibility**: Files play as audio in any standard WAV player
2. **Extensibility**: Protobuf allows forward/backward compatible schema evolution
3. **Completeness**: All wavetable metadata travels with the audio data

## File Structure

```
┌────────────────────────────────────────────────────────┐
│ RIFF Header (12 bytes)                                 │
│   - "RIFF"     : 4 bytes (FourCC identifier)           │
│   - file_size  : 4 bytes (little-endian u32)           │
│   - "WAVE"     : 4 bytes (format identifier)           │
├────────────────────────────────────────────────────────┤
│ fmt  chunk (24-26 bytes typical)                       │
│   - "fmt "     : 4 bytes (chunk ID)                    │
│   - chunk_size : 4 bytes (16 or more)                  │
│   - format     : 2 bytes (3 = IEEE float)              │
│   - channels   : 2 bytes (1 = mono)                    │
│   - sample_rate: 4 bytes (e.g., 44100)                 │
│   - byte_rate  : 4 bytes                               │
│   - block_align: 2 bytes                               │
│   - bit_depth  : 2 bytes (32)                          │
├────────────────────────────────────────────────────────┤
│ data chunk (variable size)                             │
│   - "data"     : 4 bytes (chunk ID)                    │
│   - chunk_size : 4 bytes (sample count × 4)            │
│   - samples    : float32[] (mip-major, frame-secondary)│
├────────────────────────────────────────────────────────┤
│ WTBL chunk (variable size)                             │
│   - "WTBL"     : 4 bytes (chunk ID)                    │
│   - chunk_size : 4 bytes                               │
│   - protobuf   : bytes (WavetableMetadata message)     │
└────────────────────────────────────────────────────────┘
```

## Audio Data Layout

### Sample Format

- **Format**: 32-bit IEEE 754 floating-point
- **Channels**: 1 (mono)
- **Range**: -1.0 to +1.0 (normalized)
- **Byte order**: Little-endian

### Sample Ordering

Samples are stored in **mip-major, frame-secondary** order:

```
[mip0_frame0][mip0_frame1]...[mip0_frameN-1]  // All frames at mip 0
[mip1_frame0][mip1_frame1]...[mip1_frameN-1]  // All frames at mip 1
...
[mipM_frame0][mipM_frame1]...[mipM_frameN-1]  // All frames at mip M
```

Each `mipX_frameY` contains `mip_frame_lengths[X]` consecutive samples.

### Total Sample Count

```
total_samples = sum(mip_frame_lengths[i] * num_frames for i in 0..num_mip_levels)
```

### Example: Classic Digital Wavetable

A PPG-style wavetable with 256 samples, 64 frames, 7 mip levels:

| Mip Level | Frame Length | Frames | Samples per Mip | Cumulative |
|-----------|--------------|--------|-----------------|------------|
| 0         | 256          | 64     | 16,384          | 16,384     |
| 1         | 128          | 64     | 8,192           | 24,576     |
| 2         | 64           | 64     | 4,096           | 28,672     |
| 3         | 32           | 64     | 2,048           | 30,720     |
| 4         | 16           | 64     | 1,024           | 31,744     |
| 5         | 8            | 64     | 512             | 32,256     |
| 6         | 4            | 64     | 256             | 32,512     |

**Total**: 32,512 samples × 4 bytes = 130,048 bytes audio data

## WTBL Chunk

### Chunk Header

| Offset | Size | Description |
|--------|------|-------------|
| 0      | 4    | Chunk ID: "WTBL" (0x5754424C) |
| 4      | 4    | Chunk size (little-endian u32) |
| 8      | N    | Protobuf-encoded WavetableMetadata |

### Word Alignment

RIFF chunks must be word-aligned (2-byte boundaries). If the protobuf data has odd length, a single padding byte (0x00) is appended after the data but NOT included in chunk_size.

## Protocol Buffers Schema

The metadata is encoded using Protocol Buffers (proto3). The canonical schema is located at `proto/wavetable.proto`.

### WavetableMetadata Message

```protobuf
message WavetableMetadata {
  // Core fields (required)
  uint32 schema_version = 1;       // Must be >= 1
  WavetableType wavetable_type = 2;
  uint32 frame_length = 3;         // Samples per frame at mip 0
  uint32 num_frames = 4;           // Keyframes per mip level
  uint32 num_mip_levels = 5;       // Number of mip levels
  repeated uint32 mip_frame_lengths = 6;  // Frame length per mip

  // Metadata fields (optional)
  NormalizationMethod normalization_method = 16;
  optional uint32 source_bit_depth = 17;
  optional string author = 18;
  optional string name = 19;
  optional string description = 20;
  optional float tuning_reference = 21;
  optional string generation_parameters = 22;
  optional uint32 sample_rate = 23;

  // Type-specific metadata
  oneof type_metadata {
    ClassicDigitalMetadata classic_digital = 50;
    HighResolutionMetadata high_resolution = 51;
    VintageEmulationMetadata vintage_emulation = 52;
    PcmSampleMetadata pcm_sample = 53;
  }
}
```

### Wavetable Types

| Value | Name | Description |
|-------|------|-------------|
| 0     | UNSPECIFIED | Unknown/invalid (treat as CUSTOM) |
| 1     | CLASSIC_DIGITAL | PPG Wave-style (256 samples, 64 frames, 8-bit source) |
| 2     | HIGH_RESOLUTION | Modern digital (2048+ samples, many frames) |
| 3     | VINTAGE_EMULATION | Vintage oscillator emulations (OSCar, Wasp) |
| 4     | PCM_SAMPLE | Single-cycle PCM samples (AWM/SY99-style) |
| 5     | CUSTOM | User-defined, no specific handling |

### Normalization Methods

| Value | Name | Description |
|-------|------|-------------|
| 0     | UNSPECIFIED | Not specified |
| 1     | PEAK | Normalized to peak amplitude |
| 2     | RMS | Normalized to target RMS |
| 3     | NONE | No normalization applied |

### Type-Specific Metadata

#### ClassicDigitalMetadata

```protobuf
message ClassicDigitalMetadata {
  optional uint32 original_bit_depth = 1;   // Typically 8
  optional uint32 original_sample_rate = 2;
  optional string source_hardware = 3;       // e.g., "PPG Wave 2.2"
  repeated uint32 harmonic_caps = 4;         // Max harmonics per mip
}
```

#### HighResolutionMetadata

```protobuf
message HighResolutionMetadata {
  optional uint32 max_harmonics = 1;
  InterpolationHint interpolation_hint = 2;
  optional string source_synth = 3;
}
```

#### VintageEmulationMetadata

```protobuf
message VintageEmulationMetadata {
  optional string emulated_hardware = 1;     // e.g., "OSCar", "EDP Wasp"
  optional string oscillator_type = 2;
  optional bool preserves_aliasing = 3;
}
```

#### PcmSampleMetadata

```protobuf
message PcmSampleMetadata {
  optional uint32 original_sample_rate = 1;
  optional uint32 root_note = 2;             // MIDI note (60 = C4)
  optional uint32 loop_start = 3;            // Sample index
  optional uint32 loop_end = 4;              // Sample index
}
```

## Validation Rules

### Required (MUST pass)

1. Valid RIFF/WAV structure
2. WTBL chunk present
3. Protobuf decodes without error
4. `schema_version` >= 1
5. `frame_length` > 0
6. `num_frames` > 0
7. `num_mip_levels` > 0
8. `mip_frame_lengths` has exactly `num_mip_levels` entries
9. `mip_frame_lengths[0]` equals `frame_length`
10. `mip_frame_lengths` values are decreasing (mip 0 = highest resolution)
11. Audio data length matches calculated total samples × 4
12. **FR-028**: All samples MUST be finite (no NaN or Infinity values). Files containing non-finite sample values MUST be rejected with a clear error message.
13. **FR-030b**: File size MUST NOT exceed 100 MB. Files larger than 100 MB MUST be rejected with a clear error message.

### Recommended (SHOULD pass)

1. `frame_length` is power of 2
2. All `mip_frame_lengths` values are powers of 2
3. `wavetable_type` is a known enum value

### Graceful Degradation

Readers MUST handle the following gracefully:

- **Unknown wavetable_type**: Treat as CUSTOM
- **Unknown protobuf fields**: Preserve and ignore (forward compatibility)
- **Missing optional fields**: Use sensible defaults
- **Newer schema_version with unknown fields**: Read known fields only

## Implementation Guide

### Reading a Wavetable File

1. **Check file size** (FR-030b): Verify file is ≤ 100 MB before reading
2. Parse RIFF header and verify "WAVE" format
3. Read through chunks, looking for "fmt ", "data", and "WTBL"
4. Verify audio format: 32-bit float, mono
5. Decode WTBL chunk protobuf as WavetableMetadata
6. Calculate expected sample count from metadata
7. Verify data chunk size matches expected
8. Parse audio data bytes to float samples
9. **Validate samples** (FR-028): Verify all samples are finite (no NaN/Infinity)
10. Split samples into mip levels and frames

### Writing a Wavetable File

1. Serialize WavetableMetadata to protobuf bytes
2. Calculate total audio samples
3. Write RIFF header with placeholder size
4. Write fmt chunk (32-bit float, mono, desired sample rate)
5. Write data chunk with flattened samples (mip-major order)
6. Write WTBL chunk with protobuf data
7. Seek back and update RIFF size

### Code Example (Python)

```python
import struct
from pathlib import Path

def read_wtbl_chunk(wav_path: Path) -> bytes:
    """Extract WTBL chunk from a WAV file."""
    with open(wav_path, 'rb') as f:
        # Verify RIFF header
        riff = f.read(4)
        if riff != b'RIFF':
            raise ValueError("Not a RIFF file")

        file_size = struct.unpack('<I', f.read(4))[0]
        wave = f.read(4)
        if wave != b'WAVE':
            raise ValueError("Not a WAVE file")

        # Search for WTBL chunk
        while f.tell() < file_size + 8:
            chunk_id = f.read(4)
            if len(chunk_id) < 4:
                break
            chunk_size = struct.unpack('<I', f.read(4))[0]

            if chunk_id == b'WTBL':
                return f.read(chunk_size)

            # Skip chunk (with word alignment)
            skip = chunk_size + (chunk_size % 2)
            f.seek(skip, 1)

        raise ValueError("WTBL chunk not found")


def append_wtbl_chunk(wav_path: Path, protobuf_data: bytes) -> None:
    """Append WTBL chunk to existing WAV file."""
    with open(wav_path, 'r+b') as f:
        # Read current RIFF size
        f.seek(4)
        current_size = struct.unpack('<I', f.read(4))[0]

        # Seek to end
        f.seek(0, 2)

        # Write WTBL chunk
        f.write(b'WTBL')
        f.write(struct.pack('<I', len(protobuf_data)))
        f.write(protobuf_data)

        # Pad if needed
        if len(protobuf_data) % 2:
            f.write(b'\x00')

        # Update RIFF size
        chunk_addition = 8 + len(protobuf_data) + (len(protobuf_data) % 2)
        f.seek(4)
        f.write(struct.pack('<I', current_size + chunk_addition))
```

### Code Example (Rust)

```rust
use prost::Message;
use std::io::{Read, Seek, SeekFrom, Write};

fn read_wtbl_chunk<R: Read + Seek>(reader: &mut R) -> std::io::Result<Vec<u8>> {
    // Skip RIFF header
    reader.seek(SeekFrom::Start(12))?;

    loop {
        let mut chunk_id = [0u8; 4];
        if reader.read_exact(&mut chunk_id).is_err() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "WTBL chunk not found",
            ));
        }

        let mut size_bytes = [0u8; 4];
        reader.read_exact(&mut size_bytes)?;
        let chunk_size = u32::from_le_bytes(size_bytes) as usize;

        if &chunk_id == b"WTBL" {
            let mut data = vec![0u8; chunk_size];
            reader.read_exact(&mut data)?;
            return Ok(data);
        }

        // Skip chunk with word alignment
        let skip = chunk_size + (chunk_size % 2);
        reader.seek(SeekFrom::Current(skip as i64))?;
    }
}

fn write_wtbl_chunk<W: Write + Seek>(
    writer: &mut W,
    protobuf_data: &[u8],
) -> std::io::Result<()> {
    // Seek to end
    writer.seek(SeekFrom::End(0))?;

    // Write chunk
    writer.write_all(b"WTBL")?;
    writer.write_all(&(protobuf_data.len() as u32).to_le_bytes())?;
    writer.write_all(protobuf_data)?;

    // Word alignment padding
    if protobuf_data.len() % 2 != 0 {
        writer.write_all(&[0u8])?;
    }

    Ok(())
}
```

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-19 | Initial specification |

## References

- [RIFF File Format](https://en.wikipedia.org/wiki/Resource_Interchange_File_Format)
- [WAV File Format](https://www.mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/WAVE.html)
- [Protocol Buffers](https://protobuf.dev/)
- Rigel proto schema: `proto/wavetable.proto`
