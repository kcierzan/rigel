# Quickstart: Wavetable Interchange Format

This guide explains how to create, read, and inspect wavetable files using the standardized interchange format.

## Format Overview

The wavetable interchange format uses standard WAV files with an embedded custom RIFF chunk:

```
┌────────────────────────────────────────┐
│ RIFF Header ("WAVE")                   │
├────────────────────────────────────────┤
│ fmt  chunk (audio format)              │
├────────────────────────────────────────┤
│ data chunk (waveform samples)          │
│   - 32-bit float                       │
│   - Mip-major ordering                 │
├────────────────────────────────────────┤
│ WTBL chunk (protobuf metadata)         │
│   - Schema version                     │
│   - Wavetable type                     │
│   - Frame structure                    │
│   - Optional metadata                  │
└────────────────────────────────────────┘
```

## Creating Wavetables (Python/wtgen)

### Basic Export

```python
from wtgen.format import save_wavetable_wav
from wtgen.types import WavetableType

# Generate your wavetable data
mipmaps = generate_mipmaps(...)  # List of numpy arrays

# Export with metadata
save_wavetable_wav(
    path="my_wavetable.wav",
    mipmaps=mipmaps,
    wavetable_type=WavetableType.HIGH_RESOLUTION,
    name="My Custom Wavetable",
    author="Your Name",
)
```

### With Type-Specific Metadata

```python
from wtgen.format import save_wavetable_wav, ClassicDigitalMetadata
from wtgen.types import WavetableType

# PPG-style wavetable with type-specific metadata
save_wavetable_wav(
    path="ppg_style.wav",
    mipmaps=mipmaps,
    wavetable_type=WavetableType.CLASSIC_DIGITAL,
    name="PPG Tribute",
    type_metadata=ClassicDigitalMetadata(
        original_bit_depth=8,
        source_hardware="PPG Wave 2.2",
        harmonic_caps=[128, 64, 16, 8, 4, 2],
    ),
)
```

## Reading Wavetables (Rust/rigel)

### Load Wavetable

```rust
use wavetable_io::{read_wavetable, WavetableFile};
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let wavetable = read_wavetable(Path::new("my_wavetable.wav"))?;

    println!("Name: {:?}", wavetable.metadata.name);
    println!("Type: {:?}", wavetable.metadata.wavetable_type);
    println!("Frames: {}", wavetable.metadata.num_frames);
    println!("Mip levels: {}", wavetable.metadata.num_mip_levels);

    // Access audio data
    for (mip_level, frames) in wavetable.mip_levels.iter().enumerate() {
        println!("Mip {}: {} samples/frame", mip_level, frames[0].len());
    }

    Ok(())
}
```

### Validation

```rust
use wavetable_io::{read_wavetable, validate_wavetable};

let wavetable = read_wavetable(path)?;

// Full validation
match validate_wavetable(&wavetable) {
    Ok(()) => println!("Valid wavetable"),
    Err(e) => eprintln!("Validation failed: {}", e),
}
```

## CLI Inspection Tool

### Inspect Metadata

```bash
# Basic inspection
rigel wavetable inspect my_wavetable.wav

# Output:
# Wavetable: my_wavetable.wav
# ──────────────────────────────
# Schema Version: 1
# Type: HIGH_RESOLUTION
# Name: My Custom Wavetable
# Author: Your Name
#
# Structure:
#   Frame Length: 2048 samples
#   Frames: 64 keyframes
#   Mip Levels: 8
#   Mip Frame Lengths: [2048, 1024, 512, 256, 128, 64, 32, 16]
#
# Audio Data:
#   Total Samples: 261,120
#   File Size: 1.0 MB
#   Peak: 0.98
#   RMS: 0.35
```

### Verbose Output

```bash
rigel wavetable inspect --verbose my_wavetable.wav

# Includes type-specific metadata and raw protobuf fields
```

## Wavetable Types

### CLASSIC_DIGITAL (PPG-Style)

```
Frame Length: 256 samples
Frames: 64
Mip Levels: 7
Typical Size: ~127 KB
```

Suitable for: Vintage PPG Wave emulation, 8-bit character

### HIGH_RESOLUTION

```
Frame Length: 2048+ samples
Frames: 64-256
Mip Levels: 8-11
Typical Size: 500 KB - 5 MB
```

Suitable for: Modern synths, pristine quality, complex waveforms

### VINTAGE_EMULATION

```
Frame Length: 256-512 samples
Frames: 1-16
Mip Levels: 3-5 (limited)
Typical Size: 8-64 KB
```

Suitable for: OSCar, EDP Wasp, and similar vintage oscillators

### PCM_SAMPLE

```
Frame Length: variable (64-4096)
Frames: 1 (typically)
Mip Levels: 1 (no decimation)
Typical Size: 256 B - 16 KB
```

Suitable for: Single-cycle samples, ROM playback emulation

### CUSTOM

```
Frame Length: any valid
Frames: any > 0
Mip Levels: any > 0
```

Suitable for: Experimental, non-standard wavetables

## File Compatibility

### Standard Audio Tools

The format is designed for graceful degradation:

```bash
# Works - audio plays normally
ffplay my_wavetable.wav

# Works - file opens, WTBL chunk ignored
audacity my_wavetable.wav

# Works - basic WAV info shown
mediainfo my_wavetable.wav
```

### Third-Party Creation

To create compatible files without wtgen:

1. Read the protobuf schema: `contracts/wavetable.proto`
2. Generate bindings for your language
3. Create WAV with concatenated frame data
4. Append WTBL chunk with serialized protobuf

```python
# Minimal third-party example
import struct

def append_wtbl_chunk(wav_path: str, metadata_bytes: bytes) -> None:
    """Append WTBL chunk to existing WAV file."""
    with open(wav_path, 'r+b') as f:
        # Read current RIFF size
        f.seek(4)
        current_size = struct.unpack('<I', f.read(4))[0]

        # Seek to end, write chunk
        f.seek(0, 2)
        f.write(b'WTBL')
        f.write(struct.pack('<I', len(metadata_bytes)))
        f.write(metadata_bytes)

        # Pad for word alignment
        if len(metadata_bytes) % 2:
            f.write(b'\x00')

        # Update RIFF size
        chunk_size = 8 + len(metadata_bytes) + (len(metadata_bytes) % 2)
        f.seek(4)
        f.write(struct.pack('<I', current_size + chunk_size))
```

## Error Handling

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| "WTBL chunk not found" | Missing metadata | Re-export with wtgen or add WTBL chunk |
| "Schema version unsupported" | File from future version | Update rigel or use older file |
| "Data length mismatch" | Corrupted file | Re-export or verify frame counts |
| "Invalid wavetable_type" | Unknown enum value | Update rigel (treated as CUSTOM) |

### Debugging

```bash
# Check RIFF structure
ffprobe -show_streams my_wavetable.wav

# Dump raw protobuf (requires protoc)
# Extract WTBL chunk bytes, then:
protoc --decode=rigel.wavetable.WavetableMetadata wavetable.proto < wtbl_bytes
```

## Performance Notes

- File loading: <100ms for typical wavetables
- Memory: ~2x file size during load (file + parsed data)
- Large files (>50 MB): May take several seconds to load

For real-time use, load wavetables during initialization, not during audio processing.
