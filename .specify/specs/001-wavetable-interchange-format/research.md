# Research: Wavetable Interchange Format

**Feature Branch**: `001-wavetable-interchange-format`
**Research Date**: 2026-01-19

## Research Topics

1. Protocol Buffers schema design for forward/backward compatibility
2. Custom RIFF chunk handling in Python and Rust
3. Prost crate setup and no_std considerations

---

## 1. Protocol Buffers Schema Design

### Decision: Use proto3 with versioning best practices

**Rationale**: proto3 is the modern standard with better default behavior for field presence and unknown field handling.

### Key Findings

#### Forward/Backward Compatibility
- **Adding fields is safe**: Old readers ignore unknown fields; new readers see defaults
- **Removing fields requires reservation**: Mark deleted field numbers as `reserved`
- **Never change field wire types**: Create new fields and deprecate old ones
- **Field numbers are immutable**: Once assigned, never repurpose

#### Enum Extensibility
- Always include `_UNSPECIFIED = 0` as the first value
- Unknown enum values are preserved and round-trip correctly
- Reserve old values when deprecating enum variants

**Recommended wavetable_type enum:**
```protobuf
enum WavetableType {
  WAVETABLE_TYPE_UNSPECIFIED = 0;
  WAVETABLE_TYPE_CLASSIC_DIGITAL = 1;
  WAVETABLE_TYPE_HIGH_RESOLUTION = 2;
  WAVETABLE_TYPE_VINTAGE_EMULATION = 3;
  WAVETABLE_TYPE_PCM_SAMPLE = 4;
  WAVETABLE_TYPE_CUSTOM = 5;
  // Reserve 6-20 for future types
  reserved 6 to 20;
}
```

#### Optional vs Implicit Presence
- Use `optional` for fields where "not set" differs from "set to default"
- Proto3 implicit presence cannot distinguish unset from default
- Recommended: Mark all optional metadata fields as `optional`

#### Type-Specific Metadata with oneof
- Use `oneof` for mutually exclusive variant data
- Adding new variants to `oneof` is **not backward compatible** (old readers get unknown)
- Alternative: Use enum discriminator + optional nested messages for more flexibility

**Recommended approach for type-specific metadata:**
```protobuf
message WavetableMetadata {
  // Common fields
  uint32 schema_version = 1;
  WavetableType wavetable_type = 2;
  // ...

  // Type-specific via oneof (simpler but less flexible)
  oneof type_metadata {
    ClassicDigitalMetadata classic_digital = 20;
    HighResolutionMetadata high_resolution = 21;
    VintageEmulationMetadata vintage_emulation = 22;
    PcmSampleMetadata pcm_sample = 23;
  }
}
```

#### Schema Versioning Strategy
- Use field number ranges for versioning (1-20 core v1, 21-40 v2, etc.)
- Include explicit `schema_version` field for readers to gate behavior
- Low field numbers (1-15) use 1 byte on wire; prioritize for frequent fields

### Alternatives Considered

| Format | Pros | Cons | Decision |
|--------|------|------|----------|
| JSON | Human readable, widely supported | Verbose, no schema validation, slower | Rejected |
| MessagePack | Compact binary | No schema, limited tooling | Rejected |
| FlatBuffers | Zero-copy reads | More complex, less ecosystem | Rejected |
| Protocol Buffers | Industry standard, excellent compat, typed schema, fast | Binary (not human readable) | **Selected** |

---

## 2. Custom RIFF Chunk Handling

### Decision: Manual RIFF chunk manipulation (Python), riff crate (Rust)

**Rationale**: Existing audio libraries (soundfile, scipy, hound) don't support custom chunk APIs. Manual implementation provides full control.

### Python Findings

#### Library Assessment

| Library | Custom Chunk Support | Notes |
|---------|---------------------|-------|
| soundfile | No | High-level API only |
| scipy.io.wavfile | No | Basic WAV read/write only |
| wave (stdlib) | No | Standard chunks only |
| Manual struct.pack | Yes | Full control, ~100 LOC |

**Recommendation**: Implement custom `RIFFChunkHandler` class using `struct.pack/unpack`.

**Key implementation details:**
- RIFF chunks are 8-byte header (4-byte FourCC + 4-byte size) + payload
- Chunks must be word-aligned (2-byte boundaries)
- RIFF file size at bytes 4-7 must be updated when appending chunks
- All numeric fields are little-endian

**Approach:**
1. Use soundfile to write standard WAV data
2. Append WTBL chunk after WAV data using manual binary writes
3. Update RIFF size header

#### Implementation Sketch (Python)
```python
import struct
from pathlib import Path

def append_wtbl_chunk(wav_path: Path, protobuf_data: bytes) -> None:
    """Append WTBL chunk to existing WAV file."""
    with open(wav_path, 'r+b') as f:
        # Read current RIFF size
        f.seek(4)
        current_size = struct.unpack('<I', f.read(4))[0]

        # Seek to end
        f.seek(0, 2)

        # Write WTBL chunk header + data
        chunk_size = len(protobuf_data)
        f.write(b'WTBL')
        f.write(struct.pack('<I', chunk_size))
        f.write(protobuf_data)

        # Pad if needed for word alignment
        if chunk_size % 2:
            f.write(b'\x00')
            chunk_size += 1

        # Update RIFF size (8 bytes header + data)
        new_riff_size = current_size + 8 + chunk_size
        f.seek(4)
        f.write(struct.pack('<I', new_riff_size))
```

### Rust Findings

#### Library Assessment

| Library | Custom Chunk Support | Notes |
|---------|---------------------|-------|
| hound 3.5 (current) | No | Write-only, no chunk API |
| riff 2.0 | Yes | Full RIFF support, clean API |
| Manual implementation | Yes | ~50 LOC |

**Recommendation**: Add `riff = "2.0"` crate for comprehensive RIFF support.

**Key features of riff crate:**
- `Chunk` struct with `id()`, `len()`, `content()`
- Nested chunk iteration for LIST chunks
- Writing chunks via `ChunkContents` builder

#### Implementation Sketch (Rust)
```rust
use riff::{Chunk, ChunkContents};
use std::io::{Read, Seek, Write};

fn append_wtbl_chunk<W: Write + Seek>(
    writer: &mut W,
    protobuf_data: &[u8],
) -> std::io::Result<()> {
    let wtbl_chunk = ChunkContents::Data(
        riff::ChunkId::new(b"WTBL").unwrap(),
        protobuf_data.to_vec(),
    );
    wtbl_chunk.write(writer)
}

fn read_wtbl_chunk(wav_data: &[u8]) -> Option<Vec<u8>> {
    let chunk = Chunk::new(wav_data).ok()?;
    for child in chunk.iter() {
        if child.id() == riff::ChunkId::new(b"WTBL").ok()? {
            return Some(child.content().to_vec());
        }
    }
    None
}
```

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Custom binary format | Full control | No standard tool support | Rejected |
| Separate sidecar file | Simple implementation | Two files to manage | Rejected |
| Embed in WAV comment | Uses existing chunk | Limited size, text only | Rejected |
| Custom RIFF chunk | Standard format, transparent to audio tools | Need manual implementation | **Selected** |

---

## 3. Prost Crate Setup

### Decision: Use prost with OUT_DIR code generation

**Rationale**: Prost is the Rust standard for protobuf, actively maintained, and integrates with Cargo's build system.

### Key Findings

#### Setup Requirements

**Cargo.toml:**
```toml
[dependencies]
prost = "0.14"
bytes = "1"

[build-dependencies]
prost-build = "0.14"
```

**build.rs:**
```rust
use prost_build::Config;

fn main() {
    let mut config = Config::new();
    config
        .compile_protos(
            &["proto/wavetable.proto"],
            &["proto"],
        )
        .expect("Failed to compile protobuf definitions");

    println!("cargo:rerun-if-changed=proto/");
}
```

**Including generated code (lib.rs):**
```rust
pub mod wavetable {
    include!(concat!(env!("OUT_DIR"), "/rigel.wavetable.rs"));
}
```

#### Protoc Requirement
- **Required by default**: prost-build invokes protoc during build
- **Solution for Nix**: Add `protobuf` to devenv.nix packages
- **Alternative**: Commit generated code with `skip_protoc_run()`

#### no_std Considerations
- Prost requires `alloc` crate (not truly zero-allocation)
- wavetable-io crate will NOT be no_std (file I/O requires allocation)
- This is fine per Constitution - file I/O stays out of DSP core

#### Unknown Fields
- Proto3 prost **automatically preserves unknown fields**
- Unknown fields survive encode/decode cycles
- Critical for forward compatibility

#### Error Handling
- `DecodeError`: Malformed input (best-effort error message)
- `EncodeError`: Buffer overflow only (use `encode_to_vec()` to avoid)
- Recommend using `anyhow::Context` for error chains

### Devenv Integration

Add to root `devenv.nix`:
```nix
packages = with pkgs; [
  protobuf  # For protoc compiler
  # ... existing packages
];
```

Python shell at `projects/wtgen/devenv.nix`:
```nix
packages = with pkgs; [
  protobuf  # For protoc
  # ... existing packages
];
```

---

## Summary of Decisions

| Topic | Decision | Rationale |
|-------|----------|-----------|
| Schema format | Protocol Buffers (proto3) | Industry standard, typed, excellent compat |
| Schema versioning | Field number ranges + explicit version field | Clear evolution path |
| Python RIFF handling | Manual struct.pack implementation | No library supports custom chunks |
| Rust RIFF handling | riff 2.0 crate | Clean API, full RIFF support |
| Rust protobuf | prost 0.14 with OUT_DIR generation | Standard, cargo-integrated |
| Enum default | UNSPECIFIED = 0 always first | Handles unknown values gracefully |
| Type-specific metadata | oneof in protobuf message | Type-safe, mutually exclusive |

---

## Dependencies to Add

### Rust (wavetable-io crate)
```toml
[dependencies]
prost = "0.14"
bytes = "1"
riff = "2.0"
anyhow = "1.0"

[build-dependencies]
prost-build = "0.14"
```

### Python (wtgen)
```toml
# pyproject.toml
dependencies = [
    # ... existing
    "protobuf>=5.0",
]
```

### Devenv (both shells)
```nix
packages = with pkgs; [
    protobuf  # protoc compiler
];
```
