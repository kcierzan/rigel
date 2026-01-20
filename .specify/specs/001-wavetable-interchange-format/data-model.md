# Data Model: Wavetable Interchange Format

**Feature Branch**: `001-wavetable-interchange-format`
**Date**: 2026-01-19

## Entity Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Wavetable File (WAV)                             │
├─────────────────────────────────────────────────────────────────────────┤
│  RIFF Header                                                            │
│  ├── fmt  chunk (audio format)                                          │
│  ├── data chunk (waveform samples)                                      │
│  │   └── [mip0_frame0][mip0_frame1]...[mip1_frame0][mip1_frame1]...    │
│  └── WTBL chunk (protobuf metadata)                                     │
│       └── WavetableMetadata message                                     │
│           ├── schema_version                                            │
│           ├── wavetable_type                                            │
│           ├── frame_length                                              │
│           ├── num_frames                                                │
│           ├── num_mip_levels                                            │
│           ├── mip_frame_lengths[]                                       │
│           ├── normalization_method                                      │
│           ├── source_bit_depth                                          │
│           ├── author                                                    │
│           ├── name (optional)                                           │
│           ├── description (optional)                                    │
│           ├── tuning_reference (optional)                               │
│           ├── generation_parameters (optional)                          │
│           └── type_metadata (oneof)                                     │
│               ├── ClassicDigitalMetadata                                │
│               ├── HighResolutionMetadata                                │
│               ├── VintageEmulationMetadata                              │
│               └── PcmSampleMetadata                                     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Entities

### 1. WavetableFile

**Description**: The top-level container representing a wavetable stored as a RIFF/WAV file with embedded metadata.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| riff_header | RiffHeader | Yes | Standard RIFF container with "WAVE" format |
| fmt_chunk | FmtChunk | Yes | Audio format specification |
| data_chunk | DataChunk | Yes | Concatenated waveform samples |
| wtbl_chunk | WtblChunk | Yes | Protobuf-encoded metadata |

**Constraints**:
- File MUST be valid RIFF/WAV format
- WTBL chunk MUST be present
- Audio data length MUST match metadata declarations

---

### 2. WavetableMetadata (Protobuf Message)

**Description**: The protobuf message stored in the WTBL chunk containing all wavetable metadata.

#### Core Fields (v1, field numbers 1-15)

| Field | Type | Field # | Required | Description |
|-------|------|---------|----------|-------------|
| schema_version | uint32 | 1 | Yes | Schema version for compatibility checking (current: 1) |
| wavetable_type | WavetableType | 2 | Yes | Classification of wavetable |
| frame_length | uint32 | 3 | Yes | Samples per frame at mip level 0 |
| num_frames | uint32 | 4 | Yes | Number of keyframes per mip level |
| num_mip_levels | uint32 | 5 | Yes | Number of mip levels (1-11 typical) |
| mip_frame_lengths | repeated uint32 | 6 | Yes | Frame length for each mip level |

#### Metadata Fields (v1, field numbers 16-30)

| Field | Type | Field # | Required | Description |
|-------|------|---------|----------|-------------|
| normalization_method | NormalizationMethod | 16 | No | How audio was normalized |
| source_bit_depth | uint32 | 17 | No | Original bit depth (8, 16, 24, 32) |
| author | string | 18 | No | Creator attribution |
| name | string | 19 | No | Human-readable name |
| description | string | 20 | No | Extended description |
| tuning_reference | float | 21 | No | Reference frequency in Hz (default 440.0) |
| generation_parameters | string | 22 | No | JSON-encoded generation params |

#### Type-Specific Metadata (field numbers 50-60)

| Field | Type | Field # | Required | Description |
|-------|------|---------|----------|-------------|
| type_metadata | oneof | 50-53 | No | Type-specific metadata container |

**Validation Rules**:
- schema_version MUST be >= 1
- wavetable_type MUST be valid enum value (unknown = CUSTOM)
- frame_length MUST be > 0 and power of 2
- num_frames MUST be > 0
- num_mip_levels MUST be > 0
- mip_frame_lengths MUST have exactly num_mip_levels entries
- mip_frame_lengths[0] MUST equal frame_length
- Each mip_frame_lengths[i] MUST be >= mip_frame_lengths[i+1] (decreasing)

---

### 3. WavetableType (Enum)

**Description**: Classification of the wavetable for appropriate handling.

| Value | Number | Description |
|-------|--------|-------------|
| WAVETABLE_TYPE_UNSPECIFIED | 0 | Unknown/invalid (default for missing) |
| WAVETABLE_TYPE_CLASSIC_DIGITAL | 1 | PPG Wave-style (256 samples, 64 frames, 8-bit source) |
| WAVETABLE_TYPE_HIGH_RESOLUTION | 2 | Modern digital (2048+ samples, many frames) |
| WAVETABLE_TYPE_VINTAGE_EMULATION | 3 | Vintage oscillator emulations (OSCar, Wasp) |
| WAVETABLE_TYPE_PCM_SAMPLE | 4 | Single-cycle PCM samples (AWM/SY99-style) |
| WAVETABLE_TYPE_CUSTOM | 5 | User-defined, no specific handling |

**Reserved**: Values 6-20 for future types

**Handling**:
- Unknown values from future versions MUST be treated as CUSTOM
- Reader MUST NOT fail on unknown enum values

---

### 4. NormalizationMethod (Enum)

**Description**: How the audio samples were normalized.

| Value | Number | Description |
|-------|--------|-------------|
| NORMALIZATION_UNSPECIFIED | 0 | Not specified |
| NORMALIZATION_PEAK | 1 | Normalized to peak amplitude |
| NORMALIZATION_RMS | 2 | Normalized to target RMS |
| NORMALIZATION_NONE | 3 | No normalization applied |

---

### 5. ClassicDigitalMetadata

**Description**: Type-specific metadata for CLASSIC_DIGITAL wavetables.

| Field | Type | Field # | Required | Description |
|-------|------|---------|----------|-------------|
| original_bit_depth | uint32 | 1 | No | Source bit depth (typically 8) |
| original_sample_rate | uint32 | 2 | No | Source sample rate |
| source_hardware | string | 3 | No | Original hardware (e.g., "PPG Wave 2.2") |
| harmonic_caps | repeated uint32 | 4 | No | Max harmonics per mip level |

---

### 6. HighResolutionMetadata

**Description**: Type-specific metadata for HIGH_RESOLUTION wavetables.

| Field | Type | Field # | Required | Description |
|-------|------|---------|----------|-------------|
| max_harmonics | uint32 | 1 | No | Maximum harmonic count at mip 0 |
| interpolation_hint | InterpolationHint | 2 | No | Suggested interpolation method |
| source_synth | string | 3 | No | Source synthesizer if known |

---

### 7. VintageEmulationMetadata

**Description**: Type-specific metadata for VINTAGE_EMULATION wavetables.

| Field | Type | Field # | Required | Description |
|-------|------|---------|----------|-------------|
| emulated_hardware | string | 1 | No | Target hardware (e.g., "OSCar", "EDP Wasp") |
| oscillator_type | string | 2 | No | Oscillator variant |
| preserves_aliasing | bool | 3 | No | Whether aliasing artifacts are intentional |

---

### 8. PcmSampleMetadata

**Description**: Type-specific metadata for PCM_SAMPLE wavetables.

| Field | Type | Field # | Required | Description |
|-------|------|---------|----------|-------------|
| original_sample_rate | uint32 | 1 | No | Source sample rate |
| root_note | uint32 | 2 | No | MIDI note number for unity playback |
| loop_start | uint32 | 3 | No | Loop start sample (if applicable) |
| loop_end | uint32 | 4 | No | Loop end sample (if applicable) |

---

### 9. InterpolationHint (Enum)

**Description**: Suggested interpolation method for playback.

| Value | Number | Description |
|-------|--------|-------------|
| INTERPOLATION_UNSPECIFIED | 0 | Use reader default |
| INTERPOLATION_LINEAR | 1 | Linear interpolation |
| INTERPOLATION_CUBIC | 2 | Cubic (Catmull-Rom) |
| INTERPOLATION_SINC | 3 | Windowed sinc |

---

## Data Layout

### WAV Audio Data (data chunk)

Samples are stored as **mip-major, frame-secondary** ordering:

```
[mip0_frame0][mip0_frame1]...[mip0_frameN-1]  // All frames at mip 0
[mip1_frame0][mip1_frame1]...[mip1_frameN-1]  // All frames at mip 1
...
[mipM_frame0][mipM_frame1]...[mipM_frameN-1]  // All frames at mip M
```

**Sample Format**: 32-bit IEEE 754 floating point (channels=1, sample_rate=from context)

**Total Sample Count**:
```
total_samples = sum(mip_frame_lengths[i] * num_frames for i in 0..num_mip_levels)
```

### Example: Classic Digital (256 samples, 64 frames, 7 mips)

| Mip Level | Frame Length | Frames | Samples | Cumulative |
|-----------|--------------|--------|---------|------------|
| 0 | 256 | 64 | 16,384 | 16,384 |
| 1 | 128 | 64 | 8,192 | 24,576 |
| 2 | 64 | 64 | 4,096 | 28,672 |
| 3 | 32 | 64 | 2,048 | 30,720 |
| 4 | 16 | 64 | 1,024 | 31,744 |
| 5 | 8 | 64 | 512 | 32,256 |
| 6 | 4 | 64 | 256 | 32,512 |

**File size**: 32,512 samples × 4 bytes = 130,048 bytes (~127 KB audio data)

---

## State Transitions

Wavetable files are immutable data; no state transitions apply. The format is write-once, read-many.

---

## Validation Summary

### Structural Validation (MUST pass)

1. Valid RIFF/WAV structure
2. WTBL chunk present
3. Protobuf decodes successfully
4. schema_version >= 1
5. num_frames > 0
6. num_mip_levels > 0
7. mip_frame_lengths has correct count
8. Audio data length matches declared total

### Content Validation (SHOULD pass)

1. frame_length is power of 2
2. mip_frame_lengths are decreasing powers of 2
3. Samples are finite (no NaN/Inf)
4. wavetable_type is known enum value

### Graceful Degradation

- Unknown wavetable_type → treat as CUSTOM
- Unknown protobuf fields → preserve, ignore
- Missing optional fields → use sensible defaults
- Newer schema_version with unknown fields → read known fields only
