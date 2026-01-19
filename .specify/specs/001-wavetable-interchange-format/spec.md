# Feature Specification: Wavetable Interchange Format

**Feature Branch**: `001-wavetable-interchange-format`
**Created**: 2026-01-19
**Status**: Draft
**Input**: User description: "Create and standardize on a wavetable interchange format between rigel and wtgen"
**Linear Issue**: [NEW-7](https://linear.app/new-atlantis/issue/NEW-7/standardize-on-wavetable-interchange-format)

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Generate Wavetables in wtgen for rigel (Priority: P1)

A sound designer uses wtgen (the Python toolkit) to create custom wavetables using various waveform generation methods (sawtooth, square, triangle, sine, etc.) and processing (EQ, tilt). They export these wavetables as WAV files with embedded metadata that rigel-synth can load and use for sound synthesis. The format supports multiple wavetable types including classic PPG-style, high-resolution vintage digital, vintage oscillator emulations, and short PCM samples.

**Why this priority**: This is the fundamental use case - the entire pipeline depends on being able to move wavetable data from the generation tool to the synthesizer. Without this, no other functionality matters.

**Independent Test**: Can be fully tested by generating a wavetable in wtgen, exporting it as a WAV file, and verifying the file contains valid audio data plus parseable protobuf metadata that correctly identifies the wavetable type.

**Acceptance Scenarios**:

1. **Given** wtgen has generated a wavetable with multiple keyframes and mip levels, **When** the user exports it, **Then** a standard WAV file is created containing all waveform data sequentially plus a custom RIFF chunk with protobuf metadata including the wavetable type.

2. **Given** an exported wavetable WAV file of any supported type, **When** opened in any standard audio editor, **Then** the audio data is visible and playable (the custom metadata chunk is ignored gracefully by standard tools).

3. **Given** a wavetable file with incomplete or corrupted protobuf metadata, **When** a reader attempts to parse it, **Then** a clear error is reported indicating what validation failed.

4. **Given** wavetables of different types (PPG-style, high-res, vintage emulation, PCM sample), **When** exported and re-imported, **Then** the wavetable type metadata is preserved and correctly identifies each type.

---

### User Story 2 - Inspect Wavetable Metadata (Priority: P2)

A developer wants to inspect the protobuf metadata embedded in a wavetable WAV file to understand its properties, verify schema version compatibility, identify the wavetable type, or debug format issues. They use a command-line tool in the dev environment to display the decoded metadata.

**Why this priority**: Debugging and verification are essential for iterating on wavetable designs and troubleshooting format issues. Good tooling accelerates development workflows.

**Independent Test**: Can be fully tested by running an inspection command on a wavetable file and verifying all protobuf fields are correctly decoded and displayed, including wavetable type classification.

**Acceptance Scenarios**:

1. **Given** a valid wavetable WAV file, **When** the developer runs the inspection tool, **Then** the output displays: schema version, wavetable type, frame length, number of frames, number of mip levels, normalization method, bit depth, author, and type-specific parameters.

2. **Given** a wavetable file with a newer schema version containing unknown fields, **When** the inspection tool parses it, **Then** known fields are displayed correctly and unknown fields are indicated without causing parse failures.

3. **Given** the protobuf schema definition, **When** a developer needs to decode metadata manually, **Then** they can use standard protobuf tools (protoc, prost) to inspect the WTBL chunk contents.

---

### User Story 3 - Third-Party Wavetable Creation (Priority: P3)

A developer creating tools or presets wants to generate wavetables from their own pipeline and have them work with rigel. They follow the documented RIFF/WAV format specification and protobuf schema to create compliant wavetable files for any of the supported wavetable types.

**Why this priority**: Ecosystem growth depends on third-party tools being able to create compatible wavetables. Using standard WAV format with documented protobuf schema lowers the barrier to entry.

**Independent Test**: Can be fully tested by creating a wavetable file manually following the specification and verifying it passes validation.

**Acceptance Scenarios**:

1. **Given** a developer has read the format specification and protobuf schema, **When** they create a wavetable WAV file following the documented structure for any supported type, **Then** the file passes validation and metadata inspection tools work correctly.

2. **Given** the protobuf schema file (.proto), **When** a developer generates bindings for their language, **Then** they can create valid metadata for all wavetable types without examining wtgen source code.

---

### User Story 4 - Convert Legacy Wavetable Formats (Priority: P4)

A sound designer has existing wavetables in legacy formats (raw dumps, proprietary formats, or extracted from vintage hardware) and wants to convert them to the standardized format with appropriate type metadata.

**Why this priority**: Supporting the broader ecosystem of existing wavetable content expands the library of sounds available to users.

**Independent Test**: Can be fully tested by importing a legacy format file and verifying the conversion preserves audio fidelity and assigns appropriate type metadata.

**Acceptance Scenarios**:

1. **Given** a raw PCM dump from vintage hardware, **When** imported with appropriate parameters, **Then** wtgen creates a valid wavetable file with correct type metadata (e.g., vintage_emulation or pcm_sample).

2. **Given** a high-resolution wavetable extracted from a modern synth, **When** converted to the standard format, **Then** the full resolution is preserved and metadata reflects high_resolution type.

---

### Edge Cases

- **Zero keyframes or mip levels**: Rejected as structurally invalid with clear error (see FR-030a).
- **Unknown protobuf fields (forward compatibility)**: Unknown fields from newer schema versions are preserved during read/write operations and do not cause parse failures (see FR-012).
- **Missing required protobuf fields (backward compatibility)**: Files missing required fields (schema_version, wavetable_type, frame_length, num_frames, num_mip_levels) are rejected with a clear error indicating which field is missing.
- **Extremely large wavetable files**: Files exceeding 100MB are rejected with a clear error (see FR-030b). This provides a sanity check while supporting all practical wavetable use cases.
- **Missing WTBL chunk**: Rejected with a clear error; the WTBL chunk is required for a valid wavetable file (see FR-025).
- **Data/metadata length mismatch**: Rejected with error; strict validation requires exact match (see FR-026).
- **Non-standard WAV sample rates/bit depths**: The WAV header sample rate is informational only (wavetables are single-cycle, rate-agnostic). Readers accept any valid WAV sample rate. Bit depth in WAV header MUST be 32-bit float for storage; the `source_bit_depth` metadata field captures the original source resolution.
- **NaN or infinity sample values**: Readers MUST reject wavetable files containing non-finite sample values (NaN or infinity) with a clear error (see FR-028).
- **Unknown wavetable type values (future extensibility)**: Unknown `wavetable_type` values from future versions are treated as `CUSTOM` without causing parse failures (see FR-016).
- **Variable PCM sample frame lengths**: All frames within a single mip level MUST have identical length (the `mip_frame_lengths[i]` value). Variable-length frames are structurally invalid and rejected.

## Requirements *(mandatory)*

### Functional Requirements

**RIFF/WAV Format Structure**

- **FR-001**: The format MUST use standard RIFF/WAV as the container format to ensure compatibility with existing audio tools.
- **FR-002**: Wavetable waveform data MUST be stored in the standard WAV `data` chunk as concatenated single-cycle frames.
- **FR-003**: Frames MUST be stored in a logical order: all keyframes for mip level 0, then all keyframes for mip level 1, etc. (mip-major ordering).
- **FR-004**: Each frame MUST contain exactly `frame_length` samples representing one complete waveform cycle.
- **FR-005**: Audio samples MUST be stored as 32-bit floating point (IEEE 754) for maximum precision, though the format SHOULD support 16-bit and 24-bit integer for compatibility with legacy wavetables.

**Custom WTBL Metadata Chunk**

- **FR-006**: Wavetable metadata MUST be stored in a custom RIFF chunk with the FourCC identifier `WTBL`.
- **FR-007**: The WTBL chunk MUST contain a Protocol Buffers (protobuf) encoded message as its payload.
- **FR-008**: The protobuf schema MUST include a version field to enable schema evolution and backward/forward compatibility.
- **FR-009**: The protobuf message MUST include these required fields: schema_version, wavetable_type, frame_length, num_frames, num_mip_levels.
- **FR-010**: The protobuf message MUST include these metadata fields: normalization_method, source_bit_depth, author.
- **FR-011**: The protobuf message SHOULD include optional fields for: name, description, tuning_reference, generation_parameters.
- **FR-012**: Unknown protobuf fields MUST be preserved during read/write operations to support forward compatibility.

**Wavetable Type Classification**

- **FR-013**: The format MUST include a `wavetable_type` enumeration field to classify the wavetable for appropriate handling.
- **FR-014**: The `wavetable_type` enumeration MUST include at minimum:
  - `CLASSIC_DIGITAL` - PPG Wave-style wavetables (256 samples/frame, 64 frames, 8-bit source)
  - `HIGH_RESOLUTION` - Modern/vintage digital wavetables (2048+ samples/frame, high frame counts)
  - `VINTAGE_EMULATION` - Emulations of vintage oscillators (OSCar, EDP Wasp-style)
  - `PCM_SAMPLE` - Short single-cycle PCM samples (Yamaha AWM/SY99-style)
  - `CUSTOM` - User-defined or unclassified wavetables
- **FR-015**: The enumeration MUST reserve numeric space for future wavetable types without breaking compatibility.
- **FR-016**: Readers MUST gracefully handle unknown `wavetable_type` values by treating them as `CUSTOM`.

**Type-Specific Metadata**

- **FR-017**: The protobuf schema MUST support optional type-specific metadata using a oneof field or nested messages:
  - For `CLASSIC_DIGITAL`: original_bit_depth, original_sample_rate, source_hardware
  - For `HIGH_RESOLUTION`: max_harmonics, interpolation_hint
  - For `VINTAGE_EMULATION`: emulated_hardware, oscillator_type
  - For `PCM_SAMPLE`: original_sample_rate, loop_points, root_note
- **FR-018**: Type-specific metadata MUST be optional; wavetables MUST be valid without it.

**Multi-Frame and Mipmap Structure**

- **FR-019**: The format MUST support multiple keyframes per wavetable (e.g., 64 frames for PPG-style, 256+ for high-resolution).
- **FR-020**: The format MUST support multiple mip levels per wavetable for bandwidth-limited playback at different octaves.
- **FR-021**: Mip level 0 MUST be the highest resolution (longest frame length); subsequent levels MUST have progressively shorter frame lengths.
- **FR-022**: The protobuf metadata MUST specify the frame length for each mip level OR a decimation factor that can derive them.
- **FR-023**: Total sample count in the WAV data chunk MUST equal: sum of (frame_length[mip] * num_frames) for all mip levels.
- **FR-024**: For `PCM_SAMPLE` type, the format MUST support single mip level (no decimation) as the default.

**Validation Requirements**

- **FR-025**: Readers MUST validate that the WTBL chunk exists before attempting to use the file as a wavetable.
- **FR-026**: Readers MUST validate that declared sample counts in metadata match actual WAV data length exactly; any mismatch MUST be rejected with a clear error (no permissive truncation or padding).
- **FR-027**: Readers MUST validate protobuf schema version and report clear errors for incompatible versions.
- **FR-028**: Readers MUST validate that audio samples are finite (not NaN or infinity) and MUST reject files containing non-finite values with a clear error.
- **FR-029**: Readers MUST validate that `wavetable_type` is present and handle unknown values gracefully.
- **FR-030**: Readers are NOT required to enforce defensive limits on max frame counts; structural validation (RIFF format integrity, protobuf decoding) is sufficient for frame-related checks.
- **FR-030a**: Readers MUST reject wavetable files with zero keyframes or zero mip levels as structurally invalid, reporting a clear error.
- **FR-030b**: Readers MUST reject wavetable files exceeding 100MB in size with a clear error. This limit provides a sanity check while supporting all practical wavetable use cases (typical files are under 50MB).

**Tooling Requirements**

- **FR-031**: wtgen MUST be able to export wavetables in the RIFF/WAV + WTBL format for all supported wavetable types.
- **FR-032**: The dev environment MUST include a CLI tool to inspect and display protobuf metadata from wavetable files.
- **FR-033**: The inspection tool MUST decode all known protobuf fields and display them in human-readable format.
- **FR-034**: The inspection tool MUST display the wavetable type and any type-specific metadata present.
- **FR-035**: The inspection tool MUST indicate unknown fields (from newer schema versions) without failing.
- **FR-036**: The protobuf schema definition (.proto file) MUST be included in the repository and kept in sync with implementations.
- **FR-037**: The inspection tool SHOULD provide summary statistics appropriate to each wavetable type.

### Key Entities

- **Wavetable File**: A standard WAV file containing concatenated waveform frames in the data chunk plus a WTBL metadata chunk.
- **Wavetable Type**: Classification of the wavetable that indicates its origin, intended use, and appropriate playback handling (PPG-style, high-resolution, vintage emulation, PCM sample, or custom).
- **Keyframe**: A single-cycle waveform representing one "snapshot" in a morphing wavetable. Multiple keyframes enable wavetable position scanning.
- **Mip Level**: A set of keyframes at a specific sample resolution. Lower mip levels have fewer samples and fewer harmonics for alias-free playback at higher frequencies.
- **WTBL Chunk**: A custom RIFF chunk containing protobuf-encoded metadata describing the wavetable structure, type, and properties.
- **Protobuf Schema**: The Protocol Buffers definition that specifies metadata field types, wavetable type enumeration, required vs optional fields, and version compatibility rules.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Wavetables exported by wtgen pass validation 100% of the time and produce valid WAV files that open in standard audio editors.
- **SC-002**: Protobuf metadata can be decoded using standard tools (protoc --decode) without custom code.
- **SC-003**: A developer can inspect wavetable metadata using the CLI tool in under 5 seconds for files up to 100MB.
- **SC-004**: Files created with schema version N can be read by tools expecting schema version N-1 (forward compatibility) with graceful handling of unknown fields.
- **SC-005**: Files created with schema version N-1 can be read by tools expecting schema version N (backward compatibility) with sensible defaults for missing optional fields.
- **SC-006**: A third-party developer can create a valid wavetable file using only the format documentation and .proto schema within 1 hour.
- **SC-007**: The protobuf schema file and format documentation are complete enough to implement a reader without examining wtgen source code.
- **SC-008**: All five wavetable types can be round-tripped (export/import) with 100% metadata preservation.
- **SC-009**: Wavetable type is correctly identified in 100% of inspection tool output.
- **SC-010**: Unknown wavetable types from future versions are handled gracefully without parse failures.

## Clarifications

### Session 2026-01-19

- Q: What security posture should parsers adopt for untrusted wavetable files? → A: Trust file contents; validate only structural correctness (RIFF format, protobuf decoding). No defensive limits on frame counts. (Updated: 100MB file size limit added as sanity check.)
- Q: What happens when a wavetable file has zero keyframes or zero mip levels? → A: Reject as invalid with clear error; zero frames/mips is structurally invalid.
- Q: When WAV audio data length doesn't match metadata's declared frame count, what happens? → A: Reject with error; data must match metadata exactly (strict validation).

### Session 2026-01-19 (Analysis Follow-up)

- Q: How does the system handle extremely large wavetable files (hundreds of MB)? → A: Files exceeding 100MB are rejected with a clear error. This provides a sanity check while supporting all practical wavetable use cases.
- Q: How does the system handle non-standard sample rates or bit depths in the WAV header? → A: WAV header sample rate is informational only (wavetables are single-cycle, rate-agnostic). Readers accept any valid WAV sample rate. Bit depth in WAV header MUST be 32-bit float for storage; the `source_bit_depth` metadata field captures the original source resolution.
- Q: What happens when sample data contains NaN or infinity values? → A: Readers MUST reject files containing non-finite values with a clear error. Writers SHOULD NOT produce NaN/Inf values.
- Q: What happens when PCM sample frame lengths vary within the same wavetable? → A: All frames within a single mip level MUST have identical length. Variable-length frames are structurally invalid and rejected. For multi-sample PCM content with different lengths, use separate wavetable files.

## Out of Scope

- **Loading wavetables into rigel-synth runtime structures**: This specification defines the interchange format only. Loading into performance-oriented data structures (compile-time baked wavetables, runtime user-loaded wavetables) is a separate implementation effort.
- **Wavetable playback/interpolation algorithms**: How rigel-synth uses the loaded wavetable data is not part of this specification.
- **GUI wavetable editor**: Visual editing tools are out of scope; this focuses on the file format and CLI tooling.
- **Conversion from specific proprietary formats**: While the format can store converted wavetables, specific import filters for proprietary formats (e.g., Serum .wtf, Massive .nwt) are out of scope.

## Assumptions

- Protocol Buffers provides sufficient backward/forward compatibility guarantees through its field numbering and wire format.
- RIFF/WAV is well-supported by both Python (soundfile) and Rust ecosystems for reading/writing.
- Custom RIFF chunks (like WTBL) are safely ignored by standard audio tools that don't recognize them.
- 32-bit floating point precision is sufficient for all wavetable use cases.
- Mip level frame lengths follow power-of-two decimation (e.g., 2048 -> 1024 -> 512 -> 256 -> 128 -> 64 -> 32).
- The typical wavetable will have 64-256 keyframes with 7-8 mip levels, resulting in files under 50MB.
- Wavetable type classification is sufficient to guide playback behavior without requiring complex heuristics.

## Reference: Wavetable Type Examples

### CLASSIC_DIGITAL (PPG Wave-Style)

Classic 8-bit era wavetable synthesis:
- 256 samples per frame (original PPG standard)
- 64 frames (keyframes) per wavetable
- 8-bit source resolution (stored as float for precision)
- 7 mip levels for antialiased playback with harmonic caps: 128, 64, 16, 8, 4, 2 harmonics

File structure:
- Mip 0: 64 frames × 256 samples = 16,384 samples
- Mip 1-6: progressively decimated
- **Total**: ~32,512 samples × 4 bytes = ~127 KB

### HIGH_RESOLUTION (AN1x / Modern Digital Style)

High-fidelity wavetables from 90s digital synths and modern recreations:
- 2048 samples per frame (or higher)
- 64-256 frames per wavetable
- 16-bit or higher source resolution
- Full mipmap chain for alias-free playback

File structure example (64 frames):
- Mip 0: 64 frames × 2048 samples = 131,072 samples
- Mip 1-10: progressively decimated to 1 sample
- **Total**: ~256K samples × 4 bytes = ~1 MB

### VINTAGE_EMULATION (OSCar / EDP Wasp Style)

Emulations of analog-era digital oscillators:
- 256-512 samples per frame (depending on emulation target)
- Single wave or small morphing set (1-16 frames)
- Captures character of original DAC/filter response
- May include deliberate aliasing or quantization artifacts

File structure example:
- Mip 0: 8 frames × 256 samples = 2,048 samples
- Limited mip levels (preserves character)
- **Total**: ~4K samples × 4 bytes = ~16 KB

### PCM_SAMPLE (Yamaha AWM / SY99 Style)

Short single-cycle samples used as oscillator sources:
- Variable frame length (64-4096 samples typical)
- Single frame (no morphing) or attack/sustain pairs
- Original sample rate and root note metadata important
- Usually single mip level (no decimation, uses different anti-aliasing)

File structure example:
- Mip 0: 1 frame × 2048 samples = 2,048 samples
- **Total**: 2,048 samples × 4 bytes = ~8 KB

### CUSTOM

User-defined wavetables that don't fit other categories:
- Any valid combination of frame length, frame count, and mip levels
- No assumptions about source or intended use
- Most flexible but least optimized
