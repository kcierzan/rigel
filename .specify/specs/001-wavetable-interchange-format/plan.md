# Implementation Plan: Wavetable Interchange Format

**Branch**: `001-wavetable-interchange-format` | **Date**: 2026-01-19 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `.specify/specs/001-wavetable-interchange-format/spec.md`
**Linear Issue**: [NEW-7](https://linear.app/new-atlantis/issue/NEW-7/standardize-on-wavetable-interchange-format)

## Summary

Create a standardized wavetable interchange format between wtgen (Python generation toolkit) and rigel-synth (Rust synthesizer) using RIFF/WAV container format with a custom `WTBL` chunk containing Protocol Buffers metadata. This enables wavetable generation in Python with reliable consumption in the Rust audio plugin, supporting multiple wavetable types (PPG-style, high-resolution, vintage emulation, PCM samples).

## Technical Context

**Language/Version**:
- Python 3.12+ (wtgen, existing devenv)
- Rust 1.75+ (rigel-synth, existing rust-toolchain.toml)

**Primary Dependencies**:
- Python: numpy, scipy, soundfile (existing); protobuf (to add)
- Rust: hound 3.5 (existing); prost, riff (to add)

**Storage**: Files (RIFF/WAV format with custom WTBL metadata chunk)

**Testing**:
- Python: pytest + hypothesis (existing)
- Rust: cargo test + SIMD-specific tests (existing)

**Target Platforms**:
- macOS (Apple Silicon), Linux (x86_64), Windows (x86_64) - all existing

**Project Type**: Monorepo with multiple projects (rigel-synth Rust workspace, wtgen Python package)

**Performance Goals**:
- Wavetable file loading in CLI: <5 seconds for 100MB files
- No impact on DSP core performance (file I/O stays out of no_std code)

**Constraints**:
- rigel-dsp MUST remain no_std, allocation-free (file I/O in CLI/plugin only)
- Protobuf schema must support forward/backward compatibility
- Format must be readable by third-party tools (standard WAV with ignored custom chunk)

**Scale/Scope**:
- Typical file sizes: 16KB-1MB (up to 100MB max; files >100MB rejected per FR-030b)
- 5 wavetable types to support
- 2 languages requiring protobuf bindings

**Validation Requirements** (spec update 2026-01-19):
- FR-028: NaN/Infinity sample values MUST be rejected with clear error
- FR-030b: Files exceeding 100MB MUST be rejected with clear error

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Principle I: Real-Time Safety ✅ PASS
- **Requirement**: rigel-dsp MUST remain no_std, allocation-free
- **Assessment**: All file I/O and protobuf parsing will be in CLI/plugin layers only. The DSP core will NOT include wavetable loading code. Loaded wavetables will be passed to DSP as pre-allocated references.

### Principle II: Layered Architecture ✅ PASS
- **Requirement**: DSP core → CLI → Plugin separation
- **Assessment**: Wavetable I/O will be implemented in:
  - New `wavetable-io` crate (can allocate, not no_std)
  - CLI layer for inspection tool
  - wtgen for Python export
  - DSP core remains untouched for file I/O

### Principle III: Test-Driven Validation ✅ PASS
- **Requirement**: Automated tests + audible verification
- **Assessment**:
  - Python: pytest + hypothesis for property-based testing of format
  - Rust: cargo test for reader validation
  - Round-trip tests (generate → export → load → validate)
  - Audible verification for loaded wavetables

### Principle IV: Performance Accountability ✅ PASS
- **Requirement**: No DSP performance regressions
- **Assessment**: File I/O is not on the DSP hot path. Wavetables loaded once at plugin init or user action, not during real-time processing.

### Principle V: Reproducible Environments ✅ PASS
- **Requirement**: All development in Nix/devenv shells
- **Assessment**: Existing devenv shells will be extended with:
  - `protoc` compiler for Python shell
  - `prost-build` will handle Rust codegen automatically

### Principle VI: Cross-Platform Commitment ✅ PASS
- **Requirement**: macOS, Linux, Windows support
- **Assessment**:
  - Protobuf is cross-platform
  - RIFF/WAV is platform-agnostic
  - No platform-specific file handling

### Principle VII: DSP Correctness Properties ⚠️ N/A
- **Requirement**: Wavetable DSP invariants
- **Assessment**: This feature defines the interchange format only. DSP playback is explicitly out of scope per spec. Future loading into DSP structures must respect zero-crossing alignment, RMS consistency, etc.

## Project Structure

### Documentation (this feature)

```text
.specify/specs/001-wavetable-interchange-format/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
│   └── wavetable.proto  # Protocol Buffers schema definition
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
# Shared Schema
proto/
└── wavetable.proto      # Canonical protobuf schema (symlinked or copied to contracts/)

# Rust (rigel-synth)
projects/rigel-synth/crates/
├── wavetable-io/        # NEW: Wavetable file I/O crate (NOT no_std)
│   ├── Cargo.toml
│   ├── build.rs         # prost-build for protobuf codegen
│   ├── src/
│   │   ├── lib.rs
│   │   ├── reader.rs    # WAV + WTBL chunk reading
│   │   ├── writer.rs    # WAV + WTBL chunk writing (for tests)
│   │   ├── validation.rs
│   │   └── types.rs     # Rust types for wavetable metadata
│   └── tests/
├── cli/
│   └── src/
│       └── commands/
│           └── wavetable.rs  # inspect subcommand
└── dsp/                 # UNCHANGED - no file I/O here

# Python (wtgen)
projects/wtgen/
├── proto/
│   └── wavetable.proto  # Symlink to shared schema
├── src/wtgen/
│   └── format/          # NEW: Wavetable interchange format module
│       ├── __init__.py
│       ├── proto/       # Generated protobuf bindings
│       │   └── wavetable_pb2.py
│       ├── riff.py      # RIFF/WAV chunk handling
│       ├── writer.py    # Export wavetable as WAV+WTBL
│       ├── reader.py    # Read wavetable from WAV+WTBL
│       └── validation.py
└── tests/
    ├── unit/format/
    └── integration/
```

**Structure Decision**: Multi-project monorepo structure with shared protobuf schema. The wavetable-io crate is isolated from the no_std DSP core, allowing allocations for file I/O. The Python format module extends wtgen with new export capabilities.

## Complexity Tracking

> **No constitution violations identified. All principles pass.**

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| New crate (wavetable-io) | Separate from dsp | Maintains no_std isolation per Principle I |
| Protocol Buffers | Chosen over JSON/MessagePack | Industry standard, excellent forward/backward compat, typed schema |
| RIFF/WAV container | Chosen over custom format | Standard audio format, graceful degradation in standard tools |

---

## Constitution Check - Post-Design Re-evaluation

*Evaluated after Phase 1 design completion.*

### Design Verification

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Real-Time Safety | ✅ PASS | wavetable-io crate is NOT no_std; DSP core untouched |
| II. Layered Architecture | ✅ PASS | Clear separation: wavetable-io → CLI → (future: DSP consumer) |
| III. Test-Driven Validation | ✅ PASS | Property-based tests (Hypothesis), round-trip validation planned |
| IV. Performance Accountability | ✅ PASS | File I/O not on audio path; loaded at init time only |
| V. Reproducible Environments | ✅ PASS | protoc added to devenv; prost-build handles Rust codegen |
| VI. Cross-Platform | ✅ PASS | All dependencies (prost, protobuf, riff) are cross-platform |
| VII. DSP Correctness | ⚠️ N/A | Format only; DSP consumption is future scope |

### New Dependencies Compatibility

| Dependency | Crate | no_std | Allocation-free | Platform |
|------------|-------|--------|-----------------|----------|
| prost | wavetable-io | No (requires alloc) | No | All |
| riff | wavetable-io | No | No | All |
| protobuf (Python) | wtgen | N/A | N/A | All |

All new dependencies are correctly scoped to non-DSP crates. No constitutional violations.
