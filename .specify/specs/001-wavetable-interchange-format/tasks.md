# Tasks: Wavetable Interchange Format

**Input**: Design documents from `.specify/specs/001-wavetable-interchange-format/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/wavetable.proto

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3, US4)
- Include exact file paths in descriptions

## Path Conventions

This is a monorepo with:
- **Rust**: `projects/rigel-synth/crates/` (wavetable-io, cli)
- **Python**: `projects/wtgen/` (src/wtgen/, tests/)
- **Shared**: `proto/` at repository root

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization, protobuf schema, and devenv configuration

- [X] T001 Create shared protobuf schema at proto/wavetable.proto (copy from contracts/wavetable.proto)
- [X] T002 [P] Add protobuf package to root devenv.nix
- [X] T003 [P] Add protobuf package to projects/wtgen/devenv.nix
- [X] T004 Create wavetable-io crate directory structure at projects/rigel-synth/crates/wavetable-io/
- [X] T005 Create Cargo.toml for wavetable-io with prost, bytes, riff, anyhow dependencies in projects/rigel-synth/crates/wavetable-io/Cargo.toml
- [X] T006 Add wavetable-io to workspace members in Cargo.toml (root)
- [X] T007 Create build.rs for prost-build protobuf codegen in projects/rigel-synth/crates/wavetable-io/build.rs
- [X] T008 Add protobuf dependency to wtgen in projects/wtgen/pyproject.toml

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T009 Create lib.rs with module structure and protobuf includes in projects/rigel-synth/crates/wavetable-io/src/lib.rs
- [X] T010 [P] Create types.rs with Rust wavetable types (WavetableFile, MipLevel) in projects/rigel-synth/crates/wavetable-io/src/types.rs
- [X] T011 [P] Create Python format module __init__.py in projects/wtgen/src/wtgen/format/__init__.py
- [X] T012 [P] Create Python proto directory and generate wavetable_pb2.py in projects/wtgen/src/wtgen/format/proto/
- [X] T013 Implement RIFF chunk reading utilities in projects/rigel-synth/crates/wavetable-io/src/riff.rs
- [X] T014 Implement Python RIFF chunk utilities in projects/wtgen/src/wtgen/format/riff.py

**Checkpoint**: Foundation ready - user story implementation can now begin

---

## Phase 3: User Story 1 - Generate Wavetables in wtgen for rigel (Priority: P1) 🎯 MVP

**Goal**: Enable sound designers to generate wavetables in wtgen and export them as WAV files with embedded protobuf metadata that rigel can consume

**Independent Test**: Generate a wavetable in wtgen, export as WAV, verify file contains valid audio data plus parseable protobuf metadata with correct wavetable type

### Implementation for User Story 1

- [X] T015 [P] [US1] Create validation.py with metadata validation functions in projects/wtgen/src/wtgen/format/validation.py
- [X] T016 [P] [US1] Create WavetableType enum wrapper in projects/wtgen/src/wtgen/format/types.py
- [X] T017 [US1] Implement writer.py with save_wavetable_wav() function in projects/wtgen/src/wtgen/format/writer.py
- [X] T018 [US1] Implement reader.py with load_wavetable_wav() function in projects/wtgen/src/wtgen/format/reader.py
- [X] T019 [US1] Add type-specific metadata classes (ClassicDigitalMetadata, etc.) in projects/wtgen/src/wtgen/format/types.py
- [X] T020 [US1] Implement validation.rs with metadata validation functions in projects/rigel-synth/crates/wavetable-io/src/validation.rs
- [X] T021 [US1] Implement reader.rs with read_wavetable() function in projects/rigel-synth/crates/wavetable-io/src/reader.rs
- [X] T022 [US1] Implement writer.rs with write_wavetable() function (for round-trip tests) in projects/rigel-synth/crates/wavetable-io/src/writer.rs
- [X] T023 [US1] Add unit tests for Python writer in projects/wtgen/tests/unit/format/test_writer.py
- [X] T024 [US1] Add unit tests for Python reader in projects/wtgen/tests/unit/format/test_reader.py
- [X] T025 [US1] Add unit tests for Rust reader in projects/rigel-synth/crates/wavetable-io/tests/reader_tests.rs
- [X] T026 [US1] Add round-trip integration test (Python write -> Rust read) including unknown field preservation verification per FR-012 in projects/rigel-synth/crates/wavetable-io/tests/roundtrip_tests.rs
- [X] T027 [US1] Export format module functions in projects/wtgen/src/wtgen/__init__.py

**Checkpoint**: User Story 1 complete - wtgen can export wavetables, Rust can read them

---

## Phase 4: User Story 2 - Inspect Wavetable Metadata (Priority: P2)

**Goal**: Provide a CLI tool for developers to inspect protobuf metadata in wavetable files for debugging and verification

**Independent Test**: Run inspection command on a wavetable file and verify all protobuf fields are correctly decoded and displayed

### Implementation for User Story 2

- [X] T028 [US2] Add wavetable-io dependency to CLI crate in projects/rigel-synth/crates/cli/Cargo.toml
- [X] T029 [US2] Create wavetable.rs command module with inspect subcommand in projects/rigel-synth/crates/cli/src/commands/wavetable.rs
- [X] T030 [US2] Implement metadata display formatting (human-readable output) in projects/rigel-synth/crates/cli/src/commands/wavetable.rs
- [X] T031 [US2] Implement verbose output mode with type-specific metadata and summary statistics (total samples, file size, peak/RMS per FR-037) in projects/rigel-synth/crates/cli/src/commands/wavetable.rs
- [X] T032 [US2] Add unknown field handling display (forward compatibility) in projects/rigel-synth/crates/cli/src/commands/wavetable.rs
- [X] T033 [US2] Register wavetable command in CLI main module in projects/rigel-synth/crates/cli/src/main.rs
- [X] T034 [US2] Add CLI integration tests for inspect command in projects/rigel-synth/crates/cli/tests/wavetable_cmd_tests.rs

**Checkpoint**: User Story 2 complete - CLI can inspect any wavetable file

---

## Phase 5: User Story 3 - Third-Party Wavetable Creation (Priority: P3)

**Goal**: Enable third-party developers to create compatible wavetable files using only the format documentation and protobuf schema

**Independent Test**: Create a wavetable file manually following the specification and verify it passes validation

### Implementation for User Story 3

- [X] T035 [US3] Create format specification document in docs/wavetable-format.md
- [X] T036 [US3] Add validation CLI subcommand (rigel wavetable validate <file>) in projects/rigel-synth/crates/cli/src/commands/wavetable.rs
- [X] T037 [US3] Add detailed validation error messages for third-party debugging in projects/rigel-synth/crates/wavetable-io/src/validation.rs
- [X] T038 [US3] Create example code for third-party Python implementation in docs/examples/third_party_python.py
- [X] T039 [US3] Add validation test with manually-crafted wavetable file in projects/rigel-synth/crates/wavetable-io/tests/third_party_tests.rs

**Checkpoint**: User Story 3 complete - third parties can create compatible files

---

## Phase 6: User Story 4 - Convert Legacy Wavetable Formats (Priority: P4)

**Goal**: Enable sound designers to import legacy wavetable formats and convert them to the standardized format with appropriate type metadata

**Independent Test**: Import a legacy format file and verify the conversion preserves audio fidelity and assigns appropriate type metadata

### Implementation for User Story 4

- [X] T040 [US4] Create legacy module structure in projects/wtgen/src/wtgen/format/legacy/__init__.py
- [X] T041 [P] [US4] Implement raw PCM import (raw_import.py) in projects/wtgen/src/wtgen/format/legacy/raw_import.py
- [X] T042 [P] [US4] Implement high-res wavetable import (hires_import.py) in projects/wtgen/src/wtgen/format/legacy/hires_import.py
- [X] T043 [US4] Add wavetable type inference/detection utilities in projects/wtgen/src/wtgen/format/legacy/type_inference.py
- [X] T044 [US4] Add unit tests for legacy imports in projects/wtgen/tests/unit/format/test_legacy_import.py
- [X] T045 [US4] Add integration tests for legacy -> standard format conversion in projects/wtgen/tests/integration/test_legacy_conversion.py

**Checkpoint**: User Story 4 complete - legacy wavetables can be converted

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [X] T046 [P] Add property-based tests (Hypothesis) for Python format module in projects/wtgen/tests/unit/format/test_properties.py
- [X] T047 Run full Python quality checks (ruff, mypy, basedpyright) for format module
- [X] T048 Run full Rust quality checks (cargo fmt, clippy) for wavetable-io
- [X] T049 [P] Add performance benchmark for wavetable loading (target: <5s for 100MB per SC-003) in projects/rigel-synth/crates/wavetable-io/benches/loading_bench.rs
- [X] T050 Validate quickstart.md scenarios work end-to-end
- [X] T051 Update CLAUDE.md with new Active Technologies entry for wavetable format

---

## Phase 8: Spec Update - Enhanced Validation (FR-028, FR-030b)

**Purpose**: Implement validation requirements added after initial task generation

**Context**: spec.md was updated on 2026-01-19 to strengthen validation:
- FR-028: NaN/Infinity sample values MUST be rejected (previously SHOULD warn)
- FR-030b: Files exceeding 100MB MUST be rejected (new requirement)

### NaN/Infinity Sample Validation (FR-028)

- [X] T052 [P] [US1] Add NaN/Infinity sample validation to Python reader in projects/wtgen/src/wtgen/format/reader.py - reject files containing non-finite sample values with clear error
- [X] T053 [P] [US1] Add NaN/Infinity sample validation to Rust reader in projects/rigel-synth/crates/wavetable-io/src/reader.rs - reject files containing non-finite sample values with clear error
- [X] T054 [P] [US1] Add unit tests for NaN/Infinity rejection in Python in projects/wtgen/tests/unit/format/test_validation.py
- [X] T055 [P] [US1] Add unit tests for NaN/Infinity rejection in Rust in projects/rigel-synth/crates/wavetable-io/tests/validation_tests.rs

### File Size Limit Validation (FR-030b)

- [X] T056 [P] [US1] Add 100MB file size limit validation to Python reader in projects/wtgen/src/wtgen/format/reader.py - reject files exceeding 100MB with clear error
- [X] T057 [P] [US1] Add 100MB file size limit validation to Rust reader in projects/rigel-synth/crates/wavetable-io/src/reader.rs - reject files exceeding 100MB with clear error
- [X] T058 [P] [US1] Add unit tests for 100MB file size limit in Python in projects/wtgen/tests/unit/format/test_validation.py
- [X] T059 [P] [US1] Add unit tests for 100MB file size limit in Rust in projects/rigel-synth/crates/wavetable-io/tests/validation_tests.rs

### Documentation Update

- [X] T060 [US3] Update docs/wavetable-format.md with FR-028 and FR-030b validation requirements

**Checkpoint**: Enhanced validation complete - all files validated for sample integrity and size limits

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-6)**: All depend on Foundational phase completion
  - User stories can proceed in parallel if staffed
  - Or sequentially in priority order (P1 → P2 → P3 → P4)
- **Polish (Phase 7)**: Depends on desired user stories being complete
- **Enhanced Validation (Phase 8)**: Can start after Phase 2; all tasks are parallel within phase

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Depends on US1 (needs wavetable-io reader functionality)
- **User Story 3 (P3)**: Can start after Foundational - May benefit from US2 validation command
- **User Story 4 (P4)**: Depends on US1 (needs writer functionality)

### Within Each User Story

- Python components before Rust (for Python-first round-trip testing)
- Validation before reader/writer
- Core implementation before tests
- Unit tests before integration tests

### Parallel Opportunities

- T002, T003 (devenv configs) can run in parallel
- T010, T011, T012 (initial module structure) can run in parallel
- T015, T016 (Python validation/types) can run in parallel
- T041, T042 (legacy importers) can run in parallel
- T046, T049 (property-based tests, performance benchmark) can run in parallel
- T052, T053, T056, T057 (enhanced validation implementation) can run in parallel
- T054, T055, T058, T059 (enhanced validation tests) can run in parallel after implementation

---

## Parallel Example: User Story 1 Foundation

```bash
# Launch initial Python format module components together:
Task: "Create validation.py" in projects/wtgen/src/wtgen/format/validation.py
Task: "Create WavetableType enum wrapper" in projects/wtgen/src/wtgen/format/types.py

# After dependencies resolve, launch tests in parallel:
Task: "Unit tests for Python writer" in projects/wtgen/tests/unit/format/test_writer.py
Task: "Unit tests for Python reader" in projects/wtgen/tests/unit/format/test_reader.py
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Test Python export → Rust read round-trip
5. Deploy/integrate into rigel-synth workflow

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test round-trip → MVP complete!
3. Add User Story 2 → CLI inspection tool available
4. Add User Story 3 → Third-party documentation complete
5. Add User Story 4 → Legacy format support

### Suggested MVP Scope

**User Story 1 alone provides:**
- wtgen can export wavetables with all 5 types
- Rust can read wavetable files with full validation
- Round-trip testing ensures format correctness
- All acceptance scenarios for US1 are testable

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify tests pass after each task group
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- rigel-dsp remains untouched - all file I/O in wavetable-io crate only
