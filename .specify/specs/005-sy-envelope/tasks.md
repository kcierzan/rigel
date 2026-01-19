# Tasks: SY-Style Envelope Modulation Source

**Input**: Design documents from `.specify/specs/005-sy-envelope/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/envelope-api.rs

**Tests**: Included per spec requirement ("fully tested and benchmarked")

**Organization**: Tasks are grouped by user story to enable independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Project root**: `projects/rigel-synth/crates/modulation/`
- **Source**: `src/envelope/`
- **Tests**: `tests/envelope_tests.rs`
- **Benchmarks**: `benches/envelope_bench.rs`

---

## Phase 1: Setup (Module Infrastructure)

**Purpose**: Create envelope module structure within rigel-modulation crate

- [x] T001 Create envelope module directory structure at `projects/rigel-synth/crates/modulation/src/envelope/`
- [x] T002 Create module root with re-exports in `projects/rigel-synth/crates/modulation/src/envelope/mod.rs`
- [x] T003 [P] Add envelope module to crate root in `projects/rigel-synth/crates/modulation/src/lib.rs`
- [x] T004 [P] Create test file in `projects/rigel-synth/crates/modulation/tests/envelope_tests.rs`
- [x] T005 [P] Create benchmark file in `projects/rigel-synth/crates/modulation/benches/envelope_bench.rs`

---

## Phase 2: Foundational (Core Types & Constants)

**Purpose**: Core types and constants that ALL user stories depend on

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T006 Implement EnvelopePhase enum in `projects/rigel-synth/crates/modulation/src/envelope/state.rs`
- [x] T007 [P] Implement Segment struct in `projects/rigel-synth/crates/modulation/src/envelope/segment.rs`
- [x] T008 [P] Implement LoopConfig struct in `projects/rigel-synth/crates/modulation/src/envelope/config.rs`
- [x] T009 Implement EnvelopeConfig with const generics in `projects/rigel-synth/crates/modulation/src/envelope/config.rs`
- [x] T010 Implement EnvelopeLevel type (i16/Q8) and constants in `projects/rigel-synth/crates/modulation/src/envelope/state.rs`
- [x] T011 [P] Implement LEVEL_LUT lookup table in `projects/rigel-synth/crates/modulation/src/envelope/rates.rs`
- [x] T012 [P] Implement STATICS timing table in `projects/rigel-synth/crates/modulation/src/envelope/rates.rs`
- [x] T013 Implement EnvelopeState struct with i16 fields in `projects/rigel-synth/crates/modulation/src/envelope/state.rs`
- [x] T014 Implement type aliases (FmEnvelope, AwmEnvelope, SevenSegEnvelope) in `projects/rigel-synth/crates/modulation/src/envelope/mod.rs`

**Checkpoint**: Foundation ready - core types available for all user stories

---

## Phase 3: User Story 1 - Basic Envelope Modulation (Priority: P1) 🎯 MVP

**Goal**: Basic amplitude control that responds to note-on/note-off events with attack and release transitions

**Independent Test**: Trigger a single note and measure envelope output against expected values at key time points

### Tests for User Story 1

- [x] T015 [P] [US1] Unit test for note_on triggers attack in `tests/envelope_tests.rs`
- [x] T016 [P] [US1] Unit test for note_off triggers release in `tests/envelope_tests.rs`
- [x] T017 [P] [US1] Unit test for output range 0.0-1.0 in `tests/envelope_tests.rs`
- [x] T018 [P] [US1] Unit test for segment transitions in `tests/envelope_tests.rs`

### Implementation for User Story 1

- [x] T019 [US1] Implement Envelope struct with config and state in `projects/rigel-synth/crates/modulation/src/envelope/mod.rs`
- [x] T020 [US1] Implement Envelope::new() and with_config() constructors in `projects/rigel-synth/crates/modulation/src/envelope/mod.rs`
- [x] T021 [US1] Implement note_on() method (start attack sequence) in `projects/rigel-synth/crates/modulation/src/envelope/mod.rs`
- [x] T022 [US1] Implement note_off() method (transition to release) in `projects/rigel-synth/crates/modulation/src/envelope/mod.rs`
- [x] T023 [US1] Implement level_to_linear() conversion using rigel_math::fast_exp2 in `projects/rigel-synth/crates/modulation/src/envelope/rates.rs`
- [x] T024 [US1] Implement process() method (advance one sample, return linear amplitude) in `projects/rigel-synth/crates/modulation/src/envelope/mod.rs`
- [x] T025 [US1] Implement value() method (get current level without advancing) in `projects/rigel-synth/crates/modulation/src/envelope/mod.rs`
- [x] T026 [US1] Implement advance_segment() helper for segment transitions in `projects/rigel-synth/crates/modulation/src/envelope/mod.rs`
- [x] T027 [US1] Implement reset() method in `projects/rigel-synth/crates/modulation/src/envelope/mod.rs`

**Checkpoint**: Basic envelope responds to note events with attack/release behavior

---

## Phase 4: User Story 2 - Rate Scaling by Key Position (Priority: P1)

**Goal**: Higher notes have faster envelopes based on MIDI note number

**Independent Test**: Trigger same patch at different MIDI notes and verify timing scales appropriately

### Tests for User Story 2

- [x] T028 [P] [US2] Unit test for rate scaling at MIDI 60 (baseline) in `tests/envelope_tests.rs`
- [x] T029 [P] [US2] Unit test for rate scaling at MIDI 84 (faster) in `tests/envelope_tests.rs`
- [x] T030 [P] [US2] Unit test for rate scaling disabled (sensitivity=0) in `tests/envelope_tests.rs`

### Implementation for User Story 2

- [x] T031 [US2] Implement rate_to_qrate() formula in `projects/rigel-synth/crates/modulation/src/envelope/rates.rs`
- [x] T032 [US2] Implement scale_rate() for MIDI note adjustment in `projects/rigel-synth/crates/modulation/src/envelope/rates.rs`
- [x] T033 [US2] Integrate rate scaling into note_on() with midi_note parameter in `projects/rigel-synth/crates/modulation/src/envelope/mod.rs`
- [x] T034 [US2] Cache midi_note in EnvelopeState for segment transitions in `projects/rigel-synth/crates/modulation/src/envelope/state.rs`

**Checkpoint**: Envelope timing varies naturally across keyboard range

---

## Phase 5: User Story 3 - MSFA-Compatible Rate Behavior (Priority: P1)

**Goal**: Rate calculations match original DX7/MSFA behavior including distance-dependent timing

**Independent Test**: Compare envelope curves against MSFA/Dexed reference output for identical parameters

### Tests for User Story 3

- [x] T035 [P] [US3] Unit test for MSFA rate formula (rate * 41 >> 6) in `tests/envelope_tests.rs`
- [x] T036 [P] [US3] Unit test for distance-dependent timing in `tests/envelope_tests.rs`
- [x] T037 [P] [US3] Unit test for rate 99 near-instantaneous transition in `tests/envelope_tests.rs`
- [x] T038 [P] [US3] Tolerance test against MSFA reference (0.1dB) in `tests/envelope_tests.rs`

### Implementation for User Story 3

- [x] T039 [US3] Implement calculate_increment_q8() from qRate in `projects/rigel-synth/crates/modulation/src/envelope/rates.rs`
- [x] T040 [US3] Implement scale_output_level() with LEVEL_LUT in `projects/rigel-synth/crates/modulation/src/envelope/rates.rs`
- [x] T041 [US3] Implement decay behavior (linear decrease in log domain) in `projects/rigel-synth/crates/modulation/src/envelope/mod.rs`
- [x] T042 [US3] Implement segment target level calculation in `projects/rigel-synth/crates/modulation/src/envelope/mod.rs`
- [x] T043 [US3] Implement same-level transition timing using STATICS table in `projects/rigel-synth/crates/modulation/src/envelope/rates.rs`

**Checkpoint**: Envelope curves match MSFA reference within tolerance

---

## Phase 6: User Story 7 - High-Performance Block Processing (Priority: P1)

**Goal**: Process 1536 envelopes x 64 samples in under 100µs with zero allocations

**Independent Test**: Benchmark envelope processing throughput against performance targets

### Tests for User Story 7

- [x] T044 [P] [US7] Unit test for zero allocations during processing in `tests/envelope_tests.rs`
- [x] T045 [P] [US7] Unit test for Copy/Clone trait bounds in `tests/envelope_tests.rs`

### Benchmarks for User Story 7

- [x] T046 [P] [US7] Criterion benchmark for single envelope processing in `benches/envelope_bench.rs`
- [x] T047 [P] [US7] Criterion benchmark for 1536 envelopes x 64 samples in `benches/envelope_bench.rs`
- [x] T048 [P] [US7] iai-callgrind benchmark for instruction count regression in `benches/envelope_bench.rs`

### Implementation for User Story 7

- [x] T049 [US7] Implement process_block() method for batch sample processing in `projects/rigel-synth/crates/modulation/src/envelope/mod.rs`
- [x] T050 [US7] Create EnvelopeBatch struct in `projects/rigel-synth/crates/modulation/src/envelope/batch.rs`
- [x] T051 [US7] Implement EnvelopeBatch::new() and with_config() in `projects/rigel-synth/crates/modulation/src/envelope/batch.rs`
- [x] T052 [US7] Implement EnvelopeBatch::process() with SIMD acceleration in `projects/rigel-synth/crates/modulation/src/envelope/batch.rs`
- [x] T053 [US7] Implement levels_to_linear_simd() using rigel_math in `projects/rigel-synth/crates/modulation/src/envelope/batch.rs`
- [x] T054 [US7] Implement EnvelopeBatch::process_block() in `projects/rigel-synth/crates/modulation/src/envelope/batch.rs`
- [x] T055 [US7] Verify #[derive(Copy, Clone)] on Envelope and EnvelopeState in `projects/rigel-synth/crates/modulation/src/envelope/mod.rs`

**Checkpoint**: Performance targets met, zero allocations verified

---

## Phase 7: User Story 4 - Instantaneous Attack dB Jump (Priority: P2)

**Goal**: Punchy attack transients with immediate dB level jump at attack start

**Independent Test**: Examine first samples after note-on and verify immediate level jump occurs

### Tests for User Story 4

- [x] T056 [P] [US4] Unit test for immediate jump to JUMP_TARGET_Q8 on attack in `tests/envelope_tests.rs`
- [x] T057 [P] [US4] Unit test for smooth approach after jump in `tests/envelope_tests.rs`

### Implementation for User Story 4

- [x] T058 [US4] Add JUMP_TARGET_Q8 constant (1716) in `projects/rigel-synth/crates/modulation/src/envelope/rates.rs`
- [x] T059 [US4] Implement attack_sample() with instant jump behavior in `projects/rigel-synth/crates/modulation/src/envelope/mod.rs`
- [x] T060 [US4] Implement exponential rise toward target after jump in `projects/rigel-synth/crates/modulation/src/envelope/mod.rs`

**Checkpoint**: Attack has characteristic FM "punch"

---

## Phase 8: User Story 5 - Delayed Envelope Start (Priority: P2)

**Goal**: Configurable delay before attack begins for evolving pads

**Independent Test**: Configure delay and verify envelope remains at zero for specified duration

### Tests for User Story 5

- [x] T061 [P] [US5] Unit test for delay countdown in `tests/envelope_tests.rs`
- [x] T062 [P] [US5] Unit test for note_off during delay aborts to release in `tests/envelope_tests.rs`

### Implementation for User Story 5

- [x] T063 [US5] Add delay_samples field to EnvelopeConfig in `projects/rigel-synth/crates/modulation/src/envelope/config.rs`
- [x] T064 [US5] Add delay_remaining field to EnvelopeState in `projects/rigel-synth/crates/modulation/src/envelope/state.rs`
- [x] T065 [US5] Implement Delay phase handling in process() in `projects/rigel-synth/crates/modulation/src/envelope/mod.rs`
- [x] T066 [US5] Handle note_off during Delay phase in `projects/rigel-synth/crates/modulation/src/envelope/mod.rs`

**Checkpoint**: Delayed envelope start works for evolving sounds

---

## Phase 9: User Story 6 - Looping Between Segments (Priority: P2)

**Goal**: Loop between configurable segment boundaries for rhythmic textures

**Independent Test**: Enable looping and verify envelope cycles between segments until note-off

### Tests for User Story 6

- [x] T067 [P] [US6] Unit test for loop from segment 3 to 5 in `tests/envelope_tests.rs`
- [x] T068 [P] [US6] Unit test for note_off exits loop in `tests/envelope_tests.rs`
- [x] T069 [P] [US6] Unit test for invalid loop boundaries fallback in `tests/envelope_tests.rs`

### Implementation for User Story 6

- [x] T070 [US6] Add LoopConfig validation methods in `projects/rigel-synth/crates/modulation/src/envelope/config.rs`
- [x] T071 [US6] Implement loop detection in advance_segment() in `projects/rigel-synth/crates/modulation/src/envelope/mod.rs`
- [x] T072 [US6] Implement loop exit on note_off in `projects/rigel-synth/crates/modulation/src/envelope/mod.rs`

**Checkpoint**: Looping envelopes create rhythmic/evolving textures

---

## Phase 10: User Story 8 - Variant Envelope Configurations (Priority: P3)

**Goal**: Different envelope variants (8-segment FM, 7-segment, 5+5 AWM) via const generics

**Independent Test**: Instantiate different variants and verify correct segment counts

### Tests for User Story 8

- [x] T073 [P] [US8] Unit test for FmEnvelope (6+2 segments) in `tests/envelope_tests.rs`
- [x] T074 [P] [US8] Unit test for SevenSegEnvelope (5+2 segments) in `tests/envelope_tests.rs`
- [x] T075 [P] [US8] Unit test for AwmEnvelope (5+5 segments) in `tests/envelope_tests.rs`

### Implementation for User Story 8

- [x] T076 [US8] Verify const generics work for all variants in `projects/rigel-synth/crates/modulation/src/envelope/mod.rs`
- [x] T077 [US8] Add FmEnvelopeConfig, AwmEnvelopeConfig, SevenSegEnvelopeConfig type aliases in `projects/rigel-synth/crates/modulation/src/envelope/config.rs`
- [x] T078 [US8] Add variant-specific default configurations in `projects/rigel-synth/crates/modulation/src/envelope/config.rs`

**Checkpoint**: All envelope variants work correctly

---

## Phase 11: Polish & Integration

**Purpose**: Final integration, documentation, and cross-cutting concerns

- [x] T079 Implement ModulationSource trait for Envelope in `projects/rigel-synth/crates/modulation/src/envelope/mod.rs`
- [x] T080 [P] Add inline documentation with examples in all public items
- [x] T081 [P] Verify no_std compatibility in `projects/rigel-synth/crates/modulation/Cargo.toml`
- [x] T082 Run full benchmark suite and document results in `.specify/specs/005-sy-envelope/benchmark-results.md`
- [x] T083 Validate quickstart.md examples compile and run
- [x] T084 Run `cargo:test` and `cargo:lint` for final validation

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup - BLOCKS all user stories
- **User Stories (Phases 3-10)**: All depend on Foundational completion
  - P1 stories (US1, US2, US3, US7) can proceed in parallel after foundational
  - P2 stories (US4, US5, US6) can proceed after their P1 dependencies
  - P3 stories (US8) can proceed after core implementation
- **Polish (Phase 11)**: Depends on all user stories being complete

### User Story Dependencies

| Story | Priority | Dependencies | Can Start After |
|-------|----------|--------------|-----------------|
| US1 - Basic Envelope | P1 | Foundational | Phase 2 |
| US2 - Rate Scaling | P1 | US1 (needs note_on) | T021 |
| US3 - MSFA Rates | P1 | US1 (needs process) | T024 |
| US7 - Performance | P1 | US1 (needs Envelope) | T019 |
| US4 - Attack Jump | P2 | US1 (needs attack) | T021 |
| US5 - Delay | P2 | US1 (needs phases) | T019 |
| US6 - Looping | P2 | US1 (needs segments) | T026 |
| US8 - Variants | P3 | US1 (needs Envelope) | T019 |

### Parallel Opportunities

**Within Phase 2 (Foundational):**
- T007, T008 (Segment, LoopConfig) in parallel
- T011, T012 (lookup tables) in parallel

**Within Each User Story:**
- All test tasks marked [P] can run in parallel
- Tests should be written before implementation

**Across User Stories (after Phase 2):**
- US1 must complete first (core dependency)
- US2, US3, US7 can proceed in parallel after US1 basics
- US4, US5, US6 can proceed in parallel after US1 complete
- US8 can proceed after US1 complete

---

## Parallel Example: User Story 1

```bash
# Launch all tests for User Story 1 together:
Task: "Unit test for note_on triggers attack in tests/envelope_tests.rs"
Task: "Unit test for note_off triggers release in tests/envelope_tests.rs"
Task: "Unit test for output range 0.0-1.0 in tests/envelope_tests.rs"
Task: "Unit test for segment transitions in tests/envelope_tests.rs"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (core types)
3. Complete Phase 3: User Story 1 (basic envelope)
4. **STOP and VALIDATE**: Test envelope responds to note events
5. Demo basic amplitude modulation

### Incremental Delivery

1. **Setup + Foundational** → Core types ready
2. **Add US1** → Basic envelope works → MVP!
3. **Add US2 + US3** → MSFA-compatible behavior
4. **Add US7** → Performance validated
5. **Add US4, US5, US6** → Advanced features
6. **Add US8** → All variants supported

### P1 Stories Together (Recommended)

Since US1, US2, US3, US7 are all P1 and tightly coupled:
1. Complete Phases 1-2
2. Complete Phases 3-6 together (all P1 stories)
3. Validate against MSFA reference and performance targets
4. Continue with P2 and P3 stories

---

## Task Count Summary

| Phase | Story | Task Count |
|-------|-------|------------|
| Phase 1 | Setup | 5 |
| Phase 2 | Foundational | 9 |
| Phase 3 | US1 - Basic Envelope | 13 (4 tests + 9 impl) |
| Phase 4 | US2 - Rate Scaling | 7 (3 tests + 4 impl) |
| Phase 5 | US3 - MSFA Rates | 9 (4 tests + 5 impl) |
| Phase 6 | US7 - Performance | 12 (2 tests + 3 bench + 7 impl) |
| Phase 7 | US4 - Attack Jump | 5 (2 tests + 3 impl) |
| Phase 8 | US5 - Delay | 6 (2 tests + 4 impl) |
| Phase 9 | US6 - Looping | 6 (3 tests + 3 impl) |
| Phase 10 | US8 - Variants | 6 (3 tests + 3 impl) |
| Phase 11 | Polish | 6 |
| **Total** | | **84 tasks** |

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify tests fail before implementing
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Performance targets: <50ns/sample single, <100µs for 1536x64 batch
