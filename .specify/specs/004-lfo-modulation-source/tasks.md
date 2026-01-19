# Tasks: LFO Modulation Source

**Input**: Design documents from `.specify/specs/004-lfo-modulation-source/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, contracts/

**Tests**: Unit tests and benchmarks will be included in appropriate phases.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Crate root**: `projects/rigel-synth/crates/modulation/`
- **Source**: `projects/rigel-synth/crates/modulation/src/`
- **Tests**: `projects/rigel-synth/crates/modulation/tests/`
- **Benchmarks**: `projects/rigel-synth/crates/modulation/benches/`

---

## Phase 1: Setup (Crate Initialization)

**Purpose**: Create the rigel-modulation crate structure and configure dependencies

- [X] T001 Create crate directory structure at `projects/rigel-synth/crates/modulation/`
- [X] T002 Create `Cargo.toml` with dependencies on rigel-timing, rigel-math, and no_std configuration
- [X] T003 [P] Create `src/lib.rs` with module declarations and public re-exports
- [X] T004 [P] Add rigel-modulation to workspace `Cargo.toml` members list

---

## Phase 2: Foundational (ModulationSource Trait Migration)

**Purpose**: Move ModulationSource trait from rigel-timing to rigel-modulation with backward compatibility

**CRITICAL**: No user story work can begin until this phase is complete

- [X] T005 Create `src/traits.rs` with ModulationSource re-export from rigel-timing (trait stays canonical in rigel-timing to avoid circular deps)
- [X] T006 N/A - rigel-timing keeps the trait, rigel-modulation re-exports it
- [X] T007 N/A - no changes needed to rigel-timing
- [X] T008 N/A - trait stays in rigel-timing
- [X] T009 Verified: `rigel_timing::ModulationSource` works, `rigel_modulation::ModulationSource` also works via re-export

**Checkpoint**: ModulationSource trait is now canonical in rigel-modulation, backward compat verified

---

## Phase 3: User Story 1 - Basic LFO Modulation (Priority: P1)

**Goal**: LFO generates periodic output values with 7 waveshapes at configurable Hz rate

**Independent Test**: Create LFO, set waveshape/rate, advance time, verify output oscillates within correct range

### Implementation for User Story 1

- [X] T010 [P] [US1] Create `src/rng.rs` with PCG32 Rng struct per contracts/types.rs
- [X] T011 [P] [US1] Create `src/waveshape.rs` with LfoWaveshape enum (all 7 variants) per contracts/types.rs
- [X] T012 [P] [US1] Create `src/rate.rs` with LfoRateMode::Hz variant only (TempoSync deferred to US2)
- [X] T013 [US1] Create `src/lfo.rs` with Lfo struct skeleton (fields, new(), Default) per contracts/lfo.rs
- [X] T014 [US1] Implement waveshape generation functions in `src/waveshape.rs` (sine via rigel_math::fast_sinf, triangle, saw, square formulas)
- [X] T015 [US1] Implement Lfo::update() for Hz rate mode with phase accumulation in `src/lfo.rs`
- [X] T016 [US1] Implement Lfo::value() returning cached current_value in `src/lfo.rs`
- [X] T017 [US1] Implement ModulationSource trait for Lfo (reset, update, value) in `src/lfo.rs`
- [X] T018 [US1] Implement configuration setters (set_waveshape, set_rate) in `src/lfo.rs`
- [X] T019 [US1] Implement configuration getters (waveshape, rate_mode, phase) in `src/lfo.rs`
- [X] T020 [US1] Update `src/lib.rs` to export all US1 types (Lfo, LfoWaveshape, LfoRateMode, ModulationSource, Rng)
- [X] T021 [US1] Create `tests/lfo_tests.rs` with basic LFO tests: output range, cycle frequency, waveshape correctness

**Checkpoint**: Basic LFO with Hz rate and 4 waveshapes (sine, triangle, saw, square) is functional

---

## Phase 4: User Story 2 - Tempo-Synchronized Modulation (Priority: P1)

**Goal**: LFO rate syncs to BPM with note divisions (whole, half, quarter, eighth, sixteenth, thirty-second)

**Independent Test**: Set tempo-sync mode with known BPM and division, verify effective rate matches expected Hz

### Implementation for User Story 2

- [X] T022 [P] [US2] Add NoteDivision, NoteBase, NoteModifier types to `src/rate.rs` per contracts/types.rs
- [X] T023 [P] [US2] Implement NoteDivision::multiplier() and NoteDivision::to_hz() in `src/rate.rs`
- [X] T024 [US2] Add LfoRateMode::TempoSync variant to `src/rate.rs`
- [X] T025 [US2] Implement Lfo::effective_rate_hz() for both Hz and TempoSync modes in `src/lfo.rs`
- [X] T026 [US2] Implement Lfo::set_tempo() for updating BPM in TempoSync mode in `src/lfo.rs`
- [X] T027 [US2] Update Lfo::update() to use effective_rate_hz() for phase increment in `src/lfo.rs`
- [X] T028 [US2] Update `src/lib.rs` to export NoteDivision, NoteBase, NoteModifier
- [X] T029 [US2] Add tempo sync tests to `tests/lfo_tests.rs`: rate calculation, BPM changes, dotted/triplet modifiers

**Checkpoint**: LFO tempo sync works with all note divisions and modifiers

---

## Phase 5: User Story 7 - Control Rate Processing (Priority: P1)

**Goal**: LFO integrates with ControlRateClock for efficient control-rate updates

**Independent Test**: Update LFO at different control rate intervals, verify output consistency and CPU reduction

### Implementation for User Story 7

- [X] T030 [US7] Verify Lfo::update() uses Timebase::block_size() for elapsed samples in `src/lfo.rs`
- [X] T031 [US7] Verify Lfo::value() returns cached value with zero computation in `src/lfo.rs`
- [X] T032 [US7] Add control rate integration tests to `tests/lfo_tests.rs`: different intervals (1, 32, 64, 128), output consistency
- [X] T033 [US7] Create `benches/lfo_bench.rs` with Criterion benchmarks for control rate vs sample rate comparison

**Checkpoint**: LFO works correctly at any control rate interval with verified CPU efficiency

---

## Phase 6: User Story 3 - Phase Reset on Note Trigger (Priority: P2)

**Goal**: LFO phase resets to configurable start_phase on note-on events in Retrigger mode

**Independent Test**: Set retrigger mode, trigger LFO mid-cycle, verify phase resets to start_phase

### Implementation for User Story 3

- [X] T034 [P] [US3] Add LfoPhaseMode enum (FreeRunning, Retrigger) to `src/lfo.rs` or new `src/phase.rs`
- [X] T035 [US3] Add phase_mode and start_phase fields to Lfo struct in `src/lfo.rs`
- [X] T036 [US3] Implement Lfo::set_phase_mode() and Lfo::set_start_phase() in `src/lfo.rs`
- [X] T037 [US3] Implement Lfo::trigger() method (reset phase if Retrigger mode) in `src/lfo.rs`
- [X] T038 [US3] Update `src/lib.rs` to export LfoPhaseMode
- [X] T039 [US3] Add phase reset tests to `tests/lfo_tests.rs`: retrigger mode, free-running mode, start_phase variations

**Checkpoint**: Phase reset on trigger works in Retrigger mode, ignored in FreeRunning mode

---

## Phase 7: User Story 4 - PWM Control for Pulse Wave (Priority: P2)

**Goal**: Pulse waveshape supports configurable pulse width from 1% to 99%

**Independent Test**: Set pulse waveshape with various widths, verify duty cycle matches configured value

### Implementation for User Story 4

- [X] T040 [US4] Add pulse_width field to Lfo struct in `src/lfo.rs`
- [X] T041 [US4] Implement Lfo::set_pulse_width() with validation (0.01-0.99) in `src/lfo.rs`
- [X] T042 [US4] Implement pulse waveshape generation using pulse_width in `src/waveshape.rs`
- [X] T043 [US4] Add pulse width tests to `tests/lfo_tests.rs`: various duty cycles, edge cases (min/max)

**Checkpoint**: Pulse waveshape duty cycle is controllable from 1% to 99%

---

## Phase 8: User Story 5 - Sample and Hold Modulation (Priority: P2)

**Goal**: S&H waveshape samples new random value on cycle wrap and holds it constant

**Independent Test**: Advance through multiple cycles, verify value held within cycle, changes at boundaries

### Implementation for User Story 5

- [X] T044 [US5] Add held_value field to Lfo struct in `src/lfo.rs`
- [X] T045 [US5] Implement cycle wrap detection in Lfo::update() in `src/lfo.rs`
- [X] T046 [US5] Implement S&H waveshape: sample new random on wrap, return held_value otherwise in `src/waveshape.rs`
- [X] T047 [US5] Implement Noise waveshape: new random value on every update in `src/waveshape.rs`
- [X] T048 [US5] Update Lfo::trigger() to sample new S&H value on trigger in `src/lfo.rs`
- [X] T049 [US5] Add S&H and Noise tests to `tests/lfo_tests.rs`: value stability, cycle boundary changes, trigger behavior

**Checkpoint**: S&H and Noise waveshapes produce correct random behavior

---

## Phase 9: User Story 6 - Polarity Mode Selection (Priority: P3)

**Goal**: LFO output can be bipolar [-1.0, 1.0] or unipolar [0.0, 1.0]

**Independent Test**: Switch polarity modes, verify output range changes accordingly

### Implementation for User Story 6

- [X] T050 [P] [US6] Add LfoPolarity enum (Bipolar, Unipolar) to `src/lfo.rs` or `src/types.rs`
- [X] T051 [US6] Add polarity field to Lfo struct in `src/lfo.rs`
- [X] T052 [US6] Implement Lfo::set_polarity() in `src/lfo.rs`
- [X] T053 [US6] Apply polarity scaling in Lfo::value() or waveshape generation in `src/lfo.rs`
- [X] T054 [US6] Update `src/lib.rs` to export LfoPolarity
- [X] T055 [US6] Add polarity tests to `tests/lfo_tests.rs`: bipolar range, unipolar range, mode switching

**Checkpoint**: Polarity modes produce correct output ranges

---

## Phase 10: Polish & Cross-Cutting Concerns

**Purpose**: Integration, documentation, and final validation

- [ ] T056 [P] Add rigel-modulation dependency to rigel-dsp `Cargo.toml`
- [ ] T057 [P] Add rigel-modulation dependency to rigel-plugin `Cargo.toml` (if needed)
- [X] T058 Run `cargo fmt` on rigel-modulation crate
- [X] T059 Run `cargo clippy` on rigel-modulation crate and fix warnings
- [X] T060 Run full test suite: `cargo test -p rigel-modulation`
- [X] T061 Run benchmarks: `cargo bench -p rigel-modulation`
- [X] T062 Validate quickstart.md examples compile and run correctly
- [X] T063 Verify all success criteria from spec.md (SC-001 through SC-009)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-9)**: All depend on Foundational phase completion
  - P1 stories (US1, US2, US7) should be completed first
  - P2 stories (US3, US4, US5) can proceed after P1
  - P3 stories (US6) can proceed after P2
- **Polish (Phase 10)**: Depends on all user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P1)**: Depends on US1 (needs Lfo struct and rate infrastructure)
- **User Story 7 (P1)**: Depends on US1 (needs working Lfo::update/value)
- **User Story 3 (P2)**: Depends on US1 (adds phase mode to existing Lfo)
- **User Story 4 (P2)**: Depends on US1 (adds pulse width to existing waveshape)
- **User Story 5 (P2)**: Depends on US1 (adds S&H/Noise to existing waveshape system)
- **User Story 6 (P3)**: Depends on US1 (adds polarity scaling to existing output)

### Within Each User Story

- Models/types before implementation
- Core implementation before configuration
- Configuration before tests
- Story complete before moving to next priority

### Parallel Opportunities

- Setup tasks T003, T004 can run in parallel
- Within US1: T010, T011, T012 can run in parallel (separate files)
- Within US2: T022, T023 can run in parallel
- Within US3: T034 can run in parallel with other story work
- Within US6: T050 can run in parallel

---

## Parallel Example: User Story 1

```bash
# Launch all type definitions for User Story 1 together:
Task: "Create src/rng.rs with PCG32 Rng struct"
Task: "Create src/waveshape.rs with LfoWaveshape enum"
Task: "Create src/rate.rs with LfoRateMode::Hz variant"

# Then sequentially:
Task: "Create src/lfo.rs with Lfo struct skeleton"
Task: "Implement waveshape generation functions"
Task: "Implement Lfo::update() for Hz rate mode"
# ... etc
```

---

## Implementation Strategy

### MVP First (P1 Stories Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (trait migration)
3. Complete Phase 3: User Story 1 (basic LFO)
4. Complete Phase 4: User Story 2 (tempo sync)
5. Complete Phase 5: User Story 7 (control rate)
6. **STOP and VALIDATE**: All P1 stories functional
7. Deploy/demo if ready - LFO is usable with Hz and tempo-sync rates

### Incremental Delivery

1. Setup + Foundational → Foundation ready
2. Add US1 → Basic LFO works (MVP!)
3. Add US2 → Tempo sync works
4. Add US7 → Control rate optimized
5. Add US3 → Phase reset works
6. Add US4 → Pulse PWM works
7. Add US5 → S&H and Noise work
8. Add US6 → Polarity modes work
9. Polish → Production ready

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- All types must be Copy + Clone + Send + Sync
- Zero heap allocations in update() and value()
- Use rigel_math::sin for sine waveshape per constitution
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
