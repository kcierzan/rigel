# Tasks: Fast DSP Math Library

**Input**: Design documents from `/specs/001-fast-dsp-math/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/api-surface.md

**Tests**: Property-based, accuracy, backend consistency, and integration tests are REQUIRED per constitution principle III (Test-Driven Validation) and user story US10.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

This feature creates `rigel-math` crate at `projects/rigel-synth/crates/math/` within the rigel-synth workspace.

---

## Phase 1: Setup (Project Initialization)

**Purpose**: Initialize rigel-math crate structure and add to workspace

- [ ] T001 Create rigel-math crate directory at projects/rigel-synth/crates/math/
- [ ] T002 Create Cargo.toml for rigel-math with feature flags (scalar, avx2, avx512, neon) at projects/rigel-synth/crates/math/Cargo.toml
- [ ] T003 Add rigel-math to workspace members in projects/rigel-synth/Cargo.toml
- [ ] T004 Create lib.rs with no_std attribute at projects/rigel-synth/crates/math/src/lib.rs
- [ ] T005 [P] Create benches/ directory with Cargo.toml at projects/rigel-synth/crates/math/benches/
- [ ] T006 [P] Create tests/ directory at projects/rigel-synth/crates/math/tests/
- [ ] T007 Configure proptest dependency for property-based testing in projects/rigel-synth/crates/math/Cargo.toml
- [ ] T008 Configure criterion and iai-callgrind dependencies for benchmarking in projects/rigel-synth/crates/math/Cargo.toml

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core SIMD trait abstractions and backend selection infrastructure that ALL user stories depend on

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [ ] T009 Define SimdVector trait with associated types and method signatures in projects/rigel-synth/crates/math/src/traits.rs
- [ ] T010 Define SimdMask trait with boolean and bitwise operations in projects/rigel-synth/crates/math/src/traits.rs
- [ ] T011 Create backends module with cfg-based backend selection in projects/rigel-synth/crates/math/src/backends/mod.rs
- [ ] T012 Implement ScalarVector<T> and ScalarMask in projects/rigel-synth/crates/math/src/backends/scalar.rs
- [ ] T013 Create DefaultSimdVector type alias resolving to active backend in projects/rigel-synth/crates/math/src/lib.rs
- [ ] T014 Re-export core traits and types from lib.rs for public API in projects/rigel-synth/crates/math/src/lib.rs
- [ ] T015 Add compile-time checks to prevent multiple backend features in projects/rigel-synth/crates/math/src/backends/mod.rs

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - SIMD Abstraction Layer with Trait-Based Backends (Priority: P1) 🎯 MVP

**Goal**: Provide trait-based SIMD abstraction allowing DSP code to compile to scalar, AVX2, AVX512, or NEON without #[cfg] directives

**Independent Test**: Write simple DSP algorithm (vector addition) using trait abstraction, compile with different features (scalar, avx2, avx512, neon), verify identical results and expected performance scaling

### Backend Implementations for User Story 1

- [ ] T016 [P] [US1] Implement AVX2 backend: Avx2Vector<T> wrapper around __m256/__m256d in projects/rigel-synth/crates/math/src/backends/avx2.rs
- [ ] T017 [P] [US1] Implement AVX2 mask type: Avx2Mask wrapper around __m256 in projects/rigel-synth/crates/math/src/backends/avx2.rs
- [ ] T018 [P] [US1] Implement AVX512 backend: Avx512Vector<T> wrapper around __m512/__m512d in projects/rigel-synth/crates/math/src/backends/avx512.rs
- [ ] T019 [P] [US1] Implement AVX512 mask type: Avx512Mask wrapper around __mmask16 in projects/rigel-synth/crates/math/src/backends/avx512.rs
- [ ] T020 [P] [US1] Implement NEON backend: NeonVector<T> wrapper around float32x4_t/float64x2_t in projects/rigel-synth/crates/math/src/backends/neon.rs
- [ ] T021 [P] [US1] Implement NEON mask type: NeonMask wrapper around uint32x4_t in projects/rigel-synth/crates/math/src/backends/neon.rs

### Tests for User Story 1

- [ ] T022 [P] [US1] Property-based test: SimdVector arithmetic commutativity across all backends in projects/rigel-synth/crates/math/tests/properties.rs
- [ ] T023 [P] [US1] Property-based test: SimdVector arithmetic associativity across all backends in projects/rigel-synth/crates/math/tests/properties.rs
- [ ] T024 [P] [US1] Backend consistency test: scalar vs SIMD backends produce identical results in projects/rigel-synth/crates/math/tests/backend_consistency.rs
- [ ] T025 [P] [US1] Integration test: Simple DSP algorithm compiles and runs with all backends in projects/rigel-synth/crates/math/tests/backend_consistency.rs
- [ ] T026 [P] [US1] Unit tests: Edge cases (NaN, infinity, zero) for all backend implementations in projects/rigel-synth/crates/math/src/backends/mod.rs

### Benchmarks for User Story 1

- [ ] T027 [US1] Criterion benchmark: Measure wall-clock time for vector operations across backends in projects/rigel-synth/crates/math/benches/criterion_benches.rs
- [ ] T028 [US1] iai-callgrind benchmark: Measure instruction counts for vector operations in projects/rigel-synth/crates/math/benches/iai_benches.rs
- [ ] T029 [US1] Validate performance scaling: AVX2 4-8x, AVX512 8-16x, NEON 4-8x vs scalar in projects/rigel-synth/crates/math/benches/criterion_benches.rs

**Checkpoint**: At this point, User Story 1 should be fully functional - developers can write backend-agnostic SIMD code using trait abstractions

---

## Phase 4: User Story 2 - Block Processing Pattern (Priority: P1)

**Goal**: Provide standardized block processing with fixed sizes (64/128 samples) and clear SIMD lane packing conventions

**Independent Test**: Implement block processor, verify memory alignment, measure cache efficiency, confirm SIMD lane packing conventions

### Implementation for User Story 2

- [ ] T030 [P] [US2] Define AudioBlock<T, const N: usize> struct with alignment attributes in projects/rigel-synth/crates/math/src/block.rs
- [ ] T031 [US2] Implement AudioBlock::new() and AudioBlock::from_slice() in projects/rigel-synth/crates/math/src/block.rs
- [ ] T032 [US2] Implement AudioBlock::as_chunks<V: SimdVector>() for immutable SIMD views in projects/rigel-synth/crates/math/src/block.rs
- [ ] T033 [US2] Implement AudioBlock::as_chunks_mut<V: SimdVector>() for mutable SIMD views in projects/rigel-synth/crates/math/src/block.rs
- [ ] T034 [US2] Create Block64 and Block128 type aliases in projects/rigel-synth/crates/math/src/block.rs
- [ ] T035 [US2] Document memory layout and lane packing conventions in projects/rigel-synth/crates/math/src/block.rs

### Tests for User Story 2

- [ ] T036 [P] [US2] Unit test: Verify alignment (32-byte AVX2, 64-byte AVX512, 16-byte NEON) in projects/rigel-synth/crates/math/src/block.rs
- [ ] T037 [P] [US2] Integration test: Block processing with as_chunks enables loop unrolling (assembly inspection) in projects/rigel-synth/crates/math/tests/block_processing.rs
- [ ] T038 [P] [US2] Property-based test: Block processing with arbitrary inputs maintains correctness in projects/rigel-synth/crates/math/tests/properties.rs

**Checkpoint**: Block processing infrastructure ready - enables efficient SIMD-friendly memory access patterns

---

## Phase 5: User Story 3 - Core Vector Operations (Priority: P1)

**Goal**: Provide fundamental vector operations (arithmetic, FMA, min/max, compare, horizontal) through SIMD abstraction

**Independent Test**: Test each operation across all backends, verify numerical correctness and performance with property-based tests

### Implementation for User Story 3

- [ ] T039 [P] [US3] Implement vector arithmetic operations module in projects/rigel-synth/crates/math/src/ops/arithmetic.rs
- [ ] T040 [P] [US3] Implement FMA operations module in projects/rigel-synth/crates/math/src/ops/fma.rs
- [ ] T041 [P] [US3] Implement min/max/clamp operations module in projects/rigel-synth/crates/math/src/ops/minmax.rs
- [ ] T042 [P] [US3] Implement comparison operations (lt, gt, eq) returning masks in projects/rigel-synth/crates/math/src/ops/compare.rs
- [ ] T043 [P] [US3] Implement horizontal operations (sum, max, min) module in projects/rigel-synth/crates/math/src/ops/horizontal.rs
- [ ] T044 [US3] Create ops module with re-exports in projects/rigel-synth/crates/math/src/ops/mod.rs

### Tests for User Story 3

- [ ] T045 [P] [US3] Property-based test: FMA accuracy vs separate multiply-add across backends in projects/rigel-synth/crates/math/tests/properties.rs
- [ ] T046 [P] [US3] Property-based test: Min/max operations with edge cases (NaN, infinity) in projects/rigel-synth/crates/math/tests/properties.rs
- [ ] T047 [P] [US3] Property-based test: Horizontal sum correctness across all backends in projects/rigel-synth/crates/math/tests/properties.rs
- [ ] T048 [P] [US3] Unit test: Comparison masks enable conditional logic without branching in projects/rigel-synth/crates/math/src/ops/compare.rs
- [ ] T049 [P] [US3] Backend consistency test: All vector operations produce results within error bounds in projects/rigel-synth/crates/math/tests/backend_consistency.rs

### Benchmarks for User Story 3

- [ ] T050 [US3] Criterion benchmark: Wall-clock performance of all vector operations in projects/rigel-synth/crates/math/benches/criterion_benches.rs
- [ ] T051 [US3] iai-callgrind benchmark: Verify FMA uses single instruction on supporting backends in projects/rigel-synth/crates/math/benches/iai_benches.rs

**Checkpoint**: Core vector operations complete - developers can build complex DSP algorithms from optimized primitives

---

## Phase 6: User Story 6 - Denormal Handling (Priority: P1)

**Goal**: Automatic denormal protection integrated into block processing to prevent performance degradation

**Independent Test**: Process signals decaying to denormal range, measure CPU usage remains constant, verify no audible artifacts

### Implementation for User Story 6

- [ ] T052 [P] [US6] Implement DenormalGuard struct with RAII pattern in projects/rigel-synth/crates/math/src/denormal.rs
- [ ] T053 [US6] Implement x86-64 denormal protection (FTZ/DAZ flags in MXCSR) in projects/rigel-synth/crates/math/src/denormal.rs
- [ ] T054 [US6] Implement ARM64 denormal protection (FZ flag in FPCR) in projects/rigel-synth/crates/math/src/denormal.rs
- [ ] T055 [US6] Implement Drop trait for DenormalGuard to restore FPU state in projects/rigel-synth/crates/math/src/denormal.rs
- [ ] T056 [US6] Add with_denormal_protection convenience function in projects/rigel-synth/crates/math/src/denormal.rs

### Tests for User Story 6

- [ ] T057 [P] [US6] Performance test: CPU usage remains constant when processing denormals in projects/rigel-synth/crates/math/tests/denormal_tests.rs
- [ ] T058 [P] [US6] Accuracy test: Denormal protection introduces no audible artifacts (THD+N < -96dB) in projects/rigel-synth/crates/math/tests/accuracy.rs
- [ ] T059 [P] [US6] Unit test: DenormalGuard::is_available() returns correct value per platform in projects/rigel-synth/crates/math/src/denormal.rs

**Checkpoint**: Denormal protection prevents catastrophic performance drops during silence processing

---

## Phase 7: User Story 9 - Backend Selection and Benchmarking (Priority: P1)

**Goal**: Enable compiling/running with different backends and comparing performance through comprehensive benchmarks

**Independent Test**: Run benchmark suite with different features, compare instruction counts and wall-clock times, verify scaling

### Implementation for User Story 9

- [ ] T060 [P] [US9] Create comprehensive Criterion benchmark suite covering all operations in projects/rigel-synth/crates/math/benches/criterion_benches.rs
- [ ] T061 [P] [US9] Create iai-callgrind benchmark suite for instruction count measurements in projects/rigel-synth/crates/math/benches/iai_benches.rs
- [ ] T062 [US9] Add benchmark configuration for running with different backends in projects/rigel-synth/crates/math/benches/Cargo.toml
- [ ] T063 [US9] Document benchmark execution and result interpretation in projects/rigel-synth/crates/math/README.md

### Tests for User Story 9

- [ ] T064 [P] [US9] CI test: Run benchmark suite with --features scalar and verify compilation in .github/workflows/ci.yml
- [ ] T065 [P] [US9] CI test: Run benchmark suite with --features avx2 on x86-64 runner in .github/workflows/ci.yml
- [ ] T066 [P] [US9] CI test: Run benchmark suite with --features avx512 on x86-64 runner in .github/workflows/ci.yml
- [ ] T067 [P] [US9] CI test: Run benchmark suite with --features neon on ARM64 runner in .github/workflows/ci.yml

**Checkpoint**: Backend selection and benchmarking infrastructure enables validating performance claims

---

## Phase 8: User Story 10 - Comprehensive Test Coverage (Priority: P1)

**Goal**: Comprehensive test coverage ensuring correctness and performance across all operations and backends

**Independent Test**: Run complete test suite, verify property-based tests catch invariant violations, confirm >90% line coverage

### Implementation for User Story 10

- [ ] T068 [P] [US10] Create test_utils module with reference implementations (libm) in projects/rigel-synth/crates/math/tests/test_utils.rs
- [ ] T069 [P] [US10] Create proptest strategies for normal, denormal, and edge-case floats in projects/rigel-synth/crates/math/tests/test_utils.rs
- [ ] T070 [US10] Implement assert_backend_consistency helper in projects/rigel-synth/crates/math/tests/test_utils.rs
- [ ] T071 [US10] Configure proptest to generate 10,000+ test cases per operation in projects/rigel-synth/crates/math/tests/properties.rs

### Tests for User Story 10

- [ ] T072 [P] [US10] Property-based test: Mathematical invariants hold across thousands of random inputs in projects/rigel-synth/crates/math/tests/properties.rs
- [ ] T073 [P] [US10] Documentation tests: All code examples in API docs compile and execute in projects/rigel-synth/crates/math/src/lib.rs
- [ ] T074 [P] [US10] Unit tests: Edge cases (NaN, infinity, denormals, zero, extreme values) handled gracefully in projects/rigel-synth/crates/math/src/
- [ ] T075 [P] [US10] Code coverage: Verify >90% line coverage and >95% branch coverage for critical paths using tarpaulin or llvm-cov
- [ ] T076 [US10] Performance regression test: Detect >5% instruction count or >10% wall-clock degradation in projects/rigel-synth/crates/math/tests/regression_tests.rs

**Checkpoint**: Comprehensive test infrastructure ensures library quality and catches regressions

---

## Phase 9: User Story 4 - Fast Math Kernels (Priority: P2)

**Goal**: Vectorized fast math kernels (tanh, exp, log1p, sin, cos, inverse) working through SIMD abstraction

**Independent Test**: Compare vectorized kernel outputs against reference implementations, measure error bounds, benchmark performance

### Implementation for User Story 4

- [ ] T077 [P] [US4] Implement vectorized tanh approximation in projects/rigel-synth/crates/math/src/math/tanh.rs
- [ ] T078 [P] [US4] Implement vectorized exp approximation in projects/rigel-synth/crates/math/src/math/exp.rs
- [ ] T079 [P] [US4] Implement vectorized log and log1p approximations in projects/rigel-synth/crates/math/src/math/log.rs
- [ ] T080 [P] [US4] Implement vectorized sin/cos approximations with sincos variant in projects/rigel-synth/crates/math/src/math/trig.rs
- [ ] T081 [P] [US4] Implement vectorized fast inverse (1/x) with Newton-Raphson refinement in projects/rigel-synth/crates/math/src/math/inverse.rs
- [ ] T082 [P] [US4] Implement vectorized sqrt and rsqrt in projects/rigel-synth/crates/math/src/math/sqrt.rs
- [ ] T083 [P] [US4] Implement vectorized pow function in projects/rigel-synth/crates/math/src/math/pow.rs
- [ ] T084 [US4] Create math module with re-exports in projects/rigel-synth/crates/math/src/math/mod.rs

### Tests for User Story 4

- [ ] T085 [P] [US4] Accuracy test: tanh error <0.1% vs libm reference across all backends in projects/rigel-synth/crates/math/tests/accuracy.rs
- [ ] T086 [P] [US4] Accuracy test: exp error <0.1% vs libm reference across all backends in projects/rigel-synth/crates/math/tests/accuracy.rs
- [ ] T087 [P] [US4] Accuracy test: log1p error <0.001% for frequency calculations in projects/rigel-synth/crates/math/tests/accuracy.rs
- [ ] T088 [P] [US4] Accuracy test: sin/cos harmonic distortion <-100dB across all backends in projects/rigel-synth/crates/math/tests/accuracy.rs
- [ ] T089 [P] [US4] Accuracy test: fast inverse error <0.01% at 5-10x speedup in projects/rigel-synth/crates/math/tests/accuracy.rs
- [ ] T090 [P] [US4] Property-based test: Math kernels handle edge cases (NaN, infinity, denormals) gracefully in projects/rigel-synth/crates/math/tests/properties.rs

### Benchmarks for User Story 4

- [ ] T091 [US4] Criterion benchmark: Verify tanh executes 8-16x faster than scalar in projects/rigel-synth/crates/math/benches/criterion_benches.rs
- [ ] T092 [US4] Criterion benchmark: Verify exp achieves sub-nanosecond per-sample throughput in projects/rigel-synth/crates/math/benches/criterion_benches.rs
- [ ] T093 [US4] iai-callgrind benchmark: Measure instruction counts for all math kernels in projects/rigel-synth/crates/math/benches/iai_benches.rs

**Checkpoint**: Fast math kernels enable complex mathematical transformations in DSP algorithms

---

## Phase 10: User Story 5 - Lookup Table Infrastructure (Priority: P2)

**Goal**: Vectorized lookup table mechanisms with interpolation for wavetable synthesis

**Independent Test**: Create sample wavetables, perform vectorized lookups across blocks, measure performance and interpolation quality

### Implementation for User Story 5

- [ ] T094 [P] [US5] Define LookupTable<T, const SIZE: usize> struct in projects/rigel-synth/crates/math/src/table.rs
- [ ] T095 [P] [US5] Define IndexMode enum (Wrap, Mirror, Clamp) in projects/rigel-synth/crates/math/src/table.rs
- [ ] T096 [US5] Implement LookupTable::from_fn constructor in projects/rigel-synth/crates/math/src/table.rs
- [ ] T097 [US5] Implement scalar lookup_linear and lookup_cubic in projects/rigel-synth/crates/math/src/table.rs
- [ ] T098 [US5] Implement vectorized lookup_linear_simd with SIMD gather operations in projects/rigel-synth/crates/math/src/table.rs
- [ ] T099 [US5] Implement vectorized lookup_cubic_simd with SIMD gather operations in projects/rigel-synth/crates/math/src/table.rs

### Tests for User Story 5

- [ ] T100 [P] [US5] Performance test: 64-sample block lookup completes in <640ns (<10ns/sample) in projects/rigel-synth/crates/math/tests/table_tests.rs
- [ ] T101 [P] [US5] Accuracy test: Linear interpolation maintains phase continuity across block in projects/rigel-synth/crates/math/tests/accuracy.rs
- [ ] T102 [P] [US5] Property-based test: SIMD gather provides correct per-lane indexing in projects/rigel-synth/crates/math/tests/properties.rs
- [ ] T103 [P] [US5] Unit test: IndexMode variants (Wrap, Mirror, Clamp) behave correctly at boundaries in projects/rigel-synth/crates/math/src/table.rs

**Checkpoint**: Lookup table infrastructure enables efficient wavetable synthesis

---

## Phase 11: User Story 7 - Soft Saturation and Waveshaping (Priority: P3)

**Goal**: Vectorized saturation curves (soft clip, tube-style, tape-style) for harmonic richness

**Independent Test**: Apply vectorized saturation to test signals, analyze harmonic content, compare performance

### Implementation for User Story 7

- [ ] T104 [P] [US7] Implement saturate_soft_clip using tanh-based symmetric saturation in projects/rigel-synth/crates/math/src/saturate.rs
- [ ] T105 [P] [US7] Implement saturate_tube with asymmetric warm harmonics in projects/rigel-synth/crates/math/src/saturate.rs
- [ ] T106 [P] [US7] Implement saturate_tape with high-frequency rolloff characteristics in projects/rigel-synth/crates/math/src/saturate.rs

### Tests for User Story 7

- [ ] T107 [P] [US7] Accuracy test: Soft saturation produces harmonically-rich content without digital clipping in projects/rigel-synth/crates/math/tests/accuracy.rs
- [ ] T108 [P] [US7] Property-based test: Different saturation curves produce distinct harmonic characteristics in projects/rigel-synth/crates/math/tests/properties.rs
- [ ] T109 [P] [US7] Performance test: Saturation on 64-sample blocks is 3x faster than per-sample polynomial in projects/rigel-synth/crates/math/benches/criterion_benches.rs

**Checkpoint**: Saturation utilities enhance sonic palette with musical warmth

---

## Phase 12: User Story 8 - Crossfade and Ramping Utilities (Priority: P3)

**Goal**: Vectorized crossfade and parameter ramping for click-free transitions

**Independent Test**: Measure crossfade curves for equal-power characteristics, verify no audible clicks during parameter changes

### Implementation for User Story 8

- [ ] T110 [P] [US8] Implement crossfade_linear, crossfade_equal_power, crossfade_scurve in projects/rigel-synth/crates/math/src/crossfade.rs
- [ ] T111 [US8] Implement ParameterRamp struct with linear ramping in projects/rigel-synth/crates/math/src/crossfade.rs
- [ ] T112 [US8] Implement ParameterRamp::fill_block for efficient block filling in projects/rigel-synth/crates/math/src/crossfade.rs

### Tests for User Story 8

- [ ] T113 [P] [US8] Accuracy test: Equal-power crossfade maintains constant energy (no volume dip) in projects/rigel-synth/crates/math/tests/accuracy.rs
- [ ] T114 [P] [US8] Accuracy test: Parameter ramping produces no audible clicks or zipper noise in projects/rigel-synth/crates/math/tests/accuracy.rs
- [ ] T115 [P] [US8] Property-based test: Crossfade curves are smooth and monotonic in projects/rigel-synth/crates/math/tests/properties.rs

**Checkpoint**: Crossfade and ramping utilities enable professional-quality parameter transitions

---

## Phase 13: Polish & Cross-Cutting Concerns

**Purpose**: Documentation, optimization, and final validation across all user stories

- [ ] T116 [P] Create comprehensive README.md with quick start guide at projects/rigel-synth/crates/math/README.md
- [ ] T117 [P] Document all public API functions with error bounds and performance characteristics in projects/rigel-synth/crates/math/src/
- [ ] T118 [P] Add inline documentation tests for all code examples in projects/rigel-synth/crates/math/src/
- [ ] T119 [P] Run rustfmt on all source files in projects/rigel-synth/crates/math/
- [ ] T120 [P] Run clippy and fix all warnings in projects/rigel-synth/crates/math/
- [ ] T121 Validate all quickstart.md examples compile and execute correctly
- [ ] T122 Run complete benchmark suite and verify performance targets met
- [ ] T123 Generate and review code coverage report (target: >90% line, >95% branch for critical paths)
- [ ] T124 Update workspace Cargo.toml with rigel-math documentation metadata
- [ ] T125 Add CI workflow jobs for rigel-math testing across all backends

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories P1 (Phases 3-8)**: All depend on Foundational phase completion
  - US1, US2, US3, US6, US9, US10 can proceed in parallel after Foundational
  - US9 and US10 should run continuously alongside other stories
- **User Stories P2 (Phases 9-10)**: Depend on Foundational + US1 + US2 + US3
  - US4 and US5 can proceed in parallel after prerequisites met
- **User Stories P3 (Phases 11-12)**: Depend on Foundational + US1 + US2 + US3 + US4
  - US7 and US8 can proceed in parallel after prerequisites met
- **Polish (Phase 13)**: Depends on all desired user stories being complete

### User Story Dependencies

- **US1 (SIMD Abstraction)**: Foundation only - No other story dependencies
- **US2 (Block Processing)**: Foundation only - No other story dependencies
- **US3 (Vector Operations)**: Foundation + US1 (needs trait implementations)
- **US6 (Denormal Handling)**: Foundation only - No other story dependencies
- **US9 (Benchmarking)**: Foundation + US1 + US2 + US3 (needs operations to benchmark)
- **US10 (Test Coverage)**: Foundation + US1 (needs backends to test, runs continuously)
- **US4 (Math Kernels)**: Foundation + US1 + US2 + US3 (needs vector ops as primitives)
- **US5 (Lookup Tables)**: Foundation + US1 + US2 + US3 (needs vector ops for interpolation)
- **US7 (Saturation)**: Foundation + US1 + US2 + US3 + US4 (needs math kernels like tanh)
- **US8 (Crossfade)**: Foundation + US1 + US2 + US3 + US4 (needs math kernels for curves)

### Within Each User Story

- Backend implementations before tests
- Tests written to FAIL before implementation begins
- Unit tests → property-based tests → integration tests → benchmarks
- Core implementation before optimization
- Story complete before moving to next priority

### Parallel Opportunities

- **Phase 1 (Setup)**: All tasks except T003 can run in parallel
- **Phase 2 (Foundational)**: Tasks T009-T014 can run in parallel after T009 defines traits
- **Phase 3 (US1)**: Backend implementations T016-T021 can run in parallel, tests T022-T026 can run in parallel
- **Phase 4 (US2)**: Tests T036-T038 can run in parallel
- **Phase 5 (US3)**: Operation modules T039-T043 can run in parallel, tests T045-T049 can run in parallel
- **Phase 6 (US6)**: Implementation T052-T056 in sequence, tests T057-T059 can run in parallel
- **Phase 7 (US9)**: T060-T061 can run in parallel, CI tests T064-T067 can run in parallel
- **Phase 8 (US10)**: T068-T069 can run in parallel, tests T072-T075 can run in parallel
- **Phase 9 (US4)**: All math kernel implementations T077-T083 can run in parallel, all tests T085-T090 can run in parallel
- **Phase 10 (US5)**: T094-T095 can run in parallel, tests T100-T103 can run in parallel
- **Phase 11 (US7)**: All saturation implementations T104-T106 can run in parallel, all tests T107-T109 can run in parallel
- **Phase 12 (US8)**: Tests T113-T115 can run in parallel
- **Phase 13 (Polish)**: All documentation tasks T116-T120 can run in parallel

---

## Parallel Example: User Story 1 (SIMD Abstraction)

```bash
# Launch all backend implementations in parallel:
Task: "Implement AVX2 backend in projects/rigel-synth/crates/math/src/backends/avx2.rs"
Task: "Implement AVX512 backend in projects/rigel-synth/crates/math/src/backends/avx512.rs"
Task: "Implement NEON backend in projects/rigel-synth/crates/math/src/backends/neon.rs"

# Launch all tests in parallel:
Task: "Property-based test: commutativity in projects/rigel-synth/crates/math/tests/properties.rs"
Task: "Backend consistency test in projects/rigel-synth/crates/math/tests/backend_consistency.rs"
Task: "Unit tests: edge cases in projects/rigel-synth/crates/math/src/backends/mod.rs"
```

---

## Implementation Strategy

### MVP First (P1 User Stories Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phases 3-8: US1, US2, US3, US6, US9, US10 (all P1)
4. **STOP and VALIDATE**: Run complete test suite, benchmarks, verify >90% coverage
5. At this point you have a production-ready SIMD library with comprehensive tests

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add US1 → Test independently → Zero-cost SIMD abstraction working
3. Add US2 → Test independently → Block processing enabled
4. Add US3 → Test independently → Core vector operations available
5. Add US6 → Test independently → Denormal protection prevents performance drops
6. Add US9 + US10 → Validate all above stories via benchmarks and tests (MVP COMPLETE)
7. Add US4 → Test independently → Fast math kernels available
8. Add US5 → Test independently → Wavetable synthesis enabled
9. Add US7 → Test independently → Saturation and waveshaping available
10. Add US8 → Test independently → Professional parameter transitions

### Parallel Team Strategy

With multiple developers after Foundational phase completes:

**Wave 1 (P1 - Core Infrastructure)**:
- Developer A: US1 (SIMD Abstraction - foundation for all)
- Developer B: US2 (Block Processing - independent)
- Developer C: US6 (Denormal Handling - independent)

**Wave 2 (P1 - Build on Wave 1)**:
- Developer A: US3 (Vector Operations - needs US1)
- Developer B: US9 (Benchmarking - needs US1, US2, US3)
- Developer C: US10 (Test Coverage - ongoing, needs US1)

**Wave 3 (P2 - Advanced Features)**:
- Developer A: US4 (Math Kernels - needs US1, US2, US3)
- Developer B: US5 (Lookup Tables - needs US1, US2, US3)

**Wave 4 (P3 - Polish)**:
- Developer A: US7 (Saturation - needs US4)
- Developer B: US8 (Crossfade - needs US4)

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Tests are REQUIRED per constitution principles - property-based, accuracy, backend consistency, and integration tests
- Verify tests fail before implementing (TDD approach)
- Constitution requires >90% line coverage and >95% branch coverage for critical paths
- Benchmark after each story to verify performance targets
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- All code must be no_std compatible for rigel-dsp integration
