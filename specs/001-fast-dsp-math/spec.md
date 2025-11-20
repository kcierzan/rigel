# Feature Specification: Fast DSP Math Library

**Feature Branch**: `001-fast-dsp-math`
**Created**: 2025-11-17
**Status**: Draft
**Input**: User description: "Create blazingly fast math library for use in rigels dsp-core that targets the latest CPU architectures common on modern macos/linux/windows machines. We should have functions for common DSP operations in synthesizers/effects like tanh approximations, exp, log1p, sin/cos approximations, lookup tables, denorm handling, soft-saturation, cross-fade ramps etc. We should also target AVX512 and any other bleeding edge instruction sets. There should be a single abstraction that works in no_std, has a scalar backend that is always available, has alternate backend for different instruction sets based on cfg/features and can be dropped into later DSP code via traits not #[cfg] soup. There should be a block processing pattern with a fixed block size (64 or 128 samples) with clear conventions for packing samples into SIMD lanes. We will want vector arithmetic, FMA, min/max, compare/masks. we will want horizontal operations like sum and max. we want fast math kernels like those mentioned and fast inverse and others not mentioned but typical of audio DSP. and we will want a way to compile and run with different backends as well as benchmarking processes."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - SIMD Abstraction Layer with Trait-Based Backends (Priority: P1)

As a DSP developer, I need a single trait-based SIMD abstraction that allows me to write DSP algorithms once and have them automatically compile to scalar, AVX2, AVX512, or NEON backends based on compile-time feature flags, so that I can avoid littering my code with platform-specific #[cfg] directives while still achieving maximum performance on each platform.

**Why this priority**: This is the foundational architecture that everything else builds upon. Without this abstraction layer, every DSP algorithm would need separate implementations for each SIMD backend, leading to unmaintainable code duplication. This enables "write once, run fast everywhere."

**Independent Test**: Can be fully tested by writing a simple DSP algorithm (e.g., vector addition) using only the trait abstraction, then compiling with different feature flags (scalar, avx2, avx512, neon) and verifying that: (1) the same code compiles for all backends, (2) each backend produces identical results, (3) SIMD backends show expected performance improvements over scalar.

**Acceptance Scenarios**:

1. **Given** a DSP algorithm written using SIMD abstraction traits, **When** compiled with feature flag "scalar", **Then** code compiles successfully using only scalar operations with no SIMD instructions
2. **Given** the same DSP algorithm, **When** compiled with feature flag "avx2", **Then** code compiles to AVX2 instructions and executes 4-8x faster than scalar backend
3. **Given** DSP code using trait-based abstractions, **When** switching between backends, **Then** no changes to algorithm code are required (zero #[cfg] directives in user code)
4. **Given** multiple available backends, **When** user selects backend via cargo feature, **Then** exactly one backend implementation is compiled into the binary

---

### User Story 2 - Block Processing Pattern (Priority: P1)

As a DSP developer, I need a standardized block processing pattern with fixed block sizes (64 or 128 samples) and clear conventions for packing samples into SIMD lanes, so that I can process audio in SIMD-friendly chunks with predictable memory layouts and optimal cache utilization.

**Why this priority**: Real-time audio processing requires predictable, efficient memory access patterns. Block processing with fixed sizes enables the compiler to unroll loops and optimize SIMD operations. This is essential for achieving the performance targets across all backends.

**Independent Test**: Can be fully tested by implementing a block processor that processes audio in fixed-size blocks, verifying memory alignment, measuring cache efficiency, and confirming that SIMD lanes are packed according to documented conventions (e.g., interleaved stereo vs. planar layout).

**Acceptance Scenarios**:

1. **Given** an audio stream, **When** processed in 64-sample blocks, **Then** memory accesses are properly aligned for SIMD operations (32-byte alignment for AVX2, 64-byte for AVX512, 16-byte for NEON)
2. **Given** stereo audio, **When** packing samples into SIMD lanes, **Then** library provides clear documentation on whether samples are interleaved (LRLR) or planar (LLLL then RRRR)
3. **Given** block processing with fixed size, **When** compiler optimizations are enabled, **Then** inner loops are fully unrolled and vectorized as verified by assembly inspection
4. **Given** audio buffer of arbitrary length, **When** processing in fixed-size blocks, **Then** library provides clear pattern for handling remainder samples (< block size)

---

### User Story 3 - Core Vector Operations (Priority: P1)

As a DSP developer, I need fundamental vector operations (arithmetic, FMA, min/max, compare/masks, horizontal sum/max) accessible through the SIMD abstraction, so that I can build complex DSP algorithms from well-optimized primitives without dropping down to raw intrinsics.

**Why this priority**: These operations are the building blocks for all DSP algorithms. Without them, developers cannot actually use the SIMD abstraction for real work. Every DSP algorithm requires some combination of these primitives.

**Independent Test**: Can be fully tested by implementing test cases for each operation across all backends, verifying numerical correctness and measuring performance. Each operation should have property-based tests ensuring mathematical properties hold (e.g., commutativity, associativity where applicable).

**Acceptance Scenarios**:

1. **Given** two audio vectors, **When** performing vector addition, **Then** operation completes in O(N/lanes) time with correct element-wise results across all backends
2. **Given** three audio vectors A, B, C, **When** computing A * B + C using FMA, **Then** operation uses single fused instruction on supporting backends and achieves higher accuracy than separate multiply-add
3. **Given** a vector of samples, **When** computing horizontal sum, **Then** result equals sum of all elements and executes using optimal horizontal reduction pattern for each backend
4. **Given** two vectors compared with less-than operation, **When** result is used as mask, **Then** mask can be used in subsequent blend/select operations to implement conditional logic without branching

---

### User Story 4 - Fast Math Kernels (Priority: P2)

As a DSP developer, I need vectorized fast math kernels (tanh, exp, exp2, log, log2, log1p, sin, cos, inverse, atan, pow, sqrt, polynomial saturators, sigmoid curves, interpolation, polyBLEP, and noise generation) that work through the SIMD abstraction, so that I can apply complex mathematical transformations to audio blocks without sacrificing performance.

**Why this priority**: While the abstraction layer (P1) is foundational, these math kernels are what actually make it useful for audio DSP. They depend on the abstraction being in place first. These functions directly enable synthesizer features like oscillators, envelopes, waveshaping, modulation, and alias-free synthesis.

**Independent Test**: Can be fully tested by comparing vectorized math kernel outputs against reference scalar implementations, measuring error bounds, and benchmarking performance across all backends.

**Acceptance Scenarios**:

1. **Given** a block of 64 samples, **When** applying vectorized tanh, **Then** operation executes 8-16x faster than scalar tanh while maintaining error below 0.1%
2. **Given** envelope generation requiring exp, **When** using vectorized exp kernel, **Then** block processing achieves sub-nanosecond per-sample throughput on modern CPUs
3. **Given** oscillator phase calculations, **When** using vectorized sin/cos, **Then** harmonic distortion remains below -100dB across all backends
4. **Given** need for division-free reciprocal, **When** using fast inverse (1/x), **Then** accuracy is sufficient for audio applications (< 0.01% error) at 5-10x speedup vs division
5. **Given** phase modulation or waveshaping requiring atan, **When** using vectorized atan approximation, **Then** absolute error remains below 0.001 radians (< 0.057 degrees) across full input range while executing 8-16x faster than scalar libm atan
6. **Given** pitch shifting requiring pow(2, x), **When** using vectorized exp2/log2 with pow decomposition, **Then** operations execute 10-20x faster than scalar libm with < 0.01% error
7. **Given** waveshaping distortion, **When** using polynomial saturation curves, **Then** processing completes in < 5 CPU cycles per sample on AVX2
8. **Given** wavetable oscillator requiring interpolation, **When** using cubic Hermite interpolation kernel, **Then** phase continuity maintained with smooth C1 continuity
9. **Given** sawtooth/square wave synthesis, **When** using polyBLEP kernel, **Then** alias-free output achieved with < 8 operations per transition
10. **Given** need for audio noise, **When** using vectorized white noise generation, **Then** full 64-sample block generated in < 100 CPU cycles

---

### User Story 5 - Lookup Table Infrastructure (Priority: P2)

As a DSP developer, I need vectorized lookup table mechanisms with interpolation that work through the SIMD abstraction, so that I can implement wavetable oscillators and other tabular functions efficiently in block-processing contexts.

**Why this priority**: Lookup tables are critical for wavetable synthesis (Rigel's core functionality). While important, they depend on the block processing and vector operations from P1 stories being in place first.

**Independent Test**: Can be fully tested by creating sample wavetables, performing vectorized lookups with interpolation across entire blocks, and measuring both performance and interpolation quality through harmonic analysis.

**Acceptance Scenarios**:

1. **Given** a 2048-sample wavetable, **When** performing 64-sample block lookup with linear interpolation, **Then** entire block completes in under 640 nanoseconds (< 10ns per sample)
2. **Given** vectorized table lookup, **When** accessing values at arbitrary phases, **Then** interpolation maintains phase continuity across entire block with no discontinuities
3. **Given** SIMD-based lookup, **When** accessing different table positions per lane, **Then** gather operations or equivalent techniques provide correct per-lane indexing

---

### User Story 6 - Denormal Handling (Priority: P1)

As a DSP developer, I need automatic denormal protection integrated into the block processing pattern, so that filters and reverbs maintain consistent performance even when processing silence or very quiet signals.

**Why this priority**: Denormal numbers can cause catastrophic performance drops (10-100x slowdown) in real-time audio processing. This must be built into the foundation to ensure reliable real-time performance across all backends.

**Independent Test**: Can be fully tested by processing signals that decay into the denormal range and measuring whether CPU usage remains constant versus spiking. Verify that filters processing silence maintain normal performance across all backends.

**Acceptance Scenarios**:

1. **Given** a filter processing a signal that decays to silence, **When** internal state values enter denormal range (< 1e-38), **Then** CPU usage remains constant with no performance degradation across all backends
2. **Given** various denormal protection methods (FTZ/DAZ flags, DC offset, explicit checks), **When** applied to audio blocks, **Then** no audible artifacts are introduced (THD+N < -96dB)
3. **Given** block processing pattern, **When** denormal protection is enabled, **Then** protection is applied efficiently to entire blocks without per-sample overhead

---

### User Story 7 - Soft Saturation and Waveshaping (Priority: P3)

As a DSP developer, I need vectorized saturation curves (soft clipping, tube-style, tape-style) accessible through the SIMD abstraction, so that I can add harmonic richness and prevent digital clipping when processing audio blocks.

**Why this priority**: Saturation is essential for musical sound design and preventing harsh digital clipping. However, it can be implemented later as it depends on the vectorized math kernels from P2. This enhances the sonic palette but isn't required for basic functionality.

**Independent Test**: Can be fully tested by applying vectorized saturation functions to test signal blocks, analyzing harmonic content, and comparing performance against naive implementations.

**Acceptance Scenarios**:

1. **Given** an audio block approaching 0dBFS, **When** vectorized soft saturation is applied, **Then** output remains smooth without digital clipping and introduces harmonically-rich content
2. **Given** different saturation curve options, **When** applied to audio blocks via SIMD abstraction, **Then** each produces distinct harmonic characteristics while maintaining block processing performance
3. **Given** real-time processing requirements, **When** saturation is applied to 64-sample blocks, **Then** performance is at least 3x faster than per-sample polynomial-based implementations

---

### User Story 8 - Crossfade and Ramping Utilities (Priority: P3)

As a DSP developer, I need vectorized crossfade and parameter ramping functions that work on audio blocks, so that I can eliminate clicks and zipper noise when changing parameters or switching between signals.

**Why this priority**: Smooth parameter changes are important for professional audio quality but represent polish rather than core functionality. Can be implemented later once fundamental operations are working.

**Independent Test**: Can be fully tested by measuring crossfade curves for equal-power characteristics and verifying no audible clicks when switching parameters or signals, using block processing for efficiency.

**Acceptance Scenarios**:

1. **Given** two audio blocks to blend, **When** vectorized crossfade is applied, **Then** total energy remains constant (equal-power crossfade) with no perceived volume dip
2. **Given** a parameter change (e.g., filter cutoff), **When** ramping across block boundary, **Then** no audible clicks or zipper noise are introduced
3. **Given** different crossfade curve shapes, **When** applied via SIMD abstraction, **Then** block processing maintains optimal performance

---

### User Story 9 - Backend Selection and Benchmarking (Priority: P1)

As a DSP developer, I need the ability to compile and run the library with different SIMD backends and compare their performance through comprehensive benchmarks, so that I can validate performance claims and choose appropriate backends for target platforms.

**Why this priority**: Without the ability to test and benchmark different backends, we cannot verify that the abstraction layer is actually delivering performance benefits. This is essential for validating the entire architecture and making informed deployment decisions.

**Independent Test**: Can be fully tested by running the same benchmark suite with different cargo features (scalar, avx2, avx512, neon), comparing instruction counts (iai-callgrind) and wall-clock times (Criterion), and verifying expected performance scaling.

**Acceptance Scenarios**:

1. **Given** benchmark suite, **When** run with --features scalar, **Then** benchmarks compile and execute using only scalar operations
2. **Given** the same benchmarks, **When** run with --features avx2, **Then** performance shows 4-8x improvement over scalar for vector operations
3. **Given** multiple backends, **When** benchmarks are executed, **Then** results clearly show instruction counts, wall-clock times, and speedup ratios for each backend
4. **Given** CI environment, **When** running tests, **Then** all backends can be tested in parallel to verify correctness across entire backend matrix

---

### User Story 10 - Comprehensive Test Coverage for Correctness and Performance (Priority: P1)

As a library maintainer, I need comprehensive test coverage that ensures correctness and performance of all SIMD operations, math kernels, and backend implementations, so that I can confidently ship high-quality, bug-free code that meets accuracy and performance guarantees across all platforms.

**Why this priority**: Testing is not optional for safety-critical audio DSP code. Without comprehensive tests, we cannot guarantee that the abstraction is zero-cost, that backends produce correct results, or that performance targets are met. This is foundational infrastructure that must be built alongside the implementation, not added later.

**Independent Test**: Can be fully tested by running the complete test suite across all backends, verifying that property-based tests catch violations of mathematical invariants, accuracy tests confirm error bounds, and performance tests validate speed targets. Test coverage metrics should show >90% line coverage with >95% branch coverage for critical paths.

**Acceptance Scenarios**:

1. **Given** any SIMD operation, **When** property-based tests run with thousands of random inputs, **Then** mathematical invariants (commutativity, associativity, distributivity) hold across all backends
2. **Given** any math kernel (tanh, exp, sin, etc.), **When** accuracy tests compare against reference implementations, **Then** error remains within documented bounds (<0.1% for amplitude, <0.001% for frequency, <-100dB THD for oscillators)
3. **Given** all backends, **When** backend consistency tests run, **Then** all backends produce results within acceptable error bounds for identical inputs
4. **Given** critical operations, **When** unit tests run, **Then** edge cases (NaN, infinity, denormals, zero, extreme values) are handled gracefully without panics
5. **Given** any public function, **When** documentation tests run, **Then** all code examples in documentation compile and execute correctly
6. **Given** performance-critical operations, **When** regression tests run, **Then** performance has not degraded from baseline measurements (instruction counts within 5%, wall-clock within 10%)
7. **Given** complete test suite, **When** running with code coverage tools, **Then** critical paths achieve >95% branch coverage and overall codebase achieves >90% line coverage
8. **Given** CI pipeline, **When** tests run on pull requests, **Then** all tests pass across all backends (scalar, AVX2, AVX512, NEON) on their respective platforms

---

### Edge Cases

- What happens when math functions receive NaN or infinity inputs in vector operations? (Should return safe values or saturate per lane)
- How does system handle extreme input values (e.g., very large numbers to exp) in vectorized form? (Should clamp to safe ranges)
- What happens when lookup tables are accessed with out-of-bounds indices in SIMD gather operations? (Should wrap or clamp gracefully per lane)
- How does denormal handling interact with deliberate use of very small values in block processing? (Should preserve intentional quiet signals)
- What happens when SIMD width differs between backends (e.g., AVX2=256-bit vs AVX512=512-bit)? (Should handle via lane count abstraction)
- How are audio buffers smaller than the block size handled? (Should provide clear pattern for remainder processing)
- What happens when mixing backends in the same binary? (Should be prevented at compile time)
- How does code handle platforms without hardware support for selected backend? (Should fail at compile time, not runtime)

## Requirements *(mandatory)*

### Functional Requirements

#### SIMD Abstraction and Backends

- **FR-001**: Library MUST provide trait-based SIMD abstraction that allows DSP code to be written once and compiled to multiple backends
- **FR-002**: Library MUST support scalar backend that works on all platforms without SIMD instructions (always available fallback)
- **FR-003**: Library MUST support AVX2 backend for modern x86-64 processors (baseline for x86-64 in 2026)
- **FR-004**: Library MUST support AVX512 backend for bleeding-edge x86-64 processors
- **FR-005**: Library MUST support NEON backend for ARM64/Apple Silicon processors
- **FR-006**: Backend selection MUST occur at compile time via cargo features (e.g., --features avx2)
- **FR-007**: Only one backend MUST be active per compilation (no runtime dispatch)
- **FR-008**: DSP code using trait abstractions MUST NOT require #[cfg] directives for different backends
- **FR-009**: All backends MUST produce deterministic results within documented error bounds for the same input (bit-identical for exact operations like vector arithmetic; error-bounded for approximations like math kernels)

#### Block Processing

- **FR-010**: Library MUST provide block processing pattern with fixed block sizes of 64 or 128 samples
- **FR-011**: Library MUST document clear conventions for packing samples into SIMD lanes (interleaved vs planar)
- **FR-012**: Block processing MUST ensure proper memory alignment for each backend (32-byte for AVX2, 64-byte for AVX512, 16-byte for NEON)
- **FR-013**: Library MUST provide pattern for handling remainder samples when buffer size is not multiple of block size
- **FR-014**: Block processing MUST enable compiler loop unrolling and auto-vectorization

#### Vector Operations

- **FR-015**: Library MUST provide vector arithmetic operations: add, subtract, multiply, divide
- **FR-016**: Library MUST provide fused multiply-add (FMA) operation using hardware FMA where available
- **FR-017**: Library MUST provide min/max operations (element-wise minimum and maximum)
- **FR-018**: Library MUST provide comparison operations returning masks (less-than, greater-than, equal)
- **FR-019**: Library MUST provide mask-based select/blend operations for conditional logic without branching
- **FR-020**: Library MUST provide horizontal operations: sum (reduce all lanes), max (maximum across lanes)
- **FR-021**: All vector operations MUST maintain deterministic execution time (< 10% variance) regardless of input values

#### Fast Math Kernels

- **FR-022**: Library MUST provide vectorized tanh approximation executing 8-16x faster than scalar while maintaining error below 0.1%
- **FR-023**: Library MUST provide vectorized exp approximation with sub-nanosecond per-sample throughput (< 1ns per sample on 3GHz+ x86-64 CPU with AVX2)
- **FR-024**: Library MUST provide vectorized log1p accurate to within 0.001% for frequency calculations
- **FR-025**: Library MUST provide vectorized sin/cos approximations with harmonic distortion below -100dB
- **FR-026**: Library MUST provide vectorized fast inverse (1/x) with < 0.01% error at 5-10x speedup vs division
- **FR-027**: Library MUST provide vectorized atan approximation using Remez minimax polynomial with < 0.001 radian absolute error (< 0.057 degrees) executing 8-16x faster than scalar libm atan
- **FR-028**: Library MUST provide vectorized exp2 and log2 approximations using IEEE 754 exponent manipulation with polynomial refinement, achieving < 0.01% error at 10-20x speedup vs scalar libm
- **FR-029**: Library MUST provide vectorized pow and powf using exp2(log2(x)*y) decomposition with optimized polynomial approximations
- **FR-030**: Library MUST provide polynomial saturation curves (soft clip, hard clip, asymmetric) for waveshaping with < 5 CPU cycles per sample on AVX2
- **FR-031**: Library MUST provide sigmoid curves (logistic, tanh-based, smoothstep family) with polynomial approximations maintaining smooth C1 or C2 continuity
- **FR-032**: Library MUST provide polynomial interpolation kernels (linear, cubic Hermite, quintic) for audio resampling and wavetable synthesis
- **FR-033**: Library MUST provide polyBLEP (band-limited step) kernels using 2nd-order polynomial approximation for alias-free oscillator synthesis with < 8 operations per transition
- **FR-034**: Library MUST provide vectorized random noise generation (white, pink optional) using fast PRNG suitable for audio synthesis
- **FR-035**: Library MUST provide additional vectorized math functions typical for audio DSP (sqrt, rsqrt)
- **FR-036**: All math kernels MUST operate on entire blocks for maximum efficiency

#### Lookup Tables

- **FR-037**: Library MUST provide vectorized lookup table infrastructure supporting linear and cubic interpolation
- **FR-038**: Library MUST support per-lane indexing for SIMD table lookups (gather operations or equivalent)
- **FR-039**: Lookup tables MUST support configurable sizes optimized for cache efficiency
- **FR-040**: Library MUST provide clear documentation on size/quality/performance tradeoffs for table sizes

#### Denormal Handling

- **FR-041**: Library MUST implement denormal number protection that prevents CPU performance degradation
- **FR-042**: Denormal protection MUST apply to entire blocks without per-sample overhead
- **FR-043**: Denormal protection MUST NOT introduce audible artifacts (THD+N < -96dB)
- **FR-044**: Denormal handling MUST work consistently across all backends without platform-specific code in user algorithms

#### Additional Features

- **FR-045**: Library MUST provide vectorized equal-power crossfade functions
- **FR-046**: Library MUST provide vectorized parameter ramping utilities for click-free parameter changes
- **FR-047**: All functions MUST be no_std compatible (no heap allocations) for rigel-dsp integration
- **FR-048**: Library MUST include accuracy bounds and performance characteristics documentation for each function
- **FR-049**: Library MUST provide both accuracy-prioritized and speed-prioritized variants where meaningful tradeoffs exist

#### Testing and Benchmarking

- **FR-050**: Library MUST provide comprehensive benchmark suite that runs across all backends
- **FR-051**: Benchmarks MUST measure both instruction counts (iai-callgrind) and wall-clock times (Criterion)
- **FR-052**: Library MUST enable compiling and running tests for all backends via cargo features
- **FR-053**: Benchmark results MUST clearly show performance comparison between backends (speedup ratios)
- **FR-054**: CI MUST be able to test all backends in parallel for correctness verification

#### Comprehensive Test Coverage

- **FR-055**: Library MUST include property-based tests for all SIMD operations verifying mathematical invariants (commutativity, associativity, distributivity where applicable)
- **FR-056**: Library MUST include accuracy tests for all math kernels comparing against reference implementations with error bounds verification
- **FR-057**: Library MUST include backend consistency tests ensuring all backends produce results within acceptable error bounds for identical inputs
- **FR-058**: Library MUST include unit tests covering edge cases (NaN, infinity, denormals, zero, extreme values) for all public functions
- **FR-059**: Library MUST include documentation tests ensuring all code examples in API documentation compile and execute correctly
- **FR-060**: Library MUST include performance regression tests that detect degradation from baseline measurements (instruction counts, wall-clock times)
- **FR-061**: Test suite MUST achieve >90% line coverage overall with >95% branch coverage for critical paths (verified via code coverage tools)
- **FR-062**: Test suite MUST run in CI pipeline across all backends on appropriate platforms (x86-64 for scalar/AVX2/AVX512, ARM64 for NEON)
- **FR-063**: Property-based tests MUST generate thousands of test cases including normal values, denormals, boundary conditions, and edge cases
- **FR-064**: Library MUST include integration tests demonstrating complete DSP workflows (oscillators, filters, envelopes) using the abstraction
- **FR-065**: All test failures MUST provide clear error messages indicating which backend, operation, and input values caused the failure
- **FR-066**: Test suite MUST complete in under 5 minutes for standard runs, with optional extended test modes for exhaustive validation

### Key Entities

- **SIMD Backend**: Represents a specific instruction set implementation (scalar, AVX2, AVX512, NEON) selected at compile time via features
- **SIMD Vector Trait**: Abstract interface defining vector operations that all backends must implement
- **Block Processor**: Fixed-size audio buffer (64 or 128 samples) with proper alignment and packing conventions
- **Math Kernel**: Vectorized mathematical function (tanh, exp, exp2, log2, atan, pow, etc.) with implementation for each backend
- **Saturation Curve**: Polynomial-based waveshaping function (soft clip, hard clip, asymmetric) for harmonic distortion
- **Sigmoid Curve**: Smooth interpolation function (logistic, smoothstep family) with C1/C2 continuity for parameter transitions
- **Interpolation Kernel**: Polynomial interpolation function (linear, cubic Hermite, quintic) for resampling and wavetable lookup
- **PolyBLEP Kernel**: Band-limited step function using 2nd-order polynomial for alias-free oscillator synthesis
- **Noise Generator**: Vectorized pseudo-random number generator (PRNG) for white/pink noise synthesis
- **Lookup Table**: Pre-computed function values with vectorized interpolation supporting per-lane indexing
- **Crossfade Curve**: Vectorized parameter transition function with equal-power characteristics
- **Benchmark Suite**: Collection of performance tests measuring each backend's instruction counts and wall-clock performance
- **Property Test**: Automated test generating thousands of random inputs to verify mathematical invariants and edge case handling
- **Accuracy Test**: Comparison test validating math kernel error bounds against reference implementations
- **Backend Consistency Test**: Cross-backend validation ensuring all implementations produce results within acceptable error tolerances
- **Regression Test**: Performance baseline comparison detecting degradation in instruction counts or execution time
- **Integration Test**: End-to-end test demonstrating complete DSP workflows using the library's abstractions

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: DSP algorithms written using trait abstraction compile successfully to all supported backends (scalar, AVX2, AVX512, NEON) without code changes
- **SC-002**: Vectorized operations show expected performance scaling: AVX2 4-8x faster than scalar, AVX512 8-16x faster than scalar, NEON 4-8x faster than scalar
- **SC-003**: All backends produce deterministic results within documented error bounds for the same input (bit-identical for exact operations; error-bounded for approximations; verified through backend consistency tests T024, T049)
- **SC-004**: Developer writing DSP code using abstractions requires zero #[cfg] directives for backend selection
- **SC-005**: Block processing with fixed sizes (64 or 128 samples) achieves full loop unrolling and vectorization (verified via assembly inspection)
- **SC-006**: Denormal protection maintains consistent CPU usage (< 5% variance) when processing signals that decay into denormal range across all backends
- **SC-007**: Vectorized math kernels execute 8-16x faster than scalar equivalents while maintaining audio-quality accuracy (THD+N < -100dB or < 0.1% error)
- **SC-008**: Lookup table operations complete in under 10 nanoseconds per sample including interpolation across all backends
- **SC-009**: Horizontal operations (sum, max) execute using optimal reduction patterns for each backend architecture
- **SC-010**: FMA operations use single fused instruction on supporting backends (AVX2, AVX512, NEON) verified via instruction inspection
- **SC-011**: Benchmark suite enables comparing all backends with clear speedup ratios and instruction count comparisons
- **SC-012**: All functions complete without heap allocation as verified by rigel-dsp's no_std constraints
- **SC-013**: Comprehensive DSP algorithm (e.g., complete synthesizer voice) shows at least 40% reduction in CPU usage when using optimized backends vs scalar
- **SC-014**: Documentation enables developers to understand backend selection, block processing patterns, and SIMD lane packing conventions within 10 minutes
- **SC-015**: Test suite achieves >90% line coverage and >95% branch coverage for critical code paths as measured by code coverage tools
- **SC-016**: Property-based tests generate at least 10,000 test cases per operation, catching edge cases that would be missed by hand-written tests
- **SC-017**: All math kernels pass accuracy tests showing error bounds within specified limits across all backends (verified through automated testing)
- **SC-018**: Vectorized atan approximation maintains absolute error below 0.001 radians (< 0.057 degrees) across full input range while executing 8-16x faster than scalar libm atan
- **SC-019**: Vectorized exp2/log2 approximations achieve < 0.01% error while executing 10-20x faster than scalar libm, enabling efficient pow(x,y) decomposition
- **SC-020**: Polynomial saturation curves process audio in < 5 CPU cycles per sample on AVX2, enabling real-time waveshaping distortion
- **SC-021**: Sigmoid curves (logistic, smoothstep) maintain smooth C1 or C2 continuity for artifact-free parameter transitions
- **SC-022**: Cubic Hermite interpolation maintains phase continuity and smooth derivatives for wavetable synthesis and resampling
- **SC-023**: PolyBLEP kernels achieve alias-free oscillator output with < 8 operations per transition, matching quality of 32-sample minBLEP with fraction of overhead
- **SC-024**: Vectorized white noise generation produces full 64-sample block in < 100 CPU cycles with statistical distribution passing chi-square test
- **SC-025**: Backend consistency tests confirm that all backends produce results within acceptable error tolerances (bit-identical or within documented error bounds)
- **SC-026**: Test suite runs in under 5 minutes in standard mode, enabling rapid development iteration
- **SC-027**: CI pipeline successfully runs all tests across all backends on every pull request, catching regressions before merge
- **SC-028**: Performance regression tests detect any degradation >5% in instruction counts or >10% in wall-clock time compared to baseline
- **SC-029**: Integration tests demonstrate complete real-world DSP workflows (oscillators, filters, envelopes) execute correctly using library abstractions
- **SC-030**: All code examples in API documentation compile and execute successfully as verified by documentation tests
- **SC-031**: Test failures provide actionable error messages including backend name, operation, input values, expected result, and actual result

## Assumptions

- Target x86-64 platforms have AVX2 support (2013+ CPUs, ubiquitous by 2026) as baseline; no need to support older SSE-only systems
- Target ARM64 platforms have NEON support (standard on all ARM64 processors)
- Developers using this library understand fundamental DSP concepts and can interpret error bounds and harmonic distortion measurements
- Primary use case is real-time audio synthesis/processing at sample rates from 44.1kHz to 192kHz
- Audio quality standards align with professional music production (24-bit/96kHz as reference quality)
- The library will be integrated into rigel-dsp which already enforces no_std and no-allocation constraints
- Performance measurements will use CPU instruction counting (iai-callgrind) and wall-clock timing (Criterion) as established in Rigel's benchmarking infrastructure
- Backend selection happens at compile time via cargo features; runtime CPU detection is out of scope
- Block sizes (64 or 128 samples) align with common audio buffer sizes and cache line sizes
- SIMD lane counts vary by backend: AVX2 = 8 floats (256-bit), AVX512 = 16 floats (512-bit), NEON = 4 floats (128-bit), scalar = 1 float
- Trait-based abstraction overhead is zero-cost (verified via assembly inspection showing identical codegen to hand-written intrinsics)
- Developers are willing to accept compile-time backend selection rather than runtime dispatch in exchange for zero-cost abstraction
- CI environment supports testing multiple backends (may require different runners or Docker containers for platform-specific instruction sets)
- Property-based testing framework (proptest) is acceptable dependency for generating comprehensive test inputs
- Code coverage tools (tarpaulin, llvm-cov) are available in development and CI environments
- Test suite execution time under 5 minutes is achievable with parallelized test execution
- Developers accept that comprehensive testing is foundational infrastructure, not optional polish
- Test coverage metrics (>90% line, >95% branch for critical paths) are measurable and enforceable via CI
- Property-based tests generating 10,000+ cases per operation is computationally feasible in CI environment
- Reference implementations (libm) are acceptable for accuracy comparison in tests but not in production code
- Atan approximation will use Remez minimax polynomial (odd-polynomial on restricted domain) as research shows this provides optimal error bounds for given computational cost
- Atan error tolerance of 0.001 radians (0.057 degrees) is sufficient for audio DSP applications including phase modulation and waveshaping
- exp2/log2 will use IEEE 754 exponent field manipulation combined with polynomial approximation of fractional part, as this provides 10-20x speedup with audio-sufficient accuracy
- pow(x,y) will decompose to exp2(log2(x)*y) rather than using dedicated pow approximation, as research shows this provides better performance/accuracy balance
- Polynomial saturation curves will use Chebyshev or direct polynomial forms depending on desired harmonic characteristics, optimized for < 5 cycles/sample
- Sigmoid curves will use polynomial approximations (e.g., smoothstep: 3x²-2x³) rather than expensive exp/tanh for C1/C2 continuity with minimal overhead
- Cubic Hermite interpolation chosen over higher-order (quintic) for audio resampling as it provides best balance of smoothness and computational cost
- PolyBLEP will use 2nd-order polynomial approximation requiring only 8 operations per transition, matching 32-sample minBLEP quality with fraction of cost
- White noise generation will use fast xorshift-based PRNG rather than cryptographic-quality RNG, as audio synthesis doesn't require cryptographic properties
- Pink noise generation (if included) will use Paul Kellet's economical algorithm or Voss-McCartney algorithm for 1/f spectral characteristic

## Non-Functional Requirements

- **Performance**: All operations must maintain deterministic execution time (< 10% variance) regardless of input values
- **Portability**: Library must compile and run correctly on macOS (x86-64 + ARM64), Linux (x86-64), and Windows (x86-64)
- **Zero-Cost Abstraction**: Trait-based abstraction must compile to identical code as hand-written intrinsics (verified via assembly inspection)
- **Safety**: All operations must handle edge cases (NaN, infinity, out-of-bounds) gracefully without panicking or undefined behavior
- **Maintainability**: Each backend implementation must be isolated in separate modules with shared test suite ensuring consistency
- **Testing**: All operations must have property-based tests verifying mathematical properties across all backends
- **Documentation**: Each function must document accuracy bounds, performance characteristics, and appropriate use cases
- **Compile Times**: Adding SIMD abstraction must not significantly increase compile times (< 20% increase acceptable)
- **Test Quality**: Test suite must achieve >90% line coverage overall with >95% branch coverage for critical paths
- **Test Performance**: Standard test suite must complete in under 5 minutes to enable rapid development iteration
- **Continuous Integration**: All tests must run in CI on every pull request across all supported backends and platforms
- **Regression Detection**: Performance tests must detect >5% degradation in instruction counts or >10% degradation in wall-clock time
- **Test Isolation**: Tests must be independent and parallelizable with no shared mutable state
- **Deterministic Testing**: Property-based tests must be reproducible via seed values for debugging failures
- **Build Infrastructure**: CI must build for all three platforms (macOS ARM64, Linux x86-64, Windows x86-64) on every pull request
- **Reproducible Builds**: All builds (local and CI) must run through devenv shell for consistent, reproducible environments
- **Cross-Compilation Strategy**: Windows builds may use cross-compilation where it provides value without excessive complexity; macOS↔Linux cross-compilation is not supported due to GUI dependency complexity
- **Local Development Performance**: Native builds must be prioritized for fast iteration during local development

## Dependencies

- Rigel's existing benchmark infrastructure (Criterion, iai-callgrind)
- Rigel's existing DSP testing framework for audio quality validation
- Platform SIMD intrinsics: std::arch for x86/x86-64 (AVX2, AVX512), std::arch::aarch64 for ARM64 (NEON)
- libm or equivalent for reference implementations in tests (not in production code path)
- Cargo features for compile-time backend selection
- CI infrastructure capable of testing multiple backends (may need platform-specific runners)
- proptest for property-based testing framework generating comprehensive test inputs
- Code coverage tools (tarpaulin or llvm-cov) for measuring test coverage metrics
- Test harness supporting parallel test execution for fast test suite completion
- CI runners with access to different instruction sets (x86-64 for AVX2/AVX512, ARM64 for NEON)
- Baseline measurement storage for performance regression detection (file-based or CI artifacts)

## Scope Boundaries

### In Scope
- Trait-based SIMD abstraction layer with multiple backend implementations
- Block processing pattern with fixed sizes (64 or 128 samples)
- Core vector operations (arithmetic, FMA, min/max, compare, horizontal ops)
- Fast vectorized math kernels: tanh, exp, exp2, log, log2, log1p, sin, cos, inverse, atan, pow, sqrt, rsqrt
- Polynomial saturation curves: soft clip, hard clip, asymmetric for waveshaping
- Sigmoid curves: logistic, tanh-based, smoothstep family (C1/C2 continuity)
- Polynomial interpolation kernels: linear, cubic Hermite, quintic for resampling and wavetable synthesis
- PolyBLEP (band-limited step) kernels for alias-free oscillator synthesis
- Vectorized random noise generation: white noise (pink noise optional)
- Vectorized lookup table infrastructure with interpolation
- Denormal number protection integrated into block processing
- Vectorized crossfade and parameter ramping utilities
- Comprehensive benchmarking infrastructure for backend comparison
- SIMD backends: scalar (always available), AVX2 (x86-64 baseline), AVX512 (x86-64 bleeding edge), NEON (ARM64)
- Documentation for backend selection, block processing patterns, and performance characteristics
- Comprehensive test suite including property-based, accuracy, backend consistency, regression, and integration tests
- Code coverage measurement and enforcement (>90% line, >95% branch for critical paths)
- CI pipeline testing all backends across all platforms
- Performance regression detection and baseline tracking
- Test infrastructure for edge case handling (NaN, infinity, denormals, extreme values)

### Out of Scope
- Runtime CPU feature detection and dynamic backend selection (compile-time only)
- Support for legacy x86-64 CPUs without AVX2 (pre-2013 processors)
- Backends for deprecated instruction sets (SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, AVX without AVX2)
- Automatic parallelization across CPU cores (single-threaded per DSP voice)
- Complex mathematical operations not commonly used in audio DSP (e.g., Bessel functions, gamma functions)
- Arbitrary-precision arithmetic
- Platform-specific assembly optimization beyond SIMD intrinsics
- Automatic differentiation or symbolic mathematics
- Integration with specific DAW APIs or plugin formats (that's rigel-plugin's domain)
- Variable block sizes or adaptive block sizing (fixed sizes only)
- Mixed backend execution within same binary (one backend per compilation)
- SIMD width polymorphism at runtime (SIMD width is compile-time constant per backend)
