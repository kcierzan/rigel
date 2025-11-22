# Implementation Plan: Fast DSP Math Library

**Branch**: `001-fast-dsp-math` | **Date**: 2025-11-18 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-fast-dsp-math/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Create a trait-based SIMD abstraction layer for Rigel's DSP core that provides zero-cost, compile-time backend selection (scalar, AVX2, AVX512, NEON) with block processing patterns (64/128 samples), core vector operations (arithmetic, FMA, min/max, compare, horizontal ops), and vectorized fast math kernels (tanh, exp, log1p, sin/cos, inverse, sqrt, pow). This will be implemented as a new `rigel-math` crate within the rigel-synth workspace, enabling DSP developers to write algorithms once and compile to optimal SIMD instructions for each platform without #[cfg] soup.

## Technical Context

**Language/Version**: Rust 2021 edition (workspace toolchain from rust-toolchain.toml)
**Primary Dependencies**:
  - `std::arch` for SIMD intrinsics (AVX2, AVX512, NEON)
  - `libm` for reference implementations in tests only (NOT in production code)
  - `criterion` for wall-clock benchmarking
  - `iai-callgrind` for deterministic instruction count benchmarking
  - `proptest` for property-based testing (chosen for superior SIMD edge case handling - see research.md)
**Storage**: N/A (pure computational library, no persistence)
**Testing**: `cargo test` with property-based tests for mathematical invariants, benchmark suite for performance validation
**Target Platform**: macOS (ARM64 native), Linux (x86-64 native), Windows (x86-64 cross-compiled via xwin)
**Project Type**: Library crate within rigel-synth monorepo workspace
**Performance Goals**:
  - Vectorized operations: 4-8x speedup (AVX2), 8-16x speedup (AVX512), 4-8x speedup (NEON) vs scalar
  - Math kernels: Sub-nanosecond per-sample throughput for exp/tanh/etc
  - Block processing: <10ns per sample for lookup table interpolation
  - Zero-cost abstraction: Trait calls compile to identical assembly as raw intrinsics
**Constraints**:
  - MUST be no_std compatible (no heap allocations, no std library)
  - MUST maintain deterministic execution time (<10% variance)
  - MUST produce bit-identical results across all backends
  - MUST support compile-time backend selection only (no runtime dispatch)
  - MUST integrate into rigel-dsp's real-time safety constraints
**Scale/Scope**:
  - ~4 SIMD backends (scalar, AVX2, AVX512, NEON)
  - ~20 vector operations (arithmetic, FMA, min/max, compare, horizontal)
  - 15 fast math kernels (tanh, exp, log, log1p, sin, cos, inverse, atan, exp2, log2, pow, sqrt, rsqrt, polynomial saturation, sigmoid curves, interpolation, polyBLEP, white noise)
  - Block sizes: 64 or 128 samples
  - Comprehensive benchmark suite covering all operations across all backends

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### I. Real-Time Safety (NON-NEGOTIABLE)
**Status**: ✅ PASS
- No heap allocations: Enforced by no_std requirement
- No blocking I/O: Pure computational library, no I/O operations
- No std library: Explicitly no_std compatible
- Deterministic performance: All operations target <10% variance in execution time
- All dependencies support no_std: std::arch (intrinsics), libm (test-only)

### II. Layered Architecture
**Status**: ✅ PASS
- This creates `rigel-math` as a new library crate within rigel-synth workspace
- Sits at the DSP layer, will be used by rigel-dsp for optimized math operations
- Respects layer boundaries: pure DSP algorithms with no CLI, GUI, or plugin concerns
- Enables independent testing via cargo test and benchmarks

### III. Test-Driven Validation
**Status**: ✅ PASS
- Unit tests: Embedded in crate modules for each operation
- Property-based tests: Verifying mathematical invariants (commutativity, associativity, error bounds)
- Integration tests: Cross-backend consistency testing (bit-identical results)
- Benchmark suite: Both Criterion (wall-clock) and iai-callgrind (instruction counts)
- No audio fixtures needed (pure math library, not audio processing yet)

### IV. Performance Accountability
**Status**: ✅ PASS
- Explicit performance targets: 4-16x speedups across SIMD backends
- Benchmark suite required: Criterion + iai-callgrind for all operations
- Baseline comparisons: Will use bench:baseline before changes
- Performance validation: bench:all to verify targets
- Deterministic benchmarking: iai-callgrind provides instruction-count regression detection

### V. Reproducible Environments
**Status**: ✅ PASS
- Development occurs in root devenv shell (Rust environment)
- CI commands run through devenv shell
- Uses existing workspace rust-toolchain.toml
- Cross-compilation via devenv (build:macos, build:linux, build:win)

### VI. Cross-Platform Commitment
**Status**: ✅ PASS
- Targets: aarch64-apple-darwin (NEON), x86_64-unknown-linux-gnu (AVX2/AVX512), x86_64-pc-windows-msvc (AVX2/AVX512)
- CI validates all platforms: Backend matrix testing across scalar/AVX2/AVX512/NEON
- Platform-agnostic APIs: Trait abstraction hides platform differences from users
- Bit-identical results enforced across all backends and platforms

### VII. DSP Correctness Properties
**Status**: ⚠️  PARTIAL (Not Applicable Yet)
- This library provides math primitives, not wavetable processing
- Wavetable-specific properties (zero-crossing alignment, RMS consistency, etc.) are the responsibility of code that uses this library
- Math kernel correctness enforced via:
  - Error bounds: <0.1% for amplitude functions, <0.001% for frequency functions
  - Harmonic distortion: <-100dB for sin/cos
  - Property-based tests for mathematical invariants
- When integrated into wavetable processing, those DSP properties will be validated at that layer

**Overall Gate Status**: ✅ PASS - All applicable principles satisfied, ready for Phase 0 research

## Project Structure

### Documentation (this feature)

```text
specs/[###-feature]/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

This creates a new library crate within the existing rigel-synth workspace:

```text
projects/rigel-synth/crates/
├── math/                          # NEW: rigel-math crate (this feature)
│   ├── Cargo.toml                # Crate manifest with feature flags for backends
│   ├── src/
│   │   ├── lib.rs                # Public API and re-exports
│   │   ├── traits.rs             # Core SIMD abstraction traits (SimdVector, etc.)
│   │   ├── block.rs              # Block processing types and utilities
│   │   ├── backends/             # Backend implementations
│   │   │   ├── mod.rs            # Backend selection via cfg
│   │   │   ├── scalar.rs         # Scalar backend (always available)
│   │   │   ├── avx2.rs           # AVX2 backend (x86-64)
│   │   │   ├── avx512.rs         # AVX512 backend (x86-64)
│   │   │   └── neon.rs           # NEON backend (ARM64)
│   │   ├── ops/                  # Vector operations
│   │   │   ├── mod.rs
│   │   │   ├── arithmetic.rs     # add, sub, mul, div
│   │   │   ├── fma.rs            # Fused multiply-add
│   │   │   ├── minmax.rs         # min, max operations
│   │   │   ├── compare.rs        # Comparison and masks
│   │   │   └── horizontal.rs     # Horizontal sum, max
│   │   ├── math/                 # Fast math kernels
│   │   │   ├── mod.rs
│   │   │   ├── tanh.rs           # Hyperbolic tangent approximation
│   │   │   ├── exp.rs            # Exponential approximation
│   │   │   ├── exp2_log2.rs      # exp2/log2 via IEEE 754 exponent manipulation
│   │   │   ├── log.rs            # Logarithm approximations (log1p, etc.)
│   │   │   ├── trig.rs           # sin, cos approximations
│   │   │   ├── inverse.rs        # Fast 1/x reciprocal
│   │   │   ├── atan.rs           # Arctangent approximation (Remez minimax)
│   │   │   ├── sqrt.rs           # Square root
│   │   │   └── pow.rs            # Power functions (via exp2/log2 decomposition)
│   │   ├── table.rs              # Lookup table infrastructure
│   │   ├── denormal.rs           # Denormal number protection
│   │   ├── saturate.rs           # Polynomial saturation curves (soft/hard clip, asymmetric)
│   │   ├── sigmoid.rs            # Sigmoid curves (logistic, smoothstep, C1/C2 continuity)
│   │   ├── interpolate.rs        # Polynomial interpolation (linear, cubic Hermite, quintic)
│   │   ├── polyblep.rs           # PolyBLEP (band-limited step) for alias-free oscillators
│   │   ├── noise.rs              # Vectorized random noise generation (white, pink)
│   │   └── crossfade.rs          # Crossfade and ramping utilities
│   ├── benches/                  # Benchmark suite
│   │   ├── criterion_benches.rs  # Criterion wall-clock benchmarks
│   │   └── iai_benches.rs        # iai-callgrind instruction counts
│   └── tests/                    # Integration tests
│       ├── backend_consistency.rs # Verify bit-identical results
│       ├── properties.rs         # Property-based tests
│       └── accuracy.rs           # Math kernel accuracy validation
├── dsp/                          # Existing rigel-dsp (will use rigel-math)
├── cli/                          # Existing rigel-cli
├── plugin/                       # Existing rigel-plugin
└── xtask/                        # Existing rigel-xtask

projects/rigel-synth/Cargo.toml   # Workspace manifest (add math/ member)
```

**Structure Decision**:
- New `rigel-math` library crate at `projects/rigel-synth/crates/math/`
- Integrates into existing workspace alongside dsp, cli, plugin, xtask
- Backend implementations isolated in separate modules under `backends/`
- Operations organized by category: ops/ (vector primitives), math/ (fast kernels)
- Comprehensive test coverage: unit tests in modules, integration tests in tests/, benchmarks in benches/
- Feature flags in Cargo.toml control backend selection: `scalar` (default), `avx2`, `avx512`, `neon`

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

No violations detected. All constitution principles are satisfied.

---

## Post-Design Constitution Re-Evaluation

*Re-checked after Phase 1 design completion*

### I. Real-Time Safety (NON-NEGOTIABLE)
**Status**: ✅ PASS - CONFIRMED
- API design enforces no allocations (stack-based AudioBlock, const-sized LookupTable)
- All operations are `#[inline(always)]` preventing function call overhead
- Deterministic execution via SIMD operations with bounded worst-case timing
- Zero runtime dispatch (compile-time backend selection via features)

### II. Layered Architecture
**Status**: ✅ PASS - CONFIRMED
- Clean library crate structure at `projects/rigel-synth/crates/math/`
- No dependencies on CLI, plugin, or GUI layers
- Will be consumed by rigel-dsp via simple dependency declaration
- Independent testing via cargo test, benchmarks isolated in benches/

### III. Test-Driven Validation
**Status**: ✅ PASS - CONFIRMED
- Property-based testing designed into API contracts (proptest for invariants)
- Backend consistency tests specified (bit-identical or error-bounded results)
- Accuracy validation tests for each math kernel
- Comprehensive benchmark suite (Criterion + iai-callgrind) in design

### IV. Performance Accountability
**Status**: ✅ PASS - CONFIRMED
- Explicit performance targets: 4-16x speedups documented
- Benchmark infrastructure specified in design (criterion_benches.rs, iai_benches.rs)
- Assembly inspection verification planned for zero-cost abstraction
- Error bounds documented for all math kernels

### V. Reproducible Environments
**Status**: ✅ PASS - CONFIRMED
- Uses existing workspace devenv shell
- No new environment requirements
- Cross-compilation via existing devenv infrastructure
- CI integration via existing backend matrix pattern

### VI. Cross-Platform Commitment
**Status**: ✅ PASS - CONFIRMED
- Backend design covers all targets: AVX2/AVX512 (x86-64), NEON (ARM64), scalar (fallback)
- Platform-agnostic trait API hides backend differences
- Compile-time feature selection ensures single backend per build
- CI matrix testing planned across all backends

### VII. DSP Correctness Properties
**Status**: ✅ PASS - CONFIRMED
- Math kernel error bounds explicitly documented (tanh <0.1%, sin/cos <-100dB THD)
- Property-based tests will verify error bounds across all backends
- Denormal protection designed to prevent artifacts (THD+N < -96dB)
- Lookup table interpolation preserves phase continuity

**Overall Post-Design Status**: ✅ ALL GATES PASS - Design satisfies all constitution principles, ready for Phase 2 (task generation)
