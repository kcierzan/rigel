# Implementation Plan: SY-Style Envelope Modulation Source

**Branch**: `005-sy-envelope` | **Date**: 2026-01-10 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/005-sy-envelope/spec.md`
**Linear Issue**: [NEW-5](https://linear.app/new-atlantis/issue/NEW-5/create-sy-style-envelope-modulation-source)

## Summary

Implement a Yamaha SY99-style multi-segment envelope generator that operates in the logarithmic (dB) domain internally, outputting linear amplitude values (0.0 to 1.0). The envelope mimics MSFA behavior including nonlinear rate calculations, distance-dependent timing, and instantaneous dB jumps during attack phases. Implementation uses compile-time const generics for variant support (6+2, 7, 5+5 segments) and SIMD acceleration via rigel-math for batch processing of multiple envelopes.

## Technical Context

**Language/Version**: Rust 2021 edition (from workspace `rust-toolchain.toml`)
**Primary Dependencies**:
- `rigel-timing` - Timebase for sample-accurate timing, ControlRateClock
- `rigel-math` - SIMD-optimized math (exp, pow, log) for dB↔linear conversion
- `rigel-simd` - Block processing, denormal protection
- `libm` - Fallback for precision-critical scalar math only

**Storage**: N/A (pure computational library, no persistence)
**Testing**:
- `cargo test` - Unit and integration tests
- Criterion benchmarks for performance validation
- iai-callgrind for instruction count regression detection

**Target Platform**: All (aarch64-apple-darwin, x86_64-unknown-linux-gnu, x86_64-pc-windows-msvc)
**Project Type**: Single library crate within rigel-synth workspace
**Performance Goals**:
- 1536 envelopes × 64 samples in <100µs (SC-001)
- Single envelope: <50ns/sample (SC-002)
- SIMD batch processing: 2x+ speedup over scalar (SC-008)

**Constraints**:
- no_std compatible, zero heap allocations
- Copy/Clone for efficient voice management
- Memory footprint <128 bytes per envelope (SC-003)
- Real-time safe (no blocking, deterministic)

**Internal Format**: i16/Q8 fixed-point (hardware-authentic)
- 12-bit envelope level in Q8 format (256 steps = 6dB, ~96dB range)
- Matches original DX7 EGS→OPS chip representation
- Conversion to linear amplitude via `rigel_math::fast_exp2`
- 33% smaller than Q24/i32, better L1 cache utilization

**Scale/Scope**: 12 envelopes per voice × 32 voices × 4 unison = 1536 concurrent envelopes

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### I. Real-Time Safety (NON-NEGOTIABLE) ✅ COMPLIANT

- **No heap allocations**: Envelope uses fixed-size arrays with const generics (no Vec/Box)
- **No blocking I/O**: Pure computational module, no file/network operations
- **No std library**: Uses libm for math; rigel-math for optimized approximations
- **Deterministic performance**: Fixed segment counts, no branching on data values

### II. Layered Architecture ✅ COMPLIANT

- Envelope module resides in `rigel-modulation` crate (modulation sources layer)
- No dependencies on CLI or plugin layers
- Exposes ModulationSource trait for integration with higher layers

### III. Test-Driven Validation ✅ WILL COMPLY

- Unit tests for MSFA rate calculations, segment transitions, retrigger behavior
- Property-based tests for edge cases (rate=0, rate=99, boundary conditions)
- Benchmark validation against MSFA reference output (0.1dB tolerance)
- Architecture-specific SIMD tests (AVX2, NEON)

### IV. Performance Accountability ✅ WILL COMPLY

- Criterion benchmarks for single-envelope and batch processing
- iai-callgrind for instruction count stability
- **Math Operations**: Will use rigel-math for dB↔linear conversion (exp, pow, log)
- Baseline comparison before/after SIMD optimization

### V. Reproducible Environments ✅ COMPLIANT

- All development in devenv shell
- CI runs through devenv for reproducibility

### VI. Cross-Platform Commitment ✅ WILL COMPLY

- Platform-agnostic Rust code
- SIMD via rigel-math abstraction (auto-selects AVX2/NEON/scalar)
- Tests run on all CI platforms

### VII. DSP Correctness Properties ✅ WILL COMPLY

- Linear-in-dB transitions produce exponential amplitude curves (click-free)
- Rate scaling follows MSFA lookup table (authentic behavior)
- Retrigger from current level (no discontinuities)

## Project Structure

### Documentation (this feature)

```text
specs/005-sy-envelope/
├── plan.md              # This file
├── research.md          # Phase 0: MSFA algorithm research
├── data-model.md        # Phase 1: Entity definitions
├── quickstart.md        # Phase 1: Usage examples
├── contracts/           # Phase 1: API contracts
│   └── envelope-api.rs  # Public interface definitions
└── tasks.md             # Phase 2 output (/speckit.tasks command)
```

### Source Code (repository root)

```text
projects/rigel-synth/crates/modulation/
├── src/
│   ├── lib.rs               # Crate root (add envelope re-exports)
│   ├── traits.rs            # ModulationSource trait (existing)
│   ├── lfo.rs               # LFO implementation (existing)
│   ├── rate.rs              # Rate modes (existing)
│   ├── waveshape.rs         # LFO waveshapes (existing)
│   ├── simd_rng.rs          # SIMD RNG (existing)
│   └── envelope/            # NEW: Envelope module
│       ├── mod.rs           # Module root, re-exports
│       ├── segment.rs       # Segment type and transitions
│       ├── state.rs         # EnvelopeState runtime state
│       ├── config.rs        # EnvelopeConfig immutable params
│       ├── rates.rs         # MSFA rate table and calculations
│       └── batch.rs         # SIMD batch processing
├── tests/
│   ├── lfo_tests.rs         # Existing LFO tests
│   └── envelope_tests.rs    # NEW: Envelope integration tests
└── benches/
    ├── lfo_bench.rs         # Existing LFO benchmarks
    └── envelope_bench.rs    # NEW: Envelope benchmarks
```

**Structure Decision**: The envelope module is placed in `rigel-modulation` because:
1. It implements the `ModulationSource` trait (same as LFO)
2. Follows existing pattern where LFO lives alongside other modulation sources
3. Both LFO and envelope share similar dependencies (rigel-timing, rigel-math)
4. Maintains separation between modulation sources and audio signal processing (rigel-dsp)

## Complexity Tracking

> No constitution violations identified. All requirements align with principles.

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| None | N/A | N/A |
