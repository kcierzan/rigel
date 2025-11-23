# Implementation Plan: Runtime SIMD Dispatch

**Branch**: `001-runtime-simd-dispatch` | **Date**: 2025-11-22 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/001-runtime-simd-dispatch/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Transform Rigel's SIMD backend selection from compile-time to runtime dispatch for x86_64 platforms (Linux/Windows), while maintaining compile-time NEON selection for macOS. This enables a single binary per platform that automatically selects the optimal SIMD backend (scalar → AVX2 → AVX-512) based on CPU capabilities, eliminating the need for users to understand their CPU architecture while preserving deterministic testing through build-time forcing flags.

## Technical Context

**Language/Version**: Rust 2021 edition (from rust-toolchain.toml)
**Primary Dependencies**: cpufeatures (no_std CPU feature detection for x86_64), rigel-math (trait-based SIMD abstraction library)
**Storage**: N/A (pure computational library, no persistence)
**Testing**: cargo test, architecture-specific tests (AVX2/AVX-512 on x86_64, NEON on aarch64), benchmarks (Criterion + iai-callgrind)
**Target Platform**: macOS (aarch64-apple-darwin), Linux (x86_64-unknown-linux-gnu), Windows (x86_64-pc-windows-msvc)
**Project Type**: Multi-crate workspace (rigel-math + rigel-dsp)
**Performance Goals**: Runtime dispatch overhead <1% vs compile-time SIMD, single voice CPU usage ~0.1% at 44.1kHz, full polyphonic <1% CPU
**Constraints**: no_std compliance in rigel-dsp/rigel-math, zero heap allocations, deterministic performance, no blocking I/O
**Scale/Scope**: Two-layer architecture: rigel-math provides trait-based SIMD backends, rigel-dsp adds runtime dispatch wrapper
**Architecture**: Layered design separating SIMD abstraction (rigel-math) from runtime dispatch (rigel-dsp)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### I. Real-Time Safety (NON-NEGOTIABLE)
- ✅ **PASS**: Feature maintains no_std compliance in rigel-dsp/rigel-math
- ✅ **PASS**: Runtime CPU detection must use no_std compatible library
- ✅ **PASS**: Function pointer dispatch adds negligible overhead (<1% per success criteria)
- ✅ **PASS**: No heap allocations, blocking I/O, or non-deterministic operations introduced
- ⚠️ **VERIFY IN RESEARCH**: Confirm chosen CPU detection library is no_std compatible and allocation-free

### II. Layered Architecture
- ✅ **PASS**: Changes confined to rigel-dsp layer (DSP core)
- ✅ **PASS**: CLI and plugin layers unaffected by dispatch mechanism
- ✅ **PASS**: Backend abstraction maintains clean separation between SIMD implementations

### III. Test-Driven Validation
- ✅ **PASS**: Forced-backend flags enable deterministic testing of scalar, AVX2, AVX-512 backends
- ✅ **PASS**: CI will test scalar and AVX2 backends deterministically
- ✅ **PASS**: Architecture-specific tests will validate AVX2/AVX-512 on x86_64, NEON on aarch64
- ✅ **PASS**: Runtime dispatch mode will be tested in CI to verify correct backend selection
- ⚠️ **NOTE**: AVX-512 testing is experimental (local-only, not CI)

### IV. Performance Accountability
- ✅ **PASS**: Success criteria SC-002 requires <1% runtime dispatch overhead
- ✅ **PASS**: Benchmarking (Criterion + iai-callgrind) will validate performance claims
- ✅ **PASS**: Single voice and polyphonic CPU usage targets maintained
- 📋 **ACTION REQUIRED**: Save baseline before implementation, measure overhead after

### V. Reproducible Environments
- ✅ **PASS**: All development occurs in devenv shell
- ✅ **PASS**: CI runs through devenv shell
- ✅ **PASS**: Build system modifications will integrate with existing devenv scripts

### VI. Cross-Platform Commitment
- ✅ **PASS**: macOS (aarch64, compile-time NEON), Linux (x86_64, runtime dispatch), Windows (x86_64, runtime dispatch)
- ✅ **PASS**: CI validates all platforms
- ✅ **PASS**: Platform-specific logic clearly separated (macOS vs x86_64)

### VII. DSP Correctness Properties
- ✅ **PASS**: SIMD backend selection does not affect DSP algorithm correctness
- ✅ **PASS**: All backends implement identical functional interface
- ✅ **PASS**: Wavetable DSP properties preserved across all backends

**Overall Status**: ✅ PASS with 1 research verification required (no_std CPU detection library)

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

```text
projects/rigel-synth/crates/
├── math/                        # Complete SIMD library (Layer 1) ← EXPANDED
│   ├── src/
│   │   ├── lib.rs              # Public API exports, DefaultSimdVector type alias
│   │   ├── traits.rs           # SimdVector, SimdMask, SimdInt traits
│   │   ├── backends/           # Backend implementations
│   │   │   ├── mod.rs          # Backend module organization
│   │   │   ├── scalar.rs       # ScalarVector<T> (always available)
│   │   │   ├── avx2.rs         # Avx2Vector (x86_64 + AVX2)
│   │   │   ├── avx512.rs       # Avx512Vector (x86_64 + AVX-512, experimental)
│   │   │   └── neon.rs         # NeonVector (aarch64)
│   │   ├── ops.rs              # Functional SIMD operations (add, mul, fma, etc.)
│   │   ├── math.rs             # Fast math kernels (sqrt, exp, log, sin, cos, tanh)
│   │   ├── table.rs            # Wavetable lookup with linear/cubic interpolation
│   │   ├── block.rs            # Block64, Block128 with SIMD chunk iteration
│   │   ├── simd/               # Runtime dispatch (NEW in this feature)
│   │   │   ├── mod.rs          # Public API re-exports
│   │   │   ├── backend.rs      # SimdBackend trait (unified interface)
│   │   │   ├── scalar.rs       # ScalarBackend wrapper
│   │   │   ├── avx2.rs         # Avx2Backend wrapper (x86_64)
│   │   │   ├── avx512.rs       # Avx512Backend wrapper (x86_64, experimental)
│   │   │   ├── neon.rs         # NeonBackend wrapper (aarch64)
│   │   │   ├── dispatcher.rs   # BackendDispatcher, CpuFeatures, BackendType
│   │   │   └── context.rs      # SimdContext (PRIMARY PUBLIC API)
│   │   └── denormal.rs         # Denormal protection (FTZ/DAZ)
│   ├── tests/                  # Unit tests for all backends + dispatch
│   ├── benches/                # SIMD performance benchmarks (NEW)
│   └── Cargo.toml              # Feature flags: runtime-dispatch, avx2, avx512, neon, force-*
│
├── dsp/                         # DSP core (Layer 2) - consumer of rigel-math
│   ├── src/
│   │   ├── lib.rs              # SynthEngine, oscillator, envelope (uses rigel_math::simd::SimdContext)
│   │   └── [existing DSP modules]
│   ├── benches/                # DSP-specific benchmarks
│   ├── tests/                  # DSP integration tests
│   └── Cargo.toml              # Depends on rigel-math with runtime-dispatch feature
│
├── cli/                        # Command-line test harness (no changes)
│   └── src/main.rs
│
├── plugin/                     # VST3/CLAP plugin (no changes, could use rigel-math in future for UI)
│   └── src/lib.rs
│
└── xtask/                      # Build tooling
    └── src/main.rs

.github/workflows/
└── ci.yml                      # CI pipeline updates for backend testing (MODIFIED)

devenv.nix                      # Development environment (potential script additions)
```

**Structure Decision**: rigel-math is now a complete, standalone SIMD library providing both low-level primitives (ops, math, table) and high-level unified API (SimdContext). The simd/ submodule handles runtime dispatch and backend selection. rigel-dsp becomes a pure consumer of rigel-math, importing `use rigel_math::simd::SimdContext`. This enables SIMD usage throughout the codebase (DSP, UI, future modules) without coupling to domain-specific crates.

**Public API exports** (`rigel-math/src/lib.rs`):
```rust
// Primary public SIMD API
pub mod simd {
    pub use crate::simd::context::SimdContext;  // Unified API
    pub use crate::simd::dispatcher::BackendType; // For debugging
}

// Data structures
pub use block::{Block64, Block128};

// Low-level access (advanced users)
pub use traits::{SimdVector, SimdMask, SimdInt};
pub use ops;  // Functional SIMD operations
pub use math; // Fast math kernels
```

**Typical usage** (anywhere in codebase):
```rust
use rigel_math::simd::SimdContext;
use rigel_math::Block64;

let ctx = SimdContext::new();  // Auto-selects best backend
ctx.apply_gain(&input, &mut output, 0.5);
```

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

N/A - No constitution violations. All principles satisfied.
