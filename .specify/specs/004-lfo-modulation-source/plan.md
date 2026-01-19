# Implementation Plan: LFO Modulation Source

**Branch**: `004-lfo-modulation-source` | **Date**: 2025-12-14 | **Spec**: [spec.md](./spec.md)
**Linear Issue**: [NEW-6](https://linear.app/new-atlantis/issue/NEW-6/create-lfo-modulation-source)
**Input**: Feature specification from `/specs/004-lfo-modulation-source/spec.md`

## Summary

Create a new `rigel-modulation` crate containing an LFO implementation with 7 waveshapes (sine, triangle, saw, square, pulse, sample-and-hold, noise), tempo sync, phase reset modes, polarity control, and control-rate integration. The `ModulationSource` trait will be moved from `rigel-timing` to this new crate. All code must be no_std, zero-allocation, and suitable for real-time audio processing with efficient control-rate updates for massive polyphony.

## Technical Context

**Language/Version**: Rust 2021 edition (from workspace rust-toolchain.toml)
**Primary Dependencies**: `rigel-timing` (for Timebase, ControlRateClock), `rigel-math` (for fast sin approximation per constitution)
**Storage**: N/A (pure computational library, no persistence)
**Testing**: `cargo test`, Criterion benchmarks for performance validation
**Target Platform**: All platforms (aarch64-apple-darwin, x86_64-unknown-linux-gnu, x86_64-pc-windows-msvc)
**Project Type**: Rust workspace crate
**Performance Goals**: <1 microsecond per LFO update, 0.1% of configured rate accuracy, 64 LFOs at control rate < 5% CPU vs per-sample
**Constraints**: no_std, zero heap allocations, Copy/Clone types, constant-time updates
**Scale/Scope**: Single LFO type, ~500-800 LOC expected

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Real-Time Safety | ✅ PASS | no_std, zero allocations, no blocking I/O - spec explicitly requires FR-022, FR-023, FR-024 |
| II. Layered Architecture | ✅ PASS | New `rigel-modulation` crate fits between timing and dsp layers |
| III. Test-Driven Validation | ✅ PASS | Will add unit tests, integration tests, and benchmarks |
| IV. Performance Accountability | ✅ PASS | Will use rigel-math for sin/cos per constitution mandate, add Criterion benchmarks |
| V. Reproducible Environments | ✅ PASS | Uses existing devenv shell, workspace configuration |
| VI. Cross-Platform Commitment | ✅ PASS | Pure Rust, no platform-specific code |
| VII. DSP Correctness Properties | N/A | Not wavetable-related; LFO has its own correctness properties (output range, frequency accuracy) |

**Gate Status**: PASS - No violations requiring justification.

## Project Structure

### Documentation (this feature)

```text
specs/004-lfo-modulation-source/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output (Rust trait definitions)
└── tasks.md             # Phase 2 output (from /speckit.tasks)
```

### Source Code (repository root)

```text
projects/rigel-synth/crates/
├── modulation/          # NEW: rigel-modulation crate
│   ├── Cargo.toml
│   ├── src/
│   │   ├── lib.rs       # Crate root, re-exports
│   │   ├── traits.rs    # ModulationSource trait (moved from timing)
│   │   ├── lfo.rs       # LFO implementation
│   │   ├── waveshape.rs # Waveshape enum and generation functions
│   │   ├── rate.rs      # LfoRateMode, NoteDivision types
│   │   └── rng.rs       # no_std PRNG for S&H and noise
│   ├── tests/
│   │   └── lfo_tests.rs # Integration tests
│   └── benches/
│       └── lfo_bench.rs # Criterion benchmarks
├── timing/              # MODIFIED: Remove ModulationSource, add re-export
│   └── src/
│       ├── lib.rs       # Update re-exports for backward compatibility
│       └── modulation.rs # REMOVE (move to rigel-modulation)
├── dsp/                 # MODIFIED: Add rigel-modulation dependency
│   └── Cargo.toml       # Add dependency
└── plugin/              # MODIFIED: May need rigel-modulation dependency
    └── Cargo.toml       # Add dependency if needed
```

**Structure Decision**: New crate `rigel-modulation` at `projects/rigel-synth/crates/modulation/` following established crate naming pattern. The crate depends on `rigel-timing` for `Timebase` and `ControlRateClock`, and provides `ModulationSource` trait plus LFO implementation.

## Complexity Tracking

> No violations - table not required.

## Dependencies

### Crate Dependencies

```text
rigel-modulation:
  - rigel-timing (path) - for Timebase, ControlRateClock types
  - rigel-math (path) - for fast sin approximations (per constitution IV)

rigel-timing:
  - REMOVE: modulation.rs
  - ADD: rigel-modulation (path) - for re-export backward compatibility

rigel-dsp:
  - ADD: rigel-modulation (path)
```

### Backward Compatibility Strategy

1. Move `ModulationSource` trait from `rigel-timing` to `rigel-modulation`
2. Have `rigel-timing` depend on `rigel-modulation` and re-export the trait
3. Existing code using `rigel_timing::ModulationSource` continues to work

## Key Technical Decisions

### 1. Waveshape Implementation

- **Sine**: Use `rigel_math::sin(phase * TAU)` per constitution IV mandate
- **Triangle**: Piecewise linear: `if phase < 0.5 { 4*phase - 1 } else { 3 - 4*phase }`
- **Saw**: `2*phase - 1`
- **Square**: `if phase < 0.5 { 1.0 } else { -1.0 }`
- **Pulse**: `if phase < pulse_width { 1.0 } else { -1.0 }`
- **S&H**: Store value, regenerate on phase wrap
- **Noise**: PRNG on every update

### 2. Phase Accumulation

- Store phase as f32 in [0.0, 1.0)
- Increment based on elapsed samples: `phase += rate_hz * elapsed_samples / sample_rate`
- Wrap: `phase = phase.fract()` (handles wrapping efficiently)

### 3. Random Number Generation

- Use PCG32 or similar simple PRNG (no_std compatible)
- Seed from optional user value or default
- Deterministic for reproducible patches

### 4. Tempo Sync Rate Calculation

```
rate_hz = (bpm / 60.0) * note_multiplier
where note_multiplier:
  - Whole: 0.25
  - Half: 0.5
  - Quarter: 1.0
  - Eighth: 2.0
  - etc.
  - Dotted: multiply by 2/3
  - Triplet: multiply by 3/2
```

### 5. Control Rate Integration

- `update()` reads elapsed samples from `Timebase::block_size()`
- Phase increment calculated as: `phase += rate_hz * block_size / sample_rate`
- `value()` returns cached `current_value` without computation
- Works with any `ControlRateClock` interval (1, 8, 16, 32, 64, 128)

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Phase drift over long sessions | Low | Medium | Use f64 for phase accumulation internally, cast to f32 for output |
| Sin approximation accuracy | Low | Low | Use rigel-math::sin which is already validated |
| Backward compat breakage | Medium | High | Keep rigel-timing re-export; test existing code paths |
| PRNG quality for audio | Low | Low | PCG32 is sufficient for LFO randomness |
| Control rate aliasing | Low | Medium | LFO rates typically < 100 Hz, well below Nyquist for control rate |

---

## Constitution Check (Post-Design)

*Re-evaluation after Phase 1 design completion.*

| Principle | Status | Verification |
|-----------|--------|--------------|
| I. Real-Time Safety | ✅ PASS | All types are Copy/Clone, no heap allocations in data model |
| II. Layered Architecture | ✅ PASS | rigel-modulation fits cleanly between timing and dsp |
| III. Test-Driven Validation | ✅ PASS | Contracts define testable interfaces; will add unit tests |
| IV. Performance Accountability | ✅ PASS | Using rigel-math::sin per constitution; will add benchmarks |
| V. Reproducible Environments | ✅ PASS | No new environment requirements |
| VI. Cross-Platform Commitment | ✅ PASS | Pure Rust with no platform-specific code |
| VII. DSP Correctness Properties | N/A | Not wavetable-related |

**Post-Design Gate Status**: PASS - Design conforms to all applicable principles.

---

## Generated Artifacts

| Artifact | Path | Description |
|----------|------|-------------|
| Plan | `plan.md` | This implementation plan |
| Research | `research.md` | PRNG, fast sin, trait migration decisions |
| Data Model | `data-model.md` | Entity definitions and relationships |
| Contracts | `contracts/` | Rust trait and type definitions |
| Quickstart | `quickstart.md` | Usage examples and integration guide |

## Next Steps

Run `/speckit.tasks` to generate the implementation task list.
