# Repository Guidelines

This file provides guidance for AI coding assistants and human contributors working with this repository.

## Project Overview

Rigel is a wavetable synthesizer built in Rust with a focus on performance, deterministic real-time processing, and portability. The monorepo contains:

- **rigel-synth**: Rust audio plugin and DSP core (VST3/CLAP)
- **wtgen**: Python wavetable generation and research toolkit
- **rigel-site**: Future marketing/documentation website
- **rigel-backend**: Future companion backend service

## Project Structure

```
projects/
  rigel-synth/crates/
    dsp/           # no_std DSP core - allocation-free real-time code
    timing/        # no_std timing infrastructure (Timebase, Smoother, etc.)
    math/          # no_std SIMD math library
    simd/          # SIMD vector abstractions and backends
    simd-dispatch/ # Runtime SIMD backend selection
    modulation/    # Modulation sources (LFOs, envelopes)
    cli/           # CLI tool for rendering audio and testing DSP
    plugin/        # NIH-plug wrapper for VST3/CLAP
    xtask/         # Build helpers (cargo xtask bundle)
  wtgen/           # Python research workspace (separate devenv)
  rigel-site/      # Placeholder for public site
  rigel-backend/   # Placeholder for backend service
```

## Development Environment

**All development requires Nix + devenv.** The repository has two separate devenv shells:

1. **Root shell** (Rust): For rigel-synth development
2. **wtgen shell** (Python): At `projects/wtgen/`

With direnv configured, shells activate automatically when entering directories. If devenv is broken, prefix commands with `devenv shell -- <command>`.

## Essential Commands

**Always use devenv scripts** - they handle environment setup automatically.

### Rust Development (from repository root)

```bash
# Building
build:native        # Build plugin for current platform
build:macos         # macOS native build (requires macOS)
build:linux         # Linux native build (requires Linux)
build:win           # Windows cross-build via xwin

# Testing & Quality
cargo:test          # Run all tests
cargo:fmt           # Format code
cargo:lint          # Run clippy

# SIMD-Specific Tests
test:avx2           # Test with AVX2 (requires x86_64)
test:neon           # Test with NEON (requires macOS ARM)

# Benchmarking
bench:all           # Run full benchmark suite
bench:criterion     # Wall-clock benchmarks
bench:baseline      # Save baseline for comparison
bench:flamegraph    # Generate flamegraph
```

### Python Development (from `projects/wtgen/`)

```bash
# Testing
test:full           # Full pytest suite with parallel execution
test:fast           # Single-process with early exit

# Code Quality (run all before completing changes)
lint                # Ruff lint
format              # Ruff formatter
typecheck           # ty type checker

# CLI
uv run wtgen generate <waveform> --output <file.npz>
uv run wtgen info <file.npz>
```

## Critical Constraints

### Real-Time Safety (rigel-dsp)

The DSP core must maintain:
- **No heap allocations**: No Vec, Box, String, or std collections
- **No blocking I/O**: No file operations, network calls, or locks
- **No std library**: Only libm for math operations
- **Deterministic performance**: Consistent CPU usage regardless of input

Verify new dependencies support `no_std` and don't allocate.

### wtgen Standards

- All code must pass ty type checking
- Run pytest, ty, and ruff before considering changes complete
- Add tests for new code

## Testing Guidelines

### Rust Tests

Run with `cargo:test`. Always run all tests including architecture-specific features before completing a feature:
- Unit tests in crate modules
- Integration tests in `tests/` directories
- SIMD tests (`test:avx2` on x86_64, `test:neon` on macOS ARM)

### Python Tests (wtgen)

Run with `test:full` or `test:fast`:
- Unit tests: `tests/unit/`
- Integration tests: `tests/integration/`
- Property-based testing via Hypothesis

## Coding Conventions

### Rust
- Rust 2021 edition, 4-space indentation
- snake_case for functions/variables, UpperCamelCase for types
- Keep rigel-dsp free of std, heap allocations, and blocking operations
- Document public APIs with rustdoc
- Test naming: `mod_name_behavior` pattern

### Python
- Line length: 100 characters
- Type hints required on all public functions
- Ruff for linting and formatting

### Commits & PRs
- Short, imperative messages (<72 chars)
- Each commit should remain buildable and scoped to one concern
- PRs should describe motivation, outline testing, and link issues

### Nix/DevEnv
- Pay attention to nix string interpolation conflicts with shell syntax
- When testing nix changes, tail output (last 100 lines matter most)

## CI/CD Pipeline

### Main CI (`.github/workflows/ci.yml`)

Runs on all PRs and pushes:
- `rigel-pipeline`: fmt, clippy, scalar tests, AVX2 tests
- `wtgen-pipeline`: ruff lint, ty type check, pytest
- Plugin builds: Linux (native), Windows (cross-compiled), macOS (native)

### Release Workflows

**Continuous Release**: Every successful CI on `main` updates the "latest" pre-release.

**Tagged Release**: Push a tag like `v0.2.0` to create a versioned release:
```bash
git tag v0.2.0
git push origin v0.2.0
```

### Plugin Installation

**macOS:**
```bash
tar -xzf rigel-plugin-latest-macos.tar.gz
cp -r rigel-plugin.vst3 ~/Library/Audio/Plug-Ins/VST3/
cp -r rigel-plugin.clap ~/Library/Audio/Plug-Ins/CLAP/
```

**Linux:**
```bash
tar -xzf rigel-plugin-latest-linux.tar.gz
mkdir -p ~/.vst3 ~/.clap
cp -r rigel-plugin.vst3 ~/.vst3/
cp rigel-plugin.clap ~/.clap/
```

**Windows (PowerShell):**
```powershell
tar -xzf rigel-plugin-latest-windows.tar.gz
Copy-Item -Recurse rigel-plugin.vst3 "$env:COMMONPROGRAMFILES\VST3\"
Copy-Item rigel-plugin.clap "$env:COMMONPROGRAMFILES\CLAP\"
```

## Architecture Details

### SIMD Backend

Rigel uses a layered architecture for SIMD:
- **rigel-simd**: Trait-based SIMD vector abstractions (no_std, zero-allocation)
- **rigel-simd-dispatch**: Runtime backend selection on x86_64
- **rigel-math**: Math utilities built on SIMD abstractions
- **rigel-dsp**: DSP core using the above

**Backend selection:**
- x86_64: Runtime dispatch (AVX-512 > AVX2 > scalar)
- aarch64 (macOS): Compile-time NEON selection

### rigel-timing

Infrastructure for sample-accurate timing:
- `Timebase`: Synchronized DSP module timing
- `Smoother`: Parameter smoothing (Linear, Exponential, Logarithmic)
- `ControlRateClock`: Control-rate update scheduling
- `ModulationSource` trait: LFOs, envelopes, sequencers

## Key File Locations

### Configuration
- `Cargo.toml` - Rust workspace manifest
- `rust-toolchain.toml` - Rust version and targets
- `devenv.nix` - Rust development environment
- `projects/wtgen/devenv.nix` - Python development environment
- `.github/workflows/ci.yml` - CI/CD pipeline

### Source Code
- `projects/rigel-synth/crates/dsp/src/` - Core DSP
- `projects/rigel-synth/crates/timing/src/` - Timing infrastructure
- `projects/rigel-synth/crates/math/src/` - Math utilities
- `projects/rigel-synth/crates/simd/src/` - SIMD vector abstractions
- `projects/rigel-synth/crates/simd-dispatch/src/` - Runtime backend selection
- `projects/rigel-synth/crates/modulation/src/` - Modulation sources
- `projects/rigel-synth/crates/plugin/src/` - Plugin wrapper
- `projects/rigel-synth/crates/cli/src/` - CLI tool
- `projects/wtgen/src/wtgen/` - Python package

## Cross-Platform Notes

macOS/Linux cross-compilation is not supported due to GUI library complexities. Use CI for cross-platform builds. Windows cross-compilation works via xwin.
