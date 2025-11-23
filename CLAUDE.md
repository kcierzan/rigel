# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Rigel is an advanced wavetable synthesizer built in Rust with a focus on performance, deterministic real-time processing, and portability. The monorepo contains:

- **rigel-synth**: Rust audio plugin and DSP core (VST3/CLAP)
- **wtgen**: Python wavetable generation and research toolkit
- **rigel-site**: Future marketing/documentation website
- **rigel-backend**: Future companion backend service

## Architecture

### Layered Design (rigel-synth)

1. **rigel-dsp**: No-std DSP core with zero allocations
   - Real-time safe, deterministic, portable to embedded systems
   - Single `lib.rs` containing `SynthEngine`, `SimpleOscillator`, `Envelope`
   - Located at: `projects/rigel-synth/crates/dsp/src/lib.rs`

2. **rigel-cli**: Command-line tool wrapping DSP core
   - Test harness for DSP functionality
   - Generates WAV files for validation

3. **rigel-plugin**: NIH-plug integration for DAW use
   - VST3/CLAP plugin wrapper
   - iced-based GUI

4. **rigel-xtask**: Custom cargo tasks for bundling

### wtgen Architecture

Modular DSP pipeline for wavetable asset generation located at `projects/wtgen/src/wtgen/`:

- **Wave Generation** (`dsp/waves.py`): Harmonic synthesis, polyblep, standard waveforms
- **Mipmap System** (`dsp/mipmap.py`): Bandlimited mipmaps for alias-free playback
- **Processing** (`dsp/process.py`): DC removal, normalization, zero-crossing alignment
- **EQ/Filtering** (`dsp/eq.py`, `dsp/fir.py`): Parametric EQ, rolloff methods
- **CLI** (`cli/`): Command interface and validators
- **Export** (`export.py`): NPZ and WAV output
- **Plotting** (`plotting.py`): Visualization with structured data output (PlotFigure dataclass)
- **Types** (`types.py`): Type definitions

## Development Environment

All development requires Nix + devenv. The repository has two separate devenv shells:

1. **Root shell** (Rust): For rigel-synth development
2. **wtgen shell** (Python): For wtgen development, located at `projects/wtgen/`

### Entering Shells

With direnv (automatic):
```bash
cd /Users/kylecierzan/git/rigel  # Auto-enters Rust shell
cd projects/wtgen                # Auto-enters Python shell
```

Manual:
```bash
devenv shell                     # Root: Rust environment
cd projects/wtgen && devenv shell  # wtgen: Python environment
```

### Important Environment Variables

Set by devenv:
- `RIGEL_SYNTH_ROOT`: Points to `./projects/rigel-synth`
- `RIGEL_WTGEN_ROOT`: Points to `./projects/wtgen`
- `RUST_BACKTRACE=1`
- `MACOSX_DEPLOYMENT_TARGET=11.0`

## Common Commands

### Rust Development

All commands run from repository root in devenv shell. Thanks to direnv, you're likely already in the shell. If not, prefix with `devenv shell -- <command>`:

```bash
# Building (Native only - simple and reliable!)
build:native        # Build for current platform
build:macos         # macOS build (requires macOS host)
build:linux         # Linux build (requires Linux host)
build:win           # Windows cross-build (may work from Linux via xwin)

# SIMD-Optimized Builds
# These enable specific SIMD backends with scalar fallback for remainder samples
build:avx2          # Build with AVX2 support (x86_64 Linux/Windows)
build:avx512        # Build with AVX-512 support (x86_64 Linux/Windows)
build:neon          # Build with NEON support (aarch64 macOS only)

# Testing & Quality
cargo:test          # Run all Rust tests
cargo:fmt           # Format Rust code
cargo:lint          # Run clippy linter

# SIMD-Specific Tests
test:avx2           # Test with AVX2 backend enabled
test:avx512         # Test with AVX-512 backend enabled
test:neon           # Test with NEON backend enabled (macOS only)

# Plugin Bundling
cargo xtask bundle rigel-plugin --release [--target <triple>]

# Benchmarking & Profiling
bench:all           # Run all benchmarks (Criterion + iai-callgrind)
bench:criterion     # Run Criterion benchmarks (wall-clock time)
bench:iai           # Run iai-callgrind benchmarks (instruction counts)
bench:baseline      # Save Criterion baseline for comparisons
bench:flamegraph    # Generate flamegraph for profiling
bench:instruments   # macOS only: Profile with Instruments.app

# Clean
build:clean         # Remove target directory
```

**Note on GCC Specs Directory Conflict:**
All devenv scripts (`cargo:test`, `cargo:lint`, `build:*`, `bench:*`) automatically handle the GCC specs directory workaround. The repository's `specs/` directory can confuse GCC which looks for a specs file in the current directory. The devenv scripts temporarily rename it during builds/tests. If running cargo commands directly (not through devenv scripts), you may need to manually rename the directory:

```bash
# Manual workaround (only needed if NOT using devenv scripts)
mv specs specs.bak && cargo test; mv specs.bak specs
```

### Python Development (wtgen)

All commands must run in wtgen devenv shell. Enter with `cd projects/wtgen && devenv shell`:

```bash
# Environment
uv:sync             # Sync dependencies from uv.lock
                    # (runs automatically on first shell entry)

# Testing
test:full           # Full pytest suite with xdist parallelization
test:fast           # Single-process pytest with -x --tb=short

# Run single test
uv run python -m pytest -x --tb=short <path_to_test_file>

# Control Hypothesis property testing iterations
HYPOTHESIS_MAX_EXAMPLES=5000 uv run pytest tests/ -x -n auto --tb=short

# Code Quality (ALWAYS run before considering changes complete)
lint                # Ruff lint
format              # Ruff formatter
typecheck           # Both mypy and basedpyright
typecheck:mypy      # mypy only
typecheck:pyright   # basedpyright only

# CLI Usage
uv run wtgen generate <waveform> --output <file.npz> [options]
uv run wtgen harmonic --partials "1:1.0:0.0,..." --output <file.npz>
uv run wtgen info <file.npz>
```

## Critical Constraints

### Real-Time Safety (rigel-dsp)

The DSP core must maintain:
- **No heap allocations**: No Vec, Box, String, or any std collections
- **No blocking I/O**: No file operations, network calls, or locks
- **No std library**: Only libm for math operations
- **Deterministic performance**: Consistent CPU usage regardless of input

When adding dependencies to rigel-dsp, verify they support `no_std` and don't use allocations.

### wtgen Standards

- **RMS target**: 0.35 for normalized wavetables
- **Default config**: 8 octaves, 2048 samples, Tukey rolloff
- **Mipmap system**: Must provide alias-free playback across MIDI range 0-127
- **Type checking**: All code must pass both mypy and basedpyright
- **Quality gates**: ALWAYS run pytest, mypy, basedpyright, and ruff before considering changes complete
- **Test coverage**: ALWAYS add tests for new code and run them before considering task complete

Critical DSP properties that must be preserved:
- Zero-crossing alignment across all mipmap levels
- RMS consistency validation
- DC offset removal throughout processing chain
- Antialiasing effectiveness across full MIDI range
- Phase coherence preservation
- Spectral bandlimiting for alias prevention

## Testing Infrastructure

### Rust Tests

Run with `cargo:test` or `cargo test` from root:
- Unit tests embedded in crate modules
- Integration tests in `tests/` directories
- Architecture-specific tests for SIMD optimizations (NEON on aarch64, AVX2/AVX-512 on x86_64)
- Audio validation via WAV file generation

**IMPORTANT**: Always run ALL tests including architecture-specific features available on the current host before considering a feature complete.

### Python Tests (wtgen)

103+ tests across categories:
- **Unit tests**: `tests/unit/` - DSP components (process, waves, mipmap, CLI)
- **Integration tests**: `tests/integration/` - End-to-end pipeline
- **Property-based testing**: Hypothesis generates thousands of test cases validating:
  - Mathematical invariants in DSP processing
  - Robust behavior with edge cases and unexpected inputs
  - Energy balancing without DC introduction
  - Range normalization with zero-mean constraint

Run with:
```bash
test:full    # Parallel execution with pytest-xdist
test:fast    # Single-process with early exit on failure
```

Higher `HYPOTHESIS_MAX_EXAMPLES` provides more thorough validation but takes longer:
```bash
HYPOTHESIS_MAX_EXAMPLES=5000 uv run pytest tests/ -x -n auto --tb=short
```

## CI/CD Pipeline

### Main CI Workflow (`.github/workflows/ci.yml`)

Runs on all PRs and pushes:

1. **Parallel Pipelines**:
   - `rigel-pipeline`: fmt, clippy, test
   - `wtgen-pipeline`: ruff lint, pytest

2. **Plugin Builds**:
   - Linux: Native build on ubuntu-latest (`x86_64-unknown-linux-gnu`)
   - Windows: Cross-compiled on ubuntu-latest via xwin (`x86_64-pc-windows-msvc`)
   - macOS: Native build on macos-14 runner (`aarch64-apple-darwin`)

All CI commands run through devenv shell for reproducibility.

**Build Quirks:**
- **GCC Specs Directory Workaround**: The `specs/` directory at the repository root confuses GCC, which searches for a specs file in the current directory. This causes build failures on Linux/Windows with errors like `gcc: fatal error: cannot read spec file './specs': Is a directory`.
  - **Automated in devenv**: All local devenv scripts (`cargo:test`, `build:*`, `bench:*`) automatically handle this by temporarily renaming `specs/` to `specs.bak` during execution
  - **Automated in CI**: CI workflows and build scripts (`.github/workflows/*.yml`, `ci/scripts/build-win.sh`) implement the same workaround
  - **Manual workaround**: Only needed when running cargo commands directly without devenv: `mv specs specs.bak && cargo test; mv specs.bak specs`
  - The directory is always restored via cleanup traps, even on failures

### Release Workflows

**Continuous Release** (`.github/workflows/continuous-release.yml`):
- Triggers: After CI workflow completes successfully on `main` branch
- Uses `workflow_run` to avoid duplicate test execution
- Builds plugins for Linux, Windows, macOS
- Updates "latest" pre-release with binaries
- Archive naming: `rigel-plugin-latest-{platform}.tar.gz`
- Each archive contains both VST3 and CLAP bundles from `target/bundled/`

**Tagged Release** (`.github/workflows/release.yml`):
- Triggers: Push tags matching `v*` (e.g., `v0.2.0`)
- Builds plugins for all platforms
- Creates new GitHub release with auto-generated changelog
- Archive naming: `rigel-plugin-{version}-{platform}.tar.gz`
- Published as stable releases (not pre-release)
- **Best practice**: Only tag commits that have already passed CI on `main`

**Creating a Release:**
```bash
# 1. Merge your PR to main and wait for CI to pass
# 2. Ensure version in Cargo.toml matches desired version
# 3. Tag the commit on main that passed CI
git tag v0.2.0
git push origin v0.2.0
# GitHub Actions will automatically build and create the release
```

**Workflow Execution Flow:**
1. **PR opened/updated**: CI workflow runs tests + builds
2. **PR merged to main**: CI workflow runs again
3. **After CI completes successfully**: Continuous Release workflow triggers automatically
4. **Tag pushed**: Release workflow builds and publishes version

**Notes:**
- Bundles are created in `target/bundled/` by nih_plug_xtask
- macOS binaries are currently unsigned
- Future: Add code signing via GitHub secrets

### Branch Protection

To enforce that PRs cannot be merged with failing tests or builds:

1. Go to repository **Settings** → **Branches** → **Add rule**
2. Set **Branch name pattern**: `main`
3. Enable **Require status checks to pass before merging**
4. Select required check: **CI Success**
5. Enable **Require branches to be up to date before merging** (recommended)
6. Enable **Do not allow bypassing the above settings** (recommended)

The `CI Success` job in `.github/workflows/ci.yml` aggregates all test and build results into a single status check. This job will fail if any of the following fail:
- Rigel lint & tests (`rigel-pipeline`)
- wtgen lint & tests (`wtgen-pipeline`)
- Linux & Windows plugin builds (`build-plugin-linux`)
- macOS plugin build (`build-plugin-macos`)

**Additional Recommendations:**
- Enable **Require a pull request before merging** with at least 1 approval
- Enable **Require review from Code Owners** if using a CODEOWNERS file
- Enable **Require linear history** to prevent merge commits

## Cross-Platform Support

### Build Strategy: Native-First with Windows Exception

Rigel uses **native builds** for local development where possible, with Windows cross-compilation as a pragmatic exception.

**Local Development:**
- **macOS developers**: Use `build:native` or `build:macos` (native builds only, requires macOS host)
- **Linux developers**: Use `build:native` or `build:linux` (native builds only, requires Linux host)
- **Windows developers**: Use `build:win` (cross-compilation via xwin, works on Linux/macOS)
- **macOS↔Linux cross-compilation**: **Not supported** - use CI instead

**Why Limited Cross-Compilation?**

Cross-compiling GUI applications between macOS and Linux is complex and fragile due to:
- X11/XCB/Wayland library dependencies with different ABIs
- Linker configuration conflicts between platforms
- Rust toolchain limitations for cross-platform GUI builds

The complexity and maintenance burden outweigh the benefits. **Native builds + CI** is simpler and more reliable.

**Why Windows Cross-Compilation Works:**

Windows is the exception because:
- MSVC SDK can be downloaded via `xwin` without a Windows VM
- `rust-lld` can link MSVC binaries from any platform
- No GUI-specific Linux/macOS dependencies required
- Proven reliable in CI and local development

### CI for Multi-Platform Builds

All production builds use GitHub Actions:
- **Linux builds**: `ubuntu-latest` (native x86_64 build for `x86_64-unknown-linux-gnu`)
- **Windows builds**: `ubuntu-latest` (cross-compiled via xwin for `x86_64-pc-windows-msvc`)
- **macOS builds**: `macos-14` (native Apple Silicon build for `aarch64-apple-darwin`)

This provides reliable, reproducible builds with minimal complexity.

### Recommended Workflow

1. **Develop on your platform**: Use `build:native` for fast iteration
2. **Test locally**: Native builds are fast and reliable
3. **Test other platforms**: Push to GitHub and let CI build
4. **Release**: Tag a commit and CI creates releases for all platforms

This workflow is simpler, more reliable, and matches how most cross-platform projects work.

### Supported Targets

Configured in `rust-toolchain.toml`:
- **macOS**: `aarch64-apple-darwin` (Apple Silicon)
- **Linux**: `x86_64-unknown-linux-gnu`
- **Windows**: `x86_64-pc-windows-msvc`

**Build Commands:**
```bash
# Native builds (fast and reliable!)
build:native        # Build for current platform (auto-detects)
build:macos         # macOS native build (requires macOS host)
build:linux         # Linux native build (requires Linux host)
build:win           # Windows cross-build via xwin (works on Linux/macOS)
```

**Windows Cross-Build Details:**

The `build:win` command uses `ci/scripts/build-win.sh` which:
1. Downloads Windows SDK via `xwin` (cached in `~/.cache/xwin`)
2. Configures MSVC linker environment variables (LIB, INCLUDE, LIBPATH)
3. Uses `rust-lld` and `llvm-ar` from rustc sysroot
4. Invokes `cargo xtask bundle rigel-plugin --release --target x86_64-pc-windows-msvc`

This approach avoids needing a Windows VM or GitHub runner for local testing.

## SIMD Backend Architecture

Rigel uses a two-crate layered architecture for SIMD optimization:
- **rigel-math**: Trait-based SIMD abstraction library (no_std, zero-allocation)
- **rigel-dsp**: DSP core that uses rigel-math abstractions

### Backend Selection Strategy

**x86_64 (Linux/Windows)**: Runtime dispatch with CPU feature detection
- Single binary contains scalar, AVX2, and AVX-512 backends
- CPU features detected once at startup using `cpufeatures` crate
- Optimal backend selected automatically (AVX-512 → AVX2 → scalar priority)
- <1% dispatch overhead via function pointers

**aarch64 (macOS)**: Compile-time NEON selection
- NEON always available on Apple Silicon, no runtime detection needed
- Zero-cost abstraction, compiles directly to NEON instructions
- No dispatch overhead

### rigel-math Library

Provides trait-based SIMD abstractions enabling backend-agnostic DSP code:

**Core Traits:**
- `SimdVector`: Generic SIMD vector operations (load, store, arithmetic)
- `SimdMask`: SIMD boolean masks for conditional operations
- `SimdInt`: Integer SIMD operations

**Backend Implementations:**
- `ScalarVector<f32>`: Portable scalar fallback (1 lane, always available)
- `Avx2Vector`: x86_64 AVX2 backend (8 f32 lanes)
- `Avx512Vector`: x86_64 AVX-512 backend (16 f32 lanes)
- `NeonVector`: aarch64 NEON backend (4 f32 lanes)

**Modules:**
- `ops`: Functional-style SIMD operations (add, mul, fma, min/max, etc.)
- `math`: Fast math kernels (tanh, exp, log, sin/cos, sqrt, pow)
- `block`: Fixed-size aligned buffers (`Block64`, `Block128`) for cache efficiency
- `table`: Wavetable lookup with linear/cubic interpolation
- `denormal`: FTZ/DAZ denormal protection for real-time stability

**Example Usage:**
```rust
use rigel_math::{Block64, DefaultSimdVector, DenormalGuard, SimdVector};
use rigel_math::ops::mul;

fn apply_gain(input: &Block64, output: &mut Block64, gain: f32) {
    let _guard = DenormalGuard::new();
    let gain_vec = DefaultSimdVector::splat(gain);

    for (in_chunk, mut out_chunk) in input.as_chunks::<DefaultSimdVector>()
        .iter()
        .zip(output.as_chunks_mut::<DefaultSimdVector>().iter_mut())
    {
        let value = in_chunk.load();
        out_chunk.store(mul(value, gain_vec));
    }
}
```

### rigel-dsp Integration

**Single Source of Truth:**

All SIMD functionality is provided by rigel-math. rigel-dsp no longer has its own SIMD abstractions - DSP code imports directly from rigel-math.

**Recommended API:**

The `SimdContext` type provides a consistent high-level API with 30+ math operations:

```rust
use rigel_math::{Block64, DefaultSimdVector, ops};
use rigel_math::simd::{SimdContext, ProcessParams};

// Initialize once during engine startup
let ctx = SimdContext::new();

// Query selected backend (for logging/debugging)
println!("Using SIMD backend: {}", ctx.backend_name());

// Example 1: Using SimdContext high-level operations
let mut input = Block64::new();
let mut output = Block64::new();

// Fill input with test data
for i in 0..64 {
    input[i] = i as f32;
}

// Apply gain using SimdContext
ctx.apply_gain(input.as_slice(), output.as_slice_mut(), 0.5);

// Example 2: Using process_block with ProcessParams
let params = ProcessParams {
    gain: 2.0,
    frequency: 440.0,
    sample_rate: 44100.0,
};
ctx.process_block(input.as_slice(), output.as_slice_mut(), &params);

// Example 3: Direct rigel-math usage (for custom operations)
for (in_vec, mut out_chunk) in input.as_chunks::<DefaultSimdVector>()
    .iter()
    .zip(output.as_chunks_mut::<DefaultSimdVector>().iter_mut())
{
    let result = ops::add(in_vec, DefaultSimdVector::splat(1.0));
    out_chunk.store(result);
}
```

**How it works:**
- **x86_64 with `runtime-dispatch`**: `SimdContext` uses runtime CPU detection to select optimal backend
- **aarch64 or forced backends**: `SimdContext` uses compile-time `DefaultSimdVector` (zero overhead)
- **DSP code is identical**: No platform-specific `#[cfg]` directives needed

**Low-Level API (Advanced):**

For direct control, you can use the dispatcher explicitly:

```rust
use rigel_math::simd::dispatcher::{BackendDispatcher, BackendType, CpuFeatures};

// Initialize once during engine startup
let dispatcher = BackendDispatcher::init();

// Query selected backend
match dispatcher.backend_type() {
    BackendType::Avx512 => println!("Using AVX-512"),
    BackendType::Avx2 => println!("Using AVX2"),
    BackendType::Scalar => println!("Using scalar fallback"),
    BackendType::Neon => println!("Using NEON"),
}
```

### Feature Flags

**rigel-math features:**
- `scalar`: Scalar backend (always available, default)
- `avx2`: Enable AVX2 backend compilation
- `avx512`: Enable AVX-512 backend compilation (experimental)
- `neon`: Enable NEON backend compilation
- `runtime-dispatch`: Allow multiple backends to coexist for runtime selection

**rigel-dsp features:**
- `runtime-dispatch`: Enable runtime CPU detection and backend selection (default)
- `force-scalar`: Force scalar backend for deterministic CI testing
- `force-avx2`: Force AVX2 backend for deterministic CI testing
- `force-avx512`: Force AVX-512 backend for local experimental testing

**Build Examples:**
```bash
# Default: Runtime dispatch with all backends
cargo build --release

# Force scalar (CI testing)
cargo build --release --features force-scalar

# Force AVX2 (CI testing)
cargo build --release --features force-avx2

# Test specific backend
cargo test --features force-avx2
```

### CI Backend Testing

The CI pipeline (`.github/workflows/ci.yml`) tests SIMD backends deterministically:

**`backend-tests` job**:
- **Scalar**: Tested in main `rigel-pipeline` job (default features)
- **AVX2**: ubuntu-latest with `RUSTFLAGS="-C target-feature=+avx2,+fma"`
- **NEON**: macos-latest (Apple Silicon runners)
- **AVX-512**: Not tested in CI (Rust intrinsics incomplete, runners lack support)

All backend tests must pass for CI to succeed.

### Testing Locally

**Test all available backends on your CPU:**
```bash
# Run all tests (uses default features, likely runtime-dispatch)
cargo test

# Force scalar backend
cargo test --features force-scalar

# Force AVX2 backend (requires AVX2 CPU)
test:avx2

# Force NEON backend (macOS only, requires Apple Silicon)
test:neon
```

**Check your CPU features:**
```bash
# Linux
lscpu | grep -E "avx2|avx512"

# macOS
sysctl -a | grep machdep.cpu.features
```

### Performance Validation

**Benchmarking SIMD Performance:**
```bash
# Run all benchmarks with current backend configuration
bench:all

# Save performance baseline before changes
bench:baseline

# Compare current vs baseline
bench:criterion
```

**Expected SIMD Speedups (vs scalar):**
- AVX2 (8 lanes): ~2-4x faster for block processing
- AVX-512 (16 lanes): ~4-8x faster for block processing
- NEON (4 lanes): ~2-4x faster for block processing
- Dispatch overhead: <1% of block processing time

### no_std Compliance

Both rigel-math and rigel-dsp maintain strict no_std compliance:
- No heap allocations (no Vec, Box, String, or collections)
- No blocking I/O (no file operations, network, or locks)
- Stack-only data structures
- Deterministic performance

**Dependencies:**
- `libm`: no_std math functions (exp, log, trig, etc.)
- `cpufeatures`: no_std CPU feature detection (x86_64 CPUID)

### Troubleshooting

**"Illegal instruction" errors:**
- Running AVX2/AVX-512 code on CPU without support
- Solution: Use `force-scalar` feature or check CPU capabilities

**Performance regressions:**
- Verify SIMD backend is being used (check dispatcher output)
- Ensure function pointers are inlined (`#[inline]`)
- Profile with `bench:flamegraph` to identify bottlenecks

**CI backend test failures:**
- Verify feature flags are correct in `.github/workflows/ci.yml`
- Check RUSTFLAGS match required CPU features
- Ensure devenv shell is used for reproducible builds

## Coding Conventions

### Rust
- Follow Rust 2021 edition defaults
- 4-space indentation, snake_case for functions/variables, UpperCamelCase for types
- Keep rigel-dsp free of std, heap allocations, and blocking operations
- Document public items with rustdoc comments
- Use descriptive kebab-case for CLI subcommands (e.g., `note`, `chord`, `scale`)
- Test naming: `mod_name_behavior` pattern
- For audio changes, regenerate WAV fixtures via CLI and verify before merging

### Python
- Line length: 100 characters (configured in pyproject.toml)
- Ruff for linting and formatting
- Type hints required on all public functions
- Scientific computing conventions allow magic values for DSP algorithms
- basedpyright: Suppresses warnings from scientific libraries while catching real type errors
- mypy: Strict typing with specific DSP module exemptions in `mypy.ini`
- Fix typechecking issues when encountered

### Commits & PRs
- Short, imperative messages (<72 chars)
- Each commit should remain buildable and scoped to one concern
- Match existing style: "Fix devenv comment", "Make pip install..."
- PRs should describe motivation, outline testing, and link issues
- Include screenshots or audio snippets when UI or audible behavior changes
- For DAW validation, attach bundle paths

### Nix/DevEnv Editing
- Pay close attention to nix string interpolation causing conflicts with shell
- Always escape correctly
- When testing nix/devenv changes, tail output (nix errors are very long, last 100 lines matter most)

## Key File Locations

### Configuration Files
- `Cargo.toml` - Rust workspace manifest with shared dependencies
- `rust-toolchain.toml` - Rust version and target configuration
- `devenv.nix` - Rigel (Rust) development environment
- `projects/wtgen/devenv.nix` - wtgen (Python) development environment
- `projects/wtgen/pyproject.toml` - Python package configuration
- `.github/workflows/ci.yml` - CI/CD pipeline
- `AGENTS.md` - Contributor guidelines

### Source Code
- `projects/rigel-synth/crates/dsp/src/lib.rs` - Core DSP implementation
- `projects/rigel-synth/crates/cli/src/main.rs` - CLI tool
- `projects/rigel-synth/crates/plugin/src/lib.rs` - Plugin entry point
- `projects/wtgen/src/wtgen/` - Python package root

## Nix/DevEnv Considerations

- All commands must run inside devenv shell (automatic with direnv)
- If devenv shell is broken, prefix commands with: `devenv shell -- <command>`
- Update workflow: `nix flake update` and `devenv update` together
- Rollback: `git checkout -- flake.lock devenv.lock` + `direnv reload`
- CI runs commands via: `devenv shell -- <command>`

## Asset Management

- Check in large binary assets using git-lfs
- Python experiments in `projects/wtgen/tests/` should mirror Rust expectations
- Only check in generated assets when deterministic

## Performance Targets

- Single voice CPU usage: ~0.1% at 44.1kHz (validate with `bench:all`)
- Full polyphonic target: <1% CPU usage
- Zero-allocation guarantee in DSP core
- Consistent performance across all platforms

### Benchmarking

All DSP performance is validated through comprehensive benchmarking:
- **Criterion**: Wall-clock time measurements for local optimization
- **iai-callgrind**: Deterministic instruction counts for regression detection
- **Location**: `projects/rigel-synth/crates/dsp/benches/`
- **Documentation**: See `docs/benchmarking.md` for detailed usage

Run benchmarks regularly to validate performance claims and detect regressions:
```bash
bench:all          # Run full benchmark suite
bench:baseline     # Save performance baseline before changes
bench:flamegraph   # Generate flamegraph for optimization
```

## Active Technologies
- Rust 2021 edition (workspace toolchain from rust-toolchain.toml) (001-fast-dsp-math)
- N/A (pure computational library, no persistence) (001-fast-dsp-math)
- Rust 2021 edition (from rust-toolchain.toml) (001-runtime-simd-dispatch)

## Recent Changes
- 001-fast-dsp-math: Added Rust 2021 edition (workspace toolchain from rust-toolchain.toml)
