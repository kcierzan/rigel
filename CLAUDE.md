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
# Building
build:native        # Build for current platform
build:macos         # Build for Apple Silicon macOS
build:linux         # Build for x86_64 Linux
build:win           # Build for Windows (uses xwin SDK)

# Testing & Quality
cargo:test          # Run all Rust tests
cargo:fmt           # Format Rust code
cargo:lint          # Run clippy linter

# Plugin Bundling
cargo xtask bundle rigel-plugin --release [--target <triple>]

# Clean
build:clean         # Remove target directory
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
- Audio validation via WAV file generation

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

GitHub Actions workflow (`.github/workflows/ci.yml`) runs:

1. **Container Builds**: Only when devenv config changes
   - Rigel container (Rust environment)
   - wtgen container (Python environment)

2. **Parallel Pipelines**:
   - `rigel-pipeline`: fmt, clippy, test
   - `wtgen-pipeline`: ruff lint, pytest

3. **Plugin Builds**:
   - Linux & Windows: Matrix build in container
   - macOS: Native build on macos-14 runner

All CI commands run through containers for reproducibility.

## Cross-Platform Support

Targets configured in `rust-toolchain.toml` and `devenv.nix`:
- **macOS**: `aarch64-apple-darwin` (Apple Silicon)
- **Linux**: `x86_64-unknown-linux-gnu`
- **Windows**: `x86_64-pc-windows-msvc` (via xwin cross-compilation)

Build for specific target:
```bash
build:macos    # or build:linux, build:win
```

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

- Single voice CPU usage: ~0.1% at 44.1kHz
- Full polyphonic target: <1% CPU usage
- Zero-allocation guarantee in DSP core
- Consistent performance across all platforms
