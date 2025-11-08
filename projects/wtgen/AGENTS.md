## Project Overview

wtgen is a Python library for wavetable generation and processing, specializing in harmonic
wavetable synthesis and bandlimited mipmap generation for alias-free audio playback. The codebase
focuses on high-quality DSP processing with extensive testing coverage.

## Development Setup

Use of devenv shell is ABSOLUTELY ESSENTIAL to running common development tasks for the project.
Simply entering the shell for the first time should install dependencies automatically with `uv`:

```sh
devenv shell
```

will in effect run:

```sh
uv sync --frozen --group dev
uv pip install --quiet -e .
```

This also modifies the shell path to use nix binaries for `ruff` and `basedpyright` to avoid
dynamic linking issues on NixOS.

## Running commands

All important Python repository commands are run via `devenv shell`.

### Installation

```sh
devenv shell -- uv:sync
```

### Linting

```sh
devenv shell -- lint
```

### Formatting

```sh
devenv shell -- format
```

### Typechecking

```sh
devenv shell -- typecheck
```

### Testing

For full parallel testing with xdist auto-sharding:

```sh
devenv shell -- test:full
```

For single-threaded testing with short tracebacks:

```sh
devenv shell -- test:fast
```

For running a single test:

```sh
devenv shell -- uv run python -m pytest -x --tb=short <path_to_test_file>
```

**Property-Based Testing**: Uses Hypothesis extensively. Control fuzz testing iterations: ```bash #
Increase hypothesis examples for more thorough testing HYPOTHESIS_MAX_EXAMPLES=5000 uv run pytest
tests/ -x -n auto --tb=short

# For specific hypothesis tests with higher iteration counts HYPOTHESIS_MAX_EXAMPLES=10000 uv run
pytest tests/unit/test_waves.py::TestHarmonicsToTable::test_harmonics_to_table_hypothesis -x -n auto
--tb=short ```

## Architecture

**Core DSP Modules** (`src/wtgen/dsp/`):
- `waves.py` - Harmonic wavetable synthesis from partial lists
- `mipmap.py` - Bandlimited mipmap generation with multiple rolloff methods
- `process.py` - Signal processing utilities (zero-crossing alignment, DC removal, RMS
  normalization)
- `fir.py` - FIR filter implementations

**Critical DSP Properties Tested**:
- Zero-crossing alignment across all mipmap levels
- RMS consistency validation
- DC offset removal throughout processing chain
- Antialiasing effectiveness across full MIDI range (0-127)
- Phase coherence preservation
- Spectral bandlimiting for alias prevention

## Type Checking Configuration

- **basedpyright**: Configured to suppress warnings from scientific libraries while catching real
  type errors
- **mypy**: Configured for strict typing with specific DSP module exemptions in `mypy.ini`
- Both tools support the scientific computing stack (numpy, scipy, numba)
- Planned: eventually this project is likely to migrate to [ty](https://docs.astral.sh/ty/)

## Key Testing Patterns

The codebase uses extensive property-based testing with Hypothesis to validate:
- Mathematical invariants in DSP processing
- Robust behavior with edge cases and unexpected inputs
- Energy balancing without DC introduction
- Range normalization with zero-mean constraint

When running tests, higher `HYPOTHESIS_MAX_EXAMPLES` values provide more thorough validation but
take longer to execute.
- ALWAYS run pytest, mypy, basedpyright, and ruff before considering any changes complete
- ALWAYS add tests for any new code and run them before considering the task complete
- When you encounter typechecking issues in python code, fix them
