## Project Overview

wtgen is a Python library for wavetable generation and processing, specializing in harmonic wavetable synthesis and bandlimited mipmap generation for alias-free audio playback. The codebase focuses on high-quality DSP processing with extensive testing coverage.

## Development Setup

***CURRENTLY BROKEN ON NIXOS PENDING [devenv](https://devenv.sh/) SETUP!***

Once devenv is configured, these same tools should be usable through
dedicated devenv entrypoints.

**Dependencies**: Uses `uv` for dependency management. Install with:
```bash
uv sync --extra dev
uv pip install -e .
```

**Type Checking**: Project supports both mypy and basedpyright:
```bash
uv tool run mypy src/
uv tool run basedpyright src/
```

**Code Formatting**:
```bash
uv run ruff format
uv run ruff check --fix .
```

## Testing Framework

**Basic Test Execution**:
```bash
# Run all tests
uv run python -m pytest

# Run with parallel execution (faster)
uv run python -m pytest -n auto

# Run specific test module
uv run python -m pytest tests/unit/test_process.py

# Stop on first failure with short traceback
uv run python - pytest -x --tb=short
```

**Property-Based Testing**: Uses Hypothesis extensively. Control fuzz testing iterations:
```bash
# Increase hypothesis examples for more thorough testing
HYPOTHESIS_MAX_EXAMPLES=5000 uv run pytest tests/ -x -n auto --tb=short

# For specific hypothesis tests with higher iteration counts
HYPOTHESIS_MAX_EXAMPLES=10000 uv run pytest tests/unit/test_waves.py::TestHarmonicsToTable::test_harmonics_to_table_hypothesis -x -n auto --tb=short
```

**Test Categories**:
- `tests/unit/test_process.py` - DSP processing functions (30 tests)
- `tests/unit/test_waves.py` - Harmonic synthesis (22 tests)
- `tests/unit/test_mipmap.py` - Mipmap generation (21 tests)
- `tests/integration/` - End-to-end pipeline validation

## Architecture

**Core DSP Modules** (`src/wtgen/dsp/`):
- `waves.py` - Harmonic wavetable synthesis from partial lists
- `mipmap.py` - Bandlimited mipmap generation with multiple rolloff methods
- `process.py` - Signal processing utilities (zero-crossing alignment, DC removal, RMS normalization)
- `fir.py` - FIR filter implementations

**Key Constants**:
- `WAVETABLE_SIZE = 2048` - Standard wavetable length
- `TARGET_RMS = 0.35` - Standard RMS level for generated wavetables

**Critical DSP Properties Tested**:
- Zero-crossing alignment across all mipmap levels
- RMS consistency validation
- DC offset removal throughout processing chain
- Antialiasing effectiveness across full MIDI range (0-127)
- Phase coherence preservation
- Spectral bandlimiting for alias prevention

## Type Checking Configuration

- **basedpyright**: Configured to suppress warnings from scientific libraries while catching real type errors
- **mypy**: Configured for strict typing with specific DSP module exemptions in `mypy.ini`
- Both tools support the scientific computing stack (numpy, scipy, numba)
- Planned: eventually this project is likely to migrate to [ty](https://docs.astral.sh/ty/)

## Key Testing Patterns

The codebase uses extensive property-based testing with Hypothesis to validate:
- Mathematical invariants in DSP processing
- Robust behavior with edge cases and unexpected inputs
- Energy balancing without DC introduction
- Range normalization with zero-mean constraint

When running tests, higher `HYPOTHESIS_MAX_EXAMPLES` values provide more thorough validation but take longer to execute.
- ALWAYS run pytest, mypy, basedpyright, and ruff before considering any changes complete
- ALWAYS add tests for any new code and run them before considering the task complete
- When you encounter typechecking issues in python code, fix them
