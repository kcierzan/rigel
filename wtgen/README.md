# wtgen

A Python library for wavetable generation and processing

## Features

- Harmonic wavetable synthesis from partial lists
- Bandlimited mipmap generation for alias-free playback

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. Install dependencies with:

```bash
uv sync --extra dev
```

For testing, you also need to install the package in development mode:

```bash
uv pip install -e .
```

### Prerequisites

Make sure you have the development dependencies installed:

```bash
uv sync --extra dev
uv pip install -e .
```

## Code Organization

The wavetable generation code is organized into clean, modular components:

### Core Modules:
- **`src/wtgen/dsp/`**: DSP processing (wavetable synthesis, mipmaps, etc.)

### Testing Framework

The project includes comprehensive testing with both traditional unit tests and property-based fuzz testing using Hypothesis.

#### Basic Test Execution

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test module
uv run pytest tests/unit/test_process.py
```

#### Parallel Test Execution

For faster test execution, pytest-xdist is already included. Run tests in parallel:

```bash
# Run tests in parallel (auto-detect CPU cores)
uv run pytest -n auto

# Run tests with specific number of workers
uv run pytest -n 4
```

#### Fuzz Testing with Hypothesis

The test suite includes extensive property-based testing using [Hypothesis](https://hypothesis.readthedocs.io/) to automatically generate test cases and find edge cases:

```bash
# Run tests with increased fuzz testing iterations
uv run pytest --hypothesis-show-statistics

# Run with verbose hypothesis output
uv run pytest -v --hypothesis-verbosity=verbose

# For controlling hypothesis examples and deadline, modify test settings in code
# or use profiles (see hypothesis documentation)
```

#### Test Coverage

```bash
# Run tests with coverage report
uv run pytest --cov=src/wtgen --cov-report=html

# View coverage in terminal
uv run pytest --cov=src/wtgen --cov-report=term-missing
```

#### Specific Test Categories

```bash
# Run only unit tests
uv run pytest tests/unit/

# Run only integration tests
uv run pytest tests/integration/

# Run only hypothesis fuzz tests
uv run pytest -k "hypothesis"

# Run tests for specific DSP modules
uv run pytest tests/unit/test_process.py tests/unit/test_mipmap.py tests/unit/test_waves.py
```

#### Test Debugging

```bash
# Stop on first failure
uv run pytest -x

# Show local variables on failure
uv run pytest -l

# Enter debugger on failure
uv run pytest --pdb

# Run last failed tests only
uv run pytest --lf
```

### Testing Components

#### Unit Tests
- **`tests/unit/test_process.py`** - DSP processing functions (30 tests)
- **`tests/unit/test_waves.py`** - Harmonic synthesis (22 tests)
- **`tests/unit/test_mipmap.py`** - Mipmap generation (21 tests)

#### Integration Tests
- **`tests/integration/test_full_generation.py`** - End-to-end pipeline validation (9 tests)

#### Property-Based Testing
Each test module includes Hypothesis-powered fuzz testing that:
- Generates thousands of random test cases automatically
- Tests edge cases and boundary conditions
- Validates mathematical properties and invariants
- Ensures robust behavior with unexpected inputs

#### Test Coverage Areas
- ✅ **Zero-crossing alignment** across all mipmap levels
- ✅ **RMS consistency** (TARGET_RMS = 0.35) validation
- ✅ **DC offset removal** throughout processing chain
- ✅ **Antialiasing effectiveness** across full MIDI range (0-127)
- ✅ **Phase coherence** preservation
- ✅ **Range normalization** with zero-mean constraint
- ✅ **Energy balancing** without DC introduction
- ✅ **Spectral bandlimiting** for alias prevention

### Code Formatting

```bash
uv run black .
uv run ruff check --fix .
```

### Type Checking

The project supports both mypy and basedpyright for type checking:

```bash
# Using mypy
uv run mypy src/

# Using basedpyright (configured to suppress noisy warnings from scientific libraries)
uv run basedpyright src/
```

Basedpyright is configured in `pyproject.toml` to suppress warnings about missing type stubs and unknown types from third-party scientific libraries while still catching real type errors.
