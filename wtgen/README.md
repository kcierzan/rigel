# wtgen

A Python library for wavetable generation and processing

## Features

- Harmonic wavetable synthesis from partial lists
- Bandlimited mipmap generation for alias-free playback
- Command-line interface for wavetable generation
- Multiple waveform types (sawtooth, square, pulse, triangle, polyblep)
- Configurable rolloff methods for antialiasing
- Wavetable inspection and analysis tools

## Development Environment

wtgen ships with a [devenv](https://devenv.sh/) + flakes workflow so contributors on both NixOS and nix-darwin can get a fully reproducible toolchain.

### Prerequisites

- Install [Nix](https://zero-to-nix.com/start) with flakes enabled.
- Optional: install the standalone `devenv` CLI via `nix profile install github:cachix/devenv/latest` for `devenv shell` / `devenv shell -- …`.

### Quick start

Install [devenv](https://devenv.sh/getting-started/)

Then launch the devenv shell:
```bash
devenv shell
```

On first entry the shell will:
- create `.venv/` if missing,
- run `uv sync --frozen --group dev`,
- install wtgen in editable mode (`uv pip install -e .`),
- activate the virtual environment automatically.

Subsequent entries reuse the same `.venv` unless you remove it.

### Common tasks

All of these commands can be launched from inside the shell, or via `devenv shell -- …` if you prefer not to activate it manually.

- `devenv shell -- uv:sync` – rebuild `.venv` from `uv.lock` (re-runs sync + editable install).
- `devenv shell -- lint` – Ruff lint.
- `devenv shell -- format` – Ruff formatter.
- `devenv shell -- typecheck` – mypy followed by basedpyright (use `typecheck:mypy` or `typecheck:pyright` individually as needed).
- `devenv shell -- test:full` – full pytest suite with xdist auto-sharding.
- `devenv shell -- test:fast` – single-process pytest with `-x --tb=short`.

If you are not using Nix, you can mimic the same environment locally:

```bash
uv sync --frozen --group dev
uv pip install -e .
source .venv/bin/activate
```

## CLI Usage

wtgen provides a command-line interface for generating and analyzing wavetables. The CLI supports three main commands:

### Generate Standard Waveforms

Generate wavetables from standard waveform types:

```bash
# Generate a basic sawtooth wavetable (default: 8 octaves, 2048 samples)
uv run wtgen generate sawtooth --output sawtooth.npz

# Generate a square wave with custom parameters
uv run wtgen generate square \
  --output square_wave.npz \
  --octaves 6 \
  --duty 0.3 \
  --rolloff hann \
  --size 1024

# Generate a pulse wave with specific frequency
uv run wtgen wtgen.cli generate pulse \
  --output pulse.npz \
  --frequency 2.0 \
  --duty 0.1 \
  --octaves 10

# Generate a triangle wave (smooth, contains only odd harmonics)
uv run wtgen generate triangle \
  --output triangle.npz \
  --octaves 8 \
  --rolloff blackman

# Generate decimated sawtooth mipmaps with a custom high harmonic tilt
# starting at 50% of nyquist boosting to 6dB by nyquist, creating
# wave files in addition to .npz archives
uv run wtgen generate sawtooth \
    --output test_saw.npz \
    --octaves 8 \
    --high-tilt 0.5:6.0 \
    --output-wav \
    --decimate

# Generate square mipmaps with a 6dB boost @ 60hz [Q1.0]
# and a 2dB cut @ 5k [Q0.3]
uv run wtgen generate square \
    --output test_saw.npz \
    --octaves 8 \
    --eq 60:6.0:1.0,5000:-2.0:0.3

# Generate a polyblep sawtooth (band-limited)
uv run wtgen generate polyblep_saw --output polyblep.npz
```

### Generate Harmonic Wavetables

Create wavetables from custom harmonic content:

```bash
# Generate using default sawtooth harmonics (1/n series)
uv run wtgen harmonic --output harmonic_saw.npz

# Create custom harmonic content (harmonic:amplitude:phase format)
uv run wtgen harmonic \
  --output custom_harmonic.npz \
  --partials "1:1.0:0.0,2:0.5:0.0,3:0.33:0.0,4:0.25:0.0" \
  --octaves 8

# Create a simple sine wave with a 3rd harmonic
uv run wtgen harmonic \
  --output sine_plus_third.npz \
  --partials "1:1.0:0.0,3:0.2:1.57" \
  --rolloff tukey
```

### Inspect Wavetable Files

Analyze and display information about generated wavetables:

```bash
# Show detailed information about a wavetable
uv run wtgen info sawtooth.npz

# Example output:
# Wavetable: sawtooth.npz
#   Mipmap levels: 9
#   Table size: 2048
#   Waveform: WaveformType.sawtooth
#   Rolloff: RolloffMethod.tukey
#   Frequency: 1.0
#   Duty cycle: 0.5
#   RMS levels:
#     Level 0: 0.500
#     Level 1: 0.491
#     Level 2: 0.494
#     ...
```

### CLI Options Reference

#### Waveform Types
- `sawtooth` - Classic sawtooth wave (contains all harmonics, 1/n amplitude)
- `square` - Square wave (contains odd harmonics, 1/n amplitude, configurable duty cycle)
- `pulse` - Pulse wave (configurable duty cycle, harmonic content varies with pulse width)
- `triangle` - Triangle wave (contains only odd harmonics, 1/n² amplitude, very smooth)
- `polyblep_saw` - Band-limited sawtooth using polyBLEP algorithm

#### Rolloff Methods
- `tukey` - Tukey window (good balance)
- `hann` - Hann window (smooth rolloff)
- `blackman` - Blackman window (steep rolloff)
- `raised_cosine` - Raised cosine rolloff
- `brick_wall` - Hard cutoff (may cause artifacts)
- `none` - No rolloff filtering

#### Common Parameters
- `--octaves N` - Number of mipmap octaves (default: 8)
- `--size N` - Wavetable size, must be power of 2 (default: 2048)
- `--output PATH` - Output .npz file path
- `--rolloff METHOD` - Antialiasing rolloff method
- `--frequency F` - Base frequency for generation (default: 1.0)
- `--duty F` - Duty cycle for square/pulse waves (0.0-1.0, default: 0.5)

## Mipmap Generation

wtgen automatically generates bandlimited mipmaps for alias-free playback across the full MIDI range. Here's how the mipmap system works:

### Understanding Mipmaps

Mipmaps are pre-computed, progressively band-limited versions of the base wavetable:

- **Level 0**: Full spectrum (highest notes)
- **Level 1**: Limited to Nyquist/2 (one octave down)
- **Level 2**: Limited to Nyquist/4 (two octaves down)
- **Level N**: Limited to Nyquist/(2^N) (N octaves down)

### Automatic Mipmap Selection

When playing back wavetables, select the appropriate mipmap level based on fundamental frequency:

```python
import numpy as np

def select_mipmap_level(fundamental_freq, sample_rate, num_levels):
    """Select appropriate mipmap level for given fundamental frequency."""
    nyquist = sample_rate / 2

    # Find the highest level where fundamental * max_harmonic < nyquist
    for level in range(num_levels):
        max_safe_freq = nyquist / (2 ** level)
        if fundamental_freq * 16 < max_safe_freq:  # Assume ~16 harmonics max
            return level

    return num_levels - 1  # Use most filtered level
```

### Mipmap Quality Settings

Control the quality vs. computational cost trade-off:

```bash
# High quality: more levels, smoother transitions
uv run wtgen generate sawtooth \
  --octaves 12 \
  --rolloff blackman \
  --size 4096

# Balanced: good quality, reasonable size
uv run wtgen generate sawtooth \
  --octaves 8 \
  --rolloff tukey \
  --size 2048

# Compact: fewer levels, smaller files
uv run wtgen generate sawtooth \
  --octaves 5 \
  --rolloff hann \
  --size 1024
```

### Processing Pipeline

Each mipmap level goes through a standardized processing pipeline:

1. **Band-limiting**: Apply selected rolloff filter
2. **DC Removal**: Eliminate any DC offset
3. **Normalization**: Scale to target RMS level (0.35)
4. **Zero-crossing Alignment**: Ensure consistent phase relationships

This ensures consistent playback characteristics across all mipmap levels.

## Python API Usage

For programmatic access, wtgen provides a clean Python API for wavetable generation:

### Basic Wavetable Generation

```python
import numpy as np
from wtgen.dsp.waves import harmonics_to_table, generate_polyblep_sawtooth_wavetable
from wtgen.dsp.mipmap import build_mipmap
from wtgen.dsp.process import align_to_zero_crossing, dc_remove, normalize

# Generate a sawtooth wave from harmonics
partials = [(i, 1.0/i, 0.0) for i in range(1, 17)]  # 1/n amplitude series
wavetable = harmonics_to_table(partials, table_size=2048)

# Create mipmaps for alias-free playback
mipmaps = build_mipmap(wavetable, num_octaves=8, rolloff_method='tukey')

# Process each mipmap level
processed_mipmaps = []
for mipmap in mipmaps:
    processed = align_to_zero_crossing(dc_remove(normalize(mipmap)))
    processed_mipmaps.append(processed)

# Save to file
np.savez_compressed('my_wavetable.npz',
                   mipmaps=np.array(processed_mipmaps, dtype=np.float32))
```

### Custom Harmonic Content

```python
from wtgen.dsp.waves import harmonics_to_table

# Create a custom waveform with specific harmonics
# Format: (harmonic_number, amplitude, phase_radians)
custom_partials = [
    (1, 1.0, 0.0),      # Fundamental
    (2, 0.5, 0.0),      # Second harmonic
    (3, 0.33, np.pi),   # Third harmonic (inverted)
    (5, 0.2, np.pi/2),  # Fifth harmonic (90° phase)
]

wavetable = harmonics_to_table(custom_partials, table_size=2048)
```

### Different Waveform Types

```python
from wtgen.plotting import (
    generate_sawtooth_wavetable,
    generate_square_wavetable,
    generate_pulse_wavetable,
    generate_triangle_wavetable
)

# Generate different base waveforms
_, sawtooth = generate_sawtooth_wavetable(frequency=1.0)
_, square = generate_square_wavetable(frequency=1.0, duty_cycle=0.5)
_, pulse = generate_pulse_wavetable(frequency=1.0, duty_cycle=0.1)
_, triangle = generate_triangle_wavetable(frequency=1.0)
```

### Advanced Mipmap Configuration

```python
from wtgen.dsp.mipmap import build_mipmap

# Try different rolloff methods
rolloff_methods = ['tukey', 'hann', 'blackman', 'raised_cosine']

for method in rolloff_methods:
    mipmaps = build_mipmap(
        wavetable,
        num_octaves=10,
        rolloff_method=method,
        sample_rate=44100  # Optional: specify sample rate
    )
    print(f"{method}: {len(mipmaps)} mipmap levels generated")
```

### Loading and Analyzing Wavetables

```python
import numpy as np

# Load a generated wavetable
data = np.load('my_wavetable.npz', allow_pickle=True)
mipmaps = data['mipmaps']
metadata = data.get('metadata', {})

print(f"Mipmap levels: {len(mipmaps)}")
print(f"Table size: {mipmaps[0].shape[0]}")

# Analyze RMS levels
for i, mipmap in enumerate(mipmaps):
    rms = np.sqrt(np.mean(mipmap**2))
    print(f"Level {i} RMS: {rms:.3f}")

# Frequency analysis
import scipy.fft
spectrum = np.abs(scipy.fft.fft(mipmaps[0]))
freqs = scipy.fft.fftfreq(len(spectrum))
# Plot or analyze spectrum as needed
```

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. Install dependencies with:

```bash
uv sync --group dev
```

For testing, you also need to install the package in development mode:

```bash
uv pip install -e .
```

### Prerequisites

Make sure you have the development dependencies installed:

```bash
uv sync --group dev
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
- **`tests/unit/test_cli.py`** - Command-line interface (21 tests)

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
- ✅ **Spectral bandlimiting** for alias prevention

### Code Formatting

```bash
uv run ruff check --fix .
uv run ruff format
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
