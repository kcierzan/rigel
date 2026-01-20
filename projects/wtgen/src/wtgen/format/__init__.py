"""Wavetable interchange format module.

This module provides functionality for reading and writing wavetable files
in the standardized RIFF/WAV format with embedded protobuf metadata.

Format Overview
---------------
The wavetable interchange format uses standard WAV files with a custom RIFF chunk:

    +----------------------------------------+
    | RIFF Header ("WAVE")                   |
    +----------------------------------------+
    | fmt  chunk (audio format)              |
    +----------------------------------------+
    | data chunk (waveform samples)          |
    |   - 32-bit float                       |
    |   - Mip-major ordering                 |
    +----------------------------------------+
    | WTBL chunk (protobuf metadata)         |
    |   - Schema version                     |
    |   - Wavetable type                     |
    |   - Frame structure                    |
    |   - Optional metadata                  |
    +----------------------------------------+

Example Usage
-------------
>>> from wtgen.format import save_wavetable_wav, load_wavetable_wav, WavetableType
>>> # Export a wavetable
>>> save_wavetable_wav(
...     "my_wavetable.wav",
...     mipmaps=mipmaps,
...     wavetable_type=WavetableType.HIGH_RESOLUTION,
...     name="My Custom Wavetable",
...     author="Your Name",
... )
>>> # Load a wavetable
>>> wavetable = load_wavetable_wav("my_wavetable.wav")
>>> print(wavetable.metadata.name)
"""

from wtgen.format.importers import detect_wav_wavetable
from wtgen.format.standard.reader import WavetableFile, load_wavetable_wav
from wtgen.format.standard.writer import save_wavetable_wav
from wtgen.format.types import (
    ClassicDigitalMetadata,
    HighResolutionMetadata,
    InterpolationHint,
    NormalizationMethod,
    PcmSampleMetadata,
    VintageEmulationMetadata,
    WavetableType,
)
from wtgen.format.validation import (
    ValidationError,
    validate_audio_data,
    validate_metadata,
)

__all__ = [
    # Types
    "WavetableType",
    "NormalizationMethod",
    "InterpolationHint",
    "ClassicDigitalMetadata",
    "HighResolutionMetadata",
    "VintageEmulationMetadata",
    "PcmSampleMetadata",
    # Reader
    "load_wavetable_wav",
    "WavetableFile",
    # Writer
    "save_wavetable_wav",
    # Validation
    "validate_metadata",
    "validate_audio_data",
    "ValidationError",
    # Analysis
    "detect_wav_wavetable",
]
