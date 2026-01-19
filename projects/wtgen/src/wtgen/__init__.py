"""wtgen - Wavetable generation and research toolkit.

This package provides tools for generating, analyzing, and exporting
wavetable data for use with digital synthesizers.

Wavetable Format
----------------
The format submodule provides functions for reading and writing wavetable
files in the standardized RIFF/WAV format with embedded protobuf metadata.

Example Usage
-------------
>>> from wtgen.format import save_wavetable_wav, load_wavetable_wav, WavetableType
>>> import numpy as np
>>>
>>> # Create some mipmap data
>>> mip0 = np.random.randn(64, 256).astype(np.float32)
>>>
>>> # Export wavetable
>>> save_wavetable_wav(
...     "my_wavetable.wav",
...     mipmaps=[mip0],
...     wavetable_type=WavetableType.HIGH_RESOLUTION,
...     name="My Custom Wavetable",
... )
>>>
>>> # Load wavetable
>>> wavetable = load_wavetable_wav("my_wavetable.wav")
>>> print(f"Loaded: {wavetable.name}, {wavetable.num_frames} frames")
"""

# Re-export format module for convenience
from wtgen.format import (
    ClassicDigitalMetadata,
    HighResolutionMetadata,
    InterpolationHint,
    NormalizationMethod,
    PcmSampleMetadata,
    ValidationError,
    VintageEmulationMetadata,
    WavetableFile,
    WavetableType,
    load_wavetable_wav,
    save_wavetable_wav,
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
]
