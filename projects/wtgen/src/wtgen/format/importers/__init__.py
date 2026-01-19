"""External format importers.

This subpackage provides tools for importing wavetables from external formats
and converting them to the standardized wavetable interchange format.

Supported formats:
- Raw PCM: Headerless binary audio data (8-bit, 16-bit, 24-bit, 32-bit)
- WAV: Standard WAV files containing wavetable data (Serum, Vital, etc.)

Example usage:
    >>> from wtgen.format.importers import import_raw_pcm, import_hires_wav
    >>> from wtgen.format import save_wavetable_wav
    >>>
    >>> # Import raw 8-bit PCM data
    >>> mipmaps, metadata = import_raw_pcm(
    ...     "legacy.raw",
    ...     frame_length=256,
    ...     num_frames=64,
    ...     bit_depth=8
    ... )
    >>> save_wavetable_wav("converted.wav", mipmaps, **metadata)
    >>>
    >>> # Import high-resolution WAV
    >>> mipmaps, metadata = import_hires_wav("serum_table.wav", num_frames=256)
    >>> save_wavetable_wav("converted.wav", mipmaps, **metadata)
"""

from wtgen.format.importers.raw import detect_raw_format, import_raw_pcm
from wtgen.format.importers.wav import (
    detect_wav_wavetable,
    import_hires_wav,
    import_wav_with_mips,
)

__all__ = [
    "import_raw_pcm",
    "detect_raw_format",
    "import_hires_wav",
    "detect_wav_wavetable",
    "import_wav_with_mips",
]
