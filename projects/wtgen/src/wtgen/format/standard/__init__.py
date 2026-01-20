"""Standard wavetable format reader and writer.

This subpackage provides functionality for reading and writing wavetables in
the standardized RIFF/WAV format with embedded protobuf metadata (WTBL chunk).
"""

from wtgen.format.standard.reader import WavetableFile, load_wavetable_wav
from wtgen.format.standard.writer import save_wavetable_wav

__all__ = [
    "WavetableFile",
    "load_wavetable_wav",
    "save_wavetable_wav",
]
