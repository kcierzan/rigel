"""Wavetable analysis utilities.

This subpackage provides tools for analyzing wavetable data, including:
- Harmonic content analysis via FFT
- Wavetable type inference from data characteristics
- Type-specific metadata suggestions
"""

from wtgen.format.analysis.harmonics import analyze_harmonic_content
from wtgen.format.analysis.inference import infer_wavetable_type, suggest_type_metadata

__all__ = [
    "analyze_harmonic_content",
    "infer_wavetable_type",
    "suggest_type_metadata",
]
