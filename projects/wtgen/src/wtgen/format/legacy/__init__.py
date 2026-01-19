"""Legacy wavetable format import utilities.

.. deprecated::
    This module is deprecated. Use the following imports instead:

    - ``wtgen.format.importers`` for import_raw_pcm, import_hires_wav, etc.
    - ``wtgen.format.analysis`` for infer_wavetable_type, analyze_harmonic_content
"""

# Re-export from new locations for backward compatibility
from wtgen.format.analysis import analyze_harmonic_content, infer_wavetable_type
from wtgen.format.analysis.inference import suggest_type_metadata
from wtgen.format.importers import (
    detect_raw_format,
    detect_wav_wavetable,
    import_hires_wav,
    import_raw_pcm,
    import_wav_with_mips,
)

__all__ = [
    "import_raw_pcm",
    "detect_raw_format",
    "import_hires_wav",
    "detect_wav_wavetable",
    "import_wav_with_mips",
    "infer_wavetable_type",
    "analyze_harmonic_content",
    "suggest_type_metadata",
]
