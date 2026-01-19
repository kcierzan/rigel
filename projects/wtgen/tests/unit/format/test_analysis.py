"""Unit tests for wavetable analysis utilities."""

import numpy as np

from wtgen.format.analysis import analyze_harmonic_content, infer_wavetable_type
from wtgen.format.analysis.inference import suggest_type_metadata
from wtgen.format.types import WavetableType


class TestTypeInference:
    """Tests for wavetable type inference."""

    def test_infer_classic_digital(self) -> None:
        """Test inference of classic digital type."""
        # PPG-style: 64 frames of 256 samples
        wavetable = np.random.randn(64, 256).astype(np.float32)

        wt_type, analysis = infer_wavetable_type(
            wavetable,
            source_bit_depth=8,
        )

        assert wt_type == WavetableType.CLASSIC_DIGITAL
        assert analysis["confidence"] == "high"

    def test_infer_high_resolution(self) -> None:
        """Test inference of high-resolution type."""
        # Modern synth: 256 frames of 2048 samples
        wavetable = np.random.randn(256, 2048).astype(np.float32)

        wt_type, analysis = infer_wavetable_type(wavetable)

        assert wt_type == WavetableType.HIGH_RESOLUTION
        assert "high resolution" in analysis["evidence"][0].lower()

    def test_infer_pcm_sample(self) -> None:
        """Test inference of PCM sample type."""
        # Single-cycle with a clean sine wave (no aliasing artifacts)
        # This avoids triggering vintage emulation detection
        frame_length = 512
        t = np.linspace(0, 2 * np.pi, frame_length, endpoint=False)
        sine = np.sin(t).astype(np.float32)
        wavetable = sine.reshape(1, frame_length)

        wt_type, analysis = infer_wavetable_type(wavetable)

        assert wt_type == WavetableType.PCM_SAMPLE

    def test_infer_from_hardware_ppg(self) -> None:
        """Test inference from PPG hardware signature."""
        wavetable = np.random.randn(32, 512).astype(np.float32)

        wt_type, analysis = infer_wavetable_type(
            wavetable,
            source_hardware="PPG Wave 2.3",
        )

        assert wt_type == WavetableType.CLASSIC_DIGITAL
        assert analysis["confidence"] == "high"
        assert "ppg" in analysis["evidence"][0].lower()

    def test_infer_from_hardware_serum(self) -> None:
        """Test inference from Serum hardware signature."""
        wavetable = np.random.randn(64, 512).astype(np.float32)

        wt_type, analysis = infer_wavetable_type(
            wavetable,
            source_hardware="Serum",
        )

        assert wt_type == WavetableType.HIGH_RESOLUTION

    def test_infer_custom_fallback(self) -> None:
        """Test fallback to custom type."""
        # Unusual dimensions that don't match any pattern
        wavetable = np.random.randn(50, 300).astype(np.float32)

        wt_type, analysis = infer_wavetable_type(wavetable)

        assert wt_type == WavetableType.CUSTOM
        assert analysis["confidence"] == "low"


class TestHarmonicAnalysis:
    """Tests for harmonic content analysis."""

    def test_analyze_sine_wave(self) -> None:
        """Test analysis of pure sine wave."""
        frame_length = 1024
        t = np.linspace(0, 2 * np.pi, frame_length, endpoint=False)
        sine = np.sin(t).astype(np.float32)
        wavetable = sine.reshape(1, frame_length)

        result = analyze_harmonic_content(wavetable)

        # Sine wave should have low number of significant harmonics
        assert result["num_significant_harmonics"] < 10
        assert not result["has_aliasing_artifacts"]

    def test_analyze_saw_wave(self) -> None:
        """Test analysis of saw wave (rich harmonics)."""
        frame_length = 1024
        t = np.linspace(-1, 1, frame_length, endpoint=False)
        saw = t.astype(np.float32)
        wavetable = saw.reshape(1, frame_length)

        result = analyze_harmonic_content(wavetable)

        # Saw wave should have many harmonics
        assert result["num_significant_harmonics"] > 50
        assert result["has_high_harmonics"] is True

    def test_analyze_square_wave_aliased(self) -> None:
        """Test analysis of aliased square wave."""
        frame_length = 64  # Low resolution = aliasing
        square = np.array([1.0] * 32 + [-1.0] * 32, dtype=np.float32)
        wavetable = square.reshape(1, frame_length)

        result = analyze_harmonic_content(wavetable)

        # Aliased square should show high-frequency energy
        assert result["high_freq_energy_ratio"] > 0.05

    def test_analyze_specific_frame(self) -> None:
        """Test analysis of specific frame in multi-frame wavetable."""
        wavetable = np.random.randn(4, 256).astype(np.float32)

        result = analyze_harmonic_content(wavetable, frame_index=2)

        assert "estimated_max_harmonic" in result


class TestSuggestTypeMetadata:
    """Tests for type-specific metadata suggestions."""

    def test_suggest_classic_digital_metadata(self) -> None:
        """Test metadata suggestion for classic digital."""
        wavetable = np.random.randn(64, 256).astype(np.float32)

        metadata = suggest_type_metadata(
            WavetableType.CLASSIC_DIGITAL,
            wavetable,
            source_info={"bit_depth": 8, "hardware": "PPG Wave"},
        )

        assert metadata["original_bit_depth"] == 8
        assert metadata["source_hardware"] == "PPG Wave"
        assert "harmonic_caps" in metadata

    def test_suggest_high_resolution_metadata(self) -> None:
        """Test metadata suggestion for high resolution."""
        wavetable = np.random.randn(256, 2048).astype(np.float32)

        metadata = suggest_type_metadata(
            WavetableType.HIGH_RESOLUTION,
            wavetable,
        )

        assert "max_harmonics" in metadata
        assert "interpolation_hint" in metadata
        assert metadata["interpolation_hint"] == "cubic"

    def test_suggest_pcm_sample_metadata(self) -> None:
        """Test metadata suggestion for PCM sample."""
        wavetable = np.random.randn(1, 512).astype(np.float32)

        metadata = suggest_type_metadata(
            WavetableType.PCM_SAMPLE,
            wavetable,
            source_info={"sample_rate": 48000},
        )

        assert metadata["original_sample_rate"] == 48000
        assert "root_note" in metadata

    def test_suggest_custom_metadata(self) -> None:
        """Test metadata suggestion for custom type."""
        wavetable = np.random.randn(32, 512).astype(np.float32)

        metadata = suggest_type_metadata(
            WavetableType.CUSTOM,
            wavetable,
        )

        # Custom type should return empty dict
        assert metadata == {}
