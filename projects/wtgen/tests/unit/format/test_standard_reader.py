"""Unit tests for the standard wavetable reader module."""

from pathlib import Path

import numpy as np
import pytest

from wtgen.format import (
    ClassicDigitalMetadata,
    NormalizationMethod,
    WavetableType,
    load_wavetable_wav,
    save_wavetable_wav,
)
from wtgen.format.riff import RiffError


class TestLoadWavetableWav:
    """Tests for load_wavetable_wav function."""

    def test_basic_roundtrip(self, tmp_path: Path) -> None:
        """Test basic save and load roundtrip."""
        # Create test data
        mip0 = np.array(
            [
                [0.0, 0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25],
                [0.0, -0.25, -0.5, -0.75, -1.0, -0.75, -0.5, -0.25],
            ],
            dtype=np.float32,
        )

        # Save
        output_path = tmp_path / "test_roundtrip.wav"
        save_wavetable_wav(
            output_path,
            mipmaps=[mip0],
            wavetable_type=WavetableType.CUSTOM,
            name="Test Roundtrip",
            author="Test Author",
            sample_rate=48000,
        )

        # Load
        loaded = load_wavetable_wav(output_path)

        # Verify metadata
        assert loaded.wavetable_type == WavetableType.CUSTOM
        assert loaded.name == "Test Roundtrip"
        assert loaded.author == "Test Author"
        assert loaded.sample_rate == 48000

        # Verify structure
        assert loaded.num_frames == 2
        assert loaded.frame_length == 8
        assert loaded.num_mip_levels == 1
        assert len(loaded.mipmaps) == 1

        # Verify audio data
        np.testing.assert_array_almost_equal(loaded.mipmaps[0], mip0)

    def test_multi_mip_roundtrip(self, tmp_path: Path) -> None:
        """Test roundtrip with multiple mip levels."""
        num_frames = 4
        mip0 = np.random.randn(num_frames, 256).astype(np.float32)
        mip1 = np.random.randn(num_frames, 128).astype(np.float32)
        mip2 = np.random.randn(num_frames, 64).astype(np.float32)

        output_path = tmp_path / "test_multi_mip.wav"
        save_wavetable_wav(
            output_path,
            mipmaps=[mip0, mip1, mip2],
            wavetable_type=WavetableType.HIGH_RESOLUTION,
        )

        loaded = load_wavetable_wav(output_path)

        assert loaded.num_mip_levels == 3
        assert len(loaded.mipmaps) == 3
        assert loaded.mip_frame_lengths == [256, 128, 64]

        np.testing.assert_array_almost_equal(loaded.mipmaps[0], mip0)
        np.testing.assert_array_almost_equal(loaded.mipmaps[1], mip1)
        np.testing.assert_array_almost_equal(loaded.mipmaps[2], mip2)

    def test_type_metadata_roundtrip(self, tmp_path: Path) -> None:
        """Test roundtrip with type-specific metadata."""
        mip0 = np.random.randn(64, 256).astype(np.float32)

        output_path = tmp_path / "test_type_metadata.wav"
        save_wavetable_wav(
            output_path,
            mipmaps=[mip0],
            wavetable_type=WavetableType.CLASSIC_DIGITAL,
            type_metadata=ClassicDigitalMetadata(
                original_bit_depth=8,
                source_hardware="PPG Wave 2.2",
                harmonic_caps=[128, 64, 32],
            ),
        )

        loaded = load_wavetable_wav(output_path)

        assert loaded.wavetable_type == WavetableType.CLASSIC_DIGITAL
        type_meta = loaded.get_type_metadata()
        assert isinstance(type_meta, ClassicDigitalMetadata)
        assert type_meta.original_bit_depth == 8
        assert type_meta.source_hardware == "PPG Wave 2.2"
        assert type_meta.harmonic_caps == [128, 64, 32]

    def test_file_not_found(self) -> None:
        """Test that missing file raises error."""
        with pytest.raises(RiffError):
            load_wavetable_wav(Path("/nonexistent/path/file.wav"))

    def test_invalid_file(self, tmp_path: Path) -> None:
        """Test that invalid file raises error."""
        # Create a non-WAV file
        invalid_path = tmp_path / "invalid.wav"
        invalid_path.write_bytes(b"not a wav file")

        with pytest.raises(RiffError):
            load_wavetable_wav(invalid_path)

    def test_skip_validation(self, tmp_path: Path) -> None:
        """Test that validation can be skipped on load."""
        # Create valid file first
        mip0 = np.random.randn(4, 64).astype(np.float32)
        output_path = tmp_path / "test_skip_validate.wav"
        save_wavetable_wav(
            output_path,
            mipmaps=[mip0],
            wavetable_type=WavetableType.CUSTOM,
        )

        # Load with validation disabled (should succeed)
        loaded = load_wavetable_wav(output_path, validate=False)
        assert loaded is not None


class TestWavetableFileProperties:
    """Tests for WavetableFile properties."""

    def test_total_samples(self, tmp_path: Path) -> None:
        """Test total_samples calculation."""
        # 4 frames * (256 + 128 + 64) samples per mip level = 4 * 448 = 1792
        num_frames = 4
        mip0 = np.random.randn(num_frames, 256).astype(np.float32)
        mip1 = np.random.randn(num_frames, 128).astype(np.float32)
        mip2 = np.random.randn(num_frames, 64).astype(np.float32)

        output_path = tmp_path / "test_total_samples.wav"
        save_wavetable_wav(
            output_path,
            mipmaps=[mip0, mip1, mip2],
            wavetable_type=WavetableType.CUSTOM,
        )

        loaded = load_wavetable_wav(output_path)
        assert loaded.total_samples() == 4 * (256 + 128 + 64)

    def test_normalization_method(self, tmp_path: Path) -> None:
        """Test normalization_method property."""
        mip0 = np.random.randn(4, 64).astype(np.float32)

        output_path = tmp_path / "test_normalization.wav"
        save_wavetable_wav(
            output_path,
            mipmaps=[mip0],
            wavetable_type=WavetableType.CUSTOM,
            normalization_method=NormalizationMethod.PEAK,
        )

        loaded = load_wavetable_wav(output_path)
        assert loaded.normalization_method == NormalizationMethod.PEAK

    def test_optional_fields_none(self, tmp_path: Path) -> None:
        """Test that optional fields return None when not set."""
        mip0 = np.random.randn(4, 64).astype(np.float32)

        output_path = tmp_path / "test_optional.wav"
        save_wavetable_wav(
            output_path,
            mipmaps=[mip0],
            wavetable_type=WavetableType.CUSTOM,
        )

        loaded = load_wavetable_wav(output_path)
        assert loaded.name is None
        assert loaded.author is None
        assert loaded.description is None


class TestSamplePrecision:
    """Tests for sample value precision in roundtrip."""

    def test_sample_precision(self, tmp_path: Path) -> None:
        """Test that sample values maintain precision through roundtrip."""
        # Use specific values to test precision
        values = [0.0, 0.123456789, -0.987654321, 0.5, -0.5, 1.0, -1.0, 0.001]
        mip0 = np.array([values], dtype=np.float32)

        output_path = tmp_path / "test_precision.wav"
        save_wavetable_wav(
            output_path,
            mipmaps=[mip0],
            wavetable_type=WavetableType.CUSTOM,
        )

        loaded = load_wavetable_wav(output_path)

        # 32-bit float precision should be preserved
        np.testing.assert_array_almost_equal(loaded.mipmaps[0], mip0, decimal=6)

    def test_extreme_values(self, tmp_path: Path) -> None:
        """Test handling of extreme but valid float values."""
        # Values that are valid but near limits
        values = [0.0, 1e-30, -1e-30, 0.999999, -0.999999]
        mip0 = np.array([values], dtype=np.float32)

        output_path = tmp_path / "test_extreme.wav"
        save_wavetable_wav(
            output_path,
            mipmaps=[mip0],
            wavetable_type=WavetableType.CUSTOM,
            validate=False,  # Skip validation for extreme values
        )

        loaded = load_wavetable_wav(output_path, validate=False)
        np.testing.assert_array_almost_equal(loaded.mipmaps[0], mip0, decimal=6)
