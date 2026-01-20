"""Unit tests for the standard wavetable writer module."""

from pathlib import Path

import numpy as np
import pytest

from wtgen.format import (
    ClassicDigitalMetadata,
    HighResolutionMetadata,
    NormalizationMethod,
    WavetableType,
    save_wavetable_wav,
)


class TestSaveWavetableWav:
    """Tests for save_wavetable_wav function."""

    def test_basic_save(self, tmp_path: Path) -> None:
        """Test basic wavetable save with minimal metadata."""
        # Create simple mipmap data: 2 frames of 8 samples
        mip0 = np.array(
            [
                [0.0, 0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25],
                [0.0, -0.25, -0.5, -0.75, -1.0, -0.75, -0.5, -0.25],
            ],
            dtype=np.float32,
        )
        mipmaps = [mip0]

        output_path = tmp_path / "test_basic.wav"
        save_wavetable_wav(
            output_path,
            mipmaps=mipmaps,
            wavetable_type=WavetableType.CUSTOM,
        )

        assert output_path.exists()
        # File should be non-empty
        assert output_path.stat().st_size > 0

    def test_save_with_metadata(self, tmp_path: Path) -> None:
        """Test save with all optional metadata."""
        mip0 = np.array(
            [[0.0, 0.5, 1.0, 0.5], [0.0, -0.5, -1.0, -0.5]],
            dtype=np.float32,
        )
        mipmaps = [mip0]

        output_path = tmp_path / "test_metadata.wav"
        save_wavetable_wav(
            output_path,
            mipmaps=mipmaps,
            wavetable_type=WavetableType.HIGH_RESOLUTION,
            name="Test Wavetable",
            author="Test Author",
            description="A test wavetable for unit testing",
            normalization_method=NormalizationMethod.PEAK,
            source_bit_depth=16,
            tuning_reference=440.0,
            sample_rate=48000,
        )

        assert output_path.exists()

    def test_save_with_type_metadata(self, tmp_path: Path) -> None:
        """Test save with type-specific metadata."""
        mip0 = np.array(
            [[0.0] * 256, [0.5] * 256],
            dtype=np.float32,
        )
        mipmaps = [mip0]

        output_path = tmp_path / "test_classic.wav"
        save_wavetable_wav(
            output_path,
            mipmaps=mipmaps,
            wavetable_type=WavetableType.CLASSIC_DIGITAL,
            name="PPG Tribute",
            type_metadata=ClassicDigitalMetadata(
                original_bit_depth=8,
                source_hardware="PPG Wave 2.2",
                harmonic_caps=[128, 64, 32],
            ),
        )

        assert output_path.exists()

    def test_save_multi_mip(self, tmp_path: Path) -> None:
        """Test save with multiple mip levels."""
        num_frames = 4
        mip0 = np.random.randn(num_frames, 256).astype(np.float32)
        mip1 = np.random.randn(num_frames, 128).astype(np.float32)
        mip2 = np.random.randn(num_frames, 64).astype(np.float32)
        mipmaps = [mip0, mip1, mip2]

        output_path = tmp_path / "test_multi_mip.wav"
        save_wavetable_wav(
            output_path,
            mipmaps=mipmaps,
            wavetable_type=WavetableType.HIGH_RESOLUTION,
        )

        assert output_path.exists()

    def test_validation_error_empty_mipmaps(self, tmp_path: Path) -> None:
        """Test that empty mipmaps raises ValueError."""
        output_path = tmp_path / "test_empty.wav"

        with pytest.raises(ValueError, match="cannot be empty"):
            save_wavetable_wav(
                output_path,
                mipmaps=[],
                wavetable_type=WavetableType.CUSTOM,
            )

    def test_validation_error_mismatched_frames(self, tmp_path: Path) -> None:
        """Test that mismatched frame counts raises error."""
        # Mip0 has 4 frames, mip1 has 2 frames
        mip0 = np.random.randn(4, 256).astype(np.float32)
        mip1 = np.random.randn(2, 128).astype(np.float32)

        output_path = tmp_path / "test_mismatched.wav"

        with pytest.raises(ValueError, match="frames"):
            save_wavetable_wav(
                output_path,
                mipmaps=[mip0, mip1],
                wavetable_type=WavetableType.CUSTOM,
            )

    def test_1d_array_error(self, tmp_path: Path) -> None:
        """Test that 1D arrays raise an error."""
        mip0 = np.array([0.0, 0.5, 1.0, 0.5], dtype=np.float32)

        output_path = tmp_path / "test_1d.wav"

        with pytest.raises(ValueError, match="2D"):
            save_wavetable_wav(
                output_path,
                mipmaps=[mip0],
                wavetable_type=WavetableType.CUSTOM,
            )

    def test_skip_validation(self, tmp_path: Path) -> None:
        """Test that validation can be skipped."""
        # Create data with NaN (would normally fail validation)
        mip0 = np.array(
            [[0.0, np.nan, 1.0, 0.5], [0.0, -0.5, -1.0, -0.5]],
            dtype=np.float32,
        )

        output_path = tmp_path / "test_no_validate.wav"

        # Should not raise with validate=False
        save_wavetable_wav(
            output_path,
            mipmaps=[mip0],
            wavetable_type=WavetableType.CUSTOM,
            validate=False,
        )

        assert output_path.exists()

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """Test that parent directories are created if needed."""
        mip0 = np.array([[0.0, 0.5, 1.0, 0.5]], dtype=np.float32)

        output_path = tmp_path / "subdir" / "nested" / "test.wav"

        save_wavetable_wav(
            output_path,
            mipmaps=[mip0],
            wavetable_type=WavetableType.CUSTOM,
        )

        assert output_path.exists()


class TestWavetableTypes:
    """Tests for different wavetable type saves."""

    def test_all_wavetable_types(self, tmp_path: Path) -> None:
        """Test that all wavetable types can be saved."""
        mip0 = np.random.randn(4, 64).astype(np.float32)

        for wt_type in WavetableType:
            output_path = tmp_path / f"test_{wt_type.name.lower()}.wav"
            save_wavetable_wav(
                output_path,
                mipmaps=[mip0],
                wavetable_type=wt_type,
            )
            assert output_path.exists(), f"Failed to save {wt_type.name}"

    def test_high_resolution_metadata(self, tmp_path: Path) -> None:
        """Test saving with HighResolutionMetadata."""
        mip0 = np.random.randn(64, 2048).astype(np.float32)

        output_path = tmp_path / "test_hires.wav"
        save_wavetable_wav(
            output_path,
            mipmaps=[mip0],
            wavetable_type=WavetableType.HIGH_RESOLUTION,
            type_metadata=HighResolutionMetadata(
                max_harmonics=1024,
                source_synth="AN1x",
            ),
        )

        assert output_path.exists()
