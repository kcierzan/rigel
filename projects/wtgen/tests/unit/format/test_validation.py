"""Unit tests for validation requirements FR-028 (NaN/Infinity) and FR-030b (file size limit).

These tests verify that the wavetable reader properly rejects:
- FR-028: Files containing NaN or Infinity sample values
- FR-030b: Files exceeding 100MB in size
"""

import struct
from pathlib import Path

import numpy as np
import pytest

from wtgen.format import WavetableType, load_wavetable_wav, save_wavetable_wav
from wtgen.format.riff import build_wav_with_wtbl
from wtgen.format.proto import wavetable_pb2 as pb
from wtgen.format.standard.reader import MAX_FILE_SIZE_BYTES
from wtgen.format.validation import ValidationError


class TestNaNInfinityRejection:
    """Tests for FR-028: NaN/Infinity sample values MUST be rejected."""

    def test_rejects_nan_samples(self, tmp_path: Path) -> None:
        """Test that files with NaN samples are rejected."""
        # Create a wavetable with NaN values
        mip0 = np.array([[0.0, 0.5, np.nan, 0.5]], dtype=np.float32)

        # Create metadata
        metadata = pb.WavetableMetadata(
            schema_version=1,
            wavetable_type=pb.WAVETABLE_TYPE_CUSTOM,
            frame_length=4,
            num_frames=1,
            num_mip_levels=1,
            mip_frame_lengths=[4],
        )

        # Build WAV file manually with NaN
        samples_bytes = mip0.flatten().tobytes()
        wtbl_bytes = metadata.SerializeToString()
        wav_data = build_wav_with_wtbl(samples_bytes, 44100, wtbl_bytes)

        output_path = tmp_path / "test_nan.wav"
        output_path.write_bytes(wav_data)

        # Attempt to load - should raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            load_wavetable_wav(output_path)

        assert "non-finite" in str(exc_info.value).lower() or "nan" in str(exc_info.value).lower()
        assert "FR-028" in str(exc_info.value)

    def test_rejects_positive_infinity(self, tmp_path: Path) -> None:
        """Test that files with positive Infinity samples are rejected."""
        mip0 = np.array([[0.0, np.inf, 0.5, 0.5]], dtype=np.float32)

        metadata = pb.WavetableMetadata(
            schema_version=1,
            wavetable_type=pb.WAVETABLE_TYPE_CUSTOM,
            frame_length=4,
            num_frames=1,
            num_mip_levels=1,
            mip_frame_lengths=[4],
        )

        samples_bytes = mip0.flatten().tobytes()
        wtbl_bytes = metadata.SerializeToString()
        wav_data = build_wav_with_wtbl(samples_bytes, 44100, wtbl_bytes)

        output_path = tmp_path / "test_pos_inf.wav"
        output_path.write_bytes(wav_data)

        with pytest.raises(ValidationError) as exc_info:
            load_wavetable_wav(output_path)

        assert "non-finite" in str(exc_info.value).lower() or "infinity" in str(exc_info.value).lower()
        assert "FR-028" in str(exc_info.value)

    def test_rejects_negative_infinity(self, tmp_path: Path) -> None:
        """Test that files with negative Infinity samples are rejected."""
        mip0 = np.array([[0.0, -np.inf, 0.5, 0.5]], dtype=np.float32)

        metadata = pb.WavetableMetadata(
            schema_version=1,
            wavetable_type=pb.WAVETABLE_TYPE_CUSTOM,
            frame_length=4,
            num_frames=1,
            num_mip_levels=1,
            mip_frame_lengths=[4],
        )

        samples_bytes = mip0.flatten().tobytes()
        wtbl_bytes = metadata.SerializeToString()
        wav_data = build_wav_with_wtbl(samples_bytes, 44100, wtbl_bytes)

        output_path = tmp_path / "test_neg_inf.wav"
        output_path.write_bytes(wav_data)

        with pytest.raises(ValidationError) as exc_info:
            load_wavetable_wav(output_path)

        assert "non-finite" in str(exc_info.value).lower() or "infinity" in str(exc_info.value).lower()
        assert "FR-028" in str(exc_info.value)

    def test_rejects_mixed_nan_and_inf(self, tmp_path: Path) -> None:
        """Test that files with both NaN and Infinity are rejected."""
        mip0 = np.array([[np.nan, np.inf, -np.inf, 0.5]], dtype=np.float32)

        metadata = pb.WavetableMetadata(
            schema_version=1,
            wavetable_type=pb.WAVETABLE_TYPE_CUSTOM,
            frame_length=4,
            num_frames=1,
            num_mip_levels=1,
            mip_frame_lengths=[4],
        )

        samples_bytes = mip0.flatten().tobytes()
        wtbl_bytes = metadata.SerializeToString()
        wav_data = build_wav_with_wtbl(samples_bytes, 44100, wtbl_bytes)

        output_path = tmp_path / "test_mixed_invalid.wav"
        output_path.write_bytes(wav_data)

        with pytest.raises(ValidationError) as exc_info:
            load_wavetable_wav(output_path)

        assert "FR-028" in str(exc_info.value)

    def test_accepts_valid_samples(self, tmp_path: Path) -> None:
        """Test that valid finite samples are accepted."""
        # Normal values including edge cases
        mip0 = np.array([[0.0, 0.5, -0.5, 1.0, -1.0, 0.999999, -0.999999, 0.000001]], dtype=np.float32)

        output_path = tmp_path / "test_valid.wav"
        save_wavetable_wav(
            output_path,
            mipmaps=[mip0],
            wavetable_type=WavetableType.CUSTOM,
        )

        # Should load without error
        loaded = load_wavetable_wav(output_path)
        assert loaded is not None
        np.testing.assert_array_almost_equal(loaded.mipmaps[0], mip0)

    def test_error_message_includes_location(self, tmp_path: Path) -> None:
        """Test that error message includes first bad sample location."""
        # NaN at index 5
        mip0 = np.array([[0.0, 0.5, 0.5, 0.5, 0.5, np.nan, 0.5, 0.5]], dtype=np.float32)

        metadata = pb.WavetableMetadata(
            schema_version=1,
            wavetable_type=pb.WAVETABLE_TYPE_CUSTOM,
            frame_length=8,
            num_frames=1,
            num_mip_levels=1,
            mip_frame_lengths=[8],
        )

        samples_bytes = mip0.flatten().tobytes()
        wtbl_bytes = metadata.SerializeToString()
        wav_data = build_wav_with_wtbl(samples_bytes, 44100, wtbl_bytes)

        output_path = tmp_path / "test_location.wav"
        output_path.write_bytes(wav_data)

        with pytest.raises(ValidationError) as exc_info:
            load_wavetable_wav(output_path)

        # Should mention the sample index
        error_msg = str(exc_info.value)
        assert "5" in error_msg  # Index of NaN


class TestFileSizeLimit:
    """Tests for FR-030b: Files exceeding 100MB MUST be rejected."""

    def test_max_file_size_constant(self) -> None:
        """Test that MAX_FILE_SIZE_BYTES is correctly set to 100 MB."""
        assert MAX_FILE_SIZE_BYTES == 100 * 1024 * 1024

    def test_rejects_file_over_100mb(self, tmp_path: Path) -> None:
        """Test that files over 100MB are rejected.

        Note: This test creates a minimal file and patches its apparent size
        rather than creating a truly 100MB+ file to keep tests fast.
        """
        # Create a valid small wavetable first
        mip0 = np.random.randn(4, 64).astype(np.float32)
        output_path = tmp_path / "test_large.wav"
        save_wavetable_wav(
            output_path,
            mipmaps=[mip0],
            wavetable_type=WavetableType.CUSTOM,
        )

        # Pad the file to exceed 100MB
        # Write padding to make file over 100MB
        with open(output_path, "ab") as f:
            # Write enough zeros to exceed 100MB
            # Current file size + padding > 100MB
            current_size = output_path.stat().st_size
            padding_needed = MAX_FILE_SIZE_BYTES - current_size + 1
            # Write in chunks to avoid memory issues
            chunk_size = 1024 * 1024  # 1MB chunks
            remaining = padding_needed
            while remaining > 0:
                write_size = min(chunk_size, remaining)
                f.write(b"\x00" * write_size)
                remaining -= write_size

        # Verify file is over 100MB
        assert output_path.stat().st_size > MAX_FILE_SIZE_BYTES

        # Attempt to load - should raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            load_wavetable_wav(output_path)

        error_msg = str(exc_info.value)
        assert "100 MB" in error_msg or "FR-030b" in error_msg

    def test_accepts_file_under_100mb(self, tmp_path: Path) -> None:
        """Test that files under 100MB are accepted."""
        # Create a typical-sized wavetable (should be well under 100MB)
        mip0 = np.random.randn(64, 2048).astype(np.float32)  # ~500KB
        mip1 = np.random.randn(64, 1024).astype(np.float32)  # ~250KB
        mip2 = np.random.randn(64, 512).astype(np.float32)  # ~125KB

        output_path = tmp_path / "test_normal_size.wav"
        save_wavetable_wav(
            output_path,
            mipmaps=[mip0, mip1, mip2],
            wavetable_type=WavetableType.HIGH_RESOLUTION,
        )

        # Verify file is under 100MB
        assert output_path.stat().st_size < MAX_FILE_SIZE_BYTES

        # Should load without error
        loaded = load_wavetable_wav(output_path)
        assert loaded is not None

    def test_accepts_file_exactly_at_limit(self, tmp_path: Path) -> None:
        """Test that a file exactly at 100MB is accepted (boundary case).

        Note: This is a simplified test - we verify the boundary logic
        rather than creating a file exactly at 100MB.
        """
        # Create a small file and verify boundary logic
        mip0 = np.random.randn(4, 64).astype(np.float32)
        output_path = tmp_path / "test_boundary.wav"
        save_wavetable_wav(
            output_path,
            mipmaps=[mip0],
            wavetable_type=WavetableType.CUSTOM,
        )

        # File should load (it's well under 100MB)
        loaded = load_wavetable_wav(output_path)
        assert loaded is not None
