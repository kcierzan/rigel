import tempfile
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from wtgen.export import save_mipmaps_as_wav


class TestSaveMipmapsAsWav:
    """Test save_mipmaps_as_wav functionality."""

    def test_save_basic_wav(self):
        """Test saving basic .wav files."""
        # Create test data
        wave = np.sin(2 * np.pi * np.linspace(0, 1, 1024, endpoint=False)).astype(np.float32)
        tables = {"test": [wave]}

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            save_mipmaps_as_wav(tmp_path, tables, sample_rate=44100, bit_depth=16)

            # Check that directory and file were created
            expected_file = tmp_path / "test" / "mip_00_len1024.wav"
            assert expected_file.exists()

            # Load and verify the saved wav file
            sample_rate, data = wavfile.read(str(expected_file))
            assert sample_rate == 44100
            assert len(data) == 1024
            assert data.dtype == np.int16

    def test_save_multiple_bit_depths(self):
        """Test saving .wav files with different bit depths."""
        wave = np.random.randn(512).astype(np.float32) * 0.5  # Keep within [-0.5, 0.5]
        tables = {"test": [wave]}

        for bit_depth, _expected_dtype in [(16, np.int16), (24, np.int32), (32, np.int32)]:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)

                save_mipmaps_as_wav(tmp_path, tables, sample_rate=48000, bit_depth=bit_depth)

                expected_file = tmp_path / "test" / "mip_00_len512.wav"
                assert expected_file.exists()

                sample_rate, data = wavfile.read(str(expected_file))
                assert sample_rate == 48000
                assert len(data) == 512

    def test_save_multiple_mipmaps(self):
        """Test saving multiple mipmap levels as separate .wav files."""
        # Create mipmaps of different lengths
        mipmaps = []
        for length in [2048, 1024, 512, 256]:
            wave = np.sin(2 * np.pi * np.linspace(0, 1, length, endpoint=False))
            mipmaps.append(wave.astype(np.float32))

        tables = {"multi": mipmaps}

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            save_mipmaps_as_wav(tmp_path, tables, sample_rate=44100, bit_depth=16)

            # Check all mipmap files exist
            table_dir = tmp_path / "multi"
            assert table_dir.exists()

            expected_files = [
                "mip_00_len2048.wav",
                "mip_01_len1024.wav",
                "mip_02_len512.wav",
                "mip_03_len256.wav",
            ]

            for filename in expected_files:
                filepath = table_dir / filename
                assert filepath.exists()

                sample_rate, data = wavfile.read(str(filepath))
                assert sample_rate == 44100

    def test_save_multiple_tables(self):
        """Test saving multiple tables as separate directories."""
        wave1 = np.sin(2 * np.pi * np.linspace(0, 1, 512, endpoint=False)).astype(np.float32)
        wave2 = np.cos(2 * np.pi * np.linspace(0, 1, 512, endpoint=False)).astype(np.float32)

        tables = {"sine": [wave1], "cosine": [wave2]}

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            save_mipmaps_as_wav(tmp_path, tables, sample_rate=44100, bit_depth=16)

            # Check both table directories exist with files
            for table_name in ["sine", "cosine"]:
                table_dir = tmp_path / table_name
                assert table_dir.exists()

                wav_file = table_dir / "mip_00_len512.wav"
                assert wav_file.exists()

    def test_save_clipping_behavior(self):
        """Test that values outside [-1, 1] are properly clipped."""
        # Create wave with values outside [-1, 1]
        wave = np.array([-2.0, -1.5, -1.0, 0.0, 1.0, 1.5, 2.0], dtype=np.float32)
        tables = {"clip_test": [wave]}

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            save_mipmaps_as_wav(tmp_path, tables, sample_rate=44100, bit_depth=16)

            expected_file = tmp_path / "clip_test" / "mip_00_len7.wav"
            sample_rate, data = wavfile.read(str(expected_file))

            # Convert back to float to check clipping
            data_float = data.astype(np.float32) / 32767.0

            # Check that extreme values were clipped
            assert np.max(data_float) <= 1.0
            assert np.min(data_float) >= -1.0

    def test_invalid_bit_depth(self):
        """Test that invalid bit depth raises ValueError."""
        wave = np.sin(2 * np.pi * np.linspace(0, 1, 256, endpoint=False)).astype(np.float32)
        tables = {"test": [wave]}

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            with pytest.raises(ValueError, match="Unsupported bit depth"):
                save_mipmaps_as_wav(tmp_path, tables, sample_rate=44100, bit_depth=8)
