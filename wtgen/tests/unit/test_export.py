import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from wtgen.export import (
    load_wavetable_npz,
    save_mipmaps_as_wav,
    save_wavetable_npz,
    save_wavetable_with_wav_export,
)


class TestSaveWavetableNpz:
    """Test save_wavetable_npz functionality."""

    def test_save_basic_wavetable(self):
        """Test saving a basic wavetable with single mipmap level."""
        # Create test data
        wave = np.sin(2 * np.pi * np.linspace(0, 1, 2048, endpoint=False)).astype(np.float32)
        tables = {"base": [wave]}
        meta = {
            "version": 1,
            "name": "test_wavetable",
            "author": "wtgen_test",
            "sample_rate_hz": 44100,
        }

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            # Save and verify no exceptions
            save_wavetable_npz(tmp_path, tables, meta, compress=True)
            assert tmp_path.exists()

            # Verify it's a valid ZIP file
            with zipfile.ZipFile(tmp_path, "r") as z:
                files = z.namelist()
                assert "manifest.json" in files
                assert "base/mip_00_len2048.npy" in files

        finally:
            tmp_path.unlink(missing_ok=True)

    def test_save_multiple_mipmap_levels(self):
        """Test saving wavetable with multiple mipmap levels."""
        # Create test mipmaps
        base_wave = np.sin(2 * np.pi * np.linspace(0, 1, 2048, endpoint=False)).astype(np.float32)
        mip1 = np.sin(2 * np.pi * np.linspace(0, 1, 1024, endpoint=False)).astype(np.float32)
        mip2 = np.sin(2 * np.pi * np.linspace(0, 1, 512, endpoint=False)).astype(np.float32)

        tables = {"base": [base_wave, mip1, mip2]}
        meta = {"version": 1, "name": "multi_mip_test"}

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            save_wavetable_npz(tmp_path, tables, meta, compress=False)

            with zipfile.ZipFile(tmp_path, "r") as z:
                files = z.namelist()
                assert "base/mip_00_len2048.npy" in files
                assert "base/mip_01_len1024.npy" in files
                assert "base/mip_02_len512.npy" in files

        finally:
            tmp_path.unlink(missing_ok=True)

    def test_save_multiple_tables(self):
        """Test saving multiple wavetables in single file."""
        wave1 = np.sin(2 * np.pi * np.linspace(0, 1, 1024, endpoint=False)).astype(np.float32)
        wave2 = np.cos(2 * np.pi * np.linspace(0, 1, 1024, endpoint=False)).astype(np.float32)

        tables = {"sine": [wave1], "cosine": [wave2]}
        meta = {"version": 1, "name": "multi_table_test"}

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            save_wavetable_npz(tmp_path, tables, meta, compress=True)

            with zipfile.ZipFile(tmp_path, "r") as z:
                files = z.namelist()
                assert "sine/mip_00_len1024.npy" in files
                assert "cosine/mip_00_len1024.npy" in files

        finally:
            tmp_path.unlink(missing_ok=True)

    def test_manifest_generation(self):
        """Test that manifest is properly generated with correct structure."""
        wave = np.random.randn(512).astype(np.float32)
        tables = {"test": [wave]}
        meta = {
            "version": 1,
            "name": "manifest_test",
            "author": "test_suite",
            "custom_field": "custom_value",
        }

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            save_wavetable_npz(tmp_path, tables, meta, compress=True)

            with zipfile.ZipFile(tmp_path, "r") as z:
                manifest_data = z.read("manifest.json")
                manifest = json.loads(manifest_data)

                # Check original metadata is preserved
                assert manifest["version"] == 1
                assert manifest["name"] == "manifest_test"
                assert manifest["author"] == "test_suite"
                assert manifest["custom_field"] == "custom_value"

                # Check tables are properly added
                assert "tables" in manifest
                assert len(manifest["tables"]) == 1

                table_info = manifest["tables"][0]
                assert table_info["id"] == "test"
                assert table_info["description"] == "test wavetable"
                assert len(table_info["mipmaps"]) == 1

                mipmap_info = table_info["mipmaps"][0]
                assert mipmap_info["npz_path"] == "test/mip_00_len512.npy"
                assert mipmap_info["length"] == 512

                # Check stats are generated
                assert "stats" in manifest
                assert "dc_offset_mean" in manifest["stats"]
                assert "peak" in manifest["stats"]

        finally:
            tmp_path.unlink(missing_ok=True)

    def test_empty_tables_handling(self):
        """Test handling of empty tables dict."""
        tables = {}
        meta = {"version": 1, "name": "empty_test"}

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            save_wavetable_npz(tmp_path, tables, meta, compress=True)

            with zipfile.ZipFile(tmp_path, "r") as z:
                manifest_data = z.read("manifest.json")
                manifest = json.loads(manifest_data)

                assert manifest["tables"] == []
                # Stats should not be present for empty tables
                assert "stats" not in manifest

        finally:
            tmp_path.unlink(missing_ok=True)

    def test_data_type_conversion(self):
        """Test that arrays are properly converted to float32."""
        # Create arrays of different dtypes
        wave_int = np.arange(1024, dtype=np.int16)
        wave_float64 = np.random.randn(1024).astype(np.float64)

        tables = {"int_wave": [wave_int], "float64_wave": [wave_float64]}
        meta = {"version": 1, "name": "dtype_test"}

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            save_wavetable_npz(tmp_path, tables, meta, compress=True)

            # Load back and verify dtypes
            result = load_wavetable_npz(tmp_path)
            assert result["tables"]["int_wave"][0].dtype == np.float32
            assert result["tables"]["float64_wave"][0].dtype == np.float32

        finally:
            tmp_path.unlink(missing_ok=True)


class TestLoadWavetableNpz:
    """Test load_wavetable_npz functionality."""

    def test_load_basic_wavetable(self):
        """Test loading a basic saved wavetable."""
        # Create and save test data
        wave = np.sin(2 * np.pi * np.linspace(0, 1, 1024, endpoint=False)).astype(np.float32)
        tables = {"base": [wave]}
        meta = {
            "version": 1,
            "name": "load_test",
            "author": "test_suite",
            "sample_rate_hz": 44100,
        }

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            save_wavetable_npz(tmp_path, tables, meta, compress=True)

            # Load and verify
            result = load_wavetable_npz(tmp_path)

            assert "manifest" in result
            assert "tables" in result

            # Check manifest
            manifest = result["manifest"]
            assert manifest["name"] == "load_test"
            assert manifest["author"] == "test_suite"
            assert manifest["sample_rate_hz"] == 44100

            # Check tables
            loaded_tables = result["tables"]
            assert "base" in loaded_tables
            assert len(loaded_tables["base"]) == 1

            loaded_wave = loaded_tables["base"][0]
            np.testing.assert_array_almost_equal(loaded_wave, wave, decimal=5)

        finally:
            tmp_path.unlink(missing_ok=True)

    def test_load_multiple_mipmaps(self):
        """Test loading wavetable with multiple mipmap levels."""
        # Create test mipmaps with different frequencies
        waves = []
        for i, length in enumerate([2048, 1024, 512, 256]):
            freq = 2**i  # Increasing frequency for each level
            wave = np.sin(freq * 2 * np.pi * np.linspace(0, 1, length, endpoint=False))
            waves.append(wave.astype(np.float32))

        tables = {"multi_mip": waves}
        meta = {"version": 1, "name": "multi_mip_load_test"}

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            save_wavetable_npz(tmp_path, tables, meta, compress=False)

            result = load_wavetable_npz(tmp_path)
            loaded_waves = result["tables"]["multi_mip"]

            assert len(loaded_waves) == 4
            for i, (original, loaded) in enumerate(zip(waves, loaded_waves, strict=False)):
                np.testing.assert_array_almost_equal(loaded, original, decimal=5)
                assert loaded.dtype == np.float32

        finally:
            tmp_path.unlink(missing_ok=True)

    def test_load_multiple_tables(self):
        """Test loading multiple tables from single file."""
        wave1 = np.random.randn(512).astype(np.float32)
        wave2 = np.random.randn(1024).astype(np.float32)

        tables = {"table_a": [wave1], "table_b": [wave2]}
        meta = {"version": 1, "name": "multi_table_load_test"}

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            save_wavetable_npz(tmp_path, tables, meta, compress=True)

            result = load_wavetable_npz(tmp_path)
            loaded_tables = result["tables"]

            assert "table_a" in loaded_tables
            assert "table_b" in loaded_tables

            np.testing.assert_array_almost_equal(loaded_tables["table_a"][0], wave1, decimal=5)
            np.testing.assert_array_almost_equal(loaded_tables["table_b"][0], wave2, decimal=5)

        finally:
            tmp_path.unlink(missing_ok=True)

    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file raises appropriate error."""
        nonexistent_path = Path("/tmp/nonexistent_wavetable.npz")

        with pytest.raises((FileNotFoundError, zipfile.BadZipFile)):
            load_wavetable_npz(nonexistent_path)

    def test_round_trip_consistency(self):
        """Test that save/load round trip preserves data exactly."""
        # Create complex test data
        tables = {}

        # Different wave types and sizes
        for i, (name, size) in enumerate([("sine", 2048), ("square", 1024), ("noise", 512)]):
            if name == "sine":
                wave = np.sin(2 * np.pi * np.linspace(0, 1, size, endpoint=False))
            elif name == "square":
                wave = np.sign(np.sin(2 * np.pi * np.linspace(0, 1, size, endpoint=False)))
            else:  # noise
                wave = np.random.randn(size)

            # Create multiple mipmap levels
            mipmaps = []
            current_wave = wave.astype(np.float32)
            while len(current_wave) >= 64:
                mipmaps.append(current_wave.copy())
                # Simple decimation for test
                current_wave = current_wave[::2]

            tables[name] = mipmaps

        meta = {
            "version": 1,
            "name": "round_trip_test",
            "author": "test_suite",
            "sample_rate_hz": 48000,
            "custom_data": {"test": True, "value": 42},
        }

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            # Save and load
            save_wavetable_npz(tmp_path, tables, meta, compress=True)
            result = load_wavetable_npz(tmp_path)

            # Verify manifest preservation
            loaded_manifest = result["manifest"]
            assert loaded_manifest["name"] == meta["name"]
            assert loaded_manifest["author"] == meta["author"]
            assert loaded_manifest["sample_rate_hz"] == meta["sample_rate_hz"]
            assert loaded_manifest["custom_data"] == meta["custom_data"]

            # Verify table data preservation
            loaded_tables = result["tables"]
            for table_name, original_mipmaps in tables.items():
                assert table_name in loaded_tables
                loaded_mipmaps = loaded_tables[table_name]
                assert len(loaded_mipmaps) == len(original_mipmaps)

                for orig, loaded in zip(original_mipmaps, loaded_mipmaps, strict=False):
                    np.testing.assert_array_equal(orig, loaded)
                    assert loaded.dtype == np.float32

        finally:
            tmp_path.unlink(missing_ok=True)


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

        for bit_depth, expected_dtype in [(16, np.int16), (24, np.int32), (32, np.int32)]:
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


class TestSaveWavetableWithWavExport:
    """Test save_wavetable_with_wav_export functionality."""

    def test_dual_export(self):
        """Test that both .npz and .wav files are created."""
        wave = np.sin(2 * np.pi * np.linspace(0, 1, 1024, endpoint=False)).astype(np.float32)
        tables = {"test": [wave]}
        meta = {"version": 1, "name": "dual_export_test"}

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            npz_path = tmp_path / "test.npz"
            wav_dir = tmp_path / "wav_output"

            save_wavetable_with_wav_export(
                npz_path, wav_dir, tables, meta, compress=True, sample_rate=44100, bit_depth=16
            )

            # Check NPZ file exists and is valid
            assert npz_path.exists()
            result = load_wavetable_npz(npz_path)
            assert "test" in result["tables"]

            # Check WAV files exist
            wav_file = wav_dir / "test" / "mip_00_len1024.wav"
            assert wav_file.exists()

            sample_rate, data = wavfile.read(str(wav_file))
            assert sample_rate == 44100
            assert len(data) == 1024

    def test_consistent_data_between_formats(self):
        """Test that NPZ and WAV files contain equivalent data."""
        # Create test wave with known characteristics
        wave = 0.5 * np.sin(2 * np.pi * np.linspace(0, 1, 512, endpoint=False)).astype(np.float32)
        tables = {"consistency_test": [wave]}
        meta = {"version": 1, "name": "consistency_test"}

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            npz_path = tmp_path / "test.npz"
            wav_dir = tmp_path / "wav_output"

            save_wavetable_with_wav_export(
                npz_path, wav_dir, tables, meta, compress=True, sample_rate=44100, bit_depth=16
            )

            # Load NPZ data
            npz_result = load_wavetable_npz(npz_path)
            npz_wave = npz_result["tables"]["consistency_test"][0]

            # Load WAV data
            wav_file = wav_dir / "consistency_test" / "mip_00_len512.wav"
            sample_rate, wav_data = wavfile.read(str(wav_file))

            # Convert WAV data back to float
            wav_wave = wav_data.astype(np.float32) / 32767.0

            # Check that they are very similar (within quantization error)
            np.testing.assert_array_almost_equal(npz_wave, wav_wave, decimal=4)
