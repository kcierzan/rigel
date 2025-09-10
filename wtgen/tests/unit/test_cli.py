"""Unit tests for wtgen.cli module."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from wtgen.cli.commands import app
from wtgen.dsp.waves import WaveformType
from wtgen.export import load_wavetable_npz
from zipfile import BadZipFile


class TestCliGenerate:
    """Test the generate command functionality."""

    def test_generate_default_sawtooth(self):
        """Test generate command with default parameters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_sawtooth.npz"

            result = app(["generate", "--output", str(output_path)])

            assert result == 0
            assert output_path.exists()

            # Verify file contents
            wt_data = load_wavetable_npz(output_path)
            manifest = wt_data["manifest"]
            tables = wt_data["tables"]

            mipmaps = tables["base"]
            assert len(mipmaps) == 9  # 8 octaves generates 9 levels
            assert mipmaps[0].shape[0] == 2048  # default size
            assert manifest["generation"]["waveform"] == "sawtooth"
            assert manifest["generation"]["rolloff"] == "raised_cosine"

    def test_generate_all_waveforms(self):
        """Test generate command with all supported waveform types."""
        waveforms = ["sawtooth", "square", "pulse", "triangle", "polyblep_saw"]

        with tempfile.TemporaryDirectory() as temp_dir:
            for waveform in waveforms:
                output_path = Path(temp_dir) / f"test_{waveform}.npz"

                assert app(["generate", "--waveform", waveform, "--output", str(output_path)]) == 0
                assert output_path.exists()

                wt_data = load_wavetable_npz(output_path)
                manifest = wt_data["manifest"]
                assert manifest["generation"]["waveform"] == waveform

    def test_generate_custom_parameters(self):
        """Test generate command with custom parameters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_custom.npz"

            assert (
                app(
                    [
                        "generate",
                        "--waveform",
                        "square",
                        "--output",
                        str(output_path),
                        "--octaves",
                        "6",
                        "--rolloff",
                        "hann",
                        "--frequency",
                        "2.0",
                        "--duty",
                        "0.3",
                        "--size",
                        "1024",
                    ],
                )
                == 0
            )

            assert output_path.exists()

            wt_data = load_wavetable_npz(output_path)
            manifest = wt_data["manifest"]
            tables = wt_data["tables"]
            mipmaps = tables["base"]

            assert len(mipmaps) == 7  # 6 octaves generates 7 levels
            assert mipmaps[0].shape[0] == 1024
            assert manifest["generation"]["octaves"] == 6
            assert manifest["generation"]["rolloff"] == "hann"
            assert manifest["generation"]["frequency"] == 2.0
            assert manifest["generation"]["duty"] == 0.3
            assert manifest["generation"]["size"] == 1024

    def test_generate_invalid_size(self, capsys):
        """Test generate command with invalid size (not power of 2)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_invalid.npz"

            with pytest.raises(SystemExit) as e:
                app(
                    ["generate", "--output", str(output_path), "--size", "1000"]
                )  # Not a power of 2

            assert e.value.code == 1
            assert "Size must be a power of 2" in capsys.readouterr().out
            assert not output_path.exists()

    @given(
        octaves=st.integers(min_value=1, max_value=12),
        frequency=st.floats(min_value=0.1, max_value=10.0),
        duty=st.floats(min_value=0.1, max_value=0.9),
    )
    @settings(max_examples=10)
    def test_generate_hypothesis(self, octaves, frequency, duty):
        """Property-based test for generate command."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_hypothesis.npz"

            assert (
                app(
                    [
                        "generate",
                        "square",
                        "--output",
                        str(output_path),
                        "--octaves",
                        str(octaves),
                        "--frequency",
                        str(frequency),
                        "--duty",
                        str(duty),
                    ],
                )
                == 0
            )

            assert output_path.exists()

            wt_data = load_wavetable_npz(output_path)
            manifest = wt_data["manifest"]
            tables = wt_data["tables"]
            mipmaps = tables["base"]

            assert len(mipmaps) == octaves + 1  # N octaves generates N+1 levels
            assert manifest["generation"]["frequency"] == frequency
            assert abs(manifest["generation"]["duty"] - duty) < 1e-6

    def test_generate_triangle_wave(self):
        """Test triangle wave generation with specific characteristics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_triangle.npz"

            assert (
                app(
                    [
                        "generate",
                        "--waveform",
                        "triangle",
                        "--output",
                        str(output_path),
                        "--octaves",
                        "6",
                    ]
                )
                == 0
            )
            assert output_path.exists()

            wt_data = load_wavetable_npz(output_path)
            manifest = wt_data["manifest"]
            tables = wt_data["tables"]
            mipmaps = tables["base"]

            assert len(mipmaps) == 7  # 6 octaves = 7 levels
            assert manifest["generation"]["waveform"] == "triangle"

            # Triangle waves should have specific RMS characteristics
            # (approximately 1/sqrt(3) ≈ 0.577 for normalized triangle)
            base_rms = np.sqrt(np.mean(mipmaps[0] ** 2))
            assert 0.55 < base_rms < 0.60, f"Triangle RMS {base_rms} outside expected range"


class TestCliHarmonic:
    """Test the harmonic command functionality."""

    def test_harmonic_default(self):
        """Test harmonic command with default parameters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_harmonic.npz"

            assert app(["harmonic", "--output", str(output_path)]) == 0
            assert output_path.exists()

            wt_data = load_wavetable_npz(output_path)
            manifest = wt_data["manifest"]
            tables = wt_data["tables"]
            mipmaps = tables["base"]

            # Check that partials were stored in manifest
            assert "generation" in manifest
            assert "partials" in manifest["generation"]

            assert len(mipmaps) == 9  # 8 octaves generates 9 levels
            assert manifest["generation"]["waveform"] == "harmonic"
            assert manifest["generation"]["num_partials"] == 1  # default sine

    def test_harmonic_custom_partials(self):
        """Test harmonic command with custom partials."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_custom_harmonic.npz"

            # Define custom partials: fundamental + second harmonic
            partials_str = "1:1.0:0.0,2:0.5:1.57"

            assert (
                app(
                    [
                        "harmonic",
                        "--output",
                        str(output_path),
                        "--partials",
                        partials_str,
                        "--octaves",
                        "4",
                    ],
                )
                == 0
            )
            assert output_path.exists()

            wt_data = load_wavetable_npz(output_path)
            manifest = wt_data["manifest"]
            tables = wt_data["tables"]
            mipmaps = tables["base"]
            partials = manifest["generation"]["partials"]

            assert len(mipmaps) == 5  # 4 octaves generates 5 levels
            assert manifest["generation"]["num_partials"] == 2
            assert len(partials) == 2

            # Check first partial
            assert partials[0][0] == 1  # harmonic number
            assert partials[0][1] == 1.0  # amplitude
            assert abs(partials[0][2] - 0.0) < 1e-6  # phase

    def test_harmonic_invalid_partials_format(self, capsys):
        """Test harmonic command with invalid partials format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_invalid_harmonic.npz"

            # Invalid format: missing phase
            partials_str = "1:1.0,2:0.5"

            with pytest.raises(ValueError):
                app(["harmonic", "--output", str(output_path), "--partials", partials_str])

            assert not output_path.exists()

    def test_harmonic_invalid_partials_values(self, capsys):
        """Test harmonic command with invalid partial values."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_invalid_values.npz"

            # Invalid values: non-numeric
            partials_str = "a:1.0:0.0"

            with pytest.raises(ValueError):
                app(["harmonic", "--output", str(output_path), "--partials", partials_str])
                captured = capsys.readouterr()
                assert "invalid literal for int() with base 10: a" in captured.err

            assert not output_path.exists()

    def test_harmonic_invalid_size(self, capsys):
        """Test harmonic command with invalid size."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_invalid_size.npz"

            with pytest.raises(SystemExit):
                app(["harmonic", "--output", str(output_path), "--size", "1000"])
                captured = capsys.readouterr()
                assert "Size must be a power of 2" in captured.err

            assert not output_path.exists()


class TestCliInfo:
    """Test the info command functionality."""

    def test_info_valid_file(self, capsys):
        """Test info command with valid wavetable file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # First generate a wavetable file
            output_path = Path(temp_dir) / "test_info.npz"

            app(["generate", "sawtooth", "--output", str(output_path), "--octaves", "4"])

            # Then test info command
            assert app(["info", str(output_path)]) == 0
            captured = capsys.readouterr()

            assert "Wavetable:" in captured.out
            assert "Table 'base': 5 mipmap levels" in captured.out  # 4 octaves = 5 levels
            assert "Base cycle length: 2048" in captured.out
            assert "Waveform: sawtooth" in captured.out
            assert "Rolloff: raised_cosine" in captured.out
            assert "Mipmap levels:" in captured.out

    def test_info_harmonic_file(self, capsys):
        """Test info command with harmonic wavetable file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate harmonic wavetable
            output_path = Path(temp_dir) / "test_harmonic_info.npz"

            app(
                [
                    "harmonic",
                    "--output",
                    str(output_path),
                    "--partials",
                    "1:1.0:0.0,2:0.5:0.0",
                    "--octaves",
                    "3",
                ],
            )

            # Test info command
            assert app(["info", str(output_path)]) == 0

            captured = capsys.readouterr()
            assert "Table 'base': 4 mipmap levels" in captured.out
            assert "Waveform: harmonic" in captured.out

    def test_info_nonexistent_file(self, capsys):
        """Test info command with nonexistent file."""
        with pytest.raises(ValueError):
            app(["info", "/nonexistent/file.npz"])
            captured = capsys.readouterr()
            assert "does not exist" in captured.out

    def test_info_invalid_file(self, capsys):
        """Test info command with invalid file format (missing manifest.json)."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as temp_file:
            # Create invalid npz file (missing manifest.json)
            np.savez(temp_file.name, invalid_data=np.array([1, 2, 3]))

            with pytest.raises(KeyError):
                app(["info", temp_file.name])

                captured = capsys.readouterr()
                assert "manifest.json" in captured.err

    @patch("numpy.load")
    def test_info_file_read_error(self, mock_load, capsys):
        """Test info command with file read error."""
        mock_load.side_effect = Exception("File read error")

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as temp_file:
            with pytest.raises(BadZipFile):
                app(["info", temp_file.name])
                captured = capsys.readouterr()
                assert "Error reading file" in captured.err


class TestEnumsAndTypes:
    """Test CLI enums and type definitions."""

    def test_waveform_type_enum(self):
        """Test WaveformType enum values."""
        assert WaveformType.sawtooth == "sawtooth"
        assert WaveformType.square == "square"
        assert WaveformType.pulse == "pulse"
        assert WaveformType.triangle == "triangle"
        assert WaveformType.polyblep_saw == "polyblep_saw"
