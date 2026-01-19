"""Unit tests for wtgen.cli module."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from wtgen.cli.commands import app
from wtgen.dsp.waves import WaveformType
from wtgen.format import ValidationError, load_wavetable_wav
from wtgen.format.riff import RiffError


class TestCliGenerate:
    """Test the generate command functionality."""

    def test_generate_default_sawtooth(self):
        """Test generate command with default parameters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_sawtooth.wav"

            result = app(["generate", "--output", str(output_path)])

            assert result == 0
            assert output_path.exists()

            # Verify file contents
            wt_file = load_wavetable_wav(output_path)

            assert wt_file.num_mip_levels == 9  # 8 octaves generates 9 levels
            assert wt_file.frame_length == 2048  # default size
            assert wt_file.num_frames == 1  # single-frame wavetable

            # Check generation parameters
            gen_params = json.loads(wt_file.metadata.generation_parameters)
            assert gen_params["waveform"] == "sawtooth"
            assert gen_params["rolloff"] == "raised_cosine"

    def test_generate_all_waveforms(self):
        """Test generate command with all supported waveform types."""
        waveforms = ["sawtooth", "square", "pulse", "triangle", "polyblep_saw"]

        with tempfile.TemporaryDirectory() as temp_dir:
            for waveform in waveforms:
                output_path = Path(temp_dir) / f"test_{waveform}.wav"

                assert app(["generate", "--waveform", waveform, "--output", str(output_path)]) == 0
                assert output_path.exists()

                wt_file = load_wavetable_wav(output_path)
                gen_params = json.loads(wt_file.metadata.generation_parameters)
                assert gen_params["waveform"] == waveform

    def test_generate_custom_parameters(self):
        """Test generate command with custom parameters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_custom.wav"

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

            wt_file = load_wavetable_wav(output_path)

            assert wt_file.num_mip_levels == 7  # 6 octaves generates 7 levels
            assert wt_file.frame_length == 1024

            gen_params = json.loads(wt_file.metadata.generation_parameters)
            assert gen_params["octaves"] == 6
            assert gen_params["rolloff"] == "hann"
            assert gen_params["frequency"] == 2.0
            assert gen_params["duty"] == 0.3
            assert gen_params["size"] == 1024

    def test_generate_invalid_size(self, capsys):
        """Test generate command with invalid size (not power of 2)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_invalid.wav"

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
            output_path = Path(temp_dir) / "test_hypothesis.wav"

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

            wt_file = load_wavetable_wav(output_path)
            gen_params = json.loads(wt_file.metadata.generation_parameters)

            assert wt_file.num_mip_levels == octaves + 1  # N octaves generates N+1 levels
            assert gen_params["frequency"] == frequency
            assert abs(gen_params["duty"] - duty) < 1e-6

    def test_generate_triangle_wave(self):
        """Test triangle wave generation with specific characteristics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_triangle.wav"

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

            wt_file = load_wavetable_wav(output_path)
            gen_params = json.loads(wt_file.metadata.generation_parameters)

            assert wt_file.num_mip_levels == 7  # 6 octaves = 7 levels
            assert gen_params["waveform"] == "triangle"

            # Triangle waves should have specific RMS characteristics
            # (approximately 1/sqrt(3) ≈ 0.577 for normalized triangle)
            base_mip = wt_file.mipmaps[0]
            base_rms = np.sqrt(np.mean(base_mip**2))
            assert 0.55 < base_rms < 0.60, f"Triangle RMS {base_rms} outside expected range"


class TestCliHarmonic:
    """Test the harmonic command functionality."""

    def test_harmonic_default(self):
        """Test harmonic command with default parameters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_harmonic.wav"

            assert app(["harmonic", "--output", str(output_path)]) == 0
            assert output_path.exists()

            wt_file = load_wavetable_wav(output_path)
            gen_params = json.loads(wt_file.metadata.generation_parameters)

            # Check that partials were stored in generation params
            assert "partials" in gen_params
            assert "num_partials" in gen_params

            assert wt_file.num_mip_levels == 9  # 8 octaves generates 9 levels
            assert gen_params["waveform"] == "harmonic"
            assert gen_params["num_partials"] == 1  # default sine

    def test_harmonic_custom_partials(self):
        """Test harmonic command with custom partials."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_custom_harmonic.wav"

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

            wt_file = load_wavetable_wav(output_path)
            gen_params = json.loads(wt_file.metadata.generation_parameters)
            partials = gen_params["partials"]

            assert wt_file.num_mip_levels == 5  # 4 octaves generates 5 levels
            assert gen_params["num_partials"] == 2
            assert len(partials) == 2

            # Check first partial
            assert partials[0][0] == 1  # harmonic number
            assert partials[0][1] == 1.0  # amplitude
            assert abs(partials[0][2] - 0.0) < 1e-6  # phase

    def test_harmonic_invalid_partials_format(self, capsys):
        """Test harmonic command with invalid partials format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_invalid_harmonic.wav"

            # Invalid format: missing phase
            partials_str = "1:1.0,2:0.5"

            with pytest.raises(ValueError):
                app(["harmonic", "--output", str(output_path), "--partials", partials_str])

            assert not output_path.exists()

    def test_harmonic_invalid_partials_values(self, capsys):
        """Test harmonic command with invalid partial values."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_invalid_values.wav"

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
            output_path = Path(temp_dir) / "test_invalid_size.wav"

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
            output_path = Path(temp_dir) / "test_info.wav"

            app(["generate", "sawtooth", "--output", str(output_path), "--octaves", "4"])

            # Then test info command
            assert app(["info", str(output_path)]) == 0
            captured = capsys.readouterr()

            assert "Wavetable:" in captured.out
            assert "Mip levels: 5" in captured.out  # 4 octaves = 5 levels
            assert "Frame length (mip 0): 2048" in captured.out
            assert "Waveform: sawtooth" in captured.out
            assert "Rolloff: raised_cosine" in captured.out
            assert "Mipmap levels:" in captured.out

    def test_info_harmonic_file(self, capsys):
        """Test info command with harmonic wavetable file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate harmonic wavetable
            output_path = Path(temp_dir) / "test_harmonic_info.wav"

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
            assert "Mip levels: 4" in captured.out
            assert "Waveform: harmonic" in captured.out

    def test_info_nonexistent_file(self, capsys):
        """Test info command with nonexistent file."""
        with pytest.raises(ValueError):
            app(["info", "/nonexistent/file.wav"])
            captured = capsys.readouterr()
            assert "does not exist" in captured.out

    def test_info_invalid_file(self, capsys):
        """Test info command with invalid file format (not a valid WAV)."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            # Create invalid wav file
            temp_file.write(b"not a valid wav file")
            temp_file.flush()

            with pytest.raises((RiffError, ValidationError)):
                app(["info", temp_file.name])


class TestEnumsAndTypes:
    """Test CLI enums and type definitions."""

    def test_waveform_type_enum(self):
        """Test WaveformType enum values."""
        assert WaveformType.sawtooth == "sawtooth"
        assert WaveformType.square == "square"
        assert WaveformType.pulse == "pulse"
        assert WaveformType.triangle == "triangle"
        assert WaveformType.polyblep_saw == "polyblep_saw"
