"""Unit tests for wtgen.cli module."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from typer.testing import CliRunner

from wtgen.cli import RolloffMethod, WaveformType, app
from wtgen.export import load_wavetable_npz


class TestCliGenerate:
    """Test the generate command functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_generate_default_sawtooth(self):
        """Test generate command with default parameters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_sawtooth.npz"

            result = self.runner.invoke(app, ["generate", "--output", str(output_path)])

            assert result.exit_code == 0
            assert output_path.exists()

            # Verify file contents
            wt_data = load_wavetable_npz(output_path)
            manifest = wt_data["manifest"]
            tables = wt_data["tables"]

            mipmaps = tables["base"]
            assert len(mipmaps) == 9  # 8 octaves generates 9 levels
            assert mipmaps[0].shape[0] == 2048  # default size
            assert manifest["generation"]["waveform"] == "sawtooth"
            assert manifest["generation"]["rolloff"] == "tukey"

    def test_generate_all_waveforms(self):
        """Test generate command with all supported waveform types."""
        waveforms = ["sawtooth", "square", "pulse", "triangle", "polyblep_saw"]

        with tempfile.TemporaryDirectory() as temp_dir:
            for waveform in waveforms:
                output_path = Path(temp_dir) / f"test_{waveform}.npz"

                result = self.runner.invoke(
                    app, ["generate", waveform, "--output", str(output_path)]
                )

                assert result.exit_code == 0
                assert output_path.exists()

                wt_data = load_wavetable_npz(output_path)
                manifest = wt_data["manifest"]
                assert manifest["generation"]["waveform"] == waveform

    def test_generate_custom_parameters(self):
        """Test generate command with custom parameters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_custom.npz"

            result = self.runner.invoke(
                app,
                [
                    "generate",
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

            assert result.exit_code == 0
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

    def test_generate_invalid_size(self):
        """Test generate command with invalid size (not power of 2)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_invalid.npz"

            result = self.runner.invoke(
                app,
                ["generate", "--output", str(output_path), "--size", "1000"],  # Not a power of 2
            )

            assert result.exit_code == 1
            assert "Size must be a power of 2" in result.stderr
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

            result = self.runner.invoke(
                app,
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

            assert result.exit_code == 0
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

            result = self.runner.invoke(
                app, ["generate", "triangle", "--output", str(output_path), "--octaves", "6"]
            )

            assert result.exit_code == 0
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

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_harmonic_default(self):
        """Test harmonic command with default parameters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_harmonic.npz"

            result = self.runner.invoke(app, ["harmonic", "--output", str(output_path)])

            assert result.exit_code == 0
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
            assert manifest["generation"]["num_partials"] == 16  # default sawtooth harmonics

    def test_harmonic_custom_partials(self):
        """Test harmonic command with custom partials."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_custom_harmonic.npz"

            # Define custom partials: fundamental + second harmonic
            partials_str = "1:1.0:0.0,2:0.5:1.57"

            result = self.runner.invoke(
                app,
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

            assert result.exit_code == 0
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

    def test_harmonic_invalid_partials_format(self):
        """Test harmonic command with invalid partials format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_invalid_harmonic.npz"

            # Invalid format: missing phase
            partials_str = "1:1.0,2:0.5"

            result = self.runner.invoke(
                app, ["harmonic", "--output", str(output_path), "--partials", partials_str]
            )

            assert result.exit_code == 1
            assert "Error parsing partials" in result.stderr
            assert not output_path.exists()

    def test_harmonic_invalid_partials_values(self):
        """Test harmonic command with invalid partial values."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_invalid_values.npz"

            # Invalid values: non-numeric
            partials_str = "a:1.0:0.0"

            result = self.runner.invoke(
                app, ["harmonic", "--output", str(output_path), "--partials", partials_str]
            )

            assert result.exit_code == 1
            assert "Error parsing partials" in result.stderr
            assert not output_path.exists()

    def test_harmonic_invalid_size(self):
        """Test harmonic command with invalid size."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_invalid_size.npz"

            result = self.runner.invoke(
                app,
                ["harmonic", "--output", str(output_path), "--size", "1000"],  # Not a power of 2
            )

            assert result.exit_code == 1
            assert "Size must be a power of 2" in result.stderr
            assert not output_path.exists()


class TestCliInfo:
    """Test the info command functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_info_valid_file(self):
        """Test info command with valid wavetable file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # First generate a wavetable file
            output_path = Path(temp_dir) / "test_info.npz"

            self.runner.invoke(
                app, ["generate", "sawtooth", "--output", str(output_path), "--octaves", "4"]
            )

            # Then test info command
            result = self.runner.invoke(app, ["info", str(output_path)])

            assert result.exit_code == 0
            assert "Wavetable:" in result.stdout
            assert "Table 'base': 5 mipmap levels" in result.stdout  # 4 octaves = 5 levels
            assert "Base cycle length: 2048" in result.stdout
            assert "Waveform: sawtooth" in result.stdout
            assert "Rolloff: tukey" in result.stdout
            assert "Mipmap levels:" in result.stdout

    def test_info_harmonic_file(self):
        """Test info command with harmonic wavetable file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate harmonic wavetable
            output_path = Path(temp_dir) / "test_harmonic_info.npz"

            self.runner.invoke(
                app,
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
            result = self.runner.invoke(app, ["info", str(output_path)])

            assert result.exit_code == 0
            assert "Table 'base': 4 mipmap levels" in result.stdout  # 3 octaves = 4 levels
            assert "Waveform: harmonic" in result.stdout

    def test_info_nonexistent_file(self):
        """Test info command with nonexistent file."""
        result = self.runner.invoke(app, ["info", "/nonexistent/file.npz"])

        assert result.exit_code == 1
        assert "does not exist" in result.stderr

    def test_info_invalid_file(self):
        """Test info command with invalid file format (missing manifest.json)."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as temp_file:
            # Create invalid npz file (missing manifest.json)
            np.savez(temp_file.name, invalid_data=np.array([1, 2, 3]))

            result = self.runner.invoke(app, ["info", temp_file.name])

            assert result.exit_code == 1
            assert "Error reading file:" in result.stderr
            assert "manifest.json" in result.stderr

    @patch("numpy.load")
    def test_info_file_read_error(self, mock_load):
        """Test info command with file read error."""
        mock_load.side_effect = Exception("File read error")

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as temp_file:
            result = self.runner.invoke(app, ["info", temp_file.name])

            assert result.exit_code == 1
            assert "Error reading file" in result.stderr


class TestEnumsAndTypes:
    """Test CLI enums and type definitions."""

    def test_waveform_type_enum(self):
        """Test WaveformType enum values."""
        assert WaveformType.sawtooth == "sawtooth"
        assert WaveformType.square == "square"
        assert WaveformType.pulse == "pulse"
        assert WaveformType.triangle == "triangle"
        assert WaveformType.polyblep_saw == "polyblep_saw"

    def test_rolloff_method_enum(self):
        """Test RolloffMethod enum values."""
        assert RolloffMethod.brick_wall == "brick_wall"
        assert RolloffMethod.tukey == "tukey"
        assert RolloffMethod.blackman == "blackman"
        assert RolloffMethod.raised_cosine == "raised_cosine"
        assert RolloffMethod.hann == "hann"
        assert RolloffMethod.none == "none"


class TestIntegrationScenarios:
    """Test integration scenarios across CLI commands."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_generate_then_info_workflow(self):
        """Test complete workflow: generate wavetable then get info."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "workflow_test.npz"

            # Generate wavetable
            generate_result = self.runner.invoke(
                app,
                [
                    "generate",
                    "square",
                    "--output",
                    str(output_path),
                    "--octaves",
                    "6",
                    "--duty",
                    "0.25",
                ],
            )

            assert generate_result.exit_code == 0

            # Get info about generated file
            info_result = self.runner.invoke(app, ["info", str(output_path)])

            assert info_result.exit_code == 0
            assert "Table 'base': 7 mipmap levels" in info_result.stdout  # 6 octaves = 7 levels
            assert "Waveform: square" in info_result.stdout

    def test_harmonic_then_info_workflow(self):
        """Test harmonic generation then info workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "harmonic_workflow.npz"

            # Generate harmonic wavetable
            generate_result = self.runner.invoke(
                app,
                [
                    "harmonic",
                    "--output",
                    str(output_path),
                    "--partials",
                    "1:1.0:0.0,3:0.33:1.57,5:0.2:0.0",
                    "--octaves",
                    "5",
                ],
            )

            assert generate_result.exit_code == 0

            # Get info about generated file
            info_result = self.runner.invoke(app, ["info", str(output_path)])

            assert info_result.exit_code == 0
            assert "Table 'base': 6 mipmap levels" in info_result.stdout  # 5 octaves = 6 levels
            assert "Waveform: harmonic" in info_result.stdout

    @given(
        waveform=st.sampled_from(["sawtooth", "square", "pulse", "triangle", "polyblep_saw"]),
        octaves=st.integers(min_value=2, max_value=8),
        rolloff=st.sampled_from(["tukey", "hann", "blackman"]),
    )
    @settings(max_examples=5)
    def test_generate_info_roundtrip(self, waveform, octaves, rolloff):
        """Property-based test for generate->info roundtrip."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "roundtrip_test.npz"

            # Generate
            generate_result = self.runner.invoke(
                app,
                [
                    "generate",
                    waveform,
                    "--output",
                    str(output_path),
                    "--octaves",
                    str(octaves),
                    "--rolloff",
                    rolloff,
                ],
            )

            assert generate_result.exit_code == 0

            # Info
            info_result = self.runner.invoke(app, ["info", str(output_path)])

            assert info_result.exit_code == 0
            assert (
                f"Table 'base': {octaves + 1} mipmap levels" in info_result.stdout
            )  # N octaves = N+1 levels
            assert f"Waveform: {waveform}" in info_result.stdout
            assert f"Rolloff: {rolloff}" in info_result.stdout
