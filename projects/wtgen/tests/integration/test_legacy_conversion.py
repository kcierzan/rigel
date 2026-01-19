"""
Integration tests for legacy wavetable format conversion.

Tests the complete pipeline from legacy format import through to standardized
wavetable interchange format export, including validation and round-trip verification.
"""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from wtgen.format import load_wavetable_wav, save_wavetable_wav
from wtgen.format.importers import import_wav_with_mips
from wtgen.format.legacy import (
    detect_raw_format,
    detect_wav_wavetable,
    import_hires_wav,
    import_raw_pcm,
    infer_wavetable_type,
)
from wtgen.format.types import WavetableType


class TestRawPcmToStandardFormat:
    """Test conversion from raw PCM to standardized format."""

    def test_ppg_style_conversion(self, tmp_path: Path) -> None:
        """Test full conversion of PPG-style raw wavetable."""
        # Create PPG-style raw data: 64 frames of 256 samples, 8-bit
        frame_length = 256
        num_frames = 64

        # Create actual waveform data (saw wave morphing)
        raw_data = bytearray()
        for frame in range(num_frames):
            phase_shift = frame / num_frames
            for sample in range(frame_length):
                t = sample / frame_length
                # Morphing saw wave
                value = int(((t + phase_shift) % 1.0 * 2 - 1) * 127)
                raw_data.append(value & 0xFF)

        raw_path = tmp_path / "ppg_wavetable.raw"
        raw_path.write_bytes(bytes(raw_data))

        # Import raw PCM
        mipmaps, metadata = import_raw_pcm(
            raw_path,
            frame_length=frame_length,
            num_frames=num_frames,
            bit_depth=8,
            signed=True,
        )

        # Verify import
        assert len(mipmaps) == 1
        assert mipmaps[0].shape == (64, 256)
        assert metadata["wavetable_type"] == WavetableType.CLASSIC_DIGITAL

        # Save as standardized format
        output_path = tmp_path / "converted_ppg.wav"
        save_wavetable_wav(
            output_path,
            mipmaps=mipmaps,
            wavetable_type=metadata["wavetable_type"],
            name="Converted PPG Wavetable",
            source_bit_depth=metadata["source_bit_depth"],
            type_metadata=metadata.get("type_metadata"),
        )

        assert output_path.exists()

        # Reload and verify
        loaded = load_wavetable_wav(output_path)
        assert len(loaded.mipmaps) == 1
        assert loaded.mipmaps[0].shape == (64, 256)
        assert loaded.wavetable_type == WavetableType.CLASSIC_DIGITAL

        # Audio data should be approximately equal (within normalization tolerance)
        np.testing.assert_allclose(mipmaps[0], loaded.mipmaps[0], rtol=1e-5, atol=1e-5)

    def test_16bit_conversion(self, tmp_path: Path) -> None:
        """Test conversion of 16-bit raw PCM."""
        frame_length = 512
        num_frames = 32

        # Create 16-bit sine wave data
        samples = []
        for frame in range(num_frames):
            freq_mult = 1 + frame * 0.1  # Increasing frequency per frame
            for sample in range(frame_length):
                t = sample / frame_length * 2 * np.pi * freq_mult
                value = int(np.sin(t) * 32767)
                samples.append(value)

        raw_data = np.array(samples, dtype=np.int16)
        raw_path = tmp_path / "16bit_wavetable.raw"
        raw_path.write_bytes(raw_data.tobytes())

        # Import
        mipmaps, metadata = import_raw_pcm(
            raw_path,
            frame_length=frame_length,
            num_frames=num_frames,
            bit_depth=16,
        )

        # Save and reload
        output_path = tmp_path / "converted_16bit.wav"
        save_wavetable_wav(
            output_path,
            mipmaps=mipmaps,
            wavetable_type=metadata["wavetable_type"],
        )

        loaded = load_wavetable_wav(output_path)
        np.testing.assert_allclose(mipmaps[0], loaded.mipmaps[0], rtol=1e-5, atol=1e-5)

    def test_detection_and_import_workflow(self, tmp_path: Path) -> None:
        """Test the workflow of detecting format then importing."""
        # Create file with unknown format
        raw_path = tmp_path / "mystery.raw"
        raw_path.write_bytes(b"\x00" * (256 * 64))  # PPG-style

        # Detect format
        detected = detect_raw_format(raw_path)
        assert detected["confidence"] == "exact_match"

        # Import using detected parameters
        mipmaps, metadata = import_raw_pcm(
            raw_path,
            frame_length=detected["frame_length"],
            num_frames=detected["num_frames"],
            bit_depth=detected["bit_depth"],
        )

        # Should successfully import
        assert mipmaps[0].shape == (detected["num_frames"], detected["frame_length"])


class TestHiresWavToStandardFormat:
    """Test conversion from high-resolution WAV to standardized format."""

    def test_serum_style_conversion(self, tmp_path: Path) -> None:
        """Test conversion of Serum-style wavetable WAV."""
        # Create Serum-style data: 256 frames of 2048 samples
        num_frames = 256
        frame_length = 2048

        # Generate morphing wavetable data
        audio_data = np.zeros((num_frames, frame_length), dtype=np.float32)
        for frame in range(num_frames):
            morph = frame / num_frames
            t = np.linspace(0, 2 * np.pi, frame_length, endpoint=False)
            # Morph from sine to saw
            sine = np.sin(t)
            saw = 2 * (t / (2 * np.pi)) - 1
            audio_data[frame] = (1 - morph) * sine + morph * saw

        # Flatten and save as WAV
        flat_audio = audio_data.flatten()
        wav_path = tmp_path / "serum_table.wav"
        sf.write(wav_path, flat_audio, 48000, subtype="FLOAT")

        # Import using hires importer
        mipmaps, metadata = import_hires_wav(
            wav_path,
            num_frames=num_frames,
        )

        # Verify import
        assert len(mipmaps) == 1
        assert mipmaps[0].shape == (256, 2048)
        assert metadata["wavetable_type"] == WavetableType.HIGH_RESOLUTION

        # Export to standardized format
        output_path = tmp_path / "converted_serum.wav"
        save_wavetable_wav(
            output_path,
            mipmaps=mipmaps,
            wavetable_type=metadata["wavetable_type"],
            name="Converted Serum Table",
            sample_rate=metadata["sample_rate"],
        )

        # Reload and verify
        loaded = load_wavetable_wav(output_path)
        assert loaded.mipmaps[0].shape == (256, 2048)
        assert loaded.wavetable_type == WavetableType.HIGH_RESOLUTION

    def test_detection_workflow(self, tmp_path: Path) -> None:
        """Test detecting WAV parameters and importing."""
        # Create WAV with specific dimensions
        num_frames = 64
        frame_length = 512
        audio = np.random.randn(num_frames * frame_length).astype(np.float32)
        wav_path = tmp_path / "unknown.wav"
        sf.write(wav_path, audio, 44100)

        # Detect wavetable parameters
        info = detect_wav_wavetable(wav_path)

        assert info["recommended"] is not None
        recommended = info["recommended"]

        # Import using recommended parameters
        mipmaps, metadata = import_hires_wav(
            wav_path,
            num_frames=recommended["num_frames"],
        )

        assert mipmaps[0].shape[0] == recommended["num_frames"]

    def test_stereo_to_mono_conversion(self, tmp_path: Path) -> None:
        """Test that stereo files are properly converted to mono."""
        num_frames = 32
        frame_length = 256

        # Create stereo audio
        left = np.random.randn(num_frames * frame_length).astype(np.float32)
        right = np.random.randn(num_frames * frame_length).astype(np.float32)
        stereo = np.column_stack([left, right])

        wav_path = tmp_path / "stereo.wav"
        sf.write(wav_path, stereo, 44100)

        # Import (should take first channel)
        mipmaps, _ = import_hires_wav(wav_path, num_frames=num_frames)

        # Verify shape (mono)
        assert mipmaps[0].shape == (num_frames, frame_length)

        # Export and reload
        output_path = tmp_path / "mono_converted.wav"
        save_wavetable_wav(
            output_path,
            mipmaps=mipmaps,
            wavetable_type=WavetableType.CUSTOM,
        )

        loaded = load_wavetable_wav(output_path)
        assert loaded.mipmaps[0].shape == (num_frames, frame_length)


class TestTypeInferenceIntegration:
    """Test type inference integration with conversion pipeline."""

    def test_infer_and_convert_classic_digital(self, tmp_path: Path) -> None:
        """Test inference of classic digital type and conversion."""
        # Create classic digital-style data
        wavetable = np.random.randn(64, 256).astype(np.float32)

        # Infer type
        wt_type, analysis = infer_wavetable_type(
            wavetable,
            source_bit_depth=8,
        )

        assert wt_type == WavetableType.CLASSIC_DIGITAL

        # Export with inferred type
        output_path = tmp_path / "inferred_classic.wav"
        save_wavetable_wav(
            output_path,
            mipmaps=[wavetable],
            wavetable_type=wt_type,
            source_bit_depth=8,
        )

        # Reload and verify type preserved
        loaded = load_wavetable_wav(output_path)
        assert loaded.wavetable_type == WavetableType.CLASSIC_DIGITAL

    def test_infer_and_convert_high_resolution(self, tmp_path: Path) -> None:
        """Test inference of high-resolution type and conversion."""
        # Create high-resolution style data
        wavetable = np.random.randn(256, 2048).astype(np.float32)

        # Infer type
        wt_type, analysis = infer_wavetable_type(wavetable)

        assert wt_type == WavetableType.HIGH_RESOLUTION

        # Export with inferred type
        output_path = tmp_path / "inferred_hires.wav"
        save_wavetable_wav(
            output_path,
            mipmaps=[wavetable],
            wavetable_type=wt_type,
        )

        # Reload and verify type preserved
        loaded = load_wavetable_wav(output_path)
        assert loaded.wavetable_type == WavetableType.HIGH_RESOLUTION

    def test_hardware_inference_integration(self, tmp_path: Path) -> None:
        """Test hardware-based inference with conversion."""
        wavetable = np.random.randn(64, 512).astype(np.float32)

        # Infer type from hardware name
        wt_type, _ = infer_wavetable_type(
            wavetable,
            source_hardware="Waldorf Microwave",
        )

        assert wt_type == WavetableType.CLASSIC_DIGITAL

        # Export with hardware metadata
        output_path = tmp_path / "waldorf.wav"
        save_wavetable_wav(
            output_path,
            mipmaps=[wavetable],
            wavetable_type=wt_type,
            author="Waldorf Microwave Export",
        )

        loaded = load_wavetable_wav(output_path)
        assert loaded.wavetable_type == WavetableType.CLASSIC_DIGITAL


class TestFullConversionPipeline:
    """Test complete conversion pipelines from various sources."""

    def test_raw_to_standard_round_trip(self, tmp_path: Path) -> None:
        """Test full round-trip: raw PCM -> standard -> reload."""
        frame_length = 256
        num_frames = 64

        # Create deterministic test data
        np.random.seed(42)
        raw_samples = (np.random.randn(frame_length * num_frames) * 127).astype(np.int8)

        raw_path = tmp_path / "input.raw"
        raw_path.write_bytes(raw_samples.tobytes())

        # Import raw
        mipmaps, metadata = import_raw_pcm(
            raw_path,
            frame_length=frame_length,
            num_frames=num_frames,
            bit_depth=8,
        )

        # Convert to standard format
        standard_path = tmp_path / "standard.wav"
        save_wavetable_wav(
            standard_path,
            mipmaps=mipmaps,
            wavetable_type=metadata["wavetable_type"],
            name="Round-trip Test",
            source_bit_depth=metadata["source_bit_depth"],
        )

        # Reload
        loaded = load_wavetable_wav(standard_path)

        # Verify
        assert loaded.wavetable_type == WavetableType.CLASSIC_DIGITAL
        assert loaded.mipmaps[0].shape == (64, 256)
        np.testing.assert_allclose(mipmaps[0], loaded.mipmaps[0], rtol=1e-5, atol=1e-5)

    def test_wav_to_standard_round_trip(self, tmp_path: Path) -> None:
        """Test full round-trip: WAV -> standard -> reload."""
        num_frames = 128
        frame_length = 1024

        # Create test WAV
        np.random.seed(123)
        audio = np.random.randn(num_frames * frame_length).astype(np.float32)
        source_wav = tmp_path / "source.wav"
        sf.write(source_wav, audio, 48000, subtype="FLOAT")

        # Import
        mipmaps, metadata = import_hires_wav(source_wav, num_frames=num_frames)

        # Convert to standard format
        standard_path = tmp_path / "standard_from_wav.wav"
        save_wavetable_wav(
            standard_path,
            mipmaps=mipmaps,
            wavetable_type=metadata["wavetable_type"],
            name="From WAV",
            sample_rate=metadata["sample_rate"],
        )

        # Reload and verify
        loaded = load_wavetable_wav(standard_path)
        assert loaded.wavetable_type == WavetableType.HIGH_RESOLUTION
        np.testing.assert_allclose(mipmaps[0], loaded.mipmaps[0], rtol=1e-5, atol=1e-5)

    def test_multi_mip_conversion(self, tmp_path: Path) -> None:
        """Test conversion with multiple mip levels."""
        num_frames = 64
        frame_length = 2048

        # Create source WAV
        audio = np.random.randn(num_frames * frame_length).astype(np.float32)
        source_wav = tmp_path / "source_for_mips.wav"
        sf.write(source_wav, audio, 48000)

        # Import with mip generation
        mipmaps, metadata = import_wav_with_mips(
            source_wav,
            num_frames=num_frames,
            num_mip_levels=5,
        )

        assert len(mipmaps) == 5

        # Convert to standard format
        standard_path = tmp_path / "multi_mip.wav"
        save_wavetable_wav(
            standard_path,
            mipmaps=mipmaps,
            wavetable_type=WavetableType.HIGH_RESOLUTION,
            name="Multi-Mip Test",
        )

        # Reload and verify all mip levels
        loaded = load_wavetable_wav(standard_path)
        assert len(loaded.mipmaps) == 5
        for i, (orig, load) in enumerate(zip(mipmaps, loaded.mipmaps, strict=True)):
            assert orig.shape == load.shape, f"Mip level {i} shape mismatch"

    def test_metadata_preservation(self, tmp_path: Path) -> None:
        """Test that metadata is preserved through conversion."""
        frame_length = 256
        num_frames = 64

        # Create raw data
        raw_data = bytes(frame_length * num_frames)
        raw_path = tmp_path / "meta_test.raw"
        raw_path.write_bytes(raw_data)

        # Import with metadata
        mipmaps, metadata = import_raw_pcm(
            raw_path,
            frame_length=frame_length,
            num_frames=num_frames,
            bit_depth=8,
        )

        # Export with full metadata
        output_path = tmp_path / "full_metadata.wav"
        save_wavetable_wav(
            output_path,
            mipmaps=mipmaps,
            wavetable_type=metadata["wavetable_type"],
            name="Metadata Test",
            author="Test Author",
            description="Testing metadata preservation",
            source_bit_depth=8,
            tuning_reference=432.0,
            sample_rate=44100,
        )

        # Reload and verify metadata
        loaded = load_wavetable_wav(output_path)
        assert loaded.name == "Metadata Test"
        assert loaded.author == "Test Author"
        assert loaded.description == "Testing metadata preservation"
        assert loaded.metadata.source_bit_depth == 8
        assert loaded.metadata.tuning_reference == pytest.approx(432.0)

    def test_all_wavetable_types_conversion(self, tmp_path: Path) -> None:
        """Test conversion works for all wavetable types."""
        wavetable = np.random.randn(32, 256).astype(np.float32)

        for wt_type in WavetableType:
            output_path = tmp_path / f"type_{wt_type.name.lower()}.wav"

            # Export
            save_wavetable_wav(
                output_path,
                mipmaps=[wavetable],
                wavetable_type=wt_type,
            )

            # Reload and verify type
            loaded = load_wavetable_wav(output_path)
            assert loaded.wavetable_type == wt_type, f"Type mismatch for {wt_type.name}"
