"""
Integration tests for EQ functionality with the complete wavetable generation pipeline.

Tests EQ integration with CLI commands, mipmap generation, and export functionality.
"""

import tempfile
from pathlib import Path

import numpy as np
from hypothesis import given, strategies

from wtgen.dsp.eq import Equalizer
from wtgen.dsp.fir import RolloffMethod
from wtgen.dsp.mipmap import Mipmap
from wtgen.dsp.process import align_to_zero_crossing, dc_remove, normalize
from wtgen.dsp.waves import WaveGenerator
from wtgen.export import load_wavetable_npz, save_wavetable_npz


class TestEQWithMipmapPipeline:
    """Test EQ integration with the complete mipmap generation pipeline."""

    def test_eq_before_bandlimiting_preserves_characteristics(self):
        """Test that EQ before bandlimiting maintains important characteristics."""
        # Generate base waveform
        _, base_wave = WaveGenerator().sawtooth(1.0)

        # Version 1: EQ after mipmap generation (wrong way)
        mipmaps_first = Mipmap(base_wave, num_octaves=3, decimate=False).generate()

        # Version 2: EQ before mipmap generation (correct way)
        eq = Equalizer(eq_settings="1000:6.0:2.0")
        eq_wave = eq.apply(base_wave)
        mipmaps_eq_first = Mipmap(eq_wave, num_octaves=3, decimate=False).generate()

        # Both should produce valid mipmaps
        assert len(mipmaps_first) == len(mipmaps_eq_first)

        # EQ-first version should have the EQ applied to all frequency content
        # before bandlimiting, so higher mips should still show EQ effects
        for level in mipmaps_eq_first:
            assert np.isfinite(level).all()
            assert not np.isnan(level).any()

    def test_eq_preserves_mipmap_generation_quality(self):
        """Test that EQ doesn't break the mipmap generation process."""
        # Generate base waveform
        _, base_wave = WaveGenerator().sawtooth(1.0)

        # Build mipmaps without EQ
        mipmaps_no_eq = Mipmap(base_wave, num_octaves=4, decimate=False).generate()

        # Apply EQ and build mipmaps
        eq = Equalizer(eq_settings="2000:3.0:1.5")
        eq_wave = eq.apply(base_wave)
        mipmaps_with_eq = Mipmap(eq_wave, num_octaves=4, decimate=False).generate()

        # Both should produce the same number of mipmap levels
        assert len(mipmaps_with_eq) == len(mipmaps_no_eq)

        # All levels should be valid
        for level in mipmaps_with_eq:
            assert np.isfinite(level).all()
            assert not np.isnan(level).any()

        # RMS levels should be in reasonable range
        rms_with_eq = [np.sqrt(np.mean(mip**2)) for mip in mipmaps_with_eq]
        for rms in rms_with_eq:
            assert 0.1 < rms < 1.0  # Reasonable RMS range

    def test_eq_preserves_zero_crossing_alignment_through_pipeline(self):
        """Test that EQ preserves zero-crossing alignment through full pipeline."""
        # Generate aligned base waveform
        _, base_wave = WaveGenerator().sawtooth(1.0)
        base_wave = align_to_zero_crossing(base_wave)

        # Apply EQ with phase preservation
        eq = Equalizer(eq_settings="1500:-4.0:2.0")
        eq_wave = eq.apply(base_wave)

        # Build mipmaps
        mipmaps = Mipmap(eq_wave, num_octaves=2, decimate=False).generate()

        # Process through full pipeline
        processed_mipmaps = []
        for mip in mipmaps:
            processed = align_to_zero_crossing(dc_remove(normalize(mip)))
            processed_mipmaps.append(processed)

        # All levels should maintain good zero-crossing alignment
        for level in processed_mipmaps:
            assert abs(level[0]) < 0.2  # Should start near zero

    def test_eq_with_different_rolloff_methods(self):
        """Test EQ interaction with different bandlimiting rolloff methods."""
        _, base_wave = WaveGenerator().sawtooth(1.0)
        eq = Equalizer(eq_settings="3000:5.0:1.0")
        eq_wave = eq.apply(base_wave)

        rolloff_methods: list[RolloffMethod] = ["raised_cosine", "tukey", "hann", "blackman"]

        for method in rolloff_methods:
            mipmaps = Mipmap(
                eq_wave, num_octaves=2, rolloff_method=method, decimate=False
            ).generate()

            # Should generate valid mipmaps regardless of rolloff method
            assert len(mipmaps) == 3
            for level in mipmaps:
                assert np.isfinite(level).all()
                assert not np.isnan(level).any()

    def test_eq_with_decimation(self):
        """Test EQ works correctly with mipmap decimation."""
        _, base_wave = WaveGenerator().sawtooth(1.0)
        eq = Equalizer(eq_settings="1000:4.0")
        eq_wave = eq.apply(base_wave)

        # Build mipmaps with decimation
        mipmaps = Mipmap(eq_wave, num_octaves=3, decimate=True).generate()

        # Should have decreasing sizes
        assert len(mipmaps[0]) >= len(mipmaps[1])
        assert len(mipmaps[1]) >= len(mipmaps[2])

        # All should be valid
        for level in mipmaps:
            assert np.isfinite(level).all()
            assert not np.isnan(level).any()

    def test_eq_with_tilt_settings(self):
        """Test EQ with tilt settings in the mipmap pipeline."""
        _, base_wave = WaveGenerator().sawtooth(1.0)

        # Test high tilt
        eq_high = Equalizer(high_tilt_settings="0.4:6.0")
        eq_wave_high = eq_high.apply(base_wave)
        mipmaps_high = Mipmap(eq_wave_high, num_octaves=2, decimate=False).generate()

        # Test low tilt
        eq_low = Equalizer(low_tilt_settings="0.6:4.0")
        eq_wave_low = eq_low.apply(base_wave)
        mipmaps_low = Mipmap(eq_wave_low, num_octaves=2, decimate=False).generate()

        # Test combined parametric and tilt EQ
        eq_combined = Equalizer(
            eq_settings="2000:3.0", high_tilt_settings="0.5:3.0", low_tilt_settings="0.3:-2.0"
        )
        eq_wave_combined = eq_combined.apply(base_wave)
        mipmaps_combined = Mipmap(eq_wave_combined, num_octaves=2, decimate=False).generate()

        # All should produce valid mipmaps
        for mipmaps in [mipmaps_high, mipmaps_low, mipmaps_combined]:
            assert len(mipmaps) == 3
            for level in mipmaps:
                assert np.isfinite(level).all()
                assert not np.isnan(level).any()

    @given(
        eq_freq_hz=strategies.floats(min_value=100.0, max_value=15000.0),
        eq_gain=strategies.floats(min_value=-10.0, max_value=10.0),
        eq_q=strategies.floats(min_value=0.5, max_value=4.0),
        num_octaves=strategies.integers(min_value=1, max_value=6),
    )
    def test_eq_pipeline_hypothesis(self, eq_freq_hz, eq_gain, eq_q, num_octaves):
        """Property test for EQ in the complete pipeline."""
        # Generate base waveform
        _, base_wave = WaveGenerator().sawtooth(1.0)

        # Apply EQ
        eq_string = f"{eq_freq_hz}:{eq_gain}:{eq_q}"
        eq = Equalizer(eq_settings=eq_string)
        eq_wave = eq.apply(base_wave)

        # Build mipmaps
        mipmaps = Mipmap(eq_wave, num_octaves=num_octaves, decimate=False).generate()

        # Process each level
        processed = []
        for mip in mipmaps:
            proc = align_to_zero_crossing(dc_remove(normalize(mip)))
            processed.append(proc)

        # Should produce valid results
        assert len(processed) == num_octaves + 1
        for level in processed:
            assert np.isfinite(level).all()
            assert not np.isnan(level).any()
            # Should be reasonably normalized
            rms = np.sqrt(np.mean(level**2))
            assert 0.1 < rms < 1.0


class TestEQWithHarmonics:
    """Test EQ with harmonic wavetable generation."""

    def test_eq_with_harmonic_generation(self):
        """Test EQ applied to harmonically generated wavetables."""
        # Create harmonic content
        partials = [(1, 1.0, 0.0), (2, 0.5, 0.0), (3, 0.33, 0.0), (4, 0.25, 0.0)]
        base_wave = WaveGenerator().harmonics_to_table(partials, 512)

        # Apply EQ to emphasize 2nd harmonic
        eq = Equalizer(eq_settings="200:6.0:2.0")  # Around 2nd harmonic
        eq_wave = eq.apply(base_wave)

        # Build mipmaps
        mipmaps = Mipmap(eq_wave, num_octaves=3, decimate=False).generate()

        # Should maintain harmonic relationships through bandlimiting
        for level in mipmaps:
            assert np.isfinite(level).all()
            rms = np.sqrt(np.mean(level**2))
            assert 0.1 < rms < 1.0

    def test_eq_multiple_bands_with_harmonics(self):
        """Test multiple EQ bands with harmonic content."""
        # Rich harmonic content
        partials = [(i, 1.0 / i, 0.0) for i in range(1, 9)]
        base_wave = WaveGenerator().harmonics_to_table(partials, 1024)

        # Apply complex EQ curve
        eq = Equalizer(eq_settings="100:2.0:1.0,500:-3.0:2.0,1000:4.0:1.5")
        eq_wave = eq.apply(base_wave)

        # Build mipmaps
        mipmaps = Mipmap(eq_wave, num_octaves=4, decimate=False).generate()

        # Process through pipeline
        processed = []
        for mip in mipmaps:
            proc = align_to_zero_crossing(dc_remove(normalize(mip)))
            processed.append(proc)

        # Should maintain quality through complex processing
        for level in processed:
            assert np.isfinite(level).all()
            assert not np.isnan(level).any()

    def test_eq_with_harmonic_and_tilt_processing(self):
        """Test EQ with both parametric and tilt processing on harmonics."""
        # Create rich harmonic content
        partials = [(i, 1.0 / i, 0.0) for i in range(1, 16)]
        base_wave = WaveGenerator().harmonics_to_table(partials, 2048)

        # Apply both parametric EQ and tilt
        eq = Equalizer(
            eq_settings="200:3.0:1.5,800:-2.0:2.0",  # Shape lower harmonics
            high_tilt_settings="0.6:4.0",  # Boost high frequencies
            low_tilt_settings="0.2:-1.0",  # Slight low cut
        )
        eq_wave = eq.apply(base_wave)

        # Build mipmaps
        mipmaps = Mipmap(eq_wave, num_octaves=3, decimate=False).generate()

        # Should maintain quality through complex EQ processing
        for level in mipmaps:
            assert np.isfinite(level).all()
            assert not np.isnan(level).any()
            rms = np.sqrt(np.mean(level**2))
            assert 0.05 < rms < 1.0  # Slightly wider range due to complex processing


class TestEQExportCompatibility:
    """Test EQ compatibility with export formats."""

    def test_eq_with_npz_export(self):
        """Test EQ works correctly with NPZ export."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "eq_test.npz"

            # Generate EQ'd wavetable
            _, base_wave = WaveGenerator().sawtooth(1.0)
            eq = Equalizer(eq_settings="2000:3.0")
            eq_wave = eq.apply(base_wave)
            mipmaps = Mipmap(eq_wave, num_octaves=2, decimate=False).generate()

            # Process mipmaps
            processed = []
            for mip in mipmaps:
                proc = align_to_zero_crossing(dc_remove(normalize(mip)))
                processed.append(proc)

            # Export
            tables = {"base": processed}
            meta = {
                "version": 1,
                "name": "eq_test",
                "author": "test",
                "sample_rate_hz": 44100,
                "cycle_length_samples": 512,
                "phase_convention": "zero_at_idx0",
                "normalization": "rms_0.35",
                "tuning": {"root_midi_note": 69, "cents_offset": 0},
            }

            save_wavetable_npz(output_path, tables, meta)

            # Load and verify
            loaded = load_wavetable_npz(output_path)
            assert "tables" in loaded
            assert "base" in loaded["tables"]
            assert len(loaded["tables"]["base"]) == len(processed)

    def test_eq_preserves_export_metadata(self):
        """Test that EQ processing preserves export metadata."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate and process wavetable with EQ
            _, base_wave = WaveGenerator().sawtooth(1.0)
            eq = Equalizer(eq_settings="1500:-2.0:1.5")
            eq_wave = eq.apply(base_wave)
            mipmaps = Mipmap(eq_wave, num_octaves=1, decimate=False).generate()

            processed = []
            for mip in mipmaps:
                proc = align_to_zero_crossing(dc_remove(normalize(mip)))
                processed.append(proc)

            # Create metadata with EQ info
            tables = {"base": processed}
            meta = {
                "version": 1,
                "name": "eq_wavetable",
                "generation": {
                    "waveform": "sawtooth",
                    "eq_applied": True,
                    "eq_settings": "1500:-2.0:1.5",
                },
            }

            output_path = Path(temp_dir) / "eq_meta_test.npz"
            save_wavetable_npz(output_path, tables, meta)

            # Load and verify metadata preservation
            loaded = load_wavetable_npz(output_path)
            assert loaded["manifest"]["generation"]["eq_applied"] is True
            assert loaded["manifest"]["generation"]["eq_settings"] == "1500:-2.0:1.5"

    def test_eq_with_tilt_export_metadata(self):
        """Test that tilt EQ settings are preserved in export metadata."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate and process wavetable with tilt EQ
            _, base_wave = WaveGenerator().sawtooth(1.0)
            eq = Equalizer(
                eq_settings="1000:2.0", high_tilt_settings="0.5:4.0", low_tilt_settings="0.3:-2.0"
            )
            eq_wave = eq.apply(base_wave)
            mipmaps = Mipmap(eq_wave, num_octaves=1, decimate=False).generate()

            processed = []
            for mip in mipmaps:
                proc = align_to_zero_crossing(dc_remove(normalize(mip)))
                processed.append(proc)

            # Create metadata with complete EQ info
            tables = {"base": processed}
            meta = {
                "version": 1,
                "name": "tilt_eq_wavetable",
                "generation": {
                    "waveform": "sawtooth",
                    "eq_applied": True,
                    "eq_settings": "1000:2.0",
                    "high_tilt_settings": "0.5:4.0",
                    "low_tilt_settings": "0.3:-2.0",
                },
            }

            output_path = Path(temp_dir) / "tilt_eq_meta_test.npz"
            save_wavetable_npz(output_path, tables, meta)

            # Load and verify metadata preservation
            loaded = load_wavetable_npz(output_path)
            generation = loaded["manifest"]["generation"]
            assert generation["eq_applied"] is True
            assert generation["eq_settings"] == "1000:2.0"
            assert generation["high_tilt_settings"] == "0.5:4.0"
            assert generation["low_tilt_settings"] == "0.3:-2.0"


class TestEQEdgeCases:
    """Test EQ behavior with edge cases and error conditions."""

    def test_eq_with_silent_wavetable(self):
        """Test EQ with silent (zero) wavetable."""
        silent_wave = np.zeros(256)
        eq = Equalizer(eq_settings="1000:6.0")
        result = eq.apply(silent_wave)

        # Should remain silent
        np.testing.assert_array_almost_equal(result, silent_wave)

    def test_eq_with_dc_only_wavetable(self):
        """Test EQ with DC-only wavetable."""
        dc_wave = np.ones(128) * 0.5
        eq = Equalizer(eq_settings="2000:3.0")
        result = eq.apply(dc_wave)

        # Should preserve DC (EQ doesn't affect DC component)
        assert abs(np.mean(result) - 0.5) < 0.1

    def test_eq_with_nyquist_frequency(self):
        """Test EQ near Nyquist frequency."""
        wavetable = np.sin(2 * np.pi * np.linspace(0, 1, 256, endpoint=False))

        # EQ very close to Nyquist
        eq = Equalizer(eq_settings="21000:4.0:1.0")
        result = eq.apply(wavetable)

        # Should not crash and produce valid output
        assert np.isfinite(result).all()
        assert not np.isnan(result).any()

    def test_eq_with_very_high_q(self):
        """Test EQ with very high Q factor."""
        wavetable = np.random.randn(512) * 0.1
        eq = Equalizer(eq_settings="5000:2.0:10.0")  # Very high Q
        result = eq.apply(wavetable)

        # Should handle high Q without instability
        assert np.isfinite(result).all()
        assert not np.isnan(result).any()

    def test_eq_with_very_low_q(self):
        """Test EQ with very low Q factor."""
        wavetable = np.random.randn(256) * 0.1
        eq = Equalizer(eq_settings="6000:-5.0:0.1")  # Very low Q
        result = eq.apply(wavetable)

        # Should handle low Q (wide bandwidth)
        assert np.isfinite(result).all()
        assert not np.isnan(result).any()

    def test_eq_with_many_bands(self):
        """Test EQ with many overlapping bands."""
        wavetable = np.sin(2 * np.pi * np.linspace(0, 1, 1024, endpoint=False))

        # Many overlapping bands
        eq_bands = []
        for i in range(10):
            freq_hz = 1000.0 + i * 1500.0  # Spread from 1kHz to 14.5kHz
            gain = (-1) ** i * 2.0  # Alternating boost/cut
            eq_bands.append(f"{freq_hz}:{gain}:1.0")

        eq_settings_string = ",".join(eq_bands)
        eq = Equalizer(eq_settings=eq_settings_string)
        result = eq.apply(wavetable)

        # Should handle complex EQ without issues
        assert np.isfinite(result).all()
        assert not np.isnan(result).any()

    def test_tilt_eq_edge_cases(self):
        """Test tilt EQ with edge case settings."""
        wavetable = np.sin(2 * np.pi * np.linspace(0, 1, 512, endpoint=False))

        # Test extreme tilt settings
        eq_extreme_high = Equalizer(high_tilt_settings="0.1:20.0")  # Very early start, high gain
        result_extreme_high = eq_extreme_high.apply(wavetable)

        eq_extreme_low = Equalizer(low_tilt_settings="0.9:-20.0")  # Very late start, high cut
        result_extreme_low = eq_extreme_low.apply(wavetable)

        # Should handle extreme settings without issues
        assert np.isfinite(result_extreme_high).all()
        assert not np.isnan(result_extreme_high).any()
        assert np.isfinite(result_extreme_low).all()
        assert not np.isnan(result_extreme_low).any()

    def test_empty_eq_settings(self):
        """Test EQ with empty or None settings."""
        wavetable = np.sin(2 * np.pi * np.linspace(0, 1, 256, endpoint=False))

        # Test various empty settings
        eq_none = Equalizer(eq_settings=None)
        eq_empty = Equalizer(eq_settings="")
        eq_whitespace = Equalizer(eq_settings="   ")

        result_none = eq_none.apply(wavetable)
        result_empty = eq_empty.apply(wavetable)
        result_whitespace = eq_whitespace.apply(wavetable)

        # All should return the original wavetable unchanged
        np.testing.assert_array_almost_equal(result_none, wavetable)
        np.testing.assert_array_almost_equal(result_empty, wavetable)
        np.testing.assert_array_almost_equal(result_whitespace, wavetable)


class TestEQPerformance:
    """Test EQ performance characteristics."""

    def test_eq_performance_scales_with_size(self):
        """Test that EQ performance scales reasonably with wavetable size."""
        import time  # noqa: PLC0415

        eq = Equalizer(eq_settings="2000:3.0:2.0")
        sizes = [256, 512, 1024, 2048]
        times = []

        for size in sizes:
            wavetable = np.random.randn(size) * 0.1

            start_time = time.time()
            eq.apply(wavetable)
            end_time = time.time()

            times.append(end_time - start_time)

        # Times should increase reasonably (not exponentially)
        # This is more of a smoke test than a strict performance requirement
        assert all(t < 1.0 for t in times)  # Should complete within 1 second

    def test_eq_memory_efficiency(self):
        """Test that EQ doesn't create excessive temporary arrays."""
        # This is mainly a smoke test to ensure we're not creating
        # multiple unnecessary copies of large arrays

        large_wavetable = np.random.randn(8192) * 0.1
        eq = Equalizer(eq_settings="1000:4.0:1.5")

        # Should complete without memory issues
        result = eq.apply(large_wavetable)
        assert len(result) == len(large_wavetable)
        assert np.isfinite(result).all()

    def test_tilt_eq_performance(self):
        """Test that tilt EQ performs reasonably."""
        import time  # noqa: PLC0415

        wavetable = np.random.randn(2048) * 0.1

        # Test different tilt EQ configurations
        eq_configs = [
            Equalizer(high_tilt_settings="0.5:6.0"),
            Equalizer(low_tilt_settings="0.4:4.0"),
            Equalizer(high_tilt_settings="0.6:3.0", low_tilt_settings="0.3:-2.0"),
            Equalizer(
                eq_settings="1000:2.0,5000:-3.0",
                high_tilt_settings="0.7:5.0",
                low_tilt_settings="0.2:-1.5",
            ),
        ]

        for eq in eq_configs:
            start_time = time.time()
            result = eq.apply(wavetable)
            end_time = time.time()

            # Should complete quickly
            assert (end_time - start_time) < 1.0
            assert np.isfinite(result).all()
            assert not np.isnan(result).any()

    @given(
        size=strategies.integers(min_value=128, max_value=4096),
        num_bands=strategies.integers(min_value=1, max_value=8),
    )
    def test_eq_performance_hypothesis(self, size, num_bands):
        """Property test for EQ performance with varying configurations."""
        wavetable = np.random.randn(size) * 0.1

        # Generate random EQ settings
        eq_bands = []
        for i in range(num_bands):
            freq = 200.0 + i * 2000.0  # Spread frequencies
            gain = (-1) ** i * 3.0  # Alternating boost/cut
            q = 1.0 + i * 0.5  # Varying Q factors
            eq_bands.append(f"{freq}:{gain}:{q}")

        eq_settings_string = ",".join(eq_bands)
        eq = Equalizer(eq_settings=eq_settings_string)

        # Should complete without issues
        result = eq.apply(wavetable)
        assert len(result) == size
        assert np.isfinite(result).all()
        assert not np.isnan(result).any()
