"""
Integration tests for EQ functionality with the complete wavetable generation pipeline.

Tests EQ integration with CLI commands, mipmap generation, and export functionality.
"""

import tempfile
from pathlib import Path

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from wtgen.dsp.eq import apply_parametric_eq_fft, create_eq_band
from wtgen.dsp.mipmap import build_mipmap
from wtgen.dsp.process import align_to_zero_crossing, dc_remove, normalize
from wtgen.dsp.waves import generate_sawtooth_wavetable, harmonics_to_table
from wtgen.export import load_wavetable_npz, save_wavetable_npz


class TestEQWithMipmapPipeline:
    """Test EQ integration with the complete mipmap generation pipeline."""

    def test_eq_before_bandlimiting_preserves_characteristics(self):
        """Test that EQ before bandlimiting maintains important characteristics."""
        # Generate base waveform
        _, base_wave = generate_sawtooth_wavetable(1.0)

        # Version 1: EQ after mipmap generation (wrong way)
        mipmaps_first = build_mipmap(base_wave, num_octaves=3, decimate=False)
        eq_bands = [create_eq_band(1000.0, 6.0, 2.0)]

        # Version 2: EQ before mipmap generation (correct way)
        eq_wave = apply_parametric_eq_fft(
            base_wave, eq_bands, preserve_rms=True, preserve_phase=True
        )
        mipmaps_eq_first = build_mipmap(eq_wave, num_octaves=3, decimate=False)

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
        _, base_wave = generate_sawtooth_wavetable(1.0)

        # Build mipmaps without EQ
        mipmaps_no_eq = build_mipmap(base_wave, num_octaves=4, decimate=False)

        # Apply EQ and build mipmaps
        eq_bands = [create_eq_band(2000.0, 3.0, 1.5)]
        eq_wave = apply_parametric_eq_fft(base_wave, eq_bands, preserve_rms=True)
        mipmaps_with_eq = build_mipmap(eq_wave, num_octaves=4, decimate=False)

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
        _, base_wave = generate_sawtooth_wavetable(1.0)
        base_wave = align_to_zero_crossing(base_wave)

        # Apply EQ
        eq_bands = [create_eq_band(1500.0, -4.0, 2.0)]
        eq_wave = apply_parametric_eq_fft(base_wave, eq_bands, preserve_phase=True)

        # Build mipmaps
        mipmaps = build_mipmap(eq_wave, num_octaves=2, decimate=False)

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
        _, base_wave = generate_sawtooth_wavetable(1.0)
        eq_bands = [create_eq_band(3000.0, 5.0, 1.0)]
        eq_wave = apply_parametric_eq_fft(base_wave, eq_bands)

        rolloff_methods = ["raised_cosine", "tukey", "hann", "blackman"]

        for method in rolloff_methods:
            mipmaps = build_mipmap(eq_wave, num_octaves=2, rolloff_method=method, decimate=False)

            # Should generate valid mipmaps regardless of rolloff method
            assert len(mipmaps) == 3
            for level in mipmaps:
                assert np.isfinite(level).all()
                assert not np.isnan(level).any()

    def test_eq_with_decimation(self):
        """Test EQ works correctly with mipmap decimation."""
        _, base_wave = generate_sawtooth_wavetable(1.0)
        eq_bands = [create_eq_band(1000.0, 4.0)]
        eq_wave = apply_parametric_eq_fft(base_wave, eq_bands)

        # Build mipmaps with decimation
        mipmaps = build_mipmap(eq_wave, num_octaves=3, decimate=True)

        # Should have decreasing sizes
        assert len(mipmaps[0]) >= len(mipmaps[1])
        assert len(mipmaps[1]) >= len(mipmaps[2])

        # All should be valid
        for level in mipmaps:
            assert np.isfinite(level).all()
            assert not np.isnan(level).any()

    @given(
        eq_freq_hz=st.floats(min_value=100.0, max_value=15000.0),
        eq_gain=st.floats(min_value=-10.0, max_value=10.0),
        eq_q=st.floats(min_value=0.5, max_value=4.0),
        num_octaves=st.integers(min_value=1, max_value=6),
    )
    def test_eq_pipeline_hypothesis(self, eq_freq_hz, eq_gain, eq_q, num_octaves):
        """Property test for EQ in the complete pipeline."""
        # Generate base waveform
        _, base_wave = generate_sawtooth_wavetable(1.0)

        # Apply EQ
        eq_bands = [create_eq_band(eq_freq_hz, eq_gain, eq_q)]
        eq_wave = apply_parametric_eq_fft(base_wave, eq_bands)

        # Build mipmaps
        mipmaps = build_mipmap(eq_wave, num_octaves=num_octaves, decimate=False)

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
        base_wave = harmonics_to_table(partials, 512)

        # Apply EQ to emphasize 2nd harmonic
        eq_bands = [create_eq_band(200.0, 6.0, 2.0)]  # Around 2nd harmonic
        eq_wave = apply_parametric_eq_fft(base_wave, eq_bands, preserve_rms=True)

        # Build mipmaps
        mipmaps = build_mipmap(eq_wave, num_octaves=3, decimate=False)

        # Should maintain harmonic relationships through bandlimiting
        for level in mipmaps:
            assert np.isfinite(level).all()
            rms = np.sqrt(np.mean(level**2))
            assert 0.1 < rms < 1.0

    def test_eq_multiple_bands_with_harmonics(self):
        """Test multiple EQ bands with harmonic content."""
        # Rich harmonic content
        partials = [(i, 1.0 / i, 0.0) for i in range(1, 9)]
        base_wave = harmonics_to_table(partials, 1024)

        # Apply complex EQ curve
        eq_bands = [
            create_eq_band(100.0, 2.0, 1.0),  # Boost fundamental
            create_eq_band(500.0, -3.0, 2.0),  # Cut mid harmonics
            create_eq_band(1000.0, 4.0, 1.5),  # Boost upper harmonics
        ]
        eq_wave = apply_parametric_eq_fft(base_wave, eq_bands)

        # Build mipmaps
        mipmaps = build_mipmap(eq_wave, num_octaves=4, decimate=False)

        # Process through pipeline
        processed = []
        for mip in mipmaps:
            proc = align_to_zero_crossing(dc_remove(normalize(mip)))
            processed.append(proc)

        # Should maintain quality through complex processing
        for level in processed:
            assert np.isfinite(level).all()
            assert not np.isnan(level).any()


class TestEQExportCompatibility:
    """Test EQ compatibility with export formats."""

    def test_eq_with_npz_export(self):
        """Test EQ works correctly with NPZ export."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "eq_test.npz"

            # Generate EQ'd wavetable
            _, base_wave = generate_sawtooth_wavetable(1.0)
            eq_bands = [create_eq_band(2000.0, 3.0)]
            eq_wave = apply_parametric_eq_fft(base_wave, eq_bands)
            mipmaps = build_mipmap(eq_wave, num_octaves=2, decimate=False)

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
            _, base_wave = generate_sawtooth_wavetable(1.0)
            eq_bands = [create_eq_band(1500.0, -2.0, 1.5)]
            eq_wave = apply_parametric_eq_fft(base_wave, eq_bands)
            mipmaps = build_mipmap(eq_wave, num_octaves=1, decimate=False)

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
                    "eq_bands": [{"frequency_hz": 1500.0, "gain_db": -2.0, "q_factor": 1.5}],
                },
            }

            output_path = Path(temp_dir) / "eq_meta_test.npz"
            save_wavetable_npz(output_path, tables, meta)

            # Load and verify metadata preservation
            loaded = load_wavetable_npz(output_path)
            assert loaded["manifest"]["generation"]["eq_applied"] is True
            assert len(loaded["manifest"]["generation"]["eq_bands"]) == 1


class TestEQEdgeCases:
    """Test EQ behavior with edge cases and error conditions."""

    def test_eq_with_silent_wavetable(self):
        """Test EQ with silent (zero) wavetable."""
        silent_wave = np.zeros(256)
        eq_bands = [create_eq_band(1000.0, 6.0)]
        result = apply_parametric_eq_fft(silent_wave, eq_bands)

        # Should remain silent
        np.testing.assert_array_almost_equal(result, silent_wave)

    def test_eq_with_dc_only_wavetable(self):
        """Test EQ with DC-only wavetable."""
        dc_wave = np.ones(128) * 0.5
        eq_bands = [create_eq_band(2000.0, 3.0)]
        result = apply_parametric_eq_fft(dc_wave, eq_bands)

        # Should preserve DC (EQ doesn't affect DC component)
        assert abs(np.mean(result) - 0.5) < 0.1

    def test_eq_with_nyquist_frequency(self):
        """Test EQ near Nyquist frequency."""
        wavetable = np.sin(2 * np.pi * np.linspace(0, 1, 256, endpoint=False))

        # EQ very close to Nyquist
        eq_bands = [create_eq_band(21000.0, 4.0, 1.0)]
        result = apply_parametric_eq_fft(wavetable, eq_bands)

        # Should not crash and produce valid output
        assert np.isfinite(result).all()
        assert not np.isnan(result).any()

    def test_eq_with_very_high_q(self):
        """Test EQ with very high Q factor."""
        wavetable = np.random.randn(512) * 0.1
        eq_bands = [create_eq_band(5000.0, 2.0, 10.0)]  # Very high Q
        result = apply_parametric_eq_fft(wavetable, eq_bands)

        # Should handle high Q without instability
        assert np.isfinite(result).all()
        assert not np.isnan(result).any()

    def test_eq_with_very_low_q(self):
        """Test EQ with very low Q factor."""
        wavetable = np.random.randn(256) * 0.1
        eq_bands = [create_eq_band(6000.0, -5.0, 0.1)]  # Very low Q
        result = apply_parametric_eq_fft(wavetable, eq_bands)

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
            eq_bands.append(create_eq_band(freq_hz, gain, 1.0))

        result = apply_parametric_eq_fft(wavetable, eq_bands)

        # Should handle complex EQ without issues
        assert np.isfinite(result).all()
        assert not np.isnan(result).any()


class TestEQPerformance:
    """Test EQ performance characteristics."""

    def test_eq_performance_scales_with_size(self):
        """Test that EQ performance scales reasonably with wavetable size."""
        import time  # noqa: PLC0415

        eq_bands = [create_eq_band(2000.0, 3.0, 2.0)]
        sizes = [256, 512, 1024, 2048]
        times = []

        for size in sizes:
            wavetable = np.random.randn(size) * 0.1

            start_time = time.time()
            apply_parametric_eq_fft(wavetable, eq_bands)
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
        eq_bands = [create_eq_band(1000.0, 4.0, 1.5)]

        # Should complete without memory issues
        result = apply_parametric_eq_fft(large_wavetable, eq_bands)
        assert len(result) == len(large_wavetable)
        assert np.isfinite(result).all()
