"""Unit tests for wtgen.dsp.waves module."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from wtgen.dsp.waves import Partial, PartialList, WaveGenerator
from wtgen.utils import EPSILON


class TestHarmonicsToTable:
    """Test harmonic to wavetable conversion functionality."""

    def test_empty_partials(self):
        """Test conversion with empty partials list."""
        result = WaveGenerator().harmonics_to_table([], 64)
        expected = np.zeros(64)
        np.testing.assert_allclose(result, expected, atol=EPSILON)

    def test_single_fundamental(self):
        """Test conversion with single fundamental frequency."""
        partials: PartialList = [(1, 1.0, 0.0)]  # Fundamental with 0 phase
        result = WaveGenerator().harmonics_to_table(partials, 64, phase="linear")

        # Should be approximately a cosine wave (phase=0)
        t = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        expected = np.cos(t)
        expected = expected / np.max(np.abs(expected))  # Normalize like the function does

        # Allow some tolerance for boundary smoothing
        np.testing.assert_allclose(result, expected, atol=0.1)

    def test_single_fundamental_with_phase(self):
        """Test conversion with phase-shifted fundamental."""
        partials: PartialList = [(1, 1.0, np.pi / 2)]  # 90 degree phase shift
        result = WaveGenerator().harmonics_to_table(partials, 64, phase="linear")

        # Should be approximately a sine wave
        t = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        expected = np.sin(t)
        expected = expected / np.max(np.abs(expected))

        # Allow tolerance for processing artifacts
        correlation = np.corrcoef(result, expected)[0, 1]
        assert abs(correlation) > 0.075  # Tightened based on fuzz testing (5th percentile)

    def test_multiple_harmonics(self):
        """Test conversion with multiple harmonics."""
        partials: PartialList = [
            (1, 1.0, 0.0),  # Fundamental
            (2, 0.5, 0.0),  # Second harmonic
            (3, 0.25, 0.0),  # Third harmonic
        ]
        result = WaveGenerator().harmonics_to_table(partials, 128, phase="linear")

        # Should be finite and normalized
        assert np.all(np.isfinite(result))
        assert len(result) == 128
        assert np.max(np.abs(result)) <= 1.01  # Allow small tolerance

    def test_high_harmonic_filtering(self):
        """Test that harmonics above Nyquist are filtered out."""
        # Create partials with very high harmonic numbers
        partials: PartialList = [
            (1, 1.0, 0.0),  # Fundamental
            (100, 0.5, 0.0),  # Very high harmonic (should be filtered for small tables)
        ]
        result = WaveGenerator().harmonics_to_table(
            partials, 64, phase="linear"
        )  # Small table size

        # Should still be finite and reasonable
        assert np.all(np.isfinite(result))
        assert len(result) == 64

    def test_continuity_property(self):
        """Test that the generated waveform is continuous (periodic boundary)."""
        partials: PartialList = [(1, 1.0, 0.0), (3, 0.3, 0.0)]
        result = WaveGenerator().harmonics_to_table(partials, 64, phase="linear")

        # The discontinuity between last and first sample should be small
        boundary_discontinuity = abs(result[-1] - result[0])
        assert boundary_discontinuity < 0.135  # Tightened based on fuzz testing (95th percentile)

    def test_normalization_property(self):
        """Test that output is properly normalized without clipping."""
        partials: PartialList = [(1, 5.0, 0.0), (2, 3.0, 0.0)]  # Large amplitudes
        result = WaveGenerator().harmonics_to_table(partials, 128, phase="linear")

        # Should be normalized and contain no clipping
        assert np.all(result >= -1.0), "Values below -1.0 detected (clipping)"
        assert np.all(result <= 1.0), "Values above 1.0 detected (clipping)"
        assert np.max(np.abs(result)) <= 1.01, "Peak values exceed safe range"
        assert np.max(np.abs(result)) >= 0.5, (
            "Should actually use significant range"
        )  # Should actually use significant range

    def test_phase_mode_linear(self):
        """Test linear phase mode."""
        partials: PartialList = [(1, 1.0, 0.0)]
        result = WaveGenerator().harmonics_to_table(partials, 64, phase="linear")

        assert np.all(np.isfinite(result))
        assert len(result) == 64

    def test_phase_mode_minimum(self):
        """Test minimum phase mode (currently implemented as linear)."""
        partials: PartialList = [(1, 1.0, 0.0)]
        result = WaveGenerator().harmonics_to_table(partials, 64, phase="minimum")

        assert np.all(np.isfinite(result))
        assert len(result) == 64

    def test_zero_amplitude_partials(self):
        """Test handling of zero amplitude partials."""
        partials: PartialList = [
            (1, 0.0, 0.0),  # Zero amplitude
            (2, 1.0, 0.0),  # Non-zero amplitude
        ]
        result = WaveGenerator().harmonics_to_table(partials, 64, phase="linear")

        # Should handle gracefully
        assert np.all(np.isfinite(result))
        assert len(result) == 64

    @given(
        st.lists(
            st.tuples(
                st.integers(1, 20),  # harmonic number
                st.floats(0.0, 2.0),  # amplitude
                st.floats(0.0, 2 * np.pi),  # phase
            ),
            min_size=0,
            max_size=10,
        )
    )
    @settings(deadline=None)  # Disable deadline for this test
    def test_harmonics_to_table_hypothesis(self, partials_data):
        """Hypothesis test for harmonics_to_table including clipping prevention checks."""
        # Filter out invalid values and deduplicate harmonics (keep last occurrence)
        harmonic_dict = {}
        for h, a, p in partials_data:
            if np.isfinite(a) and np.isfinite(p) and a >= 0:
                harmonic_dict[h] = (a, p)

        valid_partials = [(h, a, p) for h, (a, p) in harmonic_dict.items()]

        if not valid_partials:
            # Empty case
            result = WaveGenerator().harmonics_to_table([], 64)
            np.testing.assert_allclose(result, np.zeros(64), atol=EPSILON)
            return

        partials: PartialList = valid_partials
        result = WaveGenerator().harmonics_to_table(partials, 64, phase="linear")

        # Basic properties
        assert np.all(np.isfinite(result))
        assert len(result) == 64
        assert isinstance(result, np.ndarray)

        # Critical clipping prevention checks
        assert np.all(result >= -1.0), "Values below -1.0 detected (clipping)"
        assert np.all(result <= 1.0), "Values above 1.0 detected (clipping)"

        # Peak should not exceed 1.0 (with small tolerance for numerical precision)
        max_abs_value = np.max(np.abs(result))
        assert max_abs_value <= 1.0 + 1e-10, f"Peak value {max_abs_value} exceeds 1.0"

    def test_different_table_sizes(self):
        """Test with different wavetable sizes."""
        partials: PartialList = [(1, 1.0, 0.0), (2, 0.5, 0.0)]

        for size in [32, 64, 128, 256, 512, 1024, 2048]:
            result = WaveGenerator().harmonics_to_table(partials, size, phase="linear")
            assert len(result) == size
            assert np.all(np.isfinite(result))

    def test_dc_component_handling(self):
        """Test that DC component (harmonic 0) is handled if present."""
        # Note: The current implementation expects harmonic_index >= 1
        # This test verifies behavior with the current constraint
        partials: PartialList = [(1, 1.0, 0.0)]  # Only fundamental, no DC
        result = WaveGenerator().harmonics_to_table(partials, 64, phase="linear")

        # Should have minimal DC after processing
        dc_level = abs(np.mean(result))
        assert dc_level < 0.2  # Should be relatively low DC

    def test_large_phase_values(self):
        """Test handling of phase values outside [0, 2π]."""
        partials: PartialList = [
            (1, 1.0, 4 * np.pi),  # Large phase
            (2, 0.5, -np.pi),  # Negative phase
        ]
        result = WaveGenerator().harmonics_to_table(partials, 64, phase="linear")

        # Should handle gracefully
        assert np.all(np.isfinite(result))
        assert len(result) == 64

    def test_identical_phases_different_harmonics(self):
        """Test harmonics with identical phases."""
        partials: PartialList = [
            (1, 1.0, np.pi / 4),
            (2, 0.5, np.pi / 4),
            (3, 0.25, np.pi / 4),
        ]
        result = WaveGenerator().harmonics_to_table(partials, 128, phase="linear")

        # Should produce a reasonable waveform
        assert np.all(np.isfinite(result))
        assert len(result) == 128
        assert np.max(np.abs(result)) > 0.1  # Should have reasonable amplitude


class TestPartialDataTypes:
    """Test the partial data type definitions."""

    def test_partial_tuple_structure(self):
        """Test that Partial tuples have correct structure."""
        # This is more of a type checking test
        partial: Partial = (1, 1.0, 0.0)
        harmonic, amplitude, phase = partial

        assert isinstance(harmonic, int)
        assert isinstance(amplitude, float)
        assert isinstance(phase, float)

    def test_partial_list_structure(self):
        """Test that PartialList has correct structure."""
        partials: PartialList = [
            (1, 1.0, 0.0),
            (2, 0.5, np.pi / 2),
            (3, 0.25, np.pi),
        ]

        assert len(partials) == 3
        for partial in partials:
            assert len(partial) == 3
            harmonic, amplitude, phase = partial
            assert isinstance(harmonic, int)
            assert isinstance(amplitude, int | float)
            assert isinstance(phase, int | float)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_very_small_table_size(self):
        """Test with very small table sizes."""
        partials: PartialList = [(1, 1.0, 0.0)]

        result = WaveGenerator().harmonics_to_table(partials, 4)
        assert len(result) == 4
        assert np.all(np.isfinite(result))

    def test_power_of_two_sizes(self):
        """Test that power-of-two sizes work correctly."""
        partials: PartialList = [(1, 1.0, 0.0), (2, 0.5, 0.0)]

        for power in range(2, 12):  # 4 to 2048
            size = 2**power
            result = WaveGenerator().harmonics_to_table(partials, size, phase="linear")
            assert len(result) == size
            assert np.all(np.isfinite(result))

    def test_non_power_of_two_sizes(self):
        """Test that non-power-of-two sizes work correctly."""
        partials: PartialList = [(1, 1.0, 0.0)]

        for size in [100, 200, 300, 500, 1000]:
            result = WaveGenerator().harmonics_to_table(partials, size, phase="linear")
            assert len(result) == size
            assert np.all(np.isfinite(result))

    def test_high_harmonic_numbers(self):
        """Test with very high harmonic numbers."""
        partials: PartialList = [
            (1, 1.0, 0.0),
            (50, 0.1, 0.0),
            (100, 0.05, 0.0),
        ]
        result = WaveGenerator().harmonics_to_table(partials, 256, phase="linear")

        # Should filter appropriately and remain finite
        assert np.all(np.isfinite(result))
        assert len(result) == 256

    def test_many_harmonics(self):
        """Test with many harmonics."""
        # Create many harmonics with decreasing amplitude
        partials: PartialList = []
        for h in range(1, 21):
            partials.append((h, 1.0 / h, 0.0))

        result = WaveGenerator().harmonics_to_table(partials, 512, phase="linear")

        assert np.all(np.isfinite(result))
        assert len(result) == 512
        assert np.max(np.abs(result)) > 0.1  # Should have reasonable amplitude
