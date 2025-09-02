"""Unit tests for wtgen.dsp.process module."""

import numpy as np
from hypothesis import assume, given
from hypothesis import strategies as st

from wtgen.dsp.process import (
    align_to_zero_crossing,
    dc_remove,
    ensure_consistent_phase_alignment,
    min_phase_realign,
    normalize,
    normalize_to_range,
)


class TestDCRemove:
    """Test DC offset removal functionality."""

    def test_dc_remove_zero_mean(self):
        """Test that dc_remove produces zero-mean output."""
        wavetable = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = dc_remove(wavetable)
        assert abs(np.mean(result)) < 1e-15  # Tightened based on fuzz testing

    def test_dc_remove_already_zero_mean(self):
        """Test dc_remove on already zero-mean signal."""
        wavetable = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = dc_remove(wavetable)
        np.testing.assert_allclose(result, wavetable, atol=1e-15)

    def test_dc_remove_constant_signal(self):
        """Test dc_remove on constant signal."""
        wavetable = np.array([5.0, 5.0, 5.0, 5.0])
        result = dc_remove(wavetable)
        np.testing.assert_allclose(result, np.zeros(4), atol=1e-15)

    @given(st.lists(st.floats(-1000, 1000), min_size=1, max_size=2048))
    def test_dc_remove_hypothesis(self, values):
        """Hypothesis test for dc_remove."""
        assume(all(np.isfinite(v) for v in values))
        wavetable = np.array(values)
        result = dc_remove(wavetable)

        # Result should have zero mean
        assert (
            abs(np.mean(result)) < 1e-12
        )  # Tightened from 1e-10, balancing precision with robustness
        # Should preserve shape
        assert result.shape == wavetable.shape


class TestNormalize:
    """Test amplitude normalization functionality."""

    def test_normalize_basic(self):
        """Test basic normalization to default peak."""
        wavetable = np.array([2.0, -4.0, 1.0])
        result = normalize(wavetable)
        # The true peak (including inter-sample peaks) should not exceed 0.999
        from src.wtgen.dsp.process import _estimate_inter_sample_peak
        true_peak = _estimate_inter_sample_peak(result)
        assert true_peak <= 0.999 + 1e-10

    def test_normalize_custom_peak(self):
        """Test normalization to custom peak value."""
        wavetable = np.array([2.0, -4.0, 1.0])
        result = normalize(wavetable, peak=0.5)
        # The true peak (including inter-sample peaks) should not exceed 0.5
        from src.wtgen.dsp.process import _estimate_inter_sample_peak
        true_peak = _estimate_inter_sample_peak(result)
        assert true_peak <= 0.5 + 1e-10

    def test_normalize_zero_signal(self):
        """Test normalization of zero signal."""
        wavetable = np.zeros(4)
        result = normalize(wavetable)
        np.testing.assert_allclose(result, np.zeros(4), atol=1e-12)

    @given(st.lists(st.floats(-1000, 1000), min_size=1, max_size=2048))
    def test_normalize_hypothesis(self, values):
        """Hypothesis test for normalize."""
        assume(all(np.isfinite(v) for v in values))

        wavetable = np.array(values)
        max_val = max(abs(v) for v in values)
        result = normalize(wavetable)

        # Should preserve shape
        assert result.shape == wavetable.shape

        if max_val > 1e-12:  # Non-zero signal
            # The true peak (including inter-sample peaks) should not exceed the target
            from src.wtgen.dsp.process import _estimate_inter_sample_peak
            true_peak = _estimate_inter_sample_peak(result)
            assert true_peak <= 0.999 + 1e-10  # Allow small numerical tolerance
            
            # Sample peak may be lower than 0.999 due to inter-sample peak prevention
            sample_peak = np.max(np.abs(result))
            assert sample_peak <= 0.999 + 1e-10
        else:  # Near-zero signal should become exactly zero
            assert np.all(np.abs(result) < 1e-10)  # Near-zero signal should become very small


class TestNormalizeToRange:
    """Test range normalization functionality."""

    def test_normalize_to_range_basic(self):
        """Test basic range normalization (maintains zero mean)."""
        wavetable = np.array([1.0, 3.0, 2.0])
        result = normalize_to_range(wavetable)

        # Should have zero mean
        assert (
            abs(np.mean(result)) < 1e-12
        )  # Tightened from 1e-10, balancing precision with robustness
        # Should use target range
        range_used = np.max(result) - np.min(result)
        target_range = 0.999 - (-0.999)
        assert abs(range_used - target_range) < 1e-15

    def test_normalize_to_range_custom(self):
        """Test range normalization with custom range (maintains zero mean)."""
        wavetable = np.array([1.0, 5.0, 3.0])
        result = normalize_to_range(wavetable, target_min=-0.5, target_max=0.8)

        # Should have zero mean
        assert (
            abs(np.mean(result)) < 1e-12
        )  # Tightened from 1e-10, balancing precision with robustness
        # Should use the target range
        range_used = np.max(result) - np.min(result)
        target_range = 0.8 - (-0.5)
        assert abs(range_used - target_range) < 1e-15

    def test_normalize_to_range_constant(self):
        """Test range normalization on constant signal."""
        wavetable = np.array([2.0, 2.0, 2.0])
        result = normalize_to_range(wavetable)
        np.testing.assert_allclose(result, np.zeros(3), atol=1e-15)

    @given(st.lists(st.floats(-100, 100), min_size=2, max_size=1024))
    def test_normalize_to_range_hypothesis(self, values):
        """Hypothesis test for normalize_to_range."""
        assume(all(np.isfinite(v) for v in values))
        assume(max(values) - min(values) > 1e-12)  # Non-constant signal

        wavetable = np.array(values)
        result = normalize_to_range(wavetable)

        # Should have zero mean
        assert (
            abs(np.mean(result)) < 1e-12
        )  # Tightened from 1e-10, balancing precision with robustness
        # Should use target range
        range_used = np.max(result) - np.min(result)
        target_range = 0.999 - (-0.999)
        assert abs(range_used - target_range) < 1e-15
        # Should preserve shape
        assert result.shape == wavetable.shape


class TestAlignToZeroCrossing:
    """Test zero-crossing alignment functionality."""

    def test_align_to_zero_crossing_basic(self):
        """Test basic zero-crossing alignment."""
        # Create a sine wave starting at non-zero
        t = np.linspace(np.pi / 4, np.pi / 4 + 2 * np.pi, 100, endpoint=False)
        wavetable = np.sin(t)
        result = align_to_zero_crossing(wavetable)

        # Should start closer to zero
        assert abs(result[0]) < abs(wavetable[0])

    def test_align_to_zero_crossing_already_aligned(self):
        """Test alignment when already at zero crossing."""
        # Create signal starting at zero
        wavetable = np.array([0.0, 1.0, 0.0, -1.0])
        result = align_to_zero_crossing(wavetable)

        # Should start at or very close to zero
        assert abs(result[0]) < 1e-10

    def test_align_to_zero_crossing_no_crossing(self):
        """Test alignment when no zero crossing exists."""
        wavetable = np.array([1.0, 2.0, 3.0, 4.0])
        result = align_to_zero_crossing(wavetable)

        # Should find the sample closest to zero
        assert result[0] == 1.0  # The minimum value

    def test_align_to_zero_crossing_short_signal(self):
        """Test alignment on very short signals."""
        wavetable = np.array([1.0])
        result = align_to_zero_crossing(wavetable)
        np.testing.assert_array_equal(result, wavetable)

    @given(st.lists(st.floats(-10, 10), min_size=4, max_size=512))
    def test_align_to_zero_crossing_hypothesis(self, values):
        """Hypothesis test for align_to_zero_crossing."""
        assume(all(np.isfinite(v) for v in values))

        wavetable = np.array(values)
        result = align_to_zero_crossing(wavetable)

        # Should preserve shape and data (just rotated)
        assert result.shape == wavetable.shape
        np.testing.assert_allclose(np.sort(result), np.sort(wavetable), atol=1e-15)


class TestEnsureConsistentPhaseAlignment:
    """Test mipmap phase alignment functionality."""

    def test_ensure_consistent_phase_alignment_basic(self):
        """Test basic phase alignment across mipmap levels."""
        # Create multiple sine waves with different phases
        t = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        level1 = np.sin(t)
        level2 = np.sin(t + np.pi / 4)  # Phase shifted
        level3 = np.sin(t + np.pi / 2)  # More phase shifted

        mipmap_chain = [level1, level2, level3]
        result = ensure_consistent_phase_alignment(mipmap_chain)

        # All levels should be aligned (start closer to zero)
        for level in result:
            assert len(level) == 64
            # Each should be aligned to zero crossing
            assert abs(level[0]) <= abs(np.sin(np.pi / 4))  # Better than worst case

    def test_ensure_consistent_phase_alignment_empty(self):
        """Test phase alignment with empty input."""
        result = ensure_consistent_phase_alignment([])
        assert result == []

    def test_ensure_consistent_phase_alignment_single(self):
        """Test phase alignment with single level."""
        wavetable = np.array([1.0, 0.0, -1.0, 0.0])
        result = ensure_consistent_phase_alignment([wavetable])

        assert len(result) == 1
        assert len(result[0]) == 4

    @given(st.lists(st.lists(st.floats(-5, 5), min_size=4, max_size=128), min_size=1, max_size=10))
    def test_ensure_consistent_phase_alignment_hypothesis(self, level_data):
        """Hypothesis test for phase alignment."""
        assume(all(all(np.isfinite(v) for v in level) for level in level_data))

        mipmap_chain = [np.array(level) for level in level_data]
        result = ensure_consistent_phase_alignment(mipmap_chain)

        # Should preserve count and shapes
        assert len(result) == len(mipmap_chain)
        for i, level in enumerate(result):
            assert level.shape == mipmap_chain[i].shape


class TestMinPhaseRealign:
    """Test minimum phase realignment functionality."""

    def test_min_phase_realign_basic(self):
        """Test basic minimum phase conversion."""
        # Create a simple signal
        wavetable = np.array([1.0, 0.5, -0.5, -1.0, 0.0, 0.5, 1.0, 0.5])
        result = min_phase_realign(wavetable)

        # Should be normalized
        assert abs(np.max(np.abs(result)) - 0.999) < 1e-10
        # Should preserve length
        assert result.shape == wavetable.shape

    def test_min_phase_realign_sine(self):
        """Test minimum phase conversion on sine wave."""
        t = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        wavetable = np.sin(t)
        result = min_phase_realign(wavetable)

        # Should be normalized and finite
        assert np.all(np.isfinite(result))
        assert abs(np.max(np.abs(result)) - 0.999) < 1e-10

    @given(st.lists(st.floats(-5, 5), min_size=8, max_size=256))
    def test_min_phase_realign_hypothesis(self, values):
        """Hypothesis test for minimum phase realignment."""
        assume(all(np.isfinite(v) for v in values))
        assume(max(abs(v) for v in values) > 1e-6)  # Non-zero signal

        wavetable = np.array(values)
        result = min_phase_realign(wavetable)

        # Should be normalized and finite
        assert np.all(np.isfinite(result))
        # The true peak (including inter-sample peaks) should not exceed 0.999
        from src.wtgen.dsp.process import _estimate_inter_sample_peak
        true_peak = _estimate_inter_sample_peak(result)
        assert true_peak <= 0.999 + 1e-7
        assert result.shape == wavetable.shape


class TestIntegrationProperties:
    """Test integration properties across process functions."""

    def test_dc_remove_normalize_chain(self):
        """Test chaining DC removal and normalization."""
        wavetable = np.array([2.0, 4.0, 6.0, 8.0])  # Has DC offset

        # Chain operations
        step1 = dc_remove(wavetable)
        step2 = normalize(step1)

        # Should have zero mean and the true peak should not exceed 0.999
        assert abs(np.mean(step1)) < 1e-15
        from src.wtgen.dsp.process import _estimate_inter_sample_peak
        true_peak = _estimate_inter_sample_peak(step2)
        assert true_peak <= 0.999 + 1e-12
