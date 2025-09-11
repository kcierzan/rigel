"""
Unit tests for EQ functionality.

Tests the parametric EQ implementation for wavetables and mipmap chains,
ensuring proper frequency response, RMS preservation, and phase alignment.
"""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from wtgen.dsp.eq import (
    analyze_eq_response,
    analyze_tilt_eq_response,
    apply_parametric_eq_fft,
    apply_tilt_eq_fft,
    create_eq_band,
    create_high_tilt_eq,
    create_low_tilt_eq,
    parse_eq_string,
)
from wtgen.dsp.mipmap import Mipmap
from wtgen.dsp.process import align_to_zero_crossing
from wtgen.dsp.waves import WaveGenerator
from wtgen.utils import EPSILON


class TestCreateEQBand:
    """Test EQ band creation and validation."""

    def test_create_eq_band_valid(self):
        """Test creating valid EQ bands."""
        band = create_eq_band(1000.0, 3.0, 2.0)  # 1000 Hz
        assert band["frequency_hz"] == 1000.0
        assert band["gain_db"] == 3.0
        assert band["q_factor"] == 2.0
        # Check that normalized frequency is calculated correctly (1000/22050 ≈ 0.045)
        expected_norm = 1000.0 / 22050.0  # Default sample rate is 44100, Nyquist = 22050
        assert abs(band["frequency"] - expected_norm) < 1e-6

    def test_create_eq_band_default_q(self):
        """Test EQ band with default Q factor."""
        band = create_eq_band(5000.0, -6.0)  # 5000 Hz
        assert band["frequency_hz"] == 5000.0
        assert band["gain_db"] == -6.0
        assert band["q_factor"] == 1.0

    def test_create_eq_band_invalid_frequency(self):
        """Test that invalid frequencies raise errors."""
        with pytest.raises(ValueError, match="Frequency must be between 0.0 and 22050.0 Hz"):
            create_eq_band(25000.0, 3.0)  # Above Nyquist

        with pytest.raises(ValueError, match="Frequency must be between 0.0 and 22050.0 Hz"):
            create_eq_band(0.0, 3.0)  # DC

        with pytest.raises(ValueError, match="Frequency must be between 0.0 and 22050.0 Hz"):
            create_eq_band(-100.0, 3.0)  # Negative frequency

    @given(
        frequency_hz=st.floats(min_value=20.0, max_value=22000.0),
        gain_db=st.floats(min_value=-24.0, max_value=24.0),
        q_factor=st.floats(min_value=0.1, max_value=10.0),
    )
    def test_create_eq_band_hypothesis(self, frequency_hz, gain_db, q_factor):
        """Property test for EQ band creation."""
        band = create_eq_band(frequency_hz, gain_db, q_factor)
        assert band["frequency_hz"] == frequency_hz
        assert band["gain_db"] == gain_db
        assert band["q_factor"] == q_factor
        # Check that normalized frequency is in valid range
        assert 0.0 < band["frequency"] < 1.0


class TestParseEQString:
    """Test EQ string parsing functionality."""

    def test_parse_empty_string(self):
        """Test parsing empty EQ string."""
        result = parse_eq_string("")
        assert result == []

        result = parse_eq_string("   ")
        assert result == []

    def test_parse_single_band_with_q(self):
        """Test parsing single EQ band with Q factor."""
        result = parse_eq_string("1000:3.0:2.0")
        assert len(result) == 1
        assert result[0]["frequency_hz"] == 1000.0
        assert result[0]["gain_db"] == 3.0
        assert result[0]["q_factor"] == 2.0

    def test_parse_single_band_default_q(self):
        """Test parsing single EQ band with default Q."""
        result = parse_eq_string("5000:-6.0")
        assert len(result) == 1
        assert result[0]["frequency_hz"] == 5000.0
        assert result[0]["gain_db"] == -6.0
        assert result[0]["q_factor"] == 1.0

    def test_parse_multiple_bands(self):
        """Test parsing multiple EQ bands."""
        result = parse_eq_string("1000:3.0:2.0,5000:-6.0:1.5,10000:2.0")
        assert len(result) == 3

        assert result[0]["frequency_hz"] == 1000.0
        assert result[0]["gain_db"] == 3.0
        assert result[0]["q_factor"] == 2.0

        assert result[1]["frequency_hz"] == 5000.0
        assert result[1]["gain_db"] == -6.0
        assert result[1]["q_factor"] == 1.5

        assert result[2]["frequency_hz"] == 10000.0
        assert result[2]["gain_db"] == 2.0
        assert result[2]["q_factor"] == 1.0

    def test_parse_invalid_format(self):
        """Test that invalid formats raise errors."""
        with pytest.raises(ValueError, match="Invalid EQ band format"):
            parse_eq_string("1000")

        with pytest.raises(ValueError, match="Invalid EQ band format"):
            parse_eq_string("1000:3.0:2.0:extra")

        with pytest.raises(ValueError, match="Error parsing EQ string"):
            parse_eq_string("invalid:format")

        with pytest.raises(ValueError, match="Error parsing EQ string"):
            parse_eq_string("1000:not_a_number")


class TestApplyParametricEQFFT:
    """Test the core FFT-based EQ application."""

    def test_apply_eq_empty_wavetable(self):
        """Test EQ on empty wavetable."""
        empty = np.array([])
        eq_bands = [create_eq_band(1000.0, 3.0)]
        result = apply_parametric_eq_fft(empty, eq_bands)
        assert len(result) == 0

    def test_apply_eq_no_bands(self):
        """Test EQ with no bands returns original."""
        wavetable = np.sin(2 * np.pi * np.linspace(0, 1, 256, endpoint=False))
        result = apply_parametric_eq_fft(wavetable, [])
        np.testing.assert_array_almost_equal(result, wavetable)

    def test_apply_eq_zero_gain(self):
        """Test EQ with zero gain has minimal effect."""
        wavetable = np.sin(2 * np.pi * np.linspace(0, 1, 256, endpoint=False))
        eq_bands = [create_eq_band(1000.0, 0.0)]
        result = apply_parametric_eq_fft(wavetable, eq_bands)

        # Should be very close to original
        np.testing.assert_array_almost_equal(result, wavetable, decimal=5)

    def test_rms_preservation(self):
        """Test that RMS is preserved when requested."""
        wavetable = np.sin(2 * np.pi * np.linspace(0, 1, 1024, endpoint=False))
        original_rms = np.sqrt(np.mean(wavetable**2))

        # Apply significant EQ
        eq_bands = [create_eq_band(1000.0, 6.0, 2.0)]
        result = apply_parametric_eq_fft(wavetable, eq_bands, preserve_rms=True)
        final_rms = np.sqrt(np.mean(result**2))

        # RMS should be preserved within tolerance
        assert abs(final_rms - original_rms) < 0.01

    def test_rms_not_preserved_when_disabled(self):
        """Test that RMS changes when preservation is disabled."""
        wavetable = np.sin(2 * np.pi * np.linspace(0, 1, 1024, endpoint=False))
        original_rms = np.sqrt(np.mean(wavetable**2))

        # Apply significant boost at the fundamental frequency
        eq_bands = [create_eq_band(50.0, 18.0, 1.0)]  # Boost at low frequency with wider Q
        result = apply_parametric_eq_fft(wavetable, eq_bands, preserve_rms=False)
        final_rms = np.sqrt(np.mean(result**2))

        # RMS should have changed significantly
        assert abs(final_rms - original_rms) > 0.05  # Reduced threshold

    def test_phase_preservation(self):
        """Test that phase alignment is preserved when requested."""
        # Create a wavetable that starts at zero crossing
        wavetable = np.sin(2 * np.pi * np.linspace(0, 1, 256, endpoint=False))
        wavetable = align_to_zero_crossing(wavetable)

        # Apply EQ
        eq_bands = [create_eq_band(2000.0, 3.0)]
        result = apply_parametric_eq_fft(wavetable, eq_bands, preserve_phase=True)

        # Should still start near zero crossing
        assert abs(result[0]) < 0.1

    def test_boost_increases_energy(self):
        """Test that boost increases energy in target frequency region."""
        # Create a simple sine wave
        wavetable = np.sin(2 * np.pi * np.linspace(0, 1, 1024, endpoint=False))

        # Boost at low frequency
        eq_bands = [create_eq_band(100.0, 6.0, 1.0)]
        result = apply_parametric_eq_fft(wavetable, eq_bands, preserve_rms=False)

        # Result should have higher RMS (more energy)
        original_rms = np.sqrt(np.mean(wavetable**2))
        boosted_rms = np.sqrt(np.mean(result**2))
        assert boosted_rms > original_rms

    def test_cut_decreases_energy(self):
        """Test that cut decreases energy in target frequency region."""
        # Create a complex waveform with multiple harmonics
        t = np.linspace(0, 1, 1024, endpoint=False)
        wavetable = np.sin(2 * np.pi * t) + 0.5 * np.sin(4 * np.pi * t)

        # Cut higher frequency content
        eq_bands = [create_eq_band(200.0, -12.0, 2.0)]
        result = apply_parametric_eq_fft(wavetable, eq_bands, preserve_rms=False)

        # Result should have lower RMS
        original_rms = np.sqrt(np.mean(wavetable**2))
        cut_rms = np.sqrt(np.mean(result**2))
        assert cut_rms < original_rms

    @given(
        size=st.integers(min_value=64, max_value=2048),
        frequency_hz=st.floats(min_value=50.0, max_value=20000.0),
        gain_db=st.floats(min_value=-12.0, max_value=12.0),
        q_factor=st.floats(min_value=0.5, max_value=5.0),
    )
    def test_apply_eq_hypothesis(self, size, frequency_hz, gain_db, q_factor):
        """Property test for EQ application."""
        # Generate test wavetable
        t = np.linspace(0, 1, size, endpoint=False)
        wavetable = np.sin(2 * np.pi * t) + 0.3 * np.sin(6 * np.pi * t)

        eq_bands = [create_eq_band(frequency_hz, gain_db, q_factor)]
        result = apply_parametric_eq_fft(wavetable, eq_bands)

        # Basic sanity checks
        assert len(result) == len(wavetable)
        assert np.isfinite(result).all()
        assert not np.isnan(result).any()

        # RMS should be preserved (default behavior)
        original_rms = np.sqrt(np.mean(wavetable**2))
        result_rms = np.sqrt(np.mean(result**2))
        if original_rms > EPSILON:
            assert abs(result_rms - original_rms) < 0.01


class TestAnalyzeEQResponse:
    """Test EQ response analysis functionality."""

    def test_analyze_no_bands(self):
        """Test analysis with no EQ bands."""
        frequencies, magnitudes = analyze_eq_response([])

        assert len(frequencies) == 512  # Default num_points
        assert len(magnitudes) == 512
        np.testing.assert_array_almost_equal(magnitudes, np.zeros_like(magnitudes))

    def test_analyze_single_band(self):
        """Test analysis with single EQ band."""
        eq_bands = [create_eq_band(5000.0, 6.0, 2.0)]
        frequencies, magnitudes = analyze_eq_response(eq_bands, num_points=256)

        assert len(frequencies) == 256
        assert len(magnitudes) == 256

        # Should have peak around 5000 Hz
        peak_idx = np.argmax(magnitudes)
        peak_freq = frequencies[peak_idx]
        assert abs(peak_freq - 5000.0) < 1000.0  # Within 1kHz tolerance

        # Peak should be positive (boost)
        assert magnitudes[peak_idx] > 0

    def test_analyze_multiple_bands(self):
        """Test analysis with multiple EQ bands."""
        eq_bands = [
            create_eq_band(2000.0, 3.0),
            create_eq_band(8000.0, -4.0),
            create_eq_band(15000.0, 2.0),
        ]

        frequencies, magnitudes = analyze_eq_response(eq_bands)

        # Should show effects of all bands
        # Find approximate frequency indices
        freq_2k_idx = np.argmin(np.abs(frequencies - 2000.0))
        freq_8k_idx = np.argmin(np.abs(frequencies - 8000.0))
        freq_15k_idx = np.argmin(np.abs(frequencies - 15000.0))

        low_region = magnitudes[freq_2k_idx]
        mid_region = magnitudes[freq_8k_idx]
        high_region = magnitudes[freq_15k_idx]

        assert low_region > 0  # Boost
        assert mid_region < 0  # Cut
        assert high_region > 0  # Boost

    def test_analyze_custom_points(self):
        """Test analysis with custom number of points."""
        eq_bands = [create_eq_band(6000.0, 2.0)]
        frequencies, magnitudes = analyze_eq_response(eq_bands, num_points=128)

        assert len(frequencies) == 128
        assert len(magnitudes) == 128


class TestEQIntegration:
    """Integration tests combining EQ with other DSP functions."""

    def test_eq_with_mipmap_generation(self):
        """Test EQ integration with mipmap generation."""
        # Generate base waveform
        _, base_wave = WaveGenerator().sawtooth(1.0)

        # Apply EQ first
        eq_bands = [create_eq_band(2000.0, 4.0, 1.5)]
        eq_wave = apply_parametric_eq_fft(base_wave, eq_bands)

        # Build mipmaps from EQ'd waveform
        mipmaps = Mipmap(eq_wave, num_octaves=2, decimate=False).generate()

        # Should generate valid mipmaps
        assert len(mipmaps) == 3  # 0, 1, 2
        for level in mipmaps:
            assert np.isfinite(level).all()
            assert not np.isnan(level).any()

    def test_eq_preserves_zero_crossing_alignment(self):
        """Test that EQ works well with zero-crossing alignment."""
        # Create aligned waveform
        _, base_wave = WaveGenerator().sawtooth(1.0)
        aligned_wave = align_to_zero_crossing(base_wave)

        # Apply EQ with phase preservation
        eq_bands = [create_eq_band(1500.0, -3.0)]
        eq_wave = apply_parametric_eq_fft(aligned_wave, eq_bands, preserve_phase=True)

        # Should still be well-aligned
        assert abs(eq_wave[0]) < 0.1

    def test_eq_with_extreme_settings(self):
        """Test EQ with extreme but valid settings."""
        wavetable = np.sin(2 * np.pi * np.linspace(0, 1, 256, endpoint=False))

        # Extreme boost and cut
        eq_bands = [
            create_eq_band(1000.0, 20.0, 5.0),  # High boost, high Q
            create_eq_band(8000.0, -20.0, 0.5),  # High cut, low Q
        ]

        result = apply_parametric_eq_fft(wavetable, eq_bands)

        # Should not produce invalid values
        assert np.isfinite(result).all()
        assert not np.isnan(result).any()

        # Should be clipped to reasonable range
        assert np.max(np.abs(result)) <= 1.1  # Some tolerance for numerical precision


class TestCreateTiltEQ:
    """Test tilt EQ filter creation."""

    def test_create_high_tilt_eq_valid(self):
        """Test creating valid high tilt EQ."""
        response = create_high_tilt_eq(0.5, 6.0)
        assert len(response) == 513  # Half of 1024 + DC
        assert np.all(np.isfinite(response))
        assert response[0] == 1.0  # Unity at DC

        # Should have higher gain at higher frequencies
        assert response[-1] > response[len(response) // 2]

    def test_create_low_tilt_eq_valid(self):
        """Test creating valid low tilt EQ."""
        response = create_low_tilt_eq(0.5, 4.0)
        assert len(response) == 513  # Half of 1024 + DC
        assert np.all(np.isfinite(response))
        assert response[0] == 1.0  # Unity at DC

        # Should have higher gain at lower frequencies (after DC)
        mid_point = len(response) // 2
        # Compare early frequencies to later frequencies
        assert np.mean(response[1 : mid_point // 2]) > np.mean(response[mid_point:])

    def test_create_tilt_eq_invalid_ratio(self):
        """Test that invalid frequency ratios raise errors."""
        with pytest.raises(ValueError, match="start_freq_ratio must be between 0.0 and 1.0"):
            create_high_tilt_eq(1.5, 6.0)

        with pytest.raises(ValueError, match="start_freq_ratio must be between 0.0 and 1.0"):
            create_low_tilt_eq(-0.1, 4.0)

    def test_create_tilt_eq_zero_gain(self):
        """Test tilt EQ with zero gain."""
        response_high = create_high_tilt_eq(0.3, 0.0)
        response_low = create_low_tilt_eq(0.7, 0.0)

        np.testing.assert_array_almost_equal(response_high, np.ones_like(response_high))
        np.testing.assert_array_almost_equal(response_low, np.ones_like(response_low))

    def test_create_tilt_eq_negative_gain(self):
        """Test tilt EQ with negative gain (cut)."""
        response_high = create_high_tilt_eq(0.4, -3.0)
        response_low = create_low_tilt_eq(0.6, -2.0)

        # All gains should be <= 1.0 (cuts only)
        assert np.all(response_high <= 1.0)
        assert np.all(response_low <= 1.0)

    @given(
        start_freq_ratio=st.floats(min_value=0.01, max_value=0.99),
        tilt_db=st.floats(min_value=-12.0, max_value=12.0),
    )
    def test_create_tilt_eq_hypothesis(self, start_freq_ratio, tilt_db):
        """Property test for tilt EQ creation."""
        response_high = create_high_tilt_eq(start_freq_ratio, tilt_db)
        response_low = create_low_tilt_eq(start_freq_ratio, tilt_db)

        # Basic sanity checks
        assert len(response_high) == 513
        assert len(response_low) == 513
        assert np.all(np.isfinite(response_high))
        assert np.all(np.isfinite(response_low))
        assert np.all(response_high > 0)
        assert np.all(response_low > 0)


class TestApplyTiltEQFFT:
    """Test the core tilt EQ application."""

    def test_apply_tilt_eq_empty_wavetable(self):
        """Test tilt EQ on empty wavetable."""
        empty = np.array([])
        result_high = apply_tilt_eq_fft(empty, 0.5, 3.0, "high")
        result_low = apply_tilt_eq_fft(empty, 0.5, 3.0, "low")

        assert len(result_high) == 0
        assert len(result_low) == 0

    def test_apply_tilt_eq_zero_gain(self):
        """Test tilt EQ with zero gain has minimal effect."""
        wavetable = np.sin(2 * np.pi * np.linspace(0, 1, 256, endpoint=False))
        result_high = apply_tilt_eq_fft(wavetable, 0.3, 0.0, "high")
        result_low = apply_tilt_eq_fft(wavetable, 0.7, 0.0, "low")

        np.testing.assert_array_almost_equal(result_high, wavetable, decimal=5)
        np.testing.assert_array_almost_equal(result_low, wavetable, decimal=5)

    def test_apply_tilt_eq_invalid_type(self):
        """Test that invalid tilt types raise errors."""
        wavetable = np.sin(2 * np.pi * np.linspace(0, 1, 256, endpoint=False))

        with pytest.raises(ValueError, match="tilt_type must be 'high' or 'low'"):
            apply_tilt_eq_fft(wavetable, 0.5, 3.0, "invalid")

    def test_high_tilt_boosts_highs(self):
        """Test that high tilt actually boosts high frequencies."""
        # Create a complex waveform with multiple harmonics
        t = np.linspace(0, 1, 1024, endpoint=False)
        wavetable = (
            np.sin(2 * np.pi * t) + 0.3 * np.sin(6 * np.pi * t) + 0.1 * np.sin(10 * np.pi * t)
        )

        # Apply high tilt boost
        result = apply_tilt_eq_fft(wavetable, 0.3, 6.0, "high", preserve_rms=False)

        # Check in frequency domain that high frequencies are boosted
        original_spectrum = np.abs(np.fft.fft(wavetable))
        boosted_spectrum = np.abs(np.fft.fft(result))

        # High frequency region should be boosted
        high_freq_start = len(original_spectrum) // 4
        high_freq_end = len(original_spectrum) // 2

        original_high_energy = np.mean(original_spectrum[high_freq_start:high_freq_end])
        boosted_high_energy = np.mean(boosted_spectrum[high_freq_start:high_freq_end])

        assert boosted_high_energy > original_high_energy

    def test_low_tilt_boosts_lows(self):
        """Test that low tilt actually boosts low frequencies."""
        # Create a complex waveform
        t = np.linspace(0, 1, 1024, endpoint=False)
        wavetable = (
            np.sin(2 * np.pi * t) + 0.5 * np.sin(4 * np.pi * t) + 0.3 * np.sin(8 * np.pi * t)
        )

        # Apply low tilt boost
        result = apply_tilt_eq_fft(wavetable, 0.6, 4.0, "low", preserve_rms=False)

        # Check in frequency domain that low frequencies are boosted
        original_spectrum = np.abs(np.fft.fft(wavetable))
        boosted_spectrum = np.abs(np.fft.fft(result))

        # Low frequency region should be boosted (skip DC)
        low_freq_end = len(original_spectrum) // 8

        original_low_energy = np.mean(original_spectrum[1:low_freq_end])
        boosted_low_energy = np.mean(boosted_spectrum[1:low_freq_end])

        assert boosted_low_energy > original_low_energy

    def test_tilt_eq_rms_preservation(self):
        """Test that RMS is preserved when requested."""
        wavetable = np.sin(2 * np.pi * np.linspace(0, 1, 512, endpoint=False))
        original_rms = np.sqrt(np.mean(wavetable**2))

        # Apply significant tilt
        result_high = apply_tilt_eq_fft(wavetable, 0.2, 8.0, "high", preserve_rms=True)
        result_low = apply_tilt_eq_fft(wavetable, 0.8, 6.0, "low", preserve_rms=True)

        result_high_rms = np.sqrt(np.mean(result_high**2))
        result_low_rms = np.sqrt(np.mean(result_low**2))

        assert abs(result_high_rms - original_rms) < 0.01
        assert abs(result_low_rms - original_rms) < 0.01

    def test_tilt_eq_phase_preservation(self):
        """Test that phase alignment is preserved when requested."""
        wavetable = np.sin(2 * np.pi * np.linspace(0, 1, 256, endpoint=False))
        wavetable = align_to_zero_crossing(wavetable)

        # Apply tilt EQ
        result_high = apply_tilt_eq_fft(wavetable, 0.4, 4.0, "high", preserve_phase=True)
        result_low = apply_tilt_eq_fft(wavetable, 0.6, 3.0, "low", preserve_phase=True)

        # Should still start near zero crossing
        assert abs(result_high[0]) < 0.1
        assert abs(result_low[0]) < 0.1

    @given(
        size=st.integers(min_value=64, max_value=1024),
        start_freq_ratio=st.floats(min_value=0.1, max_value=0.9),
        tilt_db=st.floats(min_value=-12.0, max_value=12.0),
        tilt_type=st.sampled_from(["high", "low"]),
    )
    def test_apply_tilt_eq_hypothesis(self, size, start_freq_ratio, tilt_db, tilt_type):
        """Property test for tilt EQ application."""
        # Generate test wavetable
        t = np.linspace(0, 1, size, endpoint=False)
        wavetable = np.sin(2 * np.pi * t) + 0.2 * np.sin(6 * np.pi * t)

        result = apply_tilt_eq_fft(wavetable, start_freq_ratio, tilt_db, tilt_type)

        # Basic sanity checks
        assert len(result) == len(wavetable)
        assert np.isfinite(result).all()
        assert not np.isnan(result).any()

        # RMS should be preserved (default behavior)
        original_rms = np.sqrt(np.mean(wavetable**2))
        result_rms = np.sqrt(np.mean(result**2))
        if original_rms > EPSILON:
            assert abs(result_rms - original_rms) < 0.01


class TestAnalyzeTiltEQResponse:
    """Test tilt EQ response analysis functionality."""

    def test_analyze_tilt_eq_response_high(self):
        """Test analysis of high tilt EQ response."""
        frequencies, magnitudes = analyze_tilt_eq_response(0.5, 6.0, "high", num_points=256)

        assert len(frequencies) == 256
        assert len(magnitudes) == 256

        # Should have increasing gain towards higher frequencies
        mid_point = len(magnitudes) // 2
        low_region = np.mean(magnitudes[: mid_point // 2])
        high_region = np.mean(magnitudes[mid_point:])

        assert high_region > low_region

    def test_analyze_tilt_eq_response_low(self):
        """Test analysis of low tilt EQ response."""
        frequencies, magnitudes = analyze_tilt_eq_response(0.6, 4.0, "low", num_points=256)

        assert len(frequencies) == 256
        assert len(magnitudes) == 256

        # Should have increasing gain towards lower frequencies
        # Find the start frequency index
        nyquist = 22050.0  # Default sample rate / 2
        start_freq_hz = 0.6 * nyquist
        start_idx = np.argmin(np.abs(frequencies - start_freq_hz))

        # Compare regions before and after the start frequency
        if start_idx > 10:  # Ensure we have enough points to compare
            low_region = np.mean(magnitudes[1 : start_idx // 2])  # Skip DC
            mid_region = np.mean(magnitudes[start_idx:])

            assert low_region > mid_region

    def test_analyze_tilt_eq_response_zero_gain(self):
        """Test analysis with zero gain."""
        frequencies, magnitudes = analyze_tilt_eq_response(0.4, 0.0, "high")

        # Should be flat at 0 dB
        np.testing.assert_array_almost_equal(magnitudes, np.zeros_like(magnitudes), decimal=5)

    def test_analyze_tilt_eq_response_custom_points(self):
        """Test analysis with custom number of points."""
        frequencies, magnitudes = analyze_tilt_eq_response(0.3, 3.0, "high", num_points=128)

        assert len(frequencies) == 128
        assert len(magnitudes) == 128


class TestTiltEQIntegration:
    """Integration tests for tilt EQ with other DSP functions."""

    def test_tilt_eq_with_mipmap_generation(self):
        """Test tilt EQ integration with mipmap generation."""
        # Generate base waveform
        _, base_wave = WaveGenerator().sawtooth(1.0)

        # Apply tilt EQ first
        eq_wave = apply_tilt_eq_fft(base_wave, 0.4, 4.0, "high")

        # Build mipmaps from tilted waveform
        mipmaps = Mipmap(eq_wave, num_octaves=2, decimate=False).generate()

        # Should generate valid mipmaps
        assert len(mipmaps) == 3  # 0, 1, 2
        for level in mipmaps:
            assert np.isfinite(level).all()
            assert not np.isnan(level).any()

    def test_tilt_eq_preserves_zero_crossing_alignment(self):
        """Test that tilt EQ works well with zero-crossing alignment."""
        # Create aligned waveform
        _, base_wave = WaveGenerator().sawtooth(1.0)
        aligned_wave = align_to_zero_crossing(base_wave)

        # Apply tilt EQ with phase preservation
        eq_wave_high = apply_tilt_eq_fft(aligned_wave, 0.3, 3.0, "high", preserve_phase=True)
        eq_wave_low = apply_tilt_eq_fft(aligned_wave, 0.7, -2.0, "low", preserve_phase=True)

        # Should still be well-aligned
        assert abs(eq_wave_high[0]) < 0.1
        assert abs(eq_wave_low[0]) < 0.1

    def test_tilt_eq_with_extreme_settings(self):
        """Test tilt EQ with extreme but valid settings."""
        wavetable = np.sin(2 * np.pi * np.linspace(0, 1, 256, endpoint=False))

        # Extreme settings
        result_high = apply_tilt_eq_fft(wavetable, 0.1, 20.0, "high")
        result_low = apply_tilt_eq_fft(wavetable, 0.9, -20.0, "low")

        # Should not produce invalid values
        assert np.isfinite(result_high).all()
        assert not np.isnan(result_high).any()
        assert np.isfinite(result_low).all()
        assert not np.isnan(result_low).any()

        # Should be in reasonable range
        assert np.max(np.abs(result_high)) <= 1.1
        assert np.max(np.abs(result_low)) <= 1.1

    def test_combined_parametric_and_tilt_eq(self):
        """Test combining parametric EQ with tilt EQ."""
        wavetable = np.sin(2 * np.pi * np.linspace(0, 1, 512, endpoint=False))

        # Apply parametric EQ first
        eq_bands = [create_eq_band(2000.0, 3.0, 2.0)]
        parametric_result = apply_parametric_eq_fft(wavetable, eq_bands)

        # Then apply tilt EQ
        combined_result = apply_tilt_eq_fft(parametric_result, 0.6, 2.0, "high")

        # Should process without error
        assert np.isfinite(combined_result).all()
        assert not np.isnan(combined_result).any()
        assert len(combined_result) == len(wavetable)
