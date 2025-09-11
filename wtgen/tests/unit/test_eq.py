"""
Unit tests for EQ functionality.

Tests the parametric EQ implementation for wavetables and mipmap chains,
ensuring proper frequency response, RMS preservation, and phase alignment.
"""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from wtgen.dsp.eq import Equalizer
from wtgen.dsp.mipmap import Mipmap
from wtgen.dsp.process import align_to_zero_crossing
from wtgen.dsp.waves import WaveGenerator
from wtgen.utils import EPSILON


class TestEqualizerConstruction:
    """Test Equalizer class construction and validation."""

    def test_create_empty_equalizer(self):
        """Test creating equalizer with no settings."""
        eq = Equalizer()
        assert eq._eq_settings is None
        assert eq._high_tilt_settings is None
        assert eq._low_tilt_settings is None
        assert eq._sample_rate == 44100.0

    def test_create_equalizer_with_parametric_eq(self):
        """Test creating equalizer with parametric EQ settings."""
        eq = Equalizer(eq_settings="1000:3.0:2.0,5000:-6.0")
        assert eq._eq_settings == "1000:3.0:2.0,5000:-6.0"
        assert eq._sample_rate == 44100.0

    def test_create_equalizer_with_tilt_settings(self):
        """Test creating equalizer with tilt settings."""
        eq = Equalizer(
            high_tilt_settings="0.5:6.0", low_tilt_settings="0.3:4.0", sample_rate=48000.0
        )
        assert eq._high_tilt_settings == "0.5:6.0"
        assert eq._low_tilt_settings == "0.3:4.0"
        assert eq._sample_rate == 48000.0

    def test_create_equalizer_all_settings(self):
        """Test creating equalizer with all settings."""
        eq = Equalizer(
            eq_settings="2000:2.0", high_tilt_settings="0.6:3.0", low_tilt_settings="0.4:-2.0"
        )
        assert eq._eq_settings == "2000:2.0"
        assert eq._high_tilt_settings == "0.6:3.0"
        assert eq._low_tilt_settings == "0.4:-2.0"


class TestEqualizerCreateEQBand:
    """Test EQ band creation within Equalizer class."""

    def test_create_eq_band_valid(self):
        """Test creating valid EQ bands."""
        eq = Equalizer()
        band = eq._create_eq_band(1000.0, 3.0, 2.0)
        assert band["frequency_hz"] == 1000.0
        assert band["gain_db"] == 3.0
        assert band["q_factor"] == 2.0
        # Check that normalized frequency is calculated correctly (1000/22050 ≈ 0.045)
        expected_norm = 1000.0 / 22050.0  # Default sample rate is 44100, Nyquist = 22050
        assert abs(band["frequency"] - expected_norm) < 1e-6

    def test_create_eq_band_default_q(self):
        """Test EQ band with default Q factor."""
        eq = Equalizer()
        band = eq._create_eq_band(5000.0, -6.0)
        assert band["frequency_hz"] == 5000.0
        assert band["gain_db"] == -6.0
        assert band["q_factor"] == 1.0

    def test_create_eq_band_invalid_frequency(self):
        """Test that invalid frequencies raise errors."""
        eq = Equalizer()
        with pytest.raises(ValueError, match="Frequency must be between 0.0 and 22050.0 Hz"):
            eq._create_eq_band(25000.0, 3.0)  # Above Nyquist

        with pytest.raises(ValueError, match="Frequency must be between 0.0 and 22050.0 Hz"):
            eq._create_eq_band(0.0, 3.0)  # DC

        with pytest.raises(ValueError, match="Frequency must be between 0.0 and 22050.0 Hz"):
            eq._create_eq_band(-100.0, 3.0)  # Negative frequency

    @given(
        frequency_hz=st.floats(min_value=20.0, max_value=22000.0),
        gain_db=st.floats(min_value=-24.0, max_value=24.0),
        q_factor=st.floats(min_value=0.1, max_value=10.0),
    )
    def test_create_eq_band_hypothesis(self, frequency_hz, gain_db, q_factor):
        """Property test for EQ band creation."""
        eq = Equalizer()
        band = eq._create_eq_band(frequency_hz, gain_db, q_factor)
        assert band["frequency_hz"] == frequency_hz
        assert band["gain_db"] == gain_db
        assert band["q_factor"] == q_factor
        # Check that normalized frequency is in valid range
        assert 0.0 < band["frequency"] < 1.0


class TestEqualizerParseEQSettings:
    """Test EQ string parsing functionality."""

    def test_parse_empty_string(self):
        """Test parsing empty EQ string."""
        eq = Equalizer()
        result = eq._parse_eq_settings("")
        assert result == []

        result = eq._parse_eq_settings("   ")
        assert result == []

    def test_parse_single_band_with_q(self):
        """Test parsing single EQ band with Q factor."""
        eq = Equalizer()
        result = eq._parse_eq_settings("1000:3.0:2.0")
        assert len(result) == 1
        assert result[0]["frequency_hz"] == 1000.0
        assert result[0]["gain_db"] == 3.0
        assert result[0]["q_factor"] == 2.0

    def test_parse_single_band_default_q(self):
        """Test parsing single EQ band with default Q."""
        eq = Equalizer()
        result = eq._parse_eq_settings("5000:-6.0")
        assert len(result) == 1
        assert result[0]["frequency_hz"] == 5000.0
        assert result[0]["gain_db"] == -6.0
        assert result[0]["q_factor"] == 1.0

    def test_parse_multiple_bands(self):
        """Test parsing multiple EQ bands."""
        eq = Equalizer()
        result = eq._parse_eq_settings("1000:3.0:2.0,5000:-6.0:1.5,10000:2.0")
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
        eq = Equalizer()
        with pytest.raises(ValueError, match="Invalid EQ band format"):
            eq._parse_eq_settings("1000")

        with pytest.raises(ValueError, match="Invalid EQ band format"):
            eq._parse_eq_settings("1000:3.0:2.0:extra")

        with pytest.raises(ValueError, match="Error parsing EQ string"):
            eq._parse_eq_settings("invalid:format")

        with pytest.raises(ValueError, match="Error parsing EQ string"):
            eq._parse_eq_settings("1000:not_a_number")


class TestEqualizerApply:
    """Test the main apply method of Equalizer."""

    def test_apply_empty_wavetable(self):
        """Test EQ on empty wavetable."""
        empty = np.array([])
        eq = Equalizer(eq_settings="1000:3.0")
        result = eq.apply(empty)
        assert len(result) == 0

    def test_apply_no_settings(self):
        """Test EQ with no settings returns original."""
        wavetable = np.sin(2 * np.pi * np.linspace(0, 1, 256, endpoint=False))
        eq = Equalizer()
        result = eq.apply(wavetable)
        np.testing.assert_array_almost_equal(result, wavetable)

    def test_apply_parametric_eq_only(self):
        """Test apply with only parametric EQ settings."""
        # Create a richer signal with multiple harmonics that will be affected by EQ
        t = np.linspace(0, 1, 2048, endpoint=False)
        # Mix of different frequencies that will be affected by 1000 Hz EQ
        wavetable = (
            np.sin(2 * np.pi * 50 * t)  # Low frequency
            + 0.5 * np.sin(2 * np.pi * 1000 * t)  # Target frequency for EQ
            + 0.3 * np.sin(2 * np.pi * 5000 * t)  # High frequency
        )
        eq = Equalizer(eq_settings="1000:6.0:1.0", sample_rate=44100.0)
        result = eq.apply(wavetable)

        # Should be different from original due to boost at 1000 Hz
        assert not np.allclose(result, wavetable, rtol=1e-3)
        # Should have same length
        assert len(result) == len(wavetable)
        # Should be finite
        assert np.isfinite(result).all()

    def test_apply_tilt_eq_only(self):
        """Test apply with only tilt EQ settings."""
        # Create a richer signal with multiple harmonics across the spectrum
        t = np.linspace(0, 1, 2048, endpoint=False)
        wavetable = (
            np.sin(2 * np.pi * 100 * t)
            + 0.5 * np.sin(2 * np.pi * 1000 * t)
            + 0.3 * np.sin(2 * np.pi * 5000 * t)
            + 0.2 * np.sin(2 * np.pi * 10000 * t)
        )

        # Test high tilt - should affect higher frequencies
        eq_high = Equalizer(high_tilt_settings="0.3:6.0")
        result_high = eq_high.apply(wavetable)
        assert not np.allclose(result_high, wavetable, rtol=1e-3)
        assert len(result_high) == len(wavetable)
        assert np.isfinite(result_high).all()

        # Test low tilt - should affect lower frequencies
        eq_low = Equalizer(low_tilt_settings="0.7:4.0")
        result_low = eq_low.apply(wavetable)
        assert not np.allclose(result_low, wavetable, rtol=1e-3)
        assert len(result_low) == len(wavetable)
        assert np.isfinite(result_low).all()

    def test_apply_combined_settings(self):
        """Test apply with all types of settings combined."""
        # Create a rich signal with content at various frequencies
        t = np.linspace(0, 1, 2048, endpoint=False)
        wavetable = (
            np.sin(2 * np.pi * 100 * t)
            + 0.6 * np.sin(2 * np.pi * 2000 * t)  # Target for parametric EQ
            + 0.4 * np.sin(2 * np.pi * 8000 * t)  # Target for parametric EQ cut
            + 0.3 * np.sin(2 * np.pi * 12000 * t)  # High frequency content
        )
        eq = Equalizer(
            eq_settings="2000:6.0,8000:-6.0",  # More significant changes
            high_tilt_settings="0.4:4.0",
            low_tilt_settings="0.6:-2.0",
        )
        result = eq.apply(wavetable)

        # Should be significantly different from original
        assert not np.allclose(result, wavetable, rtol=1e-3)
        assert len(result) == len(wavetable)
        assert np.isfinite(result).all()
        assert not np.isnan(result).any()

    def test_apply_zero_gain(self):
        """Test EQ with zero gain has minimal effect."""
        wavetable = np.sin(2 * np.pi * np.linspace(0, 1, 256, endpoint=False))
        eq = Equalizer(eq_settings="1000:0.0")
        result = eq.apply(wavetable)

        # Should be very close to original
        np.testing.assert_array_almost_equal(result, wavetable, decimal=5)

    def test_rms_preservation(self):
        """Test that RMS is preserved by default."""
        wavetable = np.sin(2 * np.pi * np.linspace(0, 1, 1024, endpoint=False))
        original_rms = np.sqrt(np.mean(wavetable**2))

        # Apply significant EQ
        eq = Equalizer(eq_settings="1000:6.0:2.0")
        result = eq.apply(wavetable)
        final_rms = np.sqrt(np.mean(result**2))

        # RMS should be preserved within tolerance
        assert abs(final_rms - original_rms) < 0.01

    def test_phase_preservation(self):
        """Test that phase alignment is preserved by default."""
        # Create a wavetable that starts at zero crossing
        wavetable = np.sin(2 * np.pi * np.linspace(0, 1, 256, endpoint=False))
        wavetable = align_to_zero_crossing(wavetable)

        # Apply EQ
        eq = Equalizer(eq_settings="2000:3.0")
        result = eq.apply(wavetable)

        # Should still start near zero crossing
        assert abs(result[0]) < 0.1

    @given(
        size=st.integers(min_value=64, max_value=2048),
        frequency_hz=st.floats(min_value=50.0, max_value=20000.0),
        gain_db=st.floats(min_value=-12.0, max_value=12.0),
        q_factor=st.floats(min_value=0.5, max_value=5.0),
    )
    def test_apply_parametric_eq_hypothesis(self, size, frequency_hz, gain_db, q_factor):
        """Property test for parametric EQ application."""
        # Generate test wavetable
        t = np.linspace(0, 1, size, endpoint=False)
        wavetable = np.sin(2 * np.pi * t) + 0.3 * np.sin(6 * np.pi * t)

        eq_string = f"{frequency_hz}:{gain_db}:{q_factor}"
        eq = Equalizer(eq_settings=eq_string)
        result = eq.apply(wavetable)

        # Basic sanity checks
        assert len(result) == len(wavetable)
        assert np.isfinite(result).all()
        assert not np.isnan(result).any()

        # RMS should be preserved (default behavior)
        original_rms = np.sqrt(np.mean(wavetable**2))
        result_rms = np.sqrt(np.mean(result**2))
        if original_rms > EPSILON:
            assert abs(result_rms - original_rms) < 0.01


class TestEqualizerAnalyzeResponse:
    """Test EQ response analysis functionality."""

    def test_analyze_no_eq_settings(self):
        """Test analysis with no EQ settings."""
        eq = Equalizer()
        frequencies, magnitudes = eq.analyze_eq_response()

        assert len(frequencies) == 512  # Default num_points
        assert len(magnitudes) == 512
        np.testing.assert_array_almost_equal(magnitudes, np.zeros_like(magnitudes))

    def test_analyze_single_band(self):
        """Test analysis with single EQ band."""
        eq = Equalizer(eq_settings="5000:6.0:2.0")
        frequencies, magnitudes = eq.analyze_eq_response(num_points=256)

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
        eq = Equalizer(eq_settings="2000:3.0,8000:-4.0,15000:2.0")
        frequencies, magnitudes = eq.analyze_eq_response()

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
        eq = Equalizer(eq_settings="6000:2.0")
        frequencies, magnitudes = eq.analyze_eq_response(num_points=128)

        assert len(frequencies) == 128
        assert len(magnitudes) == 128


class TestEqualizerTiltResponse:
    """Test tilt EQ response analysis functionality."""

    def test_analyze_tilt_eq_response_high(self):
        """Test analysis of high tilt EQ response."""
        eq = Equalizer()
        frequencies, magnitudes = eq.analyze_tilt_eq_response(0.5, 6.0, "high", num_points=256)

        assert len(frequencies) == 256
        assert len(magnitudes) == 256

        # Should have increasing gain towards higher frequencies
        mid_point = len(magnitudes) // 2
        low_region = np.mean(magnitudes[: mid_point // 2])
        high_region = np.mean(magnitudes[mid_point:])

        assert high_region > low_region

    def test_analyze_tilt_eq_response_low(self):
        """Test analysis of low tilt EQ response."""
        eq = Equalizer()
        frequencies, magnitudes = eq.analyze_tilt_eq_response(0.6, 4.0, "low", num_points=256)

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
        eq = Equalizer()
        frequencies, magnitudes = eq.analyze_tilt_eq_response(0.4, 0.0, "high")

        # Should be flat at 0 dB
        np.testing.assert_array_almost_equal(magnitudes, np.zeros_like(magnitudes), decimal=5)

    def test_analyze_tilt_eq_response_custom_points(self):
        """Test analysis with custom number of points."""
        eq = Equalizer()
        frequencies, magnitudes = eq.analyze_tilt_eq_response(0.3, 3.0, "high", num_points=128)

        assert len(frequencies) == 128
        assert len(magnitudes) == 128


class TestEqualizerTiltSettings:
    """Test tilt setting parsing and application."""

    def test_parse_tilt_settings_valid(self):
        """Test parsing valid tilt settings."""
        eq = Equalizer()
        start_ratio, gain_db = eq._parse_tilt_settings("0.5:6.0")
        assert start_ratio == 0.5
        assert gain_db == 6.0

        start_ratio, gain_db = eq._parse_tilt_settings("0.3:-4.0")
        assert start_ratio == 0.3
        assert gain_db == -4.0

    def test_high_tilt_boosts_highs(self):
        """Test that high tilt actually boosts high frequencies."""
        # Create a complex waveform with multiple harmonics
        t = np.linspace(0, 1, 1024, endpoint=False)
        wavetable = (
            np.sin(2 * np.pi * t) + 0.3 * np.sin(6 * np.pi * t) + 0.1 * np.sin(10 * np.pi * t)
        )

        # Apply high tilt boost
        eq = Equalizer(high_tilt_settings="0.3:6.0")
        result = eq.apply(wavetable)

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
        # Create a complex waveform with specific frequency content
        t = np.linspace(0, 1, 2048, endpoint=False)
        wavetable = (
            0.8 * np.sin(2 * np.pi * 200 * t)  # Low frequency
            + 0.6 * np.sin(2 * np.pi * 1000 * t)  # Mid frequency
            + 0.4 * np.sin(2 * np.pi * 5000 * t)  # High frequency
            + 0.2 * np.sin(2 * np.pi * 10000 * t)  # Very high frequency
        )

        # Apply low tilt boost with more aggressive settings
        eq = Equalizer(low_tilt_settings="0.4:8.0")  # Start tilt earlier and boost more
        result = eq.apply(wavetable)

        # Check in frequency domain that low frequencies are boosted
        original_spectrum = np.abs(np.fft.fft(wavetable))
        boosted_spectrum = np.abs(np.fft.fft(result))

        # Compare low vs high frequency energy more directly
        # Low frequency region (below tilt start point)
        low_freq_end = len(original_spectrum) // 10  # More focused on very low frequencies
        # High frequency region (above tilt start point)
        high_freq_start = len(original_spectrum) // 4
        high_freq_end = len(original_spectrum) // 2

        original_low_energy = np.mean(original_spectrum[1:low_freq_end])
        boosted_low_energy = np.mean(boosted_spectrum[1:low_freq_end])

        original_high_energy = np.mean(original_spectrum[high_freq_start:high_freq_end])
        boosted_high_energy = np.mean(boosted_spectrum[high_freq_start:high_freq_end])

        # Low frequencies should be boosted relative to high frequencies
        original_ratio = original_low_energy / (original_high_energy + 1e-10)
        boosted_ratio = boosted_low_energy / (boosted_high_energy + 1e-10)

        assert boosted_ratio > original_ratio

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

        tilt_setting = f"{start_freq_ratio}:{tilt_db}"
        if tilt_type == "high":
            eq = Equalizer(high_tilt_settings=tilt_setting)
        else:
            eq = Equalizer(low_tilt_settings=tilt_setting)

        result = eq.apply(wavetable)

        # Basic sanity checks
        assert len(result) == len(wavetable)
        assert np.isfinite(result).all()
        assert not np.isnan(result).any()

        # RMS should be preserved (default behavior)
        original_rms = np.sqrt(np.mean(wavetable**2))
        result_rms = np.sqrt(np.mean(result**2))
        if original_rms > EPSILON:
            assert abs(result_rms - original_rms) < 0.01


class TestEqualizerIntegration:
    """Integration tests combining EQ with other DSP functions."""

    def test_eq_with_mipmap_generation(self):
        """Test EQ integration with mipmap generation."""
        # Generate base waveform
        _, base_wave = WaveGenerator().sawtooth(1.0)

        # Apply EQ first
        eq = Equalizer(eq_settings="2000:4.0:1.5")
        eq_wave = eq.apply(base_wave)

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

        # Apply EQ
        eq = Equalizer(eq_settings="1500:-3.0")
        eq_wave = eq.apply(aligned_wave)

        # Should still be well-aligned
        assert abs(eq_wave[0]) < 0.1

    def test_eq_with_extreme_settings(self):
        """Test EQ with extreme but valid settings."""
        wavetable = np.sin(2 * np.pi * np.linspace(0, 1, 256, endpoint=False))

        # Extreme boost and cut
        eq = Equalizer(eq_settings="1000:20.0:5.0,8000:-20.0:0.5")
        result = eq.apply(wavetable)

        # Should not produce invalid values
        assert np.isfinite(result).all()
        assert not np.isnan(result).any()

        # Should be clipped to reasonable range
        assert np.max(np.abs(result)) <= 1.1  # Some tolerance for numerical precision

    def test_tilt_eq_with_mipmap_generation(self):
        """Test tilt EQ integration with mipmap generation."""
        # Generate base waveform
        _, base_wave = WaveGenerator().sawtooth(1.0)

        # Apply tilt EQ first
        eq = Equalizer(high_tilt_settings="0.4:4.0")
        eq_wave = eq.apply(base_wave)

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

        # Apply tilt EQ
        eq_high = Equalizer(high_tilt_settings="0.3:3.0")
        eq_low = Equalizer(low_tilt_settings="0.7:-2.0")

        eq_wave_high = eq_high.apply(aligned_wave)
        eq_wave_low = eq_low.apply(aligned_wave)

        # Should still be well-aligned
        assert abs(eq_wave_high[0]) < 0.1
        assert abs(eq_wave_low[0]) < 0.1

    def test_tilt_eq_with_extreme_settings(self):
        """Test tilt EQ with extreme but valid settings."""
        wavetable = np.sin(2 * np.pi * np.linspace(0, 1, 256, endpoint=False))

        # Extreme settings
        eq_high = Equalizer(high_tilt_settings="0.1:20.0")
        eq_low = Equalizer(low_tilt_settings="0.9:-20.0")

        result_high = eq_high.apply(wavetable)
        result_low = eq_low.apply(wavetable)

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

        # Apply both parametric and tilt EQ
        eq = Equalizer(eq_settings="2000:3.0:2.0", high_tilt_settings="0.6:2.0")
        combined_result = eq.apply(wavetable)

        # Should process without error
        assert np.isfinite(combined_result).all()
        assert not np.isnan(combined_result).any()
        assert len(combined_result) == len(wavetable)

    def test_all_eq_types_combined(self):
        """Test all EQ types applied together."""
        wavetable = np.sin(2 * np.pi * np.linspace(0, 1, 1024, endpoint=False))

        # Apply all types of EQ
        eq = Equalizer(
            eq_settings="1000:2.0:1.5,8000:-3.0",
            high_tilt_settings="0.7:3.0",
            low_tilt_settings="0.3:1.5",
        )
        result = eq.apply(wavetable)

        # Should process without error and maintain expected properties
        assert np.isfinite(result).all()
        assert not np.isnan(result).any()
        assert len(result) == len(wavetable)

        # RMS should be preserved
        original_rms = np.sqrt(np.mean(wavetable**2))
        result_rms = np.sqrt(np.mean(result**2))
        assert abs(result_rms - original_rms) < 0.01
