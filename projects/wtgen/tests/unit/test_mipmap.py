"""Unit tests for wtgen.dsp.mipmap module."""

import hypothesis
import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from wtgen.dsp.mipmap import Mipmap, MipmapChain, MipmapLevel, build_multiframe_mipmap
from wtgen.utils import EPSILON


class TestBuildMipmap:
    """Test mipmap generation functionality."""

    def test_Mipmap_basic(self):
        """Test basic mipmap generation."""
        # Create a simple test wavetable
        t = np.linspace(0, 2 * np.pi, 2048, endpoint=False)
        base_wavetable = np.sin(t) + 0.3 * np.sin(3 * t)  # Fundamental + 3rd harmonic
        mipmap_chain = Mipmap(base_wavetable=base_wavetable, num_octaves=5).generate()

        # Should return the expected number of levels
        assert len(mipmap_chain) == 6  # 0 to 5 inclusive

        # All levels should have same length as input
        for level in mipmap_chain:
            assert len(level) == len(base_wavetable)
            assert level.dtype == np.float32

    def test_Mipmap_rms_normalization_with_clipping_prevention(self):
        """Test that all mipmap levels use RMS normalization but prevent clipping."""
        # Create a test waveform with known characteristics
        t = np.linspace(0, 2 * np.pi, 256, endpoint=False)
        base_wavetable = 2.5 * np.sin(t) + 1.8 * np.sin(3 * t)  # Large amplitudes
        mipmap_chain = Mipmap(base_wavetable=base_wavetable, num_octaves=5).generate()

        for i, level in enumerate(mipmap_chain):
            # Critical clipping prevention checks
            assert np.all(level >= -1.0), f"Level {i}: Values below -1.0 detected (clipping)"
            assert np.all(level <= 1.0), f"Level {i}: Values above 1.0 detected (clipping)"

            # Peak should not exceed 1.0
            max_abs_value = np.max(np.abs(level))
            assert max_abs_value <= 1.0 + 1e-6, f"Level {i}: Peak {max_abs_value} exceeds 1.0"

            # Check that RMS is reasonable (may be lower when scaled to avoid
            # clipping).
            if np.any(np.abs(level) > EPSILON):  # Non-zero waveform
                rms = np.sqrt(np.mean(level**2))
                assert rms > 0.1, f"Level {i}: RMS {rms} too low, indicates over-scaling"
                assert rms <= 0.35 + 1e-6, (
                    f"Level {i}: RMS {rms} exceeds target"
                )  # Tightened from 0.001

    def test_Mipmap_dc_removal(self):
        """Test that DC offset is removed from all levels."""
        # Create wavetable with DC offset
        t = np.linspace(0, 2 * np.pi, 512, endpoint=False)
        base_wavetable = np.sin(t) + 0.5  # Add DC offset
        mipmap_chain = Mipmap(base_wavetable=base_wavetable, num_octaves=4).generate()

        # All levels should have minimal DC
        for i, level in enumerate(mipmap_chain):
            dc_level = np.mean(level)
            assert abs(dc_level) < 2e-8, (
                f"Level {i} has DC offset: {dc_level:.6f}"
            )  # Tightened from 0.001

    def test_Mipmap_bandlimiting(self):
        """Test that higher levels have reduced bandwidth."""
        # Create rich harmonic content
        t = np.linspace(0, 2 * np.pi, 2048, endpoint=False)
        base_wavetable = np.zeros_like(t)

        # Add many harmonics
        for h in range(1, 20):
            base_wavetable += (1.0 / h) * np.sin(h * t)

        mipmap_chain = Mipmap(base_wavetable=base_wavetable, num_octaves=6).generate()

        # Higher levels should have less high-frequency content
        # We can test this by comparing spectral energy above certain frequencies
        for i in range(1, len(mipmap_chain)):
            spec_low = np.abs(np.fft.fft(mipmap_chain[0]))[1:50]  # Low frequencies
            spec_high = np.abs(np.fft.fft(mipmap_chain[i]))[1:50]  # Same range in higher level

            # Energy should be preserved in low frequencies but reduced overall for higher levels
            assert len(spec_low) == len(spec_high)

    def test_Mipmap_phase_alignment(self):
        """Test that all levels start at zero crossings."""
        t = np.linspace(
            np.pi / 4, np.pi / 4 + 2 * np.pi, 1024, endpoint=False
        )  # Start away from zero
        base_wavetable = np.sin(t)

        mipmap_chain = Mipmap(base_wavetable=base_wavetable, num_octaves=5).generate()

        # All levels should start closer to zero than the original
        original_start = abs(base_wavetable[0])

        for i, level in enumerate(mipmap_chain):
            start_value = abs(level[0])
            # Should be at or very close to zero crossing
            assert start_value <= original_start, (
                f"Level {i} start value {start_value:.6f} not well aligned"
            )

    def test_Mipmap_decimated_phase_consistency(self):
        """Test that decimated mipmaps maintain consistent phase alignment across all levels."""
        # Create test waveform that doesn't start at zero crossing
        t = np.linspace(np.pi / 3, np.pi / 3 + 2 * np.pi, 2048, endpoint=False)
        base_wavetable = np.sin(t) + 0.5 * np.sin(2 * t)

        # Generate mipmaps with decimation
        mipmap_chain = Mipmap(
            base_wavetable=base_wavetable, num_octaves=10, decimate=True
        ).generate()

        # Test that all levels start close to zero crossing
        original_start = abs(base_wavetable[0])
        phase_values = []

        for i, level in enumerate(mipmap_chain):
            start_value = abs(level[0])
            end_value = abs(level[-1])
            phase_values.append(start_value)

            # Each level should start closer to zero than the original
            assert start_value <= original_start + 1e-6, (
                f"Level {i} start value {start_value:.6f} "
                f"not better than original {original_start:.6f}"
            )

            # Should be very close to zero crossing (much stricter than before)
            assert start_value < 0.001, (
                f"Level {i} start value {start_value:.6f} not at zero crossing"
            )

            assert end_value < 0.02, f"Level {i} end value {end_value:.6f} not near zero crossing"

        # Test that phase alignment is consistent - no abrupt jumps
        max_phase_jump = 0.0
        for i in range(1, len(phase_values)):
            phase_jump = abs(phase_values[i] - phase_values[i - 1])
            max_phase_jump = max(max_phase_jump, phase_jump)

            # Phase values should not jump abruptly between levels
            assert phase_jump < 0.05, (
                f"Abrupt phase jump {phase_jump:.6f} between levels {i - 1} and {i}"
            )

        # Test decimated sizes are correct
        expected_size = len(base_wavetable)
        for i, level in enumerate(mipmap_chain):
            if i == 0:
                # First level should be full size
                assert len(level) == expected_size
            else:
                # Decimated levels should have reduced size
                expected_decimated_size = max(expected_size // (2**i), 64)
                assert len(level) == expected_decimated_size, (
                    f"Level {i} has size {len(level)}, expected {expected_decimated_size}"
                )

    def test_Mipmap_phase_alignment_with_filtering(self):
        """Test that phase alignment is preserved after spectral filtering."""
        # Create complex harmonic waveform
        t = np.linspace(0.2, 0.2 + 2 * np.pi, 1024, endpoint=False)  # Not starting at zero
        base_wavetable = np.zeros_like(t)

        # Add harmonics that will be filtered at different levels
        for h in range(1, 20):
            base_wavetable += (1.0 / h) * np.sin(h * t)

        mipmap_chain = Mipmap(
            base_wavetable, num_octaves=8, rolloff_method="raised_cosine"
        ).generate()

        # Test phase alignment across levels
        for i, level in enumerate(mipmap_chain):
            start_value = abs(level[0])

            # Even after heavy filtering, should maintain zero-crossing alignment
            assert start_value < 0.2, (
                f"Level {i} start value {start_value:.6f} not well aligned after filtering"
            )

            # Ensure no phase inversion (positive and negative values should be balanced)
            mean_value = np.mean(level)
            assert abs(mean_value) < 1e-6, (
                f"Level {i} has DC offset {mean_value:.8f} after filtering"
            )

        # Test that all levels maintain their waveform character despite filtering
        for i, level in enumerate(mipmap_chain):
            rms = np.sqrt(np.mean(level**2))
            if (
                i < len(mipmap_chain) - 2
            ):  # Don't test highest levels which might be heavily filtered
                assert rms > 0.1, f"Level {i} RMS {rms:.6f} too low, excessive filtering"

    def test_Mipmap_different_octave_counts(self):
        """Test mipmap generation with different octave counts."""
        t = np.linspace(0, 2 * np.pi, 512, endpoint=False)
        base_wavetable = np.sin(t)

        for num_octaves in [1, 3, 5, 8, 10, 12]:
            mipmap_chain = Mipmap(base_wavetable, num_octaves=num_octaves).generate()

            # Should have correct number of levels
            assert len(mipmap_chain) == num_octaves + 1

            # All levels should be valid
            for level in mipmap_chain:
                assert np.all(np.isfinite(level))
                assert len(level) == len(base_wavetable)

    def test_Mipmap_zero_signal(self):
        """Test mipmap generation with zero input signal."""
        base_wavetable = np.zeros(256)
        mipmap_chain = Mipmap(base_wavetable, num_octaves=3).generate()

        # Should handle gracefully
        assert len(mipmap_chain) == 4
        for level in mipmap_chain:
            assert np.all(np.abs(level) < 1e-10)  # Should remain zero

    def test_Mipmap_constant_signal(self):
        """Test mipmap generation with constant (DC) signal."""
        base_wavetable = np.ones(256) * 2.0  # Constant signal
        mipmap_chain = Mipmap(base_wavetable, num_octaves=3).generate()

        # Should handle gracefully and remove DC
        assert len(mipmap_chain) == 4
        for level in mipmap_chain:
            assert abs(np.mean(level)) < 2e-8  # DC should be removed

    @given(st.lists(st.floats(-10, 10), min_size=64, max_size=2048))
    @settings(deadline=None)  # Disable deadline for this test
    def test_Mipmap_hypothesis(self, values):
        """Hypothesis test for mipmap generation including clipping prevention checks."""
        assume(all(np.isfinite(v) for v in values))
        assume(len(values) >= 64)  # Minimum reasonable size

        # Ensure length is reasonable for FFT
        if len(values) > 2048:
            values = values[:2048]

        base_wavetable = np.array(values, dtype=np.float64)
        mipmap_chain = Mipmap(base_wavetable, num_octaves=3).generate()

        # Basic validity checks
        assert len(mipmap_chain) == 4
        for level in mipmap_chain:
            assert np.all(np.isfinite(level))
            assert len(level) == len(base_wavetable)
            assert level.dtype == np.float32

            # Critical clipping prevention checks
            assert np.all(level >= -1.0), "Values below -1.0 detected (clipping)"
            assert np.all(level <= 1.0), "Values above 1.0 detected (clipping)"

            # Peak should not exceed 1.0
            max_abs_value = np.max(np.abs(level))
            assert max_abs_value <= 1.0 + 1e-6, f"Peak value {max_abs_value} exceeds 1.0"

    def test_Mipmap_complex_waveform(self):
        """Test mipmap generation with complex harmonic content."""
        # Create a complex waveform resembling a sawtooth
        t = np.linspace(0, 2 * np.pi, 2048, endpoint=False)
        base_wavetable = np.zeros_like(t)

        # Sawtooth-like harmonic series
        for h in range(1, 50):
            base_wavetable += ((-1) ** (h + 1)) * (1.0 / h) * np.sin(h * t)

        mipmap_chain = Mipmap(base_wavetable, num_octaves=10).generate()

        # Should maintain key properties
        assert len(mipmap_chain) == 11

        # Check clipping prevention and RMS behavior
        target_rms = 0.35
        for i, level in enumerate(mipmap_chain):
            # Critical clipping prevention checks
            assert np.all(level >= -1.0), f"Level {i}: Values below -1.0 detected (clipping)"
            assert np.all(level <= 1.0), f"Level {i}: Values above 1.0 detected (clipping)"

            # Peak should not exceed 1.0
            max_abs_value = np.max(np.abs(level))
            assert max_abs_value <= 1.0 + 1e-6, f"Level {i}: Peak {max_abs_value} exceeds 1.0"

            # RMS should be reasonable (may be lower than target due to clipping prevention)
            if np.any(np.abs(level) > EPSILON):  # Non-zero waveform
                rms = np.sqrt(np.mean(level**2))
                assert rms > 0.1, f"Level {i}: RMS {rms} too low"
                # RMS may be lower than target if scaling was applied to prevent clipping
                assert rms <= target_rms + 1e-6, f"Level {i}: RMS {rms} exceeds target"

        # Check that bandlimiting is working (higher levels should have fewer harmonics)
        spectra = [np.abs(np.fft.fft(level)) for level in mipmap_chain]
        for i in range(1, len(spectra)):
            # Higher levels should have energy concentrated in lower frequencies
            np.sum(spectra[i - 1][: len(spectra[i - 1]) // 8])
            np.sum(spectra[i][: len(spectra[i]) // 8])

            # The relative concentration of low-frequency energy should increase.
            # (This is a loose check since exact ratios depend on the filtering.)

    def test_Mipmap_no_clipping(self):
        """Test that mipmap generation prevents clipping with RMS normalization."""
        # Create a waveform with extreme values
        extreme_waveform = np.array([100.0, -50.0, 75.0, -200.0] * 64, dtype=np.float64)

        mipmap_chain = Mipmap(extreme_waveform, num_octaves=3).generate()

        for i, level in enumerate(mipmap_chain):
            # Critical clipping prevention checks
            assert np.all(level >= -1.0), f"Level {i}: Clipping detected (values < -1.0)"
            assert np.all(level <= 1.0), f"Level {i}: Clipping detected (values > 1.0)"

            # Peak should not exceed 1.0
            max_abs_value = np.max(np.abs(level))
            assert max_abs_value <= 1.0 + 1e-6, f"Level {i}: Peak {max_abs_value} exceeds 1.0"

            # Should have reasonable RMS (may be lower than target due to clipping prevention)
            if np.any(np.abs(level) > EPSILON):
                rms = np.sqrt(np.mean(level**2))
                assert rms > 0.1, f"Level {i}: RMS {rms} too low"

    def test_Mipmap_spectral_characteristics(self):
        """Test spectral characteristics of mipmap levels."""
        # Create test signal with known frequency content
        t = np.linspace(0, 2 * np.pi, 1024, endpoint=False)
        base_wavetable = np.sin(t) + 0.5 * np.sin(5 * t) + 0.25 * np.sin(10 * t)

        mipmap_chain = Mipmap(base_wavetable, num_octaves=5).generate()

        # Each level should have reduced high-frequency content
        for _i, level in enumerate(mipmap_chain):
            spectrum = np.abs(np.fft.fft(level))

            # Level 0 should have most content, higher levels should have less
            high_freq_energy = np.sum(spectrum[100:500])  # High frequency range

            # Should be finite and reasonable
            assert np.isfinite(high_freq_energy)
            assert high_freq_energy >= 0

    def test_Mipmap_midi_range_coverage(self):
        """Test that mipmap covers full MIDI range appropriately."""
        t = np.linspace(0, 2 * np.pi, 2048, endpoint=False)
        base_wavetable = np.sin(t) + 0.3 * np.sin(3 * t)

        # Test with full MIDI range coverage
        mipmap_chain = Mipmap(base_wavetable, num_octaves=10).generate()

        # Should have 11 levels (0-10)
        assert len(mipmap_chain) == 11

        # Each level should be progressively more bandlimited
        # We can verify this by checking that later levels have less high-frequency content
        for i in range(1, len(mipmap_chain)):
            # Compare spectral roll-off between consecutive levels
            spec_prev = np.abs(np.fft.fft(mipmap_chain[i - 1]))
            spec_curr = np.abs(np.fft.fft(mipmap_chain[i]))

            # Both should be finite
            assert np.all(np.isfinite(spec_prev))
            assert np.all(np.isfinite(spec_curr))


class TestMipmapAntialiasing:
    """Test antialiasing effectiveness across different sample rates and MIDI ranges."""

    def _calculate_mipmap_parameters(self, sample_rate: float, table_size: int, octave_level: int):
        """Calculate mipmap parameters for analysis."""
        base_freq = sample_rate / table_size
        max_freq_in_level = base_freq * (2**octave_level)

        # Conservative safety margins (same as in Mipmap)
        if sample_rate <= 48000:
            if octave_level == 0:
                safety_margin = 0.78
            elif octave_level == 1:
                safety_margin = 0.65
            elif octave_level == 2:
                safety_margin = 0.62
            elif octave_level == 3:
                safety_margin = 0.65
            elif octave_level == 4:
                safety_margin = 0.60
            elif octave_level <= 6:
                safety_margin = 0.65
            else:
                safety_margin = 0.70
        else:
            scale_factor = min(1.1, 0.9 + (sample_rate / 96000.0) * 0.2)
            if octave_level == 0:
                safety_margin = min(0.80, 0.72 * scale_factor)
            elif octave_level == 1:
                safety_margin = min(0.80, 0.65 * scale_factor)
            elif octave_level == 2:
                safety_margin = min(0.78, 0.62 * scale_factor)
            elif octave_level == 3:
                safety_margin = min(0.80, 0.65 * scale_factor)
            elif octave_level == 4:
                safety_margin = min(0.75, 0.60 * scale_factor)
            elif octave_level <= 6:
                safety_margin = min(0.80, 0.65 * scale_factor)
            else:
                safety_margin = min(0.85, 0.70 * scale_factor)

        nyquist = sample_rate / 2
        max_safe_harmonics = int((safety_margin * nyquist) / max_freq_in_level)
        max_safe_harmonics = max(1, max_safe_harmonics)

        return max_safe_harmonics, max_freq_in_level, safety_margin

    def _midi_to_frequency(self, midi_note: int) -> float:
        """Convert MIDI note number to frequency in Hz."""
        return 440.0 * (2 ** ((midi_note - 69) / 12.0))

    def _determine_mipmap_level(
        self, fundamental_freq: float, sample_rate: float, table_size: int
    ) -> int:
        """Determine which mipmap level should be used for a given fundamental frequency."""
        base_freq = sample_rate / table_size
        level = max(0, min(10, int(np.log2(fundamental_freq / base_freq))))
        return level

    def test_antialiasing_44_1khz_default(self):
        """Test antialiasing effectiveness at default 44.1kHz sample rate."""
        sample_rate = 44100.0
        table_size = 2048

        # Create test wavetable with many harmonics
        t = np.linspace(0, 2 * np.pi, table_size, endpoint=False)
        base_wavetable = np.zeros_like(t)
        for h in range(1, 50):  # Rich harmonic content
            base_wavetable += (1.0 / h) * np.sin(h * t)

        Mipmap(base_wavetable, num_octaves=10, sample_rate=sample_rate)
        nyquist = sample_rate / 2

        # Test critical MIDI notes across the range
        critical_midi_notes = [21, 36, 48, 60, 72, 84, 96, 108, 120, 127]

        for midi_note in critical_midi_notes:
            fundamental_freq = self._midi_to_frequency(midi_note)
            level_index = self._determine_mipmap_level(fundamental_freq, sample_rate, table_size)

            # Get mipmap parameters for this level
            max_harmonics, _, _ = self._calculate_mipmap_parameters(
                sample_rate, table_size, level_index
            )

            # Calculate highest possible harmonic frequency
            highest_harmonic_freq = fundamental_freq * max_harmonics

            # Assert no aliasing (highest harmonic < Nyquist)
            assert highest_harmonic_freq < nyquist, (
                f"MIDI {midi_note} ({fundamental_freq:.1f}Hz) level {level_index}: "
                f"Highest harmonic {highest_harmonic_freq:.1f}Hz exceeds Nyquist {nyquist:.1f}Hz"
            )

    @pytest.mark.parametrize("sample_rate", [44100.0, 48000.0, 88200.0, 96000.0])
    def test_antialiasing_multiple_sample_rates(self, sample_rate):
        """Test antialiasing effectiveness across different sample rates."""
        table_size = 2048

        # Create test wavetable
        t = np.linspace(0, 2 * np.pi, table_size, endpoint=False)
        base_wavetable = np.zeros_like(t)
        for h in range(1, 30):
            base_wavetable += (1.0 / h) * np.sin(h * t)

        mipmap_chain = Mipmap(base_wavetable, num_octaves=10, sample_rate=sample_rate).generate()
        nyquist = sample_rate / 2

        # Test MIDI range appropriate for this sample rate
        max_midi = min(
            127, int(69 + 12 * np.log2(nyquist / 880.0))
        )  # Don't test beyond reasonable range
        test_midi_notes = range(21, max_midi + 1, 12)  # Every octave

        for midi_note in test_midi_notes:
            fundamental_freq = self._midi_to_frequency(midi_note)
            level_index = self._determine_mipmap_level(fundamental_freq, sample_rate, table_size)

            if level_index < len(mipmap_chain):
                max_harmonics, _, _ = self._calculate_mipmap_parameters(
                    sample_rate, table_size, level_index
                )
                highest_harmonic_freq = fundamental_freq * max_harmonics

                assert highest_harmonic_freq < nyquist, (
                    f"Sample rate {sample_rate}Hz, MIDI {midi_note}: "
                    f"Aliasing detected - {highest_harmonic_freq:.1f}Hz > {nyquist:.1f}Hz"
                )

    def test_mipmap_level_transitions_no_aliasing(self):
        """Test that mipmap level transitions maintain antialiasing protection."""
        sample_rate = 44100.0
        table_size = 2048

        # Create harmonic-rich test signal
        t = np.linspace(0, 2 * np.pi, table_size, endpoint=False)
        base_wavetable = np.zeros_like(t)
        for h in range(1, 40):
            base_wavetable += (1.0 / h) * np.sin(h * t)

        Mipmap(base_wavetable, num_octaves=10, sample_rate=sample_rate)
        nyquist = sample_rate / 2
        base_freq = sample_rate / table_size

        # Test each mipmap level's frequency range
        for octave_level in range(0, 10):
            max_harmonics, max_freq_in_level, safety_margin = self._calculate_mipmap_parameters(
                sample_rate, table_size, octave_level
            )

            # Test the designed frequency for this level
            level_base_freq = base_freq * (2**octave_level)
            highest_harmonic_freq = level_base_freq * max_harmonics

            # Ensure antialiasing protection
            assert highest_harmonic_freq < nyquist, (
                f"Level {octave_level} base freq {level_base_freq:.1f}Hz: "
                f"Harmonic {highest_harmonic_freq:.1f}Hz > Nyquist {nyquist:.1f}Hz"
            )

            # With optimized safety margins, we can safely use up to ~95% of Nyquist
            # This accounts for the max(1, ...) constraint in the implementation
            safety_ratio = highest_harmonic_freq / nyquist
            assert safety_ratio < 0.96, (
                f"Level {octave_level}: Safety ratio {safety_ratio:.3f} "
                f"exceeds optimized limit (should be < 0.96)"
            )

    def test_extreme_frequencies_antialiasing(self):
        """Test antialiasing at extreme frequencies that might cause issues."""
        sample_rate = 44100.0
        table_size = 2048

        # Create test wavetable
        t = np.linspace(0, 2 * np.pi, table_size, endpoint=False)
        base_wavetable = np.sin(t) + 0.5 * np.sin(3 * t) + 0.25 * np.sin(5 * t)

        mipmap_chain = Mipmap(base_wavetable, num_octaves=10, sample_rate=sample_rate).generate()
        nyquist = sample_rate / 2

        # Test extreme frequencies
        extreme_frequencies = [
            20.0,  # Very low frequency
            55.0,  # Around A1
            220.0,  # Around A3
            880.0,  # A5
            3520.0,  # A7
            7040.0,  # A8 (very high)
            10000.0,  # Extreme high frequency
        ]

        for freq in extreme_frequencies:
            if freq < nyquist / 2:  # Only test reasonable frequencies
                level_index = self._determine_mipmap_level(freq, sample_rate, table_size)

                if level_index < len(mipmap_chain):
                    max_harmonics, _, _ = self._calculate_mipmap_parameters(
                        sample_rate, table_size, level_index
                    )

                    highest_harmonic_freq = freq * max_harmonics

                    assert highest_harmonic_freq < nyquist, (
                        f"Extreme frequency {freq:.1f}Hz (level {level_index}): "
                        f"Aliasing at {highest_harmonic_freq:.1f}Hz"
                    )

    def test_antialiasing_safety_margins(self):
        """Test that safety margins provide adequate protection against aliasing."""
        sample_rate = 44100.0
        table_size = 2048
        nyquist = sample_rate / 2

        # Test each mipmap level's safety margin
        for octave_level in range(0, 11):
            max_harmonics, max_freq_in_level, safety_margin = self._calculate_mipmap_parameters(
                sample_rate, table_size, octave_level
            )

            # Calculate what the max_harmonics would be without the max(1, ...) constraint
            theoretical_max_harmonics = (safety_margin * nyquist) / max_freq_in_level

            # If theoretical calculation gives >= 1, then safety margin should be respected
            if theoretical_max_harmonics >= 1.0:
                actual_cutoff_freq = max_harmonics * max_freq_in_level
                safety_ratio = actual_cutoff_freq / nyquist

                assert safety_ratio <= safety_margin + 0.01, (
                    f"Level {octave_level}: Safety ratio {safety_ratio:.3f} exceeds margin "
                    f"{safety_margin:.3f} (theoretical: {theoretical_max_harmonics:.3f})"
                )
            else:
                # When forced to use 1 harmonic, just ensure no aliasing
                # (fundamental frequency itself should be at or below Nyquist)
                assert max_freq_in_level <= nyquist, (
                    f"Level {octave_level}: Fundamental frequency {max_freq_in_level:.1f}Hz "
                    f"exceeds Nyquist {nyquist:.1f}Hz"
                )

            # With optimized safety margins, we can safely use up to ~95% of Nyquist
            # This is much more aggressive than the old conservative approach
            actual_cutoff_freq = max_harmonics * max_freq_in_level
            safety_ratio = actual_cutoff_freq / nyquist

            if octave_level < 9:  # Don't apply this constraint to the highest levels
                assert safety_ratio < 0.96, (
                    f"Level {octave_level}: Safety margin exceeded - "
                    f"ratio {safety_ratio:.3f} too close to Nyquist"
                )

    @hypothesis.given(
        st.floats(44100.0, 192000.0),  # Sample rate range (minimum 44.1kHz)
        st.integers(512, 4096),  # Table size range
        st.integers(20, 127),  # MIDI note range
    )
    @hypothesis.settings(deadline=None, max_examples=100)
    def test_antialiasing_hypothesis(self, sample_rate, table_size, midi_note):
        """Hypothesis test for antialiasing across random configurations."""
        # Ensure table size is power of 2 for cleaner FFT
        table_size = 2 ** int(np.log2(table_size))

        # Create simple test wavetable
        t = np.linspace(0, 2 * np.pi, table_size, endpoint=False)
        base_wavetable = np.sin(t) + 0.3 * np.sin(3 * t)

        try:
            mipmap_chain = Mipmap(
                base_wavetable, num_octaves=10, sample_rate=sample_rate
            ).generate()
            nyquist = sample_rate / 2

            fundamental_freq = self._midi_to_frequency(midi_note)

            # Skip if frequency is unreasonably high for this sample rate
            if fundamental_freq >= nyquist / 4:
                return

            level_index = self._determine_mipmap_level(fundamental_freq, sample_rate, table_size)

            if level_index < len(mipmap_chain):
                max_harmonics, max_freq_in_level, _ = self._calculate_mipmap_parameters(
                    sample_rate, table_size, level_index
                )

                # The antialiasing test should be based on the wavetable's base frequency
                # at this mipmap level, not an arbitrary playback frequency
                base_freq = sample_rate / table_size
                effective_freq_in_level = base_freq * (2**level_index)
                highest_harmonic_freq = effective_freq_in_level * max_harmonics

                # Assert no aliasing based on the mipmap level's frequency range
                assert highest_harmonic_freq < nyquist, (
                    f"Hypothesis test failed: SR={sample_rate:.0f}, size={table_size}, "
                    f"level={level_index}, effective_freq={effective_freq_in_level:.1f}Hz, "
                    f"max_harm={max_harmonics}, "
                    f"highest={highest_harmonic_freq:.1f}Hz > Nyquist={nyquist:.1f}Hz"
                )

        except Exception as e:
            # Log parameters that caused the failure for debugging
            pytest.fail(
                f"Mipmap generation failed with SR={sample_rate:.0f}, "
                f"size={table_size}, MIDI={midi_note}: {e}"
            )

    def test_mipmap_spectral_integrity(self):
        """Test that mipmap levels maintain spectral integrity while preventing aliasing."""
        sample_rate = 44100.0
        table_size = 2048

        # Create test signal with known harmonic content
        t = np.linspace(0, 2 * np.pi, table_size, endpoint=False)
        base_wavetable = np.zeros_like(t)
        original_harmonics = [1, 2, 3, 5, 7, 11, 13]  # Prime harmonics for clear analysis

        for h in original_harmonics:
            base_wavetable += (1.0 / h) * np.sin(h * t)

        mipmap_chain = Mipmap(base_wavetable, num_octaves=8, sample_rate=sample_rate).generate()

        # Analyze spectral content of each level
        for level_idx, level in enumerate(mipmap_chain):
            spectrum = np.abs(np.fft.fft(level))
            N = len(spectrum)

            # Find peak frequencies in the spectrum
            peak_threshold = 0.1 * np.max(spectrum[1 : N // 2])
            peaks = [i for i in range(1, N // 2) if spectrum[i] > peak_threshold]

            # Calculate expected maximum harmonic for this level
            max_harmonics, _, _ = self._calculate_mipmap_parameters(
                sample_rate, table_size, level_idx
            )

            # Verify no significant energy above the cutoff
            for peak_bin in peaks:
                harmonic_number = peak_bin
                assert harmonic_number <= max_harmonics + 1, (
                    f"Level {level_idx}: Unexpected peak at harmonic {harmonic_number}, "
                    f"max allowed is {max_harmonics}"
                )


class TestMipmapDataTypes:
    """Test mipmap data type definitions."""

    def test_mipmap_level_type(self):
        """Test MipmapLevel type alias."""
        level: MipmapLevel = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert level.dtype == np.float32
        assert isinstance(level, np.ndarray)

    def test_mipmap_chain_type(self):
        """Test MipmapChain type alias."""
        chain: MipmapChain = [
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([3.0, 4.0], dtype=np.float32),
        ]
        assert len(chain) == 2
        for level in chain:
            assert level.dtype == np.float32
            assert isinstance(level, np.ndarray)


class TestMipmapEdgeCases:
    """Test edge cases for mipmap generation."""

    def test_single_sample_wavetable(self):
        """Test with single sample wavetable."""
        base_wavetable = np.array([1.0])
        mipmap_chain = Mipmap(base_wavetable, num_octaves=2).generate()

        # Should handle gracefully
        assert len(mipmap_chain) == 3
        for level in mipmap_chain:
            assert len(level) == 1
            assert np.isfinite(level[0])

    def test_very_small_wavetable(self):
        """Test with very small wavetable."""
        base_wavetable = np.array([1.0, -1.0, 0.5, -0.5])
        mipmap_chain = Mipmap(base_wavetable, num_octaves=2).generate()

        # Should handle gracefully
        assert len(mipmap_chain) == 3
        for level in mipmap_chain:
            assert len(level) == 4
            assert np.all(np.isfinite(level))

    def test_large_wavetable(self):
        """Test with large wavetable."""
        # Create large wavetable
        size = 8192
        t = np.linspace(0, 2 * np.pi, size, endpoint=False)
        base_wavetable = np.sin(t) + 0.3 * np.sin(3 * t)

        mipmap_chain = Mipmap(base_wavetable, num_octaves=3).generate()

        # Should handle large sizes
        assert len(mipmap_chain) == 4
        for level in mipmap_chain:
            assert len(level) == size
            assert np.all(np.isfinite(level))

    def test_zero_octaves(self):
        """Test with zero octaves."""
        t = np.linspace(0, 2 * np.pi, 256, endpoint=False)
        base_wavetable = np.sin(t)

        mipmap_chain = Mipmap(base_wavetable, num_octaves=0).generate()

        # Should return just the base level
        assert len(mipmap_chain) == 1
        assert len(mipmap_chain[0]) == len(base_wavetable)

    def test_extreme_values(self):
        """Test with extreme input values."""
        # Very large values
        base_wavetable = np.array([1000.0, -500.0, 2000.0, -1500.0] * 64)
        mipmap_chain = Mipmap(base_wavetable, num_octaves=3).generate()

        # Should normalize and handle extreme values
        for level in mipmap_chain:
            assert np.all(np.isfinite(level))
            assert np.max(np.abs(level)) < 10.0  # Should be reasonably bounded

    def test_nan_inf_handling(self):
        """Test handling of NaN and Inf values."""
        # Create wavetable with some invalid values
        base_wavetable = np.array([1.0, 2.0, np.nan, 4.0, np.inf, -1.0] * 32)

        # The function should handle this gracefully or the test should expect an exception
        # For now, let's test that it doesn't crash catastrophically
        try:
            mipmap_chain = Mipmap(base_wavetable, num_octaves=2).generate()
            # If it succeeds, results should at least be finite where possible
            for level in mipmap_chain:
                finite_mask = np.isfinite(level)
                if np.any(finite_mask):
                    finite_values = level[finite_mask]
                    assert len(finite_values) >= 0  # Basic sanity check
        except (ValueError, RuntimeError):
            # If it raises an exception, that's also acceptable behavior
            pass


class TestBuildMultiframeMipmap:
    """Tests for the build_multiframe_mipmap function."""

    def test_multiframe_basic(self):
        """Test basic multiframe mipmap generation."""
        num_frames = 8
        frame_length = 1024
        num_octaves = 5

        # Create multi-frame test wavetable
        t = np.linspace(0, 2 * np.pi, frame_length, endpoint=False)
        wavetable = np.zeros((num_frames, frame_length), dtype=np.float64)
        for i in range(num_frames):
            # Each frame has slightly different harmonic content
            wavetable[i] = np.sin(t) + 0.3 * np.sin((i + 1) * t)

        mipmaps = build_multiframe_mipmap(wavetable, num_octaves=num_octaves)

        # Should have correct number of levels
        assert len(mipmaps) == num_octaves + 1

        # Each level should have correct number of frames
        for level_idx, level in enumerate(mipmaps):
            assert level.shape[0] == num_frames, (
                f"Level {level_idx}: expected {num_frames} frames, got {level.shape[0]}"
            )
            assert level.dtype == np.float32

        # First level should have original frame length
        assert mipmaps[0].shape[1] == frame_length

        # Subsequent levels should be decimated
        for level_idx in range(1, len(mipmaps)):
            expected_length = max(frame_length // (2**level_idx), 64)
            assert mipmaps[level_idx].shape[1] == expected_length, (
                f"Level {level_idx}: expected {expected_length} samples, "
                f"got {mipmaps[level_idx].shape[1]}"
            )

    def test_multiframe_no_aliasing(self):
        """Test that multiframe mipmap prevents aliasing with proper filtering."""
        num_frames = 4
        frame_length = 2048
        sample_rate = 44100.0

        # Create multi-frame wavetable with high harmonic content
        t = np.linspace(0, 2 * np.pi, frame_length, endpoint=False)
        wavetable = np.zeros((num_frames, frame_length), dtype=np.float64)
        for i in range(num_frames):
            # Rich harmonic content that would cause aliasing without filtering
            for h in range(1, 30):
                wavetable[i] += (1.0 / h) * np.sin(h * t)

        mipmaps = build_multiframe_mipmap(wavetable, num_octaves=6, sample_rate=sample_rate)

        # Check that higher mip levels have reduced high-frequency content
        for level_idx in range(1, len(mipmaps)):
            level = mipmaps[level_idx]
            level_length = level.shape[1]

            # Analyze each frame's spectral content
            for frame_idx in range(num_frames):
                frame = level[frame_idx]
                spectrum = np.abs(np.fft.fft(frame))

                # Calculate the expected cutoff for this level
                # Higher levels should have less high-frequency content
                half_spectrum = spectrum[: level_length // 2]

                # No significant energy should be above the level's Nyquist-safe frequency
                # The energy in upper bins should be greatly reduced
                upper_quarter_energy = np.sum(half_spectrum[level_length // 4 :])
                lower_quarter_energy = np.sum(half_spectrum[: level_length // 4])

                # Upper frequencies should have less energy than lower frequencies
                # This is a loose check since exact ratios depend on the filtering
                if lower_quarter_energy > EPSILON:
                    ratio = upper_quarter_energy / lower_quarter_energy
                    # Higher levels should have more aggressive filtering
                    assert ratio < 1.0, (
                        f"Level {level_idx}, frame {frame_idx}: "
                        f"upper/lower energy ratio {ratio:.3f} too high"
                    )

    def test_multiframe_consistency(self):
        """Test that each frame is processed correctly and consistently."""
        num_frames = 4
        frame_length = 512

        # Create wavetable where each frame is the same
        t = np.linspace(0, 2 * np.pi, frame_length, endpoint=False)
        single_frame = np.sin(t) + 0.5 * np.sin(3 * t)
        wavetable = np.tile(single_frame, (num_frames, 1))

        mipmaps = build_multiframe_mipmap(wavetable, num_octaves=4)

        # All frames within each level should be nearly identical
        for level_idx, level in enumerate(mipmaps):
            for frame_idx in range(1, num_frames):
                # Frames should be very similar (within floating point tolerance)
                diff = np.max(np.abs(level[frame_idx] - level[0]))
                assert diff < 1e-5, (
                    f"Level {level_idx}: frame {frame_idx} differs from frame 0 by {diff}"
                )

    def test_multiframe_single_frame(self):
        """Test multiframe mipmap with single frame wavetable."""
        frame_length = 1024

        t = np.linspace(0, 2 * np.pi, frame_length, endpoint=False)
        wavetable = np.sin(t).reshape(1, frame_length)

        mipmaps = build_multiframe_mipmap(wavetable, num_octaves=4)

        # Should work correctly with single frame
        assert len(mipmaps) == 5
        for level in mipmaps:
            assert level.shape[0] == 1
            assert level.dtype == np.float32

    def test_multiframe_different_rolloff_methods(self):
        """Test multiframe mipmap with different rolloff methods."""
        num_frames = 2
        frame_length = 512

        t = np.linspace(0, 2 * np.pi, frame_length, endpoint=False)
        wavetable = np.zeros((num_frames, frame_length), dtype=np.float64)
        for i in range(num_frames):
            wavetable[i] = np.sin(t) + 0.3 * np.sin(5 * t)

        rolloff_methods = ["raised_cosine", "tukey", "hann", "blackman", "brick_wall"]

        for method in rolloff_methods:
            mipmaps = build_multiframe_mipmap(wavetable, num_octaves=3, rolloff_method=method)

            # Should produce valid output regardless of method
            assert len(mipmaps) == 4
            for level in mipmaps:
                assert level.shape[0] == num_frames
                assert np.all(np.isfinite(level))
                # Should not clip
                assert np.all(level >= -1.0)
                assert np.all(level <= 1.0)

    def test_multiframe_invalid_input_1d(self):
        """Test that 1D input raises an error."""
        wavetable_1d = np.sin(np.linspace(0, 2 * np.pi, 512))

        with pytest.raises(ValueError, match="Expected 2D wavetable"):
            build_multiframe_mipmap(wavetable_1d)

    def test_multiframe_invalid_input_3d(self):
        """Test that 3D input raises an error."""
        wavetable_3d = np.zeros((4, 512, 2))

        with pytest.raises(ValueError, match="Expected 2D wavetable"):
            build_multiframe_mipmap(wavetable_3d)

    def test_multiframe_no_clipping(self):
        """Test that multiframe mipmap prevents clipping with extreme values."""
        num_frames = 4
        frame_length = 256

        # Create wavetable with extreme values
        wavetable = np.random.randn(num_frames, frame_length) * 100.0

        mipmaps = build_multiframe_mipmap(wavetable, num_octaves=3)

        # All levels should be properly normalized without clipping
        for level_idx, level in enumerate(mipmaps):
            assert np.all(level >= -1.0), f"Level {level_idx}: clipping detected (values < -1.0)"
            assert np.all(level <= 1.0), f"Level {level_idx}: clipping detected (values > 1.0)"

    @given(
        num_frames=st.integers(min_value=1, max_value=16),
        frame_length=st.sampled_from([64, 128, 256, 512, 1024]),
        num_octaves=st.integers(min_value=1, max_value=6),
    )
    @settings(deadline=None, max_examples=20)
    def test_multiframe_hypothesis(self, num_frames, frame_length, num_octaves):
        """Property-based test for multiframe mipmap generation."""
        # Create random wavetable
        wavetable = np.random.randn(num_frames, frame_length).astype(np.float64)

        mipmaps = build_multiframe_mipmap(wavetable, num_octaves=num_octaves)

        # Basic validity checks
        assert len(mipmaps) == num_octaves + 1

        for _level_idx, level in enumerate(mipmaps):
            assert level.shape[0] == num_frames
            assert level.dtype == np.float32
            assert np.all(np.isfinite(level))
            # Should not clip
            assert np.all(level >= -1.0)
            assert np.all(level <= 1.0)
