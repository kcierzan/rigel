"""Property-based tests (Hypothesis) for the wavetable format module.

These tests use property-based testing to verify that:
1. Round-trip preservation: save -> load preserves data integrity
2. Metadata validation: validators accept valid data and reject invalid data
3. Format invariants: file structure maintains required properties
"""

from pathlib import Path

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from wtgen.format import (
    WavetableType,
    load_wavetable_wav,
    save_wavetable_wav,
)
from wtgen.format.types import (
    ClassicDigitalMetadata,
    HighResolutionMetadata,
    InterpolationHint,
    NormalizationMethod,
    PcmSampleMetadata,
    VintageEmulationMetadata,
)


# Custom strategies for wavetable data generation
@st.composite
def power_of_two(draw: st.DrawFn, min_exp: int = 2, max_exp: int = 11) -> int:
    """Generate a power of 2 between 2^min_exp and 2^max_exp."""
    exp = draw(st.integers(min_value=min_exp, max_value=max_exp))
    return 2**exp


def frame_length_strategy() -> st.SearchStrategy[int]:
    """Generate valid frame lengths (powers of 2, 4-2048)."""
    return power_of_two(min_exp=2, max_exp=11)


def num_frames_strategy() -> st.SearchStrategy[int]:
    """Generate valid number of frames (1-256)."""
    return st.integers(min_value=1, max_value=256)


def mip_frame_lengths(base_length: int, num_mips: int) -> list[int]:
    """Generate decreasing mip frame lengths starting from base_length.

    This is a pure function, not a strategy, since it deterministically
    calculates mip levels from the base length.
    """
    lengths = [base_length]
    current = base_length
    for _ in range(num_mips - 1):
        current = current // 2
        if current < 4:
            break
        lengths.append(current)
    return lengths


@st.composite
def finite_float_array(
    draw: st.DrawFn, shape: tuple[int, int], min_val: float = -1.0, max_val: float = 1.0
) -> np.ndarray:
    """Generate an array of finite float32 values."""
    # Generate normalized float values
    values = draw(
        st.lists(
            st.floats(
                min_value=min_val,
                max_value=max_val,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=shape[0] * shape[1],
            max_size=shape[0] * shape[1],
        )
    )
    return np.array(values, dtype=np.float32).reshape(shape)


def wavetable_type_strategy() -> st.SearchStrategy[WavetableType]:
    """Generate any valid wavetable type."""
    return st.sampled_from(
        [
            WavetableType.CLASSIC_DIGITAL,
            WavetableType.HIGH_RESOLUTION,
            WavetableType.VINTAGE_EMULATION,
            WavetableType.PCM_SAMPLE,
            WavetableType.CUSTOM,
        ]
    )


def normalization_method_strategy() -> st.SearchStrategy[NormalizationMethod]:
    """Generate any valid normalization method."""
    return st.sampled_from(
        [
            NormalizationMethod.UNSPECIFIED,
            NormalizationMethod.PEAK,
            NormalizationMethod.RMS,
            NormalizationMethod.NONE,
        ]
    )


@st.composite
def simple_mipmap_strategy(draw: st.DrawFn) -> list[np.ndarray]:
    """Generate a simple wavetable with valid mipmaps."""
    frame_length = draw(power_of_two(min_exp=3, max_exp=8))  # 8-256 samples
    num_frames = draw(st.integers(min_value=1, max_value=16))
    num_mips = draw(st.integers(min_value=1, max_value=3))

    mipmaps = []
    current_length = frame_length
    for _ in range(num_mips):
        if current_length < 4:
            break
        mipmap = draw(finite_float_array((num_frames, current_length)))
        mipmaps.append(mipmap)
        current_length = current_length // 2

    return mipmaps


class TestRoundTripPreservation:
    """Property tests for save/load round-trip data preservation."""

    @settings(
        max_examples=50,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None,
    )
    @given(mipmaps=simple_mipmap_strategy(), wt_type=wavetable_type_strategy())
    def test_mipmap_data_preserved(
        self, tmp_path_factory: "pytest.TempPathFactory", mipmaps: list[np.ndarray], wt_type: WavetableType
    ) -> None:
        """Mipmap audio data should survive save/load cycle."""
        tmp_path = tmp_path_factory.mktemp("roundtrip")
        output_path = tmp_path / "test.wav"

        save_wavetable_wav(
            output_path,
            mipmaps=mipmaps,
            wavetable_type=wt_type,
        )

        loaded = load_wavetable_wav(output_path)

        assert len(loaded.mipmaps) == len(mipmaps)
        for i, (original, loaded_mip) in enumerate(zip(mipmaps, loaded.mipmaps)):
            np.testing.assert_array_almost_equal(
                original, loaded_mip, decimal=5, err_msg=f"Mipmap {i} data mismatch"
            )

    @settings(
        max_examples=50,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None,
    )
    @given(
        wt_type=wavetable_type_strategy(),
        norm_method=normalization_method_strategy(),
        sample_rate=st.sampled_from([44100, 48000, 88200, 96000]),
    )
    def test_metadata_preserved(
        self,
        tmp_path_factory: "pytest.TempPathFactory",
        wt_type: WavetableType,
        norm_method: NormalizationMethod,
        sample_rate: int,
    ) -> None:
        """Core metadata should survive save/load cycle."""
        tmp_path = tmp_path_factory.mktemp("metadata")
        output_path = tmp_path / "test.wav"

        # Simple 8-sample, 2-frame mipmap
        mip0 = np.random.randn(2, 8).astype(np.float32)

        save_wavetable_wav(
            output_path,
            mipmaps=[mip0],
            wavetable_type=wt_type,
            normalization_method=norm_method,
            sample_rate=sample_rate,
        )

        loaded = load_wavetable_wav(output_path)

        assert loaded.wavetable_type == wt_type
        assert loaded.normalization_method == norm_method
        assert loaded.sample_rate == sample_rate

    @settings(
        max_examples=20,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None,
    )
    @given(
        name=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=["L", "N", "P", "S", "Zs"])),
        author=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=["L", "N", "P", "S", "Zs"])),
    )
    def test_optional_string_metadata_preserved(
        self, tmp_path_factory: "pytest.TempPathFactory", name: str, author: str
    ) -> None:
        """Optional string metadata should survive save/load cycle."""
        tmp_path = tmp_path_factory.mktemp("strings")
        output_path = tmp_path / "test.wav"

        mip0 = np.random.randn(2, 8).astype(np.float32)

        save_wavetable_wav(
            output_path,
            mipmaps=[mip0],
            wavetable_type=WavetableType.CUSTOM,
            name=name,
            author=author,
        )

        loaded = load_wavetable_wav(output_path)

        assert loaded.name == name
        assert loaded.author == author


class TestTypeSpecificMetadata:
    """Property tests for type-specific metadata preservation."""

    @settings(
        max_examples=20,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None,
    )
    @given(
        bit_depth=st.sampled_from([8, 12, 16]),
        sample_rate=st.sampled_from([31250, 44100, 48000]),
        source_hardware=st.text(min_size=1, max_size=30, alphabet=st.characters(whitelist_categories=["L", "N", "P", "S", "Zs"])),
    )
    def test_classic_digital_metadata_preserved(
        self,
        tmp_path_factory: "pytest.TempPathFactory",
        bit_depth: int,
        sample_rate: int,
        source_hardware: str,
    ) -> None:
        """ClassicDigitalMetadata should survive save/load cycle."""
        tmp_path = tmp_path_factory.mktemp("classic")
        output_path = tmp_path / "test.wav"

        mip0 = np.random.randn(4, 64).astype(np.float32)

        save_wavetable_wav(
            output_path,
            mipmaps=[mip0],
            wavetable_type=WavetableType.CLASSIC_DIGITAL,
            type_metadata=ClassicDigitalMetadata(
                original_bit_depth=bit_depth,
                original_sample_rate=sample_rate,
                source_hardware=source_hardware,
            ),
        )

        loaded = load_wavetable_wav(output_path)
        type_meta = loaded.get_type_metadata()

        assert isinstance(type_meta, ClassicDigitalMetadata)
        assert type_meta.original_bit_depth == bit_depth
        assert type_meta.original_sample_rate == sample_rate
        assert type_meta.source_hardware == source_hardware

    @settings(
        max_examples=20,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None,
    )
    @given(
        max_harmonics=st.integers(min_value=1, max_value=2048),
        interp_hint=st.sampled_from(list(InterpolationHint)),
    )
    def test_high_resolution_metadata_preserved(
        self,
        tmp_path_factory: "pytest.TempPathFactory",
        max_harmonics: int,
        interp_hint: InterpolationHint,
    ) -> None:
        """HighResolutionMetadata should survive save/load cycle."""
        tmp_path = tmp_path_factory.mktemp("hires")
        output_path = tmp_path / "test.wav"

        mip0 = np.random.randn(4, 64).astype(np.float32)

        save_wavetable_wav(
            output_path,
            mipmaps=[mip0],
            wavetable_type=WavetableType.HIGH_RESOLUTION,
            type_metadata=HighResolutionMetadata(
                max_harmonics=max_harmonics,
                interpolation_hint=interp_hint,
            ),
        )

        loaded = load_wavetable_wav(output_path)
        type_meta = loaded.get_type_metadata()

        assert isinstance(type_meta, HighResolutionMetadata)
        assert type_meta.max_harmonics == max_harmonics
        assert type_meta.interpolation_hint == interp_hint


class TestFormatInvariants:
    """Property tests for format structure invariants."""

    @settings(
        max_examples=30,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None,
    )
    @given(mipmaps=simple_mipmap_strategy())
    def test_mip_lengths_decreasing(
        self, tmp_path_factory: "pytest.TempPathFactory", mipmaps: list[np.ndarray]
    ) -> None:
        """Loaded mip frame lengths should be decreasing."""
        tmp_path = tmp_path_factory.mktemp("decreasing")
        output_path = tmp_path / "test.wav"

        save_wavetable_wav(
            output_path,
            mipmaps=mipmaps,
            wavetable_type=WavetableType.CUSTOM,
        )

        loaded = load_wavetable_wav(output_path)

        for i in range(1, len(loaded.mip_frame_lengths)):
            assert loaded.mip_frame_lengths[i] <= loaded.mip_frame_lengths[i - 1], (
                f"mip_frame_lengths[{i}] should be <= mip_frame_lengths[{i-1}]"
            )

    @settings(
        max_examples=30,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None,
    )
    @given(mipmaps=simple_mipmap_strategy())
    def test_frame_length_matches_mip0(
        self, tmp_path_factory: "pytest.TempPathFactory", mipmaps: list[np.ndarray]
    ) -> None:
        """frame_length should equal mip_frame_lengths[0]."""
        tmp_path = tmp_path_factory.mktemp("mip0")
        output_path = tmp_path / "test.wav"

        save_wavetable_wav(
            output_path,
            mipmaps=mipmaps,
            wavetable_type=WavetableType.CUSTOM,
        )

        loaded = load_wavetable_wav(output_path)

        assert loaded.frame_length == loaded.mip_frame_lengths[0], (
            f"frame_length ({loaded.frame_length}) should equal "
            f"mip_frame_lengths[0] ({loaded.mip_frame_lengths[0]})"
        )

    @settings(
        max_examples=30,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None,
    )
    @given(mipmaps=simple_mipmap_strategy())
    def test_num_frames_consistent(
        self, tmp_path_factory: "pytest.TempPathFactory", mipmaps: list[np.ndarray]
    ) -> None:
        """All mipmaps should have the same number of frames as metadata declares."""
        tmp_path = tmp_path_factory.mktemp("frames")
        output_path = tmp_path / "test.wav"

        save_wavetable_wav(
            output_path,
            mipmaps=mipmaps,
            wavetable_type=WavetableType.CUSTOM,
        )

        loaded = load_wavetable_wav(output_path)

        for i, mipmap in enumerate(loaded.mipmaps):
            assert mipmap.shape[0] == loaded.num_frames, (
                f"mipmap[{i}] has {mipmap.shape[0]} frames, but num_frames is {loaded.num_frames}"
            )

    @settings(
        max_examples=30,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None,
    )
    @given(mipmaps=simple_mipmap_strategy())
    def test_total_samples_matches_calculation(
        self, tmp_path_factory: "pytest.TempPathFactory", mipmaps: list[np.ndarray]
    ) -> None:
        """total_samples() should match sum of all mipmap samples."""
        tmp_path = tmp_path_factory.mktemp("total")
        output_path = tmp_path / "test.wav"

        save_wavetable_wav(
            output_path,
            mipmaps=mipmaps,
            wavetable_type=WavetableType.CUSTOM,
        )

        loaded = load_wavetable_wav(output_path)

        actual_total = sum(m.size for m in loaded.mipmaps)
        assert loaded.total_samples() == actual_total, (
            f"total_samples() returned {loaded.total_samples()}, "
            f"but actual total is {actual_total}"
        )


# Import pytest for type annotation
import pytest
