"""Type stubs for generated protobuf module."""

from typing import ClassVar

from google.protobuf.internal.containers import RepeatedScalarFieldContainer
from google.protobuf.message import Message

# Enum types
class WavetableType:
    WAVETABLE_TYPE_UNSPECIFIED: ClassVar[int]
    WAVETABLE_TYPE_CLASSIC_DIGITAL: ClassVar[int]
    WAVETABLE_TYPE_HIGH_RESOLUTION: ClassVar[int]
    WAVETABLE_TYPE_VINTAGE_EMULATION: ClassVar[int]
    WAVETABLE_TYPE_PCM_SAMPLE: ClassVar[int]
    WAVETABLE_TYPE_CUSTOM: ClassVar[int]

class NormalizationMethod:
    NORMALIZATION_UNSPECIFIED: ClassVar[int]
    NORMALIZATION_PEAK: ClassVar[int]
    NORMALIZATION_RMS: ClassVar[int]
    NORMALIZATION_NONE: ClassVar[int]

class InterpolationHint:
    INTERPOLATION_UNSPECIFIED: ClassVar[int]
    INTERPOLATION_LINEAR: ClassVar[int]
    INTERPOLATION_CUBIC: ClassVar[int]
    INTERPOLATION_SINC: ClassVar[int]

# Integer constants for enums
WAVETABLE_TYPE_UNSPECIFIED: int
WAVETABLE_TYPE_CLASSIC_DIGITAL: int
WAVETABLE_TYPE_HIGH_RESOLUTION: int
WAVETABLE_TYPE_VINTAGE_EMULATION: int
WAVETABLE_TYPE_PCM_SAMPLE: int
WAVETABLE_TYPE_CUSTOM: int

NORMALIZATION_UNSPECIFIED: int
NORMALIZATION_PEAK: int
NORMALIZATION_RMS: int
NORMALIZATION_NONE: int

INTERPOLATION_UNSPECIFIED: int
INTERPOLATION_LINEAR: int
INTERPOLATION_CUBIC: int
INTERPOLATION_SINC: int

# Message types
class ClassicDigitalMetadata(Message):
    original_bit_depth: int
    original_sample_rate: int
    source_hardware: str
    harmonic_caps: RepeatedScalarFieldContainer[int]

    def __init__(
        self,
        *,
        original_bit_depth: int | None = ...,
        original_sample_rate: int | None = ...,
        source_hardware: str | None = ...,
        harmonic_caps: list[int] | None = ...,
    ) -> None: ...
    def HasField(self, field_name: str) -> bool: ...
    def ClearField(self, field_name: str) -> None: ...

class HighResolutionMetadata(Message):
    max_harmonics: int
    interpolation_hint: int
    source_synth: str

    def __init__(
        self,
        *,
        max_harmonics: int | None = ...,
        interpolation_hint: int = ...,
        source_synth: str | None = ...,
    ) -> None: ...
    def HasField(self, field_name: str) -> bool: ...
    def ClearField(self, field_name: str) -> None: ...

class VintageEmulationMetadata(Message):
    emulated_hardware: str
    oscillator_type: str
    preserves_aliasing: bool

    def __init__(
        self,
        *,
        emulated_hardware: str | None = ...,
        oscillator_type: str | None = ...,
        preserves_aliasing: bool | None = ...,
    ) -> None: ...
    def HasField(self, field_name: str) -> bool: ...
    def ClearField(self, field_name: str) -> None: ...

class PcmSampleMetadata(Message):
    original_sample_rate: int
    root_note: int
    loop_start: int
    loop_end: int

    def __init__(
        self,
        *,
        original_sample_rate: int | None = ...,
        root_note: int | None = ...,
        loop_start: int | None = ...,
        loop_end: int | None = ...,
    ) -> None: ...
    def HasField(self, field_name: str) -> bool: ...
    def ClearField(self, field_name: str) -> None: ...

class WavetableMetadata(Message):
    schema_version: int
    wavetable_type: int
    frame_length: int
    num_frames: int
    num_mip_levels: int
    mip_frame_lengths: RepeatedScalarFieldContainer[int]
    normalization_method: int
    source_bit_depth: int
    author: str
    name: str
    description: str
    tuning_reference: float
    generation_parameters: str
    sample_rate: int

    @property
    def classic_digital(self) -> ClassicDigitalMetadata: ...
    @property
    def high_resolution(self) -> HighResolutionMetadata: ...
    @property
    def vintage_emulation(self) -> VintageEmulationMetadata: ...
    @property
    def pcm_sample(self) -> PcmSampleMetadata: ...

    def __init__(
        self,
        *,
        schema_version: int = ...,
        wavetable_type: int = ...,
        frame_length: int = ...,
        num_frames: int = ...,
        num_mip_levels: int = ...,
        mip_frame_lengths: list[int] | None = ...,
        normalization_method: int = ...,
        source_bit_depth: int | None = ...,
        author: str | None = ...,
        name: str | None = ...,
        description: str | None = ...,
        tuning_reference: float | None = ...,
        generation_parameters: str | None = ...,
        sample_rate: int | None = ...,
        classic_digital: ClassicDigitalMetadata | None = ...,
        high_resolution: HighResolutionMetadata | None = ...,
        vintage_emulation: VintageEmulationMetadata | None = ...,
        pcm_sample: PcmSampleMetadata | None = ...,
    ) -> None: ...
    def HasField(self, field_name: str) -> bool: ...
    def ClearField(self, field_name: str) -> None: ...
    def WhichOneof(self, oneof_group: str) -> str | None: ...
