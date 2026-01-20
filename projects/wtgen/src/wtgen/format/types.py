"""Python types for wavetable metadata.

These types provide a Pythonic interface for working with wavetable metadata,
with conversion to/from protobuf messages for serialization.
"""

from dataclasses import dataclass, field
from enum import IntEnum

# Re-export protobuf types
from wtgen.format.proto import wavetable_pb2 as pb


class WavetableType(IntEnum):
    """Classification of wavetable for handling.

    Readers should handle unknown values by treating them as CUSTOM.
    This ensures forward compatibility when new types are added.
    """

    UNSPECIFIED = 0
    """Default value for unset/unknown type."""

    CLASSIC_DIGITAL = 1
    """PPG Wave-style wavetables (256 samples, 64 frames, 8-bit source)."""

    HIGH_RESOLUTION = 2
    """Modern high-resolution wavetables (2048+ samples, many frames)."""

    VINTAGE_EMULATION = 3
    """Vintage oscillator emulations (OSCar, Wasp, etc.)."""

    PCM_SAMPLE = 4
    """Single-cycle PCM samples (AWM/SY99-style)."""

    CUSTOM = 5
    """User-defined wavetables."""

    @classmethod
    def from_proto(cls, proto_value: int) -> "WavetableType":
        """Convert from protobuf enum value, treating unknown as CUSTOM."""
        try:
            return cls(proto_value)
        except ValueError:
            return cls.CUSTOM

    def to_proto(self) -> int:
        """Convert to protobuf enum value."""
        return self.value

    @property
    def display_name(self) -> str:
        """Human-readable name for this type."""
        names = {
            self.UNSPECIFIED: "Unspecified",
            self.CLASSIC_DIGITAL: "Classic Digital",
            self.HIGH_RESOLUTION: "High Resolution",
            self.VINTAGE_EMULATION: "Vintage Emulation",
            self.PCM_SAMPLE: "PCM Sample",
            self.CUSTOM: "Custom",
        }
        return names.get(self, "Unknown")


class NormalizationMethod(IntEnum):
    """How audio samples were normalized."""

    UNSPECIFIED = 0
    """Not specified."""

    PEAK = 1
    """Normalized to peak amplitude (max sample = 1.0 or -1.0)."""

    RMS = 2
    """Normalized to target RMS level."""

    NONE = 3
    """No normalization applied (original levels preserved)."""

    @classmethod
    def from_proto(cls, proto_value: int) -> "NormalizationMethod":
        """Convert from protobuf enum value."""
        try:
            return cls(proto_value)
        except ValueError:
            return cls.UNSPECIFIED

    def to_proto(self) -> int:
        """Convert to protobuf enum value."""
        return self.value


class InterpolationHint(IntEnum):
    """Suggested interpolation method for playback."""

    UNSPECIFIED = 0
    """Use reader default."""

    LINEAR = 1
    """Linear interpolation (fast, some aliasing)."""

    CUBIC = 2
    """Cubic interpolation (Catmull-Rom, good quality)."""

    SINC = 3
    """Windowed sinc interpolation (highest quality, expensive)."""

    @classmethod
    def from_proto(cls, proto_value: int) -> "InterpolationHint":
        """Convert from protobuf enum value."""
        try:
            return cls(proto_value)
        except ValueError:
            return cls.UNSPECIFIED

    def to_proto(self) -> int:
        """Convert to protobuf enum value."""
        return self.value


@dataclass
class ClassicDigitalMetadata:
    """Type-specific metadata for CLASSIC_DIGITAL wavetables."""

    original_bit_depth: int | None = None
    """Original bit depth (typically 8 for PPG-style)."""

    original_sample_rate: int | None = None
    """Original sample rate if known."""

    source_hardware: str | None = None
    """Source hardware identifier (e.g., "PPG Wave 2.2")."""

    harmonic_caps: list[int] = field(default_factory=list)
    """Maximum harmonics per mip level."""

    def to_proto(self) -> pb.ClassicDigitalMetadata:
        """Convert to protobuf message."""
        msg = pb.ClassicDigitalMetadata()
        if self.original_bit_depth is not None:
            msg.original_bit_depth = self.original_bit_depth
        if self.original_sample_rate is not None:
            msg.original_sample_rate = self.original_sample_rate
        if self.source_hardware is not None:
            msg.source_hardware = self.source_hardware
        msg.harmonic_caps.extend(self.harmonic_caps)
        return msg

    @classmethod
    def from_proto(cls, msg: pb.ClassicDigitalMetadata) -> "ClassicDigitalMetadata":
        """Create from protobuf message."""
        return cls(
            original_bit_depth=(
                msg.original_bit_depth if msg.HasField("original_bit_depth") else None
            ),
            original_sample_rate=(
                msg.original_sample_rate if msg.HasField("original_sample_rate") else None
            ),
            source_hardware=msg.source_hardware if msg.HasField("source_hardware") else None,
            harmonic_caps=list(msg.harmonic_caps),
        )


@dataclass
class HighResolutionMetadata:
    """Type-specific metadata for HIGH_RESOLUTION wavetables."""

    max_harmonics: int | None = None
    """Maximum harmonic count at mip level 0."""

    interpolation_hint: InterpolationHint = InterpolationHint.UNSPECIFIED
    """Suggested interpolation method for playback."""

    source_synth: str | None = None
    """Source synthesizer if known (e.g., "AN1x")."""

    def to_proto(self) -> pb.HighResolutionMetadata:
        """Convert to protobuf message."""
        msg = pb.HighResolutionMetadata()
        if self.max_harmonics is not None:
            msg.max_harmonics = self.max_harmonics
        msg.interpolation_hint = self.interpolation_hint.to_proto()
        if self.source_synth is not None:
            msg.source_synth = self.source_synth
        return msg

    @classmethod
    def from_proto(cls, msg: pb.HighResolutionMetadata) -> "HighResolutionMetadata":
        """Create from protobuf message."""
        return cls(
            max_harmonics=msg.max_harmonics if msg.HasField("max_harmonics") else None,
            interpolation_hint=InterpolationHint.from_proto(msg.interpolation_hint),
            source_synth=msg.source_synth if msg.HasField("source_synth") else None,
        )


@dataclass
class VintageEmulationMetadata:
    """Type-specific metadata for VINTAGE_EMULATION wavetables."""

    emulated_hardware: str | None = None
    """Target hardware being emulated (e.g., "OSCar", "EDP Wasp")."""

    oscillator_type: str | None = None
    """Specific oscillator type/variant."""

    preserves_aliasing: bool | None = None
    """Whether aliasing artifacts are intentionally preserved."""

    def to_proto(self) -> pb.VintageEmulationMetadata:
        """Convert to protobuf message."""
        msg = pb.VintageEmulationMetadata()
        if self.emulated_hardware is not None:
            msg.emulated_hardware = self.emulated_hardware
        if self.oscillator_type is not None:
            msg.oscillator_type = self.oscillator_type
        if self.preserves_aliasing is not None:
            msg.preserves_aliasing = self.preserves_aliasing
        return msg

    @classmethod
    def from_proto(cls, msg: pb.VintageEmulationMetadata) -> "VintageEmulationMetadata":
        """Create from protobuf message."""
        return cls(
            emulated_hardware=(
                msg.emulated_hardware if msg.HasField("emulated_hardware") else None
            ),
            oscillator_type=msg.oscillator_type if msg.HasField("oscillator_type") else None,
            preserves_aliasing=(
                msg.preserves_aliasing if msg.HasField("preserves_aliasing") else None
            ),
        )


@dataclass
class PcmSampleMetadata:
    """Type-specific metadata for PCM_SAMPLE wavetables."""

    original_sample_rate: int | None = None
    """Original sample rate of the source material."""

    root_note: int | None = None
    """MIDI note number for unity playback (Middle C = 60)."""

    loop_start: int | None = None
    """Loop start sample index (0-based)."""

    loop_end: int | None = None
    """Loop end sample index (0-based, exclusive)."""

    def to_proto(self) -> pb.PcmSampleMetadata:
        """Convert to protobuf message."""
        msg = pb.PcmSampleMetadata()
        if self.original_sample_rate is not None:
            msg.original_sample_rate = self.original_sample_rate
        if self.root_note is not None:
            msg.root_note = self.root_note
        if self.loop_start is not None:
            msg.loop_start = self.loop_start
        if self.loop_end is not None:
            msg.loop_end = self.loop_end
        return msg

    @classmethod
    def from_proto(cls, msg: pb.PcmSampleMetadata) -> "PcmSampleMetadata":
        """Create from protobuf message."""
        return cls(
            original_sample_rate=(
                msg.original_sample_rate if msg.HasField("original_sample_rate") else None
            ),
            root_note=msg.root_note if msg.HasField("root_note") else None,
            loop_start=msg.loop_start if msg.HasField("loop_start") else None,
            loop_end=msg.loop_end if msg.HasField("loop_end") else None,
        )


# Type alias for type-specific metadata union
TypeMetadata = (
    ClassicDigitalMetadata | HighResolutionMetadata | VintageEmulationMetadata | PcmSampleMetadata
)
