from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

type WavetableData = NDArray[np.float32]
type MipmapList = list[WavetableData]
type WavetableTables = dict[str, MipmapList]
type WavetableMetadata = dict[str, Any]

BitDepth = Literal[16, 24, 32]


class WaveformType(str, Enum):
    sawtooth = "sawtooth"
    square = "square"
    pulse = "pulse"
    triangle = "triangle"
    sine = "sine"
    polyblep_saw = "polyblep_saw"


@dataclass
class EQBand:
    frequency: float
    gain_db: float
    q_factor: float


@dataclass
class TiltSettings:
    start_ratio: float
    gain_db: float


@dataclass
class WavetableGenerationParams:
    octaves: int = 8
    size: int = 2048
    decimate: bool = False


@dataclass
class ExportParams:
    export_wav: bool = False
    wav_dir: Path | None = None
    wav_sample_rate: int = 44100
    wav_bit_depth: int = 16


@dataclass
class ProcessingParams:
    eq: str | None = None
    high_tilt: str | None = None
    low_tilt: str | None = None


@dataclass
class HarmonicPartial:
    harmonic: int
    amplitude: float
    phase: float


type HarmonicPartialList = list[HarmonicPartial]
