# Data Model: LFO Modulation Source

**Date**: 2025-12-14
**Feature**: 004-lfo-modulation-source

## Entity Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         rigel-modulation                            │
├─────────────────────────────────────────────────────────────────────┤
│  Traits:                                                            │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ ModulationSource                                             │   │
│  │   + reset(&mut self)                                         │   │
│  │   + update(&mut self, timebase: &Timebase)                   │   │
│  │   + value(&self) -> f32                                      │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Types:                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐    │
│  │ Lfo             │  │ LfoWaveshape    │  │ LfoPhaseMode    │    │
│  │ (config+state)  │  │ (enum)          │  │ (enum)          │    │
│  └────────┬────────┘  └─────────────────┘  └─────────────────┘    │
│           │                                                         │
│           │ uses                                                    │
│           ▼                                                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐    │
│  │ LfoRateMode     │  │ NoteDivision    │  │ Rng             │    │
│  │ (enum)          │  │ (struct)        │  │ (PRNG state)    │    │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘    │
│                                                                     │
│  ┌─────────────────┐                                               │
│  │ LfoPolarity     │                                               │
│  │ (enum)          │                                               │
│  └─────────────────┘                                               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Core Entities

### ModulationSource (Trait)

Trait interface for all modulation sources. Moved from `rigel-timing`.

```rust
pub trait ModulationSource {
    /// Reset the modulation source to initial state.
    fn reset(&mut self);

    /// Update the modulation source state at control rate.
    fn update(&mut self, timebase: &Timebase);

    /// Get the current output value [-1.0, 1.0] or [0.0, 1.0].
    fn value(&self) -> f32;
}
```

**Validation Rules**:
- `value()` must return a value within the configured polarity range
- `update()` must complete in constant time
- `reset()` must restore initial state completely

---

### Lfo

Primary LFO implementation. Combines configuration and runtime state.

```rust
#[derive(Clone, Copy, Debug)]
pub struct Lfo {
    // Configuration (set once or via setters)
    waveshape: LfoWaveshape,
    rate_mode: LfoRateMode,
    phase_mode: LfoPhaseMode,
    polarity: LfoPolarity,
    start_phase: f32,       // [0.0, 1.0] normalized
    pulse_width: f32,       // [0.01, 0.99] for Pulse waveshape

    // Runtime state (changes during processing)
    phase: f32,             // [0.0, 1.0) current phase position
    current_value: f32,     // cached output value
    rng: Rng,               // PRNG for S&H and noise
    held_value: f32,        // stored value for S&H waveshape
}
```

**Field Constraints**:

| Field | Type | Valid Range | Default |
|-------|------|-------------|---------|
| waveshape | LfoWaveshape | enum variant | Sine |
| rate_mode | LfoRateMode | enum variant | Hz(1.0) |
| phase_mode | LfoPhaseMode | enum variant | FreeRunning |
| polarity | LfoPolarity | enum variant | Bipolar |
| start_phase | f32 | [0.0, 1.0] | 0.0 |
| pulse_width | f32 | [0.01, 0.99] | 0.5 |
| phase | f32 | [0.0, 1.0) | 0.0 |
| current_value | f32 | [-1.0, 1.0] or [0.0, 1.0] | 0.0 |

**Methods**:

```rust
impl Lfo {
    /// Create new LFO with default settings
    pub fn new() -> Self;

    /// Create LFO with specific configuration
    pub fn with_config(
        waveshape: LfoWaveshape,
        rate_mode: LfoRateMode,
        phase_mode: LfoPhaseMode,
        polarity: LfoPolarity,
    ) -> Self;

    // Configuration setters
    pub fn set_waveshape(&mut self, waveshape: LfoWaveshape);
    pub fn set_rate(&mut self, rate: LfoRateMode);
    pub fn set_phase_mode(&mut self, mode: LfoPhaseMode);
    pub fn set_polarity(&mut self, polarity: LfoPolarity);
    pub fn set_start_phase(&mut self, phase: f32);
    pub fn set_pulse_width(&mut self, width: f32);

    // Configuration getters
    pub fn waveshape(&self) -> LfoWaveshape;
    pub fn rate_mode(&self) -> LfoRateMode;
    pub fn phase_mode(&self) -> LfoPhaseMode;
    pub fn polarity(&self) -> LfoPolarity;
    pub fn start_phase(&self) -> f32;
    pub fn pulse_width(&self) -> f32;

    // Runtime queries
    pub fn phase(&self) -> f32;

    /// Trigger phase reset (call on note-on if in Retrigger mode)
    pub fn trigger(&mut self);

    /// Get effective rate in Hz given current tempo
    pub fn effective_rate_hz(&self, bpm: f32) -> f32;
}

impl ModulationSource for Lfo {
    fn reset(&mut self);
    fn update(&mut self, timebase: &Timebase);
    fn value(&self) -> f32;
}
```

---

### LfoWaveshape (Enum)

Determines how phase maps to output value.

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LfoWaveshape {
    Sine,
    Triangle,
    Saw,
    Square,
    Pulse,
    SampleAndHold,
    Noise,
}
```

**Waveshape Formulas** (bipolar, phase in [0.0, 1.0)):

| Waveshape | Formula |
|-----------|---------|
| Sine | `sin(phase * TAU)` |
| Triangle | `if phase < 0.5 { 4*phase - 1 } else { 3 - 4*phase }` |
| Saw | `2*phase - 1` |
| Square | `if phase < 0.5 { 1.0 } else { -1.0 }` |
| Pulse | `if phase < pulse_width { 1.0 } else { -1.0 }` |
| SampleAndHold | `held_value` (update on phase wrap) |
| Noise | `rng.next_f32()` (new value each update) |

---

### LfoPhaseMode (Enum)

Determines phase behavior on note events.

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LfoPhaseMode {
    #[default]
    FreeRunning,
    Retrigger,
}
```

**State Transitions**:

```
FreeRunning:
  note_on → (no effect, phase continues)
  reset() → phase = start_phase

Retrigger:
  note_on → phase = start_phase
  reset() → phase = start_phase
```

---

### LfoPolarity (Enum)

Determines output value range.

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LfoPolarity {
    #[default]
    Bipolar,   // [-1.0, 1.0]
    Unipolar,  // [0.0, 1.0]
}
```

**Conversion**: `unipolar = (bipolar + 1.0) * 0.5`

---

### LfoRateMode (Enum)

Determines how LFO rate is calculated.

```rust
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LfoRateMode {
    /// Fixed rate in Hz
    Hz(f32),
    /// Tempo-synchronized rate
    TempoSync {
        division: NoteDivision,
        bpm: f32,
    },
}
```

**Validation Rules**:
- `Hz` rate must be in [0.01, 100.0] Hz
- `bpm` must be in [1.0, 999.0] BPM (clamped internally)

---

### NoteDivision (Struct)

Musical note division for tempo sync.

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NoteDivision {
    pub base: NoteBase,
    pub modifier: NoteModifier,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum NoteBase {
    Whole,      // 1/1
    Half,       // 1/2
    #[default]
    Quarter,    // 1/4
    Eighth,     // 1/8
    Sixteenth,  // 1/16
    ThirtySecond, // 1/32
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum NoteModifier {
    #[default]
    Normal,
    Dotted,     // 1.5x duration
    Triplet,    // 2/3 duration
}
```

**Rate Multipliers** (at 1 BPM = 1/60 Hz base):

| Base | Multiplier |
|------|------------|
| Whole | 0.25 |
| Half | 0.5 |
| Quarter | 1.0 |
| Eighth | 2.0 |
| Sixteenth | 4.0 |
| ThirtySecond | 8.0 |

**Modifier Effects**:
- Normal: × 1.0
- Dotted: × (2/3) (longer duration = slower rate)
- Triplet: × 1.5 (shorter duration = faster rate)

---

### Rng (Struct)

PCG32 pseudo-random number generator.

```rust
#[derive(Clone, Copy, Debug)]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Create with seed
    pub const fn new(seed: u64) -> Self;

    /// Generate random u32
    pub fn next_u32(&mut self) -> u32;

    /// Generate random f32 in [-1.0, 1.0]
    pub fn next_f32(&mut self) -> f32;
}
```

**Internal State**: 8 bytes (u64)

---

## Entity Relationships

```
Lfo
 ├── has-a → LfoWaveshape (configuration)
 ├── has-a → LfoRateMode
 │            └── may contain → NoteDivision
 ├── has-a → LfoPhaseMode (configuration)
 ├── has-a → LfoPolarity (configuration)
 ├── has-a → Rng (runtime state)
 └── implements → ModulationSource (trait)

ModulationSource
 └── uses → Timebase (from rigel-timing)
```

---

## Memory Layout

| Type | Size (bytes) | Alignment |
|------|--------------|-----------|
| Lfo | ~40 | 8 |
| LfoWaveshape | 1 | 1 |
| LfoPhaseMode | 1 | 1 |
| LfoPolarity | 1 | 1 |
| LfoRateMode | 12 | 4 |
| NoteDivision | 2 | 1 |
| Rng | 8 | 8 |

All types are `Copy` + `Clone` + `Send` + `Sync`.

---

## Default Values

```rust
impl Default for Lfo {
    fn default() -> Self {
        Self {
            waveshape: LfoWaveshape::Sine,
            rate_mode: LfoRateMode::Hz(1.0),
            phase_mode: LfoPhaseMode::FreeRunning,
            polarity: LfoPolarity::Bipolar,
            start_phase: 0.0,
            pulse_width: 0.5,
            phase: 0.0,
            current_value: 0.0,
            rng: Rng::new(0x12345678),
            held_value: 0.0,
        }
    }
}
```
