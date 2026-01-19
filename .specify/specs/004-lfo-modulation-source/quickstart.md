# Quickstart: LFO Modulation Source

## Overview

The `rigel-modulation` crate provides modulation sources for the Rigel synthesizer. The primary component is an LFO (Low Frequency Oscillator) that generates periodic waveforms for parameter modulation.

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
rigel-modulation = { path = "../modulation" }
rigel-timing = { path = "../timing" }
```

## Basic Usage

### Creating an LFO

```rust
use rigel_modulation::{Lfo, LfoWaveshape, LfoRateMode, ModulationSource};
use rigel_timing::Timebase;

// Create with defaults (sine, 1 Hz, bipolar, free-running)
let mut lfo = Lfo::new();

// Or with specific configuration
let mut lfo = Lfo::with_config(
    LfoWaveshape::Triangle,
    LfoRateMode::Hz(2.0),
    LfoPhaseMode::Retrigger,
    LfoPolarity::Bipolar,
);
```

### Processing in Audio Callback

```rust
use rigel_modulation::{Lfo, ModulationSource};
use rigel_timing::Timebase;

fn process_block(
    lfo: &mut Lfo,
    timebase: &mut Timebase,
    output: &mut [f32],
    block_size: usize,
) {
    // Advance timebase
    timebase.advance_block(block_size as u32);

    // Update LFO at control rate
    lfo.update(timebase);

    // Get modulation value (constant for this block)
    let mod_value = lfo.value();

    // Apply modulation to audio
    for sample in output.iter_mut() {
        *sample *= 1.0 + mod_value * 0.5; // Example: modulate amplitude
    }
}
```

### Tempo Synchronization

```rust
use rigel_modulation::{Lfo, LfoRateMode, NoteDivision, NoteBase, NoteModifier};

let mut lfo = Lfo::new();

// Set to quarter note at 120 BPM
lfo.set_rate(LfoRateMode::TempoSync {
    division: NoteDivision::normal(NoteBase::Quarter),
    bpm: 120.0,
});

// The LFO will now cycle at 2 Hz (120 BPM = 2 beats/second)

// For dotted eighth notes:
lfo.set_rate(LfoRateMode::TempoSync {
    division: NoteDivision::dotted(NoteBase::Eighth),
    bpm: 120.0,
});

// For triplet sixteenths:
lfo.set_rate(LfoRateMode::TempoSync {
    division: NoteDivision::triplet(NoteBase::Sixteenth),
    bpm: 120.0,
});
```

### Note Triggering

```rust
use rigel_modulation::{Lfo, LfoPhaseMode};

let mut lfo = Lfo::new();
lfo.set_phase_mode(LfoPhaseMode::Retrigger);
lfo.set_start_phase(0.25); // Start at 90° (peak of sine)

// On note-on event:
fn on_note_on(lfo: &mut Lfo) {
    lfo.trigger(); // Phase resets to start_phase
}
```

### Waveshapes

```rust
use rigel_modulation::{Lfo, LfoWaveshape};

let mut lfo = Lfo::new();

// Standard waveshapes
lfo.set_waveshape(LfoWaveshape::Sine);      // Smooth oscillation
lfo.set_waveshape(LfoWaveshape::Triangle);  // Linear ramp up/down
lfo.set_waveshape(LfoWaveshape::Saw);       // Linear ramp up, instant reset
lfo.set_waveshape(LfoWaveshape::Square);    // 50% duty cycle
lfo.set_waveshape(LfoWaveshape::Pulse);     // Variable duty cycle
lfo.set_waveshape(LfoWaveshape::SampleAndHold); // Random steps
lfo.set_waveshape(LfoWaveshape::Noise);     // Continuous random

// For pulse waveshape, set the duty cycle:
lfo.set_waveshape(LfoWaveshape::Pulse);
lfo.set_pulse_width(0.25); // 25% duty cycle (short spike)
```

### Polarity Modes

```rust
use rigel_modulation::{Lfo, LfoPolarity};

let mut lfo = Lfo::new();

// Bipolar: output in [-1.0, 1.0]
lfo.set_polarity(LfoPolarity::Bipolar);

// Unipolar: output in [0.0, 1.0]
lfo.set_polarity(LfoPolarity::Unipolar);
```

## Control Rate Integration

For efficient processing with massive polyphony, LFOs should be updated at control rate (typically every 32-64 samples) rather than every sample.

```rust
use rigel_modulation::{Lfo, ModulationSource};
use rigel_timing::{Timebase, ControlRateClock};

fn process_block_with_control_rate(
    lfo: &mut Lfo,
    timebase: &mut Timebase,
    clock: &mut ControlRateClock,
    output: &mut [f32],
    block_size: u32,
) {
    // Advance timebase for this block
    timebase.advance_block(block_size);

    // Update LFO at control rate intervals
    for _offset in clock.advance(block_size) {
        lfo.update(timebase);
    }

    // lfo.value() returns cached value - very cheap to call per-sample
    let mod_value = lfo.value();

    // Apply modulation
    for sample in output.iter_mut() {
        *sample *= 1.0 + mod_value * 0.5;
    }
}
```

### Multiple LFOs with Shared Control Rate

```rust
use rigel_modulation::{Lfo, ModulationSource};
use rigel_timing::{Timebase, ControlRateClock};

struct PolyphonicSynth {
    voices: [Voice; 64],
    clock: ControlRateClock,
    timebase: Timebase,
}

struct Voice {
    lfo: Lfo,
    // ... other components
}

impl PolyphonicSynth {
    fn process_block(&mut self, block_size: u32) {
        self.timebase.advance_block(block_size);

        // Update all LFOs at control rate - efficient for 64+ voices
        for _offset in self.clock.advance(block_size) {
            for voice in &mut self.voices {
                voice.lfo.update(&self.timebase);
            }
        }

        // Per-sample processing uses cached lfo.value()
        // ...
    }
}
```

### Performance Characteristics

| Control Rate Interval | Updates per 1024-sample block | 64-voice LFO updates |
|----------------------|------------------------------|----------------------|
| 1 (per-sample) | 1024 | 65,536 |
| 32 | 32 | 2,048 |
| 64 | 16 | 1,024 |
| 128 | 8 | 512 |

Using 64-sample control rate reduces CPU usage by ~64× compared to per-sample processing.

## Integration with Voice System

```rust
use rigel_modulation::{Lfo, LfoPhaseMode, ModulationSource};
use rigel_timing::Timebase;

struct Voice {
    lfo: Lfo,
    // ... other voice components
}

impl Voice {
    fn new() -> Self {
        let mut lfo = Lfo::new();
        lfo.set_phase_mode(LfoPhaseMode::Retrigger);
        Self { lfo }
    }

    fn note_on(&mut self) {
        self.lfo.trigger();
    }

    fn process(&mut self, timebase: &Timebase) -> f32 {
        self.lfo.update(timebase);
        self.lfo.value()
    }
}
```

## Implementing Custom Modulation Sources

```rust
use rigel_modulation::ModulationSource;
use rigel_timing::Timebase;

/// Simple envelope follower
#[derive(Clone, Copy)]
struct EnvelopeFollower {
    current: f32,
    attack: f32,
    release: f32,
}

impl ModulationSource for EnvelopeFollower {
    fn reset(&mut self) {
        self.current = 0.0;
    }

    fn update(&mut self, _timebase: &Timebase) {
        // Implementation would track input level
    }

    fn value(&self) -> f32 {
        self.current
    }
}
```

## Real-Time Safety Checklist

When using `rigel-modulation` in real-time audio:

- ✅ All types are `Copy` - no reference counting overhead
- ✅ No heap allocations in `update()` or `value()`
- ✅ Constant-time operations regardless of configuration
- ✅ Safe to use in `#![no_std]` environments

## Common Patterns

### Modulating Filter Cutoff

```rust
fn modulate_filter(
    base_cutoff: f32,
    lfo_value: f32,  // [-1.0, 1.0]
    mod_amount: f32, // 0.0 to 1.0
) -> f32 {
    // Map LFO to octaves
    let octave_shift = lfo_value * mod_amount * 4.0; // ±4 octaves
    base_cutoff * 2.0_f32.powf(octave_shift)
}
```

### Vibrato (Pitch Modulation)

```rust
fn apply_vibrato(
    base_pitch: f32,
    lfo_value: f32,   // [-1.0, 1.0]
    depth_cents: f32, // e.g., 50.0 for ±50 cents
) -> f32 {
    let semitone_shift = lfo_value * (depth_cents / 100.0);
    base_pitch * 2.0_f32.powf(semitone_shift / 12.0)
}
```

### Tremolo (Amplitude Modulation)

```rust
fn apply_tremolo(
    sample: f32,
    lfo_value: f32, // [0.0, 1.0] unipolar
    depth: f32,     // 0.0 to 1.0
) -> f32 {
    let gain = 1.0 - depth + (lfo_value * depth);
    sample * gain
}
```
