# Contracts: LFO Modulation Source

This directory contains the API contracts (signatures and documentation) for the `rigel-modulation` crate.

## Files

| File | Description |
|------|-------------|
| `traits.rs` | `ModulationSource` trait definition |
| `lfo.rs` | `Lfo` struct API and methods |
| `types.rs` | Supporting types (enums, NoteDivision, Rng) |

## Usage

These files define the **public API surface** of the crate. Implementation must conform to these signatures.

The actual implementation will be in:
- `projects/rigel-synth/crates/modulation/src/`

## Trait Hierarchy

```
ModulationSource (trait)
    └── Lfo (implements)
```

## Type Summary

```rust
// Primary type
pub struct Lfo { ... }

// Configuration enums
pub enum LfoWaveshape { Sine, Triangle, Saw, Square, Pulse, SampleAndHold, Noise }
pub enum LfoPhaseMode { FreeRunning, Retrigger }
pub enum LfoPolarity { Bipolar, Unipolar }
pub enum LfoRateMode { Hz(f32), TempoSync { division, bpm } }

// Supporting types
pub struct NoteDivision { base: NoteBase, modifier: NoteModifier }
pub enum NoteBase { Whole, Half, Quarter, Eighth, Sixteenth, ThirtySecond }
pub enum NoteModifier { Normal, Dotted, Triplet }
pub struct Rng { state: u64 }
```

## Real-Time Guarantees

All types in this crate provide:
- `Copy` + `Clone` semantics
- Zero heap allocations
- Constant-time operations
- `Send` + `Sync` thread safety
