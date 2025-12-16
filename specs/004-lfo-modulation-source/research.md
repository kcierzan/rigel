# Research: LFO Modulation Source

**Date**: 2025-12-14
**Feature**: 004-lfo-modulation-source

## Research Questions

1. What no_std PRNG is suitable for S&H and noise waveshapes?
2. Does rigel-math have fast sine approximations for LFO use?
3. How should the ModulationSource trait migration work?

---

## 1. Random Number Generation for Audio LFOs

### Decision: Use inline PCG32 implementation

### Rationale

- **PCG (Permuted Congruential Generator)** offers better statistical quality than xorshift for audio applications
- Lower bits of xorshift exhibit non-random patterns; PCG distributes randomness evenly across all bits
- Only ~5-10% slower than xorshift (imperceptible in real-time audio)
- No external dependencies required - can be implemented inline

### Implementation

```rust
/// PCG32 PRNG optimized for audio synthesis
#[derive(Copy, Clone, Debug)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub const fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x2C9277B5_27D4EB2D_u64),
        }
    }

    #[inline(always)]
    pub fn next_u32(&mut self) -> u32 {
        let old_state = self.state;
        self.state = old_state
            .wrapping_mul(6364136223846793005_u64)
            .wrapping_add(1442695040888963407_u64);

        let xor_shifted = (((old_state >> 18) ^ old_state) >> 27) as u32;
        let rot = (old_state >> 59) as u32;
        xor_shifted.rotate_right(rot)
    }

    /// Generate random f32 in [-1.0, 1.0]
    #[inline(always)]
    pub fn next_f32(&mut self) -> f32 {
        let value = self.next_u32();
        // Use high 24 bits for better distribution
        let normalized = ((value >> 8) as f32) / 16777215.0;
        normalized * 2.0 - 1.0
    }
}
```

### Alternatives Considered

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| Existing xorshift (rigel-math) | Already in codebase, 4-byte state | Lower quality in lower bits | Not for S&H/noise |
| oorandom crate | Battle-tested, zero dependencies | External dependency | Acceptable fallback |
| rand_pcg crate | Multiple PCG variants | 2 dependencies | Overkill for this use |
| **PCG inline** | Zero deps, excellent quality | 8-byte state vs 4 | **CHOSEN** |

---

## 2. Fast Sine for LFO Waveshape

### Decision: Use rigel-math `sin()` function

### Rationale

The rigel-math crate already provides production-ready fast sine:

- **Location**: `rigel-math/src/math/trig.rs`
- **Algorithm**: Cody-Waite FMA range reduction + 7th-order minimax polynomial
- **Accuracy**: <0.01 absolute error (specifically validated for LFO use)
- **Performance (AVX2)**: 2.8-5.4x faster than libm::sinf

### LFO Test Coverage

The crate already includes LFO-specific accuracy tests:

```rust
#[test]
fn accuracy_sin_lfo_range() {
    // Test sin() for LFO: [0, 2*pi]
    let test_values: Vec<f32> = (0..=100)
        .map(|i| (i as f32) * core::f32::consts::TAU / 100.0)
        .collect();
    let max_error = max_absolute_error(&test_values, |x| sin(x), |x| libm::sinf(x));
    assert!(max_error < 0.01, "sin LFO range error: {:.6}", max_error);
}
```

### Usage Pattern

```rust
use rigel_math::math::sin;

fn sine_waveshape(phase: f32) -> f32 {
    // phase is in [0.0, 1.0), convert to radians
    sin(phase * core::f32::consts::TAU)
}
```

### Alternatives Considered

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| libm::sinf | Known accuracy | 2.8-5.4x slower | Not for hot path |
| Lookup table | Very fast | Memory overhead, interpolation | Unnecessary complexity |
| **rigel-math::sin** | Already optimized, LFO-tested | Slight approx error | **CHOSEN** |

---

## 3. ModulationSource Trait Migration

### Decision: Move trait to rigel-modulation, re-export from rigel-timing

### Rationale

1. The trait logically belongs with its implementations (LFO, future envelopes)
2. `rigel-timing` becomes purely about timing infrastructure
3. Backward compatibility via re-export prevents breaking changes

### Migration Strategy

**Step 1**: Create rigel-modulation with the trait

```rust
// rigel-modulation/src/traits.rs
use rigel_timing::Timebase;

pub trait ModulationSource {
    fn reset(&mut self);
    fn update(&mut self, timebase: &Timebase);
    fn value(&self) -> f32;
}
```

**Step 2**: Update rigel-timing to re-export

```rust
// rigel-timing/src/lib.rs
// Re-export for backward compatibility
pub use rigel_modulation::ModulationSource;
```

**Step 3**: Update Cargo.toml dependencies

```toml
# rigel-timing/Cargo.toml
[dependencies]
rigel-modulation = { path = "../modulation" }

# Re-export in lib.rs
```

### Dependency Graph (After Migration)

```
rigel-math ─────────────────────────────────┐
     │                                       │
     v                                       v
rigel-timing ──────────> rigel-modulation ──┤
     │                         │             │
     v                         v             │
rigel-dsp <───────────────────┘              │
     │                                       │
     v                                       │
rigel-plugin <───────────────────────────────┘
```

### Alternatives Considered

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| Keep trait in timing | No migration work | Wrong abstraction home | Rejected |
| Duplicate trait | No dependency change | Code duplication, drift risk | Rejected |
| **Move + re-export** | Clean separation, backward compat | Slight complexity | **CHOSEN** |

---

## 4. Tempo Sync Implementation

### Decision: Use note division multipliers with tempo-to-Hz conversion

### Rate Calculation Formula

```rust
fn tempo_to_hz(bpm: f32, division: NoteDivision) -> f32 {
    let beats_per_second = bpm / 60.0;
    beats_per_second * division.multiplier()
}

impl NoteDivision {
    pub fn multiplier(&self) -> f32 {
        let base = match self.base {
            NoteBase::Whole => 0.25,
            NoteBase::Half => 0.5,
            NoteBase::Quarter => 1.0,
            NoteBase::Eighth => 2.0,
            NoteBase::Sixteenth => 4.0,
            NoteBase::ThirtySecond => 8.0,
        };

        match self.modifier {
            NoteModifier::Normal => base,
            NoteModifier::Dotted => base * (2.0 / 3.0),  // 1.5x duration = 2/3 rate
            NoteModifier::Triplet => base * 1.5,         // 2/3 duration = 1.5x rate
        }
    }
}
```

### Example Calculations (at 120 BPM)

| Division | Multiplier | Rate (Hz) |
|----------|------------|-----------|
| 1/1 (Whole) | 0.25 | 0.5 Hz |
| 1/2 (Half) | 0.5 | 1.0 Hz |
| 1/4 (Quarter) | 1.0 | 2.0 Hz |
| 1/4 Dotted | 0.667 | 1.33 Hz |
| 1/4 Triplet | 1.5 | 3.0 Hz |
| 1/8 (Eighth) | 2.0 | 4.0 Hz |
| 1/16 (Sixteenth) | 4.0 | 8.0 Hz |

---

## 5. Control Rate Integration

### Decision: Elapsed-sample-based phase calculation using Timebase

### Rationale

The LFO must work with any `ControlRateClock` interval (1, 8, 16, 32, 64, 128 samples). Rather than assuming a fixed interval, the LFO calculates phase increment based on actual elapsed samples from the Timebase.

### Implementation

```rust
impl Lfo {
    pub fn update(&mut self, timebase: &Timebase) {
        let elapsed_samples = timebase.block_size() as f32;
        let rate_hz = self.effective_rate_hz();

        // Phase increment based on elapsed time
        let phase_increment = rate_hz * elapsed_samples / timebase.sample_rate();

        let old_phase = self.phase;
        self.phase = (self.phase + phase_increment).fract();

        // Detect cycle wrap for S&H
        let wrapped = self.phase < old_phase;

        // Generate new value based on waveshape
        self.current_value = self.generate_value(wrapped);
    }

    pub fn value(&self) -> f32 {
        // Return cached value - no computation
        self.current_value
    }
}
```

### Integration with ControlRateClock

```rust
// In audio callback
fn process_block(
    lfo: &mut Lfo,
    timebase: &mut Timebase,
    clock: &mut ControlRateClock,
    block_size: u32,
) {
    timebase.advance_block(block_size);

    for offset in clock.advance(block_size) {
        // Update LFO at control rate intervals
        lfo.update(timebase);
    }

    // lfo.value() can be called per-sample cheaply
    let mod_value = lfo.value();
}
```

### Validation

At 44100 Hz, 1 Hz LFO, 64-sample control rate:
- Phase increment per update: 1.0 / 44100.0 * 64 ≈ 0.00145
- Updates per cycle: 44100 / 64 ≈ 689
- Total phase per cycle: 0.00145 * 689 ≈ 0.999 ✓

---

## Summary

All research questions resolved. No NEEDS CLARIFICATION markers remain.

| Question | Decision | Confidence |
|----------|----------|------------|
| PRNG for S&H/Noise | PCG32 inline implementation | High |
| Fast sine function | rigel-math::sin() | High |
| Trait migration | Move + re-export pattern | High |
| Tempo sync | Note division multipliers | High |
| Control rate integration | Elapsed-sample-based calculation | High |
