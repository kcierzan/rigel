# Quickstart: SY-Style Envelope

**Feature**: SY-Style Envelope Modulation Source
**Date**: 2026-01-10

## Basic Usage

### Create and Use an FM Envelope (8-segment)

```rust
use rigel_modulation::envelope::{FmEnvelope, Segment, EnvelopeConfig, LoopConfig};

fn main() {
    // Create envelope with default settings
    let mut env = FmEnvelope::new(44100.0);

    // Trigger note-on (middle C)
    env.note_on(60);

    // Process audio samples
    for i in 0..44100 {
        let amplitude = env.process();
        // amplitude is in range [0.0, 1.0]
        // Multiply with oscillator output for gain control
    }

    // Release the note
    env.note_off();

    // Continue processing release phase
    while env.is_active() {
        let amplitude = env.process();
        // ... use amplitude
    }
}
```

### Configure Envelope Segments

```rust
use rigel_modulation::envelope::{FmEnvelope, Segment, EnvelopeConfig, LoopConfig};

fn create_punchy_bass_envelope(sample_rate: f32) -> FmEnvelope {
    let config = EnvelopeConfig {
        key_on_segments: [
            Segment::new(99, 99),  // Attack: instant to full
            Segment::new(80, 70),  // Decay 1: fast to 70%
            Segment::new(60, 50),  // Decay 2: medium to 50%
            Segment::new(40, 50),  // Sustain prep: hold at 50%
            Segment::new(40, 50),  // Sustain: hold
            Segment::new(40, 50),  // Sustain: hold
        ],
        release_segments: [
            Segment::new(50, 20),  // Release 1: medium to 20%
            Segment::new(30, 0),   // Release 2: slower to silence
        ],
        rate_scaling: 3,           // Moderate keyboard tracking
        output_level: 127,         // Full output
        delay_samples: 0,          // No delay
        loop_config: LoopConfig::disabled(),
        sample_rate,
    };

    FmEnvelope::with_config(config)
}
```

### Use Envelope with Oscillator

```rust
use rigel_dsp::{SynthEngine, SimpleOscillator};
use rigel_modulation::envelope::FmEnvelope;

fn process_voice(
    osc: &mut SimpleOscillator,
    env: &mut FmEnvelope,
    output: &mut [f32],
) {
    for sample in output.iter_mut() {
        let osc_sample = osc.process_sample();
        let env_value = env.process();
        *sample = osc_sample * env_value;
    }
}
```

## Advanced Features

### Delayed Envelope Start

```rust
use rigel_modulation::envelope::{FmEnvelope, EnvelopeConfig, Segment, LoopConfig};

fn create_delayed_pad_envelope(sample_rate: f32) -> FmEnvelope {
    // Calculate 500ms delay in samples
    let delay_samples = (0.5 * sample_rate) as u32;

    let config = EnvelopeConfig {
        key_on_segments: [
            Segment::new(40, 99),  // Slow attack
            Segment::new(50, 80),
            Segment::new(60, 70),
            Segment::new(40, 70),
            Segment::new(40, 70),
            Segment::new(40, 70),
        ],
        release_segments: [
            Segment::new(30, 30),
            Segment::new(20, 0),   // Slow fade
        ],
        rate_scaling: 0,
        output_level: 127,
        delay_samples,             // 500ms delay before attack
        loop_config: LoopConfig::disabled(),
        sample_rate,
    };

    FmEnvelope::with_config(config)
}
```

### Looping Envelope (for Rhythmic Effects)

```rust
use rigel_modulation::envelope::{FmEnvelope, EnvelopeConfig, Segment, LoopConfig};

fn create_rhythmic_envelope(sample_rate: f32) -> FmEnvelope {
    let config = EnvelopeConfig {
        key_on_segments: [
            Segment::new(99, 99),  // 0: Instant attack
            Segment::new(70, 50),  // 1: Quick decay to 50%
            Segment::new(80, 99),  // 2: Quick rise to full (loop start)
            Segment::new(70, 30),  // 3: Decay to 30%
            Segment::new(80, 99),  // 4: Rise again (loop end)
            Segment::new(50, 60),  // 5: Final sustain
        ],
        release_segments: [
            Segment::new(50, 20),
            Segment::new(30, 0),
        ],
        rate_scaling: 0,
        output_level: 127,
        delay_samples: 0,
        loop_config: LoopConfig::new(2, 4).unwrap(), // Loop between segments 2-4
        sample_rate,
    };

    FmEnvelope::with_config(config)
}
```

### Rate Scaling by Key Position

```rust
use rigel_modulation::envelope::{FmEnvelope, EnvelopeConfig, Segment, LoopConfig};

fn create_keyboard_tracked_envelope(sample_rate: f32) -> FmEnvelope {
    let config = EnvelopeConfig {
        key_on_segments: [
            Segment::new(90, 99),
            Segment::new(70, 80),
            Segment::new(50, 70),
            Segment::new(40, 70),
            Segment::new(40, 70),
            Segment::new(40, 70),
        ],
        release_segments: [
            Segment::new(50, 20),
            Segment::new(30, 0),
        ],
        rate_scaling: 7,           // Maximum keyboard tracking
        output_level: 127,
        delay_samples: 0,
        loop_config: LoopConfig::disabled(),
        sample_rate,
    };

    FmEnvelope::with_config(config)
}

// High notes will have faster envelopes:
// - C1 (MIDI 36): Normal rate
// - C4 (MIDI 60): ~1.5x faster
// - C7 (MIDI 96): ~2.5x faster
```

## Batch Processing (SIMD)

### Process Multiple Voices in Parallel

```rust
use rigel_modulation::envelope::{EnvelopeBatch, FmEnvelope};

fn process_polyphonic(
    batch: &mut EnvelopeBatch<8, 6, 2>,  // 8 voices, FM envelope
    oscillators: &mut [f32; 8],          // Oscillator outputs
    output: &mut [f32; 8],               // Final output
) {
    // Process all 8 envelopes in one SIMD operation
    let mut env_values = [0.0f32; 8];
    batch.process(&mut env_values);

    // Apply envelope to each voice
    for i in 0..8 {
        output[i] = oscillators[i] * env_values[i];
    }
}

fn setup_polyphonic_envelopes(sample_rate: f32) -> EnvelopeBatch<8, 6, 2> {
    let mut batch = EnvelopeBatch::new(sample_rate);

    // Configure each envelope for different notes
    batch.note_on(0, 48);  // Voice 0: C3
    batch.note_on(1, 55);  // Voice 1: G3
    batch.note_on(2, 60);  // Voice 2: C4
    // ... etc

    batch
}
```

### Block Processing for Efficiency

```rust
use rigel_modulation::envelope::FmEnvelope;

fn process_block_optimized(
    env: &mut FmEnvelope,
    output: &mut [f32],
) {
    // More efficient than calling process() in a loop
    env.process_block(output);
}

// Or with batch:
use rigel_modulation::envelope::EnvelopeBatch;

fn process_voices_block(
    batch: &mut EnvelopeBatch<8, 6, 2>,
    block_size: usize,
) -> [[f32; 8]; 64] {
    let mut output = [[0.0f32; 8]; 64];
    batch.process_block(&mut output[..block_size]);
    output
}
```

## State Inspection

### Query Envelope State

```rust
use rigel_modulation::envelope::{FmEnvelope, EnvelopePhase};

fn inspect_envelope(env: &FmEnvelope) {
    // Check phase
    match env.phase() {
        EnvelopePhase::Idle => println!("Not triggered"),
        EnvelopePhase::Delay => println!("Waiting to start"),
        EnvelopePhase::KeyOn => println!("In attack/decay"),
        EnvelopePhase::Sustain => println!("Sustaining"),
        EnvelopePhase::Release => println!("Releasing"),
        EnvelopePhase::Complete => println!("Finished"),
    }

    // Get current value without advancing
    let current_level = env.value();
    println!("Current amplitude: {:.3}", current_level);

    // Check if envelope is still sounding
    if env.is_active() {
        println!("Envelope is active");
    }

    // Check segment
    println!("Current segment: {}", env.current_segment());
}
```

## Variant Envelopes

### AWM-Style (5+5 Segments)

```rust
use rigel_modulation::envelope::{AwmEnvelope, EnvelopeConfig, Segment, LoopConfig};

fn create_awm_envelope(sample_rate: f32) -> AwmEnvelope {
    let config = EnvelopeConfig {
        key_on_segments: [
            Segment::new(90, 99),  // Attack
            Segment::new(70, 85),  // Decay 1
            Segment::new(60, 75),  // Decay 2
            Segment::new(50, 70),  // Decay 3
            Segment::new(40, 70),  // Sustain
        ],
        release_segments: [
            Segment::new(60, 50),  // Release 1
            Segment::new(50, 30),  // Release 2
            Segment::new(40, 15),  // Release 3
            Segment::new(30, 5),   // Release 4
            Segment::new(20, 0),   // Release 5 (to silence)
        ],
        rate_scaling: 2,
        output_level: 127,
        delay_samples: 0,
        loop_config: LoopConfig::disabled(),
        sample_rate,
    };

    AwmEnvelope::with_config(config)
}
```

### 7-Segment Envelope

```rust
use rigel_modulation::envelope::{SevenSegEnvelope, EnvelopeConfig, Segment, LoopConfig};

fn create_seven_seg_envelope(sample_rate: f32) -> SevenSegEnvelope {
    let config = EnvelopeConfig {
        key_on_segments: [
            Segment::new(99, 99),  // Attack
            Segment::new(80, 80),  // Decay 1
            Segment::new(60, 70),  // Decay 2
            Segment::new(50, 65),  // Decay 3
            Segment::new(40, 65),  // Sustain
        ],
        release_segments: [
            Segment::new(50, 20),  // Release 1
            Segment::new(30, 0),   // Release 2
        ],
        rate_scaling: 0,
        output_level: 127,
        delay_samples: 0,
        loop_config: LoopConfig::disabled(),
        sample_rate,
    };

    SevenSegEnvelope::with_config(config)
}
```

## Integration with ModulationSource Trait

```rust
use rigel_modulation::envelope::{FmEnvelope, ModulationSource};

fn apply_modulation<M: ModulationSource>(
    modulator: &mut M,
    base_value: f32,
    mod_amount: f32,
) -> f32 {
    let mod_value = modulator.tick();  // Advances modulator
    base_value + (mod_value * mod_amount)
}

fn modulate_filter_cutoff(
    env: &mut FmEnvelope,
    base_cutoff: f32,
    env_amount: f32,
) -> f32 {
    // Envelope modulates filter cutoff
    apply_modulation(env, base_cutoff, env_amount)
}
```

## Performance Tips

1. **Use batch processing** for polyphonic voices (6-7x faster with AVX2)
2. **Process in blocks** rather than sample-by-sample when possible
3. **Avoid checking `is_active()` every sample** - do it once per block
4. **Pre-configure envelopes** before note-on to avoid runtime calculations
5. **Use `value()` for UI** instead of `process()` to avoid state changes

## Internal Format: i16/Q8 Fixed-Point

The envelope uses i16/Q8 fixed-point internally (matching original DX7 hardware):

### Memory Efficiency

| Metric | Value |
|--------|-------|
| Level storage | i16 (2 bytes) |
| EnvelopeState | 16 bytes |
| Envelope<6,2> | 48 bytes |
| 1536 envelopes | 73.7 KB (fits in L2 cache) |

### Format Details

- **Q8 format**: 256 steps = 6dB (one amplitude doubling)
- **Resolution**: 0.0234 dB per step (imperceptible)
- **Dynamic range**: ~96dB (4096 steps)
- **Conversion**: Uses `rigel_math::fast_exp2` (SIMD-accelerated)

### Why This Matters

1. **33% smaller** than Q24/i32 format used by Dexed
2. **Hardware-authentic**: Matches DX7 EGS→OPS chip exactly
3. **Better cache utilization**: More envelopes fit in L1 cache
4. **SIMD-friendly**: Process 16 i16 values vs 8 i32 in AVX2 register

### Performance Targets

- **Single envelope**: <50ns per sample
- **1536 envelopes × 64 samples**: <100µs
- **SIMD batch**: 2x+ speedup over scalar
