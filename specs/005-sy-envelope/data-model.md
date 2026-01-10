# Data Model: SY-Style Envelope

**Feature**: SY-Style Envelope Modulation Source
**Date**: 2026-01-10
**Location**: `rigel-modulation` crate (alongside LFO)

## Entity Overview

```text
+-------------------+       +-------------------+       +------------------+
|  EnvelopeConfig   |       |   EnvelopeState   |       |    Envelope      |
|  (Immutable)      |       |   (Runtime)       |       |  (Main Type)     |
+-------------------+       +-------------------+       +------------------+
| - rates[]         |       | - level           |       | - config         |
| - levels[]        |       | - target_level    |       | - state          |
| - rate_scaling    |       | - stage           |       +------------------+
| - output_level    |       | - phase           |       | + note_on()      |
| - delay_samples   |       | - increment       |       | + note_off()     |
| - loop_config     |       | - rising          |       | + process()      |
+-------------------+       | - delay_remaining |       | + value()        |
                            +-------------------+       +------------------+
```

---

## Core Entities

### 1. EnvelopePhase

Current operational phase of the envelope.

```rust
/// Envelope operational phases
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnvelopePhase {
    /// Envelope is idle (note off, fully released)
    Idle,
    /// Waiting for delay period to complete
    Delay,
    /// Processing key-on segments (attack/decay)
    KeyOn,
    /// Holding at sustain level (loop point or final key-on segment)
    Sustain,
    /// Processing release segments
    Release,
    /// Envelope completed (ready to be reused)
    Complete,
}
```

**State Transitions:**

```text
Idle ──note_on()──> Delay ──delay_complete──> KeyOn
                           (or if delay=0)

KeyOn ──segment_complete──> KeyOn (next segment)
      ──loop_triggered──> KeyOn (loop start)
      ──final_segment──> Sustain
      ──note_off()──> Release

Sustain ──note_off()──> Release

Release ──segment_complete──> Release (next segment)
        ──final_segment──> Complete

Complete ──note_on()──> Delay/KeyOn
         (auto-transitions to Idle)
```

---

### 2. Segment

Individual segment configuration within an envelope.

```rust
/// A single envelope segment
#[derive(Debug, Clone, Copy, Default)]
pub struct Segment {
    /// Rate parameter (0-99 DX7 convention)
    /// Higher = faster transition
    pub rate: u8,

    /// Target level parameter (0-99 DX7 convention)
    /// 99 = 0dB (full), 0 = ~-96dB (silent)
    pub level: u8,
}

impl Segment {
    /// Create a new segment
    pub const fn new(rate: u8, level: u8) -> Self {
        Self { rate, level }
    }

    /// Maximum rate (near-instantaneous)
    pub const MAX_RATE: u8 = 99;

    /// Maximum level (0dB)
    pub const MAX_LEVEL: u8 = 99;
}
```

---

### 3. LoopConfig

Configuration for segment looping during key-on phase.

```rust
/// Loop configuration for envelope segments
#[derive(Debug, Clone, Copy, Default)]
pub struct LoopConfig {
    /// Whether looping is enabled
    pub enabled: bool,

    /// Index of first segment in loop (0-based)
    /// Must be < end_segment and within key-on range
    pub start_segment: u8,

    /// Index of last segment in loop (0-based)
    /// After completing this segment, loop back to start_segment
    pub end_segment: u8,
}

impl LoopConfig {
    /// Create disabled loop config
    pub const fn disabled() -> Self {
        Self {
            enabled: false,
            start_segment: 0,
            end_segment: 0,
        }
    }

    /// Create enabled loop config
    /// Returns None if boundaries are invalid (start >= end)
    pub fn new(start: u8, end: u8) -> Option<Self> {
        if start < end {
            Some(Self {
                enabled: true,
                start_segment: start,
                end_segment: end,
            })
        } else {
            None
        }
    }

    /// Validate loop boundaries against segment count
    pub fn is_valid(&self, key_on_segments: usize) -> bool {
        !self.enabled
            || (self.start_segment < self.end_segment
                && (self.end_segment as usize) < key_on_segments)
    }
}
```

---

### 4. EnvelopeConfig

Immutable configuration for an envelope instance.

```rust
/// Envelope configuration (immutable after creation)
/// Uses const generics for segment count optimization
#[derive(Debug, Clone, Copy)]
pub struct EnvelopeConfig<const KEY_ON_SEGS: usize, const RELEASE_SEGS: usize> {
    /// Key-on segments (attack, decay, etc.)
    pub key_on_segments: [Segment; KEY_ON_SEGS],

    /// Release segments (key-off behavior)
    pub release_segments: [Segment; RELEASE_SEGS],

    /// Rate scaling sensitivity (0-7)
    /// Higher = more rate variation across keyboard
    pub rate_scaling: u8,

    /// Output level scaling (pre-computed from operator level)
    /// In internal units (~0.75dB per step)
    pub output_level: u8,

    /// Delay before attack begins (in samples)
    pub delay_samples: u32,

    /// Loop configuration for key-on segments
    pub loop_config: LoopConfig,

    /// Sample rate for timing calculations
    pub sample_rate: f32,
}

impl<const K: usize, const R: usize> EnvelopeConfig<K, R> {
    /// Total number of segments
    pub const TOTAL_SEGMENTS: usize = K + R;

    /// Create default configuration (all segments at max rate/level)
    pub fn default_with_sample_rate(sample_rate: f32) -> Self {
        Self {
            key_on_segments: [Segment::new(99, 99); K],
            release_segments: [Segment::new(50, 0); R],
            rate_scaling: 0,
            output_level: 127,  // Full output
            delay_samples: 0,
            loop_config: LoopConfig::disabled(),
            sample_rate,
        }
    }

    /// Create configuration from raw parameters
    pub fn new(
        key_on_segments: [Segment; K],
        release_segments: [Segment; R],
        rate_scaling: u8,
        output_level: u8,
        delay_samples: u32,
        loop_config: LoopConfig,
        sample_rate: f32,
    ) -> Self {
        Self {
            key_on_segments,
            release_segments,
            rate_scaling: rate_scaling.min(7),
            output_level,
            delay_samples,
            loop_config,
            sample_rate,
        }
    }
}

// Type aliases for common configurations
pub type FmEnvelopeConfig = EnvelopeConfig<6, 2>;      // 8-segment FM
pub type AwmEnvelopeConfig = EnvelopeConfig<5, 5>;     // 5+5 AWM
pub type SevenSegEnvelopeConfig = EnvelopeConfig<5, 2>; // 7-segment
```

---

### 5. EnvelopeState

Runtime state of an envelope (mutable during processing).

```rust
/// Envelope level type in Q8 fixed-point format (matches DX7 hardware)
/// Range: 0 to 4095 (12 bits used, ~96dB dynamic range)
/// Conversion to linear: 2^(level / 256.0)
pub type EnvelopeLevel = i16;

/// Maximum envelope level (0dB, full amplitude)
pub const LEVEL_MAX: i16 = 4095;

/// Minimum envelope level (silence, ~-96dB)
pub const LEVEL_MIN: i16 = 0;

/// Runtime state for envelope processing
#[derive(Debug, Clone, Copy)]
pub struct EnvelopeState {
    /// Current level in Q8 fixed-point format (0-4095)
    /// Represents log2 amplitude (256 units = 6dB)
    level: i16,

    /// Target level for current segment (Q8)
    target_level: i16,

    /// Level change per sample (Q8, signed)
    /// Positive for rising, negative for falling
    increment: i16,

    /// Current segment index
    /// 0..(K-1) for key-on, K..(K+R-1) for release
    segment_index: u8,

    /// Current envelope phase
    phase: EnvelopePhase,

    /// True if level is rising toward target
    rising: bool,

    /// Remaining delay samples (counts down to 0)
    delay_remaining: u32,

    /// Scaled qRate for current segment (with rate scaling applied)
    current_qrate: u8,

    /// MIDI note for rate scaling (cached)
    midi_note: u8,
}

impl Default for EnvelopeState {
    fn default() -> Self {
        Self {
            level: 0,
            target_level: 0,
            increment: 0,
            segment_index: 0,
            phase: EnvelopePhase::Idle,
            rising: false,
            delay_remaining: 0,
            current_qrate: 0,
            midi_note: 60,
        }
    }
}

impl EnvelopeState {
    /// Get current level in Q8 format (0-4095)
    #[inline]
    pub fn level_q8(&self) -> i16 {
        self.level
    }

    /// Get current linear amplitude (0.0 to 1.0)
    #[inline]
    pub fn level_linear(&self) -> f32 {
        level_to_linear(self.level)
    }

    /// Get current phase
    #[inline]
    pub fn phase(&self) -> EnvelopePhase {
        self.phase
    }

    /// Check if envelope is active (not idle or complete)
    #[inline]
    pub fn is_active(&self) -> bool {
        !matches!(self.phase, EnvelopePhase::Idle | EnvelopePhase::Complete)
    }

    /// Check if envelope is in release phase
    #[inline]
    pub fn is_releasing(&self) -> bool {
        matches!(self.phase, EnvelopePhase::Release)
    }
}
```

---

### 6. Envelope (Main Type)

The complete envelope generator combining config and state.

```rust
/// SY-style multi-segment envelope generator
///
/// Operates in logarithmic (dB) domain internally, outputs linear amplitude.
/// Implements MSFA-compatible rate calculations and attack behavior.
///
/// # Type Parameters
/// * `KEY_ON_SEGS` - Number of key-on segments (attack/decay)
/// * `RELEASE_SEGS` - Number of release segments
///
/// # Example
/// ```ignore
/// use rigel_modulation::envelope::{Envelope, FmEnvelope, Segment};
///
/// // Create 8-segment FM envelope
/// let mut env = FmEnvelope::new(44100.0);
/// env.note_on(60);  // Middle C
///
/// // Process samples
/// for _ in 0..1024 {
///     let amplitude = env.process();
///     // Use amplitude for audio...
/// }
///
/// env.note_off();
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Envelope<const KEY_ON_SEGS: usize, const RELEASE_SEGS: usize> {
    /// Immutable configuration
    config: EnvelopeConfig<KEY_ON_SEGS, RELEASE_SEGS>,

    /// Mutable runtime state
    state: EnvelopeState,
}

// Type aliases
pub type FmEnvelope = Envelope<6, 2>;       // Standard 8-segment FM
pub type AwmEnvelope = Envelope<5, 5>;      // 5+5 AWM-style
pub type SevenSegEnvelope = Envelope<5, 2>; // 7-segment

impl<const K: usize, const R: usize> Envelope<K, R> {
    /// Create new envelope with default configuration
    pub fn new(sample_rate: f32) -> Self {
        Self {
            config: EnvelopeConfig::default_with_sample_rate(sample_rate),
            state: EnvelopeState::default(),
        }
    }

    /// Create envelope with specific configuration
    pub fn with_config(config: EnvelopeConfig<K, R>) -> Self {
        Self {
            config,
            state: EnvelopeState::default(),
        }
    }

    /// Trigger note-on event
    pub fn note_on(&mut self, midi_note: u8);

    /// Trigger note-off event
    pub fn note_off(&mut self);

    /// Process one sample and return linear amplitude (0.0 to 1.0)
    pub fn process(&mut self) -> f32;

    /// Get current linear amplitude without advancing state
    pub fn value(&self) -> f32;

    /// Reset to idle state
    pub fn reset(&mut self);

    /// Update configuration (takes effect on next note-on)
    pub fn set_config(&mut self, config: EnvelopeConfig<K, R>);

    /// Get current configuration
    pub fn config(&self) -> &EnvelopeConfig<K, R>;

    /// Get current state (read-only)
    pub fn state(&self) -> &EnvelopeState;
}
```

---

### 7. EnvelopeBatch (SIMD Processing)

Batch processor for multiple envelopes.

```rust
/// SIMD-accelerated batch envelope processor
///
/// Processes N envelopes in parallel using SIMD instructions.
/// N should match SIMD lane count (4/8/16 depending on platform).
///
/// # Example
/// ```ignore
/// use rigel_modulation::envelope::{EnvelopeBatch, FmEnvelope};
///
/// // Create batch of 8 envelopes (AVX2 optimal)
/// let mut batch = EnvelopeBatch::<8, 6, 2>::new(44100.0);
///
/// // Trigger all envelopes
/// for i in 0..8 {
///     batch.note_on(i, 60 + i as u8);
/// }
///
/// // Process samples
/// let mut output = [0.0f32; 8];
/// for _ in 0..1024 {
///     batch.process(&mut output);
///     // output[0..8] contains linear amplitudes
/// }
/// ```
#[derive(Debug, Clone)]
pub struct EnvelopeBatch<const N: usize, const K: usize, const R: usize> {
    /// Individual envelope instances
    envelopes: [Envelope<K, R>; N],
}

impl<const N: usize, const K: usize, const R: usize> EnvelopeBatch<N, K, R> {
    /// Create batch with default configurations
    pub fn new(sample_rate: f32) -> Self;

    /// Create batch with specific configuration for all envelopes
    pub fn with_config(config: EnvelopeConfig<K, R>) -> Self;

    /// Trigger note-on for envelope at index
    pub fn note_on(&mut self, index: usize, midi_note: u8);

    /// Trigger note-off for envelope at index
    pub fn note_off(&mut self, index: usize);

    /// Process one sample for all envelopes (SIMD accelerated)
    /// Writes N linear amplitude values to output slice
    pub fn process(&mut self, output: &mut [f32; N]);

    /// Process block of samples for all envelopes
    /// output[i][j] = envelope i, sample j
    pub fn process_block(&mut self, block_size: usize, output: &mut [[f32; N]]);
}
```

---

## Conversion Functions

```rust
/// Level lookup table for output levels 0-19
pub const LEVEL_LUT: [u8; 20] = [
    0, 5, 9, 13, 17, 20, 23, 25, 27, 29,
    31, 33, 35, 37, 39, 41, 42, 43, 45, 46
];

/// Scale output level (0-99) to internal representation
#[inline]
pub fn scale_output_level(outlevel: u8) -> u8 {
    if outlevel >= 20 {
        28 + outlevel
    } else {
        LEVEL_LUT[outlevel as usize]
    }
}

/// Convert DX7 rate (0-99) to internal qRate (0-63)
#[inline]
pub fn rate_to_qrate(rate: u8) -> u8 {
    ((rate as u32 * 41) >> 6) as u8
}

/// Calculate rate scaling adjustment
#[inline]
pub fn scale_rate(midi_note: u8, sensitivity: u8) -> u8 {
    let x = ((midi_note as i32 / 3) - 7).clamp(0, 31) as u8;
    ((sensitivity as u32 * x as u32) >> 3) as u8
}

/// Calculate increment from qRate in Q8 format
/// Scaled for per-sample processing with i16 level representation
#[inline]
pub fn calculate_increment_q8(qrate: u8) -> i16 {
    // Base increment calculation (same as MSFA)
    // Then scale down from Q24/block to Q8/sample
    let base_inc = (4 + (qrate as i32 & 3)) << (2 + (qrate as i32 >> 2));
    // Scale for Q8 per-sample operation
    (base_inc >> 8) as i16
}

/// Convert Q8 level (0-4095) to linear amplitude (0.0 to 1.0)
/// Uses fast exp2 approximation from rigel-math
#[inline]
pub fn level_to_linear(level_q8: i16) -> f32 {
    // Q8 format: 256 steps = 6dB (one octave)
    // linear = 2^(level / 256)
    let log2_gain = (level_q8 as f32) / 256.0;
    rigel_math::simd::fast_exp2(log2_gain)
}

/// Convert linear amplitude (0.0 to 1.0) to Q8 level
#[inline]
pub fn linear_to_level(linear: f32) -> i16 {
    // level = log2(linear) * 256
    let log2_val = rigel_math::simd::fast_log2(linear);
    (log2_val * 256.0) as i16
}

/// Attack jump threshold (Q8 format, ~40dB above minimum)
pub const JUMP_TARGET_Q8: i16 = 1716;
```

---

## Validation Rules

### Segment Parameters
- `rate`: 0-99, clamped if out of range
- `level`: 0-99, clamped if out of range

### Rate Scaling
- `sensitivity`: 0-7, clamped to 7 if higher

### Loop Configuration
- `start_segment` must be < `end_segment`
- `end_segment` must be < `KEY_ON_SEGS`
- Invalid loops are treated as disabled

### State Invariants
- `level` is always in valid Q8 range (0-4095)
- `phase` transitions follow defined state machine
- `segment_index` is always valid for current phase

---

## Memory Layout

Uses i16/Q8 fixed-point format (hardware-authentic, 33% smaller than Q24).

```text
EnvelopeConfig<6, 2> (typical FM envelope):
  - key_on_segments: [Segment; 6] = 12 bytes
  - release_segments: [Segment; 2] = 4 bytes
  - rate_scaling: u8 = 1 byte
  - output_level: u8 = 1 byte
  - delay_samples: u32 = 4 bytes
  - loop_config: LoopConfig = 3 bytes
  - sample_rate: f32 = 4 bytes
  - [padding]: 3 bytes
  Total: 32 bytes

EnvelopeState (Q8 fixed-point):
  - level: i16 = 2 bytes
  - target_level: i16 = 2 bytes
  - increment: i16 = 2 bytes
  - segment_index: u8 = 1 byte
  - phase: EnvelopePhase = 1 byte
  - rising: bool = 1 byte
  - delay_remaining: u32 = 4 bytes
  - current_qrate: u8 = 1 byte
  - midi_note: u8 = 1 byte
  - [padding]: 1 byte
  Total: 16 bytes

Envelope<6, 2>:
  - config: 32 bytes
  - state: 16 bytes
  Total: 48 bytes (well under 128 byte target)

1536 envelopes: 48 × 1536 = 73.7 KB (fits comfortably in L2 cache)
```
