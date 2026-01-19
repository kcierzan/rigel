# Feature Specification: SY-Style Envelope Modulation Source

**Feature Branch**: `005-sy-envelope`
**Created**: 2026-01-10
**Status**: Complete
**Completed**: 2026-01-19
**Input**: User description: "Implement a Yamaha SY-style envelope modulation source that meets the requirements described in the Linear NEW-5 task. The envelope should closely follow the functionality in MSFA and should exhibit the same rate scaling and instantaneous dB jump during initial attack. The implementation should serve as the basis for variants like the full 7-segment envelope as well as the 5+5 envelope in the AWM section. We should use rigel-math and SIMD functionality to make these envelopes as performant as possible as there will be at least 12 per voice x polyphony x unison. They should be fully tested and benchmarked."

## Clarifications

### Session 2026-01-10

- Q: What is the expected output domain and range for the envelope? → A: Linear amplitude (0.0 to 1.0) after internal dB→linear conversion
- Q: What should happen to the envelope level when a new note-on occurs while the envelope is still active? → A: Start from current level; attack transitions from current position toward target
- Q: How should envelope segment variants be implemented? → A: Compile-time generics (e.g., `Envelope<6, 2>` for 6 key-on + 2 release segments)
- Q: Should the envelope support level scaling by key position? → A: No, out of scope; apply externally via modulation routing if needed
- Q: How should SIMD acceleration be applied to envelope processing? → A: Batch multiple envelopes together (4-8 voices per SIMD operation)

## Overview

This specification defines a Yamaha SY99-style multi-segment envelope generator that operates in the logarithmic (dB) domain internally, outputting linear amplitude values (0.0 to 1.0) for direct signal multiplication. The envelope mimics the behavior of the original DX7/SY99 hardware as implemented in the MSFA (Music Synthesizer for Android) engine, including nonlinear rate calculations, distance-dependent timing, and instantaneous dB jumps during attack phases.

The implementation will serve as the foundation for multiple envelope variants:
- **8-segment FM envelope**: 6 key-on segments + 2 release segments (primary)
- **7-segment envelope**: Standard variant for general use
- **5+5 AWM envelope**: 5 key-on + 5 key-off segments for sample-based synthesis

**Out of Scope**: Level scaling by key position (adjusting envelope amplitude based on MIDI note) is not included in this envelope module; such scaling should be applied externally via the modulation routing system.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Basic Envelope Modulation (Priority: P1)

A synthesizer voice needs amplitude control that responds naturally to note events, providing the characteristic punchy attack and smooth decay of FM synthesis.

**Why this priority**: Without basic envelope functionality, no sound can be generated. This is the foundational capability that all other features depend upon.

**Independent Test**: Can be fully tested by triggering a single note and measuring the envelope output against expected dB values at key time points. Delivers basic amplitude control for any oscillator.

**Acceptance Scenarios**:

1. **Given** an envelope with default parameters, **When** a note-on event occurs, **Then** the envelope transitions through attack segments toward peak level within the expected time
2. **Given** an envelope at sustain level, **When** a note-off event occurs, **Then** the envelope transitions through release segments toward zero (silence)
3. **Given** an envelope in any segment, **When** the output value is sampled, **Then** it returns a linear amplitude value in the range [0.0, 1.0] suitable for direct signal multiplication

---

### User Story 2 - Rate Scaling by Key Position (Priority: P1)

A sound designer wants higher notes to have faster envelopes (like acoustic instruments where shorter strings vibrate faster), so envelope rates should automatically adjust based on the played note.

**Why this priority**: Rate scaling is essential for playable patches that sound natural across the keyboard. Without it, bass notes sound too short and high notes drag.

**Independent Test**: Can be tested by triggering the same patch at different MIDI notes and verifying that envelope timing scales appropriately.

**Acceptance Scenarios**:

1. **Given** rate scaling is enabled, **When** a note is played at MIDI note 60 (middle C), **Then** envelope rates match the base timing
2. **Given** rate scaling is enabled, **When** a note is played two octaves higher (MIDI 84), **Then** envelope rates are faster than at middle C
3. **Given** rate scaling is disabled (set to 0), **When** notes are played at different pitches, **Then** envelope timing remains constant regardless of key position

---

### User Story 3 - MSFA-Compatible Rate Behavior (Priority: P1)

A developer wants to accurately recreate classic DX7/SY99 sounds, so the envelope must exhibit the same nonlinear rate behavior and distance-dependent timing as the original hardware.

**Why this priority**: Accurate MSFA compatibility ensures that classic FM patches translate correctly, which is a primary design goal for the synthesizer.

**Independent Test**: Can be tested by comparing envelope output curves against reference MSFA/Dexed output for identical parameter sets.

**Acceptance Scenarios**:

1. **Given** envelope rate and level parameters matching a known DX7 patch, **When** processed, **Then** the envelope curve matches MSFA reference within acceptable tolerance
2. **Given** a rate parameter of 99 (maximum), **When** transitioning between levels, **Then** the transition is nearly instantaneous (characteristic DX7 "click")
3. **Given** two segments with the same rate but different level distances, **When** processed, **Then** the segment with greater distance takes longer (distance-dependent timing)

---

### User Story 4 - Instantaneous Attack dB Jump (Priority: P2)

A patch programmer wants punchy attack transients, so the envelope should support an instantaneous dB level jump at the start of attack (as in the original DX7/SY99).

**Why this priority**: This characteristic attack behavior is what gives FM synthesis its distinctive punch. Important for authenticity but not blocking for basic operation.

**Independent Test**: Can be tested by examining the first few samples after note-on and verifying an immediate level jump occurs.

**Acceptance Scenarios**:

1. **Given** an envelope starting from zero, **When** note-on triggers with attack rate > 0, **Then** the output immediately jumps to a non-zero dB value on the first sample
2. **Given** the instantaneous jump has occurred, **When** attack continues, **Then** the envelope smoothly approaches the attack target level

---

### User Story 5 - Delayed Envelope Start (Priority: P2)

A sound designer wants to create evolving pads where certain operators fade in after a delay, so envelopes must support a configurable delay before the attack begins.

**Why this priority**: Delay enables more complex sound design but basic envelope shapes work without it.

**Independent Test**: Can be tested by configuring a delay and verifying the envelope remains at zero for the specified duration before attack begins.

**Acceptance Scenarios**:

1. **Given** a delay time of 500ms is configured, **When** note-on occurs, **Then** the envelope output remains at minimum for approximately 500ms before attack begins
2. **Given** a delay is in progress, **When** note-off occurs, **Then** the envelope immediately transitions to release behavior (delay is aborted)

---

### User Story 6 - Looping Between Segments (Priority: P2)

A sound designer wants to create rhythmic or evolving textures, so envelopes must support looping between configurable segment boundaries during key-on.

**Why this priority**: Looping adds creative capability for modern sound design but is not required for traditional FM patches.

**Independent Test**: Can be tested by enabling looping and verifying the envelope cycles between specified segments until note-off.

**Acceptance Scenarios**:

1. **Given** looping is enabled from segment 3 to segment 5, **When** the envelope reaches segment 5, **Then** it loops back to segment 3 and continues
2. **Given** looping is active, **When** note-off occurs, **Then** looping stops and the envelope transitions to release segments
3. **Given** loop boundaries are invalid (start > end), **When** the envelope is configured, **Then** looping is disabled with a sensible fallback

---

### User Story 7 - High-Performance Block Processing (Priority: P1)

A synthesizer engine needs to process hundreds of envelopes per audio block (12 envelopes x 32 voices x 4 unison = 1536 envelopes) without causing audio dropouts.

**Why this priority**: Performance is critical for polyphonic synthesis. Without efficient processing, the feature is unusable in practice.

**Independent Test**: Can be tested by benchmarking envelope processing throughput and verifying it meets performance targets.

**Acceptance Scenarios**:

1. **Given** 1536 envelopes processing simultaneously, **When** generating a 64-sample audio block at 44.1kHz, **Then** total envelope processing time is under 1% of available CPU budget
2. **Given** SIMD acceleration is available, **When** processing envelope blocks, **Then** performance improves compared to scalar processing
3. **Given** any envelope configuration, **When** processing, **Then** no heap allocations occur

---

### User Story 8 - Variant Envelope Configurations (Priority: P3)

A synthesizer framework needs different envelope variants for different synthesis sections (FM operators vs. AWM samples), so the envelope implementation should be configurable for different segment counts.

**Why this priority**: Variant support enables code reuse but can be implemented after the core 8-segment envelope works.

**Independent Test**: Can be tested by instantiating different envelope variants and verifying they behave correctly with their respective segment counts.

**Acceptance Scenarios**:

1. **Given** an 8-segment envelope configuration (6+2), **When** instantiated, **Then** all 8 segments are available for use
2. **Given** a 7-segment envelope configuration, **When** instantiated, **Then** exactly 7 segments are available
3. **Given** a 5+5 envelope configuration, **When** instantiated, **Then** 5 key-on and 5 release segments are available

---

### Edge Cases

- What happens when rate scaling pushes a rate beyond valid range? (Clamp to maximum valid rate of 63 qRate units)
- How does the envelope handle sample rate changes during playback? (Recalculate internal coefficients using sample rate multiplier)
- What happens when loop start equals loop end? (Single-segment loop, effectively holds at that level)
- How does the envelope behave with rate = 0? (Holds at current level indefinitely until note-off)
- What happens if note-off occurs during delay? (Skip remaining delay and proceed to release segments)
- How does the envelope handle very short notes (note-off before attack completes)? (Immediately transition to release from current level)

## Requirements *(mandatory)*

### Functional Requirements

#### Core Envelope Behavior

- **FR-001**: Envelope MUST operate internally in logarithmic domain (dB) and output linear amplitude (0.0 to 1.0) after dB→linear conversion
- **FR-002**: Envelope MUST support configurable segment counts via compile-time const generics (e.g., `Envelope<KEY_ON_SEGS, RELEASE_SEGS>`) for 6+2, 7, 5+5 configurations
- **FR-003**: Each segment MUST have independent rate and level parameters
- **FR-004**: Segment transitions MUST be linear in the logarithmic domain (linear-in-dB, exponential in linear amplitude)
- **FR-005**: Rate values MUST follow MSFA convention (0-99 range mapping to internal timing)
- **FR-006**: Level values MUST follow MSFA convention (0-99 range mapping to dB values)

#### Rate and Timing

- **FR-007**: Envelope rates MUST be nonlinear following the MSFA rate-step table formula
- **FR-008**: Segment timing MUST be distance-dependent (longer distance = longer time for same rate)
- **FR-009**: Rate scaling MUST adjust segment rates based on MIDI note number
- **FR-010**: Rate scaling MUST accept values in qRate units (0-63 range)
- **FR-011**: Maximum rate (99) MUST produce near-instantaneous transitions

#### Attack Behavior

- **FR-012**: Attack phase MUST implement instantaneous dB jump on first sample (MSFA-style behavior)
- **FR-013**: After initial jump, attack MUST smoothly approach target level using exponential curve
- **FR-014**: Attack increment MUST be proportional to remaining distance to target

#### Delay and Looping

- **FR-015**: Envelope MUST support configurable delay before attack begins
- **FR-016**: Delay time MUST be specified in time units convertible to samples via Timebase
- **FR-017**: Envelope MUST support looping between arbitrary segment boundaries
- **FR-018**: Loop boundaries MUST be configurable as start and end segment indices
- **FR-019**: Looping MUST only occur during key-on phase

#### Note Events

- **FR-020**: Envelope MUST respond to note-on by starting attack sequence (after delay if configured)
- **FR-021**: Envelope MUST respond to note-off by transitioning to release segments
- **FR-022**: Envelope MUST support retrigger behavior: on new note-on, restart attack sequence from current level (no hard reset to zero) to avoid clicks

#### Integration

- **FR-023**: Envelope MUST implement the ModulationSource trait for integration with modulation routing
- **FR-024**: Envelope MUST use Timebase for sample-accurate timing
- **FR-025**: Envelope MUST be no_std compatible with zero heap allocations
- **FR-026**: Envelope MUST be Copy and Clone for efficient voice management

#### Performance

- **FR-027**: Envelope MUST support efficient block processing for multiple samples
- **FR-028**: Envelope SHOULD use SIMD operations via rigel-math by batching multiple envelopes (4-8 voices) per SIMD operation for parallel processing
- **FR-029**: Envelope MUST maintain real-time safety (constant-time operations, no blocking)

### Key Entities

- **Segment**: A transition between two levels at a specified rate; contains rate (0-99), target level (0-99), and computed increment
- **Envelope\<K, R\>**: Generic envelope type parameterized by key-on segment count (K) and release segment count (R); enables compile-time optimization per variant
- **EnvelopeState**: Current runtime state including active segment index, current level (in dB), samples remaining in delay, loop state
- **EnvelopeConfig**: Immutable configuration including rates array, levels array, rate scaling, delay time, loop boundaries
- **EnvelopePhase**: Current phase of envelope operation (Delay, Attack, Sustain, Release, Complete)

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Processing 1536 envelopes for a 64-sample block completes in under 100 microseconds on modern hardware
- **SC-002**: Single envelope processing overhead is under 50 nanoseconds per sample
- **SC-003**: Memory footprint per envelope instance is under 128 bytes
- **SC-004**: Envelope output values match MSFA reference implementation within 0.1 dB tolerance
- **SC-005**: All envelope parameter combinations produce valid, artifact-free output
- **SC-006**: Envelope transitions between segments produce no audible clicks or discontinuities
- **SC-007**: Rate scaling produces keyboard tracking that sounds natural across full MIDI range (0-127)
- **SC-008**: Block processing performance improves by at least 2x when SIMD is available compared to scalar

## Assumptions

- The MSFA rate-step table formula will be used for rate calculations
- Level scaling will use the MSFA lookup table for values 0-19 and linear formula for 20-99
- Sample rate will typically be 44.1kHz or 48kHz; sample rate compensation will be applied for other rates
- The envelope will output linear amplitude values (0.0 to 1.0) suitable for direct multiplication with oscillator output; internal processing uses dB domain (-96dB to 0dB) converted to linear on output
- rigel-math SIMD abstractions will be used for vectorized processing
- The 8-segment envelope (6 key-on + 2 release) is the primary variant; others derive from the same core implementation
