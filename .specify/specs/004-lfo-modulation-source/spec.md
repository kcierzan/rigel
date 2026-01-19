# Feature Specification: LFO Modulation Source

**Feature Branch**: `004-lfo-modulation-source`
**Created**: 2025-12-14
**Status**: Draft
**Linear Issue**: [NEW-6](https://linear.app/new-atlantis/issue/NEW-6/create-lfo-modulation-source)
**Input**: User description: "Create an LFO-style modulation source that exposes modulation data that should be usable in a forthcoming modulation bus. The lfo should support phase-reset on keydown and free-running modes. It should support unipolar/bipolar modes (this may be a feature only of the forthcoming modulation matrix system). It should be timebase aware and should support tempo sync. At this time, it should support basic waveshapes like sine, triangle, saw, square, pulse (with pwm), sample and hold, and noise (random). We may want to consider a new modulation crate that imports from the timing crate and is imported in turn by rigel-plugin."

## Clarifications

### Session 2025-12-14

- Q: Should the LFO be implemented in a new `rigel-modulation` crate, or added to an existing crate? → A: Create a new dedicated `rigel-modulation` crate that depends on `rigel-timing` and is imported by `rigel-dsp`/`rigel-plugin`. This crate will house all modulation sources (LFO, future envelopes, sequencers, etc.) independently from timing infrastructure.
- Q: Where should the `ModulationSource` trait live? → A: Move the `ModulationSource` trait from `rigel-timing` to `rigel-modulation`. The trait defines the interface for modulation sources, so it belongs with its implementations. `rigel-timing` provides only foundational types (`Timebase`, `Smoother`, `ControlRateClock`).
- Q: Should LFOs be per-voice or global? → A: LFOs are currently "global" (not per-voice) as there is no complete polyphonic voice architecture yet. However, the implementation must avoid any design choices that would preclude future per-voice LFOs or other modulation sources. All types should be Copy/Clone and self-contained to enable per-voice instantiation later.
- Q: How should LFO updates be scheduled? → A: The LFO MUST run at control rate using the `ControlRateClock` infrastructure from `rigel-timing`. The LFO's `update()` method is designed to be called at control rate intervals (typically every 32-64 samples), not every sample. This is critical for efficiency with massive polyphony/unison where per-voice LFOs would otherwise be prohibitively expensive. The LFO must calculate phase increments based on the control rate interval, not assuming per-sample updates.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Basic LFO Modulation (Priority: P1)

A sound designer wants to add movement to a synthesizer patch by using an LFO to modulate a filter cutoff. They select a sine wave LFO, set the rate to 2 Hz, and connect it to the filter cutoff parameter. The filter cutoff smoothly oscillates, adding expressive movement to the sound.

**Why this priority**: This is the core value proposition of an LFO - providing cyclical modulation. Without this fundamental capability, the LFO cannot fulfill any other requirements.

**Independent Test**: Can be fully tested by creating an LFO instance, setting a waveshape and rate, advancing time, and verifying the output oscillates with the expected frequency and waveform characteristics.

**Acceptance Scenarios**:

1. **Given** an LFO configured with a sine waveshape at 1 Hz rate, **When** one second of time passes, **Then** the LFO completes exactly one cycle returning values in the range [-1.0, 1.0]
2. **Given** an LFO configured with any supported waveshape, **When** the LFO is queried for its value, **Then** the value is always within the expected output range for its polarity mode
3. **Given** an LFO with a rate of 0.5 Hz, **When** two seconds of time passes, **Then** the LFO completes exactly one full cycle

---

### User Story 2 - Tempo-Synchronized Modulation (Priority: P1)

A musician wants their LFO modulation to stay locked to the song tempo for rhythmic effects. They select a tempo-sync mode with a 1/4 note division. When the DAW tempo changes from 120 BPM to 140 BPM, the LFO automatically adjusts its rate so that the modulation cycle always aligns with quarter note boundaries.

**Why this priority**: Tempo synchronization is essential for modern electronic music production where modulation must align with the beat. This is a fundamental feature expected by users.

**Independent Test**: Can be tested by providing tempo information to the LFO and verifying the cycle period matches the expected note division at various tempos.

**Acceptance Scenarios**:

1. **Given** an LFO in tempo-sync mode with 1/4 note division at 120 BPM, **When** queried for its effective rate, **Then** the rate equals 2 Hz (120 beats/60 seconds = 2 quarter notes per second)
2. **Given** an LFO in tempo-sync mode, **When** the tempo changes from 120 BPM to 60 BPM, **Then** the LFO cycle period doubles
3. **Given** an LFO in tempo-sync mode with 1/8 note triplet division, **When** at 120 BPM, **Then** the LFO completes 6 cycles per second (2 beats × 3 triplets)

---

### User Story 3 - Phase Reset on Note Trigger (Priority: P2)

A sound designer wants consistent attack transients where the LFO always starts from the same position when a new note is played. They enable "retrigger" mode on the LFO. Now whenever a new key is pressed, the LFO phase resets to zero, ensuring the modulation effect is identical for every note.

**Why this priority**: Phase reset enables predictable, repeatable sound design. While not strictly necessary for basic modulation, it is essential for professional sound design workflows.

**Independent Test**: Can be tested by triggering a note event and verifying the LFO phase resets to zero (or the configured start phase) after the trigger.

**Acceptance Scenarios**:

1. **Given** an LFO in retrigger mode at mid-cycle, **When** a note-on event occurs, **Then** the LFO phase resets to zero (or configured start phase)
2. **Given** an LFO in free-running mode at mid-cycle, **When** a note-on event occurs, **Then** the LFO phase continues uninterrupted
3. **Given** an LFO in retrigger mode with a start phase of 90 degrees, **When** a note-on event occurs, **Then** the LFO phase resets to 90 degrees (0.25 in normalized phase)

---

### User Story 4 - PWM Control for Pulse Wave (Priority: P2)

A sound designer wants to create rhythmic gating effects using a pulse wave LFO. They select the pulse waveshape and adjust the pulse width from 50% (square wave) to 10% (short spike). This creates sharp rhythmic pulses that gate other parameters in the synthesizer.

**Why this priority**: Pulse width modulation is a unique feature of pulse waves that enables creative effects not possible with other waveshapes. Important for sound design flexibility.

**Independent Test**: Can be tested by setting different pulse widths and verifying the duty cycle of the resulting waveform matches the configured value.

**Acceptance Scenarios**:

1. **Given** a pulse LFO with 50% pulse width, **When** one cycle completes, **Then** the output is high for exactly 50% of the cycle and low for 50%
2. **Given** a pulse LFO with 10% pulse width, **When** one cycle completes, **Then** the output is high for 10% of the cycle and low for 90%
3. **Given** a pulse LFO with 90% pulse width, **When** one cycle completes, **Then** the output is high for 90% of the cycle and low for 10%

---

### User Story 5 - Sample and Hold Modulation (Priority: P2)

A sound designer wants to create stepped random modulation for experimental effects. They select the sample-and-hold waveshape. The LFO generates a new random value at each cycle boundary and holds it constant until the next cycle, creating a stepped random pattern.

**Why this priority**: Sample-and-hold provides unique modulation character essential for classic synthesizer sounds and experimental design.

**Independent Test**: Can be tested by advancing through multiple cycles and verifying the output remains constant within each cycle but changes at cycle boundaries.

**Acceptance Scenarios**:

1. **Given** a sample-and-hold LFO at mid-cycle, **When** the phase is still within the same cycle, **Then** the output value remains constant
2. **Given** a sample-and-hold LFO approaching a cycle boundary, **When** the cycle boundary is crossed, **Then** a new random value is sampled and held
3. **Given** a sample-and-hold LFO, **When** multiple cycles complete, **Then** each cycle produces a different random value within the output range

---

### User Story 6 - Polarity Mode Selection (Priority: P3)

A sound designer wants to use an LFO to control a parameter that only accepts positive values (like an oscillator volume). They switch the LFO from bipolar mode [-1.0, 1.0] to unipolar mode [0.0, 1.0]. The modulation now only adds to the base parameter value rather than oscillating around it.

**Why this priority**: While useful for parameter mapping, polarity conversion could potentially be handled at the modulation matrix level. Still valuable for the LFO itself to support this.

**Independent Test**: Can be tested by switching polarity modes and verifying the output range changes accordingly.

**Acceptance Scenarios**:

1. **Given** an LFO in bipolar mode with a sine wave, **When** the LFO completes a cycle, **Then** values range from -1.0 to 1.0
2. **Given** an LFO in unipolar mode with a sine wave, **When** the LFO completes a cycle, **Then** values range from 0.0 to 1.0
3. **Given** an LFO switching from bipolar to unipolar mode, **When** the output was at -1.0, **Then** the output becomes 0.0

---

### User Story 7 - Control Rate Processing for Polyphony Efficiency (Priority: P1)

A synthesizer running 64 voices of unison with per-voice LFOs needs to minimize CPU usage. The system uses control rate processing to update LFOs every 64 samples instead of every sample. With a 1024-sample block at 44.1kHz, this reduces LFO computations from 65,536 per block (64 voices × 1024 samples) to just 1,024 per block (64 voices × 16 updates).

**Why this priority**: Control rate processing is essential for scalability with massive polyphony. Without it, per-voice LFOs become a CPU bottleneck, limiting the synthesizer's polyphony and unison capabilities. This is a fundamental architectural requirement.

**Independent Test**: Can be tested by measuring CPU time for updating N LFO instances at different control rate intervals and verifying the computational cost scales with the number of updates, not the block size.

**Acceptance Scenarios**:

1. **Given** an LFO with a 1 Hz rate and control rate interval of 64 samples at 44100 Hz, **When** `update()` is called with a timebase advanced by 64 samples, **Then** the phase increments by exactly (1.0 / 44100.0) × 64 = 0.00145 (approximately)
2. **Given** an LFO scheduled via `ControlRateClock` with interval 64, **When** a 256-sample block is processed, **Then** `update()` is called exactly 4 times (at sample offsets 0, 64, 128, 192)
3. **Given** an LFO integrated with `ControlRateClock`, **When** the control rate interval changes from 64 to 32 samples, **Then** the LFO continues to produce correct output with twice as many updates per block
4. **Given** 64 per-voice LFO instances, **When** updated at control rate (64-sample interval) vs sample rate, **Then** CPU usage is reduced by approximately 64× while maintaining acceptable modulation quality

---

### Edge Cases

- What happens when the LFO rate is set to 0 Hz? The LFO should output a constant value based on the current phase position.
- What happens when tempo is set to 0 BPM in tempo-sync mode? The LFO should fall back to a minimum safe rate (e.g., 0.001 Hz) or use the last valid tempo.
- What happens when sample rate changes mid-playback? The LFO phase should be preserved while internal timing calculations are updated.
- What happens when pulse width is set to 0% or 100%? The output should clamp to the minimum/maximum practical value (e.g., 1%/99%) to ensure some transition occurs.
- What happens to sample-and-hold when reset is triggered? A new random value should be sampled immediately.
- What happens when LFO is processing faster than the audio thread provides updates? The LFO should remain stable and produce valid output based on the timebase.

## Requirements *(mandatory)*

### Functional Requirements

#### Core LFO Behavior
- **FR-001**: LFO MUST generate periodic output values based on the selected waveshape
- **FR-002**: LFO MUST support rate control in Hz (free-running mode) from 0.01 Hz to 100 Hz
- **FR-003**: LFO MUST synchronize with the existing Timebase infrastructure for sample-accurate timing
- **FR-004**: LFO MUST implement the ModulationSource trait (defined in `rigel-modulation`)

#### Waveshapes
- **FR-005**: LFO MUST support sine waveshape (smooth sinusoidal oscillation)
- **FR-006**: LFO MUST support triangle waveshape (linear ramp up and down)
- **FR-007**: LFO MUST support sawtooth waveshape (linear ramp up, instant reset)
- **FR-008**: LFO MUST support square waveshape (50% duty cycle)
- **FR-009**: LFO MUST support pulse waveshape with configurable pulse width from 1% to 99%
- **FR-010**: LFO MUST support sample-and-hold waveshape (random value held for one cycle)
- **FR-011**: LFO MUST support noise waveshape (continuously varying random values)

#### Phase and Triggering
- **FR-012**: LFO MUST support free-running mode where phase continues uninterrupted
- **FR-013**: LFO MUST support retrigger mode where phase resets on note-on events
- **FR-014**: LFO MUST support configurable start phase from 0 to 360 degrees (0.0 to 1.0 normalized) for retrigger
- **FR-015**: LFO MUST expose a trigger method that can be called on note-on events

#### Polarity
- **FR-016**: LFO MUST support bipolar output mode with range [-1.0, 1.0]
- **FR-017**: LFO MUST support unipolar output mode with range [0.0, 1.0]

#### Tempo Synchronization
- **FR-018**: LFO MUST support tempo-sync mode where rate is derived from BPM and note division
- **FR-019**: LFO MUST support standard note divisions: 1/1, 1/2, 1/4, 1/8, 1/16, 1/32
- **FR-020**: LFO MUST support dotted note divisions (1.5× duration)
- **FR-021**: LFO MUST support triplet note divisions (2/3× duration)

#### Real-time Constraints
- **FR-022**: LFO MUST be no_std compatible (no heap allocations, no standard library)
- **FR-023**: LFO MUST process updates in constant time regardless of configuration
- **FR-024**: LFO MUST be Copy/Clone for efficient real-time handling

#### Future Compatibility
- **FR-029**: LFO implementation MUST be self-contained with no global state, enabling future per-voice instantiation
- **FR-030**: All LFO types MUST be Copy/Clone and small enough to be efficiently copied per-voice

#### Control Rate Integration
- **FR-031**: LFO `update()` method MUST be designed for control-rate invocation, not per-sample processing
- **FR-032**: LFO MUST calculate phase increments based on elapsed samples since last update (from `Timebase`), not assuming a fixed update interval
- **FR-033**: LFO MUST integrate seamlessly with `ControlRateClock` from `rigel-timing` for scheduling updates
- **FR-034**: LFO MUST produce correct output regardless of control rate interval (1, 8, 16, 32, 64, or 128 samples)
- **FR-035**: LFO `value()` method MUST return the last computed value without additional computation, enabling cheap per-sample reads between control rate updates

#### Crate Organization
- **FR-025**: LFO MUST be implemented in a new `rigel-modulation` crate separate from `rigel-timing`
- **FR-026**: The `rigel-modulation` crate MUST depend on `rigel-timing` for `Timebase` access
- **FR-027**: The `ModulationSource` trait MUST be moved from `rigel-timing` to `rigel-modulation` and re-exported for backward compatibility

### Key Entities

- **ModulationSource**: Trait interface for all modulation sources. Defines `reset()`, `update(timebase)`, and `value()` methods. Moved from `rigel-timing` to `rigel-modulation` as the canonical location for modulation abstractions.

- **Lfo**: The primary LFO implementation that generates modulation values. Holds configuration (waveshape, rate, polarity, phase mode) and runtime state (current phase). Implements ModulationSource trait.

- **LfoWaveshape**: Enumeration of available waveshapes (Sine, Triangle, Saw, Square, Pulse, SampleAndHold, Noise). Determines how phase position maps to output value.

- **LfoPhaseMode**: Enumeration of phase behavior (FreeRunning, Retrigger). Determines whether phase resets on note events.

- **LfoPolarity**: Enumeration of output range modes (Bipolar, Unipolar). Determines output value range scaling.

- **LfoRateMode**: Configuration for how rate is determined - either direct Hz value or tempo-synchronized with note division and BPM.

- **NoteDivision**: Enumeration of musical note divisions for tempo sync (Whole, Half, Quarter, Eighth, Sixteenth, ThirtySecond, with optional Dotted/Triplet modifiers).

## Assumptions

- **Crate organization**: The LFO will be implemented in a new `rigel-modulation` crate that depends on `rigel-timing` for `Timebase`. The `ModulationSource` trait will be moved from `rigel-timing` to `rigel-modulation` (with re-export for backward compatibility). This crate is designed to house all future modulation sources (envelopes, sequencers, etc.) independently from core timing infrastructure.
- **Random number generation**: The LFO will use a deterministic pseudo-random number generator seeded from the system or provided seed for sample-and-hold and noise modes, ensuring reproducibility when desired.
- **Tempo information source**: Tempo (BPM) will be provided externally via the Timebase or a separate method call, as the LFO itself does not manage transport state.
- **Trigger event delivery**: Note-on events for phase reset will be delivered via explicit method calls from the voice/note management layer.
- **Output range guarantee**: All waveshapes will produce values strictly within their polarity range; no overshoots or undershoots will occur.
- **Phase continuity**: Phase is maintained as a continuous 0.0-1.0 value that wraps smoothly, avoiding discontinuities except at explicit resets.
- **Global vs per-voice**: LFOs are currently used globally (not per-voice) as there is no complete polyphonic voice architecture. However, the design explicitly supports future per-voice instantiation by ensuring all types are Copy/Clone, self-contained, and free of global state.
- **Control rate scheduling**: The LFO's `update()` method is designed to be called at control rate intervals via `ControlRateClock`, not every sample. The caller is responsible for scheduling updates using `ControlRateClock::advance()`. The LFO computes phase increments based on elapsed samples from `Timebase`, making it agnostic to the specific control rate interval used. This enables efficient processing with massive polyphony.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: LFO produces output values that remain strictly within the configured polarity range ([-1.0, 1.0] for bipolar, [0.0, 1.0] for unipolar) across all waveshapes and configurations
- **SC-002**: LFO cycle frequency accuracy is within 0.1% of the configured rate when measured over 10 or more cycles
- **SC-003**: Tempo-synchronized LFO maintains lock with BPM changes, with cycle boundaries aligning to beat divisions within 1 sample of accuracy
- **SC-004**: LFO update computation completes within a fixed time budget suitable for real-time audio (less than 1 microsecond per update on target hardware)
- **SC-005**: LFO uses zero heap allocations during normal operation, validated through code analysis
- **SC-006**: All seven waveshapes produce distinct, correct waveform characteristics as validated against reference waveforms
- **SC-007**: Phase reset on trigger occurs within the same audio block as the trigger event (sample-accurate)
- **SC-008**: LFO produces identical output when updated at different control rate intervals (1, 32, 64, 128 samples), within floating-point precision tolerance
- **SC-009**: Processing 64 LFO instances at control rate (64-sample interval) uses less than 5% of the CPU time compared to per-sample processing
