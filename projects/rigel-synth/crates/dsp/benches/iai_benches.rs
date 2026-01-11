use iai_callgrind::{library_benchmark, library_benchmark_group, main};
use rigel_dsp::{
    midi_to_freq, soft_clip, ControlRateClock, SimpleOscillator, Smoother, SmoothingMode,
    SynthEngine, SynthParams, Timebase,
};
use rigel_modulation::envelope::{FmEnvelope, FmEnvelopeConfig};
use std::hint::black_box;

// ==============================================================================
// Utility Functions
// ==============================================================================

#[library_benchmark]
fn iai_midi_to_freq() -> f32 {
    black_box(midi_to_freq(black_box(60)))
}

#[library_benchmark]
fn iai_soft_clip() -> f32 {
    black_box(soft_clip(black_box(1.5)))
}

// ==============================================================================
// Clamp Implementation Comparison
// ==============================================================================

// Alternative clamp implementations for benchmarking
#[inline]
fn clamp_branching(value: f32, min: f32, max: f32) -> f32 {
    if value < min {
        min
    } else if value > max {
        max
    } else {
        value
    }
}

#[inline]
fn clamp_branchless_max_min(value: f32, min: f32, max: f32) -> f32 {
    value.max(min).min(max)
}

#[inline]
fn clamp_builtin(value: f32, min: f32, max: f32) -> f32 {
    value.clamp(min, max)
}

#[library_benchmark]
#[benches::with_setup(args = ["in_range", "below_min", "above_max"])]
fn iai_clamp_branching(scenario: &str) -> f32 {
    let value = match scenario {
        "in_range" => 0.5,
        "below_min" => -0.5,
        "above_max" => 1.5,
        _ => 0.5,
    };
    black_box(clamp_branching(
        black_box(value),
        black_box(0.0),
        black_box(1.0),
    ))
}

#[library_benchmark]
#[benches::with_setup(args = ["in_range", "below_min", "above_max"])]
fn iai_clamp_max_min(scenario: &str) -> f32 {
    let value = match scenario {
        "in_range" => 0.5,
        "below_min" => -0.5,
        "above_max" => 1.5,
        _ => 0.5,
    };
    black_box(clamp_branchless_max_min(
        black_box(value),
        black_box(0.0),
        black_box(1.0),
    ))
}

#[library_benchmark]
#[benches::with_setup(args = ["in_range", "below_min", "above_max"])]
fn iai_clamp_builtin(scenario: &str) -> f32 {
    let value = match scenario {
        "in_range" => 0.5,
        "below_min" => -0.5,
        "above_max" => 1.5,
        _ => 0.5,
    };
    black_box(clamp_builtin(
        black_box(value),
        black_box(0.0),
        black_box(1.0),
    ))
}

#[library_benchmark]
fn iai_clamp_branching_mixed() -> f32 {
    let values = [-0.5, 0.3, 0.7, 1.5, 0.1, 2.0, -1.0, 0.5];
    let mut sum = 0.0;
    for &v in values.iter() {
        sum += clamp_branching(v, 0.0, 1.0);
    }
    black_box(sum)
}

#[library_benchmark]
fn iai_clamp_max_min_mixed() -> f32 {
    let values = [-0.5, 0.3, 0.7, 1.5, 0.1, 2.0, -1.0, 0.5];
    let mut sum = 0.0;
    for &v in values.iter() {
        sum += clamp_branchless_max_min(v, 0.0, 1.0);
    }
    black_box(sum)
}

#[library_benchmark]
fn iai_clamp_builtin_mixed() -> f32 {
    let values = [-0.5, 0.3, 0.7, 1.5, 0.1, 2.0, -1.0, 0.5];
    let mut sum = 0.0;
    for &v in values.iter() {
        sum += clamp_builtin(v, 0.0, 1.0);
    }
    black_box(sum)
}

// ==============================================================================
// Oscillator
// ==============================================================================

#[library_benchmark]
fn iai_oscillator_single_sample() -> f32 {
    let mut osc = SimpleOscillator::new();
    osc.set_frequency(440.0, 44100.0);
    black_box(osc.process_sample())
}

#[library_benchmark]
#[benches::with_setup(args = [64, 128, 256, 512, 1024])]
fn iai_oscillator_buffer(buffer_size: usize) -> f32 {
    let mut osc = SimpleOscillator::new();
    osc.set_frequency(440.0, 44100.0);

    let mut sum = 0.0;
    for _ in 0..buffer_size {
        sum += osc.process_sample();
    }
    black_box(sum)
}

// ==============================================================================
// Envelope
// ==============================================================================

#[library_benchmark]
fn iai_envelope_attack() -> f32 {
    let config = FmEnvelopeConfig::adsr(0.1, 0.1, 0.7, 0.1, 44100.0);
    let mut env = FmEnvelope::with_config(config);
    env.note_on(60);
    black_box(env.process())
}

#[library_benchmark]
fn iai_envelope_sustain() -> f32 {
    let config = FmEnvelopeConfig::adsr(0.001, 0.001, 0.7, 0.1, 44100.0);
    let mut env = FmEnvelope::with_config(config);
    env.note_on(60);
    // Process through attack and decay to reach sustain
    for _ in 0..500 {
        env.process();
    }
    black_box(env.process())
}

#[library_benchmark]
fn iai_envelope_release() -> f32 {
    let config = FmEnvelopeConfig::adsr(0.001, 0.1, 0.7, 0.1, 44100.0);
    let mut env = FmEnvelope::with_config(config);
    env.note_on(60);
    for _ in 0..100 {
        env.process();
    }
    env.note_off();
    black_box(env.process())
}

// ==============================================================================
// SynthEngine - Critical Path
// ==============================================================================

#[library_benchmark]
fn iai_synth_single_sample() -> f32 {
    let mut engine = SynthEngine::new(44100.0);
    let params = SynthParams::default();
    engine.note_on(60, 0.8);
    black_box(engine.process_sample(&params))
}

#[library_benchmark]
#[benches::with_setup(args = [64, 128, 256, 512, 1024])]
fn iai_synth_buffer(buffer_size: usize) -> f32 {
    let mut engine = SynthEngine::new(44100.0);
    let params = SynthParams::default();
    engine.note_on(60, 0.8);

    let mut sum = 0.0;
    for _ in 0..buffer_size {
        sum += engine.process_sample(&params);
    }
    black_box(sum)
}

#[library_benchmark]
fn iai_synth_with_pitch_mod() -> f32 {
    let mut engine = SynthEngine::new(44100.0);
    let params = SynthParams {
        pitch_offset: 2.0, // +2 semitones
        ..Default::default()
    };
    engine.note_on(60, 0.8);
    black_box(engine.process_sample(&params))
}

#[library_benchmark]
fn iai_synth_sustained_note() -> f32 {
    let mut engine = SynthEngine::new(44100.0);
    let params = SynthParams::default();
    engine.note_on(60, 0.8);

    // Process enough samples to reach sustain stage
    for _ in 0..10000 {
        engine.process_sample(&params);
    }

    black_box(engine.process_sample(&params))
}

#[library_benchmark]
fn iai_synth_note_lifecycle() -> f32 {
    let mut engine = SynthEngine::new(44100.0);
    let params = SynthParams::default();

    engine.note_on(60, 0.8);

    // Process 10ms of audio
    let mut sum = 0.0;
    for _ in 0..441 {
        sum += engine.process_sample(&params);
    }

    engine.note_off(60);

    // Process another 10ms
    for _ in 0..441 {
        sum += engine.process_sample(&params);
    }

    black_box(sum)
}

// ==============================================================================
// Polyphonic Extrapolation
// ==============================================================================

#[library_benchmark]
#[benches::with_setup(args = [1, 4, 8])]
fn iai_polyphonic_voices(voice_count: usize) -> f32 {
    let params = SynthParams::default();
    let mut engines: Vec<SynthEngine> = (0..voice_count)
        .map(|i| {
            let mut engine = SynthEngine::new(44100.0);
            engine.note_on(60 + (i % 12) as u8, 0.8);
            engine
        })
        .collect();

    let mut sum = 0.0;
    for engine in engines.iter_mut() {
        sum += engine.process_sample(&params);
    }
    black_box(sum)
}

// ==============================================================================
// Timebase
// ==============================================================================

#[library_benchmark]
fn iai_timebase_advance_block() -> u64 {
    let mut timebase = Timebase::new(44100.0);
    timebase.advance_block(64);
    black_box(timebase.sample_position())
}

#[library_benchmark]
fn iai_timebase_samples_to_seconds() -> f64 {
    let timebase = Timebase::new(44100.0);
    black_box(timebase.samples_to_seconds(44100))
}

#[library_benchmark]
fn iai_timebase_ms_to_samples() -> u32 {
    let timebase = Timebase::new(44100.0);
    black_box(timebase.ms_to_samples(10.0))
}

// ==============================================================================
// Smoother
// ==============================================================================

#[library_benchmark]
fn iai_smoother_linear_single() -> f32 {
    let mut smoother = Smoother::new(0.0, SmoothingMode::Linear, 10.0, 44100.0);
    smoother.set_target(1.0);
    black_box(smoother.process_sample())
}

#[library_benchmark]
fn iai_smoother_exponential_single() -> f32 {
    let mut smoother = Smoother::new(0.0, SmoothingMode::Exponential, 10.0, 44100.0);
    smoother.set_target(1.0);
    black_box(smoother.process_sample())
}

#[library_benchmark]
fn iai_smoother_logarithmic_single() -> f32 {
    let mut smoother = Smoother::new(100.0, SmoothingMode::Logarithmic, 10.0, 44100.0);
    smoother.set_target(1000.0);
    black_box(smoother.process_sample())
}

#[library_benchmark]
fn iai_smoother_instant_single() -> f32 {
    let mut smoother = Smoother::new(0.0, SmoothingMode::Instant, 10.0, 44100.0);
    smoother.set_target(1.0);
    black_box(smoother.process_sample())
}

#[library_benchmark]
fn iai_smoother_inactive_single() -> f32 {
    let mut smoother = Smoother::new(0.5, SmoothingMode::Linear, 10.0, 44100.0);
    black_box(smoother.process_sample())
}

#[library_benchmark]
#[benches::with_setup(args = [64, 128, 256])]
fn iai_smoother_linear_block(buffer_size: usize) -> f32 {
    let mut smoother = Smoother::new(0.0, SmoothingMode::Linear, 10.0, 44100.0);
    smoother.set_target(1.0);
    let mut buffer = vec![0.0f32; buffer_size];
    smoother.process_block(&mut buffer);
    black_box(buffer[buffer_size - 1])
}

#[library_benchmark]
fn iai_smoother_set_target() -> f32 {
    let mut smoother = Smoother::new(0.0, SmoothingMode::Linear, 10.0, 44100.0);
    smoother.set_target(black_box(1.0));
    black_box(smoother.current())
}

// ==============================================================================
// ControlRateClock
// ==============================================================================

#[library_benchmark]
fn iai_control_rate_clock_advance_64() -> usize {
    let mut clock = ControlRateClock::new(64);
    black_box(clock.advance(128).count())
}

#[library_benchmark]
fn iai_control_rate_clock_advance_32() -> usize {
    let mut clock = ControlRateClock::new(32);
    black_box(clock.advance(128).count())
}

#[library_benchmark]
#[benches::with_setup(args = [64, 128, 256])]
fn iai_control_rate_clock_block(block_size: u32) -> usize {
    let mut clock = ControlRateClock::new(64);
    black_box(clock.advance(block_size).count())
}

// ==============================================================================
// Benchmark Groups and Main Configuration
// ==============================================================================

library_benchmark_group!(
    name = utility_benches;
    benchmarks = iai_midi_to_freq, iai_soft_clip
);

library_benchmark_group!(
    name = clamp_benches;
    benchmarks =
        iai_clamp_branching,
        iai_clamp_max_min,
        iai_clamp_builtin,
        iai_clamp_branching_mixed,
        iai_clamp_max_min_mixed,
        iai_clamp_builtin_mixed
);

library_benchmark_group!(
    name = oscillator_benches;
    benchmarks = iai_oscillator_single_sample, iai_oscillator_buffer
);

library_benchmark_group!(
    name = envelope_benches;
    benchmarks = iai_envelope_attack, iai_envelope_sustain, iai_envelope_release
);

library_benchmark_group!(
    name = synth_engine_benches;
    benchmarks =
        iai_synth_single_sample,
        iai_synth_buffer,
        iai_synth_with_pitch_mod,
        iai_synth_sustained_note,
        iai_synth_note_lifecycle
);

library_benchmark_group!(
    name = polyphonic_benches;
    benchmarks = iai_polyphonic_voices
);

library_benchmark_group!(
    name = timebase_benches;
    benchmarks = iai_timebase_advance_block, iai_timebase_samples_to_seconds, iai_timebase_ms_to_samples
);

library_benchmark_group!(
    name = smoother_benches;
    benchmarks =
        iai_smoother_linear_single,
        iai_smoother_exponential_single,
        iai_smoother_logarithmic_single,
        iai_smoother_instant_single,
        iai_smoother_inactive_single,
        iai_smoother_linear_block,
        iai_smoother_set_target
);

library_benchmark_group!(
    name = control_rate_clock_benches;
    benchmarks = iai_control_rate_clock_advance_64, iai_control_rate_clock_advance_32, iai_control_rate_clock_block
);

main!(
    library_benchmark_groups = utility_benches,
    clamp_benches,
    oscillator_benches,
    envelope_benches,
    synth_engine_benches,
    polyphonic_benches,
    timebase_benches,
    smoother_benches,
    control_rate_clock_benches
);
