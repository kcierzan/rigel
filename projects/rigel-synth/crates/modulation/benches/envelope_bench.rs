//! Benchmarks for SY-Style Envelope module.
//!
//! Performance targets from spec:
//! - Single envelope: <50ns per sample
//! - 1536 envelopes × 64 samples: <100µs
//! - SIMD batch: 2x+ speedup over scalar
//!
//! Benchmarks process realistic 1-second envelope lifecycles to capture
//! the full attack/decay/release behavior including exponential decay phases.

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use rigel_modulation::envelope::{
    ControlRateFmEnvelope, EnvelopeBatch, FmEnvelope, FmEnvelopeBatch8, FmEnvelopeConfig,
};
use std::hint::black_box;

/// Sample rate for all benchmarks
const SAMPLE_RATE: f32 = 44100.0;

/// Block size for block processing benchmarks
const BLOCK_SIZE: usize = 64;

/// Number of blocks per second at 44100Hz with 64-sample blocks
const BLOCKS_PER_SECOND: usize = (SAMPLE_RATE as usize) / BLOCK_SIZE; // ~689 blocks

/// Samples for ~1 second of audio
const SAMPLES_PER_SECOND: usize = SAMPLE_RATE as usize;

// =============================================================================
// Single Envelope Benchmarks (Realistic 1-Second Lifecycle)
// =============================================================================

fn bench_single_envelope_process(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_envelope");

    // Per-sample throughput for 1 second of audio
    group.throughput(Throughput::Elements(SAMPLES_PER_SECOND as u64));

    // Benchmark processing 1 second of samples (attack + decay + release)
    group.bench_function("1_second_lifecycle", |b| {
        // Configure realistic ADSR: 100ms attack, 200ms decay, sustain, 500ms release
        let config = FmEnvelopeConfig::adsr(0.1, 0.2, 0.7, 0.5, SAMPLE_RATE);
        let mut env = FmEnvelope::with_config(config);

        b.iter(|| {
            env.reset();
            env.note_on(60);

            // Process ~500ms of key-on (attack + decay + sustain)
            for _ in 0..22050 {
                black_box(env.process());
            }

            // Trigger release
            env.note_off();

            // Process ~500ms of release
            for _ in 0..22050 {
                black_box(env.process());
            }

            env.value()
        })
    });

    // Block processing: 1 second in 64-sample blocks
    group.throughput(Throughput::Elements(
        (BLOCKS_PER_SECOND * BLOCK_SIZE) as u64,
    ));

    group.bench_function("1_second_block_64", |b| {
        let config = FmEnvelopeConfig::adsr(0.1, 0.2, 0.7, 0.5, SAMPLE_RATE);
        let mut env = FmEnvelope::with_config(config);
        let mut output = [0.0f32; BLOCK_SIZE];

        b.iter(|| {
            env.reset();
            env.note_on(60);

            // Process ~500ms in blocks (attack + decay + sustain)
            for _ in 0..(BLOCKS_PER_SECOND / 2) {
                env.process_block(&mut output);
            }

            env.note_off();

            // Process ~500ms of release in blocks
            for _ in 0..(BLOCKS_PER_SECOND / 2) {
                env.process_block(&mut output);
            }

            black_box(output[0])
        })
    });

    // 128-sample blocks for comparison
    group.bench_function("1_second_block_128", |b| {
        let config = FmEnvelopeConfig::adsr(0.1, 0.2, 0.7, 0.5, SAMPLE_RATE);
        let mut env = FmEnvelope::with_config(config);
        let mut output = [0.0f32; 128];
        let blocks_per_second_128 = SAMPLES_PER_SECOND / 128;

        b.iter(|| {
            env.reset();
            env.note_on(60);

            for _ in 0..(blocks_per_second_128 / 2) {
                env.process_block(&mut output);
            }

            env.note_off();

            for _ in 0..(blocks_per_second_128 / 2) {
                env.process_block(&mut output);
            }

            black_box(output[0])
        })
    });

    group.finish();
}

// =============================================================================
// Batch Envelope Benchmarks (Realistic 1-Second Lifecycle)
// =============================================================================

fn bench_batch_envelope(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_envelope");

    // 8 envelopes × 1 second = 8 × 44100 samples
    group.throughput(Throughput::Elements((8 * SAMPLES_PER_SECOND) as u64));

    // Batch of 8 envelopes processing full 1-second lifecycle
    group.bench_function("batch_8_1_second", |b| {
        let config = FmEnvelopeConfig::adsr(0.1, 0.2, 0.7, 0.5, SAMPLE_RATE);
        let mut batch = FmEnvelopeBatch8::with_config(config);
        let mut output = [[0.0f32; 8]; BLOCK_SIZE];

        b.iter(|| {
            // Reset and trigger all 8 envelopes
            batch.reset_all();
            for i in 0..8 {
                batch.note_on(i, 60 + i as u8);
            }

            // Process ~500ms in blocks (attack + decay + sustain)
            for _ in 0..(BLOCKS_PER_SECOND / 2) {
                batch.process_block(&mut output);
            }

            // Release all envelopes
            for i in 0..8 {
                batch.note_off(i);
            }

            // Process ~500ms of release
            for _ in 0..(BLOCKS_PER_SECOND / 2) {
                batch.process_block(&mut output);
            }

            black_box(output[0][0])
        })
    });

    // Staggered note-on/off to stress segment transition branch prediction
    group.bench_function("batch_8_staggered_notes", |b| {
        let config = FmEnvelopeConfig::adsr(0.05, 0.1, 0.8, 0.3, SAMPLE_RATE);
        let mut batch = FmEnvelopeBatch8::with_config(config);
        let mut output = [[0.0f32; 8]; BLOCK_SIZE];

        b.iter(|| {
            batch.reset_all();

            // Stagger note-on events across 8 voices
            for voice in 0..8 {
                batch.note_on(voice, 60 + voice as u8);

                // Process some blocks between each note-on
                for _ in 0..50 {
                    batch.process_block(&mut output);
                }
            }

            // Process through decay phases
            for _ in 0..200 {
                batch.process_block(&mut output);
            }

            // Stagger note-off events
            for voice in 0..8 {
                batch.note_off(voice);

                // Process between each note-off
                for _ in 0..50 {
                    batch.process_block(&mut output);
                }
            }

            // Process remaining release
            for _ in 0..100 {
                batch.process_block(&mut output);
            }

            black_box(output[0][0])
        })
    });

    group.finish();
}

// =============================================================================
// Polyphonic Workload Benchmarks (Realistic 1-Second Lifecycle)
// =============================================================================

fn bench_polyphonic_workload(c: &mut Criterion) {
    let mut group = c.benchmark_group("polyphonic_workload");

    // 1536 envelopes × 1 second lifecycle
    // Throughput: 1536 × 44100 = ~67.7M samples
    const NUM_BATCHES_1536: usize = 192; // 192 * 8 = 1536 envelopes
    group.throughput(Throughput::Elements(
        (NUM_BATCHES_1536 * 8 * SAMPLES_PER_SECOND) as u64,
    ));

    group.bench_function("1536_envelopes_1_second", |b| {
        let config = FmEnvelopeConfig::adsr(0.1, 0.2, 0.7, 0.5, SAMPLE_RATE);

        let mut batches: Vec<EnvelopeBatch<8, 6, 2>> = (0..NUM_BATCHES_1536)
            .map(|_| {
                let mut batch = FmEnvelopeBatch8::with_config(config);
                for i in 0..8 {
                    batch.note_on(i, 60 + (i % 12) as u8);
                }
                batch
            })
            .collect();

        let mut output = [[0.0f32; 8]; BLOCK_SIZE];

        b.iter(|| {
            // Reset all envelopes
            for batch in batches.iter_mut() {
                batch.reset_all();
                for i in 0..8 {
                    batch.note_on(i, 60 + (i % 12) as u8);
                }
            }

            // Process ~500ms (attack + decay + sustain)
            for _ in 0..(BLOCKS_PER_SECOND / 2) {
                for batch in batches.iter_mut() {
                    batch.process_block(&mut output);
                }
            }

            // Release all envelopes
            for batch in batches.iter_mut() {
                for i in 0..8 {
                    batch.note_off(i);
                }
            }

            // Process ~500ms of release
            for _ in 0..(BLOCKS_PER_SECOND / 2) {
                for batch in batches.iter_mut() {
                    batch.process_block(&mut output);
                }
            }

            black_box(output[0][0])
        })
    });

    // 32 voices × 12 envelopes × 1 second = 384 envelopes
    const NUM_VOICES: usize = 32;
    const ENVS_PER_VOICE: usize = 12;
    const NUM_BATCHES_384: usize = (NUM_VOICES * ENVS_PER_VOICE).div_ceil(8); // 48 batches

    group.throughput(Throughput::Elements(
        (NUM_BATCHES_384 * 8 * SAMPLES_PER_SECOND) as u64,
    ));

    group.bench_function("32_voices_12_env_1_second", |b| {
        let config = FmEnvelopeConfig::adsr(0.1, 0.2, 0.7, 0.5, SAMPLE_RATE);

        let mut batches: Vec<EnvelopeBatch<8, 6, 2>> = (0..NUM_BATCHES_384)
            .map(|_| {
                let mut batch = FmEnvelopeBatch8::with_config(config);
                for i in 0..8 {
                    batch.note_on(i, 60);
                }
                batch
            })
            .collect();

        let mut output = [[0.0f32; 8]; BLOCK_SIZE];

        b.iter(|| {
            // Reset and trigger
            for batch in batches.iter_mut() {
                batch.reset_all();
                for i in 0..8 {
                    batch.note_on(i, 60);
                }
            }

            // Process ~500ms
            for _ in 0..(BLOCKS_PER_SECOND / 2) {
                for batch in batches.iter_mut() {
                    batch.process_block(&mut output);
                }
            }

            // Release
            for batch in batches.iter_mut() {
                for i in 0..8 {
                    batch.note_off(i);
                }
            }

            // Process ~500ms release
            for _ in 0..(BLOCKS_PER_SECOND / 2) {
                for batch in batches.iter_mut() {
                    batch.process_block(&mut output);
                }
            }

            black_box(output[0][0])
        })
    });

    // Segment transition stress test - many rapid segment changes
    group.bench_function("segment_transition_stress", |b| {
        // Very short ADSR to force rapid segment transitions
        let config = FmEnvelopeConfig::adsr(0.02, 0.03, 0.5, 0.05, SAMPLE_RATE);

        let mut batches: Vec<EnvelopeBatch<8, 6, 2>> = (0..48)
            .map(|_| FmEnvelopeBatch8::with_config(config))
            .collect();

        let mut output = [[0.0f32; 8]; BLOCK_SIZE];

        b.iter(|| {
            // Multiple note-on/off cycles to stress segment transitions
            for _cycle in 0..10 {
                // Trigger all
                for batch in batches.iter_mut() {
                    for i in 0..8 {
                        batch.note_on(i, 60);
                    }
                }

                // Process through attack/decay (short ~50ms)
                for _ in 0..35 {
                    for batch in batches.iter_mut() {
                        batch.process_block(&mut output);
                    }
                }

                // Release
                for batch in batches.iter_mut() {
                    for i in 0..8 {
                        batch.note_off(i);
                    }
                }

                // Process release (short ~50ms)
                for _ in 0..35 {
                    for batch in batches.iter_mut() {
                        batch.process_block(&mut output);
                    }
                }
            }

            black_box(output[0][0])
        })
    });

    group.finish();
}

// =============================================================================
// Level Conversion Benchmarks
// =============================================================================

fn bench_level_conversion(c: &mut Criterion) {
    use rigel_modulation::envelope::{level_to_linear, linear_to_param_level, param_to_level_q8};

    let mut group = c.benchmark_group("level_conversion");

    group.bench_function("param_to_level_q8", |b| {
        let params: Vec<u8> = (0..100).collect();
        let mut idx = 0;

        b.iter(|| {
            let param = params[idx % params.len()];
            idx += 1;
            black_box(param_to_level_q8(black_box(param)))
        })
    });

    group.bench_function("linear_to_param_level", |b| {
        let linears: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
        let mut idx = 0;

        b.iter(|| {
            let linear = linears[idx % linears.len()];
            idx += 1;
            black_box(linear_to_param_level(black_box(linear)))
        })
    });

    // Benchmark level_to_linear (the hot path we optimized)
    group.bench_function("level_to_linear", |b| {
        let levels: Vec<i16> = (0..4096).map(|i| i as i16).collect();
        let mut idx = 0;

        b.iter(|| {
            let level = levels[idx % levels.len()];
            idx += 1;
            black_box(level_to_linear(black_box(level)))
        })
    });

    // Benchmark level_to_linear with typical envelope levels (not edge cases)
    group.bench_function("level_to_linear_typical", |b| {
        // Typical levels seen during envelope processing (1716-4095 range)
        let levels: Vec<i16> = (1716..4096).map(|i| i as i16).collect();
        let mut idx = 0;

        b.iter(|| {
            let level = levels[idx % levels.len()];
            idx += 1;
            black_box(level_to_linear(black_box(level)))
        })
    });

    group.finish();
}

// =============================================================================
// Rate Calculation Benchmarks
// =============================================================================

fn bench_rate_calculations(c: &mut Criterion) {
    use rigel_modulation::envelope::{calculate_increment_f32, scale_rate, seconds_to_rate};

    let mut group = c.benchmark_group("rate_calculations");

    group.bench_function("scale_rate", |b| {
        let notes: Vec<u8> = (21..108).collect();
        let mut idx = 0;

        b.iter(|| {
            let note = notes[idx % notes.len()];
            idx += 1;
            black_box(scale_rate(black_box(note), 7))
        })
    });

    group.bench_function("calculate_increment_f32", |b| {
        let rates: Vec<u8> = (0..100).collect();
        let mut idx = 0;

        b.iter(|| {
            let rate = rates[idx % rates.len()];
            idx += 1;
            black_box(calculate_increment_f32(black_box(rate), 44100.0))
        })
    });

    group.bench_function("seconds_to_rate", |b| {
        let times: Vec<f32> = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
            .into_iter()
            .collect();
        let mut idx = 0;

        b.iter(|| {
            let time = times[idx % times.len()];
            idx += 1;
            black_box(seconds_to_rate(black_box(time), 44100.0))
        })
    });

    group.finish();
}

// =============================================================================
// Note Event Benchmarks
// =============================================================================

fn bench_note_events(c: &mut Criterion) {
    let mut group = c.benchmark_group("note_events");

    group.bench_function("note_on", |b| {
        let mut env = FmEnvelope::new(44100.0);

        b.iter(|| {
            env.reset();
            env.note_on(black_box(60));
            black_box(env.phase())
        })
    });

    group.bench_function("note_off", |b| {
        let mut env = FmEnvelope::new(44100.0);
        env.note_on(60);

        b.iter(|| {
            // Setup: ensure we're in KeyOn
            if !env.is_active() {
                env.note_on(60);
                for _ in 0..100 {
                    env.process();
                }
            }
            env.note_off();
            black_box(env.phase())
        })
    });

    group.finish();
}

// =============================================================================
// Control Rate vs Per-Sample Comparison Benchmarks
// =============================================================================

fn bench_control_rate_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("control_rate_vs_per_sample");

    // Throughput: 1 second of audio samples
    group.throughput(Throughput::Elements(SAMPLES_PER_SECOND as u64));

    // Per-sample baseline: standard envelope processing
    group.bench_function("per_sample_baseline", |b| {
        let config = FmEnvelopeConfig::adsr(0.1, 0.2, 0.7, 0.5, SAMPLE_RATE);
        let mut env = FmEnvelope::with_config(config);

        b.iter(|| {
            env.reset();
            env.note_on(60);

            // Process ~500ms key-on
            for _ in 0..22050 {
                black_box(env.process());
            }

            env.note_off();

            // Process ~500ms release
            for _ in 0..22050 {
                black_box(env.process());
            }

            env.value()
        })
    });

    // Control-rate with 64-sample interval (~1.45ms at 44.1kHz)
    group.bench_function("control_rate_64_samples", |b| {
        let config = FmEnvelopeConfig::adsr(0.1, 0.2, 0.7, 0.5, SAMPLE_RATE);
        let mut env = ControlRateFmEnvelope::new(config, 64);

        b.iter(|| {
            env.reset();
            env.note_on(60);

            // Process ~500ms key-on with control-rate ticks
            let key_on_blocks = 22050 / 64;
            for _ in 0..key_on_blocks {
                env.tick();
                for _ in 0..64 {
                    black_box(env.sample());
                }
            }

            env.note_off();

            // Process ~500ms release
            let release_blocks = 22050 / 64;
            for _ in 0..release_blocks {
                env.tick();
                for _ in 0..64 {
                    black_box(env.sample());
                }
            }

            env.current_value()
        })
    });

    // Control-rate with 32-sample interval (~0.73ms at 44.1kHz)
    group.bench_function("control_rate_32_samples", |b| {
        let config = FmEnvelopeConfig::adsr(0.1, 0.2, 0.7, 0.5, SAMPLE_RATE);
        let mut env = ControlRateFmEnvelope::new(config, 32);

        b.iter(|| {
            env.reset();
            env.note_on(60);

            // Process ~500ms key-on
            let key_on_blocks = 22050 / 32;
            for _ in 0..key_on_blocks {
                env.tick();
                for _ in 0..32 {
                    black_box(env.sample());
                }
            }

            env.note_off();

            // Process ~500ms release
            let release_blocks = 22050 / 32;
            for _ in 0..release_blocks {
                env.tick();
                for _ in 0..32 {
                    black_box(env.sample());
                }
            }

            env.current_value()
        })
    });

    // Control-rate with generate_block for efficient block processing
    group.bench_function("control_rate_64_block", |b| {
        let config = FmEnvelopeConfig::adsr(0.1, 0.2, 0.7, 0.5, SAMPLE_RATE);
        let mut env = ControlRateFmEnvelope::new(config, 64);
        let mut output = [0.0f32; 64];

        b.iter(|| {
            env.reset();
            env.note_on(60);

            // Process ~500ms key-on
            let key_on_blocks = 22050 / 64;
            for _ in 0..key_on_blocks {
                env.generate_block(&mut output);
            }

            env.note_off();

            // Process ~500ms release
            let release_blocks = 22050 / 64;
            for _ in 0..release_blocks {
                env.generate_block(&mut output);
            }

            black_box(output[0])
        })
    });

    group.finish();
}

// =============================================================================
// Polyphonic Control-Rate Benchmarks (768 envelopes = 64 voices × 12 operators)
// =============================================================================

fn bench_polyphonic_control_rate(c: &mut Criterion) {
    let mut group = c.benchmark_group("polyphonic_control_rate");

    const NUM_ENVELOPES: usize = 768;
    const UPDATE_INTERVAL: u32 = 64;

    // Throughput: 768 envelopes × 1 second
    group.throughput(Throughput::Elements(
        (NUM_ENVELOPES * SAMPLES_PER_SECOND) as u64,
    ));

    // Per-sample: 768 individual envelopes
    group.bench_function("per_sample_768_env", |b| {
        let config = FmEnvelopeConfig::adsr(0.1, 0.2, 0.7, 0.5, SAMPLE_RATE);
        let mut envelopes: Vec<FmEnvelope> = (0..NUM_ENVELOPES)
            .map(|_| FmEnvelope::with_config(config))
            .collect();

        b.iter(|| {
            // Reset and trigger all
            for env in envelopes.iter_mut() {
                env.reset();
                env.note_on(60);
            }

            // Process ~500ms key-on
            for _ in 0..22050 {
                for env in envelopes.iter_mut() {
                    black_box(env.process());
                }
            }

            // Release all
            for env in envelopes.iter_mut() {
                env.note_off();
            }

            // Process ~500ms release
            for _ in 0..22050 {
                for env in envelopes.iter_mut() {
                    black_box(env.process());
                }
            }

            envelopes[0].value()
        })
    });

    // Control-rate: 768 individual envelopes
    group.bench_function("control_rate_768_env", |b| {
        let config = FmEnvelopeConfig::adsr(0.1, 0.2, 0.7, 0.5, SAMPLE_RATE);
        let mut envelopes: Vec<ControlRateFmEnvelope> = (0..NUM_ENVELOPES)
            .map(|_| ControlRateFmEnvelope::new(config, UPDATE_INTERVAL))
            .collect();

        b.iter(|| {
            // Reset and trigger all
            for env in envelopes.iter_mut() {
                env.reset();
                env.note_on(60);
            }

            // Process ~500ms key-on
            let key_on_blocks = 22050 / UPDATE_INTERVAL as usize;
            for _ in 0..key_on_blocks {
                // Control-rate tick for all envelopes
                for env in envelopes.iter_mut() {
                    env.tick();
                }

                // Sample all envelopes
                for _ in 0..UPDATE_INTERVAL {
                    for env in envelopes.iter_mut() {
                        black_box(env.sample());
                    }
                }
            }

            // Release all
            for env in envelopes.iter_mut() {
                env.note_off();
            }

            // Process ~500ms release
            let release_blocks = 22050 / UPDATE_INTERVAL as usize;
            for _ in 0..release_blocks {
                for env in envelopes.iter_mut() {
                    env.tick();
                }

                for _ in 0..UPDATE_INTERVAL {
                    for env in envelopes.iter_mut() {
                        black_box(env.sample());
                    }
                }
            }

            envelopes[0].current_value()
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_single_envelope_process,
    bench_batch_envelope,
    bench_polyphonic_workload,
    bench_level_conversion,
    bench_rate_calculations,
    bench_note_events,
    bench_control_rate_comparison,
    bench_polyphonic_control_rate,
);

criterion_main!(benches);
