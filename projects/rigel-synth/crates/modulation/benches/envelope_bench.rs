//! Benchmarks for SY-Style Envelope module.
//!
//! Performance targets from spec:
//! - Single envelope: <50ns per sample
//! - 1536 envelopes × 64 samples: <100µs
//! - SIMD batch: 2x+ speedup over scalar

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use rigel_modulation::envelope::{EnvelopeBatch, FmEnvelope, FmEnvelopeBatch8};
use std::hint::black_box;

// =============================================================================
// Single Envelope Benchmarks
// =============================================================================

fn bench_single_envelope_process(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_envelope");
    group.throughput(Throughput::Elements(1));

    group.bench_function("process_sample", |b| {
        let mut env = FmEnvelope::new(44100.0);
        env.note_on(60);

        b.iter(|| black_box(env.process()))
    });

    group.bench_function("process_block_64", |b| {
        let mut env = FmEnvelope::new(44100.0);
        env.note_on(60);
        let mut output = [0.0f32; 64];

        b.iter(|| {
            env.process_block(&mut output);
            black_box(output[0])
        })
    });

    group.bench_function("process_block_128", |b| {
        let mut env = FmEnvelope::new(44100.0);
        env.note_on(60);
        let mut output = [0.0f32; 128];

        b.iter(|| {
            env.process_block(&mut output);
            black_box(output[0])
        })
    });

    group.finish();
}

// =============================================================================
// Batch Envelope Benchmarks
// =============================================================================

fn bench_batch_envelope(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_envelope");

    // Batch of 8 envelopes (AVX2 optimal)
    group.bench_function("batch_8_process", |b| {
        let mut batch = FmEnvelopeBatch8::new(44100.0);
        for i in 0..8 {
            batch.note_on(i, 60 + i as u8);
        }
        let mut output = [0.0f32; 8];

        b.iter(|| {
            batch.process(&mut output);
            black_box(output[0])
        })
    });

    group.bench_function("batch_8_block_64", |b| {
        let mut batch = FmEnvelopeBatch8::new(44100.0);
        for i in 0..8 {
            batch.note_on(i, 60 + i as u8);
        }
        let mut output = [[0.0f32; 8]; 64];

        b.iter(|| {
            batch.process_block(&mut output);
            black_box(output[0][0])
        })
    });

    group.finish();
}

// =============================================================================
// Polyphonic Workload Benchmarks
// =============================================================================

fn bench_polyphonic_workload(c: &mut Criterion) {
    let mut group = c.benchmark_group("polyphonic_workload");

    // 1536 envelopes × 64 samples (spec target: <100µs)
    // Using 192 batches of 8 envelopes each
    group.bench_function("1536_envelopes_64_samples", |b| {
        const NUM_BATCHES: usize = 192; // 192 * 8 = 1536 envelopes
        const BLOCK_SIZE: usize = 64;

        let mut batches: Vec<EnvelopeBatch<8, 6, 2>> = (0..NUM_BATCHES)
            .map(|_| {
                let mut batch = FmEnvelopeBatch8::new(44100.0);
                for i in 0..8 {
                    batch.note_on(i, 60 + (i % 12) as u8);
                }
                batch
            })
            .collect();

        let mut output = [[0.0f32; 8]; BLOCK_SIZE];

        b.iter(|| {
            for batch in batches.iter_mut() {
                batch.process_block(&mut output);
            }
            black_box(output[0][0])
        })
    });

    // 32 voices × 12 envelopes × 64 samples
    group.bench_function("32_voices_12_env_64_samples", |b| {
        const NUM_VOICES: usize = 32;
        const ENVS_PER_VOICE: usize = 12;
        const BLOCK_SIZE: usize = 64;

        // Use 48 batches of 8 (= 384 = 32 * 12 envelopes)
        const NUM_BATCHES: usize = (NUM_VOICES * ENVS_PER_VOICE + 7) / 8;

        let mut batches: Vec<EnvelopeBatch<8, 6, 2>> = (0..NUM_BATCHES)
            .map(|_| {
                let mut batch = FmEnvelopeBatch8::new(44100.0);
                for i in 0..8 {
                    batch.note_on(i, 60);
                }
                batch
            })
            .collect();

        let mut output = [[0.0f32; 8]; BLOCK_SIZE];

        b.iter(|| {
            for batch in batches.iter_mut() {
                batch.process_block(&mut output);
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
    use rigel_modulation::envelope::{level_to_linear, linear_to_level, LEVEL_MAX};

    let mut group = c.benchmark_group("level_conversion");

    group.bench_function("level_to_linear", |b| {
        let levels: Vec<i16> = (0..LEVEL_MAX).step_by(16).collect();
        let mut idx = 0;

        b.iter(|| {
            let level = levels[idx % levels.len()];
            idx += 1;
            black_box(level_to_linear(black_box(level)))
        })
    });

    group.bench_function("linear_to_level", |b| {
        let linears: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
        let mut idx = 0;

        b.iter(|| {
            let linear = linears[idx % linears.len()];
            idx += 1;
            black_box(linear_to_level(black_box(linear)))
        })
    });

    group.finish();
}

// =============================================================================
// Rate Calculation Benchmarks
// =============================================================================

fn bench_rate_calculations(c: &mut Criterion) {
    use rigel_modulation::envelope::{calculate_increment_q8, rate_to_qrate, scale_rate};

    let mut group = c.benchmark_group("rate_calculations");

    group.bench_function("rate_to_qrate", |b| {
        let rates: Vec<u8> = (0..100).collect();
        let mut idx = 0;

        b.iter(|| {
            let rate = rates[idx % rates.len()];
            idx += 1;
            black_box(rate_to_qrate(black_box(rate)))
        })
    });

    group.bench_function("scale_rate", |b| {
        let notes: Vec<u8> = (21..108).collect();
        let mut idx = 0;

        b.iter(|| {
            let note = notes[idx % notes.len()];
            idx += 1;
            black_box(scale_rate(black_box(note), 7))
        })
    });

    group.bench_function("calculate_increment_q8", |b| {
        let qrates: Vec<u8> = (0..64).collect();
        let mut idx = 0;

        b.iter(|| {
            let qrate = qrates[idx % qrates.len()];
            idx += 1;
            black_box(calculate_increment_q8(black_box(qrate)))
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

criterion_group!(
    benches,
    bench_single_envelope_process,
    bench_batch_envelope,
    bench_polyphonic_workload,
    bench_level_conversion,
    bench_rate_calculations,
    bench_note_events,
);

criterion_main!(benches);
