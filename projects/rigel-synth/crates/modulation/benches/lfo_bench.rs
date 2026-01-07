//! LFO benchmarks for real-time performance validation.
//!
//! These benchmarks validate that the LFO implementation is suitable for
//! heavy-polyphony scenarios (4 LFOs × 16 voices = 64 instances at 44.1kHz).
//!
//! ## Performance Targets
//!
//! | Metric | Target | Rationale |
//! |--------|--------|-----------|
//! | Single `update()` | < 500 ns | Control-rate overhead |
//! | Single `generate_block(64)` | < 500 ns | Block processing |
//! | 64 LFOs full cycle | < 64 µs | 1 µs per LFO budget |
//! | 64 LFOs 1-second simulation | < 50 ms | < 5% CPU at 44.1kHz |

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rigel_modulation::{
    InterpolationStrategy, Lfo, LfoRateMode, LfoWaveshape, ModulationSource, SimdXorshift128,
};
use rigel_simd_dispatch::SimdContext;
use rigel_timing::Timebase;
use std::hint::black_box;

// =============================================================================
// Existing Benchmarks (kept for regression tracking)
// =============================================================================

fn bench_lfo_update(c: &mut Criterion) {
    let mut lfo = Lfo::new();
    lfo.set_waveshape(LfoWaveshape::Sine);
    lfo.set_rate(LfoRateMode::Hz(1.0));

    let mut timebase = Timebase::new(44100.0);
    timebase.advance_block(64);

    c.bench_function("lfo_update_sine", |b| {
        b.iter(|| {
            lfo.update(black_box(&timebase));
            black_box(lfo.value())
        })
    });
}

fn bench_lfo_control_rate(c: &mut Criterion) {
    let mut lfo = Lfo::new();
    lfo.set_waveshape(LfoWaveshape::Sine);
    lfo.set_rate(LfoRateMode::Hz(1.0));

    let mut timebase = Timebase::new(44100.0);

    // Simulate 1024 sample block at 64-sample control rate (16 updates)
    c.bench_function("lfo_control_rate_64_1024_block", |b| {
        b.iter(|| {
            for _ in 0..16 {
                timebase.advance_block(64);
                lfo.update(black_box(&timebase));
            }
            black_box(lfo.value())
        })
    });
}

fn bench_lfo_waveshapes(c: &mut Criterion) {
    let mut group = c.benchmark_group("waveshapes_update");

    let waveshapes = [
        ("sine", LfoWaveshape::Sine),
        ("triangle", LfoWaveshape::Triangle),
        ("saw", LfoWaveshape::Saw),
        ("square", LfoWaveshape::Square),
        ("pulse", LfoWaveshape::Pulse),
        ("sample_hold", LfoWaveshape::SampleAndHold),
        ("noise", LfoWaveshape::Noise),
    ];

    for (name, waveshape) in waveshapes {
        let mut lfo = Lfo::new();
        lfo.set_waveshape(waveshape);
        lfo.set_rate(LfoRateMode::Hz(1.0));

        let mut timebase = Timebase::new(44100.0);
        timebase.advance_block(64);

        group.bench_function(name, |b| {
            b.iter(|| {
                lfo.update(black_box(&timebase));
                black_box(lfo.value())
            })
        });
    }

    group.finish();
}

// =============================================================================
// New Benchmarks: Block Generation (Critical Gap)
// =============================================================================

/// Benchmark generate_block() with different interpolation strategies and block sizes.
/// Target: Linear < 200 ns, CubicHermite < 400 ns for 64 samples.
fn bench_generate_block_interpolation(c: &mut Criterion) {
    let mut group = c.benchmark_group("generate_block_interpolation");
    let simd_ctx = SimdContext::new();

    let interpolations = [
        ("linear", InterpolationStrategy::Linear),
        ("cubic_hermite", InterpolationStrategy::CubicHermite),
    ];

    let block_sizes: [usize; 4] = [32, 64, 128, 256];

    for (interp_name, interp) in interpolations {
        for &size in &block_sizes {
            let mut lfo = Lfo::new();
            lfo.set_waveshape(LfoWaveshape::Sine);
            lfo.set_rate(LfoRateMode::Hz(5.0));
            lfo.set_interpolation(interp);

            let mut timebase = Timebase::new(44100.0);
            timebase.advance_block(size as u32);
            lfo.update(&timebase);

            let mut output = vec![0.0f32; size];

            group.throughput(Throughput::Elements(size as u64));
            group.bench_with_input(BenchmarkId::new(interp_name, size), &size, |b, _| {
                b.iter(|| {
                    lfo.generate_block(black_box(&mut output), &simd_ctx);
                    black_box(&output);
                })
            });
        }
    }

    group.finish();
}

/// Benchmark all waveshapes with generate_block() to identify outliers.
fn bench_waveshape_block_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("waveshape_generate_block");
    let simd_ctx = SimdContext::new();

    let waveshapes = [
        ("sine", LfoWaveshape::Sine),
        ("triangle", LfoWaveshape::Triangle),
        ("saw", LfoWaveshape::Saw),
        ("square", LfoWaveshape::Square),
        ("pulse", LfoWaveshape::Pulse),
        ("sample_hold", LfoWaveshape::SampleAndHold),
        ("noise", LfoWaveshape::Noise),
    ];

    for (name, waveshape) in waveshapes {
        let mut lfo = Lfo::new();
        lfo.set_waveshape(waveshape);
        lfo.set_rate(LfoRateMode::Hz(5.0));

        let mut timebase = Timebase::new(44100.0);
        timebase.advance_block(64);
        lfo.update(&timebase);

        let mut output = [0.0f32; 64];

        group.throughput(Throughput::Elements(64));
        group.bench_function(name, |b| {
            b.iter(|| {
                lfo.generate_block(black_box(&mut output), &simd_ctx);
                black_box(&output);
            })
        });
    }

    group.finish();
}

// =============================================================================
// New Benchmarks: Sample Method with Caching
// =============================================================================

/// Benchmark single-sample access via sample() method.
/// Target: Cache hit < 10 ns, 64 consecutive < 500 ns.
fn bench_sample_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("sample_access");

    // Single sample (cache hit after priming)
    {
        let mut lfo = Lfo::new();
        lfo.set_waveshape(LfoWaveshape::Sine);
        lfo.set_rate(LfoRateMode::Hz(5.0));

        let mut timebase = Timebase::new(44100.0);
        timebase.advance_block(64);
        lfo.update(&timebase);
        // Prime the cache
        let _ = lfo.sample();

        group.bench_function("cache_hit_single", |b| b.iter(|| black_box(lfo.sample())));
    }

    // 64 consecutive samples (full cache usage)
    {
        let mut lfo = Lfo::new();
        lfo.set_waveshape(LfoWaveshape::Sine);
        lfo.set_rate(LfoRateMode::Hz(5.0));

        let mut timebase = Timebase::new(44100.0);
        timebase.advance_block(64);
        lfo.update(&timebase);

        group.throughput(Throughput::Elements(64));
        group.bench_function("64_consecutive", |b| {
            b.iter(|| {
                let mut sum = 0.0f32;
                for _ in 0..64 {
                    sum += lfo.sample();
                }
                black_box(sum)
            })
        });
    }

    // Sample with cache miss (update invalidates cache)
    {
        let mut lfo = Lfo::new();
        lfo.set_waveshape(LfoWaveshape::Sine);
        lfo.set_rate(LfoRateMode::Hz(5.0));

        let mut timebase = Timebase::new(44100.0);

        group.bench_function("cache_miss_update_then_sample", |b| {
            b.iter(|| {
                timebase.advance_block(64);
                lfo.update(black_box(&timebase));
                black_box(lfo.sample())
            })
        });
    }

    group.finish();
}

// =============================================================================
// New Benchmarks: Multi-LFO Polyphony Scenarios (Critical)
// =============================================================================

/// Benchmark realistic polyphony scenarios: 4 LFOs per voice.
/// Target: 64 LFOs full cycle < 64 µs (1 µs per LFO).
fn bench_multi_lfo_polyphony(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_lfo_polyphony");
    let simd_ctx = SimdContext::new();

    // Voice counts: 4, 8, 16 voices with 4 LFOs each
    let voice_counts: [usize; 3] = [4, 8, 16];
    const LFOS_PER_VOICE: usize = 4;

    for &voices in &voice_counts {
        let lfo_count = voices * LFOS_PER_VOICE;

        // Create LFOs with varied configurations (realistic scenario)
        let mut lfos: Vec<Lfo> = (0..lfo_count)
            .map(|i| {
                let mut lfo = Lfo::new();
                // Vary waveshapes across the 4 LFOs per voice
                lfo.set_waveshape(match i % 4 {
                    0 => LfoWaveshape::Sine,     // Pitch mod
                    1 => LfoWaveshape::Triangle, // Filter mod
                    2 => LfoWaveshape::Saw,      // Pan mod
                    _ => LfoWaveshape::Square,   // Gate mod
                });
                // Vary rates
                lfo.set_rate(LfoRateMode::Hz(0.5 + (i as f32 * 0.1)));
                lfo
            })
            .collect();

        let mut timebase = Timebase::new(44100.0);
        let mut outputs: Vec<[f32; 64]> = vec![[0.0; 64]; lfo_count];

        // Benchmark: update all LFOs
        let label = format!("{}v_{}lfos", voices, lfo_count);
        group.throughput(Throughput::Elements(lfo_count as u64));
        group.bench_with_input(
            BenchmarkId::new("update_all", &label),
            &lfo_count,
            |b, _| {
                b.iter(|| {
                    timebase.advance_block(64);
                    for lfo in lfos.iter_mut() {
                        lfo.update(black_box(&timebase));
                    }
                })
            },
        );

        // Benchmark: generate_block for all LFOs
        // First do an update to ensure valid state
        timebase.advance_block(64);
        for lfo in lfos.iter_mut() {
            lfo.update(&timebase);
        }

        group.bench_with_input(
            BenchmarkId::new("generate_all", &label),
            &lfo_count,
            |b, _| {
                b.iter(|| {
                    for (lfo, output) in lfos.iter().zip(outputs.iter_mut()) {
                        lfo.generate_block(black_box(output), &simd_ctx);
                    }
                    black_box(&outputs);
                })
            },
        );

        // Benchmark: full cycle (update + generate) - most realistic
        group.bench_with_input(
            BenchmarkId::new("full_cycle", &label),
            &lfo_count,
            |b, _| {
                b.iter(|| {
                    timebase.advance_block(64);
                    for (lfo, output) in lfos.iter_mut().zip(outputs.iter_mut()) {
                        lfo.update(black_box(&timebase));
                        lfo.generate_block(black_box(output), &simd_ctx);
                    }
                    black_box(&outputs);
                })
            },
        );
    }

    group.finish();
}

/// Benchmark 64 LFOs processing 1 second of audio at control rate.
/// Target: < 50 ms (< 5% CPU at 44.1kHz).
fn bench_control_rate_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("control_rate_efficiency");
    let simd_ctx = SimdContext::new();

    const LFO_COUNT: usize = 64;
    const BLOCK_SIZE: usize = 64;
    const SAMPLE_RATE: usize = 44100;
    const BLOCKS_PER_SECOND: usize = SAMPLE_RATE / BLOCK_SIZE; // 689

    // Create 64 LFOs with mixed configurations
    let mut lfos: Vec<Lfo> = (0..LFO_COUNT)
        .map(|i| {
            let mut lfo = Lfo::new();
            lfo.set_waveshape(match i % 7 {
                0 => LfoWaveshape::Sine,
                1 => LfoWaveshape::Triangle,
                2 => LfoWaveshape::Saw,
                3 => LfoWaveshape::Square,
                4 => LfoWaveshape::Pulse,
                5 => LfoWaveshape::SampleAndHold,
                _ => LfoWaveshape::Noise,
            });
            lfo.set_rate(LfoRateMode::Hz(0.1 + (i as f32 * 0.05)));
            lfo
        })
        .collect();

    let mut timebase = Timebase::new(44100.0);
    let mut outputs: Vec<[f32; 64]> = vec![[0.0; 64]; LFO_COUNT];

    // 1 second of audio at control rate
    group.throughput(Throughput::Elements((LFO_COUNT * BLOCKS_PER_SECOND) as u64));
    group.bench_function("64_lfos_1_second", |b| {
        b.iter(|| {
            for _ in 0..BLOCKS_PER_SECOND {
                timebase.advance_block(64);
                for (lfo, output) in lfos.iter_mut().zip(outputs.iter_mut()) {
                    lfo.update(black_box(&timebase));
                    lfo.generate_block(black_box(output), &simd_ctx);
                }
            }
            black_box(&outputs);
        })
    });

    group.finish();
}

// =============================================================================
// New Benchmarks: SIMD RNG Performance
// =============================================================================

/// Benchmark the SIMD Xorshift128+ RNG used for Noise waveshape.
fn bench_simd_rng(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_rng");

    let mut rng = SimdXorshift128::new(12345);

    // Single f32 value
    group.bench_function("next_f32_single", |b| b.iter(|| black_box(rng.next_f32())));

    // Full lane (SIMD width)
    group.bench_function("next_lane_f32", |b| {
        b.iter(|| black_box(rng.next_lane_f32()))
    });

    // Buffer fills at various sizes
    let buffer_sizes: [usize; 3] = [64, 256, 1024];
    for &size in &buffer_sizes {
        let mut buffer = vec![0.0f32; size];

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("fill_buffer", size), &size, |b, _| {
            b.iter(|| {
                rng.fill_buffer(black_box(&mut buffer));
                black_box(&buffer);
            })
        });
    }

    group.finish();
}

// =============================================================================
// Criterion Configuration
// =============================================================================

criterion_group!(
    benches,
    // Existing benchmarks
    bench_lfo_update,
    bench_lfo_control_rate,
    bench_lfo_waveshapes,
    // New block generation benchmarks
    bench_generate_block_interpolation,
    bench_waveshape_block_generation,
    // New sample access benchmarks
    bench_sample_access,
    // New polyphony benchmarks (critical)
    bench_multi_lfo_polyphony,
    bench_control_rate_efficiency,
    // New SIMD RNG benchmarks
    bench_simd_rng,
);

criterion_main!(benches);
