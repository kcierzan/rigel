use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rigel_dsp::{
    clamp, lerp, midi_to_freq, soft_clip, ControlRateClock, Envelope, NoteNumber, SimpleOscillator,
    Smoother, SmoothingMode, SynthEngine, SynthParams, Timebase,
};
use std::time::Duration;

fn configure_criterion() -> Criterion {
    Criterion::default()
        .measurement_time(Duration::from_secs(5))
        .warm_up_time(Duration::from_secs(1))
        .sample_size(100)
        .noise_threshold(0.05) // 5% threshold for DSP
}

// ==============================================================================
// Utility Functions Benchmarks
// ==============================================================================

fn bench_utility_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("utility_functions");

    // MIDI to frequency conversion
    group.bench_function("midi_to_freq_middle_c", |b| {
        b.iter(|| black_box(midi_to_freq(black_box(60))))
    });

    group.bench_function("midi_to_freq_low_note", |b| {
        b.iter(|| black_box(midi_to_freq(black_box(21))))
    });

    group.bench_function("midi_to_freq_high_note", |b| {
        b.iter(|| black_box(midi_to_freq(black_box(108))))
    });

    // Linear interpolation
    group.bench_function("lerp", |b| {
        b.iter(|| black_box(lerp(black_box(0.0), black_box(1.0), black_box(0.5))))
    });

    // Clamping - Current implementation
    group.bench_function("clamp_in_range", |b| {
        b.iter(|| black_box(clamp(black_box(0.5), black_box(0.0), black_box(1.0))))
    });

    group.bench_function("clamp_below_min", |b| {
        b.iter(|| black_box(clamp(black_box(-0.5), black_box(0.0), black_box(1.0))))
    });

    group.bench_function("clamp_above_max", |b| {
        b.iter(|| black_box(clamp(black_box(1.5), black_box(0.0), black_box(1.0))))
    });

    // Soft clipping
    group.bench_function("soft_clip_in_range", |b| {
        b.iter(|| black_box(soft_clip(black_box(0.5))))
    });

    group.bench_function("soft_clip_above_max", |b| {
        b.iter(|| black_box(soft_clip(black_box(2.0))))
    });

    group.bench_function("soft_clip_below_min", |b| {
        b.iter(|| black_box(soft_clip(black_box(-2.0))))
    });

    group.finish();
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

fn bench_clamp_implementations(c: &mut Criterion) {
    let mut group = c.benchmark_group("clamp_implementations");

    // Test cases: in_range, below_min, above_max
    let test_cases = [("in_range", 0.5), ("below_min", -0.5), ("above_max", 1.5)];

    for (name, value) in test_cases.iter() {
        // Current implementation (branching)
        group.bench_function(format!("branching_{}", name), |b| {
            b.iter(|| {
                black_box(clamp_branching(
                    black_box(*value),
                    black_box(0.0),
                    black_box(1.0),
                ))
            })
        });

        // Branchless using max/min
        group.bench_function(format!("max_min_{}", name), |b| {
            b.iter(|| {
                black_box(clamp_branchless_max_min(
                    black_box(*value),
                    black_box(0.0),
                    black_box(1.0),
                ))
            })
        });

        // Built-in clamp
        group.bench_function(format!("builtin_{}", name), |b| {
            b.iter(|| {
                black_box(clamp_builtin(
                    black_box(*value),
                    black_box(0.0),
                    black_box(1.0),
                ))
            })
        });
    }

    // Realistic usage pattern: mix of all three cases
    group.bench_function("branching_mixed", |b| {
        let values = [-0.5, 0.3, 0.7, 1.5, 0.1, 2.0, -1.0, 0.5];
        b.iter(|| {
            let mut sum = 0.0;
            for &v in values.iter() {
                sum += clamp_branching(black_box(v), black_box(0.0), black_box(1.0));
            }
            black_box(sum)
        })
    });

    group.bench_function("max_min_mixed", |b| {
        let values = [-0.5, 0.3, 0.7, 1.5, 0.1, 2.0, -1.0, 0.5];
        b.iter(|| {
            let mut sum = 0.0;
            for &v in values.iter() {
                sum += clamp_branchless_max_min(black_box(v), black_box(0.0), black_box(1.0));
            }
            black_box(sum)
        })
    });

    group.bench_function("builtin_mixed", |b| {
        let values = [-0.5, 0.3, 0.7, 1.5, 0.1, 2.0, -1.0, 0.5];
        b.iter(|| {
            let mut sum = 0.0;
            for &v in values.iter() {
                sum += clamp_builtin(black_box(v), black_box(0.0), black_box(1.0));
            }
            black_box(sum)
        })
    });

    group.finish();
}

// ==============================================================================
// Oscillator Benchmarks
// ==============================================================================

fn bench_oscillator(c: &mut Criterion) {
    let mut group = c.benchmark_group("oscillator");

    // Single sample generation
    group.throughput(Throughput::Elements(1));
    group.bench_function("single_sample", |b| {
        let mut osc = SimpleOscillator::new();
        osc.set_frequency(440.0, 44100.0);
        b.iter(|| black_box(osc.process_sample()))
    });

    // Amortized overhead (100 samples)
    group.throughput(Throughput::Elements(100));
    group.bench_function("100_samples", |b| {
        let mut osc = SimpleOscillator::new();
        osc.set_frequency(440.0, 44100.0);
        b.iter(|| {
            for _ in 0..100 {
                black_box(osc.process_sample());
            }
        });
    });

    // One second at 44.1kHz
    group.throughput(Throughput::Elements(44100));
    group.bench_function("one_second_44_1k", |b| {
        let mut osc = SimpleOscillator::new();
        osc.set_frequency(440.0, 44100.0);
        b.iter(|| {
            for _ in 0..44100 {
                black_box(osc.process_sample());
            }
        });
    });

    // Frequency changes
    group.bench_function("set_frequency", |b| {
        let mut osc = SimpleOscillator::new();
        b.iter(|| osc.set_frequency(black_box(440.0), black_box(44100.0)))
    });

    group.bench_function("set_frequency_and_sample", |b| {
        let mut osc = SimpleOscillator::new();
        b.iter(|| {
            osc.set_frequency(black_box(440.0), black_box(44100.0));
            black_box(osc.process_sample())
        })
    });

    group.finish();
}

// ==============================================================================
// Envelope Benchmarks
// ==============================================================================

fn bench_envelope(c: &mut Criterion) {
    let mut group = c.benchmark_group("envelope");
    let sample_rate = 44100.0;

    // Single sample processing in each stage
    group.throughput(Throughput::Elements(1));

    // Attack stage
    group.bench_function("attack_stage", |b| {
        let mut env = Envelope::new(sample_rate);
        let params = SynthParams {
            env_attack: 0.1,
            ..Default::default()
        };
        env.note_on();
        b.iter(|| black_box(env.process_sample(black_box(&params))))
    });

    // Decay stage
    group.bench_function("decay_stage", |b| {
        let mut env = Envelope::new(sample_rate);
        let params = SynthParams {
            env_attack: 0.0,
            env_decay: 0.1,
            env_sustain: 0.5,
            ..Default::default()
        };
        env.note_on();
        // Process through attack instantly
        for _ in 0..10 {
            env.process_sample(&params);
        }
        b.iter(|| black_box(env.process_sample(black_box(&params))))
    });

    // Sustain stage
    group.bench_function("sustain_stage", |b| {
        let mut env = Envelope::new(sample_rate);
        let params = SynthParams {
            env_attack: 0.0,
            env_decay: 0.0,
            env_sustain: 0.7,
            ..Default::default()
        };
        env.note_on();
        // Process through attack and decay
        for _ in 0..100 {
            env.process_sample(&params);
        }
        b.iter(|| black_box(env.process_sample(black_box(&params))))
    });

    // Release stage
    group.bench_function("release_stage", |b| {
        let mut env = Envelope::new(sample_rate);
        let params = SynthParams {
            env_attack: 0.0,
            env_release: 0.1,
            ..Default::default()
        };
        env.note_on();
        env.note_off();
        b.iter(|| black_box(env.process_sample(black_box(&params))))
    });

    // Full ADSR cycle
    group.bench_function("full_adsr_cycle", |b| {
        let mut env = Envelope::new(sample_rate);
        let params = SynthParams {
            env_attack: 0.01,
            env_decay: 0.1,
            env_sustain: 0.7,
            env_release: 0.2,
            ..Default::default()
        };

        b.iter(|| {
            env.note_on();
            // Attack + Decay
            for _ in 0..(sample_rate as usize / 10) {
                black_box(env.process_sample(&params));
            }
            // Sustain
            for _ in 0..1000 {
                black_box(env.process_sample(&params));
            }
            // Release
            env.note_off();
            for _ in 0..(sample_rate as usize / 5) {
                black_box(env.process_sample(&params));
            }
        })
    });

    // Fast envelope (percussive)
    group.bench_function("percussive_envelope", |b| {
        let mut env = Envelope::new(sample_rate);
        let params = SynthParams {
            env_attack: 0.001,
            env_decay: 0.05,
            env_sustain: 0.0,
            env_release: 0.1,
            ..Default::default()
        };

        b.iter(|| {
            env.note_on();
            for _ in 0..4410 {
                // 0.1 seconds
                black_box(env.process_sample(&params));
            }
        })
    });

    // Slow envelope (pad)
    group.bench_function("pad_envelope", |b| {
        let mut env = Envelope::new(sample_rate);
        let params = SynthParams {
            env_attack: 0.5,
            env_decay: 0.5,
            env_sustain: 0.8,
            env_release: 1.0,
            ..Default::default()
        };

        b.iter(|| {
            env.note_on();
            for _ in 0..22050 {
                // 0.5 seconds
                black_box(env.process_sample(&params));
            }
        })
    });

    group.finish();
}

// ==============================================================================
// SynthEngine Benchmarks
// ==============================================================================

fn bench_synth_engine_single(c: &mut Criterion) {
    let mut group = c.benchmark_group("synth_engine");
    let sample_rate = 44100.0;
    let params = SynthParams::default();

    // Single sample processing
    group.throughput(Throughput::Elements(1));
    group.bench_function("single_sample", |b| {
        let mut engine = SynthEngine::new(sample_rate);
        engine.note_on(60, 0.8);
        b.iter(|| black_box(engine.process_sample(black_box(&params))))
    });

    // Note lifecycle
    group.bench_function("note_on", |b| {
        let mut engine = SynthEngine::new(sample_rate);
        b.iter(|| {
            engine.note_on(black_box(60), black_box(0.8));
        })
    });

    group.bench_function("note_off", |b| {
        let mut engine = SynthEngine::new(sample_rate);
        engine.note_on(60, 0.8);
        b.iter(|| {
            engine.note_off(black_box(60));
        })
    });

    group.bench_function("note_on_to_off_cycle", |b| {
        let mut engine = SynthEngine::new(sample_rate);
        b.iter(|| {
            engine.note_on(black_box(60), black_box(0.8));
            for _ in 0..441 {
                // 10ms of audio
                black_box(engine.process_sample(black_box(&params)));
            }
            engine.note_off(black_box(60));
            for _ in 0..441 {
                black_box(engine.process_sample(black_box(&params)));
            }
        })
    });

    // Rapid retriggering
    group.bench_function("rapid_retriggering", |b| {
        let mut engine = SynthEngine::new(sample_rate);
        b.iter(|| {
            for _ in 0..10 {
                engine.note_on(black_box(60), black_box(0.8));
                for _ in 0..44 {
                    // 1ms between triggers
                    black_box(engine.process_sample(black_box(&params)));
                }
            }
        })
    });

    // Pitch modulation
    group.bench_function("with_pitch_modulation", |b| {
        let mut engine = SynthEngine::new(sample_rate);
        engine.note_on(60, 0.8);
        let mut params = SynthParams::default();
        b.iter(|| {
            params.pitch_offset = black_box(2.0); // +2 semitones
            black_box(engine.process_sample(black_box(&params)))
        })
    });

    group.finish();
}

fn bench_synth_engine_buffers(c: &mut Criterion) {
    let mut group = c.benchmark_group("synth_engine_buffers");
    let sample_rate = 44100.0;
    let params = SynthParams::default();
    let buffer_sizes = [64, 128, 256, 512, 1024, 2048];

    for size in buffer_sizes.iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // Standard buffer processing
        group.bench_with_input(
            BenchmarkId::new("buffer_processing", size),
            size,
            |b, &size| {
                let mut engine = SynthEngine::new(sample_rate);
                engine.note_on(60, 0.8);

                b.iter(|| {
                    for _ in 0..size {
                        black_box(engine.process_sample(black_box(&params)));
                    }
                });
            },
        );

        // Sustained note (envelope in sustain stage)
        group.bench_with_input(
            BenchmarkId::new("sustained_note", size),
            size,
            |b, &size| {
                let mut engine = SynthEngine::new(sample_rate);
                engine.note_on(60, 0.8);
                // Process enough samples to reach sustain stage
                for _ in 0..10000 {
                    engine.process_sample(&params);
                }

                b.iter(|| {
                    for _ in 0..size {
                        black_box(engine.process_sample(black_box(&params)));
                    }
                });
            },
        );

        // Percussive note (fast attack/release)
        group.bench_with_input(
            BenchmarkId::new("percussive_note", size),
            size,
            |b, &size| {
                let mut engine = SynthEngine::new(sample_rate);
                let params = SynthParams {
                    env_attack: 0.001,
                    env_decay: 0.05,
                    env_sustain: 0.0,
                    env_release: 0.1,
                    ..Default::default()
                };

                b.iter(|| {
                    engine.note_on(60, 0.8);
                    for _ in 0..size {
                        black_box(engine.process_sample(black_box(&params)));
                    }
                });
            },
        );
    }

    group.finish();
}

// ==============================================================================
// Timebase Benchmarks
// ==============================================================================

fn bench_timebase(c: &mut Criterion) {
    let mut group = c.benchmark_group("timebase");

    // Single advance_block call
    group.throughput(Throughput::Elements(1));
    group.bench_function("advance_block_64", |b| {
        let mut timebase = Timebase::new(44100.0);
        b.iter(|| {
            timebase.advance_block(black_box(64));
        })
    });

    group.bench_function("advance_block_128", |b| {
        let mut timebase = Timebase::new(44100.0);
        b.iter(|| {
            timebase.advance_block(black_box(128));
        })
    });

    group.bench_function("advance_block_256", |b| {
        let mut timebase = Timebase::new(44100.0);
        b.iter(|| {
            timebase.advance_block(black_box(256));
        })
    });

    // Time conversions
    group.bench_function("samples_to_seconds", |b| {
        let timebase = Timebase::new(44100.0);
        b.iter(|| black_box(timebase.samples_to_seconds(black_box(44100))))
    });

    group.bench_function("seconds_to_samples", |b| {
        let timebase = Timebase::new(44100.0);
        b.iter(|| black_box(timebase.seconds_to_samples(black_box(1.0))))
    });

    group.bench_function("ms_to_samples", |b| {
        let timebase = Timebase::new(44100.0);
        b.iter(|| black_box(timebase.ms_to_samples(black_box(10.0))))
    });

    group.finish();
}

// ==============================================================================
// Smoother Benchmarks
// ==============================================================================

fn bench_smoother_single(c: &mut Criterion) {
    let mut group = c.benchmark_group("smoother_single");
    let sample_rate = 44100.0;

    // Single sample processing for each mode
    group.throughput(Throughput::Elements(1));

    // Linear mode
    group.bench_function("linear_active", |b| {
        let mut smoother = Smoother::new(0.0, SmoothingMode::Linear, 10.0, sample_rate);
        smoother.set_target(1.0);
        b.iter(|| black_box(smoother.process_sample()))
    });

    group.bench_function("linear_inactive", |b| {
        let smoother = Smoother::new(0.5, SmoothingMode::Linear, 10.0, sample_rate);
        let mut s = smoother;
        b.iter(|| black_box(s.process_sample()))
    });

    // Exponential mode
    group.bench_function("exponential_active", |b| {
        let mut smoother = Smoother::new(0.0, SmoothingMode::Exponential, 10.0, sample_rate);
        smoother.set_target(1.0);
        b.iter(|| black_box(smoother.process_sample()))
    });

    group.bench_function("exponential_inactive", |b| {
        let smoother = Smoother::new(0.5, SmoothingMode::Exponential, 10.0, sample_rate);
        let mut s = smoother;
        b.iter(|| black_box(s.process_sample()))
    });

    // Logarithmic mode
    group.bench_function("logarithmic_active", |b| {
        let mut smoother = Smoother::new(100.0, SmoothingMode::Logarithmic, 10.0, sample_rate);
        smoother.set_target(1000.0);
        b.iter(|| black_box(smoother.process_sample()))
    });

    group.bench_function("logarithmic_inactive", |b| {
        let smoother = Smoother::new(500.0, SmoothingMode::Logarithmic, 10.0, sample_rate);
        let mut s = smoother;
        b.iter(|| black_box(s.process_sample()))
    });

    // Instant mode (should be fastest)
    group.bench_function("instant", |b| {
        let mut smoother = Smoother::new(0.0, SmoothingMode::Instant, 10.0, sample_rate);
        smoother.set_target(1.0);
        b.iter(|| black_box(smoother.process_sample()))
    });

    // set_target overhead
    group.bench_function("set_target_linear", |b| {
        let mut smoother = Smoother::new(0.0, SmoothingMode::Linear, 10.0, sample_rate);
        let mut target = 0.0;
        b.iter(|| {
            target = 1.0 - target;
            smoother.set_target(black_box(target));
        })
    });

    group.finish();
}

fn bench_smoother_block(c: &mut Criterion) {
    let mut group = c.benchmark_group("smoother_block");
    let sample_rate = 44100.0;
    let buffer_sizes = [64, 128, 256];

    for size in buffer_sizes.iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // Linear mode block processing
        group.bench_with_input(
            BenchmarkId::new("linear_active", size),
            size,
            |b, &size| {
                let mut smoother = Smoother::new(0.0, SmoothingMode::Linear, 10.0, sample_rate);
                smoother.set_target(1.0);
                let mut buffer = vec![0.0f32; size];

                b.iter(|| {
                    smoother.process_block(black_box(&mut buffer));
                });
            },
        );

        // Exponential mode block processing
        group.bench_with_input(
            BenchmarkId::new("exponential_active", size),
            size,
            |b, &size| {
                let mut smoother =
                    Smoother::new(0.0, SmoothingMode::Exponential, 10.0, sample_rate);
                smoother.set_target(1.0);
                let mut buffer = vec![0.0f32; size];

                b.iter(|| {
                    smoother.process_block(black_box(&mut buffer));
                });
            },
        );

        // Inactive smoother (should be very fast - just fills buffer)
        group.bench_with_input(BenchmarkId::new("inactive", size), size, |b, &size| {
            let smoother = Smoother::new(0.5, SmoothingMode::Linear, 10.0, sample_rate);
            let mut s = smoother;
            let mut buffer = vec![0.0f32; size];

            b.iter(|| {
                s.process_block(black_box(&mut buffer));
            });
        });
    }

    group.finish();
}

// ==============================================================================
// ControlRateClock Benchmarks
// ==============================================================================

fn bench_control_rate_clock(c: &mut Criterion) {
    let mut group = c.benchmark_group("control_rate_clock");

    // Single advance call
    group.throughput(Throughput::Elements(1));

    // Different intervals
    let intervals = [32, 64, 128];

    for interval in intervals.iter() {
        group.bench_with_input(
            BenchmarkId::new("advance_64_samples", interval),
            interval,
            |b, &interval| {
                let mut clock = ControlRateClock::new(interval);
                b.iter(|| {
                    for _ in clock.advance(black_box(64)) {
                        black_box(());
                    }
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("advance_128_samples", interval),
            interval,
            |b, &interval| {
                let mut clock = ControlRateClock::new(interval);
                b.iter(|| {
                    for _ in clock.advance(black_box(128)) {
                        black_box(());
                    }
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("advance_256_samples", interval),
            interval,
            |b, &interval| {
                let mut clock = ControlRateClock::new(interval);
                b.iter(|| {
                    for _ in clock.advance(black_box(256)) {
                        black_box(());
                    }
                });
            },
        );
    }

    // Iterator overhead (count vs explicit loop)
    group.bench_function("iterator_count_64_interval", |b| {
        let mut clock = ControlRateClock::new(64);
        b.iter(|| black_box(clock.advance(black_box(128)).count()));
    });

    group.bench_function("iterator_collect_64_interval", |b| {
        let mut clock = ControlRateClock::new(64);
        b.iter(|| {
            let offsets: Vec<u32> = clock.advance(black_box(128)).collect();
            black_box(offsets);
        });
    });

    group.finish();
}

// ==============================================================================
// Throughput and CPU Usage Validation
// ==============================================================================

fn bench_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput");
    let params = SynthParams::default();

    // Different sample rates
    let sample_rates = [44100.0, 48000.0, 96000.0];

    for &rate in &sample_rates {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("sample_rate", rate as u32),
            &rate,
            |b, &rate| {
                let mut engine = SynthEngine::new(rate);
                engine.note_on(60, 0.8);

                b.iter(|| black_box(engine.process_sample(black_box(&params))));
            },
        );
    }

    group.finish();
}

fn bench_cpu_usage_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_usage_validation");
    let sample_rate = 44100.0;
    let params = SynthParams::default();

    // Calculate available time per sample at 44.1kHz
    // 1 / 44100 = 22.675 microseconds per sample
    // For 0.1% CPU usage, processing should take < 22.7 nanoseconds
    // For 1% CPU usage, processing should take < 227 nanoseconds
    // For 10% CPU usage, processing should take < 2.27 microseconds

    group.throughput(Throughput::Elements(1));
    group.bench_function("single_voice_processing_time", |b| {
        let mut engine = SynthEngine::new(sample_rate);
        engine.note_on(60, 0.8);

        b.iter(|| black_box(engine.process_sample(black_box(&params))));
    });

    // Extrapolate to polyphonic usage
    let voice_counts = [1, 4, 8, 16];
    for &voices in &voice_counts {
        group.bench_with_input(
            BenchmarkId::new("polyphonic_extrapolation", voices),
            &voices,
            |b, &voices| {
                let mut engines: Vec<SynthEngine> = (0..voices)
                    .map(|i| {
                        let mut engine = SynthEngine::new(sample_rate);
                        engine.note_on(60 + (i % 12) as NoteNumber, 0.8);
                        engine
                    })
                    .collect();

                b.iter(|| {
                    for engine in engines.iter_mut() {
                        black_box(engine.process_sample(black_box(&params)));
                    }
                });
            },
        );
    }

    group.finish();
}

// ==============================================================================
// Criterion Configuration
// ==============================================================================

criterion_group! {
    name = benches;
    config = configure_criterion();
    targets = bench_utility_functions,
              bench_clamp_implementations,
              bench_oscillator,
              bench_envelope,
              bench_synth_engine_single,
              bench_synth_engine_buffers,
              bench_timebase,
              bench_smoother_single,
              bench_smoother_block,
              bench_control_rate_clock,
              bench_throughput,
              bench_cpu_usage_validation
}

criterion_main!(benches);
