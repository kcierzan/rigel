//! LFO benchmarks.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rigel_modulation::{Lfo, LfoRateMode, LfoWaveshape, ModulationSource};
use rigel_timing::Timebase;

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
    let mut group = c.benchmark_group("waveshapes");

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

criterion_group!(
    benches,
    bench_lfo_update,
    bench_lfo_control_rate,
    bench_lfo_waveshapes
);
criterion_main!(benches);
