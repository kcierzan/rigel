//! iai-callgrind benchmarks for rigel-math (T028)
//!
//! Measures instruction counts for vector operations (deterministic, cachegrind-based).
//! Run with: cargo bench --bench iai_benches

use iai_callgrind::{library_benchmark, library_benchmark_group, main};
use rigel_math::ops::{add, clamp, div, fma, max, min, mul, sub};
use rigel_math::{DefaultSimdVector, SimdVector};
use std::hint::black_box;

// Arithmetic operations

#[library_benchmark]
fn bench_add() -> DefaultSimdVector {
    let a = black_box(DefaultSimdVector::splat(2.0));
    let b = black_box(DefaultSimdVector::splat(3.0));
    black_box(a.add(b))
}

#[library_benchmark]
fn bench_sub() -> DefaultSimdVector {
    let a = black_box(DefaultSimdVector::splat(2.0));
    let b = black_box(DefaultSimdVector::splat(3.0));
    black_box(a.sub(b))
}

#[library_benchmark]
fn bench_mul() -> DefaultSimdVector {
    let a = black_box(DefaultSimdVector::splat(2.0));
    let b = black_box(DefaultSimdVector::splat(3.0));
    black_box(a.mul(b))
}

#[library_benchmark]
fn bench_div() -> DefaultSimdVector {
    let a = black_box(DefaultSimdVector::splat(2.0));
    let b = black_box(DefaultSimdVector::splat(3.0));
    black_box(a.div(b))
}

#[library_benchmark]
fn bench_fma() -> DefaultSimdVector {
    let a = black_box(DefaultSimdVector::splat(2.0));
    let b = black_box(DefaultSimdVector::splat(3.0));
    let c = black_box(DefaultSimdVector::splat(1.0));
    black_box(a.fma(b, c))
}

// Min/Max operations

#[library_benchmark]
fn bench_min() -> DefaultSimdVector {
    let a = black_box(DefaultSimdVector::splat(2.0));
    let b = black_box(DefaultSimdVector::splat(3.0));
    black_box(a.min(b))
}

#[library_benchmark]
fn bench_max() -> DefaultSimdVector {
    let a = black_box(DefaultSimdVector::splat(2.0));
    let b = black_box(DefaultSimdVector::splat(3.0));
    black_box(a.max(b))
}

// Horizontal operations

#[library_benchmark]
fn bench_horizontal_sum() -> f32 {
    let vec = black_box(DefaultSimdVector::splat(2.0));
    black_box(vec.horizontal_sum())
}

#[library_benchmark]
fn bench_horizontal_max() -> f32 {
    let vec = black_box(DefaultSimdVector::splat(2.0));
    black_box(vec.horizontal_max())
}

#[library_benchmark]
fn bench_horizontal_min() -> f32 {
    let vec = black_box(DefaultSimdVector::splat(2.0));
    black_box(vec.horizontal_min())
}

// Memory operations

#[library_benchmark]
fn bench_from_slice() -> DefaultSimdVector {
    let data = black_box([1.0f32; 16]); // Large enough for all backends
    black_box(DefaultSimdVector::from_slice(&data))
}

#[library_benchmark]
fn bench_to_slice() -> [f32; 16] {
    let vec = black_box(DefaultSimdVector::splat(2.0));
    let mut output = black_box([0.0f32; 16]);
    vec.to_slice(&mut output);
    black_box(output)
}

// Block processing

#[library_benchmark]
fn bench_block_processing_64() -> Vec<f32> {
    const BLOCK_SIZE: usize = 64;
    let input = black_box(vec![1.0f32; BLOCK_SIZE]);
    let mut output = black_box(vec![0.0f32; BLOCK_SIZE]);
    let gain = black_box(DefaultSimdVector::splat(0.5));

    let lanes = DefaultSimdVector::LANES;
    for chunk_start in (0..BLOCK_SIZE).step_by(lanes) {
        if chunk_start + lanes <= BLOCK_SIZE {
            let input_vec = DefaultSimdVector::from_slice(&input[chunk_start..]);
            let output_vec = input_vec.mul(gain);
            output_vec.to_slice(&mut output[chunk_start..]);
        }
    }

    black_box(output)
}

library_benchmark_group!(
    name = arithmetic_group;
    benchmarks = bench_add, bench_sub, bench_mul, bench_div, bench_fma
);

library_benchmark_group!(
    name = minmax_group;
    benchmarks = bench_min, bench_max
);

library_benchmark_group!(
    name = horizontal_group;
    benchmarks = bench_horizontal_sum, bench_horizontal_max, bench_horizontal_min
);

library_benchmark_group!(
    name = memory_group;
    benchmarks = bench_from_slice, bench_to_slice
);

library_benchmark_group!(
    name = block_group;
    benchmarks = bench_block_processing_64
);

// Ops module benchmarks (T052)

#[library_benchmark]
fn bench_ops_add() -> DefaultSimdVector {
    let a = black_box(DefaultSimdVector::splat(2.0));
    let b = black_box(DefaultSimdVector::splat(3.0));
    black_box(add(a, b))
}

#[library_benchmark]
fn bench_ops_mul() -> DefaultSimdVector {
    let a = black_box(DefaultSimdVector::splat(2.0));
    let b = black_box(DefaultSimdVector::splat(3.0));
    black_box(mul(a, b))
}

#[library_benchmark]
fn bench_ops_sub() -> DefaultSimdVector {
    let a = black_box(DefaultSimdVector::splat(5.0));
    let b = black_box(DefaultSimdVector::splat(3.0));
    black_box(sub(a, b))
}

#[library_benchmark]
fn bench_ops_div() -> DefaultSimdVector {
    let a = black_box(DefaultSimdVector::splat(6.0));
    let b = black_box(DefaultSimdVector::splat(3.0));
    black_box(div(a, b))
}

#[library_benchmark]
fn bench_ops_fma() -> DefaultSimdVector {
    let a = black_box(DefaultSimdVector::splat(2.0));
    let b = black_box(DefaultSimdVector::splat(3.0));
    let c = black_box(DefaultSimdVector::splat(1.0));
    black_box(fma(a, b, c))
}

#[library_benchmark]
fn bench_ops_min() -> DefaultSimdVector {
    let a = black_box(DefaultSimdVector::splat(2.0));
    let b = black_box(DefaultSimdVector::splat(3.0));
    black_box(min(a, b))
}

#[library_benchmark]
fn bench_ops_max() -> DefaultSimdVector {
    let a = black_box(DefaultSimdVector::splat(2.0));
    let b = black_box(DefaultSimdVector::splat(3.0));
    black_box(max(a, b))
}

#[library_benchmark]
fn bench_ops_clamp() -> DefaultSimdVector {
    let value = black_box(DefaultSimdVector::splat(5.0));
    let min_val = black_box(DefaultSimdVector::splat(0.0));
    let max_val = black_box(DefaultSimdVector::splat(3.0));
    black_box(clamp(value, min_val, max_val))
}

library_benchmark_group!(
    name = ops_group;
    benchmarks = bench_ops_add, bench_ops_mul, bench_ops_sub, bench_ops_div,
                 bench_ops_fma, bench_ops_min, bench_ops_max, bench_ops_clamp
);

// Math kernel benchmarks (T117)

use rigel_math::simd::*;

#[library_benchmark]
fn bench_math_tanh() -> DefaultSimdVector {
    let x = black_box(DefaultSimdVector::splat(1.5));
    black_box(tanh(x))
}

#[library_benchmark]
fn bench_math_exp() -> DefaultSimdVector {
    let x = black_box(DefaultSimdVector::splat(2.0));
    black_box(exp(x))
}

#[library_benchmark]
fn bench_math_log() -> DefaultSimdVector {
    let x = black_box(DefaultSimdVector::splat(2.5));
    black_box(log(x))
}

#[library_benchmark]
fn bench_math_log1p() -> DefaultSimdVector {
    let x = black_box(DefaultSimdVector::splat(0.5));
    black_box(log1p(x))
}

#[library_benchmark]
fn bench_math_log2() -> DefaultSimdVector {
    let x = black_box(DefaultSimdVector::splat(2.5));
    black_box(log2(x))
}

#[library_benchmark]
fn bench_math_log10() -> DefaultSimdVector {
    let x = black_box(DefaultSimdVector::splat(2.5));
    black_box(log10(x))
}

#[library_benchmark]
fn bench_math_sin() -> DefaultSimdVector {
    let x = black_box(DefaultSimdVector::splat(0.7));
    black_box(sin(x))
}

#[library_benchmark]
fn bench_math_cos() -> DefaultSimdVector {
    let x = black_box(DefaultSimdVector::splat(0.7));
    black_box(cos(x))
}

#[library_benchmark]
fn bench_math_sincos() -> (DefaultSimdVector, DefaultSimdVector) {
    let x = black_box(DefaultSimdVector::splat(0.7));
    black_box(sincos(x))
}

#[library_benchmark]
fn bench_math_atan() -> DefaultSimdVector {
    let x = black_box(DefaultSimdVector::splat(0.8));
    black_box(atan(x))
}

#[library_benchmark]
fn bench_math_atan2() -> DefaultSimdVector {
    let y = black_box(DefaultSimdVector::splat(0.8));
    let x = black_box(DefaultSimdVector::splat(1.0));
    black_box(atan2(y, x))
}

#[library_benchmark]
fn bench_math_recip() -> DefaultSimdVector {
    let x = black_box(DefaultSimdVector::splat(2.5));
    black_box(recip(x))
}

#[library_benchmark]
fn bench_math_sqrt() -> DefaultSimdVector {
    let x = black_box(DefaultSimdVector::splat(4.0));
    black_box(sqrt(x))
}

#[library_benchmark]
fn bench_math_rsqrt() -> DefaultSimdVector {
    let x = black_box(DefaultSimdVector::splat(4.0));
    black_box(rsqrt(x))
}

#[library_benchmark]
fn bench_math_pow() -> DefaultSimdVector {
    let base = black_box(DefaultSimdVector::splat(2.0));
    black_box(pow(base, 3.0))
}

#[library_benchmark]
fn bench_math_fast_exp2() -> DefaultSimdVector {
    let x = black_box(DefaultSimdVector::splat(2.5));
    black_box(fast_exp2(x))
}

#[library_benchmark]
fn bench_math_fast_log2() -> DefaultSimdVector {
    let x = black_box(DefaultSimdVector::splat(2.5));
    black_box(fast_log2(x))
}

library_benchmark_group!(
    name = math_group;
    benchmarks = bench_math_tanh, bench_math_exp, bench_math_log, bench_math_log1p,
                 bench_math_log2, bench_math_log10, bench_math_sin, bench_math_cos,
                 bench_math_sincos, bench_math_atan, bench_math_atan2, bench_math_recip,
                 bench_math_sqrt, bench_math_rsqrt, bench_math_pow, bench_math_fast_exp2,
                 bench_math_fast_log2
);

// DSP utility benchmarks (T117)

use rigel_math::antialias::polyblep;
use rigel_math::interpolate::{cubic_hermite, lerp};
use rigel_math::noise::{white_noise, NoiseState};
use rigel_math::saturate::{hard_clip, soft_clip};
use rigel_math::sigmoid::{smootherstep, smoothstep};

#[library_benchmark]
fn bench_dsp_soft_clip() -> DefaultSimdVector {
    let x = black_box(DefaultSimdVector::splat(2.0));
    black_box(soft_clip(x))
}

#[library_benchmark]
fn bench_dsp_hard_clip() -> DefaultSimdVector {
    let x = black_box(DefaultSimdVector::splat(2.0));
    black_box(hard_clip(x, 1.0))
}

#[library_benchmark]
fn bench_dsp_smoothstep() -> DefaultSimdVector {
    let t = black_box(DefaultSimdVector::splat(0.5));
    black_box(smoothstep(t))
}

#[library_benchmark]
fn bench_dsp_smootherstep() -> DefaultSimdVector {
    let t = black_box(DefaultSimdVector::splat(0.5));
    black_box(smootherstep(t))
}

#[library_benchmark]
fn bench_dsp_lerp() -> DefaultSimdVector {
    let y0 = black_box(DefaultSimdVector::splat(0.0));
    let y1 = black_box(DefaultSimdVector::splat(1.0));
    let frac = black_box(DefaultSimdVector::splat(0.5));
    black_box(lerp(y0, y1, frac))
}

#[library_benchmark]
fn bench_dsp_cubic_hermite() -> DefaultSimdVector {
    let y0 = black_box(DefaultSimdVector::splat(0.0));
    let y1 = black_box(DefaultSimdVector::splat(1.0));
    let m0 = black_box(DefaultSimdVector::splat(0.5));
    let m1 = black_box(DefaultSimdVector::splat(0.5));
    let frac = black_box(DefaultSimdVector::splat(0.5));
    black_box(cubic_hermite(y0, y1, m0, m1, frac))
}

#[library_benchmark]
fn bench_dsp_polyblep() -> DefaultSimdVector {
    let phase = black_box(DefaultSimdVector::splat(0.01));
    let dt = black_box(DefaultSimdVector::splat(0.01)); // typical dt for 440Hz at 44.1kHz
    black_box(polyblep(phase, dt))
}

#[library_benchmark]
fn bench_dsp_white_noise() -> DefaultSimdVector {
    let mut state = black_box(NoiseState::new(12345));
    black_box(white_noise::<DefaultSimdVector>(&mut state))
}

library_benchmark_group!(
    name = dsp_group;
    benchmarks = bench_dsp_soft_clip, bench_dsp_hard_clip, bench_dsp_smoothstep,
                 bench_dsp_smootherstep, bench_dsp_lerp, bench_dsp_cubic_hermite,
                 bench_dsp_polyblep, bench_dsp_white_noise
);

main!(
    library_benchmark_groups = arithmetic_group,
    minmax_group,
    horizontal_group,
    memory_group,
    block_group,
    ops_group,
    math_group,
    dsp_group
);
