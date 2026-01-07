//! Criterion benchmarks for rigel-math (T027, T029)
//!
//! Measures wall-clock time for vector operations across backends.
//! Run with: cargo bench --bench criterion_benches

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rigel_math::ops::{add, clamp, div, fma, max, min, mul, sub};
use rigel_math::simd::*;
use rigel_math::{DefaultSimdVector, SimdVector};
use std::hint::black_box;

/// Benchmark basic arithmetic operations
fn bench_arithmetic(c: &mut Criterion) {
    let mut group = c.benchmark_group("arithmetic");

    let a = DefaultSimdVector::splat(2.0);
    let b = DefaultSimdVector::splat(3.0);

    group.bench_function("add", |bencher| {
        bencher.iter(|| black_box(a.add(black_box(b))))
    });

    group.bench_function("sub", |bencher| {
        bencher.iter(|| black_box(a.sub(black_box(b))))
    });

    group.bench_function("mul", |bencher| {
        bencher.iter(|| black_box(a.mul(black_box(b))))
    });

    group.bench_function("div", |bencher| {
        bencher.iter(|| black_box(a.div(black_box(b))))
    });

    group.finish();
}

/// Benchmark FMA operation
fn bench_fma(c: &mut Criterion) {
    let a = DefaultSimdVector::splat(2.0);
    let b = DefaultSimdVector::splat(3.0);
    let add_c = DefaultSimdVector::splat(1.0);

    c.bench_function("fma", |bencher| {
        bencher.iter(|| black_box(a.fma(black_box(b), black_box(add_c))))
    });
}

/// Benchmark min/max operations
fn bench_minmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("minmax");

    let a = DefaultSimdVector::splat(2.0);
    let b = DefaultSimdVector::splat(3.0);

    group.bench_function("min", |bencher| {
        bencher.iter(|| black_box(a.min(black_box(b))))
    });

    group.bench_function("max", |bencher| {
        bencher.iter(|| black_box(a.max(black_box(b))))
    });

    group.finish();
}

/// Benchmark comparison operations
fn bench_comparisons(c: &mut Criterion) {
    let mut group = c.benchmark_group("comparisons");

    let a = DefaultSimdVector::splat(2.0);
    let b = DefaultSimdVector::splat(3.0);

    group.bench_function("lt", |bencher| {
        bencher.iter(|| black_box(a.lt(black_box(b))))
    });

    group.bench_function("gt", |bencher| {
        bencher.iter(|| black_box(a.gt(black_box(b))))
    });

    group.bench_function("eq", |bencher| {
        bencher.iter(|| black_box(a.eq(black_box(b))))
    });

    group.finish();
}

/// Benchmark horizontal operations
fn bench_horizontal(c: &mut Criterion) {
    let mut group = c.benchmark_group("horizontal");

    let vec = DefaultSimdVector::splat(2.0);

    group.bench_function("sum", |bencher| {
        bencher.iter(|| black_box(vec.horizontal_sum()))
    });

    group.bench_function("max", |bencher| {
        bencher.iter(|| black_box(vec.horizontal_max()))
    });

    group.bench_function("min", |bencher| {
        bencher.iter(|| black_box(vec.horizontal_min()))
    });

    group.finish();
}

/// Benchmark memory operations
fn bench_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory");

    let data = vec![1.0f32; DefaultSimdVector::LANES * 2];
    let mut output = vec![0.0f32; DefaultSimdVector::LANES * 2];

    group.bench_function("load", |bencher| {
        bencher.iter(|| black_box(DefaultSimdVector::from_slice(black_box(&data))))
    });

    group.bench_function("store", |bencher| {
        let vec = DefaultSimdVector::splat(2.0);
        bencher.iter(|| vec.to_slice(black_box(&mut output)))
    });

    group.finish();
}

/// Benchmark block processing (64 samples)
fn bench_block_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("block_processing");
    const BLOCK_SIZE: usize = 64;

    let input = vec![1.0f32; BLOCK_SIZE];
    let mut output = vec![0.0f32; BLOCK_SIZE];
    let gain = 0.5;

    group.bench_function("gain_64_samples", |bencher| {
        bencher.iter(|| {
            let gain_vec = DefaultSimdVector::splat(gain);
            let lanes = DefaultSimdVector::LANES;

            for chunk_start in (0..BLOCK_SIZE).step_by(lanes) {
                if chunk_start + lanes <= BLOCK_SIZE {
                    let input_vec = DefaultSimdVector::from_slice(&input[chunk_start..]);
                    let output_vec = input_vec.mul(gain_vec);
                    output_vec.to_slice(&mut output[chunk_start..]);
                }
            }
            black_box(&output);
        })
    });

    group.finish();
}

/// Benchmark throughput at different block sizes
fn bench_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput");

    for size in [64, 128, 256, 512].iter() {
        let input = vec![1.0f32; *size];
        let mut output = vec![0.0f32; *size];
        let gain = DefaultSimdVector::splat(0.5);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bencher, &size| {
            bencher.iter(|| {
                let lanes = DefaultSimdVector::LANES;
                for chunk_start in (0..size).step_by(lanes) {
                    if chunk_start + lanes <= size {
                        let input_vec = DefaultSimdVector::from_slice(&input[chunk_start..]);
                        let output_vec = input_vec.mul(gain);
                        output_vec.to_slice(&mut output[chunk_start..]);
                    }
                }
                black_box(&output);
            })
        });
    }

    group.finish();
}

/// Benchmark ops module functions (T051)
fn bench_ops_module(c: &mut Criterion) {
    let mut group = c.benchmark_group("ops_module");

    let a = DefaultSimdVector::splat(2.0);
    let b = DefaultSimdVector::splat(3.0);
    let c_val = DefaultSimdVector::splat(1.0);

    // Arithmetic ops
    group.bench_function("ops_add", |bencher| {
        bencher.iter(|| black_box(add(black_box(a), black_box(b))))
    });

    group.bench_function("ops_mul", |bencher| {
        bencher.iter(|| black_box(mul(black_box(a), black_box(b))))
    });

    group.bench_function("ops_sub", |bencher| {
        bencher.iter(|| black_box(sub(black_box(a), black_box(b))))
    });

    group.bench_function("ops_div", |bencher| {
        bencher.iter(|| black_box(div(black_box(a), black_box(b))))
    });

    // FMA
    group.bench_function("ops_fma", |bencher| {
        bencher.iter(|| black_box(fma(black_box(a), black_box(b), black_box(c_val))))
    });

    // MinMax
    group.bench_function("ops_min", |bencher| {
        bencher.iter(|| black_box(min(black_box(a), black_box(b))))
    });

    group.bench_function("ops_max", |bencher| {
        bencher.iter(|| black_box(max(black_box(a), black_box(b))))
    });

    // Clamp
    let value = DefaultSimdVector::splat(5.0);
    let min_val = DefaultSimdVector::splat(0.0);
    let max_val = DefaultSimdVector::splat(3.0);
    group.bench_function("ops_clamp", |bencher| {
        bencher.iter(|| {
            black_box(clamp(
                black_box(value),
                black_box(min_val),
                black_box(max_val),
            ))
        })
    });
    group.finish();
}

/// Benchmark DSP pipeline using ops (T051)
fn bench_ops_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("ops_dsp_pipeline");
    const BLOCK_SIZE: usize = 64;

    let input = vec![1.0f32; BLOCK_SIZE];
    let mut output = vec![0.0f32; BLOCK_SIZE];

    group.bench_function("crossfade_64_samples", |bencher| {
        bencher.iter(|| {
            let signal_a_gain = DefaultSimdVector::splat(0.7);
            let signal_b_gain = DefaultSimdVector::splat(0.3);
            let lanes = DefaultSimdVector::LANES;

            for chunk_start in (0..BLOCK_SIZE).step_by(lanes) {
                if chunk_start + lanes <= BLOCK_SIZE {
                    let input_vec = DefaultSimdVector::from_slice(&input[chunk_start..]);
                    // Simple crossfade: a * 0.7 + b * 0.3
                    let weighted_a = mul(input_vec, signal_a_gain);
                    let weighted_b = mul(input_vec, signal_b_gain);
                    let mixed = add(weighted_a, weighted_b);
                    mixed.to_slice(&mut output[chunk_start..]);
                }
            }
            black_box(&output);
        })
    });

    group.finish();
}

/// Benchmark SIMD math functions vs scalar libm
fn bench_math_simd_vs_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("math_simd_vs_scalar");
    let exp_val = 2.0;
    let log_val = 2.5;
    let trig_val = 0.7;
    let atan_val = 0.8;
    let tanh_val = 1.5;
    let pow_base = 2.0;
    let pow_exp = 3.0;

    // Exponential functions
    group.bench_function("exp_simd", |bencher| {
        let x = DefaultSimdVector::splat(exp_val);
        bencher.iter(|| black_box(exp(black_box(x))))
    });

    group.bench_function("exp_scalar", |bencher| {
        bencher.iter(|| black_box(libm::expf(black_box(exp_val))))
    });

    // Logarithm functions
    group.bench_function("log_simd", |bencher| {
        let x = DefaultSimdVector::splat(log_val);
        bencher.iter(|| black_box(log(black_box(x))))
    });

    group.bench_function("log_scalar", |bencher| {
        bencher.iter(|| black_box(libm::logf(black_box(log_val))))
    });
    group.bench_function("log2_simd", |bencher| {
        let x = DefaultSimdVector::splat(log_val);
        bencher.iter(|| black_box(log2(black_box(x))))
    });

    group.bench_function("log2_scalar", |bencher| {
        bencher.iter(|| black_box(libm::log2f(black_box(log_val))))
    });

    group.bench_function("log10_simd", |bencher| {
        let x = DefaultSimdVector::splat(log_val);
        bencher.iter(|| black_box(log10(black_box(x))))
    });

    group.bench_function("log10_scalar", |bencher| {
        bencher.iter(|| black_box(libm::log10f(black_box(log_val))))
    });

    // Trigonometric functions
    group.bench_function("sin_simd", |bencher| {
        let x = DefaultSimdVector::splat(trig_val);
        bencher.iter(|| black_box(sin(black_box(x))))
    });

    group.bench_function("sin_scalar", |bencher| {
        bencher.iter(|| black_box(libm::sinf(black_box(trig_val))))
    });

    group.bench_function("cos_simd", |bencher| {
        let x = DefaultSimdVector::splat(trig_val);
        bencher.iter(|| black_box(cos(black_box(x))))
    });

    group.bench_function("cos_scalar", |bencher| {
        bencher.iter(|| black_box(libm::cosf(black_box(trig_val))))
    });

    group.bench_function("sincos_simd", |bencher| {
        let x = DefaultSimdVector::splat(trig_val);
        bencher.iter(|| black_box(sincos(black_box(x))))
    });

    group.bench_function("sincos_scalar", |bencher| {
        bencher.iter(|| {
            let s = libm::sinf(trig_val);
            let c = libm::cosf(trig_val);
            black_box((s, c))
        })
    });

    // Inverse trig functions
    group.bench_function("atan_simd", |bencher| {
        let x = DefaultSimdVector::splat(atan_val);
        bencher.iter(|| black_box(atan(black_box(x))))
    });

    group.bench_function("atan_scalar", |bencher| {
        bencher.iter(|| black_box(libm::atanf(black_box(atan_val))))
    });

    group.bench_function("atan2_simd", |bencher| {
        let y = DefaultSimdVector::splat(atan_val);
        let x = DefaultSimdVector::splat(1.0);
        bencher.iter(|| black_box(atan2(black_box(y), black_box(x))))
    });

    group.bench_function("atan2_scalar", |bencher| {
        bencher.iter(|| black_box(libm::atan2f(black_box(atan_val), black_box(1.0))))
    });

    // Hyperbolic functions
    group.bench_function("tanh_simd", |bencher| {
        let x = DefaultSimdVector::splat(tanh_val);
        bencher.iter(|| black_box(tanh(black_box(x))))
    });

    group.bench_function("tanh_scalar", |bencher| {
        bencher.iter(|| black_box(libm::tanhf(black_box(tanh_val))))
    });

    group.bench_function("tanh_fast_simd", |bencher| {
        let x = DefaultSimdVector::splat(tanh_val);
        bencher.iter(|| black_box(tanh_fast(black_box(x))))
    });

    // Power function
    group.bench_function("pow_simd", |bencher| {
        let base = DefaultSimdVector::splat(pow_base);
        bencher.iter(|| black_box(pow(black_box(base), black_box(pow_exp))))
    });

    group.bench_function("pow_scalar", |bencher| {
        bencher.iter(|| black_box(libm::powf(black_box(pow_base), black_box(pow_exp))))
    });

    // Fast exp2
    group.bench_function("fast_exp2_simd", |bencher| {
        let x = DefaultSimdVector::splat(log_val);
        bencher.iter(|| black_box(fast_exp2(black_box(x))))
    });

    group.bench_function("fast_exp2_scalar", |bencher| {
        bencher.iter(|| black_box(libm::exp2f(black_box(log_val))))
    });

    // ========================================================================
    // FAIR COMPARISONS: Scalar 8x vs SIMD (apples-to-apples)
    // ========================================================================
    // NOTE: The benchmarks above compare SIMD (processing 8 f32 values) against
    // scalar (processing 1 f32 value). The benchmarks below provide fair
    // comparisons by running scalar operations 8 times to match SIMD vector width.
    //
    // To calculate per-value performance:
    // - SIMD per-value = simd_time / 8
    // - Scalar per-value = scalar_8x_time / 8
    // - Speedup = scalar_8x_time / simd_time

    group.bench_function("exp_scalar_8x", |bencher| {
        bencher.iter(|| {
            for _ in 0..8 {
                black_box(libm::expf(black_box(exp_val)));
            }
        })
    });

    group.bench_function("log_scalar_8x", |bencher| {
        bencher.iter(|| {
            for _ in 0..8 {
                black_box(libm::logf(black_box(log_val)));
            }
        })
    });

    group.bench_function("log2_scalar_8x", |bencher| {
        bencher.iter(|| {
            for _ in 0..8 {
                black_box(libm::log2f(black_box(log_val)));
            }
        })
    });

    group.bench_function("log10_scalar_8x", |bencher| {
        bencher.iter(|| {
            for _ in 0..8 {
                black_box(libm::log10f(black_box(log_val)));
            }
        })
    });

    group.bench_function("sin_scalar_8x", |bencher| {
        bencher.iter(|| {
            for _ in 0..8 {
                black_box(libm::sinf(black_box(trig_val)));
            }
        })
    });

    group.bench_function("cos_scalar_8x", |bencher| {
        bencher.iter(|| {
            for _ in 0..8 {
                black_box(libm::cosf(black_box(trig_val)));
            }
        })
    });

    group.bench_function("sincos_scalar_8x", |bencher| {
        bencher.iter(|| {
            for _ in 0..8 {
                let s = libm::sinf(trig_val);
                let c = libm::cosf(trig_val);
                black_box((s, c));
            }
        })
    });

    group.bench_function("atan_scalar_8x", |bencher| {
        bencher.iter(|| {
            for _ in 0..8 {
                black_box(libm::atanf(black_box(atan_val)));
            }
        })
    });

    group.bench_function("atan2_scalar_8x", |bencher| {
        bencher.iter(|| {
            for _ in 0..8 {
                black_box(libm::atan2f(black_box(atan_val), black_box(1.0)));
            }
        })
    });

    group.bench_function("tanh_scalar_8x", |bencher| {
        bencher.iter(|| {
            for _ in 0..8 {
                black_box(libm::tanhf(black_box(tanh_val)));
            }
        })
    });

    group.bench_function("pow_scalar_8x", |bencher| {
        bencher.iter(|| {
            for _ in 0..8 {
                black_box(libm::powf(black_box(pow_base), black_box(pow_exp)));
            }
        })
    });

    // ========================================================================
    // SIMD vs OUR SCALAR APPROXIMATIONS (pure vectorization benefit)
    // ========================================================================
    // NOTE: These benchmarks compare our SIMD approximations vs our scalar
    // approximations (both using the same polynomial approximations).
    // This isolates the benefit of vectorization from approximation quality.
    //
    // Comparison methodology:
    // 1. SIMD approximation (8 values) vs libm reference (8 calls) = accuracy trade-off
    // 2. SIMD approximation (8 values) vs scalar approximation (8 values) = pure vectorization
    // 3. Scalar approximation (1 value) vs libm reference (1 value) = approximation quality

    group.bench_function("exp_rigel_scalar", |bencher| {
        use rigel_math::backends::scalar::ScalarVector;
        let x = ScalarVector::<f32>::splat(exp_val);
        bencher.iter(|| black_box(exp(black_box(x))))
    });

    group.bench_function("log_rigel_scalar", |bencher| {
        use rigel_math::backends::scalar::ScalarVector;
        let x = ScalarVector::<f32>::splat(log_val);
        bencher.iter(|| black_box(log(black_box(x))))
    });

    group.bench_function("log2_rigel_scalar", |bencher| {
        use rigel_math::backends::scalar::ScalarVector;
        let x = ScalarVector::<f32>::splat(log_val);
        bencher.iter(|| black_box(log2(black_box(x))))
    });

    group.bench_function("sin_rigel_scalar", |bencher| {
        use rigel_math::backends::scalar::ScalarVector;
        let x = ScalarVector::<f32>::splat(trig_val);
        bencher.iter(|| black_box(sin(black_box(x))))
    });

    group.bench_function("cos_rigel_scalar", |bencher| {
        use rigel_math::backends::scalar::ScalarVector;
        let x = ScalarVector::<f32>::splat(trig_val);
        bencher.iter(|| black_box(cos(black_box(x))))
    });

    group.bench_function("tanh_rigel_scalar", |bencher| {
        use rigel_math::backends::scalar::ScalarVector;
        let x = ScalarVector::<f32>::splat(tanh_val);
        bencher.iter(|| black_box(tanh(black_box(x))))
    });

    group.finish();
}

/// Benchmark math functions on audio block (64 samples)
fn bench_math_audio_block(c: &mut Criterion) {
    let mut group = c.benchmark_group("math_audio_block");
    const BLOCK_SIZE: usize = 64;

    let input = vec![0.5f32; BLOCK_SIZE];
    let mut output = vec![0.0f32; BLOCK_SIZE];

    // SIMD exp on block
    group.bench_function("exp_block_simd", |bencher| {
        bencher.iter(|| {
            let lanes = DefaultSimdVector::LANES;
            for chunk_start in (0..BLOCK_SIZE).step_by(lanes) {
                if chunk_start + lanes <= BLOCK_SIZE {
                    let input_vec = DefaultSimdVector::from_slice(&input[chunk_start..]);
                    let result = exp(input_vec);
                    result.to_slice(&mut output[chunk_start..]);
                }
            }
            black_box(&output);
        })
    });

    // Scalar exp on block
    group.bench_function("exp_block_scalar", |bencher| {
        bencher.iter(|| {
            for i in 0..BLOCK_SIZE {
                output[i] = libm::expf(input[i]);
            }
            black_box(&output);
        })
    });

    // SIMD log on block
    group.bench_function("log_block_simd", |bencher| {
        bencher.iter(|| {
            let lanes = DefaultSimdVector::LANES;
            for chunk_start in (0..BLOCK_SIZE).step_by(lanes) {
                if chunk_start + lanes <= BLOCK_SIZE {
                    let input_vec = DefaultSimdVector::from_slice(&input[chunk_start..]);
                    let result = log(input_vec);
                    result.to_slice(&mut output[chunk_start..]);
                }
            }
            black_box(&output);
        })
    });

    // Scalar log on block
    group.bench_function("log_block_scalar", |bencher| {
        bencher.iter(|| {
            for i in 0..BLOCK_SIZE {
                output[i] = libm::logf(input[i]);
            }
            black_box(&output);
        })
    });

    // SIMD sin on block
    group.bench_function("sin_block_simd", |bencher| {
        bencher.iter(|| {
            let lanes = DefaultSimdVector::LANES;
            for chunk_start in (0..BLOCK_SIZE).step_by(lanes) {
                if chunk_start + lanes <= BLOCK_SIZE {
                    let input_vec = DefaultSimdVector::from_slice(&input[chunk_start..]);
                    let result = sin(input_vec);
                    result.to_slice(&mut output[chunk_start..]);
                }
            }
            black_box(&output);
        })
    });

    // Scalar sin on block
    group.bench_function("sin_block_scalar", |bencher| {
        bencher.iter(|| {
            for i in 0..BLOCK_SIZE {
                output[i] = libm::sinf(input[i]);
            }
            black_box(&output);
        })
    });

    // SIMD tanh on block
    group.bench_function("tanh_block_simd", |bencher| {
        bencher.iter(|| {
            let lanes = DefaultSimdVector::LANES;
            for chunk_start in (0..BLOCK_SIZE).step_by(lanes) {
                if chunk_start + lanes <= BLOCK_SIZE {
                    let input_vec = DefaultSimdVector::from_slice(&input[chunk_start..]);
                    let result = tanh(input_vec);
                    result.to_slice(&mut output[chunk_start..]);
                }
            }
            black_box(&output);
        })
    });

    // Scalar tanh on block
    group.bench_function("tanh_block_scalar", |bencher| {
        bencher.iter(|| {
            for i in 0..BLOCK_SIZE {
                output[i] = libm::tanhf(input[i]);
            }
            black_box(&output);
        })
    });

    group.finish();
}

/// Benchmark scalar backend: libm-optimized methods vs generic polynomial implementations
fn bench_scalar_libm_vs_polynomial(c: &mut Criterion) {
    use rigel_math::backends::scalar::ScalarVector;

    let mut group = c.benchmark_group("scalar_libm_vs_polynomial");

    // Test values
    let exp_val = ScalarVector(2.0);
    let log_val = ScalarVector(2.5);
    let trig_val = ScalarVector(0.7);
    let atan_val = ScalarVector(0.8);
    let tanh_val = ScalarVector(1.5);

    // Exponential
    group.bench_function("exp_libm", |bencher| {
        bencher.iter(|| black_box(black_box(exp_val).exp_libm()))
    });
    group.bench_function("exp_polynomial", |bencher| {
        bencher.iter(|| black_box(exp(black_box(exp_val))))
    });

    // Natural logarithm
    group.bench_function("log_libm", |bencher| {
        bencher.iter(|| black_box(black_box(log_val).log_libm()))
    });
    group.bench_function("log_polynomial", |bencher| {
        bencher.iter(|| black_box(log(black_box(log_val))))
    });

    // Sine
    group.bench_function("sin_libm", |bencher| {
        bencher.iter(|| black_box(black_box(trig_val).sin_libm()))
    });
    group.bench_function("sin_polynomial", |bencher| {
        bencher.iter(|| black_box(sin(black_box(trig_val))))
    });

    // Cosine
    group.bench_function("cos_libm", |bencher| {
        bencher.iter(|| black_box(black_box(trig_val).cos_libm()))
    });
    group.bench_function("cos_polynomial", |bencher| {
        bencher.iter(|| black_box(cos(black_box(trig_val))))
    });

    // Arctangent
    group.bench_function("atan_libm", |bencher| {
        bencher.iter(|| black_box(black_box(atan_val).atan_libm()))
    });
    group.bench_function("atan_polynomial", |bencher| {
        bencher.iter(|| black_box(atan(black_box(atan_val))))
    });

    // Hyperbolic tangent
    group.bench_function("tanh_libm", |bencher| {
        bencher.iter(|| black_box(black_box(tanh_val).tanh_libm()))
    });
    group.bench_function("tanh_polynomial", |bencher| {
        bencher.iter(|| black_box(tanh(black_box(tanh_val))))
    });

    group.finish();
}

/// Benchmark DSP utility functions (T114, T115, T116)
fn bench_dsp_utilities(c: &mut Criterion) {
    use rigel_math::antialias::polyblep;
    use rigel_math::interpolate::{cubic_hermite, lerp};
    use rigel_math::noise::{white_noise, NoiseState};
    use rigel_math::saturate::{hard_clip, soft_clip};
    use rigel_math::sigmoid::{smootherstep, smoothstep};

    let mut group = c.benchmark_group("dsp_utilities");

    let x = DefaultSimdVector::splat(2.0);

    // T114: Polynomial saturation
    group.bench_function("soft_clip", |bencher| {
        bencher.iter(|| black_box(soft_clip(black_box(x))))
    });

    group.bench_function("hard_clip", |bencher| {
        bencher.iter(|| black_box(hard_clip(black_box(x), 1.0)))
    });

    // Sigmoid curves
    let t = DefaultSimdVector::splat(0.5);
    group.bench_function("smoothstep", |bencher| {
        bencher.iter(|| black_box(smoothstep(black_box(t))))
    });

    group.bench_function("smootherstep", |bencher| {
        bencher.iter(|| black_box(smootherstep(black_box(t))))
    });

    // Interpolation
    let y0 = DefaultSimdVector::splat(0.0);
    let y1 = DefaultSimdVector::splat(1.0);
    let frac = DefaultSimdVector::splat(0.5);

    group.bench_function("lerp", |bencher| {
        bencher.iter(|| black_box(lerp(black_box(y0), black_box(y1), black_box(frac))))
    });

    let m0 = DefaultSimdVector::splat(0.5);
    let m1 = DefaultSimdVector::splat(0.5);
    group.bench_function("cubic_hermite", |bencher| {
        bencher.iter(|| {
            black_box(cubic_hermite(
                black_box(y0),
                black_box(y1),
                black_box(m0),
                black_box(m1),
                black_box(frac),
            ))
        })
    });

    // T115: PolyBLEP
    let phase = DefaultSimdVector::splat(0.01);
    let dt = DefaultSimdVector::splat(0.01); // typical dt for 440Hz at 44.1kHz
    group.bench_function("polyblep", |bencher| {
        bencher.iter(|| black_box(polyblep(black_box(phase), black_box(dt))))
    });

    // T116: White noise
    let mut state = NoiseState::new(12345);
    group.bench_function("white_noise", |bencher| {
        bencher.iter(|| black_box(white_noise::<DefaultSimdVector>(&mut state)))
    });

    group.finish();
}

/// Benchmark DSP utilities on 64-sample blocks (T116)
fn bench_dsp_utilities_block(c: &mut Criterion) {
    use rigel_math::noise::{white_noise, NoiseState};
    use rigel_math::saturate::soft_clip;
    use rigel_math::Block64;

    let mut group = c.benchmark_group("dsp_utilities_block");
    #[allow(dead_code)]
    const BLOCK_SIZE: usize = 64;

    // T114: Soft clipping a block
    group.bench_function("soft_clip_block_64", |bencher| {
        bencher.iter(|| {
            let mut block = Block64::new();
            for i in 0..64 {
                block[i] = ((i as f32 / 64.0) * 4.0) - 2.0; // Range [-2, 2]
            }

            for mut chunk in block.as_chunks_mut::<DefaultSimdVector>().iter_mut() {
                let val = chunk.load();
                let result = soft_clip(val);
                chunk.store(result);
            }
            black_box(&block);
        })
    });

    // T116: White noise block generation
    group.bench_function("white_noise_block_64", |bencher| {
        let mut state = NoiseState::new(12345);
        let mut block = Block64::new();

        bencher.iter(|| {
            for mut chunk in block.as_chunks_mut::<DefaultSimdVector>().iter_mut() {
                let noise = white_noise::<DefaultSimdVector>(&mut state);
                chunk.store(noise);
            }
            black_box(&block);
        })
    });

    group.finish();
}

/// Benchmark specific performance targets (T110, T111, T112, T113)
fn bench_performance_targets(c: &mut Criterion) {
    let mut group = c.benchmark_group("performance_targets");

    // T110: tanh speedup verification
    let tanh_val = DefaultSimdVector::splat(1.5);
    group.bench_function("tanh_simd", |bencher| {
        bencher.iter(|| black_box(tanh(black_box(tanh_val))))
    });
    group.bench_function("tanh_scalar_libm", |bencher| {
        bencher.iter(|| black_box(libm::tanhf(1.5)))
    });

    // T111: exp sub-nanosecond throughput
    let exp_val = DefaultSimdVector::splat(2.0);
    group.bench_function("exp_per_sample", |bencher| {
        bencher.iter(|| black_box(exp(black_box(exp_val))))
    });

    // T112: atan speedup verification
    let atan_val = DefaultSimdVector::splat(0.8);
    group.bench_function("atan_simd", |bencher| {
        bencher.iter(|| black_box(atan(black_box(atan_val))))
    });
    group.bench_function("atan_scalar_libm", |bencher| {
        bencher.iter(|| black_box(libm::atanf(0.8)))
    });

    // T113: exp2/log2 speedup verification
    let log2_val = DefaultSimdVector::splat(2.5);
    group.bench_function("fast_exp2_simd", |bencher| {
        bencher.iter(|| black_box(fast_exp2(black_box(log2_val))))
    });
    group.bench_function("fast_exp2_scalar_libm", |bencher| {
        bencher.iter(|| black_box(libm::exp2f(2.5)))
    });

    let fast_log2_val = DefaultSimdVector::splat(2.5);
    group.bench_function("fast_log2_simd", |bencher| {
        bencher.iter(|| black_box(fast_log2(black_box(fast_log2_val))))
    });
    group.bench_function("fast_log2_scalar_libm", |bencher| {
        bencher.iter(|| black_box(libm::log2f(2.5)))
    });

    group.finish();
}

/// Benchmark exp2 implementations (optimized vs previous)
fn bench_exp2_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("exp2_comparison");

    let x = DefaultSimdVector::splat(3.5);

    // New optimized implementation using bit manipulation + polynomial
    group.bench_function("fast_exp2_optimized", |bencher| {
        bencher.iter(|| black_box(fast_exp2(black_box(x))))
    });

    // Previous implementation (via exp)
    group.bench_function("exp2_via_exp", |bencher| {
        let ln_2 = DefaultSimdVector::splat(core::f32::consts::LN_2);
        bencher.iter(|| {
            let scaled = black_box(x).mul(ln_2);
            black_box(exp(scaled))
        })
    });

    group.finish();
}

/// Benchmark MIDI-to-frequency conversion (real-world use case)
fn bench_midi_to_frequency(c: &mut Criterion) {
    let mut group = c.benchmark_group("midi_to_frequency");

    let midi_notes = DefaultSimdVector::splat(60.0); // Middle C
    let a4_midi = DefaultSimdVector::splat(69.0);
    let semitones_from_a4 = midi_notes.sub(a4_midi);
    let octaves = semitones_from_a4.div(DefaultSimdVector::splat(12.0));

    // Using optimized fast_exp2
    group.bench_function("fast_exp2_method", |bencher| {
        bencher.iter(|| {
            let ratio = fast_exp2(black_box(octaves));
            black_box(ratio.mul(DefaultSimdVector::splat(440.0)))
        })
    });

    // Using exp(x * ln(2))
    group.bench_function("exp_method", |bencher| {
        let ln_2 = DefaultSimdVector::splat(core::f32::consts::LN_2);
        bencher.iter(|| {
            let scaled = black_box(octaves).mul(ln_2);
            let ratio = exp(scaled);
            black_box(ratio.mul(DefaultSimdVector::splat(440.0)))
        })
    });

    group.finish();
}

/// Benchmark pow implementations (optimized vs old decomposition)
fn bench_pow_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("pow_comparison");

    let base = DefaultSimdVector::splat(2.0);
    let exponent = 3.5;

    // New optimized implementation using fast_exp2/fast_log2
    group.bench_function("pow_optimized_exp2_log2", |bencher| {
        bencher.iter(|| black_box(pow(black_box(base), black_box(exponent))))
    });

    // Old implementation using exp/ln (for comparison)
    group.bench_function("pow_old_exp_ln", |bencher| {
        let exp_scalar = DefaultSimdVector::splat(exponent);
        bencher.iter(|| {
            let ln_base = log(black_box(base));
            let product = ln_base.mul(exp_scalar);
            black_box(exp(product))
        })
    });

    // Scalar libm baseline
    group.bench_function("pow_scalar_libm", |bencher| {
        bencher.iter(|| black_box(libm::powf(2.0, black_box(exponent))))
    });

    group.finish();
}

/// Benchmark pow for harmonic series generation (audio use case)
fn bench_pow_harmonic_series(c: &mut Criterion) {
    let mut group = c.benchmark_group("pow_harmonic_series");

    // Generate amplitude falloff for harmonics: amplitude_n = 1/n^falloff
    // Common in additive synthesis
    let falloff = 0.8;

    group.bench_function("harmonic_series_optimized", |bencher| {
        bencher.iter(|| {
            let mut sum = DefaultSimdVector::splat(0.0);
            for n in 1..=8 {
                let harmonic = DefaultSimdVector::splat(n as f32);
                let amplitude = pow(harmonic, falloff);
                sum = sum.add(amplitude);
            }
            black_box(sum)
        })
    });

    group.bench_function("harmonic_series_scalar", |bencher| {
        bencher.iter(|| {
            let mut sum = 0.0f32;
            for n in 1..=8 {
                sum += libm::powf(n as f32, falloff);
            }
            black_box(sum)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_arithmetic,
    bench_fma,
    bench_minmax,
    bench_comparisons,
    bench_horizontal,
    bench_memory,
    bench_block_processing,
    bench_throughput,
    bench_ops_module,
    bench_ops_pipeline,
    bench_math_simd_vs_scalar,
    bench_math_audio_block,
    bench_scalar_libm_vs_polynomial,
    bench_dsp_utilities,
    bench_dsp_utilities_block,
    bench_performance_targets,
    bench_exp2_comparison,
    bench_midi_to_frequency,
    bench_pow_comparison,
    bench_pow_harmonic_series
);
criterion_main!(benches);
