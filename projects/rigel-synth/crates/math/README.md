# rigel-math

Trait-based SIMD abstraction library for real-time audio DSP.

## Features

- **Zero-cost SIMD abstractions**: Write once, compile to optimal SIMD for each platform
- **Backend support**: Scalar, AVX2, AVX512, NEON
- **Real-time safe**: No allocations, deterministic execution
- **Comprehensive**: Block processing, fast math, lookup tables, denormal protection

## Quick Start

```rust,ignore
use rigel_math::{Block64, DefaultSimdVector};
use rigel_math::ops::mul;

fn apply_gain(input: &Block64, output: &mut Block64, gain: f32) {
    let gain_vec = DefaultSimdVector::splat(gain);

    for (in_chunk, out_chunk) in input.as_chunks::<DefaultSimdVector>()
        .iter()
        .zip(output.as_chunks_mut::<DefaultSimdVector>().iter_mut())
    {
        *out_chunk = mul(*in_chunk, gain_vec);
    }
}
```

## Backend Selection

Choose backend via cargo features (mutually exclusive):

```bash
cargo build --features scalar   # Default, always available
cargo build --features avx2     # x86-64 with AVX2
cargo build --features avx512   # x86-64 with AVX-512
cargo build --features neon     # ARM64 with NEON
```

## Math Function Implementations

rigel-math provides two complementary implementations for fast math functions:

### Scalar Backend: libm-Optimized

The scalar backend uses **libm** for maximum performance on single values:

```rust
use rigel_math::backends::scalar::ScalarVector;

let x = ScalarVector(2.0);
let result = x.exp_libm();  // Direct libm::expf call
```

**Available libm methods:**
- `exp_libm`, `log_libm`, `log2_libm`, `log10_libm`, `log1p_libm`
- `sin_libm`, `cos_libm`, `sincos_libm`, `tan_libm`
- `atan_libm`, `atan2_libm`, `tanh_libm`
- `pow_libm`, `sqrt_libm`

**Performance vs polynomial implementations:**
- exp: **1.9x faster** (3.85ns vs 7.30ns)
- log: **1.7x faster** (2.79ns vs 4.68ns)
- sin/cos: **1.6x faster** (~2.5ns vs ~4.0ns)
- atan: **1.5x faster** (2.96ns vs 4.30ns)

### SIMD Backends: Polynomial Approximations

AVX2, AVX512, and NEON backends use **vectorized polynomial approximations**:

```rust
use rigel_math::{DefaultSimdVector, SimdVector};
use rigel_math::math::exp;

let x = DefaultSimdVector::splat(2.0);
let result = exp(x);  // Padé[5/5] approximation, all lanes in parallel
```

**Why not libm for SIMD?**
- libm provides scalar functions only
- Vectorizing libm calls loses parallelism benefits
- Custom polynomials achieve **4-16x speedup** vs scalar libm

**Accuracy guarantees:**
- exp: <0.12% error for audio ranges
- log: <1.1% relative error
- sin/cos: <0.016% error
- atan: <1.5° (0.026 radians)

All errors are **perceptually imperceptible** in audio applications.

### Backend Consistency

When processing audio blocks with SIMD, remainder samples use the scalar backend:

```rust
// Example: Processing 66 samples with AVX2 (8 lanes)
// Samples 0-63: AVX2 polynomial approximations
// Samples 64-65: Scalar libm functions
```

**Mixing impact:**
- Maximum difference: 0.2% (2.24e-3 for exp)
- Below 16-bit audio resolution
- Non-cumulative (doesn't build up over time)
- Masked by dither in production

This design provides **optimal performance on both scalar and SIMD paths** with inaudible mixing artifacts.

## Benchmarking

rigel-math includes comprehensive benchmarks to validate performance claims across backends.

### Running Benchmarks

Two benchmark suites are provided:

1. **Criterion** (wall-clock time measurements):
   ```bash
   cargo bench --bench criterion_benches
   ```

2. **iai-callgrind** (instruction count measurements):
   ```bash
   cargo bench --bench iai_benches
   ```

### Backend Comparison

Run benchmarks with different backends to compare performance:

```bash
# Scalar baseline (always available)
cargo bench --bench criterion_benches --features scalar

# AVX2 backend (x86-64 with AVX2)
cargo bench --bench criterion_benches --features avx2

# AVX512 backend (x86-64 with AVX-512)
cargo bench --bench criterion_benches --features avx512

# NEON backend (ARM64 with NEON)
cargo bench --bench criterion_benches --features neon
```

### Saving and Comparing Baselines

Save a performance baseline before making changes:

```bash
# Save baseline with current backend
cargo bench --bench criterion_benches -- --save-baseline before

# Make changes...

# Compare against baseline
cargo bench --bench criterion_benches -- --baseline before
```

### Interpreting Results

**Criterion output:**
- Shows wall-clock time (mean, median, std dev)
- Reports change from previous run (±% with confidence interval)
- Generates HTML reports in `target/criterion/`

Expected performance scaling vs scalar:
- AVX2: 4-8x speedup (8 f32 lanes)
- AVX512: 8-16x speedup (16 f32 lanes)
- NEON: 4-8x speedup (4 f32 lanes)

**iai-callgrind output:**
- Shows instruction counts (deterministic, no timing variance)
- Cache statistics (L1, L2, LLC hits/misses)
- Useful for detecting algorithmic regressions

Example:
```text
arithmetic_group::bench_add
  Instructions:     45 (No change)
  L1 Accesses:      67 (No change)
  L2 Accesses:       0 (No change)
```

### Performance Targets

Single-operation targets:
- Vector add/mul/sub: <10 cycles/operation
- FMA: <5 cycles/operation (single instruction on AVX2/AVX512)
- Horizontal sum: <20 cycles
- Block processing (64 samples): <640ns (<10ns/sample)

## Documentation

`rigel-math` includes comprehensive documentation for development, testing, and performance optimization:

### User Guides

- **[API Reference](docs/api-reference.md)** - Complete function reference with error bounds, performance characteristics, and usage examples
- **[Testing Guide](docs/testing.md)** - How to run unit tests, doctests, and property-based tests across all backends
- **[Benchmarking Guide](docs/benchmarking.md)** - Performance measurement with Criterion and iai-callgrind
- **[Coverage Guide](docs/coverage.md)** - Generating and interpreting code coverage reports

### Development

- **[DEVELOPMENT.md](DEVELOPMENT.md)** - Complete development workflow, pre-commit checklist, and contribution guidelines

### Quick Links

- **[Main README](../../README.md)** - Rigel synthesizer project overview
- **[Root Documentation](../../docs/)** - Additional rigel-synth documentation

## License

MIT OR Apache-2.0
