# Quick Start: rigel-math

**Feature**: 001-fast-dsp-math
**Date**: 2025-11-18

## Overview

This guide shows how to use the rigel-math SIMD library for common DSP tasks. All examples work across all backends (scalar, AVX2, AVX512, NEON) without code changes.

---

## Setup

### Add Dependency

```toml
# In your Cargo.toml
[dependencies]
rigel-math = { path = "../math", default-features = false }

# Choose backend via features (mutually exclusive):
# --features scalar   (default, always available)
# --features avx2     (x86-64 with AVX2)
# --features avx512   (x86-64 with AVX-512)
# --features neon     (ARM64 with NEON)
```

### Import Types

```rust
use rigel_math::{
    DefaultSimdVector,  // Resolves to selected backend
    Block64,            // 64-sample fixed-size block
    DenormalGuard,      // Denormal protection
};
use rigel_math::ops;     // Vector operations
use rigel_math::math;    // Fast math kernels
```

---

## Example 1: Simple Gain (Vector Multiplication)

**User Story**: US3 (Core Vector Operations)

```rust
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

#[test]
fn test_apply_gain() {
    let mut input = Block64::new();
    let mut output = Block64::new();
    
    // Fill input with 1.0
    for i in 0..64 {
        input[i] = 1.0;
    }

    apply_gain(&input, &mut output, 2.0);

    // Verify all samples doubled
    for i in 0..64 {
        assert_eq!(output[i], 2.0);
    }
}
```

**Key Points**:
- `DefaultSimdVector::splat(value)` broadcasts scalar to all SIMD lanes
- `as_chunks()` views block as SIMD vectors (immutable)
- `as_chunks_mut()` views block as SIMD vectors (mutable)
- Same code works for scalar (1 lane), AVX2 (8 lanes), AVX512 (16 lanes), NEON (4 lanes)

---

## Example 2: Soft Clipping with Tanh

**User Story**: US4 (Fast Math Kernels), US7 (Saturation)

```rust
use rigel_math::{Block64, DefaultSimdVector, DenormalGuard};
use rigel_math::ops::mul;
use rigel_math::math::tanh;

fn soft_clip(input: &Block64, output: &mut Block64, drive: f32) {
    let _guard = DenormalGuard::new(); // Enable denormal protection
    
    let drive_vec = DefaultSimdVector::splat(drive);
    
    for (in_chunk, out_chunk) in input.as_chunks::<DefaultSimdVector>()
        .iter()
        .zip(output.as_chunks_mut::<DefaultSimdVector>().iter_mut())
    {
        let driven = mul(*in_chunk, drive_vec);
        *out_chunk = tanh(driven);
    }
}

#[test]
fn test_soft_clip() {
    let mut input = Block64::new();
    let mut output = Block64::new();
    
    // Test with extreme value
    for i in 0..64 {
        input[i] = 10.0; // Well beyond [-1, 1]
    }
    
    soft_clip(&input, &mut output, 1.0);
    
    // Verify saturation near 1.0 (tanh saturates to ±1)
    for i in 0..64 {
        assert!(output[i] > 0.99 && output[i] < 1.0);
    }
}
```

**Key Points**:
- `DenormalGuard` automatically enables FTZ/DAZ flags (x86-64) or FZ (ARM64)
- Denormal protection prevents performance degradation during silence
- `tanh` provides smooth saturation with <0.1% error vs libm reference

---

## Example 3: Exponential Envelope Generator

**User Story**: US4 (Fast Math Kernels)

```rust
use rigel_math::{Block64, DefaultSimdVector};
use rigel_math::math::exp;
use rigel_math::ops::mul;

fn generate_exponential_decay(
    output: &mut Block64,
    decay_rate: f32, // Negative value for decay
    initial_value: f32,
) {
    let mut time = 0.0;
    let dt = 1.0 / 44100.0; // Assuming 44.1kHz sample rate
    
    let decay_vec = DefaultSimdVector::splat(decay_rate);
    let initial_vec = DefaultSimdVector::splat(initial_value);
    
    for out_chunk in output.as_chunks_mut::<DefaultSimdVector>() {
        // Create time vector for this chunk
        let time_offsets: Vec<f32> = (0..DefaultSimdVector::LANES)
            .map(|i| time + (i as f32) * dt)
            .collect();
        
        let time_vec = DefaultSimdVector::from_slice(&time_offsets);
        let exponent = mul(decay_vec, time_vec);
        let envelope = mul(initial_vec, exp(exponent));
        
        *out_chunk = envelope;
        time += (DefaultSimdVector::LANES as f32) * dt;
    }
}

#[test]
fn test_exponential_decay() {
    let mut output = Block64::new();
    
    generate_exponential_decay(&mut output, -5.0, 1.0);
    
    // First sample should be close to 1.0
    assert!((output[0] - 1.0).abs() < 0.01);
    
    // Verify decay (each sample smaller than previous)
    for i in 1..64 {
        assert!(output[i] < output[i - 1]);
    }
}
```

**Key Points**:
- `exp` achieves sub-nanosecond per-sample throughput
- Vectorized time computation for SIMD efficiency
- Error bounds: <0.1% vs libm reference

---

## Example 4: Wavetable Oscillator

**User Story**: US5 (Lookup Table Infrastructure)

```rust
use rigel_math::{LookupTable, DefaultSimdVector, Block64, IndexMode};

const TABLE_SIZE: usize = 2048;
type WaveTable = LookupTable<f32, TABLE_SIZE>;

fn generate_sine_table() -> WaveTable {
    WaveTable::from_fn(|i| {
        let phase = (i as f32 / TABLE_SIZE as f32) * 2.0 * std::f32::consts::PI;
        phase.sin()
    })
}

fn oscillator_tick(
    table: &WaveTable,
    output: &mut Block64,
    frequency: f32, // Hz
    sample_rate: f32,
    phase: &mut f32, // Mutable phase accumulator
) {
    let phase_increment = frequency / sample_rate * (TABLE_SIZE as f32);
    
    for out_chunk in output.as_chunks_mut::<DefaultSimdVector>() {
        // Create phase vector for this chunk
        let phase_offsets: Vec<f32> = (0..DefaultSimdVector::LANES)
            .map(|i| *phase + (i as f32) * phase_increment)
            .collect();
        
        let phase_vec = DefaultSimdVector::from_slice(&phase_offsets);
        *out_chunk = table.lookup_linear_simd(phase_vec, IndexMode::Wrap);
        
        *phase += (DefaultSimdVector::LANES as f32) * phase_increment;
        
        // Wrap phase to [0, TABLE_SIZE)
        *phase = *phase % (TABLE_SIZE as f32);
    }
}

#[test]
fn test_wavetable_oscillator() {
    let table = generate_sine_table();
    let mut output = Block64::new();
    let mut phase = 0.0;
    
    oscillator_tick(&table, &mut output, 440.0, 44100.0, &mut phase);
    
    // Verify output is in valid range [-1, 1]
    for i in 0..64 {
        assert!(output[i] >= -1.0 && output[i] <= 1.0);
    }
}
```

**Key Points**:
- `LookupTable::from_fn` generates table at compile-time or initialization
- `lookup_linear_simd` uses SIMD gather operations for per-lane indexing
- `IndexMode::Wrap` for periodic waveforms (modulo wrapping)
- Performance: <10ns per sample including interpolation

---

## Example 5: Equal-Power Crossfade

**User Story**: US8 (Crossfade and Ramping Utilities)

```rust
use rigel_math::{Block64, DefaultSimdVector};
use rigel_math::crossfade::crossfade_equal_power;

fn crossfade_buffers(
    buffer_a: &Block64,
    buffer_b: &Block64,
    output: &mut Block64,
    mix: f32, // 0.0 = all A, 1.0 = all B
) {
    for (a_chunk, (b_chunk, out_chunk)) in buffer_a.as_chunks::<DefaultSimdVector>()
        .iter()
        .zip(
            buffer_b.as_chunks::<DefaultSimdVector>()
                .iter()
                .zip(output.as_chunks_mut::<DefaultSimdVector>())
        )
    {
        *out_chunk = crossfade_equal_power(*a_chunk, *b_chunk, mix);
    }
}

#[test]
fn test_crossfade() {
    let mut buffer_a = Block64::new();
    let mut buffer_b = Block64::new();
    let mut output = Block64::new();
    
    // Fill A with 1.0, B with 0.0
    for i in 0..64 {
        buffer_a[i] = 1.0;
        buffer_b[i] = 0.0;
    }
    
    // 50% mix should preserve energy
    crossfade_buffers(&buffer_a, &buffer_b, &mut output, 0.5);
    
    // Equal-power: sqrt(0.5) ≈ 0.707
    for i in 0..64 {
        assert!((output[i] - 0.707).abs() < 0.01);
    }
}
```

**Key Points**:
- Equal-power crossfade maintains constant perceived loudness
- Mix parameter: 0.0 = 100% A, 1.0 = 100% B
- No clicks or zipper noise

---

## Example 6: Parameter Ramping (Click-Free Parameter Changes)

**User Story**: US8 (Crossfade and Ramping Utilities)

```rust
use rigel_math::{Block64, ParameterRamp};

fn apply_ramped_gain(
    input: &Block64,
    output: &mut Block64,
    ramp: &mut ParameterRamp<f32>,
) {
    let mut gain_block = Block64::new();
    ramp.fill_block(&mut gain_block);
    
    for i in 0..64 {
        output[i] = input[i] * gain_block[i];
    }
}

#[test]
fn test_parameter_ramp() {
    let mut input = Block64::new();
    let mut output = Block64::new();
    
    // Fill input with 1.0
    for i in 0..64 {
        input[i] = 1.0;
    }
    
    // Ramp from 0.0 to 1.0 over 64 samples
    let mut ramp = ParameterRamp::new(0.0, 1.0, 64);
    
    apply_ramped_gain(&input, &mut output, &mut ramp);
    
    // First sample should be near 0.0
    assert!(output[0] < 0.02);
    
    // Last sample should be near 1.0
    assert!(output[63] > 0.98);
    
    // Verify smooth increase
    for i in 1..64 {
        assert!(output[i] >= output[i - 1]);
    }
}
```

**Key Points**:
- `ParameterRamp` generates smooth transitions over specified duration
- `fill_block` efficiently fills entire block with ramped values
- Prevents clicks when changing filter cutoff, gain, or other parameters

---

## Example 7: Backend Consistency Testing

**User Story**: US10 (Comprehensive Test Coverage)

```rust
#[cfg(test)]
mod backend_tests {
    use rigel_math::{ScalarVector, DefaultSimdVector};
    use rigel_math::ops::{add, mul};
    
    #[test]
    fn test_backend_consistency_add() {
        let a_scalar = ScalarVector::splat(2.0);
        let b_scalar = ScalarVector::splat(3.0);
        let result_scalar = add(a_scalar, b_scalar);
        
        let a_simd = DefaultSimdVector::splat(2.0);
        let b_simd = DefaultSimdVector::splat(3.0);
        let result_simd = add(a_simd, b_simd);
        
        // Verify all SIMD lanes match scalar result
        assert_eq!(
            result_scalar.horizontal_sum(),
            5.0
        );
        
        assert_eq!(
            result_simd.horizontal_sum(),
            5.0 * (DefaultSimdVector::LANES as f32)
        );
    }
}
```

---

## Example 8: Property-Based Testing with Proptest

**User Story**: US10 (Comprehensive Test Coverage)

```rust
#[cfg(test)]
mod property_tests {
    use proptest::prelude::*;
    use rigel_math::{DefaultSimdVector};
    use rigel_math::ops::{add, mul};
    
    proptest! {
        #[test]
        fn test_addition_commutativity(
            a in proptest::num::f32::NORMAL,
            b in proptest::num::f32::NORMAL,
        ) {
            let a_vec = DefaultSimdVector::splat(a);
            let b_vec = DefaultSimdVector::splat(b);
            
            let result1 = add(a_vec, b_vec);
            let result2 = add(b_vec, a_vec);
            
            // a + b == b + a
            prop_assert!((result1.horizontal_sum() - result2.horizontal_sum()).abs() < 1e-6);
        }
        
        #[test]
        fn test_multiplication_associativity(
            a in proptest::num::f32::NORMAL,
            b in proptest::num::f32::NORMAL,
            c in proptest::num::f32::NORMAL,
        ) {
            let a_vec = DefaultSimdVector::splat(a);
            let b_vec = DefaultSimdVector::splat(b);
            let c_vec = DefaultSimdVector::splat(c);
            
            let result1 = mul(mul(a_vec, b_vec), c_vec);
            let result2 = mul(a_vec, mul(b_vec, c_vec));
            
            // (a * b) * c == a * (b * c)
            prop_assert!((result1.horizontal_sum() - result2.horizontal_sum()).abs() < 1e-5);
        }
    }
}
```

**Key Points**:
- Proptest generates thousands of test cases automatically
- `proptest::num::f32::NORMAL` generates normal floating-point values
- Can combine strategies: `NORMAL | SUBNORMAL` for edge cases
- Tests run across all backends via `DefaultSimdVector`

---

## Performance Validation

### Benchmarking Example

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rigel_math::{Block64, DefaultSimdVector};
use rigel_math::ops::mul;

fn benchmark_gain(c: &mut Criterion) {
    let mut input = Block64::new();
    let mut output = Block64::new();
    
    c.bench_function("gain_simd", |b| {
        b.iter(|| {
            let gain_vec = DefaultSimdVector::splat(2.0);
            for (in_chunk, out_chunk) in input.as_chunks::<DefaultSimdVector>()
                .iter()
                .zip(output.as_chunks_mut::<DefaultSimdVector>())
            {
                *out_chunk = mul(black_box(*in_chunk), black_box(gain_vec));
            }
        })
    });
}

criterion_group!(benches, benchmark_gain);
criterion_main!(benches);
```

Run benchmarks:
```bash
# Wall-clock time (Criterion)
cargo bench --features avx2

# Instruction counts (iai-callgrind)
cargo bench --bench iai_benches --features avx2
```

---

## Common Patterns

### Pattern 1: Block Processing Loop

```rust
for (in_chunk, out_chunk) in input.as_chunks::<DefaultSimdVector>()
    .iter()
    .zip(output.as_chunks_mut::<DefaultSimdVector>())
{
    // Process SIMD chunk
    *out_chunk = process(*in_chunk);
}
```

### Pattern 2: Denormal Protection

```rust
fn process_audio(...) {
    let _guard = DenormalGuard::new();
    // All processing happens with denormal protection
    // Guard restores FPU state when dropped
}
```

### Pattern 3: Backend-Agnostic Code

```rust
// Write once, compiles to optimal SIMD for each backend
fn generic_process<V: SimdVector>(input: V, gain: V::Scalar) -> V {
    let gain_vec = V::splat(gain);
    mul(input, gain_vec)
}
```

---

## Next Steps

1. **Explore User Stories**: Each example maps to specific user stories in spec.md
2. **Run Tests**: `cargo test --features avx2` (or scalar, avx512, neon)
3. **Benchmark**: `cargo bench --features avx2`
4. **Integrate**: Add rigel-math to rigel-dsp and replace scalar math operations

For detailed API documentation, see `contracts/api-surface.md`.
