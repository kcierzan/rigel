# API Surface: rigel-math

**Feature**: 001-fast-dsp-math
**Date**: 2025-11-18

## Overview

This document defines the public API surface for the rigel-math SIMD library. The API is designed as a Rust library with trait-based abstractions, not a REST/GraphQL API.

---

## Public Modules

### `rigel_math::traits`

Core SIMD abstraction traits that all backends implement.

```rust
pub trait SimdVector: Copy + Clone {
    type Scalar;
    type Mask: SimdMask;
    
    const LANES: usize;
    
    // Construction
    fn splat(value: Self::Scalar) -> Self;
    fn from_slice(slice: &[Self::Scalar]) -> Self;
    fn to_slice(self, slice: &mut [Self::Scalar]);
    
    // Arithmetic
    fn add(self, rhs: Self) -> Self;
    fn sub(self, rhs: Self) -> Self;
    fn mul(self, rhs: Self) -> Self;
    fn div(self, rhs: Self) -> Self;
    fn neg(self) -> Self;
    fn abs(self) -> Self;
    
    // FMA
    fn fma(self, b: Self, c: Self) -> Self; // self * b + c
    
    // Min/Max
    fn min(self, rhs: Self) -> Self;
    fn max(self, rhs: Self) -> Self;
    
    // Comparison (returns masks)
    fn lt(self, rhs: Self) -> Self::Mask;
    fn gt(self, rhs: Self) -> Self::Mask;
    fn eq(self, rhs: Self) -> Self::Mask;
    
    // Blending
    fn select(mask: Self::Mask, true_val: Self, false_val: Self) -> Self;
    
    // Horizontal operations
    fn horizontal_sum(self) -> Self::Scalar;
    fn horizontal_max(self) -> Self::Scalar;
    fn horizontal_min(self) -> Self::Scalar;
}

pub trait SimdMask: Copy + Clone {
    fn all(self) -> bool;
    fn any(self) -> bool;
    fn none(self) -> bool;
    
    fn and(self, rhs: Self) -> Self;
    fn or(self, rhs: Self) -> Self;
    fn not(self) -> Self;
    fn xor(self, rhs: Self) -> Self;
}
```

**User Story Mapping**: US1 (SIMD Abstraction Layer)

---

### `rigel_math::backends`

Backend implementations (selected at compile time via features).

```rust
// Scalar backend (always available)
pub struct ScalarVector<T>(T);
pub struct ScalarMask(bool);

// AVX2 backend (x86-64 with AVX2, feature="avx2")
#[cfg(feature = "avx2")]
pub struct Avx2Vector<T>(/* __m256 or __m256d */);
#[cfg(feature = "avx2")]
pub struct Avx2Mask(/* __m256 */);

// AVX512 backend (x86-64 with AVX512, feature="avx512")
#[cfg(feature = "avx512")]
pub struct Avx512Vector<T>(/* __m512 or __m512d */);
#[cfg(feature = "avx512")]
pub struct Avx512Mask(/* __mmask16 */);

// NEON backend (ARM64 with NEON, feature="neon")
#[cfg(feature = "neon")]
pub struct NeonVector<T>(/* float32x4_t or float64x2_t */);
#[cfg(feature = "neon")]
pub struct NeonMask(/* uint32x4_t */);

// Default backend type alias (resolves based on active feature)
pub type DefaultSimdVector<T> = /* ScalarVector or Avx2Vector or ... */;
```

**User Story Mapping**: US1 (SIMD Abstraction Layer), US9 (Backend Selection)

---

### `rigel_math::block`

Fixed-size audio buffer types for block processing.

```rust
#[repr(align(64))]
pub struct AudioBlock<T, const N: usize> {
    samples: [T; N],
}

impl<T, const N: usize> AudioBlock<T, N> {
    pub fn new() -> Self;
    pub fn from_slice(slice: &[T]) -> Self;
    pub fn as_chunks<V: SimdVector>(&self) -> &[V];
    pub fn as_chunks_mut<V: SimdVector>(&mut self) -> &mut [V];
    pub fn len(&self) -> usize;
}

// Type aliases
pub type Block64 = AudioBlock<f32, 64>;
pub type Block128 = AudioBlock<f32, 128>;
```

**User Story Mapping**: US2 (Block Processing Pattern)

---

### `rigel_math::ops`

High-level vector operations (convenience wrappers around trait methods).

```rust
// Arithmetic
pub fn add<V: SimdVector>(a: V, b: V) -> V;
pub fn sub<V: SimdVector>(a: V, b: V) -> V;
pub fn mul<V: SimdVector>(a: V, b: V) -> V;
pub fn div<V: SimdVector>(a: V, b: V) -> V;

// FMA
pub fn fma<V: SimdVector>(a: V, b: V, c: V) -> V;

// Min/Max/Clamp
pub fn min<V: SimdVector>(a: V, b: V) -> V;
pub fn max<V: SimdVector>(a: V, b: V) -> V;
pub fn clamp<V: SimdVector>(value: V, min_val: V, max_val: V) -> V;

// Comparison
pub fn lt<V: SimdVector>(a: V, b: V) -> V::Mask;
pub fn gt<V: SimdVector>(a: V, b: V) -> V::Mask;
pub fn eq<V: SimdVector>(a: V, b: V) -> V::Mask;

// Horizontal operations
pub fn horizontal_sum<V: SimdVector>(vec: V) -> V::Scalar;
pub fn horizontal_max<V: SimdVector>(vec: V) -> V::Scalar;
pub fn horizontal_min<V: SimdVector>(vec: V) -> V::Scalar;
```

**User Story Mapping**: US3 (Core Vector Operations)

---

### `rigel_math::math`

Fast math kernels for DSP (tanh, exp, log, sin/cos, inverse, etc.).

```rust
// Hyperbolic tangent (waveshaping, soft clipping)
pub fn tanh<V: SimdVector>(x: V) -> V;
pub fn tanh_fast<V: SimdVector>(x: V) -> V; // Lower precision, faster

// Exponential (envelopes, decay)
pub fn exp<V: SimdVector>(x: V) -> V;
pub fn exp_envelope<V: SimdVector>(x: V) -> V; // Optimized for envelope generation

// Logarithms (frequency calculations)
pub fn log<V: SimdVector>(x: V) -> V;
pub fn log1p<V: SimdVector>(x: V) -> V; // log(1 + x), accurate near zero
pub fn log2<V: SimdVector>(x: V) -> V;
pub fn log10<V: SimdVector>(x: V) -> V;

// Trigonometric (oscillators, phase modulation)
pub fn sin<V: SimdVector>(x: V) -> V;
pub fn cos<V: SimdVector>(x: V) -> V;
pub fn sincos<V: SimdVector>(x: V) -> (V, V); // Simultaneous sin+cos

// Inverse/reciprocal (division-free algorithms)
pub fn recip<V: SimdVector>(x: V) -> V; // 1/x with Newton-Raphson refinement
pub fn recip_rough<V: SimdVector>(x: V) -> V; // Hardware RCP estimate only (faster, lower precision)

// Square root
pub fn sqrt<V: SimdVector>(x: V) -> V;
pub fn rsqrt<V: SimdVector>(x: V) -> V; // 1/sqrt(x)

// Power
pub fn pow<V: SimdVector>(base: V, exponent: V::Scalar) -> V; // Vectorized base, scalar exponent

// Atan (phase calculations)
pub fn atan<V: SimdVector>(x: V) -> V;
pub fn atan2<V: SimdVector>(y: V, x: V) -> V;
```

**User Story Mapping**: US4 (Fast Math Kernels)

---

### `rigel_math::table`

Lookup table infrastructure with interpolation.

```rust
pub struct LookupTable<T, const SIZE: usize> {
    values: [T; SIZE],
}

pub enum IndexMode {
    Wrap,    // Modulo wrapping
    Mirror,  // Reflect at boundaries
    Clamp,   // Saturate at edges
}

impl<T, const SIZE: usize> LookupTable<T, SIZE> {
    pub fn from_fn<F>(f: F) -> Self
    where
        F: Fn(usize) -> T;
    
    // Scalar lookups
    pub fn lookup_linear(&self, phase: T, mode: IndexMode) -> T;
    pub fn lookup_cubic(&self, phase: T, mode: IndexMode) -> T;
    
    // SIMD lookups
    pub fn lookup_linear_simd<V: SimdVector>(&self, phases: V, mode: IndexMode) -> V;
    pub fn lookup_cubic_simd<V: SimdVector>(&self, phases: V, mode: IndexMode) -> V;
}
```

**User Story Mapping**: US5 (Lookup Table Infrastructure)

---

### `rigel_math::denormal`

Denormal number protection for real-time performance stability.

```rust
pub struct DenormalGuard {
    previous_state: u32,
}

impl DenormalGuard {
    /// Enable denormal protection (FTZ/DAZ on x86-64, FZ on ARM64)
    pub fn new() -> Self;
    
    /// Check if denormal protection is available on this platform
    pub fn is_available() -> bool;
}

// Drop implementation restores previous FPU state
impl Drop for DenormalGuard {
    fn drop(&mut self);
}

// Convenience function for block processing
pub fn with_denormal_protection<F, R>(f: F) -> R
where
    F: FnOnce() -> R;
```

**User Story Mapping**: US6 (Denormal Handling)

---

### `rigel_math::saturate`

Saturation curves for waveshaping and anti-clipping.

```rust
// Soft clipping (tanh-based, symmetric)
pub fn saturate_soft_clip<V: SimdVector>(x: V) -> V;

// Tube-style saturation (asymmetric, warm harmonics)
pub fn saturate_tube<V: SimdVector>(x: V, drive: V::Scalar) -> V;

// Tape-style saturation (high-frequency rolloff)
pub fn saturate_tape<V: SimdVector>(x: V, drive: V::Scalar) -> V;
```

**User Story Mapping**: US7 (Soft Saturation and Waveshaping)

---

### `rigel_math::crossfade`

Crossfade and parameter ramping for click-free transitions.

```rust
// Crossfade curves
pub fn crossfade_linear<V: SimdVector>(a: V, b: V, mix: V::Scalar) -> V;
pub fn crossfade_equal_power<V: SimdVector>(a: V, b: V, mix: V::Scalar) -> V;
pub fn crossfade_scurve<V: SimdVector>(a: V, b: V, mix: V::Scalar) -> V;

// Parameter ramping
pub struct ParameterRamp<T> {
    start: T,
    end: T,
    samples_remaining: usize,
    increment: T,
}

impl<T> ParameterRamp<T> {
    pub fn new(start: T, end: T, duration_samples: usize) -> Self;
    pub fn next_sample(&mut self) -> T;
    pub fn is_complete(&self) -> bool;
    
    // Fill entire block with ramped values
    pub fn fill_block<V: SimdVector, const N: usize>(&mut self, block: &mut AudioBlock<T, N>);
}
```

**User Story Mapping**: US8 (Crossfade and Ramping Utilities)

---

## Feature Flags

The library uses Cargo features for compile-time backend selection:

```toml
[features]
default = ["scalar"]
scalar = []      # Always-available fallback (no SIMD)
avx2 = []        # x86-64 with AVX2 support
avx512 = []      # x86-64 with AVX-512 support
neon = []        # ARM64 with NEON support
```

**Mutual Exclusivity**: Only one backend feature should be active per build. Attempting to enable multiple SIMD backends will result in a compile error.

**User Story Mapping**: US9 (Backend Selection and Benchmarking)

---

## Usage Examples

### Example 1: Simple Vector Addition (Backend-Agnostic)

```rust
use rigel_math::DefaultSimdVector;
use rigel_math::ops::add;

// Works with any backend (scalar, AVX2, AVX512, NEON)
let a = DefaultSimdVector::splat(1.0);
let b = DefaultSimdVector::splat(2.0);
let result = add(a, b);

assert_eq!(result.horizontal_sum(), 3.0 * DefaultSimdVector::LANES as f32);
```

### Example 2: Block Processing with Denormal Protection

```rust
use rigel_math::{Block64, DefaultSimdVector, DenormalGuard};
use rigel_math::ops::mul;

fn process_audio(input: &Block64, output: &mut Block64, gain: f32) {
    let _guard = DenormalGuard::new(); // Enable denormal protection
    
    let gain_vec = DefaultSimdVector::splat(gain);
    
    for (in_chunk, out_chunk) in input.as_chunks::<DefaultSimdVector>()
        .iter()
        .zip(output.as_chunks_mut::<DefaultSimdVector>().iter_mut())
    {
        *out_chunk = mul(*in_chunk, gain_vec);
    }
}
```

### Example 3: Fast Math Kernel Usage

```rust
use rigel_math::{Block64, DefaultSimdVector};
use rigel_math::math::tanh;

fn waveshaper(input: &Block64, output: &mut Block64, drive: f32) {
    let drive_vec = DefaultSimdVector::splat(drive);
    
    for (in_chunk, out_chunk) in input.as_chunks::<DefaultSimdVector>()
        .iter()
        .zip(output.as_chunks_mut::<DefaultSimdVector>().iter_mut())
    {
        let driven = mul(*in_chunk, drive_vec);
        *out_chunk = tanh(driven);
    }
}
```

### Example 4: Wavetable Oscillator with Lookup Table

```rust
use rigel_math::{LookupTable, DefaultSimdVector, Block64, IndexMode};

const TABLE_SIZE: usize = 2048;
type SineTable = LookupTable<f32, TABLE_SIZE>;

fn generate_wavetable() -> SineTable {
    SineTable::from_fn(|i| {
        let phase = (i as f32 / TABLE_SIZE as f32) * 2.0 * std::f32::consts::PI;
        phase.sin()
    })
}

fn oscillator_process(
    table: &SineTable,
    phase_block: &Block64,
    output: &mut Block64
) {
    for (phase_chunk, out_chunk) in phase_block.as_chunks::<DefaultSimdVector>()
        .iter()
        .zip(output.as_chunks_mut::<DefaultSimdVector>().iter_mut())
    {
        *out_chunk = table.lookup_linear_simd(*phase_chunk, IndexMode::Wrap);
    }
}
```

---

## Error Bounds and Performance Guarantees

All public functions document their error bounds and performance characteristics:

| Function | Error Bound | THD (if applicable) | Performance Target |
|----------|-------------|---------------------|--------------------|
| `tanh` | <0.1% | N/A | 8-16x vs scalar |
| `exp` | <0.1% | N/A | Sub-nanosecond/sample |
| `log1p` | <0.001% | N/A | 8-16x vs scalar |
| `sin` / `cos` | <0.1% amplitude | <-100dB | 8-16x vs scalar |
| `recip` | <0.01% | N/A | 5-10x vs division |
| `recip_rough` | <0.1% | N/A | 10-20x vs division |
| `lookup_linear_simd` | Exact (fp32 precision) | N/A | <10ns/sample |
| `lookup_cubic_simd` | Exact (fp32 precision) | N/A | <20ns/sample |

---

## Testing API (Test-Only Exports)

For testing purposes, the library exposes additional utilities:

```rust
#[cfg(test)]
pub mod test_utils {
    // Reference implementations using libm (for accuracy comparison)
    pub fn reference_tanh(x: f32) -> f32;
    pub fn reference_exp(x: f32) -> f32;
    pub fn reference_sin(x: f32) -> f32;
    pub fn reference_cos(x: f32) -> f32;
    
    // Property test strategies (proptest)
    pub fn normal_floats() -> impl Strategy<Value = f32>;
    pub fn denormal_floats() -> impl Strategy<Value = f32>;
    pub fn edge_case_floats() -> impl Strategy<Value = f32>; // NaN, infinity, zero
    
    // Backend consistency helpers
    pub fn assert_backend_consistency<V: SimdVector>(
        scalar_result: V,
        simd_result: V,
        tolerance: f32
    );
}
```

**User Story Mapping**: US10 (Comprehensive Test Coverage)

---

## Summary

The rigel-math API provides:
- **Trait-based SIMD abstraction** (`SimdVector`, `SimdMask`) for backend-agnostic code
- **Compile-time backend selection** via cargo features (scalar, avx2, avx512, neon)
- **Block processing types** (`AudioBlock`) for efficient SIMD-friendly layouts
- **Core vector operations** (arithmetic, FMA, min/max, compare, horizontal)
- **Fast math kernels** (tanh, exp, log, sin/cos, inverse, sqrt, pow)
- **Lookup table infrastructure** with linear/cubic interpolation
- **Denormal protection** for real-time performance stability
- **Saturation and crossfade utilities** for musical DSP

All APIs enforce no_std, no-allocation, deterministic execution constraints required for real-time audio DSP.
