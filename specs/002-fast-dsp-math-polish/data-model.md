# Data Model: Fast DSP Math Library

**Feature**: 001-fast-dsp-math
**Date**: 2025-11-18

## Overview

This document describes the key types and abstractions in the rigel-math SIMD library. Since this is a pure computational library (not a data-oriented application), the "data model" consists of trait definitions, type abstractions, and their relationships rather than persistent entities.

---

## Core Abstractions

### SimdVector Trait

**Purpose**: Core abstraction for SIMD vector operations across all backends

**Type Parameters**:
- `Self::Scalar`: Underlying scalar type (f32 or f64)
- `Self::Mask`: Associated mask type for comparison results

**Operations**:
- Arithmetic: `add`, `sub`, `mul`, `div`, `neg`, `abs`
- FMA: `fma(a, b, c)` → `a * b + c`
- Min/Max: `min`, `max`
- Comparison: `lt`, `gt`, `eq` → returns `Self::Mask`
- Blending: `select(mask, true_val, false_val)`
- Horizontal: `horizontal_sum`, `horizontal_max`, `horizontal_min`
- Load/Store: `from_slice`, `to_slice`

**Implementations**:
- `ScalarVector<T>`: Always available, wraps single `T`
- `Avx2Vector<T>`: 8 floats (256-bit) for x86-64 with AVX2
- `Avx512Vector<T>`: 16 floats (512-bit) for x86-64 with AVX512
- `NeonVector<T>`: 4 floats (128-bit) for ARM64 with NEON

**Invariants**:
- All backends must produce results within documented error bounds for same input
- Deterministic execution: same input → same output across runs
- No allocations, no panics (graceful saturation for edge cases)

---

### SimdMask Trait

**Purpose**: Type-safe mask type for conditional SIMD operations

**Operations**:
- `all()`: Returns true if all lanes are set
- `any()`: Returns true if any lane is set
- `none()`: Returns true if no lanes are set
- Bitwise: `and`, `or`, `not`, `xor`

**Implementations**:
- `ScalarMask`: Single bool
- `Avx2Mask`: 8-lane mask (256-bit)
- `Avx512Mask`: 16-lane mask (512-bit)
- `NeonMask`: 4-lane mask (128-bit)

---

### AudioBlock<T, const N: usize>

**Purpose**: Fixed-size aligned audio buffer for block processing

**Fields**:
- `samples: [T; N]`: Fixed-size array of samples

**Type Parameters**:
- `T`: Sample type (usually f32)
- `N`: Block size (64 or 128 samples)

**Type Aliases**:
- `Block64 = AudioBlock<f32, 64>`
- `Block128 = AudioBlock<f32, 128>`

**Methods**:
- `new() -> Self`: Create zero-initialized block
- `from_slice(slice: &[T]) -> Self`: Copy from slice (panics if wrong size)
- `as_chunks<V: SimdVector>() -> &[V]`: View as SIMD chunks (immutable)
- `as_chunks_mut<V: SimdVector>() -> &mut [V]`: View as SIMD chunks (mutable)
- `len() -> usize`: Always returns `N`

**Alignment**:
- AVX2: 32-byte aligned
- AVX512: 64-byte aligned
- NEON: 16-byte aligned
- Scalar: Natural alignment

**Memory Layout**:
- Samples are tightly packed in memory
- For stereo: Use two separate blocks (planar layout) or interleaved access patterns
- Remainder handling: Documented pattern for buffers not evenly divisible by SIMD width

---

### LookupTable<T, const SIZE: usize>

**Purpose**: Pre-computed function values for wavetable synthesis and tabular functions

**Fields**:
- `values: [T; SIZE]`: Fixed-size array of table values

**Type Parameters**:
- `T`: Sample type (usually f32)
- `SIZE`: Table size (power of 2, typically 256-8192)

**Methods**:
- `from_fn<F>(f: F) -> Self` where `F: Fn(usize) -> T`: Generate table from function
- `lookup_linear(phase: T) -> T`: Linear interpolation
- `lookup_linear_simd<V: SimdVector>(phases: V) -> V`: Vectorized linear interpolation
- `lookup_cubic(phase: T) -> T`: Cubic interpolation
- `lookup_cubic_simd<V: SimdVector>(phases: V) -> V`: Vectorized cubic interpolation

**Index Wrapping**:
- `Wrap`: phase % SIZE (modulo wrapping)
- `Mirror`: phase reflects at boundaries
- `Clamp`: phase saturates at 0 and SIZE-1

**Invariants**:
- SIZE must be compile-time constant (const generic)
- Interpolation preserves phase continuity
- SIMD gather operations provide correct per-lane indexing

---

### DenormalGuard

**Purpose**: RAII wrapper for enabling denormal protection

**Fields**:
- `previous_state: u32` (x86-64 MXCSR or ARM64 FPCR state)

**Methods**:
- `new() -> Self`: Enable FTZ/DAZ flags, save previous state
- `Drop`: Restore previous state

**Platform-Specific Behavior**:
- **x86-64**: Sets FTZ (flush-to-zero) and DAZ (denormals-are-zero) flags in MXCSR
- **ARM64**: Sets FZ (flush-to-zero) flag in FPCR
- **Fallback**: DC offset if hardware flags unavailable

**Invariants**:
- No audible artifacts (THD+N < -96dB)
- Prevents performance degradation when processing silence
- Thread-local (does not affect other threads)

---

## Type Relationships

```text
┌─────────────────────────────────────────────────────────────┐
│                     SimdVector Trait                        │
│  (add, sub, mul, div, fma, min, max, lt, gt, select, ...)  │
└─────────────────────────────────────────────────────────────┘
                            ▲
                            │ implements
         ┌──────────────────┼──────────────────┬─────────────┐
         │                  │                  │             │
┌────────┴────────┐  ┌──────┴─────┐  ┌────────┴────┐  ┌─────┴──────┐
│ ScalarVector    │  │ Avx2Vector │  │Avx512Vector │  │ NeonVector │
│ (1 lane)        │  │ (8 lanes)  │  │ (16 lanes)  │  │ (4 lanes)  │
└─────────────────┘  └────────────┘  └─────────────┘  └────────────┘
                                                   
┌─────────────────────────────────────────────────────────────┐
│                 AudioBlock<T, const N: usize>               │
│  (Fixed-size aligned buffer for block processing)          │
└─────────────────────────────────────────────────────────────┘
         │
         │ contains / views as
         ▼
┌─────────────────────────────────────────────────────────────┐
│                   &[SimdVector]                             │
│  (Iterator over SIMD chunks via as_chunks/as_chunks_mut)   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              LookupTable<T, const SIZE: usize>              │
│  (Pre-computed function values with interpolation)          │
└─────────────────────────────────────────────────────────────┘
         │
         │ provides
         ▼
┌─────────────────────────────────────────────────────────────┐
│      lookup_linear_simd(phases: V) -> V                     │
│  (Vectorized table lookup using gather operations)          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     DenormalGuard                           │
│  (RAII wrapper for FTZ/DAZ denormal protection)            │
└─────────────────────────────────────────────────────────────┘
```

---

## Backend Selection

Backends are selected at compile time via cargo features:

```rust
// In user code (rigel-dsp):
use rigel_math::DefaultSimdVector; // Resolves to selected backend

// With --features scalar
type DefaultSimdVector = ScalarVector<f32>;

// With --features avx2
type DefaultSimdVector = Avx2Vector<f32>;

// With --features avx512
type DefaultSimdVector = Avx512Vector<f32>;

// With --features neon
type DefaultSimdVector = NeonVector<f32>;
```

**Feature Flags** (mutually exclusive):
- `scalar` (default): Always-available fallback
- `avx2`: Requires x86-64 with AVX2 support
- `avx512`: Requires x86-64 with AVX512 support
- `neon`: Requires ARM64 with NEON support

**Compile-Time Enforcement**:
- Only one backend feature can be active per build
- Backend mismatch detected at compile time (not runtime)
- Zero runtime dispatch overhead

---

## Error Handling

Since this is a no_std real-time library, error handling follows these patterns:

**No Panics**: All operations handle edge cases gracefully:
- NaN inputs → saturate to safe value or preserve NaN per IEEE 754
- Infinity inputs → saturate or preserve per operation semantics
- Out-of-bounds table access → wrap, mirror, or clamp per IndexMode
- Division by zero → saturate to ±infinity or max value

**Deterministic**: All operations have bounded, predictable worst-case execution time:
- No dynamic allocation
- No branching on data (only on compile-time configuration)
- SIMD operations with consistent lane behavior

**Error Bounds**: Math kernels document maximum error:
- `tanh`: <0.1% error
- `exp`: Sub-nanosecond throughput, documented error bounds
- `log1p`: <0.001% error for frequency calculations
- `sin/cos`: <-100dB THD harmonic distortion

---

## Validation Rules

**Property-Based Testing** (via proptest):
- Commutativity: `a + b == b + a` (for commutative ops)
- Associativity: `(a + b) + c == a + (b + c)` (for associative ops)
- Distributivity: `a * (b + c) == a * b + a * c` (for applicable ops)
- Error bounds: `|result - reference| < tolerance`
- Backend consistency: `scalar(x) ≈ simd(x)` within error bounds

**Backend Consistency**:
- All backends produce bit-identical results OR results within documented error tolerance
- Tested across normal values, denormals, edge cases (NaN, infinity, zero)

**Performance Validation**:
- AVX2: 4-8x speedup vs scalar
- AVX512: 8-16x speedup vs scalar
- NEON: 4-8x speedup vs scalar
- Zero-cost abstraction: trait calls compile to raw intrinsics (assembly inspection)

---

## Summary

The rigel-math data model consists of:
1. **Trait abstractions** (`SimdVector`, `SimdMask`) for platform-independent SIMD operations
2. **Backend implementations** (scalar, AVX2, AVX512, NEON) selected at compile time
3. **Block processing types** (`AudioBlock`) for efficient SIMD-friendly memory layouts
4. **Lookup table infrastructure** (`LookupTable`) for wavetable synthesis
5. **Denormal protection** (`DenormalGuard`) for real-time performance stability

All types enforce no_std, no-allocation, deterministic execution constraints required for real-time audio DSP.
