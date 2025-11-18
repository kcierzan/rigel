# SIMD Math Library Research

This document summarizes research findings for implementing a high-performance SIMD math library for audio DSP in Rust. The goal is to provide deterministic, real-time-safe mathematical operations with zero allocations for the `rigel-dsp` crate.

## 1. Property-Based Testing Framework

### Decision: Use Proptest

**Recommendation:** Adopt `proptest` (v1.5.0+) for property-based testing of SIMD mathematical operations.

### Rationale

1. **Flexibility**: Proptest uses explicit `Strategy` objects, allowing multiple different strategies for the same type. This is crucial for testing edge cases in SIMD operations (denormals, infinities, NaN handling, range boundaries).

2. **Composability**: Strategies can be composed through `prop_map` and combinators, with automatic shrinking in terms of input types. For SIMD testing, this means we can define strategies for vector inputs and have meaningful shrinking when tests fail.

3. **Constraint Handling**: Proptest strategies are aware of simple constraints and avoid generating values that violate them, which is essential for testing mathematical invariants.

4. **Integration**: Works seamlessly with `cargo test` and integrates well with Criterion benchmarks through feature flags.

5. **Maintenance Status**: While in passive maintenance mode (last commit October 2024), the crate is mature and feature-complete with 116 open issues and 18 PRs, indicating continued community support.

### Alternatives Considered

**QuickCheck:**
- **Pros**:
  - Simpler API, minimal macros
  - 2x-10x faster for generating complex values due to stateless shrinking
  - More mature (older codebase)
- **Cons**:
  - Only one generator/shrinker per type (inflexible for SIMD edge cases)
  - Less compositional for complex test scenarios
  - Harder to express constraints on generated values

**Verdict**: QuickCheck's performance advantage is offset by Proptest's superior flexibility for testing mathematical invariants across different SIMD backends with varying edge case requirements.

### Testing Mathematical Invariants

Key properties to test for SIMD math operations:

1. **Commutativity**: `f(a, b) == f(b, a)` for operations like addition, multiplication
2. **Associativity**: `f(f(a, b), c) == f(a, f(b, c))`
3. **Consistency across backends**: AVX2, AVX-512, and NEON implementations produce bit-identical results
4. **Error bounds**: Approximations stay within specified error tolerances
5. **Special value handling**: Correct behavior for `±0.0`, `±∞`, `NaN`, denormals

Example strategy for SIMD vector testing:

```rust
use proptest::prelude::*;

prop_compose! {
    fn simd_f32_vector()(values in prop::array::uniform4(-1000.0f32..1000.0f32)) -> [f32; 4] {
        values
    }
}

proptest! {
    #[test]
    fn test_tanh_approximation_bounds(x in simd_f32_vector()) {
        let result = fast_tanh_simd(x);
        for i in 0..4 {
            prop_assert!(result[i].abs() <= 1.0);
            prop_assert!((result[i] - x[i].tanh()).abs() < 0.001); // <0.1% error
        }
    }
}
```

### References

- [Proptest GitHub Repository](https://github.com/proptest-rs/proptest)
- [Proptest vs QuickCheck Comparison](https://altsysrq.github.io/proptest-book/proptest/vs-quickcheck.html)
- [Introduction to Property-Based Testing in Rust](https://lpalmieri.com/posts/an-introduction-to-property-based-testing-in-rust/)

---

## 2. Rust SIMD Best Practices (2024-2025)

### Decision: Use Trait-Based Abstractions with Runtime Dispatch

**Recommendation:** Implement trait-based SIMD abstractions using the `pulp` crate (v0.22.2+) for the initial implementation, with a path to migrate to `std::simd` once stabilized.

### Rationale

1. **Portability**: Trait abstractions allow writing SIMD code once that compiles for multiple architectures.

2. **Zero-Cost**: Properly designed traits with inline functions and monomorphization provide zero runtime overhead.

3. **Backend Selection**: Compile-time selection via cargo features combined with runtime CPU feature detection for optimal performance.

4. **Maintenance**: `pulp` is mature, powers the high-performance `faer` linear algebra library, and provides built-in multiversioning.

### Implementation Pattern

```rust
// Define a trait for SIMD operations
pub trait SimdMath {
    type F32x;

    fn fast_tanh(self, x: Self::F32x) -> Self::F32x;
    fn fast_exp(self, x: Self::F32x) -> Self::F32x;
    // ... other operations
}

// Runtime dispatch based on CPU features
pub fn get_simd_backend() -> impl SimdMath {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return Avx512Backend;
        } else if is_x86_feature_detected!("avx2") {
            return Avx2Backend;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return NeonBackend;
    }

    ScalarFallback
}
```

### Architecture-Specific Considerations

#### x86-64 (AVX2)
- **Availability**: ~75% of systems (Firefox hardware survey)
- **Alignment**: 32-byte (256-bit registers)
- **Key Instructions**: `_mm256_*` intrinsics from `std::arch::x86_64`
- **Enable**: `#[target_feature(enable = "avx2,fma")]`

#### x86-64 (AVX-512)
- **Availability**: Server/workstation class, newer consumer CPUs
- **Alignment**: 64-byte (512-bit registers)
- **Key Instructions**: `_mm512_*` intrinsics
- **Benefits**: 2x width over AVX2, improved gather/scatter, mask operations
- **Enable**: `#[target_feature(enable = "avx512f")]`

#### ARM (NEON)
- **Availability**: All ARMv8+ (Apple Silicon, modern ARM servers)
- **Alignment**: 16-byte (128-bit registers)
- **Key Instructions**: `vst1q_f32`, `vld1q_f32`, etc.
- **Note**: ARMv7 NEON forces flush-to-zero for denormals
- **Enable**: Enabled by default on `aarch64-apple-darwin`

### Ensuring Bit-Identical Results

**Challenge**: Different SIMD backends may produce slightly different results due to:
- Fused multiply-add (FMA) availability
- Different instruction scheduling
- Rounding mode differences

**Solution**:
1. Use identical polynomial coefficients across backends
2. Explicitly control FMA usage (enable for all or none)
3. Property-based testing to verify error bounds are consistent
4. Document that exact bit-for-bit reproducibility is not guaranteed across architectures, only error bounds

```rust
#[cfg(test)]
mod tests {
    proptest! {
        #[test]
        fn consistency_across_backends(x in -10.0f32..10.0f32) {
            let scalar = scalar_tanh(x);
            let avx2 = unsafe { avx2_tanh(x) };
            let neon = unsafe { neon_tanh(x) };

            // Verify all backends stay within error tolerance
            prop_assert!((scalar - avx2).abs() < 0.001);
            prop_assert!((scalar - neon).abs() < 0.001);
        }
    }
}
```

### Common Pitfalls to Avoid

1. **Misaligned Memory Access**: Always align SIMD data structures properly
   ```rust
   #[repr(align(32))]  // For AVX2
   struct AlignedBuffer([f32; 8]);
   ```

2. **Feature Detection Without Guards**: Never call SIMD intrinsics without verifying CPU support
   ```rust
   // WRONG
   let result = unsafe { _mm256_add_ps(a, b) };  // May crash!

   // CORRECT
   #[target_feature(enable = "avx2")]
   unsafe fn add_avx2(a: __m256, b: __m256) -> __m256 {
       _mm256_add_ps(a, b)
   }
   ```

3. **Over-Abstracting**: Keep hot paths simple; excessive abstraction can inhibit inlining
   ```rust
   // Mark critical functions as #[inline(always)]
   #[inline(always)]
   pub fn process_sample(x: f32) -> f32 {
       // Direct implementation, not through trait objects
   }
   ```

4. **Ignoring Remainder Handling**: SIMD processes fixed-width chunks; handle leftover samples
   ```rust
   fn process_buffer(samples: &mut [f32]) {
       let (chunks, remainder) = samples.as_chunks_mut::<8>();
       for chunk in chunks {
           // SIMD path
       }
       for sample in remainder {
           // Scalar fallback
       }
   }
   ```

5. **Assuming FTZ/DAZ is Set**: Always explicitly set denormal handling flags
   ```rust
   #[cfg(target_arch = "x86_64")]
   unsafe {
       use std::arch::x86_64::*;
       _mm_setcsr(_mm_getcsr() | 0x8040); // Set FTZ and DAZ
   }
   ```

### std::arch Usage Patterns

The `std::arch` module provides direct access to CPU intrinsics:

```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[target_feature(enable = "avx2,fma")]
unsafe fn fast_tanh_avx2(x: __m256) -> __m256 {
    // Polynomial coefficients
    let c1 = _mm256_set1_ps(1.0);
    let c3 = _mm256_set1_ps(0.33333333);
    let c5 = _mm256_set1_ps(0.13333333);

    let x2 = _mm256_mul_ps(x, x);
    let x3 = _mm256_mul_ps(x2, x);
    let x5 = _mm256_mul_ps(x3, x2);

    // tanh(x) ≈ x - x³/3 + x⁵/5 (for small x)
    let term1 = _mm256_mul_ps(x3, c3);
    let term2 = _mm256_mul_ps(x5, c5);

    _mm256_fmadd_ps(term2, c1, _mm256_sub_ps(x, term1))
}
```

### Compile-Time Backend Selection

Use cargo features for compile-time specialization:

```toml
[features]
default = ["runtime-dispatch"]
runtime-dispatch = []
force-avx2 = []
force-avx512 = []
force-neon = []
```

```rust
#[cfg(feature = "force-avx2")]
type SimdBackend = Avx2Backend;

#[cfg(feature = "force-avx512")]
type SimdBackend = Avx512Backend;

#[cfg(feature = "runtime-dispatch")]
fn get_backend() -> impl SimdMath {
    // Runtime CPU detection
}
```

### SIMD Library Ecosystem

#### pulp (Recommended for Initial Implementation)
- **Version**: 0.22.2
- **Backends**: AVX2, AVX-512 (x86-v3, x86-v4), NEON
- **Features**: Built-in multiversioning, safe abstractions
- **Used By**: `faer` (proven performance)
- **License**: MIT
- **Status**: Actively maintained

```toml
[dependencies]
pulp = { version = "0.22", default-features = false, features = ["std", "x86-v3"] }
```

#### macerator (Alternative with More Backends)
- **Version**: 0.2.9
- **Backends**: Extended instruction set support beyond pulp
- **Features**: Type-generic SIMD, generic programming support
- **License**: MIT OR Apache-2.0
- **Rust Version**: 1.81+
- **Status**: Actively developed (fork of pulp)

#### std::simd (Future Migration Path)
- **Status**: Unstable (nightly-only as of late 2024)
- **Pros**: Official Rust portable SIMD, will be standard
- **Cons**: Not yet stabilized, API may change
- **Timeline**: No confirmed stabilization date
- **Strategy**: Prepare for migration by keeping abstractions flexible

### References

- [The State of SIMD in Rust (2025)](https://shnatsel.medium.com/the-state-of-simd-in-rust-in-2025-32c263e5f53d)
- [std::arch Documentation](https://doc.rust-lang.org/std/arch/index.html)
- [pulp Crate](https://crates.io/crates/pulp)
- [macerator Crate](https://crates.io/crates/macerator)
- [Rust SIMD Performance Guide](https://rust-lang.github.io/packed_simd/perf-guide/)

---

## 3. Fast Math Approximation Techniques

### Overview

Audio DSP requires high-performance approximations that balance accuracy with computational efficiency. All approximations must vectorize well and maintain acceptable error bounds for audio quality (typically <0.1% for most functions, <-100dB THD for oscillators).

---

### 3.1 Hyperbolic Tangent (tanh)

**Use Cases**: Soft clipping, saturation, waveshaping, sigmoid activation

**Decision**: Fifth-order polynomial preprocessing

#### Recommended Algorithm

```rust
// Raph Levien's fifth-order polynomial (2e-4 accuracy, 0.55ns/sample)
#[inline(always)]
fn fast_tanh(x: f32) -> f32 {
    let x2 = x * x;
    let x3 = x2 * x;
    let x5 = x3 * x2;

    // Pre-process through odd polynomial
    let numerator = x + 0.16489087 * x3 + 0.00985468 * x5;

    // Clamp to prevent overflow
    let clamped = numerator.clamp(-3.0, 3.0);

    // Final rational approximation
    clamped * (27.0 + clamped * clamped) / (27.0 + 9.0 * clamped * clamped)
}
```

**Error Bounds**: <2e-4 (0.02%) maximum error

**Performance**: 0.55ns per sample on modern x86-64

**Vectorization**: Excellent - polynomial operations vectorize perfectly

#### Alternative: Rational Function (Padé Approximation)

```rust
// musicdsp.org rational approximation (2.6% max error, fastest)
#[inline(always)]
fn fast_tanh_rational(x: f32) -> f32 {
    let x_clamped = x.clamp(-3.0, 3.0);
    x_clamped * (27.0 + x_clamped * x_clamped) / (27.0 + 9.0 * x_clamped * x_clamped)
}
```

**Error Bounds**: ~2.6% maximum error in [-4.5, 4.5] range

**Advantages**:
- Fastest tanh approximation available
- C2-continuous at clipping boundaries
- Excellent for SSE/AVX with MIN/MAX and RCP instructions

**Trade-offs**: Lower accuracy but acceptable for overdrive/distortion effects

#### SIMD Implementation (AVX2)

```rust
#[target_feature(enable = "avx2,fma")]
unsafe fn tanh_avx2(x: __m256) -> __m256 {
    let c27 = _mm256_set1_ps(27.0);
    let c9 = _mm256_set1_ps(9.0);
    let c_min = _mm256_set1_ps(-3.0);
    let c_max = _mm256_set1_ps(3.0);

    // Clamp input
    let x_clamp = _mm256_min_ps(_mm256_max_ps(x, c_min), c_max);

    // x²
    let x2 = _mm256_mul_ps(x_clamp, x_clamp);

    // numerator = 27 + x²
    let num = _mm256_add_ps(c27, x2);

    // denominator = 27 + 9x²
    let denom = _mm256_fmadd_ps(c9, x2, c27);

    // result = x * num / denom
    _mm256_mul_ps(x_clamp, _mm256_div_ps(num, denom))
}
```

#### Musical Suitability

Per Raph Levien's analysis, tanh is the most musically appropriate sigmoid for audio because it models "the response of differential transistor pairs as used in the Moog ladder filter." It produces smooth distortion with favorable harmonic characteristics.

#### Alternatives Considered

- **Error Function (erf)**: Sharper response, 5e-4 accuracy, 0.86ns/sample - good for specific filter modeling
- **Reciprocal Square Root**: `x / √(1 + x²)` - very fast but more low-level distortion
- **Lookup Tables**: Rejected - slower than polynomial, worse for cache, interpolation overhead

#### References

- [Raph Levien: A Few of My Favorite Sigmoids](https://raphlinus.github.io/audio/2018/09/05/sigmoid.html)
- [musicdsp.org: Rational tanh Approximation](https://www.musicdsp.org/en/latest/Other/238-rational-tanh-approximation.html)
- [musicdsp.org: Reasonably Accurate/Fast tanh](https://www.musicdsp.org/en/latest/Other/178-reasonably-accurate-fastish-tanh-approximation.html)

---

### 3.2 Exponential (exp)

**Use Cases**: Envelope generation (ADSR), exponential decay, frequency conversion

**Decision**: Chebyshev polynomial with range reduction

#### Recommended Algorithm (AVX2)

Based on `avx_mathfun` implementation:

```rust
// Range reduction: exp(x) = 2^(x/ln2) = 2^floor(x/ln2) * 2^frac(x/ln2)
#[target_feature(enable = "avx2,fma")]
unsafe fn exp_avx2(x: __m256) -> __m256 {
    let ln2_inv = _mm256_set1_ps(1.4426950408889634); // 1/ln(2)
    let ln2_hi = _mm256_set1_ps(0.693359375);
    let ln2_lo = _mm256_set1_ps(-2.12194440e-4);

    // Clamp to prevent overflow
    let x_clamped = _mm256_min_ps(x, _mm256_set1_ps(88.0));

    // Range reduction
    let fx = _mm256_mul_ps(x_clamped, ln2_inv);
    let fx_rounded = _mm256_round_ps(fx, _MM_FROUND_TO_NEAREST_INT);

    // Reconstruct reduced range
    let x_reduced = _mm256_fnmadd_ps(fx_rounded, ln2_hi, x_clamped);
    let x_reduced = _mm256_fnmadd_ps(fx_rounded, ln2_lo, x_reduced);

    // Chebyshev polynomial approximation for 2^x in [0,1]
    let c1 = _mm256_set1_ps(1.0);
    let c2 = _mm256_set1_ps(0.693147180559945);
    let c3 = _mm256_set1_ps(0.240226506959101);
    let c4 = _mm256_set1_ps(0.055504108664821);
    let c5 = _mm256_set1_ps(0.009618129107597);
    let c6 = _mm256_set1_ps(0.001333355814670);

    let x2 = _mm256_mul_ps(x_reduced, x_reduced);

    // Polynomial evaluation using Horner's method with FMA
    let poly = c6;
    let poly = _mm256_fmadd_ps(poly, x_reduced, c5);
    let poly = _mm256_fmadd_ps(poly, x_reduced, c4);
    let poly = _mm256_fmadd_ps(poly, x_reduced, c3);
    let poly = _mm256_fmadd_ps(poly, x_reduced, c2);
    let poly = _mm256_fmadd_ps(poly, x_reduced, c1);

    // Scale by 2^floor(x/ln2) using exponent manipulation
    // ... (bitwise operations to reconstruct full result)
}
```

**Error Bounds**:
- With AVX2 (no FMA): 4.1e-6 maximum relative error for x ∈ [-84, 84]
- With FMA: 3.0e-7 maximum relative error

**Performance**: ~2x faster than standard library exp

#### Fast Exponential for Audio Envelopes

For ADSR envelopes specifically, a simpler incremental approach is often sufficient:

```rust
// Fast exponential patch computation (musicdsp.org)
struct ExponentialEnvelope {
    current: f32,
    multiplier: f32,  // r = exp(rate)
    delta: f32,       // d = small offset for stability
}

impl ExponentialEnvelope {
    #[inline(always)]
    fn next_sample(&mut self) -> f32 {
        // Only 1 multiplication + 1 addition per sample!
        self.current = self.current * self.multiplier + self.delta;
        self.current
    }
}
```

**Performance**: 1 multiply + 1 add per sample (extremely fast)

**Use Case**: Attack, decay, release envelope stages

#### Alternatives Considered

- **Lookup Tables with Interpolation**: 4x slower than polynomial, rejected
- **herumi/fmath Library**: 15.7-18.5x speedup for array operations (AVX-512), but requires minimum array size of 16+ elements and has undefined exact error bounds

#### References

- [avx_mathfun on GitHub](https://github.com/reyoung/avx_mathfun)
- [herumi/fmath Library](https://github.com/herumi/fmath)
- [musicdsp.org: Fast Exponential Envelope Generator](https://www.musicdsp.org/en/latest/Synthesis/189-fast-exponential-envelope-generator.html)
- [Fast Vectorizable Math Approximations (INRIA)](http://gallium.inria.fr/blog/fast-vectorizable-math-approx/)

---

### 3.3 Logarithm (log, log1p)

**Use Cases**: Frequency calculations, dB conversion (lin2db), log-frequency spacing

**Decision**: Bit manipulation + polynomial approximation

#### Recommended Algorithm

From INRIA Gallium fast vectorizable math:

```rust
#[inline(always)]
fn fast_log(x: f32) -> f32 {
    // Extract exponent and mantissa via bit manipulation
    let bits = x.to_bits();
    let exponent = ((bits >> 23) & 0xFF) as i32 - 127;
    let mantissa_bits = (bits & 0x007FFFFF) | 0x3F800000; // Normalized to [1, 2)
    let mantissa = f32::from_bits(mantissa_bits);

    // Polynomial approximation for log in [1, 2) using Remez/Sollya
    let c0 = -1.7417939;
    let c1 = 2.8212026;
    let c2 = -1.4699568;
    let c3 = 0.3969973;

    let m2 = mantissa * mantissa;
    let poly = c0 + c1 * mantissa + c2 * m2 + c3 * m2 * mantissa;

    // Combine: log(x) = log(2^exp * mantissa) = exp*ln(2) + log(mantissa)
    exponent as f32 * 0.69314718 + poly
}
```

**Error Bounds**:
- Mean absolute error: 3.756e-5
- Max absolute error: 6.104e-5

**Performance**: 7.5x faster than standard library, vectorizes perfectly

#### log1p (log(1 + x)) for Small Values

For frequency calculations often need `log(1 + x)` where x is small:

```rust
#[inline(always)]
fn fast_log1p(x: f32) -> f32 {
    if x.abs() < 0.1 {
        // Taylor series for small x: log(1+x) ≈ x - x²/2 + x³/3
        let x2 = x * x;
        x * (1.0 - x * 0.5 + x2 * 0.33333333)
    } else {
        fast_log(1.0 + x)
    }
}
```

#### Vectorization Performance

Per INRIA results:
- Vectorized log: 38x speedup vs. standard library
- Vectorized exp: 21x speedup vs. standard library

The key advantage is avoiding lookup tables which inhibit compiler auto-vectorization due to random-access memory patterns.

#### Audio-Specific: dB Conversion (lin2db)

```rust
// 20 * log10(x) = 20 * log2(x) / log2(10)
#[inline(always)]
fn lin2db(x: f32) -> f32 {
    const LOG2_10_INV: f32 = 0.30102999566; // 1/log2(10)
    20.0 * fast_log2(x) * LOG2_10_INV
}

#[inline(always)]
fn fast_log2(x: f32) -> f32 {
    // Similar to fast_log but using base-2
    let bits = x.to_bits();
    let exponent = ((bits >> 23) & 0xFF) as i32 - 127;
    let mantissa_bits = (bits & 0x007FFFFF) | 0x3F800000;
    let mantissa = f32::from_bits(mantissa_bits);

    // Polynomial for log2(m) where m ∈ [1, 2)
    let poly = /* ... coefficients for log2 ... */;

    exponent as f32 + poly
}
```

**Note**: For GUI meter displays, fast approximation may not be necessary since rendering is the bottleneck. Use for real-time envelope followers and dynamics processing.

#### References

- [Fast Vectorizable Math Approximations (INRIA)](http://gallium.inria.fr/blog/fast-vectorizable-math-approx/)
- [DSP Trick: Quick and Dirty Logarithms](https://dspguru.com/dsp/tricks/quick-and-dirty-logarithms/)

---

### 3.4 Sine/Cosine

**Use Cases**: Oscillators (LFOs, audio-rate), wavetable interpolation, modulation

**Target Accuracy**: <-100dB THD (20-bit precision) for high-quality oscillators

**Decision**: FABE13-HX Ψ-Hyperbasis algorithm or Julien Pommier SSE/AVX implementation

#### Recommended Algorithm: FABE13-HX

Uses innovative Ψ-Hyperbasis rational transformation:

```
Ψ(x) = x / (1 + (3/8)x²)
```

Both sin(x) and cos(x) leverage this unified base for enhanced vectorization.

**Accuracy**: Maximum error ≤ 2e-11 compared to libm (~24-bit precision)

**Performance**:
- Up to 8.4x faster than standard math libraries
- 614.85 M ops/sec vs. libm's 138.34 M ops/sec (4.44x)
- Scales to 5.82x for billion-element arrays

**SIMD Support**: AVX2+FMA, AVX-512F (x86-64), NEON (ARM AArch64)

**Features**:
- Branchless quadrant correction
- Prefetch-optimized memory access
- Correct handling of ±0, ±∞, NaN

#### Alternative: Julien Pommier SSE Implementation

```rust
// Widely used in audio software, proven accuracy
#[target_feature(enable = "sse2")]
unsafe fn sincos_sse(x: __m128) -> (__m128, __m128) {
    // Polynomial approximation developed for audio
    // Max absolute error on sines: 2^-24 for x ∈ [-8192, 8192]
    // ... implementation details
}
```

**Accuracy**: 2^-24 (~24-bit precision, exceeds 20-bit target)

**Performance**: Industry standard, widely adopted

**Range**: Valid for [-8192, 8192] - sufficient for audio oscillators

#### Why Not Lookup Tables?

Per Jatin Chowdhury's research, lookup tables are problematic:
- Slower than polynomial approximations (2-4x)
- Poor cache behavior for scattered access patterns
- Interpolation adds overhead
- No vectorization benefit

**Verdict**: Polynomial approximations dominate for modern audio DSP.

#### THD Requirements for Audio

For high-quality audio oscillators:
- **16-bit quality**: ~4 significant digits, THD <0.01% (~-80dB)
- **20-bit quality**: THD <0.001% (~-100dB)
- **24-bit quality**: THD <0.0001% (~-120dB)

Both FABE13-HX and Pommier implementations exceed these requirements.

#### SIMD Implementation Strategy

```rust
// Use FABE13-HX or compile Pommier's code
// Both provide sin/cos simultaneously (sincos)
#[inline(always)]
pub fn fast_sincos(x: f32) -> (f32, f32) {
    #[cfg(target_feature = "avx2")]
    unsafe { sincos_avx2(x) }

    #[cfg(target_feature = "sse2")]
    unsafe { sincos_sse2(x) }

    #[cfg(not(any(target_feature = "avx2", target_feature = "sse2")))]
    (x.sin(), x.cos()) // Fallback to libm
}
```

#### References

- [FABE13-HX GitHub Repository](https://github.com/farukalpay/FABE/)
- [Julien Pommier SSE Math Functions](http://gruntthepeon.free.fr/ssemath/)
- [Fast SIMD Sine and Cosine (DSPRelated)](https://www.dsprelated.com/showcode/59.php)
- [Lookup Tables Are Bad Approximators (Jatin Chowdhury)](https://jatinchowdhury18.medium.com/lookup-tables-are-bad-approximators-6e4a712b912e)

---

### 3.5 Fast Reciprocal (1/x)

**Use Cases**: Division avoidance, normalization, filter coefficients

**Target Accuracy**: <0.01% error for audio applications

**Decision**: Hardware RCP instruction + Newton-Raphson refinement

#### Recommended Algorithm

```rust
#[target_feature(enable = "sse2")]
unsafe fn fast_reciprocal(x: f32) -> f32 {
    // Hardware reciprocal approximation (~12-bit precision)
    let x_vec = _mm_set_ss(x);
    let rcp_approx = _mm_rcp_ss(x_vec);

    // Newton-Raphson refinement: r' = r * (2 - x * r)
    // Brings precision from 12-bit to ~23-bit (full f32)
    let two = _mm_set_ss(2.0);
    let x_r = _mm_mul_ss(x_vec, rcp_approx);
    let two_minus = _mm_sub_ss(two, x_r);
    let refined = _mm_mul_ss(rcp_approx, two_minus);

    _mm_cvtss_f32(refined)
}
```

**Error Without Refinement**: |Relative Error| ≤ 1.5 * 2^-12 (~0.037%)

**Error With One Newton-Raphson Step**: ~23-bit precision (<0.00001%)

**Performance**:
- RCP: 1-2 cycles throughput
- Division: 20+ cycles
- With refinement: ~5 cycles (still 4x faster than division)

#### Vectorized Version (AVX2)

```rust
#[target_feature(enable = "avx2")]
unsafe fn fast_reciprocal_avx2(x: __m256) -> __m256 {
    let rcp_approx = _mm256_rcp_ps(x);

    // Newton-Raphson: r' = r * (2 - x * r)
    let two = _mm256_set1_ps(2.0);
    let x_r = _mm256_mul_ps(x, rcp_approx);
    let two_minus = _mm256_sub_ps(two, x_r);
    _mm256_mul_ps(rcp_approx, two_minus)
}
```

**Processes**: 8 reciprocals in parallel (AVX2), 16 with AVX-512

#### When to Use

- **Good for**: Tight loops with many divisions, normalization, filter coefficient computation
- **Not needed for**: One-off divisions, already optimized with compiler `-ffast-math`
- **Alternative**: Enable compiler reciprocal optimization with `-mrecip` flag

#### Compiler Flag Approach

```bash
RUSTFLAGS="-C target-feature=+avx2 -C target-cpu=native" cargo build --release
```

With `-ffast-math` equivalent flags, compilers may automatically use RCP + refinement.

#### References

- [Fast 1/x Division (Stack Overflow)](https://stackoverflow.com/questions/9939322/fast-1-x-division-reciprocal)
- [Fast Vectorized rsqrt and Reciprocal with SSE/AVX](https://stackoverflow.com/questions/31555260/fast-vectorized-rsqrt-and-reciprocal-with-sse-avx-depending-on-precision)
- [RCPSS Instruction Reference](https://www.felixcloutier.com/x86/rcpss)

---

## 4. Denormal Number Protection

### Overview

Denormal (subnormal) floating-point numbers occur when values approach zero. Instead of rounding to zero, IEEE 754 allows the mantissa to become denormalized (0.000...1), providing extended precision near zero but causing 100x+ performance degradation due to microcode exception handling.

### Decision: Multi-Layered Approach

1. **Hardware FTZ/DAZ flags** (primary)
2. **DC offset injection** (fallback/portable)
3. **Per-block application** (optimal performance)

### Rationale

Audio DSP frequently encounters denormals in:
- Reverb tails decaying to silence
- IIR filter state (lowpass, feedback loops)
- Envelope generators approaching zero
- Effects during transport stop (buffer silence)

**Performance Impact Example**: Modest DSP algorithm with denormals can slow by 100x on older processors, still 10-50x on modern CPUs. CPU meters show normal usage during playback, then spike dramatically seconds after stopping as filter states decay.

---

### 4.1 Hardware FTZ/DAZ Flags (Primary Method)

#### x86-64 (SSE2+)

```rust
#[cfg(target_arch = "x86_64")]
pub fn enable_denormal_protection() {
    use std::arch::x86_64::*;

    unsafe {
        // Get current MXCSR register value
        let mut mxcsr = _mm_getcsr();

        // Enable FTZ (Flush-To-Zero): bit 15
        // Enable DAZ (Denormals-Are-Zero): bit 6
        mxcsr |= 0x8040;

        // Set modified MXCSR
        _mm_setcsr(mxcsr);
    }
}
```

**Flags**:
- **FTZ (Flush-To-Zero)**: Flush denormal *results* to zero
- **DAZ (Denormals-Are-Zero)**: Treat denormal *inputs* as zero

**When to Call**:
- At audio thread initialization (per-thread, not global)
- MXCSR is thread-local - must set for each audio processing thread

**Platform Support**:
- SSE2+ (virtually all x86-64 CPUs from ~2004+)
- Pentium 4 and newer

**Compatibility**: Not IEEE 754 compliant - only use when strict compliance isn't required (acceptable for audio)

#### ARM NEON

**ARMv7**: Automatically forces FTZ for SIMD operations (no configuration needed)

**ARMv8/ARM64**: Supports denormals in both scalar and SIMD. Flush-to-zero flag affects both:

```rust
#[cfg(target_arch = "aarch64")]
pub fn enable_denormal_protection() {
    // FPCR (Floating-Point Control Register) manipulation
    // Specific implementation depends on platform (typically through compiler intrinsics)
    // Many platforms have this enabled by default for audio workloads
}
```

**Apple Silicon Note**: macOS audio APIs may automatically enable FTZ for audio threads.

---

### 4.2 DC Offset Injection (Portable Fallback)

When hardware flags are unavailable or insufficient:

```rust
pub struct DenormalProtection {
    dc_offset: f32,
    flip_sign: bool,
}

impl DenormalProtection {
    pub fn new() -> Self {
        Self {
            dc_offset: 1e-15, // ~-300dB, far below noise floor
            flip_sign: false,
        }
    }

    pub fn process_buffer(&mut self, samples: &mut [f32]) {
        // Apply DC offset once per buffer (not per sample!)
        let offset = if self.flip_sign { -self.dc_offset } else { self.dc_offset };
        self.flip_sign = !self.flip_sign;

        samples[0] += offset;
    }
}
```

**DC Offset Magnitude**: 1e-15 to 1e-25
- **1e-15**: ~-300dB below full scale
- **1e-25**: ~-500dB (used in some implementations)

**Sign Alternation**: Flip polarity each buffer to prevent DC-blocking circuits from removing the offset

**Application Point**: Input stage, once per buffer

---

### 4.3 Per-Block vs. Per-Sample Protection

#### Per-Block (Recommended)

```rust
#[inline]
fn process_audio_block(buffer: &mut [f32]) {
    // Set FTZ/DAZ once at function entry
    enable_denormal_protection();

    // Or inject DC offset once
    buffer[0] += 1e-20;

    // Process entire buffer...
    for sample in buffer.iter_mut() {
        *sample = process_sample(*sample);
    }
}
```

**Advantages**:
- Minimal overhead (one-time cost per buffer)
- Sufficient for most use cases
- Typical buffer sizes: 64-512 samples

**Application**: Initialize at audio callback entry

#### Per-Sample (Unnecessary)

```rust
// DON'T DO THIS
#[inline(never)] // Effectively what happens
fn process_sample(x: f32) -> f32 {
    let protected = x + 1e-20; // Overhead on every sample!
    // ... processing
}
```

**Why Avoid**:
- Adds 1 addition per sample (unnecessary overhead)
- FTZ/DAZ flags remain active for entire thread
- DC injection once per buffer is sufficient

**Exception**: Only needed if processing non-contiguous samples from different sources with varying denormal likelihood

---

### 4.4 Impact on Audio Quality (THD+N)

**Theoretical Impact**: None measurable

**1e-15 DC offset**: -300dB below full scale
**1e-25 DC offset**: -500dB below full scale

**Audio System Noise Floors**:
- 16-bit: ~-96dB
- 24-bit: ~-144dB
- 32-bit float: ~-150dB (practical)

**Verdict**: DC offsets at 1e-15 or smaller are **completely inaudible** - hundreds of dB below the noise floor of any audio system.

**FTZ/DAZ Impact**: No audible effect. Denormals represent values far below the perceptual threshold. Flushing them to zero is sonically transparent.

**THD+N Measurements**: No measurable increase in Total Harmonic Distortion + Noise from denormal protection methods.

---

### 4.5 Implementation Strategy for rigel-dsp

```rust
// src/lib.rs
#![no_std]

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

pub struct SynthEngine {
    // ... fields
}

impl SynthEngine {
    pub fn process_block(&mut self, output: &mut [f32]) {
        // Enable denormal protection once per block
        #[cfg(target_arch = "x86_64")]
        unsafe {
            _mm_setcsr(_mm_getcsr() | 0x8040);
        }

        // For platforms without hardware support, DC injection fallback
        #[cfg(not(target_arch = "x86_64"))]
        if !output.is_empty() {
            output[0] += 1e-20;
        }

        // Process samples...
        for sample in output.iter_mut() {
            *sample = self.process_sample();
        }
    }
}
```

**Key Points**:
1. Set FTZ/DAZ once per audio callback (per-block)
2. Thread-local setting - initialize on audio thread startup
3. DC injection as fallback for platforms without hardware flags
4. Use #[cfg] for platform-specific implementations

---

### 4.6 Testing Denormal Scenarios

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_denormal_protection() {
        enable_denormal_protection();

        // Create denormal value
        let denormal = f32::from_bits(0x00000001); // Smallest positive denormal

        let mut buffer = vec![denormal; 1024];
        let start = std::time::Instant::now();

        // Process buffer - should not slow down
        for sample in buffer.iter_mut() {
            *sample = process_sample(*sample);
        }

        let duration = start.elapsed();

        // Should complete in microseconds, not milliseconds
        assert!(duration.as_micros() < 100);
    }
}
```

---

### References

- [EarLevel Engineering: Floating Point Denormals](https://www.earlevel.com/main/2019/04/19/floating-point-denormals/)
- [EarLevel Engineering: A Note About De-normalization](https://www.earlevel.com/main/2012/12/03/a-note-about-de-normalization/)
- [Intel: Reducing the Impact of Denormal Exceptions](https://www.cita.utoronto.ca/~merz/intel_f10b/main_for/mergedProjects/fpops_for/common/fpops_reduce_denorm.htm)
- [Stack Overflow: Flushing Denormalized Numbers to Zero](https://stackoverflow.com/questions/11671430/flushing-denormalised-numbers-to-zero)
- [JUCE Forum: State of the Art Denormal Prevention](https://forum.juce.com/t/state-of-the-art-denormal-prevention/16802)

---

## 5. Block Processing Patterns

### Overview

Block processing is fundamental to efficient SIMD audio DSP. Understanding optimal block sizes, alignment requirements, and memory access patterns is critical for achieving target performance (<0.1% CPU per voice).

---

### 5.1 Optimal Block Sizes

#### Recommended: 64 or 128 samples

**Rationale**:

1. **Cache Efficiency**:
   - L1 cache line: 64 bytes (16x f32 samples)
   - Processing 64-128 samples fits comfortably in L1 cache
   - Minimizes cache misses

2. **SIMD Width Alignment**:
   - AVX2: 8x f32 per instruction
   - 64 samples = 8 SIMD operations (perfect fit)
   - 128 samples = 16 SIMD operations

3. **Latency vs. Throughput**:
   - Smaller blocks: Lower latency (better for real-time)
   - Larger blocks: Better throughput (amortized overhead)
   - 64-128 samples: Sweet spot at 44.1kHz (~1.4-2.9ms latency)

4. **Diminishing Returns**:
   - Per musicdsp.org, most efficiency gains realized by 64 samples for IIR filters
   - Larger blocks (256+) show minimal additional benefit

**Block Size Requirements**:
- Must be even number (SIMD operations on pairs)
- Minimum: 8 samples (for pipeline behavior)
- Typical DAW buffers: 64, 128, 256, 512, 1024 samples

#### Performance Impact of Odd/Misaligned Sizes

Example from Medium article on buffer sizes:
- **512 samples (aligned)**: AVX-256 convolution reverb achieves 3.2 samples/cycle
- **189 samples (odd)**: Forces scalar fallback, drops to 1.1 samples/cycle (6-8x slower!)

**Lesson**: Vectorization thrives on predictability. Odd buffer sizes destroy SIMD efficiency.

---

### 5.2 Memory Alignment Requirements

#### Alignment by SIMD Width

| SIMD Type | Register Width | Required Alignment | Typical Use |
|-----------|----------------|-------------------|-------------|
| SSE/SSE2  | 128-bit        | 16 bytes          | x86-64 baseline |
| AVX/AVX2  | 256-bit        | 32 bytes          | Modern x86-64 |
| AVX-512   | 512-bit        | 64 bytes          | Server/HEDT |
| NEON      | 128-bit        | 16 bytes          | ARM/Apple Silicon |

#### Rust Alignment Syntax

```rust
// Align structure for AVX2
#[repr(align(32))]
pub struct AlignedBuffer {
    data: [f32; 64],
}

// Align individual array
#[repr(C, align(32))]
pub struct AudioBuffer {
    samples: [f32; 128],
}
```

#### Dynamic Allocation with Alignment

```rust
use std::alloc::{alloc, dealloc, Layout};

pub struct AlignedVec {
    ptr: *mut f32,
    len: usize,
    capacity: usize,
}

impl AlignedVec {
    pub fn new(capacity: usize, alignment: usize) -> Self {
        let layout = Layout::from_size_align(
            capacity * std::mem::size_of::<f32>(),
            alignment,
        ).unwrap();

        let ptr = unsafe { alloc(layout) as *mut f32 };

        Self {
            ptr,
            len: 0,
            capacity,
        }
    }
}

// For rigel-dsp (no_std), use fixed-size aligned arrays
```

#### Alignment Performance Impact

**Aligned Access (32-byte for AVX2)**:
- Single instruction load/store
- ~3-4 cycle latency
- Full throughput

**Misaligned Access**:
- May cross cache line boundaries
- Split loads/stores: 2x memory operations
- 2x-10x slowdown depending on microarchitecture
- Modern CPUs (Haswell+) have improved handling, but aligned is still faster

**Cache Line Splits**:
- Cache lines: 64 bytes
- Unaligned data spanning cache line: Two cache line fetches required
- Critical for streaming SIMD operations

---

### 5.3 Handling Remainder Samples

When buffer size is not a multiple of SIMD width:

#### Pattern 1: Process Chunks + Scalar Remainder

```rust
pub fn process_buffer_avx2(samples: &mut [f32]) {
    // Process 8-sample chunks with AVX2
    let chunks = samples.len() / 8;
    let remainder_start = chunks * 8;

    for i in 0..chunks {
        let offset = i * 8;
        unsafe {
            let input = _mm256_loadu_ps(samples.as_ptr().add(offset));
            let output = process_simd_avx2(input);
            _mm256_storeu_ps(samples.as_mut_ptr().add(offset), output);
        }
    }

    // Scalar fallback for remainder
    for i in remainder_start..samples.len() {
        samples[i] = process_scalar(samples[i]);
    }
}
```

**Characteristics**:
- Simple, explicit
- Minimal overhead for small remainders
- Clear code structure

#### Pattern 2: Overlapping SIMD (Advanced)

```rust
pub fn process_buffer_overlap(samples: &mut [f32]) {
    if samples.len() >= 8 {
        let chunks = (samples.len() - 8) / 8 + 1;

        for i in 0..chunks {
            let offset = i * 8;
            // Process in aligned chunks
        }

        // Process last 8 samples even if overlapping
        let last_offset = samples.len() - 8;
        unsafe {
            let input = _mm256_loadu_ps(samples.as_ptr().add(last_offset));
            let output = process_simd_avx2(input);
            _mm256_storeu_ps(samples.as_mut_ptr().add(last_offset), output);
        }
    } else {
        // Full scalar fallback for buffers < 8 samples
        for sample in samples.iter_mut() {
            *sample = process_scalar(*sample);
        }
    }
}
```

**Characteristics**:
- Avoids scalar path entirely for ≥8 samples
- Slight redundancy (overlapping processing)
- Better for algorithms where redundancy is acceptable

#### Pattern 3: Rust Slice Chunking (Safest)

```rust
pub fn process_buffer_chunks(samples: &mut [f32]) {
    // Safe Rust array chunking
    let (chunks, remainder) = samples.as_chunks_mut::<8>();

    for chunk in chunks {
        // SIMD path - chunk is guaranteed [f32; 8]
        *chunk = process_simd_array(*chunk);
    }

    for sample in remainder {
        // Scalar fallback
        *sample = process_scalar(*sample);
    }
}
```

**Characteristics**:
- Leverages Rust's type system
- Compile-time guarantees on chunk size
- Cleanest, safest approach (recommended for rigel-dsp)

---

### 5.4 Cache-Friendly Access Patterns

#### Sequential Access (Optimal)

```rust
// GOOD: Sequential, predictable access
for i in 0..samples.len() {
    output[i] = process(input[i]);
}
```

**Why**: CPU prefetcher loads next cache lines automatically, minimal cache misses.

#### Strided Access (Acceptable)

```rust
// ACCEPTABLE: Fixed stride (e.g., stereo interleaved)
for i in (0..samples.len()).step_by(2) {
    output[i] = process_left(input[i]);
    output[i + 1] = process_right(input[i + 1]);
}
```

**Why**: Prefetcher can detect regular strides, some benefit remains.

#### Random Access (Avoid)

```rust
// BAD: Random/scattered access (e.g., table lookups)
for i in 0..samples.len() {
    let index = lookup_table[i]; // Unpredictable!
    output[i] = data[index];
}
```

**Why**: Destroys prefetcher efficiency, high cache miss rate. This is why lookup tables underperform polynomial approximations.

#### SIMD Gather/Scatter (Use Sparingly)

```rust
// AVX2 gather (slow but sometimes necessary)
#[target_feature(enable = "avx2")]
unsafe fn gather_example(base: *const f32, indices: __m256i) -> __m256 {
    _mm256_i32gather_ps(base, indices, 4) // 4 = scale factor
}
```

**Why Slow**:
- Requires multiple cache line fetches
- Doesn't benefit from sequential prefetching
- ~10-20x slower than sequential loads
- Only use when algorithm absolutely requires it

#### Deinterleaving for SIMD

```rust
// BETTER: Deinterleave stereo to separate L/R buffers for SIMD
pub fn deinterleave_stereo(interleaved: &[f32], left: &mut [f32], right: &mut [f32]) {
    for (i, chunk) in interleaved.chunks_exact(2).enumerate() {
        left[i] = chunk[0];
        right[i] = chunk[1];
    }

    // Now process left and right with full SIMD width
    process_buffer_simd(left);
    process_buffer_simd(right);
}
```

**Trade-off**: Extra copy overhead but enables full SIMD throughput on each channel.

---

### 5.5 Implementation Strategy for rigel-dsp

```rust
// rigel-dsp/src/lib.rs
#![no_std]

/// Process audio in SIMD-friendly blocks
pub struct BlockProcessor {
    // Internal state...
}

impl BlockProcessor {
    pub fn process(&mut self, output: &mut [f32]) {
        // Verify block size meets minimum requirements
        assert!(output.len() >= 8, "Minimum block size is 8 samples");
        assert!(output.len() % 2 == 0, "Block size must be even");

        #[cfg(target_feature = "avx2")]
        self.process_avx2(output);

        #[cfg(all(target_feature = "sse2", not(target_feature = "avx2")))]
        self.process_sse2(output);

        #[cfg(not(any(target_feature = "avx2", target_feature = "sse2")))]
        self.process_scalar(output);
    }

    #[cfg(target_feature = "avx2")]
    fn process_avx2(&mut self, output: &mut [f32]) {
        let (chunks, remainder) = output.as_chunks_mut::<8>();

        for chunk in chunks {
            // SIMD processing with alignment guaranteed
            *chunk = self.process_simd_chunk(*chunk);
        }

        for sample in remainder {
            *sample = self.process_sample();
        }
    }
}
```

**Alignment in rigel-dsp**:
- Use `#[repr(align(32))]` for internal buffers
- Document alignment requirements for users
- Provide helpers for aligned allocation if needed

**Block Size Recommendations**:
- Document: "Optimal performance with buffer sizes of 64, 128, or 256 samples"
- Support arbitrary sizes but warn about performance implications
- Minimum: 8 samples
- Requirement: Even number of samples

---

### References

- [SIMD Optimization Techniques for Embedded DSP](https://runtimerec.com/simd-optimization-techniques-for-embedded-dsp-boosting-performance-in-resource-constrained-systems/)
- [Stop Watching Your CPU Meter Dance — Lock the Buffer Size](https://medium.com/@12264447666.williamashley/stop-watching-your-cpu-meter-dance-lock-the-buffer-size-and-let-the-dsp-breathe-1d1426fdd017)
- [JUCE SIMDRegister Optimization Tutorial](https://docs.juce.com/master/tutorial_simd_register_optimisation.html)
- [unevens/audio-dsp: SIMD Audio DSP Collection](https://github.com/unevens/audio-dsp)

---

## Summary and Recommendations

### For rigel-dsp SIMD Math Library

1. **Property Testing**: Use `proptest` for comprehensive SIMD math validation
2. **SIMD Abstraction**: Start with `pulp` crate, plan migration to `std::simd` when stable
3. **Math Approximations**:
   - tanh: Fifth-order polynomial (2e-4 error)
   - exp: Chebyshev with range reduction (3e-7 error with FMA)
   - log: Bit manipulation + Remez polynomial (6e-5 error)
   - sin/cos: FABE13-HX or Pommier (2e-11 to 2^-24 error)
   - reciprocal: RCP + Newton-Raphson (23-bit precision)
4. **Denormal Protection**: FTZ/DAZ per block + DC injection fallback
5. **Block Processing**: Target 64-128 samples, 32-byte alignment, handle remainders with `as_chunks_mut`

### Performance Targets

- Single voice CPU: ~0.1% at 44.1kHz
- Zero allocations (no_std compatible)
- Bit-identical results within error bounds across backends
- <0.1% error for approximations (except where tighter bounds specified)

### Next Steps

1. Set up property-based test infrastructure with proptest
2. Implement core SIMD trait abstractions using pulp
3. Port proven approximation algorithms (tanh, exp, sin/cos)
4. Benchmark against targets using existing criterion infrastructure
5. Validate accuracy with property tests across all backends

---

*Document Version: 1.0*
*Last Updated: 2025-11-17*
*Author: Research compilation based on industry sources and academic literature*
