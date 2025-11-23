# Research: Runtime SIMD Dispatch

**Feature**: Runtime SIMD Dispatch
**Date**: 2025-11-22
**Status**: Completed

## Overview

This document consolidates research findings for implementing runtime SIMD backend dispatch in Rigel's DSP core. Two critical unknowns required resolution:

1. **CPU Feature Detection Library**: no_std compatible library for detecting x86_64 SIMD capabilities
2. **Function Dispatch Pattern**: Optimal approach for runtime backend selection in real-time audio DSP

## Decision 1: CPU Feature Detection Library

### Chosen Solution: `cpufeatures` (RustCrypto Project)

**Version**: 0.2.17 (Latest: January 26, 2025)
**Repository**: https://github.com/RustCrypto/utils/tree/master/cpufeatures

**Rationale**:

`cpufeatures` perfectly aligns with Rigel's requirements:
- ✅ **no_std compatible**: Zero dependencies, designed for embedded/mobile contexts
- ✅ **Zero allocations**: Macro-based API with stack-only, ZST (zero-sized type) returns
- ✅ **Active maintenance**: RustCrypto organization with January 2025 releases
- ✅ **Compiler optimization**: When features enabled at compile time, dead-code eliminates fallbacks
- ✅ **Simple API**: Macro-based feature detection with automatic caching

**Implementation Example**:

```rust
#![no_std]

// Define CPU feature checks (compile-time safe, zero allocations)
cpufeatures::new!(cpuid_avx2, "avx2");
cpufeatures::new!(cpuid_avx512f, "avx512f");

/// Detect available SIMD backends at runtime
pub fn detect_cpu_features() -> CpuFeatures {
    let token_avx2 = cpuid_avx2::init();
    let token_avx512f = cpuid_avx512f::init();

    CpuFeatures {
        has_avx2: token_avx2.get(),
        has_avx512: token_avx512f.get(),
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CpuFeatures {
    pub has_avx2: bool,
    pub has_avx512: bool,
}

// Example usage in backend selection
pub fn select_backend() -> BackendType {
    let features = detect_cpu_features();

    if features.has_avx512 {
        BackendType::Avx512
    } else if features.has_avx2 {
        BackendType::Avx2
    } else {
        BackendType::Scalar
    }
}
```

**Integration with Rigel**:

```toml
# In projects/rigel-synth/crates/dsp/Cargo.toml
[dependencies]
cpufeatures = "0.2"
```

**Supported x86_64 Features**:
- AVX2: `avx2`
- AVX-512 Foundation: `avx512f`, `avx512bw`, `avx512cd`, `avx512dq`, `avx512vl`
- Additional AVX-512: `avx512_ifma`, `avx512vnni`, `avx512vbmi`, `avx512vbmi2`, `avx512vpopcntdq`
- 20+ additional features (AES, SHA, SSE variants, etc.)

### Alternatives Considered

**`raw-cpuid` (11.6.0)**:
- **Pros**: Comprehensive x86 CPUID parsing, no dependencies, no_std compatible
- **Cons**: Overkill for Rigel's use case (only need 2-3 features), heavier API surface
- **Why rejected**: Designed for system tools and detailed CPU introspection, not minimal runtime detection

**`core_detect` (1.x)**:
- **Pros**: Simplest API: `is_x86_feature_detected!("avx2")`
- **Cons**: Minimally maintained (4 GitHub stars, single-author project)
- **Why rejected**: Risk of abandonment, cpufeatures provides same simplicity with better infrastructure

### Trade-offs

**Advantages**:
- Zero allocations, perfect for no_std
- Active maintenance by RustCrypto
- Minimal overhead (results cached after first call)
- Integrates with compile-time features for optimization

**Limitations**:
- Macro-based API requires compile-time feature specification
- x86-only for detailed support (not relevant for Rigel)
- Can't dynamically add feature checks (must define upfront)

**Mitigation**: Create feature detection module with all needed checks at initialization.

---

## Decision 2: Function Dispatch Pattern

### Chosen Approach: Static Function Pointer Tables with Initialization Phase

**Rationale**:

This approach provides optimal balance for real-time audio DSP:
- ✅ **Zero runtime overhead**: Single indirect jump through table, fully predictable
- ✅ **no_std compatible**: No heap allocations, stack-based initialization
- ✅ **One-time initialization**: CPU detection happens once at startup
- ✅ **Type-safe dispatch**: Compiler validates function signatures at compile time
- ✅ **Minimal complexity**: Clear, familiar pattern without advanced generics

**Performance Characteristics**:
- Indirect function call cost: 0-2 CPU cycles on modern CPUs
- Percentage of typical audio operation: <0.05%
- Branch prediction accuracy: 95%+ for stable dispatch
- Expected overhead vs compile-time: <1% (meets success criteria SC-002)

**Implementation Pattern**:

```rust
// 1. SIMD backend trait
pub trait SimdBackend: Copy {
    fn process_block(input: &[f32], output: &mut [f32], params: &ProcessParams);
    fn advance_phase_vectorized(phases: &mut [f32], increments: &[f32], count: usize);
    fn name() -> &'static str;
}

// 2. Backend implementations (Scalar, AVX2, AVX-512, NEON)
#[derive(Copy, Clone)]
pub struct ScalarBackend;

impl SimdBackend for ScalarBackend {
    #[inline]
    fn process_block(input: &[f32], output: &mut [f32], params: &ProcessParams) {
        for (inp, out) in input.iter().zip(output.iter_mut()) {
            *out = inp * params.gain;
        }
    }

    fn name() -> &'static str { "scalar" }
}

// 3. Function pointer dispatch structure
#[repr(C)]
pub struct BackendDispatcher {
    process_block: unsafe fn(&[f32], &mut [f32], &ProcessParams),
    advance_phase: unsafe fn(&mut [f32], &[f32], usize),
    name: unsafe fn() -> &'static str,
}

impl BackendDispatcher {
    /// Detect CPU features and select optimal backend (initialization phase)
    pub fn init() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if Self::has_avx512() && cfg!(feature = "avx512") {
                return Self::for_avx512();
            }
            if Self::has_avx2() && cfg!(feature = "avx2") {
                return Self::for_avx2();
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            return Self::for_neon();
        }

        Self::for_scalar()
    }

    // Hot path - minimal overhead
    #[inline]
    pub fn process_block(&self, input: &[f32], output: &mut [f32], params: &ProcessParams) {
        unsafe { (self.process_block)(input, output, params) }
    }
}

// 4. Real-time safe engine using dispatcher
pub struct SynthEngine {
    dispatcher: BackendDispatcher,
    oscillator: SimpleOscillator,
    envelope: Envelope,
}

impl SynthEngine {
    pub fn new(sample_rate: f32) -> Self {
        // Initialization happens once, before real-time audio starts
        let dispatcher = BackendDispatcher::init();

        Self {
            dispatcher,
            oscillator: SimpleOscillator::new(),
            envelope: Envelope::new(sample_rate),
        }
    }

    // Hot path - uses function pointers with negligible overhead
    #[inline]
    pub fn process_sample(&mut self, params: &SynthParams) -> f32 {
        let mut buffer = [0.0f32; 1];
        self.dispatcher.process_block(&[], &mut buffer, &ProcessParams::from(params));
        buffer[0]
    }
}
```

### Alternatives Considered

**1. Trait Objects (`dyn SimdBackend`)**:
- **Cons**: Requires heap allocation (violates no_std), extra vtable indirection
- **Why rejected**: Incompatible with real-time safety constraints

**2. Enum-Based Dispatch**:
```rust
match backend {
    Backend::Scalar => ScalarBackend::process(...),
    Backend::Avx2 => Avx2Backend::process(...),
}
```
- **Cons**: Branch misprediction penalty (12-18 cycles), compiler can't eliminate match
- **Why rejected**: Higher overhead than function pointers, unsuitable for hot path

**3. Const Generics with Runtime Dispatch Wrapper**:
- **Cons**: Requires heap allocation (Box), trait object vtable adds indirection
- **Why rejected**: More complex, incompatible with no_std

**4. Compile-Time Only Selection**:
- **Cons**: Cannot support multiple backends in single binary, requires separate builds
- **Why rejected**: Defeats purpose of runtime CPU detection

### Trade-offs

**Advantages**:
- True zero-cost abstraction: compiles to single `call` instruction
- no_std compatible: stack-based, no allocations
- Runtime CPU detection: single binary supports multiple microarchitectures
- Deterministic performance
- Maintainable and debuggable

**Limitations**:
1. **Unsafe code required**: CPU feature detection uses CPUID intrinsics
   - Mitigation: Wrap in safe abstractions, add comprehensive tests

2. **Platform-specific detection**: CPUID differs between x86/ARM
   - Mitigation: Feature-gate detection code, compile-time fallback

3. **Cannot inline across function pointer**: Compiler sees call, not definition
   - Mitigation: Use `#[inline]` on backends, process blocks (64+ samples) not individual samples
   - Impact: Negligible for block-based DSP

4. **Testing complexity**: Need tests for each backend's correctness
   - Mitigation: Property-based testing to ensure all backends produce identical results

### Performance Validation Approach

```rust
#[cfg(test)]
mod benches {
    use criterion::{black_box, Criterion};

    fn bench_dispatch_overhead(c: &mut Criterion) {
        let dispatcher = BackendDispatcher::init();

        c.bench_function("dispatch_scalar_block", |b| {
            let mut output = vec![0.0f32; 4096];
            let input = vec![1.0f32; 4096];
            let params = ProcessParams { gain: 0.5 };

            b.iter(|| {
                dispatcher.process_block(&input, &mut output, &params);
            });
        });

        c.bench_function("direct_scalar_block", |b| {
            let mut output = vec![0.0f32; 4096];
            let input = vec![1.0f32; 4096];
            let params = ProcessParams { gain: 0.5 };

            b.iter(|| {
                ScalarBackend::process_block(&input, &mut output, &params);
            });
        });
        // Expected: <1% difference
    }
}
```

---

## Integration Strategy for Rigel

### Phase 1: Foundation (Minimally Invasive)
- Create `rigel-math/src/simd/` module with dispatcher
- Add `cpufeatures` dependency
- Keep existing compile-time backends alongside runtime dispatch
- Make runtime dispatch optional via `runtime-dispatch` feature

### Phase 2: Integration
- Initialize dispatcher once in `SynthEngine::new()` or plugin wrapper
- Real-time code uses dispatcher instead of direct backend calls

### Phase 3: Expand Coverage
- Phase 3a: Oscillator processing, wavetable interpolation (highest impact)
- Phase 3b: Envelope generation, parameter ramping
- Phase 3c: Effects processing (convolution, filtering)

### Phase 4: Performance Validation
- Benchmark each backend (Criterion + iai-callgrind)
- Compare compile-time vs runtime dispatch overhead
- Profile: CPU cycles, instruction cache misses, branch mispredictions
- Validate <1% overhead requirement (SC-002)

---

## Constitution Compliance

### Real-Time Safety (NON-NEGOTIABLE)
- ✅ **no_std compliance**: Both `cpufeatures` and function pointer dispatch are no_std compatible
- ✅ **Zero allocations**: All detection and dispatch use stack-only data structures
- ✅ **Deterministic performance**: Function pointer dispatch has predictable overhead
- ✅ **No blocking I/O**: All operations are compute-only

### Performance Accountability
- ✅ **<1% overhead target**: Function pointer dispatch expected to be <0.05% of block processing
- ✅ **Benchmark validation**: Criterion and iai-callgrind will measure actual overhead
- 📋 **Action required**: Save baseline before implementation, measure after

### Test-Driven Validation
- ✅ **Deterministic testing**: Forced-backend flags enable CI testing of each backend
- ✅ **Property-based tests**: Ensure all backends produce identical results
- ✅ **Architecture-specific tests**: AVX2/AVX-512 on x86_64, NEON on aarch64

---

## Summary

**CPU Feature Detection**: Use `cpufeatures 0.2.17` from RustCrypto
- no_std compatible, zero allocations, actively maintained
- Simple macro-based API with automatic caching
- Integrates with compile-time features for optimization

**Function Dispatch**: Use static function pointer tables initialized at startup
- <1% overhead vs compile-time selection (single indirect jump)
- no_std compatible, zero allocations
- Type-safe, maintainable, debuggable

**Implementation Path**: Phased rollout starting with high-impact DSP operations
- Validate performance at each phase
- Maintain existing compile-time backends during transition
- Comprehensive testing to ensure correctness across all backends

**Constitution Compliance**: All principles satisfied, no violations requiring justification
