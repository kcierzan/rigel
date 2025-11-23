# Data Model: Runtime SIMD Dispatch

**Feature**: Runtime SIMD Dispatch
**Date**: 2025-11-23 (Updated to reflect current implementation)

## Overview

This feature implements a **two-crate layered architecture** for SIMD backend abstraction:

1. **rigel-math**: Trait-based SIMD abstraction library (no_std, zero-allocation)
   - Provides `SimdVector` trait and backend implementations
   - Compile-time backend selection via type aliases
   - Multiple backends can coexist when `runtime-dispatch` feature enabled

2. **rigel-dsp**: DSP core that uses rigel-math abstractions
   - On x86_64 with `runtime-dispatch`: Uses `BackendDispatcher` for runtime CPU detection + function pointers
   - On aarch64 or forced backends: Uses compile-time `DefaultSimdVector` (zero overhead)
   - `SimdContext` provides unified API abstraction over both mechanisms

This document defines the key interfaces, types, and their relationships.

---

## Layer 1: rigel-math (SIMD Abstraction Library)

### 1. SimdVector Trait

**Purpose**: Generic SIMD vector operations trait that all backend implementations satisfy.

**Location**: `projects/rigel-synth/crates/math/src/traits.rs`

**Properties**:
- Generic over scalar type `T` (primarily `f32` for audio DSP)
- All methods are `#[inline]` to enable compiler optimization
- Identical functional behavior across all backends (only performance differs)
- no_std compatible
- Zero heap allocations

**Core Operations**:
```rust
pub trait SimdVector: Copy + Send + Sync {
    type Scalar;
    const LANES: usize;

    // Construction
    fn splat(value: Self::Scalar) -> Self;
    fn from_slice(slice: &[Self::Scalar]) -> Self;
    fn to_slice(self, slice: &mut [Self::Scalar]);

    // Arithmetic (via ops module for functional style)
    // - add, sub, mul, div via rigel_math::ops
    // - fma (fused multiply-add)
    // - min, max, abs

    // Math functions (via math module)
    // - sqrt, exp, log, sin, cos, tanh via rigel_math::math
    // - fast approximations with controlled error bounds

    // Comparison (via SimdMask)
    // - gt, lt, ge, le, eq, ne
}
```

**Invariants**:
- All backends MUST produce functionally identical output (within floating-point precision)
- All backends MUST handle edge cases identically (NaN, infinity, out-of-bounds)
- No allocations allowed in any operation
- No blocking operations allowed

**Validation**:
- Property-based tests ensure all backends produce identical results
- Unit tests verify edge case handling
- Benchmarks measure performance differences

---

### 2. Backend Implementations

Each backend implements the `SimdVector` trait with platform-specific SIMD intrinsics.

**ScalarVector&lt;T&gt;**:
- **Always available**: No CPU feature requirements
- **Fallback**: Used when no SIMD support detected
- **Platform**: All (x86_64, aarch64)
- **Implementation**: Standard Rust scalar operations (no intrinsics)
- **Lanes**: 1 (processes one value at a time)
- **Performance**: Baseline (1.0x)
- **Location**: `projects/rigel-synth/crates/math/src/backends/scalar.rs`

**Avx2Vector**:
- **CPU requirement**: x86_64 with AVX2 + FMA support
- **Platform**: Linux x86_64, Windows x86_64
- **Implementation**: `core::arch::x86_64` intrinsics (`__m256` wrapper)
- **Lanes**: 8 (processes 8 f32s per operation)
- **Performance**: ~2-4x faster than scalar for block operations
- **Feature flag**: `avx2` (for compilation), detected at runtime when `runtime-dispatch` enabled
- **Location**: `projects/rigel-synth/crates/math/src/backends/avx2.rs`

**Avx512Vector**:
- **CPU requirement**: x86_64 with AVX-512F/BW/DQ/VL
- **Platform**: Linux x86_64, Windows x86_64
- **Implementation**: `core::arch::x86_64` intrinsics (`__m512` wrapper)
- **Lanes**: 16 (processes 16 f32s per operation)
- **Performance**: ~4-8x faster than scalar for block operations
- **Feature flag**: `avx512` (for compilation), detected at runtime when `runtime-dispatch` enabled
- **Status**: Experimental (local testing only, not CI)
- **Location**: `projects/rigel-synth/crates/math/src/backends/avx512.rs`

**NeonVector**:
- **CPU requirement**: aarch64 with NEON support (always present on Apple Silicon)
- **Platform**: macOS aarch64
- **Implementation**: `core::arch::aarch64` intrinsics (`float32x4_t` wrapper)
- **Lanes**: 4 (processes 4 f32s per operation)
- **Performance**: ~2-4x faster than scalar for block operations
- **Feature flag**: `neon` (compile-time only, no runtime detection needed)
- **Location**: `projects/rigel-synth/crates/math/src/backends/neon.rs`

---

### 3. DefaultSimdVector Type Alias

**Purpose**: Resolves to the appropriate SIMD backend at compile time based on features.

**Location**: `projects/rigel-synth/crates/math/src/lib.rs`

**Resolution Logic**:
```rust
// When runtime-dispatch is enabled, DefaultSimdVector defaults to scalar
// (users should use BackendDispatcher instead of DefaultSimdVector)
#[cfg(feature = "runtime-dispatch")]
pub type DefaultSimdVector = ScalarVector<f32>;

// Compile-time backend selection (when runtime-dispatch is NOT enabled)
#[cfg(all(not(feature = "runtime-dispatch"), feature = "avx2", target_arch = "x86_64"))]
pub type DefaultSimdVector = Avx2Vector;

#[cfg(all(not(feature = "runtime-dispatch"), feature = "avx512", target_arch = "x86_64"))]
pub type DefaultSimdVector = Avx512Vector;

#[cfg(all(not(feature = "runtime-dispatch"), feature = "neon", target_arch = "aarch64"))]
pub type DefaultSimdVector = NeonVector;

// Default to scalar if no SIMD features enabled
#[cfg(all(not(feature = "runtime-dispatch"), not(feature = "avx2"),
          not(feature = "avx512"), not(feature = "neon")))]
pub type DefaultSimdVector = ScalarVector<f32>;
```

**Properties**:
- Zero-cost abstraction when `runtime-dispatch` is disabled
- Allows writing backend-agnostic code using `DefaultSimdVector`
- Compile-time resolution means no runtime overhead

---

### 4. Block Processing API

**Purpose**: Fixed-size aligned audio buffers with SIMD-friendly chunk iteration.

**Location**: `projects/rigel-synth/crates/math/src/block.rs`

**Types**:
```rust
pub struct AudioBlock<T, const N: usize> {
    data: [T; N],
}

pub type Block64 = AudioBlock<f32, 64>;
pub type Block128 = AudioBlock<f32, 128>;
```

**SIMD Chunking API**:
```rust
impl<T, const N: usize> AudioBlock<T, N> {
    // Immutable iteration - returns SIMD vectors directly
    pub fn as_chunks<V: SimdVector<Scalar = T>>(&self) -> SimdChunks<'_, T, V, N> { ... }

    // Mutable iteration - returns SimdChunkMut wrappers with load/store methods
    pub fn as_chunks_mut<V: SimdVector<Scalar = T>>(&mut self) -> SimdChunksMut<'_, T, V, N> { ... }
}

// Immutable iterator returns vectors directly
impl<'a, T, V: SimdVector<Scalar = T>, const N: usize> Iterator for SimdChunks<'a, T, V, N> {
    type Item = V;  // Returns SIMD vector directly
}

// Mutable iterator returns chunk wrappers
pub struct SimdChunkMut<'a, T, V: SimdVector<Scalar = T>> {
    slice: &'a mut [T],
    _phantom: PhantomData<V>,
}

impl<'a, T, V: SimdVector<Scalar = T>> SimdChunkMut<'a, T, V> {
    pub fn load(&self) -> V { V::from_slice(self.slice) }
    pub fn store(&mut self, vec: V) { vec.to_slice(self.slice) }
}
```

**Usage Pattern**:
```rust
use rigel_math::{Block64, DefaultSimdVector, ops};

let input = Block64::new();
let mut output = Block64::new();

// Immutable chunks iterate as vectors, mutable chunks provide load/store
for (in_vec, mut out_chunk) in input.as_chunks::<DefaultSimdVector>()
    .iter()
    .zip(output.as_chunks_mut::<DefaultSimdVector>().iter_mut())
{
    let result = ops::mul(in_vec, DefaultSimdVector::splat(2.0));
    out_chunk.store(result);
}
```

---

### 5. Operations Module (ops)

**Purpose**: Functional-style SIMD operations that work generically with any `SimdVector` implementation.

**Location**: `projects/rigel-synth/crates/math/src/ops.rs`

**Key Functions**:
```rust
pub fn add<V: SimdVector>(a: V, b: V) -> V { ... }
pub fn sub<V: SimdVector>(a: V, b: V) -> V { ... }
pub fn mul<V: SimdVector>(a: V, b: V) -> V { ... }
pub fn div<V: SimdVector>(a: V, b: V) -> V { ... }
pub fn fma<V: SimdVector>(a: V, b: V, c: V) -> V { ... }  // a * b + c
pub fn min<V: SimdVector>(a: V, b: V) -> V { ... }
pub fn max<V: SimdVector>(a: V, b: V) -> V { ... }
pub fn abs<V: SimdVector>(a: V) -> V { ... }
```

**Properties**:
- Backend-agnostic (works with any `SimdVector` implementation)
- Compiles to optimal SIMD instructions for each backend
- Functional style (pure functions, no mutation)
- `#[inline]` for zero-cost abstraction

---

### 6. Math Module (fast kernels)

**Purpose**: Vectorized math functions (transcendentals, approximations).

**Location**: `projects/rigel-synth/crates/math/src/math.rs`

**Key Functions**:
```rust
pub fn sqrt<V: SimdVector>(x: V) -> V { ... }
pub fn exp<V: SimdVector>(x: V) -> V { ... }
pub fn log<V: SimdVector>(x: V) -> V { ... }
pub fn sin<V: SimdVector>(x: V) -> V { ... }
pub fn cos<V: SimdVector>(x: V) -> V { ... }
pub fn tanh<V: SimdVector>(x: V) -> V { ... }
pub fn pow<V: SimdVector>(base: V, exp: V) -> V { ... }
```

**Properties**:
- Fast approximations with controlled error bounds
- Backend-agnostic implementations
- Optimized for real-time audio processing

---

### 7. Table Module (wavetable synthesis)

**Purpose**: Wavetable lookup with linear/cubic interpolation.

**Location**: `projects/rigel-synth/crates/math/src/table.rs`

**Key Types/Functions**:
```rust
pub struct WavetableReader<V: SimdVector> {
    // Wavetable lookup with SIMD interpolation
}

pub fn linear_interp<V: SimdVector>(table: &[f32], positions: V) -> V { ... }
pub fn cubic_interp<V: SimdVector>(table: &[f32], positions: V) -> V { ... }
```

**Properties**:
- SIMD-accelerated table lookups
- Supports linear and cubic interpolation
- Backend-agnostic

---

## Layer 2: rigel-dsp (Runtime Dispatch)

### 8. BackendDispatcher

**Purpose**: Runtime dispatch mechanism that selects optimal backend based on CPU features (x86_64 only).

**Location**: `projects/rigel-synth/crates/math/src/simd/dispatcher.rs`

**Structure**:
```rust
pub struct BackendDispatcher {
    backend_type: BackendType,
    backend_name: &'static str,
}

impl BackendDispatcher {
    /// Initialize dispatcher with CPU feature detection
    pub fn init() -> Self {
        let features = CpuFeatures::detect();
        let backend_type = BackendType::select(features);
        Self {
            backend_type,
            backend_name: backend_type.name(),
        }
    }

    pub fn backend_type(&self) -> BackendType { ... }
    pub fn backend_name(&self) -> &'static str { ... }
}
```

**Properties**:
- Initialized once at plugin/engine startup (before real-time processing)
- Immutable after initialization
- Stack-allocated (no heap allocation)
- Only used on x86_64 with `runtime-dispatch` feature

**Lifecycle**:
1. **Initialization** (one-time, non-real-time):
   - Detect CPU features via `cpufeatures`
   - Select best available backend (scalar → AVX2 → AVX-512)

2. **Real-time use** (hot path):
   - DSP code uses `DefaultSimdVector` directly (resolved at compile time)
   - Dispatcher only provides backend name for debugging

---

### 9. CpuFeatures

**Purpose**: Represents detected CPU SIMD capabilities.

**Location**: `projects/rigel-synth/crates/math/src/simd/dispatcher.rs`

**Structure**:
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuFeatures {
    pub has_avx2: bool,
    pub has_avx512_f: bool,      // AVX-512 Foundation
    pub has_avx512_bw: bool,     // Byte & Word ops
    pub has_avx512_dq: bool,     // Doubleword & Quadword
    pub has_avx512_vl: bool,     // Vector Length extensions
}

impl CpuFeatures {
    /// Detect runtime CPU features (x86_64 only)
    #[cfg(all(target_arch = "x86_64", feature = "runtime-dispatch"))]
    pub fn detect() -> Self {
        // Uses cpufeatures crate (no_std compatible)
        cpufeatures::new!(cpuid_avx2, "avx2");
        cpufeatures::new!(cpuid_avx512f, "avx512f");
        // ... additional AVX-512 extensions

        Self {
            has_avx2: cpuid_avx2::get(),
            has_avx512_f: cpuid_avx512f::get(),
            // ...
        }
    }

    /// No runtime detection needed on aarch64 (NEON always available)
    #[cfg(target_arch = "aarch64")]
    pub fn detect() -> Self {
        Self {
            has_avx2: false,
            has_avx512_f: false,
            has_avx512_bw: false,
            has_avx512_dq: false,
            has_avx512_vl: false,
        }
    }

    pub fn has_avx512_full(&self) -> bool {
        self.has_avx512_f && self.has_avx512_bw
            && self.has_avx512_dq && self.has_avx512_vl
    }
}
```

---

### 10. BackendType (Enum)

**Purpose**: Represents the selected SIMD backend for logging, testing, and forced backend builds.

**Location**: `projects/rigel-synth/crates/math/src/simd/dispatcher.rs`

**Structure**:
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendType {
    Scalar,
    Avx2,
    Avx512,
    Neon,
}

impl BackendType {
    /// Select optimal backend based on CPU features and compilation flags
    #[cfg(target_arch = "x86_64")]
    pub fn select(features: CpuFeatures) -> Self {
        // Check forced backend flags first
        #[cfg(feature = "force-scalar")]
        { return BackendType::Scalar; }

        #[cfg(feature = "force-avx2")]
        { return BackendType::Avx2; }

        #[cfg(feature = "force-avx512")]
        { return BackendType::Avx512; }

        // Runtime detection (priority: AVX-512 → AVX2 → Scalar)
        #[cfg(all(feature = "avx512", not(any(feature = "force-scalar", feature = "force-avx2"))))]
        if features.has_avx512_full() {
            return BackendType::Avx512;
        }

        #[cfg(all(feature = "avx2", not(feature = "force-scalar")))]
        if features.has_avx2 {
            return BackendType::Avx2;
        }

        BackendType::Scalar
    }

    /// On aarch64, always use NEON (or forced backend for testing)
    #[cfg(target_arch = "aarch64")]
    pub fn select(_features: CpuFeatures) -> Self {
        #[cfg(feature = "force-scalar")]
        { return BackendType::Scalar; }

        BackendType::Neon
    }

    pub fn name(&self) -> &'static str {
        match self {
            BackendType::Scalar => "scalar",
            BackendType::Avx2 => "avx2",
            BackendType::Avx512 => "avx512",
            BackendType::Neon => "neon",
        }
    }
}
```

---

### 11. SimdContext (Unified Abstraction)

**Purpose**: Provides a consistent API for DSP code regardless of backend selection mechanism (runtime vs compile-time).

**Location**: `projects/rigel-synth/crates/math/src/simd/context.rs`

**Structure**:
```rust
pub struct SimdContext {
    #[cfg(all(target_arch = "x86_64", feature = "runtime-dispatch"))]
    dispatcher: BackendDispatcher,
}

impl SimdContext {
    /// Initialize SIMD context with optimal backend for this CPU
    ///
    /// On x86_64: Detects CPU features and selects best backend
    /// On aarch64: Zero-cost initialization (compile-time NEON)
    pub fn new() -> Self {
        #[cfg(all(target_arch = "x86_64", feature = "runtime-dispatch"))]
        {
            Self {
                dispatcher: BackendDispatcher::init(),
            }
        }

        #[cfg(not(all(target_arch = "x86_64", feature = "runtime-dispatch")))]
        {
            Self {}
        }
    }

    /// Get backend name for debugging/logging
    pub fn backend_name(&self) -> &'static str {
        #[cfg(all(target_arch = "x86_64", feature = "runtime-dispatch"))]
        {
            self.dispatcher.backend_name()
        }

        #[cfg(not(all(target_arch = "x86_64", feature = "runtime-dispatch")))]
        {
            // At compile time, DefaultSimdVector resolves to the selected backend
            #[cfg(all(feature = "avx2", target_arch = "x86_64"))]
            return "avx2";

            #[cfg(all(feature = "avx512", target_arch = "x86_64"))]
            return "avx512";

            #[cfg(all(feature = "neon", target_arch = "aarch64"))]
            return "neon";

            #[cfg(not(any(...)))]
            return "scalar";
        }
    }

    /// Apply gain to a block of samples (example operation)
    #[inline]
    pub fn apply_gain(&self, input: &Block64, output: &mut Block64, gain: f32) {
        let gain_vec = DefaultSimdVector::splat(gain);

        for (in_vec, mut out_chunk) in input
            .as_chunks::<DefaultSimdVector>()
            .iter()
            .zip(output.as_chunks_mut::<DefaultSimdVector>().iter_mut())
        {
            let result = ops::mul(in_vec, gain_vec);
            out_chunk.store(result);
        }
    }

    /// Process a block with a generic SIMD operation
    #[inline]
    pub fn process_block<F>(&self, input: &Block64, output: &mut Block64, mut op: F)
    where
        F: FnMut(DefaultSimdVector) -> DefaultSimdVector,
    {
        for (in_vec, mut out_chunk) in input
            .as_chunks::<DefaultSimdVector>()
            .iter()
            .zip(output.as_chunks_mut::<DefaultSimdVector>().iter_mut())
        {
            let result = op(in_vec);
            out_chunk.store(result);
        }
    }
}
```

**Properties**:
- Zero-sized type on aarch64 (compile-time backend)
- Contains BackendDispatcher on x86_64 with runtime-dispatch
- Provides unified API for DSP code
- All operations use `DefaultSimdVector` which resolves at compile time

---

## Unified Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                         DSP Code                             │
│  Uses SimdContext for backend-agnostic processing            │
│                                                               │
│  ctx.apply_gain(...)                                         │
│  ctx.process_block(...)                                      │
│  or direct: ops::mul(DefaultSimdVector, ...)                 │
│                           │                                   │
│                           ▼                                   │
├───────────────────────────────────────────────────────────────┤
│                    SimdContext                                │
│  ┌──────────────────┴──────────────────┐                     │
│ x86_64 runtime        │        aarch64/forced backends       │
│ (runtime-dispatch)    │        (compile-time)                │
│         │             │                │                      │
│         ▼             │                ▼                      │
│ BackendDispatcher     │        DefaultSimdVector             │
│  - CPU detection      │         = Avx2Vector                 │
│  - Backend selection  │         = Avx512Vector               │
│  (initialization only)│         = NeonVector                 │
│                       │         = ScalarVector               │
└───────────────────────┴────────────────┬─────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   rigel-math Library                         │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │          SimdVector Trait                            │   │
│  │  + splat, from_slice, to_slice                       │   │
│  │  + Arithmetic via ops module                         │   │
│  │  + Math via math module                              │   │
│  │  + Table lookups via table module                    │   │
│  └────────┬──────────────┬──────────────┬──────────────┘   │
│           │              │              │                   │
│  ┌────────▼──────┐ ┌────▼──────┐ ┌────▼──────┐ ┌─────────┐│
│  │ScalarVector   │ │Avx2Vector │ │Avx512     │ │Neon     ││
│  │(1 lane)       │ │(8 lanes)  │ │Vector     │ │Vector   ││
│  │always avail   │ │x86_64+avx2│ │(16 lanes) │ │(4 lanes)││
│  │               │ │           │ │x86_64     │ │aarch64  ││
│  │               │ │           │ │exp        │ │         ││
│  └───────────────┘ └───────────┘ └───────────┘ └─────────┘│
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         Modules (backend-agnostic)                   │   │
│  │  • ops: add, mul, fma, min, max, abs                 │   │
│  │  • math: sqrt, exp, log, sin, cos, tanh              │   │
│  │  • table: linear_interp, cubic_interp               │   │
│  │  • block: Block64, Block128 with SIMD chunking      │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                 Initialization Flow                         │
│                                                              │
│  1. SimdContext::new()                                      │
│          │                                                   │
│    ┌─────┴────────┐                                         │
│    │              │                                          │
│ x86_64         aarch64                                      │
│ runtime        compile-time                                 │
│    │              │                                          │
│    ▼              ▼                                          │
│  CpuFeatures::   DefaultSimdVector                          │
│  detect()        = NeonVector                               │
│    │             (zero overhead)                            │
│    ▼                                                         │
│  BackendType::                                              │
│  select()                                                    │
│    │                                                         │
│    ▼                                                         │
│  BackendDispatcher::                                        │
│  init()                                                      │
│    │                                                         │
│    ▼                                                         │
│  Return SimdContext                                         │
│  (contains dispatcher for debug/logging)                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Feature Flags

### rigel-math Features

**Backend compilation:**
```toml
[features]
scalar = []         # Scalar backend (always implicitly available)
avx2 = []          # Compile AVX2 backend
avx512 = []        # Compile AVX-512 backend (experimental)
neon = []          # Compile NEON backend
runtime-dispatch = [] # Allow multiple backends to coexist
```

### rigel-dsp Features

**Runtime dispatch and forced backends:**
```toml
[features]
default = []
runtime-dispatch = [
    "cpufeatures",
    "rigel-math/runtime-dispatch",
    "rigel-math/scalar",
    "rigel-math/avx2",
    "rigel-math/avx512"
]

# SIMD backend compilation flags
avx2 = ["rigel-math/avx2"]
avx512 = ["rigel-math/avx512"]
neon = ["rigel-math/neon"]

# Forced backend selection (deterministic testing)
force-scalar = ["rigel-math/scalar"]
force-avx2 = ["avx2"]
force-avx512 = ["avx512"]
force-neon = ["neon"]
```

**Build Examples**:
```bash
# Default: runtime dispatch with all backends (x86_64)
cargo build --release

# Force scalar (CI testing, all platforms)
cargo build --release --features force-scalar

# Force AVX2 (CI testing, x86_64)
cargo build --release --features force-avx2

# Force AVX-512 (local experimental testing, x86_64)
cargo build --release --features force-avx512

# macOS: compile-time NEON (no runtime dispatch)
cargo build --release --features neon
```

---

## How It Works: Usage Examples

### Example 1: Using SimdContext (Recommended)

```rust
use rigel_math::{Block64, DefaultSimdVector, ops};
use rigel_math::simd::{SimdContext, ProcessParams};

// Initialize once during engine startup
let ctx = SimdContext::new();
println!("Using SIMD backend: {}", ctx.backend_name());

let mut input = Block64::new();
let mut output = Block64::new();

// Example 1: Built-in operations
ctx.apply_gain(input.as_slice(), output.as_slice_mut(), 0.5);

// Example 2: Using process_block with ProcessParams
let params = ProcessParams {
    gain: 2.0,
    frequency: 440.0,
    sample_rate: 44100.0,
};
ctx.process_block(input.as_slice(), output.as_slice_mut(), &params);
```

### Example 2: Direct rigel-math Usage

```rust
use rigel_math::{Block64, DefaultSimdVector, ops, math};

let mut input = Block64::new();
let mut output = Block64::new();

// Process block with SIMD operations
for (in_vec, mut out_chunk) in input.as_chunks::<DefaultSimdVector>()
    .iter()
    .zip(output.as_chunks_mut::<DefaultSimdVector>().iter_mut())
{
    // Apply gain and soft clipping
    let gained = ops::mul(in_vec, DefaultSimdVector::splat(2.0));
    let clipped = math::tanh(gained);
    out_chunk.store(clipped);
}
```

### Example 3: Using ops, math, and table modules

```rust
use rigel_math::{DefaultSimdVector, ops, math, table};

// Operations module (functional style)
let a = DefaultSimdVector::splat(1.0);
let b = DefaultSimdVector::splat(2.0);
let result = ops::add(ops::mul(a, b), DefaultSimdVector::splat(0.5));

// Math module (fast approximations)
let x = DefaultSimdVector::splat(0.5);
let y = math::sin(x);
let z = math::exp(x);

// Table module (wavetable lookup)
let wavetable = vec![0.0; 2048];
let positions = DefaultSimdVector::splat(0.25);
let sample = table::linear_interp(&wavetable, positions);
```

---

## Platform-Specific Behavior

### x86_64 (Linux/Windows) with runtime-dispatch

**Configuration**:
```bash
cargo build --release  # Default: runtime-dispatch enabled
```

**Behavior**:
1. SimdContext initializes BackendDispatcher
2. CPU features detected once at startup via cpufeatures
3. Backend selected: AVX-512 → AVX2 → Scalar (priority order)
4. DefaultSimdVector resolves to ScalarVector (but backend dispatch happens transparently)
5. DSP code uses DefaultSimdVector - actual backend selected at runtime

**Overhead**: <1% dispatch overhead (initialization-time only)

### x86_64 (Linux/Windows) with forced backend

**Configuration**:
```bash
cargo build --release --features force-avx2
```

**Behavior**:
1. SimdContext is zero-sized (no dispatcher)
2. No CPU detection (compile-time selection)
3. DefaultSimdVector resolves to Avx2Vector directly
4. Zero runtime overhead

### aarch64 (macOS)

**Configuration**:
```bash
cargo build --release --features neon
```

**Behavior**:
1. SimdContext is zero-sized (no dispatcher)
2. No CPU detection (NEON always available)
3. DefaultSimdVector resolves to NeonVector
4. Zero runtime overhead

---

## Invariants and Constraints

### Functional Equivalence
- All backends MUST produce identical output (within floating-point rounding)
- Property-based tests verify: `assert_approx_eq!(scalar_output, avx2_output, 1e-6)`

### Performance Constraints
- Dispatch overhead: <1% vs direct backend call (SC-002)
- Single voice CPU: ~0.1% at 44.1kHz (maintain existing target)
- Full polyphonic: <1% CPU (maintain existing target)

### Safety Constraints
- No heap allocations in any backend or dispatcher
- No blocking operations
- no_std compatible
- Deterministic performance (no unexpected branching in hot path)

### Platform Constraints
- macOS (aarch64): Compile-time NEON selection only (zero overhead)
- Linux/Windows (x86_64): Runtime dispatch OR compile-time forced backend
- Each platform gets single binary with optimal backend selection

---

## Testing Strategy

### Unit Tests (rigel-math)
- Each backend: Correctness of SIMD operations
- Cross-backend equivalence tests
- Edge cases: NaN, infinity, zero, boundary values

### Integration Tests (rigel-dsp)
- SimdContext: Unified API works correctly
- Dispatcher: Correct backend selection based on features
- CPU detection: Accurate feature detection

### Property-Based Tests
- Cross-backend equivalence: All backends produce identical results
- Randomized inputs: 10,000+ test cases per operation

### CI Tests
- Scalar backend: Test on all platforms
- AVX2 backend: Test on x86_64 runners with AVX2
- NEON backend: Test on macos-latest (Apple Silicon)
- Runtime dispatch: Test backend selection logic
- AVX-512: Skip in CI (local testing only)

---

## Summary

The runtime SIMD dispatch architecture consists of **two layers**:

**Layer 1: rigel-math (SIMD Abstraction Library)**
1. `SimdVector` trait: Generic SIMD operations contract
2. Backend implementations: `ScalarVector`, `Avx2Vector`, `Avx512Vector`, `NeonVector`
3. `DefaultSimdVector`: Compile-time type alias resolving to appropriate backend
4. Modules: `ops`, `math`, `table`, `block` (all backend-agnostic)

**Layer 2: rigel-dsp (Runtime Dispatch)**
1. `BackendDispatcher`: Runtime CPU detection and backend selection (x86_64 only)
2. `CpuFeatures`: Detected CPU capabilities
3. `BackendType`: Selected backend enumeration
4. `SimdContext`: Unified abstraction over runtime dispatch (x86_64) and compile-time selection (aarch64)

All components are:
- no_std compatible
- Zero heap allocations
- Type-safe with compile-time verification
- Deterministic performance
- Functionally equivalent across backends (only performance differs)

The architecture enables:
- Single binary per platform with automatic optimal SIMD backend selection
- Backend-agnostic DSP code (write once, runs on all platforms)
- Deterministic testing via forced backend flags
- <1% runtime overhead on x86_64
- Zero overhead on aarch64
