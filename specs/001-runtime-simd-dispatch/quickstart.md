# Quickstart: Runtime SIMD Dispatch Implementation

**Feature**: Runtime SIMD Dispatch
**Target**: Rigel DSP Core (projects/rigel-synth/crates/dsp)
**Goal**: Transform compile-time SIMD selection to runtime dispatch for x86_64, maintain compile-time for aarch64

---

## Overview

This quickstart guides you through implementing runtime SIMD backend dispatch in Rigel's DSP core. After completion:

- ✅ Single binary per platform (Linux, Windows, macOS)
- ✅ Automatic optimal SIMD backend selection (scalar → AVX2 → AVX-512)
- ✅ Deterministic backend testing via force flags
- ✅ <1% runtime dispatch overhead
- ✅ Maintains no_std compliance

---

## Prerequisites

Before starting implementation:

1. **Read design documents**:
   - [research.md](research.md) - CPU detection library and dispatch pattern decisions
   - [data-model.md](data-model.md) - Backend contracts and relationships
   - [contracts/](contracts/) - Detailed API specifications

2. **Save performance baseline**:
   ```bash
   bench:baseline
   ```

3. **Ensure clean working directory**:
   ```bash
   git status  # Should be clean or on feature branch
   ```

4. **Verify devenv shell active**:
   ```bash
   echo $RIGEL_SYNTH_ROOT  # Should print path
   ```

---

## Implementation Phases

### Phase 1: Foundation (No Breaking Changes)

**Goal**: Add SIMD backend trait and scalar implementation without changing existing code.

**Location**: `projects/rigel-synth/crates/math/src/simd/` (new directory)

**Tasks**:

1. **Create SIMD module structure**:
   ```bash
   mkdir -p projects/rigel-synth/crates/math/src/simd
   ```

2. **Create `src/simd/mod.rs`**:
   ```rust
   //! SIMD backend dispatch for runtime CPU feature detection

   pub mod backend;
   pub mod dispatcher;
   pub mod scalar;

   pub use backend::SimdBackend;
   pub use dispatcher::{BackendDispatcher, CpuFeatures, BackendType};
   pub use scalar::ScalarBackend;
   ```

3. **Implement `src/simd/backend.rs`** (trait definition):
   - Copy contract from [contracts/simd_backend.rs](contracts/simd_backend.rs)
   - Define `SimdBackend` trait
   - Define `ProcessParams` struct
   - Add inline hints and documentation

4. **Implement `src/simd/scalar.rs`** (scalar backend):
   ```rust
   use super::backend::{SimdBackend, ProcessParams};

   #[derive(Copy, Clone)]
   pub struct ScalarBackend;

   impl SimdBackend for ScalarBackend {
       // Implement each method with scalar operations
       // See contracts/simd_backend.rs for full API
   }
   ```

5. **Add `cpufeatures` dependency** to `Cargo.toml`:
   ```toml
   [dependencies]
   cpufeatures = "0.2"

   [features]
   default = []
   runtime-dispatch = ["cpufeatures"]
   ```

6. **Test scalar backend**:
   ```bash
   cargo test -p rigel-dsp
   ```

**Exit Criteria**: Scalar backend compiles and passes basic tests, no existing code changed.

---

### Phase 2: CPU Feature Detection

**Goal**: Implement CPU feature detection using `cpufeatures` crate.

**Location**: `projects/rigel-synth/crates/math/src/simd/dispatcher.rs`

**Tasks**:

1. **Implement `CpuFeatures` struct**:
   ```rust
   #[derive(Debug, Clone, Copy)]
   pub struct CpuFeatures {
       pub has_avx2: bool,
       pub has_avx512_f: bool,
       pub has_avx512_bw: bool,
       pub has_avx512_dq: bool,
       pub has_avx512_vl: bool,
   }

   impl CpuFeatures {
       #[cfg(target_arch = "x86_64")]
       pub fn detect() -> Self {
           cpufeatures::new!(cpuid_avx2, "avx2");
           cpufeatures::new!(cpuid_avx512f, "avx512f");
           cpufeatures::new!(cpuid_avx512bw, "avx512bw");
           cpufeatures::new!(cpuid_avx512dq, "avx512dq");
           cpufeatures::new!(cpuid_avx512vl, "avx512vl");

           Self {
               has_avx2: cpuid_avx2::init().get(),
               has_avx512_f: cpuid_avx512f::init().get(),
               has_avx512_bw: cpuid_avx512bw::init().get(),
               has_avx512_dq: cpuid_avx512dq::init().get(),
               has_avx512_vl: cpuid_avx512vl::init().get(),
           }
       }

       pub fn has_avx512_full(&self) -> bool {
           self.has_avx512_f && self.has_avx512_bw
               && self.has_avx512_dq && self.has_avx512_vl
       }
   }
   ```

2. **Add unit test for CPU detection**:
   ```rust
   #[test]
   fn test_cpu_detection() {
       let features = CpuFeatures::detect();
       // Just verify it doesn't panic
       println!("Detected features: {:?}", features);
   }
   ```

3. **Run test**:
   ```bash
   cargo test -p rigel-dsp test_cpu_detection -- --nocapture
   ```

**Exit Criteria**: CPU feature detection works on all platforms without panics.

---

### Phase 3: AVX2 Backend Implementation

**Goal**: Implement AVX2 SIMD backend using x86_64 intrinsics.

**Location**: `projects/rigel-synth/crates/math/src/simd/avx2.rs`

**Tasks**:

1. **Create `src/simd/avx2.rs`**:
   ```rust
   #![cfg(all(target_arch = "x86_64", feature = "avx2"))]

   use core::arch::x86_64::*;
   use super::backend::{SimdBackend, ProcessParams};

   #[derive(Copy, Clone)]
   pub struct Avx2Backend;

   impl SimdBackend for Avx2Backend {
       #[inline]
       fn process_block(input: &[f32], output: &mut [f32], params: &ProcessParams) {
           let gain_vec = unsafe { _mm256_set1_ps(params.gain) };
           let mut i = 0;

           // Process 8 f32s at a time
           while i + 8 <= input.len() {
               unsafe {
                   let input_vec = _mm256_loadu_ps(&input[i]);
                   let result = _mm256_mul_ps(input_vec, gain_vec);
                   _mm256_storeu_ps(&mut output[i], result);
               }
               i += 8;
           }

           // Scalar fallback for remainder
           while i < input.len() {
               output[i] = input[i] * params.gain;
               i += 1;
           }
       }

       fn name() -> &'static str { "avx2" }
   }
   ```

2. **Add property-based test**:
   ```rust
   #[cfg(all(test, target_arch = "x86_64", feature = "avx2"))]
   mod tests {
       use super::*;
       use proptest::prelude::*;

       proptest! {
           #[test]
           fn avx2_matches_scalar(input in prop::collection::vec(-10.0f32..10.0f32, 0..1024)) {
               let mut scalar_out = vec![0.0f32; input.len()];
               let mut avx2_out = vec![0.0f32; input.len()];
               let params = ProcessParams { gain: 0.5, frequency: 440.0, sample_rate: 44100.0 };

               ScalarBackend::process_block(&input, &mut scalar_out, &params);
               Avx2Backend::process_block(&input, &mut avx2_out, &params);

               for (s, a) in scalar_out.iter().zip(avx2_out.iter()) {
                   assert!((s - a).abs() < 1e-6, "Scalar and AVX2 results differ");
               }
           }
       }
   }
   ```

3. **Run tests with AVX2 enabled**:
   ```bash
   test:avx2
   ```

**Exit Criteria**: AVX2 backend produces identical results to scalar (within 1e-6 tolerance).

---

### Phase 4: Dispatcher Implementation

**Goal**: Create function pointer dispatcher for runtime backend selection.

**Location**: `projects/rigel-synth/crates/math/src/simd/dispatcher.rs`

**Tasks**:

1. **Implement `BackendDispatcher`**:
   ```rust
   #[repr(C)]
   pub struct BackendDispatcher {
       process_block: unsafe fn(&[f32], &mut [f32], &ProcessParams),
       advance_phase: unsafe fn(&mut [f32], &[f32], usize),
       interpolate: unsafe fn(&[f32], &[f32], &mut [f32]),
       name: &'static str,
   }

   impl BackendDispatcher {
       pub fn init() -> Self {
           let features = CpuFeatures::detect();

           #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
           if features.has_avx2 {
               return Self::for_avx2();
           }

           Self::for_scalar()
       }

       fn for_scalar() -> Self {
           BackendDispatcher {
               process_block: |input, output, params| unsafe {
                   ScalarBackend::process_block(input, output, params)
               },
               // ... other function pointers
               name: "scalar",
           }
       }

       #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
       fn for_avx2() -> Self {
           BackendDispatcher {
               process_block: |input, output, params| unsafe {
                   Avx2Backend::process_block(input, output, params)
               },
               // ... other function pointers
               name: "avx2",
           }
       }

       #[inline]
       pub fn process_block(&self, input: &[f32], output: &mut [f32], params: &ProcessParams) {
           unsafe { (self.process_block)(input, output, params) }
       }
   }
   ```

2. **Add dispatcher test**:
   ```rust
   #[test]
   fn test_dispatcher_selection() {
       let dispatcher = BackendDispatcher::init();
       println!("Selected backend: {}", dispatcher.backend_name());
       assert!(!dispatcher.backend_name().is_empty());
   }
   ```

3. **Run test**:
   ```bash
   cargo test -p rigel-dsp test_dispatcher_selection -- --nocapture
   ```

**Exit Criteria**: Dispatcher correctly selects backend based on CPU features.

---

### Phase 5: Integration with SynthEngine

**Goal**: Use dispatcher in Rigel's `SynthEngine`.

**Location**: `projects/rigel-synth/crates/dsp/src/lib.rs`

**Tasks**:

1. **Add dispatcher to `SynthEngine`**:
   ```rust
   use crate::simd::BackendDispatcher;

   pub struct SynthEngine {
       dispatcher: BackendDispatcher,
       oscillator: SimpleOscillator,
       envelope: Envelope,
   }

   impl SynthEngine {
       pub fn new(sample_rate: f32) -> Self {
           let dispatcher = BackendDispatcher::init();

           #[cfg(debug_assertions)]
           eprintln!("Rigel using SIMD backend: {}", dispatcher.backend_name());

           Self {
               dispatcher,
               oscillator: SimpleOscillator::new(),
               envelope: Envelope::new(sample_rate),
           }
       }

       #[inline]
       pub fn process_sample(&mut self, params: &SynthParams) -> f32 {
           // Use dispatcher for SIMD operations
           // ...
       }
   }
   ```

2. **Run all tests**:
   ```bash
   cargo:test
   ```

**Exit Criteria**: SynthEngine initializes with dispatcher, all tests pass.

---

### Phase 6: Forced Backend Flags

**Goal**: Add build-time flags to force specific backends for CI testing.

**Location**: `projects/rigel-synth/crates/dsp/Cargo.toml`

**Tasks**:

1. **Add feature flags**:
   ```toml
   [features]
   default = ["runtime-dispatch"]
   runtime-dispatch = ["cpufeatures"]

   # SIMD backend compilation flags
   avx2 = []
   avx512 = []
   neon = []

   # Forced backend selection (deterministic testing)
   force-scalar = []
   force-avx2 = ["avx2"]
   force-avx512 = ["avx512"]
   force-neon = ["neon"]
   ```

2. **Update dispatcher to respect force flags**:
   ```rust
   impl BackendDispatcher {
       pub fn init() -> Self {
           #[cfg(feature = "force-scalar")]
           { return Self::for_scalar(); }

           #[cfg(feature = "force-avx2")]
           { return Self::for_avx2(); }

           // Runtime detection if no force flag
           let features = CpuFeatures::detect();
           // ...
       }
   }
   ```

3. **Test forced flags**:
   ```bash
   cargo test -p rigel-dsp --features force-scalar
   cargo test -p rigel-dsp --features force-avx2
   ```

**Exit Criteria**: Forced backend flags override runtime detection correctly.

---

### Phase 7: Performance Validation

**Goal**: Benchmark dispatch overhead and SIMD speedup.

**Location**: `projects/rigel-synth/crates/dsp/benches/dispatch_overhead.rs`

**Tasks**:

1. **Create benchmark**:
   ```rust
   use criterion::{black_box, criterion_group, criterion_main, Criterion};
   use rigel_math::simd::{BackendDispatcher, ProcessParams};
   use rigel_math::simd::scalar::ScalarBackend;

   fn bench_dispatch_overhead(c: &mut Criterion) {
       let dispatcher = BackendDispatcher::init();
       let mut output = vec![0.0f32; 4096];
       let input = vec![1.0f32; 4096];
       let params = ProcessParams { gain: 0.5, frequency: 440.0, sample_rate: 44100.0 };

       c.bench_function("dispatch_runtime", |b| {
           b.iter(|| {
               dispatcher.process_block(
                   black_box(&input),
                   black_box(&mut output),
                   black_box(&params)
               );
           });
       });

       c.bench_function("direct_scalar", |b| {
           b.iter(|| {
               ScalarBackend::process_block(
                   black_box(&input),
                   black_box(&mut output),
                   black_box(&params)
               );
           });
       });
   }

   criterion_group!(benches, bench_dispatch_overhead);
   criterion_main!(benches);
   ```

2. **Run benchmarks**:
   ```bash
   bench:criterion
   ```

3. **Validate overhead**:
   - Calculate: `(dispatch_time - direct_time) / direct_time`
   - Assert: overhead < 1% (success criteria SC-002)

**Exit Criteria**: Dispatch overhead is <1% of block processing time.

---

### Phase 8: CI Integration

**Goal**: Update CI pipeline to test all backends deterministically.

**Location**: `.github/workflows/ci.yml`

**Tasks**:

1. **Add backend-specific test jobs**:
   ```yaml
   test-backends:
     runs-on: ubuntu-latest
     strategy:
       matrix:
         backend: [scalar, avx2]
     steps:
       - uses: actions/checkout@v4
       - name: Test ${{ matrix.backend }} backend
         run: |
           devenv shell -- cargo test -p rigel-dsp --features force-${{ matrix.backend }}
   ```

2. **Add runtime dispatch test**:
   ```yaml
   test-runtime-dispatch:
     runs-on: ubuntu-latest
     steps:
       - uses: actions/checkout@v4
       - name: Test runtime dispatch
         run: |
           devenv shell -- cargo test -p rigel-dsp --features runtime-dispatch
   ```

3. **Push to CI**:
   ```bash
   git add .github/workflows/ci.yml
   git commit -m "Add SIMD backend testing to CI"
   git push
   ```

**Exit Criteria**: CI tests scalar, AVX2, and runtime dispatch modes successfully.

---

## Validation Checklist

Before considering implementation complete:

- [ ] All scalar backend tests pass
- [ ] AVX2 backend produces identical results to scalar (property-based tests)
- [ ] Dispatcher correctly selects backend based on CPU features
- [ ] SimdContext API provides 32 production-ready operations (2 advanced methods deferred for future iteration: select, cubic interpolation)
- [ ] Forced backend flags work in CI
- [ ] Dispatch overhead <1% (benchmark validation)
- [ ] no_std compliance maintained (compilation check)
- [ ] All architecture-specific tests pass (NEON on aarch64, AVX2 on x86_64)
- [ ] Build times within 10% of baseline
- [ ] Binary size increase <20%
- [ ] CI pipeline tests all backends deterministically
- [ ] Performance baseline comparison shows no regressions

---

## Common Issues & Solutions

### Issue: AVX2 tests fail with "illegal instruction"

**Cause**: Running AVX2 code on CPU without AVX2 support

**Solution**: Use forced backend flags or check CPU features:
```bash
# Test on AVX2-capable machine
lscpu | grep avx2

# Or use forced scalar for testing
cargo test --features force-scalar
```

### Issue: Dispatch overhead >1%

**Cause**: Function pointers not inlined or cache misses

**Solution**:
1. Ensure dispatcher methods marked `#[inline]`
2. Verify dispatcher stored in hot data structure (SynthEngine)
3. Profile with `cargo flamegraph` to identify bottlenecks

### Issue: SIMD backends produce different results

**Cause**: Floating-point rounding differences or incorrect intrinsics

**Solution**:
1. Use looser epsilon in tests (1e-6 instead of exact equality)
2. Verify SIMD intrinsics match scalar operations
3. Check for uninitialized memory or alignment issues

---

## Next Steps

After implementation is complete and validated:

1. **Create PR**: Push to GitHub and create pull request
2. **Wait for CI**: Ensure all CI checks pass
3. **Review checklist**: Complete specification quality checklist
4. **Proceed to tasks**: Run `/speckit.tasks` to generate implementation tasks

---

## References

- [research.md](research.md) - Technology choices and rationale
- [data-model.md](data-model.md) - Backend contracts and structure
- [contracts/](contracts/) - Detailed API specifications
- [CLAUDE.md](/home/kylecierzan/src/rigel/CLAUDE.md) - Rigel development guide
- [Constitution](/home/kylecierzan/src/rigel/.specify/memory/constitution.md) - Project principles
