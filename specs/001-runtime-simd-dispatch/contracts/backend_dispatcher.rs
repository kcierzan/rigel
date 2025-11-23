/// Backend Dispatcher Contract
///
/// This file defines the contract for the runtime SIMD backend dispatcher.
/// It is a specification document, not executable code.
///
/// Location: projects/rigel-synth/crates/math/src/simd/dispatcher.rs

/// CPU Feature Detection Results
///
/// Represents the SIMD capabilities detected on the current CPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuFeatures {
    /// AVX2 support available (x86_64)
    pub has_avx2: bool,

    /// AVX-512 Foundation (x86_64)
    pub has_avx512_f: bool,

    /// AVX-512 Byte & Word operations (x86_64)
    pub has_avx512_bw: bool,

    /// AVX-512 Doubleword & Quadword operations (x86_64)
    pub has_avx512_dq: bool,

    /// AVX-512 Vector Length extensions (x86_64)
    pub has_avx512_vl: bool,
}

impl CpuFeatures {
    /// Detect CPU features at runtime
    ///
    /// # Platform Behavior
    /// - **x86_64**: Runtime CPUID detection using `cpufeatures` crate
    /// - **aarch64**: All fields set to false (NEON assumed always present)
    ///
    /// # Performance
    /// - First call: ~100-200 CPU cycles (CPUID instruction)
    /// - Subsequent calls: Cached, near-zero cost
    ///
    /// # Safety
    /// - no_std compatible
    /// - Zero heap allocations
    /// - No undefined behavior
    ///
    /// # Example
    /// ```rust
    /// let features = CpuFeatures::detect();
    /// if features.has_avx2 {
    ///     println!("AVX2 available");
    /// }
    /// ```
    pub fn detect() -> Self;

    /// Check if full AVX-512 support is available
    ///
    /// Full AVX-512 requires Foundation + common extensions:
    /// - AVX-512F (Foundation)
    /// - AVX-512BW (Byte & Word)
    /// - AVX-512DQ (Doubleword & Quadword)
    /// - AVX-512VL (Vector Length)
    ///
    /// # Returns
    /// `true` if all required AVX-512 features are present
    pub fn has_avx512_full(&self) -> bool {
        self.has_avx512_f
            && self.has_avx512_bw
            && self.has_avx512_dq
            && self.has_avx512_vl
    }
}

/// Backend Type Enumeration
///
/// Represents the selected SIMD backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendType {
    /// Scalar fallback (no SIMD, always available)
    Scalar,

    /// AVX2 backend (x86_64, 256-bit SIMD)
    Avx2,

    /// AVX-512 backend (x86_64, 512-bit SIMD, experimental)
    Avx512,

    /// NEON backend (aarch64, 128-bit SIMD)
    Neon,
}

impl BackendType {
    /// Select optimal backend based on CPU features
    ///
    /// # Selection Priority (when runtime-dispatch enabled)
    /// 1. AVX-512 (if `avx512` feature compiled AND CPU supports full AVX-512)
    /// 2. AVX2 (if `avx2` feature compiled AND CPU supports AVX2)
    /// 3. NEON (if `aarch64` platform)
    /// 4. Scalar (fallback, always available)
    ///
    /// # Forced Backend Selection (CI testing)
    /// When `force-scalar`, `force-avx2`, or `force-avx512` features are enabled,
    /// this function ignores CPU detection and returns the forced backend.
    ///
    /// # Parameters
    /// - `features`: Detected CPU features from `CpuFeatures::detect()`
    ///
    /// # Example
    /// ```rust
    /// let features = CpuFeatures::detect();
    /// let backend = BackendType::select(features);
    /// println!("Selected backend: {}", backend.name());
    /// ```
    pub fn select(features: CpuFeatures) -> Self;

    /// Backend name for logging/debugging
    ///
    /// # Returns
    /// Static string: "scalar", "avx2", "avx512", or "neon"
    pub fn name(&self) -> &'static str {
        match self {
            BackendType::Scalar => "scalar",
            BackendType::Avx2 => "avx2",
            BackendType::Avx512 => "avx512",
            BackendType::Neon => "neon",
        }
    }
}

/// Runtime Backend Dispatcher
///
/// Function pointer table for dispatching SIMD operations to the selected backend.
///
/// # Lifecycle
/// 1. **Initialization** (one-time, before real-time processing):
///    - Detect CPU features via `CpuFeatures::detect()`
///    - Select optimal backend via `BackendType::select()`
///    - Initialize function pointer table
///
/// 2. **Real-time use** (hot path):
///    - Call dispatch methods (`process_block`, `advance_phase`, etc.)
///    - Single indirect jump through function pointer
///    - No branching, no feature detection, predictable performance
///
/// # Memory Layout
/// The struct uses `#[repr(C)]` to ensure stable ABI and predictable layout:
/// - 3 function pointers (24 bytes on 64-bit systems)
/// - 1 static string reference (8 bytes on 64-bit systems)
/// - Total: 32 bytes (fits in single cache line)
///
/// # Example
/// ```rust
/// // Initialization (before real-time processing starts)
/// let dispatcher = BackendDispatcher::init();
/// println!("Using backend: {}", dispatcher.backend_name());
///
/// // Real-time hot path
/// let input = [1.0f32; 1024];
/// let mut output = [0.0f32; 1024];
/// let params = ProcessParams { gain: 0.5, frequency: 440.0, sample_rate: 44100.0 };
///
/// dispatcher.process_block(&input, &mut output, &params);
/// ```
#[repr(C)]
pub struct BackendDispatcher {
    /// Process audio block function pointer
    process_block_fn: unsafe fn(&[f32], &mut [f32], &ProcessParams),

    /// Advance oscillator phases function pointer
    advance_phase_fn: unsafe fn(&mut [f32], &[f32], usize),

    /// Wavetable interpolation function pointer
    interpolate_fn: unsafe fn(&[f32], &[f32], &mut [f32]),

    /// Backend name for logging/debugging
    backend_name: &'static str,
}

impl BackendDispatcher {
    /// Initialize dispatcher with optimal backend
    ///
    /// This function:
    /// 1. Detects CPU features (via `cpufeatures` crate)
    /// 2. Selects best available backend based on features + compilation flags
    /// 3. Initializes function pointer table
    ///
    /// # Platform Behavior
    /// - **macOS (aarch64)**: Always selects NEON backend (compile-time)
    /// - **Linux/Windows (x86_64)**: Runtime detection, selects AVX-512 → AVX2 → Scalar
    ///
    /// # Performance
    /// - First call: ~100-200 CPU cycles (CPUID + function pointer setup)
    /// - Should be called once at plugin initialization, not in real-time path
    ///
    /// # Safety
    /// - no_std compatible
    /// - Zero heap allocations
    /// - No undefined behavior
    ///
    /// # Example
    /// ```rust
    /// // Call once at plugin initialization
    /// let dispatcher = BackendDispatcher::init();
    /// // Store in SynthEngine or plugin state
    /// ```
    pub fn init() -> Self;

    /// Process a block of audio samples
    ///
    /// Dispatches to the selected backend's `process_block` implementation.
    ///
    /// # Performance
    /// - Overhead: Single indirect jump through function pointer (<1% vs direct call)
    /// - Branch prediction: 95%+ accuracy (stable function pointer)
    /// - I-cache friendly: Function pointer in hot cache line
    ///
    /// # Parameters
    /// - `input`: Input audio buffer
    /// - `output`: Output audio buffer (same length as input)
    /// - `params`: Processing parameters
    ///
    /// # Example
    /// ```rust
    /// dispatcher.process_block(&input, &mut output, &params);
    /// ```
    #[inline]
    pub fn process_block(&self, input: &[f32], output: &mut [f32], params: &ProcessParams) {
        unsafe { (self.process_block_fn)(input, output, params) }
    }

    /// Advance oscillator phases with SIMD vectorization
    ///
    /// Dispatches to the selected backend's `advance_phase_vectorized` implementation.
    ///
    /// # Parameters
    /// - `phases`: Current phase values (mutable, wrapped to [0, TAU))
    /// - `phase_increments`: Phase delta per sample
    /// - `count`: Number of phases to advance
    ///
    /// # Example
    /// ```rust
    /// dispatcher.advance_phase(&mut phases, &increments, 64);
    /// ```
    #[inline]
    pub fn advance_phase(&self, phases: &mut [f32], increments: &[f32], count: usize) {
        unsafe { (self.advance_phase_fn)(phases, increments, count) }
    }

    /// Wavetable interpolation with SIMD
    ///
    /// Dispatches to the selected backend's `interpolate_wavetable` implementation.
    ///
    /// # Parameters
    /// - `wavetable`: Source wavetable data
    /// - `positions`: Normalized read positions (0.0 to 1.0)
    /// - `output`: Interpolated output samples
    ///
    /// # Example
    /// ```rust
    /// dispatcher.interpolate(&wavetable, &positions, &mut output);
    /// ```
    #[inline]
    pub fn interpolate(&self, wavetable: &[f32], positions: &[f32], output: &mut [f32]) {
        unsafe { (self.interpolate_fn)(wavetable, positions, output) }
    }

    /// Get backend name for logging/debugging
    ///
    /// # Returns
    /// Static string: "scalar", "avx2", "avx512", or "neon"
    ///
    /// # Example
    /// ```rust
    /// println!("Using backend: {}", dispatcher.backend_name());
    /// ```
    pub fn backend_name(&self) -> &'static str {
        self.backend_name
    }

    /// Query selected backend type
    ///
    /// Returns the `BackendType` enum for the currently active backend.
    ///
    /// # Use Cases
    /// - Unit tests: Verify forced backend flags work correctly
    /// - Benchmarks: Ensure specific backend is being measured
    /// - Debugging: Display backend selection logic
    ///
    /// # Example
    /// ```rust
    /// let backend_type = dispatcher.backend_type();
    /// assert_eq!(backend_type, BackendType::Avx2);
    /// ```
    pub fn backend_type(&self) -> BackendType {
        match self.backend_name {
            "scalar" => BackendType::Scalar,
            "avx2" => BackendType::Avx2,
            "avx512" => BackendType::Avx512,
            "neon" => BackendType::Neon,
            _ => unreachable!("Invalid backend name"),
        }
    }
}

/// Testing Requirements
///
/// The dispatcher must pass these tests:
///
/// 1. **Correct Backend Selection**:
///    ```rust
///    #[test]
///    fn test_backend_selection() {
///        let dispatcher = BackendDispatcher::init();
///        let features = CpuFeatures::detect();
///
///        if features.has_avx512_full() && cfg!(feature = "avx512") {
///            assert_eq!(dispatcher.backend_type(), BackendType::Avx512);
///        } else if features.has_avx2 && cfg!(feature = "avx2") {
///            assert_eq!(dispatcher.backend_type(), BackendType::Avx2);
///        } else {
///            assert_eq!(dispatcher.backend_type(), BackendType::Scalar);
///        }
///    }
///    ```
///
/// 2. **Forced Backend Flags**:
///    ```rust
///    #[test]
///    #[cfg(feature = "force-avx2")]
///    fn test_force_avx2() {
///        let dispatcher = BackendDispatcher::init();
///        assert_eq!(dispatcher.backend_type(), BackendType::Avx2);
///    }
///    ```
///
/// 3. **Dispatch Overhead Benchmark**:
///    ```rust
///    #[bench]
///    fn bench_dispatch_overhead(c: &mut Criterion) {
///        let dispatcher = BackendDispatcher::init();
///        let mut output = vec![0.0f32; 4096];
///        let input = vec![1.0f32; 4096];
///
///        c.bench_function("dispatched", |b| {
///            b.iter(|| dispatcher.process_block(&input, &mut output, &params));
///        });
///
///        c.bench_function("direct_scalar", |b| {
///            b.iter(|| ScalarBackend::process_block(&input, &mut output, &params));
///        });
///        // Assert: (dispatched_time - direct_time) / direct_time < 0.01
///    }
///    ```
///
/// 4. **Cross-Platform Consistency**:
///    ```rust
///    #[test]
///    fn test_cross_platform_init() {
///        // Should not panic on any platform
///        let dispatcher = BackendDispatcher::init();
///        assert!(!dispatcher.backend_name().is_empty());
///    }
///    ```

/// Safety Invariants
///
/// The dispatcher maintains these safety guarantees:
///
/// 1. **Function Pointer Validity**:
///    - All function pointers initialized to valid backend implementations
///    - Never null or dangling
///    - Type signatures match exactly across all backends
///
/// 2. **Immutability After Init**:
///    - Dispatcher is immutable after `init()` returns
///    - No dynamic backend switching during execution
///    - Thread-safe: Can be shared across threads (if SynthEngine is Send)
///
/// 3. **no_std Compliance**:
///    - Zero heap allocations
///    - Stack-only data structures
///    - No dependencies on std library
///
/// 4. **Real-Time Safety**:
///    - Deterministic performance (no unexpected branching)
///    - Predictable CPU usage
///    - No blocking operations
