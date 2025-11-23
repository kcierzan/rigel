//! Runtime Backend Dispatcher
//!
//! This module provides runtime CPU feature detection and SIMD backend selection.
//! On x86_64, it detects AVX2/AVX-512 support and selects the optimal backend.
//! On aarch64, NEON is assumed always present (compile-time selection).

#![allow(unused)]

use super::backend::{ProcessParams, SimdBackend};

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
    /// - Subsequent calls: Cached by cpufeatures, near-zero cost
    ///
    /// # Safety
    /// - no_std compatible
    /// - Zero heap allocations
    /// - No undefined behavior
    ///
    /// # Example
    /// ```ignore
    /// let features = CpuFeatures::detect();
    /// if features.has_avx2 {
    ///     println!("AVX2 available");
    /// }
    /// ```
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            cpufeatures::new!(cpuid_avx2, "avx2");
            cpufeatures::new!(cpuid_avx512f, "avx512f");
            cpufeatures::new!(cpuid_avx512bw, "avx512bw");
            cpufeatures::new!(cpuid_avx512dq, "avx512dq");
            cpufeatures::new!(cpuid_avx512vl, "avx512vl");

            Self {
                has_avx2: cpuid_avx2::get(),
                has_avx512_f: cpuid_avx512f::get(),
                has_avx512_bw: cpuid_avx512bw::get(),
                has_avx512_dq: cpuid_avx512dq::get(),
                has_avx512_vl: cpuid_avx512vl::get(),
            }
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            // On aarch64, NEON is assumed always available
            // No runtime detection needed
            Self {
                has_avx2: false,
                has_avx512_f: false,
                has_avx512_bw: false,
                has_avx512_dq: false,
                has_avx512_vl: false,
            }
        }
    }

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
        self.has_avx512_f && self.has_avx512_bw && self.has_avx512_dq && self.has_avx512_vl
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

// Type aliases for function pointers to reduce complexity
type BinaryOpFn = fn(&[f32], &[f32], &mut [f32]);
type TernaryOpFn = fn(&[f32], &[f32], &[f32], &mut [f32]);
type UnaryOpFn = fn(&[f32], &mut [f32]);
type ProcessBlockFn = fn(&[f32], &mut [f32], &ProcessParams);
type AdvancePhaseFn = fn(&mut [f32], &[f32], usize);

/// Runtime Backend Dispatcher
///
/// Function pointer table for dispatching SIMD operations to the selected backend.
/// This struct is initialized once at startup and provides zero-overhead dispatch
/// through function pointers.
#[repr(C)]
#[derive(Clone, Debug)]
pub struct BackendDispatcher {
    /// Process audio block function pointer
    process_block_fn: ProcessBlockFn,

    /// Advance oscillator phases function pointer
    advance_phase_fn: AdvancePhaseFn,

    /// Wavetable interpolation function pointer
    interpolate_fn: BinaryOpFn,

    // ========================================================================
    // Arithmetic Operations (Binary)
    // ========================================================================
    add_fn: BinaryOpFn,
    sub_fn: BinaryOpFn,
    mul_fn: BinaryOpFn,
    div_fn: BinaryOpFn,
    fma_fn: TernaryOpFn,

    // ========================================================================
    // Arithmetic Operations (Unary)
    // ========================================================================
    neg_fn: UnaryOpFn,
    abs_fn: UnaryOpFn,

    // ========================================================================
    // Comparison Operations
    // ========================================================================
    min_fn: BinaryOpFn,
    max_fn: BinaryOpFn,

    // ========================================================================
    // Basic Math Functions
    // ========================================================================
    sqrt_fn: UnaryOpFn,
    exp_fn: UnaryOpFn,
    log_fn: UnaryOpFn,
    log2_fn: UnaryOpFn,
    log10_fn: UnaryOpFn,
    pow_fn: BinaryOpFn,

    // ========================================================================
    // Trigonometric Functions
    // ========================================================================
    sin_fn: UnaryOpFn,
    cos_fn: UnaryOpFn,
    tan_fn: UnaryOpFn,
    asin_fn: UnaryOpFn,
    acos_fn: UnaryOpFn,
    atan_fn: UnaryOpFn,
    atan2_fn: BinaryOpFn,

    // ========================================================================
    // Hyperbolic Functions
    // ========================================================================
    sinh_fn: UnaryOpFn,
    cosh_fn: UnaryOpFn,
    tanh_fn: UnaryOpFn,

    // ========================================================================
    // Rounding Functions
    // ========================================================================
    floor_fn: UnaryOpFn,
    ceil_fn: UnaryOpFn,
    round_fn: UnaryOpFn,
    trunc_fn: UnaryOpFn,

    /// Backend name for logging/debugging
    backend_name: &'static str,
}

impl BackendDispatcher {
    /// Initialize dispatcher with optimal backend
    ///
    /// This function:
    /// 1. Detects CPU features (via `cpufeatures` crate on x86_64)
    /// 2. Selects best available backend based on features + compilation flags
    /// 3. Initializes function pointer table
    ///
    /// # Platform Behavior
    /// - **x86_64**: Runtime detection, selects AVX-512 → AVX2 → Scalar
    /// - **aarch64**: Always selects NEON backend (or scalar if NEON not compiled)
    ///
    /// # Performance
    /// - First call: ~100-200 CPU cycles (CPUID + function pointer setup)
    /// - Should be called once at plugin initialization, not in real-time path
    pub fn init() -> Self {
        let features = CpuFeatures::detect();
        let backend_type = BackendType::select(features);

        match backend_type {
            BackendType::Scalar => Self::for_scalar(),

            #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
            BackendType::Avx2 => Self::for_avx2(),

            #[cfg(all(feature = "avx512", any(target_arch = "x86", target_arch = "x86_64")))]
            BackendType::Avx512 => Self::for_avx512(),

            #[cfg(all(feature = "neon", target_arch = "aarch64"))]
            BackendType::Neon => Self::for_neon(),

            // Fallback to scalar if specific backend not compiled
            #[allow(unreachable_patterns)]
            _ => Self::for_scalar(),
        }
    }

    /// Create dispatcher for scalar backend
    fn for_scalar() -> Self {
        use super::scalar::ScalarBackend;
        Self {
            process_block_fn: ScalarBackend::process_block,
            advance_phase_fn: ScalarBackend::advance_phase_vectorized,
            interpolate_fn: ScalarBackend::interpolate_wavetable,
            // Arithmetic Operations (Binary)
            add_fn: ScalarBackend::add,
            sub_fn: ScalarBackend::sub,
            mul_fn: ScalarBackend::mul,
            div_fn: ScalarBackend::div,
            fma_fn: ScalarBackend::fma,
            // Arithmetic Operations (Unary)
            neg_fn: ScalarBackend::neg,
            abs_fn: ScalarBackend::abs,
            // Comparison Operations
            min_fn: ScalarBackend::min,
            max_fn: ScalarBackend::max,
            // Basic Math Functions
            sqrt_fn: ScalarBackend::sqrt,
            exp_fn: ScalarBackend::exp,
            log_fn: ScalarBackend::log,
            log2_fn: ScalarBackend::log2,
            log10_fn: ScalarBackend::log10,
            pow_fn: ScalarBackend::pow,
            // Trigonometric Functions
            sin_fn: ScalarBackend::sin,
            cos_fn: ScalarBackend::cos,
            tan_fn: ScalarBackend::tan,
            asin_fn: ScalarBackend::asin,
            acos_fn: ScalarBackend::acos,
            atan_fn: ScalarBackend::atan,
            atan2_fn: ScalarBackend::atan2,
            // Hyperbolic Functions
            sinh_fn: ScalarBackend::sinh,
            cosh_fn: ScalarBackend::cosh,
            tanh_fn: ScalarBackend::tanh,
            // Rounding Functions
            floor_fn: ScalarBackend::floor,
            ceil_fn: ScalarBackend::ceil,
            round_fn: ScalarBackend::round,
            trunc_fn: ScalarBackend::trunc,
            backend_name: "scalar",
        }
    }

    /// Create dispatcher for AVX2 backend
    #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
    fn for_avx2() -> Self {
        use super::avx2::Avx2Backend;
        Self {
            process_block_fn: Avx2Backend::process_block,
            advance_phase_fn: Avx2Backend::advance_phase_vectorized,
            interpolate_fn: Avx2Backend::interpolate_wavetable,
            // Arithmetic Operations (Binary)
            add_fn: Avx2Backend::add,
            sub_fn: Avx2Backend::sub,
            mul_fn: Avx2Backend::mul,
            div_fn: Avx2Backend::div,
            fma_fn: Avx2Backend::fma,
            // Arithmetic Operations (Unary)
            neg_fn: Avx2Backend::neg,
            abs_fn: Avx2Backend::abs,
            // Comparison Operations
            min_fn: Avx2Backend::min,
            max_fn: Avx2Backend::max,
            // Basic Math Functions
            sqrt_fn: Avx2Backend::sqrt,
            exp_fn: Avx2Backend::exp,
            log_fn: Avx2Backend::log,
            log2_fn: Avx2Backend::log2,
            log10_fn: Avx2Backend::log10,
            pow_fn: Avx2Backend::pow,
            // Trigonometric Functions
            sin_fn: Avx2Backend::sin,
            cos_fn: Avx2Backend::cos,
            tan_fn: Avx2Backend::tan,
            asin_fn: Avx2Backend::asin,
            acos_fn: Avx2Backend::acos,
            atan_fn: Avx2Backend::atan,
            atan2_fn: Avx2Backend::atan2,
            // Hyperbolic Functions
            sinh_fn: Avx2Backend::sinh,
            cosh_fn: Avx2Backend::cosh,
            tanh_fn: Avx2Backend::tanh,
            // Rounding Functions
            floor_fn: Avx2Backend::floor,
            ceil_fn: Avx2Backend::ceil,
            round_fn: Avx2Backend::round,
            trunc_fn: Avx2Backend::trunc,
            backend_name: "avx2",
        }
    }

    /// Create dispatcher for AVX-512 backend
    #[cfg(all(feature = "avx512", any(target_arch = "x86", target_arch = "x86_64")))]
    fn for_avx512() -> Self {
        use super::avx512::Avx512Backend;
        Self {
            process_block_fn: Avx512Backend::process_block,
            advance_phase_fn: Avx512Backend::advance_phase_vectorized,
            interpolate_fn: Avx512Backend::interpolate_wavetable,
            // Arithmetic Operations (Binary)
            add_fn: Avx512Backend::add,
            sub_fn: Avx512Backend::sub,
            mul_fn: Avx512Backend::mul,
            div_fn: Avx512Backend::div,
            fma_fn: Avx512Backend::fma,
            // Arithmetic Operations (Unary)
            neg_fn: Avx512Backend::neg,
            abs_fn: Avx512Backend::abs,
            // Comparison Operations
            min_fn: Avx512Backend::min,
            max_fn: Avx512Backend::max,
            // Basic Math Functions
            sqrt_fn: Avx512Backend::sqrt,
            exp_fn: Avx512Backend::exp,
            log_fn: Avx512Backend::log,
            log2_fn: Avx512Backend::log2,
            log10_fn: Avx512Backend::log10,
            pow_fn: Avx512Backend::pow,
            // Trigonometric Functions
            sin_fn: Avx512Backend::sin,
            cos_fn: Avx512Backend::cos,
            tan_fn: Avx512Backend::tan,
            asin_fn: Avx512Backend::asin,
            acos_fn: Avx512Backend::acos,
            atan_fn: Avx512Backend::atan,
            atan2_fn: Avx512Backend::atan2,
            // Hyperbolic Functions
            sinh_fn: Avx512Backend::sinh,
            cosh_fn: Avx512Backend::cosh,
            tanh_fn: Avx512Backend::tanh,
            // Rounding Functions
            floor_fn: Avx512Backend::floor,
            ceil_fn: Avx512Backend::ceil,
            round_fn: Avx512Backend::round,
            trunc_fn: Avx512Backend::trunc,
            backend_name: "avx512",
        }
    }

    /// Create dispatcher for NEON backend
    #[cfg(all(feature = "neon", target_arch = "aarch64"))]
    fn for_neon() -> Self {
        use super::neon::NeonBackend;
        Self {
            process_block_fn: NeonBackend::process_block,
            advance_phase_fn: NeonBackend::advance_phase_vectorized,
            interpolate_fn: NeonBackend::interpolate_wavetable,
            // Arithmetic Operations (Binary)
            add_fn: NeonBackend::add,
            sub_fn: NeonBackend::sub,
            mul_fn: NeonBackend::mul,
            div_fn: NeonBackend::div,
            fma_fn: NeonBackend::fma,
            // Arithmetic Operations (Unary)
            neg_fn: NeonBackend::neg,
            abs_fn: NeonBackend::abs,
            // Comparison Operations
            min_fn: NeonBackend::min,
            max_fn: NeonBackend::max,
            // Basic Math Functions
            sqrt_fn: NeonBackend::sqrt,
            exp_fn: NeonBackend::exp,
            log_fn: NeonBackend::log,
            log2_fn: NeonBackend::log2,
            log10_fn: NeonBackend::log10,
            pow_fn: NeonBackend::pow,
            // Trigonometric Functions
            sin_fn: NeonBackend::sin,
            cos_fn: NeonBackend::cos,
            tan_fn: NeonBackend::tan,
            asin_fn: NeonBackend::asin,
            acos_fn: NeonBackend::acos,
            atan_fn: NeonBackend::atan,
            atan2_fn: NeonBackend::atan2,
            // Hyperbolic Functions
            sinh_fn: NeonBackend::sinh,
            cosh_fn: NeonBackend::cosh,
            tanh_fn: NeonBackend::tanh,
            // Rounding Functions
            floor_fn: NeonBackend::floor,
            ceil_fn: NeonBackend::ceil,
            round_fn: NeonBackend::round,
            trunc_fn: NeonBackend::trunc,
            backend_name: "neon",
        }
    }

    /// Process a block of audio samples
    ///
    /// Dispatches to the selected backend's `process_block` implementation.
    ///
    /// # Performance
    /// - Overhead: Single indirect jump through function pointer (<1% vs direct call)
    /// - Branch prediction: 95%+ accuracy (stable function pointer)
    #[inline]
    pub fn process_block(&self, input: &[f32], output: &mut [f32], params: &ProcessParams) {
        (self.process_block_fn)(input, output, params)
    }

    /// Advance oscillator phases with SIMD vectorization
    ///
    /// Dispatches to the selected backend's `advance_phase_vectorized` implementation.
    #[inline]
    pub fn advance_phase(&self, phases: &mut [f32], increments: &[f32], count: usize) {
        (self.advance_phase_fn)(phases, increments, count)
    }

    /// Wavetable interpolation with SIMD
    ///
    /// Dispatches to the selected backend's `interpolate_wavetable` implementation.
    #[inline]
    pub fn interpolate(&self, wavetable: &[f32], positions: &[f32], output: &mut [f32]) {
        (self.interpolate_fn)(wavetable, positions, output)
    }

    // ========================================================================
    // Arithmetic Operations (Binary)
    // ========================================================================

    /// Element-wise addition: output[i] = a[i] + b[i]
    #[inline]
    pub fn add(&self, a: &[f32], b: &[f32], output: &mut [f32]) {
        (self.add_fn)(a, b, output)
    }

    /// Element-wise subtraction: output[i] = a[i] - b[i]
    #[inline]
    pub fn sub(&self, a: &[f32], b: &[f32], output: &mut [f32]) {
        (self.sub_fn)(a, b, output)
    }

    /// Element-wise multiplication: output[i] = a[i] * b[i]
    #[inline]
    pub fn mul(&self, a: &[f32], b: &[f32], output: &mut [f32]) {
        (self.mul_fn)(a, b, output)
    }

    /// Element-wise division: output[i] = a[i] / b[i]
    #[inline]
    pub fn div(&self, a: &[f32], b: &[f32], output: &mut [f32]) {
        (self.div_fn)(a, b, output)
    }

    /// Element-wise fused multiply-add: output[i] = a[i] * b[i] + c[i]
    #[inline]
    pub fn fma(&self, a: &[f32], b: &[f32], c: &[f32], output: &mut [f32]) {
        (self.fma_fn)(a, b, c, output)
    }

    // ========================================================================
    // Arithmetic Operations (Unary)
    // ========================================================================

    /// Element-wise negation: output[i] = -input[i]
    #[inline]
    pub fn neg(&self, input: &[f32], output: &mut [f32]) {
        (self.neg_fn)(input, output)
    }

    /// Element-wise absolute value: output[i] = |input[i]|
    #[inline]
    pub fn abs(&self, input: &[f32], output: &mut [f32]) {
        (self.abs_fn)(input, output)
    }

    // ========================================================================
    // Comparison Operations
    // ========================================================================

    /// Element-wise minimum: output[i] = min(a[i], b[i])
    #[inline]
    pub fn min(&self, a: &[f32], b: &[f32], output: &mut [f32]) {
        (self.min_fn)(a, b, output)
    }

    /// Element-wise maximum: output[i] = max(a[i], b[i])
    #[inline]
    pub fn max(&self, a: &[f32], b: &[f32], output: &mut [f32]) {
        (self.max_fn)(a, b, output)
    }

    // ========================================================================
    // Basic Math Functions
    // ========================================================================

    /// Element-wise square root: output[i] = sqrt(input[i])
    #[inline]
    pub fn sqrt(&self, input: &[f32], output: &mut [f32]) {
        (self.sqrt_fn)(input, output)
    }

    /// Element-wise exponential: output[i] = e^input[i]
    #[inline]
    pub fn exp(&self, input: &[f32], output: &mut [f32]) {
        (self.exp_fn)(input, output)
    }

    /// Element-wise natural logarithm: output[i] = ln(input[i])
    #[inline]
    pub fn log(&self, input: &[f32], output: &mut [f32]) {
        (self.log_fn)(input, output)
    }

    /// Element-wise base-2 logarithm: output[i] = log2(input[i])
    #[inline]
    pub fn log2(&self, input: &[f32], output: &mut [f32]) {
        (self.log2_fn)(input, output)
    }

    /// Element-wise base-10 logarithm: output[i] = log10(input[i])
    #[inline]
    pub fn log10(&self, input: &[f32], output: &mut [f32]) {
        (self.log10_fn)(input, output)
    }

    /// Element-wise power: output[i] = base[i]^exponent[i]
    #[inline]
    pub fn pow(&self, base: &[f32], exponent: &[f32], output: &mut [f32]) {
        (self.pow_fn)(base, exponent, output)
    }

    // ========================================================================
    // Trigonometric Functions
    // ========================================================================

    /// Element-wise sine: output[i] = sin(input[i])
    #[inline]
    pub fn sin(&self, input: &[f32], output: &mut [f32]) {
        (self.sin_fn)(input, output)
    }

    /// Element-wise cosine: output[i] = cos(input[i])
    #[inline]
    pub fn cos(&self, input: &[f32], output: &mut [f32]) {
        (self.cos_fn)(input, output)
    }

    /// Element-wise tangent: output[i] = tan(input[i])
    #[inline]
    pub fn tan(&self, input: &[f32], output: &mut [f32]) {
        (self.tan_fn)(input, output)
    }

    /// Element-wise arcsine: output[i] = asin(input[i])
    #[inline]
    pub fn asin(&self, input: &[f32], output: &mut [f32]) {
        (self.asin_fn)(input, output)
    }

    /// Element-wise arccosine: output[i] = acos(input[i])
    #[inline]
    pub fn acos(&self, input: &[f32], output: &mut [f32]) {
        (self.acos_fn)(input, output)
    }

    /// Element-wise arctangent: output[i] = atan(input[i])
    #[inline]
    pub fn atan(&self, input: &[f32], output: &mut [f32]) {
        (self.atan_fn)(input, output)
    }

    /// Element-wise two-argument arctangent: output[i] = atan2(y[i], x[i])
    #[inline]
    pub fn atan2(&self, y: &[f32], x: &[f32], output: &mut [f32]) {
        (self.atan2_fn)(y, x, output)
    }

    // ========================================================================
    // Hyperbolic Functions
    // ========================================================================

    /// Element-wise hyperbolic sine: output[i] = sinh(input[i])
    #[inline]
    pub fn sinh(&self, input: &[f32], output: &mut [f32]) {
        (self.sinh_fn)(input, output)
    }

    /// Element-wise hyperbolic cosine: output[i] = cosh(input[i])
    #[inline]
    pub fn cosh(&self, input: &[f32], output: &mut [f32]) {
        (self.cosh_fn)(input, output)
    }

    /// Element-wise hyperbolic tangent: output[i] = tanh(input[i])
    #[inline]
    pub fn tanh(&self, input: &[f32], output: &mut [f32]) {
        (self.tanh_fn)(input, output)
    }

    // ========================================================================
    // Rounding Functions
    // ========================================================================

    /// Element-wise floor: output[i] = floor(input[i])
    #[inline]
    pub fn floor(&self, input: &[f32], output: &mut [f32]) {
        (self.floor_fn)(input, output)
    }

    /// Element-wise ceiling: output[i] = ceil(input[i])
    #[inline]
    pub fn ceil(&self, input: &[f32], output: &mut [f32]) {
        (self.ceil_fn)(input, output)
    }

    /// Element-wise rounding: output[i] = round(input[i])
    #[inline]
    pub fn round(&self, input: &[f32], output: &mut [f32]) {
        (self.round_fn)(input, output)
    }

    /// Element-wise truncation: output[i] = trunc(input[i])
    #[inline]
    pub fn trunc(&self, input: &[f32], output: &mut [f32]) {
        (self.trunc_fn)(input, output)
    }

    /// Get backend name for logging/debugging
    ///
    /// # Returns
    /// Static string: "scalar", "avx2", "avx512", or "neon"
    pub fn backend_name(&self) -> &'static str {
        self.backend_name
    }

    /// Query selected backend type
    ///
    /// Returns the `BackendType` enum for the currently active backend.
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
    /// When `force-scalar`, `force-avx2`, `force-avx512`, or `force-neon` features are enabled,
    /// this function ignores CPU detection and returns the forced backend.
    ///
    /// # Parameters
    /// - `features`: Detected CPU features from `CpuFeatures::detect()`
    ///
    /// # Example
    /// ```ignore
    /// let features = CpuFeatures::detect();
    /// let backend = BackendType::select(features);
    /// println!("Selected backend: {}", backend.name());
    /// ```
    pub fn select(features: CpuFeatures) -> Self {
        // Check forced backend flags first (deterministic testing)
        #[cfg(feature = "force-scalar")]
        {
            return BackendType::Scalar;
        }

        #[cfg(feature = "force-avx2")]
        {
            return BackendType::Avx2;
        }

        #[cfg(feature = "force-avx512")]
        {
            return BackendType::Avx512;
        }

        #[cfg(feature = "force-neon")]
        {
            return BackendType::Neon;
        }

        // Runtime detection (priority: AVX-512 → AVX2 → NEON → Scalar)
        #[cfg(target_arch = "x86_64")]
        {
            #[cfg(feature = "avx512")]
            if features.has_avx512_full() {
                return BackendType::Avx512;
            }

            #[cfg(feature = "avx2")]
            if features.has_avx2 {
                return BackendType::Avx2;
            }

            BackendType::Scalar
        }

        #[cfg(target_arch = "aarch64")]
        {
            #[cfg(feature = "neon")]
            {
                BackendType::Neon
            }

            #[cfg(not(feature = "neon"))]
            {
                BackendType::Scalar
            }
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            BackendType::Scalar
        }
    }

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
