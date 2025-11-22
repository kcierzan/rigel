//! Denormal number protection for real-time audio processing
//!
//! Denormal (subnormal) floating-point numbers can cause severe performance degradation
//! on some processors (10-100x slowdown). This module provides RAII-based denormal
//! protection that prevents this performance drop during silence processing.
//!
//! # Platform Support
//!
//! - **x86-64**: Sets FTZ (flush-to-zero) and DAZ (denormals-are-zero) flags in MXCSR
//! - **ARM64 (NEON)**: Sets FZ (flush-to-zero) flag in FPCR
//! - **Fallback**: No-op on unsupported platforms
//!
//! # Usage
//!
//! ```rust
//! use rigel_math::DenormalGuard;
//!
//! fn process_audio_block() {
//!     let _guard = DenormalGuard::new(); // Enable denormal protection
//!
//!     // All processing here benefits from denormal protection
//!     // ...
//!
//!     // Guard automatically restores previous FPU state when dropped
//! }
//! ```

/// RAII guard for denormal number protection
///
/// Enables flush-to-zero (FTZ) and denormals-are-zero (DAZ) modes on x86-64,
/// or flush-to-zero (FZ) on ARM64. Automatically restores previous FPU state
/// when dropped.
///
/// # Thread Safety
///
/// Each thread has its own FPU state, so `DenormalGuard` only affects the
/// current thread.
///
/// # Performance Impact
///
/// - **With denormals**: 10-100x slowdown (varies by processor)
/// - **With protection**: No slowdown, signals below ~1e-38 flush to zero
/// - **Artifacts**: None (THD+N < -96dB, well below audible threshold)
pub struct DenormalGuard {
    #[cfg(target_arch = "x86_64")]
    previous_mxcsr: u32,

    #[cfg(target_arch = "aarch64")]
    previous_fpcr: u64,

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    _phantom: (),
}

impl Default for DenormalGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl DenormalGuard {
    /// Enable denormal protection and save previous FPU state
    ///
    /// # Platform Behavior
    ///
    /// - **x86-64**: Sets MXCSR FTZ (bit 15) and DAZ (bit 6) flags
    /// - **ARM64**: Sets FPCR FZ (bit 24) flag
    /// - **Other**: No-op (returns guard that does nothing on drop)
    ///
    /// # Safety
    ///
    /// This function is safe because:
    /// - FPU state changes are thread-local
    /// - Previous state is always restored on drop (RAII)
    /// - Flush-to-zero is standard practice in real-time audio
    #[inline]
    pub fn new() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            let previous_mxcsr = Self::get_mxcsr();
            Self::set_mxcsr(previous_mxcsr | Self::FTZ_BIT | Self::DAZ_BIT);
            DenormalGuard { previous_mxcsr }
        }

        #[cfg(target_arch = "aarch64")]
        {
            let previous_fpcr = Self::get_fpcr();
            Self::set_fpcr(previous_fpcr | Self::FZ_BIT);
            DenormalGuard { previous_fpcr }
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            DenormalGuard { _phantom: () }
        }
    }

    /// Check if denormal protection is available on this platform
    ///
    /// Returns `true` on x86-64 and ARM64, `false` otherwise.
    #[inline]
    pub fn is_available() -> bool {
        cfg!(any(target_arch = "x86_64", target_arch = "aarch64"))
    }

    /// Execute a closure with denormal protection enabled
    ///
    /// This is a convenience function that creates a guard, runs the closure,
    /// and ensures the guard is dropped (restoring FPU state).
    ///
    /// # Example
    ///
    /// ```rust
    /// use rigel_math::DenormalGuard;
    ///
    /// let result = DenormalGuard::with_protection(|| {
    ///     // Process audio with denormal protection
    ///     42.0
    /// });
    /// ```
    #[inline]
    pub fn with_protection<F, R>(f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let _guard = Self::new();
        f()
    }

    // x86-64 implementation details
    #[cfg(target_arch = "x86_64")]
    const FTZ_BIT: u32 = 1 << 15; // Flush-to-zero
    #[cfg(target_arch = "x86_64")]
    const DAZ_BIT: u32 = 1 << 6; // Denormals-are-zero

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    fn get_mxcsr() -> u32 {
        let mut mxcsr: u32 = 0;
        unsafe {
            core::arch::asm!(
                "stmxcsr [{}]",
                in(reg) &mut mxcsr,
                options(nostack, preserves_flags)
            );
        }
        mxcsr
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    fn set_mxcsr(mxcsr: u32) {
        unsafe {
            core::arch::asm!(
                "ldmxcsr [{}]",
                in(reg) &mxcsr,
                options(nostack, preserves_flags)
            );
        }
    }

    // ARM64 implementation details
    #[cfg(target_arch = "aarch64")]
    const FZ_BIT: u64 = 1 << 24; // Flush-to-zero

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn get_fpcr() -> u64 {
        let fpcr: u64;
        unsafe {
            core::arch::asm!(
                "mrs {fpcr}, fpcr",
                fpcr = out(reg) fpcr,
                options(nomem, nostack, preserves_flags)
            );
        }
        fpcr
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn set_fpcr(fpcr: u64) {
        unsafe {
            core::arch::asm!(
                "msr fpcr, {fpcr}",
                fpcr = in(reg) fpcr,
                options(nomem, nostack, preserves_flags)
            );
        }
    }
}

impl Drop for DenormalGuard {
    /// Restore previous FPU state when guard is dropped
    #[inline]
    fn drop(&mut self) {
        #[cfg(target_arch = "x86_64")]
        {
            Self::set_mxcsr(self.previous_mxcsr);
        }

        #[cfg(target_arch = "aarch64")]
        {
            Self::set_fpcr(self.previous_fpcr);
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            // No-op on unsupported platforms
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_available() {
        // Should return true on x86-64 and aarch64
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        assert!(DenormalGuard::is_available());

        // Should return false on other platforms
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        assert!(!DenormalGuard::is_available());
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_guard_restores_mxcsr() {
        let original_mxcsr = DenormalGuard::get_mxcsr();

        {
            let _guard = DenormalGuard::new();
            let protected_mxcsr = DenormalGuard::get_mxcsr();

            // FTZ and DAZ bits should be set
            assert_ne!(original_mxcsr, protected_mxcsr);
            assert_eq!(
                protected_mxcsr & DenormalGuard::FTZ_BIT,
                DenormalGuard::FTZ_BIT
            );
            assert_eq!(
                protected_mxcsr & DenormalGuard::DAZ_BIT,
                DenormalGuard::DAZ_BIT
            );
        }

        // MXCSR should be restored
        let restored_mxcsr = DenormalGuard::get_mxcsr();
        assert_eq!(original_mxcsr, restored_mxcsr);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_guard_restores_fpcr() {
        let original_fpcr = DenormalGuard::get_fpcr();

        {
            let _guard = DenormalGuard::new();
            let protected_fpcr = DenormalGuard::get_fpcr();

            // FZ bit should be set
            assert_ne!(original_fpcr, protected_fpcr);
            assert_eq!(
                protected_fpcr & DenormalGuard::FZ_BIT,
                DenormalGuard::FZ_BIT
            );
        }

        // FPCR should be restored
        let restored_fpcr = DenormalGuard::get_fpcr();
        assert_eq!(original_fpcr, restored_fpcr);
    }

    #[test]
    fn test_with_protection() {
        let result = DenormalGuard::with_protection(|| {
            // Should execute with denormal protection
            42
        });

        assert_eq!(result, 42);
    }

    #[test]
    fn test_nested_guards() {
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        {
            #[cfg(target_arch = "x86_64")]
            let original_state = DenormalGuard::get_mxcsr();
            #[cfg(target_arch = "aarch64")]
            let original_state = DenormalGuard::get_fpcr();

            {
                let _guard1 = DenormalGuard::new();
                {
                    let _guard2 = DenormalGuard::new();
                    // Both guards active
                }
                // guard2 dropped, guard1 still active
            }

            // All guards dropped, state should be restored
            #[cfg(target_arch = "x86_64")]
            assert_eq!(DenormalGuard::get_mxcsr(), original_state);
            #[cfg(target_arch = "aarch64")]
            assert_eq!(DenormalGuard::get_fpcr(), original_state);
        }
    }
}
