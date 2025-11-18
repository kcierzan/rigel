# Research: Fast DSP Math Library

**Feature**: 001-fast-dsp-math
**Date**: 2025-11-18
**Status**: Complete

## Overview

This document captures technical research decisions made during the planning phase for the rigel-math SIMD abstraction library.

---

## Decision: Property-Based Testing Framework

### Decision
**proptest**

### Rationale

1. **Superior Strategy Composition for SIMD Edge Cases**: Proptest's explicit `Strategy` trait enables multiple, composable strategies per type—critical for SIMD testing where you need to generate distinct test cases for normal values, denormals, NaN, infinity, and boundary conditions. QuickCheck's single-strategy-per-type limitation would require newtype wrappers for each floating-point variant, creating significant boilerplate. Proptest provides built-in strategies (`NORMAL`, `SUBNORMAL`, `INFINITE`, `QUIET_NAN`, `SIGNALING_NAN`, `ZERO`, `POSITIVE`, `NEGATIVE`) that you can directly compose using `prop_oneof!` and bitwise OR operators (e.g., `POSITIVE | INFINITE`).

2. **Specialized Floating-Point Support for IEEE 754 Invariants**: Proptest's `proptest::num::f32` and `proptest::num::f64` modules provide explicit handling of floating-point edge cases that are essential for testing mathematical invariants (commutativity, associativity, error bounds). The framework's `BinarySearch` shrinking specifically targets floats by minimizing toward zero, making it ideal for isolating failures in fast math kernels. Proptest's awareness of IEEE 754 constraints (subnormal ranges, NaN bit patterns) prevents generating or shrinking to invalid values, reducing flaky test behavior.

3. **Deterministic Reproducibility with Seed Control**: Proptest persists failing test cases with seed values, enabling reproducible debugging across SIMD backends and architectures. While QuickCheck supports replay functionality (in Haskell primarily), Rust's proptest provides seamless seed integration with its test runner and property macro system, making it easier to reproduce failures discovered during CI runs on different platforms.

### Alternatives Considered

**QuickCheck** is simpler to get started with and has lower generation overhead, making it suitable for basic testing scenarios. However, its single-strategy-per-type design is fundamentally misaligned with SIMD testing requirements. Supporting multiple floating-point classes (normal, denormal, infinite, NaN) would require creating newtype wrappers (`struct Normal(f32)`, `struct Denormal(f32)`, etc.) and manually implementing `Arbitrary` for each, introducing complexity that negates the claimed simplicity advantage. QuickCheck's reliance on the `Arbitrary` trait also makes it harder to express constraints (e.g., "generate positive floats near zero without generating subnormals"), leading to higher rejection rates and slower test execution. Additionally, quickcheck has been abandoned for 4.8+ years with no recent releases, while proptest (though in passive maintenance) continues to see updates and remains the standard in the Rust ecosystem.

### Implementation Notes

1. **Strategy Design Pattern**: Structure your SIMD math library tests using proptest's strategy composition:
   ```rust
   use proptest::num::f32;
   use proptest::prelude::*;

   proptest! {
       #[test]
       fn test_simd_add_commutativity(
           a in f32::NORMAL | f32::SUBNORMAL,
           b in f32::NORMAL | f32::SUBNORMAL,
       ) {
           // Test a + b == b + a across trait implementations
       }
   }
   ```

2. **Multi-Backend Testing**: Leverage proptest's deterministic seeds to test invariants across different SIMD backends (scalar fallback, AVX2, AVX512, NEON):
   ```rust
   proptest! {
       #![proptest_config(ProptestConfig::with_cases(10000))]
       #[test]
       fn test_across_backends(values in prop::collection::vec(f32::ANY, 1..=256)) {
           assert_eq!(scalar_impl(&values), simd_impl(&values));
       }
   }
   ```

3. **Edge Case Coverage**: Combine proptest's built-in float constants with custom strategies:
   ```rust
   let edge_cases = prop_oneof![
       f32::ZERO,
       f32::SUBNORMAL,
       f32::NORMAL,
       f32::INFINITE,
       f32::QUIET_NAN,
   ];
   ```

4. **Performance at Scale**: Configure proptest for 10,000+ test cases per operation:
   - Set `PROPTEST_CASES=10000` environment variable or use `ProptestConfig::with_cases(10000)`
   - Be aware that shrinking is limited to 4x the case count by default (40,000 iterations max)
   - Use `-j 1` flag with test runner to avoid parallel shrinking contention
   - Consider forking implications if running on resource-constrained hardware

5. **Shrinking for Floating-Point Failures**: Proptest's binary search shrinking automatically minimizes floating-point failures toward zero, which is ideal for isolating boundary conditions in fast math kernels. However, for some math properties (e.g., range preservation), you may need custom shrink hints via `prop_map_into`.

6. **Deterministic Test Data**: Always use seeds in CI pipelines and document failure-inducing seeds:
   ```rust
   // Proptest persists failures automatically to `proptest-regressions/`
   // directory—commit this to git for reproducible CI failures
   ```

### References

- **Official Proptest Documentation**: https://docs.rs/proptest/latest/proptest/
- **Proptest Book (Strategy Guide)**: https://altsysrq.github.io/proptest-book/
- **Proptest vs QuickCheck Comparison**: https://altsysrq.github.io/proptest-book/proptest/vs-quickcheck.html
- **QuickCheck (Rust)**: https://github.com/BurntSushi/quickcheck
- **Proptest GitHub**: https://github.com/proptest-rs/proptest
- **Floating-Point Strategies**: https://docs.rs/proptest/latest/proptest/num/f64/index.html
- **Strategy Composition Guide**: https://altsysrq.github.io/proptest-book/proptest/tutorial/strategy-basics.html
- **Arbitrary Trait Documentation**: https://docs.rs/proptest/latest/proptest/arbitrary/trait.Arbitrary.html
- **Configuring PropTest for Large Test Suites**: https://altsysrq.github.io/proptest-book/proptest/tutorial/config.html

---

## Summary

For a SIMD math library testing IEEE 754 invariants with thousands of test cases, **proptest is the clear choice**. Its strategy-based architecture directly addresses the complexity of generating multiple categories of floating-point edge cases, providing superior composability and constraint awareness. While slightly slower than QuickCheck for simple cases, the performance overhead is negligible at 10,000 test cases and is offset by deterministic reproducibility, professional-grade shrinking, and active maintenance within the Rust ecosystem.
