//! Core SIMD abstraction traits
//!
//! This module defines the fundamental traits that all SIMD backends must implement.
//! These traits enable writing platform-agnostic DSP code that compiles to optimal
//! SIMD instructions for each target architecture.

/// Core SIMD vector abstraction trait
///
/// All SIMD backends (scalar, AVX2, AVX512, NEON) implement this trait,
/// enabling zero-cost abstraction for vectorized operations.
///
/// # Type Parameters
///
/// - `Scalar`: The underlying scalar type (f32 or f64)
/// - `Mask`: Associated mask type for comparison results
///
/// # Example
///
/// ```rust
/// use rigel_math::{DefaultSimdVector, SimdVector};
///
/// let a = DefaultSimdVector::splat(2.0);
/// let b = DefaultSimdVector::splat(3.0);
/// let result = a.add(b);
/// assert_eq!(result.horizontal_sum(), 5.0 * DefaultSimdVector::LANES as f32);
/// ```
pub trait SimdVector: Copy + Clone + Sized {
    /// The underlying scalar type (f32 or f64)
    type Scalar: Copy;

    /// Associated mask type for comparison operations
    type Mask: SimdMask;

    /// Number of SIMD lanes (1 for scalar, 4 for NEON, 8 for AVX2, 16 for AVX512)
    const LANES: usize;

    // Construction

    /// Broadcast a scalar value to all SIMD lanes
    ///
    /// # Example
    ///
    /// ```rust
    /// use rigel_math::{DefaultSimdVector, SimdVector};
    /// let vec = DefaultSimdVector::splat(2.0);
    /// assert_eq!(vec.horizontal_sum(), 2.0 * DefaultSimdVector::LANES as f32);
    /// ```
    fn splat(value: Self::Scalar) -> Self;

    /// Load from a slice (must have at least LANES elements)
    ///
    /// # Panics
    ///
    /// Panics if slice has fewer than LANES elements
    fn from_slice(slice: &[Self::Scalar]) -> Self;

    /// Store to a slice (must have at least LANES elements)
    ///
    /// # Panics
    ///
    /// Panics if slice has fewer than LANES elements
    fn to_slice(self, slice: &mut [Self::Scalar]);

    // Arithmetic operations

    /// Element-wise addition
    fn add(self, rhs: Self) -> Self;

    /// Element-wise subtraction
    fn sub(self, rhs: Self) -> Self;

    /// Element-wise multiplication
    fn mul(self, rhs: Self) -> Self;

    /// Element-wise division
    fn div(self, rhs: Self) -> Self;

    /// Element-wise negation
    fn neg(self) -> Self;

    /// Element-wise absolute value
    fn abs(self) -> Self;

    // Fused multiply-add

    /// Fused multiply-add: self * b + c
    ///
    /// On supporting backends (AVX2, AVX512, NEON), this compiles to a single FMA instruction.
    ///
    /// # Example
    ///
    /// ```rust
    /// use rigel_math::{DefaultSimdVector, SimdVector};
    /// let a = DefaultSimdVector::splat(2.0);
    /// let b = DefaultSimdVector::splat(3.0);
    /// let c = DefaultSimdVector::splat(1.0);
    /// let result = a.fma(b, c); // 2 * 3 + 1 = 7
    /// assert_eq!(result.horizontal_sum(), 7.0 * DefaultSimdVector::LANES as f32);
    /// ```
    fn fma(self, b: Self, c: Self) -> Self;

    // Min/Max operations

    /// Element-wise minimum
    fn min(self, rhs: Self) -> Self;

    /// Element-wise maximum
    fn max(self, rhs: Self) -> Self;

    // Comparison operations (return masks)

    /// Element-wise less-than comparison
    ///
    /// Returns a mask where each lane is set if self[i] < rhs[i]
    fn lt(self, rhs: Self) -> Self::Mask;

    /// Element-wise greater-than comparison
    ///
    /// Returns a mask where each lane is set if self[i] > rhs[i]
    fn gt(self, rhs: Self) -> Self::Mask;

    /// Element-wise equality comparison
    ///
    /// Returns a mask where each lane is set if self[i] == rhs[i]
    fn eq(self, rhs: Self) -> Self::Mask;

    // Blending

    /// Select values based on mask
    ///
    /// For each lane: mask[i] ? true_val[i] : false_val[i]
    ///
    /// # Example
    ///
    /// ```rust
    /// use rigel_math::{DefaultSimdVector, SimdVector};
    /// let a = DefaultSimdVector::splat(1.0);
    /// let b = DefaultSimdVector::splat(2.0);
    /// let mask = a.lt(b); // All lanes true
    /// let result = DefaultSimdVector::select(mask, a, b);
    /// assert_eq!(result.horizontal_sum(), 1.0 * DefaultSimdVector::LANES as f32);
    /// ```
    fn select(mask: Self::Mask, true_val: Self, false_val: Self) -> Self;

    // Horizontal operations

    /// Sum all SIMD lanes into a scalar
    ///
    /// # Example
    ///
    /// ```rust
    /// use rigel_math::{DefaultSimdVector, SimdVector};
    /// let vec = DefaultSimdVector::splat(2.0);
    /// assert_eq!(vec.horizontal_sum(), 2.0 * DefaultSimdVector::LANES as f32);
    /// ```
    fn horizontal_sum(self) -> Self::Scalar;

    /// Maximum value across all SIMD lanes
    fn horizontal_max(self) -> Self::Scalar;

    /// Minimum value across all SIMD lanes
    fn horizontal_min(self) -> Self::Scalar;
}

/// Mask type for conditional SIMD operations
///
/// Masks represent per-lane boolean values, enabling branchless conditional logic.
///
/// # Example
///
/// ```rust
/// use rigel_math::{DefaultSimdVector, SimdVector, SimdMask};
/// let a = DefaultSimdVector::splat(1.0);
/// let b = DefaultSimdVector::splat(2.0);
/// let mask = a.lt(b);
/// assert!(mask.all()); // All lanes are true (1.0 < 2.0)
/// ```
pub trait SimdMask: Copy + Clone + Sized {
    /// Returns true if all lanes are set
    fn all(self) -> bool;

    /// Returns true if any lane is set
    fn any(self) -> bool;

    /// Returns true if no lanes are set
    fn none(self) -> bool;

    /// Bitwise AND of two masks
    fn and(self, rhs: Self) -> Self;

    /// Bitwise OR of two masks
    fn or(self, rhs: Self) -> Self;

    /// Bitwise NOT of mask
    fn not(self) -> Self;

    /// Bitwise XOR of two masks
    fn xor(self, rhs: Self) -> Self;
}
