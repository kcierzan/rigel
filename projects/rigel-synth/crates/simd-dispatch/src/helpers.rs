//! Generic helper functions for bridging slice-based backend operations to vector-based math kernels
//!
//! This module provides the core chunking logic that enables SIMD backends to leverage
//! the optimized math module implementations without code duplication.

use crate::traits::SimdVector;

/// Process a unary operation on slices using a SIMD vector kernel
///
/// This is the core helper that bridges slice-based backend APIs to vector-based math kernels.
/// It handles:
/// - Chunking input slice into SIMD vectors
/// - Calling the kernel function on each chunk
/// - Storing results back to output slice
/// - Processing remainder with scalar fallback
///
/// # Type Parameters
///
/// - `V`: The SIMD vector type (ScalarVector, Avx2Vector, etc.)
/// - `F`: The kernel function type (e.g., `fn(V) -> V`)
///
/// # Arguments
///
/// - `input`: Input slice to process
/// - `output`: Output slice for results (must be same length as input)
/// - `kernel`: SIMD kernel function from the math module
/// - `scalar_fallback`: Fallback function for remainder samples (uses libm)
///
/// # Example
///
/// ```rust,ignore
/// use rigel_simd::simd::exp;
/// use rigel_simd::backends::ScalarVector;
/// use rigel_simd_dispatch::helpers::process_unary;
///
/// fn exp_impl(input: &[f32], output: &mut [f32]) {
///     process_unary::<ScalarVector<f32>, _>(
///         input,
///         output,
///         exp,
///         libm::expf
///     );
/// }
/// ```
#[inline]
pub fn process_unary<V, F>(
    input: &[f32],
    output: &mut [f32],
    kernel: F,
    scalar_fallback: fn(f32) -> f32,
) where
    V: SimdVector<Scalar = f32>,
    F: Fn(V) -> V,
{
    assert_eq!(
        input.len(),
        output.len(),
        "Input and output slices must have the same length"
    );

    let lanes = V::LANES;
    let len = input.len();

    // Process full SIMD chunks
    let chunks = len / lanes;
    let remainder = len % lanes;

    for i in 0..chunks {
        let start = i * lanes;
        let end = start + lanes;

        // Load vector from input
        let vec = V::from_slice(&input[start..end]);

        // Apply kernel
        let result = kernel(vec);

        // Store to output
        result.to_slice(&mut output[start..end]);
    }

    // Process remainder with scalar fallback
    let remainder_start = chunks * lanes;
    for i in 0..remainder {
        output[remainder_start + i] = scalar_fallback(input[remainder_start + i]);
    }
}

/// Process a binary operation on slices using a SIMD vector kernel
///
/// Similar to `process_unary` but for binary operations like `add`, `mul`, `pow`, etc.
///
/// # Type Parameters
///
/// - `V`: The SIMD vector type
/// - `F`: The kernel function type (e.g., `fn(V, V) -> V`)
///
/// # Arguments
///
/// - `a`: First input slice
/// - `b`: Second input slice
/// - `output`: Output slice for results (must be same length as inputs)
/// - `kernel`: SIMD kernel function from the math module
/// - `scalar_fallback`: Fallback function for remainder samples
///
/// # Example
///
/// ```rust,ignore
/// use rigel_simd::ops::add;
/// use rigel_simd::backends::Avx2Vector;
/// use rigel_simd::simd::helpers::process_binary;
///
/// fn add_impl(a: &[f32], b: &[f32], output: &mut [f32]) {
///     process_binary::<Avx2Vector, _>(
///         a,
///         b,
///         output,
///         add,
///         |x, y| x + y
///     );
/// }
/// ```
#[inline]
pub fn process_binary<V, F>(
    a: &[f32],
    b: &[f32],
    output: &mut [f32],
    kernel: F,
    scalar_fallback: fn(f32, f32) -> f32,
) where
    V: SimdVector<Scalar = f32>,
    F: Fn(V, V) -> V,
{
    assert_eq!(a.len(), b.len(), "Input slices must have the same length");
    assert_eq!(
        a.len(),
        output.len(),
        "Input and output slices must have the same length"
    );

    let lanes = V::LANES;
    let len = a.len();

    // Process full SIMD chunks
    let chunks = len / lanes;
    let remainder = len % lanes;

    for i in 0..chunks {
        let start = i * lanes;
        let end = start + lanes;

        // Load vectors from inputs
        let vec_a = V::from_slice(&a[start..end]);
        let vec_b = V::from_slice(&b[start..end]);

        // Apply kernel
        let result = kernel(vec_a, vec_b);

        // Store to output
        result.to_slice(&mut output[start..end]);
    }

    // Process remainder with scalar fallback
    let remainder_start = chunks * lanes;
    for i in 0..remainder {
        output[remainder_start + i] =
            scalar_fallback(a[remainder_start + i], b[remainder_start + i]);
    }
}

/// Process a ternary operation on slices using a SIMD vector kernel
///
/// Similar to `process_unary` and `process_binary` but for ternary operations like `fma` (fused multiply-add).
///
/// # Type Parameters
///
/// - `V`: The SIMD vector type
/// - `F`: The kernel function type (e.g., `fn(V, V, V) -> V`)
///
/// # Arguments
///
/// - `a`: First input slice
/// - `b`: Second input slice
/// - `c`: Third input slice
/// - `output`: Output slice for results (must be same length as inputs)
/// - `kernel`: SIMD kernel function from the math module
/// - `scalar_fallback`: Fallback function for remainder samples
///
/// # Example
///
/// ```rust,ignore
/// use rigel_simd::backends::ScalarVector;
/// use rigel_simd::simd::helpers::process_ternary;
///
/// fn fma_impl(a: &[f32], b: &[f32], c: &[f32], output: &mut [f32]) {
///     process_ternary::<ScalarVector<f32>, _>(
///         a,
///         b,
///         c,
///         output,
///         |x, y, z| x.fma(y, z),
///         |x, y, z| libm::fmaf(x, y, z)
///     );
/// }
/// ```
#[inline]
pub fn process_ternary<V, F>(
    a: &[f32],
    b: &[f32],
    c: &[f32],
    output: &mut [f32],
    kernel: F,
    scalar_fallback: fn(f32, f32, f32) -> f32,
) where
    V: SimdVector<Scalar = f32>,
    F: Fn(V, V, V) -> V,
{
    assert_eq!(a.len(), b.len(), "Input slices must have the same length");
    assert_eq!(a.len(), c.len(), "Input slices must have the same length");
    assert_eq!(
        a.len(),
        output.len(),
        "Input and output slices must have the same length"
    );

    let lanes = V::LANES;
    let len = a.len();

    // Process full SIMD chunks
    let chunks = len / lanes;
    let remainder = len % lanes;

    for i in 0..chunks {
        let start = i * lanes;
        let end = start + lanes;

        // Load vectors from inputs
        let vec_a = V::from_slice(&a[start..end]);
        let vec_b = V::from_slice(&b[start..end]);
        let vec_c = V::from_slice(&c[start..end]);

        // Apply kernel
        let result = kernel(vec_a, vec_b, vec_c);

        // Store to output
        result.to_slice(&mut output[start..end]);
    }

    // Process remainder with scalar fallback
    let remainder_start = chunks * lanes;
    for i in 0..remainder {
        output[remainder_start + i] = scalar_fallback(
            a[remainder_start + i],
            b[remainder_start + i],
            c[remainder_start + i],
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd;
    use crate::ScalarVector;

    #[test]
    fn test_process_unary_exact_chunks() {
        // Test with input that's exactly one SIMD chunk (1 lane for scalar)
        let input = [2.0_f32];
        let mut output = [0.0_f32; 1];

        process_unary::<ScalarVector<f32>, _>(&input, &mut output, simd::exp::exp, libm::expf);

        // exp(2) ≈ 7.389
        assert!((output[0] - 7.389).abs() < 0.01);
    }

    #[test]
    fn test_process_unary_with_remainder() {
        // Test with input that has remainder samples
        // For ScalarVector (1 lane), any length > 1 will have remainder
        let input = [1.0, 2.0, 3.0];
        let mut output = [0.0; 3];

        process_unary::<ScalarVector<f32>, _>(&input, &mut output, simd::exp::exp, libm::expf);

        // Check all values are processed
        assert!(output[0] > 0.0);
        assert!(output[1] > 0.0);
        assert!(output[2] > 0.0);
    }

    #[test]
    fn test_process_binary_basic() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let mut output = [0.0; 3];

        process_binary::<ScalarVector<f32>, _>(&a, &b, &mut output, |x, y| x.add(y), |x, y| x + y);

        assert_eq!(output[0], 5.0);
        assert_eq!(output[1], 7.0);
        assert_eq!(output[2], 9.0);
    }

    #[test]
    fn test_process_ternary_fma() {
        let a = [2.0, 3.0];
        let b = [4.0, 5.0];
        let c = [1.0, 2.0];
        let mut output = [0.0; 2];

        process_ternary::<ScalarVector<f32>, _>(
            &a,
            &b,
            &c,
            &mut output,
            |x, y, z| x.fma(y, z),
            |x, y, z| x * y + z,
        );

        // fma(2, 4, 1) = 2*4 + 1 = 9
        assert_eq!(output[0], 9.0);
        // fma(3, 5, 2) = 3*5 + 2 = 17
        assert_eq!(output[1], 17.0);
    }

    #[test]
    #[should_panic(expected = "Input and output slices must have the same length")]
    fn test_process_unary_length_mismatch() {
        let input = [1.0, 2.0];
        let mut output = [0.0; 3]; // Wrong length

        process_unary::<ScalarVector<f32>, _>(&input, &mut output, simd::exp::exp, libm::expf);
    }
}
