//! Block processing with fixed-size aligned buffers (T030-T035)
//!
//! Provides standardized block processing with fixed sizes (64/128 samples) and clear SIMD lane packing conventions.
//! This enables efficient SIMD-friendly memory access patterns with guaranteed alignment.

use crate::traits::SimdVector;
use core::marker::PhantomData;

/// Fixed-size audio block with alignment for SIMD operations
///
/// # Memory Layout and Lane Packing Conventions (T035)
///
/// Audio samples are stored sequentially in memory and processed in SIMD chunks:
///
/// ```text
/// Block64 with AVX2 (8 lanes):
/// [0  1  2  3  4  5  6  7] [8  9  10 11 12 13 14 15] ... [56 57 58 59 60 61 62 63]
///  └──────── chunk 0 ─────┘ └──────── chunk 1 ──────┘     └──────── chunk 7 ──────┘
/// ```
///
/// **Alignment Requirements:**
/// - **AVX2**: 32-byte alignment (8 f32 values)
/// - **AVX512**: 64-byte alignment (16 f32 values)
/// - **NEON**: 16-byte alignment (4 f32 values)
/// - **Scalar**: No special alignment required
///
/// **Lane Packing:**
/// - Samples are stored in **time-sequential order** (not interleaved)
/// - Each SIMD vector processes N consecutive samples
/// - No special lane ordering - natural sequential access
///
/// # Example
///
/// ```rust
/// use rigel_math::{Block64, DefaultSimdVector, SimdVector};
///
/// let mut block = Block64::new();
///
/// // Fill with test signal
/// for i in 0..64 {
///     block[i] = (i as f32) / 64.0;
/// }
///
/// // Process in SIMD chunks
/// let gain = DefaultSimdVector::splat(0.5);
/// for mut chunk in block.as_chunks_mut::<DefaultSimdVector>().iter_mut() {
///     let value = chunk.load();
///     chunk.store(value.mul(gain));
/// }
/// ```
#[repr(C, align(64))] // 64-byte alignment supports AVX-512
pub struct AudioBlock<T, const N: usize> {
    data: [T; N],
}

impl<T: Copy + Default, const N: usize> AudioBlock<T, N> {
    /// Create a new audio block filled with default values (T030, T031)
    ///
    /// # Example
    ///
    /// ```rust
    /// use rigel_math::Block64;
    ///
    /// let block = Block64::new();
    /// assert_eq!(block.len(), 64);
    /// ```
    #[inline]
    pub fn new() -> Self {
        Self {
            data: [T::default(); N],
        }
    }

    /// Create an audio block from a slice (T031)
    ///
    /// # Panics
    ///
    /// Panics if slice length doesn't match block size N
    ///
    /// # Example
    ///
    /// ```rust
    /// use rigel_math::Block64;
    ///
    /// let samples = [0.0f32; 64];
    /// let block = Block64::from_slice(&samples);
    /// ```
    #[inline]
    pub fn from_slice(slice: &[T]) -> Self {
        assert_eq!(
            slice.len(),
            N,
            "Slice length {} doesn't match block size {}",
            slice.len(),
            N
        );
        let mut data = [T::default(); N];
        data.copy_from_slice(slice);
        Self { data }
    }

    /// Returns the number of samples in this block
    #[inline]
    pub const fn len(&self) -> usize {
        N
    }

    /// Returns true if the block is empty (always false for const-sized blocks)
    #[inline]
    pub const fn is_empty(&self) -> bool {
        N == 0
    }

    /// Returns a slice view of the entire block
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Returns a mutable slice view of the entire block
    #[inline]
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Get immutable SIMD view of the block (T032)
    ///
    /// Returns an iterator-like structure that allows SIMD-chunked access.
    /// The block is divided into chunks matching the SIMD vector lane count.
    ///
    /// # Type Parameters
    ///
    /// - `V`: The SIMD vector type (e.g., `DefaultSimdVector`, `Avx2Vector`)
    ///
    /// # Panics
    ///
    /// Panics if block size N is not evenly divisible by the SIMD lane count.
    ///
    /// # Example
    ///
    /// ```rust
    /// use rigel_math::{Block64, DefaultSimdVector, SimdVector};
    ///
    /// let block = Block64::new();
    /// let chunks = block.as_chunks::<DefaultSimdVector>();
    ///
    /// for chunk in chunks.iter() {
    ///     let sum = chunk.horizontal_sum();
    ///     // Process each SIMD chunk...
    /// }
    /// ```
    #[inline]
    pub fn as_chunks<V: SimdVector<Scalar = T>>(&self) -> SimdChunks<'_, T, V, N> {
        assert_eq!(
            N % V::LANES,
            0,
            "Block size {} must be divisible by SIMD lane count {}",
            N,
            V::LANES
        );
        SimdChunks {
            data: &self.data,
            _phantom: PhantomData,
        }
    }

    /// Get mutable SIMD view of the block (T033)
    ///
    /// Returns an iterator-like structure that allows mutable SIMD-chunked access.
    ///
    /// # Example
    ///
    /// ```rust
    /// use rigel_math::{Block64, DefaultSimdVector, SimdVector};
    ///
    /// let mut block = Block64::new();
    /// let gain = DefaultSimdVector::splat(0.5);
    ///
    /// for mut chunk in block.as_chunks_mut::<DefaultSimdVector>().iter_mut() {
    ///     let value = chunk.load();
    ///     chunk.store(value.mul(gain));
    /// }
    /// ```
    #[inline]
    pub fn as_chunks_mut<V: SimdVector<Scalar = T>>(&mut self) -> SimdChunksMut<'_, T, V, N> {
        assert_eq!(
            N % V::LANES,
            0,
            "Block size {} must be divisible by SIMD lane count {}",
            N,
            V::LANES
        );
        SimdChunksMut {
            data: &mut self.data,
            _phantom: PhantomData,
        }
    }
}

impl<T: Copy, const N: usize> Default for AudioBlock<T, N>
where
    T: Default,
{
    fn default() -> Self {
        Self::new()
    }
}

// Index access
impl<T, const N: usize> core::ops::Index<usize> for AudioBlock<T, N> {
    type Output = T;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<T, const N: usize> core::ops::IndexMut<usize> for AudioBlock<T, N> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

/// Immutable SIMD chunk view
pub struct SimdChunks<'a, T, V: SimdVector<Scalar = T>, const N: usize> {
    data: &'a [T; N],
    _phantom: PhantomData<V>,
}

impl<'a, T, V: SimdVector<Scalar = T>, const N: usize> SimdChunks<'a, T, V, N> {
    /// Get number of SIMD chunks
    #[inline]
    pub const fn len(&self) -> usize {
        N / V::LANES
    }

    /// Returns true if there are no chunks
    #[inline]
    pub const fn is_empty(&self) -> bool {
        N == 0
    }

    /// Iterate over SIMD chunks
    #[inline]
    pub fn iter(&self) -> SimdChunksIter<'_, T, V> {
        SimdChunksIter {
            data: self.data,
            index: 0,
            _phantom: PhantomData,
        }
    }
}

/// Iterator over immutable SIMD chunks
pub struct SimdChunksIter<'a, T, V: SimdVector<Scalar = T>> {
    data: &'a [T],
    index: usize,
    _phantom: PhantomData<V>,
}

impl<'a, T, V: SimdVector<Scalar = T>> Iterator for SimdChunksIter<'a, T, V> {
    type Item = V;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.index + V::LANES <= self.data.len() {
            let chunk = V::from_slice(&self.data[self.index..]);
            self.index += V::LANES;
            Some(chunk)
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = (self.data.len() - self.index) / V::LANES;
        (remaining, Some(remaining))
    }
}

impl<'a, T, V: SimdVector<Scalar = T>> ExactSizeIterator for SimdChunksIter<'a, T, V> {}

/// Mutable SIMD chunk view
pub struct SimdChunksMut<'a, T, V: SimdVector<Scalar = T>, const N: usize> {
    data: &'a mut [T; N],
    _phantom: PhantomData<V>,
}

impl<'a, T, V: SimdVector<Scalar = T>, const N: usize> SimdChunksMut<'a, T, V, N> {
    /// Get number of SIMD chunks
    #[inline]
    pub const fn len(&self) -> usize {
        N / V::LANES
    }

    /// Returns true if there are no chunks
    #[inline]
    pub const fn is_empty(&self) -> bool {
        N == 0
    }

    /// Iterate over mutable SIMD chunks
    #[inline]
    pub fn iter_mut(&mut self) -> SimdChunksMutIter<'_, T, V> {
        SimdChunksMutIter {
            data: self.data,
            index: 0,
            _phantom: PhantomData,
        }
    }
}

/// Iterator over mutable SIMD chunks
pub struct SimdChunksMutIter<'a, T, V: SimdVector<Scalar = T>> {
    data: &'a mut [T],
    index: usize,
    _phantom: PhantomData<V>,
}

impl<'a, T, V: SimdVector<Scalar = T>> Iterator for SimdChunksMutIter<'a, T, V> {
    type Item = SimdChunkMut<'a, T, V>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.index + V::LANES <= self.data.len() {
            // SAFETY: We're splitting the slice into non-overlapping mutable chunks
            let ptr = self.data.as_mut_ptr();
            let chunk_slice =
                unsafe { core::slice::from_raw_parts_mut(ptr.add(self.index), V::LANES) };
            self.index += V::LANES;
            Some(SimdChunkMut {
                slice: chunk_slice,
                _phantom: PhantomData,
            })
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = (self.data.len() - self.index) / V::LANES;
        (remaining, Some(remaining))
    }
}

impl<'a, T, V: SimdVector<Scalar = T>> ExactSizeIterator for SimdChunksMutIter<'a, T, V> {}

/// Mutable reference to a SIMD chunk
pub struct SimdChunkMut<'a, T, V: SimdVector<Scalar = T>> {
    slice: &'a mut [T],
    _phantom: PhantomData<V>,
}

impl<'a, T, V: SimdVector<Scalar = T>> SimdChunkMut<'a, T, V> {
    /// Load the current values as a SIMD vector
    #[inline]
    pub fn load(&self) -> V {
        V::from_slice(self.slice)
    }

    /// Store a SIMD vector to this chunk
    #[inline]
    pub fn store(&mut self, vec: V) {
        vec.to_slice(self.slice)
    }
}

// Note: Deref is not implemented for SimdChunkMut because we can't return a reference
// to a temporary V. Users should use load() and store() methods instead.

/// Block64 type alias (T034)
///
/// Standard 64-sample audio block (common for low-latency processing)
pub type Block64 = AudioBlock<f32, 64>;

/// Block128 type alias (T034)
///
/// Standard 128-sample audio block (balanced latency/efficiency)
pub type Block128 = AudioBlock<f32, 128>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::scalar::ScalarVector;

    /// Test T036: Verify alignment
    #[test]
    fn test_block_alignment() {
        let block64 = Block64::new();
        let block128 = Block128::new();

        // Check 64-byte alignment
        let ptr64 = &block64 as *const _ as usize;
        let ptr128 = &block128 as *const _ as usize;

        assert_eq!(ptr64 % 64, 0, "Block64 not 64-byte aligned");
        assert_eq!(ptr128 % 64, 0, "Block128 not 64-byte aligned");
    }

    /// Test basic block operations
    #[test]
    fn test_block_creation() {
        let block = Block64::new();
        assert_eq!(block.len(), 64);
        assert!(!block.is_empty());

        let samples = [1.0f32; 64];
        let block = Block64::from_slice(&samples);
        assert_eq!(block[0], 1.0);
        assert_eq!(block[63], 1.0);
    }

    /// Test SIMD chunk iteration
    #[test]
    fn test_simd_chunks() {
        let mut block = Block64::new();

        // Fill with sequential values
        for i in 0..64 {
            block[i] = i as f32;
        }

        // Read chunks
        let chunks = block.as_chunks::<ScalarVector<f32>>();
        assert_eq!(chunks.len(), 64); // Scalar has 1 lane

        let mut sum = 0.0;
        for chunk in chunks.iter() {
            sum += chunk.horizontal_sum();
        }
        assert_eq!(sum, (0..64).sum::<i32>() as f32);
    }

    /// Test mutable SIMD chunks
    #[test]
    fn test_simd_chunks_mut() {
        let mut block = Block64::new();

        // Fill with ones
        for i in 0..64 {
            block[i] = 1.0;
        }

        // Multiply by 2 using SIMD
        let two = ScalarVector(2.0);
        for mut chunk in block.as_chunks_mut::<ScalarVector<f32>>().iter_mut() {
            let val = chunk.load();
            chunk.store(val.mul(two));
        }

        // Verify
        for i in 0..64 {
            assert_eq!(block[i], 2.0);
        }
    }
}
