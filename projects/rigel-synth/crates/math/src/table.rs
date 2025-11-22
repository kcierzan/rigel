//! Lookup table infrastructure for wavetable synthesis and function approximation
//!
//! This module provides efficient lookup table structures with support for:
//! - Const-generic table sizes for compile-time optimization
//! - Linear and cubic interpolation
//! - SIMD gather operations for vectorized lookups
//! - Multiple index wrapping modes (Wrap, Mirror, Clamp)
//!
#![allow(clippy::needless_range_loop)]
//! # Example
//!
//! ```rust
//! use rigel_math::table::{LookupTable, IndexMode};
//! use rigel_math::{DefaultSimdVector, SimdVector};
//!
//! // Create a sine wave lookup table
//! let table = LookupTable::<f32, 1024>::from_fn(|i, size| {
//!     let phase = (i as f32 / size as f32) * 2.0 * std::f32::consts::PI;
//!     phase.sin()
//! });
//!
//! // Scalar linear interpolation lookup
//! let value = table.lookup_linear(0.5, IndexMode::Wrap);
//!
//! // SIMD cubic interpolation lookup
//! let indices = DefaultSimdVector::splat(256.5);
//! let values = table.lookup_cubic_simd(indices, IndexMode::Wrap);
//! ```

use crate::SimdVector;

/// Index wrapping mode for table lookups
///
/// Determines how out-of-bounds indices are handled during table lookups.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexMode {
    /// Wrap indices modulo table size (periodic)
    ///
    /// Example: For size=100, index 105 wraps to 5, index -3 wraps to 97
    Wrap,

    /// Mirror indices at table boundaries (pingpong)
    ///
    /// Example: For size=100, index 105 mirrors to 95, index -3 mirrors to 3
    Mirror,

    /// Clamp indices to [0, size-1] range
    ///
    /// Example: For size=100, index 105 clamps to 99, index -3 clamps to 0
    Clamp,
}

/// Lookup table with const-generic size
///
/// Provides efficient table-based function approximation with support for
/// linear and cubic interpolation, both scalar and SIMD vectorized.
///
/// The table size is a const generic parameter, allowing the compiler to
/// optimize table operations and eliminate bounds checks in many cases.
///
/// # Type Parameters
///
/// * `T` - Element type (typically `f32` for audio DSP)
/// * `SIZE` - Table size as a const generic
#[derive(Debug, Clone)]
pub struct LookupTable<T, const SIZE: usize> {
    /// Table data
    data: [T; SIZE],
}

impl<T: Copy + Default, const SIZE: usize> LookupTable<T, SIZE> {
    /// Create a new lookup table from a generator function
    ///
    /// The generator function receives the current index and total size,
    /// allowing easy creation of periodic functions like sine waves.
    ///
    /// # Arguments
    ///
    /// * `f` - Generator function: `fn(index, size) -> T`
    ///
    /// # Example
    ///
    /// ```rust
    /// use rigel_math::table::LookupTable;
    ///
    /// // Create a ramp from 0.0 to 1.0
    /// let table = LookupTable::<f32, 256>::from_fn(|i, size| {
    ///     i as f32 / size as f32
    /// });
    /// ```
    pub fn from_fn<F>(f: F) -> Self
    where
        F: Fn(usize, usize) -> T,
    {
        let mut data: [T; SIZE] = [T::default(); SIZE];
        for i in 0..SIZE {
            data[i] = f(i, SIZE);
        }
        Self { data }
    }

    /// Get reference to underlying table data
    pub fn data(&self) -> &[T; SIZE] {
        &self.data
    }

    /// Get table size
    pub const fn size(&self) -> usize {
        SIZE
    }
}

impl<const SIZE: usize> LookupTable<f32, SIZE> {
    /// Apply index mode to wrap/mirror/clamp a fractional index
    ///
    /// Returns the wrapped index and the fractional part for interpolation.
    fn apply_index_mode(&self, index: f32, mode: IndexMode) -> (usize, f32) {
        let size = self.size() as f32;

        let (wrapped_index, frac) = match mode {
            IndexMode::Wrap => {
                // Modulo wrapping
                let mut idx = index;
                while idx < 0.0 {
                    idx += size;
                }
                while idx >= size {
                    idx -= size;
                }
                let i0 = libm::floorf(idx) as usize;
                let frac = idx - i0 as f32;
                (i0 % self.size(), frac)
            }
            IndexMode::Mirror => {
                // Ping-pong mirroring
                let mut idx = index;
                let double_size = size * 2.0;

                // Wrap to [0, 2*size) first
                while idx < 0.0 {
                    idx += double_size;
                }
                while idx >= double_size {
                    idx -= double_size;
                }

                // Mirror second half
                if idx >= size {
                    idx = double_size - idx;
                }

                let i0 = libm::floorf(idx) as usize;
                let frac = idx - i0 as f32;
                (i0.min(self.size() - 1), frac)
            }
            IndexMode::Clamp => {
                // Clamp to valid range
                let idx = libm::fmaxf(0.0, libm::fminf(index, size - 1.0));
                let i0 = libm::floorf(idx) as usize;
                let frac = idx - i0 as f32;
                (i0.min(self.size() - 1), frac)
            }
        };

        (wrapped_index, frac)
    }

    /// Perform scalar linear interpolation lookup
    ///
    /// # Arguments
    ///
    /// * `index` - Fractional index into the table
    /// * `mode` - Index wrapping mode
    ///
    /// # Returns
    ///
    /// Linearly interpolated value at the given index
    ///
    /// # Example
    ///
    /// ```rust
    /// use rigel_math::table::{LookupTable, IndexMode};
    ///
    /// let table = LookupTable::<f32, 256>::from_fn(|i, size| i as f32);
    /// let value = table.lookup_linear(10.5, IndexMode::Wrap);
    /// // Returns interpolation between table[10] and table[11]
    /// ```
    pub fn lookup_linear(&self, index: f32, mode: IndexMode) -> f32 {
        let (i0, frac) = self.apply_index_mode(index, mode);
        let i1 = match mode {
            IndexMode::Wrap => (i0 + 1) % self.size(),
            IndexMode::Mirror | IndexMode::Clamp => (i0 + 1).min(self.size() - 1),
        };

        let y0 = self.data[i0];
        let y1 = self.data[i1];

        // Linear interpolation: y0 + frac * (y1 - y0)
        y0 + frac * (y1 - y0)
    }

    /// Perform scalar cubic (Hermite) interpolation lookup
    ///
    /// Uses 4-point cubic interpolation for smoother results than linear.
    ///
    /// # Arguments
    ///
    /// * `index` - Fractional index into the table
    /// * `mode` - Index wrapping mode
    ///
    /// # Returns
    ///
    /// Cubic-interpolated value at the given index
    ///
    /// # Example
    ///
    /// ```rust
    /// use rigel_math::table::{LookupTable, IndexMode};
    ///
    /// let table = LookupTable::<f32, 256>::from_fn(|i, size| {
    ///     let x = i as f32 / size as f32 * 2.0 * std::f32::consts::PI;
    ///     x.sin()
    /// });
    /// let value = table.lookup_cubic(64.3, IndexMode::Wrap);
    /// ```
    pub fn lookup_cubic(&self, index: f32, mode: IndexMode) -> f32 {
        let (i1, frac) = self.apply_index_mode(index, mode);

        // Get 4 points for cubic interpolation: y0, y1, y2, y3
        let size = self.size();

        let (i0, i2, i3) = match mode {
            IndexMode::Wrap => (
                if i1 == 0 { size - 1 } else { i1 - 1 },
                (i1 + 1) % size,
                (i1 + 2) % size,
            ),
            IndexMode::Mirror | IndexMode::Clamp => (
                if i1 == 0 { 0 } else { i1 - 1 },
                (i1 + 1).min(size - 1),
                (i1 + 2).min(size - 1),
            ),
        };

        let y0 = self.data[i0];
        let y1 = self.data[i1];
        let y2 = self.data[i2];
        let y3 = self.data[i3];

        // 4-point cubic interpolation (Catmull-Rom spline)
        let c0 = y1;
        let c1 = 0.5 * (y2 - y0);
        let c2 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3;
        let c3 = 0.5 * (y3 - y0) + 1.5 * (y1 - y2);

        // Evaluate polynomial: c0 + c1*t + c2*t^2 + c3*t^3
        c0 + frac * (c1 + frac * (c2 + frac * c3))
    }
}

impl<const SIZE: usize> LookupTable<f32, SIZE> {
    /// Perform SIMD linear interpolation lookup with gather operations
    ///
    /// Looks up multiple table values simultaneously using SIMD gather operations.
    ///
    /// # Arguments
    ///
    /// * `indices` - SIMD vector of fractional indices
    /// * `mode` - Index wrapping mode
    ///
    /// # Returns
    ///
    /// SIMD vector of linearly interpolated values
    ///
    /// # Example
    ///
    /// ```rust
    /// use rigel_math::table::{LookupTable, IndexMode};
    /// use rigel_math::{DefaultSimdVector, SimdVector};
    ///
    /// let table = LookupTable::<f32, 256>::from_fn(|i, _| i as f32);
    /// let indices = DefaultSimdVector::from_slice(&[10.5, 20.3, 30.7, 40.1]);
    /// let values = table.lookup_linear_simd(indices, IndexMode::Wrap);
    /// ```
    pub fn lookup_linear_simd<V: SimdVector<Scalar = f32>>(
        &self,
        indices: V,
        mode: IndexMode,
    ) -> V {
        // For now, implement as scalar fallback
        // TODO: Optimize with actual SIMD gather when available
        let mut indices_buf = [0.0f32; 16]; // Max SIMD width
        indices.to_slice(&mut indices_buf);

        let mut results = [0.0f32; 16];
        for lane in 0..V::LANES {
            let index = indices_buf[lane];
            results[lane] = self.lookup_linear(index, mode);
        }
        V::from_slice(&results)
    }

    /// Perform SIMD cubic interpolation lookup with gather operations
    ///
    /// Looks up multiple table values simultaneously using SIMD gather operations
    /// with 4-point cubic interpolation.
    ///
    /// # Arguments
    ///
    /// * `indices` - SIMD vector of fractional indices
    /// * `mode` - Index wrapping mode
    ///
    /// # Returns
    ///
    /// SIMD vector of cubic-interpolated values
    ///
    /// # Example
    ///
    /// ```rust
    /// use rigel_math::table::{LookupTable, IndexMode};
    /// use rigel_math::{DefaultSimdVector, SimdVector};
    ///
    /// let table = LookupTable::<f32, 1024>::from_fn(|i, size| {
    ///     let x = i as f32 / size as f32 * 2.0 * std::f32::consts::PI;
    ///     x.sin()
    /// });
    /// let indices = DefaultSimdVector::from_slice(&[256.5, 512.3, 768.7, 1000.1]);
    /// let values = table.lookup_cubic_simd(indices, IndexMode::Wrap);
    /// ```
    pub fn lookup_cubic_simd<V: SimdVector<Scalar = f32>>(&self, indices: V, mode: IndexMode) -> V {
        // For now, implement as scalar fallback
        // TODO: Optimize with actual SIMD gather when available
        let mut indices_buf = [0.0f32; 16]; // Max SIMD width
        indices.to_slice(&mut indices_buf);

        let mut results = [0.0f32; 16];
        for lane in 0..V::LANES {
            let index = indices_buf[lane];
            results[lane] = self.lookup_cubic(index, mode);
        }
        V::from_slice(&results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_fn() {
        let table = LookupTable::<f32, 10>::from_fn(|i, _size| i as f32);
        assert_eq!(table.data[0], 0.0);
        assert_eq!(table.data[5], 5.0);
        assert_eq!(table.data[9], 9.0);
    }

    #[test]
    fn test_lookup_linear_exact() {
        let table = LookupTable::<f32, 10>::from_fn(|i, _size| (i * 10) as f32);

        // Exact indices should return exact values
        assert_eq!(table.lookup_linear(0.0, IndexMode::Wrap), 0.0);
        assert_eq!(table.lookup_linear(5.0, IndexMode::Wrap), 50.0);
        assert_eq!(table.lookup_linear(9.0, IndexMode::Wrap), 90.0);
    }

    #[test]
    fn test_lookup_linear_interpolation() {
        let table = LookupTable::<f32, 10>::from_fn(|i, _size| (i * 10) as f32);

        // Test interpolation
        let value = table.lookup_linear(5.5, IndexMode::Wrap);
        assert!((value - 55.0).abs() < 0.001);
    }

    #[test]
    fn test_index_mode_wrap() {
        let table = LookupTable::<f32, 10>::from_fn(|i, _size| i as f32);

        // Wrapping beyond size
        let value = table.lookup_linear(10.0, IndexMode::Wrap);
        assert!((value - 0.0).abs() < 0.001);

        let value = table.lookup_linear(15.0, IndexMode::Wrap);
        assert!((value - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_index_mode_clamp() {
        let table = LookupTable::<f32, 10>::from_fn(|i, _size| i as f32);

        // Clamping beyond size
        let value = table.lookup_linear(15.0, IndexMode::Clamp);
        assert!((value - 9.0).abs() < 0.001);

        let value = table.lookup_linear(-5.0, IndexMode::Clamp);
        assert!((value - 0.0).abs() < 0.001);
    }
}
