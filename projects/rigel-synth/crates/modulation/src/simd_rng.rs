//! SIMD-optimized Xorshift128+ random number generator.
//!
//! Based on [Daniel Lemire's SIMDxorshift](https://github.com/lemire/SIMDxorshift),
//! adapted for rigel-math's backend system.

use rigel_math::{DefaultSimdVector, SimdVector};

/// Maximum number of parallel generators (for AVX-512 with 16 lanes).
const MAX_LANES: usize = 16;

/// SIMD-optimized Xorshift128+ random number generator.
///
/// Generates `DefaultSimdVector::LANES` random values in parallel:
/// - AVX-512: 16 values per call
/// - AVX2: 8 values per call
/// - NEON: 4 values per call
/// - Scalar: 1 value per call (fallback)
///
/// Uses interleaved state arrays to avoid data dependencies.
///
/// # Real-Time Safety
///
/// - No heap allocations
/// - Constant-time operations
/// - Copy/Clone semantics
///
/// # Example
///
/// ```ignore
/// use rigel_modulation::SimdXorshift128;
///
/// let mut rng = SimdXorshift128::new(12345);
/// let mut buffer = [0.0f32; 64];
/// rng.fill_buffer(&mut buffer);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct SimdXorshift128 {
    // State arrays sized for max SIMD width (16 for AVX-512)
    // Only first LANES elements used per backend
    state0: [u64; MAX_LANES],
    state1: [u64; MAX_LANES],
}

impl SimdXorshift128 {
    /// Create a new RNG with the given seed.
    ///
    /// Initializes parallel generator states with distinct seeds
    /// derived from the base seed using splitmix64.
    pub fn new(seed: u64) -> Self {
        let mut s0 = [0u64; MAX_LANES];
        let mut s1 = [0u64; MAX_LANES];
        let mut current_seed = seed;

        for i in 0..MAX_LANES {
            // Splitmix64 to derive independent seeds
            current_seed = current_seed.wrapping_add(0x9E37_79B9_7F4A_7C15_u64);
            let mut z = current_seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9_u64);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB_u64);
            s0[i] = z ^ (z >> 31);

            current_seed = current_seed.wrapping_add(0x9E37_79B9_7F4A_7C15_u64);
            z = current_seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9_u64);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB_u64);
            s1[i] = z ^ (z >> 31);
        }

        Self {
            state0: s0,
            state1: s1,
        }
    }

    /// Generate LANES random f32 values in [-1.0, 1.0].
    ///
    /// Number of values depends on SIMD backend:
    /// - AVX-512: 16 values
    /// - AVX2: 8 values
    /// - NEON: 4 values
    /// - Scalar: 1 value
    #[inline]
    pub fn next_lane_f32(&mut self) -> [f32; MAX_LANES] {
        let lanes = DefaultSimdVector::LANES;
        let mut result = [0.0f32; MAX_LANES];

        #[allow(clippy::needless_range_loop)] // Uses i to index multiple arrays
        for i in 0..lanes {
            // Xorshift128+ algorithm
            let s1 = self.state0[i];
            let s0 = self.state1[i];
            self.state0[i] = s0;
            let s1_shifted = s1 ^ (s1 << 23);
            self.state1[i] = s1_shifted ^ s0 ^ (s1_shifted >> 18) ^ (s0 >> 5);
            let raw = self.state1[i].wrapping_add(s0);

            // Convert to f32 in [-1.0, 1.0]
            // Use high 24 bits for better distribution
            result[i] = ((raw >> 40) as f32) / 16_777_215.0 * 2.0 - 1.0;
        }

        result
    }

    /// Generate a single random f32 value in [-1.0, 1.0].
    ///
    /// Uses the first generator only. For bulk generation, use
    /// `fill_buffer()` which is more efficient.
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        // Xorshift128+ algorithm on first generator
        let s1 = self.state0[0];
        let s0 = self.state1[0];
        self.state0[0] = s0;
        let s1_shifted = s1 ^ (s1 << 23);
        self.state1[0] = s1_shifted ^ s0 ^ (s1_shifted >> 18) ^ (s0 >> 5);
        let raw = self.state1[0].wrapping_add(s0);

        // Convert to f32 in [-1.0, 1.0]
        ((raw >> 40) as f32) / 16_777_215.0 * 2.0 - 1.0
    }

    /// Generate a single random f32 value in [0.0, 1.0].
    #[inline]
    pub fn next_f32_unipolar(&mut self) -> f32 {
        // Xorshift128+ algorithm on first generator
        let s1 = self.state0[0];
        let s0 = self.state1[0];
        self.state0[0] = s0;
        let s1_shifted = s1 ^ (s1 << 23);
        self.state1[0] = s1_shifted ^ s0 ^ (s1_shifted >> 18) ^ (s0 >> 5);
        let raw = self.state1[0].wrapping_add(s0);

        // Convert to f32 in [0.0, 1.0]
        ((raw >> 40) as f32) / 16_777_215.0
    }

    /// Fill a buffer with random f32 values in [-1.0, 1.0].
    ///
    /// Uses SIMD-optimized batch generation. The number of values
    /// generated per iteration depends on the SIMD backend.
    pub fn fill_buffer(&mut self, output: &mut [f32]) {
        let lanes = DefaultSimdVector::LANES;
        let mut idx = 0;

        // Process in SIMD chunks
        while idx + lanes <= output.len() {
            let values = self.next_lane_f32();
            output[idx..idx + lanes].copy_from_slice(&values[..lanes]);
            idx += lanes;
        }

        // Handle remainder
        if idx < output.len() {
            let values = self.next_lane_f32();
            for (i, sample) in output[idx..].iter_mut().enumerate() {
                *sample = values[i];
            }
        }
    }

    /// Fill a buffer with random f32 values in [0.0, 1.0].
    pub fn fill_buffer_unipolar(&mut self, output: &mut [f32]) {
        self.fill_buffer(output);
        // Convert from [-1.0, 1.0] to [0.0, 1.0]
        for sample in output.iter_mut() {
            *sample = (*sample + 1.0) * 0.5;
        }
    }
}

impl Default for SimdXorshift128 {
    fn default() -> Self {
        Self::new(0x1234_5678)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_creates_distinct_states() {
        let rng = SimdXorshift128::new(12345);
        // Each lane should have different state
        assert_ne!(rng.state0[0], rng.state0[1]);
        assert_ne!(rng.state1[0], rng.state1[1]);
    }

    #[test]
    fn test_next_f32_in_range() {
        let mut rng = SimdXorshift128::new(12345);
        for _ in 0..1000 {
            let value = rng.next_f32();
            assert!(
                (-1.0..=1.0).contains(&value),
                "Value {} out of range",
                value
            );
        }
    }

    #[test]
    fn test_next_f32_unipolar_in_range() {
        let mut rng = SimdXorshift128::new(12345);
        for _ in 0..1000 {
            let value = rng.next_f32_unipolar();
            assert!((0.0..=1.0).contains(&value), "Value {} out of range", value);
        }
    }

    #[test]
    fn test_fill_buffer_all_in_range() {
        let mut rng = SimdXorshift128::new(12345);
        let mut buffer = [0.0f32; 256];
        rng.fill_buffer(&mut buffer);

        for (i, &value) in buffer.iter().enumerate() {
            assert!(
                (-1.0..=1.0).contains(&value),
                "Value {} at index {} out of range",
                value,
                i
            );
        }
    }

    #[test]
    fn test_fill_buffer_unique_values() {
        let mut rng = SimdXorshift128::new(12345);
        let mut buffer = [0.0f32; 64];
        rng.fill_buffer(&mut buffer);

        // Count unique values (with tolerance for floating point)
        let mut unique_count = 0;
        for i in 1..buffer.len() {
            if (buffer[i] - buffer[i - 1]).abs() > 0.0001 {
                unique_count += 1;
            }
        }

        // Should have mostly unique values (noise characteristic)
        assert!(
            unique_count >= 50,
            "Not enough unique values: {}",
            unique_count
        );
    }

    #[test]
    fn test_deterministic_output() {
        let mut rng1 = SimdXorshift128::new(42);
        let mut rng2 = SimdXorshift128::new(42);

        for _ in 0..100 {
            assert_eq!(rng1.next_f32(), rng2.next_f32());
        }
    }

    #[test]
    fn test_different_seeds_different_output() {
        let mut rng1 = SimdXorshift128::new(1);
        let mut rng2 = SimdXorshift128::new(2);

        let v1 = rng1.next_f32();
        let v2 = rng2.next_f32();
        assert_ne!(v1, v2);
    }

    #[test]
    fn test_copy_semantics() {
        let mut rng1 = SimdXorshift128::new(12345);
        let _ = rng1.next_f32();

        let mut rng2 = rng1; // Copy
        assert_eq!(rng1.next_f32(), rng2.next_f32());
    }
}
