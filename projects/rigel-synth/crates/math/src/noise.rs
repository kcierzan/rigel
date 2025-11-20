//! Vectorized noise generation
//!
//! Provides fast pseudo-random noise generation for audio DSP.

use crate::traits::SimdVector;

/// Simple xorshift PRNG state
#[derive(Copy, Clone)]
pub struct NoiseState {
    state: u32,
}

impl NoiseState {
    /// Create a new noise generator with a seed
    pub fn new(seed: u32) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    /// Generate next random u32
    #[inline(always)]
    fn next(&mut self) -> u32 {
        // Xorshift algorithm
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.state = x;
        x
    }

    /// Generate random f32 in [-1, 1]
    #[inline(always)]
    pub fn next_f32(&mut self) -> f32 {
        // Convert u32 to f32 in [-1, 1]
        let value = self.next();
        let normalized = (value as f32) / (u32::MAX as f32);
        normalized * 2.0 - 1.0
    }
}

/// Generate vectorized white noise
///
/// Fills a SIMD vector with independent random values in the range [-1, 1].
///
/// # Example
///
/// ```rust
/// use rigel_math::{DefaultSimdVector, SimdVector};
/// use rigel_math::noise::{NoiseState, white_noise};
///
/// let mut state = NoiseState::new(12345);
/// let noise = white_noise::<DefaultSimdVector>(&mut state);
/// ```
#[inline(always)]
pub fn white_noise<V: SimdVector<Scalar = f32>>(state: &mut NoiseState) -> V {
    // Generate V::LANES independent random values
    let mut values = [0.0f32; 16]; // Max SIMD width (AVX512)

    for i in 0..V::LANES {
        values[i] = state.next_f32();
    }

    V::from_slice(&values[..V::LANES])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DefaultSimdVector;

    #[test]
    fn test_noise_state_basic() {
        let mut state = NoiseState::new(12345);
        let value = state.next_f32();
        assert!(value >= -1.0 && value <= 1.0, "Noise should be in [-1, 1]");
    }

    #[test]
    fn test_white_noise() {
        let mut state = NoiseState::new(67890);
        let noise = white_noise::<DefaultSimdVector>(&mut state);

        // Verify all lanes are in valid range
        let mut values = [0.0f32; 16];
        noise.to_slice(&mut values);

        for i in 0..DefaultSimdVector::LANES {
            assert!(
                values[i] >= -1.0 && values[i] <= 1.0,
                "Noise lane {} out of range: {}",
                i,
                values[i]
            );
        }
    }

    #[test]
    fn test_noise_deterministic() {
        // Same seed should produce same sequence
        let mut state1 = NoiseState::new(42);
        let mut state2 = NoiseState::new(42);

        for _ in 0..10 {
            assert_eq!(
                state1.next(),
                state2.next(),
                "Same seed should produce same sequence"
            );
        }
    }
}
