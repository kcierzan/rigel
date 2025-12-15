//! PCG32 pseudo-random number generator.
//!
//! Used internally for sample-and-hold and noise waveshapes.
//! Provides deterministic, reproducible random sequences.

/// PCG32 pseudo-random number generator.
///
/// A minimal, high-quality PRNG suitable for audio synthesis.
/// Implements the PCG (Permuted Congruential Generator) algorithm.
///
/// # Properties
///
/// - 64-bit state, 32-bit output
/// - Period: 2^64
/// - Passes statistical tests (TestU01, PractRand)
/// - Deterministic: same seed produces same sequence
/// - Copy/Clone for real-time safety
#[derive(Clone, Copy, Debug)]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Create a new RNG with the given seed.
    ///
    /// # Arguments
    /// * `seed` - Initial seed value
    pub const fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x2C9277B5_27D4EB2D_u64),
        }
    }

    /// Generate the next random u32.
    ///
    /// Uses PCG-XSH-RR algorithm for high-quality randomness.
    #[inline(always)]
    pub fn next_u32(&mut self) -> u32 {
        let old_state = self.state;
        self.state = old_state
            .wrapping_mul(6364136223846793005_u64)
            .wrapping_add(1442695040888963407_u64);

        let xor_shifted = (((old_state >> 18) ^ old_state) >> 27) as u32;
        let rot = (old_state >> 59) as u32;
        xor_shifted.rotate_right(rot)
    }

    /// Generate a random f32 in [-1.0, 1.0].
    ///
    /// Uses high 24 bits for better distribution.
    #[inline(always)]
    pub fn next_f32(&mut self) -> f32 {
        let value = self.next_u32();
        // Use high 24 bits for better distribution
        let normalized = ((value >> 8) as f32) / 16777215.0;
        normalized * 2.0 - 1.0
    }

    /// Generate a random f32 in [0.0, 1.0].
    #[inline(always)]
    pub fn next_f32_unipolar(&mut self) -> f32 {
        let value = self.next_u32();
        ((value >> 8) as f32) / 16777215.0
    }
}

impl Default for Rng {
    fn default() -> Self {
        Self::new(0x12345678)
    }
}
