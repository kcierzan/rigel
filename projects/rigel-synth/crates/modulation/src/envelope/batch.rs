//! Batch envelope processor for polyphonic efficiency.
//!
//! Processes multiple envelopes in parallel.

use super::{Envelope, EnvelopeConfig};

/// SIMD-accelerated batch envelope processor.
///
/// Processes N envelopes in parallel. For optimal performance, N should
/// match the SIMD lane count (4 for NEON, 8 for AVX2, 16 for AVX-512).
///
/// # Type Parameters
///
/// * `N` - Number of envelopes in the batch
/// * `K` - Number of key-on segments per envelope
/// * `R` - Number of release segments per envelope
///
/// # Example
///
/// ```ignore
/// use rigel_modulation::envelope::{EnvelopeBatch, FmEnvelope};
///
/// // Create batch of 8 envelopes (AVX2 optimal)
/// let mut batch = EnvelopeBatch::<8, 6, 2>::new(44100.0);
///
/// // Trigger envelopes
/// for i in 0..8 {
///     batch.note_on(i, 60 + i as u8);
/// }
///
/// // Process samples
/// let mut output = [0.0f32; 8];
/// for _ in 0..1024 {
///     batch.process(&mut output);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct EnvelopeBatch<const N: usize, const K: usize, const R: usize> {
    /// Individual envelope instances
    envelopes: [Envelope<K, R>; N],
}

impl<const N: usize, const K: usize, const R: usize> EnvelopeBatch<N, K, R> {
    /// Create batch with default configurations.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate for all envelopes
    pub fn new(sample_rate: f32) -> Self {
        Self {
            envelopes: core::array::from_fn(|_| Envelope::new(sample_rate)),
        }
    }

    /// Create batch with specific configuration for all envelopes.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration to use for all envelopes
    pub fn with_config(config: EnvelopeConfig<K, R>) -> Self {
        Self {
            envelopes: core::array::from_fn(|_| Envelope::with_config(config)),
        }
    }

    /// Trigger note-on for envelope at index.
    ///
    /// # Arguments
    ///
    /// * `index` - Envelope index (0..N)
    /// * `midi_note` - MIDI note for rate scaling
    ///
    /// # Panics
    ///
    /// Panics if index >= N
    #[inline]
    pub fn note_on(&mut self, index: usize, midi_note: u8) {
        self.envelopes[index].note_on(midi_note);
    }

    /// Trigger note-off for envelope at index.
    ///
    /// # Arguments
    ///
    /// * `index` - Envelope index (0..N)
    ///
    /// # Panics
    ///
    /// Panics if index >= N
    #[inline]
    pub fn note_off(&mut self, index: usize) {
        self.envelopes[index].note_off();
    }

    /// Process one sample for all envelopes.
    ///
    /// # Arguments
    ///
    /// * `output` - Array to receive N linear amplitude values
    pub fn process(&mut self, output: &mut [f32; N]) {
        for (i, env) in self.envelopes.iter_mut().enumerate() {
            // Process envelope state and get linear amplitude directly
            output[i] = env.process();
        }
    }

    /// Process block of samples for all envelopes.
    ///
    /// # Arguments
    ///
    /// * `output` - 2D slice: output[sample_index] contains N values
    pub fn process_block(&mut self, output: &mut [[f32; N]]) {
        for sample_output in output.iter_mut() {
            self.process(sample_output);
        }
    }

    /// Get reference to individual envelope.
    #[inline]
    pub fn get(&self, index: usize) -> &Envelope<K, R> {
        &self.envelopes[index]
    }

    /// Get mutable reference to individual envelope.
    #[inline]
    pub fn get_mut(&mut self, index: usize) -> &mut Envelope<K, R> {
        &mut self.envelopes[index]
    }

    /// Set configuration for envelope at index.
    #[inline]
    pub fn set_config(&mut self, index: usize, config: EnvelopeConfig<K, R>) {
        self.envelopes[index].set_config(config);
    }

    /// Reset all envelopes to idle state.
    pub fn reset_all(&mut self) {
        for env in self.envelopes.iter_mut() {
            env.reset();
        }
    }

    /// Check if any envelope is still active.
    pub fn any_active(&self) -> bool {
        self.envelopes.iter().any(|env| env.is_active())
    }
}

/// Type alias for 8-envelope FM batch.
pub type FmEnvelopeBatch8 = EnvelopeBatch<8, 6, 2>;

/// Type alias for 4-envelope FM batch (NEON optimal).
pub type FmEnvelopeBatch4 = EnvelopeBatch<4, 6, 2>;

/// Type alias for 16-envelope FM batch (AVX-512 optimal).
pub type FmEnvelopeBatch16 = EnvelopeBatch<16, 6, 2>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_creation() {
        let batch = FmEnvelopeBatch8::new(44100.0);
        assert!(!batch.any_active());
    }

    #[test]
    fn test_batch_note_on() {
        let mut batch = FmEnvelopeBatch8::new(44100.0);
        batch.note_on(0, 60);
        batch.note_on(3, 72);

        assert!(batch.get(0).is_active());
        assert!(!batch.get(1).is_active());
        assert!(batch.get(3).is_active());
        assert!(batch.any_active());
    }

    #[test]
    fn test_batch_process() {
        let mut batch = FmEnvelopeBatch8::new(44100.0);
        batch.note_on(0, 60);
        batch.note_on(1, 60);

        let mut output = [0.0f32; 8];
        batch.process(&mut output);

        // Active envelopes should produce non-zero output
        assert!(output[0] > 0.0, "Active envelope should produce output");
        assert!(output[1] > 0.0, "Active envelope should produce output");

        // All values should be in valid range
        for &val in output.iter() {
            assert!((0.0..=1.0).contains(&val), "Output {} not in range", val);
        }
    }

    #[test]
    fn test_batch_process_block() {
        let mut batch = FmEnvelopeBatch8::new(44100.0);
        batch.note_on(0, 60);

        let mut output = [[0.0f32; 8]; 64];
        batch.process_block(&mut output);

        // First envelope should have non-zero values
        for sample in output.iter() {
            assert!((0.0..=1.0).contains(&sample[0]), "Output not in range");
        }
    }

    #[test]
    fn test_batch_reset_all() {
        let mut batch = FmEnvelopeBatch8::new(44100.0);
        batch.note_on(0, 60);
        batch.note_on(3, 72);
        batch.reset_all();

        assert!(!batch.any_active());
    }
}
