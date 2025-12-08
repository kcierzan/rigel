//! SynthEngine Timebase Integration Tests
//!
//! Tests for timebase integration with SynthEngine.
//! Pure Timebase unit tests are in the rigel-timing crate.

use rigel_dsp::{SynthEngine, SynthParams};

#[test]
fn test_synth_engine_timebase_initialized() {
    let engine = SynthEngine::new(44100.0);

    // Timebase should be initialized with correct sample rate
    assert_eq!(engine.timebase().sample_rate(), 44100.0);
    assert_eq!(engine.timebase().sample_position(), 0);
}

#[test]
fn test_synth_engine_process_block_advances_timebase() {
    let mut engine = SynthEngine::new(44100.0);
    let params = SynthParams::default();

    // Initial position should be 0
    assert_eq!(engine.timebase().sample_position(), 0);

    // Process first block
    let mut output = [0.0f32; 64];
    engine.process_block(&mut output, &params);

    // Timebase should have advanced by block size
    assert_eq!(engine.timebase().sample_position(), 64);
    assert_eq!(engine.timebase().block_start(), 0);
    assert_eq!(engine.timebase().block_size(), 64);

    // Process second block
    engine.process_block(&mut output, &params);

    // Timebase should continue advancing
    assert_eq!(engine.timebase().sample_position(), 128);
    assert_eq!(engine.timebase().block_start(), 64);
}

#[test]
fn test_synth_engine_reset_resets_timebase() {
    let mut engine = SynthEngine::new(44100.0);
    let params = SynthParams::default();

    // Process some blocks
    let mut output = [0.0f32; 64];
    engine.process_block(&mut output, &params);
    engine.process_block(&mut output, &params);

    assert_eq!(engine.timebase().sample_position(), 128);

    // Reset engine
    engine.reset();

    // Timebase should be reset
    assert_eq!(engine.timebase().sample_position(), 0);
    assert_eq!(engine.timebase().block_start(), 0);
    assert_eq!(engine.timebase().block_size(), 0);
}

#[test]
fn test_synth_engine_timebase_mut_allows_modification() {
    let mut engine = SynthEngine::new(44100.0);

    // Modify timebase through mutable accessor
    engine.timebase_mut().set_sample_rate(96000.0);

    assert_eq!(engine.timebase().sample_rate(), 96000.0);
}
