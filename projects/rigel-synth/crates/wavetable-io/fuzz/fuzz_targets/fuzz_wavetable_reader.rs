//! Fuzz target for wavetable file reading.
//!
//! This fuzzer tests the full wavetable reader pipeline with arbitrary byte
//! sequences to catch panics, infinite loops, or memory issues when processing
//! potentially malicious or malformed WAV files with WTBL chunks.
//!
//! Run with: `cargo +nightly fuzz run fuzz_wavetable_reader`

#![no_main]

use libfuzzer_sys::fuzz_target;
use wavetable_io::reader::read_wavetable_from_bytes;

fuzz_target!(|data: &[u8]| {
    // Test the full reader pipeline - should never panic
    // It may return errors for invalid data, which is expected
    let _ = read_wavetable_from_bytes(data);
});
