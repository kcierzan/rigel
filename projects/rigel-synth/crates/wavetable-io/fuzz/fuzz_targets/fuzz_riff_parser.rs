//! Fuzz target for RIFF/WAV parsing.
//!
//! This fuzzer tests the RIFF parsing code with arbitrary byte sequences
//! to catch panics, infinite loops, or memory issues when processing
//! potentially malicious or malformed WAV files.
//!
//! Run with: `cargo +nightly fuzz run fuzz_riff_parser`

#![no_main]

use libfuzzer_sys::fuzz_target;
use wavetable_io::riff::{
    extract_wtbl_chunk, parse_riff_chunks, read_data_chunk, read_fmt_chunk,
};

fuzz_target!(|data: &[u8]| {
    // Test parse_riff_chunks - should never panic
    let _ = parse_riff_chunks(data);

    // Test extract_wtbl_chunk - should never panic
    let _ = extract_wtbl_chunk(data);

    // Test read_fmt_chunk - should never panic
    let _ = read_fmt_chunk(data);

    // Test read_data_chunk - should never panic
    let _ = read_data_chunk(data);

    // Additional test: Ensure that valid RIFF headers with corrupted
    // chunk data are handled gracefully
    if data.len() >= 12 {
        // Try with a minimal valid RIFF/WAVE header
        let mut modified = data.to_vec();
        if modified.len() >= 12 {
            modified[0..4].copy_from_slice(b"RIFF");
            modified[8..12].copy_from_slice(b"WAVE");
            let _ = parse_riff_chunks(&modified);
        }
    }
});
