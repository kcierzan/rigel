//! Performance benchmarks for wavetable loading.
//!
//! Tests loading performance against SC-003 requirement:
//! <5 seconds for 100MB of wavetable data.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use prost::Message;
use wavetable_io::proto;
use wavetable_io::reader::read_wavetable_from_bytes;
use wavetable_io::riff::build_wav_with_wtbl;

/// Generate test wavetable data of specified size.
fn generate_test_wavetable(num_frames: u32, frame_length: u32, num_mip_levels: u32) -> Vec<u8> {
    // Generate mip frame lengths (halving each level)
    let mut mip_frame_lengths = Vec::with_capacity(num_mip_levels as usize);
    let mut current_length = frame_length;
    for _ in 0..num_mip_levels {
        mip_frame_lengths.push(current_length);
        current_length /= 2;
        if current_length < 4 {
            break;
        }
    }
    let actual_mip_levels = mip_frame_lengths.len() as u32;

    let metadata = proto::WavetableMetadata {
        schema_version: 1,
        wavetable_type: proto::WavetableType::HighResolution.into(),
        frame_length,
        num_frames,
        num_mip_levels: actual_mip_levels,
        mip_frame_lengths: mip_frame_lengths.clone(),
        name: Some("Benchmark Wavetable".to_string()),
        ..Default::default()
    };

    let wtbl_data = metadata.encode_to_vec();

    // Calculate total samples
    let total_samples: usize = mip_frame_lengths
        .iter()
        .map(|&len| len as usize * num_frames as usize)
        .sum();

    // Generate sample data (sine wave pattern)
    let samples: Vec<f32> = (0..total_samples)
        .map(|i| {
            let phase = (i as f32 / total_samples as f32) * std::f32::consts::TAU * 100.0;
            phase.sin()
        })
        .collect();

    build_wav_with_wtbl(&samples, 48000, &wtbl_data)
}

/// Benchmark small wavetable loading (typical single wavetable).
fn bench_small_wavetable(c: &mut Criterion) {
    // ~127KB: 64 frames x 256 samples x 7 mip levels
    let data = generate_test_wavetable(64, 256, 7);

    let mut group = c.benchmark_group("small_wavetable");
    group.throughput(Throughput::Bytes(data.len() as u64));

    group.bench_function("load_127kb", |b| {
        b.iter(|| read_wavetable_from_bytes(black_box(&data)).unwrap())
    });

    group.finish();
}

/// Benchmark medium wavetable loading (high-resolution).
fn bench_medium_wavetable(c: &mut Criterion) {
    // ~1MB: 64 frames x 2048 samples x 8 mip levels
    let data = generate_test_wavetable(64, 2048, 8);

    let mut group = c.benchmark_group("medium_wavetable");
    group.throughput(Throughput::Bytes(data.len() as u64));

    group.bench_function("load_1mb", |b| {
        b.iter(|| read_wavetable_from_bytes(black_box(&data)).unwrap())
    });

    group.finish();
}

/// Benchmark large wavetable loading (SC-003 target).
fn bench_large_wavetable(c: &mut Criterion) {
    // ~10MB: 256 frames x 4096 samples x 10 mip levels
    let data = generate_test_wavetable(256, 4096, 10);

    let mut group = c.benchmark_group("large_wavetable");
    group.throughput(Throughput::Bytes(data.len() as u64));

    group.bench_function("load_10mb", |b| {
        b.iter(|| read_wavetable_from_bytes(black_box(&data)).unwrap())
    });

    group.finish();
}

/// Benchmark scaling behavior across different wavetable sizes.
fn bench_scaling(c: &mut Criterion) {
    let configs: Vec<(u32, u32, u32, &str)> = vec![
        (16, 256, 4, "16KB"),
        (64, 256, 7, "127KB"),
        (64, 1024, 8, "500KB"),
        (64, 2048, 8, "1MB"),
        (128, 2048, 9, "4MB"),
    ];

    let mut group = c.benchmark_group("scaling");

    for (num_frames, frame_length, mip_levels, label) in configs {
        let data = generate_test_wavetable(num_frames, frame_length, mip_levels);
        let size_bytes = data.len() as u64;

        group.throughput(Throughput::Bytes(size_bytes));
        group.bench_with_input(BenchmarkId::new("load", label), &data, |b, data| {
            b.iter(|| read_wavetable_from_bytes(black_box(data)).unwrap())
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_small_wavetable,
    bench_medium_wavetable,
    bench_large_wavetable,
    bench_scaling
);
criterion_main!(benches);
