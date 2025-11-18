# Benchmarking and Performance Testing

This document describes the benchmarking infrastructure for the `rigel-dsp` crate and how to use it to measure and validate performance.

## Overview

Rigel uses a dual-benchmark approach:

1. **Criterion** - Wall-clock time measurements with statistical analysis
   - Best for: Local development, detailed profiling, optimization work
   - Provides: Timing distributions, regression detection, HTML reports
   - Trade-off: Results vary based on system load and hardware

2. **iai-callgrind** - Deterministic instruction-count measurements
   - Best for: CI/CD, cross-platform comparisons, exact regression detection
   - Provides: CPU instruction counts, cache statistics, deterministic results
   - Trade-off: Doesn't reflect actual wall-clock time

## Quick Start

### Prerequisites

Ensure you're in the devenv shell (automatic with direnv):
```bash
# Should already be active if using direnv
# Otherwise: devenv shell
```

### Running Benchmarks

```bash
# Run all benchmarks (both Criterion and iai-callgrind)
bench:all

# Run only Criterion benchmarks (wall-clock time)
bench:criterion

# Run only iai-callgrind benchmarks (instruction counts)
bench:iai

# Save a Criterion baseline for future comparisons
bench:baseline

# Generate a flamegraph for optimization work
bench:flamegraph

# macOS only: Profile with Instruments.app (detailed analysis)
bench:instruments
```

### Platform-Specific Setup

#### macOS
iai-callgrind requires Valgrind, which must be installed via Homebrew:
```bash
brew install valgrind
```

Note: Some Valgrind features may require disabling System Integrity Protection (SIP), but basic benchmarking works without this.

#### Linux
All tools are provided by devenv.nix automatically. For hardware performance counters with perf:
```bash
# May need to adjust perf_event_paranoid
sudo sysctl -w kernel.perf_event_paranoid=-1
```

## Benchmark Suite Coverage

### Utility Functions (`utility_functions` group)
- `midi_to_freq` - MIDI note to frequency conversion (uses `libm::powf`)
- `lerp` - Linear interpolation
- `clamp` - Value clamping (in range, below min, above max)
- `soft_clip` - Soft clipping with exponential curves

### Oscillator (`oscillator` group)
- Single sample generation
- Buffer processing (100 samples, 44100 samples)
- Frequency changes and combined operations

### Envelope (`envelope` group)
- Per-stage processing (Attack, Decay, Sustain, Release)
- Full ADSR cycles
- Percussive envelopes (fast attack/release)
- Pad envelopes (slow attack/release)

### SynthEngine (`synth_engine*` groups)
- Single sample processing
- Buffer processing (64, 128, 256, 512, 1024, 2048 samples)
- Note lifecycle (note_on → sustain → note_off)
- Rapid retriggering
- Pitch modulation
- Sustained vs percussive notes

### Throughput (`throughput` group)
- Different sample rates (44.1kHz, 48kHz, 96kHz)
- Samples per second measurements

### CPU Usage Validation (`cpu_usage_validation` group)
- Single voice processing time
- Polyphonic extrapolation (1, 4, 8, 16 voices)

## Interpreting Results

### Criterion Results

Criterion generates detailed HTML reports in `target/criterion/`:
```bash
# Open the HTML report in your browser
open target/criterion/report/index.html
```

Key metrics to watch:
- **Mean time**: Average execution time
- **Std deviation**: Consistency of performance
- **Throughput**: Samples per second (for buffer benchmarks)

#### Example Output
```
synth_engine/single_sample
                        time:   [45.234 ns 45.678 ns 46.123 ns]
                        thrpt:  [21.7 Msamples/s 21.9 Msamples/s 22.1 Msamples/s]
```

Interpretation:
- Processing one sample takes ~45.7 nanoseconds
- Throughput is ~21.9 million samples/second
- At 44.1kHz, available time per sample: 22,675 ns
- CPU usage: (45.7 / 22,675) × 100 = **0.2%** per voice

### iai-callgrind Results

iai-callgrind outputs instruction counts and cache statistics:
```bash
# Results are printed to terminal and saved to target/iai/
```

Key metrics:
- **Instructions**: Total CPU instructions executed
- **L1 Hits/Misses**: Level 1 cache performance
- **L2 Hits/Misses**: Level 2 cache performance
- **RAM Hits**: Main memory accesses

#### Example Output
```
iai_synth_single_sample
  Instructions:     1,234
  L1 Hits:          1,180
  L1 Misses:           54
  L2 Hits:             50
  L2 Misses:            4
```

Lower instruction counts and fewer cache misses = better performance.

## Performance Targets

### Single Voice CPU Usage

**Formula**: `CPU% = (processing_time / available_time) × 100`

At 44.1kHz sample rate:
- Available time per sample: 1/44100 = 22.675 microseconds = 22,675 nanoseconds
- Target processing time: < 2,268 ns for <10% CPU per voice
- Realistic target: 200-500 ns for ~1-2% CPU per voice

### Polyphonic Targets

For 8-voice polyphony:
- Single voice at 1% CPU → 8 voices at 8% CPU
- Leaves 92% CPU for GUI, effects, other processing

### Memory Allocations

The DSP core is `no_std` and must never allocate:
- **Target**: Zero heap allocations in benchmark runs
- Check with: `cargo flamegraph` or Instruments (macOS)

## Optimization Workflow

### 1. Establish Baseline
```bash
# Save current performance as baseline
bench:baseline
```

### 2. Make Changes
Edit DSP code in `projects/rigel-synth/crates/dsp/src/lib.rs`

### 3. Compare Performance
```bash
# Criterion automatically compares to baseline
bench:criterion

# Check for regressions
# Look for "Performance has regressed" in output
```

### 4. Profile Hot Spots
```bash
# Generate flamegraph to identify bottlenecks
bench:flamegraph

# Open the flamegraph
open flamegraph.svg
```

### 5. Validate with iai-callgrind
```bash
# Get deterministic instruction counts
bench:iai

# Compare instruction counts to previous runs
```

## Flamegraph Analysis

Flamegraphs visually show where CPU time is spent:

```bash
bench:flamegraph
open flamegraph.svg
```

Reading flamegraphs:
- **Width** = time spent (wider = slower)
- **Height** = call stack depth
- **Color** = different code modules (not meaningful)

Look for:
- Wide bars at the top → hot spots to optimize
- Unexpected library calls
- Deep call stacks (may indicate inlining opportunities)

## Advanced Profiling

### macOS Instruments

For detailed profiling on macOS with Apple's Instruments.app:

```bash
# Run Instruments profiling (automatically installs cargo-instruments if needed)
bench:instruments

# Opens Instruments.app with detailed profiling data
```

**Prerequisites:**
- Xcode Command Line Tools must be installed:
  ```bash
  xcode-select --install
  ```
- `cargo-instruments` is automatically provided by the devenv shell (installed via Nix)
- No manual installation required

**Available Templates:**
```bash
# Time profiling (default) - shows where time is spent
bench:instruments

# For other templates, run cargo-instruments directly:
# Allocations profiling
cargo instruments --bench criterion_benches --template alloc

# System trace (activity, threads, memory)
cargo instruments --bench criterion_benches --template sys

# List all available templates
cargo instruments --list-templates
```

**Reading Instruments Results:**
- Time Profiler: Shows CPU time per function (look for wide bars = hotspots)
- Allocations: Tracks heap allocations (DSP core should show zero!)
- System Trace: Shows thread activity and system calls

### Linux perf

For hardware performance counters on Linux:
```bash
# Basic profiling
perf record cargo bench --bench criterion_benches
perf report

# Detailed cache analysis
perf stat -e cache-references,cache-misses cargo bench --bench criterion_benches
```

## Baseline Management

### Saving Baselines

```bash
# Save baseline for the main branch
bench:baseline

# Baselines are stored in target/criterion/*/base/
```

### Comparing Branches

```bash
# On main branch
git checkout main
bench:baseline

# On feature branch
git checkout feature/optimization
bench:criterion

# Criterion automatically compares to main baseline
```

### Platform-Specific Baselines

Store baselines per platform in `benches/baselines/`:
```
benches/baselines/
  ├── darwin-aarch64/   # Apple Silicon
  ├── darwin-x86_64/    # Intel Mac
  └── linux-x86_64/     # Linux
```

## Benchmark Development

### Adding New Benchmarks

Edit `projects/rigel-synth/crates/dsp/benches/criterion_benches.rs`:

```rust
fn bench_new_feature(c: &mut Criterion) {
    let mut group = c.benchmark_group("new_feature");

    group.bench_function("feature_name", |b| {
        // Setup code here
        let mut engine = SynthEngine::new(44100.0);

        b.iter(|| {
            // Code to benchmark goes here
            black_box(engine.process_sample(black_box(&params)))
        });
    });

    group.finish();
}
```

Add to the criterion_group!:
```rust
criterion_group! {
    name = benches;
    config = configure_criterion();
    targets = /* existing targets */,
              bench_new_feature  // Add new benchmark
}
```

### Benchmark Best Practices

1. **Use `black_box`** - Prevents compiler from optimizing away code
   ```rust
   black_box(function(black_box(input)))
   ```

2. **Setup outside b.iter()** - Don't benchmark setup code
   ```rust
   let mut engine = SynthEngine::new(44100.0);  // Outside
   b.iter(|| {
       black_box(engine.process_sample(&params))  // Only this is timed
   });
   ```

3. **Test realistic scenarios** - Match actual usage patterns
   ```rust
   // Good: Realistic buffer size
   for _ in 0..512 { process_sample() }

   // Bad: Unrealistic workload
   for _ in 0..1_000_000 { process_sample() }
   ```

4. **Set appropriate throughput** - For meaningful samples/sec metrics
   ```rust
   group.throughput(Throughput::Elements(512));  // 512 samples per iteration
   ```

## Continuous Integration (Future)

When CI integration is added, benchmarks will:
- Run iai-callgrind on every PR (fast, deterministic)
- Fail PRs with >5% instruction count regression
- Run full Criterion suite nightly
- Publish reports to GitHub Pages

## Troubleshooting

### "'instruments' command not found" (macOS)

If you see this error when running `bench:instruments`, install Xcode Command Line Tools:
```bash
xcode-select --install
```

The `instruments` binary is part of Apple's development tools and is required for profiling.

### "Valgrind not found" (macOS)
```bash
brew install valgrind
```

### "Permission denied" when using perf (Linux)
```bash
sudo sysctl -w kernel.perf_event_paranoid=-1
```

### Noisy Criterion results
- Close other applications
- Disable CPU frequency scaling (if available)
- Run benchmarks multiple times and compare
- Use iai-callgrind for deterministic results

### Flamegraph not generating
```bash
# Ensure flamegraph is installed
which flamegraph

# Try generating manually
cargo build --bench criterion_benches --release
perf record --call-graph=dwarf target/release/criterion_benches-* --bench
perf script | flamegraph.pl > flamegraph.svg
```

## Performance Regression Detection

### Setting Regression Thresholds

Edit `criterion_benches.rs` to customize noise thresholds:
```rust
fn configure_criterion() -> Criterion {
    Criterion::default()
        .noise_threshold(0.05)  // 5% = default
        // .noise_threshold(0.02)  // 2% = stricter
}
```

### Automated Comparison Script

Create `scripts/compare-benchmarks.sh`:
```bash
#!/bin/bash
set -euo pipefail

git checkout main
bench:baseline

git checkout "$1"
bench:criterion

# Criterion will automatically compare and report regressions
```

## Resources

- [Criterion.rs User Guide](https://bheisler.github.io/criterion.rs/book/)
- [iai-callgrind Documentation](https://docs.rs/iai-callgrind/)
- [Flamegraph Guide](https://www.brendangregg.com/flamegraphs.html)
- [Linux perf Tutorial](https://perf.wiki.kernel.org/index.php/Tutorial)
