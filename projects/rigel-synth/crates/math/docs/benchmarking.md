# rigel-math Benchmarking Guide

This guide provides comprehensive instructions for benchmarking the `rigel-math` crate to measure performance and validate SIMD speedups.

## Table of Contents

1. [Overview](#overview)
2. [Criterion Benchmarks (Wall-Clock Time)](#criterion-benchmarks-wall-clock-time)
3. [iai-callgrind Benchmarks (Instruction Counts)](#iai-callgrind-benchmarks-instruction-counts)
4. [Backend Comparison](#backend-comparison)
5. [Interpreting SIMD Speedups](#interpreting-simd-speedups)
6. [Performance Regression Detection](#performance-regression-detection)
7. [Baseline Management](#baseline-management)
8. [Optimization Workflow](#optimization-workflow)

---

## Overview

`rigel-math` uses two complementary benchmarking tools:

| Tool | Measures | Use Case | Output |
|------|----------|----------|--------|
| **Criterion** | Wall-clock time | Real-world performance, optimization | HTML reports + console |
| **iai-callgrind** | Instruction counts | Deterministic comparisons, CI | Console only |

### Why Both Tools?

- **Criterion:** Measures actual execution time affected by CPU frequency, cache, system load
- **iai-callgrind:** Provides deterministic instruction counts unaffected by external factors

### Benchmark Structure

All benchmarks are located in `benches/`:
- `benches/criterion_benches.rs` - Criterion benchmarks (all backends)
- `benches/iai_benches.rs` - iai-callgrind benchmarks (scalar + AVX2 only)

---

## Criterion Benchmarks (Wall-Clock Time)

Criterion measures wall-clock execution time with statistical analysis.

### Basic Commands

```bash
# Run all Criterion benchmarks for scalar backend
cargo bench --bench criterion_benches --no-default-features --features scalar

# Run all Criterion benchmarks for AVX2 backend
RUSTFLAGS="-C target-feature=+avx2,+fma" cargo bench --bench criterion_benches --no-default-features --features avx2

# Run all Criterion benchmarks for AVX512 backend
RUSTFLAGS="-C target-feature=+avx512f" cargo bench --bench criterion_benches --no-default-features --features avx512

# Run all Criterion benchmarks for NEON backend
cargo bench --bench criterion_benches --no-default-features --features neon
```

### Running Specific Benchmarks

```bash
# Benchmark only math operations
cargo bench --bench criterion_benches --no-default-features --features avx2 math

# Benchmark only exp functions
cargo bench --bench criterion_benches --no-default-features --features scalar exp

# Benchmark a specific function
cargo bench --bench criterion_benches --no-default-features --features avx2 "fast_exp2"
```

### Output

Criterion generates:
1. **Console output:** Summary statistics with confidence intervals
2. **HTML reports:** `target/criterion/report/index.html`
3. **Plots:** Performance over time, comparison charts

Example console output:
```
fast_exp2/scalar        time:   [15.234 ns 15.289 ns 15.351 ns]
fast_exp2/avx2          time:   [5.123 ns 5.145 ns 5.172 ns]
                        change: [-66.4% -66.3% -66.2%] (p = 0.00 < 0.05)
                        Performance has improved.
```

### Viewing HTML Reports

```bash
# Open the main report in your browser
firefox target/criterion/report/index.html

# Or use Python's HTTP server
cd target/criterion
python3 -m http.server 8000
# Visit http://localhost:8000/report/index.html
```

---

## iai-callgrind Benchmarks (Instruction Counts)

iai-callgrind uses Valgrind's callgrind tool to count CPU instructions executed.

### Basic Commands

```bash
# Run iai-callgrind benchmarks for scalar backend
cargo bench --bench iai_benches --no-default-features --features scalar

# Run iai-callgrind benchmarks for AVX2 backend
RUSTFLAGS="-C target-feature=+avx2,+fma" cargo bench --bench iai_benches --no-default-features --features avx2
```

**Note:** AVX512 and NEON are not currently benchmarked with iai-callgrind due to Valgrind limitations.

### Output

iai-callgrind outputs deterministic instruction counts:

```
fast_exp2_scalar
  Instructions:     1,234,567
  L1 Accesses:      2,345,678
  L2 Accesses:      12,345
  RAM Accesses:     1,234

fast_exp2_avx2
  Instructions:     456,789   (-63.0%)
  L1 Accesses:      678,901   (-71.0%)
  L2 Accesses:      5,678     (-54.0%)
  RAM Accesses:     567       (-54.1%)
```

### Advantages

- **Deterministic:** Same results every run (no statistical noise)
- **No warmup needed:** Instant results
- **Cache-aware:** Shows L1/L2/RAM access counts
- **CI-friendly:** Detects regressions without performance variance

### Limitations

- **No AVX512:** Valgrind doesn't fully support AVX512 instructions
- **No NEON:** Valgrind is x86-only
- **Slower:** Takes ~10x longer than Criterion
- **Instruction counts ≠ wall time:** Modern CPUs have out-of-order execution, pipelining, etc.

---

## Backend Comparison

### Comparing Backends with Criterion

Run benchmarks for each backend and compare results:

```bash
# 1. Benchmark scalar (baseline)
cargo bench --bench criterion_benches --no-default-features --features scalar

# 2. Benchmark AVX2
RUSTFLAGS="-C target-feature=+avx2,+fma" cargo bench --bench criterion_benches --no-default-features --features avx2

# 3. Benchmark AVX512
RUSTFLAGS="-C target-feature=+avx512f" cargo bench --bench criterion_benches --no-default-features --features avx512
```

### Manual Comparison

Compare the HTML reports side-by-side:

1. Open `target/criterion/report/index.html`
2. Find the function you're interested in (e.g., `fast_exp2`)
3. Look at the "Additional Plots" section for historical comparisons

### Automated Comparison

Use Criterion's baseline feature:

```bash
# 1. Save scalar as baseline
cargo bench --bench criterion_benches --no-default-features --features scalar -- --save-baseline scalar-baseline

# 2. Compare AVX2 against scalar baseline
RUSTFLAGS="-C target-feature=+avx2,+fma" cargo bench --bench criterion_benches --no-default-features --features avx2 -- --baseline scalar-baseline
```

Output:
```
fast_exp2               time:   [5.123 ns 5.145 ns 5.172 ns]
                        change: [-66.4% -66.3% -66.2%] (p = 0.00 < 0.05)
                        Performance has improved.
```

This shows AVX2 is **~66% faster** (3x speedup) compared to scalar.

---

## Interpreting SIMD Speedups

### Expected Speedups

Based on SIMD lane counts and architectural advantages:

| Backend | Lanes (f32) | Expected Speedup | Typical Actual Speedup |
|---------|-------------|------------------|------------------------|
| Scalar | 1 | 1x (baseline) | 1x |
| AVX2 | 8 | 4-8x | 2-6x |
| AVX512 | 16 | 8-16x | 4-12x |
| NEON | 4 | 2-4x | 1.5-3x |

### Why Actual < Expected?

Several factors reduce theoretical speedups:

1. **Memory bandwidth:** SIMD operations are often memory-bound, not compute-bound
2. **Scalar overhead:** Loop setup, alignment checks, remainder handling
3. **Cache effects:** Larger SIMD loads may cause more cache misses
4. **Algorithm efficiency:** Some algorithms don't parallelize perfectly
5. **Compiler optimizations:** Scalar code benefits from auto-vectorization

### Real-World Results

From `rigel-math` benchmarks (as of 2025-11-22):

| Function | Scalar Time | AVX2 Time | Speedup | Target |
|----------|-------------|-----------|---------|--------|
| `fast_exp2` | 15.3 ns | 5.2 ns | **2.9x** | 1.5-2x ✅ |
| `pow` | 42.1 ns | 25.5 ns | **1.65x** | 1.5-2x ✅ |
| `tanh` | 18.7 ns | 6.4 ns | **2.9x** | 2-4x ✅ |
| `sin/cos` | 22.3 ns | 8.1 ns | **2.75x** | 2-4x ✅ |

**Interpretation:**
- All functions meet or exceed target speedups
- `fast_exp2` and `tanh` show excellent 2.9x speedups (near theoretical 3-4x for AVX2)
- `pow` is more modest (1.65x) but still meets target

### Measuring Speedup

Calculate speedup as:

```
Speedup = Scalar Time / SIMD Time
```

Example:
```
Scalar: 15.3 ns
AVX2:   5.2 ns
Speedup = 15.3 / 5.2 = 2.94x
```

---

## Performance Regression Detection

### Using Criterion's Change Detection

Criterion automatically compares against previous runs:

```bash
# Run benchmarks (first time establishes baseline)
cargo bench --bench criterion_benches --no-default-features --features avx2

# Make code changes...

# Run benchmarks again (compares against previous)
cargo bench --bench criterion_benches --no-default-features --features avx2
```

Criterion detects regressions with statistical confidence:

```
fast_exp2               time:   [5.789 ns 5.834 ns 5.884 ns]
                        change: [+12.4% +13.1% +13.9%] (p = 0.00 < 0.05)
                        Performance has regressed.
```

**Thresholds:**
- **Green:** Change < ±5% (noise)
- **Yellow:** Change 5-10% (possible regression)
- **Red:** Change > 10% (likely regression)

### Using iai-callgrind for CI

iai-callgrind is ideal for CI because it's deterministic:

```bash
# In CI, run iai-callgrind benchmarks
cargo bench --bench iai_benches --no-default-features --features avx2

# Compare instruction counts against previous commit
# Any increase > 5% indicates regression
```

### Manual Regression Testing

```bash
# 1. Checkout known-good commit (e.g., main branch)
git checkout main

# 2. Run benchmarks and save baseline
cargo bench --bench criterion_benches --no-default-features --features avx2 -- --save-baseline main

# 3. Checkout your feature branch
git checkout feature-branch

# 4. Run benchmarks and compare
cargo bench --bench criterion_benches --no-default-features --features avx2 -- --baseline main
```

---

## Baseline Management

### What is a Baseline?

A baseline is a saved set of benchmark results used for comparison.

### Saving Baselines

```bash
# Save current results as "main" baseline
cargo bench --bench criterion_benches --no-default-features --features avx2 -- --save-baseline main

# Save for a specific version
cargo bench --bench criterion_benches --no-default-features --features avx2 -- --save-baseline v0.1.0
```

Baselines are stored in `target/criterion/<benchmark-name>/base/`.

### Comparing Against Baselines

```bash
# Compare current results against "main" baseline
cargo bench --bench criterion_benches --no-default-features --features avx2 -- --baseline main
```

### Listing Baselines

```bash
# List all saved baselines
ls target/criterion/*/base/
```

### Deleting Baselines

```bash
# Delete a specific baseline
rm -rf target/criterion/*/main

# Delete all baselines
cargo clean -p criterion
```

### Recommended Baseline Strategy

1. **main baseline:** Save main branch results before starting work
2. **release baselines:** Save results for each release (e.g., v0.1.0, v0.2.0)
3. **feature baselines:** Save results before major optimizations

Example workflow:
```bash
# Before starting optimization work
git checkout main
cargo bench --bench criterion_benches --no-default-features --features avx2 -- --save-baseline main

# Switch to feature branch and optimize
git checkout optimize-exp2

# Compare after changes
cargo bench --bench criterion_benches --no-default-features --features avx2 -- --baseline main
```

---

## Optimization Workflow

### Step-by-Step Optimization Process

#### 1. Identify Bottlenecks

Run benchmarks to find slow functions:

```bash
cargo bench --bench criterion_benches --no-default-features --features scalar
```

Look for functions with high execution times (e.g., > 50 ns).

#### 2. Establish Baseline

Save current performance:

```bash
cargo bench --bench criterion_benches --no-default-features --features avx2 -- --save-baseline before-opt
```

#### 3. Profile with Flamegraph

Generate a flamegraph to see where time is spent:

```bash
# From repository root
cargo flamegraph --bench criterion_benches --no-default-features --features avx2
```

This creates `flamegraph.svg` showing function call stacks and time distribution.

#### 4. Optimize Code

Make targeted improvements based on profiling data.

#### 5. Benchmark Again

```bash
cargo bench --bench criterion_benches --no-default-features --features avx2 -- --baseline before-opt
```

Look for green "Performance has improved" messages.

#### 6. Validate with iai-callgrind

Ensure instruction counts also improved:

```bash
cargo bench --bench iai_benches --no-default-features --features avx2
```

#### 7. Test All Backends

Verify optimization works across all backends:

```bash
# Scalar
cargo bench --bench criterion_benches --no-default-features --features scalar

# AVX2
RUSTFLAGS="-C target-feature=+avx2,+fma" cargo bench --bench criterion_benches --no-default-features --features avx2

# AVX512
RUSTFLAGS="-C target-feature=+avx512f" cargo bench --bench criterion_benches --no-default-features --features avx512
```

### Common Optimization Techniques

#### 1. **Reduce Instruction Count**

Replace expensive operations:
```rust
// Before: Division (slow)
let result = x / y;

// After: Reciprocal + multiply (faster)
let result = x * recip(y);
```

#### 2. **Minimize Memory Accesses**

Reuse loaded values:
```rust
// Before: Multiple loads
for i in 0..len {
    result[i] = data[i] + data[i];
}

// After: Load once, reuse
for i in 0..len {
    let val = data[i];
    result[i] = val + val;
}
```

#### 3. **Use FMA Instructions**

Fused multiply-add is faster than separate ops:
```rust
// Before: Separate multiply and add
let result = a * b + c;

// After: FMA intrinsic
let result = a.fma(b, c); // SimdVector::fma
```

#### 4. **Align Data**

Ensure SIMD loads/stores are aligned:
```rust
#[repr(C, align(32))] // AVX2 alignment
struct AlignedBuffer {
    data: [f32; 1024],
}
```

#### 5. **Loop Unrolling**

Reduce loop overhead:
```rust
// Before: Single iteration
for i in 0..len {
    process(data[i]);
}

// After: Unroll 4x
for i in (0..len).step_by(4) {
    process(data[i]);
    process(data[i+1]);
    process(data[i+2]);
    process(data[i+3]);
}
```

### Profiling Tools

#### macOS: Instruments

```bash
# Run benchmarks under Instruments
cargo instruments --bench criterion_benches --no-default-features --features neon -- --bench
```

#### Linux: perf

```bash
# Profile with perf
perf record cargo bench --bench criterion_benches --no-default-features --features avx2
perf report
```

#### Cross-Platform: Flamegraph

```bash
# Generate flamegraph
cargo flamegraph --bench criterion_benches --no-default-features --features avx2

# View flamegraph.svg in browser
firefox flamegraph.svg
```

---

## Benchmark Execution Times

### Criterion Benchmarks

| Backend | Benchmark Count | Execution Time | Notes |
|---------|----------------|----------------|-------|
| Scalar | ~50 functions | ~2-3 minutes | Baseline |
| AVX2 | ~50 functions | ~2-3 minutes | Similar to scalar |
| AVX512 | ~50 functions | ~2-3 minutes | Similar to scalar |
| NEON | ~50 functions | ~2-3 minutes | Similar to scalar |

**Total time for all backends:** ~8-12 minutes

### iai-callgrind Benchmarks

| Backend | Benchmark Count | Execution Time | Notes |
|---------|----------------|----------------|-------|
| Scalar | ~50 functions | ~20-30 minutes | 10x slower than Criterion |
| AVX2 | ~50 functions | ~20-30 minutes | 10x slower than Criterion |

**Total time for both backends:** ~40-60 minutes

### Recommendations

- **During development:** Use Criterion for fast feedback (~3 min/backend)
- **Before PR:** Run Criterion for all backends (~12 min total)
- **Pre-release:** Run iai-callgrind for regression detection (~60 min total)

---

## FAQ

**Q: Why does Criterion run each benchmark multiple times?**

A: Criterion uses statistical sampling to measure mean time and confidence intervals, filtering out noise.

**Q: Can I reduce benchmark time for faster feedback?**

A: Yes, use `--bench` flag to run each benchmark only once (no statistical analysis):
```bash
cargo bench --bench criterion_benches --no-default-features --features scalar -- --bench
```

**Q: What's a good speedup target for SIMD?**

A: Aim for 2-4x for AVX2, 4-8x for AVX512, 1.5-3x for NEON. Actual speedup depends on algorithm complexity and memory access patterns.

**Q: Why do iai-callgrind and Criterion show different speedups?**

A: iai-callgrind counts instructions (architecture-independent), while Criterion measures wall-clock time (affected by CPU frequency, cache, pipelining). Both are valid metrics for different purposes.

**Q: Should I optimize for instruction count or wall-clock time?**

A: Optimize for wall-clock time (Criterion) since that's what users experience. Use iai-callgrind to detect regressions in CI.

**Q: Can I run benchmarks in parallel?**

A: No, benchmarks must run sequentially for accurate timing. Criterion handles this automatically.

**Q: How do I benchmark only a subset of functions?**

A: Use Criterion's filter syntax:
```bash
cargo bench --bench criterion_benches --no-default-features --features avx2 "fast_exp"
```

---

## Additional Resources

- **Testing Guide:** [`docs/testing.md`](./testing.md)
- **Coverage Guide:** [`docs/coverage.md`](./coverage.md)
- **Development Workflow:** [`DEVELOPMENT.md`](../DEVELOPMENT.md)
- **Root Benchmarking Guide:** [`/docs/benchmarking.md`](/docs/benchmarking.md) (rigel-dsp specific)

---

*Last updated: 2025-11-22*
