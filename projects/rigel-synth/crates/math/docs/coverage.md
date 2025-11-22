# rigel-math Coverage Guide

This guide provides comprehensive instructions for generating and interpreting code coverage reports for `rigel-math`.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Using the Coverage Script](#using-the-coverage-script)
4. [Manual Coverage Generation](#manual-coverage-generation)
5. [Interpreting HTML Coverage Reports](#interpreting-html-coverage-reports)
6. [Critical Path Coverage Analysis](#critical-path-coverage-analysis)
7. [Coverage Targets](#coverage-targets)
8. [Addressing Coverage Gaps](#addressing-coverage-gaps)
9. [CI Integration (Future Work)](#ci-integration-future-work)

---

## Overview

Code coverage measures which lines and branches of code are executed during tests. `rigel-math` uses:

- **Tool:** `cargo-llvm-cov` (LLVM-based coverage tool)
- **Metrics:** Line coverage, function coverage, region coverage
- **Output:** HTML reports with syntax-highlighted source code
- **Strategy:** Per-backend coverage (scalar, AVX2, AVX512, NEON)

### Why Per-Backend Coverage?

Each SIMD backend has different implementations behind compile-time feature flags. Testing one backend doesn't exercise code for other backends, so we measure coverage separately for each.

---

## Quick Start

```bash
# From repository root

# Generate coverage for all backends
./ci/scripts/measure-coverage.sh

# View AVX2 coverage report
firefox coverage/avx2/html/index.html

# View scalar coverage report
firefox coverage/scalar/html/index.html
```

The script generates HTML reports in:
- `coverage/scalar/html/index.html`
- `coverage/avx2/html/index.html`
- `coverage/avx512/html/index.html` (if AVX512 supported)
- `coverage/neon/html/index.html` (on ARM64/macOS)

---

## Using the Coverage Script

The `ci/scripts/measure-coverage.sh` script automates coverage generation for all backends.

### Script Location

```bash
/home/kylecierzan/src/rigel/ci/scripts/measure-coverage.sh
```

### How It Works

1. **Automatically handles GCC specs directory workaround** (renames `specs/` to `specs.tmp` during execution)
2. Detects available SIMD backends based on CPU capabilities
3. Runs tests for each backend with coverage instrumentation
4. Generates HTML reports in `coverage/<backend>/html/`
5. Outputs summary statistics to console
6. Restores `specs/` directory even on errors

### Running the Script

```bash
# From repository root
./ci/scripts/measure-coverage.sh
```

### Expected Output

```
=== Measuring Coverage for rigel-math ===

Testing scalar backend...
   Compiling rigel-math v0.1.0
    Finished `test` profile [unoptimized + debuginfo] target(s) in 3.2s
     Running unittests src/lib.rs
test result: ok. 310 passed; 0 failed; 1 ignored
Generating HTML report...
Coverage report: coverage/scalar/html/index.html

Testing avx2 backend...
   Compiling rigel-math v0.1.0
    Finished `test` profile [unoptimized + debuginfo] target(s) in 3.4s
     Running unittests src/lib.rs
test result: ok. 310 passed; 0 failed; 1 ignored
Generating HTML report...
Coverage report: coverage/avx2/html/index.html

=== Coverage Summary ===
Scalar: 96.3% lines, 91.2% functions
AVX2:   96.3% lines, 91.2% functions

View reports:
  firefox coverage/scalar/html/index.html
  firefox coverage/avx2/html/index.html
```

### Script Features

- **Automatic backend detection:** Only tests backends your CPU supports
- **Parallel execution:** Tests backends sequentially but efficiently
- **Preserved output:** HTML reports remain after script completes
- **Error handling:** Stops on compilation errors

---

## Manual Coverage Generation

For more control, generate coverage manually:

### Scalar Backend

```bash
cd projects/rigel-synth/crates/math

# Run tests with coverage
cargo llvm-cov \
    --no-default-features \
    --features scalar \
    --html \
    --output-dir ../../coverage/scalar \
    test

# View report
firefox ../../coverage/scalar/html/index.html
```

### AVX2 Backend

```bash
cd projects/rigel-synth/crates/math

# Run tests with coverage
cargo llvm-cov \
    --no-default-features \
    --features avx2 \
    --html \
    --output-dir ../../coverage/avx2 \
    test

# View report
firefox ../../coverage/avx2/html/index.html
```

### AVX512 Backend

```bash
cd projects/rigel-synth/crates/math

# Run tests with coverage
RUSTFLAGS="-C target-feature=+avx512f" \
cargo llvm-cov \
    --no-default-features \
    --features avx512 \
    --html \
    --output-dir ../../coverage/avx512 \
    test

# View report
firefox ../../coverage/avx512/html/index.html
```

### NEON Backend (ARM64/macOS)

```bash
cd projects/rigel-synth/crates/math

# Run tests with coverage
cargo llvm-cov \
    --no-default-features \
    --features neon \
    --html \
    --output-dir ../../coverage/neon \
    test

# View report
open ../../coverage/neon/html/index.html
```

### Text-Only Output (No HTML)

For quick feedback without HTML:

```bash
cargo llvm-cov \
    --no-default-features \
    --features avx2 \
    test
```

Output:
```
Filename                      Regions    Missed Regions     Cover   Functions  Missed Functions  Executed       Lines      Missed Lines     Cover    Branches   Missed Branches     Cover
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
src/block.rs                       45                 2    95.56%          12                 0   100.00%          89                 3    96.63%           0                 0         -
src/math/exp.rs                    38                 1    97.37%           8                 0   100.00%         102                 1    99.02%           0                 0         -
...
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
TOTAL                             842                31    96.32%         187                16    91.44%        2134                78    96.35%           0                 0         -
```

---

## Interpreting HTML Coverage Reports

### Opening the Report

```bash
firefox coverage/avx2/html/index.html
```

### Report Structure

#### 1. **Summary Page** (`index.html`)

Shows overall coverage statistics:

| Metric | Description | Target |
|--------|-------------|--------|
| **Lines** | % of lines executed | > 90% |
| **Functions** | % of functions called | > 85% |
| **Regions** | % of code regions executed | > 90% |
| **Branches** | % of conditional branches taken | > 80% (if available) |

**Interpret the summary:**
- **Green (90-100%):** Excellent coverage
- **Yellow (70-90%):** Acceptable but improvable
- **Red (<70%):** Needs more tests

#### 2. **File List**

Click on a filename to see detailed coverage for that file.

**Color coding:**
- **Dark green:** 100% coverage
- **Light green:** 90-99% coverage
- **Yellow:** 70-89% coverage
- **Orange:** 50-69% coverage
- **Red:** <50% coverage

#### 3. **Source Code View**

Each source file shows:
- **Line numbers**
- **Execution counts** (how many times each line was executed)
- **Highlighted code:**
  - **Green background:** Line was executed
  - **Red background:** Line was NOT executed
  - **No highlight:** Non-executable line (comments, blank lines)

### Example Interpretation

```rust
 42 | pub fn fast_exp2(x: f32) -> f32 {               // Green: executed 1000+ times
 43 |     if x.is_nan() {                            // Green: executed
 44 |         return f32::NAN;                       // Red: never executed (no NaN inputs in tests)
 45 |     }                                          // Green: executed
 46 |     // ... rest of function
 47 | }
```

**Analysis:** Line 44 is uncovered because tests don't pass `NaN` inputs. Add a test:

```rust
#[test]
fn test_fast_exp2_nan() {
    assert!(fast_exp2(f32::NAN).is_nan());
}
```

---

## Critical Path Coverage Analysis

### What is a Critical Path?

A **critical path** is code that executes in the hot loop of real-time audio processing. For `rigel-math`, critical paths are:

- SIMD vector operations (`ops/*`)
- Fast math functions (`math/exp.rs`, `math/tanh.rs`, etc.)
- Lookup table interpolation (`table.rs`)
- Audio block processing (`block.rs`)

### Non-Critical Paths

Less critical code includes:
- Error handling for invalid inputs
- Debugging utilities
- Rarely-used edge cases (e.g., handling infinity)

### Coverage Target Priorities

| Module | Priority | Line Coverage Target | Why |
|--------|----------|---------------------|------|
| `ops/*` | **Critical** | > 95% | Used in every audio sample |
| `math/*` | **Critical** | > 95% | DSP algorithms depend on accuracy |
| `table.rs` | **Critical** | > 95% | Wavetable oscillators use this |
| `block.rs` | **Critical** | > 90% | Block processing infrastructure |
| `traits.rs` | **High** | > 90% | Core abstractions |
| `crossfade.rs` | **Medium** | > 85% | Used for parameter ramping |
| `denormal.rs` | **Medium** | > 85% | Platform-specific edge case handling |

### Actual Coverage Results

From `coverage/ANALYSIS.md` (AVX2 backend):

| Module | Lines Covered | Total Lines | Coverage | Status |
|--------|---------------|-------------|----------|--------|
| `ops/arithmetic.rs` | 189/196 | 196 | **96.4%** | ✅ Excellent |
| `math/exp.rs` | 101/102 | 102 | **99.0%** | ✅ Excellent |
| `math/tanh.rs` | 67/68 | 68 | **98.5%** | ✅ Excellent |
| `table.rs` | 142/148 | 148 | **95.9%** | ✅ Excellent |
| `block.rs` | 86/89 | 89 | **96.6%** | ✅ Excellent |
| `traits.rs` | 234/243 | 243 | **96.3%** | ✅ Excellent |
| **TOTAL (Critical)** | 819/846 | 846 | **96.8%** | ✅ Exceeds target |

**Conclusion:** All critical paths exceed the 90% target, with most above 95%. Non-critical error handling accounts for most gaps.

---

## Coverage Targets

### Overall Targets

| Metric | Target | Actual (AVX2) | Status |
|--------|--------|---------------|--------|
| **Line Coverage** | > 90% | 96.3% | ✅ Pass |
| **Function Coverage** | > 85% | 91.2% | ✅ Pass |
| **Critical Path Coverage** | > 95% | 96.8% | ✅ Pass |

### Per-Module Targets

**Critical modules:**
- `ops/*`: 95%+
- `math/*`: 95%+
- `table.rs`: 95%+
- `block.rs`: 90%+

**Non-critical modules:**
- `crossfade.rs`: 85%+
- `denormal.rs`: 85%+
- `noise.rs`: 80%+

### Backend Consistency

All backends (scalar, AVX2, AVX512, NEON) should have similar coverage percentages because they share the same test suite.

**Expected variance:** ±2% between backends

**Example:**
- Scalar: 96.3% ± 2%
- AVX2: 96.3% ± 2%
- AVX512: 96.0% ± 2% (acceptable)
- NEON: 95.8% ± 2% (acceptable)

---

## Addressing Coverage Gaps

### Identifying Gaps

1. **Generate coverage report:**
   ```bash
   cargo llvm-cov --no-default-features --features avx2 --html --output-dir coverage/avx2 test
   ```

2. **Open HTML report:**
   ```bash
   firefox coverage/avx2/html/index.html
   ```

3. **Sort files by coverage** (click "Lines" column header)

4. **Click on files with <90% coverage**

5. **Examine red-highlighted lines**

### Classifying Gaps

For each uncovered line, determine:

**A. Critical gap** (needs tests):
- Main code path in hot loop
- Error handling that could realistically trigger
- Documented behavior or public API

**B. Non-critical gap** (acceptable):
- Debug-only code paths
- Impossible edge cases (e.g., `unreachable!()`)
- Platform-specific code not applicable to current backend

### Writing Tests to Fill Gaps

#### Example 1: Uncovered Error Handling

**Uncovered code:**
```rust
pub fn log(x: f32) -> f32 {
    if x <= 0.0 {
        return f32::NEG_INFINITY;  // Red: never tested
    }
    // ... implementation
}
```

**Add test:**
```rust
#[test]
fn test_log_negative() {
    assert_eq!(log(-1.0), f32::NEG_INFINITY);
    assert_eq!(log(0.0), f32::NEG_INFINITY);
}
```

#### Example 2: Uncovered Edge Case

**Uncovered code:**
```rust
pub fn normalize(values: &mut [f32]) {
    let max = values.iter().copied().fold(0.0f32, f32::max);
    if max == 0.0 {
        return;  // Red: never tested
    }
    for val in values {
        *val /= max;
    }
}
```

**Add test:**
```rust
#[test]
fn test_normalize_all_zeros() {
    let mut values = [0.0; 10];
    normalize(&mut values);
    assert_eq!(values, [0.0; 10]);  // Should remain unchanged
}
```

#### Example 3: Uncovered Branch

**Uncovered code:**
```rust
pub fn clamp(x: f32, min: f32, max: f32) -> f32 {
    if x < min {
        min           // Green: tested
    } else if x > max {
        max           // Red: never tested
    } else {
        x             // Green: tested
    }
}
```

**Add test:**
```rust
#[test]
fn test_clamp_above_max() {
    assert_eq!(clamp(15.0, 0.0, 10.0), 10.0);  // Tests the `x > max` branch
}
```

### Prioritizing Gap Fixes

1. **Critical path gaps:** Fix immediately
2. **Public API gaps:** Fix before release
3. **Error handling gaps:** Fix if realistic
4. **Platform-specific gaps:** Accept or mock
5. **Debug-only gaps:** Accept

---

## CI Integration (Future Work)

### Planned CI Coverage Workflow

```yaml
# .github/workflows/coverage.yml (future)
name: Coverage

on: [push, pull_request]

jobs:
  coverage:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Install coverage tools
        run: cargo install cargo-llvm-cov

      - name: Generate coverage
        run: ./ci/scripts/measure-coverage.sh

      - name: Upload to Codecov
        uses: codecov/codecov-action@v3
        with:
          files: coverage/*/html/index.html

      - name: Check coverage threshold
        run: |
          COVERAGE=$(cargo llvm-cov --no-default-features --features avx2 test --json | jq '.data[0].totals.lines.percent')
          if [ $(echo "$COVERAGE < 90" | bc) -eq 1 ]; then
            echo "Coverage $COVERAGE% is below 90% threshold"
            exit 1
          fi
```

### Coverage Badges

Add to README:

```markdown
[![Coverage](https://codecov.io/gh/user/rigel/branch/main/graph/badge.svg)](https://codecov.io/gh/user/rigel)
```

### Per-PR Coverage Comments

GitHub Actions can comment on PRs with coverage deltas:

```
Coverage changed: 96.3% → 95.8% (-0.5%)

⚠️ Coverage decreased on:
- src/math/new_function.rs: 0% (new file, no tests)

Please add tests before merging.
```

---

## Troubleshooting

### `cargo-llvm-cov` Not Found

**Error:**
```
error: no such command: `llvm-cov`
```

**Solution:**
```bash
cargo install cargo-llvm-cov
```

### Coverage Report Shows 0%

**Possible causes:**
1. Tests didn't run (compilation error)
2. Feature flag mismatch
3. Wrong output directory

**Solution:**
```bash
# Ensure tests pass first
cargo test --no-default-features --features avx2

# Then generate coverage
cargo llvm-cov --no-default-features --features avx2 test
```

### Branch Coverage Shows 0/0

**Note:** `cargo-llvm-cov` may not report branch coverage for all code. This is a known limitation.

**Workaround:** Focus on line coverage and region coverage instead.

### HTML Report Not Generated

**Possible causes:**
1. Missing `--html` flag
2. Wrong `--output-dir` path

**Solution:**
```bash
cargo llvm-cov \
    --no-default-features \
    --features avx2 \
    --html \
    --output-dir ../../coverage/avx2 \
    test
```

---

## FAQ

**Q: Why is coverage different between backends?**

A: Each backend has different code paths (e.g., AVX2 intrinsics vs scalar loops). Tests may exercise different branches depending on SIMD capabilities.

**Q: What's a good line coverage percentage?**

A: Aim for > 90% overall, with > 95% for critical paths (ops, math, table).

**Q: Should I aim for 100% coverage?**

A: No. 100% coverage is impractical and often counterproductive. Focus on critical paths and realistic edge cases.

**Q: Why does the script take so long?**

A: Coverage instrumentation adds overhead to test execution. Expect ~2-3x longer than normal test runs.

**Q: Can I exclude files from coverage?**

A: Yes, use `#[cfg(not(tarpaulin_include))]` or similar attributes, but this is rarely needed.

**Q: How do I see coverage for a specific function?**

A: Open the HTML report, navigate to the file containing the function, and scroll to the function. Execution counts appear next to each line.

---

## Additional Resources

- **Testing Guide:** [`docs/testing.md`](./testing.md)
- **Benchmarking Guide:** [`docs/benchmarking.md`](./benchmarking.md)
- **Development Workflow:** [`DEVELOPMENT.md`](../DEVELOPMENT.md)
- **Coverage Analysis:** [`/coverage/ANALYSIS.md`](/coverage/ANALYSIS.md)
- **cargo-llvm-cov Docs:** [https://github.com/taiki-e/cargo-llvm-cov](https://github.com/taiki-e/cargo-llvm-cov)

---

*Last updated: 2025-11-22*
