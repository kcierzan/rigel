# rigel-math Development Guide

Complete workflow guide for developing and contributing to `rigel-math`.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Development Environment](#development-environment)
3. [Pre-Commit Checklist](#pre-commit-checklist)
4. [Local Validation Workflow](#local-validation-workflow)
5. [Adding New Features](#adding-new-features)
6. [Common Pitfalls](#common-pitfalls)
7. [CI Testing Limitations](#ci-testing-limitations)
8. [Release Process](#release-process)

---

## Quick Start

### Clone and Setup

```bash
# Clone repository
git clone https://github.com/kcierzan/rigel.git
cd rigel

# Enter development shell (requires Nix + devenv)
devenv shell

# Navigate to rigel-math
cd projects/rigel-synth/crates/math
```

### Verify Environment

```bash
# Check Rust version
cargo --version  # Should be 1.91.0+

# Check CPU capabilities
lscpu | grep -E "avx2|avx512"

# Run tests
cargo test --no-default-features --features scalar
```

---

## Development Environment

### Required Tools

All tools are provided by the `devenv` shell:

- **Rust 1.91.0+** (workspace toolchain)
- **cargo-llvm-cov** (coverage)
- **cargo-flamegraph** (profiling)
- **criterion** (benchmarking)
- **iai-callgrind** (deterministic benchmarking)

### Editor Setup

**VS Code:**
```json
{
  "rust-analyzer.cargo.features": ["scalar"],
  "rust-analyzer.checkOnSave.command": "clippy",
  "rust-analyzer.checkOnSave.extraArgs": ["--", "-D", "warnings"]
}
```

**Vim/Neovim:**
```lua
-- Using rust-tools.nvim or rustaceanvim
{
  cargo = {
    features = { "scalar" }
  }
}
```

---

## Pre-Commit Checklist

**Run these commands before every commit:**

###  1. Format Code

```bash
cargo fmt
```

### 2. Lint Code

```bash
cargo clippy --no-default-features --features scalar -- -D warnings
```

Fix all warnings. Common issues:
- Unused variables: Remove or prefix with `_`
- Unnecessary clones: Use references where possible
- Missing documentation: Add rustdoc comments to public items

### 3. Run Tests (Scalar Backend)

```bash
cargo test --no-default-features --features scalar
```

**Must pass** (310 tests).

### 4. Run Tests (Available SIMD Backends)

```bash
# AVX2 (if CPU supports)
RUSTFLAGS="-C target-feature=+avx2,+fma" cargo test --no-default-features --features avx2

# AVX512 (if CPU supports)
RUSTFLAGS="-C target-feature=+avx512f" cargo test --no-default-features --features avx512

# NEON (on ARM64/macOS)
cargo test --no-default-features --features neon
```

### 5. Run Documentation Tests

**Using devenv (automated workaround):**
```bash
# From repository root - devenv scripts handle specs directory automatically
cargo:test
```

**Manual cargo commands (manual workaround required):**

If running cargo commands directly without devenv scripts, you need to handle the specs directory conflict manually (see [testing.md](docs/testing.md) for details):

```bash
# From repository root
cd /home/kylecierzan/src/rigel

# Move specs directory
mv specs specs.bak

# Run doctests
cd projects/rigel-synth/crates/math
cargo test --doc --no-default-features --features scalar

# Restore specs directory
cd /home/kylecierzan/src/rigel
mv specs.bak specs
```

**Shortcut (one-liner):**
```bash
cd /home/kylecierzan/src/rigel && mv specs specs.bak && cd projects/rigel-synth/crates/math && cargo test --doc --no-default-features --features scalar; cd /home/kylecierzan/src/rigel && mv specs.bak specs
```

### 6. Property-Based Tests (Extended)

```bash
PROPTEST_CASES=10000 cargo test --lib --no-default-features --features avx2
```

**Should pass** without failures. If fails, investigate the failing test case and add regression tests.

### 7. Verify No Performance Regressions

```bash
# Save baseline before changes
cargo bench --bench criterion_benches --no-default-features --features avx2 -- --save-baseline main

# Make changes...

# Compare against baseline
cargo bench --bench criterion_benches --no-default-features --features avx2 -- --baseline main
```

Look for green "Performance has improved" or yellow "No change" messages. **Red "Performance has regressed"** requires investigation.

---

## Local Validation Workflow

### Full Validation (Before PR)

Run this complete suite before opening a pull request:

```bash
#!/usr/bin/env bash
# save as: scripts/validate.sh

set -euo pipefail

echo "=== rigel-math Full Validation ==="

# 1. Format
echo "Step 1/8: Format"
cargo fmt --check || { echo "❌ Format failed. Run 'cargo fmt'"; exit 1; }

# 2. Clippy
echo "Step 2/8: Clippy"
cargo clippy --no-default-features --features scalar -- -D warnings || { echo "❌ Clippy failed"; exit 1; }

# 3. Unit tests - Scalar
echo "Step 3/8: Unit tests (scalar)"
cargo test --lib --no-default-features --features scalar || { echo "❌ Scalar tests failed"; exit 1; }

# 4. Unit tests - AVX2 (if available)
if lscpu | grep -q avx2; then
    echo "Step 4/8: Unit tests (AVX2)"
    RUSTFLAGS="-C target-feature=+avx2,+fma" cargo test --lib --no-default-features --features avx2 || { echo "❌ AVX2 tests failed"; exit 1; }
else
    echo "Step 4/8: Skipped (AVX2 not supported)"
fi

# 5. Unit tests - AVX512 (if available)
if lscpu | grep -q avx512; then
    echo "Step 5/8: Unit tests (AVX512)"
    RUSTFLAGS="-C target-feature=+avx512f" cargo test --lib --no-default-features --features avx512 || { echo "❌ AVX512 tests failed"; exit 1; }
else
    echo "Step 5/8: Skipped (AVX512 not supported)"
fi

# 6. Documentation tests (with specs workaround)
echo "Step 6/8: Documentation tests"
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"
if [ -d "specs" ]; then
    mv specs specs.bak
    RESTORE_SPECS=1
else
    RESTORE_SPECS=0
fi

cd projects/rigel-synth/crates/math
cargo test --doc --no-default-features --features scalar || { echo "❌ Doctests failed"; exit 1; }

cd "$REPO_ROOT"
if [ "$RESTORE_SPECS" -eq 1 ]; then
    mv specs.bak specs
fi

# 7. Property tests (extended)
echo "Step 7/8: Property tests (10k cases)"
PROPTEST_CASES=10000 cargo test --lib --no-default-features --features scalar || { echo "❌ Property tests failed"; exit 1; }

# 8. Coverage
echo "Step 8/8: Coverage"
cd "$REPO_ROOT"
./ci/scripts/measure-coverage.sh || { echo "❌ Coverage failed"; exit 1; }

echo "✅ All validation steps passed!"
```

**Usage:**
```bash
chmod +x scripts/validate.sh
./scripts/validate.sh
```

---

## Adding New Features

### Step-by-Step Process

#### 1. Write Tests First (TDD)

Create a test for the new feature **before** implementing it:

```rust
// tests/new_feature_test.rs
#[cfg(test)]
mod tests {
    use rigel_math::*;

    #[test]
    fn test_new_feature() {
        let input = DefaultSimdVector::splat(1.0);
        let result = new_feature(input);
        // Assert expected behavior
        assert!((result.extract(0) - expected).abs() < 1e-6);
    }
}
```

Run the test (it should fail):
```bash
cargo test test_new_feature --no-default-features --features scalar
```

#### 2. Implement the Feature

Add implementation in appropriate module (e.g., `src/math/new_feature.rs`):

```rust
use crate::SimdVector;

pub fn new_feature<V: SimdVector>(x: V) -> V {
    // Implementation here
    x.mul(V::splat(2.0))
}
```

#### 3. Add Documentation

Add rustdoc comments with error bounds and usage examples:

```rust
/// Computes new_feature(x) using <algorithm details>.
///
/// # Error Bounds
/// - **SIMD (polynomial):** < 0.1% error for x in [0, 10]
/// - **Scalar (libm):** Machine precision
///
/// # Performance
/// - **Scalar:** ~5 ns/operation
/// - **AVX2:** ~3 ns/operation (expected 2x speedup)
///
/// # Examples
/// ```
/// use rigel_math::{DefaultSimdVector, SimdVector, new_feature};
///
/// let x = DefaultSimdVector::splat(1.0);
/// let result = new_feature(x);
/// ```
pub fn new_feature<V: SimdVector>(x: V) -> V {
    // ...
}
```

#### 4. Run Tests

```bash
cargo test --no-default-features --features scalar
```

#### 5. Add Property-Based Tests

Add proptest for invariant validation:

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_new_feature_invariant(x in -10.0f32..10.0f32) {
        let result = new_feature(ScalarVector(x));
        // Validate invariant
        prop_assert!(result.0 > 0.0);
    }
}
```

#### 6. Add Benchmarks

Add to `benches/criterion_benches.rs`:

```rust
fn bench_new_feature(c: &mut Criterion) {
    c.bench_function("new_feature/scalar", |b| {
        let x = DefaultSimdVector::splat(1.0);
        b.iter(|| new_feature(black_box(x)))
    });
}

criterion_group!(new_feature_group, bench_new_feature);
```

Run benchmarks:
```bash
cargo bench --bench criterion_benches --no-default-features --features scalar
```

#### 7. Verify All Backends

Test on all available backends:

```bash
cargo test --no-default-features --features scalar new_feature
RUSTFLAGS="-C target-feature=+avx2,+fma" cargo test --no-default-features --features avx2 new_feature
RUSTFLAGS="-C target-feature=+avx512f" cargo test --no-default-features --features avx512 new_feature
```

#### 8. Generate Coverage

```bash
./ci/scripts/measure-coverage.sh
firefox coverage/avx2/html/index.html
```

Verify your new feature has > 90% line coverage.

#### 9. Update Documentation

Add entry to:
- `docs/api-reference.md` (error bounds, performance)
- `README.md` (if major feature)
- `CHANGELOG.md` (if applicable)

#### 10. Commit

```bash
git add .
git commit -m "Add new_feature with <0.1% error bound"
```

---

## Common Pitfalls

### 1. **Forgetting Feature Flags**

**Problem:**
```bash
cargo test  # ❌ Uses default features
```

**Solution:**
```bash
cargo test --no-default-features --features scalar  # ✅ Explicit backend
```

### 2. **GCC Specs Directory Conflict**

**Problem:**
```
gcc: fatal error: cannot read spec file './specs': Is a directory
```

**Solution (Automatic):** Use devenv scripts (`cargo:test`, `build:*`) which automatically handle this.

**Solution (Manual):** If running cargo commands directly, rename `specs/` directory before running commands (see [testing.md](docs/testing.md) for details).

### 3. **Mixing Backends in Tests**

**Problem:**
```rust
#[test]
fn test_mixed_backends() {
    #[cfg(feature = "scalar")]
    let x = ScalarVector(1.0);

    #[cfg(feature = "avx2")]
    let x = Avx2Vector::splat(1.0);  // Won't work - features are mutually exclusive
}
```

**Solution:** Write tests that use `DefaultSimdVector` (works for all backends).

### 4. **Ignoring Proptest Failures**

**Problem:**
```
proptest: Test failed: assertion failed
proptest: minimal failing case: x = 1.23e-45
```

**Solution:**
1. Note the failing value
2. Add regression test:
   ```rust
   #[test]
   fn test_edge_case_denormal() {
       let x = ScalarVector(1.23e-45);
       let result = my_function(x);
       // Assert correct behavior
   }
   ```
3. Fix implementation to handle edge case
4. Re-run proptest

### 5. **Not Checking Performance Regressions**

**Problem:** Code changes cause 20% slowdown but tests still pass.

**Solution:** Always run benchmarks with baseline comparison:
```bash
cargo bench --no-default-features --features avx2 -- --baseline main
```

### 6. **Incorrect Error Bounds**

**Problem:** Documenting < 0.01% error but actual error is 0.5%.

**Solution:** Validate error bounds with proptest:
```rust
proptest! {
    #[test]
    fn test_error_bound(x in -10.0f32..10.0f32) {
        let result = my_approx(ScalarVector(x));
        let reference = libm::my_reference(x);
        let error = ((result.0 - reference) / reference).abs();
        prop_assert!(error < 0.005);  // Verify < 0.5% error
    }
}
```

---

## CI Testing Limitations

### What CI Tests

| Backend | Platform | Runner | Status |
|---------|----------|--------|--------|
| Scalar | Linux x86_64 | `ubuntu-latest` | ✅ Tested |
| AVX2 | Linux x86_64 | `ubuntu-latest` | ✅ Tested |
| NEON | macOS ARM64 | `macos-latest` | ✅ Tested |
| AVX512 | N/A | N/A | ❌ **Not tested** |

### What CI Doesn't Test

1. **AVX512 backend:** No GitHub runners with AVX512 support
2. **Coverage reports:** Not uploaded to Codecov yet (planned)

### Workarounds

**AVX512 testing:**
- Must test locally on AVX512-capable hardware
- Verify before merging PRs affecting AVX512 code
- Consider self-hosted runner (future)

**Note:** Documentation tests now work in CI with automated specs directory workaround.

---

## Release Process

### Pre-Release Checklist

1. **Run full validation:**
   ```bash
   ./scripts/validate.sh
   ```

2. **Extended property tests:**
   ```bash
   PROPTEST_CASES=100000 cargo test --lib --no-default-features --features avx2
   ```

3. **Benchmark all backends:**
   ```bash
   cargo bench --bench criterion_benches --no-default-features --features scalar
   cargo bench --bench criterion_benches --no-default-features --features avx2
   cargo bench --bench criterion_benches --no-default-features --features avx512
   ```

4. **Update version in `Cargo.toml`:**
   ```toml
   [package]
   version = "0.2.0"
   ```

5. **Update CHANGELOG.md:**
   ```markdown
   ## [0.2.0] - 2025-11-22
   ### Added
   - New feature X with < 0.1% error bound
   - Benchmark suite for Y

   ### Changed
   - Improved performance of Z by 20%

   ### Fixed
   - Edge case in W causing incorrect results
   ```

6. **Commit and tag:**
   ```bash
   git add Cargo.toml CHANGELOG.md
   git commit -m "Release v0.2.0"
   git tag v0.2.0
   git push origin main v0.2.0
   ```

7. **Publish to crates.io:**
   ```bash
   cargo publish
   ```

---

## Debugging Tips

### Viewing SIMD Assembly

```bash
# Generate assembly for specific function
cargo rustc --no-default-features --features avx2 --release -- --emit asm

# View assembly
cat target/release/deps/rigel_math-*.s | grep -A 20 my_function
```

### Profiling with Flamegraph

```bash
cargo flamegraph --bench criterion_benches --no-default-features --features avx2
firefox flamegraph.svg
```

### Checking SIMD Code Generation

```bash
# Install cargo-asm
cargo install cargo-asm

# View assembly for function
cargo asm --no-default-features --features avx2 rigel_math::math::exp --rust
```

Look for AVX2 instructions (`vaddps`, `vmulps`, `vfmadd231ps`).

### Testing Single Function

```bash
cargo test --no-default-features --features scalar test_specific_function -- --nocapture
```

---

## Additional Resources

- **Testing Guide:** [`docs/testing.md`](docs/testing.md)
- **Benchmarking Guide:** [`docs/benchmarking.md`](docs/benchmarking.md)
- **Coverage Guide:** [`docs/coverage.md`](docs/coverage.md)
- **API Reference:** [`docs/api-reference.md`](docs/api-reference.md)
- **Main README:** [`README.md`](README.md)
- **Root CLAUDE.md:** [`/CLAUDE.md`](/CLAUDE.md)

---

## Getting Help

**Issues or Questions?**
1. Check existing documentation (this file, guides, README)
2. Search GitHub issues: https://github.com/kcierzan/rigel/issues
3. Open a new issue with reproduction steps

**Contributing:**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-math-function`)
3. Follow this development workflow
4. Open a pull request with clear description

---

*Last updated: 2025-11-22*
