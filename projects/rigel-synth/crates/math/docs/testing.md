# rigel-math Testing Guide

This guide provides comprehensive instructions for testing the `rigel-math` crate across all SIMD backends.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Running Unit Tests](#running-unit-tests)
3. [Running Documentation Tests](#running-documentation-tests)
4. [Property-Based Testing](#property-based-testing)
5. [CI Testing Status](#ci-testing-status)
6. [Debugging Test Failures](#debugging-test-failures)
7. [Test Execution Times](#test-execution-times)

---

## Quick Start

### Using devenv Scripts (Recommended)

```bash
# Run all tests - devenv automatically handles specs directory
cargo:test
```

The devenv script automatically handles the GCC specs directory conflict and other environment setup.

### Using Direct cargo Commands

```bash
# Run all tests for scalar backend (always works)
cargo test --no-default-features --features scalar

# Run all tests for AVX2 backend (requires AVX2 CPU)
RUSTFLAGS="-C target-feature=+avx2,+fma" cargo test --no-default-features --features avx2

# Run all tests for AVX512 backend (requires AVX512 CPU)
RUSTFLAGS="-C target-feature=+avx512f" cargo test --no-default-features --features avx512

# Run all tests for NEON backend (requires ARM64/Apple Silicon)
cargo test --no-default-features --features neon
```

**⚠️ Important:** If you encounter linker errors about `./specs: Is a directory`, see the [Documentation Tests section](#running-documentation-tests) for the workaround, or use the `cargo:test` devenv script which handles this automatically.

---

## Running Unit Tests

Unit tests validate individual functions and modules. They are located in:
- `src/*.rs` - Inline tests in each module
- `tests/` - Integration tests

### Basic Commands

```bash
# Run ONLY unit tests (skip doctests) - scalar backend
cargo test --lib --no-default-features --features scalar

# Run ONLY unit tests - AVX2 backend
RUSTFLAGS="-C target-feature=+avx2,+fma" cargo test --lib --no-default-features --features avx2

# Run ONLY unit tests - AVX512 backend
RUSTFLAGS="-C target-feature=+avx512f" cargo test --lib --no-default-features --features avx512

# Run ONLY unit tests - NEON backend
cargo test --lib --no-default-features --features neon
```

### Running Specific Test Modules

```bash
# Run tests from a specific module
cargo test --lib --no-default-features --features avx2 ops::arithmetic

# Run a specific test function
cargo test --lib --no-default-features --features scalar test_add_scalar

# Run tests matching a pattern
cargo test --lib --no-default-features --features avx2 simd
```

### Verbose Output

```bash
# Show test output even for passing tests
cargo test --lib --no-default-features --features scalar -- --show-output

# Show full test names and progress
cargo test --lib --no-default-features --features avx2 -- --nocapture
```

---

## Running Documentation Tests

Documentation tests validate code examples in rustdoc comments. There are **93 doctests** in `rigel-math`.

### Automated Workaround (Recommended)

**When using devenv scripts, the specs directory workaround is handled automatically:**

```bash
# From repository root - specs directory handled automatically
cargo:test
```

All devenv scripts (`cargo:test`, `build:*`, `bench:*`) automatically handle the GCC specs directory conflict, so you don't need to manually rename the directory.

### Manual Workaround (For Direct cargo Commands)

If you're running cargo commands directly (not through devenv scripts), you'll need to handle the specs directory conflict manually.

#### GCC Specs Directory Conflict

**Problem:** The `specs/` directory in the repository root conflicts with GCC's spec file lookup, causing linker errors:

```
gcc: fatal error: cannot read spec file './specs': Is a directory
```

**Solution:** Temporarily rename the `specs/` directory before running doctests:

```bash
# From repository root
mv specs specs.bak

# Run doctests
cd projects/rigel-synth/crates/math
cargo test --doc --no-default-features --features scalar

# Restore specs directory
cd /path/to/repo/root
mv specs.bak specs
```

### One-Liner Workarounds

```bash
# Scalar backend (from repo root)
mv specs specs.bak && cd projects/rigel-synth/crates/math && cargo test --doc --no-default-features --features scalar; cd /path/to/repo && mv specs.bak specs

# AVX2 backend (from repo root)
mv specs specs.bak && cd projects/rigel-synth/crates/math && RUSTFLAGS="-C target-feature=+avx2,+fma" cargo test --doc --no-default-features --features avx2; cd /path/to/repo && mv specs.bak specs

# AVX512 backend (from repo root)
mv specs specs.bak && cd projects/rigel-synth/crates/math && RUSTFLAGS="-C target-feature=+avx512f" cargo test --doc --no-default-features --features avx512; cd /path/to/repo && mv specs.bak specs
```

### Helper Script

For convenience, you can create a helper script `scripts/test-doctests.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

# Backup specs directory
if [ -d "specs" ]; then
    mv specs specs.bak
    RESTORE_SPECS=1
else
    RESTORE_SPECS=0
fi

# Run doctests
cd projects/rigel-synth/crates/math
cargo test --doc --no-default-features --features "${1:-scalar}"

# Restore specs directory
cd "$REPO_ROOT"
if [ "$RESTORE_SPECS" -eq 1 ]; then
    mv specs.bak specs
fi
```

Usage:
```bash
# Run scalar doctests
./scripts/test-doctests.sh scalar

# Run AVX2 doctests
RUSTFLAGS="-C target-feature=+avx2,+fma" ./scripts/test-doctests.sh avx2
```

---

## Property-Based Testing

`rigel-math` uses [proptest](https://docs.rs/proptest/) for property-based testing, generating thousands of test cases to validate invariants.

### Default Test Cases

By default, proptest generates **256 test cases** per property. This takes ~5-10 seconds per backend.

```bash
# Run with default 256 cases
cargo test --lib --no-default-features --features scalar
```

### Increasing Test Coverage

For more thorough validation (e.g., before a release), increase the number of test cases:

```bash
# Run with 1,000 test cases (takes ~15-30 seconds)
PROPTEST_CASES=1000 cargo test --lib --no-default-features --features avx2

# Run with 10,000 test cases (takes ~1-2 minutes, recommended for releases)
PROPTEST_CASES=10000 cargo test --lib --no-default-features --features avx2

# Run with 100,000 test cases (takes ~10-15 minutes, exhaustive testing)
PROPTEST_CASES=100000 cargo test --lib --no-default-features --features avx512
```

### Property Test Validation

Property tests validate:
- **Mathematical invariants**: `exp(log(x)) ≈ x`, `sin²(x) + cos²(x) ≈ 1`
- **Error bounds**: `|fast_tanh(x) - libm::tanh(x)| < 0.001`
- **SIMD consistency**: Results match scalar reference within tolerance
- **Edge cases**: Handles `NaN`, `Inf`, denormals, zero, negative zero
- **Monotonicity**: `x < y => log(x) < log(y)` (for positive x, y)

---

## CI Testing Status

### Currently Tested in CI

The `.github/workflows/ci.yml` workflow tests the following backends:

| Backend | Platform | Runner | Status |
|---------|----------|--------|--------|
| **Scalar** | Linux x86_64 | `ubuntu-latest` | ✅ Tested (default features) |
| **AVX2** | Linux x86_64 | `ubuntu-latest` | ✅ Tested (explicit `RUSTFLAGS`) |
| **NEON** | macOS ARM64 | `macos-latest` | ✅ Tested |
| **AVX512** | Linux x86_64 | N/A | ⚠️ **Not tested** (no runner support) |

### CI Test Commands

```yaml
# Scalar (rigel-pipeline job)
cargo test

# AVX2 (backend-tests job)
RUSTFLAGS="-C target-feature=+avx2,+fma" cargo test --features avx2

# NEON (backend-tests job)
cargo test --features neon
```

### Limitations

**AVX512 Testing:**
- **Not tested in CI** due to lack of GitHub-hosted runners with AVX512 support
- Must be tested **locally** on AVX512-capable hardware
- Use `lscpu | grep avx512` to check if your CPU supports AVX512

**Workaround:**
- Test AVX512 locally before merging PRs
- Consider using a self-hosted runner with AVX512 support (future work)

**Documentation Tests:**
- Not currently run in CI due to the `specs/` directory conflict
- Plan to add CI support by temporarily renaming `specs/` (similar to existing CI workaround for builds)

---

## Debugging Test Failures

### Common Issues

#### 1. **Linker Error: `cannot read spec file './specs'`**

**Cause:** GCC is trying to read the repository's `specs/` directory as a spec file.

**Solution:** Rename `specs/` directory temporarily (see [Documentation Tests](#running-documentation-tests)).

#### 2. **Feature Flag Mismatch**

**Cause:** Running tests without the correct backend feature flag.

**Error:**
```
error: The trait `SimdVector` is not implemented for `f32`
```

**Solution:** Always specify `--no-default-features --features <backend>`:
```bash
cargo test --no-default-features --features scalar
```

#### 3. **Missing CPU Features**

**Cause:** Testing AVX2/AVX512 backend on a CPU without support.

**Error:**
```
Illegal instruction (core dumped)
```

**Solution:** Check CPU capabilities:
```bash
# Check for AVX2
lscpu | grep avx2

# Check for AVX512
lscpu | grep avx512

# If not supported, use scalar backend instead
cargo test --no-default-features --features scalar
```

#### 4. **Property Test Failures**

**Cause:** Random test case found a bug or edge case.

**Error:**
```
proptest: Test failed: assertion failed
proptest: minimal failing case: x = 1.23e-45
```

**Solution:**
1. Note the failing input value
2. Add a regression test for that specific value
3. Fix the implementation to handle the edge case
4. Re-run proptest to verify the fix

#### 5. **Floating-Point Comparison Failures**

**Cause:** Exact equality comparisons on floating-point results.

**Error:**
```
assertion `left == right` failed
  left: 0.9999999
 right: 1.0
```

**Solution:** Use approximate comparisons with tolerance:
```rust
assert!((result - expected).abs() < 1e-6);
```

---

## Test Execution Times

### Unit Tests (310 tests)

| Backend | Test Count | Execution Time | Notes |
|---------|-----------|----------------|-------|
| Scalar | 310 | ~0.3s | Baseline performance |
| AVX2 | 310 | ~0.4s | Slightly slower due to SIMD setup |
| AVX512 | 310 | ~0.5s | More complex SIMD operations |
| NEON | 310 | ~0.4s | ARM64 SIMD |

### Documentation Tests (93 tests)

| Backend | Test Count | Execution Time | Notes |
|---------|-----------|----------------|-------|
| Scalar | 93 | ~0.4s | Compilation time dominates |
| AVX2 | 93 | ~0.5s | Requires `specs/` workaround |
| AVX512 | 93 | ~0.6s | Requires `specs/` workaround |

### Property-Based Tests (varies by `PROPTEST_CASES`)

| Cases | Per Backend | Notes |
|-------|-------------|-------|
| 256 (default) | ~5-10s | Quick feedback |
| 1,000 | ~15-30s | Pre-commit validation |
| 10,000 | ~1-2min | Pre-release validation |
| 100,000 | ~10-15min | Exhaustive testing |

### Full Test Suite (unit + doc + proptest with 10,000 cases)

| Backend | Total Time | Command |
|---------|-----------|---------|
| Scalar | ~2-3min | `PROPTEST_CASES=10000 cargo test --no-default-features --features scalar` |
| AVX2 | ~2-4min | `PROPTEST_CASES=10000 RUSTFLAGS="-C target-feature=+avx2,+fma" cargo test --no-default-features --features avx2` |

---

## Pre-Commit Testing Checklist

Before pushing changes, run this checklist:

```bash
# 1. Format code
cargo fmt

# 2. Lint code
cargo clippy -- -D warnings

# 3. Test scalar backend (always works)
cargo test --no-default-features --features scalar

# 4. Test AVX2 backend (if available)
RUSTFLAGS="-C target-feature=+avx2,+fma" cargo test --no-default-features --features avx2

# 5. Test AVX512 backend (if available)
RUSTFLAGS="-C target-feature=+avx512f" cargo test --no-default-features --features avx512

# 6. Run doctests with specs workaround (scalar backend)
mv specs specs.bak && cargo test --doc --no-default-features --features scalar; mv specs.bak specs

# 7. Run property tests with higher case count
PROPTEST_CASES=10000 cargo test --lib --no-default-features --features avx2

# 8. Generate coverage report (see coverage.md)
./ci/scripts/measure-coverage.sh
```

**Minimum for PR approval:**
- Steps 1-3 must pass (format, lint, scalar tests)
- Step 6 (doctests) must pass
- At least one SIMD backend (AVX2 or NEON) must pass

---

## Additional Resources

- **Benchmarking Guide:** [`docs/benchmarking.md`](./benchmarking.md)
- **Coverage Guide:** [`docs/coverage.md`](./coverage.md)
- **Development Workflow:** [`DEVELOPMENT.md`](../DEVELOPMENT.md)
- **API Reference:** [`docs/api-reference.md`](./api-reference.md)
- **Main README:** [`README.md`](../README.md)

---

## Troubleshooting

### Tests Hang or Run Forever

**Cause:** Infinite loop or deadlock in test code.

**Solution:**
- Kill the test with `Ctrl+C`
- Add `--test-threads=1` to run tests sequentially and identify the hanging test:
  ```bash
  cargo test --lib --no-default-features --features scalar -- --test-threads=1
  ```

### Out of Memory During Tests

**Cause:** Proptest generating too many test cases or large SIMD vectors.

**Solution:**
- Reduce `PROPTEST_CASES`:
  ```bash
  PROPTEST_CASES=256 cargo test --lib --no-default-features --features scalar
  ```

### Tests Pass Locally but Fail in CI

**Possible causes:**
1. **Missing feature flag:** CI uses explicit features, local might use defaults
2. **Platform differences:** CI runs on Linux/macOS, local might be different
3. **CPU features:** CI might not have AVX512 support

**Solution:**
- Check CI logs for exact test command
- Run the same command locally
- Verify feature flags match

---

## FAQ

**Q: Why do I need to move the `specs/` directory to run doctests?**

A: GCC searches for a file called `specs` in the current directory to determine compilation settings. The repository has a `specs/` directory for specification documents, which GCC tries to read as a file, causing a linker error. This is a known limitation documented in the CI workflow.

**Q: Can I run tests for multiple backends at once?**

A: No. Each backend requires different compiler flags and features. You must run tests for each backend separately.

**Q: Why are AVX512 tests not in CI?**

A: GitHub-hosted runners don't have AVX512 CPU support. We test AVX512 locally and may add self-hosted runners in the future.

**Q: How do I know if my CPU supports AVX2/AVX512?**

A: Run `lscpu | grep -E "avx2|avx512"` on Linux/macOS. On Windows, use CPU-Z or similar tools.

**Q: What's the difference between `cargo test` and `cargo test --lib`?**

A: `cargo test` runs all tests (unit tests, integration tests, and doctests). `cargo test --lib` runs only unit tests in `src/`, skipping doctests and integration tests.

**Q: Why do property tests take so long?**

A: Proptest generates thousands of random inputs and runs the test for each one. This is intentional for thorough validation. Reduce `PROPTEST_CASES` for faster feedback during development.

---

*Last updated: 2025-11-22*
