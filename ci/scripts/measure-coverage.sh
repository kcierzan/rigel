#!/usr/bin/env bash
set -euo pipefail

# Coverage measurement script for rigel-math
# Measures coverage per backend since --all-features enables multiple backends

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MATH_CRATE="$REPO_ROOT/projects/rigel-synth/crates/math"
COVERAGE_DIR="$REPO_ROOT/coverage"

echo "==================================================================="
echo "rigel-math Coverage Measurement"
echo "==================================================================="
echo "Math crate: $MATH_CRATE"
echo "Coverage output: $COVERAGE_DIR"
echo ""

# Workaround for GCC specs directory conflict
# The specs/ directory confuses GCC, so we temporarily rename it
SPECS_RENAMED=0
if [[ -d "$REPO_ROOT/specs" ]]; then
    echo "Temporarily renaming specs/ to avoid GCC conflict..."
    mv "$REPO_ROOT/specs" "$REPO_ROOT/specs.tmp"
    SPECS_RENAMED=1
fi

# Ensure specs/ is restored on exit
cleanup() {
    if [[ $SPECS_RENAMED -eq 1 && -d "$REPO_ROOT/specs.tmp" ]]; then
        echo "Restoring specs/ directory..."
        mv "$REPO_ROOT/specs.tmp" "$REPO_ROOT/specs"
    fi
}
trap cleanup EXIT

# Clean previous coverage data
rm -rf "$COVERAGE_DIR"
mkdir -p "$COVERAGE_DIR"

# Change to math crate directory
cd "$MATH_CRATE"

# Array to store coverage results
declare -A line_coverage
declare -A branch_coverage
declare -A backends_tested

# Function to run coverage for a backend
run_backend_coverage() {
    local backend="$1"
    local feature="$2"

    echo "-------------------------------------------------------------------"
    echo "Running coverage for backend: $backend (feature: $feature)"
    echo "-------------------------------------------------------------------"

    # Clean previous build artifacts
    cargo llvm-cov clean --workspace

    # Run coverage with specific backend feature and generate HTML report
    if cargo llvm-cov \
        --no-default-features \
        --features "$feature" \
        --html \
        --output-dir "$COVERAGE_DIR/$backend" \
        test; then

        backends_tested["$backend"]=1

        # Generate text summary report from the collected coverage data
        echo ""
        echo "Coverage Summary for $backend:"
        cargo llvm-cov report | tee "$COVERAGE_DIR/$backend/summary.txt"

        echo ""
        echo "✓ HTML report: file://$COVERAGE_DIR/$backend/index.html"
        echo ""
    else
        echo "✗ Coverage failed for $backend backend"
        backends_tested["$backend"]=0
        echo ""
    fi
}

# Test scalar backend (always available)
run_backend_coverage "scalar" "scalar"

# Test AVX2 backend (x86-64 with AVX2 support)
if grep -q avx2 /proc/cpuinfo 2>/dev/null; then
    run_backend_coverage "avx2" "avx2"
else
    echo "⊘ AVX2 not supported on this CPU, skipping"
    echo ""
fi

# Test AVX512 backend (x86-64 with AVX512 support)
if grep -q avx512 /proc/cpuinfo 2>/dev/null; then
    run_backend_coverage "avx512" "avx512"
else
    echo "⊘ AVX512 not supported on this CPU, skipping"
    echo ""
fi

# Test NEON backend (ARM64 with NEON support)
if [[ "$(uname -m)" == "aarch64" ]] || [[ "$(uname -m)" == "arm64" ]]; then
    run_backend_coverage "neon" "neon"
else
    echo "⊘ NEON requires ARM64 architecture, skipping (use CI ARM64 runner)"
    echo ""
fi

# Generate summary report
echo "==================================================================="
echo "Coverage Summary"
echo "==================================================================="
echo ""

for backend in scalar avx2 avx512 neon; do
    if [[ ${backends_tested[$backend]:-0} -eq 1 ]]; then
        echo "Backend: $backend"
        echo "  HTML Report: file://$COVERAGE_DIR/$backend/index.html"
        if [[ -f "$COVERAGE_DIR/$backend/summary.txt" ]]; then
            echo "  Coverage:"
            cat "$COVERAGE_DIR/$backend/summary.txt"
        fi
        echo ""
    fi
done

echo "==================================================================="
echo "Next Steps"
echo "==================================================================="
echo "1. Open HTML reports in browser to analyze coverage gaps"
echo "2. Focus on critical DSP paths (math kernels, SIMD operations)"
echo "3. Add targeted tests for uncovered branches"
echo "4. Target: >90% line coverage, >95% branch coverage for critical paths"
echo ""
echo "To view a report:"
echo "  firefox $COVERAGE_DIR/scalar/index.html"
echo "  # or your preferred browser"
echo ""
