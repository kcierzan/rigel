<!--
Sync Impact Report:
Version: 1.1.0 → 1.1.1 (PATCH: Update type checker references from mypy/basedpyright to ty)
Modified Principles:
  - Principle III (Test-Driven Validation): Updated type checker reference to ty
  - Type Safety section: Updated type checking requirement to ty
Added Sections: None
Removed Sections: None
Templates Requiring Updates: None (no template changes needed for type checker tool update)
Follow-up TODOs: None
-->

# Rigel Constitution

## Core Principles

### I. Real-Time Safety (NON-NEGOTIABLE)

The DSP core (`rigel-dsp`) MUST maintain deterministic, allocation-free operation suitable for real-time audio processing. This is the foundation upon which all audio functionality is built.

**Requirements**:
- No heap allocations: No Vec, Box, String, or any std collections
- No blocking I/O: No file operations, network calls, or locks
- No std library: Only libm for math operations
- Deterministic performance: Consistent CPU usage regardless of input
- All dependencies added to rigel-dsp MUST support `no_std` and contain no allocations

**Rationale**: Real-time audio processing operates on strict latency budgets (typically 5-10ms). Any allocation, blocking operation, or non-deterministic behavior causes audible artifacts (clicks, pops, dropouts) that render the plugin unusable. The no_std constraint also ensures portability to embedded systems.

### II. Layered Architecture

Every component MUST respect the layered design: DSP core → CLI → Plugin, with each layer having clear responsibilities and boundaries.

**Structure**:
- **rigel-dsp**: Pure DSP algorithms, no_std, zero allocations
- **rigel-cli**: Command-line test harness wrapping DSP core, generates WAV files
- **rigel-plugin**: VST3/CLAP wrapper with GUI, integrates DSP core
- **rigel-xtask**: Build tooling for bundling

**Rationale**: This separation enables independent testing of DSP algorithms via CLI before plugin integration, ensures the core remains portable, and allows future reuse in other contexts (embedded, web, mobile).

### III. Test-Driven Validation

All DSP changes MUST be validated through both automated tests and audible verification before merging.

**Requirements**:
- Rust: Unit tests embedded in crates, integration tests in `tests/` directories
- Rust: ALWAYS run `cargo fmt`, `cargo clippy`, and `cargo test` before considering changes complete
- Rust: ALWAYS run architecture-specific tests for features available on the current host (NEON on aarch64, AVX2/AVX-512 on x86_64)
- Rust: ALWAYS add tests for new code and run them before task completion
- Python (wtgen): 103+ tests including property-based testing with Hypothesis
- Python (wtgen): ALWAYS run pytest, ty, and ruff before considering changes complete
- Python (wtgen): ALWAYS add tests for new code and run them before task completion
- For audio changes: Regenerate WAV fixtures via CLI and verify audibly

**Rationale**: Audio bugs are often imperceptible in code review but immediately audible. Combining automated testing (for regression prevention) with audible verification (for quality assurance) ensures both correctness and musicality. Architecture-specific SIMD optimizations (NEON, AVX) require dedicated test coverage to ensure correctness across all supported platforms.

### IV. Performance Accountability

All performance claims MUST be validated through comprehensive benchmarking, and regressions MUST be detected before merging. Mathematical operations MUST use optimized implementations.

**CPU Targets**:
- Single voice CPU usage: ~0.1% at 44.1kHz
- Full polyphonic target: <1% CPU usage

**Benchmarking Requirements**:
- Benchmark suite: Criterion (wall-clock) + iai-callgrind (instruction counts)
- Location: `projects/rigel-synth/crates/dsp/benches/`
- Save baseline before changes: `bench:baseline`
- Validate performance: `bench:all`

**Math Operation Requirements**:
- All mathematical operations MUST prefer `rigel-math` optimized approximations over libm or stdlib math functions
- Precision-critical operations MAY use standard math functions ONLY when approximation error is unacceptable AND this is explicitly documented with justification
- Operations inside loops MUST use vectorized fast math operations from `rigel-math`
- When a required math operation is not available in `rigel-math`, developers MUST implement a performant approximation in `rigel-math` and use that implementation rather than falling back to slow standard library calls
- New `rigel-math` additions MUST include accuracy benchmarks comparing against reference implementations

**Rationale**: Performance degradation in audio plugins compounds with polyphony and accumulates with other plugins in a DAW session. Deterministic benchmarking catches regressions early, and baseline comparisons prevent performance debt from accumulating. Standard math library functions (libm, stdlib) prioritize precision over speed and are not vectorized, making them unsuitable for real-time audio where thousands of operations occur per audio buffer. The `rigel-math` library provides SIMD-optimized approximations that are 3-6x faster while maintaining sufficient accuracy for audio applications.

### V. Reproducible Environments

All development MUST occur within Nix/devenv shells to ensure deterministic builds and reproducible environments across platforms.

**Structure**:
- Root shell (Rust): For rigel-synth development
- wtgen shell (Python): For wtgen development at `projects/wtgen/`
- All CI commands run through devenv shell
- Environment variables: `RIGEL_SYNTH_ROOT`, `RIGEL_WTGEN_ROOT`, `RUST_BACKTRACE=1`, `MACOSX_DEPLOYMENT_TARGET=11.0`

**Rationale**: Audio plugin development involves complex cross-compilation (macOS, Linux, Windows), platform-specific SDKs, and intricate dependency chains. Nix/devenv ensures every developer and CI runner has identical environments, eliminating "works on my machine" issues.

### VI. Cross-Platform Commitment

All code MUST build and function correctly on macOS (Apple Silicon), Linux (x86_64), and Windows (x86_64).

**Requirements**:
- Targets: `aarch64-apple-darwin`, `x86_64-unknown-linux-gnu`, `x86_64-pc-windows-msvc`
- CI validates all platforms before merge
- Use platform-agnostic APIs in DSP core
- Test cross-platform builds: `build:macos`, `build:linux`, `build:win`

**Rationale**: DAW users exist across all major platforms. Platform-specific bugs or missing builds fragment the user base and create support burden. Cross-platform CI prevents platform-specific regressions.

### VII. DSP Correctness Properties

All wavetable DSP operations MUST preserve mathematical invariants to ensure alias-free, phase-coherent audio.

**Critical Properties** (from wtgen):
- Zero-crossing alignment across all mipmap levels
- RMS consistency validation (target: 0.35 for normalized wavetables)
- DC offset removal throughout processing chain
- Antialiasing effectiveness across full MIDI range (0-127)
- Phase coherence preservation
- Spectral bandlimiting for alias prevention

**Rationale**: Violations of these properties create audible artifacts: aliasing (harsh digital sound), DC offset (speaker damage risk), phase issues (chorusing/flanging), or amplitude inconsistencies (volume jumps). These are non-negotiable for professional audio quality.

## Type Safety & Code Quality

### Python (wtgen)

- Line length: 100 characters (pyproject.toml)
- Type hints REQUIRED on all public functions
- MUST pass ty type checking
- Ruff for linting and formatting
- Scientific computing conventions allow magic values for DSP algorithms

### Rust

- Rust 2021 edition
- 4-space indentation, snake_case for functions/variables, UpperCamelCase for types
- Keep rigel-dsp free of std, heap allocations, and blocking operations
- Document public items with rustdoc comments
- Use descriptive kebab-case for CLI subcommands
- Test naming: `mod_name_behavior` pattern

### Commits & Pull Requests

- Short, imperative messages (<72 chars)
- Each commit MUST remain buildable and scoped to one concern
- PRs MUST describe motivation, outline testing, and link issues
- Include screenshots or audio snippets when UI or audible behavior changes
- PRs CANNOT merge with failing CI (enforced by branch protection)

## Governance

### Constitution Authority

This constitution supersedes all other development practices. All PRs and code reviews MUST verify compliance with these principles.

### Amendment Process

1. Constitution amendments MUST be proposed via PR with:
   - Justification for the change
   - Impact analysis on existing code
   - Migration plan if changes are breaking
2. Version MUST be incremented according to semantic versioning:
   - MAJOR: Backward incompatible principle removals or redefinitions
   - MINOR: New principle/section added or materially expanded guidance
   - PATCH: Clarifications, wording, typo fixes, non-semantic refinements
3. All dependent templates and documentation MUST be updated in the same PR
4. Sync Impact Report MUST be prepended to constitution.md as HTML comment

### Complexity Justification

Any violation of these principles (e.g., introducing allocations in DSP core, skipping benchmarks, platform-specific code) MUST be explicitly justified with:
- The specific problem being solved
- Why simpler alternatives are insufficient
- Migration path to principle-compliant approach if temporary

### Compliance Review

All PRs MUST pass:
- Automated CI checks (fmt, clippy, tests, builds for all platforms)
- Performance benchmarks (no regressions without justification)
- Code review verifying principle adherence
- For DSP changes: Audible verification by reviewer

### Runtime Guidance

Development guidance for AI agents (Claude Code) is maintained in `CLAUDE.md`. This file provides:
- Architecture details and file locations
- Common commands and workflows
- Critical constraints and coding conventions
- CI/CD pipeline information
- Performance targets and benchmarking procedures

**Version**: 1.1.1 | **Ratified**: 2025-11-17 | **Last Amended**: 2026-01-19
