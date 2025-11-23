# Feature Specification: Runtime SIMD Dispatch

**Feature Branch**: `001-runtime-simd-dispatch`
**Created**: 2025-11-22
**Status**: Draft (Updated: 2025-11-23)

> **Implementation Note**: This feature is implemented using a two-crate layered architecture:
> - **rigel-math**: Complete SIMD library (trait-based backends + runtime dispatch + `SimdContext` unified public API)
> - **rigel-dsp**: DSP algorithms (consumer of `rigel_math::simd::SimdContext`)
>
> All SIMD code lives in rigel-math as a standalone, codebase-wide library. See `data-model.md` and `plan.md` for architectural details.

**Input**: User description: "In the interest of simplifying installations of the rigel plugin, I want to modify the SIMD backend build features that specify NEON/AVX2/AVX512/SCALAR and pick a SIMD backend purely based on compile-time features to an architecture where: macOS builds always imply NEON is present and the backend is purely chosen at compile time; linux and windows builds - the default build will use function pointers to create a runtime abstraction where we use the best backend (scalar -> avx2 -> avx512) based on what the host CPU supports. Additionally, we should be able to create builds for linux and windows that force one of those backends only. This should allow users to simply download the binary for their OS and get the best possible performance out of the plugin without needing to know anything about their CPU architecture. The forcing will ensure we have deterministic testing of backends where we can be sure a particular backend is being used regardless of host (especially important in CI). We should absolutely maintain the no_std principle of rigel-math and the dsp core while also using the most performant possible abstraction to handle runtime checking of CPU features/running SIMD functions based on CPU features."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - End User Installation (Priority: P1)

A music producer downloads the Rigel plugin for their DAW. They don't know or care about CPU instruction sets - they just want the plugin to work with optimal performance on their machine.

**Why this priority**: This is the primary value proposition of the feature. End users should get optimal performance automatically without technical knowledge or configuration.

**Independent Test**: Can be fully tested by installing a single pre-built binary on various machines (with different CPU capabilities: old CPU without AVX2, modern CPU with AVX2, high-end CPU with AVX-512) and verifying the plugin runs at optimal performance on each.

**Acceptance Scenarios**:

1. **Given** a user has a modern x86_64 Windows/Linux CPU with AVX2 support, **When** they install the default Rigel plugin binary, **Then** the plugin automatically uses the AVX2 backend without user configuration
2. **Given** a user has an older x86_64 Windows/Linux CPU without AVX2, **When** they install the default Rigel plugin binary, **Then** the plugin automatically falls back to the scalar backend and runs correctly
3. **Given** a user has a high-end x86_64 Windows/Linux CPU with AVX-512 support, **When** they install the default Rigel plugin binary, **Then** the plugin automatically uses the AVX-512 backend for maximum performance
4. **Given** a user has an Apple Silicon Mac (aarch64), **When** they install the Rigel plugin, **Then** the plugin uses NEON SIMD instructions (compile-time decision)

---

### User Story 2 - Developer Testing Specific Backends (Priority: P2)

A Rigel developer needs to test a specific SIMD backend (e.g., AVX2 or scalar) in isolation to debug an issue or validate correctness, regardless of their development machine's actual CPU capabilities.

**Why this priority**: Critical for development and debugging but doesn't affect end users. Enables deterministic testing of SIMD implementations.

**Independent Test**: Can be tested by building with a backend-forcing flag (e.g., `--features force-avx2`) and verifying through tests or instrumentation that only the specified backend is used, even on a machine with different CPU capabilities.

**Acceptance Scenarios**:

1. **Given** a developer builds Rigel with a force-scalar flag on a machine with AVX2 support, **When** the plugin runs, **Then** only the scalar backend executes (verified by tests or debug logs)
2. **Given** a developer builds Rigel with a force-avx2 flag, **When** the plugin runs on a machine with AVX2 support, **Then** only the AVX2 backend executes
3. **Given** a developer builds Rigel with a force-avx512 flag on a local Linux/Windows machine with AVX-512 support, **When** the plugin runs, **Then** only the AVX-512 backend executes (experimental, local testing only)

---

### User Story 3 - CI Deterministic Backend Testing (Priority: P2)

The CI system needs to test scalar and AVX2 backends deterministically to ensure correctness of each implementation, without relying on the specific CPU capabilities of CI runners.

**Why this priority**: Equal priority to User Story 2 as it serves the same purpose (deterministic testing) but in an automated environment. Essential for maintaining code quality for production backends.

**Independent Test**: Can be tested by running CI jobs with different backend-forcing flags and verifying that test results are consistent and backend-specific tests pass.

**Acceptance Scenarios**:

1. **Given** CI builds Rigel with force-scalar flag, **When** CI runs the test suite, **Then** all scalar backend tests pass and backend-specific tests correctly identify scalar mode
2. **Given** CI builds Rigel with force-avx2 flag on an AVX2-capable runner, **When** CI runs the test suite, **Then** all AVX2 backend tests pass and backend-specific tests correctly identify AVX2 mode
3. **Given** CI builds Rigel with default runtime dispatch enabled, **When** CI runs the test suite, **Then** tests verify that runtime dispatch correctly selects and uses available backends

---

### Edge Cases

- What happens when a user builds with `force-avx2` but runs on a CPU without AVX2 support? (Should fail gracefully or be prevented at runtime)
- What happens when a user builds with `force-avx512` but runs on a CPU without AVX-512 support? (Should fail gracefully or be prevented at runtime)
- How does the system handle future SIMD instruction sets (e.g., AVX-512 variants, future x86 extensions)?
- What is the performance overhead of runtime dispatch compared to compile-time SIMD selection?
- How does the system detect CPU features in a no_std environment?
- How are AVX-512 builds tested if CI doesn't have AVX-512 runners?

### Edge Case Resolution: Forced Backends on Incompatible CPUs

**Decision**: Forced backend builds (`force-avx2`, `force-avx512`) are developer/testing tools that bypass runtime CPU detection. Safety is the developer's responsibility.

**Behavior**:
- If a forced backend binary runs on a CPU lacking required features, it will crash with "Illegal instruction" signal
- This is INTENTIONAL: forced builds are for deterministic testing where CPU capabilities are known
- End users should NEVER receive forced backend builds - only runtime-dispatch builds

**Mitigation**:
- CI uses forced backends only on runners with matching CPU features
- Documentation clearly marks forced builds as "developer/CI only"
- Release binaries ALWAYS use runtime-dispatch (automatic safe fallback)

**Future Enhancement** (out of scope): Could add compile-time CPU feature assertions in debug builds

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: macOS builds MUST use NEON SIMD backend with compile-time selection (no runtime detection)
- **FR-002**: Linux and Windows default builds MUST detect CPU features at runtime and select the best available SIMD backend (scalar → AVX2 → AVX-512)
- **FR-003**: System MUST provide a scalar fallback backend for x86_64 CPUs without AVX2 support
- **FR-004**: System MUST provide build-time flags to force specific backends (scalar, AVX2, AVX-512) on Linux and Windows
- **FR-005**: Forced-backend builds MUST use only the specified backend without runtime detection or fallback
- **FR-006**: System MUST maintain no_std compliance in rigel-dsp and rigel-math crates
- **FR-007**: Runtime CPU feature detection MUST work in a no_std environment
- **FR-008**: Runtime dispatch mechanism MUST use function pointers for backend selection
- **FR-009**: System MUST ensure that runtime dispatch overhead is negligible compared to DSP computation cost
- **FR-010**: System MUST validate CPU feature availability before using a SIMD backend (prevent illegal instruction crashes)
- **FR-011**: System MUST provide a way to query which backend is currently active (for debugging/testing)
- **FR-012**: Default builds (without force flags) MUST automatically select the best backend based on runtime CPU feature detection
- **FR-013**: AVX-512 backend MUST be available for forced builds but treated as experimental (not required for CI testing)

### Key Entities

- **SIMD Backend**: Represents a specific implementation of DSP operations using particular instruction sets (Scalar, AVX2, AVX-512, NEON). Each backend has the same functional interface but different performance characteristics.
- **CPU Feature Set**: Represents the instruction set capabilities of the host CPU, detected at runtime (for x86_64) or known at compile-time (for macOS/aarch64).
- **Function Dispatch Table**: Represents the mechanism for selecting and invoking the appropriate backend's functions based on runtime CPU feature detection (x86_64 only).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: End users can download and install a single pre-built binary per platform (Linux, Windows, macOS) and achieve optimal SIMD performance without configuration
- **SC-002**: Runtime dispatch overhead adds less than 1% CPU usage compared to compile-time SIMD selection (measured via benchmarks)
- **SC-003**: Scalar and AVX2 backends can be tested deterministically in CI using build flags
- **SC-004**: Plugin operates correctly on CPUs ranging from old (no AVX2) to modern (AVX2), verified by CI testing on at least 2 different CPU capability levels
- **SC-005**: No_std compliance is maintained in rigel-dsp and rigel-math crates (verified by compilation checks and dependency audits)
- **SC-006**: Build times for default (runtime dispatch) builds are within 10% of current compile-time SIMD builds
- **SC-007**: Single binary size increase is less than 20% compared to single-backend builds (due to including multiple backend implementations)
- **SC-008**: AVX-512 backend can be built and tested locally on ≥1 developer machine with AVX-512 support, verified by successful test suite execution (`cargo test --features force-avx512`) and manual validation report (experimental status; not required for production release)

## Assumptions

- Apple Silicon Macs (aarch64) universally support NEON instructions, so runtime detection is unnecessary
- The performance overhead of function pointer dispatch is negligible for DSP workloads where computation time dominates
- CI runners have access to both scalar-only and AVX2-capable machines for testing
- CI runners do NOT have AVX-512 capabilities; AVX-512 testing is experimental and performed only on local Linux/Windows machines
- Users prefer a single binary per platform over multiple CPU-specific builds
- The current SIMD implementations (NEON, AVX2, AVX-512, Scalar) are functionally equivalent and differ only in performance
- CPU feature detection can be performed at plugin initialization time (once) rather than per-sample or per-buffer
- AVX-512 support is considered experimental due to limited testing infrastructure and potential performance variability across CPUs

## Dependencies

- Runtime CPU feature detection library compatible with no_std (cpufeatures crate for x86_64 CPUID)
- Build system modifications to support both default (runtime dispatch) and forced-backend builds in rigel-math
- Function pointer abstraction layer in rigel-math for backend dispatch (BackendDispatcher + SimdContext)
- rigel-dsp dependency update to use rigel-math with runtime-dispatch feature
- Access to AVX2-capable CI runners for automated testing
- Access to developer machines with AVX-512 support for experimental local testing

## Out of Scope

- Dynamic switching between backends during plugin operation (backend is selected once at initialization)
- Support for x86 32-bit architectures
- Support for ARM 32-bit (only aarch64 macOS with NEON)
- Auto-tuning or performance profiling to select backends based on workload characteristics
- Support for mixed-mode execution (using different backends for different operations)
- Automated CI testing of AVX-512 backend (experimental, local testing only)
