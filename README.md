# Rigel

An advanced wavetable synthesizer built in Rust with a focus on performance, deterministic real-time processing, and portability.

## Overview

Rigel is designed as a modular wavetable synthesizer consisting of:

- **`rigel-dsp`**: A `no_std` DSP core providing fast, deterministic audio processing
- **`rigel-cli`**: A command-line tool for generating test audio and development
- **`rigel-plugin`**: NIH-plug based VST3/CLAP plugin (planned)

The architecture prioritizes real-time safety with a `no_std` DSP core that avoids allocations and uses only deterministic operations suitable for audio processing.

### wtgen

The project also features a Python environment at `wtgen/` for DSP experimentation,
prototyping, and research. `wtgen` is under development but can be used to generate
`.npz` and `.wav` format wavetables that Rigel will soon support.

## Features

### Current
- **Monophonic synthesis** with sine wave oscillators
- **ADSR envelope generator** with configurable attack, decay, sustain, and release
- **Pitch modulation** with semitone precision
- **CLI audio generation** for testing and development
- **WAV file export** for offline rendering
- **No-std DSP core** for real-time safety and portability

### Planned
- Wavetable synthesis with morphing between tables
- Polyphonic voice management
- Audio filters (low-pass, high-pass, band-pass)
- LFO modulation system
- Effects processing
- Plugin interface (VST3/CLAP)
- Iced-based GUI editor

## Quick Start

### Prerequisites

- Rust toolchain (1.70 or later)
- Audio playback software to test generated files

### Building

```bash
# Clone the repository
git clone https://github.com/kylecierzan/rigel.git
cd rigel

# Build all components
cargo build --release

# Or build just the CLI for testing
cargo build --release --bin rigel

# Build the plugin
cargo build --release -p rigel-plugin
```

### Usage Examples

#### Generate a single note
```bash
# Generate middle C for 2 seconds
cargo run --bin rigel -- note --note 60 --duration 2.0 --output middle_c.wav

# Generate A4 with custom velocity
cargo run --bin rigel -- note --note 69 --velocity 0.9 --output a4.wav
```

#### Generate chords
```bash
# Generate a major chord
cargo run --bin rigel -- chord --root 60 --chord-type major --output c_major.wav

# Generate a minor 7th chord
cargo run --bin rigel -- chord --root 57 --chord-type min7 --output a_min7.wav
```

#### Generate scales
```bash
# Generate a C major scale over 2 octaves
cargo run --bin rigel -- scale --start-note 60 --octaves 2 --scale-type major --output c_major_scale.wav

# Generate a chromatic scale
cargo run --bin rigel -- scale --start-note 60 --scale-type chromatic --output chromatic.wav
```

#### Test pitch morphing
```bash
# Generate morphing pitch modulation
cargo run --bin rigel -- morph --note 60 --duration 4.0 --morph-speed 0.5 --output morph.wav
```

### Plugin Usage

The plugin is built as a headless instrument plugin that can be loaded into any VST3 or CLAP compatible DAW:

#### Building the plugin library

From the devenv shell
```bash
# Build the plugin for your current platform
build:native
```

# For aarch64 Mac (Apple Silicon) specifically:
build:macos
```

The build will create:
- **macOS**: `target/release/librigel_plugin.dylib` (or `target/aarch64-apple-darwin/release/librigel_plugin.dylib` for cross-compilation)
- **Windows**: `target/release/rigel_plugin.dll`
- **Linux**: `target/release/librigel_plugin.so`

#### Creating plugin bundles

Rigel uses [nih-plug-xtask](https://github.com/robbert-vdh/nih-plug/tree/master/nih_plug_xtask)
and a cargo alias `xtask` for building the CLAP and VST3 bundles:

```bash
cargo xtask bundle rigel-plugin --release
```

**N.B.**: `--target` and `--profile` flags are also supported when running `cargo xtask bundle`.

#### Installing in Your DAW

1. Build the plugin and create the VST3 bundle using the commands above
2. Copy the `Rigel.vst3` bundle to your VST3 plugins directory:
   - **macOS**: `~/Library/Audio/Plug-Ins/VST3/` or `/Library/Audio/Plug-Ins/VST3/`
   - **Windows**: `C:\Program Files\Common Files\VST3\`
   - **Linux**: `~/.vst3/`
3. Rescan plugins in your DAW
4. Load "Rigel" as an instrument plugin
5. The plugin currently provides basic monophonic synthesis with ADSR envelope controls

## Architecture

### DSP Core (`rigel-dsp`)

The heart of Rigel is a `no_std` DSP library designed for real-time audio processing:

- **Real-time safe**: No allocations, no standard library dependencies
- **Deterministic**: Consistent performance suitable for audio threads
- **Portable**: Works on embedded systems and any Rust target
- **Fast math**: Uses `libm` for efficient mathematical operations

Key components:
- `SynthEngine`: Main synthesis coordinator
- `SimpleOscillator`: Phase-accumulation based sine wave generator
- `Envelope`: ADSR envelope generator with linear interpolation
- Utility functions for MIDI note conversion, clipping, and interpolation

### CLI Tool (`rigel-cli`)

A command-line interface for:
- Testing DSP algorithms
- Generating reference audio for development
- Batch audio processing
- Algorithm validation and debugging

### Plugin (`rigel-plugin`)

A NIH-plug based headless audio plugin providing:
- VST3 and CLAP format support
- Cross-platform compatibility (macOS, Windows, Linux)
- Real-time parameter control via DAW automation
- DAW integration for music production
- GUI interface (planned for future release)

## Development

### Project Structure

```
rigel/
├── Cargo.toml              # Workspace configuration
├── crates/
│   ├── dsp/                # No-std DSP core
│   │   ├── Cargo.toml
│   │   └── src/
│   │       └── lib.rs      # All DSP code in single file
│   ├── cli/                # Command-line tool
│   │   ├── Cargo.toml
│   │   └── src/
│   │       └── main.rs
│   └── plugin/             # Plugin
│       ├── Cargo.toml
│       └── src/
├── README.md
├── TODO.md
└── .gitignore
```

### Development Environment

- `direnv` drops you into the Nix shell automatically; without it run `nix develop` (macOS/Linux) or `devenv shell`.
- The flake exports the same module to both commands, so every dependency lives in `devenv.nix`. Day-to-day work just uses the default (impure) shell for convenience.
- When you need a fully isolated build, invoke `nix develop --pure` or `devenv shell --pure` to hide host PATH entries without any custom shims.
- Common editors and TUI helpers (`neovim`, `ripgrep`, `fd`, `fzf`, `eza`, `yazi`, `lazygit`, `delta`, `starship`, etc.) now come from the dev shell, keeping tooling consistent across macOS and Linux.
- Cross-compilation helpers remain available through the `build:*` scripts described in `devenv.nix`, so `build:linux`, `build:macos`, and `build:win` continue to work unchanged inside `nix develop`.

### Maintaining the Nix Environment

Rigel's `flake.nix` simply re-exports the shell described in `devenv.nix`, so keeping either up to date keeps both `nix develop` and `devenv shell` aligned. When touching either file, stay inside the devenv shell (direnv drops you into it automatically in this repo).

**Update workflow**

1. Start from a clean working tree (`git status` should show only intentional edits).
2. Refresh the flake inputs (nixpkgs + devenv) so both `nix develop` and `devenv` agree:
   ```bash
   nix flake update        # rewrites flake.lock with the latest rolling inputs
   devenv update           # refreshes devenv.lock (rust-overlay, git-hooks, etc.)
   ```
3. Re-enter the shell (`direnv reload` or `devenv shell`) so the new inputs are evaluated.
4. Sanity-check the toolchain before committing:
   ```bash
   cargo fmt -- --check
   cargo clippy --all-targets --all-features -- -D warnings
   cargo test
   ```
   (This matches the `ci:check` task defined in `devenv.nix`.)

**Rollback instructions**

- If a new input breaks the shell or builds, restore the previous locks and reload the shell:
  ```bash
  git checkout -- flake.lock devenv.lock
  direnv reload   # or restart `devenv shell`
  ```
- For partial rollbacks, you can selectively keep either lock file (e.g., keep `flake.lock` but revert `devenv.lock`) with the same command.
- Once things look good again, rerun the sanity-check commands above and commit both the locks and any related changes to `devenv.nix`/`flake.nix`.

### Building for Different Targets

The `no_std` DSP core can be built for various targets:

```bash
# Standard desktop targets
cargo build --target x86_64-apple-darwin     # macOS Intel
cargo build --target aarch64-apple-darwin    # macOS Apple Silicon
cargo build --target x86_64-pc-windows-msvc  # Windows

# Embedded targets (DSP core only)
cargo build -p rigel-dsp --target thumbv7em-none-eabihf
```

### Testing

```bash
# Run all tests
cargo test

# Test specific component
cargo test -p rigel-dsp

# Generate test audio files
cargo run --bin rigel -- note --duration 0.5 --output test_note.wav
```

### Code Guidelines

- **DSP Core**: Must remain `no_std` and allocation-free
- **Real-time Safety**: No blocking operations in audio processing paths
- **Documentation**: All public APIs must be documented
- **Testing**: Audio algorithms should have corresponding CLI test commands

## Performance

The DSP core is optimized for real-time audio processing:

- **Zero allocations** in audio processing paths
- **SIMD ready** (planned feature for vectorized operations)
- **Cache friendly** data structures and access patterns
- **Minimal dependencies** (only `libm` for math functions)

Typical performance on modern hardware:
- Single voice: ~0.1% CPU usage at 44.1kHz
- Target: <1% CPU for full polyphonic operation

## License

This project is licensed under the MIT OR Apache-2.0 license.

## Acknowledgments

- [NIH-plug](https://github.com/robbert-vdh/nih-plug) for the plugin framework
- [libm](https://github.com/rust-lang/libm) for no-std mathematical functions
- The Rust audio community for inspiration and guidance
