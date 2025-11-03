# Repository Guidelines

## Project Structure & Module Organization
- `crates/dsp`: `no_std` DSP core; keep all real-time code allocation-free.
- `crates/cli`: CLI entry point (`rigel`) for rendering audio and exercising
  DSP paths.
- `crates/plugin`: NIH-plug wrapper for building VST3/CLAP binaries; interacts
  with DAWs.
- `crates/xtask`: helper tasks including `cargo xtask bundle` for plugin
  packaging.
- `wtgen/`: Python research workspace (nix devenv managed) for generating
  wavetable assets used during development.

## Build, Test, and Development Commands
- `cargo build --workspace` builds every crate; add `--release` when profiling
  performance.
- `cargo run --bin rigel -- note --note 60 --duration 2 --output middle_c.wav`
  renders quick regression audio.
- `cargo xtask bundle rigel-plugin --release` emits VST3/CLAP bundles under
  `target/`.
- `cargo fmt` and `cargo clippy --all-targets --all-features` must pass before
  reviews.
  pytest`.

## Coding Style & Naming Conventions
- Follow Rust 2021 defaults: 4-space indentation, snake_case modules,
  UpperCamelCase types.
- Keep `rigel-dsp` free of `std`, heap use, and dynamic allocation.
- Expose parameters and commands with descriptive, lower-kebab subcommands
  (`note`, `chord`, etc.).
- Document public items with Rustdoc when adding APIs that surface outside the
  crate.

## Testing Guidelines
- Primary suite runs via `cargo test`; target crate-level coverage for DSP math
  and CLI parsing.
- Add focused tests under `crates/dsp` or module-specific files; name tests
  `mod_name_behavior`.
- For audio changes, regenerate short WAV fixtures via the CLI and listen or
  diff waveforms before merging.
- Python experiments in `wtgen/tests/` should mirror Rust expectations; check
  in generated assets only when deterministic.

## Commit & Pull Request Guidelines
- Recent commits use short, imperative subjects (`Fix devenv comment`, `Make
  pip install ...`); match that style and keep under ~72 chars.
- Each commit should remain buildable and scoped to one concern (DSP, CLI,
  plugin, or tooling).
- Pull requests should describe motivation, outline testing (`cargo test`,
  audio renders, DAW smoke checks), and link open issues or TODO entries.
- Include screenshots or audio snippets when UI or audible behavior changes;
  attach bundle paths if asking for DAW validation.

## Security & Configuration Tips
- Verify new dependencies maintain `no_std` compatibility before adding them to
  `rigel-dsp`.
- Avoid blocking I/O or file access from real-time audio callbacks; stage work
  on auxiliary threads.
