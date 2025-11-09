# Repository Guidelines

## Project Structure & Module Organization
- `projects/rigel-synth/crates/dsp`: `no_std` DSP core; keep all real-time code
  allocation-free.
- `projects/rigel-synth/crates/cli`: CLI entry point (`rigel`) for rendering
  audio and exercising DSP paths.
- `projects/rigel-synth/crates/plugin`: NIH-plug wrapper for building VST3/CLAP
  binaries; interacts with DAWs.
- `projects/rigel-synth/crates/xtask`: helper tasks including `cargo xtask
  bundle` for plugin packaging.
- `projects/wtgen/`: Python research workspace (nix devenv managed) for
  generating wavetable assets used during development.
- `projects/rigel-site` & `projects/rigel-backend`: placeholders for the public
  site + future backend service; use them for planning docs and tracking TODOs.

## Essential commands

All commands must ALWAYS be executed in the devenv shell environment. Thanks to the
direnv setup, you will likely be running in the devenv shell environment so all
of these commands can be run as-is. If for some reason the devenv shell is
broken, invoke them like: `devenv shell -- <command>`, fixing the devenv env
immediately.

### Building

- Build for the host platform: `build:native`
- Build for macOS: `build:macos`
- Build for x86_64 Linux: `build:linux`
- Build for x86_64 Windows: `build:win`

### Testing, Linting, and Formatting

- Test: `cago:test`
- Format: `cargo:fmt`
- Lint: `cargo:lint`

## Coding Style & Naming Conventions
- Follow Rust 2021 defaults: 4-space indentation, snake_case modules,
  UpperCamelCase types.
- Keep `rigel-dsp` free of `std`, heap use, and dynamic allocation.
- Expose parameters and commands with descriptive, lower-kebab subcommands
  (`note`, `chord`, etc.).
- Document public items with Rustdoc when adding APIs that surface outside the
  crate.
- When making edits to any nix files, pay close attention to nix string interpolation
  causing conflicts with shell and always escape correctly

## Testing Guidelines
- Primary suite runs via `cargo:test`; target crate-level coverage for DSP math
  and CLI parsing.
- Add focused tests under `projects/rigel-synth/crates/dsp` or module-specific
  files; name tests
  `mod_name_behavior`.
- For audio changes, regenerate short WAV fixtures via the CLI and listen or
  diff waveforms before merging.
- Python experiments in `projects/wtgen/tests/` should mirror Rust expectations; check
  in generated assets only when deterministic. Any large binary assets should
  be checked in using git-lfs.
- When testing any nix/devenv changes, tail the output as nix errors tend to
  be very long with only the last 100 or so lines mattering much.

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
