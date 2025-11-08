# Rigel Monorepo

This repository now houses every Rigel project under a single roof.  It holds
production code for the audio plugin, the Python wavetable research toolkit,
and placeholders for the future web site + backend service.

```
projects/
├── rigel-synth/      # Rust workspace with the DSP core, CLI, plugin + xtask
├── wtgen/            # Python wavetable research + asset generation toolkit
├── rigel-site/       # Placeholder for the public marketing/docs site
└── rigel-backend/    # Placeholder for the companion backend service
```

## Getting Started

1. **Enter the Nix/devenv shell** from the repository root so the Rust toolchain
   is configured and the helper scripts (`cargo:fmt`, `build:native`, etc.) are
   available:
   ```bash
   devenv shell
   ```
2. **Rigel synth workspace** lives in `projects/rigel-synth`.  Run `cargo`
   commands from the repo root (the workspace manifest stays at the top level)
   or `cd projects/rigel-synth` if you prefer.
3. **wtgen** has its own devenv/flake stack.  `cd projects/wtgen` and run
   `devenv shell` (or `nix develop`) to enter the Python environment.  The
   project includes helper scripts such as `devenv shell -- test:full` for
   pytest.

## Project Directory Notes

- `projects/rigel-synth` contains everything that ships inside the Rigel audio
  plugin: `rigel-dsp` (`no_std` core), `rigel-cli`, `rigel-plugin`, and `rigel-xtask`.
  See `projects/rigel-synth/README.md` for details, commands, and architecture.
- `projects/wtgen` focuses on wavetable research, fixture generation, and DSP
  prototyping.  It produces `.npz` and `.wav` assets that eventually feed into
  the Rust plugin.
- `projects/rigel-site` and `projects/rigel-backend` act as stubs for upcoming
  work.  The README files inside outline the scope and technology decisions we
  expect to make when those tracks begin.

## Repository Tooling

- `devenv.nix` configures the shared Rust toolchain, cross-compilation SDKs, and
  helper scripts for the `rigel-synth` workspace.
- `projects/wtgen/devenv.nix` describes the Python toolchain using uv for
  dependency management.  The lockfile lives next to it so the environment stays
  deterministic.
- `docs/development-environments.md` captures the entry points and environment
  variables for every project-level devenv shell.
- `cargo fmt`, `cargo clippy`, `cargo test`, and `cargo xtask bundle` continue to
  work from the repository root.  The workspace automatically points to the new
  project paths.

## Contributing

1. Use the appropriate devenv shell for the project you are working on.
2. Keep Rust changes `no_std`-friendly in the DSP crate and document new APIs
   with Rustdoc when they are public.
3. For Python changes, keep `projects/wtgen/tests` up to date and regenerate fixtures when
   algorithms change.
4. Before sending a PR, run the Rust and Python test suites from the root:
   ```bash
   # Rust
   devenv shell -- cargo:test

   # Python (inside projects/wtgen)
   cd projects/wtgen && devenv shell -- test:full
   ```

See the individual project READMEs for deeper guidance.
