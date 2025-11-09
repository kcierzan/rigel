# Development Environments

Rigel now ships with two dedicated devenv shells—one for the Rust workspace and
one for the Python wtgen toolkit.  Use this document as a quick reference for
which shell to enter and how to run common workflows across the monorepo.

## Rigel Synth (Rust)
- **Path:** `projects/rigel-synth`
- **Enter shell:** `devenv shell` from the repository root (direnv/flake aware).
- **Key scripts:** `cargo:fmt`, `cargo:lint`, `cargo:test`, `build:*`.
- **Environment vars:** `RIGEL_SYNTH_ROOT` (autofilled by devenv.nix) always
  points at `./projects/rigel-synth` so helper scripts can locate assets.

## wtgen (Python)
- **Path:** `projects/wtgen`
- **Enter shell:** `cd projects/wtgen && devenv shell` (or `nix develop`).
- **Key scripts:** `test:full`, `test:fast`, `typecheck`, `lint`, `uv:sync`.
- **Environment vars:** `RIGEL_WTGEN_ROOT` and `RIGEL_SYNTH_ROOT` are exported so
  CLI tools can hand assets back to the Rust workspace without hardcoding
  relative paths.

## Cross-Project Tips
- When generating wavetables with wtgen for the plugin, write them under
  `$RIGEL_SYNTH_ROOT/assets/` (create the directory if it does not exist).  Both
  shells expose the same path so scripts can coordinate.
- Run both test suites before pushing:
  ```bash
  devenv shell -- cargo:test
  (cd projects/wtgen && devenv shell -- test:full)
  ```
- The placeholder `projects/rigel-site` and `projects/rigel-backend` folders can
  adopt their own devenv modules later; document them here when they do.
