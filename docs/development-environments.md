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

## Continuous Integration
- GitHub Actions run from `.github/workflows/ci.yml`.  Every job installs Nix
  via `DeterminateSystems/nix-installer-action` and then provisions the `devenv`
  CLI so that all commands execute through `devenv shell -- …`, matching the
  local workflow described above.
- The `lint-test` job (Ubuntu) runs `cargo fmt -- --check`, `cargo clippy`,
  `cargo test`, and the wtgen `test:full` suite (`working-directory:
  projects/wtgen`).  Reuse the `.devenv/state` and `projects/wtgen/.venv`
  caches locally if you want parity with CI runtimes.
- The `build-plugin` matrix produces release bundles for macOS
  (`build:macos` → `target/aarch64-apple-darwin/release/bundles`), Linux
  (`build:linux` → `target/x86_64-unknown-linux-gnu/release/bundles`), and
  Windows (`build:win` → `target/x86_64-pc-windows-msvc/release/bundles`).
  Pull requests run the full build matrix for regression coverage, while pushes
  to `main` additionally archive each bundle directory and upload
  `rigel-plugin-<platform>.tar.gz` artifacts for DAW testing.
- Windows bundles depend on the `xwin` SDK download.  The script already accepts
  the license flag, but budget additional cache space in CI so `target/xwin*`
  can persist between runs if builds start to thrash the job disk.
- Codesigning/notarization is not automated yet.  To publish notarized artifacts
  you will need to add the usual Apple Developer credentials (base64-encoded
  certificate, password, team ID, notarization Apple ID/app-password) as GitHub
  secrets and extend the workflow accordingly.  For DAW testing before release,
  the unsigned bundles produced here are sufficient.
