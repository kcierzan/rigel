# -----------------------------------------------------------------------------
# devenv.nix
# -----------------------------------------------------------------------------
# This file defines the development shell for wtgen.  The intent is that someone
# with zero nix/devenv experience can scroll through and understand what each
# block does.  Every major section is documented inline.
# -----------------------------------------------------------------------------
{
  pkgs,
  lib,
  config,
  inputs,
  system,
  ...
}:

let
  # We target Python 3.11 because it is the lowest supported version in
  # pyproject.toml and still the most stable release for NumPy/Numba in late
  # 2024.  Using a single constant makes it trivial to bump in the future.
  pythonVersion = "3.11";

  # Convenience booleans used in a few conditional package selections.
  isLinux = pkgs.stdenv.isLinux;
  isDarwin = pkgs.stdenv.isDarwin;
in
{
  # Tell devenv where the project root lives when used through flakes.
  # Prefer the calling shell's $PWD (requires configurable-impure-env) so the
  # root resolves to the actual checkout instead of the Nix store copy.
  devenv.root = lib.mkDefault (
    let
      envPwd = builtins.getEnv "PWD";
      fallback = toString ./.;
    in
    if envPwd != "" then envPwd else fallback
  );

  # ---------------------------------------------------------------------------
  # Essential command line tools that should exist in the shell *before*
  # Python/uv installs anything.
  # ---------------------------------------------------------------------------
  # - uv: extremely fast package installer/resolver used by this repo.
  # - git: quality-of-life so git always matches the environment.
  # - pkg-config / cmake / ninja: common C/C++ build helpers.  They are cheap
  #   to include and prevent cryptic build failures when wheels fall back to
  #   source builds (NumPy, SciPy, soundfile).
  # - gfortran/openblas/lapack: when binary wheels are not available (for
  #   example, when new Python versions land), SciPy falls back to a Fortran
  #   build; these provide the necessary toolchain.  We only install them on
  #   Linux because macOS wheels routinely bundle Accelerate/OpenBLAS.
  # - libsndfile: runtime dependency for the `soundfile` Python package.  The
  #   upstream wheels vendored it for Linux, but supplying the system library
  #   keeps us future-proof, especially on Darwin where dynamic linking differs.
  packages =
    with pkgs;
    [
      uv
      git
      pkg-config
      cmake
      ninja
      libsndfile
      ruff # Provide a native binary so linting works on NixOS without patching wheels.
      basedpyright
    ]
    ++ lib.optionals isLinux [
      gcc
      gfortran
      openblas
      lapack
    ]
    ++ lib.optionals isDarwin [
      llvmPackages.openmp # Needed for SciPy/NumPy OpenMP support on macOS.
      libiconv # Some Python wheels still expect libiconv.dylib.
    ];

  # ---------------------------------------------------------------------------
  # Python/uv integration.
  # ---------------------------------------------------------------------------
  languages.python = {
    enable = true;

    # Using a string forces devenv to fetch the matching interpreter from the
    # pinned nixpkgs.  Developers can bump the version here without learning
    # any other nix constructs.
    version = pythonVersion;

    # Allow uv to reuse manylinux wheels instead of forcing source builds on
    # Nix.  This keeps dependency installation fast and identical to other
    # platforms while still letting us override libraries above when required.
    # Do not enable this on macOS because the derivation depends on Linux-only
    # glibc toolchains which are unavailable on Darwin hosts.
    manylinux.enable = isLinux;

    uv = {
      enable = true;

      # devenv can optionally run `uv sync` on entry.  We keep the behaviour
      # explicit (see the `uv:sync` script and `enterShell` hook below) so the
      # first shell entrance stays predictable, but we still surface the option
      # here for discoverability.
      sync = {
        enable = true;

        groups = [
          "dev"
        ];

        # In case someone flips the switch above, default to the frozen lock
        # file and install both the default + dev dependency groups.  The same
        # command string is exposed as a script for day-to-day use.
        arguments = [
          "--frozen"
        ];
      };
    };
  };

  # ---------------------------------------------------------------------------
  # Environment variables shared by both the shell and `uv` tooling.
  # ---------------------------------------------------------------------------
  env = {
    # Place the virtual environment alongside the repo so IDEs can auto-detect
    # it.  The directory lives inside .gitignore.
    UV_PROJECT_ENVIRONMENT = lib.mkForce ".venv";
    UV_VENV_DIR = lib.mkForce ".venv";

    # Ask uv to manage downloading CPython.  This avoids relying on the nix
    # interpreter for wheel compatibility (especially on NixOS).
    UV_PYTHON_DOWNLOADS = lib.mkDefault "python-managed";

    # Copy rather than symlink Python to survive aggressive cleanups.
    UV_LINK_MODE = lib.mkDefault "copy";

    # Hypothesis occasionally benefits from more examples; expose a friendly
    # default developers can override per-command.
    HYPOTHESIS_PROFILE = "dev";
  };

  # ---------------------------------------------------------------------------
  # Shell entry behaviour.  The goal is to keep the first-time experience nice:
  # - Print clear messaging if dependencies have not been synced yet.
  # - Install dependencies via uv when `.venv` is missing.
  # - Activate the virtual environment so `python`, `pytest`, etc. work
  #   immediately.
  # ---------------------------------------------------------------------------
  enterShell = ''
    set -euo pipefail

    # If there is not a uv venv already, create one
    if [ ! -d ".venv" ]; then
      uv venv
    fi

    # Install the project in editable mode for pytest runs
    uv pip install --quiet -e .

    # Make sure native nixpkgs tooling (like ruff) takes precedence over wheel
    # binaries inside .venv to avoid dynamic loader issues on NixOS.
    export PATH="${pkgs.ruff}/bin:${pkgs.basedpyright}/bin:$PATH"

    printf '\nwtgen devenv ready (Python %s)\n\n' "$(python -c 'import platform; print(platform.python_version())')"
    printf 'Helpful commands:\n'
    printf '  - devenv shell -- test:full            (pytest -n auto)\n'
    printf '  - devenv shell -- test:fast            (pytest -x --tb=short)\n'
    printf '  - devenv shell -- typecheck            (mypy + basedpyright)\n'
    printf '  - devenv shell -- lint / format        (Ruff lint/format)\n'
    printf '  - devenv shell -- uv:sync              (refresh .venv)\n\n'

    # Reset strict shell flags so the interactive session behaves normally.
    set +e
    set +u
    set +o pipefail
  '';

  # ---------------------------------------------------------------------------
  # Re-usable scripts surfaced via `devenv shell -- <name>`.  Keeping the
  # canonical project workflows here means they show up in `devenv help` and
  # provide documentation that stays in sync with what CI expects.
  # ---------------------------------------------------------------------------
  scripts = {
    "uv:sync" = {
      # Install the full dependency stack (application + dev extras).  This is
      # the same command we run implicitly from `enterShell`.
      description = "Install/update the .venv using uv and the frozen lockfile";
      exec = "uv sync --frozen --group dev && uv pip install --quiet -e .";
    };

    lint = {
      description = "Run Ruff (lint only)";
      exec = "ruff check .";
    };

    format = {
      description = "Format code with Ruff";
      exec = "ruff format";
    };

    "typecheck:mypy" = {
      description = "Strict type checking via mypy";
      exec = "uv run mypy src/";
    };

    "typecheck:pyright" = {
      description = "Type checking via basedpyright";
      exec = "basedpyright src/";
    };

    typecheck = {
      description = "Run both mypy and basedpyright";
      exec = "uv run mypy src/ && basedpyright src/";
    };

    "test:full" = {
      description = "Run the full pytest suite with xdist auto-sharding";
      exec = "uv run python -m pytest -n auto";
    };

    "test:fast" = {
      description = "Pytest in single-threaded mode with short tracebacks";
      exec = "uv run python -m pytest -x --tb=short";
    };
  };

  # ---------------------------------------------------------------------------
  # Long-running processes.  Developers can spawn them via `devenv up` to keep
  # background tooling (such as file watchers) alive during a session.
  # ---------------------------------------------------------------------------
  processes = {
    # Simple example: keep pytest running in watch mode.  Disable by default so
    # non-fans are not surprised, but it is handy to document here.
    "pytest-watch".exec = "uv run python -m pytest --maxfail=1 -f";
  };

  tasks."wtgen:uv-state-dir" = {
    description = "Ensure uv state directory exists";
    exec = ''
      mkdir -p "$DEVENV_STATE/venv"
    '';
    before = [ "devenv:python:uv" ];
  };

  # ---------------------------------------------------------------------------
  # Quality-of-life defaults for interactive shells.
  # ---------------------------------------------------------------------------
  enterTest = ''
    # Guarantee deterministic locale-sensitive tests (NumPy, soundfile).
    export LC_ALL=C.UTF-8
    export LANG=C.UTF-8
  '';
}
