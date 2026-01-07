{
  pkgs,
  lib,
  config,
  inputs,
  ...
}:

let
  isLinux = pkgs.stdenv.isLinux;
  isDarwin = pkgs.stdenv.isDarwin;
  appleSdk = if isDarwin then pkgs.apple-sdk_15 else null;
  rustToolchainSpecifier = "1.91.0";

  # Build cargo-instruments from crates.io (macOS only)
  cargo-instruments = pkgs.rustPlatform.buildRustPackage rec {
    pname = "cargo-instruments";
    version = "0.4.13";

    src = pkgs.fetchCrate {
      inherit pname version;
      sha256 = "sha256-rK++Z3Ni4yfkb36auyWJ9Eiqi2ATeEyQ6J4synRTpbM=";
    };

    cargoHash = "sha256-hRpWBt00MHMBZCHAsbFU0rwpsoavv6PUNj6owFHRNEw=";

    nativeBuildInputs = [ pkgs.pkg-config ];
    buildInputs = [ pkgs.openssl ];

    # Only build on macOS since it requires Instruments.app
    meta = {
      platforms = pkgs.lib.platforms.darwin;
    };
  };

  # Rust triples we support from the primary macOS dev host. Windows/Linux builds
  # need additional SDK/tooling shims so we keep the list centralised here.
  rustTargets = [
    "aarch64-apple-darwin"
    "x86_64-pc-windows-msvc"
    "x86_64-unknown-linux-gnu"
  ];
  rustTargetList = lib.concatStringsSep " " rustTargets;
  rustComponents = [
    "rustc"
    "cargo"
    "rustfmt"
    "clippy"
    "rust-src"
    "rust-analyzer"
    "llvm-tools-preview"
  ];
  rustComponentList = lib.concatStringsSep " " rustComponents;

  # Helper function to wrap cargo commands with proper environment setup
  cargoScript = command: ''
    set -euo pipefail
    export PATH="''${DEVENV_PROFILE}/bin:$PATH"

    state_dir="''${DEVENV_STATE:-$PWD/.devenv/state}"
    export DEVENV_STATE="$state_dir"
    export CARGO_HOME="$state_dir/cargo"
    export RUSTUP_HOME="$state_dir/rustup"
    mkdir -p "$CARGO_HOME/bin" "$RUSTUP_HOME"

    exec ${command}
  '';

  # Workaround for GCC specs directory conflict
  # The specs/ directory at repo root confuses GCC which looks for a "specs" file
  # in the current directory. We temporarily rename it during cargo operations.
  withSpecsWorkaround = command: ''
    set -euo pipefail

    # Determine repository root
    if command -v git >/dev/null 2>&1 && git rev-parse --git-dir >/dev/null 2>&1; then
      REPO_ROOT="$(git rev-parse --show-toplevel)"
    else
      REPO_ROOT="$PWD"
    fi

    # Rename specs directory if it exists
    SPECS_RENAMED=0
    if [ -d "$REPO_ROOT/specs" ]; then
      mv "$REPO_ROOT/specs" "$REPO_ROOT/specs.bak"
      SPECS_RENAMED=1
    fi

    # Ensure specs/ is restored on exit (even if command fails)
    cleanup() {
      if [ $SPECS_RENAMED -eq 1 ] && [ -d "$REPO_ROOT/specs.bak" ]; then
        mv "$REPO_ROOT/specs.bak" "$REPO_ROOT/specs"
      fi
    }
    trap cleanup EXIT

    # Execute the command
    ${command}
  '';
in
{
  name = "rigel";
  devenv.root = lib.mkDefault (
    let
      envPwd = builtins.getEnv "PWD";
      fallback = toString ./.;
    in
    if envPwd != "" then envPwd else fallback
  );

  # Enable Cachix binary cache for faster Nix builds
  cachix.enable = true;
  cachix.pull = [ "kcierzan-rigel" ];
  cachix.push = "kcierzan-rigel";

  env =
    let
      hostPath = builtins.getEnv "PATH";
    in
    {
      RUST_BACKTRACE = lib.mkDefault "1";
      PKG_CONFIG_ALLOW_CROSS = lib.mkDefault "1";
      DEVENV_HOST_PATH = hostPath;
      RIGEL_SYNTH_ROOT = lib.mkDefault (toString ./projects/rigel-synth);
      RIGEL_WTGEN_ROOT = lib.mkDefault (toString ./projects/wtgen);
      # Scrub host linker/compiler hints so only devenv/Nix values leak into builds.
      LIBRARY_PATH = lib.mkForce "";
      LDFLAGS = lib.mkForce "";
      CPPFLAGS = lib.mkForce "";
    }
    // lib.optionalAttrs isDarwin {
      # macOS-specific environment variables
      MACOSX_DEPLOYMENT_TARGET = "11.0";
    };

  languages.rust = {
    enable = true;
    channel = "stable";
    # Pin the toolchain so CI and local shells use the exact same Rust.
    version = rustToolchainSpecifier;
    components = rustComponents;
    targets = rustTargets;
  };

  packages =
    with pkgs;
    [
      python3
      basedpyright
      git
      pkg-config
      cmake
      ninja
      zip
      unzip
      just
      # xwin downloads Windows SDK/MSVC redistributables so we can link MSVC builds
      # without requiring a Windows VM.
      xwin
      # Benchmarking and profiling tools
      gnuplot # Criterion chart generation
      flamegraph # Flamegraph generation (uses perf on Linux, DTrace on macOS)
    ]
    ++ lib.optionals isLinux [
      alsa-lib
      libGL
      libxkbcommon
      wayland
      xorg.libX11
      xorg.libXcursor
      xorg.libXi
      xorg.libXrandr
      xorg.libxcb
      xorg.xcbutil
      xorg.xcbutilimage
      xorg.xcbutilrenderutil
      xorg.xcbutilwm
      xorg.xcbutilkeysyms
      # Linux-specific profiling tools
      valgrind # Required for iai-callgrind benchmarks
      perf # Hardware performance counters
      # Fast linker for faster builds
      mold
      clang # Linker driver for mold
    ]
    ++ lib.optionals isDarwin [
      # macOS-specific dependencies
      appleSdk # Xcode headers
      llvmPackages.openmp # OpenMP runtime for SIMD code
      libiconv # For CLI/build tooling
      cargo-instruments # Instruments.app integration for profiling
      # Note: Valgrind for iai-callgrind can be installed via Homebrew:
      # brew install valgrind
      # DTrace (built into macOS) is used by flamegraph
    ];

  scripts = {
    "cargo:fmt".exec = cargoScript "cargo fmt";
    "cargo:lint".exec = withSpecsWorkaround (cargoScript "cargo clippy --all-targets");
    "cargo:test".exec = withSpecsWorkaround (cargoScript "cargo test");

    # SIMD-specific tests
    "test:avx2".exec = withSpecsWorkaround ''
      set -euo pipefail
      export PATH="''${DEVENV_PROFILE}/bin:$PATH"

      state_dir="''${DEVENV_STATE:-$PWD/.devenv/state}"
      export DEVENV_STATE="$state_dir"
      export CARGO_HOME="$state_dir/cargo"
      export RUSTUP_HOME="$state_dir/rustup"
      mkdir -p "$CARGO_HOME/bin" "$RUSTUP_HOME"

      export RUSTFLAGS="-C target-feature=+avx2,+fma"
      exec cargo test --features avx2
    '';

    "test:avx512".exec = withSpecsWorkaround ''
      set -euo pipefail
      export PATH="''${DEVENV_PROFILE}/bin:$PATH"

      state_dir="''${DEVENV_STATE:-$PWD/.devenv/state}"
      export DEVENV_STATE="$state_dir"
      export CARGO_HOME="$state_dir/cargo"
      export RUSTUP_HOME="$state_dir/rustup"
      mkdir -p "$CARGO_HOME/bin" "$RUSTUP_HOME"

      export RUSTFLAGS="-C target-feature=+avx512f,+avx512dq,+avx512cd,+avx512bw,+avx512vl"
      exec cargo test --features avx512
    '';

    "test:neon".exec = withSpecsWorkaround ''
      set -euo pipefail
      export PATH="''${DEVENV_PROFILE}/bin:$PATH"

      state_dir="''${DEVENV_STATE:-$PWD/.devenv/state}"
      export DEVENV_STATE="$state_dir"
      export CARGO_HOME="$state_dir/cargo"
      export RUSTUP_HOME="$state_dir/rustup"
      mkdir -p "$CARGO_HOME/bin" "$RUSTUP_HOME"

      os_type=$(uname -s)
      if [ "$os_type" != "Darwin" ]; then
        echo 'Error: test:neon is only available on macOS (aarch64-apple-darwin)'
        exit 1
      fi

      export RUSTFLAGS="-C target-feature=+neon"
      exec cargo test --features neon
    '';

    "build:cli".exec = withSpecsWorkaround (cargoScript "cargo build --release --bin rigel");

    # Native build for current platform
    "build:native".exec = withSpecsWorkaround (cargoScript "cargo xtask bundle rigel-plugin --release");

    # Platform-specific builds (work only on their respective platforms)
    "build:macos".exec = withSpecsWorkaround (
      cargoScript "cargo xtask bundle rigel-plugin --release --target aarch64-apple-darwin"
    );
    "build:linux".exec = withSpecsWorkaround (
      cargoScript "cargo xtask bundle rigel-plugin --release --target x86_64-unknown-linux-gnu"
    );

    # Windows cross-compilation (uses xwin for MSVC import libs)
    # The script populates a shared cache under target/ then wires the toolchain
    # env vars Cargo expects for the MSVC target.
    "build:win".exec = "${./ci/scripts/build-win.sh}";

    "build:clean".exec = "rm -rf target";

    # SIMD-specific builds
    # These enable SIMD features and fall back to scalar for remainder samples
    "build:avx2".exec = withSpecsWorkaround ''
      set -euo pipefail
      export PATH="''${DEVENV_PROFILE}/bin:$PATH"

      state_dir="''${DEVENV_STATE:-$PWD/.devenv/state}"
      export DEVENV_STATE="$state_dir"
      export CARGO_HOME="$state_dir/cargo"
      export RUSTUP_HOME="$state_dir/rustup"
      mkdir -p "$CARGO_HOME/bin" "$RUSTUP_HOME"

      export RUSTFLAGS="-C target-feature=+avx2,+fma"
      exec cargo xtask bundle rigel-plugin --release --features avx2 --target x86_64-unknown-linux-gnu
    '';

    "build:avx512".exec = withSpecsWorkaround ''
      set -euo pipefail
      export PATH="''${DEVENV_PROFILE}/bin:$PATH"

      state_dir="''${DEVENV_STATE:-$PWD/.devenv/state}"
      export DEVENV_STATE="$state_dir"
      export CARGO_HOME="$state_dir/cargo"
      export RUSTUP_HOME="$state_dir/rustup"
      mkdir -p "$CARGO_HOME/bin" "$RUSTUP_HOME"

      export RUSTFLAGS="-C target-feature=+avx512f,+avx512dq,+avx512cd,+avx512bw,+avx512vl"
      exec cargo xtask bundle rigel-plugin --release --features avx512 --target x86_64-unknown-linux-gnu
    '';

    "build:neon".exec = withSpecsWorkaround ''
      set -euo pipefail
      export PATH="''${DEVENV_PROFILE}/bin:$PATH"

      state_dir="''${DEVENV_STATE:-$PWD/.devenv/state}"
      export DEVENV_STATE="$state_dir"
      export CARGO_HOME="$state_dir/cargo"
      export RUSTUP_HOME="$state_dir/rustup"
      mkdir -p "$CARGO_HOME/bin" "$RUSTUP_HOME"

      os_type=$(uname -s)
      if [ "$os_type" != "Darwin" ]; then
        echo 'Error: build:neon is only available on macOS (aarch64-apple-darwin)'
        exit 1
      fi

      export RUSTFLAGS="-C target-feature=+neon"
      exec cargo xtask bundle rigel-plugin --release --features neon --target aarch64-apple-darwin
    '';

    # Benchmarking and profiling
    "bench:criterion".exec = withSpecsWorkaround (cargoScript "cargo bench --bench criterion_benches");
    "bench:iai".exec = withSpecsWorkaround (cargoScript "cargo bench --bench iai_benches");
    "bench:all".exec = withSpecsWorkaround (cargoScript "cargo bench");
    "bench:baseline".exec = withSpecsWorkaround (
      cargoScript "cargo bench --bench criterion_benches -- --save-baseline main"
    );
    "bench:flamegraph".exec = withSpecsWorkaround (
      cargoScript "cargo flamegraph --bench criterion_benches -- --bench"
    );
    # macOS Instruments profiling (requires Xcode Command Line Tools)
    "bench:instruments".exec = withSpecsWorkaround ''
      set -euo pipefail
      export PATH="''${DEVENV_PROFILE}/bin:$PATH"

      state_dir="''${DEVENV_STATE:-$PWD/.devenv/state}"
      export DEVENV_STATE="$state_dir"
      export CARGO_HOME="$state_dir/cargo"
      export RUSTUP_HOME="$state_dir/rustup"
      mkdir -p "$CARGO_HOME/bin" "$RUSTUP_HOME"

      os_type=$(uname -s)
      if [ "$os_type" != "Darwin" ]; then
        echo 'Error: bench:instruments is only available on macOS'
        exit 1
      fi

      exec cargo instruments --bench criterion_benches --template time
    '';
  };

  enterShell = ''
    set -euo pipefail
    export PATH="''${DEVENV_PROFILE}/bin:$PATH"

    # Stage per-project cargo/rustup state so builds stay project-local.
    state_dir="''${DEVENV_STATE:-$PWD/.devenv/state}"
    export DEVENV_STATE="$state_dir"
    export CARGO_HOME="$state_dir/cargo"
    export RUSTUP_HOME="$state_dir/rustup"
    mkdir -p "$CARGO_HOME/bin" "$RUSTUP_HOME"

    toolchain_bin=""
    if command -v rustup >/dev/null 2>&1; then
      desired_toolchain="${rustToolchainSpecifier}"
      ensure_toolchain() {
        local toolchain="$1"
        if ! rustup toolchain list | grep -q "^$toolchain"; then
          rustup toolchain install "$toolchain" >/dev/null 2>&1
        fi
        rustup default "$toolchain" >/dev/null 2>&1 || true
        for component in ${rustComponentList}; do
          rustup component add --toolchain "$toolchain" "$component" >/dev/null 2>&1 || true
        done
        for target in ${rustTargetList}; do
          rustup target add --toolchain "$toolchain" "$target" >/dev/null 2>&1 || true
        done
      }
      ensure_toolchain "$desired_toolchain"
      active_toolchain="$(rustup show active-toolchain 2>/dev/null | awk 'NR==1 {print $1}')"
      if [ -n "$active_toolchain" ]; then
        toolchain_bin="$RUSTUP_HOME/toolchains/$active_toolchain/bin"
        export CLIPPY_DRIVER_PATH="$toolchain_bin/clippy-driver"
        export RUSTC="$toolchain_bin/rustc"
      fi
    fi

    if [ -n "$toolchain_bin" ]; then
      export PATH="$toolchain_bin:$PATH"
    fi
    export PATH="$CARGO_HOME/bin:$PATH"

    host_path="''${DEVENV_HOST_PATH:-}"
    if [ -n "$host_path" ]; then
      export PATH="$PATH:$host_path"
    fi

    echo "Rust toolchain: $(rustc --version)"
    echo "Cargo version: $(cargo --version)"
  '';

  enterTest = ''
    set -euo pipefail

    # Workaround for GCC specs directory conflict
    if command -v git >/dev/null 2>&1 && git rev-parse --git-dir >/dev/null 2>&1; then
      REPO_ROOT="$(git rev-parse --show-toplevel)"
    else
      REPO_ROOT="$PWD"
    fi

    SPECS_RENAMED=0
    if [ -d "$REPO_ROOT/specs" ]; then
      mv "$REPO_ROOT/specs" "$REPO_ROOT/specs.bak"
      SPECS_RENAMED=1
    fi

    cleanup() {
      if [ $SPECS_RENAMED -eq 1 ] && [ -d "$REPO_ROOT/specs.bak" ]; then
        mv "$REPO_ROOT/specs.bak" "$REPO_ROOT/specs"
      fi
    }
    trap cleanup EXIT

    cargo fmt -- --check
    cargo clippy --all-targets -- -D warnings
    cargo test
  '';

  tasks = {
    "ci:check".description = "Run formatting, linting, and tests (matches enterTest)";
    "ci:check".exec = ''
      set -euo pipefail

      # Workaround for GCC specs directory conflict
      if command -v git >/dev/null 2>&1 && git rev-parse --git-dir >/dev/null 2>&1; then
        REPO_ROOT="$(git rev-parse --show-toplevel)"
      else
        REPO_ROOT="$PWD"
      fi

      SPECS_RENAMED=0
      if [ -d "$REPO_ROOT/specs" ]; then
        mv "$REPO_ROOT/specs" "$REPO_ROOT/specs.bak"
        SPECS_RENAMED=1
      fi

      cleanup() {
        if [ $SPECS_RENAMED -eq 1 ] && [ -d "$REPO_ROOT/specs.bak" ]; then
          mv "$REPO_ROOT/specs.bak" "$REPO_ROOT/specs"
        fi
      }
      trap cleanup EXIT

      cargo fmt -- --check
      cargo clippy --all-targets -- -D warnings
      cargo test
    '';
  };

  containers.shell = {
    name = "rigel-shell";
    version =
      let
        fromEnv = builtins.getEnv "DEVENV_CONTAINER_VERSION";
      in
      if fromEnv != "" then fromEnv else "latest";
  };
}
