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
  linuxCross = pkgs.pkgsCross.gnu64;
  linuxCrossCc = linuxCross.stdenv.cc;
  linuxCrossBinutils = linuxCross.buildPackages.binutils;
  linuxTargetPrefix = linuxCrossCc.targetPrefix or "";
  # Pull Linux libraries from nixpkgs' x86_64 set; these are only used when cross
  # compiling GUI crates that want pkg-config discovery (iced, baseview, ...).
  linuxPkgs = inputs.nixpkgs.legacyPackages."x86_64-linux";
  linuxXorg = if linuxPkgs ? xorg then linuxPkgs.xorg else { };
  # Teach macOS pkg-config how to find Linux GUI deps by enumerating every
  # optional pkg-config directory we might need from the x86_64 nixpkgs set.
  linuxPkgConfigPaths = lib.concatLists [
    (lib.optionals (linuxPkgs ? libGL) [
      "${linuxPkgs.libGL.dev}/lib/pkgconfig"
    ])
    (lib.optionals (linuxXorg ? libX11) [
      "${linuxXorg.libX11.dev}/lib/pkgconfig"
    ])
    (lib.optionals (linuxXorg ? libxcb) [
      "${linuxXorg.libxcb.dev}/lib/pkgconfig"
    ])
    (lib.optionals (linuxXorg ? xcbutil) [
      "${linuxXorg.xcbutil.dev}/lib/pkgconfig"
    ])
    (lib.optionals (linuxXorg ? xcbutilimage) [
      "${linuxXorg.xcbutilimage.dev}/lib/pkgconfig"
    ])
    (lib.optionals (linuxXorg ? xcbutilrenderutil) [
      "${linuxXorg.xcbutilrenderutil.dev}/lib/pkgconfig"
    ])
    (lib.optionals (linuxXorg ? xcbutilwm) [
      "${linuxXorg.xcbutilwm.dev}/lib/pkgconfig"
    ])
    (lib.optionals (linuxXorg ? xcbutilkeysyms) [
      "${linuxXorg.xcbutilkeysyms.dev}/lib/pkgconfig"
    ])
    (lib.optionals (linuxXorg ? libXcursor) [
      "${linuxXorg.libXcursor.dev}/lib/pkgconfig"
    ])
    (lib.optionals (linuxXorg ? libXi) [
      "${linuxXorg.libXi.dev}/lib/pkgconfig"
    ])
    (lib.optionals (linuxXorg ? libXrandr) [
      "${linuxXorg.libXrandr.dev}/lib/pkgconfig"
    ])
    (lib.optionals (linuxPkgs ? xorgproto) [
      "${linuxPkgs.xorgproto}/share/pkgconfig"
    ])
  ];
  linuxLibraryPaths = lib.concatLists [
    (lib.optionals (linuxPkgs ? libGL) [
      "${linuxPkgs.libGL.out}/lib"
    ])
    (lib.optionals (linuxXorg ? libX11) [
      "${linuxXorg.libX11.out}/lib"
    ])
    (lib.optionals (linuxXorg ? libxcb) [
      "${linuxXorg.libxcb.out}/lib"
    ])
    (lib.optionals (linuxXorg ? xcbutil) [
      "${linuxXorg.xcbutil.out}/lib"
    ])
    (lib.optionals (linuxXorg ? xcbutilimage) [
      "${linuxXorg.xcbutilimage.out}/lib"
    ])
    (lib.optionals (linuxXorg ? xcbutilrenderutil) [
      "${linuxXorg.xcbutilrenderutil.out}/lib"
    ])
    (lib.optionals (linuxXorg ? xcbutilwm) [
      "${linuxXorg.xcbutilwm.out}/lib"
    ])
    (lib.optionals (linuxXorg ? xcbutilkeysyms) [
      "${linuxXorg.xcbutilkeysyms.out}/lib"
    ])
    (lib.optionals (linuxXorg ? libXcursor) [
      "${linuxXorg.libXcursor.out}/lib"
    ])
    (lib.optionals (linuxXorg ? libXi) [
      "${linuxXorg.libXi.out}/lib"
    ])
    (lib.optionals (linuxXorg ? libXrandr) [
      "${linuxXorg.libXrandr.out}/lib"
    ])
  ];
  hostPkgConfigPath = builtins.getEnv "PKG_CONFIG_PATH";
  linuxPkgConfigSearchPaths =
    linuxPkgConfigPaths ++ lib.optional (hostPkgConfigPath != "") hostPkgConfigPath;
  hasLinuxPkgConfig = linuxPkgConfigSearchPaths != [ ];
  linuxPkgConfigPath = lib.concatStringsSep ":" linuxPkgConfigSearchPaths;
  hasLinuxLibraryPaths = linuxLibraryPaths != [ ];
  linuxLdFlags = lib.concatStringsSep " " (map (path: "-L${path}") linuxLibraryPaths);
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

  env =
    let
      hostPath = builtins.getEnv "PATH";
    in
    {
      RUST_BACKTRACE = lib.mkDefault "1";
      MACOSX_DEPLOYMENT_TARGET = "11.0";
      PKG_CONFIG_ALLOW_CROSS = lib.mkDefault "1";
      DEVENV_HOST_PATH = hostPath;
      RIGEL_SYNTH_ROOT = lib.mkDefault (toString ./projects/rigel-synth);
      RIGEL_WTGEN_ROOT = lib.mkDefault (toString ./projects/wtgen);
      # Scrub host linker/compiler hints so only devenv/Nix values leak into builds.
      LIBRARY_PATH = lib.mkForce "";
      LDFLAGS = lib.mkForce "";
      CPPFLAGS = lib.mkForce "";
    }
    // lib.optionalAttrs (isDarwin && hasLinuxPkgConfig) {
      PKG_CONFIG_PATH = linuxPkgConfigPath;
      PKG_CONFIG_PATH_x86_64_unknown_linux_gnu = linuxPkgConfigPath;
    }
    // lib.optionalAttrs (isDarwin && hasLinuxLibraryPaths) {
      NIX_LDFLAGS = linuxLdFlags;
      NIX_LDFLAGS_x86_64_unknown_linux_gnu = linuxLdFlags;
    }
    // lib.optionalAttrs isDarwin {
      CC_x86_64_unknown_linux_gnu = "${linuxCrossCc}/bin/${linuxTargetPrefix}cc";
      CXX_x86_64_unknown_linux_gnu = "${linuxCrossCc}/bin/${linuxTargetPrefix}c++";
      AR_x86_64_unknown_linux_gnu = "${linuxCrossBinutils}/bin/${linuxTargetPrefix}ar";
      RANLIB_x86_64_unknown_linux_gnu = "${linuxCrossBinutils}/bin/${linuxTargetPrefix}ranlib";
      CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER = "${linuxCrossCc}/bin/${linuxTargetPrefix}cc";
      CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_AR = "${linuxCrossBinutils}/bin/${linuxTargetPrefix}ar";
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
      fd
      ripgrep
      neovim
      lazygit
      fzf
      delta
      eza
      yazi
      starship
      # xwin downloads Windows SDK/MSVC redistributables so we can link MSVC builds
      # without requiring a Windows VM.
      xwin
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
    ]
    ++ lib.optionals isDarwin [
      # macOS host prerequisites: Xcode headers, OpenMP runtime for SIMD code,
      # libiconv for CLI/build tooling, plus cross-compilers for the Linux target.
      appleSdk
      llvmPackages.openmp
      libiconv
      linuxCrossCc
      linuxCrossBinutils
    ];

  scripts = {
    "cargo:fmt".exec = cargoScript "cargo fmt";
    "cargo:lint".exec = cargoScript "cargo clippy --all-targets --all-features";
    "cargo:test".exec = cargoScript "cargo test";
    "build:cli".exec = cargoScript "cargo build --release --bin rigel";
    "build:native".exec = cargoScript "cargo xtask bundle rigel-plugin --release";
    "build:linux".exec =
      cargoScript "cargo xtask bundle rigel-plugin --release --target x86_64-unknown-linux-gnu";
    "build:macos".exec =
      cargoScript "cargo xtask bundle rigel-plugin --release --target aarch64-apple-darwin";
    # Windows bundles (VST3/CLAP) require MSVC import libs and link.exe normally.
    # xwin replicates those redistributables so we can link with rust-lld instead.
    # The script populates a shared cache under target/ then wires the toolchain
    # env vars Cargo expects for the MSVC target.
    "build:win".exec = "${./ci/scripts/build-win.sh}";
    "build:clean".exec = "rm -rf target";
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
    cargo fmt -- --check
    cargo clippy --all-targets --all-features -- -D warnings
    cargo test
  '';

  tasks = {
    "ci:check".description = "Run formatting, linting, and tests (matches enterTest)";
    "ci:check".exec = ''
      set -euo pipefail
      cargo fmt -- --check
      cargo clippy --all-targets --all-features -- -D warnings
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
