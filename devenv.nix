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

  rustTargets = [
    "x86_64-apple-darwin"
    "aarch64-apple-darwin"
    "x86_64-pc-windows-msvc"
  ];
in
{
  devenv.root = lib.mkDefault (
    let
      envPwd = builtins.getEnv "PWD";
      fallback = toString ./.;
    in
    if envPwd != "" then envPwd else fallback
  );

  env = {
    RUST_BACKTRACE = lib.mkDefault "1";
    MACOSX_DEPLOYMENT_TARGET = lib.mkIf isDarwin "11.0";
  };

  languages.rust = {
    enable = true;
    channel = "stable";
    components = [
      "rustfmt"
      "clippy"
      "rust-src"
      "rust-analyzer"
    ];
    targets = rustTargets;
  };

  packages =
    with pkgs;
    [
      git
      pkg-config
      cmake
      ninja
      zip
      unzip
      just
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
    ]
    ++ lib.optionals isDarwin [
      appleSdk
      llvmPackages.openmp
      libiconv
    ];

  scripts = {
    "cargo:fmt".exec = "cargo fmt";
    "cargo:lint".exec = "cargo clippy --all-targets --all-features";
    "cargo:test".exec = "cargo test";
    "plugin:bundle".exec = "cargo xtask bundle rigel-plugin --release";
  };

  enterShell = ''
    set -euo pipefail

    # Ensure rustup's toolchain shims are preferred (fixes cargo:lint clippy driver path).
    export PATH="$HOME/.cargo/bin:$PATH"

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
}
