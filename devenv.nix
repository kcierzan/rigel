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
    "aarch64-apple-darwin"
    "x86_64-apple-darwin"
    "x86_64-pc-windows-msvc"
    "x86_64-unknown-linux-gnu"
  ];
  linuxCross = pkgs.pkgsCross.gnu64;
  linuxCrossCc = linuxCross.stdenv.cc;
  linuxCrossBinutils = linuxCross.buildPackages.binutils;
  linuxTargetPrefix = linuxCrossCc.targetPrefix or "";
  linuxPkgs = inputs.nixpkgs.legacyPackages."x86_64-linux";
  linuxXorg = if linuxPkgs ? xorg then linuxPkgs.xorg else { };
  linuxPkgConfigPaths =
    lib.concatLists [
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
  linuxLibraryPaths =
    lib.concatLists [
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
    linuxPkgConfigPaths
    ++ lib.optional (hostPkgConfigPath != "") hostPkgConfigPath;
  hasLinuxPkgConfig = linuxPkgConfigSearchPaths != [ ];
  linuxPkgConfigPath =
    lib.concatStringsSep ":" linuxPkgConfigSearchPaths;
  hasLinuxLibraryPaths = linuxLibraryPaths != [ ];
  linuxLdFlags =
    lib.concatStringsSep " " (map (path: "-L${path}") linuxLibraryPaths);
in
{
  devenv.root = lib.mkDefault (
    let
      envPwd = builtins.getEnv "PWD";
      fallback = toString ./.;
    in
    if envPwd != "" then envPwd else fallback
  );

  env =
    {
      RUST_BACKTRACE = lib.mkDefault "1";
      MACOSX_DEPLOYMENT_TARGET = lib.mkIf isDarwin "11.0";
      PKG_CONFIG_ALLOW_CROSS = lib.mkDefault "1";
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
      linuxCrossCc
      linuxCrossBinutils
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
