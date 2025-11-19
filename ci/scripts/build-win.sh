#!/usr/bin/env bash
set -euo pipefail

# Mirror devenv's cargoScript helper so CI and containers reuse the same layout.
state_dir="${DEVENV_STATE:-$PWD/.devenv/state}"
export DEVENV_STATE="$state_dir"
export CARGO_HOME="${CARGO_HOME:-$state_dir/cargo}"
export RUSTUP_HOME="${RUSTUP_HOME:-$state_dir/rustup}"
mkdir -p "$CARGO_HOME/bin" "$RUSTUP_HOME"

# Share the xwin download/splat output across invocations; this keeps the
# sync time manageable while letting CI point at a custom CARGO_TARGET_DIR.
TARGET_DIR="${CARGO_TARGET_DIR:-$PWD/target}"
XWIN_CACHE="${XWIN_CACHE:-$TARGET_DIR/xwin-cache}"
XWIN_OUT="${XWIN_OUT:-$TARGET_DIR/xwin}"
mkdir -p "$XWIN_CACHE" "$XWIN_OUT"

# Stage the Windows SDK + MSVC CRT into the local cache. download/unpack only
# run when new manifests show up; splat adjusts casing/symlinks for POSIX fs.
xwin --accept-license --cache-dir "$XWIN_CACHE" download
xwin --accept-license --cache-dir "$XWIN_CACHE" unpack
xwin --accept-license --cache-dir "$XWIN_CACHE" splat \
  --output "$XWIN_OUT" \
  --include-debug-libs \
  --include-debug-symbols

# rustup exposes rust-lld/llvm-ar under the active host triple. rust-lld can
# consume the MSVC import libraries, so we avoid shipping link.exe.
SYSROOT="$(rustc --print sysroot)"
HOST_TRIPLE="$(rustc -vV | sed -n 's/^host: //p')"
RUST_LLD="$SYSROOT/lib/rustlib/$HOST_TRIPLE/bin/rust-lld"
LLVM_AR="$SYSROOT/lib/rustlib/$HOST_TRIPLE/bin/llvm-ar"

if [ ! -x "$RUST_LLD" ]; then
  echo "error: rust-lld not found at $RUST_LLD; ensure llvm-tools-preview is installed." >&2
  exit 1
fi
if [ ! -x "$LLVM_AR" ]; then
  echo "error: llvm-ar not found at $LLVM_AR; ensure llvm-tools-preview is installed." >&2
  exit 1
fi

export CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER="$RUST_LLD"
export CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_AR="$LLVM_AR"

# Set up LIB/LIBPATH/INCLUDE so the MSVC target sees the Windows SDK + CRT.
# The `;` separators mirror how MSVC tooling expects these values.
SDK_LIB_UM="$XWIN_OUT/sdk/lib/um/x86_64"
SDK_LIB_UCRT="$XWIN_OUT/sdk/lib/ucrt/x86_64"
CRT_LIB="$XWIN_OUT/crt/lib/x86_64"

export LIB="$SDK_LIB_UM;$SDK_LIB_UCRT;$CRT_LIB"
export LIBPATH="$LIB"
export INCLUDE="$XWIN_OUT/sdk/include/ucrt;$XWIN_OUT/sdk/include/um;$XWIN_OUT/sdk/include/shared"

# Some crates (bindgen build scripts, etc.) rely on MSVC helper binaries,
# so we temporarily prepend the extracted toolchain bin directory if found.
TOOLS_BIN="$(find "$XWIN_OUT/tools" -maxdepth 3 -type d -name bin 2>/dev/null | head -n1 || true)"
if [ -n "$TOOLS_BIN" ]; then
  export PATH="$TOOLS_BIN:$PATH"
fi

export RUSTFLAGS="${RUSTFLAGS:-} -Lnative=$SDK_LIB_UM -Lnative=$SDK_LIB_UCRT -Lnative=$CRT_LIB"

# Pass through any additional arguments (e.g., --verbose)
cargo xtask bundle rigel-plugin --release --target x86_64-pc-windows-msvc "$@"
