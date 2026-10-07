#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
CACHE="${NATIVE_CACHE:-$ROOT/cache/native-build}"
SDK="$CACHE/mlx-sdk"
DOWNLOADS="$CACHE/downloads"
WITH_MLX="${DECKARD_BOOTSTRAP_MLX:-0}"
case "$WITH_MLX" in 0|1) ;; *) echo "DECKARD_BOOTSTRAP_MLX must be 0 or 1." >&2; exit 1 ;; esac
command -v cmake >/dev/null || { echo "Install CMake first (for example: brew install cmake)." >&2; exit 1; }
command -v curl >/dev/null || { echo "Install curl first." >&2; exit 1; }
if command -v shasum >/dev/null 2>&1; then
  SHASUM="shasum -a 256"
else
  SHASUM="sha256sum"
fi

fetch() {
  url=$1
  file=$2
  expected=$3
  if [ ! -f "$file" ]; then
    curl --fail --location --proto '=https' --proto-redir '=https' \
      --connect-timeout 30 --max-time 900 --output "$file.partial" "$url"
    actual=$($SHASUM "$file.partial" | cut -d' ' -f1)
    [ "$actual" = "$expected" ] || { echo "Download checksum mismatch." >&2; exit 1; }
    mv "$file.partial" "$file"
  fi
  actual=$($SHASUM "$file" | cut -d' ' -f1)
  [ "$actual" = "$expected" ] || { echo "Cached dependency checksum mismatch: $file" >&2; exit 1; }
}

bootstrap_apple() {
  case "$(uname -s)/$(uname -m)" in
    Darwin/arm64) ;;
    *) echo "The Gradient Core ML runtime requires Apple Silicon macOS 15 or newer." >&2; exit 1 ;;
  esac
  MAJOR=$(sw_vers -productVersion | cut -d. -f1)
  [ "$MAJOR" -ge 15 ] || { echo "macOS 15 or newer is required." >&2; exit 1; }
  xcrun --find clang++ >/dev/null
  if [ "$CACHE" != "$ROOT/cache/native-build" ]; then
    for required in json/include/nlohmann/json.hpp cargo/registry \
      rustup/toolchains/1.90.0-aarch64-apple-darwin/bin/cargo \
      licenses/NLOHMANN-LICENSE; do
      [ -e "$CACHE/$required" ] || {
        echo "External NATIVE_CACHE is read-only and incomplete: $required" >&2
        exit 1
      }
    done
    if [ "$WITH_MLX" = 1 ]; then
      for required in mlx-sdk/mlx/share/cmake/MLX/MLXConfig.cmake licenses/MLX-LICENSE; do
        [ -e "$CACHE/$required" ] || {
          echo "External NATIVE_CACHE is read-only and incomplete: $required" >&2
          exit 1
        }
      done
    fi
    echo "Using existing read-only native dependencies in $CACHE"
    exit 0
  fi
  mkdir -p "$DOWNLOADS"
  mkdir -p "$CACHE/json/include/nlohmann" "$CACHE/licenses"

  # Optional SDK for historical MLX research, never needed by production Core ML.
  if [ "$WITH_MLX" = 1 ]; then
  mkdir -p "$SDK"
  fetch 'https://files.pythonhosted.org/packages/79/ec/34f37376e26d537fadffb99af3a760d6545e37f5e1a30a552baadf237fc5/mlx_metal-0.32.2-py3-none-macosx_15_0_arm64.whl' \
    "$DOWNLOADS/mlx-metal.whl" 55a369250d220b2cf10213a87a2ac1b1a420608c5b35b1df4e7147ac8e32f121
  unzip -oq "$DOWNLOADS/mlx-metal.whl" 'mlx/include/*' 'mlx/share/*' 'mlx/lib/*' -d "$SDK"
  unzip -p "$DOWNLOADS/mlx-metal.whl" 'mlx_metal-0.32.2.dist-info/licenses/LICENSE' > "$CACHE/licenses/MLX-LICENSE"
  fi
  fetch 'https://raw.githubusercontent.com/nlohmann/json/9cca280a4d0ccf0c08f47a99aa71d1b0e52f8d03/single_include/nlohmann/json.hpp' \
    "$CACHE/json/include/nlohmann/json.hpp" 9bea4c8066ef4a1c206b2be5a36302f8926f7fdc6087af5d20b417d0cf103ea6
  fetch 'https://raw.githubusercontent.com/nlohmann/json/9cca280a4d0ccf0c08f47a99aa71d1b0e52f8d03/LICENSE.MIT' \
    "$CACHE/licenses/NLOHMANN-LICENSE" 86b998c792894ccb911a1cb7994f7a9652894e7a094c0b5e45be2f553f45cf14
  fetch 'https://huggingface.co/ShantanuT01/gradient-ai-text-detector/raw/c2e8b6df87f8a211cbffb713fa9873a0c3a9713f/README.md' \
    "$CACHE/licenses/GRADIENT-MODEL-CARD.md" a147869ab24ad59172bcdc7cc69cf716c646ddf0745a0e1fbb12515a2c6e752a

  export CARGO_HOME="$CACHE/cargo"
  export RUSTUP_HOME="$CACHE/rustup"
  export RUSTUP_TOOLCHAIN=1.90.0
  export CARGO_BUILD_JOBS=2
  if [ ! -x "$CARGO_HOME/bin/cargo" ]; then
    rustup_url='https://static.rust-lang.org/rustup/archive/1.28.2/aarch64-apple-darwin/rustup-init'
    curl --fail --location --proto '=https' --proto-redir '=https' \
      --output "$DOWNLOADS/rustup-init.sha256" "$rustup_url.sha256"
    rustup_sha=$(cut -d' ' -f1 "$DOWNLOADS/rustup-init.sha256")
    [ "${#rustup_sha}" -eq 64 ] || { echo "Invalid Rust bootstrap checksum." >&2; exit 1; }
    case "$rustup_sha" in *[!0-9a-f]*) echo "Invalid Rust bootstrap checksum." >&2; exit 1 ;; esac
    fetch "$rustup_url" "$DOWNLOADS/rustup-init" "$rustup_sha"
    chmod 700 "$DOWNLOADS/rustup-init"
    "$DOWNLOADS/rustup-init" -y --no-modify-path --profile minimal --default-toolchain 1.90.0
  fi
  "$CARGO_HOME/bin/rustc" --version
  "$CARGO_HOME/bin/cargo" fetch --locked --manifest-path "$ROOT/native-cli/tokenizer/Cargo.toml"
  echo "Native dependencies ready in $CACHE"
}

bootstrap_linux() {
  case "$(uname -s)/$(uname -m)" in
    Linux/x86_64) ;;
    *) echo "The Gradient Candle runtime requires Linux x86_64." >&2; exit 1 ;;
  esac
  if [ -n "${CC:-}" ] && ! command -v "$CC" >/dev/null 2>&1; then
    echo "CC points to an unavailable compiler: $CC" >&2
    exit 1
  fi
  command -v "${CC:-cc}" >/dev/null 2>&1 || { echo "Install a C++ compiler (for example: sudo apt install g++)." >&2; exit 1; }
  command -v pkg-config >/dev/null || { echo "Install pkg-config first." >&2; exit 1; }
  pkg-config --exists openssl || { echo "OpenSSL development headers are required (for example: sudo apt install libssl-dev)." >&2; exit 1; }
  pkg-config --exists libcurl || { echo "libcurl development headers are required (for example: sudo apt install libcurl4-openssl-dev)." >&2; exit 1; }
  if [ "$CACHE" != "$ROOT/cache/native-build" ]; then
    for required in json/include/nlohmann/json.hpp cargo/registry \
      rustup/toolchains/1.90.0-x86_64-unknown-linux-gnu/bin/cargo \
      licenses/NLOHMANN-LICENSE onnxruntime/include/onnxruntime_c_api.h onnxruntime/lib/libonnxruntime.so \
      licenses/ONNXRUNTIME-LICENSE onnxruntime-openvino/libonnxruntime.so.1.24.1 \
      licenses/ONNXRUNTIME-OPENVINO-LICENSE; do
      [ -e "$CACHE/$required" ] || {
        echo "External NATIVE_CACHE is read-only and incomplete: $required" >&2
        exit 1
      }
    done
    echo "Using existing read-only native dependencies in $CACHE"
    exit 0
  fi
  mkdir -p "$DOWNLOADS"
  mkdir -p "$CACHE/json/include/nlohmann" "$CACHE/licenses"
  fetch 'https://raw.githubusercontent.com/nlohmann/json/9cca280a4d0ccf0c08f47a99aa71d1b0e52f8d03/single_include/nlohmann/json.hpp' \
    "$CACHE/json/include/nlohmann/json.hpp" 9bea4c8066ef4a1c206b2be5a36302f8926f7fdc6087af5d20b417d0cf103ea6
  fetch 'https://raw.githubusercontent.com/nlohmann/json/9cca280a4d0ccf0c08f47a99aa71d1b0e52f8d03/LICENSE.MIT' \
    "$CACHE/licenses/NLOHMANN-LICENSE" 86b998c792894ccb911a1cb7994f7a9652894e7a094c0b5e45be2f553f45cf14
  fetch 'https://huggingface.co/ShantanuT01/gradient-ai-text-detector/raw/c2e8b6df87f8a211cbffb713fa9873a0c3a9713f/README.md' \
    "$CACHE/licenses/GRADIENT-MODEL-CARD.md" a147869ab24ad59172bcdc7cc69cf716c646ddf0745a0e1fbb12515a2c6e752a
  # ONNX Runtime: embedded by CMake into the deckard binary and dlopened over
  # an anonymous memfd by onnx_backend.cpp whenever model.onnx is present. It
  # is staged into the build cache only, so deckard never depends on a system
  # package or an external runtime file.
  fetch 'https://github.com/microsoft/onnxruntime/releases/download/v1.30.0/onnxruntime-linux-x64-1.30.0.tgz' \
    "$DOWNLOADS/onnxruntime-linux-x64-1.30.0.tgz" a5ed5a3cac51fbb2e90da632ae43d19212faaa20e76484e62bcb7c23ddb3b3fd
  mkdir -p "$CACHE/onnxruntime"
  tar xzf "$DOWNLOADS/onnxruntime-linux-x64-1.30.0.tgz" -C "$CACHE/onnxruntime" --strip-components=1
  cp "$CACHE/onnxruntime/LICENSE" "$CACHE/licenses/ONNXRUNTIME-LICENSE"
  # Intel iGPU GPU execution provider. ONNX Runtime's OpenVINO EP lives only in
  # the ORT 1.24.x "openvino" release (API v24 structurally ABI-compatible with
  # our v1.30 consumer headers); newer releases no longer ship a GPU EP. The
  # wheel carries the core, the OpenVINO provider, and the full OpenVINO 2025.4
  # runtime as regular .so files that onnx_backend.cpp embeds verbatim and
  # materializes to the per-user cache directory only when an Intel GPU is
  # present. The OpenVINO CPU plugin is omitted: CPU inference falls back to the
  # ONNX Runtime CPU EP built into the same core.
  fetch 'https://files.pythonhosted.org/packages/99/16/69ca742f0b65c40d4de3ff44bb6abc23c47b23e932bc901116176ae69922/onnxruntime_openvino-1.24.1-cp311-cp311-manylinux_2_28_x86_64.whl' \
    "$DOWNLOADS/onnxruntime-openvino-1.24.1.whl" 3007c803634cc69c6d52af1dea7ce729d9bb62b9a11070fd2f959119199007a8
  mkdir -p "$CACHE/onnxruntime-openvino"
  unzip -joq "$DOWNLOADS/onnxruntime-openvino-1.24.1.whl" \
    'onnxruntime/capi/libonnxruntime.so.1.24.1' \
    'onnxruntime/capi/libonnxruntime_providers_openvino.so' \
    'onnxruntime/capi/libonnxruntime_providers_shared.so' \
    'onnxruntime/capi/libopenvino.so.2541' \
    'onnxruntime/capi/libopenvino_c.so.2541' \
    'onnxruntime/capi/libopenvino_onnx_frontend.so.2541' \
    'onnxruntime/capi/libopenvino_intel_gpu_plugin.so' \
    'onnxruntime/capi/libtbb.so.12' \
    'onnxruntime/capi/libtbbmalloc.so' \
    -d "$CACHE/onnxruntime-openvino"
  unzip -poq "$DOWNLOADS/onnxruntime-openvino-1.24.1.whl" \
    'onnxruntime/LICENSE' > "$CACHE/licenses/ONNXRUNTIME-OPENVINO-LICENSE"
  unzip -poq "$DOWNLOADS/onnxruntime-openvino-1.24.1.whl" \
    'onnxruntime/ThirdPartyNotices.txt' > "$CACHE/licenses/ONNXRUNTIME-OPENVINO-THIRD-PARTY-NOTICES.txt"
  # A symlink clone of an existing system toolchain satisfies the pinned SDK
  # toolchain path offline, without ever contacting static.rust-lang.org.
  TOOLCHAIN="$CACHE/rustup/toolchains/1.90.0-x86_64-unknown-linux-gnu"
  if [ ! -x "$TOOLCHAIN/bin/cargo" ]; then
    existing=''
    for dir in "${RUSTUP_HOME:-$HOME/.rustup}" "$HOME/.rustup"; do
      [ -d "$dir/toolchains" ] || continue
      for path in "$dir"/toolchains/*-x86_64-unknown-linux-gnu; do
        [ -x "$path/bin/cargo" ] && existing=$path && break 2
      done
    done
    [ -n "$existing" ] || {
      echo "No x86_64-unknown-linux-gnu Rust toolchain found. Install Rust (rustup-init) and re-run bootstrap." >&2
      exit 1
    }
    mkdir -p "$(dirname "$TOOLCHAIN")"
    ln -s "$existing" "$TOOLCHAIN"
  fi
  "$TOOLCHAIN/bin/rustc" --version
  export CARGO_HOME="$CACHE/cargo"
  export RUSTUP_HOME="$CACHE/rustup"
  export CARGO_BUILD_JOBS=2
  "$TOOLCHAIN/bin/cargo" fetch --locked --manifest-path "$ROOT/native-cli/tokenizer/Cargo.toml"
  "$TOOLCHAIN/bin/cargo" fetch --locked --manifest-path "$ROOT/native-cli/candle/Cargo.toml"
  echo "Native dependencies ready in $CACHE"
}

case "$(uname -s)/$(uname -m)" in
  Darwin/arm64) bootstrap_apple ;;
  Linux/x86_64) bootstrap_linux ;;
  *) echo "The Gradient runtime requires Apple Silicon macOS 15 or newer, or Linux x86_64." >&2; exit 1 ;;
esac
