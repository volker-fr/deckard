#!/bin/sh
# Linux x86_64 counterpart of bootstrap.sh, which execs this script on Linux.
# Fetches the pinned build dependencies of the Candle backend into NATIVE_CACHE:
# nlohmann/json, license notices, the Rust 1.90.0 toolchain (via a checksummed
# rustup-init, exactly like the macOS bootstrap) and the offline crate registry
# for the tokenizer and Candle crates. System packages (a C++ compiler, CMake,
# pkg-config, OpenSSL and libcurl development headers) are only checked, never
# installed.
# It also fetches ONNX Runtime for the ONNX Runtime backend: the headers from
# the official release, the libraries (core, OpenVINO execution provider, the
# OpenVINO parts it needs for Intel GPUs) from the onnxruntime-openvino wheel,
# and their license notices.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
CACHE="${NATIVE_CACHE:-$ROOT/cache/native-build}"
DOWNLOADS="$CACHE/downloads"
HOST=x86_64-unknown-linux-gnu
TOOLCHAIN="$CACHE/rustup/toolchains/1.90.0-$HOST"
# ONNX Runtime and the OpenVINO in its wheel; the versions must match linux.cmake.
ORT_VERSION=1.24.1
OPENVINO_VERSION=2025.4.1
ORT="$CACHE/onnxruntime-$ORT_VERSION"

fail() { echo "$*" >&2; exit 1; }

[ "$(uname -m)" = x86_64 ] || fail "The Gradient Candle runtime requires Linux x86_64."
for tool in cmake curl pkg-config unzip objcopy; do
  command -v "$tool" >/dev/null || fail "Install $tool first."
done
command -v "${CXX:-c++}" >/dev/null || fail "Install a C++20 compiler (g++ or clang++) first."
pkg-config --exists openssl || fail "Install the OpenSSL development headers (libssl-dev, openssl-devel or openssl)."
pkg-config --exists libcurl || fail "Install the libcurl development headers (libcurl4-openssl-dev, libcurl-devel or curl)."

# A toolchain directory that is not really 1.90.0 (for example a symlink to a
# system toolchain) would silently break the pinned, reproducible build.
require_pinned_toolchain() {
  version=$("$TOOLCHAIN/bin/rustc" --version 2>/dev/null) || fail "The Rust toolchain in $TOOLCHAIN is not runnable."
  case "$version" in
    "rustc 1.90.0 "*) ;;
    *) fail "$TOOLCHAIN is $version, not the pinned 1.90.0. Remove it and re-run bootstrap." ;;
  esac
}

if [ "$CACHE" != "$ROOT/cache/native-build" ]; then
  for required in json/include/nlohmann/json.hpp cargo/registry "rustup/toolchains/1.90.0-$HOST/bin/cargo" \
    licenses/NLOHMANN-LICENSE "onnxruntime-$ORT_VERSION/include/onnxruntime_c_api.h" licenses/ONNXRUNTIME-LICENSE \
    "onnxruntime-$ORT_VERSION/lib/libonnxruntime.so.$ORT_VERSION" licenses/ONNXRUNTIME-THIRD-PARTY-NOTICES.txt \
    "onnxruntime-$ORT_VERSION/lib/libopenvino_intel_gpu_plugin.so" licenses/OPENVINO-LICENSE \
    licenses/OPENVINO-THIRD-PARTY-PROGRAMS.txt; do
    [ -e "$CACHE/$required" ] || fail "External NATIVE_CACHE is read-only and incomplete: $required"
  done
  require_pinned_toolchain
  echo "Using existing read-only native dependencies in $CACHE"
  exit 0
fi
mkdir -p "$DOWNLOADS" "$CACHE/json/include/nlohmann" "$CACHE/licenses"

fetch() {
  url=$1
  file=$2
  expected=$3
  if [ ! -f "$file" ]; then
    curl --fail --location --proto '=https' --proto-redir '=https' \
      --connect-timeout 30 --max-time 900 --output "$file.partial" "$url"
    actual=$(sha256sum "$file.partial" | cut -d' ' -f1)
    [ "$actual" = "$expected" ] || fail "Download checksum mismatch."
    mv "$file.partial" "$file"
  fi
  actual=$(sha256sum "$file" | cut -d' ' -f1)
  [ "$actual" = "$expected" ] || fail "Cached dependency checksum mismatch: $file"
}

fetch 'https://raw.githubusercontent.com/nlohmann/json/9cca280a4d0ccf0c08f47a99aa71d1b0e52f8d03/single_include/nlohmann/json.hpp' \
  "$CACHE/json/include/nlohmann/json.hpp" 9bea4c8066ef4a1c206b2be5a36302f8926f7fdc6087af5d20b417d0cf103ea6
fetch 'https://raw.githubusercontent.com/nlohmann/json/9cca280a4d0ccf0c08f47a99aa71d1b0e52f8d03/LICENSE.MIT' \
  "$CACHE/licenses/NLOHMANN-LICENSE" 86b998c792894ccb911a1cb7994f7a9652894e7a094c0b5e45be2f553f45cf14
fetch 'https://huggingface.co/ShantanuT01/gradient-ai-text-detector/raw/c2e8b6df87f8a211cbffb713fa9873a0c3a9713f/README.md' \
  "$CACHE/licenses/GRADIENT-MODEL-CARD.md" a147869ab24ad59172bcdc7cc69cf716c646ddf0745a0e1fbb12515a2c6e752a

ORT_ARCHIVE="onnxruntime-linux-x64-$ORT_VERSION"
fetch "https://github.com/microsoft/onnxruntime/releases/download/v$ORT_VERSION/$ORT_ARCHIVE.tgz" \
  "$DOWNLOADS/$ORT_ARCHIVE.tgz" 9142552248b735920f9390027e4512a2cacf8946a1ffcbe9071a5c210531026f
mkdir -p "$ORT/lib"
tar -xzf "$DOWNLOADS/$ORT_ARCHIVE.tgz" -C "$ORT" --strip-components=1 "$ORT_ARCHIVE/include"
# The libraries come from the onnxruntime-openvino wheel, the only build that
# can load the OpenVINO execution provider: the core, the provider and the
# parts of OpenVINO it needs for Intel GPUs (no CPU or NPU plugin). deckard
# embeds them and writes them next to each other; their RPATH is $ORIGIN.
ORT_WHEEL="onnxruntime_openvino-$ORT_VERSION-cp311-cp311-manylinux_2_28_x86_64.whl"
fetch "https://files.pythonhosted.org/packages/99/16/69ca742f0b65c40d4de3ff44bb6abc23c47b23e932bc901116176ae69922/$ORT_WHEEL" \
  "$DOWNLOADS/$ORT_WHEEL" 3007c803634cc69c6d52af1dea7ce729d9bb62b9a11070fd2f959119199007a8
# OpenVINO 2025.4.1 names its libraries *.so.2541.
OPENVINO_SO=$(echo "$OPENVINO_VERSION" | cut -c3- | tr -d .)
unzip -joq "$DOWNLOADS/$ORT_WHEEL" -d "$ORT/lib" \
  "onnxruntime/capi/libonnxruntime.so.$ORT_VERSION" \
  onnxruntime/capi/libonnxruntime_providers_shared.so \
  onnxruntime/capi/libonnxruntime_providers_openvino.so \
  "onnxruntime/capi/libopenvino.so.$OPENVINO_SO" \
  "onnxruntime/capi/libopenvino_onnx_frontend.so.$OPENVINO_SO" \
  onnxruntime/capi/libopenvino_intel_gpu_plugin.so \
  onnxruntime/capi/libtbb.so.12 \
  onnxruntime/capi/libtbbmalloc.so
unzip -poq "$DOWNLOADS/$ORT_WHEEL" onnxruntime/LICENSE > "$CACHE/licenses/ONNXRUNTIME-LICENSE"
unzip -poq "$DOWNLOADS/$ORT_WHEEL" onnxruntime/ThirdPartyNotices.txt \
  > "$CACHE/licenses/ONNXRUNTIME-THIRD-PARTY-NOTICES.txt"
# The wheel ships OpenVINO and oneTBB without their notices; OpenVINO's
# third-party programs file covers oneTBB.
fetch "https://raw.githubusercontent.com/openvinotoolkit/openvino/$OPENVINO_VERSION/LICENSE" \
  "$CACHE/licenses/OPENVINO-LICENSE" c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4
fetch "https://raw.githubusercontent.com/openvinotoolkit/openvino/$OPENVINO_VERSION/licensing/third-party-programs.txt" \
  "$CACHE/licenses/OPENVINO-THIRD-PARTY-PROGRAMS.txt" 687a0c5323cddd59f361dd9819dd5f48dddd967b0c1770c75d1085843407ecb4

export CARGO_HOME="$CACHE/cargo"
export RUSTUP_HOME="$CACHE/rustup"
export RUSTUP_TOOLCHAIN=1.90.0
export CARGO_BUILD_JOBS=2
if [ ! -x "$CARGO_HOME/bin/cargo" ]; then
  rustup_url="https://static.rust-lang.org/rustup/archive/1.28.2/$HOST/rustup-init"
  curl --fail --location --proto '=https' --proto-redir '=https' \
    --output "$DOWNLOADS/rustup-init.sha256" "$rustup_url.sha256"
  rustup_sha=$(cut -d' ' -f1 "$DOWNLOADS/rustup-init.sha256")
  [ "${#rustup_sha}" -eq 64 ] || fail "Invalid Rust bootstrap checksum."
  case "$rustup_sha" in *[!0-9a-f]*) fail "Invalid Rust bootstrap checksum." ;; esac
  fetch "$rustup_url" "$DOWNLOADS/rustup-init" "$rustup_sha"
  chmod 700 "$DOWNLOADS/rustup-init"
  "$DOWNLOADS/rustup-init" -y --no-modify-path --profile minimal --default-toolchain 1.90.0
fi
require_pinned_toolchain
"$CARGO_HOME/bin/cargo" fetch --locked --manifest-path "$ROOT/native-cli/tokenizer/Cargo.toml"
"$CARGO_HOME/bin/cargo" fetch --locked --manifest-path "$ROOT/native-cli/candle/Cargo.toml"
echo "Native dependencies ready in $CACHE"
