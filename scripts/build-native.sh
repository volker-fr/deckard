#!/bin/bash
set -euo pipefail
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
export NATIVE_CACHE="${NATIVE_CACHE:-$ROOT/cache/native-build}"
BUILD_DIR="${BUILD_DIR:-$ROOT/native-cli/build}"
[ "$#" -eq 0 ] || { echo 'Usage: NATIVE_CACHE=... BUILD_DIR=... scripts/build-native.sh' >&2; exit 1; }
run() {
  case "$(uname -s)" in
    Darwin) /usr/sbin/taskpolicy -b "$@" ;;
    Linux) "$@" ;;
    *) echo 'Deckard supports Apple Silicon macOS and Linux x86_64 only.' >&2; exit 1 ;;
  esac
}
for legacy in "$BUILD_DIR/dist/lib/libmlx.dylib" "$BUILD_DIR/dist/lib/mlx.metallib" \
  "$BUILD_DIR/dist/share/licenses/MLX-LICENSE" "$BUILD_DIR/licenses/MLX-LICENSE"; do
  if [ -e "$legacy" ] || [ -L "$legacy" ]; then
    printf 'Legacy MLX build artifact: %s\nChoose a fresh BUILD_DIR; existing output was left untouched.\n' "$legacy" >&2
    exit 1
  fi
done
run bash "$ROOT/native-cli/bootstrap.sh"
run cmake -S "$ROOT/native-cli" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release -DNATIVE_CACHE="$NATIVE_CACHE" \
  -DCMAKE_INSTALL_PREFIX="$BUILD_DIR/dist"
run cmake --build "$BUILD_DIR" --parallel 2
run cmake --install "$BUILD_DIR"
printf 'Built native distribution: %s/dist\n' "$BUILD_DIR"
