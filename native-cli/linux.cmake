# Linux x86_64 build of the deckard binary, included from CMakeLists.txt.
#
# Inference runs on the Candle backend (candle/), a Rust staticlib that loads the
# pinned fp32 Hugging Face checkpoint (config.json + model.safetensors) on the
# CPU. It is built offline with the same pinned toolchain and registry snapshot
# as the tokenizer. This file sets DECKARD_PLATFORM_SOURCES,
# DECKARD_PLATFORM_LIBRARIES and EXTRA_CARGO_LOCKS for the shared target.
#
# ONNX Runtime runs the pinned model.onnx when it is installed (see
# docs/LINUX.md). Its shared library is embedded in deckard and written to the
# user cache on first use (onnx_runtime.cpp), so no system package is needed.

set(CANDLE_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/candle")
set(CANDLE_TARGET "${CMAKE_CURRENT_BINARY_DIR}/candle-rust")
set(CANDLE_LIBRARY "${CANDLE_TARGET}/release/libai_hider_candle.a")
add_custom_command(
  OUTPUT "${CANDLE_LIBRARY}"
  COMMAND "${CMAKE_COMMAND}" -E env
    "CARGO_HOME=${BUILD_CARGO_HOME}" "RUSTUP_HOME=${NATIVE_CACHE}/rustup"
    "RUSTC=${RUST_TOOLCHAIN}/bin/rustc"
    "CARGO_ENCODED_RUSTFLAGS=${RUST_PATH_FLAGS}"
    "RUSTUP_TOOLCHAIN=1.90.0" "CARGO_TARGET_DIR=${CANDLE_TARGET}" "CARGO_BUILD_JOBS=2"
    "${CARGO}" build --release --locked --offline --manifest-path "${CANDLE_SOURCE}/Cargo.toml"
  DEPENDS "${CANDLE_SOURCE}/Cargo.toml" "${CANDLE_SOURCE}/Cargo.lock" "${CANDLE_SOURCE}/src/lib.rs"
  VERBATIM)
add_custom_target(candle_build DEPENDS "${CANDLE_LIBRARY}")
# Both crates share one CARGO_HOME; building them one after the other avoids
# concurrent writes to its package cache.
add_dependencies(candle_build tokenizer_build)
add_library(candle_native STATIC IMPORTED)
set_target_properties(candle_native PROPERTIES IMPORTED_LOCATION "${CANDLE_LIBRARY}")
add_dependencies(candle_native candle_build)

# The model.onnx pin has its own file: install.json records the SHA-256 of
# model-assets-linux.json, so editing that file would leave existing
# installations unable to upgrade or uninstall.
set(ONNX_ASSETS_FILE "${CMAKE_CURRENT_SOURCE_DIR}/model-assets-onnx.json")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${ONNX_ASSETS_FILE}")
file(READ "${ONNX_ASSETS_FILE}" ONNX_ASSETS_JSON)
configure_file(src/onnx_assets.hpp.in "${CMAKE_CURRENT_BINARY_DIR}/generated/onnx_assets.hpp" @ONLY)

# ONNX Runtime and OpenVINO from the onnxruntime-openvino wheel; the versions
# must match bootstrap-linux.sh.
set(ONNXRUNTIME_VERSION 1.24.1)
set(OPENVINO_VERSION 2025.4.1)
# OpenVINO 2025.4.1 names its libraries *.so.2541.
string(SUBSTRING "${OPENVINO_VERSION}" 2 -1 OPENVINO_SO)
string(REPLACE "." "" OPENVINO_SO "${OPENVINO_SO}")
set(ONNXRUNTIME "${NATIVE_CACHE}/onnxruntime-${ONNXRUNTIME_VERSION}")

# Embeds each library as the read-only symbols _binary_<name>_bin_start/_end
# and lists it as DECKARD_EMBEDDED(name, "file", "sha256") in
# generated/embedded_libraries.inc; onnx_runtime.cpp writes them next to each
# other and checks each copy against its SHA-256. The core comes first.
set(EMBEDDED "${CMAKE_CURRENT_BINARY_DIR}/embedded")
file(MAKE_DIRECTORY "${EMBEDDED}")
set(EMBEDDED_OBJECTS)
set(EMBEDDED_ENTRIES "")
foreach(entry
    "onnxruntime:libonnxruntime.so.${ONNXRUNTIME_VERSION}"
    "ort_providers_shared:libonnxruntime_providers_shared.so"
    "ort_provider_openvino:libonnxruntime_providers_openvino.so"
    "openvino:libopenvino.so.${OPENVINO_SO}"
    "openvino_onnx_frontend:libopenvino_onnx_frontend.so.${OPENVINO_SO}"
    "openvino_intel_gpu_plugin:libopenvino_intel_gpu_plugin.so"
    "tbb:libtbb.so.12"
    "tbbmalloc:libtbbmalloc.so")
  string(REPLACE ":" ";" entry "${entry}")
  list(GET entry 0 name)
  list(GET entry 1 file)
  set(library "${ONNXRUNTIME}/lib/${file}")
  set(object "${EMBEDDED}/${name}.o")
  # objcopy names the symbols after its input path, so it runs next to it.
  add_custom_command(OUTPUT "${object}"
    COMMAND "${CMAKE_COMMAND}" -E copy "${library}" "${name}.bin"
    COMMAND objcopy -I binary -O elf64-x86-64 -B i386:x86-64
      --rename-section .data=.rodata,alloc,load,readonly,data,contents "${name}.bin" "${name}.o"
    COMMAND "${CMAKE_COMMAND}" -E remove "${name}.bin"
    WORKING_DIRECTORY "${EMBEDDED}" DEPENDS "${library}" VERBATIM)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${library}")
  file(SHA256 "${library}" sha256)
  list(APPEND EMBEDDED_OBJECTS "${object}")
  string(APPEND EMBEDDED_ENTRIES "DECKARD_EMBEDDED(${name}, \"${file}\", \"${sha256}\")\n")
endforeach()
file(CONFIGURE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/generated/embedded_libraries.inc" CONTENT "${EMBEDDED_ENTRIES}" @ONLY)
set_source_files_properties(src/onnx_runtime.cpp PROPERTIES
  COMPILE_DEFINITIONS "ONNXRUNTIME_VERSION=\"${ONNXRUNTIME_VERSION}\";OPENVINO_VERSION=\"${OPENVINO_VERSION}\"")

set(DECKARD_PLATFORM_SOURCES src/onnx_runtime.cpp src/onnx_providers.cpp src/onnx_gradient.cpp
  src/intel_gpu.cpp src/system_check.cpp ${EMBEDDED_OBJECTS})
set(DECKARD_PLATFORM_INCLUDES "${ONNXRUNTIME}/include")
# SHA-256 comes from OpenSSL's libcrypto EVP API on Linux (CommonCrypto on
# macOS). The Rust staticlibs' std needs libdl on glibc older than 2.34.
set(DECKARD_PLATFORM_LIBRARIES candle_native "-lcrypto" ${CMAKE_DL_LIBS})
set(EXTRA_CARGO_LOCKS "${CANDLE_SOURCE}/Cargo.lock")
