cmake_minimum_required(VERSION 3.24)
project(deckard VERSION 0.6.4 LANGUAGES CXX OBJCXX)
if(APPLE AND NOT CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
  message(FATAL_ERROR "The Core ML CPU/Neural Engine backend requires Apple Silicon macOS.")
elseif(UNIX AND NOT APPLE AND NOT CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
  message(FATAL_ERROR "The Candle backend requires Linux x86_64.")
elseif(NOT APPLE AND NOT UNIX)
  message(FATAL_ERROR "Deckard supports Apple Silicon macOS and Linux x86_64 only.")
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
set(CMAKE_OSX_DEPLOYMENT_TARGET "15.0")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin")

# -----------------------------------------------------------------------------
# Linux CUDA backend (always ON)
# The Candle GPU backend on Linux is compiled with nvcc, part of the NVIDIA
# CUDA toolkit that is NOT installed system-wide on this machine. We therefore
# compile the CUDA candle static library inside a throwaway container (no
# Dockerfile), using the NVIDIA CUDA devel image as the compiler host. CMake
# auto-detects the engine (docker, then podman); a Linux build fails without
# either of them.
#
# CUDA Compute Capability 75 (Turing):
#  1. Backward compatibility: natively targets entry-level and workstation
#     GPUs such as the NVIDIA T1000, T4 and RTX 20-series (TU117/Turing).
#  2. Latest minimum toolchain baseline: CUDA 12+ removed support for
#     Maxwell/Kepler (<6.0) and deprecated Pascal (6.x). Compute 75 is the
#     lowest baseline modern toolchains accept without throwing compilation
#     errors in candle-kernels.
#  3. Forward compatibility via driver JIT: PTX generated for Compute 75 is
#     automatically forward-compiled by the NVIDIA driver JIT at runtime on
#     newer architectures (Ampere 8.0/8.6, Ada Lovelace 8.9, Hopper 9.0,
#     Blackwell 10.0).
#
# The container build is the equivalent of running (from the repository root):
#
#   docker run --rm \
#     -e CUDA_COMPUTE_CAP=75 \
#     -e CUDARC_CUDA_VERSION=11080 \
#     -e CARGO_TARGET_DIR=/usr/src/app/native-cli/build/candle-cuda \
#     -e RUSTUP_HOME=/usr/src/app/cache/docker-rustup \
#     -e CARGO_HOME=/usr/src/app/cache/docker-cargo \
#     -v "$(pwd)":/usr/src/app \
#     -w /usr/src/app \
#     docker.io/nvidia/cuda:11.8.0-devel-ubuntu22.04 \
#     bash -lc 'apt-get update && apt-get install -y --no-install-recommends \
#       curl build-essential pkg-config libssl-dev ca-certificates && \
#       if [ ! -x "$CARGO_HOME/bin/cargo" ]; then \
#         curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
#           --no-modify-path; fi && \
#       export PATH="$CARGO_HOME/bin:$PATH" && \
#       cargo build --release --locked --features cuda \
#         --manifest-path native-cli/candle/Cargo.toml && \
#       mkdir -p native-cli/build/candle-cuda/lib && \
#       cp -a /usr/local/cuda/lib64/libcublas.so* /usr/local/cuda/lib64/libcublasLt.so* \
#         /usr/local/cuda/lib64/libcurand.so* /usr/local/cuda/lib64/libcudart.so* \
#         /usr/local/cuda/lib64/libnvrtc.so* \
#         native-cli/build/candle-cuda/lib/ && \
#       cp -a "$(find native-cli/build/candle-cuda/release/build \
#         -type f -name libmoe.a | head -n1)" native-cli/build/candle-cuda/lib/'
#
# (replacing `docker` with `podman` if that is the installed engine). The
# cache/docker-rustup and cache/docker-cargo mounts persist the toolchain and
# the crates.io download cache across builds, so dependencies are downloaded
# only once. CUDARC_CUDA_VERSION selects cudarc's CUDA bindings without a host
# nvcc and must match the devel image (CUDA 11.8 -> 11080). The same container
# step copies the CUDA runtime libraries (cudart/cublas/cublasLt/curand/nvrtc)
# into native-cli/build/candle-cuda/lib for runtime dlopen and the candle-kernels
# libmoe.a archive for linking. cudarc runs in dynamic-loading mode (see
# native-cli/candle/Cargo.toml and the vendored candle-core patch), so deckard
# itself links no CUDA library and starts without an NVIDIA driver: CUDA is used
# only when the driver is reachable, otherwise the model runs on CPU.
# -----------------------------------------------------------------------------

set(DECKARD_CUDA_HOME "/usr/local/cuda" CACHE PATH "CUDA toolkit root whose lib64 is added to the deckard runtime search path (libcublas/libcudart/driver are dlopened).")
set(DECKARD_CUDA_COMPUTE_CAP "75" CACHE STRING "CUDA compute capability targeted by the containerized candle-kernels build (see the 75 rationale above).")
set(DECKARD_CUDA_IMAGE "docker.io/nvidia/cuda:11.8.0-devel-ubuntu22.04" CACHE STRING "Fully-qualified CUDA devel container image used to compile the candle CUDA backend (podman requires the registry prefix). CUDA 11.8 is used because the build targets compute capability 75: candle-kernels 0.11.0 supplies its own __hmax_nan/__hmin_nan shims for sm<80, and CUDA 12 headers define those functions unconditionally, so any CUDA 12.x toolkit fails to compile the kernel crate for sm_75.")
set(DECKARD_CUDARC_CUDA_VERSION "11080" CACHE STRING "cudarc CUDA-version selector; must match DECKARD_CUDA_IMAGE (CUDA 11.8 -> 11080).")
get_filename_component(AI_HIDER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
if(DEFINED ENV{NATIVE_CACHE})
  set(NATIVE_CACHE "$ENV{NATIVE_CACHE}" CACHE PATH "Native dependency cache")
else()
  set(NATIVE_CACHE "${AI_HIDER_ROOT}/cache/native-build" CACHE PATH "Native dependency cache")
endif()
add_library(ai_hider_json INTERFACE)
target_include_directories(ai_hider_json INTERFACE "${NATIVE_CACHE}/json/include")

if(APPLE)
  set(MODEL_ASSETS_FILE "${CMAKE_CURRENT_SOURCE_DIR}/model-assets.json")
elseif(UNIX AND NOT APPLE)
  set(MODEL_ASSETS_FILE "${CMAKE_CURRENT_SOURCE_DIR}/model-assets-linux.json")
endif()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${MODEL_ASSETS_FILE}")
file(READ "${MODEL_ASSETS_FILE}" MODEL_ASSETS_JSON)
file(SHA256 "${MODEL_ASSETS_FILE}" MODEL_ASSETS_SHA256)
configure_file(src/model_assets.hpp.in "${CMAKE_CURRENT_BINARY_DIR}/generated/model_assets.hpp" @ONLY)

if(APPLE)
  set(RUST_TOOLCHAIN "${NATIVE_CACHE}/rustup/toolchains/1.90.0-aarch64-apple-darwin")
  set(RUSTC "${RUST_TOOLCHAIN}/bin/rustc")
elseif(UNIX AND NOT APPLE)
  if(EXISTS "${NATIVE_CACHE}/rustup/toolchains/1.90.0-x86_64-unknown-linux-gnu/bin/cargo")
    set(RUST_TOOLCHAIN "${NATIVE_CACHE}/rustup/toolchains/1.90.0-x86_64-unknown-linux-gnu")
    set(RUSTC "${RUST_TOOLCHAIN}/bin/rustc")
  else()
    # Offline Linux bootstrap: use the system Rust toolchain when the pinned
    # SDK toolchain is not present in NATIVE_CACHE (bootstrap.sh links it).
    find_program(SYSTEM_CARGO cargo REQUIRED)
    find_program(SYSTEM_RUSTC rustc REQUIRED)
    set(RUST_TOOLCHAIN "")
    set(CARGO "${SYSTEM_CARGO}" CACHE FILEPATH "Cargo executable")
    set(RUSTC "${SYSTEM_RUSTC}")
  endif()
endif()
if(NOT DEFINED CARGO OR CARGO STREQUAL "CARGO-NOTFOUND")
  set(CARGO "${RUST_TOOLCHAIN}/bin/cargo" CACHE FILEPATH "Cargo executable")
endif()
set(BUILD_CARGO_HOME "${CMAKE_CURRENT_BINARY_DIR}/cargo")
file(MAKE_DIRECTORY "${BUILD_CARGO_HOME}")
# Cargo may update lock/cache metadata even offline. Never point CARGO_HOME at
# the shared SDK: keep its registry snapshot in the writable build directory.
file(COPY "${NATIVE_CACHE}/cargo/registry" DESTINATION "${BUILD_CARGO_HOME}")
set(RUST_TARGET "${CMAKE_CURRENT_BINARY_DIR}/rust")
string(ASCII 31 RUST_FLAG_SEPARATOR)
set(RUST_PATH_FLAGS "--remap-path-prefix=${AI_HIDER_ROOT}=/deckard-source${RUST_FLAG_SEPARATOR}--remap-path-prefix=${NATIVE_CACHE}=/deckard-sdk")
set(tokenizer_env "CARGO_HOME=${BUILD_CARGO_HOME}" "RUSTUP_HOME=${NATIVE_CACHE}/rustup")
if(APPLE)
  list(APPEND tokenizer_env "MACOSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}")
endif()
list(APPEND tokenizer_env "RUSTC=${RUSTC}" "CARGO_ENCODED_RUSTFLAGS=${RUST_PATH_FLAGS}")
if(RUST_TOOLCHAIN)
  list(APPEND tokenizer_env "RUSTUP_TOOLCHAIN=1.90.0")
endif()
list(APPEND tokenizer_env "CARGO_TARGET_DIR=${RUST_TARGET}" "CARGO_BUILD_JOBS=2")
add_custom_command(
  OUTPUT "${RUST_TARGET}/release/libai_hider_tokenizer.a"
  COMMAND "${CMAKE_COMMAND}" -E env
    ${tokenizer_env}
    "${CARGO}" build --release --locked --offline --manifest-path "${CMAKE_CURRENT_SOURCE_DIR}/tokenizer/Cargo.toml"
  DEPENDS tokenizer/Cargo.toml tokenizer/Cargo.lock tokenizer/src/lib.rs
  VERBATIM)
add_custom_target(tokenizer_build DEPENDS "${RUST_TARGET}/release/libai_hider_tokenizer.a")
add_library(tokenizer_native STATIC IMPORTED)
set_target_properties(tokenizer_native PROPERTIES IMPORTED_LOCATION "${RUST_TARGET}/release/libai_hider_tokenizer.a")
add_dependencies(tokenizer_native tokenizer_build)
add_custom_target(dependency_notices ALL
  COMMAND "${CMAKE_COMMAND}" "-DNATIVE_CACHE=${NATIVE_CACHE}"
    "-DRUST_TOOLCHAIN=${RUST_TOOLCHAIN}"
    "-DTOKENIZER_DIR=${CMAKE_CURRENT_SOURCE_DIR}/tokenizer"
    "-DCANDLE_DIR=$<IF:$<BOOL:${APPLE}>,,${CMAKE_CURRENT_SOURCE_DIR}/candle>"
    "-DDESTINATION=${CMAKE_CURRENT_BINARY_DIR}/licenses"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/collect-licenses.cmake"
  DEPENDS tokenizer_build VERBATIM)

if(APPLE)
  add_executable(deckard src/main.cpp src/coreml_gradient.mm src/tokenizer.cpp src/support.cpp src/host.cpp src/setup.cpp src/system_check.cpp)
  target_include_directories(deckard PRIVATE src "${CMAKE_CURRENT_BINARY_DIR}/generated")
  set_source_files_properties(src/coreml_gradient.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
  target_compile_options(deckard PRIVATE -Wall -Wextra
    "-ffile-prefix-map=${AI_HIDER_ROOT}=/deckard-source"
    "-ffile-prefix-map=${NATIVE_CACHE}=/deckard-sdk")
  target_link_libraries(deckard PRIVATE tokenizer_native ai_hider_json "-lcurl" "-framework Security"
    "-framework CoreFoundation" "-framework Foundation" "-framework CoreML")
elseif(UNIX AND NOT APPLE)
  # A container engine is mandatory: the Candle CUDA backend is compiled with
  # nvcc inside a container (see the CUDA comment above); no host nvcc exists.
  find_program(CONTAINER_ENGINE docker)
  if(NOT CONTAINER_ENGINE)
    find_program(CONTAINER_ENGINE podman)
    if(NOT CONTAINER_ENGINE)
      message(FATAL_ERROR "Linux builds require Docker or Podman to run nvcc in a "
        "container for the Candle CUDA backend, and neither was found on PATH. "
        "Install either engine and reconfigure.")
    endif()
  endif()
  set(CANDLE_SRC "${CMAKE_CURRENT_SOURCE_DIR}/candle")
  # Fixed host location, matching the container's CARGO_TARGET_DIR mount
  # (/usr/src/app/native-cli/build/candle-cuda): the container writes the
  # static library here regardless of the CMake build directory.
  set(CANDLE_CUDA_DIR "${AI_HIDER_ROOT}/native-cli/build/candle-cuda")
  set(CANDLE_CUDA_A "${CANDLE_CUDA_DIR}/release/libai_hider_candle.a")
  file(MAKE_DIRECTORY "${AI_HIDER_ROOT}/cache/docker-cargo" "${AI_HIDER_ROOT}/cache/docker-rustup")
  set(CUDA_CONTAINER_SCRIPT
    "apt-get update && apt-get install -y --no-install-recommends curl build-essential pkg-config libssl-dev ca-certificates && if [ ! -x \"\$CARGO_HOME/bin/cargo\" ]; then curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path; fi && export PATH=\"\$CARGO_HOME/bin:\$PATH\" && cargo build --release --locked --features cuda --manifest-path native-cli/candle/Cargo.toml && mkdir -p native-cli/build/candle-cuda/lib && cp -a /usr/local/cuda/lib64/libcublas.so* /usr/local/cuda/lib64/libcublasLt.so* /usr/local/cuda/lib64/libcurand.so* /usr/local/cuda/lib64/libcudart.so* /usr/local/cuda/lib64/libnvrtc.so* native-cli/build/candle-cuda/lib/ && cp -a \"\$(find native-cli/build/candle-cuda/release/build -type f -name libmoe.a | head -n1)\" native-cli/build/candle-cuda/lib/")
  add_custom_command(
    OUTPUT "${CANDLE_CUDA_A}"
    COMMAND "${CONTAINER_ENGINE}" run --rm
      -e CUDA_COMPUTE_CAP=${DECKARD_CUDA_COMPUTE_CAP}
      -e CUDARC_CUDA_VERSION=${DECKARD_CUDARC_CUDA_VERSION}
      -e CARGO_TARGET_DIR=/usr/src/app/native-cli/build/candle-cuda
      -e RUSTUP_HOME=/usr/src/app/cache/docker-rustup
      -e CARGO_HOME=/usr/src/app/cache/docker-cargo
      -v "${AI_HIDER_ROOT}:/usr/src/app"
      -w /usr/src/app
      "${DECKARD_CUDA_IMAGE}"
      bash -lc "${CUDA_CONTAINER_SCRIPT}"
    DEPENDS "${CANDLE_SRC}/Cargo.toml" "${CANDLE_SRC}/Cargo.lock" "${CANDLE_SRC}/src/lib.rs"
    VERBATIM)
  add_custom_target(candle_build DEPENDS "${CANDLE_CUDA_A}")
  add_dependencies(candle_build tokenizer_build)
  add_library(candle_native STATIC IMPORTED)
  set_target_properties(candle_native PROPERTIES IMPORTED_LOCATION "${CANDLE_CUDA_A}")
  add_dependencies(candle_native candle_build)
  add_executable(deckard src/main.cpp src/gradient_backend.cpp src/onnx_backend.cpp src/onnx_install.cpp src/tokenizer.cpp src/support.cpp src/host.cpp src/setup.cpp src/system_check.cpp)
  target_include_directories(deckard PRIVATE src "${CMAKE_CURRENT_BINARY_DIR}/generated")
  target_compile_options(deckard PRIVATE -Wall -Wextra
    "-ffile-prefix-map=${AI_HIDER_ROOT}=/deckard-source"
    "-ffile-prefix-map=${NATIVE_CACHE}=/deckard-sdk")
  set(DECKARD_RPATHS)
  # ONNX Runtime: the pinned SDK ships no static archive, so libonnxruntime.so
  # is embedded verbatim into the deckard binary and dlopened from memory
  # (memfd_create) by onnx_backend.cpp. No runtime file or OS package is ever
  # required. ORT successor dependencies (libstdc++/libm/libc) are the same
  # base libraries deckard itself links.
  #
  # Intel iGPU execution uses a second embedded core: the ORT 1.24.x "openvino"
  # wheel (API v24, structurally ABI-compatible with our v1.30 consumer
  # headers) plus the OpenVINO 2025.4 runtime it was built against. Those blobs
  # are materialized to the per-user cache directory only when an Intel GPU is
  # actually usable; otherwise the memfd CPU core above remains in effect, so a
  # system without OpenVINO never touches disk.
  function(embed_blob source_file blob_name)
    set(BLOB "${CMAKE_CURRENT_BINARY_DIR}/embedded/${blob_name}.bin")
    set(BLOB_OBJECT "${CMAKE_CURRENT_BINARY_DIR}/embedded/${blob_name}.o")
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/embedded")
    add_custom_command(
      OUTPUT "${BLOB}" "${BLOB_OBJECT}"
      COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${source_file}" "${BLOB}"
      COMMAND objcopy -I binary -O elf64-x86-64 -B i386:x86-64
        --rename-section .data=.rodata,alloc,load,readonly,data,contents
        "${blob_name}.bin" "${blob_name}.o"
      WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/embedded"
      DEPENDS "${source_file}"
      VERBATIM)
    add_custom_target(${blob_name}_target DEPENDS "${BLOB_OBJECT}")
    add_dependencies(deckard ${blob_name}_target)
    target_link_libraries(deckard PRIVATE "${BLOB_OBJECT}")
  endfunction()
  set(ONNXRUNTIME_INCLUDE "${NATIVE_CACHE}/onnxruntime/include")
  if(EXISTS "${ONNXRUNTIME_INCLUDE}/onnxruntime_c_api.h")
    target_include_directories(deckard PRIVATE "${ONNXRUNTIME_INCLUDE}")
    set(ONNXRUNTIME_LIB "${NATIVE_CACHE}/onnxruntime/lib")
    if(EXISTS "${ONNXRUNTIME_LIB}/libonnxruntime.so.1.30.0")
      embed_blob("${ONNXRUNTIME_LIB}/libonnxruntime.so.1.30.0" onnxruntime_blob)
    endif()
    set(ONNXRUNTIME_OV "${NATIVE_CACHE}/onnxruntime-openvino")
    if(EXISTS "${ONNXRUNTIME_OV}/libonnxruntime.so.1.24.1")
      embed_blob("${ONNXRUNTIME_OV}/libonnxruntime.so.1.24.1" openvino_ortcore)
      embed_blob("${ONNXRUNTIME_OV}/libonnxruntime_providers_openvino.so" openvino_provider)
      embed_blob("${ONNXRUNTIME_OV}/libonnxruntime_providers_shared.so" openvino_providershared)
      embed_blob("${ONNXRUNTIME_OV}/libopenvino.so.2541" openvino_runtime)
      embed_blob("${ONNXRUNTIME_OV}/libopenvino_c.so.2541" openvino_capi)
      embed_blob("${ONNXRUNTIME_OV}/libopenvino_onnx_frontend.so.2541" openvino_frontend)
      embed_blob("${ONNXRUNTIME_OV}/libopenvino_intel_gpu_plugin.so" openvino_gpuplugin)
      embed_blob("${ONNXRUNTIME_OV}/libtbb.so.12" openvino_tbb)
      embed_blob("${ONNXRUNTIME_OV}/libtbbmalloc.so" openvino_tbbmalloc)
    endif()
  endif()
  if(EXISTS "${CANDLE_CUDA_DIR}/lib")
    target_link_directories(deckard PRIVATE "${CANDLE_CUDA_DIR}/lib")
    list(APPEND DECKARD_RPATHS "${CANDLE_CUDA_DIR}/lib")
  endif()
  if(EXISTS "${DECKARD_CUDA_HOME}/lib64")
    target_link_directories(deckard PRIVATE "${DECKARD_CUDA_HOME}/lib64")
    list(APPEND DECKARD_RPATHS "${DECKARD_CUDA_HOME}/lib64")
  endif()
  # The ONNX exporter (export.py + its pinned Dockerfile) is embedded verbatim
  # so "deckard install" can regenerate the pinned model.onnx automatically in
  # a container image, with no external file or Python package on the host.
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx/export.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx/export/Dockerfile"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/onnx_export_embedded.hpp.in")
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/onnx/export.py" ONNX_EXPORT_PY)
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/onnx/export/Dockerfile" ONNX_EXPORT_DOCKERFILE)
  configure_file(src/onnx_export_embedded.hpp.in
    "${CMAKE_CURRENT_BINARY_DIR}/generated/onnx_export_embedded.hpp" @ONLY)
  if(DECKARD_RPATHS)
    list(REMOVE_DUPLICATES DECKARD_RPATHS)
    set_target_properties(deckard PROPERTIES
      BUILD_RPATH "${DECKARD_RPATHS}"
      INSTALL_RPATH "${DECKARD_RPATHS}")
  endif()
  # libmoe.a is the candle-kernels static archive (nvcc-compiled mm kernels).
  # Its host-side stubs call the cudart API (__cudaRegisterFatBinaryEnd, ...),
  # so -lcudart is required to link. All other CUDA libraries are dlopened by
  # cudarc at runtime; -lmoe must follow the candle static archive because a
  # static archive only satisfies references that already appeared earlier in
  # the link order.
  target_link_libraries(deckard PRIVATE tokenizer_native candle_native ai_hider_json
    "${CMAKE_DL_LIBS}" "-lcurl" "-lssl" "-lcrypto" "-lcudart" "-lmoe")
else()
  message(FATAL_ERROR "Deckard supports Apple silicon and Linux only.")
endif()
install(TARGETS deckard RUNTIME DESTINATION bin)
install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/licenses/" DESTINATION share/licenses)

enable_testing()
add_executable(deckard-process-metrics tests/process-metrics.cpp)
add_test(NAME cli_help COMMAND deckard --help)
add_test(NAME native_unit COMMAND deckard self-test)
