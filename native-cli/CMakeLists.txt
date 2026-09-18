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
  add_executable(deckard src/main.cpp src/coreml_gradient.mm src/tokenizer.cpp src/support.cpp src/host.cpp src/setup.cpp)
  target_include_directories(deckard PRIVATE src "${CMAKE_CURRENT_BINARY_DIR}/generated")
  set_source_files_properties(src/coreml_gradient.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
  target_compile_options(deckard PRIVATE -Wall -Wextra
    "-ffile-prefix-map=${AI_HIDER_ROOT}=/deckard-source"
    "-ffile-prefix-map=${NATIVE_CACHE}=/deckard-sdk")
  target_link_libraries(deckard PRIVATE tokenizer_native ai_hider_json "-lcurl" "-framework Security"
    "-framework CoreFoundation" "-framework Foundation" "-framework CoreML")
elseif(UNIX AND NOT APPLE)
  set(CANDLE_SRC "${CMAKE_CURRENT_SOURCE_DIR}/candle")
  set(CANDLE_TARGET "${CMAKE_CURRENT_BINARY_DIR}/candle-rust")
  set(candle_env "CARGO_HOME=${BUILD_CARGO_HOME}" "RUSTUP_HOME=${NATIVE_CACHE}/rustup")
  list(APPEND candle_env "RUSTC=${RUSTC}" "CARGO_ENCODED_RUSTFLAGS=${RUST_PATH_FLAGS}")
  if(RUST_TOOLCHAIN)
    list(APPEND candle_env "RUSTUP_TOOLCHAIN=1.90.0")
  endif()
  list(APPEND candle_env "CARGO_TARGET_DIR=${CANDLE_TARGET}" "CARGO_BUILD_JOBS=2")
  add_custom_command(
    OUTPUT "${CANDLE_TARGET}/release/libai_hider_candle.a"
    COMMAND "${CMAKE_COMMAND}" -E env
      ${candle_env}
      "${CARGO}" build --release --locked --offline --manifest-path "${CANDLE_SRC}/Cargo.toml"
    DEPENDS "${CANDLE_SRC}/Cargo.toml" "${CANDLE_SRC}/Cargo.lock" "${CANDLE_SRC}/src/lib.rs"
    VERBATIM)
  add_custom_target(candle_build DEPENDS "${CANDLE_TARGET}/release/libai_hider_candle.a")
  add_dependencies(candle_build tokenizer_build)
  add_library(candle_native STATIC IMPORTED)
  set_target_properties(candle_native PROPERTIES IMPORTED_LOCATION "${CANDLE_TARGET}/release/libai_hider_candle.a")
  add_dependencies(candle_native candle_build)
  add_executable(deckard src/main.cpp src/gradient_backend.cpp src/tokenizer.cpp src/support.cpp src/host.cpp src/setup.cpp)
  target_include_directories(deckard PRIVATE src "${CMAKE_CURRENT_BINARY_DIR}/generated")
  target_compile_options(deckard PRIVATE -Wall -Wextra
    "-ffile-prefix-map=${AI_HIDER_ROOT}=/deckard-source"
    "-ffile-prefix-map=${NATIVE_CACHE}=/deckard-sdk")
  target_link_libraries(deckard PRIVATE tokenizer_native candle_native ai_hider_json
    "${CMAKE_DL_LIBS}" "-lcurl" "-lssl" "-lcrypto")
else()
  message(FATAL_ERROR "Deckard supports Apple silicon and Linux only.")
endif()
install(TARGETS deckard RUNTIME DESTINATION bin)
install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/licenses/" DESTINATION share/licenses)

enable_testing()
add_executable(deckard-process-metrics tests/process-metrics.cpp)
add_test(NAME cli_help COMMAND deckard --help)
add_test(NAME native_unit COMMAND deckard self-test)
