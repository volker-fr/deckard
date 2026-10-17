#pragma once

#include "support.hpp"
#include <onnxruntime_c_api.h>

namespace aihider {

// The ONNX Runtime libraries embedded in deckard (linux.cmake): the core and
// the OpenVINO execution provider with the OpenVINO parts it needs. The first
// call writes each to `<cache_directory>/onnxruntime-<version>-openvino-<version>/`
// unless a copy with its compiled-in SHA-256 is there, dlopens the core and
// returns its C API; later calls return the same API. Throws
// Error("onnx_unavailable") when the runtime cannot be loaded.
const OrtApi& onnx_runtime(const fs::path& cache_directory);

// Throws Error(code, "<context>: <ONNX Runtime message>") for a failed call.
void onnx_check(const OrtApi& api, OrtStatus* status, const char* code, const std::string& context);

}  // namespace aihider
