#pragma once

#include "support.hpp"
#include <onnxruntime_c_api.h>
#include <vector>

namespace aihider {

// An ONNX Runtime execution provider (accelerator) that OnnxGradient tries
// before ONNX Runtime's built-in CPU provider. Supporting another accelerator
// means adding one entry to execution_providers(); see docs/LINUX.md.
struct ExecutionProvider {
  const char* name;    // for logs and check-system, e.g. "openvino-gpu"
  const char* device;  // reported as ModelBackend::device(), e.g. "gpu"
  // Whether this host can use the provider. Must never crash the caller: a
  // probe that could (driver initialization) runs in a child process, before
  // the caller creates any ONNX Runtime environment or thread.
  bool (*usable)(const OrtApi& api);
  // Configures `options` for the provider. Throws Error when it cannot.
  void (*append)(const OrtApi& api, OrtSessionOptions* options);
};

// In priority order; empty when only the CPU provider is available.
const std::vector<ExecutionProvider>& execution_providers();

// Whether any execution provider can run on this host, loading ONNX Runtime
// from `cache_directory`; false when ONNX Runtime itself is unavailable.
bool accelerator_available(const fs::path& cache_directory);

}  // namespace aihider
