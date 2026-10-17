#include "onnx_providers.hpp"

#include "intel_gpu.hpp"
#include "onnx_runtime.hpp"

#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <mutex>

namespace aihider {
namespace {

// --- OpenVINO on Intel GPUs -------------------------------------------------
// ONNX Runtime's OpenVINO execution provider, compiling model.onnx for the
// Intel GPU through OpenVINO's GPU plugin (OpenCL). Why OpenVINO and which
// drivers it needs: docs/LINUX.md ("Intel GPUs").

void append_openvino_gpu(const OrtApi& api, OrtSessionOptions* options) {
  const char* keys[] = {"device_type", "precision"};
  const char* values[] = {"GPU", "FP32"};
  onnx_check(api, api.SessionOptionsAppendExecutionProvider_OpenVINO_V2(options, keys, values, 2),
             "onnx_unavailable", "Cannot attach the OpenVINO GPU provider");
}

// Attaches the provider in a child process: that loads OpenVINO and its GPU
// plugin and lets them look for the device, which can crash on a driver they
// cannot handle. Only a clean exit counts.
bool attach_in_child(const OrtApi& api) {
  const pid_t child = fork();
  if (child == 0) {
    try {
      OrtEnv* env = nullptr;
      OrtSessionOptions* options = nullptr;
      onnx_check(api, api.CreateEnv(ORT_LOGGING_LEVEL_FATAL, "deckard-probe", &env), "probe", "env");
      onnx_check(api, api.CreateSessionOptions(&options), "probe", "options");
      append_openvino_gpu(api, options);
      _exit(0);
    } catch (...) {
      _exit(1);
    }
  }
  int status = 0;
  while (child > 0 && waitpid(child, &status, 0) < 0 && errno == EINTR) {
  }
  return child > 0 && WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

// attach_in_child(), once per process.
bool openvino_gpu_starts(const OrtApi& api) {
  static std::once_flag once;
  static bool starts = false;
  std::call_once(once, [&api] { starts = attach_in_child(api); });
  return starts;
}

bool openvino_gpu_usable(const OrtApi& api) {
  return intel_gpu().usable && openvino_gpu_starts(api);
}

}  // namespace

const std::vector<ExecutionProvider>& execution_providers() {
  static const std::vector<ExecutionProvider> providers = {
      {"openvino-gpu", "gpu", openvino_gpu_usable, append_openvino_gpu},
  };
  return providers;
}

bool accelerator_available(const fs::path& cache_directory) {
  try {
    const OrtApi& api = onnx_runtime(cache_directory);
    for (const auto& provider : execution_providers()) {
      if (provider.usable(api)) return true;
    }
  } catch (const Error&) {
  }
  return false;
}

}  // namespace aihider
