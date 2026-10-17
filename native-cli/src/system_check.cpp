#include "system_check.hpp"

#include "intel_gpu.hpp"
#include "onnx_providers.hpp"
#include "onnx_runtime.hpp"

#include <sys/utsname.h>

#include <fstream>

namespace aihider {
namespace {

std::string distribution() {
  std::ifstream release("/etc/os-release");
  for (std::string line; std::getline(release, line);) {
    if (line.rfind("ID=", 0) != 0) continue;
    std::string id = line.substr(3);
    std::erase(id, '"');
    return id;
  }
  return "unknown";
}

Json model_report(const fs::path& cache) {
  bool candle = true;
  for (const auto& item : model_assets().at("files").items()) candle = candle && fs::is_regular_file(cache / item.key());
  std::string onnx = "missing";
  if (fs::is_regular_file(cache / "model.onnx")) {
    try {
      require_model_file(cache, "model.onnx");
      onnx = "ready";
    } catch (const Error&) {
      onnx = "drifted";
    }
  }
  return {{"cache", cache.string()}, {"candle", candle ? "ready" : "missing"}, {"onnx", onnx}};
}

Json accelerator_report(const fs::path& cache, Json& problems, Json& help) {
  Json accelerators = Json::array();
  const OrtApi* api = nullptr;
  try {
    api = &onnx_runtime(cache);
  } catch (const Error& error) {
    problems.push_back("ONNX Runtime cannot be loaded, so model.onnx and accelerators are unavailable");
    help.push_back(error.what());
  }
  for (const auto& provider : execution_providers()) {
    accelerators.push_back({{"name", provider.name}, {"device", provider.device},
                            {"usable", api != nullptr && provider.usable(*api)}});
  }
  return accelerators;
}

// The Intel GPU chain. The kernel driver, the OpenCL loader and Intel's
// OpenCL driver are independent, so every missing one is reported; whether
// that driver exposes the GPU, and whether OpenVINO then starts on it, can
// only be judged once everything before it is present.
void intel_gpu_problems(const IntelGpu& gpu, bool openvino_starts, Json& problems, Json& help) {
  if (!gpu.present) return;
  if (gpu.kernel_driver.empty()) {
    problems.push_back("Intel GPU present but no kernel driver (i915 or xe) is bound to it");
    help.push_back("Enable the i915 (or, for Xe2 and newer, xe) kernel driver.");
  }
  if (!gpu.opencl_loader) {
    problems.push_back("Intel GPU present but the OpenCL loader (libOpenCL.so.1) is missing");
    help.push_back("Install the OpenCL ICD loader of your distribution (ocl-icd).");
  }
  if (!gpu.opencl_driver) {
    problems.push_back("Intel GPU present but no Intel OpenCL driver is installed");
    help.push_back("Install intel-compute-runtime for Gen12 (Xe) and newer, or its legacy 24.35 branch "
                   "(intel-compute-runtime-legacy / intel-opencl-icd-legacy1) for Gen8 to Gen11 such as UHD 630.");
  }
  const bool stack = !gpu.kernel_driver.empty() && gpu.opencl_loader && gpu.opencl_driver;
  if (stack && !gpu.usable) {
    problems.push_back("Intel GPU present but the installed Intel OpenCL driver does not expose it");
    help.push_back("The driver probably does not support this GPU generation: current intel-compute-runtime "
                   "supports Gen12 and newer only; Gen8 to Gen11 need the legacy 24.35 branch.");
  }
  if (gpu.usable && !openvino_starts) {
    problems.push_back("The Intel GPU is usable through OpenCL, but OpenVINO fails to start on it");
    help.push_back("OpenVINO 2025.4 rejected or crashed on this GPU; with --gpu, Deckard uses the CPU instead.");
  }
}

bool usable(const Json& accelerators, const std::string& name) {
  for (const auto& accelerator : accelerators) {
    if (accelerator["name"] == name) return accelerator["usable"];
  }
  return false;
}

}  // namespace

Json system_check() {
  Json report = {{"command", "check-system"}, {"platform", "linux"}, {"os", distribution()},
                 {"runtimes", Json::array({"onnx", "candle"})}};
  struct utsname host {};
  if (uname(&host) == 0) report["arch"] = host.machine;
  Json problems = Json::array(), help = Json::array();

  const fs::path cache = model_cache();
  const Json models = model_report(cache);
  report["models"] = models;
  if (models["candle"] == "missing") {
    problems.push_back("The pinned model files are not in " + cache.string());
    help.push_back("Run deckard install to download them.");
  }
  if (models["onnx"] == "drifted") {
    problems.push_back("model.onnx in the model cache does not match its pin");
    help.push_back("Run scripts/export-onnx.sh to export it again.");
  }

  const IntelGpu& gpu = intel_gpu();
  report["intel_gpu"] = {{"present", gpu.present},
                         {"kernel_driver", gpu.kernel_driver.empty() ? Json() : Json(gpu.kernel_driver)},
                         {"opencl_loader", gpu.opencl_loader}, {"opencl_driver", gpu.opencl_driver},
                         {"devices", gpu.devices}, {"usable", gpu.usable}};
  report["accelerators"] = accelerator_report(cache, problems, help);
  intel_gpu_problems(gpu, usable(report["accelerators"], "openvino-gpu"), problems, help);

  report["runtime"] = models["onnx"] == "ready" ? "onnx" : "candle";
  report["problems"] = problems;
  report["help"] = help;
  return report;
}

}  // namespace aihider
