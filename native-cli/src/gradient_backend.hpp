#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

namespace aihider {

// Backend contract shared by the Core ML (macOS), Candle (Linux), and
// ONNX Runtime (Linux optional) runtimes. Model and cache directories are
// parsed at construction; inference is batched through a single logit call.
class ModelBackend {
 public:
  virtual ~ModelBackend() = default;
  virtual double logit(const std::vector<std::uint32_t>& ids,
                       const std::vector<std::uint32_t>& mask) = 0;
  // Which device actually serves inference after the runtime probe decided.
  // "cpu" is the safe default; backends that engage an accelerator override.
  virtual const char* device() const { return "cpu"; }
};

// Which native runtime to instantiate on Linux: Candle or ONNX. Auto resolves
// at model-open time to Onnx when model.onnx is present, otherwise Candle, so
// the GPU-accelerated ONNX Runtime path engages without a flag.
enum class Backend { Auto, Candle, Onnx };

// Resolves Auto to the backend that matches the model directory.
Backend resolve_backend(const std::filesystem::path& model_directory, Backend backend);

// Selects the platform runtime: CoreMLGradient on Apple, and on Linux either
// CandleGradient or (when Backend::Onnx is resolved) the ONNX Runtime model.
std::unique_ptr<ModelBackend> create_model_backend(
    const std::filesystem::path& model_directory,
    const std::filesystem::path& cache_directory, Backend backend = Backend::Auto);

}  // namespace aihider