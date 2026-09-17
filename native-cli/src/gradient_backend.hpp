#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

namespace aihider {

// Backend contract shared by the Core ML (macOS) and Candle (Linux) runtimes.
// Model and cache directories are parsed at construction; inference is batched.
class ModelBackend {
 public:
  virtual ~ModelBackend() = default;
  virtual double logit(const std::vector<std::uint32_t>& ids,
                       const std::vector<std::uint32_t>& mask) = 0;
};

// Selects the platform runtime: CoreMLGradient on Apple, CandleGradient elsewhere.
std::unique_ptr<ModelBackend> create_model_backend(
    const std::filesystem::path& model_directory,
    const std::filesystem::path& cache_directory);

}  // namespace aihider