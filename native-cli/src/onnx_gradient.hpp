#pragma once

#include "support.hpp"
#include <cstdint>
#include <memory>
#include <vector>

namespace aihider {

// The pinned model.onnx (an fp32 export of the Hugging Face checkpoint, see
// docs/LINUX.md) on ONNX Runtime, batch one, on ONNX Runtime's built-in CPU
// provider. With `use_accelerators`, the first usable execution provider from
// execution_providers() runs inference instead.
class OnnxGradient {
 public:
  OnnxGradient(const fs::path& model_directory, const fs::path& cache_directory, bool use_accelerators = false);
  ~OnnxGradient();
  OnnxGradient(const OnnxGradient&) = delete;
  OnnxGradient& operator=(const OnnxGradient&) = delete;

  double logit(const std::vector<std::uint32_t>& ids, const std::vector<std::uint32_t>& mask);
  // "cpu", or the device of the execution provider that opened the session.
  const char* device() const;
  const char* runtime() const { return "native-onnx"; }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace aihider
