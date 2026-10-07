#pragma once

#include "gradient_backend.hpp"
#include "support.hpp"
#include <cstdint>
#include <memory>
#include <vector>

namespace aihider {

// ONNX Runtime backend for the Linux Gradient decoder. libonnxruntime.so is
// embedded in this binary and dlopened from memory (memfd) at construction,
// so no ONNX Runtime file or system package is ever required, mirroring how
// the Candle CUDA backend loads its GPU runtimes lazily. A missing or broken
// blob surfaces as a clear error so a plain Candle install never depends on it.
//
// When a usable Intel GPU is present, a second embedded core (the ORT 1.24
// "openvino" wheel + OpenVINO runtime) is materialized into the per-user cache
// directory and run with the OpenVINO GPU execution provider; the CPU core is
// used otherwise. The system OpenCL stack is probed at runtime for an actual
// Intel device — never assumed from the loader being installed — and is the
// final authority. DECKARD_ONNX_OV redirects where the assembly happens.
class OnnxBackend : public ModelBackend {
 public:
  OnnxBackend(const fs::path& model_directory, const fs::path& cache_directory);
  ~OnnxBackend() override;
  OnnxBackend(const OnnxBackend&) = delete;
  OnnxBackend& operator=(const OnnxBackend&) = delete;

  double logit(const std::vector<std::uint32_t>& ids,
               const std::vector<std::uint32_t>& mask) override;
  // "gpu" only when the OpenVINO GPU provider actually started a session;
  // otherwise "cpu" (the memfd CPU core). The probe is the final authority.
  const char* device() const override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace aihider