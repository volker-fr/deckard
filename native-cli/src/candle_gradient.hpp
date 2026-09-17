#pragma once

#include "gradient_backend.hpp"
#include "support.hpp"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace aihider {
namespace {
constexpr std::uint32_t max_len = 512;
using i32_t = std::int32_t;
constexpr i32_t code_ok = 0, code_invalid = 1, code_assets = 2, code_load = 3,
    code_forward = 4, code_nonfinite = 5;
class CandleLoader {
 public:
  explicit CandleLoader(const fs::path& model_directory)
      : config_path_((model_directory / "config.json").string()),
        weights_path_((model_directory / "model.safetensors").string()) {
    if (!fs::is_regular_file(model_directory / "config.json") ||
        !fs::is_regular_file(model_directory / "model.safetensors"))
      throw Error("missing_assets", "The Candle gradient model assets are incomplete.");
  }
  const std::string& config_path() const { return config_path_; }
  const std::string& weights_path() const { return weights_path_; }
 private:
  std::string config_path_, weights_path_;
};
}  // namespace

// Shared with the Rust ai_hider_candle staticlib (native-cli/candle/src/lib.rs).
// The attention mask crosses the boundary as f32 in lockstep with the Core ML
// runtime; the host only ever sends 0/1 masks.
extern "C" {
i32_t aih_candle_open(const char* config_path, size_t config_len, const char* weights_path,
                      size_t weights_len, void** out);
i32_t aih_candle_logit(void* handle, const std::uint32_t* ids, size_t ids_len,
                       const float* mask, size_t mask_len, double* out);
i32_t aih_candle_close(void* handle);
}

class CandleGradient {
 public:
  CandleGradient(const fs::path& model_directory, const fs::path& cache_directory)
      : loader_(model_directory) {
    (void)cache_directory;
    const auto& config = loader_.config_path();
    const auto& weights = loader_.weights_path();
    i32_t status = aih_candle_open(config.data(), config.size(), weights.data(), weights.size(),
                                   &handle_);
    if (status != code_ok) throw Error(error_for(status), "The Candle gradient model failed to load.");
  }
  ~CandleGradient() { if (handle_) (void)aih_candle_close(handle_); }
  CandleGradient(const CandleGradient&) = delete;
  CandleGradient& operator=(const CandleGradient&) = delete;
  double logit(const std::vector<std::uint32_t>& ids,
               const std::vector<std::uint32_t>& mask) {
    if (ids.size() > max_len || ids.size() != mask.size())
      throw Error("invalid_input", "Gradient input exceeds the 512-token limit.");
    if (ids.size() < 3 || ids.front() != 1 || ids.back() != 2)
      throw Error("invalid_input", "Gradient input must be wrapped as CLS..SEP.");
    std::vector<float> float_mask(mask.begin(), mask.end());
    double value = 0;
    i32_t status = aih_candle_logit(handle_, ids.data(), ids.size(), float_mask.data(),
                                    float_mask.size(), &value);
    if (status != code_ok) throw Error(error_for(status), "The Candle gradient inference failed.");
    return value;
  }

 private:
  static std::string error_for(i32_t status) {
    switch (status) {
      case code_invalid: return "invalid_input";
      case code_assets: return "missing_assets";
      case code_load: return "load_failed";
      case code_forward: return "inference_failed";
      case code_nonfinite: return "invalid_output";
      default: return "inference_failed";
    }
  }
  CandleLoader loader_;
  void* handle_ = nullptr;
};
}  // namespace aihider