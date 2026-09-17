#include "gradient_backend.hpp"
#include "support.hpp"

#ifdef __APPLE__
#include "coreml_gradient.hpp"
namespace aihider {
namespace {
class CoreMLBackend : public ModelBackend {
 public:
  CoreMLBackend(const fs::path& model_directory, const fs::path& cache_directory)
      : model_(model_directory, cache_directory) {}

  double logit(const std::vector<std::uint32_t>& ids,
               const std::vector<std::uint32_t>& mask) override {
    return model_.logit(ids, mask);
  }

 private:
  CoreMLGradient model_;
};
}  // namespace
}  // namespace aihider
#elif defined(__linux__)
#include "candle_gradient.hpp"
namespace aihider {
namespace {
class CandleBackend : public ModelBackend {
 public:
  CandleBackend(const fs::path& model_directory, const fs::path& cache_directory)
      : model_(model_directory, cache_directory) {}

  double logit(const std::vector<std::uint32_t>& ids,
               const std::vector<std::uint32_t>& mask) override {
    return model_.logit(ids, mask);
  }

 private:
  CandleGradient model_;
};
}  // namespace
}  // namespace aihider
#else
namespace aihider {
namespace {
class UnsupportedBackend : public ModelBackend {
 public:
  double logit(const std::vector<std::uint32_t>&,
               const std::vector<std::uint32_t>&) override {
    throw std::runtime_error("No model backend available on this platform.");
  }
};
}  // namespace
}  // namespace aihider
#endif

namespace aihider {
std::unique_ptr<ModelBackend> create_model_backend(
    const fs::path& model_directory, const fs::path& cache_directory) {
#if defined(__APPLE__)
  return std::make_unique<CoreMLBackend>(model_directory, cache_directory);
#elif defined(__linux__)
  return std::make_unique<CandleBackend>(model_directory, cache_directory);
#else
  return std::make_unique<UnsupportedBackend>(model_directory, cache_directory);
#endif
}
}  // namespace aihider