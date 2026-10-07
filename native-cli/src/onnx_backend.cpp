#include "onnx_backend.hpp"

#include "system_check.hpp"

#include <dlfcn.h>
#include <onnxruntime_c_api.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>

namespace aihider {
namespace {
constexpr std::uint32_t max_len = 512;

// OrtApiBase/GetVersionString come from the exported C API; the rest of the
// API is reached through the versioned OrtApi struct, so no ONNX Runtime
// symbol other than OrtGetApiBase is ever resolved at load time.
using api_base_fn = const OrtApiBase* (*)();
struct ApiError : std::runtime_error {
  explicit ApiError(const std::string& message) : std::runtime_error(message) {}
};

// The ONNX Runtime shared library is embedded verbatim into this binary via
// CMake's objcopy blob (embedded/onnxruntime_blob.bin), so the deckard
// executable has no external ONNX dependency at all. It is loaded through an
// anonymous memfd so the loader can read it as a file without ever touching
// disk. Successor dependencies (libstdc++/libm/libc) are shared with deckard.
extern "C" {
extern const unsigned char _binary_onnxruntime_blob_bin_start[];
extern const unsigned char _binary_onnxruntime_blob_bin_end[];
}

// Intel iGPU execution optionally replaces the memfd CPU core with the second
// embedded core: ORT 1.24.x "openvino" (API v24, struct-ABI identical to our
// v1.30 headers) plus the OpenVINO 2025.4 runtime it was built against. These
// blobs are materialized to the per-user cache directory when an Intel GPU is
// usable, which the OpenVINO GPU plugin itself requires (it links libOpenCL).
extern "C" {
extern const unsigned char _binary_openvino_ortcore_bin_start[];
extern const unsigned char _binary_openvino_ortcore_bin_end[];
extern const unsigned char _binary_openvino_provider_bin_start[];
extern const unsigned char _binary_openvino_provider_bin_end[];
extern const unsigned char _binary_openvino_providershared_bin_start[];
extern const unsigned char _binary_openvino_providershared_bin_end[];
extern const unsigned char _binary_openvino_runtime_bin_start[];
extern const unsigned char _binary_openvino_runtime_bin_end[];
extern const unsigned char _binary_openvino_capi_bin_start[];
extern const unsigned char _binary_openvino_capi_bin_end[];
extern const unsigned char _binary_openvino_frontend_bin_start[];
extern const unsigned char _binary_openvino_frontend_bin_end[];
extern const unsigned char _binary_openvino_gpuplugin_bin_start[];
extern const unsigned char _binary_openvino_gpuplugin_bin_end[];
extern const unsigned char _binary_openvino_tbb_bin_start[];
extern const unsigned char _binary_openvino_tbb_bin_end[];
extern const unsigned char _binary_openvino_tbbmalloc_bin_start[];
extern const unsigned char _binary_openvino_tbbmalloc_bin_end[];
}

struct OpenVINOBlob {
  const char* name;
  const unsigned char* begin;
  const unsigned char* end;
};

constexpr OpenVINOBlob openvino_blobs[] = {
    {"libonnxruntime.so.1.24.1", _binary_openvino_ortcore_bin_start, _binary_openvino_ortcore_bin_end},
    {"libonnxruntime_providers_openvino.so", _binary_openvino_provider_bin_start, _binary_openvino_provider_bin_end},
    {"libonnxruntime_providers_shared.so", _binary_openvino_providershared_bin_start, _binary_openvino_providershared_bin_end},
    {"libopenvino.so.2541", _binary_openvino_runtime_bin_start, _binary_openvino_runtime_bin_end},
    {"libopenvino_c.so.2541", _binary_openvino_capi_bin_start, _binary_openvino_capi_bin_end},
    {"libopenvino_onnx_frontend.so.2541", _binary_openvino_frontend_bin_start, _binary_openvino_frontend_bin_end},
    {"libopenvino_intel_gpu_plugin.so", _binary_openvino_gpuplugin_bin_start, _binary_openvino_gpuplugin_bin_end},
    {"libtbb.so.12", _binary_openvino_tbb_bin_start, _binary_openvino_tbb_bin_end},
    {"libtbbmalloc.so", _binary_openvino_tbbmalloc_bin_start, _binary_openvino_tbbmalloc_bin_end},
};

// A usable Intel GPU probe, from system_check. It uses the system OpenCL
// loader only (dlopen): the OpenVINO GPU plugin links libOpenCL.so.1, so a
// platform that cannot enumerate an Intel GPU device would also fail to start
// the plugin. On hosts without such a device (or without libOpenCL at all)
// the CPU core is used. This is the same probe `deckard check-system` runs.

// The OpenVINO EP can only be embedded in the ORT 1.24.x core, whose c_api
// header declares ORT_API_VERSION 24. Our consumer headers expose version 30;
// because OrtApi's struct is strictly append-only, the offsets of every field
// that existed in v24 (including SessionOptionsAppendExecutionProvider_OpenVINO)
// are identical in the v30 layout we compile against.
void write_blob(const fs::path& directory, const OpenVINOBlob& blob) {
  std::ofstream stream(directory / blob.name,
                       std::ios::binary | std::ios::trunc);
  const auto size = blob.end - blob.begin;
  stream.write(reinterpret_cast<const char*>(blob.begin), size);
  if (!stream) throw ApiError(std::string("failed to materialize ") + blob.name);
}
}  // namespace

struct OnnxBackend::Impl {
  void* library = nullptr;
  const OrtApi* api = nullptr;
  OrtEnv* env = nullptr;
  OrtSessionOptions* options = nullptr;
  OrtMemoryInfo* memory = nullptr;
  OrtSession* session = nullptr;
  std::string model_path;
  std::mutex run_mutex;
  const char* device = "cpu";

  Impl() = default;
  ~Impl();
  void check(OrtStatusPtr status) const {
    if (status == nullptr) return;
    std::string message = api->GetErrorMessage(status);
    api->ReleaseStatus(status);
    throw ApiError(message);
  }
  void require(bool condition, const char* code, const std::string& message) const {
    if (!condition) throw Error(code, message);
  }
  void load_model_from_array() {
    std::ifstream stream(model_path, std::ios::binary | std::ios::ate);
    require(bool(stream), "load_failed", "The ONNX gradient model could not be read.");
    const auto size = stream.tellg();
    require(size > 0 && static_cast<uintmax_t>(size) <= size_t(-1), "load_failed",
            "The ONNX gradient model is empty or too large.");
    std::vector<char> bytes(static_cast<size_t>(size));
    stream.seekg(0);
    stream.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    check(api->CreateSessionFromArray(env, bytes.data(), bytes.size(), options, &session));
  }
  bool try_openvino(const fs::path& ov_dir);
  void open(const fs::path& model_directory, const fs::path& cache_directory);
  double run(const std::vector<std::uint32_t>& ids, const std::vector<std::uint32_t>& mask);
};

OnnxBackend::Impl::~Impl() {
  if (session) api->ReleaseSession(session);
  if (memory) api->ReleaseMemoryInfo(memory);
  if (options) api->ReleaseSessionOptions(options);
  if (env) api->ReleaseEnv(env);
  if (library) dlclose(library);
}

// Materialize the embedded OpenVINO core + runtime, load it in dependency
// order, and build a GPU session. Returns false (after fully releasing every
// handle) when the platform cannot use OpenVINO, so open() falls back to the
// memfd CPU core. Callers invoke this only after the capability probe passed.
bool OnnxBackend::Impl::try_openvino(const fs::path& ov_dir) {
  struct ReleaseOwned {
    std::vector<void*> handles = std::vector<void*>(openvino_load_order().size());
    ~ReleaseOwned() {
      for (void* handle : handles)
        if (handle) dlclose(handle);
    }
  } owned;
  try {
    fs::create_directories(ov_dir);
    std::error_code error;
    for (const OpenVINOBlob& blob : openvino_blobs) {
      if (fs::is_regular_file(ov_dir / blob.name, error)) continue;
      write_blob(ov_dir, blob);
    }

    // Prove the runtime loads in a forked child first and fall back to the CPU
    // core when it dies, since a crash in a library initializer cannot be
    // caught here.
    if (!openvino_runtime_loads(ov_dir.string())) return false;

    // Dependency order matters for the OpenVINO runtime: NEEDED resolution
    // walks the already-loaded set first. All blobs share the directory, so
    // RPATH $ORIGIN resolves the siblings; loading the core last lets ORT
    // discover the sibling providers relative to its own real location. The
    // OpenVINO provider itself is loaded by the core (see openvino_load_order).
    const auto& load_order = openvino_load_order();
    for (size_t i = 0; i < load_order.size(); ++i) {
      void* handle = dlopen((ov_dir / load_order[i]).c_str(), RTLD_NOW | RTLD_GLOBAL);
      require(handle != nullptr, "onnx_runtime_missing",
              std::string("The embedded OpenVINO runtime component failed to load: ") + load_order[i]);
      owned.handles[i] = handle;
    }
    library = owned.handles.back();
    auto* api_base = reinterpret_cast<api_base_fn>(dlsym(library, "OrtGetApiBase"))();
    require(api_base != nullptr, "onnx_setup_failed", "The OpenVINO Runtime library has no C API entry point.");
    // The 1.24 openvino wheel is API v24. Our v30 headers share the same
    // struct layout for every field that existed in v24 (append-only), so the
    // v24 pointer is directly usable through the v30 struct definition.
    api = api_base->GetApi(24);
    require(api != nullptr, "onnx_setup_failed", "The OpenVINO Runtime library does not support API v24.");

    check(api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "deckard", &env));
    check(api->CreateSessionOptions(&options));
    check(api->SetSessionGraphOptimizationLevel(options, ORT_ENABLE_ALL));
    check(api->SetIntraOpNumThreads(options, 1));
    // cache_dir (OpenVINO's compiled-model cache) is deliberately left unset.
    // Measured on a UHD 630 (Gen9.5, intel-compute-runtime-legacy 24.35) with
    // the 1.8 GB fp32 model.onnx (If-folded, before export.py added inferred
    // shapes): session load 14.5 s uncached vs 13.3 s with a warm cache, but the
    // cache blob embeds the weights and occupies 1.8 GB. Load time is dominated
    // by reading and converting the model, not by kernel compilation, so the
    // cache doubles disk use for ~1 s. The pinned export loads in ~8 s.
    OrtOpenVINOProviderOptions ov_options{};
    ov_options.device_type = "GPU_FP32";
    check(api->SessionOptionsAppendExecutionProvider_OpenVINO(options, &ov_options));
    check(api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory));
    load_model_from_array();
    // Transfer ownership: the OpenVINO core stays loaded for this instance.
    std::fill(owned.handles.begin(), owned.handles.end(), nullptr);
    return true;
  } catch (...) {
    // OpenVINO is unusable here (libOpenCL missing, the GPU plugin cannot
    // start, or the provider refused GPU_FP32). Release everything the probe
    // built so the unloaded SONAME libonnxruntime.so.1 does not alias the
    // memfd CPU core, then return false for the CPU fallback. Any genuine
    // model-load failure is re-attempted by the CPU path, which surfaces it.
    if (session) api->ReleaseSession(session);
    if (memory) api->ReleaseMemoryInfo(memory);
    if (options) api->ReleaseSessionOptions(options);
    if (env) api->ReleaseEnv(env);
    session = nullptr;
    memory = nullptr;
    options = nullptr;
    env = nullptr;
    api = nullptr;
    library = nullptr;
    return false;
  }
}

void OnnxBackend::Impl::open(const fs::path& model_directory, const fs::path& cache_directory) {
  model_path = (model_directory / "model.onnx").string();
  require(fs::is_regular_file(model_directory / "model.onnx"), "missing_assets",
          "The ONNX gradient model asset model.onnx is incomplete.");

  // DECKARD_ONNX_OV, when set (path to a directory holding a pre-staged
  // OpenVINO wheel core and runtime), redirects where the OpenVINO blobs are
  // assembled. It never overrides the capability probe: OpenVINO must not be
  // dlopened on a host without a usable Intel GPU, because the provider's
  // static initializer enumerates devices during library load and cannot
  // recover (it crashes) when none exist. The runtime probe is the final
  // authority; forcing is useful for development only on a working GPU host.
  // A device that exists but cannot be driven is caught separately by
  // openvino_runtime_loads(), which runs that same load in a forked child.
  std::string ov_dir_force;
  if (const char* env = std::getenv("DECKARD_ONNX_OV")) ov_dir_force = env;
  if (usable_intel_gpu()) {    fs::path ov_dir = ov_dir_force.empty()
                          ? fs::path(cache_directory) / "onnxruntime-openvino"
                          : fs::path(ov_dir_force);
    if (try_openvino(ov_dir)) {
      device = "gpu";
      return;
    }
  }

  // Load the embedded ONNX Runtime blob from memory. An anonymous memfd
  // (glibc >= 2.27, Linux >= 3.17) lets dlopen read the shared object as if
  // from a file while the payload lives only in RAM — no ONNX Runtime file or
  // package is expected anywhere on the system. DECKARD_ONNXRUNTIME overrides
  // this for development with another library path or file.
  const size_t embedded_size = static_cast<size_t>(_binary_onnxruntime_blob_bin_end - _binary_onnxruntime_blob_bin_start);
  std::string candidate;
  if (const char* env = std::getenv("DECKARD_ONNXRUNTIME")) candidate = env;
  if (!candidate.empty()) {
    fs::path path(candidate);
    if (fs::is_directory(path)) path /= "libonnxruntime.so";
    library = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  } else {
    int fd = static_cast<int>(syscall(SYS_memfd_create, "onnxruntime", MFD_CLOEXEC));
    if (fd >= 0) {
      for (size_t offset = 0; offset < embedded_size;) {
        ssize_t written = write(fd, _binary_onnxruntime_blob_bin_start + offset, embedded_size - offset);
        if (written < 0) break;
        offset += static_cast<size_t>(written);
      }
      if (lseek(fd, 0, SEEK_SET) == 0) {
        char path[64];
        std::snprintf(path, sizeof(path), "/proc/self/fd/%d", fd);
        library = dlopen(path, RTLD_NOW | RTLD_LOCAL);
      }
      close(fd);
    }
  }
  require(library != nullptr, "onnx_runtime_missing",
          "The ONNX Runtime library failed to load from the embedded deckard blob.");

  auto* api_base = reinterpret_cast<api_base_fn>(dlsym(library, "OrtGetApiBase"))();
  require(api_base != nullptr, "onnx_setup_failed", "The ONNX Runtime library has no C API entry point.");
  api = api_base->GetApi(ORT_API_VERSION);
  require(api != nullptr, "onnx_setup_failed", "The ONNX Runtime library does not support this API version.");

  try {
    check(api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "deckard", &env));
    check(api->CreateSessionOptions(&options));
    check(api->SetSessionGraphOptimizationLevel(options, ORT_ENABLE_ALL));
    check(api->SetIntraOpNumThreads(options, 1));
    check(api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory));
    load_model_from_array();
  } catch (const ApiError& error) {
    throw Error("onnx_load_failed", "The ONNX Runtime backend could not start: " +
                                        std::string(error.what()));
  }
}

double OnnxBackend::Impl::run(const std::vector<std::uint32_t>& ids,
                              const std::vector<std::uint32_t>& mask) {
  std::lock_guard<std::mutex> guard(run_mutex);
  const size_t seq = ids.size();
  std::vector<int64_t> input_ids(seq), attention_mask(seq);
  for (size_t i = 0; i < seq; ++i) {
    input_ids[i] = static_cast<int64_t>(ids[i]);
    attention_mask[i] = static_cast<int64_t>(mask[i]);
  }
  const int64_t shape[2] = {1, static_cast<int64_t>(seq)};
  OrtValue* inputs[2] = {nullptr, nullptr};
  OrtValue* output = nullptr;
  const char* input_names[2] = {"input_ids", "attention_mask"};
  const char* output_names[1] = {"logits"};
  struct ReleaseAndExit {
    OrtValue* inputs[2];
    OrtValue* output;
    Impl* self;
    ~ReleaseAndExit() {
      if (output) self->api->ReleaseValue(output);
      if (inputs[0]) self->api->ReleaseValue(inputs[0]);
      if (inputs[1]) self->api->ReleaseValue(inputs[1]);
    }
  } release{{inputs[0], inputs[1]}, output, this};
  try {
    check(api->CreateTensorWithDataAsOrtValue(memory, input_ids.data(), input_ids.size() * sizeof(int64_t),
                                              shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &inputs[0]));
    check(api->CreateTensorWithDataAsOrtValue(memory, attention_mask.data(),
                                              attention_mask.size() * sizeof(int64_t),
                                              shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &inputs[1]));
    check(api->Run(session, nullptr, input_names, const_cast<const OrtValue**>(inputs), 2,
                   output_names, 1, &output));
    void* raw = nullptr;
    check(api->GetTensorMutableData(output, &raw));
    require(raw != nullptr, "invalid_output", "The ONNX run returned no logits.");
    const float* logits = static_cast<const float*>(raw);
    require(std::isfinite(logits[0]), "invalid_output", "The ONNX model returned a non-finite logit.");
    return static_cast<double>(logits[0]);
  } catch (const ApiError& error) {
    throw Error("inference_failed", "The ONNX gradient inference failed: " + std::string(error.what()));
  }
}

OnnxBackend::OnnxBackend(const fs::path& model_directory, const fs::path& cache_directory)
    : impl_(std::make_unique<Impl>()) {
  impl_->open(model_directory, cache_directory);
}
const char* OnnxBackend::device() const { return impl_->device; }
OnnxBackend::~OnnxBackend() = default;

double OnnxBackend::logit(const std::vector<std::uint32_t>& ids,
                          const std::vector<std::uint32_t>& mask) {
  if (ids.size() > max_len || ids.size() != mask.size())
    throw Error("invalid_input", "Gradient input exceeds the 512-token limit.");
  if (ids.size() < 3 || ids.front() != 1 || ids.back() != 2)
    throw Error("invalid_input", "Gradient input must be wrapped as CLS..SEP.");
  return impl_->run(ids, mask);
}

}  // namespace aihider