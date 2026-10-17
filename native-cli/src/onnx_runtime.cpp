#include "onnx_runtime.hpp"

#include <dlfcn.h>
#include <unistd.h>

#include <fstream>
#include <mutex>
#include <optional>

// generated/embedded_libraries.inc lists DECKARD_EMBEDDED(symbol, file, sha256)
// for every library linux.cmake embeds; the core comes first.
#define DECKARD_EMBEDDED(symbol, file, sha256) \
  extern "C" const unsigned char _binary_##symbol##_bin_start[], _binary_##symbol##_bin_end[];
#include "embedded_libraries.inc"
#undef DECKARD_EMBEDDED

namespace aihider {
namespace {

struct EmbeddedLibrary {
  const char* file;
  const char* sha256;
  const unsigned char* begin;
  const unsigned char* end;
};

#define DECKARD_EMBEDDED(symbol, file, sha256) {file, sha256, _binary_##symbol##_bin_start, _binary_##symbol##_bin_end},
constexpr EmbeddedLibrary embedded_libraries[] = {
#include "embedded_libraries.inc"
};
#undef DECKARD_EMBEDDED

bool current(const fs::path& path, const EmbeddedLibrary& library) {
  std::error_code error;
  return fs::is_regular_file(fs::symlink_status(path, error)) && sha256(path) == library.sha256;
}

// Writes `library` into `directory` unless it is already there, through a
// temporary file and a rename, so a concurrent deckard never sees a partially
// written library.
void write_library(const fs::path& directory, const EmbeddedLibrary& library) {
  const fs::path target = directory / library.file;
  if (current(target, library)) return;
  const fs::path temporary = directory / (std::string(".") + library.file + "." + std::to_string(getpid()));
  {
    std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
    stream.write(reinterpret_cast<const char*>(library.begin), library.end - library.begin);
    if (!stream) {
      std::error_code ignored;
      fs::remove(temporary, ignored);
      throw Error("onnx_unavailable", "Cannot write the ONNX Runtime libraries to " + directory.string() + ".");
    }
  }
  fs::rename(temporary, target);
  if (!current(target, library))
    throw Error("onnx_unavailable", std::string("The written ") + library.file + " does not match its pinned SHA-256.");
}

const OrtApi& load(const fs::path& directory) {
  fs::create_directories(directory);
  for (const auto& library : embedded_libraries) write_library(directory, library);
  const fs::path core = directory / embedded_libraries[0].file;
  void* handle = dlopen(core.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle) throw Error("onnx_unavailable", std::string("Cannot load ONNX Runtime: ") + dlerror());
  using GetApiBase = const OrtApiBase* (*)();
  auto get_api_base = reinterpret_cast<GetApiBase>(dlsym(handle, "OrtGetApiBase"));
  const OrtApi* api = get_api_base ? get_api_base()->GetApi(ORT_API_VERSION) : nullptr;
  if (!api) throw Error("onnx_unavailable", "The embedded ONNX Runtime does not provide C API version " +
                                                std::to_string(ORT_API_VERSION) + ".");
  // The core stays loaded for the life of the process.
  return *api;
}

// Where the embedded libraries are written; the core loads the execution
// providers from here. One directory per build, so releases with another
// ONNX Runtime (or the CPU-only build) do not overwrite each other's files.
fs::path onnx_runtime_directory(const fs::path& cache_directory) {
  return cache_directory / ("onnxruntime-" ONNXRUNTIME_VERSION "-openvino-" OPENVINO_VERSION);
}

}  // namespace

const OrtApi& onnx_runtime(const fs::path& cache_directory) {
  static std::mutex mutex;
  static const OrtApi* api = nullptr;
  static std::optional<Error> failure;
  std::lock_guard<std::mutex> lock(mutex);
  if (api) return *api;
  if (failure) throw *failure;
  try {
    api = &load(onnx_runtime_directory(cache_directory));
    return *api;
  } catch (const Error& error) {
    failure = error;
    throw;
  } catch (const std::exception& error) {
    failure = Error("onnx_unavailable", std::string("Cannot prepare ONNX Runtime: ") + error.what());
    throw *failure;
  }
}

void onnx_check(const OrtApi& api, OrtStatus* status, const char* code, const std::string& context) {
  if (!status) return;
  const std::string message = context + ": " + api.GetErrorMessage(status);
  api.ReleaseStatus(status);
  throw Error(code, message);
}

}  // namespace aihider
