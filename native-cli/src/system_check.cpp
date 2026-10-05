#include "system_check.hpp"

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

#if defined(__linux__)
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <dlfcn.h>
#include <map>
#include <mutex>
#include <sys/utsname.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace aihider {
namespace {

#if defined(__linux__)
using clGetPlatformIDs_fn = int (*)(std::uint32_t, void**, std::uint32_t*);
using clGetPlatformInfo_fn = int (*)(void*, std::uint32_t, std::size_t, void*, std::size_t*);
using clGetDeviceIDs_fn = int (*)(void*, std::uint64_t, std::uint32_t, void*, std::uint32_t*);
using clGetDeviceInfo_fn = int (*)(void*, std::uint32_t, std::size_t, void*, std::size_t*);

constexpr std::uint32_t cl_platform_name = 0x0903;
constexpr std::uint64_t cl_device_type_gpu = 1ULL << 2;
// CL_DEVICE_VENDOR_ID (0x1001) is the 4-byte PCI vendor id; 0x102C is
// CL_DEVICE_VENDOR, a string, and must not be read into an integer.
constexpr std::uint32_t cl_device_vendor_id = 0x1001;
constexpr std::uint32_t cl_device_name = 0x102B;
constexpr std::uint32_t vendor_id_intel = 0x8086;

bool software_renderer(const std::string& name) {
  return name.find("pocl") != std::string::npos || name.find("llvmpipe") != std::string::npos ||
         name.find("cpu-haswell") != std::string::npos;
}

// True for an OpenCL platform that identifies itself as Intel. Used only as a
// fallback for drivers that do not answer CL_DEVICE_VENDOR_ID.
bool intel_platform(const std::string& platform) {
  return platform.find("Intel") != std::string::npos;
}
#endif

}  // namespace

#if defined(__linux__)

std::string which(const std::string& name) {
  const char* path = std::getenv("PATH");
  if (!path) return {};
  std::istringstream dirs(path);
  std::string dir;
  while (std::getline(dirs, dir, ':')) {
    if (dir.empty()) continue;
    const fs::path candidate = fs::path(dir) / name;
    if (::access(candidate.c_str(), X_OK) == 0) return candidate.string();
  }
  return {};
}

bool opencl_loader_present() {
  void* handle = ::dlopen("libOpenCL.so.1", RTLD_NOW | RTLD_LOCAL);
  if (!handle) return false;
  ::dlclose(handle);
  return true;
}

bool intel_gpu_present() {
  std::error_code error;
  const fs::path drm("/sys/class/drm");
  if (!fs::is_directory(drm, error)) return false;
  for (const auto& entry : fs::directory_iterator(drm, fs::directory_options::skip_permission_denied, error)) {
    if (error) break;
    std::error_code link_error;
    auto target = fs::read_symlink(entry.path() / "device/driver", link_error);
    if (link_error) continue;
    if (target.filename() == "i915") return true;
  }
  return false;
}

bool intel_icd_present() {
  std::error_code error;
  const fs::path dir("/etc/OpenCL/vendors");
  if (!fs::is_directory(dir, error)) return false;
  for (const auto& entry : fs::directory_iterator(dir, fs::directory_options::skip_permission_denied, error)) {
    if (error) break;
    if (entry.path().extension() != ".icd") continue;
    std::ifstream file(entry.path());
    std::string line;
    std::getline(file, line);
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r' || line.back() == ' ')) line.pop_back();
    if (line.find("intel") != std::string::npos || line.find("igdrcl") != std::string::npos) return true;
  }
  return false;
}

bool cuda_driver_present() {
  void* handle = ::dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
  if (!handle) return false;
  ::dlclose(handle);
  return true;
}

std::vector<OpenCLDevice> opencl_gpu_devices() {
  std::vector<OpenCLDevice> devices;
  void* ocl = ::dlopen("libOpenCL.so.1", RTLD_NOW | RTLD_LOCAL);
  if (!ocl) return devices;
  struct CloseOcl {
    ~CloseOcl() {
      if (handle) ::dlclose(handle);
    }
    void* handle;
  } close_ocl{ocl};

  auto get_platforms = reinterpret_cast<clGetPlatformIDs_fn>(::dlsym(ocl, "clGetPlatformIDs"));
  auto get_platform_info = reinterpret_cast<clGetPlatformInfo_fn>(::dlsym(ocl, "clGetPlatformInfo"));
  auto get_devices = reinterpret_cast<clGetDeviceIDs_fn>(::dlsym(ocl, "clGetDeviceIDs"));
  auto get_info = reinterpret_cast<clGetDeviceInfo_fn>(::dlsym(ocl, "clGetDeviceInfo"));
  if (!get_platforms || !get_platform_info || !get_devices || !get_info) return devices;

  std::uint32_t platforms = 0;
  if (get_platforms(0, nullptr, &platforms) != 0 || platforms == 0) return devices;
  std::vector<void*> platform_list(platforms);
  if (get_platforms(platforms, platform_list.data(), nullptr) != 0) return devices;

  for (void* platform : platform_list) {
    std::string platform_name;
    char buffer[256] = {};
    std::size_t size = 0;
    if (get_platform_info(platform, cl_platform_name, sizeof(buffer) - 1, buffer, &size) == 0)
      platform_name = buffer;
    std::uint32_t device_count = 0;
    if (get_devices(platform, cl_device_type_gpu, 0, nullptr, &device_count) != 0 || device_count == 0) continue;
    std::vector<void*> device_list(device_count);
    if (get_devices(platform, cl_device_type_gpu, device_count, device_list.data(), nullptr) != 0) continue;
    for (void* device : device_list) {
      std::uint32_t vendor = 0;
      std::size_t unused = 0;
      // A driver that does not answer CL_DEVICE_VENDOR_ID leaves the vendor
      // unknown (0); the device is still reported, and usable_intel_gpu()
      // decides whether it is an Intel GPU.
      if (get_info(device, cl_device_vendor_id, sizeof(vendor), &vendor, &unused) != 0) vendor = 0;
      char name[256] = {};
      if (get_info(device, cl_device_name, sizeof(name) - 1, name, &unused) != 0) continue;
      OpenCLDevice info = {platform_name, vendor, name};
      if (software_renderer(info.name)) info.name += " (software renderer: rejected)";
      else if (info.vendor == 0) info.name += " (vendor id unavailable)";
      devices.push_back(info);
    }
  }
  return devices;
}

bool usable_intel_gpu() {
  for (const auto& device : opencl_gpu_devices()) {
    if (software_renderer(device.name)) continue;
    // The authoritative signal is Intel's PCI vendor id (0x8086). Drivers that
    // do not answer CL_DEVICE_VENDOR_ID are accepted only when the device is on
    // an OpenCL platform that identifies itself as Intel, which still rules out
    // software renderers and foreign accelerators.
    if (device.vendor != vendor_id_intel && !(device.vendor == 0 && intel_platform(device.platform))) continue;
    return true;
  }
  return false;
}

const std::vector<std::string>& openvino_load_order() {
  static const std::vector<std::string> order = {
      "libtbb.so.12", "libtbbmalloc.so", "libopenvino_c.so.2541",
      "libopenvino.so.2541", "libopenvino_onnx_frontend.so.2541",
      "libopenvino_intel_gpu_plugin.so", "libonnxruntime_providers_shared.so",
      "libonnxruntime_providers_openvino.so", "libonnxruntime.so.1.24.1"};
  return order;
}

bool openvino_runtime_loads(const std::string& directory) {
  static std::mutex cache_mutex;
  static std::map<std::string, bool> cache;
  {
    std::lock_guard<std::mutex> lock(cache_mutex);
    const auto found = cache.find(directory);
    if (found != cache.end()) return found->second;
  }
  bool loads = false;
  const pid_t child = ::fork();
  if (child == 0) {
    // Never let the child continue past the probe, even if a library manages
    // to load: it must not write to stdout/stderr or run the runtime.
    for (const auto& name : openvino_load_order()) {
      if (::dlopen((fs::path(directory) / name).c_str(), RTLD_NOW | RTLD_GLOBAL)) continue;
      ::_exit(3);
    }
    ::_exit(0);
  }
  if (child > 0) {
    int status = 0;
    // EINTR must not be mistaken for a crash, so keep waiting on a signal.
    while (::waitpid(child, &status, 0) < 0 && errno == EINTR) {
    }
    loads = WIFEXITED(status) && WEXITSTATUS(status) == 0;
  }
  std::lock_guard<std::mutex> lock(cache_mutex);
  cache[directory] = loads;
  return loads;
}

bool container_engine(std::string& engine) {
  for (const auto& candidate : {"docker", "podman"}) {
    const std::string found = which(candidate);
    if (!found.empty()) {
      engine = found;
      return true;
    }
  }
  return false;
}

std::string distro_id() {
  std::ifstream os_file("/etc/os-release");
  std::string line;
  while (std::getline(os_file, line)) {
    if (line.rfind("ID=", 0) != 0) continue;
    auto value = line.substr(line.find('=') + 1);
    value.erase(std::remove(value.begin(), value.end(), '"'), value.end());
    std::istringstream words(value);
    std::string word;
    if (words >> word) return word;
  }
  return "unknown";
}

#endif  // __linux__

Json system_check() {
  Json report = Json::object();
  report["command"] = "check-system";
#if defined(__linux__)
  struct utsname uname_data = {};
  if (::uname(&uname_data) == 0) report["arch"] = uname_data.machine;
  report["platform"] = "linux";
  report["os"] = distro_id();
  report["embedded_runtimes"] = Json::array({"onnx", "candle"});

  const bool loader = opencl_loader_present();
  const bool i915 = intel_gpu_present();
  const bool icd = intel_icd_present();
  const bool cuda = cuda_driver_present();
  const auto devices = opencl_gpu_devices();

  Json gpu = Json::object();
  gpu["loader"] = loader;
  gpu["intel_icd"] = icd;
  gpu["i915"] = i915;
  Json device_list = Json::array();
  for (const auto& device : devices) {
    std::ostringstream vendor;
    vendor << "0x" << std::hex << std::setw(4) << std::setfill('0') << device.vendor;
    device_list.push_back(
        Json{{"platform", device.platform}, {"vendor", vendor.str()}, {"name", device.name}});
  }
  gpu["devices"] = device_list;
  gpu["usable_intel_gpu"] = usable_intel_gpu();
  report["gpu_opencl"] = gpu;

  report["nvidia_cuda"] = Json{{"driver", cuda}};

  std::string engine;
  std::string engine_name;
  if (container_engine(engine)) engine_name = fs::path(engine).filename().string();
  report["container_engine"] = engine_name.empty() ? Json() : Json(engine_name);

  const fs::path cache = model_cache();
  const auto& assets = model_assets();
  std::string onnx_model, onnx_pin;
  if (assets.contains("onnx") && assets["onnx"].contains("model") && assets["onnx"].contains("sha256")) {
    onnx_model = assets["onnx"].at("model").get<std::string>();
    onnx_pin = assets["onnx"].at("sha256").get<std::string>();
  }
  const bool candle_assets =
      fs::is_regular_file(cache / "config.json") && fs::is_regular_file(cache / "model.safetensors") &&
      fs::is_regular_file(cache / "tokenizer.json");
  const bool onnx_asset = !onnx_model.empty() && fs::is_regular_file(cache / onnx_model);
  bool onnx_pinned = false;
  if (onnx_asset) {
    try {
      onnx_pinned = sha256(cache / onnx_model) == onnx_pin;
    } catch (const Error&) {
    }
  }
  Json models = Json::object();
  models["cache"] = cache.string();
  models["candle"] = candle_assets ? "ready" : "missing";
  models["onnx"] = onnx_asset ? (onnx_pinned ? "pinned" : "present-but-drifted") : "missing";
  report["models"] = models;

  Json problems = Json::array();
  Json help = Json::array();
  help.push_back("model.onnx runs the ONNX Runtime (Intel GPU via OpenVINO when the probe finds a "
                 "usable Intel GPU, else the embedded CPU core); otherwise the Candle CPU runtime is "
                 "used. The embedded runtimes are always present, so inference never depends on a "
                 "vendor driver.");

  // The OpenVINO core is only materialized once a usable Intel GPU is found,
  // so probe the very directory the runtime would load from.
  const fs::path ov_dir = cache / "onnxruntime-openvino";
  const bool intel_gpu = usable_intel_gpu();
  const bool provider_loads = intel_gpu && openvino_runtime_loads(ov_dir.string());
  const bool accelerated = intel_gpu && provider_loads;
  if (intel_gpu) {
    Json openvino = Json::object();
    openvino["directory"] = ov_dir.string();
    openvino["provider_loads"] = provider_loads;
    report["openvino"] = openvino;
  }
  if (i915 && !loader) {
    problems.push_back("Missing OpenCL loader: libOpenCL.so.1");
    help.push_back("An Intel GPU is present but the OpenCL loader is missing. It must come from your "
                   "distribution (Deckard never installs it); the embedded ONNX CPU core is used "
                   "until the OpenCL stack is present.");
  }
  if (loader && !intel_gpu) {
    problems.push_back("No usable Intel GPU for OpenVINO: the installed OpenCL stack exposes no "
                       "Intel GPU device.");
    help.push_back("The Intel OpenCL stack is installed but does not expose a usable GPU. Current "
                   "intel-compute-runtime supports Gen12+ only; for Gen8/9/11 (e.g. UHD 630) install "
                   "the legacy1 24.35 branch (intel-opencl-icd-legacy1) instead. Until then the ONNX "
                   "CPU core is used.");
  }
  if (intel_gpu && !provider_loads) {
    problems.push_back("The Intel GPU is visible to OpenCL but the OpenVINO runtime fails to load "
                       "against it, so ONNX inference runs on the CPU core.");
    help.push_back("OpenVINO loads the GPU in a library initializer and this driver crashes there, "
                   "which Deckard detects in a forked process and works around by using the CPU core. "
                   "On Intel iGPUs the usual cause is the memlock limit: raise it (for example "
                   "'LimitMEMLOCK=infinity' in /etc/security/limits.conf, then log in again) or use a "
                   "driver generation OpenVINO supports.");
  }
  if (!cuda) {
    problems.push_back("CUDA driver (libcuda.so.1) not detected");
    help.push_back("A CUDA GPU would need the NVIDIA driver/CUDA on the system; it is not present. "
                   "The embedded Candle CPU runtime always remains available.");
  }
  if (onnx_asset && !onnx_pinned)
    help.push_back("model.onnx in the model cache does not match the pinned checksum; reinstall it "
                   "with: deckard install");
  if (!onnx_asset && engine_name.empty())
    help.push_back("model.onnx is absent and no container engine (docker/podman) is available to "
                   "recreate it; reinstall to fetch the pinned assets: deckard install");
  if (onnx_asset && !engine_name.empty())
    help.push_back("A container engine (" + engine_name +
                   ") is available, so missing/drifted model.onnx can be regenerated automatically "
                   "at install time.");

  report["problems"] = problems;
  report["help"] = help;
  report["accelerated"] = accelerated;
  report["expected_device"] = accelerated ? "gpu" : (onnx_asset ? "cpu" : "candle");
#else
  report["platform"] = "apple";
  report["note"] = "GPU capability diagnostics are Linux-only; the Core ML runtime serves macOS.";
  report["help"] = Json::array(
      {"Deckard uses the prebuilt Core ML runtime on macOS.",
       "No vendor GPU driver is required; a missing model is reinstalled with: deckard install"});
  report["accelerated"] = false;
  report["expected_device"] = "cpu";
#endif
  return report;
}

}  // namespace aihider