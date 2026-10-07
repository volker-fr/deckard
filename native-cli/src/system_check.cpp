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

namespace {
// The first PCI display controller (base class 0x03: VGA, 3D, ...) from
// `vendor`, with the kernel driver bound to it ("" when none is bound).
bool pci_display_device(const std::string& vendor, std::string& driver) {
  std::error_code error;
  const fs::path pci("/sys/bus/pci/devices");
  for (const auto& entry : fs::directory_iterator(pci, fs::directory_options::skip_permission_denied, error)) {
    std::ifstream vendor_file(entry.path() / "vendor"), class_file(entry.path() / "class");
    std::string device_vendor, device_class;
    vendor_file >> device_vendor;
    class_file >> device_class;
    if (device_vendor != vendor || device_class.rfind("0x03", 0) != 0) continue;
    std::error_code link_error;
    driver = fs::read_symlink(entry.path() / "driver", link_error).filename().string();
    if (link_error) driver.clear();
    return true;
  }
  return false;
}
}  // namespace

bool intel_gpu_present() {
  std::string driver;
  return pci_display_device("0x8086", driver);
}

std::string intel_kernel_driver() {
  std::string driver;
  pci_display_device("0x8086", driver);
  return driver;
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

bool nvidia_gpu_present() {
  std::string driver;
  return pci_display_device("0x10de", driver);
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

namespace {
bool intel_gpu_in(const std::vector<OpenCLDevice>& devices) {
  for (const auto& device : devices) {
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
}  // namespace

// Enumerating platforms makes ocl-icd dlopen every ICD into the global symbol
// scope. ICDs such as pocl export clGetExtensionFunctionAddressForPlatform
// themselves, so if this ran in-process before OpenVINO, the GPU plugin would
// later bind that symbol to pocl instead of the loader and get NULL for every
// Intel USM entry point ("clHostMemAllocINTEL is nullptr"). Probe in a forked
// child so the process that may host OpenVINO never loads an ICD first.
bool usable_intel_gpu() {
  static std::once_flag once;
  static bool usable = false;
  std::call_once(once, [] {
    const pid_t child = ::fork();
    if (child == 0) ::_exit(intel_gpu_in(opencl_gpu_devices()) ? 0 : 1);
    if (child < 0) return;
    int status = 0;
    while (::waitpid(child, &status, 0) < 0 && errno == EINTR) {
    }
    usable = WIFEXITED(status) && WEXITSTATUS(status) == 0;
  });
  return usable;
}

const std::vector<std::string>& openvino_load_order() {
  // libonnxruntime_providers_openvino.so is deliberately absent: its static
  // initializer dereferences Provider_GetHost(), which is NULL until the ORT
  // core has called Provider_SetHost. The core dlopens the provider from its
  // own directory when the OpenVINO execution provider is appended.
  static const std::vector<std::string> order = {
      "libtbb.so.12", "libtbbmalloc.so", "libopenvino_c.so.2541",
      "libopenvino.so.2541", "libopenvino_onnx_frontend.so.2541",
      "libopenvino_intel_gpu_plugin.so", "libonnxruntime_providers_shared.so",
      "libonnxruntime.so.1.24.1"};
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
  report["runtimes"] = Json::array({"onnx", "candle"});

  const bool loader = opencl_loader_present();
  const bool intel_present = intel_gpu_present();
  const std::string intel_driver = intel_kernel_driver();
  const bool icd = intel_icd_present();
  const bool cuda = cuda_driver_present();
  const bool nvidia = nvidia_gpu_present();
  const auto devices = opencl_gpu_devices();

  Json gpu = Json::object();
  gpu["present"] = intel_present;
  gpu["kernel_driver"] = intel_driver.empty() ? Json() : Json(intel_driver);
  gpu["opencl_loader"] = loader;
  gpu["opencl_driver"] = icd;
  Json device_list = Json::array();
  for (const auto& device : devices) {
    std::ostringstream vendor;
    vendor << "0x" << std::hex << std::setw(4) << std::setfill('0') << device.vendor;
    device_list.push_back(
        Json{{"platform", device.platform}, {"vendor", vendor.str()}, {"name", device.name}});
  }
  gpu["devices"] = device_list;
  gpu["usable"] = usable_intel_gpu();
  report["intel_gpu"] = gpu;

  report["nvidia_gpu"] = Json{{"present", nvidia}, {"cuda_driver", cuda}};

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
  models["onnx"] = onnx_asset ? (onnx_pinned ? "ready" : "drifted") : "missing";
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
  report["openvino"] = Json{{"directory", ov_dir.string()}, {"loads", provider_loads}};
  // The kernel driver, OpenCL loader and Intel OpenCL driver are independent,
  // so every missing one is reported. Whether that driver exposes the GPU, and
  // whether OpenVINO loads against it, can only be judged once all three are
  // present; before that, their failure is a mere consequence.
  const std::string cpu_fallback = " Until then ONNX inference runs on the CPU core.";
  const bool intel_stack = !intel_driver.empty() && loader && icd;
  if (intel_present && intel_driver.empty()) {
    problems.push_back("Intel GPU present but no kernel driver (i915 or xe) is bound to it");
    help.push_back("Enable the i915 (or, for Xe2 and newer, xe) kernel driver for the Intel GPU." +
                   cpu_fallback);
  }
  if (intel_present && !loader) {
    problems.push_back("Intel GPU present but the OpenCL loader (libOpenCL.so.1) is missing");
    help.push_back("Install the OpenCL ICD loader from your distribution (ocl-icd)." + cpu_fallback);
  }
  if (intel_present && !icd) {
    problems.push_back("Intel GPU present but no Intel OpenCL driver is installed");
    help.push_back("Install Intel's OpenCL driver from your distribution: intel-compute-runtime for "
                   "Gen12 and newer, or the legacy 24.35 branch (intel-compute-runtime-legacy / "
                   "intel-opencl-icd-legacy1) for Gen8 to Gen11 such as UHD 630." + cpu_fallback);
  }
  if (intel_present && intel_stack && !intel_gpu) {
    problems.push_back("Intel GPU present but the installed Intel OpenCL driver does not expose it");
    help.push_back("The Intel OpenCL driver probably does not support this GPU generation: current "
                   "intel-compute-runtime supports Gen12 and newer only; Gen8 to Gen11 (e.g. UHD 630) "
                   "need the legacy 24.35 branch (intel-compute-runtime-legacy / "
                   "intel-opencl-icd-legacy1)." + cpu_fallback);
  } else if (intel_gpu && !provider_loads) {
    problems.push_back("Intel GPU usable through OpenCL but the embedded OpenVINO runtime fails to "
                       "load against it");
    help.push_back("A library of the embedded OpenVINO runtime crashed or failed to load, which "
                   "Deckard detects in a forked process and works around by using the CPU core. "
                   "Check that the Intel OpenCL driver matches the GPU generation OpenVINO "
                   "supports.");
  }
  if (nvidia && !cuda) {
    problems.push_back("NVIDIA GPU present but the CUDA driver (libcuda.so.1) is not loadable");
    help.push_back("Install the NVIDIA driver from your distribution to use CUDA; until then the "
                   "embedded CPU runtimes are used.");
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
  report["runtime"] = onnx_asset ? "onnx" : "candle";
  report["device"] = accelerated ? "gpu" : "cpu";
#else
  report["platform"] = "apple";
  report["note"] = "GPU capability diagnostics are Linux-only; the Core ML runtime serves macOS.";
  report["help"] = Json::array(
      {"Deckard uses the prebuilt Core ML runtime on macOS.",
       "No vendor GPU driver is required; a missing model is reinstalled with: deckard install"});
  report["accelerated"] = false;
  report["runtime"] = "coreml";
  report["device"] = "cpu";
#endif
  return report;
}

}  // namespace aihider