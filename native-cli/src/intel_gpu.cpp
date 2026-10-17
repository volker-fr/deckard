#include "intel_gpu.hpp"

#include <dlfcn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <mutex>

namespace aihider {
namespace {

constexpr const char* intel_pci_vendor = "0x8086";
constexpr std::uint32_t intel_vendor_id = 0x8086;

// Whether there is an Intel PCI display controller (base class 0x03); sets
// `driver` to the kernel driver bound to it, "" when none is bound.
bool intel_display_device(std::string& driver) {
  std::error_code error;
  for (const auto& entry : fs::directory_iterator("/sys/bus/pci/devices", error)) {
    std::ifstream vendor_file(entry.path() / "vendor"), class_file(entry.path() / "class");
    std::string vendor, device_class;
    vendor_file >> vendor;
    class_file >> device_class;
    if (vendor != intel_pci_vendor || device_class.rfind("0x03", 0) != 0) continue;
    std::error_code link_error;
    driver = fs::read_symlink(entry.path() / "driver", link_error).filename().string();
    if (link_error) driver.clear();
    return true;
  }
  return false;
}

// Whether an ICD file of ocl-icd names Intel's OpenCL driver (current
// intel-compute-runtime or its legacy1 branch).
bool intel_icd_registered() {
  const char* override_directory = std::getenv("OCL_ICD_VENDORS");
  const fs::path directory = override_directory && fs::is_directory(override_directory)
      ? fs::path(override_directory) : fs::path("/etc/OpenCL/vendors");
  std::error_code error;
  for (const auto& entry : fs::directory_iterator(directory, error)) {
    if (entry.path().extension() != ".icd") continue;
    std::ifstream file(entry.path());
    std::string library;
    std::getline(file, library);
    if (library.find("igdrcl") != std::string::npos || library.find("intel") != std::string::npos) return true;
  }
  return false;
}

// OpenCL 1.2 entry points, resolved from the loader at run time.
using GetPlatformIds = int (*)(std::uint32_t, void**, std::uint32_t*);
using GetPlatformInfo = int (*)(void*, std::uint32_t, size_t, void*, size_t*);
using GetDeviceIds = int (*)(void*, std::uint64_t, std::uint32_t, void**, std::uint32_t*);
using GetDeviceInfo = int (*)(void*, std::uint32_t, size_t, void*, size_t*);
constexpr std::uint32_t cl_platform_name = 0x0903;
constexpr std::uint64_t cl_device_type_gpu = 1 << 2;
constexpr std::uint32_t cl_device_name = 0x102B;
// The 4-byte PCI vendor id; not CL_DEVICE_VENDOR (0x102C), which is a string.
constexpr std::uint32_t cl_device_vendor_id = 0x1001;

std::string info_string(GetDeviceInfo get, void* device, std::uint32_t parameter) {
  char value[256] = {};
  return get(device, parameter, sizeof(value) - 1, value, nullptr) == 0 ? value : "";
}

// Lists GPU devices and reports whether one is an Intel GPU. Runs in the probe
// child only.
void enumerate_opencl(void* loader, IntelGpu& gpu) {
  auto platform_ids = reinterpret_cast<GetPlatformIds>(dlsym(loader, "clGetPlatformIDs"));
  auto platform_info = reinterpret_cast<GetPlatformInfo>(dlsym(loader, "clGetPlatformInfo"));
  auto device_ids = reinterpret_cast<GetDeviceIds>(dlsym(loader, "clGetDeviceIDs"));
  auto device_info = reinterpret_cast<GetDeviceInfo>(dlsym(loader, "clGetDeviceInfo"));
  if (!platform_ids || !platform_info || !device_ids || !device_info) return;
  void* platforms[16];
  std::uint32_t platform_count = 0;
  if (platform_ids(16, platforms, &platform_count) != 0) return;
  for (std::uint32_t p = 0; p < std::min<std::uint32_t>(platform_count, 16); ++p) {
    char platform_name[256] = {};
    platform_info(platforms[p], cl_platform_name, sizeof(platform_name) - 1, platform_name, nullptr);
    void* devices[16];
    std::uint32_t device_count = 0;
    if (device_ids(platforms[p], cl_device_type_gpu, 16, devices, &device_count) != 0) continue;
    for (std::uint32_t d = 0; d < std::min<std::uint32_t>(device_count, 16); ++d) {
      std::uint32_t vendor = 0;
      device_info(devices[d], cl_device_vendor_id, sizeof(vendor), &vendor, nullptr);
      gpu.devices.push_back(std::string(platform_name) + ": " + info_string(device_info, devices[d], cl_device_name));
      if (vendor == intel_vendor_id) gpu.usable = true;
    }
  }
}

IntelGpu probe_here() {
  IntelGpu gpu;
  gpu.present = intel_display_device(gpu.kernel_driver);
  gpu.opencl_driver = intel_icd_registered();
  void* loader = dlopen("libOpenCL.so.1", RTLD_NOW | RTLD_LOCAL);
  gpu.opencl_loader = loader != nullptr;
  if (loader) enumerate_opencl(loader, gpu);
  return gpu;
}

Json to_json(const IntelGpu& gpu) {
  return {{"present", gpu.present}, {"kernel_driver", gpu.kernel_driver}, {"opencl_loader", gpu.opencl_loader},
          {"opencl_driver", gpu.opencl_driver}, {"devices", gpu.devices}, {"usable", gpu.usable}};
}

IntelGpu from_json(const Json& value) {
  IntelGpu gpu;
  gpu.present = value.at("present");
  gpu.kernel_driver = value.at("kernel_driver");
  gpu.opencl_loader = value.at("opencl_loader");
  gpu.opencl_driver = value.at("opencl_driver");
  gpu.devices = value.at("devices").get<std::vector<std::string>>();
  gpu.usable = value.at("usable");
  return gpu;
}

// Runs probe_here() in a child and reads its result back through a pipe. A
// crashed child leaves the hardware facts known so far: none.
IntelGpu probe_in_child() {
  int channel[2];
  if (pipe(channel) != 0) return {};
  const pid_t child = fork();
  if (child == 0) {
    close(channel[0]);
    const std::string report = to_json(probe_here()).dump();
    (void)!write(channel[1], report.data(), report.size());
    _exit(0);
  }
  close(channel[1]);
  std::string report;
  char buffer[4096];
  for (ssize_t count; (count = read(channel[0], buffer, sizeof(buffer))) != 0;) {
    if (count < 0 && errno == EINTR) continue;
    if (count < 0) break;
    report.append(buffer, static_cast<size_t>(count));
  }
  close(channel[0]);
  int status = 0;
  while (child > 0 && waitpid(child, &status, 0) < 0 && errno == EINTR) {
  }
  try {
    return from_json(Json::parse(report));
  } catch (const std::exception&) {
    return {};
  }
}

}  // namespace

const IntelGpu& intel_gpu() {
  static std::once_flag once;
  static IntelGpu gpu;
  std::call_once(once, [] { gpu = probe_in_child(); });
  return gpu;
}

}  // namespace aihider
