#pragma once

#include "support.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace aihider {

// Host capability diagnostics for the GPU-compute path. Every probe here is
// non-destructive: the OpenCL loader and the CUDA driver are dlopened only to
// test presence, never executed, and GPU enumeration applies the same rules
// the runtime uses, so `deckard check-system` always reports what the runtime
// would actually do rather than what the software stack looks like.
struct OpenCLDevice {
  std::string platform;  // CL_PLATFORM_NAME
  std::uint32_t vendor;  // CL_DEVICE_VENDOR_ID (PCI vendor; 0x8086 = Intel)
  std::string name;      // CL_DEVICE_NAME
};

// Whether the shared OpenCL loader (libOpenCL.so.1, package ocl-icd) exists.
bool opencl_loader_present();

// Enumerated GPU-type OpenCL devices (CL_DEVICE_TYPE_GPU = 0x4). Software
// device types (CPU/ACCELERATOR) are never queried.
std::vector<OpenCLDevice> opencl_gpu_devices();

// The runtime's probe: a device with Intel's vendor id (0x8086) that is not a
// software renderer (pocl, llvmpipe, cpu-haswell), or - for drivers that do not
// answer CL_DEVICE_VENDOR_ID - a GPU on an OpenCL platform that identifies
// itself as Intel. This is the sole authority for whether the OpenVINO GPU
// provider may be engaged.
bool usable_intel_gpu();

// Dependency order of the materialized OpenVINO wheel core + GPU provider. The
// runtime loads exactly this sequence, and the load probe below validates it.
const std::vector<std::string>& openvino_load_order();

// Whether that OpenVINO core can be dlopened from `directory` without killing
// the process. libonnxruntime_providers_openvino.so enumerates Intel GPU
// devices in a global static initializer, so an OpenCL device that exists but
// is unusable (an unsupported generation, an insufficient memlock limit) makes
// the provider segfault *during dlopen* - uncatchable by the runtime's own
// error handling. The load therefore runs in a forked child; only a clean exit
// counts as usable, so an unloadable provider degrades to the CPU core instead
// of crashing. Results are cached per directory for the process lifetime.
bool openvino_runtime_loads(const std::string& directory);

// Whether an i915 device is bound under /sys/class/drm.
bool intel_gpu_present();

// Whether an Intel OpenCL driver is registered under /etc/OpenCL/vendors.
bool intel_icd_present();

// Whether the CUDA driver (libcuda.so.1, NVIDIA) is loadable.
bool cuda_driver_present();

// First executable found on PATH for the name, or an empty string.
std::string which(const std::string& name);

// Whether docker or podman is available, returning its path in engine.
bool container_engine(std::string& engine);

// Distribution id tokens from /etc/os-release (e.g. "manjaro").
std::string distro_id();

// The `deckard check-system` report: GPU/CUDA probes, embedded runtimes,
// model assets, and actionable help when something is missing.
Json system_check();

}  // namespace aihider