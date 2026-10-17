#pragma once

#include "support.hpp"
#include <string>
#include <vector>

namespace aihider {

// What this Linux host offers for running the OpenVINO GPU plugin, which talks
// to Intel GPUs through OpenCL. See docs/LINUX.md ("Intel GPUs").
struct IntelGpu {
  bool present = false;               // an Intel display controller on PCI
  std::string kernel_driver;          // "i915", "xe", or "" when none is bound
  bool opencl_loader = false;         // libOpenCL.so.1 (ocl-icd) loads
  bool opencl_driver = false;         // an Intel OpenCL driver (ICD) is registered
  std::vector<std::string> devices;   // OpenCL GPU devices, "platform: name"
  bool usable = false;                // the Intel driver exposes an Intel GPU device
};

// Probes once per process in a child process: enumerating OpenCL platforms
// loads every installed OpenCL driver, and some (pocl) export OpenCL entry
// points that would later shadow the loader's for OpenVINO in this process.
// A driver that crashes during enumeration takes down only the child.
const IntelGpu& intel_gpu();

}  // namespace aihider
