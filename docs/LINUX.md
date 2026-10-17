# Deckard on Linux

Deckard runs on Linux x86_64 with the same Chrome extension, native-messaging
protocol, policy and model identity as on macOS. Only the inference runtime
differs:

| | macOS (Apple Silicon) | Linux (x86_64) |
|---|---|---|
| Runtime | Core ML (`coreml_gradient.mm`) | Candle (`native-cli/candle`, Rust); optionally [ONNX Runtime](#onnx-runtime-optional), on the CPU or an [Intel GPU](#intel-gpus) |
| Model | pinned fp16 Core ML export, shipped in the release bundle | pinned fp32 Hugging Face checkpoint, downloaded at install |
| Model files | `model-assets.json` | `model-assets-linux.json` |
| Install prefix | `~/Deckard` | `~/.local/share/deckard` |
| Model cache | `~/Library/Caches/Deckard/coreml` | `~/.cache/deckard` |
| Chrome host registration | `~/Library/Application Support/Google/Chrome/NativeMessagingHosts` | `~/.config/google-chrome/NativeMessagingHosts` |

Like the Core ML backend, Candle never lets masked positions contribute, and
position ids do not depend on the sequence length. Linux scores the fp32
checkpoint itself rather than the fp16 export.

There is no Linux release bundle yet, so Deckard is built from source.

## Build

System packages (names vary by distribution): a C++20 compiler, CMake 3.24+,
`pkg-config`, `curl`, `unzip`, and the OpenSSL and libcurl development headers
(for example `g++ cmake pkg-config curl unzip libssl-dev libcurl4-openssl-dev`
on Debian/Ubuntu).

```sh
scripts/build-native.sh
```

`native-cli/bootstrap.sh` hands Linux off to `native-cli/bootstrap-linux.sh`,
which fills `cache/native-build` with the pinned dependencies: nlohmann/json,
license notices, Rust 1.90.0 (installed through a checksummed `rustup-init`,
exactly like on macOS) and an offline crate registry for the tokenizer and
Candle crates. A pre-filled `NATIVE_CACHE` elsewhere is used read-only, and its
Rust toolchain must really be 1.90.0. `native-cli/linux.cmake` then builds the
Candle staticlib with that toolchain and links it into `deckard`.

It also fetches ONNX Runtime: the C headers from the official release
(`onnxruntime-linux-x64` tgz), the pre-compiled libraries from the
`onnxruntime-openvino` wheel (the core, the OpenVINO execution provider and
the OpenVINO parts it needs for [Intel GPUs](#intel-gpus)), and their license
notices; the versions are set in `bootstrap-linux.sh` and `linux.cmake`.
`linux.cmake` embeds the libraries in `deckard` with `objcopy` (from
binutils, which compilers usually pull in).

## Install

From the repository root:

```sh
native-cli/build/dist/bin/deckard install --extension-dir extension
```

A source build has no packaged extension next to the binary, so
`--extension-dir` points install at the repository's `extension/` folder; it
is copied to `~/.local/share/deckard/extension`. Install also adds deckard to
`PATH` in your shell profile (`--shell bash|zsh`, default: your login shell);
`--shell none` skips that, for example when the profile is a symlink managed
elsewhere.

Without `--model-dir`, install downloads the three pinned files
(`config.json`, `model.safetensors`, `tokenizer.json`, about 1.7 GB) from
Hugging Face into the model cache, resuming partial downloads and verifying
every SHA-256 pin. `--model-dir DIR` installs from a local copy instead (other
files in `DIR` are ignored); `--no-download` fails instead of downloading
missing files. Then load `~/.local/share/deckard/extension` unpacked in Chrome
as described in the README.

This installs Candle; with a pinned `model.onnx` in the same directory, ONNX
Runtime (see [Installing with model.onnx](#installing-with-modelonnx)).

## Performance

Measured on an Intel Core i5-8500T with Intel UHD Graphics 630:

| | Candle |
|---|---|
| Model load, once per host start | 4.6 s, of which about 4 s is the SHA-256 check of the weights |
| One window of 32 / 128 / 512 tokens | 0.65 / 1.44 / 8.2 s |
| A real page ([Prussian Union on EndlessWiki](https://www.endlesswiki.com/wiki/prussian_union)), warm host | 11.2 s |
| CPU cores used while scoring | about 1 of 6 |
| Memory of the Chrome-launched host | 1.8 GB resident; up to 3.3 GB while `deckard verify` loads and checks the model |

Window timings are the median of `deckard verify` runs after a warm-up; the
page timing is the CPU activity of the host Chrome launched while the
extension scored the page.

Candle runs almost entirely on one core, so scoring a page leaves the rest of
the machine responsive but takes several seconds. Chrome starts the host on
demand and the extension closes it after five idle minutes, so the first page
after a pause also waits for the model load. The Linux host uses more memory
than the macOS one because it runs the fp32 checkpoint rather than the fp16
Core ML export.

### Why Candle, and what a faster runtime costs

ONNX Runtime scores the same checkpoint (exported to `model.onnx`) about five
times faster on this CPU, because it spreads the work over several cores. It
costs more almost everywhere else (disk, dependencies, start-up, memory,
binary size), so Candle is the default and ONNX Runtime is an opt-in (see
[ONNX Runtime](#onnx-runtime-optional)). Measured on the same machine:

| | Candle | ONNX Runtime (CPU) |
|---|---|---|
| One window of 512 tokens | 8.2 s | 1.65 s |
| CPU cores used while scoring | about 1 | about 3 |
| Model load, once per host start | 4.6 s | 7.7 s |
| Model load plus one 512-token window | 12.8 s | 9.4 s |
| Peak memory (`deckard verify`) | 3.3 GB | 3.6 GB |
| Memory of the host after scoring one text | 1.85 GB resident | 2.8 GB resident |
| `deckard` binary | 35 MB | about 121 MB (embeds the 84 MB of ONNX Runtime and OpenVINO libraries) |
| Disk beyond the three Hugging Face files | none | `model.onnx`, 1.8 GB, in the model cache and again in the installation; the 84 MB of libraries in the model cache |
| Getting the model | download the pinned files | additionally export `model.onnx` with PyTorch (nobody publishes it): Python 3.14, about 1.3 GB of downloads and a few minutes |
| Build dependencies | Rust toolchain and crates | additionally ONNX Runtime headers and libraries (8 MB and 81 MB downloads) |
| Runtime dependencies | libc, libcrypto, libcurl | none beyond those; the libraries are written to the model cache and loaded at run time |

## ONNX Runtime (optional)

- Candle (default): needs only the Hugging Face files (it runs
  `model.safetensors`); uses one CPU core.
- ONNX Runtime (optional, needs `model.onnx`): slower startup and more memory,
  but about five times faster scoring on several cores. Candle is its
  fallback.
- Both run the same checkpoint and match the same PyTorch reference logits.
  Numbers: [Why Candle, and what a faster runtime costs](#why-candle-and-what-a-faster-runtime-costs).

`model.onnx` is pinned like the other model files: its SHA-256 is recorded in
`native-cli/model-assets-onnx.json`, and install and every model load refuse a
`model.onnx` with other bytes. It has its own file, apart from
`model-assets-linux.json`, so existing installations stay recognized. Code:
`onnx_gradient.cpp`, `onnx_runtime.cpp`.

### Which runtime runs

| Installed model | `deckard start`, `scan`, `verify` |
|---|---|
| no `model.onnx` | Candle |
| pinned `model.onnx` | ONNX Runtime; Candle if ONNX Runtime cannot start or fails during inference |
| `model.onnx` with the wrong SHA-256 | error `asset_mismatch`, never a fallback |

- `--candle` or `--onnx` forces one runtime, without fallback.
- A fallback logs `Deckard: onnx_fallback (<reason>); using Candle on the CPU.`
  on stderr.
- `deckard verify` reports the runtime that ran; the host's `ping` reports
  the serving runtime once the model is loaded.

### Why Candle is the fallback

Candle is linked into `deckard`. ONNX Runtime is a set of shared libraries
(84 MB) that `deckard` writes to
`~/.cache/deckard/onnxruntime-<version>-openvino-<version>/` and loads at run
time, which can fail:

- disk full or `~/.cache` not writable;
- `~/.cache` on a separate `noexec` mount;
- glibc older than 2.27;
- not enough memory (ONNX Runtime needs about 1 GB more).

So `model.safetensors` (Candle) stays installed next to `model.onnx`. They
cannot share bytes: `model.onnx` stores every weight again, 1.21 of its
1.84 GB transposed.

### Installing with model.onnx

Nobody publishes `model.onnx`, so `scripts/export-onnx.sh` exports it from
the checkpoint, and install takes it from where it takes the other files
(`~/.cache/deckard` or `--model-dir`). After [Install](#install), which
downloaded the checkpoint the export reads, from the repository root:

```sh
scripts/export-onnx.sh                                               # writes ~/.cache/deckard/model.onnx
native-cli/build/dist/bin/deckard install --extension-dir extension  # new release with model.onnx
```

Then click Reload on the extension in `chrome://extensions`.

- The script needs Python 3.14 with its `venv` module (for example
  `python3.14-venv` on Debian/Ubuntu); `PYTHON` names another interpreter.
  It installs the hash-locked packages of
  `native-cli/onnx/export/requirements.txt` (mainly torch 2.10 for the CPU,
  transformers, onnx; about 1.3 GB) into a temporary venv in the model
  directory (on disk, unlike a RAM-backed `/tmp`) and removes them
  afterwards. The export takes a few minutes and up to about 10 GB of memory.
- `native-cli/onnx/export.py` reproduces the pin byte for byte on any x86_64
  CPU (it forces CPU-independent PyTorch and MKL kernels); the script fails
  if it does not.
- Pinned `model.onnx` present: install installs it and ONNX Runtime runs.
  Absent: Candle. Wrong SHA-256: the install fails with `asset_mismatch` and
  changes nothing.
- After a pin change, run the script again: it replaces the old `model.onnx`.
- Back to Candle: delete `~/.cache/deckard/model.onnx` and install again.

## Intel GPUs

ONNX Runtime can run `model.onnx` on an Intel GPU through OpenVINO. This is
opt-in: `deckard install --gpu` (needs `model.onnx`) records the choice, and
the Chrome-launched host then tries the GPU first and uses the CPU if the GPU
is not usable. `start`, `scan` and `verify` take `--gpu` for one run;
`verify` reports the `device` that ran. Code: `onnx_providers.cpp`,
`intel_gpu.cpp`.

### Why it is not the default

Same i5-8500T with its UHD Graphics 630 (24 execution units),
`deckard verify --onnx [--gpu]`, median of three runs:

| | ONNX Runtime, CPU | ONNX Runtime + OpenVINO, GPU |
|---|---|---|
| Model load, once per host start | 7.8 s | 13.3 s |
| 32 / 128 / 512 tokens | 0.11 / 0.31 / 1.6–2.3 s | 0.11–0.33 / 0.39 / 2.2 s |
| Peak memory | 3.6 GB | 9.1 GB |
| CPU time for the whole run | 34 s | 15 s |

On this GPU scoring is no faster, start-up is 5.5 s slower and peak memory
exceeds the 6 GiB limit of `native-cli/tests/model-smoke.mjs`. The GPU does
leave the CPU mostly idle while pages are scored. Larger Intel GPUs (Iris Xe,
Arc) are likely faster but have not been measured.

### Why OpenVINO through ONNX Runtime

- OpenVINO is Intel's supported inference path for its GPUs; its GPU plugin
  uses Intel's OpenCL driver.
- ONNX Runtime's OpenVINO execution provider reuses `model.onnx`, the session
  code and the CPU fallback, so the GPU is one entry in
  `execution_providers()`, not another runtime.
- Only the `onnxruntime-openvino` wheel ships an ONNX Runtime core built with
  that provider, so deckard embeds that core for the CPU path too (same speed
  and memory as the official one). Of OpenVINO it embeds only what the GPU
  needs (no CPU or NPU plugin).
- Candle has no Intel GPU backend; Vulkan- or SYCL-based runtimes would need
  another model format and runtime.

### Drivers

| | Package (examples) |
|---|---|
| Kernel driver | `i915` (or `xe` for Xe2 and newer), part of the kernel |
| OpenCL loader | `ocl-icd` (`libOpenCL.so.1`) |
| Intel OpenCL driver, Gen12 (Xe) and newer | `intel-compute-runtime` |
| Intel OpenCL driver, Gen8 to Gen11 (e.g. UHD 630) | the legacy 24.35 branch: `intel-compute-runtime-legacy`, `intel-opencl-icd-legacy1` |

No reboot is needed after installing a driver.

### Safety

- `intel_gpu.cpp` enumerates OpenCL devices in a child process: enumeration
  loads every installed OpenCL driver, and some (pocl) export OpenCL entry
  points that would shadow the loader's inside deckard and break OpenVINO's
  GPU plugin.
- Attaching the OpenVINO provider happens once in a child process first: a
  driver OpenVINO cannot handle may crash it, and only a clean exit lets
  deckard use the GPU.
- `export.py` folds DeBERTa-v2's relative-position `If` nodes into their
  always-taken branch; OpenVINO's GPU plugin cannot compile the unfolded
  graph.

### check-system

`deckard check-system` prints what this host can use and what is missing:

| Key | Meaning |
|---|---|
| `models` | the model cache; the Candle files and `model.onnx` are `ready`, `missing` or `drifted` (wrong SHA-256) |
| `runtime` | the runtime `deckard start` picks: `onnx` or `candle` |
| `intel_gpu` | `present` (PCI), `kernel_driver`, `opencl_loader`, `opencl_driver`, the OpenCL `devices`, `usable` |
| `accelerators` | every entry of `execution_providers()`, its device and whether it starts here |
| `problems`, `help` | each missing piece and how to fix it |

Independent gaps (kernel driver, OpenCL loader, Intel OpenCL driver) are all
reported; checks that need them only once they are present. Probing writes
the embedded libraries to the model cache, as the first ONNX run would.

### Adding an accelerator

ONNX Runtime runs on its built-in CPU provider unless `--gpu` is set and an
entry of `execution_providers()` (`onnx_providers.cpp`) is usable. An entry
names the provider and its device, probes the host without ever crashing the
caller, and configures the session; `OnnxGradient` tries the entries in order
and logs and skips any that fail. A new accelerator is one entry plus its
libraries in `bootstrap-linux.sh` and `linux.cmake`; `check-system` lists it
automatically.

## Test

```sh
npm test
DECKARD_MODEL_DIR=~/.cache/deckard npm run test:native
```

The native suites run against the Linux binary. Tests that only apply to
macOS, such as those for the Core ML runtime or for upgrading from earlier
macOS releases, are skipped on Linux.

`linux-backends.test.mjs` covers runtime selection, fallback, installing with
and without `model.onnx` and the host's `ping`; its ONNX cases need
`model.onnx` in `DECKARD_MODEL_DIR`. `deckard self-test` checks the fallback
rules with scripted runtimes, without a model. CI exports `model.onnx`
(`scripts/export-onnx.sh`), so the ONNX cases run there too.

`linux-intel-gpu.test.mjs` covers `check-system`, a hidden OpenCL driver,
`--gpu` flag conflicts and install `--gpu` without `model.onnx`; with
`model.onnx`, `verify --gpu` must use the GPU exactly when `check-system`
says it can.

`linux-accuracy.test.mjs` checks Candle's scores, not only that it runs: with
the model available, `deckard verify` must reproduce reference logits that
PyTorch computed from the pinned checkpoint
(`native-cli/tests/fixtures/linux-reference.json`) within its 0.002 score
tolerance. When the pin in `model-assets-linux.json` changes, regenerate them
with `native-cli/tests/fixtures/linux_reference.py` (needs `torch` and
`transformers`); the test fails until then. With `model.onnx`, it also checks
ONNX Runtime (`--onnx`) against the same reference.

The Candle crate has its own unit tests (`cargo test` in `native-cli/candle`);
set `DECKARD_CANDLE_CONFIG` and `DECKARD_CANDLE_WEIGHTS` to also run one real
inference.

`linux-cli.test.mjs` covers the Linux side of `deckard install` and the host:
the default paths, the model cache (reuse without network, resumed and
corrupt files, `--no-download`) and a host started under `nice`. Downloads go
to a dead proxy there, so these tests never use the network.

## Tried and measured

Findings and experiments, kept so they are not repeated. Same machine as
[Performance](#performance), unless noted.

| Tried | Result | Decision |
|---|---|---|
| ONNX Runtime: weight prepacking off (`session.disable_prepacking`) | host 2.44 instead of 2.85 GB resident; peak unchanged (3.75 GB, at load); 512 tokens about 20% slower (noisy: measured under load) | kept the default: speed is why ONNX Runtime is there |
| ONNX Runtime: CPU memory arena and memory patterns off | no change (2.8 GB resident) | kept the default |
| ONNX Runtime library from the `onnxruntime-openvino` wheel instead of the official release | same speed and memory; 31 instead of 22 MB, 81 MB download | the wheel: it is the only build that loads the OpenVINO provider ([Intel GPUs](#intel-gpus)) |
| Export: shape inference and checker on the file instead of in memory | peak 9.6 instead of 13.3 GB, identical `model.onnx` | on the file |
