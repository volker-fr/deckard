#include "onnx_install.hpp"

#include "onnx_export_embedded.hpp"
#include "support.hpp"
#include "system_check.hpp"

#if defined(__linux__)

#include <cerrno>
#include <fstream>
#include <iostream>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace aihider {
namespace {

// Runs a program as a direct child (so it may prompt or stream to the
// terminal), returning its exit status. The exporter is the only interactive
// step; everything else has no output.
int run(const std::vector<std::string>& command) {
  const std::string executable = which(command.front());
  if (executable.empty()) return 127;
  std::vector<char*> argv;
  for (const auto& arg : command) argv.push_back(const_cast<char*>(arg.c_str()));
  argv.push_back(nullptr);
  const pid_t pid = ::fork();
  if (pid == 0) {
    ::execv(executable.c_str(), argv.data());
    ::_exit(127);
  }
  int status = 0;
  while (::waitpid(pid, &status, 0) < 0) {
    if (errno != EINTR) return 1;
  }
  return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
}

void write_text(const fs::path& path, const std::string& content) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  stream.write(content.data(), static_cast<std::streamsize>(content.size()));
  if (!stream) throw Error("onnx_export_stage", "Cannot write the ONNX exporter stage: " + path.string());
}

}  // namespace

void prepare_onnx_model(const fs::path& source_directory) {
  const auto& assets = model_assets();
  const auto onnx_it = assets.find("onnx");
  if (onnx_it == assets.end() || !onnx_it->is_object()) return;
  const auto& onnx = *onnx_it;
  if (!onnx.contains("model") || !onnx.contains("sha256") || !onnx.contains("export"))
    return;
  const std::string name = onnx.at("model").get<std::string>();
  const std::string expected = onnx.at("sha256").get<std::string>();
  const fs::path output = source_directory / name;

  if (fs::is_regular_file(output)) {
    bool valid = false;
    try { valid = sha256(output) == expected; } catch (const Error&) { valid = false; }
    if (valid) return;
    std::error_code error;
    fs::remove(output, error);
  }

  std::string engine;
  if (!container_engine(engine)) {
    std::cout << "Deckard: the ONNX Runtime model is unavailable and no container engine "
              << "(docker/podman) was found to generate it; using the Candle runtime.\n";
    return;
  }

  const auto& exporter = onnx.at("export");
  if (!exporter.contains("torch") || !exporter.contains("transformers") || !exporter.contains("onnx"))
    return;
  const std::string tag =
      std::string("deckard/onnx-export:torch-") + exporter.at("torch").get<std::string>() +
      "-transformers-" + exporter.at("transformers").get<std::string>() +
      "-onnx-" + exporter.at("onnx").get<std::string>();

  const fs::path stage = source_directory / (name + ".export-stage");
  struct RemoveStage {
    fs::path path;
    ~RemoveStage() { if (!path.empty()) { std::error_code error; fs::remove_all(path, error); } }
  } cleanup{stage};
  try {
    fs::create_directories(stage);
    write_text(stage / "export.py", onnx_export_py);
    write_text(stage / "Dockerfile", onnx_export_dockerfile);
    if (run({engine, "image", "inspect", tag}) != 0) {
      std::cout << "Deckard: preparing the pinned ONNX exporter image (one time) ..." << std::endl;
      if (run({engine, "build", "--quiet", "-t", tag, "-f", (stage / "Dockerfile").string(),
               stage.string()}) != 0) {
        std::cout << "Deckard: could not build the ONNX exporter image; using the Candle runtime.\n";
        return;
      }
    }
    std::cout << "Deckard: exporting the pinned ONNX gradient model ..." << std::endl;
    const std::string user = std::to_string(::getuid()) + ":" + std::to_string(::getgid());
    std::vector<std::string> command = {engine, "run", "--rm", "--user", user};
    // Rootless podman maps --user into a subordinate uid range that cannot
    // write the bind-mounted output directory; keep-id maps it to the caller.
    if (fs::path(engine).filename() == "podman") command.push_back("--userns=keep-id");
    // The image's ENTRYPOINT is already `python /export/export.py`.
    command.insert(command.end(), {
        "-v", source_directory.string() + ":/checkpoint:ro",
        "-v", source_directory.string() + ":/out",
        tag, "--checkpoint", "/checkpoint", "--output", "/out/" + name, "--force"});
    const int status = run(command);
    if (status != 0 || !fs::is_regular_file(output) || sha256(output) != expected) {
      std::error_code error;
      fs::remove(output, error);
      std::cout << "Deckard: the ONNX export did not reproduce the pinned artifact; using the Candle runtime.\n";
      return;
    }
    std::cout << "Deckard: prepared " << output.string() << " (sha256 " << expected << ").\n";
  } catch (const Error& error) {
    std::cout << "Deckard: ONNX model export skipped (" << error.what() << "); using the Candle runtime.\n";
  }
}

}  // namespace aihider
#endif