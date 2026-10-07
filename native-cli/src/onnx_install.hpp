#pragma once
#include <filesystem>

namespace aihider {

#if defined(__linux__)
// Generates the pinned model.onnx from the local checkpoint and verifies its
// pinned sha256, using a pinned PyTorch exporter image in the host container
// engine. Never throws for missing engines or failed exports: without the
// artifact the install falls back to the Candle runtime, which keeps installs
// fully automatic and self-contained. Vendor GPU drivers are expected on the
// system (see `deckard check-system`), never installed by Deckard.
void prepare_onnx_model(const std::filesystem::path& source_directory);
#endif

}  // namespace aihider