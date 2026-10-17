#pragma once

#include "support.hpp"

namespace aihider {

// The report of `deckard check-system` (Linux): which models are in the model
// cache, which runtime and accelerators this host can use, and, for anything
// missing, a problem and how to fix it. Keys are described in docs/LINUX.md.
Json system_check();

}  // namespace aihider
