// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
#include "md/version.hpp"

// Defined by the build from the top-level project(VERSION ...); see core/CMakeLists.txt.
#ifndef MD_VERSION
#define MD_VERSION "0.0.0"
#endif

namespace md {

std::string_view version() noexcept {
    return MD_VERSION;
}

} // namespace md
