// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
#pragma once

#include <string_view>

namespace md {

/// Semantic version of the missile-defense core simulation library.
[[nodiscard]] std::string_view version() noexcept;

} // namespace md
