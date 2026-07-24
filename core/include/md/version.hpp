#pragma once

#include <string_view>

namespace md {

/// Semantic version of the missile-defense core simulation library.
[[nodiscard]] std::string_view version() noexcept;

} // namespace md
