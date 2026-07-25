// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
#include "md/profile.hpp"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>

namespace md::prof {

namespace {

constexpr std::array<const char*, zone_count> zone_names{
    "cooldowns",  "crosshair",  "fire",        "interceptors", "blasts",
    "explosions", "smartbombs", "movethreats", "splitmirvs",   "blasthits",
    "groundhits", "waves",      "bonuscities", "termination",
};

#ifdef MD_PROFILE
struct Counters {
    std::array<std::uint64_t, zone_count> nanos{};
    std::array<std::uint64_t, zone_count> calls{};
};

// Thread-local so parallel workers never contend or corrupt each other's totals.
thread_local Counters counters{};
#endif

} // namespace

const char* name(Zone zone) noexcept {
    const auto index = static_cast<std::size_t>(zone);
    return index < zone_count ? zone_names[index] : "?";
}

#ifdef MD_PROFILE

void accumulate(Zone zone, std::uint64_t nanos_elapsed) noexcept {
    const auto index = static_cast<std::size_t>(zone);
    if (index < zone_count) {
        counters.nanos[index] += nanos_elapsed;
        counters.calls[index] += 1u;
    }
}

Scope::Scope(Zone zone) noexcept
    : zone_{zone}, start_{static_cast<std::uint64_t>(
                       std::chrono::steady_clock::now().time_since_epoch().count())} {}

Scope::~Scope() noexcept {
    const auto now =
        static_cast<std::uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
    accumulate(zone_, now - start_);
}

std::uint64_t nanos(Zone zone) noexcept {
    const auto index = static_cast<std::size_t>(zone);
    return index < zone_count ? counters.nanos[index] : 0u;
}

std::uint64_t calls(Zone zone) noexcept {
    const auto index = static_cast<std::size_t>(zone);
    return index < zone_count ? counters.calls[index] : 0u;
}

void reset() noexcept {
    counters = Counters{};
}

#else

std::uint64_t nanos(Zone /*zone*/) noexcept {
    return 0u;
}

std::uint64_t calls(Zone /*zone*/) noexcept {
    return 0u;
}

void reset() noexcept {}

#endif

} // namespace md::prof
