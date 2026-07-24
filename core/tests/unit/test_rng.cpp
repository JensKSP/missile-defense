// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
#include "md/rng.hpp"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>

using md::Pcg32;

TEST_CASE("Pcg32 is deterministic for a given seed", "[unit][rng]") {
    Pcg32 a{42};
    Pcg32 b{42};
    for (int i = 0; i < 1000; ++i) {
        REQUIRE(a.next_u32() == b.next_u32());
    }
}

TEST_CASE("Pcg32 produces different streams for different seeds", "[unit][rng]") {
    Pcg32 a{1};
    Pcg32 b{2};
    bool diverged = false;
    for (int i = 0; i < 100; ++i) {
        if (a.next_u32() != b.next_u32()) {
            diverged = true;
            break;
        }
    }
    REQUIRE(diverged);
}

TEST_CASE("Pcg32 next_float stays within the unit interval", "[unit][rng]") {
    Pcg32 r{123};
    for (int i = 0; i < 10000; ++i) {
        const float f = r.next_float();
        REQUIRE(f >= 0.0f);
        REQUIRE(f < 1.0f);
    }
}

TEST_CASE("Pcg32::below is in range and covers both endpoints", "[unit][rng]") {
    Pcg32 r{7};
    std::uint32_t lo = 0xffffffffu;
    std::uint32_t hi = 0u;
    for (int i = 0; i < 100000; ++i) {
        const std::uint32_t v = r.below(6);
        REQUIRE(v < 6u);
        lo = std::min(lo, v);
        hi = std::max(hi, v);
    }
    REQUIRE(lo == 0u);
    REQUIRE(hi == 5u);
}

TEST_CASE("Pcg32 constexpr evaluation matches runtime", "[unit][rng]") {
    // Proves the generator is usable in constant expressions.
    constexpr std::uint32_t first = [] {
        Pcg32 r{2026};
        return r.next_u32();
    }();
    Pcg32 runtime{2026};
    REQUIRE(runtime.next_u32() == first);
}
