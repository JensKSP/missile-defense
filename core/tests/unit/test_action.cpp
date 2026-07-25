// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
#include "md/action.hpp"
#include "md/config.hpp"
#include "md/vec2.hpp"

#include <catch2/catch_test_macros.hpp>

using md::Action;
using md::BaseId;
using md::Vec2;

TEST_CASE("Action::noop neither steers nor fires", "[unit][action]") {
    constexpr Action n = Action::noop();
    STATIC_REQUIRE_FALSE(n.move); // the crosshair holds its position
    STATIC_REQUIRE_FALSE(n.fire);
}

TEST_CASE("Action::aim_at steers without firing", "[unit][action]") {
    constexpr Action a = Action::aim_at(Vec2{10.0f, 20.0f});
    STATIC_REQUIRE(a.move);
    STATIC_REQUIRE_FALSE(a.fire);
    STATIC_REQUIRE(a.aim == Vec2{10.0f, 20.0f});
}

TEST_CASE("Action::fire carries base and aim, and does both", "[unit][action]") {
    constexpr Action f = Action::fire_at(BaseId::Delta, Vec2{10.0f, 20.0f});
    STATIC_REQUIRE(f.move);
    STATIC_REQUIRE(f.fire);
    STATIC_REQUIRE(f.base == BaseId::Delta);
    STATIC_REQUIRE(f.aim == Vec2{10.0f, 20.0f});
}

TEST_CASE("Action::fire_here fires without moving the crosshair", "[unit][action]") {
    constexpr Action f = Action::fire_here(BaseId::Omega);
    STATIC_REQUIRE_FALSE(f.move);
    STATIC_REQUIRE(f.fire);
    STATIC_REQUIRE(f.base == BaseId::Omega);
}
