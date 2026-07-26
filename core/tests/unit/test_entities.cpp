// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
#include "md/config.hpp"
#include "md/entities.hpp"

#include <catch2/catch_test_macros.hpp>
#include <type_traits>

using md::Base;
using md::Blast;
using md::City;
using md::Config;
using md::Interceptor;
using md::Threat;
using md::Vec2;

TEST_CASE("Entities are trivially copyable so a snapshot is a memcpy", "[unit][entities]") {
    STATIC_REQUIRE(std::is_trivially_copyable_v<City>);
    STATIC_REQUIRE(std::is_trivially_copyable_v<Base>);
    STATIC_REQUIRE(std::is_trivially_copyable_v<Interceptor>);
    STATIC_REQUIRE(std::is_trivially_copyable_v<Threat>);
    STATIC_REQUIRE(std::is_trivially_copyable_v<Blast>);
}

TEST_CASE("Entity defaults are the expected alive/empty state", "[unit][entities]") {
    REQUIRE(City{}.alive);
    REQUIRE(Base{}.ammo == 0u);
    // The transient pools carry no `active` flag: Sim keeps them compacted and
    // hands out counted spans, so everything a caller can see is live.
    REQUIRE(Threat{}.pos == Vec2{});
    REQUIRE(Interceptor{}.target == Vec2{});
    REQUIRE(Blast{}.radius == 0.0f);
}

TEST_CASE("Config defaults are sane", "[unit][entities]") {
    STATIC_REQUIRE(md::base_count == 3u);
    STATIC_REQUIRE(md::max_cities == 6u);

    constexpr Config c{};
    REQUIRE(c.world_width > 0.0f);
    REQUIRE(c.world_height > 0.0f);
    REQUIRE(c.dt > 0.0f);
    REQUIRE(c.interceptor_speed > 0.0f);
    REQUIRE(c.blast_max_radius > 0.0f);
    REQUIRE(c.ammo_per_base > 0u);
    REQUIRE(c.decision_interval == 4u);
}
