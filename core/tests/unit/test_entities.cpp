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

TEST_CASE("Entities are trivially copyable so a snapshot is a memcpy", "[unit][entities]") {
    STATIC_REQUIRE(std::is_trivially_copyable_v<City>);
    STATIC_REQUIRE(std::is_trivially_copyable_v<Base>);
    STATIC_REQUIRE(std::is_trivially_copyable_v<Interceptor>);
    STATIC_REQUIRE(std::is_trivially_copyable_v<Threat>);
    STATIC_REQUIRE(std::is_trivially_copyable_v<Blast>);
}

TEST_CASE("Entity defaults are the expected inactive/alive state", "[unit][entities]") {
    REQUIRE(City{}.alive);
    REQUIRE(Base{}.ammo == 0u);
    REQUIRE_FALSE(Interceptor{}.active);
    REQUIRE_FALSE(Threat{}.active);
    REQUIRE_FALSE(Blast{}.active);
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
}
