#include "md/action.hpp"
#include "md/config.hpp"
#include "md/vec2.hpp"

#include <catch2/catch_test_macros.hpp>

using md::Action;
using md::BaseId;
using md::Vec2;

TEST_CASE("Action::noop is the do-nothing primitive", "[unit][action]") {
    constexpr Action n = Action::noop();
    STATIC_REQUIRE(n.kind == Action::Kind::NoOp);
}

TEST_CASE("Action::fire carries base and target", "[unit][action]") {
    constexpr Action f = Action::fire(BaseId::Delta, Vec2{10.0f, 20.0f});
    STATIC_REQUIRE(f.kind == Action::Kind::Fire);
    STATIC_REQUIRE(f.base == BaseId::Delta);
    STATIC_REQUIRE(f.target == Vec2{10.0f, 20.0f});
}
