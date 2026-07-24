#include "md/vec2.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using Catch::Matchers::WithinAbs;
using md::Vec2;

TEST_CASE("Vec2 arithmetic is constexpr and correct", "[unit][vec2]") {
    constexpr Vec2 a{3.0f, 4.0f};
    constexpr Vec2 b{1.0f, 2.0f};

    STATIC_REQUIRE((a + b) == Vec2{4.0f, 6.0f});
    STATIC_REQUIRE((a - b) == Vec2{2.0f, 2.0f});
    STATIC_REQUIRE((a * 2.0f) == Vec2{6.0f, 8.0f});
    STATIC_REQUIRE((2.0f * a) == Vec2{6.0f, 8.0f});
    STATIC_REQUIRE((-a) == Vec2{-3.0f, -4.0f});
    STATIC_REQUIRE(dot(a, b) == 11.0f);
    STATIC_REQUIRE(a.length_sq() == 25.0f);
}

TEST_CASE("Vec2 compound assignment mutates in place", "[unit][vec2]") {
    Vec2 v{1.0f, 1.0f};
    v += Vec2{2.0f, 3.0f};
    REQUIRE(v == Vec2{3.0f, 4.0f});
    v -= Vec2{1.0f, 1.0f};
    REQUIRE(v == Vec2{2.0f, 3.0f});
    v *= 2.0f;
    REQUIRE(v == Vec2{4.0f, 6.0f});
}

TEST_CASE("Vec2 length and normalization", "[unit][vec2]") {
    const Vec2 a{3.0f, 4.0f};
    REQUIRE_THAT(static_cast<double>(a.length()), WithinAbs(5.0, 1e-6));
    REQUIRE_THAT(static_cast<double>(distance(Vec2{0.0f, 0.0f}, a)), WithinAbs(5.0, 1e-6));

    const Vec2 n = a.normalized();
    REQUIRE_THAT(static_cast<double>(n.length()), WithinAbs(1.0, 1e-6));

    // The zero vector normalizes to zero rather than producing NaN.
    REQUIRE(Vec2{}.normalized() == Vec2{});
}
