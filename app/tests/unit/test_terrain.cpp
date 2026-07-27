// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
//
// The landscape the cities stand on. It is decoration, so what it has to get
// right is not a number but a set of promises the drawing depends on: that
// every installation has level ground under its whole footprint, that the
// ground never rises far enough to swallow the skyline or the HUD above it, and
// that the surface is continuous — a seam between two pieces of the piecewise
// shape would draw as a cliff, and only a sampled test would ever notice.
#include "md/config.hpp"
#include "md/sim.hpp"
#include "terrain.hpp"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <vector>

namespace {

/// The landscape the game actually raises: built from a real `Sim`'s layout, so
/// a change to where the installations stand shows up here rather than as a city
/// hanging off the side of a hill.
struct Field {
    md::Sim sim{md::Config{}};
    std::vector<float> city_x;
    std::vector<float> base_x;
    md::Terrain terrain;

    Field() {
        for (const auto& city : sim.cities()) {
            city_x.push_back(city.pos.x);
        }
        for (const auto& base : sim.bases()) {
            base_x.push_back(base.pos.x);
        }
        terrain = md::Terrain{sim.config().world_width, city_x, base_x};
    }
};

/// The half-width the renderer draws each kind of installation at.
constexpr float city_half_width = 7.0f;
constexpr float base_half_width = 6.0f;

} // namespace

TEST_CASE("Every installation stands on level ground", "[unit][app][terrain]") {
    const Field field;

    // A row of towers is planted at one height. If the ground under a footprint
    // sloped, the buildings would either float or sink at one end.
    const auto level_across = [&](float centre, float half_w) {
        const float expected = field.terrain.height(centre);
        for (int i = -20; i <= 20; ++i) {
            const float x = centre + (half_w * static_cast<float>(i) / 20.0f);
            REQUIRE(field.terrain.height(x) == expected);
        }
    };

    for (const float x : field.city_x) {
        level_across(x, city_half_width);
    }
    for (const float x : field.base_x) {
        level_across(x, base_half_width);
    }
}

TEST_CASE("The plateaus are wider than the footprints they carry", "[unit][app][terrain]") {
    const Field field;
    // The check above samples the current footprints; this one states the margin
    // the layout has, so widening a city by a unit fails here with a reason
    // rather than there with a slope.
    CHECK(field.terrain.plateau_half_width() > city_half_width);
    CHECK(field.terrain.plateau_half_width() > base_half_width);
}

TEST_CASE("The batteries stand on higher ground than the towns", "[unit][app][terrain]") {
    const Field field;
    float highest_town = 0.0f;
    for (const float x : field.city_x) {
        highest_town = std::max(highest_town, field.terrain.height(x));
    }
    for (const float x : field.base_x) {
        // The arcade original puts its three launchers on mounds; so does this.
        CHECK(field.terrain.height(x) > highest_town);
    }
}

TEST_CASE("The ground stays between the world floor and the skyline", "[unit][app][terrain]") {
    const Field field;
    const float world_w = field.sim.config().world_width;

    float lowest = world_w;
    float highest = 0.0f;
    for (int i = 0; i <= 2000; ++i) {
        const float x = world_w * static_cast<float>(i) / 2000.0f;
        const float h = field.terrain.height(x);
        lowest = std::min(lowest, h);
        highest = std::max(highest, h);
    }

    CHECK(lowest > 0.0f);           // never a gap under the ground
    CHECK(highest < 10.0f);         // under the rooftops: a city is never in a pit
    CHECK(highest - lowest > 2.0f); // and a landscape, not the flat bar it replaced

    // The play-mode key hints have no scrim to hide behind — a live game is
    // underneath them — so the only thing keeping them readable is clear sky.
    // They sit at 0.115 of the world height in 0.008 glyphs, and a glyph hangs
    // 4.92*px below the top it is given.
    const float world_h = field.sim.config().world_height;
    const float lowest_hint = (world_h * 0.115f) - (4.92f * world_h * 0.008f);
    CHECK(highest < lowest_hint);
}

TEST_CASE("The surface is continuous", "[unit][app][terrain]") {
    const Field field;
    const float world_w = field.sim.config().world_width;
    // Every seam in the piecewise shape — plateau to hill, hill to plateau,
    // plateau to shoulder — is crossed by this sweep. A step at any of them
    // would draw as a cliff face the width of one column.
    constexpr int samples = 20000;
    const float step = world_w / static_cast<float>(samples);
    float previous = field.terrain.height(0.0f);
    for (int i = 1; i <= samples; ++i) {
        const float h = field.terrain.height(step * static_cast<float>(i));
        REQUIRE(std::abs(h - previous) < step * 2.0f); // no slope steeper than 2:1
        previous = h;
    }
}

TEST_CASE("The same layout always raises the same landscape", "[unit][app][terrain]") {
    // A match draws two backdrops from one heightfield and the menus sit over it
    // between runs; a landscape that reshuffled would be visible in both.
    const Field a;
    const Field b;
    const float world_w = a.sim.config().world_width;
    for (int i = 0; i <= 500; ++i) {
        const float x = world_w * static_cast<float>(i) / 500.0f;
        REQUIRE(a.terrain.height(x) == b.terrain.height(x));
    }
}

TEST_CASE("The distant ridge stays behind the game", "[unit][app][terrain]") {
    const md::Config config;
    for (int i = 0; i <= 2000; ++i) {
        const float x = config.world_width * static_cast<float>(i) / 2000.0f;
        const float h = md::Terrain::ridge(x);
        CHECK(h > 0.0f);
        // Below the play-mode key hints, whose glyphs hang to about 13.6 — the
        // ridge is dim enough to read through, but it has no business up there.
        CHECK(h < 13.6f);
    }
}

TEST_CASE("A terrain with nowhere to stand is still flat ground", "[unit][app][terrain]") {
    // The default-constructed member the renderer holds before it is built.
    const md::Terrain empty;
    CHECK(empty.height(0.0f) > 0.0f);
    CHECK(empty.height(160.0f) == empty.height(0.0f));
}
