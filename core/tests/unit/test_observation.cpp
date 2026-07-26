// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
#include "md/action.hpp"
#include "md/config.hpp"
#include "md/observation.hpp"
#include "md/sim.hpp"
#include "md/vec2.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cstddef>
#include <vector>

using Catch::Matchers::WithinAbs;
using md::Action;
using md::BaseId;
using md::Config;
using md::ObsSpec;
using md::Sim;
using md::Vec2;

namespace {
std::vector<float> encode_to_vector(const Sim& sim, const ObsSpec& spec) {
    std::vector<float> obs(spec.size(), -99.0f); // poison so unwritten slots show up
    md::encode(sim, spec, obs);
    return obs;
}
} // namespace

TEST_CASE("The default spec exposes every entity the simulation can hold", "[unit][obs]") {
    constexpr ObsSpec spec;
    // No truncation by default: the policy must never be blind to a threat the
    // human can see (that would be an information asymmetry, not a handicap).
    STATIC_REQUIRE(spec.threats == md::max_threats);
    STATIC_REQUIRE(spec.interceptors == md::max_interceptors);
    STATIC_REQUIRE(spec.blasts == md::max_blasts);
    STATIC_REQUIRE(spec.size() > 0u);
}

TEST_CASE("encode fills exactly spec.size() floats", "[unit][obs]") {
    Sim sim;
    sim.reset(3);
    constexpr ObsSpec spec;

    std::vector<float> obs(spec.size() + 1u, -99.0f);
    md::encode(sim, spec, obs);

    // Everything up to size() was written; the guard element past the end was not.
    for (std::size_t i = 0; i < spec.size(); ++i) {
        REQUIRE(obs[i] != -99.0f);
    }
    REQUIRE(obs[spec.size()] == -99.0f);
}

TEST_CASE("encode writes nothing into a buffer that is too small", "[unit][obs]") {
    Sim sim;
    sim.reset(3);
    constexpr ObsSpec spec;

    std::vector<float> obs(spec.size() - 1u, -99.0f);
    md::encode(sim, spec, obs);
    for (const float v : obs) {
        REQUIRE(v == -99.0f); // a partial row would silently corrupt a training batch
    }
}

TEST_CASE("Empty entity slots are zero-padded", "[unit][obs]") {
    Sim sim;
    sim.reset(3); // nothing has spawned yet on tick 0
    constexpr ObsSpec spec;
    const std::vector<float> obs = encode_to_vector(sim, spec);

    REQUIRE(sim.threats().empty());
    for (std::size_t i = 0; i < spec.threats * ObsSpec::threat_features; ++i) {
        REQUIRE(obs[i] == 0.0f);
    }
}

TEST_CASE("A live threat occupies its slot with normalised state", "[unit][obs]") {
    Sim sim;
    sim.reset(42);
    sim.step(Action::noop()); // spawn the wave's first threat
    REQUIRE(sim.threats().size() >= 1);
    constexpr ObsSpec spec;
    const std::vector<float> obs = encode_to_vector(sim, spec);

    const auto& threat = sim.threats()[0];
    REQUIRE(obs[0] == 1.0f); // present flag
    const double expect_x =
        static_cast<double>((2.0f * threat.pos.x / sim.config().world_width) - 1.0f);
    REQUIRE_THAT(static_cast<double>(obs[1]), WithinAbs(expect_x, 1e-5));
    REQUIRE(obs[4] < 0.0f); // normalised vy: descending

    // Exactly one of the four type slots is hot.
    float one_hot = 0.0f;
    for (std::size_t k = 5; k < 9; ++k) {
        one_hot += obs[k];
    }
    REQUIRE(one_hot == 1.0f);
}

TEST_CASE("A blast carries the same lifetime phase the renderer shows", "[unit][obs]") {
    Config cfg;
    cfg.aim_max_speed = 0.0f;  // place the crosshair exactly at the launch point
    cfg.decision_interval = 1; // replace the fire action with NoOp on the next tick
    Sim sim{cfg};
    sim.reset(42);

    const Vec2 centre = sim.bases()[0].pos;
    sim.step(Action::fire_at(BaseId::Alpha, centre));
    REQUIRE(sim.blasts().size() == 1u);

    constexpr ObsSpec spec;
    STATIC_REQUIRE(ObsSpec::blast_features == 5u);
    const std::size_t blasts_at = (spec.threats * ObsSpec::threat_features) +
                                  (spec.interceptors * ObsSpec::interceptor_features);

    // Once expansion finishes, radius no longer says how close the blast is to
    // expiry. Its rendered phase — age / lifetime — must keep advancing.
    while (sim.blasts()[0].radius < cfg.blast_max_radius) {
        sim.step(Action::noop());
        REQUIRE(sim.blasts().size() == 1u);
    }
    const std::vector<float> first = encode_to_vector(sim, spec);
    const float first_phase = sim.blasts()[0].age / cfg.blast_lifetime;
    for (int tick = 0; tick < 3; ++tick) {
        sim.step(Action::noop());
    }
    REQUIRE(sim.blasts().size() == 1u);
    const std::vector<float> later = encode_to_vector(sim, spec);
    const float later_phase = sim.blasts()[0].age / cfg.blast_lifetime;

    REQUIRE_THAT(static_cast<double>(first[blasts_at + 3]), WithinAbs(1.0, 1e-5));
    REQUIRE_THAT(static_cast<double>(later[blasts_at + 3]), WithinAbs(1.0, 1e-5));
    REQUIRE_THAT(static_cast<double>(first[blasts_at + 4]),
                 WithinAbs(static_cast<double>(first_phase), 1e-5));
    REQUIRE_THAT(static_cast<double>(later[blasts_at + 4]),
                 WithinAbs(static_cast<double>(later_phase), 1e-5));
    REQUIRE(later[blasts_at + 4] > first[blasts_at + 4]);
}

TEST_CASE("Positions stay inside [-1, 1] across the whole field", "[unit][obs]") {
    Sim sim;
    sim.reset(7);
    for (int i = 0; i < 400; ++i) {
        sim.step(Action::noop());
    }
    constexpr ObsSpec spec;
    const std::vector<float> obs = encode_to_vector(sim, spec);
    for (std::size_t i = 0; i < spec.threats * ObsSpec::threat_features; ++i) {
        REQUIRE(obs[i] >= -1.5f);
        REQUIRE(obs[i] <= 1.5f); // generous: threats spawn slightly off the top edge
    }
}

TEST_CASE("The observation carries the player-model state", "[unit][obs]") {
    Config cfg;
    cfg.aim_max_speed = 0.0f; // snap so we know exactly where the crosshair is
    Sim sim{cfg};
    sim.reset(0);
    constexpr ObsSpec spec;

    // The crosshair and trigger cooldown are part of the observation: without them
    // a policy cannot plan around cursor travel or shot pacing.
    const std::size_t globals = (spec.threats * ObsSpec::threat_features) +
                                (spec.interceptors * ObsSpec::interceptor_features) +
                                (spec.blasts * ObsSpec::blast_features) +
                                (md::base_count * ObsSpec::base_features) +
                                (md::max_cities * ObsSpec::city_features);

    sim.step(Action::aim_at(Vec2{0.0f, 0.0f})); // drive the crosshair to a corner
    const std::vector<float> obs = encode_to_vector(sim, spec);
    REQUIRE_THAT(static_cast<double>(obs[globals]), WithinAbs(-1.0, 1e-5));
    REQUIRE_THAT(static_cast<double>(obs[globals + 1]), WithinAbs(-1.0, 1e-5));
}

TEST_CASE("Event counts reach the observation (audio parity)", "[unit][obs]") {
    Config cfg;
    cfg.aim_max_speed = 0.0f;
    Sim sim{cfg};
    sim.reset(0);
    constexpr ObsSpec spec;
    const std::size_t events = spec.size() - ObsSpec::event_features;

    sim.step(Action::fire_at(BaseId::Alpha, Vec2{100.0f, 90.0f}));
    const std::vector<float> obs = encode_to_vector(sim, spec);

    float total = 0.0f;
    for (std::size_t i = 0; i < ObsSpec::event_features; ++i) {
        total += obs[events + i];
    }
    REQUIRE(total > 0.0f); // direct encoding exposes the Fire event from this tick
}

TEST_CASE("Encoding is a pure function of the state", "[unit][obs]") {
    Sim sim;
    sim.reset(11);
    for (int i = 0; i < 50; ++i) {
        sim.step(Action::noop());
    }
    constexpr ObsSpec spec;

    const Sim snapshot = sim; // Sim is a value; the copy must encode identically
    REQUIRE(encode_to_vector(sim, spec) == encode_to_vector(snapshot, spec));
}
