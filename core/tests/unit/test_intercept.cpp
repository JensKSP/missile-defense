// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
#include "md/action.hpp"
#include "md/config.hpp"
#include "md/entities.hpp"
#include "md/intercept.hpp"
#include "md/observation.hpp"
#include "md/sim.hpp"
#include "md/vec2.hpp"

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cstddef>
#include <cstdint>

using Catch::Matchers::WithinAbs;
using md::Action;
using md::BaseId;
using md::Config;
using md::Intercept;
using md::ObsSpec;
using md::Sim;
using md::Threat;
using md::Vec2;

namespace {

/// A synthetic descending threat, so the geometry under test is exact rather than
/// whatever the wave generator happened to spawn.
Threat falling(Vec2 pos, float speed) {
    Threat t{};
    t.pos = pos;
    t.origin = pos;
    t.velocity = Vec2{0.0f, -speed};
    return t;
}

} // namespace

TEST_CASE("With instant aim the solve is the classic lead-intercept", "[unit][intercept]") {
    Config cfg;
    cfg.aim_max_speed = 0.0f; // no cursor travel: flight starts immediately
    Sim sim{cfg};
    sim.reset(0);

    const Threat t = falling(Vec2{160.0f, 120.0f}, 30.0f);
    const Intercept plan = md::solve_intercept(sim, BaseId::Delta, t);

    REQUIRE(plan.feasible);
    REQUIRE(plan.aim_time == 0.0f);
    REQUIRE_THAT(static_cast<double>(plan.total_time),
                 WithinAbs(static_cast<double>(plan.fly_time), 1e-4));
    // Leading a descending threat means aiming *below* where it is now.
    REQUIRE(plan.point.y < t.pos.y);
    REQUIRE_THAT(static_cast<double>(plan.point.x), WithinAbs(160.0, 1e-4)); // falls straight down
}

TEST_CASE("The solved point is exactly where the threat will be", "[unit][intercept]") {
    Config cfg;
    cfg.aim_max_speed = 0.0f;
    Sim sim{cfg};
    sim.reset(0);

    const Threat t = falling(Vec2{100.0f, 140.0f}, 40.0f);
    const Intercept plan = md::solve_intercept(sim, BaseId::Alpha, t);
    REQUIRE(plan.feasible);

    const Vec2 where = t.pos + (t.velocity * plan.total_time);
    REQUIRE_THAT(static_cast<double>(plan.point.y), WithinAbs(static_cast<double>(where.y), 1e-3));
}

TEST_CASE("Cursor travel time pushes the intercept later and lower", "[unit][intercept]") {
    const Threat t = falling(Vec2{300.0f, 150.0f}, 40.0f); // far from the centred crosshair

    Config instant;
    instant.aim_max_speed = 0.0f;
    Sim fast{instant};
    fast.reset(0);
    const Intercept quick = md::solve_intercept(fast, BaseId::Alpha, t);

    Config slow_cfg;
    slow_cfg.aim_max_speed = 80.0f; // a sluggish cursor: travel dominates
    Sim slow{slow_cfg};
    slow.reset(0);
    const Intercept late = md::solve_intercept(slow, BaseId::Alpha, t);

    REQUIRE(quick.feasible);
    REQUIRE(late.aim_time > 0.0f);
    REQUIRE(late.total_time > quick.total_time); // waiting for the cursor costs time
    REQUIRE(late.point.y < quick.point.y);       // so the threat is lower when hit
}

TEST_CASE("A threat already at the ground is infeasible", "[unit][intercept]") {
    Sim sim;
    sim.reset(0);
    const Threat t = falling(Vec2{160.0f, 0.2f}, 60.0f); // lands within a few ticks
    REQUIRE_FALSE(md::solve_intercept(sim, BaseId::Delta, t).feasible);
}

TEST_CASE("engage steers toward the target, then fires on arrival", "[unit][intercept]") {
    Config cfg;
    cfg.aim_max_speed = 200.0f; // slow enough that arrival takes several ticks
    cfg.wave_base_threats = 1;
    cfg.threat_base_speed = 10.0f; // lingers, so the plan stays feasible
    Sim sim{cfg};
    sim.reset(42);
    sim.step(Action::noop()); // spawn one threat
    REQUIRE(sim.threats().size() == 1);

    const Action first = md::engage(sim, BaseId::Delta, 0);
    REQUIRE(first.move);
    REQUIRE_FALSE(first.fire); // still travelling: do not shoot from the wrong place

    bool fired = false;
    for (int i = 0; i < 200 && !fired; ++i) {
        const Action a = md::engage(sim, BaseId::Delta, 0);
        fired = a.fire;
        sim.step(a);
    }
    REQUIRE(fired);
    REQUIRE(sim.interceptors().size() + sim.blasts().size() >= 1); // the shot went out
}

TEST_CASE("engage on an empty slot is a NoOp", "[unit][intercept]") {
    Sim sim;
    sim.reset(0);
    REQUIRE(sim.threats().empty());
    const Action a = md::engage(sim, BaseId::Alpha, 0);
    REQUIRE_FALSE(a.fire);
    REQUIRE_FALSE(a.move);
}

TEST_CASE("The discrete action space covers NoOp plus every battery-threat pair",
          "[unit][intercept]") {
    constexpr ObsSpec spec;
    STATIC_REQUIRE(md::action_count(spec) == 1u + (md::base_count * spec.threats));
}

TEST_CASE("decode_action maps index 0 to NoOp and the rest to engagements", "[unit][intercept]") {
    Config cfg;
    cfg.wave_base_threats = 1;
    Sim sim{cfg};
    sim.reset(42);
    sim.step(Action::noop());
    REQUIRE(sim.threats().size() == 1);

    constexpr ObsSpec spec;
    const Action noop = md::decode_action(sim, spec, 0);
    REQUIRE_FALSE(noop.fire);
    REQUIRE_FALSE(noop.move);

    // Index 1 == (battery 0, threat slot 0): the same thing engage() produces.
    const Action first = md::decode_action(sim, spec, 1);
    const Action direct = md::engage(sim, BaseId::Alpha, 0);
    REQUIRE(first.move == direct.move);
    REQUIRE(first.fire == direct.fire);
    REQUIRE(first.aim == direct.aim);

    // Out of range falls back to NoOp rather than reading past the end.
    REQUIRE_FALSE(md::decode_action(sim, spec, md::action_count(spec)).fire);
}

TEST_CASE("The action mask hides empty slots and unusable batteries", "[unit][intercept]") {
    Config cfg;
    cfg.wave_base_threats = 1;
    cfg.ammo_per_base = 0; // every battery is dry
    Sim sim{cfg};
    sim.reset(42);
    sim.step(Action::noop());

    constexpr ObsSpec spec;
    static std::array<bool, md::action_count(spec)> mask{};
    md::action_mask(sim, spec, mask);

    REQUIRE(mask[0]); // NoOp is always legal
    for (std::size_t i = 1; i < mask.size(); ++i) {
        REQUIRE_FALSE(mask[i]); // no ammo anywhere -> nothing to engage with
    }
}

TEST_CASE("The action mask keeps live threats engageable while cooling down", "[unit][intercept]") {
    Config cfg;
    cfg.wave_base_threats = 1;
    cfg.aim_max_speed = 0.0f;
    Sim sim{cfg};
    sim.reset(42);
    sim.step(Action::noop());
    REQUIRE(sim.threats().size() == 1);

    // Fire, putting both the battery and the trigger on cooldown.
    sim.step(Action::fire_at(BaseId::Alpha, sim.threats()[0].pos));
    REQUIRE(sim.fire_cooldown() > 0.0f);

    constexpr ObsSpec spec;
    static std::array<bool, md::action_count(spec)> mask{};
    md::action_mask(sim, spec, mask);

    // Steering toward a target is useful work even while the trigger recovers, so
    // cooldown must not mask the action out.
    REQUIRE(mask[1]);
}
