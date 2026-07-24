#include "md/action.hpp"
#include "md/config.hpp"
#include "md/sim.hpp"
#include "md/vec2.hpp"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cstdint>
#include <type_traits>

using Catch::Matchers::WithinAbs;
using md::Action;
using md::BaseId;
using md::Config;
using md::Sim;
using md::Vec2;

TEST_CASE("Sim whole-state is trivially copyable (snapshot == memcpy)", "[unit][sim]") {
    STATIC_REQUIRE(std::is_trivially_copyable_v<Sim>);
}

TEST_CASE("reset lays out six cities and three full bases on the ground", "[unit][sim]") {
    Sim sim;
    sim.reset(123);

    REQUIRE(sim.cities().size() == md::max_cities);
    REQUIRE(sim.bases().size() == md::base_count);
    REQUIRE(sim.threats().empty());
    REQUIRE(sim.interceptors().empty());
    REQUIRE(sim.blasts().empty());
    REQUIRE(sim.score() == 0);
    REQUIRE(sim.tick() == 0u);
    REQUIRE(sim.wave() == 1u);
    REQUIRE_FALSE(sim.terminated());

    for (const auto& c : sim.cities()) {
        REQUIRE(c.alive);
        REQUIRE(c.pos.y == 0.0f);
        REQUIRE(c.pos.x > 0.0f);
        REQUIRE(c.pos.x < sim.config().world_width);
    }
    for (const auto& b : sim.bases()) {
        REQUIRE(b.alive);
        REQUIRE(b.ammo == sim.config().ammo_per_base);
    }
}

TEST_CASE("bases are ordered ALPHA < DELTA < OMEGA across the field", "[unit][sim]") {
    Sim sim;
    sim.reset(7);
    REQUIRE(sim.bases()[0].pos.x < sim.bases()[1].pos.x);
    REQUIRE(sim.bases()[1].pos.x < sim.bases()[2].pos.x);
}

TEST_CASE("a Sim is a value: copying snapshots its state", "[unit][sim]") {
    Sim sim;
    sim.reset(1);
    const Sim snapshot = sim;
    REQUIRE(snapshot.score() == sim.score());
    REQUIRE(snapshot.tick() == sim.tick());
    REQUIRE(snapshot.cities().size() == sim.cities().size());
}

TEST_CASE("step advances the tick", "[unit][sim]") {
    Sim sim;
    sim.reset(0);
    const auto result = sim.step(md::Action::noop());
    REQUIRE(sim.tick() == 1u);
    REQUIRE(result.reward == 0);
    REQUIRE_FALSE(result.terminated);
}

TEST_CASE("Fire spawns an interceptor, consumes ammo, and starts cooldown", "[unit][sim]") {
    Sim sim;
    sim.reset(0);
    const std::uint32_t ammo_before = sim.bases()[0].ammo;

    sim.step(Action::fire(BaseId::Alpha, Vec2{100.0f, 90.0f}));

    REQUIRE(sim.interceptors().size() == 1);
    REQUIRE(sim.bases()[0].ammo == ammo_before - 1);
    REQUIRE(sim.bases()[0].cooldown_remaining > 0.0f);
}

TEST_CASE("A base with no ammo cannot fire", "[unit][sim]") {
    Config cfg;
    cfg.ammo_per_base = 1;
    cfg.base_cooldown = 0.0f;
    Sim sim{cfg};
    sim.reset(0);

    sim.step(Action::fire(BaseId::Delta, Vec2{160.0f, 90.0f})); // uses the only round
    REQUIRE(sim.bases()[1].ammo == 0u);

    sim.step(Action::fire(BaseId::Delta, Vec2{160.0f, 90.0f})); // rejected: empty
    REQUIRE(sim.bases()[1].ammo == 0u);
    REQUIRE(sim.interceptors().size() == 1); // no second interceptor spawned
}

TEST_CASE("A base respects its cooldown between shots", "[unit][sim]") {
    Config cfg;
    cfg.base_cooldown = 0.5f;
    Sim sim{cfg};
    sim.reset(0);

    sim.step(Action::fire(BaseId::Alpha, Vec2{50.0f, 120.0f}));
    const std::uint32_t ammo_after_first = sim.bases()[0].ammo;

    // Immediately firing again is rejected while on cooldown.
    sim.step(Action::fire(BaseId::Alpha, Vec2{50.0f, 120.0f}));
    REQUIRE(sim.bases()[0].ammo == ammo_after_first);

    // After the cooldown elapses (~0.5 s = 30 ticks) a shot succeeds again.
    for (int i = 0; i < 40; ++i) {
        sim.step(Action::noop());
    }
    const std::uint32_t ammo_before_third = sim.bases()[0].ammo;
    sim.step(Action::fire(BaseId::Alpha, Vec2{50.0f, 120.0f}));
    REQUIRE(sim.bases()[0].ammo == ammo_before_third - 1);
}

TEST_CASE("An interceptor reaches its target and detonates into a blast", "[unit][sim]") {
    Config cfg;
    cfg.base_cooldown = 0.0f;
    Sim sim{cfg};
    sim.reset(0);

    const Vec2 target{sim.bases()[0].pos.x, 20.0f}; // straight up, close
    sim.step(Action::fire(BaseId::Alpha, target));
    REQUIRE(sim.interceptors().size() == 1);

    bool detonated = false;
    for (int i = 0; i < 30 && !detonated; ++i) {
        sim.step(Action::noop());
        detonated = !sim.blasts().empty();
    }
    REQUIRE(detonated);
    REQUIRE(sim.interceptors().empty()); // consumed on detonation
}

TEST_CASE("A blast expands to full radius then expires", "[unit][sim]") {
    Config cfg;
    cfg.base_cooldown = 0.0f;
    cfg.blast_lifetime = 0.2f;
    Sim sim{cfg};
    sim.reset(0);

    sim.step(Action::fire(BaseId::Alpha, Vec2{sim.bases()[0].pos.x, 5.0f}));

    float max_radius = 0.0f;
    for (int i = 0; i < 40; ++i) {
        sim.step(Action::noop());
        if (!sim.blasts().empty()) {
            max_radius = std::max(max_radius, sim.blasts()[0].radius);
        }
    }
    REQUIRE_THAT(static_cast<double>(max_radius),
                 WithinAbs(static_cast<double>(cfg.blast_max_radius), 1e-4));
    REQUIRE(sim.blasts().empty()); // expired and removed
}
