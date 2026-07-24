#include "md/action.hpp"
#include "md/config.hpp"
#include "md/sim.hpp"

#include <catch2/catch_test_macros.hpp>
#include <type_traits>

using md::Sim;

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
