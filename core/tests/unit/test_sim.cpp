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
using md::EventType;
using md::Sim;
using md::ThreatType;
using md::Vec2;

namespace {
bool has_event(const Sim& sim, EventType type) {
    for (const auto& event : sim.events()) {
        if (event.type == type) {
            return true;
        }
    }
    return false;
}
} // namespace

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

TEST_CASE("Threats spawn during a wave and descend from the top", "[unit][sim]") {
    Sim sim;
    sim.reset(42);
    sim.step(Action::noop()); // first threat spawns on tick 0 (spawn_timer == 0)

    REQUIRE(sim.threats().size() >= 1);
    for (const auto& threat : sim.threats()) {
        REQUIRE(threat.velocity.y < 0.0f); // heading downward
        REQUIRE(threat.pos.y <= sim.config().world_height);
    }
}

TEST_CASE("A blast destroys threats within its radius and scores", "[unit][sim]") {
    Config cfg;
    cfg.blast_max_radius = 40.0f;   // large, forgiving blast
    cfg.blast_lifetime = 3.0f;      // long-lived
    cfg.threat_base_speed = 5.0f;   // slow threats
    cfg.interceptor_speed = 400.0f; // fast interceptor
    Sim sim{cfg};
    sim.reset(1);

    sim.step(Action::noop()); // spawn a threat
    REQUIRE(sim.threats().size() >= 1);

    const std::int32_t score_before = sim.score();
    sim.step(Action::fire(BaseId::Delta, sim.threats()[0].pos));

    bool scored = false;
    for (int i = 0; i < 300 && !scored; ++i) {
        sim.step(Action::noop());
        scored = sim.score() > score_before;
    }
    REQUIRE(scored);
}

TEST_CASE("A threat reaching its city destroys it", "[unit][sim]") {
    Config cfg;
    cfg.threat_base_speed = 2000.0f; // crosses the field in a few ticks
    Sim sim{cfg};
    sim.reset(5);
    sim.step(Action::noop()); // spawn a threat heading to a city
    REQUIRE(sim.threats().size() >= 1);

    bool city_lost = false;
    for (int i = 0; i < 20 && !city_lost; ++i) {
        sim.step(Action::noop());
        for (const auto& city : sim.cities()) {
            city_lost = city_lost || !city.alive;
        }
    }
    REQUIRE(city_lost);
}

TEST_CASE("Threats can destroy bases, not just cities", "[unit][sim]") {
    Config cfg;
    cfg.threat_base_speed = 2500.0f;
    cfg.spawn_interval = 0.05f;
    cfg.wave_base_threats = 40;
    Sim sim{cfg};
    sim.reset(4);

    bool base_destroyed = false;
    for (int i = 0; i < 5000 && !base_destroyed; ++i) {
        sim.step(Action::noop());
        for (const auto& base : sim.bases()) {
            base_destroyed = base_destroyed || !base.alive;
        }
    }
    REQUIRE(base_destroyed);
}

TEST_CASE("MIRV threats split into multiple warheads", "[unit][sim]") {
    Config cfg;
    cfg.mirv_chance_per_wave = 1.0f; // force every wave-2+ spawn to be a MIRV
    Sim sim{cfg};
    sim.reset(2);

    // A split removes one parent and adds several children, so the active-threat
    // count jumps by >= 2 in a single step (a plain spawn only adds one).
    bool split_seen = false;
    bool split_event_seen = false;
    std::size_t prev = sim.threats().size();
    for (int i = 0; i < 8000 && !split_seen; ++i) {
        sim.step(Action::noop());
        const std::size_t now = sim.threats().size();
        split_seen = now >= prev + 2;
        if (split_seen) {
            split_event_seen = has_event(sim, EventType::MirvSplit);
        }
        prev = now;
    }
    REQUIRE(split_seen);
    REQUIRE(split_event_seen); // the split emits a MirvSplit event (for audio / AI)
}

TEST_CASE("Smart bombs spawn from the configured wave", "[unit][sim]") {
    Config cfg;
    cfg.smart_bomb_wave = 1;
    cfg.smart_bomb_chance = 1.0f;
    Sim sim{cfg};
    sim.reset(1);
    sim.step(Action::noop());

    REQUIRE(sim.threats().size() >= 1);
    bool has_smart = false;
    for (const auto& threat : sim.threats()) {
        has_smart = has_smart || (threat.type == ThreatType::SmartBomb);
    }
    REQUIRE(has_smart);
}

TEST_CASE("Smart bombs steer away from a nearby blast", "[unit][sim]") {
    Config cfg;
    cfg.smart_bomb_wave = 1;
    cfg.smart_bomb_chance = 1.0f;
    cfg.wave_base_threats = 1;     // a single smart bomb to track
    cfg.threat_base_speed = 20.0f; // slow, so it lingers
    cfg.blast_lifetime = 3.0f;     // long-lived blast
    cfg.blast_max_radius = 10.0f;
    cfg.interceptor_speed = 800.0f; // near-instant detonation
    cfg.base_cooldown = 0.0f;
    Sim sim{cfg};
    sim.reset(11);
    sim.step(Action::noop());
    REQUIRE(sim.threats().size() == 1);
    const Vec2 p = sim.threats()[0].pos;

    // Detonate a blast to the LEFT of the smart bomb (near, but out of kill range).
    sim.step(Action::fire(BaseId::Delta, Vec2{p.x - 18.0f, p.y}));

    float vx = 0.0f;
    for (int i = 0; i < 80; ++i) {
        sim.step(Action::noop());
        if (!sim.threats().empty()) {
            vx = sim.threats()[0].velocity.x;
        }
    }
    REQUIRE(vx > 0.0f); // steering right, away from the blast on its left
}

TEST_CASE("A destroyed city is rebuilt at the bonus-score threshold", "[unit][sim]") {
    Config cfg;
    cfg.bonus_city_score = 50; // low threshold for the test
    cfg.threat_base_speed = 2000.0f;
    cfg.spawn_interval = 0.1f;
    cfg.blast_max_radius = 50.0f; // big blast to rack up kills
    cfg.blast_lifetime = 3.0f;
    cfg.interceptor_speed = 1500.0f;
    cfg.base_cooldown = 0.0f;
    cfg.wave_base_threats = 80;
    Sim sim{cfg};
    sim.reset(6);

    const auto alive_cities = [&sim] {
        std::size_t n = 0;
        for (const auto& city : sim.cities()) {
            n += city.alive ? 1u : 0u;
        }
        return n;
    };

    // Phase 1: no defence — let at least one city fall.
    for (int i = 0; i < 40 && alive_cities() == sim.cities().size(); ++i) {
        sim.step(Action::noop());
    }
    REQUIRE(alive_cities() < sim.cities().size());

    // Phase 2: defend to earn points. A bonus city is the ONLY thing that can raise
    // the alive-city count (threats only destroy), so an increase proves the rebuild.
    bool rebuilt = false;
    std::size_t prev = alive_cities();
    for (int i = 0; i < 2000 && !rebuilt && !sim.terminated(); ++i) {
        if (!sim.threats().empty()) {
            sim.step(Action::fire(BaseId::Delta, sim.threats()[0].pos));
        } else {
            sim.step(Action::noop());
        }
        const std::size_t now = alive_cities();
        rebuilt = now > prev;
        prev = now;
    }
    REQUIRE(rebuilt);
}

TEST_CASE("Firing emits a Fire event; events are per-step", "[unit][sim]") {
    Sim sim;
    sim.reset(0);
    REQUIRE(sim.events().empty());

    sim.step(Action::fire(BaseId::Alpha, Vec2{100.0f, 90.0f}));
    REQUIRE(has_event(sim, EventType::Fire));

    sim.step(Action::noop()); // the next step clears the previous step's events
    REQUIRE_FALSE(has_event(sim, EventType::Fire));
}

TEST_CASE("Starting a wave emits a WaveStarted event (siren)", "[unit][sim]") {
    Sim sim;
    sim.reset(1);
    sim.step(Action::noop()); // wave 1 begins at game start -> siren on the first step
    REQUIRE(has_event(sim, EventType::WaveStarted));

    sim.step(Action::noop());
    REQUIRE_FALSE(has_event(sim, EventType::WaveStarted)); // only once, at the wave's start
}

TEST_CASE("A blast kill emits a ThreatKilled event", "[unit][sim]") {
    Config cfg;
    cfg.blast_max_radius = 40.0f;
    cfg.blast_lifetime = 3.0f;
    cfg.threat_base_speed = 5.0f;
    cfg.interceptor_speed = 400.0f;
    Sim sim{cfg};
    sim.reset(1);
    sim.step(Action::noop()); // spawn a threat
    REQUIRE(sim.threats().size() >= 1);
    sim.step(Action::fire(BaseId::Delta, sim.threats()[0].pos));

    bool killed = false;
    for (int i = 0; i < 300 && !killed; ++i) {
        sim.step(Action::noop());
        killed = has_event(sim, EventType::ThreatKilled);
    }
    REQUIRE(killed);
}

TEST_CASE("Losing the last city emits CityLost then GameOver", "[unit][sim]") {
    Config cfg;
    cfg.threat_base_speed = 3000.0f;
    cfg.spawn_interval = 0.05f;
    cfg.wave_base_threats = 50;
    Sim sim{cfg};
    sim.reset(9);

    bool city_lost = false;
    bool game_over = false;
    for (int i = 0; i < 5000 && !game_over; ++i) {
        sim.step(Action::noop());
        city_lost = city_lost || has_event(sim, EventType::CityLost);
        game_over = has_event(sim, EventType::GameOver);
    }
    REQUIRE(city_lost);
    REQUIRE(game_over);
    REQUIRE(sim.terminated());
}

TEST_CASE("The episode terminates when every city is destroyed", "[unit][sim]") {
    Config cfg;
    cfg.threat_base_speed = 3000.0f;
    cfg.spawn_interval = 0.05f;
    cfg.wave_base_threats = 50;
    Sim sim{cfg};
    sim.reset(9);

    bool terminated = false;
    for (int i = 0; i < 5000 && !terminated; ++i) {
        terminated = sim.step(Action::noop()).terminated;
    }

    REQUIRE(terminated);
    REQUIRE(sim.terminated());
    for (const auto& city : sim.cities()) {
        REQUIRE_FALSE(city.alive);
    }
}

TEST_CASE("Clearing a wave awards a bonus and advances to the next wave", "[unit][sim]") {
    Config cfg;
    cfg.wave_base_threats = 1;       // a single threat this wave
    cfg.threat_base_speed = 3000.0f; // reaches a city fast, clearing the wave
    cfg.wave_break = 0.1f;
    Sim sim{cfg};
    sim.reset(3);
    REQUIRE(sim.wave() == 1u);

    bool advanced = false;
    for (int i = 0; i < 600 && !advanced; ++i) {
        sim.step(Action::noop());
        advanced = sim.wave() == 2u;
    }
    REQUIRE(advanced);
    REQUIRE(sim.score() > 0); // end-of-wave bonus for surviving cities + unused ammo
}
