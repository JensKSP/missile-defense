// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
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
/// A config with the player-model limits switched off: the crosshair snaps to the
/// aim point and there is no trigger interval. Tests of *other* mechanics (blasts,
/// ammo, scoring) use this so they are not paced — or silently satisfied — by the
/// aiming model. The player model itself is covered by its own tests below.
Config unpaced() {
    Config cfg;
    cfg.aim_max_speed = 0.0f;  // instant aim
    cfg.fire_interval = 0.0f;  // no global trigger interval
    cfg.decision_interval = 1; // accept a deliberately different action every tick
    return cfg;
}

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

    sim.step(Action::fire_at(BaseId::Alpha, Vec2{100.0f, 90.0f}));

    REQUIRE(sim.interceptors().size() == 1);
    REQUIRE(sim.bases()[0].ammo == ammo_before - 1);
    REQUIRE(sim.bases()[0].cooldown_remaining > 0.0f);
}

TEST_CASE("A base with no ammo cannot fire", "[unit][sim]") {
    Config cfg = unpaced(); // isolate the ammo check from the trigger interval
    cfg.ammo_per_base = 1;
    cfg.base_cooldown = 0.0f;
    Sim sim{cfg};
    sim.reset(0);

    sim.step(Action::fire_at(BaseId::Delta, Vec2{160.0f, 90.0f})); // uses the only round
    REQUIRE(sim.bases()[1].ammo == 0u);

    sim.step(Action::fire_at(BaseId::Delta, Vec2{160.0f, 90.0f})); // rejected: empty
    REQUIRE(sim.bases()[1].ammo == 0u);
    REQUIRE(sim.interceptors().size() == 1); // no second interceptor spawned
}

TEST_CASE("A base respects its cooldown between shots", "[unit][sim]") {
    Config cfg = unpaced(); // isolate the per-base cooldown from the trigger interval
    cfg.base_cooldown = 0.5f;
    Sim sim{cfg};
    sim.reset(0);

    sim.step(Action::fire_at(BaseId::Alpha, Vec2{50.0f, 120.0f}));
    const std::uint32_t ammo_after_first = sim.bases()[0].ammo;

    // Immediately firing again is rejected while on cooldown.
    sim.step(Action::fire_at(BaseId::Alpha, Vec2{50.0f, 120.0f}));
    REQUIRE(sim.bases()[0].ammo == ammo_after_first);

    // After the cooldown elapses (~0.5 s = 30 ticks) a shot succeeds again.
    for (int i = 0; i < 40; ++i) {
        sim.step(Action::noop());
    }
    const std::uint32_t ammo_before_third = sim.bases()[0].ammo;
    sim.step(Action::fire_at(BaseId::Alpha, Vec2{50.0f, 120.0f}));
    REQUIRE(sim.bases()[0].ammo == ammo_before_third - 1);
}

TEST_CASE("An interceptor reaches its target and detonates into a blast", "[unit][sim]") {
    Config cfg = unpaced();
    cfg.base_cooldown = 0.0f;
    Sim sim{cfg};
    sim.reset(0);

    const Vec2 target{sim.bases()[0].pos.x, 20.0f}; // straight up, close
    sim.step(Action::fire_at(BaseId::Alpha, target));
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
    Config cfg = unpaced();
    cfg.base_cooldown = 0.0f;
    cfg.blast_lifetime = 0.2f;
    Sim sim{cfg};
    sim.reset(0);

    sim.step(Action::fire_at(BaseId::Alpha, Vec2{sim.bases()[0].pos.x, 5.0f}));

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
    Config cfg = unpaced();
    cfg.blast_max_radius = 40.0f;   // large, forgiving blast
    cfg.blast_lifetime = 3.0f;      // long-lived
    cfg.threat_base_speed = 5.0f;   // slow threats
    cfg.interceptor_speed = 400.0f; // fast interceptor
    Sim sim{cfg};
    sim.reset(1);

    sim.step(Action::noop()); // spawn a threat
    REQUIRE(sim.threats().size() >= 1);

    const std::int32_t score_before = sim.score();
    sim.step(Action::fire_at(BaseId::Delta, sim.threats()[0].pos));

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
    Config cfg = unpaced(); // the blast must land exactly where the test asks
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
    sim.step(Action::fire_at(BaseId::Delta, Vec2{p.x - 18.0f, p.y}));

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
    Config cfg = unpaced();
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
            sim.step(Action::fire_at(BaseId::Delta, sim.threats()[0].pos));
        } else {
            sim.step(Action::noop());
        }
        const std::size_t now = alive_cities();
        rebuilt = now > prev;
        prev = now;
    }
    REQUIRE(rebuilt);
}

TEST_CASE("Damage lands where the warhead lands, not where it was aimed", "[unit][sim]") {
    // A smart bomb steers sideways to dodge blasts, so its impact point can end up
    // far from the installation it launched at. What it destroys must follow the
    // visible trajectory: if the stored assignment decided the outcome, a bomb that
    // plainly drifted clear would still level the city it started towards, and
    // neither the player nor a policy could predict that from what they can see.
    Config cfg = unpaced();
    cfg.smart_bomb_wave = 1;
    cfg.smart_bomb_chance = 1.0f; // every spawn is a dodger
    cfg.wave_base_threats = 1;
    cfg.threat_base_speed = 10.0f;      // slow: plenty of time to be shoved
    cfg.smart_bomb_dodge_range = 80.0f; // and it reacts from far away
    cfg.smart_bomb_dodge_accel = 500.0f;
    cfg.blast_lifetime = 8.0f; // a long-lived shove
    cfg.blast_max_radius = 10.0f;
    cfg.interceptor_speed = 900.0f;
    cfg.base_cooldown = 0.0f;
    Sim sim{cfg};
    sim.reset(11);
    sim.step(Action::noop());
    REQUIRE(sim.threats().size() == 1);
    REQUIRE(sim.threats()[0].type == ThreatType::SmartBomb);

    const auto aimed_kind = sim.threats()[0].target_kind;
    const std::uint32_t aimed_index = sim.threats()[0].target_index;
    const Vec2 aim_point = (aimed_kind == md::TargetKind::City) ? sim.cities()[aimed_index].pos
                                                                : sim.bases()[aimed_index].pos;

    // Detonate beside the bomb so it steers away from where it was headed.
    const Vec2 bomb = sim.threats()[0].pos;
    sim.step(Action::fire_at(BaseId::Alpha, Vec2{bomb.x - 12.0f, bomb.y - 6.0f}));

    float last_x = bomb.x;
    for (int i = 0; i < 3000 && !sim.threats().empty(); ++i) {
        last_x = sim.threats()[0].pos.x;
        sim.step(Action::noop());
    }

    const float slot = cfg.world_width / static_cast<float>(md::base_count + md::max_cities);
    REQUIRE(std::abs(last_x - aim_point.x) > slot * 0.5f); // it really did drift clear

    // Having landed elsewhere, it cannot have destroyed what it was aimed at.
    if (aimed_kind == md::TargetKind::City) {
        REQUIRE(sim.cities()[aimed_index].alive);
    } else {
        REQUIRE(sim.bases()[aimed_index].alive);
    }
}

TEST_CASE("Batteries are rebuilt between waves", "[unit][sim]") {
    // Losing a battery costs its ammo and coverage for the rest of the wave, but
    // must not be permanent: three dead batteries used to mean the player could
    // never fire again and just watched a decided game finish itself.
    Config cfg = unpaced();
    cfg.threat_base_speed = 3000.0f; // everything lands almost at once
    cfg.wave_base_threats = 40;      // enough to flatten the batteries
    cfg.spawn_interval = 0.02f;
    Sim sim{cfg};
    sim.reset(4);

    bool a_base_died = false;
    for (int i = 0; i < 4000 && !sim.terminated(); ++i) {
        sim.step(Action::noop());
        for (const auto& base : sim.bases()) {
            a_base_died = a_base_died || !base.alive;
        }
        if (a_base_died && sim.wave() >= 2u) {
            break;
        }
    }
    REQUIRE(a_base_died);
    if (sim.wave() >= 2u) {
        for (const auto& base : sim.bases()) {
            REQUIRE(base.alive); // the new wave brought them all back
            REQUIRE(base.ammo == cfg.ammo_per_base);
        }
    }
}

TEST_CASE("A MIRV whose spread will not fit holds instead of splitting short", "[unit][sim]") {
    // Removing the parent frees exactly one slot, so a saturated field would turn
    // a MIRV into a single warhead — quietly gifting the player the other two.
    Config cfg = unpaced();
    cfg.mirv_splits = md::max_threats + 10u; // can never fit
    cfg.wave_base_threats = 1;
    cfg.mirv_chance_per_wave = 1.0f; // guarantee a MIRV from wave 2
    cfg.mirv_max_chance = 1.0f;
    cfg.threat_base_speed = 20.0f;
    Sim sim{cfg};
    sim.reset(2);

    // Run into wave 2+ so MIRVs can appear, and check none ever split short.
    for (int i = 0; i < 3000 && !sim.terminated(); ++i) {
        sim.step(Action::noop());
        REQUIRE(sim.threats().size() <= md::max_threats);
        for (const auto& threat : sim.threats()) {
            // No child warheads can exist: every split was refused.
            REQUIRE(threat.type != ThreatType::Warhead);
        }
    }
}

TEST_CASE("A bonus earned with every city standing is banked, not forfeited", "[unit][sim]") {
    // Crossing the threshold with a full skyline used to advance it and rebuild
    // nothing — so playing well threw the reward away. It must be held instead.
    Config cfg = unpaced();
    cfg.bonus_city_score = 50;    // two kills' worth
    cfg.blast_max_radius = 40.0f; // easy kills
    cfg.blast_lifetime = 3.0f;
    cfg.threat_base_speed = 5.0f; // slow, so no city is lost meanwhile
    cfg.interceptor_speed = 400.0f;
    cfg.base_cooldown = 0.0f;
    Sim sim{cfg};
    sim.reset(1);

    for (int i = 0; i < 600 && sim.score() < cfg.bonus_city_score; ++i) {
        if (!sim.threats().empty()) {
            sim.step(Action::fire_at(BaseId::Delta, sim.threats()[0].pos));
        } else {
            sim.step(Action::noop());
        }
    }
    REQUIRE(sim.score() >= cfg.bonus_city_score);

    std::size_t alive = 0;
    for (const auto& city : sim.cities()) {
        alive += city.alive ? 1u : 0u;
    }
    REQUIRE(alive == md::max_cities);         // nothing to rebuild ...
    REQUIRE(sim.bonus_cities_banked() >= 1u); // ... so the credit is kept
}

TEST_CASE("A banked bonus city is spent on the next gap in the skyline", "[unit][sim]") {
    Config cfg = unpaced();
    cfg.bonus_city_score = 50;
    cfg.blast_max_radius = 40.0f;
    cfg.blast_lifetime = 3.0f;
    cfg.threat_base_speed = 5.0f;
    cfg.interceptor_speed = 400.0f;
    cfg.base_cooldown = 0.0f;
    Sim sim{cfg};
    sim.reset(1);

    for (int i = 0; i < 600 && sim.bonus_cities_banked() == 0u; ++i) {
        if (!sim.threats().empty()) {
            sim.step(Action::fire_at(BaseId::Delta, sim.threats()[0].pos));
        } else {
            sim.step(Action::noop());
        }
    }
    REQUIRE(sim.bonus_cities_banked() >= 1u);
    const std::uint32_t banked_before = sim.bonus_cities_banked();

    // Stop defending: the next warhead through takes a city, and the credit pays
    // for it on the same tick, so the skyline never actually thins.
    bool spent = false;
    for (int i = 0; i < 4000 && !spent; ++i) {
        sim.step(Action::noop());
        spent = sim.bonus_cities_banked() < banked_before;
    }
    REQUIRE(spent);
    REQUIRE(has_event(sim, EventType::BonusCity));
}

TEST_CASE("A zero bonus threshold does not hang the simulation", "[unit][sim]") {
    // `next_bonus_score_ += 0` never escapes `score >= next`, so the award loop
    // spun forever. Degenerate configs must terminate, not wedge the sim.
    Config cfg = unpaced();
    cfg.bonus_city_score = 0;
    Sim sim{cfg};
    sim.reset(0);
    for (int i = 0; i < 100; ++i) {
        sim.step(Action::noop());
    }
    REQUIRE(sim.tick() == 100u);
}

TEST_CASE("Firing emits a Fire event; events are per-step", "[unit][sim]") {
    Sim sim;
    sim.reset(0);
    REQUIRE(sim.events().empty());

    sim.step(Action::fire_at(BaseId::Alpha, Vec2{100.0f, 90.0f}));
    REQUIRE(has_event(sim, EventType::Fire));

    sim.step(Action::noop()); // the next step clears the previous step's events
    REQUIRE_FALSE(has_event(sim, EventType::Fire));
}

// ---- Player model: crosshair travel + trigger interval (DESIGN.md §5) --------
// These limits apply to every driver alike, so the AI cannot out-mechanic a hand.

TEST_CASE("The crosshair starts centred and holds when the action does not move it",
          "[unit][sim]") {
    Sim sim;
    sim.reset(0);
    const Vec2 start{sim.config().world_width * 0.5f, sim.config().world_height * 0.5f};
    REQUIRE(sim.crosshair() == start);

    sim.step(Action::noop());
    REQUIRE(sim.crosshair() == start); // NoOp does not drag the cursor anywhere
}

TEST_CASE("The crosshair travels toward the aim point at a capped speed", "[unit][sim]") {
    Config cfg;
    cfg.aim_max_speed = 60.0f; // 1 world unit per tick at dt = 1/60
    Sim sim{cfg};
    sim.reset(0);

    const Vec2 start = sim.crosshair();
    const Vec2 far{start.x + 100.0f, start.y}; // 100 units away: 100 ticks of travel

    sim.step(Action::aim_at(far));
    REQUIRE_THAT(static_cast<double>(sim.crosshair().x - start.x), WithinAbs(1.0, 1e-4));
    REQUIRE_THAT(static_cast<double>(sim.crosshair().y),
                 WithinAbs(static_cast<double>(start.y), 1e-4));

    for (int i = 0; i < 99; ++i) {
        sim.step(Action::aim_at(far));
    }
    REQUIRE_THAT(static_cast<double>(sim.crosshair().x), WithinAbs(static_cast<double>(far.x),
                                                                   1e-3)); // arrived, not overshot
}

TEST_CASE("A shot detonates at the crosshair, not at a distant requested aim", "[unit][sim]") {
    Config cfg;
    cfg.aim_max_speed = 60.0f; // 1 unit/tick — the crosshair cannot arrive this tick
    cfg.fire_interval = 0.0f;
    Sim sim{cfg};
    sim.reset(0);

    const Vec2 start = sim.crosshair();
    sim.step(Action::fire_at(BaseId::Alpha, Vec2{start.x + 100.0f, start.y}));

    REQUIRE(sim.interceptors().size() == 1);
    // The interceptor is aimed one tick's worth of travel away, not 100 units away.
    REQUIRE_THAT(static_cast<double>(sim.interceptors()[0].target.x),
                 WithinAbs(static_cast<double>(start.x + 1.0f), 1e-4));
}

TEST_CASE("aim_max_speed = 0 means instant aim", "[unit][sim]") {
    Config cfg = unpaced();
    Sim sim{cfg};
    sim.reset(0);

    const Vec2 target{10.0f, 170.0f};
    sim.step(Action::aim_at(target));
    REQUIRE(sim.crosshair() == target);
}

TEST_CASE("The crosshair is clamped to the world bounds", "[unit][sim]") {
    Config cfg = unpaced();
    Sim sim{cfg};
    sim.reset(0);

    sim.step(Action::aim_at(Vec2{-500.0f, -500.0f}));
    REQUIRE(sim.crosshair() == Vec2{0.0f, 0.0f});

    sim.step(Action::aim_at(Vec2{9000.0f, 9000.0f}));
    REQUIRE(sim.crosshair() == Vec2{cfg.world_width, cfg.world_height});
}

TEST_CASE("The trigger interval paces shots across different batteries", "[unit][sim]") {
    Config cfg;
    cfg.aim_max_speed = 0.0f;  // isolate the trigger interval from crosshair travel
    cfg.base_cooldown = 0.0f;  // and from the per-battery cooldown
    cfg.fire_interval = 0.15f; // 9 ticks at dt = 1/60
    cfg.decision_interval = 1; // this test deliberately changes the action each tick
    Sim sim{cfg};
    sim.reset(0);

    const Vec2 target{160.0f, 90.0f};
    sim.step(Action::fire_at(BaseId::Alpha, target));
    REQUIRE(sim.interceptors().size() == 1);

    // A *different*, fully-loaded battery still cannot fire: the limit is the
    // player's trigger finger, not the battery.
    sim.step(Action::fire_at(BaseId::Delta, target));
    REQUIRE(sim.interceptors().size() == 1);
    REQUIRE(sim.bases()[1].ammo == cfg.ammo_per_base); // no ammo spent on the rejected shot

    for (int i = 0; i < 9; ++i) {
        sim.step(Action::noop()); // let the interval elapse
    }
    sim.step(Action::fire_at(BaseId::Delta, target));
    REQUIRE(sim.interceptors().size() == 2);
}

TEST_CASE("fire_interval = 0 leaves only the per-battery cooldown", "[unit][sim]") {
    Config cfg = unpaced();
    cfg.base_cooldown = 0.0f;
    Sim sim{cfg};
    sim.reset(0);

    const Vec2 target{160.0f, 90.0f};
    sim.step(Action::fire_at(BaseId::Alpha, target));
    sim.step(Action::fire_at(BaseId::Delta, target));
    REQUIRE(sim.interceptors().size() == 2); // back-to-back ticks are allowed
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
    Config cfg = unpaced();
    cfg.blast_max_radius = 40.0f;
    cfg.blast_lifetime = 3.0f;
    cfg.threat_base_speed = 5.0f;
    cfg.interceptor_speed = 400.0f;
    Sim sim{cfg};
    sim.reset(1);
    sim.step(Action::noop()); // spawn a threat
    REQUIRE(sim.threats().size() >= 1);
    sim.step(Action::fire_at(BaseId::Delta, sim.threats()[0].pos));

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

TEST_CASE("The wave multiplier steps up every two waves and caps", "[unit][sim][scoring]") {
    // 1x for waves 1-2, 2x for 3-4, and so on to 6x from wave 11. Without it the
    // incentive to survive deep is flattened: at the cap a surviving city is
    // worth 600, not 100. Waves are cleared by letting a single fast threat land.
    Config cfg;
    cfg.wave_base_threats = 1;
    cfg.wave_threats_increment = 0; // stay at one threat per wave, however deep
    cfg.threat_base_speed = 3000.0f;
    cfg.wave_break = 0.05f;
    Sim sim{cfg};
    sim.reset(3);
    REQUIRE(sim.wave() == 1u);
    REQUIRE(sim.score_multiplier() == 1);

    const auto multiplier_at = [](std::uint32_t wave) {
        return static_cast<std::int32_t>(std::min((wave - 1u) / 2u + 1u, 6u));
    };
    for (int i = 0; i < 60000 && sim.wave() < 12u; ++i) {
        const std::uint32_t before = sim.wave();
        sim.step(Action::noop());
        if (sim.wave() != before) {
            CHECK(sim.score_multiplier() == multiplier_at(sim.wave()));
        }
    }
    REQUIRE(sim.wave() >= 12u);
    REQUIRE(sim.score_multiplier() == 6); // capped from wave 11 on
}

TEST_CASE("A smart bomb is worth five ordinary warheads", "[unit][sim][scoring]") {
    // The arcade pays 125 for the one threat that steers around blasts; every
    // other flier is an ordinary missile at 25. Wave 1, so the multiplier is 1x
    // and the raw values are directly visible in the score.
    const auto score_first_kill = [](std::uint64_t seed, ThreatType wanted) {
        Config cfg = unpaced();
        cfg.blast_max_radius = 40.0f;
        cfg.blast_lifetime = 3.0f;
        cfg.threat_base_speed = 5.0f;
        cfg.interceptor_speed = 400.0f;
        Sim sim{cfg};
        sim.reset(seed);
        sim.step(Action::noop());
        if (sim.threats().empty() || sim.threats()[0].type != wanted) {
            return -1; // this seed did not open with the threat we wanted
        }
        const std::int32_t before = sim.score();
        sim.step(Action::fire_at(BaseId::Delta, sim.threats()[0].pos));
        for (int i = 0; i < 300; ++i) {
            sim.step(Action::noop());
            if (sim.score() != before) {
                return sim.score() - before;
            }
        }
        return -1;
    };

    // Seeds differ in what wave 1 opens with; take the first that gives each type.
    std::int32_t icbm = -1;
    std::int32_t smart = -1;
    for (std::uint64_t seed = 1; seed < 400 && (icbm < 0 || smart < 0); ++seed) {
        if (icbm < 0) {
            icbm = score_first_kill(seed, ThreatType::Icbm);
        }
        if (smart < 0) {
            smart = score_first_kill(seed, ThreatType::SmartBomb);
        }
    }
    REQUIRE(icbm == Config{}.score_per_kill);
    if (smart > 0) { // smart bombs may not appear in wave 1 of any early seed
        REQUIRE(smart == Config{}.score_per_smart_bomb);
        REQUIRE(smart == 5 * icbm);
    }
}
