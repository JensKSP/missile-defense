#pragma once

#include "md/config.hpp"
#include "md/vec2.hpp"

#include <cstdint>
#include <type_traits>

namespace md {

/// A defended city. Destroyed permanently when a threat reaches it.
struct City {
    Vec2 pos{};
    bool alive = true;
};

/// An interceptor battery.
struct Base {
    Vec2 pos{};
    std::uint32_t ammo = 0;
    float cooldown_remaining = 0.0f; // seconds until this base may fire again
    bool alive = true;               // v0.1: bases are not destructible
};

/// A player interceptor in flight toward its detonation point.
struct Interceptor {
    Vec2 pos{};
    Vec2 origin{}; // launch point (for the trail)
    Vec2 velocity{};
    Vec2 target{}; // point at which it detonates
    bool active = false;
};

/// An incoming enemy warhead.
struct Threat {
    Vec2 pos{};
    Vec2 origin{}; // where it entered/split from (for the trail)
    Vec2 velocity{};
    ThreatType type = ThreatType::Icbm;
    TargetKind target_kind = TargetKind::City;
    std::uint32_t target_index = 0; // index into cities or bases, per target_kind
    float split_altitude = 0.0f;    // MIRV: y at which it splits (0 = never)
    bool active = false;
};

/// An expanding explosion that destroys threats within its current radius.
struct Blast {
    Vec2 center{};
    float age = 0.0f;    // seconds since detonation
    float radius = 0.0f; // current radius, derived from age
    bool active = false;
};

/// A cosmetic ground-impact fireball (does NOT destroy threats). Spawned when a
/// threat reaches the ground; bigger when it destroyed a city or base.
struct Explosion {
    Vec2 center{};
    float age = 0.0f;
    float radius = 0.0f;
    float peak_radius = 0.0f;
    bool active = false;
};

// The determinism / parallelism contract requires every entity to be trivially
// copyable, so the whole Sim state can be snapshotted with a plain memcpy.
static_assert(std::is_trivially_copyable_v<City>);
static_assert(std::is_trivially_copyable_v<Base>);
static_assert(std::is_trivially_copyable_v<Interceptor>);
static_assert(std::is_trivially_copyable_v<Threat>);
static_assert(std::is_trivially_copyable_v<Blast>);

} // namespace md
