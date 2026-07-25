// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
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

// The transient entities below carry no `active` flag: `Sim` keeps each pool
// compacted and hands out counted spans, so every element a caller can see is by
// construction live. A flag would be redundant state that is always `true` —
// and an inviting way to write a filter that silently does nothing.

/// A player interceptor in flight toward its detonation point.
struct Interceptor {
    Vec2 pos{};
    Vec2 origin{}; // launch point (for the trail)
    Vec2 velocity{};
    Vec2 target{}; // point at which it detonates
};

/// An incoming enemy warhead.
struct Threat {
    Vec2 pos{};
    Vec2 origin{}; // where it entered/split from (for the trail)
    Vec2 velocity{};
    ThreatType type = ThreatType::Icbm;
    // The installation this threat was *launched at*, which fixes its heading at
    // spawn. Damage is resolved by where it actually lands (a smart bomb steering
    // around a blast can miss), so this is a record of intent, not of outcome.
    TargetKind target_kind = TargetKind::City;
    std::uint32_t target_index = 0; // index into cities or bases, per target_kind
    float split_altitude = 0.0f;    // MIRV: y at which it splits (0 = never)
};

/// An expanding explosion that destroys threats within its current radius.
struct Blast {
    Vec2 center{};
    float age = 0.0f;    // seconds since detonation
    float radius = 0.0f; // current radius, derived from age
    // Threats this blast has destroyed over its whole life, not per tick. A
    // training reward wants to know whether an interceptor earned its keep, and
    // that is only answerable once the blast has finished expanding.
    std::uint32_t kills = 0;
};

/// A cosmetic ground-impact fireball (does NOT destroy threats). Spawned when a
/// threat reaches the ground; bigger when it destroyed a city or base.
struct Explosion {
    Vec2 center{};
    float age = 0.0f;
    float radius = 0.0f;
    float peak_radius = 0.0f;
};

// The determinism / parallelism contract requires every entity to be trivially
// copyable, so the whole Sim state can be snapshotted with a plain memcpy.
static_assert(std::is_trivially_copyable_v<City>);
static_assert(std::is_trivially_copyable_v<Base>);
static_assert(std::is_trivially_copyable_v<Interceptor>);
static_assert(std::is_trivially_copyable_v<Threat>);
static_assert(std::is_trivially_copyable_v<Blast>);

} // namespace md
