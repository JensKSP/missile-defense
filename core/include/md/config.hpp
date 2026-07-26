// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
#pragma once

#include <cstdint>

namespace md {

/// The three interceptor batteries, left to right (arcade: ALPHA / DELTA / OMEGA).
enum class BaseId : std::uint8_t { Alpha = 0, Delta = 1, Omega = 2 };
inline constexpr std::uint32_t base_count = 3;

/// Enemy variants. Encoded as a tag (no class hierarchy) to keep entity state POD
/// and the step loop free of virtual dispatch.
enum class ThreatType : std::uint8_t { Icbm = 0, Mirv = 1, SmartBomb = 2, Warhead = 3 };

/// What a threat is aimed at — both cities and bases sit on the ground and can be
/// destroyed by a threat that reaches them.
enum class TargetKind : std::uint8_t { City = 0, Base = 1 };

// Fixed capacities. These bound the inline simulation state so a snapshot is a memcpy
// and the step loop never allocates.
inline constexpr std::uint32_t max_cities = 6;
inline constexpr std::uint32_t max_threats = 128;
inline constexpr std::uint32_t max_interceptors = 64;
inline constexpr std::uint32_t max_blasts = 64;
inline constexpr std::uint32_t max_explosions = 64;
inline constexpr std::uint32_t max_events = 128; // events emitted per step (capped)

// Resolution of the kills-per-shot histogram: bins for 0, 1, 2, 3 and "4 or more"
// threats destroyed by a single interceptor's blast. A blast can occasionally
// catch a larger MIRV cluster; those fold into the top bin. This is how "is the
// agent catching clusters or wasting shots?" reads as a distribution rather than
// a single mean (the held-out scripted baseline's is about 1.09).
inline constexpr std::uint32_t kills_per_shot_bins = 5;

/// Tunable simulation constants (see DESIGN.md §2–4). Values here are the v0.1
/// strawman defaults, finalized during playtest before the mechanics freeze.
struct Config {
    // World — world units, origin bottom-left, y-up.
    float world_width = 320.0f;
    float world_height = 180.0f;
    float dt = 1.0f / 60.0f; // fixed timestep (seconds)

    // Bases.
    std::uint32_t ammo_per_base = 10;
    float base_cooldown = 0.1f; // minimum seconds between launches from one base

    // Player model — the limits a *hand* has, applied to every driver alike so the
    // AI cannot out-mechanic the human (DESIGN.md §5). The crosshair is simulation
    // state: naming a distant point costs travel time instead of being free.
    // Strawman values; calibrate from recorded human play before the freeze.
    float aim_max_speed = 1200.0f; // crosshair top speed, world units/s (0 = instant)
    float fire_interval = 0.33f;   // min seconds between ANY two launches (~3/s; 0 = none)
    // Reaction rate: the sim samples a *new* action once per this many ticks and
    // holds it between, so no driver — human, scripted, or learned — can re-decide
    // faster than a hand can. 4 ticks ≈ 15 Hz (the training frame-skip); 1 = every
    // tick (60 Hz). Enforced in `Sim::step`, not in a driver, exactly as the aim
    // and trigger limits are — otherwise a per-tick driver has a free reflex edge.
    std::uint32_t decision_interval = 4;

    // Interceptors & blasts.
    float interceptor_speed = 220.0f; // world units / second
    float blast_max_radius = 14.0f;   // world units
    float blast_lifetime = 0.9f;      // seconds: expand -> linger -> collapse

    // Ground-impact explosions (cosmetic — do not destroy threats).
    float explosion_lifetime = 0.9f;       // seconds
    float explosion_radius_ground = 9.0f;  // nuke hits already-ruined ground
    float explosion_radius_target = 22.0f; // nuke destroys a live city or base (bigger)

    // Threats.
    float threat_base_speed = 30.0f;    // wave-1 descent speed, world units / second
    float threat_speed_per_wave = 2.5f; // added to descent speed each wave (gentle ramp)

    // Waves & spawning.
    std::uint32_t wave_base_threats = 8;      // threats spawned in wave 1
    std::uint32_t wave_threats_increment = 2; // additional threats per subsequent wave
    float spawn_interval = 0.6f;              // seconds between spawns within a wave
    float wave_break = 2.0f;                  // seconds of calm between waves

    // MIRV — splitting warheads, appearing from wave 2.
    float mirv_chance_per_wave = 0.04f; // added chance per wave that a spawn is a MIRV
    float mirv_max_chance = 0.40f;      // cap on that chance
    std::uint32_t mirv_splits = 3;      // warheads a MIRV splits into

    // Smart bombs — decoys that steer to dodge blasts, appearing from wave 5.
    std::uint32_t smart_bomb_wave = 5;    // first wave they can appear
    float smart_bomb_chance = 0.15f;      // chance a wave>=smart_bomb_wave spawn is one
    float smart_bomb_dodge_range = 22.0f; // reacts to blasts within this (world units)
    float smart_bomb_dodge_accel = 90.0f; // lateral steering accel (world units / s^2)

    // Scoring (DESIGN.md §4.3), following the 1980 arcade original.
    std::int32_t score_per_kill = 25;
    // A smart bomb is worth five ordinary warheads in the original, and it is by
    // far the hardest thing on the field to hit — it steers around blasts.
    std::int32_t score_per_smart_bomb = 125;
    std::int32_t score_per_unused_interceptor = 5;
    std::int32_t score_per_surviving_city = 100;
    std::int32_t bonus_city_score = 10000; // restore a destroyed city every N points
    // The arcade multiplies *everything* — kills and the end-of-wave bonus — by a
    // factor that steps up every two waves and caps out. It is the reason the
    // original is a game about surviving deep rather than farming early waves: at
    // the cap a surviving city is worth 600, not 100. Leaving it out, as this did
    // until now, quietly flattens the whole incentive to last.
    std::uint32_t score_multiplier_wave_step = 2; // waves per step up
    std::uint32_t score_multiplier_max = 6;       // reached at wave 11
};

} // namespace md
