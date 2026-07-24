#pragma once

#include <cstdint>

namespace md {

/// The three interceptor batteries, left to right (arcade: ALPHA / DELTA / OMEGA).
enum class BaseId : std::uint8_t { Alpha = 0, Delta = 1, Omega = 2 };
inline constexpr std::uint32_t base_count = 3;

/// Enemy variants. Encoded as a tag (no class hierarchy) to keep entity state POD
/// and the step loop free of virtual dispatch.
enum class ThreatType : std::uint8_t { Icbm = 0, Mirv = 1, SmartBomb = 2 };

// Fixed capacities. These bound the inline simulation state so a snapshot is a memcpy
// and the step loop never allocates.
inline constexpr std::uint32_t max_cities = 6;
inline constexpr std::uint32_t max_threats = 128;
inline constexpr std::uint32_t max_interceptors = 64;
inline constexpr std::uint32_t max_blasts = 64;

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

    // Interceptors & blasts.
    float interceptor_speed = 220.0f; // world units / second
    float blast_max_radius = 14.0f;   // world units
    float blast_lifetime = 0.9f;      // seconds: expand -> linger -> collapse

    // Threats.
    float threat_base_speed = 30.0f;    // wave-1 descent speed, world units / second
    float threat_speed_per_wave = 4.0f; // added to descent speed each wave

    // Waves & spawning.
    std::uint32_t wave_base_threats = 8;      // threats spawned in wave 1
    std::uint32_t wave_threats_increment = 2; // additional threats per subsequent wave
    float spawn_interval = 0.6f;              // seconds between spawns within a wave
    float wave_break = 2.0f;                  // seconds of calm between waves

    // Scoring (DESIGN.md §4.3).
    std::int32_t score_per_kill = 25;
    std::int32_t score_per_unused_interceptor = 5;
    std::int32_t score_per_surviving_city = 100;
};

} // namespace md
