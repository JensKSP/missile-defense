// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
#include "md/observation.hpp"

#include "md/config.hpp"
#include "md/entities.hpp"
#include "md/event.hpp"
#include "md/sim.hpp"
#include "md/vec2.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>

namespace md {

namespace {

/// 1/x, or 0 when x is not positive — keeps the encoder total on degenerate configs.
float inv_or_zero(float x) noexcept {
    return x > 0.0f ? 1.0f / x : 0.0f;
}

} // namespace

void encode(const Sim& sim, const ObsSpec& spec, std::span<float> out) noexcept {
    const std::size_t total = spec.size();
    if (out.size() < total) {
        return; // caller buffer too small: write nothing rather than a partial row
    }

    // Clear once, then write only what is live. The observation is mostly padding
    // — a few threats in 128 slots — so a bulk clear plus a handful of scattered
    // stores beats visiting every element, and it makes the padding contract
    // ("empty slots read zero") true by construction rather than by loop.
    float* const base = out.data();
    std::fill_n(base, total, 0.0f);

    const Config& cfg = sim.config();
    const float inv_w = inv_or_zero(cfg.world_width);
    const float inv_h = inv_or_zero(cfg.world_height);
    const float inv_speed = inv_or_zero(cfg.interceptor_speed);
    const float inv_radius = inv_or_zero(cfg.blast_max_radius);
    const float inv_blast_lifetime = inv_or_zero(cfg.blast_lifetime);
    const float inv_ammo = inv_or_zero(static_cast<float>(cfg.ammo_per_base));
    const float inv_base_cd = inv_or_zero(cfg.base_cooldown);
    const float inv_trigger = inv_or_zero(cfg.fire_interval);
    const float inv_bonus = inv_or_zero(static_cast<float>(cfg.bonus_city_score));

    // World box -> [-1, 1].
    const auto nx = [inv_w](float x) noexcept { return (2.0f * x * inv_w) - 1.0f; };
    const auto ny = [inv_h](float y) noexcept { return (2.0f * y * inv_h) - 1.0f; };

    std::size_t offset = 0;

    // ---- Threats: present, position, velocity, one-hot type ------------------
    {
        const auto threats = sim.threats();
        const std::size_t count = std::min<std::size_t>(threats.size(), spec.threats);
        for (std::size_t i = 0; i < count; ++i) {
            float* slot = base + offset + (i * ObsSpec::threat_features);
            const Threat& threat = threats[i];
            slot[0] = 1.0f;
            slot[1] = nx(threat.pos.x);
            slot[2] = ny(threat.pos.y);
            slot[3] = threat.velocity.x * inv_speed;
            slot[4] = threat.velocity.y * inv_speed;
            const auto type = static_cast<std::size_t>(threat.type);
            if (type < 4u) {
                slot[5 + type] = 1.0f; // the other three stay zero from the clear
            }
        }
        offset += static_cast<std::size_t>(spec.threats) * ObsSpec::threat_features;
    }

    // ---- Interceptors: present, position, velocity, detonation point ---------
    // The player chose that detonation point, so it is theirs to know.
    {
        const auto interceptors = sim.interceptors();
        const std::size_t count = std::min<std::size_t>(interceptors.size(), spec.interceptors);
        for (std::size_t i = 0; i < count; ++i) {
            float* slot = base + offset + (i * ObsSpec::interceptor_features);
            const Interceptor& shot = interceptors[i];
            slot[0] = 1.0f;
            slot[1] = nx(shot.pos.x);
            slot[2] = ny(shot.pos.y);
            slot[3] = shot.velocity.x * inv_speed;
            slot[4] = shot.velocity.y * inv_speed;
            slot[5] = nx(shot.target.x);
            slot[6] = ny(shot.target.y);
        }
        offset += static_cast<std::size_t>(spec.interceptors) * ObsSpec::interceptor_features;
    }

    // ---- Blasts: present, position, radius, rendered lifetime phase -----------
    // Radius alone stops carrying time once a blast reaches full size. The
    // renderer continues to show its age through the fireball phase, so expose the
    // same age/lifetime value here. This lets a policy judge how long an otherwise
    // identical full-radius blast will remain active without privileged state.
    {
        const auto blasts = sim.blasts();
        const std::size_t count = std::min<std::size_t>(blasts.size(), spec.blasts);
        for (std::size_t i = 0; i < count; ++i) {
            float* slot = base + offset + (i * ObsSpec::blast_features);
            const Blast& blast = blasts[i];
            slot[0] = 1.0f;
            slot[1] = nx(blast.center.x);
            slot[2] = ny(blast.center.y);
            slot[3] = blast.radius * inv_radius;
            slot[4] = blast.age * inv_blast_lifetime;
        }
        offset += static_cast<std::size_t>(spec.blasts) * ObsSpec::blast_features;
    }

    // ---- Batteries: alive, position, ammo, own cooldown ----------------------
    {
        std::size_t i = 0;
        for (const Base& battery : sim.bases()) {
            float* slot = base + offset + (i * ObsSpec::base_features);
            slot[0] = battery.alive ? 1.0f : 0.0f;
            slot[1] = nx(battery.pos.x);
            slot[2] = static_cast<float>(battery.ammo) * inv_ammo;
            slot[3] = battery.cooldown_remaining * inv_base_cd;
            ++i;
        }
        offset += static_cast<std::size_t>(base_count) * ObsSpec::base_features;
    }

    // ---- Cities: alive, position --------------------------------------------
    {
        std::size_t i = 0;
        for (const City& city : sim.cities()) {
            float* slot = base + offset + (i * ObsSpec::city_features);
            slot[0] = city.alive ? 1.0f : 0.0f;
            slot[1] = nx(city.pos.x);
            ++i;
        }
        offset += static_cast<std::size_t>(max_cities) * ObsSpec::city_features;
    }

    // ---- Globals: crosshair, trigger cooldown, wave, score -------------------
    // The crosshair and trigger are part of the player model, so the policy must
    // see them to plan around cursor travel and shot pacing. Score is on the HUD
    // and gates bonus cities, so it is decision-relevant, not just the return.
    {
        const Vec2 cross = sim.crosshair();
        float* slot = base + offset;
        slot[0] = nx(cross.x);
        slot[1] = ny(cross.y);
        slot[2] = sim.fire_cooldown() * inv_trigger;
        slot[3] = static_cast<float>(sim.wave()) * 0.05f;
        slot[4] = static_cast<float>(sim.score()) * inv_bonus;
        offset += ObsSpec::global_features;
    }

    // ---- Direct encode: events from this tick (DESIGN.md §13) ----------------
    {
        float* slot = base + offset;
        for (const Event& event : sim.events()) {
            const auto index = static_cast<std::size_t>(event.type);
            if (index < ObsSpec::event_features) {
                slot[index] += 0.25f;
            }
        }
    }
}

} // namespace md
