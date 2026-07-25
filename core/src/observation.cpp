// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
#include "md/observation.hpp"

#include "md/config.hpp"
#include "md/entities.hpp"
#include "md/event.hpp"
#include "md/sim.hpp"
#include "md/vec2.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace md {

namespace {

/// Sequential cursor over the output buffer. Bounds are checked once by the
/// caller; this only keeps the write position.
class Writer {
  public:
    explicit Writer(std::span<float> out) noexcept : out_{out} {}

    void put(float v) noexcept {
        if (index_ < out_.size()) {
            out_[index_] = v;
        }
        ++index_;
    }

    void pad(std::size_t n) noexcept {
        for (std::size_t i = 0; i < n; ++i) {
            put(0.0f);
        }
    }

  private:
    std::span<float> out_;
    std::size_t index_ = 0;
};

/// 1/x, or 0 when x is not positive — keeps the encoder total on degenerate configs.
float inv_or_zero(float x) noexcept {
    return x > 0.0f ? 1.0f / x : 0.0f;
}

} // namespace

void encode(const Sim& sim, const ObsSpec& spec, std::span<float> out) noexcept {
    if (out.size() < spec.size()) {
        return; // caller buffer too small: write nothing rather than a partial row
    }

    const Config& cfg = sim.config();
    Writer w{out};

    const float inv_w = inv_or_zero(cfg.world_width);
    const float inv_h = inv_or_zero(cfg.world_height);
    const float inv_speed = inv_or_zero(cfg.interceptor_speed);
    const float inv_radius = inv_or_zero(cfg.blast_max_radius);
    const float inv_ammo = inv_or_zero(static_cast<float>(cfg.ammo_per_base));
    const float inv_base_cd = inv_or_zero(cfg.base_cooldown);
    const float inv_trigger = inv_or_zero(cfg.fire_interval);
    const float inv_bonus = inv_or_zero(static_cast<float>(cfg.bonus_city_score));

    // World box -> [-1, 1].
    const auto nx = [inv_w](float x) noexcept { return (2.0f * x * inv_w) - 1.0f; };
    const auto ny = [inv_h](float y) noexcept { return (2.0f * y * inv_h) - 1.0f; };

    // ---- Threats: present, position, velocity, one-hot type ------------------
    const auto threats = sim.threats();
    for (std::uint32_t i = 0; i < spec.threats; ++i) {
        if (i >= threats.size()) {
            w.pad(ObsSpec::threat_features);
            continue;
        }
        const Threat& t = threats[i];
        w.put(1.0f);
        w.put(nx(t.pos.x));
        w.put(ny(t.pos.y));
        w.put(t.velocity.x * inv_speed);
        w.put(t.velocity.y * inv_speed);
        const auto type = static_cast<std::uint8_t>(t.type);
        for (std::uint8_t k = 0; k < 4; ++k) {
            w.put(type == k ? 1.0f : 0.0f);
        }
    }

    // ---- Interceptors: present, position, velocity, detonation point ---------
    // The player chose that detonation point, so it is theirs to know.
    const auto interceptors = sim.interceptors();
    for (std::uint32_t i = 0; i < spec.interceptors; ++i) {
        if (i >= interceptors.size()) {
            w.pad(ObsSpec::interceptor_features);
            continue;
        }
        const Interceptor& it = interceptors[i];
        w.put(1.0f);
        w.put(nx(it.pos.x));
        w.put(ny(it.pos.y));
        w.put(it.velocity.x * inv_speed);
        w.put(it.velocity.y * inv_speed);
        w.put(nx(it.target.x));
        w.put(ny(it.target.y));
    }

    // ---- Blasts: present, position, current radius ---------------------------
    const auto blasts = sim.blasts();
    for (std::uint32_t i = 0; i < spec.blasts; ++i) {
        if (i >= blasts.size()) {
            w.pad(ObsSpec::blast_features);
            continue;
        }
        const Blast& b = blasts[i];
        w.put(1.0f);
        w.put(nx(b.center.x));
        w.put(ny(b.center.y));
        w.put(b.radius * inv_radius);
    }

    // ---- Batteries: alive, position, ammo, own cooldown ----------------------
    for (const Base& b : sim.bases()) {
        w.put(b.alive ? 1.0f : 0.0f);
        w.put(nx(b.pos.x));
        w.put(static_cast<float>(b.ammo) * inv_ammo);
        w.put(b.cooldown_remaining * inv_base_cd);
    }

    // ---- Cities: alive, position --------------------------------------------
    for (const City& c : sim.cities()) {
        w.put(c.alive ? 1.0f : 0.0f);
        w.put(nx(c.pos.x));
    }

    // ---- Globals: crosshair, trigger cooldown, wave, score -------------------
    // The crosshair and trigger are part of the player model, so the policy must
    // see them to plan around cursor travel and shot pacing. Score is on the HUD
    // and gates bonus cities, so it is decision-relevant, not just the return.
    const Vec2 cross = sim.crosshair();
    w.put(nx(cross.x));
    w.put(ny(cross.y));
    w.put(sim.fire_cooldown() * inv_trigger);
    w.put(static_cast<float>(sim.wave()) * 0.05f);
    w.put(static_cast<float>(sim.score()) * inv_bonus);

    // ---- Events this tick: what the human hears (DESIGN.md §13) --------------
    std::array<float, ObsSpec::event_features> counts{};
    for (const Event& e : sim.events()) {
        const auto k = static_cast<std::size_t>(e.type);
        if (k < counts.size()) {
            counts[k] += 1.0f;
        }
    }
    for (const float c : counts) {
        w.put(c * 0.25f);
    }
}

} // namespace md
