// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
#include "md/agent/heuristic.hpp"

#include "md/action.hpp"
#include "md/config.hpp"
#include "md/entities.hpp"
#include "md/intercept.hpp"
#include "md/sim.hpp"
#include "md/vec2.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace md::agent {

namespace {

constexpr float never = 1.0e9f;
constexpr int samples = 6; // path samples when testing "already covered"

/// Seconds until a descending threat reaches the ground, or `never` if it is not
/// descending. Derived from raw state, exactly as a policy would have to.
float time_to_ground(const Threat& threat) noexcept {
    return (threat.velocity.y < 0.0f) ? (threat.pos.y / -threat.velocity.y) : never;
}

Vec2 position_at(const Threat& threat, float t) noexcept {
    return threat.pos + (threat.velocity * t);
}

/// Would `threat` pass within `radius` of `centre` at some point in [t0, t1]?
/// Sampled rather than solved: the exact quadratic buys nothing here and the
/// sampling is what a cautious player does anyway.
bool passes_through(const Threat& threat, Vec2 centre, float radius, float t0, float t1) noexcept {
    if (t1 < t0) {
        return false;
    }
    const float r_sq = radius * radius;
    const float span = t1 - t0;
    for (int i = 0; i <= samples; ++i) {
        const float t = t0 + (span * static_cast<float>(i) / static_cast<float>(samples));
        if (distance_sq(position_at(threat, t), centre) <= r_sq) {
            return true;
        }
    }
    return false;
}

/// Is this threat already dealt with — by a blast burning now, or by an
/// interceptor already on its way to a point it will fall into? Skipping these is
/// what stops the agent from emptying three batteries into one warhead.
bool already_covered(const Sim& sim, const Threat& threat) noexcept {
    const Config& cfg = sim.config();
    const float radius = cfg.blast_max_radius;

    for (const Blast& blast : sim.blasts()) {
        const float remaining = std::max(0.0f, cfg.blast_lifetime - blast.age);
        if (passes_through(threat, blast.center, radius, 0.0f, remaining)) {
            return true;
        }
    }
    if (cfg.interceptor_speed <= 0.0f) {
        return false;
    }
    for (const Interceptor& shot : sim.interceptors()) {
        const float arrive = distance(shot.pos, shot.target) / cfg.interceptor_speed;
        if (passes_through(threat, shot.target, radius, arrive, arrive + cfg.blast_lifetime)) {
            return true;
        }
    }
    return false;
}

/// What this threat will destroy if left alone — inferred from where its current
/// trajectory meets the ground, never from the simulation's target bookkeeping.
float ground_value(const Sim& sim, const Threat& threat, const Params& params) noexcept {
    const float impact = time_to_ground(threat);
    if (impact >= never) {
        return 0.0f;
    }
    const Config& cfg = sim.config();
    const float x = position_at(threat, impact).x;
    // Ground slots are evenly spaced; anything within half a slot of the impact
    // point is what this warhead is about to take out.
    const float slot = cfg.world_width / static_cast<float>(base_count + max_cities);
    const float reach = slot * 0.5f;

    for (const City& city : sim.cities()) {
        if (city.alive && std::abs(city.pos.x - x) <= reach) {
            return params.city_value;
        }
    }
    for (const Base& base : sim.bases()) {
        if (base.alive && std::abs(base.pos.x - x) <= reach) {
            return params.base_value;
        }
    }
    return params.stray_value; // heading for rubble: worth a shot only if nothing else is
}

/// How many threats a blast at `centre` would catch, `t` seconds from now.
/// Waiting for a MIRV spread to converge is worth several separate shots.
float cluster_size(const Sim& sim, Vec2 centre, float t) noexcept {
    const float radius = sim.config().blast_max_radius;
    const float r_sq = radius * radius;
    float count = 0.0f;
    for (const Threat& other : sim.threats()) {
        if (distance_sq(position_at(other, t), centre) <= r_sq) {
            count += 1.0f;
        }
    }
    return count;
}

} // namespace

Action Heuristic::act(const Sim& sim) const noexcept {
    const auto threats = sim.threats();
    const auto bases = sim.bases();

    float best_score = 0.0f;
    std::uint32_t best_slot = 0;
    BaseId best_base = BaseId::Alpha;
    bool found = false;

    for (std::uint32_t slot = 0; slot < threats.size(); ++slot) {
        const Threat& threat = threats[slot];

        const float impact = time_to_ground(threat);
        if (impact >= never || threat.pos.y <= params_.ground_guard) {
            continue; // not descending, or already too low to be worth a blast
        }
        const float value = ground_value(sim, threat, params_);
        if (value <= 0.0f || already_covered(sim, threat)) {
            continue;
        }

        for (std::uint32_t b = 0; b < bases.size(); ++b) {
            const Base& base = bases[b];
            if (!base.alive || base.ammo == 0u) {
                continue;
            }
            const auto id = static_cast<BaseId>(b);
            const Intercept plan = solve_intercept(sim, id, threat);
            if (!plan.feasible) {
                continue;
            }
            // The shot cannot start until both this battery and the trigger are
            // ready, and it has to land before the warhead does.
            const float wait = std::max(base.cooldown_remaining, sim.fire_cooldown());
            if (plan.total_time + wait + params_.safety_margin > impact) {
                continue;
            }

            const float cluster = cluster_size(sim, plan.point, plan.total_time);
            const float payoff = value + (params_.cluster_bonus * std::max(0.0f, cluster - 1.0f));
            const float urgency = 1.0f + (params_.urgency_weight / (impact + 1.0f));
            // Dividing by set-up time is what produces target commitment: whatever
            // the crosshair is already near is cheapest, so it wins ties.
            const float score = (payoff * urgency) / (plan.total_time + wait + 0.1f);

            if (score > best_score) {
                best_score = score;
                best_slot = slot;
                best_base = id;
                found = true;
            }
        }
    }

    if (!found) {
        return Action::noop();
    }
    return engage(sim, best_base, best_slot);
}

} // namespace md::agent
