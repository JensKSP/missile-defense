// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
#include "md/intercept.hpp"

#include "md/action.hpp"
#include "md/config.hpp"
#include "md/entities.hpp"
#include "md/observation.hpp"
#include "md/sim.hpp"
#include "md/vec2.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>

namespace md {

namespace {

constexpr int max_iterations = 24;   // fixed-point steps; converges in ~4 in practice
constexpr float converged = 1.0e-4f; // seconds
constexpr float never = 1.0e9f;      // "does not happen" sentinel, in seconds
constexpr float min_altitude = 0.5f; // do not bother intercepting at ground level

} // namespace

Intercept solve_intercept(const Sim& sim, BaseId base, const Threat& threat) noexcept {
    Intercept result{};
    const Config& cfg = sim.config();
    const auto bases = sim.bases();
    const auto index = static_cast<std::size_t>(base);
    if (index >= bases.size() || cfg.interceptor_speed <= 0.0f) {
        return result;
    }
    const Vec2 origin = bases[index].pos;
    const Vec2 cursor = sim.crosshair();

    // How long until the threat is on the ground — nothing later than this is worth
    // solving for, and it is the feasibility deadline.
    const float ground_time =
        (threat.velocity.y < 0.0f) ? (threat.pos.y / -threat.velocity.y) : never;

    const float inv_fly = 1.0f / cfg.interceptor_speed;
    const bool instant_aim = cfg.aim_max_speed <= 0.0f;
    const float inv_aim = instant_aim ? 0.0f : (1.0f / cfg.aim_max_speed);

    // Fixed point on t = aim(X(t)) + fly(X(t)). Seeded at 0, which is the shot we
    // could take if the threat stood still; each pass pushes t out to account for
    // how far the threat has fallen in the meantime.
    float t = 0.0f;
    for (int i = 0; i < max_iterations; ++i) {
        const Vec2 x = threat.pos + (threat.velocity * t);
        const float aim = distance(x, cursor) * inv_aim;
        const float fly = distance(x, origin) * inv_fly;
        const float next = aim + fly;
        const bool done = std::abs(next - t) < converged;
        t = next;
        if (done) {
            break;
        }
        if (t > ground_time) {
            break; // diverging past the deadline; the check below rejects it
        }
    }

    const Vec2 point = threat.pos + (threat.velocity * t);
    result.point = point;
    result.aim_time = distance(point, cursor) * inv_aim;
    result.fly_time = distance(point, origin) * inv_fly;
    result.total_time = t;
    result.feasible = (t <= ground_time) && (point.y >= min_altitude) && (point.x >= 0.0f) &&
                      (point.x <= cfg.world_width) && (point.y <= cfg.world_height);
    return result;
}

Action engage(const Sim& sim, BaseId base, std::uint32_t threat_slot) noexcept {
    const auto threats = sim.threats();
    if (threat_slot >= threats.size()) {
        return Action::noop();
    }
    const Intercept plan = solve_intercept(sim, base, threats[threat_slot]);
    if (!plan.feasible) {
        return Action::noop(); // unreachable: do not waste ammo or drag the cursor
    }

    // Fire on the tick the crosshair will actually arrive. `move_crosshair` runs
    // before `try_fire` within a step, so "within one tick of travel" means the
    // shot leaves from the right place. With instant aim it always arrives.
    const Config& cfg = sim.config();
    const float reach = cfg.aim_max_speed * cfg.dt;
    const bool arrives =
        (cfg.aim_max_speed <= 0.0f) || (distance(sim.crosshair(), plan.point) <= reach + converged);

    Action action = Action::aim_at(plan.point);
    if (arrives) {
        action.fire = true;
        action.base = base;
    }
    return action;
}

Action decode_action(const Sim& sim, const ObsSpec& spec, std::uint32_t index) noexcept {
    if (index == 0u || spec.threats == 0u || index >= action_count(spec)) {
        return Action::noop();
    }
    const std::uint32_t k = index - 1u;
    const auto base = static_cast<BaseId>(k / spec.threats);
    return engage(sim, base, k % spec.threats);
}

void action_mask(const Sim& sim, const ObsSpec& spec, std::span<bool> out) noexcept {
    const std::uint32_t n = action_count(spec);
    const auto threats = sim.threats();
    const auto bases = sim.bases();
    for (std::uint32_t i = 0; i < n && static_cast<std::size_t>(i) < out.size(); ++i) {
        if (i == 0u) {
            out[0] = true; // NoOp is always available
            continue;
        }
        const std::uint32_t k = i - 1u;
        const auto base = static_cast<std::size_t>(k / spec.threats);
        const std::uint32_t slot = k % spec.threats;
        out[static_cast<std::size_t>(i)] = (base < bases.size()) && bases[base].alive &&
                                           (bases[base].ammo > 0u) && (slot < threats.size());
    }
}

} // namespace md
