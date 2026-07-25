// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
#pragma once

#include "md/action.hpp"
#include "md/config.hpp"
#include "md/entities.hpp"
#include "md/observation.hpp"
#include "md/vec2.hpp"

#include <cstdint>
#include <span>

namespace md {

class Sim;

/// Where and when a shot fired *now* would meet a threat.
struct Intercept {
    Vec2 point{};            // aim here; the blast should appear here
    float aim_time = 0.0f;   // seconds of crosshair travel from where it is now
    float fly_time = 0.0f;   // seconds of interceptor flight
    float total_time = 0.0f; // seconds from now until detonation (aim + fly)
    bool feasible = false;   // false: the threat lands, or reaches the ground, first
};

/// Solve the lead-intercept for `threat` from `base`.
///
/// With the crosshair speed-capped (DESIGN.md §5.1) this is no longer the classic
/// quadratic: the shot cannot be fired until the *cursor* has travelled to the aim
/// point, so the flight does not start at t = 0. The point must satisfy
///
///     t = |X(t) - crosshair| / aim_max_speed + |X(t) - base| / interceptor_speed
///
/// where `X(t) = threat.pos + threat.velocity * t`. Solved by fixed-point
/// iteration, which contracts whenever threats are slower than the cursor and the
/// interceptor — true for every sane config. With `aim_max_speed == 0` the aim term
/// vanishes and this degenerates to the ordinary lead-intercept.
[[nodiscard]] Intercept solve_intercept(const Sim& sim, BaseId base, const Threat& threat) noexcept;

/// Macro-action: steer toward the intercept point for the threat in slot
/// `threat_slot`, and pull the trigger on the tick the crosshair actually arrives.
///
/// This is the bridge from a *discrete* policy ("engage threat i from base j") to
/// the continuous `Action` primitive, and it is shared by the scripted baseline and
/// the learned policy so the two cannot drift apart.
[[nodiscard]] Action engage(const Sim& sim, BaseId base, std::uint32_t threat_slot) noexcept;

/// Size of the discrete action space: NoOp plus one action per (battery, threat slot).
[[nodiscard]] constexpr std::uint32_t action_count(const ObsSpec& spec) noexcept {
    return 1u + (base_count * spec.threats);
}

/// Turn a discrete action index into the `Action` for this tick. Index 0 is NoOp;
/// thereafter index maps to (battery, threat slot) in row-major order.
[[nodiscard]] Action decode_action(const Sim& sim, const ObsSpec& spec,
                                   std::uint32_t index) noexcept;

/// Validity mask over the discrete action space, for masking a policy's logits.
///
/// An action is invalid only when it can never do anything: an empty threat slot,
/// or a dead//out-of-ammo battery. Cooldowns deliberately do **not** mask an action
/// out — while the trigger is recovering, steering toward the next target is still
/// useful work, and `engage` will fire as soon as it legally can.
void action_mask(const Sim& sim, const ObsSpec& spec, std::span<bool> out) noexcept;

} // namespace md
