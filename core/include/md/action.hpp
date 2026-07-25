// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
#pragma once

#include "md/config.hpp"
#include "md/vec2.hpp"

#include <type_traits>

namespace md {

/// The single control primitive, shared by human and AI.
///
/// The crosshair is *simulation state*, not a free parameter of the action: an
/// action asks to steer it toward `aim` (the sim caps per-tick travel at
/// `Config::aim_max_speed`) and optionally to launch from `base` at wherever the
/// crosshair actually ended up. So naming a far-away point costs **time**, exactly
/// as it does for a hand on a mouse, and shots are paced by `Config::fire_interval`
/// — the limits apply to every driver alike (DESIGN.md §5).
struct Action {
    Vec2 aim{};                  // desired crosshair position (used only when `move`)
    BaseId base = BaseId::Alpha; // battery to launch from (used only when `fire`)
    bool move = false;           // steer the crosshair toward `aim` this tick
    bool fire = false;           // launch at the crosshair's post-move position

    /// Do nothing: the crosshair holds its position.
    [[nodiscard]] static constexpr Action noop() noexcept { return Action{}; }

    /// Steer toward `target` without firing.
    [[nodiscard]] static constexpr Action aim_at(Vec2 target) noexcept {
        return Action{.aim = target, .move = true};
    }

    /// Steer toward `target` and launch from `b`. Note the shot detonates at the
    /// crosshair's position *this* tick, which may still be short of `target`;
    /// repeat the action until the crosshair arrives to hit `target` exactly.
    [[nodiscard]] static constexpr Action fire_at(BaseId b, Vec2 target) noexcept {
        return Action{.aim = target, .base = b, .move = true, .fire = true};
    }

    /// Launch from `b` at the crosshair's current position, without steering.
    [[nodiscard]] static constexpr Action fire_here(BaseId b) noexcept {
        return Action{.base = b, .fire = true};
    }
};

static_assert(std::is_trivially_copyable_v<Action>);

} // namespace md
