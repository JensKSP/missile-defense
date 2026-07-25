// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
#pragma once

#include "md/action.hpp"

namespace md {
class Sim;
}

namespace md::agent {

/// Tunables for the scripted baseline. Defaults were chosen to be reasonable, not
/// optimal — the point of the baseline is to be a *fair* yardstick, not to be
/// unbeatable.
struct Params {
    float city_value = 3.0f;     // a threat that will destroy a live city
    float base_value = 2.0f;     // ... a live battery
    float stray_value = 0.15f;   // ... only already-ruined ground: worth little
    float cluster_bonus = 1.2f;  // per *extra* threat caught in the same blast
    float urgency_weight = 1.5f; // preference for threats about to land
    float safety_margin = 0.1f;  // seconds of slack required before impact
    float ground_guard = 4.0f;   // ignore threats below this altitude: too late
};

/// A hand-coded Missile Command player — the M4 baseline the learned agent must
/// beat. No learning, no tuning against a test set, deterministic, and
/// allocation-free.
///
/// **It is deliberately held to the same information as the neural policy.** It
/// reads only what `md::encode` exposes (positions, velocities, ammo, cooldowns,
/// the crosshair) and never touches the simulation's internal bookkeeping — in
/// particular not `Threat::target_index`, which would tell it for free which city
/// a warhead is aimed at. It infers that from the trajectory, exactly as a human
/// or a policy must. Otherwise "beat the baseline" would be an unfair race.
///
/// It is also subject to the same player model (crosshair travel, trigger
/// interval), because those live in `Sim::step`, not in the driver.
///
/// Strategy, per tick:
///  1. Discard threats already doomed by an in-flight interceptor or a live blast
///     — no double-spending ammo.
///  2. Score every (threat, battery) pair by what the threat would destroy, how
///     many *other* threats one blast would take with it, how soon it lands, and
///     how long the shot takes to set up (cursor travel included).
///  3. Engage the best pair, which steers the crosshair and fires on arrival.
///
/// Scoring is a pure function of the observable state, so target commitment falls
/// out for free: the shot you are already aiming at is the cheapest to take, which
/// keeps the crosshair from oscillating between rivals.
class Heuristic {
  public:
    Heuristic() noexcept = default;

    explicit Heuristic(Params params) noexcept : params_{params} {}

    /// The action for this tick. Deterministic and allocation-free.
    [[nodiscard]] Action act(const Sim& sim) const noexcept;

    [[nodiscard]] const Params& params() const noexcept { return params_; }

  private:
    Params params_{};
};

} // namespace md::agent
