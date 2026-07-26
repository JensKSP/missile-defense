// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
#pragma once

#include "md/action.hpp"

#include <cstdint>

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

    /// How far ahead, in seconds, the agent keeps track of its own shots.
    ///
    /// **This one knob is ammunition discipline, and it is worth almost the
    /// entire score.** A threat that one of its interceptors will catch is
    /// skipped only if that interceptor detonates within this many seconds;
    /// beyond it the agent has forgotten it fired, and spends a second missile.
    /// At zero it tracks nothing in flight and wastes over half its ammunition.
    ///
    /// Graded rather than boolean on purpose. The two obvious switches — track
    /// interceptors or not, react to live blasts or not — both turned out to be
    /// nearly all-or-nothing: blasts last 0.9 s and are few, so seeing them is
    /// worth about 1,000 points, while tracking missiles in flight is worth
    /// about 78,000. A difficulty ladder needs a dial, and this is the one that
    /// has a middle: "how long is your memory" degrades smoothly where "do you
    /// have one" does not.
    float coverage_horizon = 1.0e9f;

    /// Skip a threat that a blast burning *right now* is going to catch. Cheap
    /// — the explosion is on screen — and worth about 1,000 points, so it is
    /// switched off only at the very bottom of the ladder.
    bool avoid_blast_double_spend = true;
};

/// How well the scripted agent plays. The three are *defined* by which of its
/// two deliberate behaviours are switched on, rather than by tuned magic
/// numbers, so each step down removes one identifiable idea and the score
/// difference is attributable to it.
///
/// **`Skill::high` is the published baseline** and its `Params` are exactly the
/// struct's defaults — `docs/TRAINING.md`'s yardstick is that agent and no
/// other. The lower two exist to make the game approachable and to price the
/// behaviours; they are not benchmarks unless separately measured and named.
/// `Skill::medium`'s memory, in seconds. Swept against the canonical block; the
/// value is empirical and has no meaning beyond "this is where the midpoint
/// came out". Re-sweep it if the interceptor speed or the world box changes,
/// because it is a duration and both of those move what a duration is worth.
///
/// Be aware that the response is a cliff, not a slope: 0.30 s scores ~34k and
/// 0.40 s ~85k, because it is crossing the flight time of a typical interceptor
/// and either the agent remembers a shot before it lands or it does not. 0.36
/// sits deliberately on the shoulder rather than the edge.
inline constexpr float medium_coverage_horizon = 0.36f;

enum class Skill : std::uint8_t {
    /// No ammunition discipline at all: re-engages threats that are already
    /// dead, whether the blast is on screen or its own missile is inbound.
    low,
    /// Reacts to what it can *see* — it will not fire into a blast already
    /// burning — but does not track the missiles it has in flight, so it
    /// double-spends on anything whose interceptor has not detonated yet.
    /// Nor does it wait for a spread to converge.
    medium,
    /// Everything on: the M4 baseline.
    high,
};

/// The tunables for one skill. `high` returns the defaults unchanged.
///
/// The ladder is built by removing behaviours in order of what they are worth,
/// **measured, not guessed** — see docs/ROADMAP.md for the three canonical
/// scores. Tracking your own in-flight missiles is the expensive one; waiting
/// for MIRV spreads is worth surprisingly little, which is why it goes first.
[[nodiscard]] constexpr Params params_for(Skill skill) noexcept {
    Params params{};
    if (skill == Skill::high) {
        return params;
    }
    // Waiting for a spread to converge is what turns one interceptor into two
    // or three kills; at zero the agent simply takes the cheapest shot it can.
    params.cluster_bonus = 0.0f;
    if (skill == Skill::medium) {
        // Calibrated, not guessed: swept against the canonical block to land
        // near the midpoint between the two ends. See docs/ROADMAP.md.
        params.coverage_horizon = medium_coverage_horizon;
        return params;
    }
    params.coverage_horizon = 0.0f;
    params.avoid_blast_double_spend = false;
    return params;
}

/// A hand-coded Missile Defense player — the M4 baseline the learned agent must
/// beat. No learning, no tuning against a test set, deterministic, and
/// allocation-free.
///
/// **It is deliberately held to the same information as the neural policy.** It
/// reads only what `md::encode` exposes (positions, velocities, blast lifetime
/// phase, ammo, cooldowns, the crosshair) and never touches the simulation's
/// internal bookkeeping — in particular not `Threat::target_index`, which would
/// tell it for free which city a warhead is aimed at. It infers that from the
/// trajectory, exactly as a human or a policy must. Otherwise "beat the
/// baseline" would be an unfair race.
///
/// It is also subject to the same player model (15 Hz decisions, crosshair
/// travel, trigger interval), because those live in `Sim::step`, not in the
/// driver.
///
/// Strategy, per sampled decision:
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

    /// The candidate action for this tick. `Sim` samples it at the shared player
    /// cadence and holds the chosen action between samples.
    [[nodiscard]] Action act(const Sim& sim) const noexcept;

    [[nodiscard]] const Params& params() const noexcept { return params_; }

  private:
    Params params_{};
};

} // namespace md::agent
