// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
#pragma once

#include "md/action.hpp"
#include "md/agent/eval.hpp"
#include "md/protocol.hpp"

#include <cstdint>
#include <deque>
#include <string_view>

namespace md::agent {

/// Human limits, applied to every agent alike.
///
/// This sits **between the simulation and whatever is playing it** — a
/// `Driver`, so the scripted ladder and a learned `.mdp` policy are handicapped
/// by the same code on the same terms. That placement is the point: a handicap
/// that only the scripted agent carried would make "beat the baseline" a race
/// between an agent wearing weights and one that is not.
///
/// **Why a handicap exists at all.** The agents are not superhuman because they
/// see better; they are superhuman because they never mis-click and never
/// forget what they fired. A human scores around 8,000–10,000 on this game and
/// the scripted ladder starts at 19,585, which makes its bottom rung a poor
/// first target for a person. These knobs give back the two limits a person has
/// and an agent does not.
///
/// **Measured, not guessed**, and the measurements are unkind — record them
/// here so the next person does not re-derive them:
///
/// | Handicap | MEDIUM | HIGH/MEDIUM |
/// |---|---:|---:|
/// | none | 62,523 | 1.59x |
/// | target perception delayed 500 ms | 47,057 | 1.94x |
/// | target position noise, sigma 12 | 54,515 | 1.72x |
/// | only the 2 most urgent threats visible | 62,839 | 1.56x |
/// | reaction delay 100 ms | 6,595 | 1.06x |
///
/// The first three barely register: threats fly ballistically, so an agent that
/// can compute `pos + v*t` extrapolates a delay or a jitter straight back out,
/// and there are rarely more than two urgent threats to divide attention
/// between. What *does* bite is anything that makes shots miss — and that is
/// also what flattens the ladder, because the ladder's rungs are ammunition
/// discipline, and remembering a shot is worthless when the shot misses anyway.
///
/// So these knobs trade score against ladder spread, and there is no setting
/// that is free. Pick them with `HIGH/MEDIUM` in view, not only the headline.
struct Handicap {
    /// Ticks between deciding and acting. A person sees, decides, and then moves
    /// a hand; an agent does all three in the same instant.
    ///
    /// The **whole decision** is delayed, aim included, and that is deliberate:
    /// delaying only the trigger costs almost nothing, because the crosshair
    /// goes on tracking in the meantime and the shot still lands where the agent
    /// currently wants it. It is the staleness of the aim that hurts.
    ///
    /// Steep. At 60 Hz: 1 tick (17 ms) takes MEDIUM from 62,523 to 28,648;
    /// 6 ticks (100 ms) to 6,595. Past about 3 ticks all three rungs converge.
    std::uint32_t reaction_delay = 0;

    /// How far the crosshair lags behind where the agent wants it, per decision,
    /// as a fraction in [0, 1). 0 is instant placement — what an agent does and
    /// a hand cannot. 0.88 leaves MEDIUM near 9,100 while HIGH stays at 17,421,
    /// which is the one combination found that reaches human scores and keeps a
    /// ladder worth climbing.
    float aim_trail = 0.0f;

    /// Whether anything is switched on at all.
    [[nodiscard]] constexpr bool active() const noexcept {
        return reaction_delay > 0 || aim_trail > 0.0f;
    }
};

/// The published handicap.
///
/// Everything that plays or scores this game reads it from here —
/// `md_agent_eval`, the training environment, the game's own WATCH AI — because
/// a handicap applied in one of those and not the others produces two ladders
/// with the same names and different numbers. The value lives in
/// `protocol.toml`, which generates `md/protocol.hpp`.
///
/// **Only `aim_trail`.** A reaction delay was measured and rejected: it reaches
/// the same MEDIUM but leaves HIGH just 1.4 sigma above it over 32 seeds, which
/// is no ladder at all. A perception delay was rejected for a different and
/// worse reason — it can only be applied inside the scripted agent's own
/// reasoning, so a learned policy would never carry it, and the rungs would be
/// measured under a handicap the contestant did not have.
inline constexpr Handicap canonical_handicap{.aim_trail = protocol::aim_trail};

/// Wraps any `Driver` and applies `Handicap` to what it does.
///
/// A decorator rather than a change inside each agent, so there is exactly one
/// implementation of "what a human cannot do" and no way for one contestant to
/// be handicapped differently from another. `name()` is passed through with a
/// suffix, because a handicapped result is not the same claim as an unhandicapped
/// one and a table that showed them under the same label would be lying.
class HandicappedDriver final : public Driver {
  public:
    HandicappedDriver(Driver& inner, Handicap handicap)
        : inner_{&inner}, handicap_{handicap}, name_{std::string(inner.name()) + " (handicapped)"} {
    }

    [[nodiscard]] std::string_view name() const noexcept override {
        return handicap_.active() ? name_ : inner_->name();
    }

    [[nodiscard]] std::uint32_t last_index() const noexcept override {
        return inner_->last_index();
    }

    [[nodiscard]] Action act(const Sim& sim) override {
        Action wanted = inner_->act(sim);
        if (!handicap_.active()) {
            return wanted;
        }
        if (handicap_.aim_trail > 0.0f) {
            // Ease toward the intended point instead of arriving at it. Applied
            // before the delay so the two compose the way a hand does: you start
            // moving late *and* you take time to get there.
            const float keep = handicap_.aim_trail;
            wanted.aim.x = crosshair_.x + ((wanted.aim.x - crosshair_.x) * (1.0f - keep));
            wanted.aim.y = crosshair_.y + ((wanted.aim.y - crosshair_.y) * (1.0f - keep));
            crosshair_ = wanted.aim;
        }
        if (handicap_.reaction_delay == 0) {
            return wanted;
        }
        pending_.push_back(wanted);
        if (pending_.size() <= handicap_.reaction_delay) {
            return Action::noop(); // nothing has come back from the hand yet
        }
        const Action due = pending_.front();
        pending_.pop_front();
        return due;
    }

  private:
    Driver* inner_;
    Handicap handicap_;
    std::string name_;
    Vec2 crosshair_{};
    std::deque<Action> pending_;
};

} // namespace md::agent
