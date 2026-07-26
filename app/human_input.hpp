// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: OpenAI Codex
#pragma once

#include "md/action.hpp"
#include "md/config.hpp"
#include "md/sim.hpp"

namespace md {

/// Turns an edge-triggered human click into an action the paced simulation
/// cannot miss.
///
/// A click may arrive on any rendered frame, while `Sim` samples a new action
/// only once per `Config::decision_interval`. Keep the click pending through
/// unsampled ticks, present it on the next decision tick, then consume it once.
class HumanFireLatch {
  public:
    void request() noexcept { pending_ = true; }

    void clear() noexcept { pending_ = false; }

    [[nodiscard]] bool pending() const noexcept { return pending_; }

    void apply(const Sim& sim, Action& action, BaseId base) noexcept {
        if (!pending_ || !sim.samples_action_this_tick()) {
            return;
        }
        action.fire = true;
        action.base = base;
        pending_ = false;
    }

  private:
    bool pending_ = false;
};

} // namespace md
