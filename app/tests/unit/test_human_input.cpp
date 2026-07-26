// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: OpenAI Codex
#include "human_input.hpp"
#include "md/action.hpp"
#include "md/config.hpp"
#include "md/sim.hpp"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("A human click waits for the next sampled action and is consumed once",
          "[unit][app][input]") {
    md::Config cfg;
    cfg.decision_interval = 4;
    cfg.aim_max_speed = 0.0f;
    md::Sim sim{cfg};
    md::HumanFireLatch click;

    // The first tick is a decision boundary. Put the simulation just after it,
    // which is where an edge-triggered click used to be cleared and lost.
    sim.step(md::Action::noop());
    REQUIRE(sim.tick() == 1u);
    click.request();

    for (int tick = 1; tick < 4; ++tick) {
        md::Action action = md::Action::aim_at(sim.bases()[0].pos);
        click.apply(sim, action, md::BaseId::Alpha);
        CHECK_FALSE(action.fire);
        CHECK(click.pending());
        sim.step(action);
        CHECK(sim.bases()[0].ammo == cfg.ammo_per_base);
    }

    REQUIRE(sim.samples_action_this_tick());
    md::Action accepted = md::Action::aim_at(sim.bases()[0].pos);
    click.apply(sim, accepted, md::BaseId::Alpha);
    CHECK(accepted.fire);
    CHECK_FALSE(click.pending());
    sim.step(accepted);
    CHECK(sim.bases()[0].ammo == cfg.ammo_per_base - 1u);

    // The driver presents no second edge after the sampled click.
    md::Action next = md::Action::aim_at(sim.bases()[0].pos);
    click.apply(sim, next, md::BaseId::Alpha);
    CHECK_FALSE(next.fire);
}
