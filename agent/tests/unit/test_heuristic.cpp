// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
#include "md/action.hpp"
#include "md/agent/eval.hpp"
#include "md/agent/heuristic.hpp"
#include "md/config.hpp"
#include "md/sim.hpp"
#include "md/vec2.hpp"

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include <vector>

using md::Action;
using md::Config;
using md::Sim;
using md::agent::EpisodeResult;
using md::agent::Heuristic;
using md::agent::Params;

namespace {

/// Play `ticks` of a seeded episode with the agent driving.
Sim play(std::uint64_t seed, int ticks, const Config& config = {}) {
    Sim sim{config};
    sim.reset(seed);
    const Heuristic agent{};
    for (int i = 0; i < ticks && !sim.terminated(); ++i) {
        sim.step(agent.act(sim));
    }
    return sim;
}

} // namespace

TEST_CASE("With nothing in the sky the agent holds its fire", "[unit][agent]") {
    Sim sim;
    sim.reset(0); // tick 0: no threats have spawned yet
    REQUIRE(sim.threats().empty());

    const Action a = Heuristic{}.act(sim);
    REQUIRE_FALSE(a.fire); // no ammo wasted on an empty sky
    REQUIRE_FALSE(a.move);
}

TEST_CASE("The agent engages an incoming threat", "[unit][agent]") {
    Sim sim;
    sim.reset(42);
    sim.step(Action::noop()); // spawn the first threat
    REQUIRE(sim.threats().size() >= 1);

    const Action a = Heuristic{}.act(sim);
    REQUIRE(a.move); // it is at least driving the crosshair at the problem
}

TEST_CASE("The agent is deterministic: same state, same action", "[unit][agent]") {
    Sim sim = play(7, 200);
    const Heuristic agent{};
    const Action first = agent.act(sim);
    const Action second = agent.act(sim);
    REQUIRE(first.aim == second.aim);
    REQUIRE(first.fire == second.fire);
    REQUIRE(first.move == second.move);
    REQUIRE(first.base == second.base);

    // And a snapshot of the same state yields the same action.
    const Sim copy = sim;
    const Action from_copy = agent.act(copy);
    REQUIRE(from_copy.aim == first.aim);
}

TEST_CASE("The agent actually fires and destroys threats", "[unit][agent]") {
    const Sim sim = play(42, 1200);
    // Ammo has been spent and kills scored: it is playing, not idling.
    REQUIRE(sim.score() > 0);
}

TEST_CASE("The agent defends better than doing nothing", "[unit][agent]") {
    // The honest smoke test for a baseline: it must beat the null policy on the
    // metric that matters — cities kept alive.
    constexpr std::uint64_t seed = 2024;
    constexpr int ticks = 3000;

    const Sim played = play(seed, ticks);

    Sim idle;
    idle.reset(seed);
    for (int i = 0; i < ticks && !idle.terminated(); ++i) {
        idle.step(Action::noop());
    }

    const auto alive = [](const Sim& s) {
        std::size_t n = 0;
        for (const auto& city : s.cities()) {
            if (city.alive) {
                ++n;
            }
        }
        return n;
    };
    REQUIRE(alive(played) > alive(idle));
}

TEST_CASE("The agent does not empty every battery into one warhead", "[unit][agent]") {
    // A threat already covered by an in-flight interceptor or a live blast must be
    // skipped, otherwise ammo evaporates. Kills-per-shot is the visible symptom.
    Config cfg;
    cfg.wave_base_threats = 1; // a single warhead, so over-commitment is obvious
    const EpisodeResult r = md::agent::run_episode(cfg, 42, Heuristic{}, 3000);
    REQUIRE(r.shots > 0u);
    REQUIRE(r.kills > 0u);
    REQUIRE(r.shots <= r.kills * 3u); // generous, but rules out dumping the magazine
}

TEST_CASE("Episode results are reproducible", "[unit][agent]") {
    const Config cfg;
    const Heuristic agent{};
    const EpisodeResult a = md::agent::run_episode(cfg, 99, agent, 2000);
    const EpisodeResult b = md::agent::run_episode(cfg, 99, agent, 2000);
    REQUIRE(a.score == b.score);
    REQUIRE(a.shots == b.shots);
    REQUIRE(a.kills == b.kills);
    REQUIRE(a.wave_reached == b.wave_reached);
}

TEST_CASE("The evaluation seed set is fixed and reproducible", "[unit][agent]") {
    const std::vector<std::uint64_t> a = md::agent::default_seeds(16);
    const std::vector<std::uint64_t> b = md::agent::default_seeds(16);
    REQUIRE(a == b);
    REQUIRE(a.size() == 16u);
    // A prefix of a larger set is the same set: adding seeds never invalidates
    // previously published numbers.
    const std::vector<std::uint64_t> longer = md::agent::default_seeds(32);
    REQUIRE(std::vector<std::uint64_t>(longer.begin(), longer.begin() + 16) == a);
}

TEST_CASE("evaluate aggregates over the seed set", "[unit][agent]") {
    const Config cfg;
    const std::vector<std::uint64_t> seeds = md::agent::default_seeds(4);
    const md::agent::Summary s = md::agent::evaluate(cfg, seeds, Heuristic{}, 2000);
    REQUIRE(s.episodes == 4u);
    REQUIRE(s.mean_score > 0.0);
    REQUIRE(s.min_score <= s.max_score);
}

TEST_CASE("An episode's kills, shots and histogram reconcile", "[unit][agent]") {
    const Config cfg;
    const EpisodeResult r = md::agent::run_episode(cfg, 7, Heuristic{}, 4000);
    REQUIRE(r.shots > 0u);
    REQUIRE(r.kills > 0u);

    // The histogram accounts for every interceptor that detonated: bin 0 is the
    // wasted shots, bins 1.. the ones that destroyed something.
    std::uint32_t detonated = 0;
    for (const std::uint32_t count : r.kills_per_shot) {
        detonated += count;
    }
    REQUIRE(r.wasted() == r.kills_per_shot[0]);
    REQUIRE(r.hits() == detonated - r.kills_per_shot[0]);
    // Interceptors still in flight at the end became no blast, so are counted in
    // neither bin — the histogram can only be shorter than the shots fired.
    REQUIRE(detonated <= r.shots);

    // With nothing in the top ("4+") bin, the kill count is recoverable exactly
    // from the histogram, which proves the two tallies agree rather than drift.
    if (r.kills_per_shot.back() == 0u) {
        std::uint32_t weighted = 0;
        for (std::size_t bin = 0; bin < r.kills_per_shot.size(); ++bin) {
            weighted += static_cast<std::uint32_t>(bin) * r.kills_per_shot[bin];
        }
        REQUIRE(weighted == r.kills);
    }

    REQUIRE(r.cities_left <= md::max_cities);
    REQUIRE(r.bases_left <= md::base_count);
}

TEST_CASE("summarize sums the histogram and averages the statistics", "[unit][agent]") {
    const Config cfg;
    const Heuristic agent{};
    const std::vector<std::uint64_t> seeds = md::agent::default_seeds(4);
    std::vector<EpisodeResult> episodes;
    episodes.reserve(seeds.size());
    for (const std::uint64_t seed : seeds) {
        episodes.push_back(md::agent::run_episode(cfg, seed, agent, 4000));
    }
    const md::agent::Summary sum = md::agent::summarize(episodes);

    // The aggregate histogram is exactly the per-episode ones added together.
    std::array<std::uint64_t, md::kills_per_shot_bins> expected{};
    for (const EpisodeResult& episode : episodes) {
        for (std::size_t bin = 0; bin < expected.size(); ++bin) {
            expected[bin] += episode.kills_per_shot[bin];
        }
    }
    REQUIRE(sum.kills_per_shot == expected);
    REQUIRE(sum.mean_ticks > 0.0);
    REQUIRE(sum.mean_shots > 0.0);
    REQUIRE(sum.mean_kills > 0.0);
    REQUIRE(sum.mean_hits <= sum.mean_shots);
    REQUIRE(sum.mean_cities_left <= static_cast<double>(md::max_cities));
}

TEST_CASE("Throttling the agent's decisions really holds its action", "[unit][agent]") {
    // A scripted agent that re-aims every tick is being compared against a neural
    // policy that decides every fourth, so part of any gap between them is
    // reaction speed rather than tactics. `frame_skip` removes that part by
    // holding the scripted agent's action the same way.
    //
    // That the throttle *works* is only visible when it is turned up far enough
    // to hurt: at one decision a second the agent is aiming at where the threats
    // were, and the score collapses. That collapse is the evidence the action is
    // genuinely held rather than quietly recomputed.
    const Config cfg;
    const Heuristic agent{};
    const std::vector<std::uint64_t> seeds = md::agent::default_seeds(4);
    const md::agent::Summary native = md::agent::evaluate(cfg, seeds, agent, 3000, 1);
    const md::agent::Summary crippled = md::agent::evaluate(cfg, seeds, agent, 3000, 60);
    REQUIRE(crippled.mean_score < native.mean_score / 4.0);

    // And the point of the flag: at the policy's own ~15 Hz the baseline is
    // essentially undiminished, so a learned agent held to that rate is being
    // compared against a fair opponent rather than a hobbled one.
    const md::agent::Summary policy_rate = md::agent::evaluate(cfg, seeds, agent, 3000, 4);
    REQUIRE(policy_rate.mean_score > native.mean_score * 0.9);

    // Deterministic at every rate, or a comparison between two of them is noise.
    REQUIRE(md::agent::evaluate(cfg, seeds, agent, 3000, 4).mean_score == policy_rate.mean_score);
    // Zero would step nothing; it means "every tick", which is the default.
    REQUIRE(md::agent::evaluate(cfg, seeds, agent, 3000, 0).mean_score == native.mean_score);
}

TEST_CASE("The agent obeys the player model like any other driver", "[unit][agent]") {
    // The crosshair cap and trigger interval live in Sim::step, so the agent
    // cannot opt out of them: its shots are paced exactly like a human's.
    Config cfg;
    cfg.fire_interval = 0.5f; // 30 ticks between launches
    const EpisodeResult r = md::agent::run_episode(cfg, 42, Heuristic{}, 600);
    REQUIRE(r.shots <= 600u / 30u + 1u);
}

TEST_CASE("Params are honoured", "[unit][agent]") {
    Params params;
    params.city_value = 99.0f;
    const Heuristic agent{params};
    REQUIRE(agent.params().city_value == 99.0f);
}
