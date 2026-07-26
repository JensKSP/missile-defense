// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
#pragma once

#include "md/agent/heuristic.hpp"
#include "md/config.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace md::agent {

/// Outcome of one episode.
struct EpisodeResult {
    std::uint64_t seed = 0;
    std::int32_t score = 0;
    std::uint32_t wave_reached = 0; // the wave in progress when it ended
    std::uint32_t cities_left = 0;
    std::uint64_t ticks = 0;
    std::uint32_t shots = 0; // interceptors launched
    std::uint32_t kills = 0; // threats destroyed by a blast
    bool terminated = false; // false => stopped at the tick cap, still alive

    /// Kills per interceptor spent — above 1.0 means blasts are catching clusters.
    [[nodiscard]] double accuracy() const noexcept {
        return shots == 0u ? 0.0 : static_cast<double>(kills) / static_cast<double>(shots);
    }
};

/// Aggregate over a seed set. This is the shared yardstick: the learned agent is
/// scored with the same function over the same seeds, so "beat the baseline" is a
/// concrete claim rather than a vibe.
struct Summary {
    std::size_t episodes = 0;
    double mean_score = 0.0;
    double mean_wave = 0.0;
    double mean_cities_left = 0.0;
    double mean_accuracy = 0.0;
    std::int32_t min_score = 0;
    std::int32_t max_score = 0;
    std::size_t survived = 0; // episodes that hit the tick cap without dying
};

/// The canonical evaluation seeds. Fixed on purpose: comparable across agents and
/// across time. Do not tune an agent against these — that is overfitting a
/// benchmark.
[[nodiscard]] std::vector<std::uint64_t> default_seeds(std::size_t count = 32);

/// Play one episode to termination or `max_ticks`, whichever comes first.
///
/// `frame_skip` is how many ticks the agent's action is held before it decides
/// again: 1 is its native per-tick (60 Hz) rate; 4 throttles it to the neural
/// policy's decision rate (~15 Hz), which is what makes the two comparable on
/// tactics rather than on reaction speed. 0 means 1. The physics limits in
/// `Sim::step` — the fire interval, the crosshair speed — apply regardless, so
/// this throttles deciding and never the simulation.
[[nodiscard]] EpisodeResult run_episode(const Config& config, std::uint64_t seed,
                                        const Heuristic& agent, std::uint64_t max_ticks = 120000,
                                        unsigned frame_skip = 1);

/// Aggregate episode outcomes. Split out from `evaluate` so a *learned* agent —
/// which is driven from Python and cannot be a `Heuristic` — is scored by the same
/// function over the same fields, rather than by a reimplementation that might
/// quietly differ. That is what makes "beat the baseline" a claim and not a vibe.
[[nodiscard]] Summary summarize(std::span<const EpisodeResult> episodes);

/// Play every seed and aggregate. `frame_skip` throttles the agent's decision
/// rate (see `run_episode`); the default 1 is the native per-tick baseline.
[[nodiscard]] Summary evaluate(const Config& config, std::span<const std::uint64_t> seeds,
                               const Heuristic& agent, std::uint64_t max_ticks = 120000,
                               unsigned frame_skip = 1);

} // namespace md::agent
