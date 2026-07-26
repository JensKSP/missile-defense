// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
#pragma once

#include "md/agent/heuristic.hpp"
#include "md/config.hpp"
#include "md/entities.hpp"
#include "md/event.hpp"

#include <array>
#include <cstdint>
#include <numeric>
#include <span>
#include <vector>

namespace md::agent {

/// Outcome of one episode. Every count is tallied off the deterministic event
/// stream (or the end state), so a learned policy driven from Python and the
/// scripted `Heuristic` are measured by the exact same bookkeeping.
struct EpisodeResult {
    std::uint64_t seed = 0;
    std::int32_t score = 0;
    std::uint32_t wave_reached = 0;  // the wave in progress when it ended (the "last level")
    std::uint32_t waves_cleared = 0; // waves completed outright
    std::uint64_t ticks = 0;         // survival time; ticks / 60 = seconds of play
    std::uint32_t cities_left = 0;   // cities still standing at the end
    std::uint32_t cities_lost = 0;   // cities destroyed over the episode
    std::uint32_t bases_left = 0;    // batteries still standing at the end
    std::uint32_t bases_lost = 0;    // battery destructions over the episode
    std::uint32_t ammo_left = 0;     // interceptors still loaded in surviving batteries
    std::uint32_t bonus_cities = 0;  // destroyed cities rebuilt at score thresholds
    std::uint32_t mirv_splits = 0;   // MIRV warheads that split mid-descent
    std::uint32_t shots = 0;         // interceptors launched
    std::uint32_t kills = 0;         // threats destroyed by a blast ("targets destroyed")
    // Interceptors binned by how many threats their blast destroyed: 0,1,2,3,4+.
    std::array<std::uint32_t, kills_per_shot_bins> kills_per_shot{};
    bool terminated = false; // false => stopped at the tick cap, still alive

    /// Interceptors whose blast destroyed nothing (== `kills_per_shot[0]`).
    [[nodiscard]] std::uint32_t wasted() const noexcept { return kills_per_shot[0]; }

    /// Interceptors whose blast destroyed at least one threat.
    [[nodiscard]] std::uint32_t hits() const noexcept {
        return std::accumulate(kills_per_shot.begin() + 1, kills_per_shot.end(), 0U);
    }

    /// Kills per interceptor spent — above 1.0 means blasts are catching clusters.
    [[nodiscard]] double accuracy() const noexcept {
        return shots == 0u ? 0.0 : static_cast<double>(kills) / static_cast<double>(shots);
    }

    /// Fraction of *resolved* interceptors that destroyed at least one threat.
    [[nodiscard]] double hit_rate() const noexcept {
        const std::uint32_t resolved = hits() + wasted();
        return resolved == 0u ? 0.0 : static_cast<double>(hits()) / static_cast<double>(resolved);
    }
};

/// Aggregate over a seed set. Scripted and learned drivers use this same
/// implementation on a protocol-matched split, so "beat the baseline" is a
/// concrete claim rather than a vibe.
struct Summary {
    std::size_t episodes = 0;
    double mean_score = 0.0;
    double mean_wave = 0.0;          // last level reached
    double mean_waves_cleared = 0.0; // waves completed outright
    double mean_ticks = 0.0;         // survival time; / 60 = seconds
    double mean_cities_left = 0.0;
    double mean_cities_lost = 0.0;
    double mean_bases_left = 0.0;
    double mean_bases_lost = 0.0;
    double mean_ammo_left = 0.0; // unfired interceptors at the end — ammo held in reserve
    double mean_bonus_cities = 0.0;
    double mean_mirv_splits = 0.0;
    double mean_shots = 0.0;    // interceptors fired
    double mean_kills = 0.0;    // targets destroyed
    double mean_hits = 0.0;     // interceptors that destroyed >=1 threat
    double mean_accuracy = 0.0; // mean kills per interceptor — the documented yardstick
    double mean_hit_rate = 0.0; // mean fraction of interceptors that hit
    std::int32_t min_score = 0;
    std::int32_t max_score = 0;
    std::size_t survived = 0; // episodes that hit the tick cap without dying
    // Kills-per-shot histogram summed over the whole seed set: how every
    // interceptor across the evaluation was spent, 0,1,2,3,4+ threats each.
    std::array<std::uint64_t, kills_per_shot_bins> kills_per_shot{};
};

/// A fixed deterministic seed stream. Protocols select disjoint blocks from it;
/// the prefix alone is not inherently a held-out benchmark.
[[nodiscard]] std::vector<std::uint64_t> default_seeds(std::size_t count = 32);

/// Fold one step's events into a running `EpisodeResult`. Shared between the
/// scripted `run_episode` and the Python-driven `VecEnv` so both count off the
/// same event stream and cannot drift — the same reason `summarize` is shared.
void tally_events(EpisodeResult& result, std::span<const Event> events) noexcept;

/// Bin every still-active blast by its kills at episode close, so the
/// kills-per-shot histogram accounts for interceptors whose blast had not yet
/// expired when the episode ended — the final wave especially. Without it the
/// histogram, and the `hits`/`wasted` derived from it, undercount every episode.
/// Shared for the same reason as `tally_events`.
void bin_active_blasts(EpisodeResult& result, std::span<const Blast> blasts) noexcept;

/// Play one episode to termination or `max_ticks`, whichever comes first. The
/// agent's reaction rate is the sim's own `Config::decision_interval` (the core
/// samples a new action once per that many ticks), so there is no per-driver
/// cadence knob here — a scripted agent and a learned policy are throttled
/// identically by the simulation, not by how their driver happens to loop.
[[nodiscard]] EpisodeResult run_episode(const Config& config, std::uint64_t seed,
                                        const Heuristic& agent, std::uint64_t max_ticks = 120000);

/// Aggregate episode outcomes. Split out from `evaluate` so a *learned* agent —
/// which is driven from Python and cannot be a `Heuristic` — is scored by the same
/// function over the same fields, rather than by a reimplementation that might
/// quietly differ. That is what makes "beat the baseline" a claim and not a vibe.
[[nodiscard]] Summary summarize(std::span<const EpisodeResult> episodes);

/// Play every seed and aggregate. The reaction rate comes from the `Config`
/// (`decision_interval`), the same as `run_episode`.
[[nodiscard]] Summary evaluate(const Config& config, std::span<const std::uint64_t> seeds,
                               const Heuristic& agent, std::uint64_t max_ticks = 120000);

} // namespace md::agent
