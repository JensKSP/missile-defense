// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
#include "md/agent/eval.hpp"

#include "md/action.hpp"
#include "md/agent/heuristic.hpp"
#include "md/config.hpp"
#include "md/event.hpp"
#include "md/rng.hpp"
#include "md/sim.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace md::agent {

std::vector<std::uint64_t> default_seeds(std::size_t count) {
    // Drawn from the project's own PCG32 so the set is reproducible anywhere,
    // rather than a hand-picked list that might flatter one agent.
    std::vector<std::uint64_t> seeds;
    seeds.reserve(count);
    Pcg32 rng{0xB1A5EDULL};
    for (std::size_t i = 0; i < count; ++i) {
        const auto hi = static_cast<std::uint64_t>(rng.next_u32());
        const auto lo = static_cast<std::uint64_t>(rng.next_u32());
        seeds.push_back((hi << 32U) | lo);
    }
    return seeds;
}

/// Fold one step's events into the running per-episode tallies. This reads the
/// same event stream the app's audio and the agent's observation do, so every
/// count is something the agent could itself perceive — no privileged access to
/// the simulation's internals. Every `EventType` is listed on purpose: a new one
/// must not compile until someone has decided whether it is a statistic.
void tally_events(EpisodeResult& result, std::span<const Event> events) noexcept {
    for (const Event& event : events) {
        switch (event.type) {
        case EventType::Fire:
            ++result.shots;
            break;
        case EventType::ThreatKilled:
            ++result.kills;
            break;
        case EventType::CityLost:
            ++result.cities_lost;
            break;
        case EventType::BaseLost:
            ++result.bases_lost;
            break;
        case EventType::WaveCleared:
            ++result.waves_cleared;
            break;
        case EventType::BonusCity:
            ++result.bonus_cities;
            break;
        case EventType::MirvSplit:
            ++result.mirv_splits;
            break;
        case EventType::Detonate:
        case EventType::GameOver:
        case EventType::WaveStarted:
            break;
        }
    }
}

void bin_active_blasts(EpisodeResult& result, std::span<const Blast> blasts) noexcept {
    for (const Blast& blast : blasts) {
        ++result.kills_per_shot[std::min(blast.kills, kills_per_shot_bins - 1U)];
    }
}

EpisodeResult run_episode(const Config& config, std::uint64_t seed, const Heuristic& agent,
                          std::uint64_t max_ticks, unsigned frame_skip) {
    Sim sim{config};
    sim.reset(seed);

    EpisodeResult result{};
    result.seed = seed;

    const std::uint64_t skip = frame_skip == 0 ? 1 : frame_skip;
    Action action{};
    for (std::uint64_t tick = 0; tick < max_ticks; ++tick) {
        // Decide every `skip` ticks and hold the action between. skip==1 is the
        // native per-tick rate; skip==4 throttles the agent to the neural
        // policy's ~15 Hz, so the two are scored at the same reaction rate.
        if (tick % skip == 0) {
            action = agent.act(sim);
        }
        const StepResult step = sim.step(action);
        tally_events(result, sim.events());
        for (std::size_t b = 0; b < result.kills_per_shot.size(); ++b) {
            result.kills_per_shot[b] += static_cast<std::uint32_t>(step.kills_per_shot[b]);
        }
        if (step.terminated) {
            result.terminated = true;
            break;
        }
    }

    bin_active_blasts(result, sim.blasts()); // count blasts still expanding at the end
    result.score = sim.score();
    result.wave_reached = sim.wave();
    result.ticks = sim.tick();
    for (const City& city : sim.cities()) {
        if (city.alive) {
            ++result.cities_left;
        }
    }
    for (const Base& base : sim.bases()) {
        if (base.alive) {
            ++result.bases_left;
            result.ammo_left += base.ammo;
        }
    }
    return result;
}

Summary summarize(std::span<const EpisodeResult> episodes) {
    Summary summary{};
    if (episodes.empty()) {
        return summary;
    }

    double score_sum = 0.0;
    double wave_sum = 0.0;
    double waves_cleared_sum = 0.0;
    double ticks_sum = 0.0;
    double cities_left_sum = 0.0;
    double cities_lost_sum = 0.0;
    double bases_left_sum = 0.0;
    double bases_lost_sum = 0.0;
    double ammo_left_sum = 0.0;
    double bonus_cities_sum = 0.0;
    double mirv_splits_sum = 0.0;
    double shots_sum = 0.0;
    double kills_sum = 0.0;
    double hits_sum = 0.0;
    double accuracy_sum = 0.0;
    double hit_rate_sum = 0.0;
    bool first = true;

    for (const EpisodeResult& episode : episodes) {
        score_sum += static_cast<double>(episode.score);
        wave_sum += static_cast<double>(episode.wave_reached);
        waves_cleared_sum += static_cast<double>(episode.waves_cleared);
        ticks_sum += static_cast<double>(episode.ticks);
        cities_left_sum += static_cast<double>(episode.cities_left);
        cities_lost_sum += static_cast<double>(episode.cities_lost);
        bases_left_sum += static_cast<double>(episode.bases_left);
        bases_lost_sum += static_cast<double>(episode.bases_lost);
        ammo_left_sum += static_cast<double>(episode.ammo_left);
        bonus_cities_sum += static_cast<double>(episode.bonus_cities);
        mirv_splits_sum += static_cast<double>(episode.mirv_splits);
        shots_sum += static_cast<double>(episode.shots);
        kills_sum += static_cast<double>(episode.kills);
        hits_sum += static_cast<double>(episode.hits());
        accuracy_sum += episode.accuracy();
        hit_rate_sum += episode.hit_rate();
        for (std::size_t b = 0; b < summary.kills_per_shot.size(); ++b) {
            summary.kills_per_shot[b] += episode.kills_per_shot[b];
        }
        if (!episode.terminated) {
            ++summary.survived;
        }
        if (first) {
            summary.min_score = episode.score;
            summary.max_score = episode.score;
            first = false;
        } else {
            summary.min_score = std::min(summary.min_score, episode.score);
            summary.max_score = std::max(summary.max_score, episode.score);
        }
    }

    const auto n = static_cast<double>(episodes.size());
    summary.episodes = episodes.size();
    summary.mean_score = score_sum / n;
    summary.mean_wave = wave_sum / n;
    summary.mean_waves_cleared = waves_cleared_sum / n;
    summary.mean_ticks = ticks_sum / n;
    summary.mean_cities_left = cities_left_sum / n;
    summary.mean_cities_lost = cities_lost_sum / n;
    summary.mean_bases_left = bases_left_sum / n;
    summary.mean_bases_lost = bases_lost_sum / n;
    summary.mean_ammo_left = ammo_left_sum / n;
    summary.mean_bonus_cities = bonus_cities_sum / n;
    summary.mean_mirv_splits = mirv_splits_sum / n;
    summary.mean_shots = shots_sum / n;
    summary.mean_kills = kills_sum / n;
    summary.mean_hits = hits_sum / n;
    summary.mean_accuracy = accuracy_sum / n;
    summary.mean_hit_rate = hit_rate_sum / n;
    return summary;
}

Summary evaluate(const Config& config, std::span<const std::uint64_t> seeds, const Heuristic& agent,
                 std::uint64_t max_ticks, unsigned frame_skip) {
    std::vector<EpisodeResult> episodes;
    episodes.reserve(seeds.size());
    for (const std::uint64_t seed : seeds) {
        episodes.push_back(run_episode(config, seed, agent, max_ticks, frame_skip));
    }
    return summarize(episodes);
}

} // namespace md::agent
