// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
#include "md/agent/eval.hpp"

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

EpisodeResult run_episode(const Config& config, std::uint64_t seed, const Heuristic& agent,
                          std::uint64_t max_ticks) {
    Sim sim{config};
    sim.reset(seed);

    EpisodeResult result{};
    result.seed = seed;

    for (std::uint64_t tick = 0; tick < max_ticks; ++tick) {
        const StepResult step = sim.step(agent.act(sim));
        for (const Event& event : sim.events()) {
            if (event.type == EventType::Fire) {
                ++result.shots;
            } else if (event.type == EventType::ThreatKilled) {
                ++result.kills;
            }
        }
        if (step.terminated) {
            result.terminated = true;
            break;
        }
    }

    result.score = sim.score();
    result.wave_reached = sim.wave();
    result.ticks = sim.tick();
    for (const City& city : sim.cities()) {
        if (city.alive) {
            ++result.cities_left;
        }
    }
    return result;
}

Summary evaluate(const Config& config, std::span<const std::uint64_t> seeds, const Heuristic& agent,
                 std::uint64_t max_ticks) {
    Summary summary{};
    if (seeds.empty()) {
        return summary;
    }

    double score_sum = 0.0;
    double wave_sum = 0.0;
    double cities_sum = 0.0;
    double accuracy_sum = 0.0;
    bool first = true;

    for (const std::uint64_t seed : seeds) {
        const EpisodeResult episode = run_episode(config, seed, agent, max_ticks);
        score_sum += static_cast<double>(episode.score);
        wave_sum += static_cast<double>(episode.wave_reached);
        cities_sum += static_cast<double>(episode.cities_left);
        accuracy_sum += episode.accuracy();
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

    const auto n = static_cast<double>(seeds.size());
    summary.episodes = seeds.size();
    summary.mean_score = score_sum / n;
    summary.mean_wave = wave_sum / n;
    summary.mean_cities_left = cities_sum / n;
    summary.mean_accuracy = accuracy_sum / n;
    return summary;
}

} // namespace md::agent
