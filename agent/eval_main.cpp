// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
// Runs the scripted baseline over the canonical seed set and prints the metrics
// the learned agent will be measured against.
//
//   md_agent_eval [--seeds N] [--max-ticks N] [--per-episode]
#include "md/agent/eval.hpp"
#include "md/agent/heuristic.hpp"
#include "md/config.hpp"
#include "md/version.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string_view>
#include <vector>

namespace {

std::uint64_t parse_u64(const char* text, std::uint64_t fallback) {
    char* end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    return (end == text || value == 0ULL) ? fallback : static_cast<std::uint64_t>(value);
}

} // namespace

int main(int argc, char** argv) {
    std::size_t seed_count = 32;
    std::uint64_t max_ticks = 120000;
    bool per_episode = false;

    for (int i = 1; i < argc; ++i) {
        const std::string_view arg{argv[i]};
        if (arg == "--per-episode") {
            per_episode = true;
        } else if (arg == "--seeds" && (i + 1) < argc) {
            seed_count = static_cast<std::size_t>(parse_u64(argv[++i], 32));
        } else if (arg == "--max-ticks" && (i + 1) < argc) {
            max_ticks = parse_u64(argv[++i], 120000);
        } else {
            std::fprintf(stderr, "usage: md_agent_eval [--seeds N] [--max-ticks N] [--per-episode]\n");
            return 2;
        }
    }

    const md::Config config{};
    const md::agent::Heuristic agent{};
    const std::vector<std::uint64_t> seeds = md::agent::default_seeds(seed_count);

    std::printf("missile-defense %s — scripted baseline (M4)\n", std::string(md::version()).c_str());
    std::printf("%zu episodes, cap %llu ticks (%.0f s of play)\n\n", seeds.size(),
                static_cast<unsigned long long>(max_ticks),
                static_cast<double>(max_ticks) * static_cast<double>(config.dt));

    if (per_episode) {
        std::printf("%-20s %8s %6s %7s %8s %7s %6s\n", "seed", "score", "wave", "cities", "shots",
                    "kills", "k/s");
        for (const std::uint64_t seed : seeds) {
            const md::agent::EpisodeResult r = md::agent::run_episode(config, seed, agent, max_ticks);
            std::printf("%-20llu %8d %6u %7u %8u %7u %6.2f\n",
                        static_cast<unsigned long long>(r.seed), r.score, r.wave_reached,
                        r.cities_left, r.shots, r.kills, r.accuracy());
        }
        std::printf("\n");
    }

    const md::agent::Summary s = md::agent::evaluate(config, seeds, agent, max_ticks);
    std::printf("mean score      %10.1f   [%d .. %d]\n", s.mean_score, s.min_score, s.max_score);
    std::printf("mean wave       %10.2f\n", s.mean_wave);
    std::printf("mean cities left%10.2f  of %u\n", s.mean_cities_left, md::max_cities);
    std::printf("kills per shot  %10.2f\n", s.mean_accuracy);
    std::printf("survived cap    %10zu / %zu\n", s.survived, s.episodes);
    return 0;
}
