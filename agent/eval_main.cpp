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
#include <exception>
#include <print>
#include <string_view>
#include <vector>

namespace {

std::uint64_t parse_u64(const char* text, std::uint64_t fallback) {
    char* end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    return (end == text || value == 0ULL) ? fallback : static_cast<std::uint64_t>(value);
}

int run(int argc, char** argv) {
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
            std::println(stderr,
                         "usage: md_agent_eval [--seeds N] [--max-ticks N] [--per-episode]");
            return 2;
        }
    }

    const md::Config config{};
    const md::agent::Heuristic agent{};
    const std::vector<std::uint64_t> seeds = md::agent::default_seeds(seed_count);

    std::println("missile-defense {} — scripted baseline (M4)", md::version());
    std::println("{} episodes, cap {} ticks ({:.0f} s of play)\n", seeds.size(), max_ticks,
                 static_cast<double>(max_ticks) * static_cast<double>(config.dt));

    if (per_episode) {
        std::println("{:<20} {:>8} {:>6} {:>7} {:>8} {:>7} {:>6}", "seed", "score", "wave",
                     "cities", "shots", "kills", "k/s");
        for (const std::uint64_t seed : seeds) {
            const md::agent::EpisodeResult r =
                md::agent::run_episode(config, seed, agent, max_ticks);
            std::println("{:<20} {:>8} {:>6} {:>7} {:>8} {:>7} {:>6.2f}", r.seed, r.score,
                         r.wave_reached, r.cities_left, r.shots, r.kills, r.accuracy());
        }
        std::println();
    }

    const md::agent::Summary s = md::agent::evaluate(config, seeds, agent, max_ticks);
    std::println("mean score      {:>10.1f}   [{} .. {}]", s.mean_score, s.min_score, s.max_score);
    std::println("mean wave       {:>10.2f}", s.mean_wave);
    std::println("mean cities left{:>10.2f}  of {}", s.mean_cities_left, md::max_cities);
    std::println("kills per shot  {:>10.2f}", s.mean_accuracy);
    std::println("survived cap    {:>10} / {}", s.survived, s.episodes);
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    // `main` must not let an exception escape (bugprone-exception-escape): the
    // formatting layer can throw, and there is nothing useful above us to catch
    // it. fputs is used here rather than std::println because the handler itself
    // must not be able to throw.
    try {
        return run(argc, argv);
    } catch (const std::exception& error) {
        std::fputs("md_agent_eval: ", stderr);
        std::fputs(error.what(), stderr);
        std::fputs("\n", stderr);
        return 1;
    } catch (...) {
        std::fputs("md_agent_eval: unknown error\n", stderr);
        return 1;
    }
}
