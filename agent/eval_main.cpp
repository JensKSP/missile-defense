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
#include <numeric>
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
    unsigned frame_skip = 1;

    for (int i = 1; i < argc; ++i) {
        const std::string_view arg{argv[i]};
        if (arg == "--per-episode") {
            per_episode = true;
        } else if (arg == "--seeds" && (i + 1) < argc) {
            seed_count = static_cast<std::size_t>(parse_u64(argv[++i], 32));
        } else if (arg == "--max-ticks" && (i + 1) < argc) {
            max_ticks = parse_u64(argv[++i], 120000);
        } else if (arg == "--frame-skip" && (i + 1) < argc) {
            // Throttle the agent's decision rate: 1 = native 60 Hz, 4 = the
            // neural policy's ~15 Hz, for a same-reaction-rate comparison.
            frame_skip = static_cast<unsigned>(parse_u64(argv[++i], 1));
        } else {
            std::println(stderr, "usage: md_agent_eval [--seeds N] [--max-ticks N] [--frame-skip "
                                 "N] [--per-episode]");
            return 2;
        }
    }

    const md::Config config{};
    const md::agent::Heuristic agent{};
    const std::vector<std::uint64_t> seeds = md::agent::default_seeds(seed_count);

    std::println("missile-defense {} — scripted baseline (M4)", md::version());
    std::println("{} episodes, cap {} ticks ({:.0f} s of play)", seeds.size(), max_ticks,
                 static_cast<double>(max_ticks) * static_cast<double>(config.dt));
    std::println("decisions every {} tick(s) (~{:.0f} Hz)\n", frame_skip,
                 1.0 / (static_cast<double>(config.dt) * static_cast<double>(frame_skip)));

    if (per_episode) {
        std::println("{:<20} {:>8} {:>5} {:>5} {:>8} {:>7} {:>7} {:>6} {:>6} {:>6}", "seed",
                     "score", "wave", "wvs", "ticks", "cit_ls", "bas_ls", "shots", "kills", "hits");
        for (const std::uint64_t seed : seeds) {
            const md::agent::EpisodeResult r =
                md::agent::run_episode(config, seed, agent, max_ticks, frame_skip);
            std::println("{:<20} {:>8} {:>5} {:>5} {:>8} {:>7} {:>7} {:>6} {:>6} {:>6}", r.seed,
                         r.score, r.wave_reached, r.waves_cleared, r.ticks, r.cities_lost,
                         r.bases_lost, r.shots, r.kills, r.hits());
        }
        std::println();
    }

    const md::agent::Summary s = md::agent::evaluate(config, seeds, agent, max_ticks, frame_skip);
    const auto& hist = s.kills_per_shot;
    const auto total_shots =
        std::max<std::uint64_t>(1, std::accumulate(hist.begin(), hist.end(), std::uint64_t{0}));
    const auto pct = [total_shots](std::uint64_t n) {
        return 100.0 * static_cast<double>(n) / static_cast<double>(total_shots);
    };
    std::println("mean score       {:>10.1f}   [{} .. {}]", s.mean_score, s.min_score, s.max_score);
    std::println("survived         {:>10.0f} ticks ({:.1f} s)   {} / {} reached the cap",
                 s.mean_ticks, s.mean_ticks * static_cast<double>(config.dt), s.survived,
                 s.episodes);
    std::println("last wave        {:>10.2f}   ({:.2f} cleared)", s.mean_wave,
                 s.mean_waves_cleared);
    std::println("cities           {:>10.2f} left   {:.2f} lost   {:.2f} rebuilt   (of {})",
                 s.mean_cities_left, s.mean_cities_lost, s.mean_bonus_cities, md::max_cities);
    std::println("bases            {:>10.2f} left   {:.2f} lost   (of {})", s.mean_bases_left,
                 s.mean_bases_lost, md::base_count);
    std::println("ammo unfired     {:>10.2f}   (interceptors still loaded at the end)",
                 s.mean_ammo_left);
    std::println("targets killed   {:>10.2f}   ({:.2f} MIRV splits)", s.mean_kills,
                 s.mean_mirv_splits);
    std::println("shots fired      {:>10.2f}   {:.2f} hit ({:.0f}%)   {:.2f} kills/shot",
                 s.mean_shots, s.mean_hits, 100.0 * s.mean_hit_rate, s.mean_accuracy);
    std::println("kills per shot   0:{} ({:.0f}%)  1:{} ({:.0f}%)  2:{} ({:.0f}%)  3:{} ({:.0f}%)  "
                 "4+:{} ({:.0f}%)",
                 hist[0], pct(hist[0]), hist[1], pct(hist[1]), hist[2], pct(hist[2]), hist[3],
                 pct(hist[3]), hist[4], pct(hist[4]));
    std::println("survived cap     {:>10} / {}", s.survived, s.episodes);
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
