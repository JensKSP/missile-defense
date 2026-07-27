// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
// Runs a contestant over the canonical seed set and prints the metrics that
// decide whether a learned agent has beaten the scripted baseline.
//
//   md_agent_eval [--seeds N] [--seed-offset N] [--max-ticks N]
//                 [--frame-skip N] [--per-episode]
//                 [--policy <file.mdp>] [--action-log <file>]
//
// With no `--policy` this is the M4 baseline, exactly as it always was. With one
// it is that learned policy, through the *same* episode loop, the same event
// tallying and the same `summarize` — which is the whole reason "beat the
// baseline" is a claim rather than two numbers from two programs.
#include "md/agent/eval.hpp"
#include "md/agent/handicap.hpp"
#include "md/agent/heuristic.hpp"
#include "md/agent/policy.hpp"
#include "md/config.hpp"
#include "md/observation.hpp"
#include "md/version.hpp"

#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <print>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace {

constexpr std::size_t canonical_seed_offset = 32;

std::optional<std::uint64_t> parse_u64(std::string_view text, bool allow_zero = false) {
    std::uint64_t value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || (!allow_zero && value == 0u)) {
        return std::nullopt;
    }
    return value;
}

void usage() {
    std::println(stderr, "usage: md_agent_eval [--seeds N] [--seed-offset N] [--max-ticks N] "
                         "[--frame-skip N] [--per-episode] [--skill low|medium|high] "
                         "[--policy FILE.mdp] [--action-log FILE]");
}

int run(int argc, char** argv) {
    std::size_t seed_count = 32;
    std::size_t seed_offset = canonical_seed_offset;
    std::uint64_t max_ticks = 120000;
    bool per_episode = false;
    std::string policy_path;
    // The published baseline unless asked otherwise, so an unqualified run is
    // always the number docs/TRAINING.md quotes.
    md::agent::Skill skill = md::agent::Skill::high;
    // The published handicap is the default: an evaluation without it is not a
    // comparable one, so opting out has to be typed (`--aim-trail 0`).
    md::agent::Handicap handicap = md::agent::canonical_handicap;
    std::string action_log_path;
    md::Config config{}; // defaults, including the 15 Hz decision cadence and 3/s fire

    for (int i = 1; i < argc; ++i) {
        const std::string_view arg{argv[i]};
        if (arg == "--per-episode") {
            per_episode = true;
        } else if (arg == "--seeds" && (i + 1) < argc) {
            const auto value = parse_u64(argv[++i]);
            if (!value.has_value() || *value > std::numeric_limits<std::size_t>::max()) {
                std::println(stderr, "invalid positive integer for --seeds: {}", argv[i]);
                return 2;
            }
            seed_count = static_cast<std::size_t>(*value);
        } else if (arg == "--seed-offset" && (i + 1) < argc) {
            const auto value = parse_u64(argv[++i], true);
            if (!value.has_value() || *value > std::numeric_limits<std::size_t>::max()) {
                std::println(stderr, "invalid non-negative integer for --seed-offset: {}", argv[i]);
                return 2;
            }
            seed_offset = static_cast<std::size_t>(*value);
        } else if (arg == "--max-ticks" && (i + 1) < argc) {
            const auto value = parse_u64(argv[++i]);
            if (!value.has_value()) {
                std::println(stderr, "invalid positive integer for --max-ticks: {}", argv[i]);
                return 2;
            }
            max_ticks = *value;
        } else if (arg == "--frame-skip" && (i + 1) < argc) {
            // The reaction rate, ticks per decision, straight into the sim's own
            // limit: 1 = native 60 Hz, 4 = the neural policy's ~15 Hz. The honest
            // same-reaction-rate knob for comparing the two — the sim enforces it.
            const auto value = parse_u64(argv[++i]);
            if (!value.has_value() || *value > std::numeric_limits<std::uint32_t>::max()) {
                std::println(stderr, "invalid positive integer for --frame-skip: {}", argv[i]);
                return 2;
            }
            config.decision_interval = static_cast<std::uint32_t>(*value);
        } else if (arg == "--skill" && (i + 1) < argc) {
            const std::string_view value{argv[++i]};
            if (value == "low") {
                skill = md::agent::Skill::low;
            } else if (value == "medium") {
                skill = md::agent::Skill::medium;
            } else if (value == "high") {
                skill = md::agent::Skill::high;
            } else {
                std::println(stderr, "unknown --skill '{}' (low, medium, high)", value);
                return 2;
            }
        } else if (arg == "--react-delay" && (i + 1) < argc) {
            // Milliseconds in, ticks out: a person thinks in the former and the
            // simulation only steps in the latter, so the rounding is stated
            // rather than hidden — 75 ms is 4.5 ticks and cannot be had.
            const double ms = std::atof(argv[++i]);
            handicap.reaction_delay = static_cast<std::uint32_t>(
                std::llround(ms / 1000.0 / static_cast<double>(config.dt)));
        } else if (arg == "--aim-trail" && (i + 1) < argc) {
            handicap.aim_trail = static_cast<float>(std::atof(argv[++i]));
        } else if (arg == "--policy" && (i + 1) < argc) {
            policy_path = argv[++i];
        } else if (arg == "--action-log" && (i + 1) < argc) {
            action_log_path = argv[++i];
        } else {
            usage();
            return 2;
        }
    }

    if (seed_offset > std::numeric_limits<std::size_t>::max() - seed_count) {
        std::println(stderr, "seed offset plus count is too large");
        return 2;
    }
    // The contestant. Both go through `run_episode`, so the only thing that
    // differs between a scripted and a learned evaluation is which `Driver` is
    // asked for an action — not the loop, the tallying or the aggregation.
    const md::ObsSpec spec{};
    std::optional<md::agent::Policy> loaded;
    std::unique_ptr<md::agent::Driver> driver;
    if (policy_path.empty()) {
        driver = std::make_unique<md::agent::ScriptedDriver>(skill);
    } else {
        loaded = md::agent::Policy::load(policy_path);
        driver = std::make_unique<md::agent::PolicyDriver>(*loaded, spec);
    }
    std::unique_ptr<md::agent::Driver> handicapped;
    if (handicap.active()) {
        handicapped = std::make_unique<md::agent::HandicappedDriver>(*driver, handicap);
    }
    md::agent::Driver& contestant = handicap.active() ? *handicapped : *driver;

    const std::vector<std::uint64_t> stream = md::agent::default_seeds(seed_offset + seed_count);
    const std::vector<std::uint64_t> seeds(
        stream.begin() + static_cast<std::ptrdiff_t>(seed_offset), stream.end());

    std::println("missile-defense {} — {}", md::version(), contestant.name());
    std::println("{} episodes, cap {} ticks ({:.0f} s of play)", seeds.size(), max_ticks,
                 static_cast<double>(max_ticks) * static_cast<double>(config.dt));
    std::println("seed stream offset {}", seed_offset);
    std::println(
        "decisions every {} tick(s) (~{:.0f} Hz)\n", config.decision_interval,
        1.0 / (static_cast<double>(config.dt) * static_cast<double>(config.decision_interval)));

    // One pass over the seeds, whether or not the per-episode table is printed.
    // The action log — one index per sampled decision — is what the parity e2e
    // holds against the Python evaluator's, so it is only collected when asked
    // for: a canonical run is 30,000 decisions an episode.
    std::vector<md::agent::EpisodeResult> episodes;
    episodes.reserve(seeds.size());
    std::vector<std::uint32_t> action_log;
    if (per_episode) {
        std::println("{:<20} {:>8} {:>5} {:>5} {:>8} {:>7} {:>7} {:>6} {:>6} {:>6}", "seed",
                     "score", "wave", "wvs", "ticks", "cit_ls", "bas_ls", "shots", "kills", "hits");
    }
    for (const std::uint64_t seed : seeds) {
        const md::agent::EpisodeResult r = md::agent::run_episode(
            config, seed, contestant, max_ticks, action_log_path.empty() ? nullptr : &action_log);
        if (per_episode) {
            std::println("{:<20} {:>8} {:>5} {:>5} {:>8} {:>7} {:>7} {:>6} {:>6} {:>6}", r.seed,
                         r.score, r.wave_reached, r.waves_cleared, r.ticks, r.cities_lost,
                         r.bases_lost, r.shots, r.kills, r.hits());
        }
        episodes.push_back(r);
    }
    if (per_episode) {
        std::println();
    }
    if (!action_log_path.empty()) {
        std::ofstream log{action_log_path, std::ios::trunc};
        if (!log) {
            std::println(stderr, "could not write the action log: {}", action_log_path);
            return 1;
        }
        for (const std::uint32_t index : action_log) {
            log << index << '\n';
        }
    }

    const md::agent::Summary s = md::agent::summarize(episodes);
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
