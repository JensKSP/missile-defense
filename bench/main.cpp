// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
// Throughput benchmark for the simulation — how many sims can this machine run?
//
//   md_bench [--episodes N] [--repeat N] [--threads N] [--sample] [--csv]
//
// Reports rates, not wall-clock totals, so numbers are comparable across
// machines. Every result is derived from work that is actually consumed (the
// checksums are printed) so nothing can be optimised away.
#include "md/action.hpp"
#include "md/agent/heuristic.hpp"
#include "md/config.hpp"
#include "md/observation.hpp"
#include "md/sim.hpp"
#include "md/vec_sim.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string_view>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    int episodes = 24;     // episodes per single-threaded measurement
    int repeat = 3;        // best-of, to shake out scheduler noise
    unsigned threads = 0;  // 0 = detect
    bool sample = false;   // run a long workload for an external sampler
    bool csv = false;      // machine-readable
};

template <class Fn> double time_it(Fn&& fn) {
    const auto start = Clock::now();
    fn();
    return std::chrono::duration<double>(Clock::now() - start).count();
}

/// Best-of `repeat` — reports the cleanest run rather than an average polluted
/// by whatever else the machine was doing.
template <class Fn> double best_of(int repeat, Fn&& fn) {
    double best = 0.0;
    for (int i = 0; i < repeat; ++i) {
        const double seconds = time_it(fn);
        if (i == 0 || seconds < best) {
            best = seconds;
        }
    }
    return best;
}

void row(const Options& opt, const char* name, double per_second, const char* unit) {
    if (opt.csv) {
        std::printf("%s,%.0f,%s\n", name, per_second, unit);
    } else {
        std::printf("  %-26s %14.0f  %s/s\n", name, per_second, unit);
    }
}

std::uint64_t parse_u64(const char* text, std::uint64_t fallback) {
    char* end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    return end == text ? fallback : static_cast<std::uint64_t>(value);
}

/// A long, steady workload for a sampling profiler to attach to. There is
/// deliberately no in-code phase timing: `Sim::step` is a pure function of
/// (state, action) and reads no clock, so profiling happens from outside the
/// process — see docs/PERFORMANCE.md.
void run_profiling_workload(const md::Config& config, const md::agent::Heuristic& agent,
                            std::size_t episodes) {
    std::printf("\nprofiling workload: %zu episodes, single-threaded.\n", episodes);
    std::printf("Attach a sampler to this process, e.g.\n"
                "  linux:   perf record -g -- <this binary> --sample\n"
                "  windows: wpr -start CPU -filemode ... wpr -stop md.etl  (view in WPA)\n");
    md::VecSim vec{0, config, 1};
    const std::uint64_t done = vec.run_episodes(agent, episodes, 0);
    std::printf("done: %llu episodes\n", static_cast<unsigned long long>(done));
}

} // namespace

int main(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg{argv[i]};
        if (arg == "--sample") {
            opt.sample = true;
        } else if (arg == "--csv") {
            opt.csv = true;
        } else if (arg == "--episodes" && (i + 1) < argc) {
            opt.episodes = static_cast<int>(parse_u64(argv[++i], 24));
        } else if (arg == "--repeat" && (i + 1) < argc) {
            opt.repeat = static_cast<int>(parse_u64(argv[++i], 3));
        } else if (arg == "--threads" && (i + 1) < argc) {
            opt.threads = static_cast<unsigned>(parse_u64(argv[++i], 0));
        } else {
            std::fprintf(stderr,
                         "usage: md_bench [--episodes N] [--repeat N] [--threads N] "
                         "[--sample] [--csv]\n");
            return 2;
        }
    }

    const md::Config config{};
    const md::agent::Heuristic agent{};
    const unsigned detected = md::VecSim::hardware_threads();

    if (!opt.csv) {
        std::printf("missile-defense throughput benchmark\n");
        std::printf("  hardware threads detected : %u\n", detected);
        std::printf("  episodes per measurement  : %d (best of %d)\n\n", opt.episodes, opt.repeat);
    }

    // Consumed so no measured work can be elided.
    std::uint64_t checksum = 0;

    // ---- 1. The simulation on its own ---------------------------------------
    std::uint64_t sim_ticks = 0;
    const double t_sim = best_of(opt.repeat, [&] {
        sim_ticks = 0;
        for (int e = 0; e < opt.episodes; ++e) {
            md::Sim sim{config};
            sim.reset(static_cast<std::uint64_t>(e));
            while (!sim.terminated()) {
                sim.step(md::Action::noop());
                ++sim_ticks;
            }
            checksum += static_cast<std::uint64_t>(sim.score());
        }
    });

    // ---- 2. Simulation driven by the scripted agent (what `poe eval` does) ---
    std::uint64_t agent_ticks = 0;
    int episodes_done = 0;
    const double t_agent = best_of(opt.repeat, [&] {
        agent_ticks = 0;
        episodes_done = 0;
        for (int e = 0; e < opt.episodes; ++e) {
            md::Sim sim{config};
            sim.reset(static_cast<std::uint64_t>(e));
            while (!sim.terminated()) {
                sim.step(agent.act(sim));
                ++agent_ticks;
            }
            checksum += static_cast<std::uint64_t>(sim.score());
            ++episodes_done;
        }
    });

    // ---- 3. Policy and observation cost in isolation -------------------------
    // Both scale with the number of live threats, so measure on a *busy* field
    // rather than whatever the clock happened to land on: an early, near-empty
    // state would flatter the policy enormously.
    md::Sim probe{config};
    {
        md::Sim scan{config};
        scan.reset(7);
        std::size_t busiest = 0;
        for (int i = 0; i < 40000 && !scan.terminated(); ++i) {
            scan.step(agent.act(scan));
            if (scan.threats().size() > busiest) {
                busiest = scan.threats().size();
                probe = scan; // Sim is a value: snapshotting the peak is a copy
            }
        }
    }
    const std::size_t probe_threats = probe.threats().size();

    constexpr int decisions = 100000;
    const double t_act = best_of(opt.repeat, [&] {
        for (int i = 0; i < decisions; ++i) {
            checksum += agent.act(probe).fire ? 1u : 0u;
        }
    });

    constexpr md::ObsSpec spec;
    std::vector<float> obs(spec.size());
    constexpr int encodes = 100000;
    const double t_encode = best_of(opt.repeat, [&] {
        for (int i = 0; i < encodes; ++i) {
            md::encode(probe, spec, obs);
            checksum += static_cast<std::uint64_t>(obs[0] != 0.0f);
        }
    });

    if (!opt.csv) {
        std::printf("single-threaded\n");
    }
    row(opt, "sim step (no policy)", static_cast<double>(sim_ticks) / t_sim, "ticks");
    row(opt, "sim step + agent", static_cast<double>(agent_ticks) / t_agent, "ticks");
    row(opt, "agent decision", static_cast<double>(decisions) / t_act, "acts");
    row(opt, "observation encode", static_cast<double>(encodes) / t_encode, "obs");
    row(opt, "episodes (agent)", static_cast<double>(episodes_done) / t_agent, "episodes");

    // Split policy vs simulation *on the same trajectory*. Comparing the two rates
    // above would be wrong: undefended episodes die around wave 3 with a nearly
    // empty sky, while agent-driven ones reach wave 16 with far more entities per
    // tick, so their per-tick costs are not comparable. Two clock reads per tick
    // cost something, so read this as a split, not as absolute timings.
    double act_seconds = 0.0;
    double step_seconds = 0.0;
    std::uint64_t split_ticks = 0;
    for (int e = 0; e < std::max(1, opt.episodes / 4); ++e) {
        md::Sim sim{config};
        sim.reset(static_cast<std::uint64_t>(e));
        while (!sim.terminated()) {
            const auto t0 = Clock::now();
            const md::Action action = agent.act(sim);
            const auto t1 = Clock::now();
            checksum += sim.step(action).terminated ? 1u : 0u;
            const auto t2 = Clock::now();
            act_seconds += std::chrono::duration<double>(t1 - t0).count();
            step_seconds += std::chrono::duration<double>(t2 - t1).count();
            ++split_ticks;
        }
    }

    if (!opt.csv) {
        const double total = act_seconds + step_seconds;
        std::printf("\n  busiest state seen: %zu threats; observation %zu floats\n", probe_threats,
                    spec.size());
        std::printf("  same-trajectory split: policy %.0f%% / simulation %.0f%%"
                    "  (%.0f ns + %.0f ns per tick)\n",
                    100.0 * act_seconds / total, 100.0 * step_seconds / total,
                    1.0e9 * act_seconds / static_cast<double>(split_ticks),
                    1.0e9 * step_seconds / static_cast<double>(split_ticks));
    }

    // ---- 4. Parallel throughput ---------------------------------------------
    // The number that matters for training: episodes per second with the machine
    // fully occupied, however many cores it happens to have.
    const unsigned threads = opt.threads != 0 ? opt.threads : detected;
    // Episodes vary a lot in length, so give every worker a healthy queue to pull
    // from. One episode per thread would make the slowest episode set the wall
    // clock and understate scaling badly.
    const std::size_t parallel_episodes = static_cast<std::size_t>(threads) * 8u;
    if (!opt.csv) {
        std::printf("\nparallel — %zu episodes over up to %u threads\n", parallel_episodes,
                    threads);
    }
    std::vector<unsigned> ladder;
    for (unsigned n = 1u; n < threads; n *= 2u) {
        ladder.push_back(n);
    }
    ladder.push_back(threads); // always finish at the full machine

    double single_rate = 0.0;
    for (const unsigned n : ladder) {
        // run_episodes drives its own per-worker Sim, so the batch stays empty here.
        md::VecSim vec{0, config, n};
        std::uint64_t done = 0;
        const double seconds =
            best_of(opt.repeat, [&] { done = vec.run_episodes(agent, parallel_episodes, 0); });
        const double rate = static_cast<double>(done) / seconds;
        if (n == 1u) {
            single_rate = rate;
        }
        char label[80];
        std::snprintf(label, sizeof(label), "episodes @ %2u thread%s", n, n == 1u ? "" : "s");
        if (opt.csv) {
            row(opt, label, rate, "episodes");
        } else {
            std::printf("  %-26s %14.0f  episodes/s  %5.1fx\n", label, rate,
                        single_rate > 0.0 ? rate / single_rate : 1.0);
        }
        checksum += done;
    }

    if (opt.sample) {
        run_profiling_workload(config, agent,
                               static_cast<std::size_t>(opt.episodes) * 4u);
    }
    if (!opt.csv) {
        std::printf("\n(checksum %llu — printed so nothing measured is optimised away)\n",
                    static_cast<unsigned long long>(checksum));
    }
    return 0;
}
