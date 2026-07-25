// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
#pragma once

#include "md/action.hpp"
#include "md/config.hpp"
#include "md/sim.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <span>
#include <thread>
#include <vector>

namespace md {

/// A batch of independent simulations, stepped across a worker pool.
///
/// The simulations share nothing: each owns its own `Pcg32`, all its state is
/// inline, and `step()` neither allocates nor touches globals. So the only thing
/// between one core and all of them is the work split — no lock on the hot path,
/// and no false sharing, since a `Sim` is ~12 KB, far wider than a cache line.
///
/// Thread count is **detected at run time**, not baked in, because the same build
/// is expected to run on machines with very different core counts. Pass an
/// explicit count to override; pass 0 (the default) to fill the machine.
class VecSim {
  public:
    /// `threads == 0` detects the hardware concurrency.
    explicit VecSim(std::size_t count, const Config& config = {}, unsigned threads = 0);

    /// Usable hardware threads, with a sane floor when the platform will not say.
    [[nodiscard]] static unsigned hardware_threads() noexcept;

    [[nodiscard]] std::size_t size() const noexcept { return sims_.size(); }

    [[nodiscard]] unsigned threads() const noexcept { return threads_; }

    [[nodiscard]] const Config& config() const noexcept { return config_; }

    /// Seed sim *i* with `base_seed + i`, so a batch covers distinct episodes.
    void reset(std::uint64_t base_seed);

    /// Step every simulation once. `actions` and `results` must both be `size()`
    /// long; the work is split across the pool.
    void step(std::span<const Action> actions, std::span<StepResult> results);

    [[nodiscard]] const Sim& operator[](std::size_t index) const { return sims_[index]; }

    [[nodiscard]] std::span<const Sim> sims() const noexcept { return sims_; }

    /// Play `episodes` complete episodes with `policy` driving (anything exposing
    /// `Action act(const Sim&) const`), spread across the pool. Returns how many
    /// finished.
    ///
    /// Workers pull whole episodes from a shared counter rather than lock-stepping
    /// the batch. Episodes differ a lot in length, so a barrier per tick would
    /// leave most cores waiting on the longest survivor; claiming work on demand
    /// keeps every core busy to the end. Templated on the policy so the core keeps
    /// its promise of depending on nothing — `md::agent` depends on core, never the
    /// other way round.
    template <class Policy>
    std::uint64_t run_episodes(const Policy& policy, std::size_t episodes,
                               std::uint64_t base_seed = 0, std::uint64_t max_ticks = 200000) {
        if (episodes == 0) {
            return 0;
        }
        std::atomic<std::size_t> next{0};
        std::atomic<std::uint64_t> finished{0};
        const Config cfg = config_;

        const auto worker = [&] {
            Sim sim{cfg}; // one reused sim per worker: no allocation in the loop
            for (;;) {
                const std::size_t index = next.fetch_add(1, std::memory_order_relaxed);
                if (index >= episodes) {
                    break;
                }
                sim.reset(base_seed + static_cast<std::uint64_t>(index));
                for (std::uint64_t t = 0; t < max_ticks && !sim.terminated(); ++t) {
                    sim.step(policy.act(sim));
                }
                finished.fetch_add(1, std::memory_order_relaxed);
            }
        };

        const unsigned count = std::max(1u, threads_);
        if (count == 1u) {
            worker(); // stay on this thread: measurable, and debuggable
        } else {
            std::vector<std::thread> pool;
            pool.reserve(count);
            for (unsigned i = 0; i < count; ++i) {
                pool.emplace_back(worker);
            }
            for (std::thread& thread : pool) {
                thread.join();
            }
        }
        return finished.load(std::memory_order_relaxed);
    }

  private:
    std::vector<Sim> sims_;
    Config config_{};
    unsigned threads_ = 1;
};

} // namespace md
