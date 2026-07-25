// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
#include "md/vec_sim.hpp"

#include "md/action.hpp"
#include "md/config.hpp"
#include "md/sim.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <thread>
#include <vector>

namespace md {

unsigned VecSim::hardware_threads() noexcept {
    const unsigned detected = std::thread::hardware_concurrency();
    return detected == 0u ? 1u : detected; // the standard is allowed to say "no idea"
}

VecSim::VecSim(std::size_t count, const Config& config, unsigned threads)
    : config_{config}, threads_{threads == 0u ? hardware_threads() : threads} {
    sims_.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        sims_.emplace_back(config);
    }
    reset(0);
}

void VecSim::reset(std::uint64_t base_seed) {
    for (std::size_t i = 0; i < sims_.size(); ++i) {
        sims_[i].reset(base_seed + static_cast<std::uint64_t>(i));
    }
}

void VecSim::step(std::span<const Action> actions, std::span<StepResult> results) {
    const std::size_t count = sims_.size();
    if (actions.size() < count || results.size() < count || count == 0) {
        return; // caller mis-sized the buffers: do nothing rather than run off the end
    }

    const auto chunk = [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            results[i] = sims_[i].step(actions[i]);
        }
    };

    const unsigned workers = std::min(static_cast<unsigned>(count), std::max(1u, threads_));
    if (workers == 1u) {
        chunk(0, count);
        return;
    }

    // Contiguous slices: each worker owns a disjoint run of sims, and a Sim is far
    // wider than a cache line, so neighbouring workers never share one.
    std::vector<std::thread> pool;
    pool.reserve(workers);
    const std::size_t per = (count + workers - 1u) / workers;
    for (unsigned w = 0; w < workers; ++w) {
        const std::size_t begin = std::min(count, static_cast<std::size_t>(w) * per);
        const std::size_t end = std::min(count, begin + per);
        if (begin < end) {
            pool.emplace_back(chunk, begin, end);
        }
    }
    for (std::thread& thread : pool) {
        thread.join();
    }
}

} // namespace md
