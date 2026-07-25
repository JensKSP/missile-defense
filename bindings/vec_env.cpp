// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
#include "vec_env.hpp"

#include "md/action.hpp"
#include "md/intercept.hpp"
#include "md/observation.hpp"
#include "md/sim.hpp"
#include "md/vec_sim.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <thread>
#include <vector>

namespace md::rl {

VecEnv::VecEnv(std::size_t num_envs, const Config& config, const ObsSpec& spec, unsigned threads,
               unsigned frame_skip, std::uint64_t max_ticks)
    : config_{config}, spec_{spec},
      threads_{threads == 0u ? VecSim::hardware_threads() : threads},
      frame_skip_{frame_skip == 0u ? 1u : frame_skip}, max_ticks_{max_ticks} {
    sims_.reserve(num_envs);
    for (std::size_t i = 0; i < num_envs; ++i) {
        sims_.emplace_back(config_);
    }
    episode_ticks_.assign(num_envs, 0);
}

std::uint32_t VecEnv::action_count() const noexcept {
    return md::action_count(spec_);
}

void VecEnv::encode_into(std::size_t index, float* obs) const {
    md::encode(sims_[index], spec_, std::span<float>{obs, spec_.size()});
}

void VecEnv::reset(std::uint64_t seed, float* obs) {
    const std::size_t stride = spec_.size();
    for (std::size_t i = 0; i < sims_.size(); ++i) {
        sims_[i].reset(seed + static_cast<std::uint64_t>(i));
        episode_ticks_[i] = 0;
        encode_into(i, obs + (i * stride));
    }
    // Fresh episodes after this batch continue past the seeds just used, so
    // auto-reset never replays an episode the batch has already seen.
    next_seed_ = seed + static_cast<std::uint64_t>(sims_.size());
}

void VecEnv::run_range(std::size_t begin, std::size_t end, const std::int32_t* actions, float* obs,
                       float* final_obs, float* rewards, bool* terminated,
                       bool* truncated) {
    const std::size_t stride = spec_.size();
    for (std::size_t i = begin; i < end; ++i) {
        Sim& sim = sims_[i];
        const auto index = static_cast<std::uint32_t>(std::max(0, actions[i]));

        float reward = 0.0f;
        bool done = false;
        for (unsigned k = 0; k < frame_skip_ && !done; ++k) {
            // Re-decode each tick: an engagement is a steer-then-fire macro, so
            // holding the index means "keep pursuing that target".
            const Action action = decode_action(sim, spec_, index);
            const StepResult result = sim.step(action);
            reward += static_cast<float>(result.reward);
            done = result.terminated;
            ++episode_ticks_[i];
        }

        const bool cut = !done && max_ticks_ > 0 && episode_ticks_[i] >= max_ticks_;
        terminated[i] = done;
        truncated[i] = cut;
        rewards[i] = reward;

        if (done || cut) {
            // Keep the last state of the finished episode — a truncated return has
            // to bootstrap from it — then start the next one straight away so the
            // batch never contains a dead environment.
            encode_into(i, final_obs + (i * stride));
            sim.reset(next_seed_ + static_cast<std::uint64_t>(i));
            episode_ticks_[i] = 0;
        }
        encode_into(i, obs + (i * stride));
    }
}

void VecEnv::step(const std::int32_t* actions, float* obs, float* final_obs, float* rewards,
                  bool* terminated, bool* truncated) {
    const std::size_t count = sims_.size();
    if (count == 0) {
        return;
    }
    const unsigned workers = std::min(static_cast<unsigned>(count), std::max(1u, threads_));
    if (workers == 1u) {
        run_range(0, count, actions, obs, final_obs, rewards, terminated, truncated);
    } else {
        std::vector<std::thread> pool;
        pool.reserve(workers);
        const std::size_t per = (count + workers - 1u) / workers;
        for (unsigned w = 0; w < workers; ++w) {
            const std::size_t begin = std::min(count, static_cast<std::size_t>(w) * per);
            const std::size_t end = std::min(count, begin + per);
            if (begin < end) {
                pool.emplace_back([this, begin, end, actions, obs, final_obs, rewards, terminated,
                                   truncated] {
                    run_range(begin, end, actions, obs, final_obs, rewards, terminated, truncated);
                });
            }
        }
        for (std::thread& thread : pool) {
            thread.join();
        }
    }
    // Advance the auto-reset seed pool past everything this batch could have used.
    next_seed_ += static_cast<std::uint64_t>(count);
}

void VecEnv::action_masks(bool* out) const {
    const auto width = static_cast<std::size_t>(action_count());
    for (std::size_t i = 0; i < sims_.size(); ++i) {
        md::action_mask(sims_[i], spec_, std::span<bool>{out + (i * width), width});
    }
}

} // namespace md::rl
