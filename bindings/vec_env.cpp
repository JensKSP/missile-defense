// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
#include "vec_env.hpp"

#include "md/action.hpp"
#include "md/entities.hpp"
#include "md/event.hpp"
#include "md/intercept.hpp"
#include "md/observation.hpp"
#include "md/sim.hpp"
#include "md/vec_sim.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <thread>
#include <vector>

namespace md::rl {

namespace {

// Event features are counts scaled by one quarter in md::encode. VecEnv presents
// one observation per agent decision rather than per simulation tick, so it adds
// every tick's events into that same representation across the skipped window.
void accumulate_events(std::array<float, ObsSpec::event_features>& out,
                       std::span<const Event> events) noexcept {
    for (const Event& event : events) {
        const auto index = static_cast<std::size_t>(event.type);
        if (index < out.size()) {
            out[index] += 0.25f;
        }
    }
}

} // namespace

VecEnv::VecEnv(std::size_t num_envs, const Config& config, const ObsSpec& spec, unsigned threads,
               unsigned frame_skip, std::uint64_t max_ticks)
    : config_{config}, spec_{spec}, threads_{threads == 0u ? VecSim::hardware_threads() : threads},
      frame_skip_{frame_skip == 0u ? 1u : frame_skip}, max_ticks_{max_ticks} {
    // `frame_skip` is the policy's decision cadence, not only a batching
    // optimization. Keep the core player-model limit in lockstep so changing the
    // trainer's cadence cannot quietly give it faster reactions than evaluation.
    config_.decision_interval = frame_skip_;
    sims_.reserve(num_envs);
    for (std::size_t i = 0; i < num_envs; ++i) {
        sims_.emplace_back(config_);
    }
    episode_ticks_.assign(num_envs, 0);
    wasted_.assign(num_envs, 0);
    multi_kills_.assign(num_envs, 0);
    recording_on_.assign(num_envs, 0);
    episode_seed_.assign(num_envs, 0);
    live_log_.resize(num_envs);
    finished_.resize(num_envs);
    live_result_.resize(num_envs);
    finished_result_.resize(num_envs);
}

void VecEnv::begin_episode(std::size_t index, std::uint64_t seed) {
    live_result_[index] = agent::EpisodeResult{};
    live_result_[index].seed = seed;
}

/// Close out the episode, filling the same fields `md::agent::run_episode` does so
/// the two can be aggregated by one function.
void VecEnv::finish_episode(std::size_t index, bool terminated) {
    agent::EpisodeResult& result = live_result_[index];
    const Sim& sim = sims_[index];
    result.score = sim.score();
    result.wave_reached = sim.wave();
    result.ticks = episode_ticks_[index];
    result.terminated = terminated;
    result.cities_left = 0;
    for (const City& city : sim.cities()) {
        if (city.alive) {
            ++result.cities_left;
        }
    }
    result.bases_left = 0;
    result.ammo_left = 0;
    for (const Base& base : sim.bases()) {
        if (base.alive) {
            ++result.bases_left;
            result.ammo_left += base.ammo;
        }
    }
    agent::bin_active_blasts(result, sim.blasts()); // blasts still expanding at the end
    finished_result_[index] = result;
}

std::optional<agent::EpisodeResult> VecEnv::take_episode_result(std::size_t index) {
    if (index >= sims_.size() || !finished_result_[index].has_value()) {
        return std::nullopt;
    }
    std::optional<agent::EpisodeResult> out = finished_result_[index];
    finished_result_[index].reset();
    return out;
}

void VecEnv::set_recording(std::size_t index, bool on) {
    if (index >= sims_.size()) {
        return;
    }
    if (on && recording_on_[index] != 0u) {
        return; // already logging this whole episode; do not erase its prefix
    }
    if (on && episode_ticks_[index] != 0u) {
        throw std::logic_error(
            "recording can only be enabled when the target environment is at episode tick 0; "
            "call reset() before record()");
    }
    recording_on_[index] = on ? 1u : 0u;
    live_log_[index].clear(); // a log only ever covers one whole episode
}

bool VecEnv::is_recording(std::size_t index) const {
    return index < sims_.size() && recording_on_[index] != 0u;
}

std::optional<replay::Recording> VecEnv::take_recording(std::size_t index) {
    if (index >= sims_.size() || !finished_[index].has_value()) {
        return std::nullopt;
    }
    std::optional<replay::Recording> out = std::move(finished_[index]);
    finished_[index].reset();
    return out;
}

/// Seal the episode just ended into a recording, and start the log for the next.
void VecEnv::finish_recording(std::size_t index, std::uint64_t next_episode_seed) {
    replay::Recording recording;
    recording.config = config_;
    recording.spec = spec_;
    recording.seed = episode_seed_[index];
    recording.frame_skip = frame_skip_;

    // A terminal transition or an exact time-limit cutoff can end part-way through
    // the final frame-skip window. The v1 replay format deliberately stores only a
    // fixed frame_skip, so represent that uncommon partial episode as a per-tick
    // action log instead of changing the format or replaying ticks that never ran.
    // The Config keeps the real decision_interval; repeated indices are therefore
    // latched on exactly the same ticks as they were during training.
    const std::uint64_t ticks = episode_ticks_[index];
    const std::uint64_t nominal_ticks = static_cast<std::uint64_t>(live_log_[index].size()) *
                                        static_cast<std::uint64_t>(frame_skip_);
    if (ticks < nominal_ticks) {
        std::vector<std::int32_t> tick_actions;
        tick_actions.reserve(static_cast<std::size_t>(ticks));
        std::uint64_t remaining = ticks;
        for (const std::int32_t action : live_log_[index]) {
            const auto repeats =
                static_cast<std::size_t>(std::min<std::uint64_t>(remaining, frame_skip_));
            tick_actions.insert(tick_actions.end(), repeats, action);
            remaining -= static_cast<std::uint64_t>(repeats);
            if (remaining == 0u) {
                break;
            }
        }
        recording.frame_skip = 1u;
        recording.actions = std::move(tick_actions);
    } else {
        recording.actions = std::move(live_log_[index]);
    }
    finished_[index] = std::move(recording);
    live_log_[index].clear();
    episode_seed_[index] = next_episode_seed;
}

std::uint32_t VecEnv::action_count() const noexcept {
    return md::action_count(spec_);
}

void VecEnv::encode_into(std::size_t index, float* obs,
                         const EventObservation* window_events) const {
    md::encode(sims_[index], spec_, std::span<float>{obs, spec_.size()});
    if (window_events != nullptr) {
        const std::size_t events_at = spec_.size() - ObsSpec::event_features;
        std::ranges::copy(*window_events, obs + events_at);
    }
}

void VecEnv::reset(std::uint64_t seed, float* obs) {
    const std::size_t stride = spec_.size();
    for (std::size_t i = 0; i < sims_.size(); ++i) {
        sims_[i].reset(seed + static_cast<std::uint64_t>(i));
        episode_ticks_[i] = 0;
        episode_seed_[i] = seed + static_cast<std::uint64_t>(i);
        live_log_[i].clear();
        finished_[i].reset(); // a reset discards any episode not yet collected
        finished_result_[i].reset();
        begin_episode(i, episode_seed_[i]);
        encode_into(i, obs + (i * stride));
    }
    // Fresh episodes after this batch continue past the seeds just used, so
    // auto-reset never replays an episode the batch has already seen.
    next_seed_ = seed + static_cast<std::uint64_t>(sims_.size());
}

void VecEnv::reset(std::span<const std::uint64_t> seeds, float* obs) {
    const std::size_t stride = spec_.size();
    std::uint64_t highest = 0;
    for (std::size_t i = 0; i < sims_.size(); ++i) {
        // Fewer seeds than envs would silently evaluate a seed twice, so repeat
        // deliberately only when the caller has given us nothing else to use.
        const std::uint64_t seed = seeds.empty() ? 0 : seeds[i % seeds.size()];
        sims_[i].reset(seed);
        episode_ticks_[i] = 0;
        episode_seed_[i] = seed;
        live_log_[i].clear();
        finished_[i].reset();
        finished_result_[i].reset();
        begin_episode(i, seed);
        encode_into(i, obs + (i * stride));
        highest = std::max(highest, seed);
    }
    // Auto-reset must not replay a seed this batch is already measuring.
    next_seed_ = highest + 1;
}

void VecEnv::shot_stats(std::int32_t* wasted, std::int32_t* multi_kills) const {
    std::ranges::copy(wasted_, wasted);
    std::ranges::copy(multi_kills_, multi_kills);
}

void VecEnv::run_range(std::size_t begin, std::size_t end, const std::int32_t* actions, float* obs,
                       float* final_obs, float* rewards, bool* terminated, bool* truncated) {
    const std::size_t stride = spec_.size();
    for (std::size_t i = begin; i < end; ++i) {
        Sim& sim = sims_[i];
        const auto index = static_cast<std::uint32_t>(std::max(0, actions[i]));
        if (recording_on_[i] != 0u) {
            // Log the clamped index, not the raw one: that is what was played.
            live_log_[i].push_back(static_cast<std::int32_t>(index));
        }

        float reward = 0.0f;
        bool done = false;
        EventObservation window_events{};
        wasted_[i] = 0;
        multi_kills_[i] = 0;
        for (unsigned k = 0;
             k < frame_skip_ && !done && (max_ticks_ == 0u || episode_ticks_[i] < max_ticks_);
             ++k) {
            // Re-decode each tick: an engagement is a steer-then-fire macro, so
            // holding the index means "keep pursuing that target".
            const Action action = decode_action(sim, spec_, index);
            const StepResult result = sim.step(action);
            reward += static_cast<float>(result.reward);
            wasted_[i] += result.wasted;
            multi_kills_[i] += result.multi_kills;
            done = result.terminated;
            ++episode_ticks_[i];
            accumulate_events(window_events, sim.events());
            // The full per-episode tallies, off the same event stream and the same
            // StepResult as md::agent::run_episode — the counting is *shared* code,
            // so the scripted baseline and a learned policy cannot drift apart.
            agent::tally_events(live_result_[i], sim.events());
            for (std::size_t b = 0; b < live_result_[i].kills_per_shot.size(); ++b) {
                live_result_[i].kills_per_shot[b] +=
                    static_cast<std::uint32_t>(result.kills_per_shot[b]);
            }
        }

        const bool cut = !done && max_ticks_ > 0 && episode_ticks_[i] >= max_ticks_;
        terminated[i] = done;
        truncated[i] = cut;
        rewards[i] = reward;

        if (done || cut) {
            // Keep the last state of the finished episode — a truncated return has
            // to bootstrap from it — then start the next one straight away so the
            // batch never contains a dead environment.
            encode_into(i, final_obs + (i * stride), &window_events);
            finish_episode(i, done); // truncation is "still alive", as in M4
            const auto next = next_seed_ + static_cast<std::uint64_t>(i);
            if (recording_on_[i] != 0u) {
                finish_recording(i, next);
            }
            sim.reset(next);
            episode_ticks_[i] = 0;
            begin_episode(i, next);
            encode_into(i, obs + (i * stride));
        } else {
            encode_into(i, obs + (i * stride), &window_events);
        }
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
