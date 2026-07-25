// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
#pragma once

#include "md/config.hpp"
#include "md/observation.hpp"
#include "md/replay/recording.hpp"
#include "md/sim.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace md::rl {

/// A batch of environments with the conventions a reinforcement-learning loop
/// expects: discrete actions, frame-skip, auto-reset, and the
/// terminated/truncated distinction.
///
/// Deliberately **not** part of `md::core`. None of this is a rule of the game —
/// it is how a trainer chooses to consume the game — and the core stays a pure
/// simulation that knows nothing about episodes-as-training-data.
///
/// All the heavy methods take raw pointers to caller-owned buffers so the Python
/// layer can hand over NumPy arrays directly: observations are written straight
/// into the training batch, with no intermediate copy.
class VecEnv {
  public:
    VecEnv(std::size_t num_envs, const Config& config, const ObsSpec& spec, unsigned threads,
           unsigned frame_skip, std::uint64_t max_ticks);

    [[nodiscard]] std::size_t num_envs() const noexcept { return sims_.size(); }

    [[nodiscard]] std::size_t obs_size() const noexcept { return spec_.size(); }

    [[nodiscard]] std::uint32_t action_count() const noexcept;

    [[nodiscard]] unsigned threads() const noexcept { return threads_; }

    [[nodiscard]] unsigned frame_skip() const noexcept { return frame_skip_; }

    [[nodiscard]] const Config& config() const noexcept { return config_; }

    /// Seed every environment (env *i* gets `seed + i`) and write the initial
    /// observations. `obs` must hold `num_envs * obs_size` floats.
    void reset(std::uint64_t seed, float* obs);

    /// Advance every environment by `frame_skip` ticks under its action index.
    ///
    /// The action index is re-decoded on each of those ticks against the *current*
    /// state, because an engagement is a steer-then-fire macro: holding the index
    /// means "keep pursuing that target", not "replay the same raw action".
    ///
    /// Rewards are summed across the skipped ticks. An environment that ends is
    /// reset immediately, so `obs` always holds a live state ready for the next
    /// forward pass, while `final_obs` keeps the last observation of the finished
    /// episode — which the learner needs to bootstrap a truncated return.
    void step(const std::int32_t* actions, float* obs, float* final_obs, float* rewards,
              bool* terminated, bool* truncated);

    /// Validity mask over the discrete action space, `num_envs * action_count`
    /// entries. Masking these out saves a policy from having to discover that
    /// firing an empty battery does nothing. `bool` is one byte here and NumPy's
    /// `bool_` is too, so this writes straight into the array with no conversion.
    void action_masks(bool* out) const;

    /// Start or stop logging environment `index`'s action indices.
    ///
    /// Recording one environment out of a batch is the point: a training run wants
    /// the occasional watchable episode, not a copy of every rollout. The log is
    /// four bytes per agent step, so leaving one env recording costs nothing next
    /// to the forward pass.
    void set_recording(std::size_t index, bool on);

    [[nodiscard]] bool is_recording(std::size_t index) const;

    /// Take the last *complete* episode recorded for `index`, if one has finished
    /// since the previous call. Episodes are only handed over whole: a partial log
    /// would replay into a game that stops mid-air.
    [[nodiscard]] std::optional<replay::Recording> take_recording(std::size_t index);

  private:
    void finish_recording(std::size_t index, std::uint64_t next_episode_seed);

    void encode_into(std::size_t index, float* obs) const;
    void run_range(std::size_t begin, std::size_t end, const std::int32_t* actions, float* obs,
                   float* final_obs, float* rewards, bool* terminated, bool* truncated);

    std::vector<Sim> sims_;
    std::vector<std::uint64_t> episode_ticks_;
    // Recording state, one slot per env so the worker ranges stay disjoint.
    std::vector<std::uint8_t> recording_on_; // not vector<bool>: workers write it
    std::vector<std::uint64_t> episode_seed_;
    std::vector<std::vector<std::int32_t>> live_log_;
    std::vector<std::optional<replay::Recording>> finished_;
    Config config_{};
    ObsSpec spec_{};
    unsigned threads_ = 1;
    unsigned frame_skip_ = 1;
    std::uint64_t max_ticks_ = 0;
    std::uint64_t next_seed_ = 0;
};

} // namespace md::rl
