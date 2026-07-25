// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
//
// `md::rl::VecEnv` is what the training loop sits on, so the properties pinned
// here are the ones a broken rollout would otherwise blame on the learner:
// seeding, the frame-skip contract, the terminated/truncated split, auto-reset,
// and invariance to the worker count.
#include "md/action.hpp"
#include "md/config.hpp"
#include "md/intercept.hpp"
#include "md/observation.hpp"
#include "md/sim.hpp"
#include "vec_env.hpp"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace {

using md::rl::VecEnv;

/// The caller-owned buffers, exactly as the Python layer hands them over.
/// `bool` and not `std::vector<bool>`: the mask/flag arrays are written through a
/// `bool*`, which the bitset specialisation cannot provide.
struct Batch {
    explicit Batch(const VecEnv& env)
        : obs(env.num_envs() * env.obs_size()), final_obs(env.num_envs() * env.obs_size()),
          rewards(env.num_envs()), terminated(std::make_unique<bool[]>(env.num_envs())),
          truncated(std::make_unique<bool[]>(env.num_envs())),
          mask(std::make_unique<bool[]>(env.num_envs() * env.action_count())) {}

    std::vector<float> obs;
    std::vector<float> final_obs;
    std::vector<float> rewards;
    std::unique_ptr<bool[]> terminated;
    std::unique_ptr<bool[]> truncated;
    std::unique_ptr<bool[]> mask;
};

/// One environment's row of an observation batch.
std::span<const float> row(const std::vector<float>& buffer, std::size_t index,
                           std::size_t stride) {
    return std::span<const float>{buffer.data() + (index * stride), stride};
}

std::vector<float> encode_of(const md::Sim& sim, const md::ObsSpec& spec) {
    std::vector<float> out(spec.size());
    md::encode(sim, spec, std::span<float>{out});
    return out;
}

} // namespace

TEST_CASE("reset gives environment i the seed seed+i", "[rl][vec_env]") {
    const md::Config cfg{};
    const md::ObsSpec spec{};
    VecEnv env{4, cfg, spec, 1, 1, 0};
    Batch batch{env};

    env.reset(100, batch.obs.data());

    for (std::size_t i = 0; i < env.num_envs(); ++i) {
        md::Sim reference{cfg};
        reference.reset(100 + static_cast<std::uint64_t>(i));
        const std::vector<float> expected = encode_of(reference, spec);
        const auto actual = row(batch.obs, i, env.obs_size());
        CHECK(std::equal(expected.begin(), expected.end(), actual.begin()));
    }
}

TEST_CASE("a rollout matches the same sim stepped by hand", "[rl][vec_env]") {
    // Pins the frame-skip contract: rewards summed over the window, and the action
    // index re-decoded against the *current* state on each skipped tick.
    const md::Config cfg{};
    const md::ObsSpec spec{};
    constexpr unsigned frame_skip = 4;
    VecEnv env{1, cfg, spec, 1, frame_skip, 0};
    Batch batch{env};
    env.reset(7, batch.obs.data());

    md::Sim reference{cfg};
    reference.reset(7);

    const std::int32_t action = 1; // battery 0, threat slot 0 — exercises decode
    for (int step = 0; step < 50; ++step) {
        env.step(&action, batch.obs.data(), batch.final_obs.data(), batch.rewards.data(),
                 batch.terminated.get(), batch.truncated.get());

        float expected_reward = 0.0f;
        bool done = false;
        for (unsigned k = 0; k < frame_skip && !done; ++k) {
            const md::Action decoded =
                md::decode_action(reference, spec, static_cast<std::uint32_t>(action));
            const md::StepResult result = reference.step(decoded);
            expected_reward += static_cast<float>(result.reward);
            done = result.terminated;
        }

        REQUIRE_FALSE(done); // 200 ticks is far short of a full episode
        CHECK(batch.rewards[0] == expected_reward);
        const std::vector<float> expected = encode_of(reference, spec);
        CHECK(std::equal(expected.begin(), expected.end(), batch.obs.begin()));
    }
}

TEST_CASE("hitting the tick cap truncates rather than terminates", "[rl][vec_env]") {
    const md::Config cfg{};
    const md::ObsSpec spec{};
    constexpr std::uint64_t max_ticks = 3;
    VecEnv env{1, cfg, spec, 1, 1, max_ticks};
    Batch batch{env};
    env.reset(0, batch.obs.data());

    // The state the third step is about to leave behind.
    md::Sim reference{cfg};
    reference.reset(0);

    const std::int32_t noop = 0;
    for (std::uint64_t tick = 1; tick <= max_ticks; ++tick) {
        reference.step(md::decode_action(reference, spec, 0));
        env.step(&noop, batch.obs.data(), batch.final_obs.data(), batch.rewards.data(),
                 batch.terminated.get(), batch.truncated.get());
        CHECK_FALSE(batch.terminated[0]);
        CHECK(batch.truncated[0] == (tick == max_ticks));
    }

    // `final_obs` keeps the finished episode's last state — a truncated return has
    // to bootstrap from it.
    const std::vector<float> expected_final = encode_of(reference, spec);
    CHECK(std::equal(expected_final.begin(), expected_final.end(), batch.final_obs.begin()));

    // ...while `obs` already holds a fresh episode, so the batch never carries a
    // dead environment into the next forward pass. reset(0) over 1 env leaves the
    // seed pool at 1, and each step advances it by the batch size.
    md::Sim fresh{cfg};
    fresh.reset(max_ticks);
    const std::vector<float> expected_next = encode_of(fresh, spec);
    CHECK(std::equal(expected_next.begin(), expected_next.end(), batch.obs.begin()));
}

TEST_CASE("terminated and truncated are never both set", "[rl][vec_env]") {
    const md::Config cfg{};
    const md::ObsSpec spec{};
    VecEnv env{8, cfg, spec, 1, 4, 32};
    Batch batch{env};
    env.reset(3, batch.obs.data());

    const std::vector<std::int32_t> actions(env.num_envs(), 1);
    for (int step = 0; step < 40; ++step) {
        env.step(actions.data(), batch.obs.data(), batch.final_obs.data(), batch.rewards.data(),
                 batch.terminated.get(), batch.truncated.get());
        for (std::size_t i = 0; i < env.num_envs(); ++i) {
            CHECK_FALSE((batch.terminated[i] && batch.truncated[i]));
        }
    }
}

TEST_CASE("the worker count does not change the result", "[rl][vec_env]") {
    // The pool splits the batch into disjoint ranges, so a rollout must be
    // bit-identical however many threads run it — otherwise a training run stops
    // being reproducible the moment it moves to another machine.
    const md::Config cfg{};
    const md::ObsSpec spec{};
    constexpr std::size_t count = 64;

    VecEnv single{count, cfg, spec, 1, 4, 500};
    VecEnv many{count, cfg, spec, 8, 4, 500};
    Batch a{single};
    Batch b{many};
    single.reset(11, a.obs.data());
    many.reset(11, b.obs.data());
    REQUIRE(a.obs == b.obs);

    std::vector<std::int32_t> actions(count);
    for (std::size_t i = 0; i < count; ++i) {
        actions[i] = static_cast<std::int32_t>(i % single.action_count());
    }

    for (int step = 0; step < 30; ++step) {
        single.step(actions.data(), a.obs.data(), a.final_obs.data(), a.rewards.data(),
                    a.terminated.get(), a.truncated.get());
        many.step(actions.data(), b.obs.data(), b.final_obs.data(), b.rewards.data(),
                  b.terminated.get(), b.truncated.get());
        REQUIRE(a.obs == b.obs);
        REQUIRE(a.rewards == b.rewards);
        for (std::size_t i = 0; i < count; ++i) {
            REQUIRE(a.terminated[i] == b.terminated[i]);
            REQUIRE(a.truncated[i] == b.truncated[i]);
        }
    }
}

TEST_CASE("action masks match the per-sim mask", "[rl][vec_env]") {
    const md::Config cfg{};
    const md::ObsSpec spec{};
    VecEnv env{4, cfg, spec, 1, 8, 0};
    Batch batch{env};
    env.reset(21, batch.obs.data());

    CHECK(env.action_count() == md::action_count(spec));

    // Step first, so the batch holds live threats and a spent battery or two —
    // a mask taken at tick 0 would be all-but-empty and prove little.
    const std::vector<std::int32_t> actions(env.num_envs(), 1);
    for (int step = 0; step < 20; ++step) {
        env.step(actions.data(), batch.obs.data(), batch.final_obs.data(), batch.rewards.data(),
                 batch.terminated.get(), batch.truncated.get());
    }
    env.action_masks(batch.mask.get());

    const auto width = static_cast<std::size_t>(env.action_count());
    for (std::size_t i = 0; i < env.num_envs(); ++i) {
        md::Sim reference{cfg};
        reference.reset(21 + static_cast<std::uint64_t>(i));
        // Replay this environment's ticks so the reference sits in the same state.
        for (int step = 0; step < 20; ++step) {
            for (unsigned k = 0; k < env.frame_skip(); ++k) {
                reference.step(md::decode_action(reference, spec, 1));
            }
        }
        auto expected = std::make_unique<bool[]>(width);
        md::action_mask(reference, spec, std::span<bool>{expected.get(), width});
        for (std::size_t a = 0; a < width; ++a) {
            REQUIRE(batch.mask[(i * width) + a] == expected[a]);
        }
    }
}

TEST_CASE("an empty batch is a no-op", "[rl][vec_env]") {
    const md::Config cfg{};
    const md::ObsSpec spec{};
    VecEnv env{0, cfg, spec, 4, 4, 100};
    CHECK(env.num_envs() == 0);
    // No buffers to write, and nothing may dereference them.
    env.reset(0, nullptr);
    env.step(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    SUCCEED("empty batch handled without touching the buffers");
}
