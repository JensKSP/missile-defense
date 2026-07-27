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
#include "md/event.hpp"
#include "md/intercept.hpp"
#include "md/observation.hpp"
#include "md/replay/recording.hpp"
#include "md/sim.hpp"
#include "vec_env.hpp"

#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
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

using EventObservation = std::array<float, md::ObsSpec::event_features>;

void accumulate_events(EventObservation& out, std::span<const md::Event> events) {
    for (const md::Event& event : events) {
        out[static_cast<std::size_t>(event.type)] += 0.25f;
    }
}

void replace_events(std::vector<float>& observation, const EventObservation& events) {
    std::ranges::copy(events, observation.end() - md::ObsSpec::event_features);
}

} // namespace

TEST_CASE("reset gives environment i the seed seed+i", "[rl][vec_env]") {
    const md::Config cfg{};
    const md::ObsSpec spec{};
    VecEnv env{4, cfg, spec, 1, 1, 0};
    Batch batch{env};

    env.reset(100, batch.obs.data());

    for (std::size_t i = 0; i < env.num_envs(); ++i) {
        md::Sim reference{env.config()};
        reference.reset(100 + static_cast<std::uint64_t>(i));
        const std::vector<float> expected = encode_of(reference, spec);
        const auto actual = row(batch.obs, i, env.obs_size());
        CHECK(std::equal(expected.begin(), expected.end(), actual.begin()));
    }
}

TEST_CASE("a rollout matches the same sim stepped by hand", "[rl][vec_env]") {
    // Pins the frame-skip contract: rewards summed over the window, and the action
    // index re-decoded against the *current* state on each skipped tick.
    md::Config cfg{};
    const md::ObsSpec spec{};
    constexpr unsigned frame_skip = 4;
    VecEnv env{1, cfg, spec, 1, frame_skip, 0};
    REQUIRE(env.config().decision_interval == frame_skip);
    Batch batch{env};
    env.reset(7, batch.obs.data());

    cfg.decision_interval = frame_skip;
    md::Sim reference{cfg};
    reference.reset(7);

    const std::int32_t action = 1; // battery 0, threat slot 0 — exercises decode
    for (int step = 0; step < 50; ++step) {
        env.step(&action, batch.obs.data(), batch.final_obs.data(), batch.rewards.data(),
                 batch.terminated.get(), batch.truncated.get());

        float expected_reward = 0.0f;
        bool done = false;
        EventObservation expected_events{};
        for (unsigned k = 0; k < frame_skip && !done; ++k) {
            const md::Action decoded =
                md::decode_action(reference, spec, static_cast<std::uint32_t>(action));
            const md::StepResult result = reference.step(decoded);
            expected_reward += static_cast<float>(result.reward);
            done = result.terminated;
            accumulate_events(expected_events, reference.events());
        }

        REQUIRE_FALSE(done); // 200 ticks is far short of a full episode
        CHECK(batch.rewards[0] == expected_reward);
        std::vector<float> expected = encode_of(reference, spec);
        replace_events(expected, expected_events);
        CHECK(std::equal(expected.begin(), expected.end(), batch.obs.begin()));
    }
}

TEST_CASE("frame skip is the authoritative decision cadence", "[rl][vec_env]") {
    md::Config cfg{};
    cfg.decision_interval = 11; // deliberately disagree with the trainer cadence
    const md::ObsSpec spec{};

    VecEnv env{1, cfg, spec, 1, 3, 0};

    // VecEnv owns a copy: aligning it must neither mutate the caller's Config nor
    // leave the simulation accepting stale actions at a different rate.
    CHECK(cfg.decision_interval == 11u);
    CHECK(env.frame_skip() == 3u);
    CHECK(env.config().decision_interval == env.frame_skip());
}

TEST_CASE("an agent observation includes events from every skipped tick", "[rl][vec_env]") {
    const md::Config cfg{};
    const md::ObsSpec spec{};
    VecEnv env{1, cfg, spec, 1, 4, 0};
    Batch batch{env};
    env.reset(7, batch.obs.data());

    const std::int32_t noop = 0;
    env.step(&noop, batch.obs.data(), batch.final_obs.data(), batch.rewards.data(),
             batch.terminated.get(), batch.truncated.get());

    // WaveStarted is emitted on the first tick after reset. Before window
    // aggregation it disappeared because the fourth tick's empty event block
    // overwrote it before the policy received its next observation.
    const std::size_t events_at = spec.size() - md::ObsSpec::event_features;
    const auto wave_started = static_cast<std::size_t>(md::EventType::WaveStarted);
    CHECK(batch.obs[events_at + wave_started] == 0.25f);
}

TEST_CASE("hitting the tick cap truncates rather than terminates", "[rl][vec_env]") {
    md::Config cfg{};
    const md::ObsSpec spec{};
    constexpr std::uint64_t max_ticks = 3;
    constexpr unsigned frame_skip = 4;
    VecEnv env{1, cfg, spec, 1, frame_skip, max_ticks};
    Batch batch{env};
    env.reset(0, batch.obs.data());

    // The state the one agent step is about to leave behind.
    cfg.decision_interval = frame_skip;
    md::Sim reference{cfg};
    reference.reset(0);

    const std::int32_t noop = 0;
    EventObservation expected_events{};
    for (std::uint64_t tick = 0; tick < max_ticks; ++tick) {
        reference.step(md::decode_action(reference, spec, 0));
        accumulate_events(expected_events, reference.events());
    }
    env.step(&noop, batch.obs.data(), batch.final_obs.data(), batch.rewards.data(),
             batch.terminated.get(), batch.truncated.get());
    CHECK_FALSE(batch.terminated[0]);
    CHECK(batch.truncated[0]);
    const auto result = env.take_episode_result(0);
    REQUIRE(result.has_value());
    CHECK(result->ticks == max_ticks);

    // `final_obs` keeps the finished episode's last state — a truncated return has
    // to bootstrap from it.
    std::vector<float> expected_final = encode_of(reference, spec);
    replace_events(expected_final, expected_events);
    CHECK(std::equal(expected_final.begin(), expected_final.end(), batch.final_obs.begin()));

    // ...while `obs` already holds a fresh episode, so the batch never carries a
    // dead environment into the next forward pass. reset(0) over one env leaves
    // the first auto-reset seed at 1.
    md::Sim fresh{cfg};
    fresh.reset(1);
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
        md::Sim reference{env.config()};
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

TEST_CASE("a recorded episode replays to the state the batch ended in", "[rl][vec_env]") {
    // The whole point of recording during training: what the app plays back has to
    // be the episode the learner actually experienced, not an approximation.
    const md::Config cfg{};
    const md::ObsSpec spec{};
    constexpr std::uint64_t max_ticks = 240;
    VecEnv env{4, cfg, spec, 1, 4, max_ticks};
    Batch batch{env};
    env.reset(31, batch.obs.data());
    env.set_recording(0, true);
    CHECK(env.is_recording(0));
    CHECK_FALSE(env.is_recording(1));

    std::vector<std::int32_t> actions(env.num_envs());
    bool ended = false;
    for (int step = 0; step < 200 && !ended; ++step) {
        for (std::size_t i = 0; i < env.num_envs(); ++i) {
            const auto pick = (static_cast<std::uint32_t>(step) * 7u) + 1u;
            actions[i] = static_cast<std::int32_t>(pick % env.action_count());
        }
        env.step(actions.data(), batch.obs.data(), batch.final_obs.data(), batch.rewards.data(),
                 batch.terminated.get(), batch.truncated.get());
        ended = batch.terminated[0] || batch.truncated[0];
    }
    REQUIRE(ended);

    const auto recording = env.take_recording(0);
    REQUIRE(recording.has_value());
    CHECK(recording->seed == 31); // env 0 was seeded seed + 0
    CHECK(recording->frame_skip == env.frame_skip());
    CHECK_FALSE(recording->actions.empty());

    // Replaying must land on exactly the observation the batch reported as final.
    md::replay::Player player{*recording};
    while (player.tick()) {
        // drain
    }
    std::vector<float> replayed(spec.size());
    md::encode(player.sim(), spec, std::span<float>{replayed});
    // VecEnv's event suffix is accumulated across the final decision window,
    // whereas a tick-at-a-time replay exposes the current tick's events. Compare
    // the simulation state; the driver-level event window is tested separately.
    CHECK(std::equal(replayed.begin(), replayed.end() - md::ObsSpec::event_features,
                     batch.final_obs.begin()));

    // An episode is handed over once.
    CHECK_FALSE(env.take_recording(0).has_value());
}

TEST_CASE("a partial final frame records exactly the ticks that ran", "[rl][vec_env]") {
    const md::Config cfg{};
    const md::ObsSpec spec{};
    constexpr std::uint64_t max_ticks = 10;
    VecEnv env{1, cfg, spec, 1, 4, max_ticks};
    Batch batch{env};
    env.reset(31, batch.obs.data());
    env.set_recording(0, true);

    constexpr std::array<std::int32_t, 3> decisions{0, 1, 2};
    for (const std::int32_t decision : decisions) {
        env.step(&decision, batch.obs.data(), batch.final_obs.data(), batch.rewards.data(),
                 batch.terminated.get(), batch.truncated.get());
    }
    REQUIRE(batch.truncated[0]);

    const auto recording = env.take_recording(0);
    REQUIRE(recording.has_value());
    // The existing replay format has one fixed frame_skip. A partial final window
    // is expanded to a per-tick log, retaining the effective decision cadence in
    // Config so playback remains bit-identical without a format change.
    CHECK(recording->frame_skip == 1u);
    CHECK(recording->config.decision_interval == 4u);
    CHECK(recording->actions.size() == max_ticks);
    CHECK(recording->actions == std::vector<std::int32_t>{0, 0, 0, 0, 1, 1, 1, 1, 2, 2});

    md::replay::Player player{*recording};
    while (player.tick()) {
        // drain
    }
    CHECK(player.ticks_played() == max_ticks);
    const std::vector<float> replayed = encode_of(player.sim(), spec);
    CHECK(std::equal(replayed.begin(), replayed.end() - md::ObsSpec::event_features,
                     batch.final_obs.begin()));
}

TEST_CASE("recording cannot be enabled part-way through an episode", "[rl][vec_env]") {
    const md::Config cfg{};
    const md::ObsSpec spec{};
    VecEnv env{2, cfg, spec, 1, 4, 8};
    Batch batch{env};
    env.reset(31, batch.obs.data());

    const std::int32_t actions[2] = {0, 0};
    env.step(actions, batch.obs.data(), batch.final_obs.data(), batch.rewards.data(),
             batch.terminated.get(), batch.truncated.get());
    REQUIRE_FALSE(batch.truncated[0]);

    REQUIRE_THROWS_WITH(
        env.set_recording(0, true),
        "recording can only be enabled when the target environment is at episode tick 0; "
        "call reset() before record()");
    CHECK_FALSE(env.is_recording(0));

    // Rejection must not silently reset or otherwise mutate the live episode.
    env.step(actions, batch.obs.data(), batch.final_obs.data(), batch.rewards.data(),
             batch.terminated.get(), batch.truncated.get());
    REQUIRE(batch.truncated[0]);
    CHECK_FALSE(env.take_recording(0).has_value());

    env.reset(31, batch.obs.data());
    REQUIRE_NOTHROW(env.set_recording(0, true));
    CHECK(env.is_recording(0));
}

TEST_CASE("a just-reset environment can record while its peers continue", "[rl][vec_env]") {
    const md::Config cfg{};
    const md::ObsSpec spec{};
    VecEnv env{2, cfg, spec, 1, 4, 120'000};
    Batch batch{env};
    env.reset(31, batch.obs.data());

    const std::int32_t actions[2] = {0, 0};
    std::optional<std::size_t> just_reset{};
    for (std::size_t step = 0; step < 30'000 && !just_reset.has_value(); ++step) {
        env.step(actions, batch.obs.data(), batch.final_obs.data(), batch.rewards.data(),
                 batch.terminated.get(), batch.truncated.get());
        if (batch.terminated[0] != batch.terminated[1]) {
            just_reset = batch.terminated[0] ? 0u : 1u;
        }
    }

    REQUIRE(just_reset.has_value());
    REQUIRE_NOTHROW(env.set_recording(*just_reset, true));
    CHECK(env.is_recording(*just_reset));
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
