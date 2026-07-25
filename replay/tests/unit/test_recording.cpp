// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
//
// The property everything else rests on: replaying a recording reproduces the
// recorded run exactly. If that ever stops holding, watching a training episode
// stops showing what the policy actually did.
#include "md/action.hpp"
#include "md/config.hpp"
#include "md/intercept.hpp"
#include "md/observation.hpp"
#include "md/replay/recording.hpp"
#include "md/sim.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <vector>

namespace {

using md::replay::Player;
using md::replay::Recording;

/// A scratch path that cleans itself up, so a failing assertion cannot leave files
/// behind in the temp directory.
class TempFile {
  public:
    explicit TempFile(const std::string& name)
        : path_{std::filesystem::temp_directory_path() / name} {
        std::filesystem::remove(path_);
    }

    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;
    TempFile(TempFile&&) = delete;
    TempFile& operator=(TempFile&&) = delete;

    ~TempFile() {
        std::error_code ec;
        std::filesystem::remove(path_, ec);
    }

    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

  private:
    std::filesystem::path path_;
};

std::vector<float> observation_of(const md::Sim& sim, const md::ObsSpec& spec) {
    std::vector<float> out(spec.size());
    md::encode(sim, spec, std::span<float>{out});
    return out;
}

/// A deterministic stand-in for a policy: indices that actually engage targets, so
/// the replay has real gameplay to reproduce rather than a stream of no-ops.
std::vector<std::int32_t> scripted_actions(std::size_t count, std::uint32_t action_count) {
    std::vector<std::int32_t> actions(count);
    for (std::size_t i = 0; i < count; ++i) {
        actions[i] = static_cast<std::int32_t>((i * 7 + 1) % action_count);
    }
    return actions;
}

Recording make_recording(std::uint64_t seed, std::size_t steps, std::uint32_t frame_skip) {
    Recording recording;
    recording.seed = seed;
    recording.frame_skip = frame_skip;
    recording.update = 1200;
    recording.set_label("update-1200");
    recording.actions = scripted_actions(steps, md::action_count(recording.spec));
    return recording;
}

} // namespace

TEST_CASE("a replay reproduces the recorded run tick for tick", "[replay]") {
    const Recording recording = make_recording(4242, 400, 4);

    // Drive a sim exactly as a training rollout does: hold each index across the
    // frame-skip window, re-decoding it every tick.
    md::Sim reference{recording.config};
    reference.reset(recording.seed);
    std::vector<std::vector<float>> expected;
    for (const std::int32_t index : recording.actions) {
        for (std::uint32_t k = 0; k < recording.frame_skip; ++k) {
            if (reference.terminated()) {
                break;
            }
            reference.step(
                md::decode_action(reference, recording.spec, static_cast<std::uint32_t>(index)));
            expected.push_back(observation_of(reference, recording.spec));
        }
    }
    REQUIRE_FALSE(expected.empty());

    Player player{recording};
    std::size_t at = 0;
    while (player.tick()) {
        REQUIRE(at < expected.size());
        REQUIRE(observation_of(player.sim(), recording.spec) == expected[at]);
        ++at;
    }
    CHECK(at == expected.size());
    CHECK(player.ticks_played() == static_cast<std::uint64_t>(expected.size()));
}

TEST_CASE("a recording survives a round trip through a file", "[replay]") {
    const TempFile file{"md_replay_roundtrip.mdr"};
    const Recording original = make_recording(99, 64, 3);

    REQUIRE(md::replay::save(original, file.path()));
    const auto loaded = md::replay::load(file.path());
    REQUIRE(loaded.has_value());

    CHECK(loaded->seed == original.seed);
    CHECK(loaded->frame_skip == original.frame_skip);
    CHECK(loaded->update == original.update);
    CHECK(loaded->label_text() == "update-1200");
    CHECK(loaded->actions == original.actions);
    CHECK(loaded->config.world_width == original.config.world_width);
    CHECK(loaded->config.ammo_per_base == original.config.ammo_per_base);
    CHECK(loaded->spec.threats == original.spec.threats);
}

TEST_CASE("a round-tripped recording replays identically", "[replay]") {
    // Round-tripping must not perturb the run: this is the path a training
    // artifact actually takes — written by the trainer, read by the app.
    const TempFile file{"md_replay_identical.mdr"};
    const Recording original = make_recording(7, 200, 4);
    REQUIRE(md::replay::save(original, file.path()));
    const auto loaded = md::replay::load(file.path());
    REQUIRE(loaded.has_value());

    Player from_memory{original};
    Player from_disk{*loaded};
    while (from_memory.tick()) {
        REQUIRE(from_disk.tick());
        REQUIRE(observation_of(from_memory.sim(), original.spec) ==
                observation_of(from_disk.sim(), loaded->spec));
    }
    CHECK_FALSE(from_disk.tick());
}

TEST_CASE("restart rewinds a player to the beginning", "[replay]") {
    const Recording recording = make_recording(11, 50, 2);
    Player player{recording};
    const std::vector<float> start = observation_of(player.sim(), recording.spec);

    for (int i = 0; i < 20; ++i) {
        REQUIRE(player.tick());
    }
    REQUIRE(observation_of(player.sim(), recording.spec) != start);

    player.restart();
    CHECK(player.ticks_played() == 0);
    CHECK(observation_of(player.sim(), recording.spec) == start);
}

TEST_CASE("a player reports progress and stops at the end", "[replay]") {
    const Recording recording = make_recording(3, 25, 4);
    Player player{recording};
    CHECK(player.total_ticks() == 100);
    CHECK(player.progress() == 0.0f);

    while (player.tick()) {
        // drain
    }
    CHECK(player.finished());
    CHECK_FALSE(player.tick()); // idempotent once exhausted
    CHECK(player.progress() > 0.0f);
}

TEST_CASE("an empty recording is finished immediately", "[replay]") {
    Recording recording;
    recording.seed = 1;
    Player player{recording};
    CHECK(player.finished());
    CHECK_FALSE(player.tick());
    CHECK(player.progress() == 1.0f);
}

TEST_CASE("labels are truncated rather than overflowing", "[replay]") {
    Recording recording;
    recording.set_label(std::string(200, 'x'));
    CHECK(recording.label_text().size() == recording.label.size() - 1);
    CHECK(recording.label.back() == '\0');
}

TEST_CASE("junk and truncated files are rejected, not replayed", "[replay]") {
    SECTION("a missing file") {
        CHECK_FALSE(
            md::replay::load(std::filesystem::temp_directory_path() / "md_replay_absent.mdr")
                .has_value());
    }

    SECTION("a file that is not a recording") {
        const TempFile file{"md_replay_junk.mdr"};
        {
            std::ofstream out(file.path(), std::ios::binary);
            out << "this is not a recording, it is a text file that happens to be long enough";
        }
        CHECK_FALSE(md::replay::load(file.path()).has_value());
    }

    SECTION("a recording cut short") {
        const TempFile whole{"md_replay_whole.mdr"};
        const TempFile cut{"md_replay_cut.mdr"};
        REQUIRE(md::replay::save(make_recording(5, 100, 4), whole.path()));

        // Copy everything but the last few actions: the header still promises them.
        const auto size = static_cast<std::size_t>(std::filesystem::file_size(whole.path()));
        std::vector<char> bytes(size);
        {
            std::ifstream in(whole.path(), std::ios::binary);
            in.read(bytes.data(), static_cast<std::streamsize>(size));
        }
        {
            std::ofstream out(cut.path(), std::ios::binary);
            out.write(bytes.data(), static_cast<std::streamsize>(size - 16));
        }
        CHECK_FALSE(md::replay::load(cut.path()).has_value());
    }
}
