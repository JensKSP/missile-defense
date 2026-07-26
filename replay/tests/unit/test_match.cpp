// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
//
// **Desynchronisation is the entire failure mode of a split screen.** Two
// players on two timers drift within seconds, and tick 900 beside tick 913 is
// not a comparison — it is two videos. So most of what is asserted here is one
// property stated from several directions: after any transport operation, both
// sides are on the same tick.
//
// The second theme is unequal endings, which are the *interesting* case rather
// than an edge one: one agent dying at wave 9 while the other reaches wave 14
// is what the comparison is for. The shorter side has to freeze on its final
// state while the clock keeps running, or the longer side's remaining play
// would be shown against a screen that had silently rewound.
#include "md/intercept.hpp"
#include "md/replay/match.hpp"
#include "md/replay/recording.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

using md::replay::MatchPlayer;
using md::replay::Recording;

/// A scratch directory that cleans itself up, so a failing assertion cannot
/// leave manifests and recordings behind in the temp directory.
class TempDir {
  public:
    explicit TempDir(const std::string& name)
        : path_{std::filesystem::temp_directory_path() / ("md-match-" + name)} {
        std::filesystem::remove_all(path_);
        std::filesystem::create_directories(path_);
    }

    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;
    TempDir(TempDir&&) = delete;
    TempDir& operator=(TempDir&&) = delete;

    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }

    [[nodiscard]] std::filesystem::path operator/(const std::string& name) const {
        return path_ / name;
    }

    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

  private:
    std::filesystem::path path_;
};

/// A recording with real gameplay in it, of a given length.
///
/// `steps` differ between the two sides in most tests below, because equal
/// lengths would let a broken implementation pass the synchronisation checks by
/// coincidence.
std::filesystem::path write_recording(const std::filesystem::path& path, std::uint64_t seed,
                                      std::size_t steps, std::int32_t stride) {
    Recording recording;
    recording.seed = seed;
    recording.frame_skip = 4;
    recording.config.decision_interval = 4;
    recording.set_label("side");
    const auto count = md::action_count(recording.spec);
    recording.actions.resize(steps);
    for (std::size_t i = 0; i < steps; ++i) {
        recording.actions[i] =
            static_cast<std::int32_t>((i * static_cast<std::size_t>(stride) + 1) % count);
    }
    REQUIRE(md::replay::save(recording, path));
    return path;
}

std::filesystem::path write_manifest(const std::filesystem::path& path, const std::string& body) {
    std::ofstream out{path, std::ios::trunc};
    out << body;
    return path;
}

/// A manifest in exactly the shape `md.tournament.write_manifest` produces.
std::string manifest_for(const std::string& left, const std::string& right,
                         std::uint64_t seed = 4242) {
    return R"({"version": 1, "seeds": [)" + std::to_string(seed) +
           R"(], "left": {"model_id": "a", "display_name": "Amber Anvil", "mean_score": 51000.0, )"
           R"("recording": ")" +
           left +
           R"("}, "right": {"model_id": "b", "display_name": "Brisk Harbour", )"
           R"("mean_score": 47000.0, "recording": ")" +
           right + R"("}, "ranked": true})";
}

} // namespace

TEST_CASE("A match loads both sides and names them", "[replay][match]") {
    const TempDir dir{"names"};
    write_recording(dir / "a.mdr", 4242, 200, 7);
    write_recording(dir / "b.mdr", 4242, 200, 5);
    const auto manifest = write_manifest(dir / "m.json", manifest_for("a.mdr", "b.mdr"));

    const MatchPlayer match = MatchPlayer::load(manifest);
    // Names, never paths: the whole point of the manifest carrying them.
    CHECK(match.left().name == "Amber Anvil");
    CHECK(match.right().name == "Brisk Harbour");
    // And the scores the tournament recorded, so the spectator can state what it
    // is showing rather than leaving a viewer to assume.
    REQUIRE(match.left().mean_score.has_value());
    CHECK(*match.left().mean_score == 51000.0);
    CHECK(match.seed() == 4242u);
}

TEST_CASE("Recording paths are resolved beside the manifest", "[replay][match]") {
    // Relative, so a match directory can be moved or restored from an archive
    // without every manifest in it going stale.
    const TempDir dir{"relative"};
    write_recording(dir / "a.mdr", 7, 60, 7);
    write_recording(dir / "b.mdr", 7, 60, 5);
    const auto manifest = write_manifest(dir / "m.json", manifest_for("a.mdr", "b.mdr", 7));
    CHECK_NOTHROW(MatchPlayer::load(manifest));
}

TEST_CASE("Both sides stay on the same tick, however the transport is driven", "[replay][match]") {
    const TempDir dir{"sync"};
    write_recording(dir / "a.mdr", 4242, 300, 7);
    write_recording(dir / "b.mdr", 4242, 180, 5); // deliberately shorter
    MatchPlayer match =
        MatchPlayer::load(write_manifest(dir / "m.json", manifest_for("a.mdr", "b.mdr")));

    // Ticking: the shared clock advances once per tick, never twice, never zero.
    for (int i = 0; i < 200; ++i) {
        if (!match.tick()) {
            break;
        }
        CHECK(match.tick_count() == static_cast<std::uint64_t>(i + 1));
    }

    // Seeking: both land on the same tick, and the shorter side lands on its end.
    match.seek(100);
    CHECK(match.tick_count() == 100u);
    CHECK(match.left().player.ticks_played() == 100u);
    CHECK(match.right().player.ticks_played() == 100u);

    // Restarting: back to zero, both of them.
    match.restart();
    CHECK(match.tick_count() == 0u);
    CHECK(match.left().player.ticks_played() == 0u);
    CHECK(match.right().player.ticks_played() == 0u);
}

TEST_CASE("A seek past the shorter side leaves it on its own final state", "[replay][match]") {
    // The unequal-endings case. The shorter side must not rewind, and the clock
    // must not stop for the side that is still playing.
    const TempDir dir{"unequal"};
    write_recording(dir / "a.mdr", 99, 400, 7);
    write_recording(dir / "b.mdr", 99, 40, 5);
    MatchPlayer match =
        MatchPlayer::load(write_manifest(dir / "m.json", manifest_for("a.mdr", "b.mdr", 99)));

    const std::uint64_t short_side = match.right().player.total_ticks();
    match.seek(short_side + 500);
    CHECK(match.right().player.finished());
    CHECK(match.right().player.ticks_played() == short_side);
    // The longer side went further, and the clock followed *it*.
    CHECK(match.left().player.ticks_played() > short_side);
    CHECK(match.tick_count() > short_side);
}

TEST_CASE("A match runs until both sides are done, not until the first is", "[replay][match]") {
    const TempDir dir{"until"};
    write_recording(dir / "a.mdr", 5, 240, 7);
    write_recording(dir / "b.mdr", 5, 40, 5);
    MatchPlayer match =
        MatchPlayer::load(write_manifest(dir / "m.json", manifest_for("a.mdr", "b.mdr", 5)));

    std::uint64_t ticks = 0;
    while (match.tick()) {
        ++ticks;
        REQUIRE(ticks < 100000u); // a match that never ends is the bug, not a hang
    }
    CHECK(match.finished());
    // Not `== total_ticks()`: that is the longer recording's *nominal* length,
    // and an episode ends when the game does, which is routinely sooner.
    CHECK(ticks > 0u);
    CHECK(ticks <= match.total_ticks());
    // The bar still has to read full. A finished match parked at 68% looks
    // stuck, which is a worse lie than the rounding it avoids.
    CHECK(match.progress() == 1.0F);
}

TEST_CASE("Two recordings can be paired without a manifest", "[replay][match]") {
    // The ad-hoc path: two episodes exist and nothing wrote a tournament record
    // for them. It has no scores to claim and says so by leaving them empty.
    const TempDir dir{"adhoc"};
    const auto left = write_recording(dir / "a.mdr", 1234, 80, 7);
    const auto right = write_recording(dir / "b.mdr", 1234, 80, 5);
    const MatchPlayer match = MatchPlayer::pair(left, right);
    CHECK(match.seed() == 1234u);
    CHECK_FALSE(match.left().mean_score.has_value());
    CHECK(match.left().name == "a");
}

TEST_CASE("Two recordings of different seeds are refused", "[replay][match]") {
    // Not a match — two unrelated episodes side by side, which would make every
    // visual comparison between them meaningless.
    const TempDir dir{"seeds"};
    const auto left = write_recording(dir / "a.mdr", 1, 60, 7);
    const auto right = write_recording(dir / "b.mdr", 2, 60, 5);
    CHECK_THROWS_AS(MatchPlayer::pair(left, right), MatchPlayer::Error);

    const auto manifest = write_manifest(dir / "m.json", manifest_for("a.mdr", "b.mdr", 1));
    CHECK_THROWS_AS(MatchPlayer::load(manifest), MatchPlayer::Error);
}

TEST_CASE("A manifest naming a recording that is not there says which side", "[replay][match]") {
    // With two files in play, "could not read the recording" leaves a person
    // with no idea which one is bad.
    const TempDir dir{"missing"};
    write_recording(dir / "a.mdr", 8, 40, 7);
    const auto manifest = write_manifest(dir / "m.json", manifest_for("a.mdr", "gone.mdr", 8));
    try {
        (void) MatchPlayer::load(manifest);
        FAIL("a match with a missing side loaded");
    } catch (const MatchPlayer::Error& error) {
        const std::string message{error.what()};
        CHECK(message.find("right") != std::string::npos);
        CHECK(message.find("gone.mdr") != std::string::npos);
    }
}

TEST_CASE("A manifest that is not one is refused", "[replay][match]") {
    const TempDir dir{"bad"};
    CHECK_THROWS_AS(MatchPlayer::load(dir / "absent.json"), MatchPlayer::Error);
    CHECK_THROWS_AS(MatchPlayer::load(write_manifest(dir / "junk.json", "not json")),
                    MatchPlayer::Error);
    CHECK_THROWS_AS(MatchPlayer::load(write_manifest(dir / "empty.json", "{}")),
                    MatchPlayer::Error);
    CHECK_THROWS_AS(MatchPlayer::load(write_manifest(
                        dir / "future.json", R"({"version": 99, "left": {}, "right": {}})")),
                    MatchPlayer::Error);
}
