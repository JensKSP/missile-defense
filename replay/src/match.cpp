// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
#include "md/replay/match.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <utility>

namespace md::replay {

namespace {

using json = nlohmann::json;

/// Mirrors `missile_defense.runs.tournament.write_manifest`.
constexpr int manifest_version = 1;

[[noreturn]] void fail(const std::filesystem::path& path, const std::string& why) {
    throw MatchPlayer::Error{path.string() + ": " + why};
}

/// Load one recording, or say which side could not be read.
///
/// Named rather than generic: "could not read the recording" leaves a person
/// with two files and no idea which is the bad one, and a match is precisely
/// the situation where there are two.
Player open(const std::filesystem::path& manifest, const std::filesystem::path& recording,
            const std::string& side) {
    std::optional<Recording> loaded = load(recording);
    if (!loaded.has_value()) {
        fail(manifest, "the " + side + " recording could not be read (" + recording.string() +
                           ") — it may be missing, truncated, or from another build");
    }
    return Player{std::move(*loaded)};
}

/// A manifest path resolved against the manifest's own directory.
///
/// Relative, so a match directory can be moved or restored from an archive
/// without every manifest in it going stale. An absolute path in the file is
/// honoured as given, which is what an ad-hoc writer would produce.
std::filesystem::path beside(const std::filesystem::path& manifest, const std::string& entry) {
    const std::filesystem::path path{entry};
    return path.is_absolute() ? path : manifest.parent_path() / path;
}

} // namespace

MatchPlayer::MatchPlayer(Side left, Side right, std::uint64_t seed)
    : left_{std::move(left)}, right_{std::move(right)}, seed_{seed} {}

MatchPlayer MatchPlayer::load(const std::filesystem::path& manifest) {
    std::ifstream file{manifest};
    if (!file) {
        fail(manifest, "could not be opened");
    }
    json payload;
    try {
        payload = json::parse(file);
    } catch (const json::exception& error) {
        fail(manifest, std::string{"is not readable JSON ("} + error.what() + ")");
    }
    if (!payload.is_object()) {
        fail(manifest, "is not a match manifest");
    }

    try {
        if (payload.at("version").get<int>() != manifest_version) {
            fail(manifest, "was written by a different version of this program");
        }
        const json& left = payload.at("left");
        const json& right = payload.at("right");

        const auto side = [&](const json& entry, const std::string& which) {
            const auto recording = entry.at("recording").get<std::string>();
            if (recording.empty()) {
                fail(manifest, "the " + which + " side names no recording");
            }
            Side built{open(manifest, beside(manifest, recording), which),
                       entry.value("display_name", std::string{}), std::nullopt};
            if (entry.contains("mean_score") && entry.at("mean_score").is_number()) {
                built.mean_score = entry.at("mean_score").get<double>();
            }
            if (built.name.empty()) {
                built.name = which == "left" ? "LEFT" : "RIGHT";
            }
            return built;
        };

        std::uint64_t seed = 0;
        if (const auto seeds = payload.find("seeds");
            seeds != payload.end() && seeds->is_array() && !seeds->empty()) {
            seed = seeds->front().get<std::uint64_t>();
        }

        Side left_side = side(left, "left");
        Side right_side = side(right, "right");

        // Same seed, same configuration, or it is not a paired match — it is two
        // unrelated episodes side by side, which would make every visual
        // comparison between them meaningless.
        if (left_side.player.recording().seed != right_side.player.recording().seed) {
            fail(manifest, "the two sides were recorded on different seeds; a match compares "
                           "two agents on the *same* problem");
        }
        // Read before the move, not after it: `seed != 0 ? ... : left_side...`
        // in the argument list is a use-after-move whose evaluation order is
        // unspecified — it happened to work, which is the worst kind of working.
        const std::uint64_t recorded = left_side.player.recording().seed;
        return MatchPlayer{std::move(left_side), std::move(right_side),
                           seed != 0 ? seed : recorded};
    } catch (const json::out_of_range& error) {
        fail(manifest, std::string{"is missing a required field ("} + error.what() + ")");
    } catch (const json::type_error& error) {
        fail(manifest, std::string{"has a field of the wrong type ("} + error.what() + ")");
    }
}

MatchPlayer MatchPlayer::pair(const std::filesystem::path& left,
                              const std::filesystem::path& right) {
    Side left_side{open(left, left, "left"), std::string{left.stem().string()}, std::nullopt};
    Side right_side{open(right, right, "right"), std::string{right.stem().string()}, std::nullopt};
    if (left_side.player.recording().seed != right_side.player.recording().seed) {
        throw Error{"these two recordings are of different seeds; a match compares two agents "
                    "on the *same* problem, so pairing them would show two unrelated episodes"};
    }
    const std::uint64_t seed = left_side.player.recording().seed;
    return MatchPlayer{std::move(left_side), std::move(right_side), seed};
}

bool MatchPlayer::tick() {
    // Both, or neither. A side that has already finished is skipped rather than
    // stopping the clock: the tick number has to keep meaning "the same moment"
    // for the side still playing, which is exactly the unequal-endings case.
    const auto step = [](Side& side) { return !side.player.finished() && side.player.tick(); };

    // Under wave sync the side that got to the next wave first waits at its
    // threshold. Only while *both* are still playing: a finished side has no
    // wave to reach, and holding for it would end the match early.
    bool hold_left = false;
    bool hold_right = false;
    if (wave_sync_ && !left_.player.finished() && !right_.player.finished()) {
        const std::uint32_t left_wave = left_.player.sim().wave();
        const std::uint32_t right_wave = right_.player.sim().wave();
        hold_left = left_wave > right_wave;
        hold_right = right_wave > left_wave;
    }

    bool left_played = !hold_left && step(left_);
    bool right_played = !hold_right && step(right_);
    if (!left_played && !right_played) {
        // A hold is only ever *for* the other side, and that side has just run
        // out of recording. The wait is over now rather than one frame later:
        // a frame in which nothing moved is one the caller reads as the end of
        // the match.
        left_played = hold_left && step(left_);
        right_played = hold_right && step(right_);
    }
    if (!left_played && !right_played) {
        return false;
    }
    ++tick_;
    return true;
}

void MatchPlayer::restart() {
    left_.player.restart();
    right_.player.restart();
    tick_ = 0;
}

void MatchPlayer::seek(std::uint64_t to_tick) {
    if (wave_sync_) {
        // Under sync the two sides are deliberately *not* on the same tick — a
        // held side spends frames without spending ticks — so seeking each of
        // them to `to_tick` would tear apart the alignment the mode exists to
        // create. Replayed from the start instead, which is what `Player::seek`
        // does internally for a backwards seek anyway; a match is a few
        // thousand ticks of a simulation that runs far faster than it draws.
        restart();
        while (tick_ < to_tick && tick()) {
            // the loop is the seek
        }
        return;
    }
    // `Player::seek` clamps to its own length, so a side shorter than `to_tick`
    // lands on its own end — which is the frozen-final-frame behaviour, arrived
    // at by seeking rather than by a second code path that could disagree.
    left_.player.seek(to_tick);
    right_.player.seek(to_tick);
    tick_ = std::min(to_tick, total_ticks());
}

std::uint64_t MatchPlayer::total_ticks() const noexcept {
    return std::max(left_.player.total_ticks(), right_.player.total_ticks());
}

float MatchPlayer::progress() const noexcept {
    // Finished is 1.0, whatever the arithmetic says. `total_ticks()` is the
    // longer *recording's* nominal length, and an episode routinely ends before
    // its recording runs out — the game was over. Without this a completed
    // match parks its progress bar at 68% and looks stuck, which is a worse lie
    // than the rounding it would have avoided.
    if (finished()) {
        return 1.0F;
    }
    if (wave_sync_) {
        // The clock and the recordings no longer share a scale: a held side
        // spends frames without spending ticks, so `tick_` outruns both
        // recordings and the bar would reach the end long before the match
        // does. How far the further side has got through its own episode is
        // the fraction that stays true either way.
        return std::min(1.0F, std::max(left_.player.progress(), right_.player.progress()));
    }
    const std::uint64_t total = total_ticks();
    if (total == 0) {
        return 0.0F;
    }
    return std::min(1.0F, static_cast<float>(tick_) / static_cast<float>(total));
}

bool MatchPlayer::finished() const noexcept {
    return left_.player.finished() && right_.player.finished();
}

} // namespace md::replay
