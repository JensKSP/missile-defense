// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
#pragma once

#include "md/replay/recording.hpp"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>

namespace md::replay {

/// Two recordings of the same seed, played back as one thing.
///
/// **Desynchronisation is the entire failure mode of this feature.** Two players
/// on two timers drift within seconds, and a split screen showing tick 900 next
/// to tick 913 is not a comparison — it is two videos. So there is exactly one
/// transport here: `tick`, `seek` and `restart` move *both* sides or neither,
/// and neither `Player` is reachable in a way that lets a caller advance one
/// alone.
///
/// **Unequal endings are normal, not an edge case.** One agent dying at wave 9
/// while the other reaches wave 14 is the *interesting* case — it is what the
/// comparison is for. The shorter side freezes on its final state and the
/// clock keeps running, so the two stay on the same tick number even when only
/// one of them is still doing anything.
///
/// **Wave sync** (`set_wave_sync`, on by default) is the one thing that breaks
/// the shared clock, deliberately: a faster agent reaches wave 5 while the
/// other is still in wave 4, and from that moment the two halves are answering
/// different problems. So whichever side gets to a new wave first waits at the
/// threshold, and the ticks each has played diverge by exactly the frames it
/// spent waiting. Turn it off for the strict reading — same tick, same elapsed
/// time, whoever got further got further.
///
/// A match is loaded from a manifest written by `md.tournament.write_manifest`,
/// or paired ad hoc from two recordings. The manifest carries the scores the
/// tournament recorded, so a spectator can state what it is showing rather than
/// leaving a viewer to assume; the ad-hoc path has no such claim to make and
/// says so by leaving them empty.
class MatchPlayer {
  public:
    /// One side of the screen.
    struct Side {
        Player player;
        /// The model's display name, or the recording's label when pairing two
        /// files by hand. Never a path: a path is not a name.
        std::string name;
        /// What the tournament recorded for this side, when a manifest said.
        /// Absent for an ad-hoc pairing, which has nothing to claim.
        std::optional<double> mean_score;
    };

    /// A manifest could not be used, and why.
    class Error : public std::runtime_error {
      public:
        using std::runtime_error::runtime_error;
    };

    /// Load a paired match from a manifest.
    [[nodiscard]] static MatchPlayer load(const std::filesystem::path& manifest);

    /// Pair two recordings directly, with no manifest and no claimed scores.
    ///
    /// The development and one-off path — and the one that makes the game-only
    /// exhibition possible, where two episodes exist and nothing wrote a
    /// tournament record for them.
    [[nodiscard]] static MatchPlayer pair(const std::filesystem::path& left,
                                          const std::filesystem::path& right);

    /// Advance both sides one tick. False once *both* have finished.
    ///
    /// A side that has already finished is not stepped; the tick counter still
    /// advances, which is what keeps "the same tick" true while one side is
    /// frozen on its last frame.
    bool tick();

    /// Hold whichever side has started a wave the other has not reached yet.
    ///
    /// The same tick is the *fair* comparison — both agents have had exactly
    /// the same amount of time — and it is not always the *legible* one. A
    /// stronger agent clears waves faster, so within a minute one half of the
    /// screen is fighting wave 7 while the other is on wave 5, and a viewer
    /// trying to see how the two answer the *same* problem is watching two
    /// different problems. With this on, the side that reaches a new wave first
    /// waits at its threshold until the other arrives, and both then play that
    /// wave side by side.
    ///
    /// **A finished side never holds the other.** One agent dying at wave 9
    /// while the other reaches 14 is the case the split screen is for; waiting
    /// for a dead contestant would simply stop the match.
    ///
    /// **On by default**, because the thing a split screen is for is seeing two
    /// answers to the same problem, and two agents on different waves are not
    /// answering the same problem. Off restores the strict reading, where the
    /// tick number alone says everything about where both sides are — which is
    /// the fair one when what you want is "who got further in the same time".
    void set_wave_sync(bool on) noexcept { wave_sync_ = on; }

    [[nodiscard]] bool wave_sync() const noexcept { return wave_sync_; }

    /// Rewind both to the start of their recordings.
    void restart();

    /// Move both to `to_tick`. A side shorter than that lands on its own end.
    void seek(std::uint64_t to_tick);

    [[nodiscard]] const Side& left() const noexcept { return left_; }

    [[nodiscard]] const Side& right() const noexcept { return right_; }

    /// The shared clock. Both sides are always at this tick, or finished before it.
    [[nodiscard]] std::uint64_t tick_count() const noexcept { return tick_; }

    /// The longer of the two recordings — what a progress bar spans.
    [[nodiscard]] std::uint64_t total_ticks() const noexcept;

    [[nodiscard]] float progress() const noexcept;

    /// True once neither side has anything left to play.
    [[nodiscard]] bool finished() const noexcept;

    /// The seed both recordings were made on, when the manifest named one.
    [[nodiscard]] std::uint64_t seed() const noexcept { return seed_; }

  private:
    MatchPlayer(Side left, Side right, std::uint64_t seed);

    Side left_;
    Side right_;
    std::uint64_t seed_ = 0;
    std::uint64_t tick_ = 0;
    bool wave_sync_ = true;
};

} // namespace md::replay
