// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
#pragma once

#include "md/config.hpp"
#include "md/observation.hpp"
#include "md/sim.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string_view>
#include <vector>

namespace md::replay {

/// A recorded run, and everything needed to reproduce it exactly.
///
/// What is stored is the *discrete action index* per agent step — not the decoded
/// `Action`, and not any simulation state. `decode_action` is a pure function of
/// (sim, spec, index), and the simulation is deterministic, so seed + indices
/// replay bit-identically while costing four bytes per agent step. A 20 000-step
/// episode is 80 kB, which is what makes it reasonable to drop one on disk every
/// few training updates.
///
/// `frame_skip` matters: a training driver holds one index across that many ticks,
/// re-decoding it each tick (an engagement is a steer-then-fire macro). A replay
/// has to repeat that exactly, which is what `Player` does. If an episode ends
/// part-way through its last window, the recorder may store a per-tick log
/// (`frame_skip == 1`) while retaining the original decision cadence in `config`;
/// this represents the exact tick count without changing the file format.
struct Recording {
    Config config{};
    ObsSpec spec{};
    std::uint64_t seed = 0;
    std::uint32_t frame_skip = 1;
    std::uint64_t update = 0;     // training update this came from (0 when not from training)
    std::array<char, 32> label{}; // short human-facing tag, NUL-padded
    std::vector<std::int32_t> actions;

    /// `label` as text, without the NUL padding.
    [[nodiscard]] std::string_view label_text() const noexcept;

    /// Set `label`, truncating to fit.
    void set_label(std::string_view text) noexcept;
};

/// Write `recording` to `path`, creating parent directories. False on any failure.
///
/// The on-disk form embeds the raw `Config`/`ObsSpec` bytes and is validated by
/// size on load: these are **build-local training artifacts**, not an archive
/// format. A recording is only guaranteed replayable by the build that wrote it —
/// which is the honest guarantee, since changing a `Config` default changes the
/// simulation and would silently alter the run.
bool save(const Recording& recording, const std::filesystem::path& path);

/// Read a recording back. `std::nullopt` if the file is missing, truncated, not a
/// recording, or was written by an incompatible build.
[[nodiscard]] std::optional<Recording> load(const std::filesystem::path& path);

/// Replays a `Recording` into a `Sim`, one tick at a time.
///
/// Tick-at-a-time rather than step-at-a-time so a UI can drive it at any speed and
/// still see every intermediate frame — and so the app can hand the same events to
/// the audio engine that a live game produces.
class Player {
  public:
    explicit Player(Recording recording);

    /// Reset the simulation to the recording's seed and rewind to the first action.
    void restart();

    /// Advance exactly one simulation tick. False when there is nothing left to
    /// play — the actions ran out, or the episode terminated.
    bool tick();

    [[nodiscard]] const Sim& sim() const noexcept { return sim_; }

    [[nodiscard]] const Recording& recording() const noexcept { return recording_; }

    /// True once the recording is exhausted or the episode has terminated.
    [[nodiscard]] bool finished() const noexcept;

    /// Ticks played so far, and the total the recording covers.
    [[nodiscard]] std::uint64_t ticks_played() const noexcept { return ticks_; }

    [[nodiscard]] std::uint64_t total_ticks() const noexcept;

    /// How far through the recording, in [0, 1] — for a progress bar.
    [[nodiscard]] float progress() const noexcept;

    /// Jump to `tick`, clamped to the recording's length.
    ///
    /// There is no state to interpolate to: the only way to know the simulation at
    /// tick N is to have run N ticks. Seeking forward therefore plays forward, and
    /// seeking back rewinds to the nearest earlier snapshot and plays from there —
    /// which is what keeps scrubbing backwards on a long episode responsive rather
    /// than replaying from tick zero every time.
    void seek(std::uint64_t to_tick);

  private:
    /// A restore point: `Sim` is fixed-capacity POD, so this is a memcpy.
    struct Snapshot {
        std::uint64_t tick = 0;
        std::size_t step = 0;
        std::uint32_t within = 0;
        Sim sim;
    };

    /// Ticks between snapshots — 10 seconds of play at the fixed timestep. Costs a
    /// few tens of kB across a full episode, and bounds a backwards seek to
    /// replaying at most this many ticks.
    static constexpr std::uint64_t snapshot_interval = 600;

    void capture_snapshot();

    Recording recording_;
    Sim sim_;
    std::size_t step_ = 0;     // index into recording_.actions
    std::uint32_t within_ = 0; // ticks already spent on the current action
    std::uint64_t ticks_ = 0;
    std::vector<Snapshot> snapshots_;
};

} // namespace md::replay
