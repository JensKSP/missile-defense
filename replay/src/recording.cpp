// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
#include "md/replay/recording.hpp"

#include "md/action.hpp"
#include "md/intercept.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <fstream>
#include <istream>
#include <iterator>
#include <ostream>
#include <span>
#include <system_error>
#include <utility>

namespace md::replay {

namespace {

// ---- On-disk layout ---------------------------------------------------------
// Fixed 80-byte header, then the raw Config and ObsSpec bytes, then one int32 per
// agent step. Little-endian, which every supported target is; the header stores
// the two struct sizes so a mismatched build is rejected loudly instead of
// producing a run that silently diverges.
//
//   0   8   magic "MDREPLY\0"
//   8   4   format_version
//  12   4   config_size
//  16   4   spec_size
//  20   4   frame_skip
//  24   8   seed
//  32   8   update
//  40   8   step_count
//  48  32   label
constexpr std::array<char, 8> magic{{'M', 'D', 'R', 'E', 'P', 'L', 'Y', '\0'}};
constexpr std::uint32_t format_version = 1;
constexpr std::size_t header_size = 80;

static_assert(std::endian::native == std::endian::little,
              "the replay format is little-endian; add byte swapping for big-endian targets");

template <typename T> void put(std::array<char, header_size>& buffer, std::size_t at, T value) {
    static_assert(std::is_trivially_copyable_v<T>);
    std::memcpy(buffer.data() + at, &value, sizeof(T));
}

template <typename T> T get(const std::array<char, header_size>& buffer, std::size_t at) {
    static_assert(std::is_trivially_copyable_v<T>);
    T value{};
    std::memcpy(&value, buffer.data() + at, sizeof(T));
    return value;
}

// Binary serialisation is the one place a byte view of an object is the point, and
// iostreams speak `char*`. The casts are quarantined in these two helpers rather
// than sprinkled through the reader and writer.
void write_raw(std::ostream& out, const void* data, std::size_t bytes) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes));
}

std::streamsize read_raw(std::istream& in, void* data, std::size_t bytes) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    in.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(bytes));
    return in.gcount();
}

} // namespace

std::string_view Recording::label_text() const noexcept {
    // The array is always NUL-terminated by construction: set_label caps at
    // size() - 1, and load() forces the last byte. Scanning for the terminator
    // with an iterator is not portable here — std::array's iterator is a raw
    // pointer in libstdc++/libc++ but a class type in MSVC's STL, and the two
    // spellings that satisfy each compiler are mutually exclusive.
    return std::string_view{label.data()};
}

void Recording::set_label(std::string_view text) noexcept {
    label.fill('\0');
    const std::size_t n = std::min(text.size(), label.size() - 1);
    std::ranges::copy_n(text.begin(), static_cast<std::ptrdiff_t>(n), label.begin());
}

bool save(const Recording& recording, const std::filesystem::path& path) {
    if (const std::filesystem::path parent = path.parent_path(); !parent.empty()) {
        std::error_code ec; // create_directories throws on failure without one
        std::filesystem::create_directories(parent, ec);
        if (ec) {
            return false;
        }
    }

    std::array<char, header_size> header{};
    std::ranges::copy(magic, header.begin());
    put<std::uint32_t>(header, 8, format_version);
    put<std::uint32_t>(header, 12, static_cast<std::uint32_t>(sizeof(Config)));
    put<std::uint32_t>(header, 16, static_cast<std::uint32_t>(sizeof(ObsSpec)));
    put<std::uint32_t>(header, 20, recording.frame_skip);
    put<std::uint64_t>(header, 24, recording.seed);
    put<std::uint64_t>(header, 32, recording.update);
    put<std::uint64_t>(header, 40, static_cast<std::uint64_t>(recording.actions.size()));
    std::ranges::copy(recording.label, header.begin() + 48);

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        return false;
    }
    write_raw(out, header.data(), header.size());
    write_raw(out, &recording.config, sizeof(Config));
    write_raw(out, &recording.spec, sizeof(ObsSpec));
    if (!recording.actions.empty()) {
        write_raw(out, recording.actions.data(), recording.actions.size() * sizeof(std::int32_t));
    }
    out.flush();
    return static_cast<bool>(out);
}

std::optional<Recording> load(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return std::nullopt;
    }

    std::array<char, header_size> header{};
    if (read_raw(in, header.data(), header.size()) != static_cast<std::streamsize>(header_size)) {
        return std::nullopt;
    }
    if (!std::ranges::equal(magic, std::span<const char>{header.data(), magic.size()})) {
        return std::nullopt;
    }
    if (get<std::uint32_t>(header, 8) != format_version) {
        return std::nullopt;
    }
    // A different struct size means a different simulation; refuse rather than
    // replay something that is no longer the run that was recorded.
    if (get<std::uint32_t>(header, 12) != sizeof(Config) ||
        get<std::uint32_t>(header, 16) != sizeof(ObsSpec)) {
        return std::nullopt;
    }

    Recording recording;
    recording.frame_skip = std::max<std::uint32_t>(1, get<std::uint32_t>(header, 20));
    recording.seed = get<std::uint64_t>(header, 24);
    recording.update = get<std::uint64_t>(header, 32);
    const auto count = get<std::uint64_t>(header, 40);
    std::ranges::copy_n(header.begin() + 48, static_cast<std::ptrdiff_t>(recording.label.size()),
                        recording.label.begin());
    recording.label.back() = '\0'; // a hand-edited file must not un-terminate it

    read_raw(in, &recording.config, sizeof(Config));
    read_raw(in, &recording.spec, sizeof(ObsSpec));
    if (!in) {
        return std::nullopt;
    }

    recording.actions.resize(static_cast<std::size_t>(count));
    if (count > 0) {
        const std::size_t bytes = static_cast<std::size_t>(count) * sizeof(std::int32_t);
        if (read_raw(in, recording.actions.data(), bytes) != static_cast<std::streamsize>(bytes)) {
            return std::nullopt; // truncated
        }
    }
    return recording;
}

Player::Player(Recording recording) : recording_{std::move(recording)}, sim_{recording_.config} {
    restart();
}

void Player::restart() {
    sim_.reset(recording_.seed);
    step_ = 0;
    within_ = 0;
    ticks_ = 0;
    snapshots_.clear();
    capture_snapshot(); // tick 0, so a seek back to the start needs no reset
}

void Player::capture_snapshot() {
    snapshots_.push_back(Snapshot{.tick = ticks_, .step = step_, .within = within_, .sim = sim_});
}

void Player::seek(std::uint64_t to_tick) {
    const std::uint64_t target = std::min(to_tick, total_ticks());
    if (target < ticks_) {
        // Rewind to the latest snapshot at or before the target, then play forward.
        const auto at = std::ranges::partition_point(
            snapshots_, [target](const Snapshot& s) { return s.tick <= target; });
        if (at == snapshots_.begin()) {
            restart();
        } else {
            const Snapshot& from = *std::prev(at);
            sim_ = from.sim;
            step_ = from.step;
            within_ = from.within;
            ticks_ = from.tick;
            snapshots_.erase(at, snapshots_.end()); // drop the future we just left
        }
    }
    while (ticks_ < target && tick()) {
        // play forward
    }
}

bool Player::finished() const noexcept {
    return step_ >= recording_.actions.size() || sim_.terminated();
}

std::uint64_t Player::total_ticks() const noexcept {
    return static_cast<std::uint64_t>(recording_.actions.size()) * recording_.frame_skip;
}

float Player::progress() const noexcept {
    const std::uint64_t total = total_ticks();
    if (total == 0) {
        return 1.0f;
    }
    return std::min(1.0f, static_cast<float>(ticks_) / static_cast<float>(total));
}

bool Player::tick() {
    if (finished()) {
        return false;
    }
    // Re-decode every tick against the *current* state, exactly as md::rl::VecEnv
    // does while training. Decoding once per agent step and reusing the Action
    // would drift, because an engagement steers toward a moving target.
    const auto index = static_cast<std::uint32_t>(std::max(0, recording_.actions[step_]));
    sim_.step(decode_action(sim_, recording_.spec, index));
    ++ticks_;
    if (++within_ >= recording_.frame_skip) {
        within_ = 0;
        ++step_;
    }
    // Only on an action boundary: restoring mid-window would need `within_` to line
    // up with a decode that already happened.
    if (within_ == 0 && ticks_ % snapshot_interval == 0) {
        capture_snapshot();
    }
    return true;
}

} // namespace md::replay
