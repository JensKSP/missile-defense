// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
#pragma once

#include <cstddef>
#include <cstdint>

/// Per-phase timing for `Sim::step`, so optimisation targets measured hotspots
/// rather than guesses.
///
/// **Off by default and compiled to nothing.** `MD_PROF_ZONE` expands to an empty
/// statement unless the build defines `MD_PROFILE`, so the shipping hot path keeps
/// its no-allocation, no-syscall promise. Enable with:
///
///     cmake --preset profile && cmake --build --preset profile
///     ./build/profile/bench/md_bench --profile
///
/// Counters are thread-local, so parallel runs accumulate per worker and
/// `report()` shows the calling thread's own totals.
///
/// A caveat worth stating: a clock read per zone per tick is not free, and the
/// zones here are short. Treat the table as *relative attribution* — which phase
/// dominates — not as an absolute cost model. The unprofiled build is the one
/// whose wall-clock numbers mean anything.
namespace md::prof {

enum class Zone : std::uint8_t {
    Cooldowns,
    Crosshair,
    Fire,
    Interceptors,
    Blasts,
    Explosions,
    SmartBombs,
    MoveThreats,
    SplitMirvs,
    BlastHits,
    GroundHits,
    Waves,
    BonusCities,
    Termination,
    Count,
};

inline constexpr std::size_t zone_count = static_cast<std::size_t>(Zone::Count);

/// Human-readable zone name, for reports.
[[nodiscard]] const char* name(Zone zone) noexcept;

#ifdef MD_PROFILE

/// Add `nanos` to a zone's running total (thread-local).
void accumulate(Zone zone, std::uint64_t nanos) noexcept;

/// RAII timer: charges its lifetime to `zone`.
class Scope {
  public:
    explicit Scope(Zone zone) noexcept;
    Scope(const Scope&) = delete;
    Scope(Scope&&) = delete;
    Scope& operator=(const Scope&) = delete;
    Scope& operator=(Scope&&) = delete;
    ~Scope() noexcept;

  private:
    Zone zone_;
    std::uint64_t start_;
};

// A macro, not a function: it has to expand to *nothing* when profiling is off,
// which no inline function or RAII helper can promise as unconditionally.
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define MD_PROF_ZONE(zone_name) const ::md::prof::Scope md_prof_scope_(::md::prof::Zone::zone_name)

#else

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define MD_PROF_ZONE(zone_name) ((void) 0)

#endif

/// Nanoseconds and call count charged to `zone` on this thread. Always defined —
/// both read zero in a non-profiling build, so callers need no `#ifdef`.
///
/// Reporting is deliberately left to the caller: the core does no I/O and no
/// formatting, so these are the raw counters and `bench/` renders the table.
[[nodiscard]] std::uint64_t nanos(Zone zone) noexcept;
[[nodiscard]] std::uint64_t calls(Zone zone) noexcept;

/// Clear this thread's counters.
void reset() noexcept;

} // namespace md::prof
