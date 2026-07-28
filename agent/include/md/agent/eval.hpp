// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
#pragma once

#include "md/agent/heuristic.hpp"
#include "md/agent/policy.hpp"
#include "md/config.hpp"
#include "md/entities.hpp"
#include "md/event.hpp"
#include "md/observation.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <numeric>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace md::agent {

/// Outcome of one episode. Every count is tallied off the deterministic event
/// stream (or the end state), so a learned policy driven from Python and the
/// scripted `Heuristic` are measured by the exact same bookkeeping.
struct EpisodeResult {
    std::uint64_t seed = 0;
    std::int32_t score = 0;
    std::uint32_t wave_reached = 0;  // the wave in progress when it ended (the "last level")
    std::uint32_t waves_cleared = 0; // waves completed outright
    std::uint64_t ticks = 0;         // survival time; ticks / 60 = seconds of play
    std::uint32_t cities_left = 0;   // cities still standing at the end
    std::uint32_t cities_lost = 0;   // cities destroyed over the episode
    std::uint32_t bases_left = 0;    // batteries still standing at the end
    std::uint32_t bases_lost = 0;    // battery destructions over the episode
    std::uint32_t ammo_left = 0;     // interceptors still loaded in surviving batteries
    std::uint32_t bonus_cities = 0;  // destroyed cities rebuilt at score thresholds
    std::uint32_t mirv_splits = 0;   // MIRV warheads that split mid-descent
    std::uint32_t shots = 0;         // interceptors launched
    std::uint32_t kills = 0;         // threats destroyed by a blast ("targets destroyed")
    // Where the score came from. These three sum to `score` exactly (see
    // `Sim::kill_credit`). `cities_left` and `ammo_left` above answer "what was
    // left when it was over", which for this game is always "nothing" — the
    // episode ends *because* the cities ran out. These answer the question that
    // was actually being asked: what did it hold, wave after wave, while it
    // still mattered.
    std::int32_t kill_credit = 0; // points from destroying threats
    std::int32_t city_credit = 0; // points from cities standing at each wave end
    std::int32_t ammo_credit = 0; // points from interceptors unspent at each wave end
    // Interceptors binned by how many threats their blast destroyed: 0,1,2,3,4+.
    std::array<std::uint32_t, kills_per_shot_bins> kills_per_shot{};
    bool terminated = false; // false => stopped at the tick cap, still alive

    /// Interceptors whose blast destroyed nothing (== `kills_per_shot[0]`).
    [[nodiscard]] std::uint32_t wasted() const noexcept { return kills_per_shot[0]; }

    /// Interceptors whose blast destroyed at least one threat.
    [[nodiscard]] std::uint32_t hits() const noexcept {
        return std::accumulate(kills_per_shot.begin() + 1, kills_per_shot.end(), 0U);
    }

    /// Kills per interceptor spent — above 1.0 means blasts are catching clusters.
    [[nodiscard]] double accuracy() const noexcept {
        return shots == 0u ? 0.0 : static_cast<double>(kills) / static_cast<double>(shots);
    }

    /// Fraction of *resolved* interceptors that destroyed at least one threat.
    [[nodiscard]] double hit_rate() const noexcept {
        const std::uint32_t resolved = hits() + wasted();
        return resolved == 0u ? 0.0 : static_cast<double>(hits()) / static_cast<double>(resolved);
    }
};

/// Aggregate over a seed set. Scripted and learned drivers use this same
/// implementation on a protocol-matched split, so "beat the baseline" is a
/// concrete claim rather than a vibe.
struct Summary {
    std::size_t episodes = 0;
    double mean_score = 0.0;
    double mean_wave = 0.0;          // last level reached
    double mean_waves_cleared = 0.0; // waves completed outright
    double mean_ticks = 0.0;         // survival time; / 60 = seconds
    double mean_cities_left = 0.0;
    double mean_cities_lost = 0.0;
    double mean_bases_left = 0.0;
    double mean_bases_lost = 0.0;
    double mean_ammo_left = 0.0; // unfired interceptors at the end — ammo held in reserve
    double mean_bonus_cities = 0.0;
    double mean_mirv_splits = 0.0;
    double mean_shots = 0.0;    // interceptors fired
    double mean_kills = 0.0;    // targets destroyed
    double mean_hits = 0.0;     // interceptors that destroyed >=1 threat
    double mean_accuracy = 0.0; // mean kills per interceptor — the documented yardstick
    double mean_hit_rate = 0.0; // mean fraction of interceptors that hit
    // The score, decomposed. Read as shares of `mean_score` these say what kind
    // of player a policy became: a defender's is city-heavy, a trigger-happy
    // one's is nearly all kills with no ammunition credit at all. Two policies
    // on the same total can differ completely here.
    double mean_kill_credit = 0.0;
    double mean_city_credit = 0.0;
    double mean_ammo_credit = 0.0;
    // Interceptors fired per wave cleared. Derived rather than measured — the
    // parts were always both present — and it is what shows a policy emptying
    // its magazines into the first wave it meets.
    double mean_shots_per_wave = 0.0;

    /// Share of the score that came from each source, in [0, 1]. Zero when
    /// nothing was scored, rather than a division by zero.
    [[nodiscard]] double kill_share() const noexcept { return share(mean_kill_credit); }
    [[nodiscard]] double city_share() const noexcept { return share(mean_city_credit); }
    [[nodiscard]] double ammo_share() const noexcept { return share(mean_ammo_credit); }

  private:
    [[nodiscard]] double share(double part) const noexcept {
        return mean_score == 0.0 ? 0.0 : part / mean_score;
    }

  public:
    std::int32_t min_score = 0;
    std::int32_t max_score = 0;
    std::size_t survived = 0; // episodes that hit the tick cap without dying
    // Kills-per-shot histogram summed over the whole seed set: how every
    // interceptor across the evaluation was spent, 0,1,2,3,4+ threats each.
    std::array<std::uint64_t, kills_per_shot_bins> kills_per_shot{};
};

/// A fixed deterministic seed stream. Protocols select disjoint blocks from it;
/// the prefix alone is not inherently a held-out benchmark.
[[nodiscard]] std::vector<std::uint64_t> default_seeds(std::size_t count = 32);

/// Fold one step's events into a running `EpisodeResult`. Shared between the
/// scripted `run_episode` and the Python-driven `VecEnv` so both count off the
/// same event stream and cannot drift — the same reason `summarize` is shared.
void tally_events(EpisodeResult& result, std::span<const Event> events) noexcept;

/// Bin every still-active blast by its kills at episode close, so the
/// kills-per-shot histogram accounts for interceptors whose blast had not yet
/// expired when the episode ended — the final wave especially. Without it the
/// histogram, and the `hits`/`wasted` derived from it, undercount every episode.
/// Shared for the same reason as `tally_events`.
void bin_active_blasts(EpisodeResult& result, std::span<const Blast> blasts) noexcept;

/// Something that can play: the scripted baseline, or a learned policy.
///
/// One virtual call per tick, against a `Sim::step` that is thousands of times
/// more work — and it buys the thing this whole comparison rests on, which is
/// that both contestants go through *the same* episode loop, the same event
/// tallying and the same `summarize`. A second loop for learned policies is how
/// two agents end up measured by two subtly different rulers.
class Driver {
  public:
    Driver() = default;
    virtual ~Driver() = default;

    /// The candidate action for this tick. `Sim` samples it at the shared player
    /// cadence and holds the chosen action between samples.
    [[nodiscard]] virtual Action act(const Sim& sim) = 0;

    /// The discrete action index behind the last decision, for an action log.
    /// The scripted agent has no index — it produces an `Action` directly — so
    /// it reports `no_index` and its log is empty rather than invented.
    [[nodiscard]] virtual std::uint32_t last_index() const noexcept { return no_index; }

    /// What to call this contestant in a result table. A model's display name
    /// where it has one, so a league ranks names rather than paths.
    [[nodiscard]] virtual std::string_view name() const noexcept = 0;

    static constexpr std::uint32_t no_index = 0xFFFFFFFFu;

  protected:
    // Protected rather than deleted: a concrete driver may want to be movable —
    // the game holds one in an `optional` and replaces it when the watched agent
    // changes — while slicing one through a `Driver&` stays impossible.
    Driver(const Driver&) = default;
    Driver(Driver&&) = default;
    Driver& operator=(const Driver&) = default;
    Driver& operator=(Driver&&) = default;
};

/// The M4 baseline as a `Driver`.
class ScriptedDriver final : public Driver {
  public:
    ScriptedDriver() = default;

    explicit ScriptedDriver(Params params) noexcept : agent_{params} {}

    /// Named skills carry their name into the printout, because a run at a
    /// reduced skill is *not* the published baseline and a result labelled
    /// "SCRIPTED" would be indistinguishable from one that is.
    explicit ScriptedDriver(Skill skill) noexcept
        : agent_{params_for(skill)}, name_{skill_name(skill)} {}

    [[nodiscard]] Action act(const Sim& sim) override { return agent_.act(sim); }

    [[nodiscard]] std::string_view name() const noexcept override { return name_; }

    [[nodiscard]] static constexpr std::string_view skill_name(Skill skill) noexcept {
        switch (skill) {
        case Skill::low:
            return "SCRIPTED (LOW)";
        case Skill::medium:
            return "SCRIPTED (MEDIUM)";
        case Skill::high:
            break;
        }
        return "SCRIPTED";
    }

  private:
    Heuristic agent_;
    std::string_view name_ = "SCRIPTED";
};

/// A learned `.mdp` policy as a `Driver`, through the same `Action` primitive.
///
/// The forward pass runs only on the ticks `Sim` will actually sample — one in
/// `Config::decision_interval` — because on every other tick the action is
/// ignored, and doing the arithmetic anyway would be four times the work for an
/// answer nobody reads. It is also what makes the action log line up with the
/// Python evaluator's, which steps a whole decision window per action.
class PolicyDriver final : public Driver {
  public:
    PolicyDriver(const Policy& policy, const ObsSpec& spec);

    [[nodiscard]] Action act(const Sim& sim) override;

    [[nodiscard]] std::uint32_t last_index() const noexcept override { return last_index_; }

    [[nodiscard]] std::string_view name() const noexcept override { return name_; }

    /// The critic's estimate at the last decision. An evaluator that logs it can
    /// tell "played badly" from "knew it was losing".
    [[nodiscard]] float last_value() const noexcept { return last_value_; }

  private:
    const Policy* policy_;
    ObsSpec spec_;
    std::string name_;
    std::vector<float> observation_;
    /// `action_mask` writes `bool`; `Policy::act` takes `std::uint8_t`, because
    /// a `std::span<bool>` cannot be formed over a `std::vector<bool>` and the
    /// policy's interface should not be shaped by that accident. So: a real
    /// array of `bool` for the first, a byte buffer for the second, and one
    /// copy per decision. Reinterpreting one as the other would be an aliasing
    /// violation that happens to work, which this project does not do.
    std::unique_ptr<bool[]> mask_;
    std::vector<std::uint8_t> legal_;
    /// Events seen since the last decision, in `md::encode`'s own scaling.
    ///
    /// **The observation is per *decision*, not per tick.** `md::encode` writes
    /// the current tick's events into the suffix; a policy that only ever sees
    /// the decision tick's events is blind to the three ticks in between, where
    /// most of them happen. `VecEnv` accumulates across the window and overwrites
    /// that suffix, and this has to do the identical thing — the parity e2e
    /// caught it diverging at decision 401, on the first event that fell in a
    /// skipped tick.
    std::array<float, ObsSpec::event_features> window_{};
    std::uint32_t last_index_ = 0;
    float last_value_ = 0.0F;
};

/// Play one episode to termination or `max_ticks`, whichever comes first. The
/// agent's reaction rate is the sim's own `Config::decision_interval` (the core
/// samples a new action once per that many ticks), so there is no per-driver
/// cadence knob here — a scripted agent and a learned policy are throttled
/// identically by the simulation, not by how their driver happens to loop.
///
/// `action_log`, when given, receives one index per *sampled* decision — not per
/// tick. That is the granularity at which two implementations can be compared:
/// the ticks in between are ones the simulation never asked about.
EpisodeResult run_episode(const Config& config, std::uint64_t seed, Driver& driver,
                          std::uint64_t max_ticks = 120000,
                          std::vector<std::uint32_t>* action_log = nullptr);

/// The scripted overload, kept because most callers have a `Heuristic` and not
/// a `Driver`, and wrapping one at every call site would be noise.
[[nodiscard]] EpisodeResult run_episode(const Config& config, std::uint64_t seed,
                                        const Heuristic& agent, std::uint64_t max_ticks = 120000);

/// Aggregate episode outcomes. Split out from `evaluate` so a *learned* agent —
/// which is driven from Python and cannot be a `Heuristic` — is scored by the same
/// function over the same fields, rather than by a reimplementation that might
/// quietly differ. That is what makes "beat the baseline" a claim and not a vibe.
[[nodiscard]] Summary summarize(std::span<const EpisodeResult> episodes);

/// Play every seed and aggregate. The reaction rate comes from the `Config`
/// (`decision_interval`), the same as `run_episode`.
[[nodiscard]] Summary evaluate(const Config& config, std::span<const std::uint64_t> seeds,
                               const Heuristic& agent, std::uint64_t max_ticks = 120000);

} // namespace md::agent
