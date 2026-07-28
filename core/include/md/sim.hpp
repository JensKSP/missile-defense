// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
#pragma once

#include "md/action.hpp"
#include "md/config.hpp"
#include "md/entities.hpp"
#include "md/event.hpp"
#include "md/rng.hpp"
#include "md/vec2.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <type_traits>

namespace md {

/// Result of advancing the simulation by one tick.
struct StepResult {
    std::int32_t reward = 0; // score delta this tick (the RL reward seed)
    bool terminated = false; // all cities destroyed — episode over
    // Two facts about how the ammunition was spent. The simulation reports them
    // and prices nothing: what a wasted shot or a double kill is *worth* is a
    // reward-design question, and it lives in the training layer (missile_defense.sim.env.Shaping)
    // so that the score — and therefore the benchmark every agent is measured on
    // — stays exactly what DESIGN §4.3 specifies.
    std::int32_t wasted = 0;      // blasts that expired this tick having killed nothing
    std::int32_t multi_kills = 0; // kills beyond a blast's first
    // How many threats each blast that expired this tick had destroyed, binned
    // 0,1,2,3,4+. `kills_per_shot[0]` is exactly `wasted`; the tail is what
    // "catching clusters" looks like as a distribution instead of a mean. Filled
    // at blast expiry, the one moment a blast's lifetime kill count is final.
    std::array<std::int32_t, kills_per_shot_bins> kills_per_shot{};
};

/// The Missile Command simulation: a self-contained, deterministic POD value.
///
/// `step()` advances exactly one fixed-`dt` tick and owns no timing. Drivers
/// (human app, training, replay) own the loop. All state is inline and fixed
/// capacity, so a snapshot is a `memcpy` and stepping is allocation-free.
class Sim {
  public:
    explicit Sim(const Config& config = {}) noexcept;

    /// Restart from a fresh field with the given RNG seed.
    void reset(std::uint64_t seed) noexcept;

    /// Advance exactly one fixed-`dt` tick.
    StepResult step(const Action& action) noexcept;

    // --- read-only state (shared by the renderer and future RL observations) ---
    [[nodiscard]] const Config& config() const noexcept { return config_; }

    [[nodiscard]] std::int32_t score() const noexcept { return score_; }

    /// Where the score came from. Every point this game awards passes through
    /// `score_multiplier()` and originates in exactly one of three places, so
    /// these three **sum to `score()` exactly** — an invariant the unit tests
    /// assert, and the reason a fourth source added without accounting for it
    /// here fails a test rather than silently going missing.
    ///
    /// The counts behind them were always computed and then discarded:
    /// `award_end_of_wave_bonus` already walks the cities and the magazines at
    /// every wave boundary and kept only the scalar it added. Keeping the split
    /// costs three additions and answers the question the end-of-episode
    /// `cities_left` and `ammo_left` cannot — both are ~0 in every episode ever
    /// recorded, because the game *ends* when the cities run out and the
    /// ammunition is what was spent defending them.
    ///
    /// Read as shares of the total they say what kind of player a policy became:
    /// a defender's score is city-heavy, a trigger-happy one's is nearly all
    /// kills with no ammunition credit at all.
    [[nodiscard]] std::int32_t kill_credit() const noexcept { return kill_credit_; }
    /// Cities standing at each wave end, at `score_per_surviving_city` x multiplier.
    [[nodiscard]] std::int32_t city_credit() const noexcept { return city_credit_; }
    /// Interceptors unspent at each wave end, at `score_per_unused_interceptor` x multiplier.
    [[nodiscard]] std::int32_t ammo_credit() const noexcept { return ammo_credit_; }

    [[nodiscard]] std::uint64_t tick() const noexcept { return tick_; }

    /// Whether the action passed to the next `step()` will replace the one
    /// currently held by the player-model cadence. Drivers with edge-triggered
    /// input (notably a mouse click) use this to keep the edge pending until the
    /// simulation can actually sample it.
    [[nodiscard]] bool samples_action_this_tick() const noexcept;

    [[nodiscard]] std::uint32_t wave() const noexcept { return wave_; }

    /// The arcade's wave score multiplier: 1x for waves 1-2, 2x for 3-4, and so
    /// on to `score_multiplier_max`. Applies to kills and to the end-of-wave
    /// bonus alike, so surviving deep is worth far more than playing early waves
    /// perfectly. Derived from the wave rather than stored, so it cannot drift.
    [[nodiscard]] std::int32_t score_multiplier() const noexcept;

    [[nodiscard]] bool terminated() const noexcept { return terminated_; }

    /// The aiming crosshair — simulation state, steered toward `Action::aim` at
    /// `Config::aim_max_speed`. Launches detonate at exactly this point.
    [[nodiscard]] Vec2 crosshair() const noexcept { return crosshair_; }

    /// Seconds until the next launch is allowed by the global trigger interval
    /// (`Config::fire_interval`); per-battery cooldowns apply on top.
    [[nodiscard]] float fire_cooldown() const noexcept { return fire_cooldown_remaining_; }

    /// Bonus cities earned but not yet spent, because every city was still
    /// standing when the score threshold was crossed. Banked until one falls.
    [[nodiscard]] std::uint32_t bonus_cities_banked() const noexcept { return banked_cities_; }

    [[nodiscard]] std::span<const City> cities() const noexcept {
        return {cities_.data(), cities_.size()};
    }

    [[nodiscard]] std::span<const Base> bases() const noexcept {
        return {bases_.data(), bases_.size()};
    }

    [[nodiscard]] std::span<const Threat> threats() const noexcept {
        return {threats_.data(), threat_count_};
    }

    [[nodiscard]] std::span<const Interceptor> interceptors() const noexcept {
        return {interceptors_.data(), interceptor_count_};
    }

    [[nodiscard]] std::span<const Blast> blasts() const noexcept {
        return {blasts_.data(), blast_count_};
    }

    [[nodiscard]] std::span<const Explosion> explosions() const noexcept {
        return {explosions_.data(), explosion_count_};
    }

    /// Events emitted during the most recent `step()` (cleared at the next step).
    [[nodiscard]] std::span<const Event> events() const noexcept {
        return {events_.data(), event_count_};
    }

  private:
    void push_event(EventType type, Vec2 pos) noexcept; // note: `emit` is a Qt macro

    void update_cooldowns() noexcept;
    void move_crosshair(const Action& action) noexcept;
    bool try_fire(const Action& action) noexcept;
    void advance_interceptors() noexcept;
    void advance_blasts() noexcept;
    [[nodiscard]] bool spawn_blast(Vec2 center) noexcept; // false when the pool is full
    void spawn_explosion(Vec2 center, float peak_radius) noexcept;
    void advance_explosions() noexcept;
    void move_threats() noexcept;
    void steer_smart_bombs() noexcept;
    void split_mirvs() noexcept;
    std::int32_t resolve_blast_hits() noexcept;
    [[nodiscard]] std::int32_t kill_score(ThreatType type) const noexcept;
    void resolve_ground_hits() noexcept;
    void update_waves() noexcept;
    void start_wave(std::uint32_t wave) noexcept;
    void spawn_threat() noexcept;
    void award_end_of_wave_bonus() noexcept;
    void award_bonus_cities() noexcept;
    [[nodiscard]] bool pick_target(TargetKind& kind, std::uint32_t& index) noexcept;
    [[nodiscard]] float threat_speed() const noexcept;
    [[nodiscard]] float mirv_probability() const noexcept;
    void update_termination() noexcept;

    Config config_{};
    Pcg32 rng_{};
    std::array<City, max_cities> cities_{};
    std::array<Base, base_count> bases_{};
    std::array<Threat, max_threats> threats_{};
    std::array<Interceptor, max_interceptors> interceptors_{};
    std::array<Blast, max_blasts> blasts_{};
    std::array<Explosion, max_explosions> explosions_{};
    std::array<Event, max_events> events_{};
    std::uint32_t threat_count_ = 0;
    std::uint32_t interceptor_count_ = 0;
    std::uint32_t blast_count_ = 0;
    std::uint32_t explosion_count_ = 0;
    std::uint32_t event_count_ = 0;
    Vec2 crosshair_{};                     // the shared aiming cursor (world units)
    float fire_cooldown_remaining_ = 0.0f; // global trigger interval countdown
    std::int32_t score_ = 0;
    // The three sources `score_` is made of, kept in step with it. See
    // `kill_credit()` for why they are worth their three additions.
    std::int32_t kill_credit_ = 0;
    std::int32_t city_credit_ = 0;
    std::int32_t ammo_credit_ = 0;
    std::uint64_t tick_ = 0;
    std::uint32_t wave_ = 0;
    bool terminated_ = false;

    // The action currently in force. Sampled once per `Config::decision_interval`
    // ticks and held between (see `Sim::step`), so the reaction rate is a
    // player-model limit like the aim and trigger caps — enforced here, not in a
    // driver, so no per-tick caller gets a free reflex edge over a 15 Hz one.
    Action latched_action_{};

    // Reset at the top of every step; reported in that step's StepResult. Members
    // rather than return values because they are produced in two different phases
    // (a blast expiring, and a blast killing) and consumed in one.
    std::int32_t tick_wasted_ = 0;
    std::int32_t tick_multi_kills_ = 0;
    std::array<std::int32_t, kills_per_shot_bins> tick_kills_per_shot_{};

    // Wave/spawn progression.
    std::int32_t next_bonus_score_ = 0;  // score at which the next bonus city is earned
    std::uint32_t banked_cities_ = 0;    // earned bonus cities awaiting a gap to fill
    std::uint32_t threats_to_spawn_ = 0; // remaining spawns in the current wave
    float spawn_timer_ = 0.0f;           // countdown to the next spawn
    float break_timer_ = 0.0f;           // >0 while between waves
    bool wave_started_pending_ = false;  // emit a WaveStarted event on the next step
};

static_assert(std::is_trivially_copyable_v<Sim>,
              "Sim state must be trivially copyable so snapshots are a memcpy");

} // namespace md
