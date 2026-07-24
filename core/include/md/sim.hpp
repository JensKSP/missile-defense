#pragma once

#include "md/action.hpp"
#include "md/config.hpp"
#include "md/entities.hpp"
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

    [[nodiscard]] std::uint64_t tick() const noexcept { return tick_; }

    [[nodiscard]] std::uint32_t wave() const noexcept { return wave_; }

    [[nodiscard]] bool terminated() const noexcept { return terminated_; }

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

  private:
    void update_cooldowns() noexcept;
    bool try_fire(const Action& action) noexcept;
    void advance_interceptors() noexcept;
    void advance_blasts() noexcept;
    void spawn_blast(Vec2 center) noexcept;
    void move_threats() noexcept;
    std::int32_t resolve_blast_hits() noexcept;
    void resolve_ground_hits() noexcept;
    void update_waves() noexcept;
    void start_wave(std::uint32_t wave) noexcept;
    void spawn_threat() noexcept;
    void award_end_of_wave_bonus() noexcept;
    [[nodiscard]] bool pick_target(TargetKind& kind, std::uint32_t& index) noexcept;
    void update_termination() noexcept;

    Config config_{};
    Pcg32 rng_{};
    std::array<City, max_cities> cities_{};
    std::array<Base, base_count> bases_{};
    std::array<Threat, max_threats> threats_{};
    std::array<Interceptor, max_interceptors> interceptors_{};
    std::array<Blast, max_blasts> blasts_{};
    std::uint32_t threat_count_ = 0;
    std::uint32_t interceptor_count_ = 0;
    std::uint32_t blast_count_ = 0;
    std::int32_t score_ = 0;
    std::uint64_t tick_ = 0;
    std::uint32_t wave_ = 0;
    bool terminated_ = false;

    // Wave/spawn progression.
    std::uint32_t threats_to_spawn_ = 0; // remaining spawns in the current wave
    float spawn_timer_ = 0.0f;           // countdown to the next spawn
    float break_timer_ = 0.0f;           // >0 while between waves
};

static_assert(std::is_trivially_copyable_v<Sim>,
              "Sim state must be trivially copyable so snapshots are a memcpy");

} // namespace md
