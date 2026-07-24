#include "md/sim.hpp"

#include "md/config.hpp"
#include "md/entities.hpp"
#include "md/rng.hpp"
#include "md/vec2.hpp"

#include <algorithm>
#include <array>
#include <cstdint>

namespace md {

namespace {

/// Current blast radius: expands from 0 to the max over the first 30% of its
/// lifetime, then lingers at full radius until it expires.
float blast_radius(float age, const Config& c) noexcept {
    const float expand_time = 0.3f * c.blast_lifetime;
    if (age < expand_time) {
        return c.blast_max_radius * (age / expand_time);
    }
    return c.blast_max_radius;
}

} // namespace

Sim::Sim(const Config& config) noexcept : config_{config} {
    reset(0);
}

void Sim::reset(std::uint64_t seed) noexcept {
    rng_ = Pcg32{seed};

    // Field layout: [ALPHA] c c c [DELTA] c c c [OMEGA] — nine evenly spaced
    // slots across the bottom, bases at slots 0 / 4 / 8.
    constexpr std::uint32_t slots = base_count + max_cities;
    const auto slot_x = [&](std::uint32_t i) {
        return config_.world_width * (static_cast<float>(i) + 0.5f) / static_cast<float>(slots);
    };

    const auto make_base = [&](std::uint32_t slot) {
        return Base{.pos = Vec2{slot_x(slot), 0.0f},
                    .ammo = config_.ammo_per_base,
                    .cooldown_remaining = 0.0f,
                    .alive = true};
    };
    bases_[0] = make_base(0);
    bases_[1] = make_base(4);
    bases_[2] = make_base(8);

    constexpr std::array<std::uint32_t, max_cities> city_slots{1, 2, 3, 5, 6, 7};
    for (std::uint32_t i = 0; i < max_cities; ++i) {
        cities_[i] = City{.pos = Vec2{slot_x(city_slots[i]), 0.0f}, .alive = true};
    }

    threat_count_ = 0;
    interceptor_count_ = 0;
    blast_count_ = 0;
    score_ = 0;
    tick_ = 0;
    wave_ = 1;
    terminated_ = false;
}

StepResult Sim::step(const Action& action) noexcept {
    update_cooldowns();
    try_fire(action);
    advance_interceptors();
    advance_blasts();
    // Threat spawning/waves, collisions, scoring, and termination: next increment.
    ++tick_;
    return StepResult{.reward = 0, .terminated = terminated_};
}

void Sim::update_cooldowns() noexcept {
    for (auto& base : bases_) {
        base.cooldown_remaining = std::max(0.0f, base.cooldown_remaining - config_.dt);
    }
}

bool Sim::try_fire(const Action& action) noexcept {
    if (action.kind != Action::Kind::Fire) {
        return false;
    }
    const auto index = static_cast<std::uint32_t>(action.base);
    Base& base = bases_[index];
    if (!base.alive || base.ammo == 0 || base.cooldown_remaining > 0.0f) {
        return false;
    }
    if (interceptor_count_ >= max_interceptors) {
        return false;
    }

    const Vec2 direction = (action.target - base.pos).normalized();
    interceptors_[interceptor_count_] =
        Interceptor{.pos = base.pos,
                    .velocity = direction * config_.interceptor_speed,
                    .target = action.target,
                    .active = true};
    ++interceptor_count_;
    --base.ammo;
    base.cooldown_remaining = config_.base_cooldown;
    return true;
}

void Sim::advance_interceptors() noexcept {
    const float step_len = config_.interceptor_speed * config_.dt;
    const float step_len_sq = step_len * step_len;

    std::uint32_t i = 0;
    while (i < interceptor_count_) {
        Interceptor& it = interceptors_[i];
        const Vec2 to_target = it.target - it.pos;
        if (to_target.length_sq() <= step_len_sq) {
            // Arrived (or would overshoot): detonate at the target point.
            spawn_blast(it.target);
            interceptors_[i] = interceptors_[interceptor_count_ - 1];
            --interceptor_count_;
        } else {
            it.pos += it.velocity * config_.dt;
            ++i;
        }
    }
}

void Sim::spawn_blast(Vec2 center) noexcept {
    if (blast_count_ >= max_blasts) {
        return;
    }
    blasts_[blast_count_] = Blast{.center = center, .age = 0.0f, .radius = 0.0f, .active = true};
    ++blast_count_;
}

void Sim::advance_blasts() noexcept {
    std::uint32_t i = 0;
    while (i < blast_count_) {
        Blast& blast = blasts_[i];
        blast.age += config_.dt;
        if (blast.age >= config_.blast_lifetime) {
            blasts_[i] = blasts_[blast_count_ - 1];
            --blast_count_;
        } else {
            blast.radius = blast_radius(blast.age, config_);
            ++i;
        }
    }
}

} // namespace md
