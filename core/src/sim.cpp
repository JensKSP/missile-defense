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
    terminated_ = false;
    break_timer_ = 0.0f;
    spawn_timer_ = 0.0f;
    start_wave(1);
}

StepResult Sim::step(const Action& action) noexcept {
    if (terminated_) {
        ++tick_;
        return StepResult{.reward = 0, .terminated = true};
    }

    const std::int32_t score_before = score_;

    update_cooldowns();
    try_fire(action);
    advance_interceptors();         // may spawn blasts
    advance_blasts();               // age blasts, update radius, expire
    move_threats();                 // integrate threat positions
    score_ += resolve_blast_hits(); // blasts kill threats (blasts win ties)
    resolve_city_hits();            // surviving threats at ground destroy cities
    update_waves();                 // spawn, and advance waves with end-of-wave bonus
    update_termination();           // all cities destroyed?

    ++tick_;
    return StepResult{.reward = score_ - score_before, .terminated = terminated_};
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
                    .origin = base.pos,
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

void Sim::move_threats() noexcept {
    for (std::uint32_t i = 0; i < threat_count_; ++i) {
        threats_[i].pos += threats_[i].velocity * config_.dt;
    }
}

std::int32_t Sim::resolve_blast_hits() noexcept {
    std::int32_t reward = 0;
    std::uint32_t i = 0;
    while (i < threat_count_) {
        bool killed = false;
        for (std::uint32_t b = 0; b < blast_count_; ++b) {
            const float radius = blasts_[b].radius;
            if (distance_sq(threats_[i].pos, blasts_[b].center) <= (radius * radius)) {
                killed = true;
                break;
            }
        }
        if (killed) {
            reward += config_.score_per_kill;
            threats_[i] = threats_[threat_count_ - 1];
            --threat_count_;
        } else {
            ++i;
        }
    }
    return reward;
}

void Sim::resolve_city_hits() noexcept {
    std::uint32_t i = 0;
    while (i < threat_count_) {
        const std::uint32_t target = threats_[i].target_city;
        if (threats_[i].pos.y <= cities_[target].pos.y) {
            cities_[target].alive = false; // a threat reaching its city destroys it
            threats_[i] = threats_[threat_count_ - 1];
            --threat_count_;
        } else {
            ++i;
        }
    }
}

void Sim::update_waves() noexcept {
    if (break_timer_ > 0.0f) {
        break_timer_ = std::max(0.0f, break_timer_ - config_.dt);
        if (break_timer_ <= 0.0f) {
            start_wave(wave_ + 1);
        }
        return;
    }

    if (threats_to_spawn_ > 0) {
        spawn_timer_ = std::max(0.0f, spawn_timer_ - config_.dt);
        if (spawn_timer_ <= 0.0f && threat_count_ < max_threats) {
            spawn_threat();
            --threats_to_spawn_;
            spawn_timer_ = config_.spawn_interval;
        }
    } else if (threat_count_ == 0) {
        // Wave cleared: award the end-of-wave bonus and pause before the next.
        award_end_of_wave_bonus();
        break_timer_ = config_.wave_break;
    }
}

void Sim::start_wave(std::uint32_t wave) noexcept {
    wave_ = wave;
    threats_to_spawn_ = config_.wave_base_threats + ((wave - 1) * config_.wave_threats_increment);
    spawn_timer_ = 0.0f; // first threat of the wave spawns immediately
    for (auto& base : bases_) {
        if (base.alive) {
            base.ammo = config_.ammo_per_base;
        }
    }
}

void Sim::spawn_threat() noexcept {
    const std::uint32_t city = pick_alive_city();
    if (city >= max_cities) {
        return; // nothing left to attack
    }
    const Vec2 origin{rng_.uniform(0.0f, config_.world_width), config_.world_height};
    const Vec2 target = cities_[city].pos;
    const float speed =
        config_.threat_base_speed + (static_cast<float>(wave_ - 1) * config_.threat_speed_per_wave);
    threats_[threat_count_] = Threat{.pos = origin,
                                     .origin = origin,
                                     .velocity = (target - origin).normalized() * speed,
                                     .type = ThreatType::Icbm,
                                     .target_city = city,
                                     .split_altitude = 0.0f,
                                     .active = true};
    ++threat_count_;
}

void Sim::award_end_of_wave_bonus() noexcept {
    std::int32_t bonus = 0;
    for (const auto& base : bases_) {
        bonus += (static_cast<std::int32_t>(base.ammo) * config_.score_per_unused_interceptor);
    }
    for (const auto& city : cities_) {
        if (city.alive) {
            bonus += config_.score_per_surviving_city;
        }
    }
    score_ += bonus;
}

std::uint32_t Sim::pick_alive_city() noexcept {
    std::uint32_t alive = 0;
    for (const auto& city : cities_) {
        if (city.alive) {
            ++alive;
        }
    }
    if (alive == 0) {
        return max_cities; // sentinel: no valid target
    }
    std::uint32_t nth = rng_.below(alive);
    for (std::uint32_t idx = 0; idx < max_cities; ++idx) {
        if (cities_[idx].alive) {
            if (nth == 0) {
                return idx;
            }
            --nth;
        }
    }
    return max_cities;
}

void Sim::update_termination() noexcept {
    for (const auto& city : cities_) {
        if (city.alive) {
            terminated_ = false;
            return;
        }
    }
    terminated_ = true;
}

} // namespace md
