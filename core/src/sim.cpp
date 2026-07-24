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
    explosion_count_ = 0;
    event_count_ = 0;
    score_ = 0;
    tick_ = 0;
    terminated_ = false;
    break_timer_ = 0.0f;
    spawn_timer_ = 0.0f;
    next_bonus_score_ = config_.bonus_city_score;
    start_wave(1);
}

void Sim::push_event(EventType type, Vec2 pos) noexcept {
    if (event_count_ >= max_events) {
        return;
    }
    events_[event_count_] = Event{.type = type, .pos = pos};
    ++event_count_;
}

StepResult Sim::step(const Action& action) noexcept {
    event_count_ = 0; // events are per-step

    if (terminated_) {
        ++tick_;
        return StepResult{.reward = 0, .terminated = true};
    }

    const std::int32_t score_before = score_;

    update_cooldowns();
    try_fire(action);
    advance_interceptors();         // may spawn blasts
    advance_blasts();               // age blasts, update radius, expire
    advance_explosions();           // age cosmetic ground-impact fireballs
    steer_smart_bombs();            // smart bombs adjust heading to dodge blasts
    move_threats();                 // integrate threat positions
    split_mirvs();                  // MIRVs split into child warheads at altitude
    score_ += resolve_blast_hits(); // blasts kill threats (blasts win ties)
    resolve_ground_hits();          // surviving threats at ground destroy cities/bases
    update_waves();                 // spawn, and advance waves with end-of-wave bonus
    award_bonus_cities();           // restore a destroyed city at score thresholds
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
    push_event(EventType::Fire, base.pos);
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
            push_event(EventType::Detonate, it.target);
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

void Sim::spawn_explosion(Vec2 center, float peak_radius) noexcept {
    if (explosion_count_ >= max_explosions) {
        return;
    }
    explosions_[explosion_count_] = Explosion{
        .center = center, .age = 0.0f, .radius = 0.0f, .peak_radius = peak_radius, .active = true};
    ++explosion_count_;
}

void Sim::advance_explosions() noexcept {
    const float expand = 0.25f * config_.explosion_lifetime;
    std::uint32_t i = 0;
    while (i < explosion_count_) {
        Explosion& explosion = explosions_[i];
        explosion.age += config_.dt;
        if (explosion.age >= config_.explosion_lifetime) {
            explosions_[i] = explosions_[explosion_count_ - 1];
            --explosion_count_;
        } else {
            explosion.radius = explosion.peak_radius * std::min(1.0f, explosion.age / expand);
            ++i;
        }
    }
}

void Sim::steer_smart_bombs() noexcept {
    const float range_sq = config_.smart_bomb_dodge_range * config_.smart_bomb_dodge_range;
    const float max_vx = threat_speed();
    for (std::uint32_t i = 0; i < threat_count_; ++i) {
        Threat& threat = threats_[i];
        if (threat.type != ThreatType::SmartBomb) {
            continue;
        }
        // Steer laterally away from the nearest blast within reach.
        float best_sq = range_sq;
        std::int32_t nearest = -1;
        for (std::uint32_t b = 0; b < blast_count_; ++b) {
            const float d2 = distance_sq(threat.pos, blasts_[b].center);
            if (d2 < best_sq) {
                best_sq = d2;
                nearest = static_cast<std::int32_t>(b);
            }
        }
        if (nearest < 0) {
            continue;
        }
        const Blast& blast = blasts_[static_cast<std::uint32_t>(nearest)];
        const float away = threat.pos.x >= blast.center.x ? 1.0f : -1.0f;
        threat.velocity.x += away * config_.smart_bomb_dodge_accel * config_.dt;
        threat.velocity.x = std::clamp(threat.velocity.x, -max_vx, max_vx);
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
            push_event(EventType::ThreatKilled, threats_[i].pos);
            threats_[i] = threats_[threat_count_ - 1];
            --threat_count_;
        } else {
            ++i;
        }
    }
    return reward;
}

void Sim::resolve_ground_hits() noexcept {
    std::uint32_t i = 0;
    while (i < threat_count_) {
        const Threat& threat = threats_[i];
        const bool city = threat.target_kind == TargetKind::City;
        const Vec2 target_pos =
            city ? cities_[threat.target_index].pos : bases_[threat.target_index].pos;
        if (threat.pos.y <= target_pos.y) {
            // A threat reaching its target destroys it (a dead base can no longer fire).
            // A bigger fireball if it actually took out a live city/base.
            const bool hit_live =
                city ? cities_[threat.target_index].alive : bases_[threat.target_index].alive;
            spawn_explosion({threat.pos.x, 2.0f}, hit_live ? config_.explosion_radius_target
                                                           : config_.explosion_radius_ground);
            if (hit_live) {
                push_event(city ? EventType::CityLost : EventType::BaseLost, target_pos);
            }
            if (city) {
                cities_[threat.target_index].alive = false;
            } else {
                bases_[threat.target_index].alive = false;
            }
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
        push_event(EventType::WaveCleared,
                   Vec2{config_.world_width * 0.5f, config_.world_height * 0.5f});
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

float Sim::threat_speed() const noexcept {
    return config_.threat_base_speed +
           (static_cast<float>(wave_ - 1) * config_.threat_speed_per_wave);
}

float Sim::mirv_probability() const noexcept {
    return std::min(config_.mirv_max_chance,
                    static_cast<float>(wave_ - 1) * config_.mirv_chance_per_wave);
}

void Sim::spawn_threat() noexcept {
    TargetKind kind = TargetKind::City;
    std::uint32_t index = 0;
    if (!pick_target(kind, index)) {
        return; // nothing left to attack
    }
    const Vec2 target = (kind == TargetKind::City) ? cities_[index].pos : bases_[index].pos;
    const Vec2 origin{rng_.uniform(0.0f, config_.world_width), config_.world_height};

    ThreatType type = ThreatType::Icbm;
    float split_altitude = 0.0f;
    if (wave_ >= config_.smart_bomb_wave && rng_.next_float() < config_.smart_bomb_chance) {
        type = ThreatType::SmartBomb;
    } else if (wave_ >= 2 && rng_.next_float() < mirv_probability()) {
        type = ThreatType::Mirv;
        split_altitude = config_.world_height * rng_.uniform(0.45f, 0.65f);
    }

    threats_[threat_count_] = Threat{.pos = origin,
                                     .origin = origin,
                                     .velocity = (target - origin).normalized() * threat_speed(),
                                     .type = type,
                                     .target_kind = kind,
                                     .target_index = index,
                                     .split_altitude = split_altitude,
                                     .active = true};
    ++threat_count_;
}

void Sim::split_mirvs() noexcept {
    std::uint32_t i = 0;
    while (i < threat_count_) {
        Threat& threat = threats_[i];
        if (threat.type != ThreatType::Mirv || threat.pos.y > threat.split_altitude) {
            ++i;
            continue;
        }
        // Split: remove the parent, then spawn child ICBMs from the split point.
        const Vec2 split_pos = threat.pos;
        threats_[i] = threats_[threat_count_ - 1];
        --threat_count_;
        for (std::uint32_t c = 0; c < config_.mirv_splits && threat_count_ < max_threats; ++c) {
            TargetKind kind = TargetKind::City;
            std::uint32_t index = 0;
            if (!pick_target(kind, index)) {
                break;
            }
            const Vec2 target = (kind == TargetKind::City) ? cities_[index].pos : bases_[index].pos;
            threats_[threat_count_] =
                Threat{.pos = split_pos,
                       .origin = split_pos,
                       .velocity = (target - split_pos).normalized() * threat_speed(),
                       .type = ThreatType::Icbm,
                       .target_kind = kind,
                       .target_index = index,
                       .split_altitude = 0.0f,
                       .active = true};
            ++threat_count_;
        }
        // Re-check the element swapped into slot i (do not advance i).
    }
}

void Sim::award_end_of_wave_bonus() noexcept {
    std::int32_t bonus = 0;
    for (const auto& base : bases_) {
        if (base.alive) {
            bonus += (static_cast<std::int32_t>(base.ammo) * config_.score_per_unused_interceptor);
        }
    }
    for (const auto& city : cities_) {
        if (city.alive) {
            bonus += config_.score_per_surviving_city;
        }
    }
    score_ += bonus;
}

bool Sim::pick_target(TargetKind& kind, std::uint32_t& index) noexcept {
    // Threats target any alive city or base, uniformly at random.
    std::uint32_t alive = 0;
    for (const auto& city : cities_) {
        alive += city.alive ? 1u : 0u;
    }
    for (const auto& base : bases_) {
        alive += base.alive ? 1u : 0u;
    }
    if (alive == 0) {
        return false;
    }
    std::uint32_t nth = rng_.below(alive);
    for (std::uint32_t idx = 0; idx < max_cities; ++idx) {
        if (cities_[idx].alive) {
            if (nth == 0) {
                kind = TargetKind::City;
                index = idx;
                return true;
            }
            --nth;
        }
    }
    for (std::uint32_t idx = 0; idx < base_count; ++idx) {
        if (bases_[idx].alive) {
            if (nth == 0) {
                kind = TargetKind::Base;
                index = idx;
                return true;
            }
            --nth;
        }
    }
    return false;
}

void Sim::award_bonus_cities() noexcept {
    while (score_ >= next_bonus_score_) {
        for (auto& city : cities_) {
            if (!city.alive) {
                city.alive = true; // rebuild the first destroyed city
                push_event(EventType::BonusCity, city.pos);
                break;
            }
        }
        next_bonus_score_ += config_.bonus_city_score;
    }
}

void Sim::update_termination() noexcept {
    for (const auto& city : cities_) {
        if (city.alive) {
            terminated_ = false;
            return;
        }
    }
    if (!terminated_) { // emit once, on the transition to game over
        push_event(EventType::GameOver,
                   Vec2{config_.world_width * 0.5f, config_.world_height * 0.5f});
    }
    terminated_ = true;
}

} // namespace md
