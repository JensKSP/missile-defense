// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
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
    crosshair_ = Vec2{config_.world_width * 0.5f, config_.world_height * 0.5f};
    fire_cooldown_remaining_ = 0.0f;
    score_ = 0;
    tick_ = 0;
    latched_action_ = {};
    terminated_ = false;
    break_timer_ = 0.0f;
    spawn_timer_ = 0.0f;
    next_bonus_score_ = config_.bonus_city_score;
    banked_cities_ = 0;
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
    tick_wasted_ = 0;
    tick_multi_kills_ = 0;
    tick_kills_per_shot_.fill(0);

    if (terminated_) {
        ++tick_;
        return StepResult{.reward = 0, .terminated = true};
    }

    if (wave_started_pending_) { // a wave began (this step or during reset) — sound the siren
        wave_started_pending_ = false;
        push_event(EventType::WaveStarted, Vec2{config_.world_width * 0.5f, config_.world_height});
    }

    const std::int32_t score_before = score_;

    // Reaction-rate limit (DESIGN §5): sample a new action once per
    // decision_interval ticks and hold it between, so no driver — human,
    // scripted, or learned — re-decides faster than a hand can. The crosshair
    // still steers toward the latched aim every tick, so only the *decision* is
    // paced, never the motion.
    const std::uint64_t interval = std::max<std::uint64_t>(1, config_.decision_interval);
    if (tick_ % interval == 0) {
        latched_action_ = action;
    }

    update_cooldowns();
    move_crosshair(latched_action_); // steer the shared cursor (speed-capped)
    try_fire(latched_action_);       // launches detonate at the crosshair
    advance_interceptors();         // may spawn blasts
    advance_blasts();               // age blasts, update radius, expire
    advance_explosions();           // age cosmetic ground-impact fireballs
    steer_smart_bombs();            // smart bombs adjust heading to dodge blasts
    move_threats();                 // integrate threat positions
    split_mirvs();                  // MIRVs split into child warheads at altitude
    score_ += resolve_blast_hits(); // blasts kill threats (blasts win ties)
    resolve_ground_hits();          // landings destroy whatever stands there
    update_waves();                 // spawn, and advance waves with end-of-wave bonus
    award_bonus_cities();           // rebuild a destroyed city at score thresholds
    update_termination();           // all cities destroyed?

    ++tick_;
    return StepResult{.reward = score_ - score_before,
                      .terminated = terminated_,
                      .wasted = tick_wasted_,
                      .multi_kills = tick_multi_kills_,
                      .kills_per_shot = tick_kills_per_shot_};
}

void Sim::update_cooldowns() noexcept {
    for (auto& base : bases_) {
        base.cooldown_remaining = std::max(0.0f, base.cooldown_remaining - config_.dt);
    }
    fire_cooldown_remaining_ = std::max(0.0f, fire_cooldown_remaining_ - config_.dt);
}

/// Steer the crosshair toward `action.aim`, travelling at most
/// `aim_max_speed * dt` this tick, and keep it inside the world. A distant aim
/// point therefore takes several ticks to reach — the cost a hand pays for a
/// large movement. `aim_max_speed <= 0` disables the cap (instant aim).
void Sim::move_crosshair(const Action& action) noexcept {
    if (!action.move) {
        return; // hold position
    }
    Vec2 delta = action.aim - crosshair_;
    if (config_.aim_max_speed > 0.0f) {
        const float max_step = config_.aim_max_speed * config_.dt;
        if (delta.length_sq() > max_step * max_step) {
            delta = delta.normalized() * max_step;
        }
    }
    crosshair_ += delta;
    crosshair_.x = std::clamp(crosshair_.x, 0.0f, config_.world_width);
    crosshair_.y = std::clamp(crosshair_.y, 0.0f, config_.world_height);
}

bool Sim::try_fire(const Action& action) noexcept {
    if (!action.fire) {
        return false;
    }
    if (fire_cooldown_remaining_ > 0.0f) { // global trigger interval (the finger)
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

    const Vec2 target = crosshair_; // you shoot where the crosshair actually is
    const Vec2 direction = (target - base.pos).normalized();
    interceptors_[interceptor_count_] =
        Interceptor{.pos = base.pos,
                    .origin = base.pos,
                    .velocity = direction * config_.interceptor_speed,
                    .target = target};
    ++interceptor_count_;
    --base.ammo;
    base.cooldown_remaining = config_.base_cooldown;
    fire_cooldown_remaining_ = config_.fire_interval;
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
            // Arrived (or would overshoot): detonate at the target point. If the
            // blast pool is momentarily full, hold station rather than consuming
            // the interceptor — spending a round and announcing a detonation that
            // never happened would waste ammo and lie to the audio.
            if (!spawn_blast(it.target)) {
                ++i;
                continue;
            }
            push_event(EventType::Detonate, it.target);
            interceptors_[i] = interceptors_[interceptor_count_ - 1];
            --interceptor_count_;
        } else {
            it.pos += it.velocity * config_.dt;
            ++i;
        }
    }
}

bool Sim::spawn_blast(Vec2 center) noexcept {
    if (blast_count_ >= max_blasts) {
        return false;
    }
    blasts_[blast_count_] = Blast{.center = center, .age = 0.0f, .radius = 0.0f};
    ++blast_count_;
    return true;
}

void Sim::advance_blasts() noexcept {
    std::uint32_t i = 0;
    while (i < blast_count_) {
        Blast& blast = blasts_[i];
        blast.age += config_.dt;
        if (blast.age >= config_.blast_lifetime) {
            // It has finished expanding, so this is the first moment the question
            // "did that interceptor achieve anything?" has a final answer.
            if (blast.kills == 0) {
                ++tick_wasted_;
            }
            ++tick_kills_per_shot_[std::min(blast.kills, kills_per_shot_bins - 1U)];
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
    explosions_[explosion_count_] =
        Explosion{.center = center, .age = 0.0f, .radius = 0.0f, .peak_radius = peak_radius};
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
        std::uint32_t by = 0; // which blast got it — needed to credit the kill
        for (std::uint32_t b = 0; b < blast_count_; ++b) {
            const float radius = blasts_[b].radius;
            if (distance_sq(threats_[i].pos, blasts_[b].center) <= (radius * radius)) {
                killed = true;
                by = b;
                break;
            }
        }
        if (killed) {
            // Every kill after a blast's first costs no extra ammunition — which
            // is the whole of the headroom over the scripted agent's 1.10 kills
            // per interceptor, so it is counted separately.
            if (blasts_[by].kills > 0) {
                ++tick_multi_kills_;
            }
            ++blasts_[by].kills;
            reward += kill_score(threats_[i].type) * score_multiplier();
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
    // Damage is resolved by *where the warhead lands*, not by what it was aimed at
    // when it spawned. Smart bombs steer sideways to dodge blasts, so the two can
    // differ by a long way — and if the stored assignment decided the outcome, a
    // bomb that visibly drifted clear would still level the city it started
    // towards. Neither the player nor a policy could predict that from what they
    // can see, which breaks the parity the whole observation design rests on.
    //
    // Ground slots are evenly spaced, so half a slot either side of the impact
    // point covers exactly one installation; landing anywhere else just cracks the
    // dirt.
    const float reach = config_.world_width / static_cast<float>(base_count + max_cities) * 0.5f;

    std::uint32_t i = 0;
    while (i < threat_count_) {
        const Vec2 impact = threats_[i].pos;
        if (impact.y > 0.0f) {
            ++i;
            continue;
        }

        float nearest = reach;
        bool hit = false;
        bool is_city = false;
        std::uint32_t index = 0;
        for (std::uint32_t c = 0; c < max_cities; ++c) {
            const float d = std::abs(cities_[c].pos.x - impact.x);
            if (cities_[c].alive && d <= nearest) {
                nearest = d;
                hit = true;
                is_city = true;
                index = c;
            }
        }
        for (std::uint32_t b = 0; b < base_count; ++b) {
            const float d = std::abs(bases_[b].pos.x - impact.x);
            if (bases_[b].alive && d <= nearest) {
                nearest = d;
                hit = true;
                is_city = false;
                index = b;
            }
        }

        // A bigger fireball when something actually went up.
        spawn_explosion({impact.x, 2.0f},
                        hit ? config_.explosion_radius_target : config_.explosion_radius_ground);
        if (hit && is_city) {
            cities_[index].alive = false;
            push_event(EventType::CityLost, cities_[index].pos);
        } else if (hit) {
            bases_[index].alive = false; // a dead battery cannot fire for the rest of the wave
            push_event(EventType::BaseLost, bases_[index].pos);
        }

        threats_[i] = threats_[threat_count_ - 1];
        --threat_count_;
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
    wave_started_pending_ = true; // siren on the next step (survives step()'s event reset)
    threats_to_spawn_ = config_.wave_base_threats + ((wave - 1) * config_.wave_threats_increment);
    spawn_timer_ = 0.0f; // first threat of the wave spawns immediately
    // Batteries are rebuilt between waves, as in the arcade. Losing one still
    // costs you its remaining ammo and its coverage for the rest of the wave, but
    // it is not permanent: without this, losing all three left the player unable
    // to fire ever again, watching a decided game play itself out — no agency for
    // a human, and pure noise in a training rollout.
    for (auto& base : bases_) {
        base.alive = true;
        base.ammo = config_.ammo_per_base;
        base.cooldown_remaining = 0.0f;
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
                                     .split_altitude = split_altitude};
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
        // Only split when the whole spread fits. Removing the parent frees exactly
        // one slot, so a saturated field would otherwise silently turn a MIRV into
        // one warhead instead of `mirv_splits` — quietly handing the player kills
        // it never earned. Holding the parent retries next tick instead.
        if ((threat_count_ - 1u) + config_.mirv_splits > max_threats) {
            ++i;
            continue;
        }
        // Split: remove the parent, then spawn child ICBMs from the split point.
        const Vec2 split_pos = threat.pos;
        push_event(EventType::MirvSplit, split_pos);
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
                       .type = ThreatType::Warhead, // child re-entry vehicles
                       .target_kind = kind,
                       .target_index = index,
                       .split_altitude = 0.0f};
            ++threat_count_;
        }
        // Re-check the element swapped into slot i (do not advance i).
    }
}

std::int32_t Sim::kill_score(ThreatType type) const noexcept {
    // Everything that flies is a missile and scores alike, except the smart bomb
    // — the one threat that actively evades, and the one the arcade pays extra
    // for. A MIRV and the warheads it splits into are ordinary missiles.
    return type == ThreatType::SmartBomb ? config_.score_per_smart_bomb : config_.score_per_kill;
}

std::int32_t Sim::score_multiplier() const noexcept {
    if (wave_ == 0u || config_.score_multiplier_wave_step == 0u) {
        return 1;
    }
    const std::uint32_t steps = (wave_ - 1u) / config_.score_multiplier_wave_step;
    return static_cast<std::int32_t>(std::min(steps + 1u, config_.score_multiplier_max));
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
    // The multiplier is the one for the wave just cleared: this runs before
    // start_wave advances the counter.
    score_ += bonus * score_multiplier();
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
    // Earn: bank a credit per threshold crossed. Banking (rather than rebuilding on
    // the spot) is what stops a bonus earned while all six cities still stand from
    // being silently forfeited — which used to punish playing well. The guard on a
    // non-positive threshold also keeps this loop finite for degenerate configs.
    if (config_.bonus_city_score > 0) {
        while (score_ >= next_bonus_score_) {
            banked_cities_ = std::min(banked_cities_ + 1u, max_cities);
            next_bonus_score_ += config_.bonus_city_score;
        }
    }

    // Spend: fill the first gap in the skyline, as soon as there is one.
    while (banked_cities_ > 0) {
        bool rebuilt = false;
        for (auto& city : cities_) {
            if (!city.alive) {
                city.alive = true;
                push_event(EventType::BonusCity, city.pos);
                --banked_cities_;
                rebuilt = true;
                break;
            }
        }
        if (!rebuilt) {
            break; // all six standing: hold the credit for a future loss
        }
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
