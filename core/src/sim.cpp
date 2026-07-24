#include "md/sim.hpp"

#include "md/config.hpp"
#include "md/entities.hpp"
#include "md/rng.hpp"
#include "md/vec2.hpp"

#include <array>
#include <cstdint>

namespace md {

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

StepResult Sim::step([[maybe_unused]] const Action& action) noexcept {
    // Firing, motion, spawning, collisions, and scoring arrive in the next
    // increment. For now a tick only advances time so drivers can run the loop.
    ++tick_;
    return StepResult{.reward = 0, .terminated = terminated_};
}

} // namespace md
